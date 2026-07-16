//! Anomaly scoring v2 (phase 11.B2, pure arithmetic).
//!
//! v1 was a single heuristic — the share of unexpected object classes. v2
//! combines the prediction-error-flavoured signals the pipeline already
//! computes into one calibrated [0,1] score:
//!
//! - **unexpected object** (v1, kept): a class outside the normal room/desk
//!   inventory (knife on the desk, luggage appearing);
//! - **pose OOD** (`plausibility == "low"`): the VAE says this skeleton is far
//!   from anything seen in training — a prediction-error signal;
//! - **sudden fall**: a body that was *standing* within the last
//!   [`ANOMALY_FALL_WINDOW_S`] and reads *lying* now. Long-term lying (bed,
//!   couch) carries no such transition and is not flagged.
//!
//! Components combine by noisy-OR (`1 − Π(1−cᵢ)`) — independent weak evidence
//! accumulates without a hand-tuned sum ever leaving [0,1]. The published score
//! is a decaying peak (`max(raw, prev·λ)`): a one-tick spike (a fall *is* one
//! tick) stays visible to a slow consumer for a few ticks instead of vanishing.

use crate::world::{ObjectState, PersonState};

/// Objects expected in a normal room/desk scene. Anything
/// relevant-but-not-here (sharp items, personal effects appearing/leaving)
/// nudges the anomaly score.
pub const EXPECTED_OBJECTS: &[&str] = &[
    "chair", "couch", "bed", "dining table", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "book", "clock", "vase", "potted plant", "cup",
    "bottle", "bowl", "fork", "spoon", "microwave", "oven", "toaster", "sink",
    "refrigerator",
];

/// Contribution of each unexpected object class instance.
pub const ANOMALY_OBJECT_WEIGHT: f32 = 0.25;
/// Contribution of a person whose posture plausibility is "low" (VAE OOD).
pub const ANOMALY_POSE_OOD: f32 = 0.3;
/// Contribution of a standing→lying transition inside the fall window.
pub const ANOMALY_FALL: f32 = 0.6;
/// How recently a body must have stood for its lying to count as a *fall*.
pub const ANOMALY_FALL_WINDOW_S: f64 = 3.0;
/// Peak decay per tick — a spike stays visible ~10 ticks.
pub const ANOMALY_DECAY: f32 = 0.9;
/// Bounded per-track memory (the tracker's id spaces are bounded anyway).
const STANDING_MEMORY_CAP: usize = 64;

/// Stateful anomaly scorer (phase 11.B2). Keeps the last time each track stood
/// (for the fall transition) and the decaying published peak.
#[derive(Debug, Default)]
pub struct AnomalyScorer {
    /// `(track_id, t_secs when last seen standing)`.
    standing_seen: Vec<(u32, f64)>,
    score: f32,
}

impl AnomalyScorer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance one tick and return the published score. `t_secs` is any
    /// monotonic clock in seconds (the same one the activity journal uses).
    pub fn update(&mut self, people: &[PersonState], objects: &[ObjectState], t_secs: f64) -> f32 {
        let mut components: Vec<f32> = Vec::new();

        // Unexpected objects (v1 component, per instance).
        let unexpected = objects
            .iter()
            .filter(|o| !EXPECTED_OBJECTS.contains(&o.class.as_str()))
            .count();
        if unexpected > 0 {
            components.push((unexpected as f32 * ANOMALY_OBJECT_WEIGHT).clamp(0.0, 1.0));
        }

        for p in people {
            let label = p.posture.as_ref().map(|(l, _)| l.as_str());
            if label == Some("standing") {
                match self.standing_seen.iter_mut().find(|(id, _)| *id == p.track_id) {
                    Some(e) => e.1 = t_secs,
                    None => {
                        if self.standing_seen.len() < STANDING_MEMORY_CAP {
                            self.standing_seen.push((p.track_id, t_secs));
                        }
                    }
                }
            }
            if label == Some("lying") {
                let recently_stood = self
                    .standing_seen
                    .iter()
                    .any(|(id, t)| *id == p.track_id && t_secs - *t <= ANOMALY_FALL_WINDOW_S);
                if recently_stood {
                    components.push(ANOMALY_FALL);
                }
            }
            // Pose OOD — the VAE's prediction-error flag.
            if p.plausibility.as_deref() == Some("low") {
                components.push(ANOMALY_POSE_OOD);
            }
        }

        // Forget stale standing memory (well past the window).
        self.standing_seen.retain(|(_, t)| t_secs - *t <= ANOMALY_FALL_WINDOW_S * 4.0);

        // Noisy-OR, then the decaying peak.
        let raw = 1.0 - components.iter().fold(1.0f32, |acc, c| acc * (1.0 - c.clamp(0.0, 1.0)));
        self.score = raw.max(self.score * ANOMALY_DECAY);
        if self.score < 1e-3 {
            self.score = 0.0;
        }
        self.score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::Position3D;
    use crate::BBox;
    use std::time::Instant;

    fn person(track_id: u32, posture: Option<&str>) -> PersonState {
        PersonState {
            track_id,
            bbox: BBox { x1: 0.3, y1: 0.2, x2: 0.5, y2: 0.7 },
            identity: None,
            emotion: None,
            posture: posture.map(|l| (l.to_string(), 0.9)),
            position_3d: Some(Position3D { x: 0.0, y: 0.0, z: 2.0, confidence: 0.5 }),
            emotion_distribution: None,
            wrists: Vec::new(),
            hands: Vec::new(),
            body_coverage: None,
            torso_attitude: None,
            motion_state: None,
            velocity: None,
            left_gesture: None,
            right_gesture: None,
            recon_error: None,
            plausibility: None,
            torso_flexion: None,
            head_yaw: None,
            head_facing: None,
            last_seen: Instant::now(),
        }
    }

    fn object(class: &str) -> ObjectState {
        ObjectState {
            id: 10000,
            class: class.into(),
            bbox: BBox { x1: 0.4, y1: 0.4, x2: 0.5, y2: 0.5 },
            confidence: 0.8,
            age: 2,
            position_3d: None,
            mask: None,
            remembered: false,
            last_seen: Instant::now(),
        }
    }

    #[test]
    fn calm_scene_scores_zero() {
        let mut s = AnomalyScorer::new();
        let score = s.update(&[person(0, Some("standing"))], &[object("cup")], 0.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn unexpected_object_raises() {
        let mut s = AnomalyScorer::new();
        let score = s.update(&[], &[object("knife")], 0.0);
        assert!((score - ANOMALY_OBJECT_WEIGHT).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn sudden_fall_flags_but_longterm_lying_does_not() {
        // Fall: standing at t=0 → lying at t=1 (inside the window).
        let mut s = AnomalyScorer::new();
        s.update(&[person(0, Some("standing"))], &[], 0.0);
        let fall = s.update(&[person(0, Some("lying"))], &[], 1.0);
        assert!((fall - ANOMALY_FALL).abs() < 1e-6, "got {fall}");

        // Lying with no recent standing (asleep on the couch) → no fall claim.
        let mut s2 = AnomalyScorer::new();
        let lying = s2.update(&[person(0, Some("lying"))], &[], 0.0);
        assert_eq!(lying, 0.0);

        // Standing long ago (outside the window) → no fall claim either.
        let mut s3 = AnomalyScorer::new();
        s3.update(&[person(0, Some("standing"))], &[], 0.0);
        for t in 1..40 {
            s3.update(&[person(0, Some("sitting"))], &[], t as f64);
        }
        let late = s3.update(&[person(0, Some("lying"))], &[], 40.0);
        assert!(late < 0.05, "stale standing must not read as a fall, got {late}");
    }

    #[test]
    fn knife_plus_fall_scores_above_either_alone() {
        // The plan's discriminating case: knife + sudden fall > a calm scene
        // and > each component alone (noisy-OR accumulation).
        let mut s = AnomalyScorer::new();
        s.update(&[person(0, Some("standing"))], &[], 0.0);
        let combined = s.update(&[person(0, Some("lying"))], &[object("knife")], 1.0);
        let expected = 1.0 - (1.0 - ANOMALY_FALL) * (1.0 - ANOMALY_OBJECT_WEIGHT);
        assert!((combined - expected).abs() < 1e-6, "got {combined}");
        assert!(combined > ANOMALY_FALL && combined > ANOMALY_OBJECT_WEIGHT);
        assert!(combined <= 1.0);
    }

    #[test]
    fn pose_ood_contributes() {
        let mut s = AnomalyScorer::new();
        let mut p = person(0, Some("standing"));
        p.plausibility = Some("low".into());
        let score = s.update(&[p], &[], 0.0);
        assert!((score - ANOMALY_POSE_OOD).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn spike_decays_instead_of_vanishing() {
        let mut s = AnomalyScorer::new();
        s.update(&[person(0, Some("standing"))], &[], 0.0);
        let spike = s.update(&[person(0, Some("lying"))], &[], 1.0);
        let next = s.update(&[], &[], 2.0);
        assert!((next - spike * ANOMALY_DECAY).abs() < 1e-6, "decaying peak, got {next}");
        let mut last = next;
        for t in 3..120 {
            last = s.update(&[], &[], t as f64);
        }
        assert_eq!(last, 0.0, "calm scene eventually returns to zero");
    }
}
