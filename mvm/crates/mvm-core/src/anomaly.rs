//! Anomaly scoring v2 (phase 11.B2, pure arithmetic).
//!
//! v1 was a single heuristic — the share of unexpected object classes. v2
//! combines the prediction-error-flavoured signals the pipeline already
//! computes into one calibrated [0,1] score:
//!
//! - **unexpected object** (v1, kept): a class outside the normal room/desk
//!   inventory (knife on the desk, luggage appearing);
//! - **pose OOD** (`plausibility == "low"`): the Mahalanobis OOD gate says this
//!   skeleton is far from anything seen in training — a prediction-error signal;
//! - **sudden fall** (phase 15.6, E2 scene-level): a body whose pelvis *dropped
//!   fast AND ended on the floor* — peak downward pelvis speed ≥
//!   [`E2_FALL_VEL_MS`] over a [`E2_WIN_S`] window AND a floor endpoint (height
//!   above floor ≤ [`E2_FLOOR_M`]). This replaces the old per-track
//!   standing→lying label window (LJ4: that rule false-alarmed on brisk
//!   controlled lie-downs and lost real falls to id-churn — the fall itself
//!   churns the track id, killing the per-id standing memory). The kinematic
//!   rule is id-local only through a short height ring, and the floor gate is
//!   what separates a fall from a fast crouch (a vault descends just as fast but
//!   never reaches the ground). Long-term lying (bed, couch) has no descent and
//!   is not flagged.
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
/// Contribution of a person whose posture plausibility is "low" (Mahalanobis OOD).
pub const ANOMALY_POSE_OOD: f32 = 0.3;
/// Contribution of a detected fall (E2 scene-level rule).
pub const ANOMALY_FALL: f32 = 0.6;
/// Peak downward pelvis speed (m/s) over an [`E2_WIN_S`] window that marks a fall
/// (phase 15.6, LJ2 E2). Calibrated 1.13 m/s on 60 fps CMU mocap; recalibrated to
/// the ~15 fps production cadence (the coarser tick smears the peak) → 1.05. CMU
/// falls are stunt/acrobatic, so this bounds the true clinical threshold *from
/// above* (a gentle elderly fall may descend slower).
pub const E2_FALL_VEL_MS: f32 = 1.05;
/// Floor endpoint: a pelvis-above-floor at/below this (metres) counts as having
/// *reached the ground*. This is the veto that separates a fall (descends → floor)
/// from a fast crouch or vault (descends → stays up). Approximate: it rides the
/// uncalibrated 3D scale (tail #3, ±15–30%).
pub const E2_FLOOR_M: f32 = 0.35;
/// Window (seconds) the peak descent speed is averaged over — the LJ2 robust
/// recipe (a mean descent over 0.4 s, immune to single-frame FK spikes, not a raw
/// finite difference).
pub const E2_WIN_S: f64 = 0.4;
/// Per-track pelvis-height history retained (seconds). The descent and the floor
/// endpoint must co-occur inside this window for a fall to fire — long enough to
/// span a real fall, short enough that a body lying still eventually shows ~0
/// descent and stops flagging.
pub const E2_LOOKBACK_S: f64 = 3.0;
/// Peak decay per tick — a spike stays visible ~10 ticks.
pub const ANOMALY_DECAY: f32 = 0.9;
/// Bounded per-track memory (the tracker's id spaces are bounded anyway).
const TRACK_MEMORY_CAP: usize = 64;
/// Bounded per-track height samples (E2_LOOKBACK_S · fps, with headroom).
const HEIGHT_SAMPLES_CAP: usize = 96;

/// One track's recent pelvis-height-above-floor samples `(t_secs, height_m)`,
/// time-ordered, bounded to [`E2_LOOKBACK_S`].
#[derive(Debug, Default)]
struct TrackHeights {
    id: u32,
    samples: Vec<(f64, f32)>,
}

/// Decide the E2 fall rule on one track's recent height ring. `require_floor` is
/// the shipped primary path (velocity + ground-endpoint veto). `require_floor =
/// false` is the documented **velocity-only fallback** (LJ6 P4.0): if the floor
/// proxy proves too noisy live (scale wobble) it can be disabled, at the cost of
/// the vault false-positive the floor gate catches. Pure arithmetic.
fn e2_fall(samples: &[(f64, f32)], require_floor: bool) -> bool {
    if samples.len() < 3 {
        return false;
    }
    // Median-smooth the heights (k=3) — kills single-frame 3D jitter, keeps the
    // ramp (mirrors the LJ2/LJ4 recipe).
    let med: Vec<f32> = (0..samples.len())
        .map(|i| {
            let lo = i.saturating_sub(1);
            let hi = (i + 2).min(samples.len());
            let mut w: Vec<f32> = samples[lo..hi].iter().map(|s| s.1).collect();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w[w.len() / 2]
        })
        .collect();

    // Reached the floor anywhere in the window (the endpoint of a fall).
    let reached_floor = med.iter().cloned().fold(f32::INFINITY, f32::min) <= E2_FLOOR_M;
    if require_floor && !reached_floor {
        return false;
    }

    // Peak windowed descent speed: for each later sample j, pair it with the
    // earlier sample closest to E2_WIN_S back and take the mean descent over that
    // gap. Robust to fps jitter — the window is measured in *time*, not frames.
    let mut peak = 0.0f32;
    for j in 1..samples.len() {
        let tj = samples[j].0;
        // earliest i with a gap ≥ ~half the window; prefer the one nearest E2_WIN_S.
        let mut best_i: Option<usize> = None;
        let mut best_err = f64::INFINITY;
        for i in 0..j {
            let gap = tj - samples[i].0;
            if gap < E2_WIN_S * 0.5 {
                break; // samples are time-ordered; closer ones only get smaller
            }
            let err = (gap - E2_WIN_S).abs();
            if err <= best_err {
                best_err = err;
                best_i = Some(i);
            }
        }
        if let Some(i) = best_i {
            let gap = (tj - samples[i].0) as f32;
            if gap > 0.0 {
                let descent = (med[i] - med[j]) / gap; // >0 = went DOWN
                peak = peak.max(descent);
            }
        }
    }

    peak >= E2_FALL_VEL_MS
}

/// Stateful anomaly scorer (phase 11.B2; fall rule reworked to E2 in 15.6). Keeps
/// a short per-track pelvis-height ring (for the scene-level fall rule) and the
/// decaying published peak.
#[derive(Debug)]
pub struct AnomalyScorer {
    /// Per-track pelvis-height-above-floor rings for the E2 fall rule.
    heights: Vec<TrackHeights>,
    /// Camera mount height above floor (metres); pelvis height = this − y (y-down).
    camera_height_m: f32,
    score: f32,
}

impl Default for AnomalyScorer {
    fn default() -> Self {
        Self { heights: Vec::new(), camera_height_m: 1.75, score: 0.0 }
    }
}

impl AnomalyScorer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the camera mount height used to convert `position_3d.y` (y-down) into a
    /// pelvis height above the floor. Defaults to 1.75 m (the eye-level rig).
    pub fn with_camera_height(mut self, camera_height_m: f32) -> Self {
        self.camera_height_m = camera_height_m;
        self
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

        let live_ids: Vec<u32> = people.iter().map(|p| p.track_id).collect();
        for p in people {
            // E2 scene-level fall: record pelvis height above floor, then decide on
            // the track's recent ring. Height needs a 3D position.
            if let Some(pos) = &p.position_3d {
                let height = self.camera_height_m - pos.y; // y-down → up
                let idx = match self.heights.iter().position(|h| h.id == p.track_id) {
                    Some(i) => Some(i),
                    None if self.heights.len() < TRACK_MEMORY_CAP => {
                        self.heights.push(TrackHeights { id: p.track_id, samples: Vec::new() });
                        Some(self.heights.len() - 1)
                    }
                    None => None,
                };
                if let Some(i) = idx {
                    let ring = &mut self.heights[i];
                    ring.samples.push((t_secs, height));
                    ring.samples.retain(|(t, _)| t_secs - *t <= E2_LOOKBACK_S);
                    if ring.samples.len() > HEIGHT_SAMPLES_CAP {
                        let drop = ring.samples.len() - HEIGHT_SAMPLES_CAP;
                        ring.samples.drain(0..drop);
                    }
                    if e2_fall(&ring.samples, true) {
                        components.push(ANOMALY_FALL);
                    }
                }
            }
            // Pose OOD — the Mahalanobis prediction-error flag.
            if p.plausibility.as_deref() == Some("low") {
                components.push(ANOMALY_POSE_OOD);
            }
        }

        // Forget rings for tracks gone well past the lookback (also drop absent
        // tracks whose ring has fully aged out).
        self.heights.retain(|h| {
            live_ids.contains(&h.id)
                || h.samples.iter().any(|(t, _)| t_secs - *t <= E2_LOOKBACK_S)
        });

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

    /// A person at a given pelvis height above the floor (metres). The default
    /// camera height is 1.75 m and the frame is y-down, so `position_3d.y =
    /// 1.75 − height`.
    fn person_h(track_id: u32, height: f32) -> PersonState {
        let mut p = person(track_id, Some("lying"));
        p.position_3d = Some(Position3D { x: 0.0, y: 1.75 - height, z: 2.0, confidence: 0.5 });
        p
    }

    /// Drive one track descending from `h0`→`h1` (m above floor) linearly over
    /// `dur_s` at 15 fps through a fresh scorer; return the peak published score.
    fn drive_descent(h0: f32, h1: f32, dur_s: f64, objects: &[ObjectState]) -> f32 {
        let mut s = AnomalyScorer::new();
        let fps = 15.0f64;
        let n = (dur_s * fps).round() as usize;
        let mut peak = 0.0f32;
        for k in 0..=n {
            let t = k as f64 / fps;
            let frac = if n == 0 { 1.0 } else { k as f32 / n as f32 };
            let h = h0 + (h1 - h0) * frac;
            peak = peak.max(s.update(&[person_h(0, h)], objects, t));
        }
        peak
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
    fn fast_fall_to_floor_fires() {
        // Pelvis drops ~1 m in 0.6 s and ends on the floor → E2 fires (0.6).
        let fall = drive_descent(1.1, 0.1, 0.6, &[]);
        assert!((fall - ANOMALY_FALL).abs() < 1e-6, "got {fall}");
    }

    #[test]
    fn slow_controlled_lie_down_stays_quiet() {
        // Same floor endpoint, but a 5 s controlled descent — velocity stays well
        // under threshold, so no fall (the LJ4 false-alarm the per-id rule made).
        let quiet = drive_descent(1.1, 0.15, 5.0, &[]);
        assert_eq!(quiet, 0.0, "controlled lie-down must not read as a fall, got {quiet}");
    }

    #[test]
    fn fast_crouch_above_floor_is_not_a_fall() {
        // Fast descent (~1.6 m/s) that never reaches the floor (min 0.6 m) — the
        // vault/crouch confound. The floor veto keeps it quiet.
        let quiet = drive_descent(1.4, 0.6, 0.5, &[]);
        assert_eq!(quiet, 0.0, "a fast crouch that stays up is not a fall, got {quiet}");
    }

    #[test]
    fn already_lying_still_does_not_flag() {
        // On the floor the whole time (no descent) → quiet, even though the floor
        // endpoint is satisfied (bed/couch long-term lying).
        let mut s = AnomalyScorer::new();
        let mut last = 0.0;
        for k in 0..60 {
            last = s.update(&[person_h(0, 0.15)], &[], k as f64 / 15.0);
        }
        assert_eq!(last, 0.0, "static lying must not flag, got {last}");
    }

    #[test]
    fn velocity_only_fallback_path() {
        // The documented fallback (P4.0): velocity-only, floor gate disabled. A
        // fast crouch that stays above the floor is a fall under velocity-only but
        // NOT under the shipped floor-gated rule — this pins both code paths.
        let fps = 15.0f64;
        let n = (0.5 * fps).round() as usize;
        let samples: Vec<(f64, f32)> = (0..=n)
            .map(|k| {
                let frac = k as f32 / n as f32;
                (k as f64 / fps, 1.4 + (0.6 - 1.4) * frac) // 1.4 → 0.6 m, never reaches floor
            })
            .collect();
        assert!(!e2_fall(&samples, true), "floor-gated rule must veto the vault");
        assert!(e2_fall(&samples, false), "velocity-only fallback must fire on the fast descent");
    }

    #[test]
    fn fall_plus_knife_scores_above_either_alone() {
        // The plan's discriminating case: knife + fall > a calm scene and > each
        // component alone (noisy-OR accumulation).
        let combined = drive_descent(1.1, 0.1, 0.6, &[object("knife")]);
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
        // Drive a fall, then the person leaves the frame — the fall spike must
        // decay smoothly rather than vanish in one tick.
        let mut s = AnomalyScorer::new();
        let fps = 15.0f64;
        let n = (0.6 * fps).round() as usize;
        let mut spike = 0.0f32;
        for k in 0..=n {
            let frac = k as f32 / n as f32;
            let h = 1.1 + (0.1 - 1.1) * frac;
            spike = spike.max(s.update(&[person_h(0, h)], &[], k as f64 / fps));
        }
        assert!((spike - ANOMALY_FALL).abs() < 1e-6, "fall should peak at ANOMALY_FALL, got {spike}");
        // Person gone → the ring stops updating, the published peak decays.
        let next = s.update(&[], &[], (n as f64 + 1.0) / fps);
        assert!((next - spike * ANOMALY_DECAY).abs() < 1e-6, "decaying peak, got {next}");
        let mut last = next;
        for t in 3..120 {
            last = s.update(&[], &[], 10.0 + t as f64);
        }
        assert_eq!(last, 0.0, "calm scene eventually returns to zero");
    }
}
