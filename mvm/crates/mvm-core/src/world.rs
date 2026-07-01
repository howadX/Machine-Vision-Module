//! WorldState — the Agent Mesh blackboard (phase 8.A/8.B).
//!
//! Projection types that detector *agents* write and fusion reads. They are an
//! **outward projection** of the internal state (`PersonTrack`, `mvm-scene`'s
//! `TrackedObject`), not a new source of truth: `PersonState::from_track` flattens
//! a track into what downstream needs.
//!
//! **Per-entity freshness (decision 8.0.1).** Each entity carries its own
//! `last_seen`/`timestamp` — agents update at different cadences (human ~every
//! frame, objects every 3rd, depth every 30th, audio every 10th), so fusion must
//! see "person fresh (28 ms) / object stale (99 ms)" and avoid building edges on
//! stale nodes. A single global timestamp would hide that desync.
//!
//! Nothing consumes `WorldState` yet in the synchronous CLI path; it is the
//! foundation the threads-mesh (8.C, `mvm-mesh`) will build the blackboard on.

use crate::{BBox, PersonTrack};
use std::time::Instant;

/// Metric 3D position in metres, origin = camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Confidence of the estimate [0,1] (0 = no trustworthy depth).
    pub confidence: f32,
}

/// Outward projection of a tracked person (group G₁ Human).
#[derive(Debug, Clone)]
pub struct PersonState {
    /// Shown id (reserved 0–9 for gallery users, ≥10 ephemeral) — `PersonTrack::display_id`.
    pub track_id: u32,
    pub bbox: BBox,
    /// `(name, confidence)` only when a recognised gallery user holds this track.
    pub identity: Option<(String, f32)>,
    pub emotion: Option<(String, f32)>,
    pub posture: Option<(String, f32)>,
    pub position_3d: Option<Position3D>,
    /// Full 8-class emotion softmax (for the Scene JSON contract); `None` when the
    /// track has no Rust-model distribution.
    pub emotion_distribution: Option<[f32; 8]>,
    /// Normalised [x,y] of this person's visible wrist landmarks (BlazePose 15/16).
    /// The always-available interaction anchor — on a ceiling cam the wrists reach
    /// the desk even when the body bbox (head/shoulders) does not.
    pub wrists: Vec<[f32; 2]>,
    /// Normalised hand landmark sets attached to this person (one inner vec per
    /// detected hand). Empty when the hand model found none. Filled by the agent
    /// after tracking — `from_track` leaves it empty.
    pub hands: Vec<Vec<[f32; 2]>>,
    /// Body-visibility regime ("full"/"upper"/"head") — occlusion-aware posture.
    /// `upper`/`head` mean the lower body was not seen, so `posture` may honestly
    /// be `None`; fusion reads this (with `torso_attitude` + context) instead.
    pub body_coverage: Option<&'static str>,
    /// 2D image-plane torso attitude ("upright"/"inclined"/"horizontal") — flows
    /// even when `posture` is withheld under occlusion.
    pub torso_attitude: Option<&'static str>,
    /// Coarse locomotion state read straight off the track (phase 9.8): `"idle"` /
    /// `"moving"`. `None` until the motion ring has committed a state (`Unknown`).
    pub motion_state: Option<&'static str>,
    /// Centroid speed in torso-lengths/sec (phase 9.8) — distance-invariant.
    pub velocity: Option<f32>,
    pub last_seen: Instant,
}

impl PersonState {
    /// Project a `PersonTrack` into the blackboard view. `identity` follows the
    /// reserved-id model (`shown_identity()` → name only for reserved slots).
    pub fn from_track(t: &PersonTrack, now: Instant) -> Self {
        let position_3d = match (t.position_x, t.position_y, t.position_z) {
            (Some(x), Some(y), Some(z)) => Some(Position3D {
                x,
                y,
                z,
                confidence: t.ws_confidence.unwrap_or(0.0),
            }),
            _ => None,
        };
        Self {
            track_id: t.display_id,
            bbox: t.bbox,
            identity: t
                .shown_identity()
                .map(|n| (n.clone(), t.identity_confidence.unwrap_or(0.0))),
            emotion: t.emotion.clone().map(|l| (l, t.emotion_confidence.unwrap_or(0.0))),
            posture: t.posture.clone().map(|l| (l, t.posture_confidence.unwrap_or(0.0))),
            position_3d,
            emotion_distribution: t.emotion_distribution,
            // Landmarks aren't on the track — the agent attaches them post-tracking
            // from the per-body pose/hand data (see `project_people`).
            wrists: Vec::new(),
            hands: Vec::new(),
            body_coverage: t.body_coverage,
            torso_attitude: t.torso_attitude,
            // Motion comes straight off the track (phase 9.8) — no round-trip.
            // Emit a committed state only (skip the startup `Unknown`).
            motion_state: match t.motion.state() {
                crate::motion::MotionState::Unknown => None,
                s => Some(s.as_str()),
            },
            velocity: if t.motion.is_empty() { None } else { Some(t.motion.speed()) },
            last_seen: now,
        }
    }
}

/// Outward projection of a tracked object (group G₂ Scene). Mirrors `mvm-scene`'s
/// `TrackedObject` but lives in `mvm-core` (it can't depend on `mvm-scene`); the
/// consumer builds this from a `TrackedObject`.
#[derive(Debug, Clone)]
pub struct ObjectState {
    /// Stable scene-graph id (≥ 10000).
    pub id: u32,
    pub class: String,
    pub bbox: BBox,
    pub confidence: f32,
    /// Frames this object has been tracked (temporal persistence, mirrors
    /// `TrackedObject::age`).
    pub age: u32,
    /// Metric position from depth-anchoring; `None` when no depth anchor.
    pub position_3d: Option<Position3D>,
    pub last_seen: Instant,
}

/// An audio event (group G₄ Environment — YAMNet, phase 8.E.3, not yet wired).
#[derive(Debug, Clone)]
pub struct AudioEvent {
    pub class: String,
    pub confidence: f32,
    pub timestamp: Instant,
}

/// A cross-group activity inferred by fusion (e.g. "typing", "two people talking").
#[derive(Debug, Clone)]
pub struct Activity {
    pub label: String,
    pub confidence: f32,
    /// Entity ids (people/objects) participating.
    pub participants: Vec<u32>,
}

/// The shared blackboard the Agent Mesh fuses over. Agents update their slice
/// independently (each entity carries `last_seen`); a global `frame_id`/`timestamp`
/// marks the most recent frame any agent processed.
#[derive(Debug, Clone)]
pub struct WorldState {
    pub frame_id: u64,
    pub timestamp: Instant,
    pub people: Vec<PersonState>,
    pub objects: Vec<ObjectState>,
    pub audio_events: Vec<AudioEvent>,
    pub activities: Vec<Activity>,
    pub anomaly_score: f32,
}

impl WorldState {
    /// Empty world stamped `now`.
    pub fn new(now: Instant) -> Self {
        Self {
            frame_id: 0,
            timestamp: now,
            people: Vec::new(),
            objects: Vec::new(),
            audio_events: Vec::new(),
            activities: Vec::new(),
            anomaly_score: 0.0,
        }
    }

    /// People whose `last_seen` is within `max_age` of `now` — fusion should only
    /// relate fresh nodes (per-entity freshness, decision 8.0.1).
    pub fn fresh_people(&self, now: Instant, max_age: std::time::Duration) -> impl Iterator<Item = &PersonState> {
        self.people.iter().filter(move |p| now.duration_since(p.last_seen) <= max_age)
    }

    /// Objects whose `last_seen` is within `max_age` of `now` — the object-side
    /// twin of [`fresh_people`](Self::fresh_people). Fusion relates only fresh
    /// nodes, so a stale object (its agent fell behind) drops out of the graph.
    pub fn fresh_objects(&self, now: Instant, max_age: std::time::Duration) -> impl Iterator<Item = &ObjectState> {
        self.objects.iter().filter(move |o| now.duration_since(o.last_seen) <= max_age)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(display_id: u32) -> PersonTrack {
        let mut t = PersonTrack::new(99, BBox { x1: 0.2, y1: 0.3, x2: 0.5, y2: 0.7 });
        t.display_id = display_id;
        t
    }

    #[test]
    fn position_3d_fields_roundtrip() {
        let p = Position3D { x: 1.0, y: -0.5, z: 2.0, confidence: 0.8 };
        assert_eq!(p.x, 1.0);
        assert_eq!(p.z, 2.0);
        assert!((p.confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn person_state_from_track_projects_id_and_bbox() {
        let now = Instant::now();
        let ps = PersonState::from_track(&track(10), now);
        assert_eq!(ps.track_id, 10);
        assert!((ps.bbox.x1 - 0.2).abs() < 1e-6);
        assert!(ps.identity.is_none(), "ephemeral track has no shown identity");
        assert!(ps.position_3d.is_none(), "no 3D set yet");
    }

    #[test]
    fn person_state_with_reserved_identity() {
        let now = Instant::now();
        let mut t = track(0); // reserved slot → identity is shown
        t.identity = Some("Вадим".into());
        t.identity_confidence = Some(0.57);
        t.emotion = Some("Neutral".into());
        t.emotion_confidence = Some(0.4);
        t.position_x = Some(0.9);
        t.position_y = Some(0.4);
        t.position_z = Some(1.1);
        t.ws_confidence = Some(0.65);
        let ps = PersonState::from_track(&t, now);
        assert_eq!(ps.identity, Some(("Вадим".to_string(), 0.57)));
        assert_eq!(ps.emotion, Some(("Neutral".to_string(), 0.4)));
        let pos = ps.position_3d.expect("3D present");
        assert!((pos.z - 1.1).abs() < 1e-6 && (pos.confidence - 0.65).abs() < 1e-6);
    }

    #[test]
    fn object_state_holds_class_and_bbox() {
        let o = ObjectState {
            id: 10000,
            class: "cup".into(),
            bbox: BBox { x1: 0.4, y1: 0.4, x2: 0.5, y2: 0.55 },
            confidence: 0.82,
            age: 5,
            position_3d: Some(Position3D { x: 0.18, y: 0.65, z: 0.68, confidence: 0.4 }),
            last_seen: Instant::now(),
        };
        assert_eq!(o.id, 10000);
        assert_eq!(o.class, "cup");
        assert!(o.position_3d.is_some());
    }

    #[test]
    fn world_state_new_empty() {
        let w = WorldState::new(Instant::now());
        assert_eq!(w.frame_id, 0);
        assert!(w.people.is_empty() && w.objects.is_empty());
        assert_eq!(w.anomaly_score, 0.0);
    }

    #[test]
    fn world_state_add_person_and_clone_independent() {
        let now = Instant::now();
        let mut w = WorldState::new(now);
        w.people.push(PersonState::from_track(&track(10), now));
        let snapshot = w.clone();
        w.people.push(PersonState::from_track(&track(11), now));
        assert_eq!(snapshot.people.len(), 1, "clone is an independent snapshot");
        assert_eq!(w.people.len(), 2);
    }

    #[test]
    fn fresh_people_filters_by_age() {
        let now = Instant::now();
        let mut w = WorldState::new(now);
        let mut stale = PersonState::from_track(&track(10), now);
        stale.last_seen = now - std::time::Duration::from_millis(500);
        let fresh = PersonState::from_track(&track(11), now);
        w.people.push(stale);
        w.people.push(fresh);
        let n = w.fresh_people(now, std::time::Duration::from_millis(100)).count();
        assert_eq!(n, 1, "only the fresh person within 100ms");
    }

    #[test]
    fn fresh_objects_filters_by_age() {
        let now = Instant::now();
        let mut w = WorldState::new(now);
        let mk = |id: u32, age_ms: u64| ObjectState {
            id,
            class: "cup".into(),
            bbox: BBox { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2 },
            confidence: 0.8,
            age: 1,
            position_3d: None,
            last_seen: now - std::time::Duration::from_millis(age_ms),
        };
        w.objects.push(mk(10000, 500)); // stale
        w.objects.push(mk(10001, 10)); // fresh
        let ids: Vec<u32> = w
            .fresh_objects(now, std::time::Duration::from_millis(100))
            .map(|o| o.id)
            .collect();
        assert_eq!(ids, vec![10001], "only the fresh object within 100ms");
    }
}
