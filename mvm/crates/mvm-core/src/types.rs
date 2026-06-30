//! MVM core types — bounding boxes, detections, person tracks.
//!
//! These types are shared across all MVM modules. They carry identity,
//! emotion, posture, and 3D workspace data with confidence scores.

/// A bounding box in normalised frame coordinates (0.0..1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub fn area(&self) -> f32 {
        let w = (self.x2 - self.x1).max(0.0);
        let h = (self.y2 - self.y1).max(0.0);
        w * h
    }

    pub fn iou(&self, other: &BBox) -> f32 {
        let xi1 = self.x1.max(other.x1);
        let yi1 = self.y1.max(other.y1);
        let xi2 = self.x2.min(other.x2);
        let yi2 = self.y2.min(other.y2);
        let inter = ((xi2 - xi1).max(0.0)) * ((yi2 - yi1).max(0.0));
        let union = self.area() + other.area() - inter;
        if union < 1e-6 { 0.0 } else { inter / union }
    }
}

/// A single detection from the vision pipeline, with optional enrichment.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox,
    // Identity
    pub identity: Option<String>,
    pub identity_confidence: Option<f32>,
    // Emotion
    pub emotion: Option<String>,
    pub emotion_confidence: Option<f32>,
    // Posture
    pub posture: Option<String>,
    pub posture_confidence: Option<f32>,
    // 3D Workspace
    pub distance: Option<f32>,
    pub position_x: Option<f32>,
    pub position_y: Option<f32>,
    pub position_z: Option<f32>,
    pub ws_confidence: Option<f32>,
    // Posture quality
    pub recon_error: Option<f32>,
    pub plausibility: Option<String>,
    /// Full softmax distribution over 8 emotion classes (from Rust ONNX model).
    /// None when coming from Python bridge (which only passes label+conf).
    pub emotion_distribution: Option<[f32; 8]>,
    /// SFace 128-d L2-normalised face embedding (phase 7.5). Carries through to
    /// the tracker for margin-identity (#5) and cosine re-link (#6). None when
    /// no native embedding was computed (e.g. Python-only path).
    pub embedding: Option<Vec<f32>>,
}

impl Detection {
    /// Convenience constructor: bbox-only detection, all enrichment None.
    pub fn bare(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            bbox: BBox { x1, y1, x2, y2 },
            identity: None, identity_confidence: None,
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            emotion_distribution: None,
            embedding: None,
        }
    }
}

/// A **lean** body observation — the inverted association input (phase 9.5).
///
/// Where [`Detection`] is fat (it carries every enrichment a model could attach),
/// a `BodyObservation` is only what association actually needs: geometry + the
/// detector score (for ByteTrack high/low gating) + the raw landmarks (reserved
/// for the track-owned motion ring in 9.8). The tracker associates these FIRST,
/// returns a stable id per body, and enrichment (SFace/emotion/posture/3D) is
/// applied **after**, per track — so 9.6/9.7 can run those models only on the
/// tracks that need them instead of on every body every frame.
#[derive(Debug, Clone)]
pub struct BodyObservation {
    /// Body bbox, normalised [0..1].
    pub bbox: BBox,
    /// Person-detector confidence (ByteTrack gating). 1.0 for legacy callers.
    pub score: f32,
    /// 33 raw landmarks `[x,y,z,visibility]`, if posed. `None` for a bbox-only
    /// (low-confidence) box. Persisted on the track in 9.8; unused by association.
    pub landmarks: Option<crate::motion::Landmarks33>,
}

impl BodyObservation {
    /// Bbox-only observation (high-confidence by default, no landmarks).
    pub fn bare(bbox: BBox) -> Self {
        Self { bbox, score: 1.0, landmarks: None }
    }
}

/// Track ids below this are reserved for **recognised gallery users** (stable id
/// per person, e.g. Вадим = 0). Anonymous bodies get ephemeral ids ≥ this value,
/// allocated on appearance and freed on exit (phase 8 identity model).
pub const FIRST_EPHEMERAL_ID: u32 = 10;

/// Consecutive cosine-stable face observations after which a track's identity is
/// **locked** (phase 9.6) — the same face vector seen enough times that SFace can
/// stop re-running on it. The headline cadence gate: a locked track skips SFace.
pub const ID_LOCK_FRAMES: u32 = 3;

/// Frames a track coasts on its fused emotion before EmotiEffLib re-runs on it
/// (phase 9.7). Between runs the label is **frozen** (a matched track does not
/// decay), so the cadence saves the classify pass without the label jittering —
/// the temporal-PoE fusion already smooths across the (now sparser) observations.
pub const EMOTION_CADENCE: u32 = 3;

/// Explicit lifecycle state of a track (phase 9.3). Until now the track's status
/// was reasoned out ad-hoc from raw `age`/`missed` at each call site; this names
/// it once as a tested state machine and is the **scaffold for state-gated
/// enrichment** (9.6 cadenced SFace, 9.7 cadenced emotion) and Scene-JSON output.
///
/// Lifecycle: `Tentative → Confirmed → IdLocked` as trust accrues; a gap sends an
/// established track to `Occluded`, a long gap to `Lost`. An unconfirmed track
/// that misses drops straight to `Lost` (likely a false positive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleState {
    /// Just born — too few matches to trust. Dropped on the first miss.
    Tentative,
    /// Matched enough frames to be a real track.
    Confirmed,
    /// Confirmed **and** identity locked (a recognised gallery user) — the
    /// highest-trust state; enrichment can be cadenced hardest here.
    IdLocked,
    /// Was established but is currently unmatched (a short gap) — alive on
    /// inertia, eligible for observation-centric recovery (9.2).
    Occluded,
    /// Unmatched too long — slated for expiry.
    Lost,
}

impl LifecycleState {
    /// Lowercase tag for serialization / Scene JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            LifecycleState::Tentative => "tentative",
            LifecycleState::Confirmed => "confirmed",
            LifecycleState::IdLocked => "id_locked",
            LifecycleState::Occluded => "occluded",
            LifecycleState::Lost => "lost",
        }
    }

    /// True once the track is trustworthy (Confirmed or IdLocked) — the gate for
    /// downstream consumers that should ignore flickering Tentative detections.
    pub fn is_established(self) -> bool {
        matches!(self, LifecycleState::Confirmed | LifecycleState::IdLocked)
    }
}

/// A tracked person — persistent across frames.
#[derive(Debug, Clone)]
pub struct PersonTrack {
    /// Internal stable key for this track's lifetime (always ephemeral, ≥ `FIRST_EPHEMERAL_ID`).
    pub id: u32,
    /// Resolved id shown downstream (phase 8): the user's reserved id (< `FIRST_EPHEMERAL_ID`)
    /// when this track currently holds a recognised user, else the ephemeral `id`.
    /// Recomputed each tick by `PersonTracker::tick`.
    pub display_id: u32,
    pub bbox: BBox,
    /// Kalman filter maintaining a smoothed bbox estimate with velocity model.
    pub kalman: crate::kalman::KalmanBBox,
    pub age: u64,
    pub missed: u64,
    /// Best identity seen for this track (highest confidence).
    pub identity: Option<String>,
    pub identity_confidence: Option<f32>,
    /// Most recent emotion.
    pub emotion: Option<String>,
    pub emotion_confidence: Option<f32>,
    /// Most recent posture.
    pub posture: Option<String>,
    pub posture_confidence: Option<f32>,
    /// Most recent 3D position.
    pub distance: Option<f32>,
    pub position_x: Option<f32>,
    pub position_y: Option<f32>,
    pub position_z: Option<f32>,
    pub ws_confidence: Option<f32>,
    /// Posture quality.
    pub recon_error: Option<f32>,
    pub plausibility: Option<String>,
    /// EMA state for native depth smoothing (phase 6.2). None until first native 3D estimate.
    pub depth_ema: Option<crate::space::DepthEma>,
    /// Latest emotion distribution from Rust model (8 classes). Used by temporal PoE (6.4b).
    pub emotion_distribution: Option<[f32; 8]>,
    /// Latest SFace embedding (phase 7.5). Used for cosine re-link (#6) so two
    /// distinct people are never merged into one id under a single-name gallery.
    pub embedding: Option<Vec<f32>>,
    /// Centre `[cx, cy]` of the most recent **observed** (matched) bbox — the
    /// anchor for observation-centric tracking (OC-SORT, phase 9.2). Distinct
    /// from `bbox` (which is the Kalman-smoothed/inertia estimate): this is the
    /// last raw measurement, so recovery extrapolates from a real observation,
    /// not a drifted filter state. `None` until the first observation.
    pub last_obs_center: Option<[f32; 2]>,
    /// EMA-smoothed **observation-centric velocity** `[vx, vy]` in normalised
    /// units per frame (OC-SORT momentum, phase 9.2). Built from the deltas
    /// between successive *observations* (gap-normalised), not the Kalman
    /// velocity — robust through occlusion. Its *direction* drives the momentum
    /// term in association (OCM) and the virtual trajectory in recovery (OCR).
    /// `None` until two observations exist.
    pub obs_velocity: Option<[f32; 2]>,
    /// Explicit lifecycle state (phase 9.3). Recomputed each tick by the tracker
    /// from `age`/`missed` + identity lock. Born `Tentative`.
    pub lifecycle: LifecycleState,
    /// Consecutive cosine-stable face observations (phase 9.6). Climbs while the
    /// incoming embedding agrees with the track's; resets when a different face
    /// arrives (re-lock on doubt). Drives [`PersonTrack::is_id_locked`].
    pub id_stable_count: u32,
    /// Frames since EmotiEffLib last enriched this track (phase 9.7). Climbs each
    /// tick, resets to 0 when an emotion observation lands. Drives
    /// [`PersonTrack::needs_emotion`] / the emotion cadence gate.
    pub frames_since_emotion: u32,
}

impl PersonTrack {
    /// Create a bare track from an id + bbox: Kalman seeded from the bbox, all
    /// enrichment fields `None`. `display_id` starts equal to `id` (the tracker
    /// resolves it each tick). Convenience for projection/tests; the tracker's
    /// live path builds tracks inline with native-3D priority.
    pub fn new(id: u32, bbox: BBox) -> Self {
        Self {
            id,
            display_id: id,
            bbox,
            kalman: crate::kalman::KalmanBBox::new(bbox),
            age: 1,
            missed: 0,
            identity: None,
            identity_confidence: None,
            emotion: None,
            emotion_confidence: None,
            posture: None,
            posture_confidence: None,
            distance: None,
            position_x: None,
            position_y: None,
            position_z: None,
            ws_confidence: None,
            recon_error: None,
            plausibility: None,
            depth_ema: None,
            emotion_distribution: None,
            embedding: None,
            // Seed the observation anchor from the birth bbox; velocity is unknown
            // until a second observation establishes a direction.
            last_obs_center: Some([(bbox.x1 + bbox.x2) * 0.5, (bbox.y1 + bbox.y2) * 0.5]),
            obs_velocity: None,
            lifecycle: LifecycleState::Tentative,
            id_stable_count: 0,
            frames_since_emotion: u32::MAX, // never enriched → first sight needs emotion
        }
    }

    /// Whether this track's emotion is stale enough to re-run EmotiEffLib (phase
    /// 9.7): it has coasted on its fused label for at least [`EMOTION_CADENCE`]
    /// frames. A fresh track (counter below the cadence) reuses its label and
    /// skips the classify pass.
    pub fn needs_emotion(&self) -> bool {
        self.frames_since_emotion >= EMOTION_CADENCE
    }

    /// Whether this track's identity is **locked** (phase 9.6) — either it holds a
    /// reserved gallery id (a recognised user, already high-trust) or its face
    /// embedding has been cosine-stable for [`ID_LOCK_FRAMES`] observations. A
    /// locked track no longer needs SFace each frame (the cadence perf win); the
    /// re-check cadence and the stability reset still defend it against drift.
    pub fn is_id_locked(&self) -> bool {
        self.display_id < FIRST_EPHEMERAL_ID || self.id_stable_count >= ID_LOCK_FRAMES
    }

    /// Identity to SHOW (phase 8 identity model): the name only when this track
    /// currently holds a reserved user slot (`display_id < FIRST_EPHEMERAL_ID`).
    /// A track that matched a user label but lost the exclusive binding (e.g. the
    /// #5 single-gallery stranger) reads as anonymous — exactly one body can be a
    /// given user at a time.
    pub fn shown_identity(&self) -> Option<&String> {
        if self.display_id < FIRST_EPHEMERAL_ID {
            self.identity.as_ref()
        } else {
            None
        }
    }
}

// ── Scene-graph types (phase 6.4) ─────────────────────────────────────────────

/// Kind of entity in the scene-graph. Object(class) is the extension point for phase 7.
#[derive(Debug, Clone, PartialEq)]
pub enum EntityKind {
    Person,
    Face,
    Hand,
    BodyPart,
    Object(String),
}

/// A scene entity: one node in the 3D scene-graph.
/// Constructed from PersonTrack (face/person) or future detectors (objects).
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: u32,
    pub kind: EntityKind,
    pub bbox: BBox,
    /// Metric 3D position in metres, origin = camera. [x, y, z].
    pub pos_3d: [f32; 3],
    /// Approximate radius in metres derived from bbox + depth. Used for proximity zones.
    pub radius_m: f32,
    /// Confidence of the 3D position estimate (from ws_confidence).
    pub pos_conf: f32,
    // Flat attributes — Attributes struct introduced in phase 7 when more types land.
    pub identity: Option<String>,
    pub emotion: Option<String>,
    pub emotion_distribution: Option<[f32; 8]>,
}

/// Kind of relation between two entities.
#[derive(Debug, Clone, PartialEq)]
pub enum RelationKind {
    /// Closer than the proximity threshold.
    Near,
    /// Bboxes overlap (IoU > 0).
    Overlapping,
    /// One entity is spatially contained within another (future use).
    PartOf,
}

/// A directed edge in the scene-graph.
#[derive(Debug, Clone)]
pub struct Relation {
    pub a: u32,
    pub b: u32,
    pub distance_m: f32,
    pub overlap: f32,
    pub kind: RelationKind,
}

/// The full metric 3D scene-graph for one frame.
/// Entities = nodes with 3D positions; relations = weighted edges;
/// zones = connected components (contextual interaction groups).
#[derive(Debug, Clone)]
pub struct Scene {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    /// Each zone is a list of entity IDs that are spatially connected.
    pub zones: Vec<Vec<u32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BBox ──────────────────────────────────────────────

    #[test]
    fn bbox_area_unit_square() {
        let b = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert!((b.area() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bbox_area_half_frame() {
        let b = BBox { x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.75 };
        assert!((b.area() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn bbox_area_zero_width() {
        let b = BBox { x1: 0.5, y1: 0.0, x2: 0.5, y2: 1.0 };
        assert_eq!(b.area(), 0.0);
    }

    #[test]
    fn bbox_area_inverted() {
        let b = BBox { x1: 0.8, y1: 0.8, x2: 0.2, y2: 0.2 };
        assert_eq!(b.area(), 0.0);
    }

    #[test]
    fn iou_identical() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let b = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert!((a.iou(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_half_overlap() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 0.6, y2: 1.0 };
        let b = BBox { x1: 0.4, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert!((a.iou(&b) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn iou_no_overlap() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 0.4, y2: 1.0 };
        let b = BBox { x1: 0.6, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn iou_one_inside_another() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let b = BBox { x1: 0.2, y1: 0.2, x2: 0.8, y2: 0.8 };
        assert!((a.iou(&b) - 0.36).abs() < 1e-5);
    }

    #[test]
    fn iou_edge_touching() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 0.5, y2: 1.0 };
        let b = BBox { x1: 0.5, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn iou_symmetric() {
        let a = BBox { x1: 0.1, y1: 0.2, x2: 0.7, y2: 0.9 };
        let b = BBox { x1: 0.3, y1: 0.1, x2: 0.9, y2: 0.8 };
        assert_eq!(a.iou(&b), b.iou(&a));
    }
}
