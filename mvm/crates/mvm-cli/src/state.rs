//! State file writer. Produces key: value format compatible with
//! Hermes Sight agent and sight-journal skill.
//!
//! Ported from sightd v0.3 `src/state.rs`.

use anyhow::Result;
use chrono::{FixedOffset, Local};
use mvm_core::PersonTracker;
use mvm_scene::TrackedObject;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

/// Per-face emotion A/B comparison captured before confidence-based fallback.
pub struct EmotionDebugEntry {
    pub id: u32,
    pub rust_emotion: String,
    pub rust_conf: f32,
    pub py_emotion: Option<String>,
    pub py_conf: Option<f32>,
    /// Which source won: "rust", "python", or "none" (rust not run).
    pub winner: &'static str,
}

/// Path to the Sight state file.
const STATE_PATH: &str = ".hermes/profiles/sight/workspace/current_state.md";

#[derive(Serialize)]
struct JsonEmotionDebug {
    rust: String,
    rust_conf: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    py: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    py_conf: Option<f32>,
    winner: &'static str,
}

#[derive(Serialize)]
struct JsonFaceDebug {
    id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    identity_conf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion_conf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    posture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    posture_conf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dist_m: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pos: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion_debug: Option<JsonEmotionDebug>,
}

#[derive(Serialize)]
struct JsonFrameDebug<'a> {
    ts: String,
    fps: f64,
    status: &'a str,
    faces: Vec<JsonFaceDebug>,
}

#[derive(Serialize)]
struct JsonFace {
    id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    identity_conf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion_conf: Option<f32>,
    /// Full 8-class softmax distribution from Rust ONNX model (phase 6.4).
    /// Absent when only Python bridge prediction is available.
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion_distribution: Option<[f32; 8]>,
    /// Face bbox height (normalised 0..1) — diagnostic field for 3D calibration.
    bbox_h: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    posture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    posture_conf: Option<f32>,
    /// Body-visibility regime ("full"/"upper"/"head") — occlusion-aware posture.
    /// With "upper"/"head" the lower body was NOT seen and `posture` is honestly
    /// absent; read `torso_attitude` + fusion activities instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    body_coverage: Option<&'static str>,
    /// 2D torso attitude ("upright"/"inclined"/"horizontal") — flows even when
    /// `posture` is withheld under occlusion.
    #[serde(skip_serializing_if = "Option::is_none")]
    torso_attitude: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dist_m: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pos: Option<[f32; 3]>,
    /// Coarse locomotion state (phase 9.8): "idle" / "moving". Absent until the
    /// motion ring commits a state.
    #[serde(skip_serializing_if = "Option::is_none")]
    motion: Option<&'static str>,
    /// Centroid speed in torso-lengths/sec (phase 9.8).
    #[serde(skip_serializing_if = "Option::is_none")]
    velocity: Option<f32>,
    /// Coarse left-arm gesture off the motion ring ("reaching"/"waving", phase
    /// 10.0). Absent when the arm is resting.
    #[serde(skip_serializing_if = "Option::is_none")]
    left_gesture: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    right_gesture: Option<&'static str>,
    /// Visible wrist anchors, normalised [x,y] (BlazePose 15/16, phase 10.0).
    /// Mesh path only — the sync tracker path carries no landmarks.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    wrists: Vec<[f32; 2]>,
    /// Detected hands: presence + side always; 21 landmarks only under
    /// `--dump-landmarks` (heavy payload, phase 10.0 opt-in).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    hands: Vec<JsonHand>,
    /// Normalised Shannon entropy of the emotion distribution [0,1] — the
    /// pre-baked "how reliable is `emotion`" signal (phase 10.0). 1 ≈ uniform.
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion_entropy: Option<f32>,
    /// Posture VAE reconstruction error — diagnostic (phase 10.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    recon_error: Option<f32>,
    /// Posture OOD plausibility flag — diagnostic (phase 10.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    plausibility: Option<String>,
    /// Coarse head yaw, degrees (phase 10.3): 0 = frontal, − nose toward image
    /// left, + toward image right. Latest face-matched snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    head_yaw: Option<f32>,
    /// Categorical head direction ("camera"/"left"/"right", phase 10.3).
    #[serde(skip_serializing_if = "Option::is_none")]
    head_facing: Option<&'static str>,
}

/// One hand in the Scene JSON (phase 10.0/10.2).
#[derive(Serialize)]
struct JsonHand {
    /// "left"/"right" — the body wrist that drove the ROI (authoritative).
    side: &'static str,
    /// Hand-model presence score [0,1] (vis-downweighted, phase 10.2).
    presence: f32,
    /// Raw model handedness ~[0,1] — disambiguation next to `side` (phase 10.2).
    handedness: f32,
    /// Static gesture ("fist"/"open"/"point"/"pinch"), absent when unclear.
    #[serde(skip_serializing_if = "Option::is_none")]
    gesture: Option<&'static str>,
    /// Normalised 21-point landmarks; present only under `--dump-landmarks`.
    #[serde(skip_serializing_if = "Option::is_none")]
    landmarks: Option<Vec<[f32; 2]>>,
}

#[derive(Serialize)]
struct JsonRelation {
    a: u32,
    b: u32,
    dist_m: f32,
    overlap: f32,
    kind: &'static str,
}

/// One object node in the scene-graph JSON (phase 8.E, G₂ Scene group).
#[derive(Serialize)]
struct JsonObject {
    /// Stable tracked id (≥ OBJECT_ID_BASE) — matches ids in zones/relations and
    /// is the same physical object across frames (phase 8.E filter B).
    id: u32,
    class: String,
    score: f32,
    /// Frames this object has been tracked (temporal persistence).
    age: u32,
    /// Metric 3D position from depth-anchoring (phase 8.E.2); absent without depth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pos: Option<[f32; 3]>,
    /// Normalised [x1,y1,x2,y2] in the original frame.
    bbox: [f32; 4],
}

/// A cross-group activity in the JSON (phase 8.F).
#[derive(Serialize)]
struct JsonActivity {
    label: String,
    confidence: f32,
    participants: Vec<u32>,
}

#[derive(Serialize)]
struct JsonFrame<'a> {
    ts: String,
    fps: f64,
    status: &'a str,
    faces: Vec<JsonFace>,
    /// Object detections (YOLO 80-class) participating in the scene-graph (8.E).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    objects: Vec<JsonObject>,
    /// Relational scene-graph: zones (connected components) and edges (phase 6.4).
    zones: Vec<Vec<u32>>,
    relations: Vec<JsonRelation>,
    /// Cross-group activities (phase 8.F); omitted when empty (e.g. sync path).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    activities: Vec<JsonActivity>,
    /// Scene anomaly score [0,1] (phase 8.F).
    anomaly_score: f32,
}

/// Build scene-graph entities for tracked objects (phase 8.E / 8.E.2).
///
/// When depth-anchoring (phase 8.E.2) has set `pos_3d`, the object carries a metric
/// position and `pos_conf > 0`, so `build_scene` forms real 3D `Near` edges with
/// people and other objects. Without depth it falls back to `[0,0,0]`/`conf 0` —
/// `build_scene` then only relates it by 2D bbox-overlap (the "person ↔ held
/// object" signal), never fabricating a metric distance.
fn object_entities(objects: &[TrackedObject]) -> Vec<mvm_core::types::Entity> {
    use mvm_core::types::{Entity, EntityKind};
    objects
        .iter()
        .map(|o| {
            let (pos_3d, pos_conf) = match o.pos_3d {
                Some(p) => (p, o.pos_conf),
                None => ([0.0, 0.0, 0.0], 0.0),
            };
            Entity {
                id: o.id,
                kind: EntityKind::Object(o.class_name.to_string()),
                bbox: o.bbox,
                pos_3d,
                radius_m: 0.1,
                pos_conf,
                identity: None,
                emotion: None,
                emotion_distribution: None,
            }
        })
        .collect()
}

// ── WorldState projections (phase 8.C.7, mesh StateFileWriter) ────────────────
// The mesh serialises the blackboard (PersonState/ObjectState) instead of a
// PersonTracker, but must emit the *same* Scene JSON contract. These mirror the
// tracker-keyed builders above.

fn person_to_json_face(p: &mvm_core::PersonState, dump_landmarks: bool) -> JsonFace {
    JsonFace {
        id: p.track_id,
        identity: p.identity.as_ref().map(|(n, _)| n.clone()),
        identity_conf: p.identity.as_ref().map(|(_, c)| *c),
        emotion: p.emotion.as_ref().map(|(e, _)| e.clone()),
        emotion_conf: p.emotion.as_ref().map(|(_, c)| *c),
        emotion_distribution: p.emotion_distribution,
        bbox_h: p.bbox.y2 - p.bbox.y1,
        posture: p.posture.as_ref().map(|(s, _)| s.clone()),
        posture_conf: p.posture.as_ref().map(|(_, c)| *c),
        body_coverage: p.body_coverage,
        torso_attitude: p.torso_attitude,
        dist_m: p.position_3d.map(|q| q.z),
        pos: p.position_3d.map(|q| [q.x, q.y, q.z]),
        motion: p.motion_state,
        velocity: p.velocity,
        left_gesture: p.left_gesture,
        right_gesture: p.right_gesture,
        wrists: p.wrists.clone(),
        hands: p.hands.iter().map(|h| hand_to_json(h, dump_landmarks)).collect(),
        emotion_entropy: p.emotion_distribution.map(|d| mvm_core::emotion_entropy(&d)),
        recon_error: p.recon_error,
        plausibility: p.plausibility.clone(),
        head_yaw: p.head_yaw,
        head_facing: p.head_facing,
    }
}

fn hand_to_json(h: &mvm_core::HandState, dump_landmarks: bool) -> JsonHand {
    JsonHand {
        side: h.side,
        presence: h.presence,
        handedness: h.handedness,
        gesture: h.gesture,
        landmarks: dump_landmarks.then(|| h.landmarks.clone()),
    }
}

fn person_to_entity(p: &mvm_core::PersonState) -> mvm_core::types::Entity {
    use mvm_core::types::{Entity, EntityKind};
    let (pos_3d, pos_conf) = match p.position_3d {
        Some(q) => ([q.x, q.y, q.z], q.confidence),
        None => ([0.0, 0.0, 0.0], 0.0),
    };
    Entity {
        id: p.track_id,
        kind: EntityKind::Person,
        bbox: p.bbox,
        pos_3d,
        radius_m: 0.15,
        pos_conf,
        identity: p.identity.as_ref().map(|(n, _)| n.clone()),
        emotion: p.emotion.as_ref().map(|(e, _)| e.clone()),
        emotion_distribution: p.emotion_distribution,
    }
}

fn object_state_to_entity(o: &mvm_core::ObjectState) -> mvm_core::types::Entity {
    use mvm_core::types::{Entity, EntityKind};
    let (pos_3d, pos_conf) = match o.position_3d {
        Some(q) => ([q.x, q.y, q.z], q.confidence),
        None => ([0.0, 0.0, 0.0], 0.0),
    };
    Entity {
        id: o.id,
        kind: EntityKind::Object(o.class.clone()),
        bbox: o.bbox,
        pos_3d,
        radius_m: 0.1,
        pos_conf,
        identity: None,
        emotion: None,
        emotion_distribution: None,
    }
}

fn object_state_to_json(o: &mvm_core::ObjectState) -> JsonObject {
    JsonObject {
        id: o.id,
        class: o.class.clone(),
        score: o.confidence,
        age: o.age,
        pos: o.position_3d.map(|q| [q.x, q.y, q.z]),
        bbox: [o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2],
    }
}

pub struct StateFile {
    path: PathBuf,
    tz: FixedOffset,
    /// Include raw 21-point hand landmarks in the Scene JSON (`--dump-landmarks`,
    /// phase 10.0 opt-in — heavy payload, off by default).
    pub dump_landmarks: bool,
}

impl StateFile {
    /// Production: uses HOME/.hermes/profiles/sight/workspace/current_state.md
    pub fn open() -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let path = PathBuf::from(home).join(STATE_PATH);
        Self::open_at(path)
    }

    /// Test constructor: writes to an arbitrary path.
    pub fn open_at(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tz = FixedOffset::east_opt(3 * 3600).unwrap();
        Ok(Self { path, tz, dump_landmarks: false })
    }

    /// Write current state to file.
    pub fn write(&self, tracker: &PersonTracker, fps: f64, objects: &[TrackedObject]) -> Result<()> {
        let now = Local::now().with_timezone(&self.tz);
        let timestamp = now.format("%Y-%m-%dT%H:%M:%S%:z").to_string();

        let mut active_tracks: Vec<_> = tracker.active().collect();
        active_tracks.sort_by_key(|t| t.display_id);
        let face_count = active_tracks.len();

        let status = if face_count > 0 {
            "occupied"
        } else {
            "empty"
        };

        let dominant_emotion = active_tracks
            .iter()
            .filter_map(|t| t.emotion.as_deref())
            .next()
            .unwrap_or("none");

        let mut lines = vec![
            "# Camera State (mvm v0.4)".to_string(),
            format!("- Last checked: {timestamp}"),
            "- Camera: /dev/video0".to_string(),
            "- Uptime: —".to_string(),
            format!("- FPS: {fps:.1}"),
            format!("- Face count: {face_count}"),
            format!("- Dominant emotion: {dominant_emotion}"),
        ];

        if face_count > 0 {
            let names: Vec<String> = active_tracks
                .iter()
                .map(|t| t.shown_identity().map(|s| s.as_str()).unwrap_or("unknown").to_string())
                .collect();
            lines.push(format!("- Persons visible: {}", names.join(", ")));

            for (i, t) in active_tracks.iter().enumerate() {
                let identity = t.shown_identity().map(|s| s.as_str()).unwrap_or("unknown");
                let id_conf = t
                    .shown_identity()
                    .and(t.identity_confidence)
                    .map(|c| format!(" ({c:.2})"))
                    .unwrap_or_default();
                let emotion = t.emotion.as_deref().unwrap_or("none");
                let emo_conf = t
                    .emotion_confidence
                    .map(|c| format!(" ({c:.2})"))
                    .unwrap_or_default();
                let posture = t.posture.as_deref().unwrap_or("—");
                let pos_conf = t
                    .posture_confidence
                    .map(|c| format!(" ({c:.2})"))
                    .unwrap_or_default();

                lines.push(format!("- Person_{}:", i + 1));
                lines.push(format!("  - id: {}", t.display_id));
                lines.push(format!("  - identity: {identity}{id_conf}"));
                lines.push(format!(
                    "  - bbox: {:.2},{:.2},{:.2},{:.2}",
                    t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2
                ));
                lines.push(format!("  - age: {} frames", t.age));
                lines.push(format!("  - emotion: {emotion}{emo_conf}"));
                lines.push(format!("  - posture: {posture}{pos_conf}"));
                // Occlusion-aware regime: name what the camera actually saw when
                // the lower body is occluded (posture is then honestly "—").
                if let Some(cov) = t.body_coverage {
                    if cov != "full" {
                        let att = t.torso_attitude.unwrap_or("?");
                        lines.push(format!("  - body_coverage: {cov} (torso {att})"));
                    }
                }
                if let Some(d) = t.distance {
                    lines.push(format!("  - distance: {d:.2}"));
                }
                if let (Some(x), Some(y), Some(z)) =
                    (t.position_x, t.position_y, t.position_z)
                {
                    lines.push(format!("  - position: ({x:.2}, {y:.2}, {z:.2})m"));
                }
                if let Some(ws) = t.ws_confidence {
                    lines.push(format!("  - ws_confidence: {ws:.2}"));
                }
                if let Some(re) = t.recon_error {
                    lines.push(format!("  - recon_error: {re:.4}"));
                }
                if let Some(ref pl) = t.plausibility {
                    lines.push(format!("  - plausibility: {pl}"));
                }
            }
        } else {
            lines.push("- Persons visible: none".to_string());
        }

        // Objects (phase 8.E, G₂ Scene group): scene context beyond the person.
        if objects.is_empty() {
            lines.push("- Objects visible: none".to_string());
        } else {
            let summary: Vec<String> = objects
                .iter()
                .map(|o| format!("{} ({:.2})", o.class_name, o.score))
                .collect();
            lines.push(format!("- Objects visible: {}", summary.join(", ")));
            for (i, o) in objects.iter().enumerate() {
                lines.push(format!("- Object_{}:", i + 1));
                lines.push(format!("  - id: {}", o.id));
                lines.push(format!("  - class: {}", o.class_name));
                lines.push(format!("  - score: {:.2}", o.score));
                lines.push(format!("  - age: {} frames", o.age));
                lines.push(format!(
                    "  - bbox: {:.2},{:.2},{:.2},{:.2}",
                    o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2
                ));
            }
        }

        lines.push(format!("- Status: {status}"));
        lines.push(String::new());

        fs::write(&self.path, lines.join("\n"))?;

        eprintln!(
            "  state written: {face_count} faces, {fps:.1} fps"
        );
        Ok(())
    }

    /// Serialise current tracker state to a single JSON line.
    /// Includes relational scene-graph (zones + edges) from mvm-fusion (phase 6.4).
    pub fn to_json_line(&self, tracker: &PersonTracker, fps: f64, objects: &[TrackedObject]) -> String {
        use mvm_core::types::{Entity, EntityKind};

        let now = Local::now().with_timezone(&self.tz);
        let ts = now.format("%Y-%m-%dT%H:%M:%S%:z").to_string();

        let mut active: Vec<_> = tracker.active().collect();
        active.sort_by_key(|t| t.display_id);
        let status = if active.is_empty() { "empty" } else { "occupied" };

        // Build entities for the scene-graph. Entities without a 3D position use (0,0,0)
        // with pos_conf=0 so they still participate in 2D overlap relations.
        let mut entities: Vec<Entity> = active.iter().map(|t| {
            let pos_3d = match (t.position_x, t.position_y, t.position_z) {
                (Some(x), Some(y), Some(z)) => [x, y, z],
                _ => [0.0, 0.0, 0.0],
            };
            Entity {
                id: t.display_id,
                kind: EntityKind::Person,
                bbox: t.bbox,
                pos_3d,
                radius_m: 0.15, // rough head radius
                pos_conf: t.ws_confidence.unwrap_or(0.0),
                identity: t.shown_identity().cloned(),
                emotion: t.emotion.clone(),
                emotion_distribution: t.emotion_distribution,
            }
        }).collect();

        // Object nodes (phase 8.E): fed into the same object-agnostic graph so
        // person↔object overlap relations ("context outward from the human") emerge.
        entities.extend(object_entities(objects));

        let json_objects: Vec<JsonObject> = objects.iter().map(|o| JsonObject {
            id: o.id,
            class: o.class_name.to_string(),
            score: o.score,
            age: o.age,
            pos: o.pos_3d,
            bbox: [o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2],
        }).collect();

        let scene = mvm_fusion::build_scene(entities);

        let relations: Vec<JsonRelation> = scene.relations.iter().map(|r| {
            use mvm_core::types::RelationKind;
            let kind = match r.kind {
                RelationKind::Near        => "near",
                RelationKind::Overlapping => "overlapping",
                RelationKind::PartOf      => "part_of",
            };
            JsonRelation { a: r.a, b: r.b, dist_m: r.distance_m, overlap: r.overlap, kind }
        }).collect();

        let faces: Vec<JsonFace> = active.iter().map(|t| {
            let pos = match (t.position_x, t.position_y, t.position_z) {
                (Some(x), Some(y), Some(z)) => Some([x, y, z]),
                _ => None,
            };
            JsonFace {
                id: t.display_id,
                identity: t.shown_identity().cloned(),
                identity_conf: t.shown_identity().and(t.identity_confidence),
                emotion: t.emotion.clone(),
                emotion_conf: t.emotion_confidence,
                emotion_distribution: t.emotion_distribution,
                posture: t.posture.clone(),
                posture_conf: t.posture_confidence,
                body_coverage: t.body_coverage,
                torso_attitude: t.torso_attitude,
                dist_m: t.distance,
                pos,
                bbox_h: t.bbox.y2 - t.bbox.y1,
                // Motion straight off the track (phase 9.8) — skip the startup Unknown.
                motion: match t.motion.state() {
                    mvm_core::motion::MotionState::Unknown => None,
                    s => Some(s.as_str()),
                },
                velocity: if t.motion.is_empty() { None } else { Some(t.motion.speed()) },
                // Phase 10.0: emit what the track already computes. A resting arm
                // maps to an absent field, mirroring PersonState::from_track.
                left_gesture: match t.motion.left_gesture() {
                    mvm_core::motion::ArmGesture::None => None,
                    g => Some(g.as_str()),
                },
                right_gesture: match t.motion.right_gesture() {
                    mvm_core::motion::ArmGesture::None => None,
                    g => Some(g.as_str()),
                },
                // Landmarks live on the mesh path only; the sync tracker has none.
                wrists: Vec::new(),
                hands: Vec::new(),
                emotion_entropy: t.emotion_distribution.map(|d| mvm_core::emotion_entropy(&d)),
                recon_error: t.recon_error,
                plausibility: t.plausibility.clone(),
                head_yaw: t.head_yaw,
                head_facing: t.head_facing,
            }
        }).collect();

        let frame = JsonFrame {
            ts, fps, status, faces, objects: json_objects, zones: scene.zones, relations,
            activities: Vec::new(), // sync path computes no activities
            anomaly_score: 0.0,
        };
        serde_json::to_string(&frame).unwrap_or_else(|_| "{}".to_string())
    }

    /// Serialise a `WorldState` snapshot to the Scene JSON line (phase 8.C.7).
    /// Same contract as [`to_json_line`](Self::to_json_line) — faces + objects +
    /// zones + relations — but sourced from the blackboard the agents fill.
    pub fn to_json_line_world(&self, world: &mvm_core::WorldState, fps: f64) -> String {
        let now = Local::now().with_timezone(&self.tz);
        let ts = now.format("%Y-%m-%dT%H:%M:%S%:z").to_string();
        let status = if world.people.is_empty() { "empty" } else { "occupied" };

        // Entities for the relational graph (people + objects, object-agnostic).
        let mut entities: Vec<mvm_core::types::Entity> =
            world.people.iter().map(person_to_entity).collect();
        entities.extend(world.objects.iter().map(object_state_to_entity));

        let faces: Vec<JsonFace> = world
            .people
            .iter()
            .map(|p| person_to_json_face(p, self.dump_landmarks))
            .collect();
        let json_objects: Vec<JsonObject> = world.objects.iter().map(object_state_to_json).collect();

        let scene = mvm_fusion::build_scene(entities);
        let relations: Vec<JsonRelation> = scene.relations.iter().map(|r| {
            use mvm_core::types::RelationKind;
            let kind = match r.kind {
                RelationKind::Near => "near",
                RelationKind::Overlapping => "overlapping",
                RelationKind::PartOf => "part_of",
            };
            JsonRelation { a: r.a, b: r.b, dist_m: r.distance_m, overlap: r.overlap, kind }
        }).collect();

        let activities: Vec<JsonActivity> = world
            .activities
            .iter()
            .map(|a| JsonActivity {
                label: a.label.clone(),
                confidence: a.confidence,
                participants: a.participants.clone(),
            })
            .collect();

        let frame = JsonFrame {
            ts, fps, status, faces, objects: json_objects, zones: scene.zones, relations,
            activities,
            anomaly_score: world.anomaly_score,
        };
        serde_json::to_string(&frame).unwrap_or_else(|_| "{}".to_string())
    }

    /// Write a `WorldState` snapshot to the Hermes StateFile (phase 8.C.7) — the
    /// markdown twin of [`write`](Self::write), sourced from the blackboard.
    pub fn write_world(&self, world: &mvm_core::WorldState, fps: f64) -> Result<()> {
        let now = Local::now().with_timezone(&self.tz);
        let timestamp = now.format("%Y-%m-%dT%H:%M:%S%:z").to_string();

        let mut people: Vec<&mvm_core::PersonState> = world.people.iter().collect();
        people.sort_by_key(|p| p.track_id);
        let face_count = people.len();
        let status = if face_count > 0 { "occupied" } else { "empty" };
        let dominant_emotion = people
            .iter()
            .filter_map(|p| p.emotion.as_ref().map(|(e, _)| e.as_str()))
            .next()
            .unwrap_or("none");

        let mut lines = vec![
            "# Camera State (mvm v0.4)".to_string(),
            format!("- Last checked: {timestamp}"),
            "- Camera: /dev/video0".to_string(),
            "- Uptime: —".to_string(),
            format!("- FPS: {fps:.1}"),
            format!("- Face count: {face_count}"),
            format!("- Dominant emotion: {dominant_emotion}"),
        ];

        if face_count > 0 {
            let names: Vec<String> = people
                .iter()
                .map(|p| p.identity.as_ref().map(|(n, _)| n.clone()).unwrap_or_else(|| "unknown".into()))
                .collect();
            lines.push(format!("- Persons visible: {}", names.join(", ")));
            for (i, p) in people.iter().enumerate() {
                let identity = p.identity.as_ref().map(|(n, _)| n.as_str()).unwrap_or("unknown");
                let id_conf = p.identity.as_ref().map(|(_, c)| format!(" ({c:.2})")).unwrap_or_default();
                let emotion = p.emotion.as_ref().map(|(e, _)| e.as_str()).unwrap_or("none");
                let emo_conf = p.emotion.as_ref().map(|(_, c)| format!(" ({c:.2})")).unwrap_or_default();
                let posture = p.posture.as_ref().map(|(s, _)| s.as_str()).unwrap_or("—");
                let pos_conf = p.posture.as_ref().map(|(_, c)| format!(" ({c:.2})")).unwrap_or_default();
                lines.push(format!("- Person_{}:", i + 1));
                lines.push(format!("  - id: {}", p.track_id));
                lines.push(format!("  - identity: {identity}{id_conf}"));
                lines.push(format!(
                    "  - bbox: {:.2},{:.2},{:.2},{:.2}",
                    p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2
                ));
                lines.push(format!("  - emotion: {emotion}{emo_conf}"));
                lines.push(format!("  - posture: {posture}{pos_conf}"));
                // Occlusion-aware regime (mirrors the sync-path StateFile).
                if let Some(cov) = p.body_coverage {
                    if cov != "full" {
                        let att = p.torso_attitude.unwrap_or("?");
                        lines.push(format!("  - body_coverage: {cov} (torso {att})"));
                    }
                }
                if let Some(q) = p.position_3d {
                    lines.push(format!("  - distance: {:.2}", q.z));
                    lines.push(format!("  - position: ({:.2}, {:.2}, {:.2})m", q.x, q.y, q.z));
                    lines.push(format!("  - ws_confidence: {:.2}", q.confidence));
                }
            }
        } else {
            lines.push("- Persons visible: none".to_string());
        }

        if world.objects.is_empty() {
            lines.push("- Objects visible: none".to_string());
        } else {
            let summary: Vec<String> = world
                .objects
                .iter()
                .map(|o| format!("{} ({:.2})", o.class, o.confidence))
                .collect();
            lines.push(format!("- Objects visible: {}", summary.join(", ")));
            for (i, o) in world.objects.iter().enumerate() {
                lines.push(format!("- Object_{}:", i + 1));
                lines.push(format!("  - id: {}", o.id));
                lines.push(format!("  - class: {}", o.class));
                lines.push(format!("  - score: {:.2}", o.confidence));
                lines.push(format!("  - age: {} frames", o.age));
                lines.push(format!(
                    "  - bbox: {:.2},{:.2},{:.2},{:.2}",
                    o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2
                ));
            }
        }

        // Cross-group activities + anomaly (phase 8.F).
        if world.activities.is_empty() {
            lines.push("- Activities: none".to_string());
        } else {
            let summary: Vec<String> = world
                .activities
                .iter()
                .map(|a| format!("{} ({:.2})", a.label, a.confidence))
                .collect();
            lines.push(format!("- Activities: {}", summary.join(", ")));
        }
        lines.push(format!("- Anomaly: {:.2}", world.anomaly_score));

        lines.push(format!("- Status: {status}"));
        lines.push(String::new());
        fs::write(&self.path, lines.join("\n"))?;
        Ok(())
    }

    /// Print a formatted table of all tracked persons to stderr.
    pub fn dump_table(&self, tracker: &PersonTracker, fps: f64) {
        let now = Local::now().with_timezone(&self.tz);
        let ts = now.format("%Y-%m-%dT%H:%M:%S%:z");

        let active: Vec<_> = tracker.active().collect();
        let face_count = active.len();
        let status = if face_count > 0 { "occupied" } else { "empty" };

        eprintln!();
        eprintln!("  ╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("  ║  MVM Rust Pipeline — Frame Analysis                               ║");
        eprintln!("  ╠══════════════════════════════════════════════════════════════════════╣");
        eprintln!("  ║  Time: {ts} | FPS: {fps:.1} | Status: {status:<8} | Faces: {face_count:<3}         ║", 
            ts = ts, fps = fps, status = status, face_count = face_count);
        eprintln!("  ╠══════════════════════════════════════════════════════════════════════╣");

        if face_count > 0 {
            eprintln!("  ║ {:>3} │ {:<12} │ {:>5} │ {:<12} │ {:>5} │ {:<13} │ {:>5} │ {:>6} │ {:>18} ║",
                "ID", "Identity", "Conf", "Emotion", "Conf", "Posture", "Conf", "Dist,m", "Position (x,y,z)m");
            eprintln!("  ╟─────┼──────────────┼───────┼──────────────┼───────┼───────────────┼───────┼────────┼────────────────────╢");

            for t in &active {
                let id = t.display_id;
                let identity = t.shown_identity().map(|s| s.as_str()).unwrap_or("—");
                let id_conf = t.shown_identity().and(t.identity_confidence).map(|c| format!("{c:.2}")).unwrap_or_else(|| "—".into());
                let emotion = t.emotion.as_deref().unwrap_or("—");
                let emo_conf = t.emotion_confidence.map(|c| format!("{c:.2}")).unwrap_or_else(|| "—".into());
                let posture = t.posture.as_deref().unwrap_or("—");
                let pos_conf = t.posture_confidence.map(|c| format!("{c:.2}")).unwrap_or_else(|| "—".into());
                let dist = t.distance.map(|d| format!("{d:.2}")).unwrap_or_else(|| "—".into());
                let position = match (t.position_x, t.position_y, t.position_z) {
                    (Some(x), Some(y), Some(z)) => format!("({x:.2}, {y:.2}, {z:.2})"),
                    _ => "—".into(),
                };

                eprintln!("  ║ {:>3} │ {:<12} │ {:>5} │ {:<12} │ {:>5} │ {:<13} │ {:>5} │ {:>6} │ {:>18} ║",
                    id, identity, id_conf, emotion, emo_conf, posture, pos_conf, dist, position);
            }
        } else {
            eprintln!("  ║  No faces detected                                                ║");
        }

        eprintln!("  ╚══════════════════════════════════════════════════════════════════════╝");
        eprintln!();
    }

    /// Like `dump_table`, but appends an emotion A/B section when entries are present.
    pub fn dump_table_debug(&self, tracker: &PersonTracker, fps: f64, debug: &[EmotionDebugEntry]) {
        self.dump_table(tracker, fps);
        if debug.is_empty() {
            return;
        }
        eprintln!("  ╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("  ║  Emotion A/B Debug (EmotiEffLib=Rust  HSEmotion=Python)           ║");
        eprintln!("  ╠══════════════════════════════════════════════════════════════════════╣");
        eprintln!("  ║ {:>3} │ {:<12} │ {:>5} │ {:<12} │ {:>5} │ {:>6} ║",
            "ID", "Rust Emo", "Conf", "Py Emo", "Conf", "Winner");
        eprintln!("  ╟─────┼──────────────┼───────┼──────────────┼───────┼────────╢");
        for e in debug {
            let py_emo  = e.py_emotion.as_deref().unwrap_or("—");
            let py_conf = e.py_conf.map(|c| format!("{c:.2}")).unwrap_or_else(|| "—".into());
            eprintln!("  ║ {:>3} │ {:<12} │ {:>5.2} │ {:<12} │ {:>5} │ {:>6} ║",
                e.id, e.rust_emotion, e.rust_conf, py_emo, py_conf, e.winner);
        }
        eprintln!("  ╚══════════════════════════════════════════════════════════════════════╝");
        eprintln!();
    }

    /// Like `to_json_line`, but adds `emotion_debug` object to each face with a debug entry.
    pub fn to_json_line_debug(&self, tracker: &PersonTracker, fps: f64, debug: &[EmotionDebugEntry]) -> String {
        let now = Local::now().with_timezone(&self.tz);
        let ts = now.format("%Y-%m-%dT%H:%M:%S%:z").to_string();

        let mut active: Vec<_> = tracker.active().collect();
        active.sort_by_key(|t| t.display_id);
        let status = if active.is_empty() { "empty" } else { "occupied" };

        let faces: Vec<JsonFaceDebug> = active.iter().map(|t| {
            let pos = match (t.position_x, t.position_y, t.position_z) {
                (Some(x), Some(y), Some(z)) => Some([x, y, z]),
                _ => None,
            };
            let emotion_debug = debug.iter().find(|e| e.id == t.display_id).map(|e| JsonEmotionDebug {
                rust: e.rust_emotion.clone(),
                rust_conf: e.rust_conf,
                py: e.py_emotion.clone(),
                py_conf: e.py_conf,
                winner: e.winner,
            });
            JsonFaceDebug {
                id: t.display_id,
                identity: t.shown_identity().cloned(),
                identity_conf: t.shown_identity().and(t.identity_confidence),
                emotion: t.emotion.clone(),
                emotion_conf: t.emotion_confidence,
                posture: t.posture.clone(),
                posture_conf: t.posture_confidence,
                dist_m: t.distance,
                pos,
                emotion_debug,
            }
        }).collect();

        let frame = JsonFrameDebug { ts, fps, status, faces };
        serde_json::to_string(&frame).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{Detection, PersonTracker};

    fn temp_state_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("mvm_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn read_state(path: &PathBuf) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    fn det(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection::bare(x1, y1, x2, y2)
    }

    #[test]
    fn state_empty_room() {
        let path = temp_state_path("test_empty.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 30.0, &[]).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Status: empty"));
        assert!(content.contains("- Face count: 0"));
        assert!(content.contains("- Dominant emotion: none"));
        assert!(content.contains("- Persons visible: none"));
        assert!(!content.contains("Person_1:"));
    }

    #[test]
    fn state_enriched_person() {
        let path = temp_state_path("test_rich.md");
        let sf = StateFile::open_at(path.clone()).unwrap();

        let mut tracker = PersonTracker::new();
        tracker.register_user("Вадим");
        tracker.tick(&[Detection {
            bbox: mvm_core::BBox {
                x1: 0.2,
                y1: 0.3,
                x2: 0.5,
                y2: 0.7,
            },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: Some("Neutral".into()),
            emotion_confidence: Some(0.65),
            posture: Some("сидит".into()),
            posture_confidence: Some(0.95),
            distance: None,
            position_x: None,
            position_y: None,
            position_z: None,
            ws_confidence: None,
            recon_error: None,
            plausibility: None,
            body_coverage: None,
            torso_attitude: None,
            emotion_distribution: None, embedding: None,
 head_yaw: None, head_facing: None,
        }]);

        sf.write(&tracker, 25.5, &[]).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Status: occupied"));
        assert!(content.contains("- Dominant emotion: Neutral"));
        assert!(content.contains("- Persons visible: Вадим"));
        assert!(content.contains("  - identity: Вадим (0.72)"));
        assert!(content.contains("  - bbox: 0.20,0.30,0.50,0.70"));
        assert!(content.contains("  - emotion: Neutral (0.65)"));
        assert!(content.contains("  - posture: сидит (0.95)"));
    }

    #[test]
    fn state_unknown_person() {
        let path = temp_state_path("test_unknown.md");
        let sf = StateFile::open_at(path.clone()).unwrap();

        let mut tracker = PersonTracker::new();
        tracker.tick(&[det(0.2, 0.3, 0.5, 0.7)]);

        sf.write(&tracker, 30.0, &[]).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Persons visible: unknown"));
        assert!(content.contains("  - identity: unknown"));
        assert!(content.contains("  - emotion: none"));
        assert!(content.contains("  - posture: —"));
    }

    #[test]
    fn state_multiple_people() {
        let path = temp_state_path("test_multi.md");
        let sf = StateFile::open_at(path.clone()).unwrap();

        let mut tracker = PersonTracker::new();
        tracker.register_user("Вадим");
        tracker.register_user("Даня");
        tracker.tick(&[
            Detection {
                bbox: mvm_core::BBox {
                    x1: 0.1,
                    y1: 0.1,
                    x2: 0.3,
                    y2: 0.5,
                },
                identity: Some("Вадим".into()),
                identity_confidence: Some(0.8),
                emotion: Some("Neutral".into()),
                emotion_confidence: Some(0.6),
                posture: Some("сидит".into()),
                posture_confidence: Some(0.9),
                distance: None,
                position_x: None,
                position_y: None,
                position_z: None,
                ws_confidence: None,
                recon_error: None,
                plausibility: None,
                body_coverage: None,
                torso_attitude: None,
                emotion_distribution: None, embedding: None,
 head_yaw: None, head_facing: None,
            },
            Detection {
                bbox: mvm_core::BBox {
                    x1: 0.6,
                    y1: 0.2,
                    x2: 0.9,
                    y2: 0.7,
                },
                identity: Some("Даня".into()),
                identity_confidence: Some(0.75),
                emotion: Some("Happiness".into()),
                emotion_confidence: Some(0.7),
                posture: Some("стоит".into()),
                posture_confidence: Some(0.85),
                distance: None,
                position_x: None,
                position_y: None,
                position_z: None,
                ws_confidence: None,
                recon_error: None,
                plausibility: None,
                body_coverage: None,
                torso_attitude: None,
                emotion_distribution: None, embedding: None,
 head_yaw: None, head_facing: None,
            },
        ]);

        sf.write(&tracker, 29.9, &[]).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Face count: 2"));
        assert!(content.contains("- Persons visible: Вадим, Даня"));
        assert!(content.contains("  - identity: Вадим (0.80)"));
        assert!(content.contains("  - identity: Даня (0.75)"));
        assert!(content.contains("  - emotion: Happiness (0.70)"));
        assert!(content.contains("  - posture: стоит (0.85)"));
    }

    #[test]
    fn state_timestamp_is_iso8601() {
        let path = temp_state_path("test_ts.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 0.0, &[]).unwrap();

        let content = read_state(&path);
        let ts_line = content
            .lines()
            .find(|l| l.starts_with("- Last checked:"))
            .unwrap();
        let ts = ts_line.strip_prefix("- Last checked: ").unwrap();
        assert!(ts.contains('T'));
        assert!(ts.ends_with("+03:00") || ts.ends_with("+0300"));
    }

    #[test]
    fn state_file_creates_parent_dirs() {
        let path = std::env::temp_dir()
            .join("mvm_test_nested")
            .join("a")
            .join("b")
            .join("test_nested.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 0.0, &[]).unwrap();
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(
            path.parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap(),
        );
    }

    #[test]
    fn state_header_present() {
        let path = temp_state_path("test_header.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 15.0, &[]).unwrap();

        let content = read_state(&path);
        assert!(content.contains("Camera: /dev/video0"));
    }

    #[test]
    fn json_line_is_valid_json() {
        let sf = StateFile::open_at(temp_state_path("test_json.md")).unwrap();
        let tracker = PersonTracker::new();
        let line = sf.to_json_line(&tracker, 30.0, &[]);
        let v: serde_json::Value = serde_json::from_str(&line).expect("must be valid JSON");
        assert!(v["ts"].is_string());
        assert!(v["fps"].is_number());
        assert!(v["status"].is_string());
        assert!(v["faces"].is_array());
    }

    #[test]
    fn json_line_contains_face_fields() {
        let sf = StateFile::open_at(temp_state_path("test_json_face.md")).unwrap();
        let mut tracker = PersonTracker::new();
        tracker.register_user("Вадим");
        tracker.tick(&[Detection {
            bbox: mvm_core::BBox { x1: 0.2, y1: 0.3, x2: 0.5, y2: 0.7 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.69),
            emotion: Some("Happiness".into()),
            emotion_confidence: Some(1.0),
            posture: Some("сидит_за_ПК".into()),
            posture_confidence: Some(0.89),
            distance: Some(2.75),
            position_x: Some(0.75),
            position_y: Some(0.36),
            position_z: Some(2.75),
            ws_confidence: Some(0.9),
            recon_error: None,
            plausibility: None,
            body_coverage: None,
            torso_attitude: None,
            emotion_distribution: None, embedding: None,
 head_yaw: None, head_facing: None,
        }]);
        let line = sf.to_json_line(&tracker, 29.9, &[]);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let face = &v["faces"][0];
        assert_eq!(face["identity"], "Вадим");
        assert_eq!(face["emotion"], "Happiness");
        assert_eq!(face["posture"], "сидит_за_ПК");
        assert!((face["identity_conf"].as_f64().unwrap() - 0.69).abs() < 0.01);
        assert!((face["emotion_conf"].as_f64().unwrap() - 1.0).abs() < 0.01);
        assert!((face["dist_m"].as_f64().unwrap() - 2.75).abs() < 0.01);
        let pos = &face["pos"];
        assert!(pos.is_array());
        assert_eq!(pos.as_array().unwrap().len(), 3);
    }

    #[test]
    fn json_line_empty_room() {
        let sf = StateFile::open_at(temp_state_path("test_json_empty.md")).unwrap();
        let tracker = PersonTracker::new();
        let line = sf.to_json_line(&tracker, 0.0, &[]);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["status"], "empty");
        assert_eq!(v["faces"].as_array().unwrap().len(), 0);
    }

    // ── Object scene-nodes (phase 8.E) ──────────────────────────────────────

    fn obj(id_offset: u32, class_name: &'static str, x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> TrackedObject {
        TrackedObject {
            id: mvm_scene::OBJECT_ID_BASE + id_offset,
            class_name,
            bbox: mvm_core::BBox { x1, y1, x2, y2 },
            score,
            age: 1,
            missed: 0,
            pos_3d: None,
            pos_conf: 0.0,
        }
    }

    #[test]
    fn json_line_lists_object_nodes() {
        let sf = StateFile::open_at(temp_state_path("test_json_obj.md")).unwrap();
        let tracker = PersonTracker::new();
        let objects = vec![obj(0, "cup", 0.4, 0.4, 0.5, 0.55, 0.81)];
        let line = sf.to_json_line(&tracker, 30.0, &objects);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let objs = v["objects"].as_array().expect("objects array present");
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0]["class"], "cup");
        assert_eq!(objs[0]["id"].as_u64().unwrap(), mvm_scene::OBJECT_ID_BASE as u64);
        assert!((objs[0]["score"].as_f64().unwrap() - 0.81).abs() < 0.01);
    }

    #[test]
    fn json_line_omits_objects_when_none() {
        let sf = StateFile::open_at(temp_state_path("test_json_noobj.md")).unwrap();
        let tracker = PersonTracker::new();
        let line = sf.to_json_line(&tracker, 30.0, &[]);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        // skip_serializing_if = empty → field omitted entirely.
        assert!(v["objects"].is_null());
    }

    #[test]
    fn person_overlapping_object_gets_relation() {
        let sf = StateFile::open_at(temp_state_path("test_json_rel.md")).unwrap();
        let mut tracker = PersonTracker::new();
        // Anonymous body → ephemeral id 10 (FIRST_EPHEMERAL_ID).
        tracker.tick(&[det(0.3, 0.3, 0.6, 0.8)]);
        let pid = tracker.active().next().unwrap().display_id;
        // A cup whose bbox overlaps the body bbox → "overlapping" edge (no 3D needed).
        let oid = mvm_scene::OBJECT_ID_BASE;
        let objects = vec![obj(0, "cup", 0.4, 0.4, 0.55, 0.6, 0.8)];
        let line = sf.to_json_line(&tracker, 30.0, &objects);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let rels = v["relations"].as_array().unwrap();
        let edge = rels.iter().find(|r| {
            let a = r["a"].as_u64().unwrap() as u32;
            let b = r["b"].as_u64().unwrap() as u32;
            (a == pid && b == oid) || (a == oid && b == pid)
        });
        assert!(edge.is_some(), "expected person↔object overlap edge, got {rels:?}");
        assert_eq!(edge.unwrap()["kind"], "overlapping");
    }

    #[test]
    fn markdown_lists_objects() {
        let sf = StateFile::open_at(temp_state_path("test_md_obj.md")).unwrap();
        let tracker = PersonTracker::new();
        let objects = vec![obj(0, "chair", 0.1, 0.1, 0.4, 0.9, 0.77)];
        sf.write(&tracker, 30.0, &objects).unwrap();
        let content = read_state(&sf.path);
        assert!(content.contains("Objects visible: chair"), "got:\n{content}");
        assert!(content.contains("class: chair"));
    }

    #[test]
    fn json_debug_contains_both_emotions() {
        let sf = StateFile::open_at(temp_state_path("test_json_debug.md")).unwrap();
        let mut tracker = PersonTracker::new();
        tracker.register_user("Вадим");
        tracker.tick(&[Detection {
            bbox: mvm_core::BBox { x1: 0.2, y1: 0.3, x2: 0.5, y2: 0.7 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.75),
            emotion: Some("Happiness".into()),
            emotion_confidence: Some(0.80),
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            body_coverage: None, torso_attitude: None, emotion_distribution: None, embedding: None,
 head_yaw: None, head_facing: None,
        }]);
        let debug = vec![EmotionDebugEntry {
            id: 0,
            rust_emotion: "Happiness".into(),
            rust_conf: 0.80,
            py_emotion: Some("Neutral".into()),
            py_conf: Some(0.55),
            winner: "rust",
        }];
        let line = sf.to_json_line_debug(&tracker, 29.9, &debug);
        let v: serde_json::Value = serde_json::from_str(&line).expect("must be valid JSON");
        let face = &v["faces"][0];
        assert_eq!(face["emotion"], "Happiness");
        let dbg = &face["emotion_debug"];
        assert_eq!(dbg["rust"], "Happiness");
        assert!((dbg["rust_conf"].as_f64().unwrap() - 0.80).abs() < 0.01);
        assert_eq!(dbg["py"], "Neutral");
        assert!((dbg["py_conf"].as_f64().unwrap() - 0.55).abs() < 0.01);
        assert_eq!(dbg["winner"], "rust");
    }

    #[test]
    fn json_debug_no_entries_omits_emotion_debug_field() {
        let sf = StateFile::open_at(temp_state_path("test_json_nodebug.md")).unwrap();
        let mut tracker = PersonTracker::new();
        tracker.tick(&[Detection {
            bbox: mvm_core::BBox { x1: 0.2, y1: 0.3, x2: 0.5, y2: 0.7 },
            identity: None, identity_confidence: None,
            emotion: Some("Sadness".into()), emotion_confidence: Some(0.6),
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            body_coverage: None, torso_attitude: None, emotion_distribution: None, embedding: None,
 head_yaw: None, head_facing: None,
        }]);
        let line = sf.to_json_line_debug(&tracker, 29.9, &[]);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert!(v["faces"][0]["emotion_debug"].is_null());
    }

    // ── WorldState serialisation (phase 8.C.7, mesh StateFileWriter) ─────────

    fn person_state(id: u32, bbox: mvm_core::BBox) -> mvm_core::PersonState {
        mvm_core::PersonState {
            track_id: id,
            bbox,
            identity: None,
            emotion: None,
            posture: None,
            position_3d: None,
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
            head_yaw: None, head_facing: None,
            last_seen: std::time::Instant::now(),
        }
    }

    fn object_state(id: u32, class: &str, bbox: mvm_core::BBox) -> mvm_core::ObjectState {
        mvm_core::ObjectState {
            id,
            class: class.into(),
            bbox,
            confidence: 0.8,
            age: 2,
            position_3d: None,
            last_seen: std::time::Instant::now(),
        }
    }

    #[test]
    fn world_json_valid_and_empty_status() {
        let sf = StateFile::open_at(temp_state_path("test_world_empty.md")).unwrap();
        let world = mvm_core::WorldState::new(std::time::Instant::now());
        let line = sf.to_json_line_world(&world, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).expect("valid JSON");
        assert_eq!(v["status"], "empty");
        assert_eq!(v["faces"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn world_json_person_object_overlap_relation() {
        let sf = StateFile::open_at(temp_state_path("test_world_rel.md")).unwrap();
        let mut world = mvm_core::WorldState::new(std::time::Instant::now());
        world.people.push(person_state(10, mvm_core::BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 }));
        world.objects.push(object_state(
            mvm_scene::OBJECT_ID_BASE,
            "cup",
            mvm_core::BBox { x1: 0.4, y1: 0.4, x2: 0.55, y2: 0.6 },
        ));
        let line = sf.to_json_line_world(&world, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["status"], "occupied");
        assert_eq!(v["objects"][0]["class"], "cup");
        assert_eq!(v["objects"][0]["age"].as_u64().unwrap(), 2);
        let rels = v["relations"].as_array().unwrap();
        let oid = mvm_scene::OBJECT_ID_BASE;
        let edge = rels.iter().find(|r| {
            let a = r["a"].as_u64().unwrap() as u32;
            let b = r["b"].as_u64().unwrap() as u32;
            (a == 10 && b == oid) || (a == oid && b == 10)
        });
        assert_eq!(edge.expect("person↔object edge")["kind"], "overlapping");
    }

    #[test]
    fn world_md_lists_person_and_object() {
        let sf = StateFile::open_at(temp_state_path("test_world_md.md")).unwrap();
        let mut world = mvm_core::WorldState::new(std::time::Instant::now());
        world.people.push(person_state(10, mvm_core::BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 }));
        world.objects.push(object_state(
            mvm_scene::OBJECT_ID_BASE,
            "chair",
            mvm_core::BBox { x1: 0.1, y1: 0.1, x2: 0.4, y2: 0.9 },
        ));
        sf.write_world(&world, 25.0).unwrap();
        let content = read_state(&sf.path);
        assert!(content.contains("- Status: occupied"), "got:\n{content}");
        assert!(content.contains("Objects visible: chair"));
        assert!(content.contains("  - id: 10"));
        assert!(content.contains("class: chair"));
    }

    #[test]
    fn world_json_surfaces_motion() {
        // Phase 9.8 contract: a person's motion_state/velocity reach the Scene JSON.
        let sf = StateFile::open_at(temp_state_path("test_world_motion.md")).unwrap();
        let mut world = mvm_core::WorldState::new(std::time::Instant::now());
        let mut p = person_state(0, mvm_core::BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 });
        p.motion_state = Some("moving");
        p.velocity = Some(0.8);
        world.people.push(p);
        let line = sf.to_json_line_world(&world, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["faces"][0]["motion"], "moving");
        assert!((v["faces"][0]["velocity"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn world_json_omits_motion_when_absent() {
        // No motion committed → the fields are skipped (contract unchanged).
        let sf = StateFile::open_at(temp_state_path("test_world_nomotion.md")).unwrap();
        let mut world = mvm_core::WorldState::new(std::time::Instant::now());
        world.people.push(person_state(0, mvm_core::BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 }));
        let line = sf.to_json_line_world(&world, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert!(v["faces"][0].get("motion").is_none(), "motion omitted when absent");
    }

    // ── Phase 10.0: emit what we already compute ─────────────────────────────

    #[test]
    fn world_json_surfaces_gestures_hands_and_posture_quality() {
        let sf = StateFile::open_at(temp_state_path("test_world_p10.md")).unwrap();
        let mut world = mvm_core::WorldState::new(std::time::Instant::now());
        let mut p = person_state(0, mvm_core::BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 });
        p.left_gesture = Some("waving");
        p.wrists = vec![[0.4, 0.6]];
        p.hands = vec![mvm_core::HandState {
            landmarks: vec![[0.41, 0.61]; 21],
            presence: 0.87,
            side: "left",
            handedness: 0.12,
            gesture: Some("open"),
        }];
        p.recon_error = Some(0.031);
        p.plausibility = Some("ok".into());
        world.people.push(p);
        let line = sf.to_json_line_world(&world, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let f = &v["faces"][0];
        assert_eq!(f["left_gesture"], "waving");
        assert!(f.get("right_gesture").is_none(), "resting arm omitted");
        assert_eq!(f["wrists"][0][0].as_f64().unwrap(), 0.4);
        assert_eq!(f["hands"][0]["side"], "left");
        assert!((f["hands"][0]["presence"].as_f64().unwrap() - 0.87).abs() < 0.01);
        assert!((f["hands"][0]["handedness"].as_f64().unwrap() - 0.12).abs() < 0.01);
        assert_eq!(f["hands"][0]["gesture"], "open");
        assert!(f["hands"][0].get("landmarks").is_none(), "landmarks opt-in only");
        assert!((f["recon_error"].as_f64().unwrap() - 0.031).abs() < 1e-6);
        assert_eq!(f["plausibility"], "ok");
    }

    #[test]
    fn world_json_dump_landmarks_includes_hand_points() {
        let mut sf = StateFile::open_at(temp_state_path("test_world_lm.md")).unwrap();
        sf.dump_landmarks = true;
        let mut world = mvm_core::WorldState::new(std::time::Instant::now());
        let mut p = person_state(0, mvm_core::BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 });
        p.hands = vec![mvm_core::HandState {
            landmarks: vec![[0.5, 0.5]; 21],
            presence: 0.9,
            side: "right",
            handedness: 0.9,
            gesture: None,
        }];
        world.people.push(p);
        let line = sf.to_json_line_world(&world, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let lm = v["faces"][0]["hands"][0]["landmarks"].as_array().expect("landmarks dumped");
        assert_eq!(lm.len(), 21);
    }

    #[test]
    fn world_json_surfaces_emotion_entropy() {
        // The distribution is on the person → its entropy is pre-baked into JSON.
        let sf = StateFile::open_at(temp_state_path("test_world_ent.md")).unwrap();
        let mut world = mvm_core::WorldState::new(std::time::Instant::now());
        let mut p = person_state(0, mvm_core::BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 });
        p.emotion_distribution = Some([0.125; 8]); // uniform → entropy 1.0
        world.people.push(p);
        let line = sf.to_json_line_world(&world, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert!((v["faces"][0]["emotion_entropy"].as_f64().unwrap() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn world_json_surfaces_head_orientation() {
        // Phase 10.3: head_yaw/head_facing flow from the person to the JSON.
        let sf = StateFile::open_at(temp_state_path("test_world_head.md")).unwrap();
        let mut world = mvm_core::WorldState::new(std::time::Instant::now());
        let mut p = person_state(0, mvm_core::BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 });
        p.head_yaw = Some(-32.5);
        p.head_facing = Some("left");
        world.people.push(p);
        let line = sf.to_json_line_world(&world, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert!((v["faces"][0]["head_yaw"].as_f64().unwrap() + 32.5).abs() < 1e-3);
        assert_eq!(v["faces"][0]["head_facing"], "left");
    }

    #[test]
    fn sync_json_surfaces_head_orientation_from_track() {
        let sf = StateFile::open_at(temp_state_path("test_sync_head.md")).unwrap();
        let mut tracker = PersonTracker::new();
        let mut d = det(0.2, 0.3, 0.5, 0.7);
        d.head_yaw = Some(5.0);
        d.head_facing = Some("camera");
        tracker.tick(&[d]);
        let line = sf.to_json_line(&tracker, 30.0, &[]);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert!((v["faces"][0]["head_yaw"].as_f64().unwrap() - 5.0).abs() < 1e-3);
        assert_eq!(v["faces"][0]["head_facing"], "camera");
    }

    #[test]
    fn sync_json_surfaces_entropy_and_posture_quality() {
        // The tracker path emits the same phase-10.0 fields off the track.
        let sf = StateFile::open_at(temp_state_path("test_sync_p10.md")).unwrap();
        let mut tracker = PersonTracker::new();
        let mut one_hot = [0.0f32; 8];
        one_hot[3] = 1.0;
        let mut det = det(0.2, 0.3, 0.5, 0.7);
        det.emotion_distribution = Some(one_hot);
        det.recon_error = Some(0.12);
        det.plausibility = Some("low".into());
        tracker.tick(&[det]);
        let line = sf.to_json_line(&tracker, 30.0, &[]);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let f = &v["faces"][0];
        assert!(f["emotion_entropy"].as_f64().unwrap() < 1e-4, "one-hot → entropy 0");
        assert!((f["recon_error"].as_f64().unwrap() - 0.12).abs() < 1e-6);
        assert_eq!(f["plausibility"], "low");
        assert!(f.get("left_gesture").is_none(), "no motion ring → no gesture");
        assert!(f.get("hands").is_none(), "sync path carries no hands");
    }

    #[test]
    fn world_json_surfaces_activities_and_anomaly() {
        let sf = StateFile::open_at(temp_state_path("test_world_act.md")).unwrap();
        let mut world = mvm_core::WorldState::new(std::time::Instant::now());
        world.people.push(person_state(0, mvm_core::BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 }));
        world.activities.push(mvm_core::Activity {
            label: "typing".into(),
            confidence: 0.8,
            participants: vec![0, 10000],
        });
        world.anomaly_score = 0.25;
        let line = sf.to_json_line_world(&world, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["activities"][0]["label"], "typing");
        assert_eq!(v["activities"][0]["participants"][1].as_u64().unwrap(), 10000);
        assert!((v["anomaly_score"].as_f64().unwrap() - 0.25).abs() < 1e-6);
    }
}
