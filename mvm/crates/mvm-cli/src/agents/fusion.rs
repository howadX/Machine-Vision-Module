//! FusionAgent (phase 8.C.7) — the cross-group step. It is the only agent that
//! sees **both** people and objects plus the shared depth map, so depth-anchoring
//! lives here (decision 8.C.5): the relative depth map is anchored to people's
//! metric torso-Z, giving objects a metric position in the same frame → real
//! person↔object `Near` edges (built later by the StateFileWriter / 8.F).
//!
//! It reads the depth handle from the shared `DepthSlot`, the frame dimensions
//! from the latest frame (the same pixel space the body-primary 3D uses), and
//! read-modify-writes `WorldState.objects` under a single write lock so a
//! concurrent `SceneAgent` swap can't clobber the anchored positions.

use crate::agents::DepthSlot;
use mvm_core::{Activity, BBox, ObjectState, PersonState, Position3D, WorldState};
use mvm_mesh::{Agent, AgentHealth, MeshContext};

pub(crate) struct FusionAgent {
    config: mvm_core::Config,
    depth_slot: DepthSlot,
}

impl FusionAgent {
    pub(crate) fn new(config: mvm_core::Config, depth_slot: DepthSlot) -> Self {
        Self { config, depth_slot }
    }
}

impl Agent for FusionAgent {
    fn name(&self) -> String {
        "fusion".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let pinhole = mvm_core::space::Pinhole::from_config(&self.config);
        let mut ticks = 0u64;
        while ctx.running() {
            // Cheaply grab the latest depth handle (clone the inner Arc, release
            // the slot lock immediately) + the frame's pixel dimensions.
            let depth = self.depth_slot.lock().ok().and_then(|g| g.clone());
            let dims = ctx
                .frames
                .latest()
                .map(|f| (f.image.width() as f32, f.image.height() as f32));
            if let Ok(mut w) = ctx.world.write() {
                let WorldState { people, objects, activities, anomaly_score, .. } = &mut *w;
                // Anchor objects to metric 3D when a depth map is available (the
                // person↔person activities below don't need it — people carry
                // torso-3D directly).
                if let (Some(d), Some((fw, fh))) = (depth.as_ref(), dims) {
                    anchor_objects(people.as_slice(), objects.as_mut_slice(), d, &pinhole, fw, fh);
                }
                // Cross-group inference (phase 8.F): pure arithmetic over the
                // fused entities.
                *activities = infer_activities(people.as_slice(), objects.as_slice());
                *anomaly_score = anomaly_from_objects(objects.as_slice());
                ticks += 1;
            }
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        AgentHealth { name: self.name(), ticks, errors: 0 }
    }
}

/// Anchor objects to metric 3D using people as the scale reference (phase 8.E.2,
/// now on the fusion step). For each person with a metric torso-Z and a depth
/// sample, `k_i = Z_i·d_i`; the median `k` converts each object's depth value to
/// metric `Z = k / d_obj`, back-projected through the same pinhole. With no
/// person anchor, objects are left depthless (overlap-only) — no fabricated
/// distance. Mirrors the synchronous `anchor_objects_depth`.
pub(crate) fn anchor_objects(
    people: &[PersonState],
    objects: &mut [ObjectState],
    depth: &mvm_depth::DepthMap,
    pinhole: &mvm_core::space::Pinhole,
    fw: f32,
    fh: f32,
) {
    const OBJ_DEPTH_CONF: f32 = 0.4;
    const Z_MIN: f32 = 0.3;
    const Z_MAX: f32 = 8.0;

    let mut ks: Vec<f32> = Vec::new();
    for p in people {
        if let Some(q) = p.position_3d {
            if q.z > 0.0 {
                if let Some(d) = depth.median_in_bbox(&p.bbox) {
                    if d > 1e-3 {
                        ks.push(q.z * d);
                    }
                }
            }
        }
    }
    if ks.is_empty() {
        for o in objects.iter_mut() {
            o.position_3d = None;
        }
        return;
    }
    ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ks[ks.len() / 2];

    for o in objects.iter_mut() {
        match depth.median_in_bbox(&o.bbox) {
            Some(d) if d > 1e-3 => {
                let z = (k / d).clamp(Z_MIN, Z_MAX);
                let cx = (o.bbox.x1 + o.bbox.x2) * 0.5 * fw;
                let cy = (o.bbox.y1 + o.bbox.y2) * 0.5 * fh;
                let p3 = pinhole.pixel_to_3d(cx, cy, z);
                o.position_3d = Some(Position3D { x: p3[0], y: p3[1], z: p3[2], confidence: OBJ_DEPTH_CONF });
            }
            _ => o.position_3d = None,
        }
    }
}

// ── Cross-group inference (phase 8.F) — pure arithmetic ───────────────────────

/// Two people closer than this (metric 3D) count as "together".
const ACTIVITY_NEAR_M: f32 = 1.5;
/// An object at least this fraction inside a person's bbox counts as in use.
const CONTAINMENT_MIN: f32 = 0.3;

/// The activity label a person↔object interaction yields, or `None` for objects
/// that are context only (furniture you don't "use" as an activity).
fn interaction_label(class: &str) -> Option<&'static str> {
    match class {
        "keyboard" | "laptop" => Some("typing"),
        "mouse" => Some("using_mouse"),
        "cell phone" => Some("using_phone"),
        "cup" | "bottle" | "wine glass" | "bowl" => Some("holding_drink"),
        "book" => Some("reading"),
        "remote" | "tv" => Some("watching_tv"),
        _ => None,
    }
}

/// Fraction of `inner` that lies inside `outer` (intersection / inner-area).
fn containment(outer: &BBox, inner: &BBox) -> f32 {
    let ix1 = outer.x1.max(inner.x1);
    let iy1 = outer.y1.max(inner.y1);
    let ix2 = outer.x2.min(inner.x2);
    let iy2 = outer.y2.min(inner.y2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let inner_area = (inner.x2 - inner.x1) * (inner.y2 - inner.y1);
    if inner_area > 0.0 { inter / inner_area } else { 0.0 }
}

fn dist3(a: Position3D, b: Position3D) -> f32 {
    let (dx, dy, dz) = (a.x - b.x, a.y - b.y, a.z - b.z);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Infer cross-group activities from the fused world (phase 8.F). Pure arithmetic:
///   - person↔person within `ACTIVITY_NEAR_M` (metric) → "people_together";
///   - person↔interactive-object (object mostly inside the person bbox) →
///     a class-specific label ("typing", "holding_drink", …).
/// Furniture is context only (no activity). Hand-precise variants ("hand∩keyboard")
/// need hand landmarks on the blackboard (or masks) — a later refinement.
pub(crate) fn infer_activities(people: &[PersonState], objects: &[ObjectState]) -> Vec<Activity> {
    let mut acts = Vec::new();

    // person ↔ person proximity (needs both torso-3D positions).
    for i in 0..people.len() {
        for j in (i + 1)..people.len() {
            if let (Some(pa), Some(pb)) = (people[i].position_3d, people[j].position_3d) {
                let d = dist3(pa, pb);
                if d < ACTIVITY_NEAR_M {
                    acts.push(Activity {
                        label: "people_together".into(),
                        confidence: (1.0 - d / ACTIVITY_NEAR_M).clamp(0.0, 1.0),
                        participants: vec![people[i].track_id, people[j].track_id],
                    });
                }
            }
        }
    }

    // person ↔ interactive object. Two triggers:
    //   1. strong — the object sits mostly inside the person's bbox (e.g. a cup
    //      held against the torso);
    //   2. posture-gated — a desk object (keyboard/mouse/laptop) anywhere while
    //      the person's posture says they're at a desk. On a ceiling cam the body
    //      bbox is head/shoulders and never reaches the desk, so containment alone
    //      misses "typing"; the posture classifier already encodes "at a desk", so
    //      we lean on it. Hand-precise contact ("hand∩keyboard") needs hand
    //      landmarks on the blackboard (a deferred tail) or masks.
    for p in people {
        let at_desk = p
            .posture
            .as_ref()
            .map(|(s, _)| s.contains("desk") || s == "sitting")
            .unwrap_or(false);
        let posture_conf = p.posture.as_ref().map(|(_, c)| *c).unwrap_or(0.5);
        for o in objects {
            if let Some(label) = interaction_label(&o.class) {
                let c = containment(&p.bbox, &o.bbox);
                let desk_obj = matches!(o.class.as_str(), "keyboard" | "mouse" | "laptop");
                if c >= CONTAINMENT_MIN {
                    acts.push(Activity {
                        label: label.into(),
                        confidence: (0.5 * o.confidence + 0.5 * c).clamp(0.0, 1.0),
                        participants: vec![p.track_id, o.id],
                    });
                } else if at_desk && desk_obj {
                    acts.push(Activity {
                        label: label.into(),
                        confidence: (0.5 * posture_conf + 0.5 * o.confidence).clamp(0.0, 1.0),
                        participants: vec![p.track_id, o.id],
                    });
                }
            }
        }
    }

    acts
}

/// Objects expected in a normal room/desk scene. Anything relevant-but-not-here
/// (sharp items, personal effects appearing/leaving) nudges the anomaly score.
const EXPECTED_OBJECTS: &[&str] = &[
    "chair", "couch", "bed", "dining table", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "book", "clock", "vase", "potted plant", "cup",
    "bottle", "bowl", "fork", "spoon", "microwave", "oven", "toaster", "sink",
    "refrigerator",
];

/// A simple anomaly heuristic (phase 8.F, v1): the share of detected objects
/// whose class is *not* part of the expected indoor inventory, scaled and
/// clamped. A placeholder — to be enriched with audio events (8.E.3) and learned
/// priors. Sound-driven anomalies are out of scope until YAMNet lands.
pub(crate) fn anomaly_from_objects(objects: &[ObjectState]) -> f32 {
    let unexpected = objects
        .iter()
        .filter(|o| !EXPECTED_OBJECTS.contains(&o.class.as_str()))
        .count();
    (unexpected as f32 * 0.25).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::Config;
    use std::time::Instant;

    fn pinhole() -> mvm_core::space::Pinhole {
        mvm_core::space::Pinhole::from_config(&Config::default())
    }

    fn person(z: f32) -> PersonState {
        PersonState {
            track_id: 10,
            bbox: BBox { x1: 0.3, y1: 0.2, x2: 0.5, y2: 0.7 },
            identity: None,
            emotion: None,
            posture: None,
            position_3d: Some(Position3D { x: 0.0, y: 0.0, z, confidence: 0.5 }),
            emotion_distribution: None,
            last_seen: Instant::now(),
        }
    }

    fn object() -> ObjectState {
        ObjectState {
            id: 10000,
            class: "cup".into(),
            bbox: BBox { x1: 0.45, y1: 0.45, x2: 0.6, y2: 0.6 },
            confidence: 0.8,
            age: 2,
            position_3d: Some(Position3D { x: 9.0, y: 9.0, z: 9.0, confidence: 0.4 }), // stale
            last_seen: Instant::now(),
        }
    }

    /// Uniform depth map → every `median_in_bbox` returns the same value.
    fn uniform_depth(v: f32) -> mvm_depth::DepthMap {
        mvm_depth::DepthMap { data: vec![v; 64], w: 8, h: 8 }
    }

    #[test]
    fn no_people_leaves_objects_depthless() {
        let mut objs = vec![object()];
        anchor_objects(&[], &mut objs, &uniform_depth(0.5), &pinhole(), 640.0, 480.0);
        assert!(objs[0].position_3d.is_none(), "no anchor → depthless");
    }

    #[test]
    fn person_anchor_sets_object_metric_z() {
        // person Z=2.0, d_person=0.5 → k=1.0; object d=0.5 → Z = 1.0/0.5 = 2.0.
        let mut objs = vec![object()];
        anchor_objects(&[person(2.0)], &mut objs, &uniform_depth(0.5), &pinhole(), 640.0, 480.0);
        let q = objs[0].position_3d.expect("object anchored");
        assert!((q.z - 2.0).abs() < 1e-3, "metric Z from people anchor, got {}", q.z);
        assert!((q.confidence - 0.4).abs() < 1e-6);
    }

    #[test]
    fn degenerate_object_bbox_yields_no_position() {
        // A zero-area object bbox → median_in_bbox None → depthless even with anchor.
        let mut o = object();
        o.bbox = BBox { x1: 0.5, y1: 0.5, x2: 0.5, y2: 0.5 };
        let mut objs = vec![o];
        anchor_objects(&[person(2.0)], &mut objs, &uniform_depth(0.5), &pinhole(), 640.0, 480.0);
        assert!(objs[0].position_3d.is_none(), "degenerate bbox → None");
    }

    // ── activities + anomaly (phase 8.F) ──

    fn obj_class(id: u32, class: &str, bbox: BBox) -> ObjectState {
        ObjectState {
            id,
            class: class.into(),
            bbox,
            confidence: 0.8,
            age: 1,
            position_3d: None,
            last_seen: Instant::now(),
        }
    }

    #[test]
    fn typing_when_keyboard_inside_person() {
        // person bbox is {0.3,0.2,0.5,0.7}; keyboard fully inside it.
        let p = person(1.0);
        let kb = obj_class(10001, "keyboard", BBox { x1: 0.35, y1: 0.55, x2: 0.45, y2: 0.62 });
        let acts = infer_activities(&[p], &[kb]);
        assert!(
            acts.iter().any(|a| a.label == "typing" && a.participants == vec![10, 10001]),
            "got {acts:?}"
        );
    }

    #[test]
    fn no_activity_when_object_outside_and_not_at_desk() {
        // person(1.0) has posture None → no desk gate, keyboard outside bbox → nothing.
        let p = person(1.0);
        let kb = obj_class(10001, "keyboard", BBox { x1: 0.8, y1: 0.8, x2: 0.9, y2: 0.85 });
        assert!(infer_activities(&[p], &[kb]).is_empty());
    }

    #[test]
    fn desk_posture_infers_activity_even_without_containment() {
        // Ceiling-cam case: mouse on the desk, outside the head/shoulders bbox, but
        // posture says the person is at a desk → posture-gated "using_mouse".
        let mut p = person(1.0);
        p.posture = Some(("sitting_at_desk".into(), 0.8));
        let mouse = obj_class(10001, "mouse", BBox { x1: 0.8, y1: 0.8, x2: 0.85, y2: 0.85 });
        let acts = infer_activities(&[p], &[mouse]);
        assert!(
            acts.iter().any(|a| a.label == "using_mouse" && a.participants == vec![10, 10001]),
            "got {acts:?}"
        );
    }

    #[test]
    fn furniture_is_context_not_activity() {
        let p = person(1.0);
        let chair = obj_class(10001, "chair", BBox { x1: 0.35, y1: 0.55, x2: 0.45, y2: 0.65 });
        assert!(infer_activities(&[p], &[chair]).is_empty(), "chair is context, not an activity");
    }

    #[test]
    fn people_together_when_near() {
        let mut a = person(1.0);
        a.track_id = 10;
        let mut b = person(1.0);
        b.track_id = 11;
        let acts = infer_activities(&[a, b], &[]);
        assert!(
            acts.iter().any(|x| x.label == "people_together" && x.participants == vec![10, 11]),
            "got {acts:?}"
        );
    }

    #[test]
    fn anomaly_rises_for_unexpected_object() {
        let cup = obj_class(10001, "cup", BBox { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2 });
        let knife = obj_class(10002, "knife", BBox { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2 });
        assert_eq!(anomaly_from_objects(&[cup]), 0.0, "cup is expected → no anomaly");
        assert!(anomaly_from_objects(&[knife]) > 0.0, "knife is unexpected → anomaly");
    }
}
