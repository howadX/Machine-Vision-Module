//! Mesh agent wrappers (phase 8.C.5).
//!
//! Each detector chain becomes an `mvm_mesh::Agent` running on its own thread at
//! its own cadence, reading the latest frame and writing its slice of the shared
//! `WorldState` (or, for depth, a shared raw-map slot). The heavy pieces stay in
//! their crates; these are thin glue + the deterministic projection/cadence
//! helpers (the unit-tested surface — the `run()` loops need real models and are
//! exercised live in the `mesh` subcommand, 8.D).
//!
//! Decision 8.C.5: depth-anchoring is **not** here — it runs on the fusion step
//! (8.C.7) where both people and objects are visible. The `DepthAgent` only
//! publishes the raw `DepthMap` to a shared slot. Keeping that slot in `mvm-cli`
//! (not `MeshContext`) lets `mvm-mesh` stay detector-agnostic (no `mvm-depth` dep
//! → no cycle).

mod capture;
mod depth;
mod fusion;
mod human;
mod scene;
mod statefile;

pub(crate) use capture::CaptureAgent;
pub(crate) use depth::{DepthAgent, DepthSlot};
pub(crate) use fusion::{infer_activities, FusionAgent};
pub(crate) use human::HumanAgent;
pub(crate) use scene::SceneAgent;
pub(crate) use statefile::StateFileWriter;

use crate::pipeline::BodyData;
use mvm_core::{BBox, ObjectState, PersonState, PersonTracker, Position3D, WorldState};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// A cadence gate: run on frame 0 and every `cadence`-th frame after. `cadence
/// <= 1` means every frame. Mirrors the synchronous loop's
/// `frame_id % object_cadence == 0`, now per-agent.
pub(crate) fn cadence_due(frame_id: u64, cadence: u32) -> bool {
    cadence <= 1 || frame_id % cadence as u64 == 0
}

/// Project the tracker's active people into `WorldState.people` and advance the
/// frame stamp. Each person's wrists/hands are attached from the per-body
/// landmark data by best bbox match (phase 8.F: interaction by landmarks). Called
/// by the `HumanAgent` every frame (it owns the frame_id / timestamp advance).
pub(crate) fn project_people(
    world: &Arc<RwLock<WorldState>>,
    tracker: &PersonTracker,
    bodies: &[BodyData],
    frame_id: u64,
    now: Instant,
) {
    let people: Vec<PersonState> = tracker
        .active()
        .map(|t| {
            let mut p = PersonState::from_track(t, now);
            if let Some(b) = best_body_for_bbox(&p.bbox, bodies) {
                p.wrists = b.wrists.clone();
                p.hands = b.hands.clone();
            }
            p
        })
        .collect();
    if let Ok(mut w) = world.write() {
        w.people = people;
        w.frame_id = frame_id;
        w.timestamp = now;
    }
}

/// The body whose bbox overlaps `target` most (IoU), above a small floor. Ties a
/// track back to the pose/hand data it came from.
fn best_body_for_bbox<'a>(target: &BBox, bodies: &'a [BodyData]) -> Option<&'a BodyData> {
    bodies
        .iter()
        .map(|b| (b, bbox_iou(target, &b.bbox)))
        .filter(|(_, iou)| *iou > 0.1)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(b, _)| b)
}

fn bbox_iou(a: &BBox, b: &BBox) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    if ua > 0.0 { inter / ua } else { 0.0 }
}

/// Project tracked objects into `WorldState.objects`. 3D is whatever the tracked
/// object already carries (`None` until the fusion step anchors them, 8.C.7).
pub(crate) fn project_objects(
    world: &Arc<RwLock<WorldState>>,
    objects: &[mvm_scene::TrackedObject],
    now: Instant,
) {
    let objs: Vec<ObjectState> = objects
        .iter()
        .map(|t| ObjectState {
            id: t.id,
            class: t.class_name.to_string(),
            bbox: t.bbox,
            confidence: t.score,
            age: t.age,
            position_3d: t
                .pos_3d
                .map(|[x, y, z]| Position3D { x, y, z, confidence: t.pos_conf }),
            mask: t.mask.clone(),
            last_seen: now,
        })
        .collect();
    if let Ok(mut w) = world.write() {
        w.objects = objs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{BBox, Config, Detection};
    use std::time::Instant;

    #[test]
    fn cadence_due_every_frame_when_one() {
        assert!(cadence_due(0, 1));
        assert!(cadence_due(7, 1));
        assert!(cadence_due(7, 0)); // 0 also means "every frame"
    }

    #[test]
    fn cadence_due_every_nth() {
        assert!(cadence_due(0, 3));
        assert!(!cadence_due(1, 3));
        assert!(!cadence_due(2, 3));
        assert!(cadence_due(3, 3));
        assert!(cadence_due(6, 3));
    }

    #[test]
    fn project_people_writes_active_tracks() {
        let config = Config::default();
        let mut tracker = PersonTracker::with_config(&config);
        // One body detection → one active track after a tick.
        let mut det = Detection::bare(0.2, 0.2, 0.4, 0.6);
        det.position_z = Some(1.5);
        tracker.tick(&[det]);

        let world = Arc::new(RwLock::new(WorldState::new(Instant::now())));
        let now = Instant::now();
        project_people(&world, &tracker, &[], 42, now);

        let w = world.read().unwrap();
        assert_eq!(w.frame_id, 42);
        assert_eq!(w.people.len(), 1, "one active track → one PersonState");
    }

    #[test]
    fn project_objects_maps_fields_and_3d() {
        let world = Arc::new(RwLock::new(WorldState::new(Instant::now())));
        let now = Instant::now();
        let objs = vec![
            mvm_scene::TrackedObject {
                id: 10000,
                class_name: "cup",
                bbox: BBox { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2 },
                score: 0.8,
                age: 3,
                missed: 0,
                pos_3d: Some([0.5, 0.1, 0.7]),
                pos_conf: 0.4,
                mask: None,
            },
            mvm_scene::TrackedObject {
                id: 10001,
                class_name: "bed",
                bbox: BBox { x1: 0.6, y1: 0.6, x2: 0.9, y2: 0.9 },
                score: 0.6,
                age: 1,
                missed: 0,
                pos_3d: None,
                pos_conf: 0.0,
                mask: None,
            },
        ];
        project_objects(&world, &objs, now);

        let w = world.read().unwrap();
        assert_eq!(w.objects.len(), 2);
        assert_eq!(w.objects[0].id, 10000);
        assert_eq!(w.objects[0].class, "cup");
        let p = w.objects[0].position_3d.as_ref().expect("cup has 3D");
        assert!((p.z - 0.7).abs() < 1e-6);
        assert!(w.objects[1].position_3d.is_none(), "bed has no depth anchor → None");
    }
}
