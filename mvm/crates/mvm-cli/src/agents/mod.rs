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
//! (8.C.7) where both people and objects are visible. The monocular provider only
//! publishes the raw `RelativeDepthMap` to a shared slot. Keeping that slot in `mvm-cli`
//! (not `MeshContext`) lets `mvm-mesh` stay detector-agnostic (no `mvm-depth` dep
//! → no cycle).

mod capture;
mod cortex;
pub(crate) use crate::cortex_report::CortexReportWriter;
mod dataset_writer;
mod depth;
mod fusion;
/// Golden-scenario CI gate (phase 15.4): replays compiled scenarios through the
/// production temporal stack — model-free, `--stub`-class.
#[cfg(test)]
mod golden;
mod human;
mod human_enrichment;
mod observations;
mod policy;
mod scene;
mod statefile;
mod tracker_capture;
mod tracking;

pub(crate) use capture::CaptureAgent;
pub(crate) use cortex::CortexAgent;
pub(crate) use dataset_writer::DatasetWriterAgent;
pub(crate) use depth::MonocularDepthProvider;
pub(crate) use fusion::{FusionAgent, infer_activities};
pub(crate) use human::LegacyHumanAgent;
pub(crate) use human_enrichment::HumanAgent;
pub(crate) use observations::{
    AssociationMeasurementContext, AttentionPolicyBus, CortexFrameBus, DepthEvidence,
    DepthObservationBus, DepthObservationFrame, FrameEvidenceHandles, FusedFrameBus,
    FusedFramePacket, HumanEnrichmentBus, HumanEnrichmentFrame, HumanEnrichmentItem,
    HumanObservationBus, HumanObservationFrame, HumanWorkBus, HumanWorkItem, ObservationStamp,
    PolicyOutcomeBus, SceneObservationBus, SceneObservationFrame, TrackedPersonWork,
    registered_metric_from_frame,
};
pub(crate) use policy::{
    DepthExecutionControl, MetricAgent, PolicySchedulerAgent, PolicySchedulerWiring,
    SceneExecutionControl, SchedulingMetrics, SchedulingMode,
};
pub(crate) use scene::SceneAgent;
pub(crate) use statefile::{JournalCfg, StateFileWriter};
pub(crate) use tracker_capture::{ModelDigest, TrackerCalibrationCapture, TrackerCaptureMetadata};
pub(crate) use tracking::TrackingAgent;

use crate::pipeline::BodyData;
use mvm_core::{BBox, ObjectState, PersonState, PersonTracker, Position3D};
use std::time::Instant;

/// A cadence gate: run on frame 0 and every `cadence`-th frame after. `cadence
/// <= 1` means every frame. Mirrors the synchronous loop's
/// `frame_id % object_cadence == 0`, now per-agent.
pub(crate) fn cadence_due(frame_id: u64, cadence: u32) -> bool {
    cadence <= 1 || frame_id.is_multiple_of(cadence as u64)
}

/// Project the tracker's active people into an immutable expert observation.
/// Each person's wrists/hands are attached from the per-body landmark data by
/// best bbox match (phase 8.F: interaction by landmarks).
pub(crate) fn collect_people(
    tracker: &PersonTracker,
    bodies: &[BodyData],
    now: Instant,
) -> Vec<PersonState> {
    tracker
        .published()
        .map(|t| {
            let mut p = PersonState::from_track(t, now);
            if let Some(b) = best_body_for_bbox(&p.bbox, bodies) {
                p.wrists = b.wrists.clone();
                p.hands = b.hands.clone();
                p.face_bbox = b.face_bbox; // this frame's matched face (16.5 privacy)
            }
            p
        })
        .collect()
}

pub(crate) fn collect_occluded_people(tracker: &PersonTracker, now: Instant) -> Vec<PersonState> {
    tracker
        .occluded()
        .map(|track| PersonState::from_track(track, now))
        .collect()
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

/// Project tracked objects into an immutable expert observation. 3D is whatever
/// the tracked object already carries (`None` until canonical fusion anchors it).
pub(crate) fn collect_objects(
    objects: &[mvm_scene::TrackedObject],
    now: Instant,
) -> Vec<ObjectState> {
    objects
        .iter()
        .map(|t| ObjectState {
            id: t.id,
            class: t.class_name.to_string(),
            bbox: t.bbox,
            confidence: t.score,
            age: t.age,
            position_3d: t.pos_3d.map(|[x, y, z]| Position3D {
                x,
                y,
                z,
                confidence: t.pos_conf,
            }),
            mask: t.mask.clone(),
            // Object-permanence: a held (occluded) static fixture has missed > 0
            // but is still emitted (phase 11) — surface it on mesh (12.B3 parity).
            remembered: t.missed > 0,
            last_seen: now,
        })
        .collect()
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
    fn tentative_person_is_not_projected_downstream() {
        let mut tracker = PersonTracker::new();
        tracker.tick(&[Detection::bare(0.2, 0.1, 0.5, 0.9)]);
        assert_eq!(tracker.active().count(), 1, "tracker keeps its hypothesis");
        assert!(
            collect_people(&tracker, &[], Instant::now()).is_empty(),
            "one-frame tentative detection must not become a PersonState"
        );
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
        // Publication is bounded by the explicit confirmation window.
        let mut det = Detection::bare(0.2, 0.2, 0.4, 0.6);
        det.position_z = Some(1.5);
        for _ in 0..mvm_core::tracker::PERSON_CONFIRM_AGE {
            tracker.tick(&[det.clone()]);
        }

        let now = Instant::now();
        let people = collect_people(&tracker, &[], now);

        assert_eq!(people.len(), 1, "one confirmed track → one PersonState");
    }

    #[test]
    fn project_objects_maps_fields_and_3d() {
        let now = Instant::now();
        let objs = vec![
            mvm_scene::TrackedObject {
                id: 10000,
                class_name: "cup",
                bbox: BBox {
                    x1: 0.1,
                    y1: 0.1,
                    x2: 0.2,
                    y2: 0.2,
                },
                score: 0.8,
                age: 3,
                missed: 0,
                lifecycle: mvm_core::types::LifecycleState::Confirmed,
                pos_3d: Some([0.5, 0.1, 0.7]),
                pos_conf: 0.4,
                mask: None,
            },
            mvm_scene::TrackedObject {
                id: 10001,
                class_name: "bed",
                bbox: BBox {
                    x1: 0.6,
                    y1: 0.6,
                    x2: 0.9,
                    y2: 0.9,
                },
                score: 0.6,
                age: 1,
                missed: 0,
                lifecycle: mvm_core::types::LifecycleState::Tentative,
                pos_3d: None,
                pos_conf: 0.0,
                mask: None,
            },
        ];
        let objects = collect_objects(&objs, now);

        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].id, 10000);
        assert_eq!(objects[0].class, "cup");
        let p = objects[0].position_3d.as_ref().expect("cup has 3D");
        assert!((p.z - 0.7).abs() < 1e-6);
        assert!(
            objects[1].position_3d.is_none(),
            "bed has no depth anchor → None"
        );
    }

    #[test]
    fn only_fusion_writes_the_compatibility_world() {
        for (name, source) in [
            ("human", include_str!("human.rs")),
            ("scene", include_str!("scene.rs")),
            ("depth", include_str!("depth.rs")),
            ("statefile", include_str!("statefile.rs")),
            ("dataset_writer", include_str!("dataset_writer.rs")),
        ] {
            assert!(
                !source.contains("ctx.world.write()"),
                "{name} must consume observations/canonical frames read-only"
            );
        }
        assert_eq!(
            include_str!("fusion.rs")
                .matches("ctx.world.write()")
                .count(),
            1,
            "Fusion owns the sole compatibility projection write-site"
        );
    }
}
