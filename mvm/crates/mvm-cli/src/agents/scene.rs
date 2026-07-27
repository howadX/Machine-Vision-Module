//! SceneAgent (phase 8.C.5) — group G₂. Owns the YOLO detector + its
//! `ObjectTracker`, runs on its own cadence (default every 3rd frame), and
//! publishes a stamped immutable observation. Object 3D is left `None` here — depth-anchoring
//! happens on the fusion step (8.C.7) where both people and objects are visible.

use crate::agents::{
    ObservationStamp, SceneObservationBus, SceneObservationFrame, cadence_due, collect_objects,
};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::sync::Arc;

/// Mask-IoU floor for greedily attaching a YOLO-seg mask onto a tracked object
/// (mirrors `pipeline::SEG_ATTACH_IOU` on the sync path).
const SEG_ATTACH_IOU: f32 = 0.5;

pub(crate) struct SceneAgent {
    detector: mvm_scene::YoloDetector,
    /// Optional YOLO-seg expert (phase 10.4, `--masks`). One forward yields both
    /// object masks (attached to tracked objects → mask-contact / mask-depth) and,
    /// in phase 11.A3, person masks published to the tracker.
    seg_detector: Option<mvm_scene::YoloSegDetector>,
    observations: SceneObservationBus,
    tracker: mvm_scene::ObjectTracker,
    animal_tracker: mvm_scene::ObjectTracker,
    score_thresh: f32,
    min_area: f32,
    cadence: u32,
}

impl SceneAgent {
    pub(crate) fn new(
        detector: mvm_scene::YoloDetector,
        seg_detector: Option<mvm_scene::YoloSegDetector>,
        observations: SceneObservationBus,
        tracker: mvm_scene::ObjectTracker,
        score_thresh: f32,
        min_area: f32,
        cadence: u32,
    ) -> Self {
        Self {
            detector,
            seg_detector,
            observations,
            tracker,
            animal_tracker: mvm_scene::ObjectTracker::with_params(0.3, 8, 0.5),
            score_thresh,
            min_area,
            cadence,
        }
    }
}

impl Agent for SceneAgent {
    fn name(&self) -> String {
        "scene".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0u64;
        let mut errors = 0u64;
        let mut last_id: Option<u64> = None;
        while ctx.running() {
            if let Some(frame) = ctx.frames.latest()
                && last_id != Some(frame.id)
                && cadence_due(frame.id, self.cadence)
            {
                last_id = Some(frame.id);
                match self.detector.detect(frame.image.as_ref()) {
                    Ok(objs) => {
                        // Animal-class detections (dog/cat/…) are NOT scene
                        // nodes (dropped by `is_relevant`), but fusion needs
                        // them to drop a pet-overlapping body from relational
                        // activities (anti-pet gate) — capture before filtering.
                        let animal_detections: Vec<_> = objs
                            .iter()
                            .filter(|o| mvm_scene::is_animal(o.class_name))
                            .cloned()
                            .collect();
                        let animals: Vec<_> = self
                            .animal_tracker
                            .update(&animal_detections)
                            .into_iter()
                            .map(|animal| mvm_fusion::canonical::AnimalObservation {
                                track_id: animal.id,
                                class_name: animal.class_name.to_owned(),
                                bbox: animal.bbox,
                                confidence: animal.score,
                                remembered: animal.missed > 0,
                            })
                            .collect();
                        // Filters A/C/D (mirror the synchronous loop): relevant
                        // classes, per-class score floor, minimum bbox area.
                        let filtered: Vec<_> = objs
                            .into_iter()
                            .filter(|o| mvm_scene::is_relevant(o.class_name))
                            .filter(|o| {
                                o.score
                                    >= mvm_scene::class_score_threshold(
                                        o.class_name,
                                        self.score_thresh,
                                    )
                            })
                            .filter(|o| {
                                let area = (o.bbox.x2 - o.bbox.x1) * (o.bbox.y2 - o.bbox.y1);
                                area >= self.min_area
                            })
                            .collect();
                        let mut tracked = self.tracker.update(&filtered);
                        let mut person_masks = Vec::new();

                        // Phase 10.4/11: the YOLO-seg expert (opt-in `--masks`)
                        // runs over the same frame; its object masks are attached
                        // onto the tracked objects (YOLO26n stays the detection
                        // authority) → mask-contact + mask-median depth on the
                        // production mesh path. Person masks (11.A3) are handled
                        // separately (published to the tracker, not attached).
                        if let Some(seg) = &mut self.seg_detector {
                            match seg.detect(frame.image.as_ref()) {
                                Ok(segs) => {
                                    // Split the single seg forward: object masks
                                    // attach to tracked objects; person masks are
                                    // published for the body tracker (11.A3).
                                    let mut obj_segs = Vec::new();
                                    for s in segs {
                                        if s.det.class_name == "person" {
                                            person_masks.push((s.det.bbox, s.mask));
                                        } else if mvm_scene::is_relevant(s.det.class_name) {
                                            obj_segs.push(s);
                                        }
                                    }
                                    mvm_scene::attach_masks(
                                        &mut tracked,
                                        &obj_segs,
                                        SEG_ATTACH_IOU,
                                    );
                                    if std::env::var("MVM_SEG_DEBUG").is_ok() {
                                        let attached =
                                            tracked.iter().filter(|o| o.mask.is_some()).count();
                                        eprintln!(
                                            "  seg-dbg: {} obj-seg → {}/{} objects masked; {} person masks published",
                                            obj_segs.len(),
                                            attached,
                                            tracked.len(),
                                            person_masks.len()
                                        );
                                    }
                                }
                                Err(e) => eprintln!("  seg: detect error: {e}"),
                            }
                        }

                        let stamp = ObservationStamp::from_frame(&frame);
                        self.observations.publish(SceneObservationFrame {
                            stamp,
                            objects: Arc::new(collect_objects(&tracked, stamp.produced_at)),
                            animals: Arc::new(animals),
                            person_masks: Arc::new(person_masks),
                        });
                        ticks += 1;
                    }
                    Err(e) => {
                        eprintln!("  scene: detect error: {e}");
                        errors += 1;
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        AgentHealth {
            name: self.name(),
            ticks,
            errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cadence_gate_runs_every_third_frame() {
        // The scene agent only works frames where its cadence is due.
        assert!(cadence_due(0, 3));
        assert!(!cadence_due(4, 3));
        assert!(cadence_due(9, 3));
    }

    #[test]
    fn relevant_filter_excludes_person() {
        // person is owned by the body-primary tracker (G₁) → never an object node.
        assert!(!mvm_scene::is_relevant("person"));
        assert!(mvm_scene::is_relevant("cup"));
    }

    #[test]
    fn person_masks_keep_the_scene_source_stamp() {
        let bus = SceneObservationBus::new();
        let mask = mvm_core::InstanceMask::from_bits(vec![
            0u8;
            mvm_core::mask::MASK_RES
                * mvm_core::mask::MASK_RES
        ]);
        bus.publish(SceneObservationFrame {
            stamp: ObservationStamp::new(9, std::time::Instant::now(), 1_000),
            objects: Arc::new(Vec::new()),
            animals: Arc::new(Vec::new()),
            person_masks: Arc::new(vec![(
                mvm_core::BBox {
                    x1: 0.1,
                    y1: 0.1,
                    x2: 0.3,
                    y2: 0.6,
                },
                mask,
            )]),
        });
        let got = bus.latest().expect("published scene observation");
        assert_eq!(got.stamp.source_frame_id, 9);
        assert_eq!(got.person_masks.len(), 1);
    }
}
