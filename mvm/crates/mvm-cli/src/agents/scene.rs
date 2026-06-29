//! SceneAgent (phase 8.C.5) — group G₂. Owns the YOLO detector + its
//! `ObjectTracker`, runs on its own cadence (default every 3rd frame), and
//! writes `WorldState.objects`. Object 3D is left `None` here — depth-anchoring
//! happens on the fusion step (8.C.7) where both people and objects are visible.

use crate::agents::{cadence_due, project_objects};
use mvm_mesh::{Agent, AgentHealth, MeshContext};

pub(crate) struct SceneAgent {
    detector: mvm_scene::YoloDetector,
    tracker: mvm_scene::ObjectTracker,
    score_thresh: f32,
    min_area: f32,
    cadence: u32,
}

impl SceneAgent {
    pub(crate) fn new(
        detector: mvm_scene::YoloDetector,
        tracker: mvm_scene::ObjectTracker,
        score_thresh: f32,
        min_area: f32,
        cadence: u32,
    ) -> Self {
        Self { detector, tracker, score_thresh, min_area, cadence }
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
            if let Some(frame) = ctx.frames.latest() {
                if last_id != Some(frame.id) && cadence_due(frame.id, self.cadence) {
                    last_id = Some(frame.id);
                    match self.detector.detect(frame.image.as_ref()) {
                        Ok(objs) => {
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
                            let tracked = self.tracker.update(&filtered);
                            project_objects(&ctx.world, &tracked, frame.captured);
                            ticks += 1;
                        }
                        Err(e) => {
                            eprintln!("  scene: detect error: {e}");
                            errors += 1;
                        }
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        AgentHealth { name: self.name(), ticks, errors }
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
}
