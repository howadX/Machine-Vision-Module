//! HumanAgent (phase 8.C.5) — group G₁. Wraps a (human-only) `BodyPrimaryPipeline`
//! and owns its own `PersonTracker`. Every frame: process → track → project the
//! active tracks into `WorldState.people`. The pipeline is constructed with no
//! scene/depth detector (those are their own agents now), so it does only the
//! body-primary chain; objects/depth come from `SceneAgent`/`DepthAgent`.

use crate::agents::project_people;
use crate::pipeline::BodyPrimaryPipeline;
use mvm_mesh::{Agent, AgentHealth, MeshContext};

pub(crate) struct HumanAgent {
    pipeline: BodyPrimaryPipeline,
    tracker: mvm_core::PersonTracker,
}

impl HumanAgent {
    pub(crate) fn new(pipeline: BodyPrimaryPipeline, tracker: mvm_core::PersonTracker) -> Self {
        Self { pipeline, tracker }
    }
}

impl Agent for HumanAgent {
    fn name(&self) -> String {
        "human".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0u64;
        let mut last_id: Option<u64> = None;
        let mut epoch: Option<std::time::Instant> = None; // motion timestamp origin (9.8)
        while ctx.running() {
            if let Some(frame) = ctx.frames.latest() {
                if last_id != Some(frame.id) {
                    last_id = Some(frame.id);
                    let prev_tracks: Vec<mvm_core::BBox> =
                        self.tracker.active().map(|t| t.bbox).collect();
                    // Id-locked tracks (phase 9.6) → their bodies skip SFace this frame.
                    let locked_tracks: Vec<mvm_core::BBox> =
                        self.tracker.active().filter(|t| t.is_id_locked()).map(|t| t.bbox).collect();
                    // Emotion-fresh tracks (phase 9.7) → their bodies skip EmotiEffLib.
                    let emotion_fresh: Vec<mvm_core::BBox> =
                        self.tracker.active().filter(|t| !t.needs_emotion()).map(|t| t.bbox).collect();
                    let result = self.pipeline.process(
                        frame.image.as_ref(), frame.id, &prev_tracks, &locked_tracks, &emotion_fresh,
                    );
                    // Monotonic seconds since the first frame → motion velocities (9.8).
                    let t_secs = frame
                        .captured
                        .duration_since(*epoch.get_or_insert(frame.captured))
                        .as_secs_f64();
                    self.tracker.tick_motion(
                        &result.detections, &result.low_conf, &result.landmarks, t_secs,
                    );
                    project_people(&ctx.world, &self.tracker, &result.bodies, frame.id, frame.captured);
                    ticks += 1;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        AgentHealth { name: self.name(), ticks, errors: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{Config, Detection, WorldState};
    use std::sync::{Arc, RwLock};
    use std::time::Instant;

    /// An anonymous body (no matched face) projects to a person with no identity
    /// — the body-primary FRONTIER invariant (track without a frontal face).
    #[test]
    fn anonymous_body_has_no_identity() {
        let config = Config::default();
        let mut tracker = mvm_core::PersonTracker::with_config(&config);
        tracker.tick(&[Detection::bare(0.2, 0.2, 0.4, 0.6)]);

        let world = Arc::new(RwLock::new(WorldState::new(Instant::now())));
        project_people(&world, &tracker, &[], 1, Instant::now());

        let w = world.read().unwrap();
        assert_eq!(w.people.len(), 1);
        assert!(w.people[0].identity.is_none(), "anonymous body → identity None");
    }

    /// Two distinct bodies → two independent people in the world (multi-persona).
    #[test]
    fn two_bodies_project_two_people() {
        let config = Config::default();
        let mut tracker = mvm_core::PersonTracker::with_config(&config);
        tracker.tick(&[
            Detection::bare(0.05, 0.2, 0.25, 0.7),
            Detection::bare(0.65, 0.2, 0.85, 0.7),
        ]);

        let world = Arc::new(RwLock::new(WorldState::new(Instant::now())));
        project_people(&world, &tracker, &[], 2, Instant::now());

        assert_eq!(world.read().unwrap().people.len(), 2);
    }
}
