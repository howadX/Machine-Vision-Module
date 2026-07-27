//! DepthAgent (phase 8.C.5) — group G₄ (environment). Owns the heavy
//! Depth-Anything estimator (~230ms), runs on a sparse cadence (default every
//! 30th frame), and publishes a stamped immutable depth observation.
//!
//! It does NOT anchor objects — anchoring needs both people and objects and runs
//! on the fusion step (8.C.7), which reads this same slot. The slot lives in
//! `mvm-cli` (not `MeshContext`) so `mvm-mesh` need not depend on `mvm-depth`.

use crate::agents::{DepthObservationBus, DepthObservationFrame, ObservationStamp, cadence_due};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::sync::Arc;

pub(crate) struct DepthAgent {
    estimator: mvm_depth::DepthEstimator,
    observations: DepthObservationBus,
    cadence: u32,
}

impl DepthAgent {
    pub(crate) fn new(
        estimator: mvm_depth::DepthEstimator,
        observations: DepthObservationBus,
        cadence: u32,
    ) -> Self {
        Self {
            estimator,
            observations,
            cadence,
        }
    }
}

impl Agent for DepthAgent {
    fn name(&self) -> String {
        "depth".into()
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
                match self.estimator.estimate(frame.image.as_ref()) {
                    Ok(map) => {
                        self.observations.publish(DepthObservationFrame {
                            stamp: ObservationStamp::new(frame.id, frame.captured),
                            map: Arc::new(map),
                        });
                        ticks += 1;
                    }
                    Err(e) => {
                        eprintln!("  depth: estimate error: {e}");
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
    fn bus_starts_empty_and_accepts_a_stamped_map() {
        let bus = DepthObservationBus::new();
        assert!(bus.latest().is_none());
        bus.publish(DepthObservationFrame {
            stamp: ObservationStamp::new(3, std::time::Instant::now()),
            map: Arc::new(mvm_depth::DepthMap {
                data: vec![0.5; 4],
                w: 2,
                h: 2,
            }),
        });
        let got = bus.latest().expect("fusion sees the published map");
        assert_eq!(got.stamp.source_frame_id, 3);
        assert_eq!(got.map.w, 2);
    }

    #[test]
    fn depth_cadence_is_sparse() {
        // Default depth cadence (30) is due rarely.
        assert!(cadence_due(0, 30));
        assert!(!cadence_due(1, 30));
        assert!(!cadence_due(29, 30));
        assert!(cadence_due(30, 30));
    }
}
