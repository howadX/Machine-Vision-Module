//! MonocularDepthProvider — group G₄ (environment). Owns the heavy
//! Depth-Anything estimator (~230ms), runs on a sparse cadence (default every
//! 30th frame), and publishes stamped immutable relative-depth evidence.
//! Registered metric evidence suppresses inference on that RGB tick.
//!
//! It does NOT anchor objects — anchoring needs both people and objects and runs
//! on the fusion step (8.C.7), which reads this same slot. The slot lives in
//! `mvm-cli` (not `MeshContext`) so `mvm-mesh` need not depend on `mvm-depth`.

use crate::agents::{
    DepthEvidence, DepthExecutionControl, DepthObservationBus, DepthObservationFrame,
    ObservationStamp, cadence_due, registered_metric_from_frame,
};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::sync::Arc;

pub(crate) struct MonocularDepthProvider {
    estimator: mvm_depth::DepthEstimator,
    observations: DepthObservationBus,
    cadence: u32,
    control: DepthExecutionControl,
    metrics: super::SchedulingMetrics,
    suppress_for_metric: bool,
    max_pair_skew_ns: u64,
}

impl MonocularDepthProvider {
    pub(crate) fn new(
        estimator: mvm_depth::DepthEstimator,
        observations: DepthObservationBus,
        cadence: u32,
        control: DepthExecutionControl,
        metrics: super::SchedulingMetrics,
        suppress_for_metric: bool,
        max_pair_skew_ns: u64,
    ) -> Self {
        Self {
            estimator,
            observations,
            cadence,
            control,
            metrics,
            suppress_for_metric,
            max_pair_skew_ns,
        }
    }
}

impl Agent for MonocularDepthProvider {
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
            {
                let permit = self.control.take_for(frame.id);
                let metric_is_primary = self.suppress_for_metric
                    && registered_metric_from_frame(&frame, self.max_pair_skew_ns).is_some();
                if !metric_is_primary && (cadence_due(frame.id, self.cadence) || permit.is_some()) {
                    if permit.is_some() && !cadence_due(frame.id, self.cadence) {
                        self.metrics
                            .record_extra_run(mvm_cortex::ObservationTarget::Depth);
                    }
                    last_id = Some(frame.id);
                    match self.estimator.estimate(frame.image.as_ref()) {
                        Ok(map) => {
                            self.observations.publish(DepthObservationFrame {
                                stamp: ObservationStamp::from_frame(&frame),
                                evidence: DepthEvidence::Relative(Arc::new(map)),
                            });
                            ticks += 1;
                        }
                        Err(e) => {
                            eprintln!("  depth: estimate error: {e}");
                            errors += 1;
                        }
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
            stamp: ObservationStamp::new(3, std::time::Instant::now(), 1_000),
            evidence: DepthEvidence::Relative(Arc::new(mvm_depth::RelativeDepthMap {
                data: vec![0.5; 4],
                w: 2,
                h: 2,
            })),
        });
        let got = bus.latest().expect("fusion sees the published map");
        assert_eq!(got.stamp.source_frame_id, 3);
        assert_eq!(got.relative_map().unwrap().w, 2);
    }

    #[test]
    fn depth_cadence_is_sparse() {
        // Default depth cadence (30) is due rarely.
        assert!(cadence_due(0, 30));
        assert!(!cadence_due(1, 30));
        assert!(!cadence_due(29, 30));
        assert!(cadence_due(30, 30));
    }

    #[test]
    fn depth_permit_is_one_shot_and_does_not_change_cadence() {
        use mvm_cortex::{EvidenceRef, ObservationRequest, ObservationTarget, ReasonCode};

        let control = DepthExecutionControl::default();
        control.grant(ObservationRequest {
            request_id: 9,
            source_scene_frame_id: 7,
            target: ObservationTarget::Depth,
            reason: ReasonCode::StaleSource,
            evidence: vec![EvidenceRef {
                scene_frame_id: 7,
                entity: None,
                event_seq: None,
            }],
            priority: 200,
            max_extra_runs: 1,
            expires_at_frame: 10,
        });

        assert_eq!(control.take_for(8).unwrap().request_id, 9);
        assert!(control.take_for(8).is_none());
        assert!(!cadence_due(8, 30));
    }
}
