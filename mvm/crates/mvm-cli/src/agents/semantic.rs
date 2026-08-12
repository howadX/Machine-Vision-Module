//! Default-off measurement-only SegFormer shadow agent.

use crate::agents::{SemanticObservationBus, SemanticObservationFrame, cadence_due};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::sync::Arc;

pub(crate) const DEFAULT_CADENCE: u32 = 30;
pub(crate) const DEFAULT_MAX_AGE_FRAMES: u64 = 60;

pub(crate) fn should_publish(source_frame_id: u64, current_frame_id: u64, max_age: u64) -> bool {
    current_frame_id
        .checked_sub(source_frame_id)
        .is_some_and(|age| age <= max_age)
}

pub(crate) struct SemanticAgent {
    estimator: mvm_semantic::SemanticEstimator,
    observations: SemanticObservationBus,
    cadence: u32,
    max_age_frames: u64,
}

impl SemanticAgent {
    pub(crate) fn new(
        estimator: mvm_semantic::SemanticEstimator,
        observations: SemanticObservationBus,
        cadence: u32,
        max_age_frames: u64,
    ) -> Self {
        Self {
            estimator,
            observations,
            cadence,
            max_age_frames,
        }
    }
}

impl Agent for SemanticAgent {
    fn name(&self) -> String {
        "semantic-shadow".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0;
        let mut errors = 0;
        let mut last_id = None;
        while ctx.running() {
            if let Some(frame) = ctx.frames.latest()
                && last_id != Some(frame.id)
                && cadence_due(frame.id, self.cadence)
            {
                last_id = Some(frame.id);
                match self.estimator.estimate(frame.image.as_ref()) {
                    Ok(grid) => {
                        let current = ctx.frames.latest().map_or(frame.id, |latest| latest.id);
                        if should_publish(frame.id, current, self.max_age_frames) {
                            self.observations.publish(SemanticObservationFrame {
                                stamp: crate::agents::ObservationStamp::from_frame(&frame),
                                grid: Arc::new(grid),
                            });
                            ticks += 1;
                        } else {
                            errors += 1;
                            eprintln!(
                                "  semantic-shadow: dropped stale result from frame {} at frame {current}",
                                frame.id
                            );
                        }
                    }
                    Err(error) => {
                        errors += 1;
                        eprintln!("  semantic-shadow: estimate error: {error}");
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
    fn semantic_results_reject_future_and_stale_sources() {
        assert!(should_publish(10, 10, 60));
        assert!(should_publish(10, 70, 60));
        assert!(!should_publish(10, 71, 60));
        assert!(!should_publish(11, 10, 60));
    }

    #[test]
    fn semantic_default_cadence_is_sparse() {
        assert_eq!(DEFAULT_CADENCE, 30);
        assert!(crate::agents::cadence_due(0, DEFAULT_CADENCE));
        assert!(!crate::agents::cadence_due(1, DEFAULT_CADENCE));
    }

    #[test]
    fn shadow_name_cannot_be_confused_with_canonical_semantics() {
        assert_eq!("semantic-shadow", "semantic-shadow");
        assert_eq!(DEFAULT_MAX_AGE_FRAMES, 60);
    }
}
