//! AgentMesh adapter for `mvm-cortex`.
//!
//! It reads immutable Fusion packets and publishes a separate cognitive view;
//! it never writes `WorldState` or scene facts.

use super::{AttentionPolicyBus, CortexFrameBus, FusedFrameBus};
use mvm_cortex::{CortexConfig, CortexInput, VisualCortex};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::sync::Arc;

pub(crate) struct CortexAgent {
    fused_frames: FusedFrameBus,
    cortex_frames: CortexFrameBus,
    policies: AttentionPolicyBus,
    cortex: VisualCortex,
    config: mvm_core::Config,
    event_cursor: Option<u64>,
    previous_scene: Option<Arc<mvm_core::canonical::FusedSceneFrame>>,
    last_scene_frame: Option<u64>,
}

impl CortexAgent {
    pub(crate) fn new(
        fused_frames: FusedFrameBus,
        cortex_frames: CortexFrameBus,
        policies: AttentionPolicyBus,
        config: mvm_core::Config,
        cortex_config: CortexConfig,
    ) -> Self {
        Self {
            fused_frames,
            cortex_frames,
            policies,
            cortex: VisualCortex::new(cortex_config),
            config,
            event_cursor: None,
            previous_scene: None,
            last_scene_frame: None,
        }
    }
}

impl Agent for CortexAgent {
    fn name(&self) -> String {
        "cortex".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0;
        while ctx.running() {
            if self.cortex.config().enabled
                && let Some(packet) = self.fused_frames.latest()
                && self.last_scene_frame != Some(packet.scene.stamp.frame_id)
            {
                self.last_scene_frame = Some(packet.scene.stamp.frame_id);
                let delta = packet.scene.events.delta_after(self.event_cursor);
                let scene_event_gap = delta.gap.is_some();
                self.event_cursor = delta.next_cursor;
                let events = delta.events.into_iter().cloned().collect::<Vec<_>>();
                let input = CortexInput {
                    previous_scene: self.previous_scene.as_deref(),
                    current_scene: packet.scene.as_ref(),
                    scene_events: &events,
                    scene_event_gap,
                    config: &self.config,
                    now_unix_ms: packet.scene.stamp.published_at_unix_ms,
                };
                let policy = self.cortex.attention_policy(CortexInput {
                    previous_scene: input.previous_scene,
                    current_scene: input.current_scene,
                    scene_events: input.scene_events,
                    scene_event_gap: input.scene_event_gap,
                    config: input.config,
                    now_unix_ms: input.now_unix_ms,
                });
                let frame = self.cortex.update(input);
                self.previous_scene = Some(Arc::clone(&packet.scene));
                self.cortex_frames.publish(frame);
                self.policies.publish(policy);
                ticks += 1;
            }
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        AgentHealth {
            name: self.name(),
            ticks,
            errors: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_defaults_disabled_when_constructed_directly() {
        let agent = CortexAgent::new(
            FusedFrameBus::new(),
            CortexFrameBus::new(),
            AttentionPolicyBus::new(),
            mvm_core::Config::default(),
            CortexConfig::default(),
        );
        assert_eq!(agent.name(), "cortex");
        assert!(!agent.cortex.config().enabled);
    }
}
