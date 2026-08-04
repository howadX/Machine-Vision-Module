use crate::{
    AttentionPolicy, CortexConfig, CortexFrame, CortexInput, EvidenceRef,
    affordance::AffordanceLayer, attention::AttentionLayer, events::EventInterpretationLayer,
    memory::EventMemoryLayer, predict::PredictiveLayer, report::ReportLayer,
    self_model::SelfModelLayer, uncertainty::UncertaintyLayer,
};

/// Owner of Cortex ordering and runtime-only temporal state. Feature logic is
/// intentionally absent until later Phase 18 slices.
pub struct VisualCortex {
    config: CortexConfig,
    predictor: PredictiveLayer,
    events: EventInterpretationLayer,
    memory: EventMemoryLayer,
    affordance: AffordanceLayer,
    attention: AttentionLayer,
    self_model: SelfModelLayer,
    uncertainty: UncertaintyLayer,
    report: ReportLayer,
}

impl VisualCortex {
    pub fn new(config: CortexConfig) -> Self {
        Self {
            config,
            predictor: PredictiveLayer,
            events: EventInterpretationLayer,
            memory: EventMemoryLayer,
            affordance: AffordanceLayer,
            attention: AttentionLayer,
            self_model: SelfModelLayer,
            uncertainty: UncertaintyLayer,
            report: ReportLayer,
        }
    }

    pub fn config(&self) -> CortexConfig {
        self.config
    }

    /// Pure proposal generation; scheduler acceptance and expert execution are
    /// runtime-owned and cannot alter this Cortex state.
    pub fn attention_policy(&self, input: CortexInput<'_>) -> AttentionPolicy {
        self.events
            .policy(input.current_scene.stamp.frame_id, input.scene_events)
    }

    /// Interpret only canonical Fusion events. Cortex never performs a second
    /// scene diff: each asserted interpretation carries its event evidence.
    pub fn update(&mut self, input: CortexInput<'_>) -> CortexFrame {
        let _ = (
            &self.predictor,
            &self.events,
            &self.memory,
            &self.affordance,
            &self.attention,
            &self.self_model,
            &self.uncertainty,
            &self.report,
            input.previous_scene,
            input.config,
        );
        CortexFrame {
            timestamp_unix_ms: input.now_unix_ms,
            scene_frame_id: input.current_scene.stamp.frame_id,
            claims: self.events.interpret(input.scene_events),
            scene_event_refs: input
                .scene_events
                .iter()
                .map(|event| EvidenceRef {
                    scene_frame_id: event.source_frame_id,
                    entity: event.entity_keys.first().cloned(),
                    event_seq: Some(event.event_seq),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClaimStatus, CognitiveClaim, ReasonCode};
    use mvm_core::canonical::{
        EventTailView, FactEvidenceRef, FactReason, FactSource, FactStatus, FrameStamp,
        FusedSceneFrame, SceneEvent, SceneEventKind, SceneFact, SceneQuality,
    };

    fn scene() -> FusedSceneFrame {
        FusedSceneFrame {
            stamp: FrameStamp::new("test", 7, 1_000, 1_001).unwrap(),
            sources: Vec::new(),
            entities: Vec::new(),
            relations: Vec::new(),
            zones: Vec::new(),
            activities: Vec::new(),
            events: EventTailView::default(),
            anomaly: SceneFact::unknown(FactReason::MissingSource),
            quality: SceneQuality::default(),
        }
    }

    #[test]
    fn skeleton_emits_no_unevidenced_claims() {
        let scene = scene();
        let mut cortex = VisualCortex::new(CortexConfig::default());
        let frame = cortex.update(CortexInput {
            previous_scene: None,
            current_scene: &scene,
            scene_events: &[],
            config: &mvm_core::Config::default(),
            now_unix_ms: 1_002,
        });
        assert_eq!(frame.scene_frame_id, 7);
        assert!(frame.claims.is_empty());
        assert_eq!(
            serde_json::to_value(&frame).unwrap()["claims"],
            serde_json::json!([])
        );
    }

    #[test]
    fn missing_claim_cannot_become_negative_fact() {
        let claim = CognitiveClaim::<String>::withheld(ReasonCode::Skeleton, Vec::new());
        assert_eq!(claim.status, ClaimStatus::Withheld);
        assert!(claim.value.is_none());
        assert!(claim.confidence.is_none());
        assert!(claim.validate().is_ok());
    }

    #[test]
    fn asserted_claim_requires_evidence() {
        assert!(CognitiveClaim::asserted("forecast".to_string(), 0.8, Vec::new()).is_err());
    }

    #[test]
    fn interpretation_is_linked_to_a_canonical_event_not_a_scene_diff() {
        let scene = scene();
        let event = SceneEvent {
            event_seq: 12,
            source_frame_id: 7,
            source_timestamp_unix_ms: 1_001,
            kind: SceneEventKind::AnomalySpike,
            entity_keys: Vec::new(),
            fact: SceneFact::claimed(
                FactStatus::Temporal,
                true,
                0.8,
                vec![FactEvidenceRef {
                    source: FactSource::Fusion,
                    source_frame_id: 7,
                    entity: None,
                    event_seq: Some(12),
                }],
            )
            .unwrap(),
            payload: None,
        };
        let mut cortex = VisualCortex::new(CortexConfig::default());
        let frame = cortex.update(CortexInput {
            previous_scene: None,
            current_scene: &scene,
            scene_events: &[event],
            config: &mvm_core::Config::default(),
            now_unix_ms: 1_002,
        });
        assert_eq!(
            frame.claims[0].value.as_deref(),
            Some("anomaly_requires_attention")
        );
        assert_eq!(frame.claims[0].evidence[0].event_seq, Some(12));
    }

    #[test]
    fn unexpected_appearance_proposes_one_bounded_scene_refresh() {
        let scene = scene();
        let event = SceneEvent {
            event_seq: 13,
            source_frame_id: 7,
            source_timestamp_unix_ms: 1_001,
            kind: SceneEventKind::EntityAppeared,
            entity_keys: Vec::new(),
            fact: SceneFact::claimed(
                FactStatus::Temporal,
                true,
                0.8,
                vec![FactEvidenceRef {
                    source: FactSource::Fusion,
                    source_frame_id: 7,
                    entity: None,
                    event_seq: Some(13),
                }],
            )
            .unwrap(),
            payload: None,
        };
        let cortex = VisualCortex::new(CortexConfig::default());
        let policy = cortex.attention_policy(CortexInput {
            previous_scene: None,
            current_scene: &scene,
            scene_events: &[event],
            config: &mvm_core::Config::default(),
            now_unix_ms: 1_002,
        });
        assert_eq!(policy.requests.len(), 1);
        assert_eq!(policy.requests[0].max_extra_runs, 1);
        assert_eq!(policy.requests[0].evidence[0].event_seq, Some(13));
    }
}
