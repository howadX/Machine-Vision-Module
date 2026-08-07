use crate::{
    AttentionPolicy, CortexConfig, CortexFrame, CortexInput, EvidenceRef, ObservationRequest,
    ObservationTarget, ReasonCode, UncertaintyCode, affordance::AffordanceLayer,
    attention::AttentionLayer, events::EventInterpretationLayer, memory::EventMemoryLayer,
    predict::PredictiveLayer, report::ReportLayer, self_model::SelfModelLayer,
    uncertainty::UncertaintyLayer,
};

/// Owner of Cortex ordering and runtime-only temporal state.
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
            predictor: PredictiveLayer::default(),
            events: EventInterpretationLayer,
            memory: EventMemoryLayer::default(),
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
        self.events.policy(input.current_scene, input.scene_events)
    }

    /// Derive bounded advisory requests from this tick's actual cognitive
    /// output. This runs after `update`: uncertainty and prediction errors may
    /// request another observation, but cannot alter the fused facts that
    /// produced them.
    pub fn attention_policy_for_frame(
        &self,
        input: CortexInput<'_>,
        frame: &CortexFrame,
    ) -> AttentionPolicy {
        let mut policy = self.attention_policy(input);
        let scene_frame_id = frame.scene_frame_id;
        if !policy
            .requests
            .iter()
            .any(|request| request.target == ObservationTarget::Scene)
            && frame.uncertainty.claims.iter().any(|claim| {
                claim.code == UncertaintyCode::SourceFreshness
                    && claim.reasons.iter().any(|reason| {
                        matches!(reason, ReasonCode::StaleSource | ReasonCode::CoastedSource)
                    })
            })
        {
            policy.requests.push(ObservationRequest {
                request_id: (1_u64 << 61) | scene_frame_id,
                source_scene_frame_id: scene_frame_id,
                target: ObservationTarget::Scene,
                reason: ReasonCode::HighUncertaintySceneSource,
                evidence: vec![EvidenceRef {
                    scene_frame_id,
                    entity: None,
                    event_seq: None,
                }],
                priority: 192,
                max_extra_runs: 1,
                expires_at_frame: scene_frame_id.saturating_add(2),
            });
        }
        if !policy
            .requests
            .iter()
            .any(|request| request.target == ObservationTarget::Depth)
            && !frame.prediction_errors.is_empty()
        {
            let evidence = frame
                .prediction_errors
                .iter()
                .flat_map(|error| error.evidence.iter().cloned())
                .collect::<Vec<_>>();
            policy.requests.push(ObservationRequest {
                request_id: (1_u64 << 60) | scene_frame_id,
                source_scene_frame_id: scene_frame_id,
                target: ObservationTarget::Depth,
                reason: ReasonCode::Other("prediction_error_reobservation".into()),
                evidence: if evidence.is_empty() {
                    vec![EvidenceRef {
                        scene_frame_id,
                        entity: None,
                        event_seq: None,
                    }]
                } else {
                    evidence
                },
                priority: 208,
                max_extra_runs: 1,
                expires_at_frame: scene_frame_id.saturating_add(2),
            });
        }
        policy.requests.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.request_id.cmp(&right.request_id))
        });
        policy
    }

    /// Interpret only canonical Fusion events. Cortex never performs a second
    /// scene diff: each asserted interpretation carries its event evidence.
    pub fn update(&mut self, input: CortexInput<'_>) -> CortexFrame {
        let (forecasts, prediction_errors) = self.predictor.update(input.current_scene);
        let forecast = forecasts
            .iter()
            .find(|forecast| forecast.horizon_ms == 1_000)
            .cloned();
        let memory = self
            .memory
            .update(input.current_scene, input.scene_events, input.now_unix_ms);
        let self_model = self.self_model.build(input.current_scene, input.config);
        let uncertainty = self.uncertainty.build(
            input.current_scene,
            &self_model,
            input.config,
            input.scene_event_gap,
        );
        let affordances = self.affordance.build(input.current_scene, &memory);
        let attention = self.attention.build(
            input.current_scene,
            &prediction_errors,
            &memory,
            &affordances,
            &uncertainty,
            &self_model,
        );
        let _ = (
            &self.events,
            &self.affordance,
            &self.report,
            input.previous_scene,
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
            forecast,
            forecasts,
            prediction_errors,
            memory,
            self_model,
            uncertainty,
            affordances,
            attention,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ClaimStatus, CognitiveClaim, ObservationTarget, PredictionError, PredictionErrorKind,
        ReasonCode,
    };
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
            hand_interactions: Vec::new(),
            zones: Vec::new(),
            activities: Vec::new(),
            spatial: Default::default(),
            events: EventTailView::default(),
            anomaly: SceneFact::unknown(FactReason::MissingSource),
            quality: SceneQuality::default(),
        }
    }

    #[test]
    fn empty_scene_emits_no_unevidenced_interpretations() {
        let scene = scene();
        let mut cortex = VisualCortex::new(CortexConfig::default());
        let frame = cortex.update(CortexInput {
            previous_scene: None,
            current_scene: &scene,
            scene_events: &[],
            scene_event_gap: false,
            config: &mvm_core::Config::default(),
            now_unix_ms: 1_002,
        });
        assert_eq!(frame.scene_frame_id, 7);
        assert_eq!(
            frame
                .forecasts
                .iter()
                .map(|forecast| forecast.horizon_ms)
                .collect::<Vec<_>>(),
            vec![1_000, 2_000]
        );
        assert_eq!(
            frame.forecast.as_ref().map(|forecast| forecast.horizon_ms),
            Some(1_000)
        );
        assert!(frame.claims.is_empty());
        assert_eq!(frame.attention.source_scene_frame_id, 7);
        assert!(frame.attention.validate().is_ok());
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
    fn older_cortex_frame_without_memory_remains_readable() {
        let frame: CortexFrame = serde_json::from_value(serde_json::json!({
            "timestamp_unix_ms": 1_002,
            "scene_frame_id": 7,
            "claims": []
        }))
        .unwrap();
        assert!(frame.memory.recent.is_empty());
        assert!(frame.memory.relevant.is_empty());
        assert!(frame.forecast.is_none());
        assert!(frame.forecasts.is_empty());
        assert!(frame.prediction_errors.is_empty());
        assert_eq!(frame.self_model.source_scene_frame_id, 0);
        assert_eq!(frame.uncertainty.source_scene_frame_id, 0);
    }

    #[test]
    fn asserted_claim_requires_evidence() {
        assert!(CognitiveClaim::asserted("forecast".to_string(), 0.8, Vec::new()).is_err());
    }

    #[test]
    fn prediction_error_creates_a_bounded_depth_reobservation_request() {
        let scene = scene();
        let config = mvm_core::Config::default();
        let mut cortex = VisualCortex::new(CortexConfig::default());
        let mut frame = cortex.update(CortexInput {
            previous_scene: None,
            current_scene: &scene,
            scene_events: &[],
            scene_event_gap: false,
            config: &config,
            now_unix_ms: 1_002,
        });
        frame.prediction_errors.push(PredictionError {
            kind: PredictionErrorKind::BboxDrift,
            forecast_source_scene_frame_id: 6,
            forecast_horizon_ms: 1_000,
            observed_scene_frame_id: 7,
            entities: Vec::new(),
            magnitude: Some(1.0),
            confidence: 0.9,
            evidence: vec![EvidenceRef {
                scene_frame_id: 7,
                entity: None,
                event_seq: None,
            }],
        });
        let policy = cortex.attention_policy_for_frame(
            CortexInput {
                previous_scene: None,
                current_scene: &scene,
                scene_events: &[],
                scene_event_gap: false,
                config: &config,
                now_unix_ms: 1_002,
            },
            &frame,
        );
        assert!(policy.requests.iter().any(|request| {
            request.target == ObservationTarget::Depth
                && request.reason == ReasonCode::Other("prediction_error_reobservation".into())
                && request.max_extra_runs == 1
                && request.expires_at_frame == 9
        }));
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
            scene_event_gap: false,
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
            scene_event_gap: false,
            config: &mvm_core::Config::default(),
            now_unix_ms: 1_002,
        });
        assert_eq!(policy.requests.len(), 1);
        assert_eq!(policy.requests[0].max_extra_runs, 1);
        assert_eq!(policy.requests[0].evidence[0].event_seq, Some(13));
    }
}
