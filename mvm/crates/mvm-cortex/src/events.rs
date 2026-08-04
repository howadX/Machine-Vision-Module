use crate::{
    AttentionPolicy, CognitiveClaim, EvidenceRef, ObservationRequest, ObservationTarget, ReasonCode,
};
use mvm_core::canonical::{SceneEvent, SceneEventKind};

/// Maps observed Fusion transitions to deliberately small, evidence-linked
/// interpretations. It does not diff frames or create scene events itself.
pub(crate) struct EventInterpretationLayer;

impl EventInterpretationLayer {
    pub(crate) fn policy(&self, scene_frame_id: u64, events: &[SceneEvent]) -> AttentionPolicy {
        let requests = events
            .iter()
            .filter(|event| event.kind == SceneEventKind::EntityAppeared)
            .map(|event| ObservationRequest {
                request_id: event.event_seq,
                source_scene_frame_id: scene_frame_id,
                target: ObservationTarget::Scene,
                reason: ReasonCode::UnexpectedAppearance,
                evidence: event
                    .entity_keys
                    .iter()
                    .map(|entity| EvidenceRef {
                        scene_frame_id: event.source_frame_id,
                        entity: Some(entity.clone()),
                        event_seq: Some(event.event_seq),
                    })
                    .chain(event.entity_keys.is_empty().then_some(EvidenceRef {
                        scene_frame_id: event.source_frame_id,
                        entity: None,
                        event_seq: Some(event.event_seq),
                    }))
                    .collect(),
                priority: 255,
                max_extra_runs: 1,
                expires_at_frame: scene_frame_id.saturating_add(2),
            })
            .collect();
        AttentionPolicy {
            source_scene_frame_id: scene_frame_id,
            requests,
        }
    }

    pub(crate) fn interpret(&self, events: &[SceneEvent]) -> Vec<CognitiveClaim<String>> {
        events
            .iter()
            .filter_map(|event| {
                let value = match event.kind {
                    SceneEventKind::EntityAppeared | SceneEventKind::PersonEntered => {
                        "unexpected_appearance"
                    }
                    SceneEventKind::ObjectTouched
                    | SceneEventKind::ObjectHeld
                    | SceneEventKind::ObjectReleased => "high_salience_interaction",
                    SceneEventKind::AnomalySpike => "anomaly_requires_attention",
                    _ => return None,
                };
                let evidence = event
                    .entity_keys
                    .iter()
                    .map(|entity| EvidenceRef {
                        scene_frame_id: event.source_frame_id,
                        entity: Some(entity.clone()),
                        event_seq: Some(event.event_seq),
                    })
                    .chain(event.entity_keys.is_empty().then_some(EvidenceRef {
                        scene_frame_id: event.source_frame_id,
                        entity: None,
                        event_seq: Some(event.event_seq),
                    }))
                    .collect();
                CognitiveClaim::asserted(
                    value.to_owned(),
                    event.fact.confidence.unwrap_or(0.0),
                    evidence,
                )
                .ok()
            })
            .collect()
    }
}
