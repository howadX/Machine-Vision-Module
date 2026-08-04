use crate::{CognitiveClaim, EvidenceRef};
use mvm_core::canonical::{SceneEvent, SceneEventKind};

/// Maps observed Fusion transitions to deliberately small, evidence-linked
/// interpretations. It does not diff frames or create scene events itself.
pub(crate) struct EventInterpretationLayer;

impl EventInterpretationLayer {
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
