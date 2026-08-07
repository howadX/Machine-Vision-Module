use crate::{
    AttentionPolicy, CognitiveClaim, EvidenceRef, ObservationRequest, ObservationTarget, ReasonCode,
};
use mvm_core::canonical::{
    FactSource, FusedSceneFrame, SceneEvent, SceneEventKind, SourceFreshness,
};

/// Maps observed Fusion transitions to deliberately small, evidence-linked
/// interpretations. It does not diff frames or create scene events itself.
pub(crate) struct EventInterpretationLayer;

impl EventInterpretationLayer {
    pub(crate) fn policy(&self, scene: &FusedSceneFrame, events: &[SceneEvent]) -> AttentionPolicy {
        let scene_frame_id = scene.stamp.frame_id;
        let mut requests: Vec<_> = events
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
        let source_needs_refresh = |kind| {
            scene.sources.iter().any(|source| {
                source.source == kind
                    && matches!(
                        source.freshness,
                        SourceFreshness::Stale | SourceFreshness::Unavailable
                    )
            })
        };
        let scene_source_needs_refresh = source_needs_refresh(FactSource::Scene);
        if scene_source_needs_refresh
            && !requests
                .iter()
                .any(|request| request.target == ObservationTarget::Scene)
        {
            requests.push(ObservationRequest {
                request_id: (1_u64 << 63) | scene_frame_id,
                source_scene_frame_id: scene_frame_id,
                target: ObservationTarget::Scene,
                reason: ReasonCode::StaleSource,
                evidence: vec![EvidenceRef {
                    scene_frame_id,
                    entity: None,
                    event_seq: None,
                }],
                priority: 224,
                max_extra_runs: 1,
                expires_at_frame: scene_frame_id.saturating_add(2),
            });
        }
        if source_needs_refresh(FactSource::Depth) {
            requests.push(ObservationRequest {
                request_id: (1_u64 << 62) | scene_frame_id,
                source_scene_frame_id: scene_frame_id,
                target: ObservationTarget::Depth,
                reason: ReasonCode::StaleSource,
                evidence: vec![EvidenceRef {
                    scene_frame_id,
                    entity: None,
                    event_seq: None,
                }],
                priority: 208,
                max_extra_runs: 1,
                expires_at_frame: scene_frame_id.saturating_add(2),
            });
        }
        requests.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.request_id.cmp(&right.request_id))
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::canonical::{
        EventTailView, FactReason, FactSource, FrameStamp, FusedSceneFrame, SceneFact,
        SceneQuality, SourceFreshness, SourceStamp,
    };

    fn scene_with_scene_source(freshness: SourceFreshness) -> FusedSceneFrame {
        FusedSceneFrame {
            stamp: FrameStamp::new("run", 12, 1_200, 1_201).unwrap(),
            sources: vec![SourceStamp {
                source: FactSource::Scene,
                source_frame_id: 8,
                produced_at_unix_ms: 900,
                age_frames: 4,
                freshness,
            }],
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
    fn stale_scene_source_proposes_one_deterministic_refresh() {
        let layer = EventInterpretationLayer;
        let scene = scene_with_scene_source(SourceFreshness::Stale);

        let first = layer.policy(&scene, &[]);
        let second = layer.policy(&scene, &[]);

        assert_eq!(first, second);
        assert_eq!(first.requests.len(), 1);
        assert_eq!(first.requests[0].target, ObservationTarget::Scene);
        assert_eq!(first.requests[0].reason, ReasonCode::StaleSource);
        assert_eq!(first.requests[0].max_extra_runs, 1);
        assert!(!first.requests[0].evidence.is_empty());
    }

    #[test]
    fn stale_depth_source_proposes_bounded_depth_refresh() {
        let layer = EventInterpretationLayer;
        let mut scene = scene_with_scene_source(SourceFreshness::Fresh);
        scene.sources.push(SourceStamp {
            source: FactSource::Depth,
            source_frame_id: 1,
            produced_at_unix_ms: 500,
            age_frames: 11,
            freshness: SourceFreshness::Stale,
        });

        let policy = layer.policy(&scene, &[]);

        assert_eq!(policy.requests.len(), 1);
        assert_eq!(policy.requests[0].target, ObservationTarget::Depth);
        assert_eq!(policy.requests[0].reason, ReasonCode::StaleSource);
        assert_eq!(policy.requests[0].max_extra_runs, 1);
    }
}
