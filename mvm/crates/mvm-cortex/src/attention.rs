use crate::{
    AffordanceResult, AttentionBreakdown, AttentionMap, AttentionScore, AttentionTarget,
    ClaimStatus, EvidenceRef, MemoryContext, PredictionError, UncertaintyReport, VisualSelfModel,
};
use mvm_core::canonical::{
    FactEvidenceRef, FusedSceneFrame, SceneEntityKind, SceneEventKind, SceneFact,
};
use std::collections::BTreeMap;

const TOP_K: usize = 3;
const RECENT_EVENT_WINDOW_MS: i64 = 30_000;

/// Deterministic read-only ranking over already-published cognitive inputs.
/// It does not create observation requests; the bounded advisory policy remains
/// the explicit follow-up scope of Phase 18.8b.
pub(crate) struct AttentionLayer;

impl AttentionLayer {
    #[allow(
        clippy::too_many_arguments,
        reason = "Phase 18.8 has a fixed, explicit input contract"
    )]
    pub(crate) fn build(
        &self,
        scene: &FusedSceneFrame,
        prediction_errors: &[PredictionError],
        memory: &MemoryContext,
        affordances: &[AffordanceResult],
        uncertainty: &UncertaintyReport,
        self_model: &VisualSelfModel,
    ) -> AttentionMap {
        let mut candidates = BTreeMap::<AttentionTarget, Candidate>::new();

        for error in prediction_errors {
            let value =
                error.magnitude.unwrap_or(0.5).clamp(0.0, 1.0) * error.confidence.clamp(0.0, 1.0);
            for target in targets_for_entities(&error.entities) {
                add(
                    &mut candidates,
                    target,
                    Component::PredictionError,
                    value,
                    &error.evidence,
                );
            }
        }

        for event in &memory.recent {
            let age_ms = scene
                .stamp
                .captured_at_unix_ms
                .saturating_sub(event.source_timestamp_unix_ms)
                .max(0);
            let recency = 1.0 - (age_ms as f32 / RECENT_EVENT_WINDOW_MS as f32).clamp(0.0, 1.0);
            let evidence = event_evidence(event);
            for target in targets_for_entities(&event.entities) {
                add(
                    &mut candidates,
                    target.clone(),
                    Component::EventRecent,
                    recency * event.confidence.clamp(0.0, 1.0),
                    &evidence,
                );
                if matches!(
                    event.kind,
                    SceneEventKind::EntityAppeared
                        | SceneEventKind::EntityDisappeared
                        | SceneEventKind::PersonEntered
                        | SceneEventKind::PersonExited
                ) {
                    add(
                        &mut candidates,
                        target,
                        Component::Novelty,
                        event.confidence.clamp(0.0, 1.0),
                        &evidence,
                    );
                }
            }
        }

        for activity in &scene.activities {
            if asserted(&activity.fact) {
                let evidence = fact_evidence(scene, None, &activity.fact.evidence);
                let value = activity.fact.confidence.unwrap_or(0.0);
                for target in targets_for_entities(&activity.participants) {
                    add(
                        &mut candidates,
                        target,
                        Component::Activity,
                        value,
                        &evidence,
                    );
                }
            }
        }

        if asserted(&scene.anomaly) {
            add(
                &mut candidates,
                AttentionTarget::Scene,
                Component::Anomaly,
                scene.anomaly.value.unwrap_or(0.0).clamp(0.0, 1.0)
                    * scene.anomaly.confidence.unwrap_or(0.0),
                &fact_evidence(scene, None, &scene.anomaly.evidence),
            );
        }

        for entity in &scene.entities {
            if let SceneEntityKind::Person(person) = &entity.kind
                && asserted(&person.gallery_identity)
            {
                add(
                    &mut candidates,
                    AttentionTarget::Entity(entity.key.clone()),
                    Component::Identity,
                    person.gallery_identity.confidence.unwrap_or(0.0),
                    &fact_evidence(scene, Some(&entity.key), &person.gallery_identity.evidence),
                );
            }
        }

        for affordance in affordances {
            if affordance.claim.status == ClaimStatus::Asserted
                && affordance.claim.value == Some(true)
            {
                let value = affordance.claim.confidence.unwrap_or(0.0);
                let mut targeted = false;
                if let Some(target) = &affordance.target {
                    add(
                        &mut candidates,
                        AttentionTarget::Entity(target.clone()),
                        Component::Affordance,
                        value,
                        &affordance.claim.evidence,
                    );
                    targeted = true;
                }
                if let Some(actor) = &affordance.actor {
                    add(
                        &mut candidates,
                        AttentionTarget::Entity(actor.clone()),
                        Component::Affordance,
                        value,
                        &affordance.claim.evidence,
                    );
                    targeted = true;
                }
                if !targeted {
                    add(
                        &mut candidates,
                        AttentionTarget::Scene,
                        Component::Affordance,
                        value,
                        &affordance.claim.evidence,
                    );
                }
            }
        }

        for claim in &uncertainty.claims {
            let target = claim
                .entity
                .as_ref()
                .cloned()
                .map(AttentionTarget::Entity)
                .or_else(|| claim.zone_id.map(AttentionTarget::Zone))
                .unwrap_or(AttentionTarget::Scene);
            add(
                &mut candidates,
                target,
                Component::Uncertainty,
                uncertainty_priority(claim.reasons.first()),
                &fallback_evidence(scene, &claim.evidence),
            );
        }

        // Zone visibility has no equivalent to a zone-rule violation in the
        // canonical frame. It can increase re-observation priority through the
        // uncertainty component, but `zone_violation` remains zero until
        // Fusion publishes an explicit observed violation contract.
        for zone in &self_model.zones {
            if zone.visibility.status != ClaimStatus::Asserted {
                add(
                    &mut candidates,
                    AttentionTarget::Zone(zone.zone_id),
                    Component::Uncertainty,
                    uncertainty_priority(zone.visibility.reasons.first()),
                    &fallback_evidence(scene, &zone.visibility.evidence),
                );
            }
        }

        let mut top_k = candidates
            .into_iter()
            .filter_map(|(target, candidate)| candidate.score(target))
            .collect::<Vec<_>>();
        top_k.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.target.cmp(&right.target))
        });
        top_k.truncate(TOP_K);

        let map = AttentionMap {
            source_scene_frame_id: scene.stamp.frame_id,
            top_k,
        };
        debug_assert!(map.validate().is_ok());
        map
    }
}

#[derive(Default)]
struct Candidate {
    breakdown: AttentionBreakdown,
    evidence: Vec<EvidenceRef>,
}

impl Candidate {
    fn score(self, target: AttentionTarget) -> Option<AttentionScore> {
        let breakdown = self.breakdown;
        let score = (0.20 * breakdown.novelty
            + 0.20 * breakdown.prediction_error
            + 0.15 * breakdown.uncertainty
            + 0.12 * breakdown.event_recent
            + 0.08 * breakdown.activity
            + 0.13 * breakdown.anomaly
            + 0.04 * breakdown.identity
            + 0.08 * breakdown.affordance
            + 0.0 * breakdown.zone_violation)
            .clamp(0.0, 1.0);
        (score > 0.0).then_some(AttentionScore {
            target,
            score,
            breakdown,
            evidence: self.evidence,
        })
    }
}

#[derive(Clone, Copy)]
enum Component {
    Novelty,
    PredictionError,
    Uncertainty,
    EventRecent,
    Activity,
    Anomaly,
    Identity,
    Affordance,
}

fn add(
    candidates: &mut BTreeMap<AttentionTarget, Candidate>,
    target: AttentionTarget,
    component: Component,
    value: f32,
    evidence: &[EvidenceRef],
) {
    let candidate = candidates.entry(target).or_default();
    let value = value.clamp(0.0, 1.0);
    match component {
        Component::Novelty => candidate.breakdown.novelty = candidate.breakdown.novelty.max(value),
        Component::PredictionError => {
            candidate.breakdown.prediction_error = candidate.breakdown.prediction_error.max(value);
        }
        Component::Uncertainty => {
            candidate.breakdown.uncertainty = candidate.breakdown.uncertainty.max(value);
        }
        Component::EventRecent => {
            candidate.breakdown.event_recent = candidate.breakdown.event_recent.max(value);
        }
        Component::Activity => {
            candidate.breakdown.activity = candidate.breakdown.activity.max(value)
        }
        Component::Anomaly => candidate.breakdown.anomaly = candidate.breakdown.anomaly.max(value),
        Component::Identity => {
            candidate.breakdown.identity = candidate.breakdown.identity.max(value)
        }
        Component::Affordance => {
            candidate.breakdown.affordance = candidate.breakdown.affordance.max(value);
        }
    }
    for item in evidence {
        if !candidate.evidence.contains(item) {
            candidate.evidence.push(item.clone());
        }
    }
}

fn asserted<T>(fact: &SceneFact<T>) -> bool {
    fact.status.is_asserted()
        && fact.value.is_some()
        && fact
            .confidence
            .is_some_and(|confidence| confidence.is_finite() && confidence > 0.0)
}

fn targets_for_entities(entities: &[mvm_core::canonical::EntityKey]) -> Vec<AttentionTarget> {
    if entities.is_empty() {
        vec![AttentionTarget::Scene]
    } else {
        entities
            .iter()
            .cloned()
            .map(AttentionTarget::Entity)
            .collect()
    }
}

fn event_evidence(event: &crate::MemoryEvent) -> Vec<EvidenceRef> {
    event
        .entities
        .iter()
        .cloned()
        .map(|entity| EvidenceRef {
            scene_frame_id: event.source_frame_id,
            entity: Some(entity),
            event_seq: Some(event.event_seq),
        })
        .chain(event.entities.is_empty().then_some(EvidenceRef {
            scene_frame_id: event.source_frame_id,
            entity: None,
            event_seq: Some(event.event_seq),
        }))
        .collect()
}

fn fact_evidence(
    scene: &FusedSceneFrame,
    fallback_entity: Option<&mvm_core::canonical::EntityKey>,
    evidence: &[FactEvidenceRef],
) -> Vec<EvidenceRef> {
    let evidence = evidence
        .iter()
        .map(|item| EvidenceRef {
            scene_frame_id: item.source_frame_id,
            entity: item.entity.clone().or_else(|| fallback_entity.cloned()),
            event_seq: item.event_seq,
        })
        .collect::<Vec<_>>();
    fallback_evidence(scene, &evidence)
}

fn fallback_evidence(scene: &FusedSceneFrame, evidence: &[EvidenceRef]) -> Vec<EvidenceRef> {
    if evidence.is_empty() {
        vec![EvidenceRef {
            scene_frame_id: scene.stamp.frame_id,
            entity: None,
            event_seq: None,
        }]
    } else {
        evidence.to_vec()
    }
}

fn uncertainty_priority(reason: Option<&crate::ReasonCode>) -> f32 {
    match reason {
        Some(crate::ReasonCode::EventGap)
        | Some(crate::ReasonCode::StaleSource)
        | Some(crate::ReasonCode::CoastedSource) => 0.8,
        Some(crate::ReasonCode::MissingSource) => 0.6,
        Some(crate::ReasonCode::Occluded) => 0.5,
        _ => 0.4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AffordanceKind, AffordanceResult, ClaimStatus, CognitiveClaim, EvidenceRef, MemoryEvent,
        PredictionErrorKind, ReasonCode, UncertainClaim, UncertaintyCode,
    };
    use mvm_core::canonical::{
        EventTailView, FactEvidenceRef, FactSource, FactStatus, FrameStamp, FusedSceneFrame,
        SceneActivity, SceneFact, SceneQuality,
    };
    use serde::Deserialize;

    fn scene() -> FusedSceneFrame {
        FusedSceneFrame {
            stamp: FrameStamp::new("attention", 7, 1_000, 1_001).unwrap(),
            sources: Vec::new(),
            entities: Vec::new(),
            relations: Vec::new(),
            hand_interactions: Vec::new(),
            zones: Vec::new(),
            activities: Vec::new(),
            spatial: Default::default(),
            events: EventTailView::default(),
            anomaly: SceneFact::claimed(
                FactStatus::Temporal,
                0.9,
                0.9,
                vec![FactEvidenceRef {
                    source: FactSource::Fusion,
                    source_frame_id: 7,
                    entity: None,
                    event_seq: None,
                }],
            )
            .unwrap(),
            quality: SceneQuality::default(),
        }
    }

    #[test]
    fn anomaly_and_prediction_error_rank_scene_with_evidence() {
        let scene = scene();
        let errors = vec![PredictionError {
            kind: PredictionErrorKind::EntityAppeared,
            forecast_source_scene_frame_id: 6,
            forecast_horizon_ms: 1_000,
            observed_scene_frame_id: 7,
            entities: Vec::new(),
            magnitude: Some(1.0),
            confidence: 0.8,
            evidence: vec![EvidenceRef {
                scene_frame_id: 7,
                entity: None,
                event_seq: None,
            }],
        }];

        let map = AttentionLayer.build(
            &scene,
            &errors,
            &MemoryContext::default(),
            &[],
            &UncertaintyReport::default(),
            &VisualSelfModel::default(),
        );

        assert_eq!(map.source_scene_frame_id, 7);
        assert_eq!(map.top_k.len(), 1);
        assert_eq!(map.top_k[0].target, AttentionTarget::Scene);
        assert!(map.top_k[0].breakdown.anomaly > 0.0);
        assert!(map.top_k[0].breakdown.prediction_error > 0.0);
        assert!(map.top_k[0].score > 0.0);
        assert!(!map.top_k[0].evidence.is_empty());
        assert!(map.validate().is_ok());
    }

    #[derive(Deserialize)]
    struct Fixture {
        status: String,
        cases: Vec<FixtureCase>,
    }

    #[derive(Deserialize)]
    struct FixtureCase {
        name: String,
        signal: String,
        expected_target: String,
    }

    #[test]
    fn top_three_attention_covers_phase_gate_on_benchmark() {
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../../benchmark/scenarios/018.json")).unwrap();
        assert_eq!(fixture.status, "synthetic_replay");
        let mut covered = 0usize;

        for case in &fixture.cases {
            let mut frame = scene();
            frame.anomaly = SceneFact::unknown(mvm_core::canonical::FactReason::MissingSource);
            let mut errors = Vec::new();
            let mut memory = MemoryContext::default();
            let mut affordances = Vec::new();
            let mut uncertainty = UncertaintyReport::default();

            match case.signal.as_str() {
                "anomaly" => frame.anomaly = scene().anomaly,
                "prediction_error" => errors.push(PredictionError {
                    kind: PredictionErrorKind::EntityAppeared,
                    forecast_source_scene_frame_id: 6,
                    forecast_horizon_ms: 1_000,
                    observed_scene_frame_id: 7,
                    entities: Vec::new(),
                    magnitude: Some(1.0),
                    confidence: 0.8,
                    evidence: vec![frame_evidence()],
                }),
                "novelty" => memory.recent.push(MemoryEvent {
                    event_seq: 1,
                    source_frame_id: 7,
                    source_timestamp_unix_ms: 1_000,
                    kind: SceneEventKind::EntityAppeared,
                    entities: Vec::new(),
                    confidence: 0.9,
                    object_classes: Vec::new(),
                    contact_duration_ms: None,
                }),
                "uncertainty" => uncertainty.claims.push(UncertainClaim {
                    code: UncertaintyCode::EventDelivery,
                    status: ClaimStatus::Withheld,
                    entity: None,
                    zone_id: None,
                    source: None,
                    reasons: vec![ReasonCode::EventGap],
                    evidence: vec![frame_evidence()],
                }),
                "activity" => frame.activities.push(SceneActivity {
                    label: "using_object(cup)".into(),
                    participants: Vec::new(),
                    fact: SceneFact::claimed(
                        FactStatus::Temporal,
                        true,
                        0.8,
                        vec![FactEvidenceRef {
                            source: FactSource::Fusion,
                            source_frame_id: 7,
                            entity: None,
                            event_seq: None,
                        }],
                    )
                    .unwrap(),
                }),
                "affordance" => affordances.push(AffordanceResult {
                    kind: AffordanceKind::Interactable,
                    target: None,
                    actor: None,
                    claim: CognitiveClaim::asserted(true, 0.8, vec![frame_evidence()]).unwrap(),
                }),
                other => panic!("unknown benchmark signal {other}"),
            }

            let map = AttentionLayer.build(
                &frame,
                &errors,
                &memory,
                &affordances,
                &uncertainty,
                &VisualSelfModel::default(),
            );
            let is_covered = case.expected_target == "scene"
                && map
                    .top_k
                    .iter()
                    .any(|score| score.target == AttentionTarget::Scene);
            covered += usize::from(is_covered);
            assert!(is_covered, "case {}", case.name);
        }

        let coverage = covered as f32 / fixture.cases.len() as f32;
        assert!(coverage > 0.8, "top-3 attention coverage={coverage}");
    }

    fn frame_evidence() -> EvidenceRef {
        EvidenceRef {
            scene_frame_id: 7,
            entity: None,
            event_seq: None,
        }
    }
}
