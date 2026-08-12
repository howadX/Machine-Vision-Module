use crate::{
    AffordanceKind, AffordanceResult, CognitiveClaim, EvidenceRef, MemoryContext, ReasonCode,
};
use mvm_core::canonical::{
    FactEvidenceRef, FusedSceneFrame, HandInteractionFact, HandInteractionKind, SceneActivity,
    SceneEntityKind,
};

pub(crate) struct AffordanceLayer;

impl AffordanceLayer {
    pub(crate) fn build(
        &self,
        scene: &FusedSceneFrame,
        memory: &MemoryContext,
    ) -> Vec<AffordanceResult> {
        let mut results = Vec::new();
        for interaction in &scene.hand_interactions {
            if !asserted(&interaction.fact) {
                continue;
            }
            let evidence = interaction_evidence(scene, interaction);
            let confidence = interaction.fact.confidence.unwrap_or(0.0);
            if interaction.kind == HandInteractionKind::TouchingCandidate {
                results.push(asserted_result(
                    AffordanceKind::Reachable,
                    interaction,
                    confidence,
                    evidence.clone(),
                ));
                results.push(asserted_result(
                    AffordanceKind::Interactable,
                    interaction,
                    confidence,
                    evidence.clone(),
                ));
            }
            results.push(conditional_result(
                AffordanceKind::Held,
                interaction,
                active_activity(scene, "holding_drink", interaction),
                evidence.clone(),
            ));
            results.push(conditional_result(
                AffordanceKind::LikelyUsed,
                interaction,
                active_using_activity(scene, interaction)
                    || recently_interacted(memory, interaction),
                evidence.clone(),
            ));
            // A head-pose observation has no target geometry in Phase 18. It
            // therefore cannot become gaze, attention, or intent.
            results.push(withheld_result(
                AffordanceKind::Attended,
                interaction,
                ReasonCode::InsufficientEvidence,
                evidence.clone(),
            ));
            // Occupancy/free-space evidence belongs to Phase 19. Unknown space
            // is never interpreted as blocked or passable here.
            results.push(withheld_result(
                AffordanceKind::Blocked,
                interaction,
                ReasonCode::MissingSource,
                evidence,
            ));
        }
        for entity in &scene.entities {
            if let SceneEntityKind::Object(facts) = &entity.kind
                && facts.class_name.value.as_deref() == Some("chair")
            {
                results.push(AffordanceResult {
                    kind: AffordanceKind::Sittable,
                    target: Some(entity.key.clone()),
                    actor: None,
                    claim: CognitiveClaim::withheld(
                        ReasonCode::InsufficientEvidence,
                        fact_evidence(scene, &entity.key, &facts.class_name.evidence),
                    ),
                });
            }
        }
        results.sort_by(|a, b| (&a.target, &a.actor, a.kind).cmp(&(&b.target, &b.actor, b.kind)));
        results.dedup_by(|a, b| a.kind == b.kind && a.target == b.target && a.actor == b.actor);
        debug_assert!(results.iter().all(|result| result.validate().is_ok()));
        results
    }
}

fn asserted(fact: &mvm_core::canonical::SceneFact<bool>) -> bool {
    fact.status.is_asserted()
        && fact.value == Some(true)
        && fact
            .confidence
            .is_some_and(|value| value.is_finite() && value >= 0.5)
}

fn asserted_result(
    kind: AffordanceKind,
    interaction: &HandInteractionFact,
    confidence: f32,
    evidence: Vec<EvidenceRef>,
) -> AffordanceResult {
    AffordanceResult {
        kind,
        target: Some(interaction.target_entity.clone()),
        actor: Some(interaction.person.clone()),
        claim: CognitiveClaim::asserted(true, confidence, evidence)
            .expect("hand interaction evidence supports asserted affordance"),
    }
}

fn conditional_result(
    kind: AffordanceKind,
    interaction: &HandInteractionFact,
    supported: bool,
    evidence: Vec<EvidenceRef>,
) -> AffordanceResult {
    if supported {
        asserted_result(
            kind,
            interaction,
            interaction.fact.confidence.unwrap_or(0.0),
            evidence,
        )
    } else {
        withheld_result(
            kind,
            interaction,
            ReasonCode::InsufficientEvidence,
            evidence,
        )
    }
}

fn withheld_result(
    kind: AffordanceKind,
    interaction: &HandInteractionFact,
    reason: ReasonCode,
    evidence: Vec<EvidenceRef>,
) -> AffordanceResult {
    AffordanceResult {
        kind,
        target: Some(interaction.target_entity.clone()),
        actor: Some(interaction.person.clone()),
        claim: CognitiveClaim::withheld(reason, evidence),
    }
}

fn active_activity(
    scene: &FusedSceneFrame,
    label: &str,
    interaction: &HandInteractionFact,
) -> bool {
    scene.activities.iter().any(|activity| {
        activity.label == label && asserted(&activity.fact) && contains_pair(activity, interaction)
    })
}

fn active_using_activity(scene: &FusedSceneFrame, interaction: &HandInteractionFact) -> bool {
    scene.activities.iter().any(|activity| {
        activity.label.starts_with("using_object(")
            && asserted(&activity.fact)
            && contains_pair(activity, interaction)
    })
}

fn contains_pair(activity: &SceneActivity, interaction: &HandInteractionFact) -> bool {
    activity.participants.contains(&interaction.person)
        && activity.participants.contains(&interaction.target_entity)
}

fn recently_interacted(memory: &MemoryContext, interaction: &HandInteractionFact) -> bool {
    memory.recent.iter().any(|event| {
        matches!(
            event.kind,
            mvm_core::canonical::SceneEventKind::ObjectTouched
                | mvm_core::canonical::SceneEventKind::ObjectHeld
        ) && event.entities.contains(&interaction.person)
            && event.entities.contains(&interaction.target_entity)
    })
}

fn interaction_evidence(
    scene: &FusedSceneFrame,
    interaction: &HandInteractionFact,
) -> Vec<EvidenceRef> {
    fact_evidence(scene, &interaction.person, &interaction.fact.evidence)
}

fn fact_evidence(
    scene: &FusedSceneFrame,
    entity: &mvm_core::canonical::EntityKey,
    evidence: &[FactEvidenceRef],
) -> Vec<EvidenceRef> {
    if evidence.is_empty() {
        return vec![EvidenceRef {
            scene_frame_id: scene.stamp.frame_id,
            entity: Some(entity.clone()),
            event_seq: None,
        }];
    }
    evidence
        .iter()
        .map(|item| EvidenceRef {
            scene_frame_id: item.source_frame_id,
            entity: item.entity.clone().or_else(|| Some(entity.clone())),
            event_seq: item.event_seq,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClaimStatus;
    use mvm_core::BBox;
    use mvm_core::canonical::{
        EntityKey, EventTailView, FactEvidenceRef, FactReason, FactSource, FactStatus, FrameStamp,
        FusedSceneFrame, HandInteractionFact, HandInteractionKind, HandSide, HeadFacing,
        HeadOrientationFact, ObjectFacts, PersonSensoryFacts, SceneActivity, SceneEntity,
        SceneEntityKind, SceneFact, SceneQuality, TrackDomain,
    };
    use serde::Deserialize;

    #[test]
    fn affordance_result_preserves_withheld_status_without_negative_value() {
        let result = AffordanceResult {
            kind: AffordanceKind::Blocked,
            target: None,
            actor: None,
            claim: CognitiveClaim::<bool>::withheld(ReasonCode::MissingSource, Vec::new()),
        };
        assert_eq!(result.claim.status, ClaimStatus::Withheld);
        assert!(result.claim.value.is_none());
        assert!(result.validate().is_ok());
    }

    #[derive(Deserialize)]
    struct Fixture {
        status: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        kind: AffordanceKind,
        expected_status: ClaimStatus,
    }

    fn key(domain: TrackDomain, id: u32) -> EntityKey {
        EntityKey {
            run_id: "affordance-test".into(),
            track_domain: domain,
            track_id: id,
            incarnation: 0,
        }
    }

    fn evidence(key: &EntityKey) -> Vec<FactEvidenceRef> {
        vec![FactEvidenceRef {
            source: FactSource::Human,
            source_frame_id: 7,
            entity: Some(key.clone()),
            event_seq: None,
        }]
    }

    fn fact<T>(key: &EntityKey, value: T) -> SceneFact<T> {
        SceneFact::claimed(FactStatus::Observed, value, 0.9, evidence(key)).unwrap()
    }

    fn scene() -> FusedSceneFrame {
        let person = key(TrackDomain::Person, 10);
        let object = key(TrackDomain::Object, 20);
        FusedSceneFrame {
            stamp: FrameStamp::new("affordance-test", 7, 1_000, 1_001).unwrap(),
            sources: Vec::new(),
            entities: vec![
                SceneEntity {
                    key: person.clone(),
                    display_id: 10,
                    kind: SceneEntityKind::Person(mvm_core::canonical::PersonFacts {
                        gallery_identity: SceneFact::unknown(FactReason::MissingSource),
                        posture: SceneFact::unknown(FactReason::MissingSource),
                        emotion: SceneFact::unknown(FactReason::MissingSource),
                        sensory: Box::new(PersonSensoryFacts {
                            body_coverage: SceneFact::unknown(FactReason::MissingSource),
                            head_orientation: fact(
                                &person,
                                HeadOrientationFact {
                                    yaw_deg: Some(20.0),
                                    facing: Some(HeadFacing::Right),
                                },
                            ),
                            hands: SceneFact::unknown(FactReason::MissingSource),
                        }),
                    }),
                    existence: fact(&person, mvm_core::canonical::Existence::Present),
                    bbox: fact(
                        &person,
                        BBox {
                            x1: 0.1,
                            y1: 0.1,
                            x2: 0.4,
                            y2: 0.9,
                        },
                    ),
                    position_3d: SceneFact::unknown(FactReason::Uncalibrated),
                    geometry: Box::default(),
                    kinematics: SceneFact::unknown(FactReason::MissingSource),
                },
                SceneEntity {
                    key: object.clone(),
                    display_id: 20,
                    kind: SceneEntityKind::Object(ObjectFacts {
                        class_name: fact(&object, "cup".to_owned()),
                        remembered: false,
                    }),
                    existence: fact(&object, mvm_core::canonical::Existence::Present),
                    bbox: fact(
                        &object,
                        BBox {
                            x1: 0.4,
                            y1: 0.2,
                            x2: 0.6,
                            y2: 0.6,
                        },
                    ),
                    position_3d: SceneFact::unknown(FactReason::Uncalibrated),
                    geometry: Box::default(),
                    kinematics: SceneFact::unknown(FactReason::MissingSource),
                },
            ],
            relations: Vec::new(),
            hand_interactions: vec![HandInteractionFact {
                person: person.clone(),
                hand_side: HandSide::Right,
                target_entity: object.clone(),
                kind: HandInteractionKind::TouchingCandidate,
                fact: SceneFact::claimed(FactStatus::Derived, true, 0.9, evidence(&person))
                    .unwrap(),
            }],
            zones: Vec::new(),
            activities: vec![SceneActivity {
                label: "holding_drink".into(),
                participants: vec![person, object],
                fact: SceneFact::claimed(
                    FactStatus::Temporal,
                    true,
                    0.9,
                    vec![FactEvidenceRef {
                        source: FactSource::Activity,
                        source_frame_id: 7,
                        entity: None,
                        event_seq: None,
                    }],
                )
                .unwrap(),
            }],
            events: EventTailView::default(),
            anomaly: SceneFact::unknown(FactReason::MissingSource),
            quality: SceneQuality::default(),
        }
    }

    #[test]
    fn affordance_v1_exceeds_phase_gate_on_synthetic_replay() {
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../../benchmark/scenarios/017.json")).unwrap();
        assert_eq!(fixture.status, "synthetic_replay");
        let results = AffordanceLayer.build(&scene(), &MemoryContext::default());
        let correct = fixture
            .cases
            .iter()
            .filter(|case| {
                results.iter().any(|result| {
                    result.kind == case.kind && result.claim.status == case.expected_status
                })
            })
            .count();
        let accuracy = correct as f32 / fixture.cases.len() as f32;
        assert_eq!(correct, fixture.cases.len());
        assert!(accuracy > 0.85, "affordance accuracy={accuracy}");
    }
}
