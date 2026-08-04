use crate::{
    ClaimStatus, EvidenceRef, ReasonCode, UncertainClaim, UncertaintyCode, UncertaintyReport,
    VisualSelfModel,
};
use mvm_core::Config;
use mvm_core::canonical::{
    EntityKey, Existence, FactEvidenceRef, FactReason, FactSource, FactStatus, FusedSceneFrame,
    SceneEntityKind, SceneFact, SourceFreshness,
};

pub(crate) struct UncertaintyLayer;

impl UncertaintyLayer {
    pub(crate) fn build(
        &self,
        scene: &FusedSceneFrame,
        self_model: &VisualSelfModel,
        config: &Config,
        scene_event_gap: bool,
    ) -> UncertaintyReport {
        let mut claims = Vec::new();
        claims.push(uncertain(
            UncertaintyCode::CameraWorldPose,
            ClaimStatus::Unknown,
            None,
            None,
            None,
            ReasonCode::MissingSource,
            Vec::new(),
        ));
        let mut metric_calibration = uncertain(
            UncertaintyCode::MetricCalibration,
            ClaimStatus::Withheld,
            None,
            None,
            None,
            ReasonCode::Uncalibrated,
            frame_evidence(scene),
        );
        metric_calibration.reasons.push(ReasonCode::OrdinalOnly);
        claims.push(metric_calibration);

        for source in &scene.sources {
            match source.freshness {
                SourceFreshness::Coasted => claims.push(uncertain(
                    UncertaintyCode::SourceFreshness,
                    ClaimStatus::Withheld,
                    None,
                    None,
                    Some(source.source),
                    ReasonCode::CoastedSource,
                    frame_evidence(scene),
                )),
                SourceFreshness::Stale => claims.push(uncertain(
                    UncertaintyCode::SourceFreshness,
                    ClaimStatus::Withheld,
                    None,
                    None,
                    Some(source.source),
                    ReasonCode::StaleSource,
                    frame_evidence(scene),
                )),
                SourceFreshness::Unavailable => claims.push(uncertain(
                    UncertaintyCode::MissingExpert,
                    ClaimStatus::Unknown,
                    None,
                    None,
                    Some(source.source),
                    ReasonCode::MissingSource,
                    Vec::new(),
                )),
                SourceFreshness::Fresh => {}
            }
        }
        if scene_event_gap || scene.quality.event_gap {
            claims.push(uncertain(
                UncertaintyCode::EventDelivery,
                ClaimStatus::Withheld,
                None,
                None,
                None,
                ReasonCode::EventGap,
                frame_evidence(scene),
            ));
        }

        for entity in &scene.entities {
            inspect_existence(scene, entity, &mut claims);
            inspect_position(scene, entity, &mut claims);
            if let SceneEntityKind::Person(person) = &entity.kind {
                inspect_fact(
                    scene,
                    &entity.key,
                    UncertaintyCode::Posture,
                    &person.posture,
                    0.5,
                    &mut claims,
                );
                inspect_fact(
                    scene,
                    &entity.key,
                    UncertaintyCode::Identity,
                    &person.gallery_identity,
                    config.identity_threshold,
                    &mut claims,
                );
                // Canonical PersonFacts currently preserves the emotion label
                // but not its probability distribution or entropy. Make that
                // observer limitation explicit instead of reconstructing it.
                claims.push(uncertain(
                    UncertaintyCode::EmotionDistribution,
                    ClaimStatus::Unknown,
                    Some(entity.key.clone()),
                    None,
                    Some(FactSource::Human),
                    ReasonCode::MissingSource,
                    fact_evidence(scene, &entity.key, &person.emotion.evidence),
                ));
            }
        }

        for zone in &self_model.zones {
            claims.push(uncertain(
                UncertaintyCode::ZoneVisibility,
                ClaimStatus::Withheld,
                None,
                Some(zone.zone_id),
                None,
                ReasonCode::InsufficientEvidence,
                zone.visibility.evidence.clone(),
            ));
        }
        if scene.quality.withheld_facts > 0 || scene.quality.unknown_facts > 0 {
            claims.push(uncertain(
                UncertaintyCode::SceneFactInventory,
                ClaimStatus::Withheld,
                None,
                None,
                None,
                ReasonCode::InsufficientEvidence,
                frame_evidence(scene),
            ));
        }

        claims.sort_by(|a, b| {
            (&a.code, &a.entity, &a.zone_id, &a.source)
                .cmp(&(&b.code, &b.entity, &b.zone_id, &b.source))
        });
        claims.dedup_by(|a, b| {
            a.code == b.code
                && a.entity == b.entity
                && a.zone_id == b.zone_id
                && a.source == b.source
        });
        debug_assert!(claims.iter().all(|claim| claim.validate().is_ok()));
        UncertaintyReport {
            source_scene_frame_id: scene.stamp.frame_id,
            claims,
        }
    }
}

fn inspect_existence(
    scene: &FusedSceneFrame,
    entity: &mvm_core::canonical::SceneEntity,
    claims: &mut Vec<UncertainClaim>,
) {
    let reason = match entity.existence.value {
        Some(Existence::Occluded) => Some(ReasonCode::Occluded),
        Some(Existence::Coasted) => Some(ReasonCode::StaleSource),
        Some(Existence::Present) if safely_asserted(&entity.existence, 0.5) => None,
        _ => Some(reason_for_fact(&entity.existence)),
    };
    if let Some(reason) = reason {
        claims.push(uncertain(
            UncertaintyCode::Existence,
            ClaimStatus::Withheld,
            Some(entity.key.clone()),
            None,
            None,
            reason,
            fact_evidence(scene, &entity.key, &entity.existence.evidence),
        ));
    }
}

fn inspect_position(
    scene: &FusedSceneFrame,
    entity: &mvm_core::canonical::SceneEntity,
    claims: &mut Vec<UncertainClaim>,
) {
    if !safely_asserted(&entity.position_3d, 0.5)
        || entity.position_3d.reason == Some(FactReason::Uncalibrated)
    {
        claims.push(uncertain(
            UncertaintyCode::EntityMetricPosition,
            ClaimStatus::Withheld,
            Some(entity.key.clone()),
            None,
            None,
            reason_for_fact(&entity.position_3d),
            fact_evidence(scene, &entity.key, &entity.position_3d.evidence),
        ));
    }
}

fn inspect_fact<T>(
    scene: &FusedSceneFrame,
    key: &EntityKey,
    code: UncertaintyCode,
    fact: &SceneFact<T>,
    threshold: f32,
    claims: &mut Vec<UncertainClaim>,
) {
    if safely_asserted(fact, threshold) {
        return;
    }
    claims.push(uncertain(
        code,
        if fact.status == FactStatus::Unknown {
            ClaimStatus::Unknown
        } else {
            ClaimStatus::Withheld
        },
        Some(key.clone()),
        None,
        None,
        reason_for_fact(fact),
        fact_evidence(scene, key, &fact.evidence),
    ));
}

fn safely_asserted<T>(fact: &SceneFact<T>, threshold: f32) -> bool {
    matches!(
        fact.status,
        FactStatus::Observed | FactStatus::Derived | FactStatus::Temporal
    ) && fact
        .confidence
        .is_some_and(|confidence| confidence.is_finite() && confidence >= threshold)
}

fn reason_for_fact<T>(fact: &SceneFact<T>) -> ReasonCode {
    match fact.reason.as_ref() {
        Some(FactReason::StaleSource) => ReasonCode::StaleSource,
        Some(FactReason::Remembered) => ReasonCode::RememberedSource,
        Some(FactReason::Occluded) => ReasonCode::Occluded,
        Some(FactReason::InsufficientCoverage) => ReasonCode::InsufficientEvidence,
        Some(FactReason::Uncalibrated) => ReasonCode::Uncalibrated,
        Some(FactReason::LowConfidence | FactReason::OutOfDistribution) => {
            ReasonCode::LowConfidence
        }
        Some(FactReason::MissingSource) => ReasonCode::MissingSource,
        Some(FactReason::Other(reason)) => ReasonCode::Other(reason.clone()),
        None if matches!(fact.status, FactStatus::Remembered) => ReasonCode::RememberedSource,
        None if matches!(fact.status, FactStatus::Weak) => ReasonCode::WeakSource,
        None if fact.confidence.is_some_and(|confidence| confidence < 0.5) => {
            ReasonCode::LowConfidence
        }
        None => ReasonCode::InsufficientEvidence,
    }
}

fn frame_evidence(scene: &FusedSceneFrame) -> Vec<EvidenceRef> {
    vec![EvidenceRef {
        scene_frame_id: scene.stamp.frame_id,
        entity: None,
        event_seq: None,
    }]
}

fn fact_evidence(
    scene: &FusedSceneFrame,
    key: &EntityKey,
    evidence: &[FactEvidenceRef],
) -> Vec<EvidenceRef> {
    if evidence.is_empty() {
        return vec![EvidenceRef {
            scene_frame_id: scene.stamp.frame_id,
            entity: Some(key.clone()),
            event_seq: None,
        }];
    }
    evidence
        .iter()
        .map(|item| EvidenceRef {
            scene_frame_id: item.source_frame_id,
            entity: item.entity.clone().or_else(|| Some(key.clone())),
            event_seq: item.event_seq,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn uncertain(
    code: UncertaintyCode,
    status: ClaimStatus,
    entity: Option<EntityKey>,
    zone_id: Option<u64>,
    source: Option<FactSource>,
    reason: ReasonCode,
    evidence: Vec<EvidenceRef>,
) -> UncertainClaim {
    UncertainClaim {
        code,
        status,
        entity,
        zone_id,
        source,
        reasons: vec![reason],
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{UncertaintyCode, self_model::SelfModelLayer};
    use mvm_core::canonical::{
        EntityKey, EventTailView, Existence, FactEvidenceRef, FactReason, FactSource, FactStatus,
        FrameStamp, FusedSceneFrame, GalleryIdentity, PersonFacts, SceneEntity, SceneEntityKind,
        SceneFact, SceneQuality, SourceFreshness, SourceStamp, TrackDomain,
    };
    use mvm_core::{BBox, Position3D};
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        status: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        signal: String,
        expected_code: UncertaintyCode,
        expected_present: bool,
    }

    fn evidence(frame_id: u64, key: &EntityKey) -> Vec<FactEvidenceRef> {
        vec![FactEvidenceRef {
            source: FactSource::Human,
            source_frame_id: frame_id,
            entity: Some(key.clone()),
            event_seq: None,
        }]
    }

    fn fact<T>(frame_id: u64, key: &EntityKey, value: T, confidence: f32) -> SceneFact<T> {
        SceneFact::claimed(
            FactStatus::Observed,
            value,
            confidence,
            evidence(frame_id, key),
        )
        .unwrap()
    }

    fn base_scene() -> FusedSceneFrame {
        let key = EntityKey {
            run_id: "uncertainty-test".into(),
            track_domain: TrackDomain::Person,
            track_id: 10,
            incarnation: 0,
        };
        FusedSceneFrame {
            stamp: FrameStamp::new("uncertainty-test", 7, 1_000, 1_001).unwrap(),
            sources: vec![SourceStamp {
                source: FactSource::Human,
                source_frame_id: 7,
                produced_at_unix_ms: 1_000,
                age_frames: 0,
                freshness: SourceFreshness::Fresh,
            }],
            entities: vec![SceneEntity {
                key: key.clone(),
                display_id: 10,
                kind: SceneEntityKind::Person(PersonFacts {
                    gallery_identity: fact(
                        7,
                        &key,
                        GalleryIdentity {
                            gallery_id: 1,
                            name: "known".into(),
                        },
                        0.9,
                    ),
                    posture: fact(7, &key, "standing".into(), 0.9),
                    emotion: fact(7, &key, "neutral".into(), 0.9),
                }),
                existence: fact(7, &key, Existence::Present, 0.9),
                bbox: fact(
                    7,
                    &key,
                    BBox {
                        x1: 0.2,
                        y1: 0.1,
                        x2: 0.5,
                        y2: 0.9,
                    },
                    0.9,
                ),
                position_3d: fact(
                    7,
                    &key,
                    Position3D {
                        x: 0.0,
                        y: 0.2,
                        z: 2.0,
                        confidence: 0.9,
                    },
                    0.9,
                ),
                kinematics: SceneFact::unknown(FactReason::MissingSource),
            }],
            relations: Vec::new(),
            zones: Vec::new(),
            activities: Vec::new(),
            events: EventTailView::default(),
            anomaly: SceneFact::unknown(FactReason::MissingSource),
            quality: SceneQuality::default(),
        }
    }

    fn apply_signal(scene: &mut FusedSceneFrame, signal: &str) {
        let key = scene.entities[0].key.clone();
        let person = match &mut scene.entities[0].kind {
            SceneEntityKind::Person(person) => person,
            _ => unreachable!(),
        };
        match signal {
            "posture_withheld" => {
                person.posture =
                    SceneFact::withheld(FactReason::InsufficientCoverage, evidence(7, &key));
            }
            "identity_low" => person.gallery_identity.confidence = Some(0.2),
            "existence_occluded" => scene.entities[0].existence.value = Some(Existence::Occluded),
            "source_stale" => scene.sources[0].freshness = SourceFreshness::Stale,
            "source_unavailable" => scene.sources[0] = SourceStamp::unavailable(FactSource::Human),
            "event_gap" => {}
            "position_uncalibrated" => {
                scene.entities[0].position_3d =
                    SceneFact::withheld(FactReason::Uncalibrated, evidence(7, &key));
            }
            "posture_observed"
            | "identity_high"
            | "existence_present"
            | "source_fresh"
            | "event_complete"
            | "position_observed"
            | "emotion_distribution_absent"
            | "camera_world_pose_absent" => {}
            other => panic!("unknown benchmark signal {other}"),
        }
    }

    #[test]
    fn uncertain_claim_inventory_exceeds_phase_gate_on_benchmark() {
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../../benchmark/scenarios/016.json")).unwrap();
        assert_eq!(fixture.status, "synthetic_replay");
        let mut correct = 0usize;

        for case in &fixture.cases {
            let mut scene = base_scene();
            apply_signal(&mut scene, &case.signal);
            let self_model = SelfModelLayer.build(&scene, &mvm_core::Config::default());
            let scene_event_gap = case.signal == "event_gap";
            let report = UncertaintyLayer.build(
                &scene,
                &self_model,
                &mvm_core::Config::default(),
                scene_event_gap,
            );
            let present = report
                .claims
                .iter()
                .any(|claim| claim.code == case.expected_code);
            correct += usize::from(present == case.expected_present);
            assert_eq!(present, case.expected_present, "case {}", case.name);
        }

        let accuracy = correct as f32 / fixture.cases.len() as f32;
        assert!(accuracy > 0.8, "uncertainty accuracy={accuracy}");
    }
}
