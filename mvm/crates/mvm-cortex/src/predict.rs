use crate::{
    ActivityForecast, ClaimStatus, CognitiveClaim, EntityForecast, EvidenceRef, ForecastPosition3D,
    PredictionError, PredictionErrorKind, ReasonCode, RelationForecast, SceneForecast,
};
use mvm_core::canonical::{
    EntityKey, Existence, FactStatus, FusedSceneFrame, SceneEntityKind, SceneFact, SceneRelation,
    SceneRelationKind,
};
use mvm_core::{BBox, Position3D};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const FORECAST_HORIZON_MS: i64 = 1_000;
const FORECAST_CONFIDENCE_DECAY: f32 = 0.9;
const MAX_PENDING_FORECASTS: usize = 128;

/// Runtime-only constant-velocity and persistence model. Pending forecasts are
/// bounded and disappear with the process; they are not a second history store.
#[derive(Default)]
pub(crate) struct PredictiveLayer {
    pending: VecDeque<SceneForecast>,
    run_id: Option<String>,
    last_timestamp_unix_ms: Option<i64>,
}

impl PredictiveLayer {
    pub(crate) fn update(
        &mut self,
        current: &FusedSceneFrame,
    ) -> (SceneForecast, Vec<PredictionError>) {
        self.reset_if_timeline_changed(current);

        let mut due = None;
        while self.pending.front().is_some_and(|forecast| {
            forecast.target_timestamp_unix_ms <= current.stamp.captured_at_unix_ms
        }) {
            due = self.pending.pop_front();
        }
        let errors = due
            .as_ref()
            .map_or_else(Vec::new, |forecast| compare(forecast, current));
        let forecast = build_forecast(current);
        if self.pending.len() == MAX_PENDING_FORECASTS {
            self.pending.pop_front();
        }
        self.pending.push_back(forecast.clone());
        self.last_timestamp_unix_ms = Some(current.stamp.captured_at_unix_ms);
        (forecast, errors)
    }

    fn reset_if_timeline_changed(&mut self, current: &FusedSceneFrame) {
        let run_changed = self.run_id.as_deref() != Some(current.stamp.run_id.as_str());
        let time_regressed = self
            .last_timestamp_unix_ms
            .is_some_and(|last| current.stamp.captured_at_unix_ms < last);
        if run_changed || time_regressed {
            self.pending.clear();
        }
        self.run_id = Some(current.stamp.run_id.clone());
    }
}

fn build_forecast(scene: &FusedSceneFrame) -> SceneForecast {
    let horizon_seconds = FORECAST_HORIZON_MS as f32 / 1_000.0;
    let active = active_entity_keys(scene);
    let entities = scene
        .entities
        .iter()
        .map(|entity| {
            let evidence = entity_evidence(scene.stamp.frame_id, &entity.key);
            let kinematics = safe_value(&entity.kinematics).filter(|value| {
                value.bbox_velocity.iter().all(|v| v.is_finite())
                    && value
                        .velocity_3d_mps
                        .is_none_or(|velocity| velocity.iter().all(|v| v.is_finite()))
            });
            let bbox = forecast_claim(&entity.bbox, evidence.clone(), |bbox| {
                valid_bbox(bbox).then(|| {
                    let velocity = kinematics.map_or([0.0, 0.0], |value| value.bbox_velocity);
                    translate_bbox(*bbox, velocity, horizon_seconds)
                })
            });
            let position_3d = forecast_claim(&entity.position_3d, evidence.clone(), |position| {
                valid_position(position).then(|| {
                    let velocity = kinematics
                        .and_then(|value| value.velocity_3d_mps)
                        .unwrap_or([0.0; 3]);
                    ForecastPosition3D {
                        x: position.x + velocity[0] * horizon_seconds,
                        y: position.y + velocity[1] * horizon_seconds,
                        z: position.z + velocity[2] * horizon_seconds,
                    }
                })
            });
            let posture = match &entity.kind {
                SceneEntityKind::Person(facts) => {
                    forecast_claim(&facts.posture, evidence.clone(), |value| {
                        Some(value.clone())
                    })
                }
                _ => CognitiveClaim::unknown(ReasonCode::MissingSource),
            };
            EntityForecast {
                key: entity.key.clone(),
                existence: forecast_claim(&entity.existence, evidence, |value| Some(*value)),
                bbox,
                position_3d,
                posture,
            }
        })
        .collect();

    let activities = scene
        .activities
        .iter()
        .map(|activity| ActivityForecast {
            label: activity.label.clone(),
            participants: activity.participants.clone(),
            persistence: forecast_claim(
                &activity.fact,
                activity_evidence(scene.stamp.frame_id, &activity.participants),
                |active| (*active).then_some(true),
            ),
        })
        .collect();
    let relations = scene
        .relations
        .iter()
        .filter(|relation| active.contains(&relation.a) && active.contains(&relation.b))
        .map(|relation| RelationForecast {
            a: relation.a.clone(),
            b: relation.b.clone(),
            kind: relation.kind,
            persistence: relation_persistence(scene.stamp.frame_id, relation),
        })
        .collect();

    SceneForecast {
        source_scene_frame_id: scene.stamp.frame_id,
        source_timestamp_unix_ms: scene.stamp.captured_at_unix_ms,
        target_timestamp_unix_ms: scene
            .stamp
            .captured_at_unix_ms
            .saturating_add(FORECAST_HORIZON_MS),
        horizon_ms: FORECAST_HORIZON_MS,
        entities,
        activities,
        relations,
    }
}

fn compare(forecast: &SceneForecast, observed: &FusedSceneFrame) -> Vec<PredictionError> {
    let mut errors = Vec::new();
    let predicted_by_key: BTreeMap<_, _> = forecast
        .entities
        .iter()
        .map(|entity| (&entity.key, entity))
        .collect();
    let observed_by_key: BTreeMap<_, _> = observed
        .entities
        .iter()
        .map(|entity| (&entity.key, entity))
        .collect();

    for entity in &forecast.entities {
        let Some(current) = observed_by_key.get(&entity.key) else {
            if entity.existence.status == ClaimStatus::Asserted {
                errors.push(prediction_error(
                    PredictionErrorKind::EntityDisappeared,
                    forecast,
                    observed,
                    vec![entity.key.clone()],
                    None,
                    entity.existence.confidence.unwrap_or(0.0),
                    entity.existence.evidence.clone(),
                ));
            }
            continue;
        };

        if let (Some(predicted), Some(actual)) = (
            asserted_value(&entity.bbox),
            safe_value(&current.bbox).filter(|bbox| valid_bbox(bbox)),
        ) {
            errors.push(prediction_error(
                PredictionErrorKind::BboxDrift,
                forecast,
                observed,
                vec![entity.key.clone()],
                Some((1.0 - predicted.iou(actual)).clamp(0.0, 1.0)),
                combined_confidence(entity.bbox.confidence, current.bbox.confidence),
                merge_evidence(
                    entity.bbox.evidence.clone(),
                    entity_evidence(observed.stamp.frame_id, &entity.key),
                ),
            ));
        }

        if let (Some(predicted), Some(actual)) = (
            asserted_value(&entity.position_3d),
            safe_value(&current.position_3d).filter(|position| valid_position(position)),
        ) {
            let drift = ((predicted.x - actual.x).powi(2)
                + (predicted.y - actual.y).powi(2)
                + (predicted.z - actual.z).powi(2))
            .sqrt();
            errors.push(prediction_error(
                PredictionErrorKind::Position3dDrift,
                forecast,
                observed,
                vec![entity.key.clone()],
                Some(drift),
                combined_confidence(
                    entity.position_3d.confidence,
                    current.position_3d.confidence,
                ),
                merge_evidence(
                    entity.position_3d.evidence.clone(),
                    entity_evidence(observed.stamp.frame_id, &entity.key),
                ),
            ));
        }

        let current_posture = match &current.kind {
            SceneEntityKind::Person(facts) => safe_value(&facts.posture),
            _ => None,
        };
        if let (Some(predicted), Some(actual)) = (asserted_value(&entity.posture), current_posture)
            && predicted != actual
        {
            errors.push(prediction_error(
                PredictionErrorKind::PostureChanged,
                forecast,
                observed,
                vec![entity.key.clone()],
                None,
                entity.posture.confidence.unwrap_or(0.0),
                merge_evidence(
                    entity.posture.evidence.clone(),
                    entity_evidence(observed.stamp.frame_id, &entity.key),
                ),
            ));
        }
    }

    for entity in &observed.entities {
        if !predicted_by_key.contains_key(&entity.key)
            && let Some(confidence) = safe_true_existence(&entity.existence)
        {
            errors.push(prediction_error(
                PredictionErrorKind::EntityAppeared,
                forecast,
                observed,
                vec![entity.key.clone()],
                None,
                confidence,
                entity_evidence(observed.stamp.frame_id, &entity.key),
            ));
        }
    }

    compare_activities(forecast, observed, &mut errors);
    compare_relations(forecast, observed, &mut errors);
    errors.sort_by(|a, b| (&a.kind, &a.entities).cmp(&(&b.kind, &b.entities)));
    errors
}

fn compare_activities(
    forecast: &SceneForecast,
    observed: &FusedSceneFrame,
    errors: &mut Vec<PredictionError>,
) {
    let predicted: BTreeMap<_, _> = forecast
        .activities
        .iter()
        .filter(|activity| asserted_value(&activity.persistence) == Some(&true))
        .map(|activity| {
            (
                (activity.label.clone(), activity.participants.clone()),
                activity,
            )
        })
        .collect();
    let current: BTreeMap<_, _> = observed
        .activities
        .iter()
        .filter(|activity| safe_value(&activity.fact) == Some(&true))
        .map(|activity| {
            (
                (activity.label.clone(), activity.participants.clone()),
                activity,
            )
        })
        .collect();

    for (key, activity) in &predicted {
        if !current.contains_key(key) {
            errors.push(prediction_error(
                PredictionErrorKind::ActivityChanged,
                forecast,
                observed,
                activity.participants.clone(),
                None,
                activity.persistence.confidence.unwrap_or(0.0),
                activity.persistence.evidence.clone(),
            ));
        }
    }
    for (key, activity) in &current {
        if !predicted.contains_key(key) {
            errors.push(prediction_error(
                PredictionErrorKind::ActivityChanged,
                forecast,
                observed,
                activity.participants.clone(),
                None,
                activity.fact.confidence.unwrap_or(0.0),
                activity_evidence(observed.stamp.frame_id, &activity.participants),
            ));
        }
    }
}

fn compare_relations(
    forecast: &SceneForecast,
    observed: &FusedSceneFrame,
    errors: &mut Vec<PredictionError>,
) {
    let predicted = forecast
        .relations
        .iter()
        .filter(|relation| asserted_value(&relation.persistence) == Some(&true))
        .map(|relation| (relation.a.clone(), relation.b.clone(), relation.kind))
        .collect::<BTreeSet<_>>();
    let active = active_entity_keys(observed);
    let current = observed
        .relations
        .iter()
        .filter(|relation| active.contains(&relation.a) && active.contains(&relation.b))
        .filter(|relation| relation_is_supported(relation))
        .map(|relation| (relation.a.clone(), relation.b.clone(), relation.kind))
        .collect::<BTreeSet<_>>();
    for (a, b, kind) in predicted.difference(&current) {
        let relation = forecast
            .relations
            .iter()
            .find(|relation| relation.a == *a && relation.b == *b && relation.kind == *kind)
            .expect("key came from forecast relations");
        errors.push(prediction_error(
            PredictionErrorKind::RelationBroken,
            forecast,
            observed,
            vec![a.clone(), b.clone()],
            None,
            relation.persistence.confidence.unwrap_or(0.0),
            relation.persistence.evidence.clone(),
        ));
    }
}

fn prediction_error(
    kind: PredictionErrorKind,
    forecast: &SceneForecast,
    observed: &FusedSceneFrame,
    entities: Vec<EntityKey>,
    magnitude: Option<f32>,
    confidence: f32,
    evidence: Vec<EvidenceRef>,
) -> PredictionError {
    PredictionError {
        kind,
        forecast_source_scene_frame_id: forecast.source_scene_frame_id,
        observed_scene_frame_id: observed.stamp.frame_id,
        entities,
        magnitude: magnitude.filter(|value| value.is_finite()),
        confidence: if confidence.is_finite() {
            confidence.clamp(0.0, 1.0)
        } else {
            0.0
        },
        evidence,
    }
}

fn forecast_claim<T, U>(
    fact: &SceneFact<T>,
    evidence: Vec<EvidenceRef>,
    transform: impl FnOnce(&T) -> Option<U>,
) -> CognitiveClaim<U> {
    match fact.status {
        FactStatus::Observed | FactStatus::Derived | FactStatus::Temporal => {
            let Some(value) = fact.value.as_ref().and_then(transform) else {
                return CognitiveClaim::withheld(
                    ReasonCode::Other("invalid_forecast_input".into()),
                    evidence,
                );
            };
            let Some(confidence) = fact.confidence.filter(|value| value.is_finite()) else {
                return CognitiveClaim::withheld(
                    ReasonCode::Other("invalid_forecast_confidence".into()),
                    evidence,
                );
            };
            CognitiveClaim::asserted(
                value,
                (confidence * FORECAST_CONFIDENCE_DECAY).clamp(0.0, 1.0),
                evidence,
            )
            .unwrap_or_else(|_| {
                CognitiveClaim::withheld(
                    ReasonCode::Other("invalid_forecast_claim".into()),
                    Vec::new(),
                )
            })
        }
        FactStatus::Remembered => CognitiveClaim::withheld(ReasonCode::RememberedSource, evidence),
        FactStatus::Weak => CognitiveClaim::withheld(ReasonCode::WeakSource, evidence),
        FactStatus::Withheld => {
            CognitiveClaim::withheld(ReasonCode::InsufficientEvidence, evidence)
        }
        FactStatus::Unknown => CognitiveClaim::unknown(ReasonCode::MissingSource),
    }
}

fn safe_value<T>(fact: &SceneFact<T>) -> Option<&T> {
    matches!(
        fact.status,
        FactStatus::Observed | FactStatus::Derived | FactStatus::Temporal
    )
    .then_some(())
    .and(fact.value.as_ref())
    .filter(|_| fact.confidence.is_some_and(|value| value.is_finite()))
}

fn asserted_value<T>(claim: &CognitiveClaim<T>) -> Option<&T> {
    (claim.status == ClaimStatus::Asserted)
        .then_some(())
        .and(claim.value.as_ref())
}

fn safe_true_existence(fact: &SceneFact<Existence>) -> Option<f32> {
    safe_value(fact).and(fact.confidence)
}

fn active_entity_keys(scene: &FusedSceneFrame) -> BTreeSet<EntityKey> {
    scene
        .entities
        .iter()
        .filter(|entity| safe_true_existence(&entity.existence).is_some())
        .map(|entity| entity.key.clone())
        .collect()
}

fn relation_persistence(frame_id: u64, relation: &SceneRelation) -> CognitiveClaim<bool> {
    let evidence = activity_evidence(frame_id, &[relation.a.clone(), relation.b.clone()]);
    let fact = relation_support_fact(relation);
    fact.map_or_else(
        || CognitiveClaim::unknown(ReasonCode::MissingSource),
        |fact| forecast_claim(fact, evidence, |_| Some(true)),
    )
}

fn relation_support_fact(relation: &SceneRelation) -> Option<&SceneFact<f32>> {
    let preferred = match relation.kind {
        SceneRelationKind::Near => &relation.distance_m,
        SceneRelationKind::Overlapping
        | SceneRelationKind::PartOf
        | SceneRelationKind::Touching
        | SceneRelationKind::HeldBy => &relation.overlap,
    };
    safe_value(preferred)
        .map(|_| preferred)
        .or_else(|| safe_value(&relation.distance_m).map(|_| &relation.distance_m))
        .or_else(|| safe_value(&relation.overlap).map(|_| &relation.overlap))
}

fn relation_is_supported(relation: &SceneRelation) -> bool {
    relation_support_fact(relation).is_some()
}

fn entity_evidence(frame_id: u64, entity: &EntityKey) -> Vec<EvidenceRef> {
    vec![EvidenceRef {
        scene_frame_id: frame_id,
        entity: Some(entity.clone()),
        event_seq: None,
    }]
}

fn activity_evidence(frame_id: u64, entities: &[EntityKey]) -> Vec<EvidenceRef> {
    if entities.is_empty() {
        vec![EvidenceRef {
            scene_frame_id: frame_id,
            entity: None,
            event_seq: None,
        }]
    } else {
        entities
            .iter()
            .map(|entity| EvidenceRef {
                scene_frame_id: frame_id,
                entity: Some(entity.clone()),
                event_seq: None,
            })
            .collect()
    }
}

fn merge_evidence(mut a: Vec<EvidenceRef>, b: Vec<EvidenceRef>) -> Vec<EvidenceRef> {
    for evidence in b {
        if !a.contains(&evidence) {
            a.push(evidence);
        }
    }
    a
}

fn combined_confidence(a: Option<f32>, b: Option<f32>) -> f32 {
    match (a, b) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => a.min(b),
        _ => 0.0,
    }
}

fn valid_bbox(bbox: &BBox) -> bool {
    [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
        .iter()
        .all(|value| value.is_finite())
        && bbox.x2 > bbox.x1
        && bbox.y2 > bbox.y1
}

fn translate_bbox(bbox: BBox, velocity: [f32; 2], horizon_seconds: f32) -> BBox {
    let width = bbox.x2 - bbox.x1;
    let height = bbox.y2 - bbox.y1;
    let x1 = (bbox.x1 + velocity[0] * horizon_seconds).clamp(0.0, (1.0 - width).max(0.0));
    let y1 = (bbox.y1 + velocity[1] * horizon_seconds).clamp(0.0, (1.0 - height).max(0.0));
    BBox {
        x1,
        y1,
        x2: x1 + width,
        y2: y1 + height,
    }
}

fn valid_position(position: &Position3D) -> bool {
    [position.x, position.y, position.z, position.confidence]
        .iter()
        .all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClaimStatus, PredictionErrorKind};
    use mvm_core::canonical::{
        EntityKey, EventTailView, Existence, FactEvidenceRef, FactReason, FactSource, FactStatus,
        FrameStamp, FusedSceneFrame, Kinematics, PersonFacts, PersonSensoryFacts, SceneActivity,
        SceneEntity, SceneEntityKind, SceneFact, SceneQuality, SceneRelation, SceneRelationKind,
        TrackDomain,
    };
    use mvm_core::{BBox, Position3D};
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct ForecastFixture {
        scenario_id: String,
        status: String,
        horizon_ms: i64,
        source_bbox: [f32; 4],
        bbox_velocity_per_second: [f32; 2],
        observed_bbox: [f32; 4],
    }

    fn evidence(frame_id: u64, key: &EntityKey) -> Vec<FactEvidenceRef> {
        vec![FactEvidenceRef {
            source: FactSource::Human,
            source_frame_id: frame_id,
            entity: Some(key.clone()),
            event_seq: None,
        }]
    }

    fn fact<T>(frame_id: u64, key: &EntityKey, value: T) -> SceneFact<T> {
        SceneFact::claimed(FactStatus::Observed, value, 0.9, evidence(frame_id, key)).unwrap()
    }

    fn bbox(values: [f32; 4]) -> BBox {
        BBox {
            x1: values[0],
            y1: values[1],
            x2: values[2],
            y2: values[3],
        }
    }

    fn person(
        frame_id: u64,
        track_id: u32,
        bbox: BBox,
        velocity: [f32; 2],
        posture: &str,
    ) -> SceneEntity {
        let key = EntityKey {
            run_id: "predict-test".into(),
            track_domain: TrackDomain::Person,
            track_id,
            incarnation: 0,
        };
        SceneEntity {
            key: key.clone(),
            display_id: track_id,
            kind: SceneEntityKind::Person(PersonFacts {
                gallery_identity: SceneFact::unknown(FactReason::MissingSource),
                posture: fact(frame_id, &key, posture.to_owned()),
                emotion: SceneFact::unknown(FactReason::MissingSource),
                sensory: Box::new(PersonSensoryFacts::default()),
            }),
            existence: fact(frame_id, &key, Existence::Present),
            bbox: fact(frame_id, &key, bbox),
            position_3d: fact(
                frame_id,
                &key,
                Position3D {
                    x: bbox.x1,
                    y: bbox.y1,
                    z: 2.0,
                    confidence: 0.9,
                },
            ),
            kinematics: SceneFact::claimed(
                FactStatus::Temporal,
                Kinematics {
                    bbox_velocity: velocity,
                    velocity_3d_mps: Some([velocity[0], velocity[1], 0.0]),
                },
                0.9,
                evidence(frame_id, &key),
            )
            .unwrap(),
        }
    }

    fn scene(frame_id: u64, timestamp_ms: i64, bbox: BBox, velocity: [f32; 2]) -> FusedSceneFrame {
        FusedSceneFrame {
            stamp: FrameStamp::new("predict-test", frame_id, timestamp_ms, timestamp_ms).unwrap(),
            sources: Vec::new(),
            entities: vec![person(frame_id, 10, bbox, velocity, "standing")],
            relations: Vec::new(),
            hand_interactions: Vec::new(),
            zones: Vec::new(),
            activities: Vec::new(),
            events: EventTailView::default(),
            anomaly: SceneFact::unknown(FactReason::MissingSource),
            quality: SceneQuality::default(),
        }
    }

    fn fixtures() -> Vec<ForecastFixture> {
        [
            include_str!("../../../benchmark/scenarios/013.json"),
            include_str!("../../../benchmark/scenarios/014.json"),
            include_str!("../../../benchmark/scenarios/015.json"),
        ]
        .into_iter()
        .map(|raw| serde_json::from_str(raw).unwrap())
        .collect()
    }

    #[test]
    fn forecast_at_one_second_beats_last_observation_on_three_benchmark_scenarios() {
        for fixture in fixtures() {
            assert_eq!(fixture.status, "synthetic_replay");
            assert_eq!(fixture.horizon_ms, 1_000);
            let source_bbox = bbox(fixture.source_bbox);
            let observed_bbox = bbox(fixture.observed_bbox);
            let mut predictor = PredictiveLayer::default();
            let (_, first_errors) =
                predictor.update(&scene(1, 0, source_bbox, fixture.bbox_velocity_per_second));
            assert!(first_errors.is_empty());
            let (_, errors) = predictor.update(&scene(
                2,
                fixture.horizon_ms,
                observed_bbox,
                fixture.bbox_velocity_per_second,
            ));
            let candidate_iou = 1.0
                - errors
                    .iter()
                    .find(|error| error.kind == PredictionErrorKind::BboxDrift)
                    .and_then(|error| error.magnitude)
                    .expect("bbox error");
            let baseline_iou = source_bbox.iou(&observed_bbox);
            assert!(
                candidate_iou > baseline_iou,
                "scenario {}: candidate={candidate_iou}, baseline={baseline_iou}",
                fixture.scenario_id
            );
        }
    }

    #[test]
    fn remembered_bbox_is_withheld_instead_of_upgraded() {
        let mut input = scene(1, 0, bbox([0.1, 0.2, 0.3, 0.6]), [0.2, 0.0]);
        let key = input.entities[0].key.clone();
        input.entities[0].bbox = SceneFact::claimed(
            FactStatus::Remembered,
            bbox([0.1, 0.2, 0.3, 0.6]),
            0.7,
            evidence(1, &key),
        )
        .unwrap();

        let (forecast, _) = PredictiveLayer::default().update(&input);
        assert_eq!(forecast.entities[0].bbox.status, ClaimStatus::Withheld);
        assert!(forecast.entities[0].bbox.value.is_none());
    }

    #[test]
    fn due_forecast_reports_the_v1_prediction_error_taxonomy_with_evidence() {
        let mut source = scene(1, 0, bbox([0.1, 0.1, 0.3, 0.5]), [0.0, 0.0]);
        let departed = person(1, 11, bbox([0.6, 0.1, 0.8, 0.5]), [0.0, 0.0], "standing");
        let departed_key = departed.key.clone();
        let retained_key = source.entities[0].key.clone();
        source.entities.push(departed);
        source.activities.push(SceneActivity {
            label: "talking".into(),
            participants: vec![retained_key.clone(), departed_key.clone()],
            fact: fact(1, &retained_key, true),
        });
        source.relations.push(SceneRelation {
            a: retained_key.clone(),
            b: departed_key.clone(),
            kind: SceneRelationKind::Near,
            distance_m: fact(1, &retained_key, 0.5),
            overlap: SceneFact::unknown(FactReason::MissingSource),
        });

        let mut observed = scene(2, 1_000, bbox([0.4, 0.2, 0.6, 0.6]), [0.0, 0.0]);
        observed.entities[0] = person(2, 10, bbox([0.4, 0.2, 0.6, 0.6]), [0.0, 0.0], "sitting");
        observed.entities.push(person(
            2,
            12,
            bbox([0.7, 0.2, 0.9, 0.6]),
            [0.0, 0.0],
            "standing",
        ));

        let mut predictor = PredictiveLayer::default();
        predictor.update(&source);
        let (_, errors) = predictor.update(&observed);
        let kinds = errors
            .iter()
            .map(|error| error.kind)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            kinds,
            BTreeSet::from([
                PredictionErrorKind::BboxDrift,
                PredictionErrorKind::Position3dDrift,
                PredictionErrorKind::EntityAppeared,
                PredictionErrorKind::EntityDisappeared,
                PredictionErrorKind::ActivityChanged,
                PredictionErrorKind::RelationBroken,
                PredictionErrorKind::PostureChanged,
            ])
        );
        assert!(errors.iter().all(|error| !error.evidence.is_empty()));
        assert!(errors.iter().all(|error| {
            error.forecast_source_scene_frame_id == 1 && error.observed_scene_frame_id == 2
        }));
    }
}
