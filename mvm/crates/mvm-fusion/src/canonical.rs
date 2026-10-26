//! Canonical scene assembler (phase 18.1).
//!
//! This module is the only owner of canonical entity incarnations, relations,
//! persistent zones and the sequenced scene-event tail. Inputs are immutable
//! expert observations; output revisions are deterministic after canonicalizing.

use crate::{ZoneTracker, build_scene};
use mvm_core::canonical::{
    AnimalFacts, BodyCoverage, DirectMetricGeometry, EntityGeometryFacts, EntityKey, EventTailView,
    Existence, FactEvidenceRef, FactReason, FactSource, FactStatus, FrameStamp, FusedSceneFrame,
    GalleryIdentity, GeometryReferenceFrame, HandFact, HandGesture, HandInteractionFact,
    HandInteractionKind, HandSide, HeadFacing, HeadOrientationFact, IncarnationRegistry,
    Kinematics, ObjectFacts, PersonFacts, PersonSensoryFacts, SceneActivity, SceneEntity,
    SceneEntityKind, SceneEvent, SceneEventKind, SceneEventPayload, SceneFact, SceneQuality,
    SceneRelation, SceneRelationKind, SceneZone, SourceFreshness, SourceStamp, TrackDomain,
};
use mvm_core::types::{Entity, EntityKind, RelationKind};
use mvm_core::{Activity, ActivityEvent, BBox, ObjectState, PersonState, Position3D};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const EVENT_TAIL_CAP: usize = 256;

#[derive(Debug, Clone)]
pub struct AnimalObservation {
    pub track_id: u32,
    pub class_name: String,
    pub bbox: BBox,
    pub confidence: f32,
    pub remembered: bool,
}

pub struct CanonicalFusionInput<'a> {
    pub stamp: FrameStamp,
    pub sources: Vec<SourceStamp>,
    pub people: &'a [PersonState],
    pub objects: &'a [ObjectState],
    pub animals: &'a [AnimalObservation],
    pub activities: &'a [Activity],
    pub ended_activities: &'a [ActivityEvent],
    pub anomaly_score: f32,
    pub metric_geometry: &'a [MetricGeometryObservation],
    /// Established local ids that remain alive even without a current
    /// observation. They reserve incarnation only and never create entities.
    pub retained_people: &'a [u32],
    pub retired_people: &'a [u32],
    pub retained_objects: &'a [u32],
    pub retired_objects: &'a [u32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricGeometryObservation {
    pub track_domain: TrackDomain,
    pub track_id: u32,
    pub source_frame_id: u64,
    pub produced_at_unix_ms: i64,
    pub source_freshness: SourceFreshness,
    pub centroid_m: [f32; 3],
    pub aabb_min_m: [f32; 3],
    pub aabb_max_m: [f32; 3],
    pub median_z_m: f32,
    pub median_absolute_deviation_m: f32,
    pub sample_count: u32,
    pub valid_ratio: f32,
    pub boundary_contamination: f32,
    pub calibration_revision: String,
    pub mask_aware: bool,
    pub weak: bool,
}

pub struct CanonicalFusion {
    registry: IncarnationRegistry,
    active: BTreeMap<(TrackDomain, u32), EntityKey>,
    surrogate_ids: BTreeMap<EntityKey, u32>,
    next_surrogate_id: u32,
    zones: ZoneTracker,
    events: VecDeque<SceneEvent>,
    next_event_seq: u64,
    previous_poses: BTreeMap<EntityKey, PreviousPose>,
    previous_entities: BTreeMap<EntityKey, PreviousEntity>,
    previous_activities: BTreeMap<ActivityKey, Vec<EntityKey>>,
    previous_zones: BTreeMap<u64, Vec<EntityKey>>,
    previous_anomaly_score: Option<f32>,
    has_previous_tick: bool,
}

#[derive(Debug, Clone, Copy)]
struct PreviousPose {
    captured_at_unix_ms: i64,
    bbox_center: [f32; 2],
    position_3d: Option<Position3D>,
}

#[derive(Debug, Clone)]
struct PreviousEntity {
    kind: SceneEntityKind,
    bbox: Option<BBox>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ActivityKey {
    label: String,
    participants: Vec<EntityKey>,
}

fn validate_metric_geometry_input(input: &CanonicalFusionInput<'_>) -> anyhow::Result<()> {
    let mut seen = BTreeSet::new();
    for value in input.metric_geometry {
        anyhow::ensure!(
            value.source_frame_id <= input.stamp.frame_id,
            "metric geometry for {:?}/{} comes from future frame {} > {}",
            value.track_domain,
            value.track_id,
            value.source_frame_id,
            input.stamp.frame_id
        );
        anyhow::ensure!(
            seen.insert((value.track_domain, value.track_id)),
            "duplicate metric geometry for {:?}/{}",
            value.track_domain,
            value.track_id
        );
    }
    Ok(())
}

impl CanonicalFusion {
    pub fn new(run_id: impl Into<String>) -> Self {
        let run_id = run_id.into();
        Self {
            registry: IncarnationRegistry::new(run_id),
            active: BTreeMap::new(),
            surrogate_ids: BTreeMap::new(),
            next_surrogate_id: 0,
            zones: ZoneTracker::new(),
            events: VecDeque::new(),
            next_event_seq: 0,
            previous_poses: BTreeMap::new(),
            previous_entities: BTreeMap::new(),
            previous_activities: BTreeMap::new(),
            previous_zones: BTreeMap::new(),
            previous_anomaly_score: None,
            has_previous_tick: false,
        }
    }

    pub fn assemble(&mut self, input: CanonicalFusionInput<'_>) -> anyhow::Result<FusedSceneFrame> {
        validate_metric_geometry_input(&input)?;
        self.sync_tracked_domain(
            TrackDomain::Person,
            input.people.iter().map(|person| person.tracker_id),
            input.retained_people,
            input.retired_people,
        )?;
        self.sync_tracked_domain(
            TrackDomain::Object,
            input.objects.iter().map(|object| object.id),
            input.retained_objects,
            input.retired_objects,
        )?;
        self.sync_domain(
            TrackDomain::Animal,
            input.animals.iter().map(|animal| animal.track_id),
        );
        let retained_entity_keys: BTreeSet<EntityKey> = input
            .retained_people
            .iter()
            .filter_map(|id| self.active.get(&(TrackDomain::Person, *id)).cloned())
            .chain(
                input
                    .retained_objects
                    .iter()
                    .filter_map(|id| self.active.get(&(TrackDomain::Object, *id)).cloned()),
            )
            .collect();

        let human_source = source(&input.sources, FactSource::Human);
        let scene_source = source(&input.sources, FactSource::Scene);
        let depth_source = source(&input.sources, FactSource::Depth);

        let mut entities = Vec::new();
        for person in input.people {
            let key = self.key(TrackDomain::Person, person.tracker_id);
            entities.push(person_entity(
                person,
                key,
                human_source,
                metric_geometry(
                    input.metric_geometry,
                    TrackDomain::Person,
                    person.tracker_id,
                ),
            ));
        }
        for object in input.objects {
            let key = self.key(TrackDomain::Object, object.id);
            entities.push(object_entity(
                object,
                key,
                scene_source,
                depth_source,
                metric_geometry(input.metric_geometry, TrackDomain::Object, object.id),
            ));
        }
        for animal in input.animals {
            let key = self.key(TrackDomain::Animal, animal.track_id);
            entities.push(animal_entity(animal, key, scene_source));
        }
        entities.sort_by(|a, b| a.key.cmp(&b.key));
        self.apply_kinematics(&mut entities, &input.stamp)?;

        let (relations, zones) = self.relations_and_zones(&entities, input.stamp.frame_id)?;
        let hand_interactions = canonical_hand_interactions(
            input.people,
            input.objects,
            &entities,
            human_source,
            scene_source,
        )?;
        let activities = canonical_activities(input.activities, &entities, input.stamp.frame_id)?;
        self.append_events(
            input.ended_activities,
            &entities,
            &activities,
            &zones,
            input.anomaly_score,
            (&input.stamp, &retained_entity_keys),
        )?;

        let anomaly = if input.anomaly_score.is_finite() {
            SceneFact::claimed(
                FactStatus::Temporal,
                input.anomaly_score.clamp(0.0, 1.0),
                1.0,
                vec![FactEvidenceRef {
                    source: FactSource::Fusion,
                    source_frame_id: input.stamp.frame_id,
                    entity: None,
                    event_seq: None,
                }],
            )?
        } else {
            SceneFact::unknown(FactReason::MissingSource)
        };

        let first_seq = self
            .events
            .front()
            .map_or(self.next_event_seq, |event| event.event_seq);
        let mut frame = FusedSceneFrame {
            stamp: input.stamp,
            sources: input.sources,
            entities,
            relations,
            hand_interactions,
            zones,
            activities,
            events: EventTailView {
                first_seq,
                next_seq: self.next_event_seq,
                events: self.events.iter().cloned().collect(),
            },
            anomaly,
            quality: SceneQuality::default(),
        };
        frame.canonicalize();
        frame.validate()?;
        Ok(frame)
    }

    fn sync_domain(&mut self, domain: TrackDomain, ids: impl Iterator<Item = u32>) {
        let current: BTreeSet<u32> = ids.collect();
        let retired: Vec<(TrackDomain, u32)> = self
            .active
            .keys()
            .filter(|(active_domain, id)| *active_domain == domain && !current.contains(id))
            .copied()
            .collect();
        for slot in retired {
            if let Some(key) = self.active.remove(&slot) {
                self.registry.retire(&key);
            }
        }
        for id in current {
            self.active
                .entry((domain, id))
                .or_insert_with(|| self.registry.activate(domain, id));
        }
    }

    fn sync_tracked_domain(
        &mut self,
        domain: TrackDomain,
        current_ids: impl Iterator<Item = u32>,
        retained_ids: &[u32],
        retirements: &[u32],
    ) -> anyhow::Result<()> {
        let current: BTreeSet<u32> = current_ids.collect();
        let retained: BTreeSet<u32> = retained_ids.iter().copied().collect();
        let retired: BTreeSet<u32> = retirements.iter().copied().collect();
        anyhow::ensure!(
            current.is_disjoint(&retired) && retained.is_disjoint(&retired),
            "{domain:?} tracker handoff retains and retires the same id"
        );

        for id in &retired {
            if let Some(key) = self.active.remove(&(domain, *id)) {
                self.registry.retire(&key);
            }
        }

        let desired: BTreeSet<u32> = current.union(&retained).copied().collect();
        let implicitly_retired: Vec<(TrackDomain, u32)> = self
            .active
            .keys()
            .filter(|(active_domain, id)| *active_domain == domain && !desired.contains(id))
            .copied()
            .collect();
        for slot in implicitly_retired {
            if let Some(key) = self.active.remove(&slot) {
                self.registry.retire(&key);
            }
        }

        // A retained-only id never creates a canonical identity out of thin air;
        // it can only keep an already-active incarnation reserved.
        for id in current {
            self.active
                .entry((domain, id))
                .or_insert_with(|| self.registry.activate(domain, id));
        }
        Ok(())
    }

    fn key(&self, domain: TrackDomain, id: u32) -> EntityKey {
        self.active[&(domain, id)].clone()
    }

    fn surrogate_id(&mut self, key: &EntityKey) -> u32 {
        if let Some(id) = self.surrogate_ids.get(key) {
            return *id;
        }
        let id = self.next_surrogate_id;
        self.next_surrogate_id = self.next_surrogate_id.saturating_add(1);
        self.surrogate_ids.insert(key.clone(), id);
        id
    }

    fn relations_and_zones(
        &mut self,
        entities: &[SceneEntity],
        source_frame_id: u64,
    ) -> anyhow::Result<(Vec<SceneRelation>, Vec<SceneZone>)> {
        let mut legacy = Vec::with_capacity(entities.len());
        let mut keys = BTreeMap::new();
        for entity in entities {
            let id = self.surrogate_id(&entity.key);
            keys.insert(id, entity.key.clone());
            legacy.push(Entity {
                id,
                kind: legacy_kind(&entity.kind),
                bbox: entity.bbox.value.unwrap_or(BBox {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 0.0,
                    y2: 0.0,
                }),
                pos_3d: entity
                    .position_3d
                    .value
                    .map_or([0.0; 3], |position| [position.x, position.y, position.z]),
                radius_m: 0.25,
                pos_conf: entity
                    .position_3d
                    .value
                    .map_or(0.0, |position| position.confidence),
                identity: None,
                emotion: None,
                emotion_distribution: None,
            });
        }
        let scene = build_scene(legacy);
        let mut relations = Vec::with_capacity(scene.relations.len());
        for relation in &scene.relations {
            let a = keys[&relation.a].clone();
            let b = keys[&relation.b].clone();
            let evidence = vec![
                evidence(FactSource::Fusion, source_frame_id, Some(a.clone())),
                evidence(FactSource::Fusion, source_frame_id, Some(b.clone())),
            ];
            relations.push(SceneRelation {
                a,
                b,
                kind: match relation.kind {
                    RelationKind::Near => SceneRelationKind::Near,
                    RelationKind::Overlapping => SceneRelationKind::Overlapping,
                    RelationKind::PartOf => SceneRelationKind::PartOf,
                },
                distance_m: SceneFact::claimed(
                    FactStatus::Derived,
                    relation.distance_m,
                    1.0,
                    evidence.clone(),
                )?,
                overlap: SceneFact::claimed(FactStatus::Derived, relation.overlap, 1.0, evidence)?,
            });
        }
        let zones = self
            .zones
            .update(&scene.zones, &scene.entities)
            .iter()
            .filter_map(|zone| {
                let members: Vec<_> = zone
                    .members
                    .iter()
                    .filter_map(|id| keys.get(id).cloned())
                    .collect();
                (!members.is_empty()).then(|| SceneZone {
                    id: u64::from(zone.id),
                    label: zone.label.map(str::to_owned),
                    members,
                    age_frames: u64::from(zone.age),
                })
            })
            .collect();
        Ok((relations, zones))
    }

    fn append_events(
        &mut self,
        ended: &[ActivityEvent],
        entities: &[SceneEntity],
        activities: &[SceneActivity],
        zones: &[SceneZone],
        anomaly_score: f32,
        context: (&FrameStamp, &BTreeSet<EntityKey>),
    ) -> anyhow::Result<()> {
        let (stamp, retained_entity_keys) = context;
        // Snapshot before emitting: publication mutates the event tail, never
        // the previous-tick evidence used to classify this transition.
        let previous_entities = self.previous_entities.clone();
        let previous_activities = self.previous_activities.clone();
        let previous_zones = self.previous_zones.clone();
        let have_previous_tick = self.has_previous_tick;
        let mut current: BTreeMap<EntityKey, PreviousEntity> = entities
            .iter()
            .map(|entity| {
                (
                    entity.key.clone(),
                    PreviousEntity {
                        kind: entity.kind.clone(),
                        bbox: entity.bbox.value,
                    },
                )
            })
            .collect();
        for (key, entity) in &current {
            if have_previous_tick && !previous_entities.contains_key(key) {
                self.push_event(
                    SceneEventKind::EntityAppeared,
                    vec![key.clone()],
                    1.0,
                    stamp,
                    None,
                )?;
                if matches!(entity.kind, SceneEntityKind::Person(_)) {
                    self.push_event(
                        SceneEventKind::PersonEntered,
                        vec![key.clone()],
                        1.0,
                        stamp,
                        None,
                    )?;
                }
            }
        }
        for (key, entity) in &previous_entities {
            if have_previous_tick
                && !current.contains_key(key)
                && !retained_entity_keys.contains(key)
            {
                self.push_event(
                    SceneEventKind::EntityDisappeared,
                    vec![key.clone()],
                    1.0,
                    stamp,
                    None,
                )?;
                if matches!(entity.kind, SceneEntityKind::Person(_)) {
                    self.push_event(
                        SceneEventKind::PersonExited,
                        vec![key.clone()],
                        1.0,
                        stamp,
                        None,
                    )?;
                }
            }
        }
        for (key, entity) in &current {
            if !have_previous_tick {
                continue;
            }
            let Some(previous) = previous_entities.get(key) else {
                continue;
            };
            if let (SceneEntityKind::Person(now), SceneEntityKind::Person(before)) =
                (&entity.kind, &previous.kind)
            {
                let now_posture = now.posture.value.as_deref();
                let before_posture = before.posture.value.as_deref();
                if now_posture == Some("sitting") && before_posture != Some("sitting") {
                    self.push_event(
                        SceneEventKind::PersonSatDown,
                        vec![key.clone()],
                        now.posture.confidence.unwrap_or(0.0),
                        stamp,
                        None,
                    )?;
                }
                if now_posture == Some("standing") && before_posture == Some("sitting") {
                    self.push_event(
                        SceneEventKind::PersonStoodUp,
                        vec![key.clone()],
                        now.posture.confidence.unwrap_or(0.0),
                        stamp,
                        None,
                    )?;
                }
            }
            if matches!(entity.kind, SceneEntityKind::Object(_))
                && let (Some(now), Some(before)) = (entity.bbox, previous.bbox)
                && bbox_center_distance(now, before) >= 0.05
            {
                self.push_event(
                    SceneEventKind::ObjectMoved,
                    vec![key.clone()],
                    1.0,
                    stamp,
                    None,
                )?;
            }
        }
        let current_activities: BTreeMap<ActivityKey, (Vec<EntityKey>, f32)> = activities
            .iter()
            .map(|activity| {
                let mut participants = activity.participants.clone();
                participants.sort();
                (
                    ActivityKey {
                        label: activity.label.clone(),
                        participants: participants.clone(),
                    },
                    (participants, activity.fact.confidence.unwrap_or(0.0)),
                )
            })
            .collect();
        for (key, (participants, confidence)) in &current_activities {
            if have_previous_tick && !previous_activities.contains_key(key) {
                self.push_event(
                    SceneEventKind::ActivityStarted,
                    participants.clone(),
                    *confidence,
                    stamp,
                    None,
                )?;
                let interaction = if key.label == "holding_drink" {
                    Some(SceneEventKind::ObjectHeld)
                } else if key.label.starts_with("using_object(") {
                    Some(SceneEventKind::ObjectTouched)
                } else {
                    None
                };
                if let Some(kind) = interaction {
                    self.push_event(kind, participants.clone(), *confidence, stamp, None)?;
                }
            }
        }
        let mut current_zones: BTreeMap<u64, Vec<EntityKey>> = zones
            .iter()
            .filter(|zone| !zone.members.is_empty())
            .map(|zone| (zone.id, zone.members.clone()))
            .collect();
        for (id, members) in &current_zones {
            if have_previous_tick
                && previous_zones
                    .get(id)
                    .is_none_or(|previous| members.iter().any(|member| !previous.contains(member)))
            {
                self.push_event(
                    SceneEventKind::ZoneOccupied,
                    members.clone(),
                    1.0,
                    stamp,
                    None,
                )?;
            }
        }
        for (id, members) in &previous_zones {
            if have_previous_tick
                && !current_zones.contains_key(id)
                && !members
                    .iter()
                    .any(|member| retained_entity_keys.contains(member))
            {
                self.push_event(SceneEventKind::ZoneFreed, members.clone(), 1.0, stamp, None)?;
            }
        }
        if let Some(previous) = self.previous_anomaly_score
            && anomaly_score.is_finite()
            && anomaly_score >= 0.5
            && anomaly_score - previous >= 0.25
        {
            self.push_event(
                SceneEventKind::AnomalySpike,
                Vec::new(),
                anomaly_score,
                stamp,
                None,
            )?;
        }
        for event in ended {
            let entity_keys = participant_keys(&event.participants, entities);
            let payload = SceneEventPayload::ActivityEpisode {
                label: event.label.clone(),
                started_at_s: event.started_at,
                ended_at_s: event.ended_at,
                duration_s: event.duration_s,
                peak_confidence: event.peak_confidence,
                stable_participants: event.stable_participants,
            };
            self.push_event(
                SceneEventKind::ActivityEnded,
                entity_keys.clone(),
                event.peak_confidence,
                stamp,
                Some(payload),
            )?;
            if event.label == "holding_drink" || event.label.starts_with("using_object(") {
                self.push_event(
                    SceneEventKind::ObjectReleased,
                    entity_keys,
                    event.peak_confidence,
                    stamp,
                    None,
                )?;
            }
        }
        // Retention reserves identity and transition history, not presence. This
        // prevents a short accepted occlusion from fabricating exit/enter and
        // zone-free/occupied event pairs while the current frame stays empty.
        for key in retained_entity_keys {
            if !current.contains_key(key)
                && let Some(previous) = previous_entities.get(key)
            {
                current.insert(key.clone(), previous.clone());
            }
        }
        for (id, members) in previous_zones {
            if let std::collections::btree_map::Entry::Vacant(entry) = current_zones.entry(id) {
                let retained_members: Vec<_> = members
                    .into_iter()
                    .filter(|member| retained_entity_keys.contains(member))
                    .collect();
                if !retained_members.is_empty() {
                    entry.insert(retained_members);
                }
            }
        }
        self.previous_entities = current;
        self.previous_activities = current_activities
            .into_iter()
            .map(|(key, (participants, _))| (key, participants))
            .collect();
        self.previous_zones = current_zones;
        self.previous_anomaly_score = anomaly_score.is_finite().then_some(anomaly_score);
        self.has_previous_tick = true;
        Ok(())
    }

    fn push_event(
        &mut self,
        kind: SceneEventKind,
        entity_keys: Vec<EntityKey>,
        confidence: f32,
        stamp: &FrameStamp,
        payload: Option<SceneEventPayload>,
    ) -> anyhow::Result<()> {
        let seq = self.next_event_seq;
        self.next_event_seq = self.next_event_seq.saturating_add(1);
        self.events.push_back(SceneEvent {
            event_seq: seq,
            source_frame_id: stamp.frame_id,
            source_timestamp_unix_ms: stamp.published_at_unix_ms,
            kind,
            entity_keys,
            fact: SceneFact::claimed(
                FactStatus::Temporal,
                true,
                confidence.clamp(0.0, 1.0),
                vec![FactEvidenceRef {
                    source: FactSource::Fusion,
                    source_frame_id: stamp.frame_id,
                    entity: None,
                    event_seq: Some(seq),
                }],
            )?,
            payload,
        });
        while self.events.len() > EVENT_TAIL_CAP {
            self.events.pop_front();
        }
        Ok(())
    }

    fn apply_kinematics(
        &mut self,
        entities: &mut [SceneEntity],
        stamp: &FrameStamp,
    ) -> anyhow::Result<()> {
        let mut current = BTreeMap::new();
        for entity in entities.iter_mut() {
            let Some(bbox) = entity.bbox.value else {
                continue;
            };
            let pose = PreviousPose {
                captured_at_unix_ms: stamp.captured_at_unix_ms,
                bbox_center: [(bbox.x1 + bbox.x2) * 0.5, (bbox.y1 + bbox.y2) * 0.5],
                position_3d: entity.position_3d.value,
            };
            if let Some(previous) = self.previous_poses.get(&entity.key) {
                let dt_s =
                    (pose.captured_at_unix_ms - previous.captured_at_unix_ms) as f32 / 1_000.0;
                if dt_s > 0.0 {
                    let velocity_3d_mps = previous
                        .position_3d
                        .zip(pose.position_3d)
                        .map(|(a, b)| [(b.x - a.x) / dt_s, (b.y - a.y) / dt_s, (b.z - a.z) / dt_s]);
                    entity.kinematics = SceneFact::claimed(
                        FactStatus::Temporal,
                        Kinematics {
                            bbox_velocity: [
                                (pose.bbox_center[0] - previous.bbox_center[0]) / dt_s,
                                (pose.bbox_center[1] - previous.bbox_center[1]) / dt_s,
                            ],
                            velocity_3d_mps,
                        },
                        1.0,
                        vec![evidence(
                            FactSource::Fusion,
                            stamp.frame_id,
                            Some(entity.key.clone()),
                        )],
                    )?;
                }
            }
            current.insert(entity.key.clone(), pose);
        }
        self.previous_poses = current;
        Ok(())
    }
}

fn source(sources: &[SourceStamp], wanted: FactSource) -> Option<&SourceStamp> {
    sources.iter().find(|source| source.source == wanted)
}

fn source_evidence(
    source: FactSource,
    stamp: Option<&SourceStamp>,
    entity: &EntityKey,
) -> Vec<FactEvidenceRef> {
    stamp
        .filter(|stamp| stamp.freshness != SourceFreshness::Unavailable)
        .map(|stamp| evidence(source, stamp.source_frame_id, Some(entity.clone())))
        .into_iter()
        .collect()
}

fn evidence(
    source: FactSource,
    source_frame_id: u64,
    entity: Option<EntityKey>,
) -> FactEvidenceRef {
    FactEvidenceRef {
        source,
        source_frame_id,
        entity,
        event_seq: None,
    }
}

fn bbox_center_distance(a: BBox, b: BBox) -> f32 {
    let dx = (a.x1 + a.x2 - b.x1 - b.x2) * 0.5;
    let dy = (a.y1 + a.y2 - b.y1 - b.y2) * 0.5;
    (dx * dx + dy * dy).sqrt()
}

fn observed<T>(
    value: T,
    confidence: f32,
    source: FactSource,
    stamp: Option<&SourceStamp>,
    entity: &EntityKey,
) -> SceneFact<T> {
    let refs = source_evidence(source, stamp, entity);
    if refs.is_empty() {
        SceneFact::unknown(FactReason::MissingSource)
    } else {
        SceneFact::claimed(
            FactStatus::Observed,
            value,
            confidence.clamp(0.0, 1.0),
            refs,
        )
        .expect("clamped observed fact is valid")
    }
}

fn position_fact(
    position: Option<Position3D>,
    source: FactSource,
    stamp: Option<&SourceStamp>,
    entity: &EntityKey,
) -> SceneFact<Position3D> {
    match position {
        Some(position) if stamp.is_some_and(|stamp| stamp.freshness != SourceFreshness::Stale) => {
            observed(position, position.confidence, source, stamp, entity)
        }
        Some(_) => SceneFact::unknown(FactReason::StaleSource),
        None => SceneFact::unknown(FactReason::Uncalibrated),
    }
}

fn metric_geometry(
    observations: &[MetricGeometryObservation],
    domain: TrackDomain,
    track_id: u32,
) -> Option<&MetricGeometryObservation> {
    observations
        .iter()
        .find(|value| value.track_domain == domain && value.track_id == track_id)
}

fn direct_metric_fact(
    measurement: Option<&MetricGeometryObservation>,
    entity: &EntityKey,
) -> SceneFact<DirectMetricGeometry> {
    let Some(value) = measurement else {
        return SceneFact::unknown(FactReason::MissingSource);
    };
    if matches!(
        value.source_freshness,
        SourceFreshness::Stale | SourceFreshness::Unavailable
    ) {
        return SceneFact::unknown(FactReason::StaleSource);
    }
    let confidence = (value.valid_ratio * (1.0 - value.boundary_contamination)).clamp(0.0, 1.0);
    let status = if value.weak || value.boundary_contamination > 0.5 || confidence < 0.5 {
        FactStatus::Weak
    } else {
        FactStatus::Observed
    };
    SceneFact::claimed(
        status,
        DirectMetricGeometry {
            centroid_m: value.centroid_m,
            aabb_min_m: value.aabb_min_m,
            aabb_max_m: value.aabb_max_m,
            median_z_m: value.median_z_m,
            median_absolute_deviation_m: value.median_absolute_deviation_m,
            sample_count: value.sample_count,
            valid_ratio: value.valid_ratio,
            boundary_contamination: value.boundary_contamination,
            reference_frame: GeometryReferenceFrame::CameraOptical,
            calibration_revision: value.calibration_revision.clone(),
            mask_aware: value.mask_aware,
        },
        confidence,
        vec![evidence(
            FactSource::MetricDepth,
            value.source_frame_id,
            Some(entity.clone()),
        )],
    )
    .expect("validated direct metric measurement is canonical")
}

fn preferred_position(
    direct: &SceneFact<DirectMetricGeometry>,
    fallback: &SceneFact<Position3D>,
) -> SceneFact<Position3D> {
    direct.value.as_ref().map_or_else(
        || fallback.clone(),
        |metric| {
            SceneFact::claimed(
                direct.status,
                Position3D {
                    x: metric.centroid_m[0],
                    y: metric.centroid_m[1],
                    z: metric.centroid_m[2],
                    confidence: direct.confidence.unwrap_or(0.0),
                },
                direct.confidence.unwrap_or(0.0),
                direct.evidence.clone(),
            )
            .expect("direct metric position is canonical")
        },
    )
}

fn person_entity(
    person: &PersonState,
    key: EntityKey,
    source: Option<&SourceStamp>,
    metric: Option<&MetricGeometryObservation>,
) -> SceneEntity {
    let posture = match &person.posture {
        Some((label, confidence)) => {
            observed(label.clone(), *confidence, FactSource::Human, source, &key)
        }
        None if matches!(person.body_coverage, Some("upper" | "head")) => SceneFact::withheld(
            FactReason::InsufficientCoverage,
            source_evidence(FactSource::Human, source, &key),
        ),
        None => SceneFact::unknown(FactReason::MissingSource),
    };
    let emotion = person.emotion.as_ref().map_or_else(
        || SceneFact::unknown(FactReason::MissingSource),
        |(label, confidence)| observed(label.clone(), *confidence, FactSource::Human, source, &key),
    );
    let gallery_identity = person.identity.as_ref().map_or_else(
        || SceneFact::unknown(FactReason::MissingSource),
        |(name, confidence)| {
            observed(
                GalleryIdentity {
                    gallery_id: person.track_id,
                    name: name.clone(),
                },
                *confidence,
                FactSource::Human,
                source,
                &key,
            )
        },
    );
    let fallback_position = position_fact(person.position_3d, FactSource::Human, source, &key);
    let direct_metric = direct_metric_fact(metric, &key);
    let position_3d = preferred_position(&direct_metric, &fallback_position);
    SceneEntity {
        key: key.clone(),
        display_id: person.track_id,
        kind: SceneEntityKind::Person(PersonFacts {
            gallery_identity,
            posture,
            emotion,
            sensory: Box::new(PersonSensoryFacts {
                body_coverage: body_coverage_fact(person.body_coverage, source, &key),
                head_orientation: head_orientation_fact(person, source, &key),
                hands: hands_fact(person, source, &key),
            }),
        }),
        existence: observed(Existence::Present, 1.0, FactSource::Human, source, &key),
        bbox: observed(person.bbox, 1.0, FactSource::Human, source, &key),
        position_3d,
        geometry: Box::new(EntityGeometryFacts {
            direct_metric,
            fallback_position,
        }),
        // Legacy `PersonState.velocity` is scalar speed without direction; it
        // cannot honestly populate vector-valued canonical kinematics.
        kinematics: SceneFact::unknown(FactReason::MissingSource),
    }
}

fn body_coverage_fact(
    coverage: Option<&str>,
    source: Option<&SourceStamp>,
    key: &EntityKey,
) -> SceneFact<BodyCoverage> {
    let value = match coverage {
        Some("full") => Some(BodyCoverage::Full),
        Some("upper") => Some(BodyCoverage::Upper),
        Some("head") => Some(BodyCoverage::Head),
        Some("unknown") => Some(BodyCoverage::Unknown),
        _ => None,
    };
    value.map_or_else(
        || SceneFact::unknown(FactReason::MissingSource),
        |value| observed(value, 1.0, FactSource::Human, source, key),
    )
}

fn head_orientation_fact(
    person: &PersonState,
    source: Option<&SourceStamp>,
    key: &EntityKey,
) -> SceneFact<HeadOrientationFact> {
    let yaw_deg = person.head_yaw.filter(|yaw| yaw.is_finite());
    let facing = match person.head_facing {
        Some("camera") => Some(HeadFacing::Camera),
        Some("left") => Some(HeadFacing::Left),
        Some("right") => Some(HeadFacing::Right),
        _ => None,
    };
    if yaw_deg.is_none() && facing.is_none() {
        return SceneFact::unknown(FactReason::MissingSource);
    }
    // The tracker currently exposes no orientation confidence. Preserve this as
    // a weak visual observation instead of inventing detector certainty.
    SceneFact::claimed(
        FactStatus::Weak,
        HeadOrientationFact { yaw_deg, facing },
        0.5,
        source_evidence(FactSource::Human, source, key),
    )
    .expect("finite head orientation is a valid weak fact")
}

fn hands_fact(
    person: &PersonState,
    source: Option<&SourceStamp>,
    key: &EntityKey,
) -> SceneFact<Vec<HandFact>> {
    if person.hands.is_empty() {
        return SceneFact::unknown(FactReason::MissingSource);
    }
    let hands = person
        .hands
        .iter()
        .filter_map(|hand| {
            let side = match hand.side {
                "left" => HandSide::Left,
                "right" => HandSide::Right,
                _ => return None,
            };
            let gesture = match hand.gesture {
                Some("open") => Some(HandGesture::Open),
                Some("fist") => Some(HandGesture::Fist),
                Some("point") => Some(HandGesture::Point),
                Some("pinch") => Some(HandGesture::Pinch),
                Some(_) => Some(HandGesture::Unknown),
                None => None,
            };
            Some(HandFact {
                side,
                presence: hand.presence.is_finite() && hand.presence > 0.0,
                gesture,
            })
        })
        .collect::<Vec<_>>();
    if hands.is_empty() {
        return SceneFact::unknown(FactReason::MissingSource);
    }
    let confidence = person
        .hands
        .iter()
        .filter_map(|hand| hand.presence.is_finite().then_some(hand.presence))
        .fold(1.0f32, f32::min)
        .clamp(0.0, 1.0);
    observed(hands, confidence, FactSource::Human, source, key)
}

fn canonical_hand_interactions(
    people: &[PersonState],
    objects: &[ObjectState],
    entities: &[SceneEntity],
    human_source: Option<&SourceStamp>,
    scene_source: Option<&SourceStamp>,
) -> anyhow::Result<Vec<HandInteractionFact>> {
    let person_keys: BTreeMap<_, _> = entities
        .iter()
        .filter(|entity| matches!(entity.kind, SceneEntityKind::Person(_)))
        .map(|entity| (entity.display_id, entity.key.clone()))
        .collect();
    let object_keys: BTreeMap<_, _> = entities
        .iter()
        .filter(|entity| matches!(entity.kind, SceneEntityKind::Object(_)))
        .map(|entity| (entity.display_id, entity.key.clone()))
        .collect();
    let mut interactions = Vec::new();
    for person in people {
        let Some(person_key) = person_keys.get(&person.track_id) else {
            continue;
        };
        for hand in &person.hands {
            let hand_side = match hand.side {
                "left" => HandSide::Left,
                "right" => HandSide::Right,
                _ => continue,
            };
            if !hand.presence.is_finite() || hand.presence <= 0.0 {
                continue;
            }
            for object in objects {
                let Some(target_entity) = object_keys.get(&object.id) else {
                    continue;
                };
                if !hand
                    .landmarks
                    .iter()
                    .any(|point| point_in_bbox(*point, object.bbox))
                {
                    continue;
                }
                let evidence = [
                    source_evidence(FactSource::Human, human_source, person_key),
                    source_evidence(FactSource::Scene, scene_source, target_entity),
                ]
                .into_iter()
                .flatten()
                .collect();
                interactions.push(HandInteractionFact {
                    person: person_key.clone(),
                    hand_side,
                    target_entity: target_entity.clone(),
                    kind: HandInteractionKind::TouchingCandidate,
                    fact: SceneFact::claimed(
                        FactStatus::Derived,
                        true,
                        hand.presence.min(object.confidence).clamp(0.0, 1.0),
                        evidence,
                    )?,
                });
            }
        }
    }
    Ok(interactions)
}

fn point_in_bbox(point: [f32; 2], bbox: BBox) -> bool {
    point[0].is_finite()
        && point[1].is_finite()
        && point[0] >= bbox.x1
        && point[0] <= bbox.x2
        && point[1] >= bbox.y1
        && point[1] <= bbox.y2
}

fn object_entity(
    object: &ObjectState,
    key: EntityKey,
    scene: Option<&SourceStamp>,
    depth: Option<&SourceStamp>,
    metric: Option<&MetricGeometryObservation>,
) -> SceneEntity {
    let status = if object.remembered {
        FactStatus::Remembered
    } else {
        FactStatus::Observed
    };
    let refs = source_evidence(FactSource::Scene, scene, &key);
    let class_name = if refs.is_empty() {
        SceneFact::unknown(FactReason::MissingSource)
    } else {
        SceneFact::claimed(
            status,
            object.class.clone(),
            object.confidence.clamp(0.0, 1.0),
            refs.clone(),
        )
        .expect("tracked object confidence is valid")
    };
    let fallback_position = position_fact(object.position_3d, FactSource::Depth, depth, &key);
    let direct_metric = direct_metric_fact(metric, &key);
    let position_3d = preferred_position(&direct_metric, &fallback_position);
    SceneEntity {
        key: key.clone(),
        display_id: object.id,
        kind: SceneEntityKind::Object(ObjectFacts {
            class_name,
            remembered: object.remembered,
        }),
        existence: if refs.is_empty() {
            SceneFact::unknown(FactReason::MissingSource)
        } else {
            SceneFact::claimed(
                status,
                if object.remembered {
                    Existence::Occluded
                } else {
                    Existence::Present
                },
                object.confidence.clamp(0.0, 1.0),
                refs,
            )
            .expect("tracked object existence is valid")
        },
        bbox: observed(
            object.bbox,
            object.confidence,
            FactSource::Scene,
            scene,
            &key,
        ),
        position_3d,
        geometry: Box::new(EntityGeometryFacts {
            direct_metric,
            fallback_position,
        }),
        kinematics: SceneFact::unknown(FactReason::MissingSource),
    }
}

fn animal_entity(
    animal: &AnimalObservation,
    key: EntityKey,
    source: Option<&SourceStamp>,
) -> SceneEntity {
    let status = if animal.remembered {
        FactStatus::Remembered
    } else {
        FactStatus::Observed
    };
    let refs = source_evidence(FactSource::Scene, source, &key);
    let class_name = if refs.is_empty() {
        SceneFact::unknown(FactReason::MissingSource)
    } else {
        SceneFact::claimed(
            status,
            animal.class_name.clone(),
            animal.confidence.clamp(0.0, 1.0),
            refs.clone(),
        )
        .expect("tracked animal class fact is valid")
    };
    SceneEntity {
        key: key.clone(),
        display_id: animal.track_id,
        kind: SceneEntityKind::Animal(AnimalFacts { class_name }),
        existence: if refs.is_empty() {
            SceneFact::unknown(FactReason::MissingSource)
        } else {
            SceneFact::claimed(
                status,
                if animal.remembered {
                    Existence::Coasted
                } else {
                    Existence::Present
                },
                animal.confidence.clamp(0.0, 1.0),
                refs,
            )
            .expect("tracked animal existence fact is valid")
        },
        bbox: observed(
            animal.bbox,
            animal.confidence,
            FactSource::Scene,
            source,
            &key,
        ),
        position_3d: SceneFact::unknown(FactReason::Uncalibrated),
        geometry: Box::default(),
        kinematics: SceneFact::unknown(FactReason::MissingSource),
    }
}

fn legacy_kind(kind: &SceneEntityKind) -> EntityKind {
    match kind {
        SceneEntityKind::Person(_) => EntityKind::Person,
        SceneEntityKind::Object(facts) => EntityKind::Object(
            facts
                .class_name
                .value
                .clone()
                .unwrap_or_else(|| "unknown".into()),
        ),
        SceneEntityKind::Animal(facts) => EntityKind::Object(
            facts
                .class_name
                .value
                .clone()
                .unwrap_or_else(|| "animal".into()),
        ),
    }
}

fn participant_keys(ids: &[u32], entities: &[SceneEntity]) -> Vec<EntityKey> {
    ids.iter()
        .filter_map(|id| {
            entities
                .iter()
                .find(|entity| entity.display_id == *id)
                .map(|entity| entity.key.clone())
        })
        .collect()
}

fn canonical_activities(
    activities: &[Activity],
    entities: &[SceneEntity],
    source_frame_id: u64,
) -> anyhow::Result<Vec<SceneActivity>> {
    activities
        .iter()
        .map(|activity| {
            let participants = participant_keys(&activity.participants, entities);
            Ok(SceneActivity {
                label: activity.label.clone(),
                participants,
                fact: SceneFact::claimed(
                    FactStatus::Temporal,
                    true,
                    activity.confidence.clamp(0.0, 1.0),
                    vec![evidence(FactSource::Activity, source_frame_id, None)],
                )?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn person(id: u32) -> PersonState {
        PersonState {
            tracker_id: id,
            track_id: id,
            bbox: BBox {
                x1: 0.1,
                y1: 0.1,
                x2: 0.3,
                y2: 0.7,
            },
            face_bbox: None,
            identity: None,
            emotion: None,
            posture: None,
            position_3d: None,
            emotion_distribution: None,
            wrists: Vec::new(),
            hands: Vec::new(),
            body_coverage: Some("upper"),
            torso_attitude: Some("upright"),
            motion_state: None,
            velocity: None,
            left_gesture: None,
            right_gesture: None,
            recon_error: None,
            plausibility: None,
            torso_flexion: None,
            head_yaw: None,
            head_facing: None,
            last_seen: Instant::now(),
        }
    }

    fn object(id: u32) -> ObjectState {
        ObjectState {
            id,
            class: "cup".into(),
            bbox: BBox {
                x1: 0.15,
                y1: 0.2,
                x2: 0.25,
                y2: 0.4,
            },
            confidence: 0.9,
            age: 1,
            position_3d: None,
            mask: None,
            remembered: false,
            last_seen: Instant::now(),
        }
    }

    fn input<'a>(frame_id: u64, people: &'a [PersonState]) -> CanonicalFusionInput<'a> {
        let captured_at = 1_000 + i64::try_from(frame_id).unwrap() * 100;
        CanonicalFusionInput {
            stamp: FrameStamp::new("run", frame_id, captured_at, captured_at + 1).unwrap(),
            sources: vec![
                SourceStamp::at_frame(FactSource::Human, frame_id, captured_at + 1)
                    .relative_to(frame_id, 2, 4)
                    .unwrap(),
            ],
            people,
            objects: &[],
            animals: &[],
            activities: &[],
            ended_activities: &[],
            anomaly_score: 0.0,
            metric_geometry: &[],
            retained_people: &[],
            retired_people: &[],
            retained_objects: &[],
            retired_objects: &[],
        }
    }

    fn metric(domain: TrackDomain, track_id: u32) -> MetricGeometryObservation {
        MetricGeometryObservation {
            track_domain: domain,
            track_id,
            source_frame_id: 1,
            produced_at_unix_ms: 1_001,
            source_freshness: SourceFreshness::Fresh,
            centroid_m: [0.1, -0.2, 1.5],
            aabb_min_m: [0.0, -0.3, 1.4],
            aabb_max_m: [0.2, -0.1, 1.6],
            median_z_m: 1.5,
            median_absolute_deviation_m: 0.02,
            sample_count: 128,
            valid_ratio: 0.9,
            boundary_contamination: 0.1,
            calibration_revision: "a".repeat(64),
            mask_aware: false,
            weak: false,
        }
    }

    fn metric_sources(frame_id: u64) -> Vec<SourceStamp> {
        vec![
            SourceStamp::at_frame(FactSource::Scene, frame_id, 1_001)
                .relative_to(frame_id, 0, 0)
                .unwrap(),
            SourceStamp::at_frame(FactSource::Depth, frame_id, 1_001)
                .relative_to(frame_id, 0, 0)
                .unwrap(),
            SourceStamp::at_frame(FactSource::MetricDepth, frame_id, 1_001)
                .relative_to(frame_id, 0, 0)
                .unwrap(),
        ]
    }

    #[test]
    fn object_without_person_uses_direct_metric_position_and_provenance() {
        let mut fusion = CanonicalFusion::new("run");
        let object = object(10_000);
        let geometry = [metric(TrackDomain::Object, object.id)];
        let frame = fusion
            .assemble(CanonicalFusionInput {
                sources: metric_sources(1),
                objects: std::slice::from_ref(&object),
                metric_geometry: &geometry,
                ..input(1, &[])
            })
            .unwrap();
        let entity = &frame.entities[0];
        assert_eq!(entity.position_3d.value.unwrap().z, 1.5);
        assert_eq!(
            entity.geometry.direct_metric.evidence[0].source,
            FactSource::MetricDepth
        );
        assert_eq!(entity.geometry.direct_metric.evidence[0].source_frame_id, 1);
        assert_eq!(
            entity
                .geometry
                .direct_metric
                .value
                .as_ref()
                .unwrap()
                .reference_frame,
            GeometryReferenceFrame::CameraOptical
        );
    }

    #[test]
    fn future_or_duplicate_metric_geometry_is_rejected_before_fusion() {
        let object = object(10_000);
        let mut future = metric(TrackDomain::Object, object.id);
        future.source_frame_id = 2;
        let mut fusion = CanonicalFusion::new("run");
        assert!(
            fusion
                .assemble(CanonicalFusionInput {
                    objects: std::slice::from_ref(&object),
                    metric_geometry: std::slice::from_ref(&future),
                    ..input(1, &[])
                })
                .unwrap_err()
                .to_string()
                .contains("future frame")
        );

        let duplicate = metric(TrackDomain::Object, object.id);
        let geometry = [duplicate.clone(), duplicate];
        assert!(
            fusion
                .assemble(CanonicalFusionInput {
                    objects: std::slice::from_ref(&object),
                    metric_geometry: &geometry,
                    ..input(1, &[])
                })
                .unwrap_err()
                .to_string()
                .contains("duplicate metric geometry")
        );
    }

    #[test]
    fn retained_gap_keeps_incarnation_until_explicit_retirement() {
        let mut fusion = CanonicalFusion::new("run");
        let observed = person(10);
        let retained = [10];
        let first = fusion
            .assemble(CanonicalFusionInput {
                retained_people: &retained,
                ..input(1, std::slice::from_ref(&observed))
            })
            .unwrap();
        let first_key = first.entities[0].key.clone();

        let gap = fusion
            .assemble(CanonicalFusionInput {
                retained_people: &retained,
                ..input(2, &[])
            })
            .unwrap();
        assert!(gap.entities.is_empty());
        assert!(gap.activities.is_empty());
        assert!(gap.relations.is_empty());
        assert!(gap.zones.is_empty());
        assert!(
            gap.events.events.iter().all(|event| !matches!(
                event.kind,
                SceneEventKind::EntityDisappeared | SceneEventKind::PersonExited
            )),
            "retained absence is not a canonical disappearance"
        );

        let returned = fusion
            .assemble(CanonicalFusionInput {
                retained_people: &retained,
                ..input(3, std::slice::from_ref(&observed))
            })
            .unwrap();
        assert_eq!(returned.entities[0].key, first_key);
        assert!(
            returned.events.events.iter().all(|event| !matches!(
                event.kind,
                SceneEventKind::EntityAppeared | SceneEventKind::PersonEntered
            )),
            "return inside retention must not fabricate a new arrival"
        );

        let retired = [10];
        let expiry = fusion
            .assemble(CanonicalFusionInput {
                retired_people: &retired,
                ..input(4, &[])
            })
            .unwrap();
        assert!(expiry.events.events.iter().any(|event| {
            event.kind == SceneEventKind::EntityDisappeared
                && event.entity_keys == [first_key.clone()]
        }));
        assert!(expiry.events.events.iter().any(|event| {
            event.kind == SceneEventKind::PersonExited && event.entity_keys == [first_key.clone()]
        }));
        let reincarnated = fusion
            .assemble(CanonicalFusionInput {
                retained_people: &retained,
                ..input(5, std::slice::from_ref(&observed))
            })
            .unwrap();
        assert_ne!(reincarnated.entities[0].key, first_key);
        assert_eq!(reincarnated.entities[0].key.incarnation, 1);
    }

    #[test]
    fn gallery_recognition_changes_display_id_without_reincarnating_person() {
        let mut fusion = CanonicalFusion::new("run");
        let mut anonymous = person(10);
        let first = fusion
            .assemble(input(1, std::slice::from_ref(&anonymous)))
            .unwrap();
        let key = first.entities[0].key.clone();
        assert_eq!(first.entities[0].display_id, 10);

        anonymous.track_id = 0;
        anonymous.identity = Some(("alice".into(), 0.9));
        let recognized = fusion
            .assemble(input(2, std::slice::from_ref(&anonymous)))
            .unwrap();

        assert_eq!(recognized.entities[0].key, key);
        assert_eq!(recognized.entities[0].display_id, 0);
        assert!(recognized.events.events.iter().all(|event| !matches!(
            event.kind,
            SceneEventKind::EntityAppeared
                | SceneEventKind::EntityDisappeared
                | SceneEventKind::PersonEntered
                | SceneEventKind::PersonExited
        )));
    }

    #[test]
    fn mask_aware_direct_centroid_and_reference_frame_serialize() {
        let mut fusion = CanonicalFusion::new("run");
        let object = object(10_000);
        let mut measured = metric(TrackDomain::Object, object.id);
        measured.mask_aware = true;
        measured.centroid_m = [0.3, 0.4, 0.8];
        let geometry = [measured];
        let frame = fusion
            .assemble(CanonicalFusionInput {
                sources: metric_sources(1),
                objects: std::slice::from_ref(&object),
                metric_geometry: &geometry,
                ..input(1, &[])
            })
            .unwrap();
        let json = serde_json::to_value(&frame).unwrap();
        assert_eq!(
            json["entities"][0]["geometry"]["direct_metric"]["value"]["mask_aware"],
            true
        );
        assert_eq!(
            json["entities"][0]["geometry"]["direct_metric"]["value"]["reference_frame"],
            "camera_optical"
        );
    }

    #[test]
    fn invalid_boundary_coverage_downgrades_direct_measurement_to_weak() {
        let mut fusion = CanonicalFusion::new("run");
        let object = object(10_000);
        let mut measured = metric(TrackDomain::Object, object.id);
        measured.boundary_contamination = 0.75;
        let geometry = [measured];
        let frame = fusion
            .assemble(CanonicalFusionInput {
                sources: metric_sources(1),
                objects: std::slice::from_ref(&object),
                metric_geometry: &geometry,
                ..input(1, &[])
            })
            .unwrap();
        assert_eq!(
            frame.entities[0].geometry.direct_metric.status,
            FactStatus::Weak
        );
        assert_eq!(
            frame.entities[0].geometry.direct_metric.confidence,
            Some(0.225)
        );
    }

    #[test]
    fn person_direct_geometry_preserves_separately_stamped_fallback() {
        let mut fusion = CanonicalFusion::new("run");
        let mut person = person(10);
        person.position_3d = Some(Position3D {
            x: 0.0,
            y: 0.0,
            z: 2.2,
            confidence: 0.6,
        });
        let geometry = [metric(TrackDomain::Person, person.tracker_id)];
        let mut sources = metric_sources(1);
        sources.push(
            SourceStamp::at_frame(FactSource::Human, 1, 1_001)
                .relative_to(1, 0, 0)
                .unwrap(),
        );
        let frame = fusion
            .assemble(CanonicalFusionInput {
                sources,
                metric_geometry: &geometry,
                ..input(1, std::slice::from_ref(&person))
            })
            .unwrap();
        let entity = &frame.entities[0];
        assert_eq!(entity.position_3d.value.unwrap().z, 1.5);
        assert_eq!(entity.geometry.fallback_position.value.unwrap().z, 2.2);
        assert_eq!(
            entity.geometry.fallback_position.evidence[0].source,
            FactSource::Human
        );
    }

    #[test]
    fn assembler_preserves_withheld_posture_and_source_frame() {
        let mut fusion = CanonicalFusion::new("run");
        let people = [person(10)];
        let frame = fusion.assemble(input(7, &people)).unwrap();
        assert_eq!(frame.sources[0].source_frame_id, 7);
        let SceneEntityKind::Person(facts) = &frame.entities[0].kind else {
            panic!("person entity");
        };
        assert_eq!(facts.posture.status, FactStatus::Withheld);
        assert!(facts.posture.value.is_none());
    }

    #[test]
    fn canonical_person_semantics_and_hand_contact_keep_evidence_boundary() {
        let mut fusion = CanonicalFusion::new("run");
        let mut person = person(10);
        person.body_coverage = Some("upper");
        person.head_yaw = Some(18.0);
        person.head_facing = Some("right");
        person.hands = vec![mvm_core::HandState {
            landmarks: vec![[0.2, 0.3]],
            presence: 0.9,
            side: "right",
            handedness: 0.9,
            gesture: Some("pinch"),
        }];
        let object = object(20);
        let people = [person];
        let mut frame_input = input(7, &people);
        frame_input.objects = std::slice::from_ref(&object);
        frame_input.sources.push(
            SourceStamp::at_frame(FactSource::Scene, 7, 1_701)
                .relative_to(7, 2, 4)
                .unwrap(),
        );
        let frame = fusion.assemble(frame_input).unwrap();
        let SceneEntityKind::Person(facts) = &frame.entities[0].kind else {
            panic!("person entity");
        };
        assert_eq!(facts.sensory.body_coverage.value, Some(BodyCoverage::Upper));
        assert_eq!(facts.sensory.head_orientation.status, FactStatus::Weak);
        assert_eq!(
            facts
                .sensory
                .head_orientation
                .value
                .as_ref()
                .unwrap()
                .yaw_deg,
            Some(18.0)
        );
        assert_eq!(
            facts.sensory.hands.value.as_ref().unwrap()[0].side,
            HandSide::Right
        );
        assert_eq!(
            facts.sensory.hands.value.as_ref().unwrap()[0].gesture,
            Some(HandGesture::Pinch)
        );
        assert_eq!(frame.hand_interactions.len(), 1);
        assert_eq!(
            frame.hand_interactions[0].kind,
            HandInteractionKind::TouchingCandidate
        );
        assert!(
            frame.hand_interactions[0]
                .fact
                .evidence
                .iter()
                .any(|item| item.source == FactSource::Human)
        );
        assert!(
            frame.hand_interactions[0]
                .fact
                .evidence
                .iter()
                .any(|item| item.source == FactSource::Scene)
        );
        assert!(frame.hand_interactions[0].fact.value == Some(true));
    }

    #[test]
    fn absent_then_reused_local_id_gets_new_incarnation() {
        let mut fusion = CanonicalFusion::new("run");
        let people = [person(10)];
        let first = fusion.assemble(input(1, &people)).unwrap().entities[0]
            .key
            .clone();
        fusion.assemble(input(2, &[])).unwrap();
        let reused = fusion.assemble(input(3, &people)).unwrap().entities[0]
            .key
            .clone();
        assert_eq!(reused.incarnation, first.incarnation + 1);
    }

    #[test]
    fn kinematics_are_derived_from_same_entity_key_across_camera_ticks() {
        let mut fusion = CanonicalFusion::new("run");
        let first = [person(10)];
        fusion.assemble(input(1, &first)).unwrap();

        let mut moved = person(10);
        moved.bbox.x1 += 0.1;
        moved.bbox.x2 += 0.1;
        let second = fusion.assemble(input(2, &[moved])).unwrap();
        let kinematics = second.entities[0]
            .kinematics
            .value
            .expect("same incarnation has temporal kinematics");
        assert!((kinematics.bbox_velocity[0] - 1.0).abs() < 1e-5);
        assert!(kinematics.velocity_3d_mps.is_none());
        assert_eq!(second.entities[0].kinematics.status, FactStatus::Temporal);
    }

    #[test]
    fn ended_activity_becomes_typed_sequenced_scene_event() {
        let mut fusion = CanonicalFusion::new("run");
        let people = [person(10)];
        let ended = [ActivityEvent {
            label: "working".into(),
            participants: vec![10],
            started_at: 1.0,
            ended_at: 4.5,
            duration_s: 3.5,
            peak_confidence: 0.9,
            stable_participants: true,
        }];
        let mut frame_input = input(7, &people);
        frame_input.ended_activities = &ended;

        let frame = fusion.assemble(frame_input).unwrap();
        let event = frame
            .events
            .events
            .iter()
            .find(|event| event.kind == SceneEventKind::ActivityEnded)
            .expect("activity-ended event");
        assert_eq!(event.kind, SceneEventKind::ActivityEnded);
        assert_eq!(event.entity_keys, vec![frame.entities[0].key.clone()]);
        assert_eq!(
            event.payload,
            Some(SceneEventPayload::ActivityEpisode {
                label: "working".into(),
                started_at_s: 1.0,
                ended_at_s: 4.5,
                duration_s: 3.5,
                peak_confidence: 0.9,
                stable_participants: true,
            })
        );
        frame.validate().unwrap();
    }

    #[test]
    fn replay_is_deterministic_independent_of_sink_read_cadence() {
        let mut eager = CanonicalFusion::new("run");
        let mut sparse = CanonicalFusion::new("run");
        let mut eager_last = None;
        let mut sparse_last = None;

        for frame_id in 1..=3 {
            let mut observed = person(10);
            observed.bbox.x1 += frame_id as f32 * 0.01;
            observed.bbox.x2 += frame_id as f32 * 0.01;
            let people = [observed];
            eager_last = Some(eager.assemble(input(frame_id, &people)).unwrap());
            let published = sparse.assemble(input(frame_id, &people)).unwrap();
            if frame_id == 3 {
                sparse_last = Some(published);
            }
        }

        assert_eq!(eager_last.unwrap(), sparse_last.unwrap());
    }

    #[test]
    fn synthetic_replay_emits_observed_transition_taxonomy() {
        let mut fusion = CanonicalFusion::new("run");
        let mut standing = person(10);
        standing.posture = Some(("standing".into(), 0.9));
        let cup = object(10_000);
        fusion
            .assemble(CanonicalFusionInput {
                objects: std::slice::from_ref(&cup),
                anomaly_score: 0.1,
                sources: sources(1),
                ..input(1, &[standing])
            })
            .unwrap();

        let mut sitting = person(10);
        sitting.posture = Some(("sitting".into(), 0.9));
        let newcomer = person(11);
        let mut moved_cup = cup.clone();
        moved_cup.bbox.x1 += 0.1;
        moved_cup.bbox.x2 += 0.1;
        let activities = [
            Activity {
                label: "holding_drink".into(),
                confidence: 0.8,
                participants: vec![10, 10_000],
            },
            Activity {
                label: "using_object(cup)".into(),
                confidence: 0.8,
                participants: vec![10, 10_000],
            },
        ];
        let second = fusion
            .assemble(CanonicalFusionInput {
                objects: &[moved_cup],
                activities: &activities,
                anomaly_score: 0.7,
                sources: sources(2),
                ..input(2, &[sitting, newcomer])
            })
            .unwrap();
        let ended = [ActivityEvent {
            label: "holding_drink".into(),
            participants: vec![10, 10_000],
            started_at: 1.0,
            ended_at: 2.0,
            duration_s: 1.0,
            peak_confidence: 0.8,
            stable_participants: true,
        }];
        let mut restanding = person(10);
        restanding.posture = Some(("standing".into(), 0.9));
        fusion
            .assemble(CanonicalFusionInput {
                ended_activities: &ended,
                anomaly_score: 0.7,
                sources: sources(3),
                ..input(3, &[restanding])
            })
            .unwrap();
        let fourth = fusion
            .assemble(CanonicalFusionInput {
                anomaly_score: 0.7,
                sources: sources(4),
                ..input(4, &[])
            })
            .unwrap();
        let kinds: Vec<_> = fourth
            .events
            .events
            .iter()
            .map(|event| event.kind)
            .collect();
        for kind in [
            SceneEventKind::EntityAppeared,
            SceneEventKind::PersonEntered,
            SceneEventKind::EntityDisappeared,
            SceneEventKind::PersonExited,
            SceneEventKind::PersonSatDown,
            SceneEventKind::PersonStoodUp,
            SceneEventKind::ObjectMoved,
            SceneEventKind::ObjectTouched,
            SceneEventKind::ObjectHeld,
            SceneEventKind::ObjectReleased,
            SceneEventKind::ActivityStarted,
            SceneEventKind::ActivityEnded,
            SceneEventKind::AnomalySpike,
            SceneEventKind::ZoneOccupied,
            SceneEventKind::ZoneFreed,
        ] {
            assert!(kinds.contains(&kind), "missing {kind:?}; {kinds:?}");
        }
        assert!(second.events.events.iter().all(|event| {
            event
                .fact
                .evidence
                .iter()
                .any(|evidence| evidence.event_seq == Some(event.event_seq))
        }));
    }

    fn sources(frame_id: u64) -> Vec<SourceStamp> {
        let at = 1_000 + i64::try_from(frame_id).unwrap() * 100 + 1;
        vec![
            SourceStamp::at_frame(FactSource::Human, frame_id, at)
                .relative_to(frame_id, 2, 4)
                .unwrap(),
            SourceStamp::at_frame(FactSource::Scene, frame_id, at)
                .relative_to(frame_id, 2, 4)
                .unwrap(),
        ]
    }
}
