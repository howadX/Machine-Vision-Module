//! Canonical immutable scene contract (phase 18.0).
//!
//! This module defines the facts substrate consumed by runtime sinks and the
//! future visual cortex. It deliberately contains no detector orchestration and
//! no raw RGB/depth/mask payloads. Wire timestamps use Unix milliseconds so
//! replay fixtures are portable; runtime monotonic clocks stay in the mesh.

use crate::{Activity, BBox, ObjectState, PersonState, Position3D, WorldState};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::Deref;
use std::time::Instant;

/// One published camera-driven scene revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameStamp {
    pub run_id: String,
    pub frame_id: u64,
    pub captured_at_unix_ms: i64,
    pub published_at_unix_ms: i64,
}

impl FrameStamp {
    pub fn new(
        run_id: impl Into<String>,
        frame_id: u64,
        captured_at_unix_ms: i64,
        published_at_unix_ms: i64,
    ) -> Result<Self, CanonicalContractError> {
        let stamp = Self {
            run_id: run_id.into(),
            frame_id,
            captured_at_unix_ms,
            published_at_unix_ms,
        };
        if stamp.run_id.is_empty() {
            return Err(CanonicalContractError::new("run_id must not be empty"));
        }
        if stamp.published_at_unix_ms < stamp.captured_at_unix_ms {
            return Err(CanonicalContractError::new(
                "published timestamp precedes capture timestamp",
            ));
        }
        Ok(stamp)
    }
}

/// Producer of a scene fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactSource {
    Capture,
    Human,
    Scene,
    Depth,
    MetricDepth,
    Tracker,
    Fusion,
    Activity,
    Compatibility,
}

/// Freshness of one expert result relative to the fused camera revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFreshness {
    Fresh,
    Coasted,
    Stale,
    Unavailable,
}

/// Causal stamp for one expert output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceStamp {
    pub source: FactSource,
    pub source_frame_id: u64,
    pub produced_at_unix_ms: i64,
    pub age_frames: u64,
    pub freshness: SourceFreshness,
}

impl SourceStamp {
    /// Construct an available source before classifying it against a fused tick.
    pub fn at_frame(source: FactSource, source_frame_id: u64, produced_at_unix_ms: i64) -> Self {
        Self {
            source,
            source_frame_id,
            produced_at_unix_ms,
            age_frames: 0,
            freshness: SourceFreshness::Fresh,
        }
    }

    /// Classify freshness using frame budgets. Future-source claims are invalid:
    /// a late result may join a later tick, never rewrite an earlier one.
    pub fn relative_to(
        mut self,
        fused_frame_id: u64,
        fresh_through: u64,
        coasted_through: u64,
    ) -> Result<Self, CanonicalContractError> {
        if coasted_through < fresh_through {
            return Err(CanonicalContractError::new(
                "coasted freshness budget is smaller than fresh budget",
            ));
        }
        self.age_frames = fused_frame_id
            .checked_sub(self.source_frame_id)
            .ok_or_else(|| CanonicalContractError::new("source frame is newer than fused frame"))?;
        self.freshness = if self.age_frames <= fresh_through {
            SourceFreshness::Fresh
        } else if self.age_frames <= coasted_through {
            SourceFreshness::Coasted
        } else {
            SourceFreshness::Stale
        };
        Ok(self)
    }

    pub fn unavailable(source: FactSource) -> Self {
        Self {
            source,
            source_frame_id: 0,
            produced_at_unix_ms: 0,
            age_frames: 0,
            freshness: SourceFreshness::Unavailable,
        }
    }
}

/// Honesty status of an evidence-backed world fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactStatus {
    Observed,
    Derived,
    Temporal,
    Remembered,
    Weak,
    Withheld,
    Unknown,
}

impl FactStatus {
    pub fn is_asserted(self) -> bool {
        matches!(
            self,
            Self::Observed | Self::Derived | Self::Temporal | Self::Remembered | Self::Weak
        )
    }

    /// Withheld and unknown are missing claims, never negative supervision.
    pub fn can_be_negative_class(self) -> bool {
        !matches!(self, Self::Withheld | Self::Unknown)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactReason {
    InsufficientCoverage,
    LowConfidence,
    OutOfDistribution,
    StaleSource,
    MissingSource,
    Occluded,
    Uncalibrated,
    Remembered,
    Other(String),
}

/// Stable reference from a fact to the expert observation that supports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactEvidenceRef {
    pub source: FactSource,
    pub source_frame_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity: Option<EntityKey>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_seq: Option<u64>,
}

/// One typed fact. Missing claims carry neither a value nor fake confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFact<T> {
    pub status: FactStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<FactEvidenceRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<FactReason>,
}

impl<T> SceneFact<T> {
    pub fn claimed(
        status: FactStatus,
        value: T,
        confidence: f32,
        evidence: Vec<FactEvidenceRef>,
    ) -> Result<Self, CanonicalContractError> {
        let fact = Self {
            status,
            value: Some(value),
            confidence: Some(confidence),
            evidence,
            reason: None,
        };
        fact.validate()?;
        Ok(fact)
    }

    pub fn withheld(reason: FactReason, evidence: Vec<FactEvidenceRef>) -> Self {
        Self {
            status: FactStatus::Withheld,
            value: None,
            confidence: None,
            evidence,
            reason: Some(reason),
        }
    }

    pub fn unknown(reason: FactReason) -> Self {
        Self {
            status: FactStatus::Unknown,
            value: None,
            confidence: None,
            evidence: Vec::new(),
            reason: Some(reason),
        }
    }

    pub fn validate(&self) -> Result<(), CanonicalContractError> {
        if self.status.is_asserted() {
            if self.value.is_none() {
                return Err(CanonicalContractError::new(
                    "asserted fact must carry a value",
                ));
            }
            let confidence = self.confidence.ok_or_else(|| {
                CanonicalContractError::new("asserted fact must carry confidence")
            })?;
            if !(0.0..=1.0).contains(&confidence) || !confidence.is_finite() {
                return Err(CanonicalContractError::new(
                    "fact confidence must be finite and within [0,1]",
                ));
            }
            if self.evidence.is_empty() {
                return Err(CanonicalContractError::new(
                    "asserted fact must cite evidence",
                ));
            }
        } else if self.value.is_some() || self.confidence.is_some() {
            return Err(CanonicalContractError::new(
                "withheld/unknown fact must not carry value or confidence",
            ));
        }
        Ok(())
    }
}

/// Tracker namespace. Local ids are only unique within one domain and incarnation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackDomain {
    Person,
    Object,
    Animal,
}

/// Run-scoped identity of one concrete tracker incarnation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityKey {
    pub run_id: String,
    pub track_domain: TrackDomain,
    pub track_id: u32,
    pub incarnation: u32,
}

#[derive(Debug, Clone, Copy)]
struct IncarnationSlot {
    incarnation: u32,
    active: bool,
}

/// Assigns a fresh canonical identity whenever a tracker reuses a local id.
#[derive(Debug, Clone)]
pub struct IncarnationRegistry {
    run_id: String,
    slots: BTreeMap<(TrackDomain, u32), IncarnationSlot>,
}

impl IncarnationRegistry {
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            slots: BTreeMap::new(),
        }
    }

    pub fn activate(&mut self, domain: TrackDomain, track_id: u32) -> EntityKey {
        let slot = self
            .slots
            .entry((domain, track_id))
            .and_modify(|slot| {
                if !slot.active {
                    slot.incarnation = slot.incarnation.saturating_add(1);
                    slot.active = true;
                }
            })
            .or_insert(IncarnationSlot {
                incarnation: 0,
                active: true,
            });
        EntityKey {
            run_id: self.run_id.clone(),
            track_domain: domain,
            track_id,
            incarnation: slot.incarnation,
        }
    }

    pub fn retire(&mut self, key: &EntityKey) -> bool {
        if key.run_id != self.run_id {
            return false;
        }
        match self.slots.get_mut(&(key.track_domain, key.track_id)) {
            Some(slot) if slot.active && slot.incarnation == key.incarnation => {
                slot.active = false;
                true
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Existence {
    Present,
    Coasted,
    Occluded,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Kinematics {
    pub bbox_velocity: [f32; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity_3d_mps: Option<[f32; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GalleryIdentity {
    pub gallery_id: u32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonFacts {
    pub gallery_identity: SceneFact<GalleryIdentity>,
    pub posture: SceneFact<String>,
    pub emotion: SceneFact<String>,
    /// Compact sensory observations, boxed to keep object/animal scene entities
    /// cheap while retaining an additive canonical contract.
    #[serde(default)]
    pub sensory: Box<PersonSensoryFacts>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonSensoryFacts {
    /// Compact observation regime. This is evidence about visible body coverage,
    /// not an inferred posture.
    #[serde(default = "unknown_fact")]
    pub body_coverage: SceneFact<BodyCoverage>,
    /// Visual head pose only. It is never a gaze target, attention, or intent.
    #[serde(default = "unknown_fact")]
    pub head_orientation: SceneFact<HeadOrientationFact>,
    /// Bounded hand summary; raw hand landmarks remain outside canonical facts.
    #[serde(default = "unknown_fact")]
    pub hands: SceneFact<Vec<HandFact>>,
}

impl Default for PersonSensoryFacts {
    fn default() -> Self {
        Self {
            body_coverage: unknown_fact(),
            head_orientation: unknown_fact(),
            hands: unknown_fact(),
        }
    }
}

fn unknown_fact<T>() -> SceneFact<T> {
    SceneFact::unknown(FactReason::MissingSource)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyCoverage {
    Full,
    Upper,
    Head,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadFacing {
    Camera,
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadOrientationFact {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub yaw_deg: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub facing: Option<HeadFacing>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandGesture {
    Open,
    Fist,
    Point,
    Pinch,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandFact {
    pub side: HandSide,
    pub presence: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gesture: Option<HandGesture>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectFacts {
    pub class_name: SceneFact<String>,
    pub remembered: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimalFacts {
    pub class_name: SceneFact<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "facts", rename_all = "snake_case")]
pub enum SceneEntityKind {
    Person(PersonFacts),
    Object(ObjectFacts),
    Animal(AnimalFacts),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntity {
    pub key: EntityKey,
    pub display_id: u32,
    pub kind: SceneEntityKind,
    pub existence: SceneFact<Existence>,
    pub bbox: SceneFact<BBox>,
    pub position_3d: SceneFact<Position3D>,
    /// Additive metric geometry contract. `position_3d` remains the preferred
    /// compatibility projection; this field preserves direct and fallback
    /// evidence separately.
    #[serde(default)]
    pub geometry: Box<EntityGeometryFacts>,
    pub kinematics: SceneFact<Kinematics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityGeometryFacts {
    #[serde(default = "unknown_fact")]
    pub direct_metric: SceneFact<DirectMetricGeometry>,
    #[serde(default = "unknown_fact")]
    pub fallback_position: SceneFact<Position3D>,
}

impl Default for EntityGeometryFacts {
    fn default() -> Self {
        Self {
            direct_metric: unknown_fact(),
            fallback_position: unknown_fact(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryReferenceFrame {
    CameraOptical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectMetricGeometry {
    pub centroid_m: [f32; 3],
    pub aabb_min_m: [f32; 3],
    pub aabb_max_m: [f32; 3],
    pub median_z_m: f32,
    pub median_absolute_deviation_m: f32,
    pub sample_count: u32,
    pub valid_ratio: f32,
    pub boundary_contamination: f32,
    pub reference_frame: GeometryReferenceFrame,
    pub calibration_revision: String,
    pub mask_aware: bool,
}

impl DirectMetricGeometry {
    fn validate(&self) -> Result<(), CanonicalContractError> {
        let finite = self
            .centroid_m
            .iter()
            .chain(self.aabb_min_m.iter())
            .chain(self.aabb_max_m.iter())
            .chain([
                &self.median_z_m,
                &self.median_absolute_deviation_m,
                &self.valid_ratio,
                &self.boundary_contamination,
            ])
            .all(|value| value.is_finite());
        if !finite
            || self.sample_count == 0
            || self.median_z_m <= 0.0
            || self.median_absolute_deviation_m < 0.0
            || !(0.0..=1.0).contains(&self.valid_ratio)
            || !(0.0..=1.0).contains(&self.boundary_contamination)
            || self.calibration_revision.len() != 64
            || !self
                .calibration_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || (0..3).any(|axis| self.aabb_min_m[axis] > self.aabb_max_m[axis])
        {
            return Err(CanonicalContractError::new(
                "invalid direct metric geometry",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneRelationKind {
    Near,
    Overlapping,
    PartOf,
    Touching,
    HeldBy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneRelation {
    pub a: EntityKey,
    pub b: EntityKey,
    pub kind: SceneRelationKind,
    pub distance_m: SceneFact<f32>,
    pub overlap: SceneFact<f32>,
}

/// Hand-to-object candidate derived by Fusion from present hand evidence. A
/// candidate is deliberately not equivalent to a physical hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandInteractionKind {
    Near,
    TouchingCandidate,
    HoldingCandidate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandInteractionFact {
    pub person: EntityKey,
    pub hand_side: HandSide,
    pub target_entity: EntityKey,
    pub kind: HandInteractionKind,
    pub fact: SceneFact<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneZone {
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub members: Vec<EntityKey>,
    pub age_frames: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneActivity {
    pub label: String,
    pub participants: Vec<EntityKey>,
    pub fact: SceneFact<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneEventKind {
    EntityAppeared,
    EntityDisappeared,
    PersonEntered,
    PersonExited,
    PersonSatDown,
    PersonStoodUp,
    ObjectTouched,
    ObjectHeld,
    ObjectReleased,
    ObjectMoved,
    ActivityStarted,
    ActivityEnded,
    AnomalySpike,
    ZoneOccupied,
    ZoneFreed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SceneEventPayload {
    ActivityEpisode {
        label: String,
        started_at_s: f64,
        ended_at_s: f64,
        duration_s: f64,
        peak_confidence: f32,
        stable_participants: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEvent {
    pub event_seq: u64,
    pub source_frame_id: u64,
    pub source_timestamp_unix_ms: i64,
    pub kind: SceneEventKind,
    pub entity_keys: Vec<EntityKey>,
    pub fact: SceneFact<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<SceneEventPayload>,
}

/// Bounded immutable event suffix published with a scene frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EventTailView {
    pub first_seq: u64,
    pub next_seq: u64,
    pub events: Vec<SceneEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventGap {
    pub requested_after: u64,
    pub available_from: u64,
}

#[derive(Debug)]
pub struct EventDelta<'a> {
    pub events: Vec<&'a SceneEvent>,
    pub next_cursor: Option<u64>,
    pub gap: Option<EventGap>,
}

impl EventTailView {
    /// Read non-destructively after the consumer's last seen sequence.
    ///
    /// A cursor older than the bounded tail produces an explicit gap while still
    /// returning every event that remains available. Re-reading the same cursor
    /// is idempotent and never mutates the tail.
    pub fn delta_after(&self, cursor: Option<u64>) -> EventDelta<'_> {
        let gap = cursor.and_then(|requested_after| {
            if requested_after.saturating_add(1) < self.first_seq {
                Some(EventGap {
                    requested_after,
                    available_from: self.first_seq,
                })
            } else {
                None
            }
        });
        let events: Vec<&SceneEvent> = self
            .events
            .iter()
            .filter(|event| cursor.is_none_or(|seq| event.event_seq > seq))
            .collect();
        let next_cursor = events.last().map(|event| event.event_seq).or(cursor);
        EventDelta {
            events,
            next_cursor,
            gap,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SceneQuality {
    pub withheld_facts: u32,
    pub unknown_facts: u32,
    pub stale_sources: u32,
    pub event_gap: bool,
}

/// The single immutable authority for all facts of a published scene revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedSceneFrame {
    pub stamp: FrameStamp,
    pub sources: Vec<SourceStamp>,
    pub entities: Vec<SceneEntity>,
    pub relations: Vec<SceneRelation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hand_interactions: Vec<HandInteractionFact>,
    pub zones: Vec<SceneZone>,
    pub activities: Vec<SceneActivity>,
    pub events: EventTailView,
    pub anomaly: SceneFact<f32>,
    pub quality: SceneQuality,
}

impl FusedSceneFrame {
    /// Stable ordering is part of the replay/wire contract.
    pub fn canonicalize(&mut self) {
        self.sources.sort_by_key(|source| source.source);
        self.entities.sort_by(|a, b| a.key.cmp(&b.key));
        self.relations
            .sort_by(|a, b| (&a.a, &a.b, a.kind).cmp(&(&b.a, &b.b, b.kind)));
        self.hand_interactions.sort_by(|a, b| {
            (&a.person, a.hand_side, &a.target_entity, a.kind).cmp(&(
                &b.person,
                b.hand_side,
                &b.target_entity,
                b.kind,
            ))
        });
        for zone in &mut self.zones {
            zone.members.sort();
        }
        self.zones.sort_by_key(|zone| zone.id);
        for activity in &mut self.activities {
            activity.participants.sort();
        }
        self.activities
            .sort_by(|a, b| (&a.label, &a.participants).cmp(&(&b.label, &b.participants)));
        for event in &mut self.events.events {
            event.entity_keys.sort();
        }
        self.events.events.sort_by_key(|event| event.event_seq);
    }

    pub fn validate(&self) -> Result<(), CanonicalContractError> {
        for source in &self.sources {
            if source.freshness != SourceFreshness::Unavailable
                && source.source_frame_id > self.stamp.frame_id
            {
                return Err(CanonicalContractError::new(
                    "source frame is newer than fused frame",
                ));
            }
        }
        self.anomaly.validate()?;
        for entity in &self.entities {
            if entity.key.run_id != self.stamp.run_id {
                return Err(CanonicalContractError::new(
                    "entity key run_id differs from frame run_id",
                ));
            }
            entity.existence.validate()?;
            entity.bbox.validate()?;
            entity.position_3d.validate()?;
            entity.geometry.direct_metric.validate()?;
            entity.geometry.fallback_position.validate()?;
            if let Some(metric) = &entity.geometry.direct_metric.value {
                metric.validate()?;
                if entity
                    .geometry
                    .direct_metric
                    .evidence
                    .iter()
                    .any(|evidence| evidence.source != FactSource::MetricDepth)
                {
                    return Err(CanonicalContractError::new(
                        "direct metric geometry must cite metric_depth evidence",
                    ));
                }
            }
            entity.kinematics.validate()?;
            match &entity.kind {
                SceneEntityKind::Person(facts) => {
                    facts.gallery_identity.validate()?;
                    facts.posture.validate()?;
                    facts.emotion.validate()?;
                    facts.sensory.body_coverage.validate()?;
                    facts.sensory.head_orientation.validate()?;
                    facts.sensory.hands.validate()?;
                }
                SceneEntityKind::Object(facts) => facts.class_name.validate()?,
                SceneEntityKind::Animal(facts) => facts.class_name.validate()?,
            }
        }
        for relation in &self.relations {
            if relation.a.run_id != self.stamp.run_id || relation.b.run_id != self.stamp.run_id {
                return Err(CanonicalContractError::new(
                    "relation endpoint run_id differs from frame run_id",
                ));
            }
            relation.distance_m.validate()?;
            relation.overlap.validate()?;
        }
        for interaction in &self.hand_interactions {
            if interaction.person.run_id != self.stamp.run_id
                || interaction.target_entity.run_id != self.stamp.run_id
            {
                return Err(CanonicalContractError::new(
                    "hand interaction endpoint run_id differs from frame run_id",
                ));
            }
            interaction.fact.validate()?;
        }
        for zone in &self.zones {
            if zone
                .members
                .iter()
                .any(|member| member.run_id != self.stamp.run_id)
            {
                return Err(CanonicalContractError::new(
                    "zone member run_id differs from frame run_id",
                ));
            }
        }
        for activity in &self.activities {
            activity.fact.validate()?;
            if activity
                .participants
                .iter()
                .any(|participant| participant.run_id != self.stamp.run_id)
            {
                return Err(CanonicalContractError::new(
                    "activity participant run_id differs from frame run_id",
                ));
            }
        }
        if let Some(first) = self.events.events.first()
            && first.event_seq != self.events.first_seq
        {
            return Err(CanonicalContractError::new(
                "event tail first_seq differs from first event",
            ));
        }
        if let Some(last) = self.events.events.last() {
            if last.event_seq.saturating_add(1) != self.events.next_seq {
                return Err(CanonicalContractError::new(
                    "event tail next_seq does not follow last event",
                ));
            }
        } else if self.events.first_seq != self.events.next_seq {
            return Err(CanonicalContractError::new(
                "empty event tail must have first_seq equal to next_seq",
            ));
        }
        let mut previous = None;
        for event in &self.events.events {
            event.fact.validate()?;
            match (&event.kind, &event.payload) {
                (
                    SceneEventKind::ActivityEnded,
                    Some(SceneEventPayload::ActivityEpisode { .. }),
                )
                | (_, None) => {}
                _ => {
                    return Err(CanonicalContractError::new(
                        "scene event payload does not match event kind",
                    ));
                }
            }
            if event.source_frame_id > self.stamp.frame_id {
                return Err(CanonicalContractError::new(
                    "event source frame is newer than fused frame",
                ));
            }
            if previous.is_some_and(|seq| event.event_seq <= seq) {
                return Err(CanonicalContractError::new(
                    "event sequence is not strictly increasing",
                ));
            }
            if event
                .entity_keys
                .iter()
                .any(|entity| entity.run_id != self.stamp.run_id)
            {
                return Err(CanonicalContractError::new(
                    "event entity run_id differs from frame run_id",
                ));
            }
            previous = Some(event.event_seq);
        }
        Ok(())
    }

    /// Build the legacy state as an owned, read-only view. Phase 18.1 will move
    /// existing sinks to this projection and remove all direct agent write-sites.
    pub fn compatibility_world_state(&self, now: Instant) -> WorldStateProjection {
        let mut people = Vec::new();
        let mut objects = Vec::new();
        let mut animals = Vec::new();

        for entity in &self.entities {
            let Some(bbox) = entity.bbox.value else {
                continue;
            };
            match &entity.kind {
                SceneEntityKind::Person(facts) => {
                    people.push(PersonState {
                        tracker_id: entity.key.track_id,
                        track_id: entity.display_id,
                        bbox,
                        face_bbox: None,
                        identity: facts.gallery_identity.value.as_ref().map(|identity| {
                            (
                                identity.name.clone(),
                                facts.gallery_identity.confidence.unwrap_or(0.0),
                            )
                        }),
                        emotion: fact_label(&facts.emotion),
                        posture: fact_label(&facts.posture),
                        position_3d: entity.position_3d.value,
                        emotion_distribution: None,
                        wrists: Vec::new(),
                        hands: Vec::new(),
                        body_coverage: None,
                        torso_attitude: None,
                        motion_state: None,
                        velocity: entity.kinematics.value.map(|kinematics| {
                            let [x, y] = kinematics.bbox_velocity;
                            x.hypot(y)
                        }),
                        left_gesture: None,
                        right_gesture: None,
                        recon_error: None,
                        plausibility: None,
                        torso_flexion: None,
                        head_yaw: None,
                        head_facing: None,
                        last_seen: now,
                    });
                }
                SceneEntityKind::Object(facts) => {
                    objects.push(ObjectState {
                        id: entity.display_id,
                        class: facts
                            .class_name
                            .value
                            .clone()
                            .unwrap_or_else(|| "unknown".into()),
                        bbox,
                        confidence: facts.class_name.confidence.unwrap_or(0.0),
                        age: 0,
                        position_3d: entity.position_3d.value,
                        mask: None,
                        remembered: facts.remembered,
                        last_seen: now,
                    });
                }
                SceneEntityKind::Animal(_) => animals.push(bbox),
            }
        }

        let activities = self
            .activities
            .iter()
            .filter(|activity| activity.fact.status.is_asserted())
            .map(|activity| Activity {
                label: activity.label.clone(),
                confidence: activity.fact.confidence.unwrap_or(0.0),
                participants: activity
                    .participants
                    .iter()
                    .map(|key| key.track_id)
                    .collect(),
            })
            .collect();

        WorldStateProjection(WorldState {
            frame_id: self.stamp.frame_id,
            timestamp: now,
            people,
            objects,
            animals,
            audio_events: Vec::new(),
            activities,
            events: Vec::new(),
            activity_instances: Vec::new(),
            anomaly_score: self.anomaly.value.unwrap_or(0.0),
            current_session_id: None,
            last_people_seen: None,
        })
    }
}

fn fact_label(fact: &SceneFact<String>) -> Option<(String, f32)> {
    fact.value
        .as_ref()
        .map(|value| (value.clone(), fact.confidence.unwrap_or(0.0)))
}

/// Compatibility wrapper intentionally exposes no mutable dereference.
#[derive(Debug, Clone)]
pub struct WorldStateProjection(WorldState);

impl Deref for WorldStateProjection {
    type Target = WorldState;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalContractError {
    message: String,
}

impl CanonicalContractError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CanonicalContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for CanonicalContractError {}
