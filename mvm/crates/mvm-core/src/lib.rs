//! mvm-core — фундамент Machine Vision Module.
//!
//! Содержит:
//! - Типы: [`BBox`], [`Detection`], [`PersonTrack`]
//! - Трекер: [`PersonTracker`] — temporal fusion
//! - Контракт: [`Module`] trait — каждый модуль MVM реализует его
//! - Конфигурация: [`Config`] — глобальные настройки

pub mod activity;
pub mod anomaly;
pub mod assignment;
pub mod canonical;
pub mod config;
pub mod emotion;
pub mod kalman;
pub mod mask;
pub mod module;
pub mod motion;
pub mod posture;
pub mod recap;
pub mod sensor;
pub mod space;
pub mod tracker;
pub mod tracking;
pub mod types;
pub mod world;

pub use activity::{ActivityEvent, ActivityInstance, ActivityJournal, ActivityKey, ActivityPhase};
pub use anomaly::AnomalyScorer;
pub use config::Config;
pub use emotion::{
    EMOTION_LABELS, EMOTION_MAX_ENTROPY, emotion_entropy, emotion_label, emotion_label_gated,
    fuse_emotion_temporal,
};
pub use mask::{InstanceMask, MASK_RES};
pub use module::{Module, ModuleMetadata};
pub use motion::{LandmarkHistory, Landmarks33};
pub use sensor::{
    CalibrationRef, DepthGeometry, DepthStatus, MetricDepthFrame, RegisteredDepthOwners,
    SensorClockDomain, SensorHealth, SensorReplayEnvelope, SensorSourceId, SensorStamp,
};
pub use space::ScaleCalibrator;
pub use tracker::{EnrichmentApply, PersonEnrichment, PersonTrackHandle, PersonTracker};
pub use tracking::{
    AssociationGeometry, AssociationGeometryStatus, AssociationObservation, AxisAlignedExtent,
    GeometryReason, GeometrySampling, MetricCandidateCue, MetricEvidenceQuality,
    PersonBirthConflict, SceneObjectEvidence, StaticDisplayClass, TrackedAssociationGeometry,
    TrackerDecision, TrackerDecisionKind, TrackerDiagnostics, TrackerDiagnosticsSnapshot,
    TrackerDomain, TrackerFrameBinding, metric_candidate_cue, metric_evidence_quality,
};
pub use types::{
    BBox, BodyObservation, Detection, Entity, EntityKind, PersonTrack, Relation, RelationKind,
    Scene,
};
pub use world::{
    Activity, AudioEvent, HandState, ObjectState, PersonState, Position3D, WorldState,
};
