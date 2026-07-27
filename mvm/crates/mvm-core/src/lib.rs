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
pub mod space;
pub mod tracker;
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
pub use space::ScaleCalibrator;
pub use tracker::PersonTracker;
pub use types::{BBox, Detection, Entity, EntityKind, PersonTrack, Relation, RelationKind, Scene};
pub use world::{
    Activity, AudioEvent, HandState, ObjectState, PersonState, Position3D, WorldState,
};
