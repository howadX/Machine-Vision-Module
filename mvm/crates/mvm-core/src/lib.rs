//! mvm-core — фундамент Machine Vision Module.
//!
//! Содержит:
//! - Типы: [`BBox`], [`Detection`], [`PersonTrack`]
//! - Трекер: [`PersonTracker`] — temporal fusion
//! - Контракт: [`Module`] trait — каждый модуль MVM реализует его
//! - Конфигурация: [`Config`] — глобальные настройки

pub mod types;
pub mod tracker;
pub mod module;
pub mod config;
pub mod kalman;
pub mod space;
pub mod emotion;
pub mod world;
pub mod motion;
pub mod assignment;
pub mod posture;
pub mod mask;
pub mod activity;

pub use types::{BBox, Detection, PersonTrack, EntityKind, Entity, RelationKind, Relation, Scene};
pub use motion::{Landmarks33, LandmarkHistory};
pub use world::{WorldState, PersonState, ObjectState, Position3D, AudioEvent, Activity, HandState};
pub use mask::{InstanceMask, MASK_RES};
pub use activity::{ActivityJournal, ActivityInstance, ActivityEvent, ActivityKey, ActivityPhase};
pub use emotion::{EMOTION_LABELS, EMOTION_MAX_ENTROPY, emotion_entropy, emotion_label, emotion_label_gated, fuse_emotion_temporal};
pub use tracker::PersonTracker;
pub use module::{Module, ModuleMetadata};
pub use config::Config;
