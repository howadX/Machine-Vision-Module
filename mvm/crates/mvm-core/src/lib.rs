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

pub use types::{BBox, Detection, PersonTrack, EntityKind, Entity, RelationKind, Relation, Scene};
pub use world::{WorldState, PersonState, ObjectState, Position3D, AudioEvent, Activity};
pub use emotion::{EMOTION_LABELS, emotion_entropy, emotion_label, fuse_emotion_temporal};
pub use tracker::PersonTracker;
pub use module::{Module, ModuleMetadata};
pub use config::Config;
