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

pub use types::{BBox, Detection, PersonTrack};
pub use tracker::PersonTracker;
pub use module::{Module, ModuleMetadata};
pub use config::Config;
