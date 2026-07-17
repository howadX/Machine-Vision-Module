//! mvm-dataset — Dataset Factory: семантический экспортный слой MVM (фаза 16).
//!
//! Крейт — чистый потребитель [`mvm_core::WorldState`], как и StateFileWriter.
//! Он **не** зависит от детекторов/fusion и **не** добавляет вычислений в hot-path.
//! Задача слоя — компилировать наблюдения в обучающие представления:
//! `register → route → encode → validate → bind → index → export`.
//!
//! Модули:
//! - [`observation`] — Canonical Observation: стабильная схема наблюдения,
//!   статусы разметки, provenance, ключи связывания (§16.0.1–16.0.2).
//!
//! ML-фреймворки (PyTorch/TF) сюда не протекают — экспорт в ML-форматы живёт
//! в Python-утилитах (`prototype/pipeline/dataset/`).

pub mod observation;

pub use observation::{
    CanonicalObservation, EntityRecord, FrameMeta, Label, LabelStatus, Provenance, QualityBlock,
    SemanticGroup, SessionRef, SCHEMA_VERSION,
};
