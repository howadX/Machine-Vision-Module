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
//! - [`router`] — Semantic Group Router: чистое детерминированное отображение
//!   снимка [`mvm_core::WorldState`] в наблюдение по ветвям G₁–G₅ (§16.0.3).
//! - [`config`] — [`DatasetConfig`]: секция `[dataset]` конфига (§16.1.1).
//! - [`writer`] — [`DatasetWriter`]: оркестрация сессий, JSONL labels, session.toml (§16.1.2).
//! - [`privacy`] — вычисление регионов размытия лиц по режиму приватности (§16.6).
//! - [`session`] — модель `session.toml` + чтение сессий с диска для `mvm dataset` (§16.3).
//!
//! ML-фреймворки (PyTorch/TF) сюда не протекают — экспорт в ML-форматы живёт
//! в Python-утилитах (`prototype/pipeline/dataset/`).

pub mod config;
pub mod observation;
pub mod privacy;
pub mod router;
pub mod session;
pub mod validate;
pub mod writer;

pub use config::{DatasetConfig, ExportConfig, ExportFormat, FrameFormat, Privacy};
pub use observation::{
    CanonicalObservation, EntityRecord, FrameMeta, Label, LabelStatus, Provenance, QualityBlock,
    SCHEMA_VERSION, SemanticGroup, SessionRef,
};
pub use privacy::{PrivacyPerson, blur_regions};
pub use router::{EdgeRel, RouteContext, SceneRelations, ZoneRel, route, route_with_relations};
pub use session::{
    DatasetStats, PrunePolicy, SessionInfo, SessionMeta, aggregate, list_sessions, partition_prune,
};
pub use validate::{
    CheckStats, DatasetReport, Finding, SessionReport, Severity, check_dataset, check_session,
};
pub use writer::{DatasetWriter, FrameRecord, RecordResult};
