//! Canonical Observation — стабильная внутренняя схема наблюдения (§16.0.1–16.0.2).
//!
//! Схема не зависит от PyTorch/TF. Она сохраняет полное наблюдение (сущности,
//! отношения, события, семантические группы, целевые метки, quality, provenance),
//! а Dataset Views строятся поверх неё по группам и задачам (§16.4).
//!
//! Два инвариантных правила:
//! 1. **No premature collapse.** Наблюдение не схлопывается в единый embedding —
//!    сохраняются сущности, их атрибуты, связи и provenance.
//! 2. **`withheld`/`unknown` ≠ отрицательный класс.** Отсутствие утверждения
//!    нельзя автоматически превращать в negative label — см. [`LabelStatus::can_be_negative_class`].

use serde::{Deserialize, Serialize};

/// Версия схемы Canonical Observation. Пишется в каждую запись (ключ связывания).
///
/// `16.7.0` (Semantic Session Log) — аддитивная достройка отложенных §16.3:
/// роутер заполняет `relations` (зоны+рёбра) и `events` (эпизоды) и добавлено
/// новое поле `activities` (per-frame активности). Все добавления читаются старыми
/// парсерами (`#[serde(default)]`), так что 16.0.1-строки остаются валидными.
pub const SCHEMA_VERSION: &str = "16.8.0";
pub const LEGACY_SCHEMA_VERSION: &str = "16.7.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthEncoding {
    RelativeAffineU16,
    MetricMmU16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthUnits {
    AffineRelative,
    Millimetres,
}

/// Self-describing dense-depth artifact metadata. The PNG payload remains at
/// `depth_path`; this block prevents a relative visualization from ever being
/// interpreted as millimetres and preserves metric sensor provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthFrameMetadata {
    pub encoding: DepthEncoding,
    pub units: DepthUnits,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invalid_value: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry: Option<mvm_core::DepthGeometry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<mvm_core::SensorSourceId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rgb_stamp: Option<mvm_core::SensorStamp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_stamp: Option<mvm_core::SensorStamp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rgb_depth_skew_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<mvm_core::CalibrationRef>,
}

/// Семантическая группа-ветвь роутера (§16.0.3). Кадр маршрутизируется только
/// в релевантные ветви (selective routing), сущность несёт свою первичную группу.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticGroup {
    /// G₁ — поза / identity / emotion / landmarks.
    Human,
    /// G₂ — объекты / глубина / зоны / permanence.
    Scene,
    /// G₃ — руки / жесты / контакты / активности.
    Interaction,
    /// G₄ — переходы / эпизоды / prediction targets.
    Temporal,
    /// G₅ — confidence / entropy / OOD / provenance.
    Quality,
}

impl SemanticGroup {
    /// Стабильный короткий код ветви (`G1`..`G5`) для индексов и путей.
    pub fn code(self) -> &'static str {
        match self {
            SemanticGroup::Human => "G1",
            SemanticGroup::Scene => "G2",
            SemanticGroup::Interaction => "G3",
            SemanticGroup::Temporal => "G4",
            SemanticGroup::Quality => "G5",
        }
    }

    /// Все ветви в каноническом порядке.
    pub const ALL: [SemanticGroup; 5] = [
        SemanticGroup::Human,
        SemanticGroup::Scene,
        SemanticGroup::Interaction,
        SemanticGroup::Temporal,
        SemanticGroup::Quality,
    ];
}

/// Явный статус каждого значения разметки (§16.0.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelStatus {
    /// Непосредственно получено детектором/экспертом.
    Observed,
    /// Вычислено Fusion или правилом над наблюдениями.
    Derived,
    /// Выведено из последовательности кадров.
    Temporal,
    /// Pseudo-label / эвристика, требующая аудита.
    Weak,
    /// Данных недостаточно — утверждение намеренно не делается.
    Withheld,
    /// Свойство не наблюдалось или не включено.
    Unknown,
}

impl LabelStatus {
    /// Является ли значение утверждением (несёт положительный сигнал), а не пропуском.
    /// `withheld`/`unknown` — пропуски, не утверждения.
    pub fn is_asserted(self) -> bool {
        matches!(
            self,
            LabelStatus::Observed
                | LabelStatus::Derived
                | LabelStatus::Temporal
                | LabelStatus::Weak
        )
    }

    /// Критическое правило §16.0.2: `withheld` и `unknown` **нельзя** автоматически
    /// превращать в отрицательный класс. Экспортёр обязан спрашивать это перед тем,
    /// как трактовать отсутствие метки как negative label.
    pub fn can_be_negative_class(self) -> bool {
        !matches!(self, LabelStatus::Withheld | LabelStatus::Unknown)
    }
}

/// Provenance derived/weak-метки (§16.0.2, принцип «Provenance is mandatory»):
/// известны модель, версия, источник и правило вывода.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Модель или правило-источник (напр. `"EmotiEffLib-B0"`, `"fusion:zone"`).
    pub model: String,
    /// Версия модели/правила/схемы.
    pub version: String,
    /// Класс источника: `"detector" | "fusion" | "temporal" | "heuristic"`.
    pub source: String,
    /// Правило вывода для derived/weak (напр. `"pelvis_speed>1.1 && on_floor"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule: Option<String>,
}

/// Интерпретируемое кодирование одной метки (принцип «Interpretable encoding»):
/// значение, статус, confidence, источник и ссылка на исходный сигнал.
///
/// Обобщён по типу значения — конкретные типы (поза, эмоция, …) навешиваются
/// в group-writer'ах (§16.3). `value == None` при `withheld`/`unknown`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Label<T> {
    /// Значение метки. `None` для `withheld`/`unknown`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<T>,
    /// Статус — как получено значение.
    pub status: LabelStatus,
    /// Уверенность источника, если применима.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Provenance — обязателен для `derived`/`weak`, опционален иначе.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    /// Ссылка на исходный сигнал (напр. `"person.track_id=1.posture"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_ref: Option<String>,
}

impl<T> Label<T> {
    /// Наблюдённая метка от детектора/эксперта.
    pub fn observed(value: T, confidence: f32) -> Self {
        Label {
            value: Some(value),
            status: LabelStatus::Observed,
            confidence: Some(confidence),
            provenance: None,
            signal_ref: None,
        }
    }

    /// Derived-метка (Fusion/правило) — provenance обязателен.
    pub fn derived(value: T, confidence: f32, provenance: Provenance) -> Self {
        Label {
            value: Some(value),
            status: LabelStatus::Derived,
            confidence: Some(confidence),
            provenance: Some(provenance),
            signal_ref: None,
        }
    }

    /// Намеренный пропуск: данных недостаточно, утверждение не делается.
    pub fn withheld() -> Self {
        Label {
            value: None,
            status: LabelStatus::Withheld,
            confidence: None,
            provenance: None,
            signal_ref: None,
        }
    }

    /// Свойство не наблюдалось / не включено.
    pub fn unknown() -> Self {
        Label {
            value: None,
            status: LabelStatus::Unknown,
            confidence: None,
            provenance: None,
            signal_ref: None,
        }
    }

    /// Прикрепить ссылку на исходный сигнал.
    pub fn with_signal_ref(mut self, r: impl Into<String>) -> Self {
        self.signal_ref = Some(r.into());
        self
    }
}

/// Метаданные кадра.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameMeta {
    /// ISO-8601 timestamp кадра.
    pub timestamp: String,
    /// Относительный путь к сохранённому RGB-кадру (`frames/XXXXXX.jpg`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_path: Option<String>,
    /// Относительный путь к карте глубины (`depth/XXXXXX.png`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_path: Option<String>,
    /// `[min, max]` сырой относительной глубины, отображённой в 16-битный PNG
    /// (§16.2.2). Делает нормализацию обратимой: `raw = min + (u16/65535)·(max−min)`.
    /// Глубина affine-relative (не метры), поэтому диапазон — на кадр.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_range: Option<[f32; 2]>,
    /// Typed encoding/units/provenance for newly recorded depth artifacts.
    /// Absent on pre-16.8 legacy relative-depth sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<DepthFrameMetadata>,
}

/// Привязка кадра к сессии.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRef {
    /// Порядковый индекс сохранённого кадра внутри сессии.
    pub seq_in_session: u64,
}

/// Одна сущность наблюдения с ключами связывания `entity_id`/`track_id`/`group`.
/// Типизированные атрибуты (поза/эмоция/объект/…) навешиваются в §16.3;
/// здесь payload остаётся структурно открытым, чтобы не схлопывать схему преждевременно.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecord {
    /// Стабильный id сущности в пределах наблюдения.
    pub entity_id: String,
    /// Id трека (тело человека / объект), если сущность трекается.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_id: Option<u64>,
    /// Первичная семантическая группа сущности.
    pub group: SemanticGroup,
    /// Вид сущности: `"person" | "object"`.
    pub kind: String,
    /// Типизированные метки-атрибуты (уточняются в §16.3).
    pub attributes: serde_json::Value,
}

/// Quality-блок наблюдения (§16.0.1, группа G₅) — «quality is data». Заполняется
/// роутером из уже наблюдённых сигналов.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityBlock {
    /// Кадр пригоден (не пустой/не битый).
    pub frame_ok: bool,
    /// Сколько людей в кадре (после трекера).
    pub people_count: u32,
    /// Сколько объектов-узлов в кадре.
    pub object_count: u32,
    /// Anomaly-score кадра (decaying peak, §11.B2).
    pub anomaly_score: f32,
    /// Свободная заметка о качестве кадра.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// Canonical Observation — одна строка `labels.jsonl` (§16.0.1).
///
/// Ключи связывания уровня наблюдения: `dataset_id`, `schema_version`,
/// `session_id`, `frame_id`. Ключи уровня сущности (`entity_id`, `track_id`,
/// `group`) — в [`EntityRecord`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalObservation {
    // --- ключи связывания (уровень наблюдения) ---
    /// Стабильный id датасета (корень хранилища).
    pub dataset_id: String,
    /// Версия схемы — [`SCHEMA_VERSION`] на момент записи.
    pub schema_version: String,
    /// Id сессии (`YYYY-MM-DD_HHMMSS`).
    pub session_id: String,
    /// Монотонный id кадра из пайплайна.
    pub frame_id: u64,

    // --- секции наблюдения ---
    /// Метаданные кадра и пути к артефактам.
    pub frame: FrameMeta,
    /// Привязка к сессии.
    pub session: SessionRef,
    /// Сущности (люди/объекты) с ключами связывания.
    pub entities: Vec<EntityRecord>,
    /// Реляционный слой (§16.7): гетерогенные записи `{kind:"zone",…}` (зоны с
    /// `label`/`members`/`persistent_id`) и `{kind:"edge",a,b,rel}` (рёбра близости/
    /// пересечения). Заполняется роутером из посчитанных вызывающим зон/рёбер.
    pub relations: Vec<serde_json::Value>,
    /// Завершённые эпизоды кадра (§16.7): `{kind:"episode",label,started_at,ended_at,
    /// duration_s,participants}` — инлайн на кадре, где эпизод закрылся.
    pub events: Vec<serde_json::Value>,
    /// Активные активности кадра (§16.7, G₃/G₄): `{label,participants,started_at,
    /// duration_s,stable}`. Новое поле с 16.7.0 — `#[serde(default)]` держит
    /// обратную совместимость с 16.0.1-строками.
    #[serde(default)]
    pub activities: Vec<serde_json::Value>,
    /// Семантические группы, в которые маршрутизирован кадр (selective routing).
    pub semantic_groups: Vec<SemanticGroup>,
    /// Целевые метки для обучения (prediction targets, уточняются в §16.4).
    pub derived_targets: serde_json::Value,
    /// Quality-блок (группа G₅).
    pub quality: QualityBlock,
    /// Provenance-цепочка уровня наблюдения.
    pub provenance: Vec<Provenance>,
}

impl CanonicalObservation {
    /// Пустой скелет наблюдения с обязательными ключами связывания.
    /// Секции заполняются group-writer'ами роутера.
    pub fn new(
        dataset_id: impl Into<String>,
        session_id: impl Into<String>,
        frame_id: u64,
        timestamp: impl Into<String>,
        seq_in_session: u64,
    ) -> Self {
        CanonicalObservation {
            dataset_id: dataset_id.into(),
            schema_version: SCHEMA_VERSION.to_string(),
            session_id: session_id.into(),
            frame_id,
            frame: FrameMeta {
                timestamp: timestamp.into(),
                frame_path: None,
                depth_path: None,
                depth_range: None,
                depth: None,
            },
            session: SessionRef { seq_in_session },
            entities: Vec::new(),
            relations: Vec::new(),
            events: Vec::new(),
            activities: Vec::new(),
            semantic_groups: Vec::new(),
            derived_targets: serde_json::Value::Null,
            quality: QualityBlock {
                frame_ok: true,
                people_count: 0,
                object_count: 0,
                anomaly_score: 0.0,
                notes: None,
            },
            provenance: Vec::new(),
        }
    }

    /// Сериализовать в одну JSONL-строку (без переводов строк внутри).
    pub fn to_jsonl_line(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Разобрать одну JSONL-строку в наблюдение.
    pub fn from_jsonl_line(line: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(line)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CanonicalObservation {
        let mut obs = CanonicalObservation::new(
            "mvm_sessions",
            "2026-07-17_153000",
            12_345,
            "2026-07-17T15:30:00.123",
            7,
        );
        obs.entities.push(EntityRecord {
            entity_id: "p1".to_string(),
            track_id: Some(1),
            group: SemanticGroup::Human,
            kind: "person".to_string(),
            attributes: serde_json::json!({ "posture": "sitting" }),
        });
        obs.semantic_groups = vec![SemanticGroup::Human, SemanticGroup::Quality];
        obs
    }

    #[test]
    fn round_trip_jsonl_is_single_line_and_stable() {
        let obs = sample();
        let line = obs.to_jsonl_line().expect("serialize");
        // JSONL-инвариант: одна запись = одна строка, без внутренних переводов строк.
        assert!(!line.contains('\n'), "jsonl line must not contain newlines");
        let back = CanonicalObservation::from_jsonl_line(&line).expect("parse");
        assert_eq!(obs, back, "round-trip must preserve observation exactly");
    }

    #[test]
    fn mandatory_binding_keys_present() {
        let obs = sample();
        let v: serde_json::Value = serde_json::from_str(&obs.to_jsonl_line().unwrap()).unwrap();
        // Уровень наблюдения.
        for key in ["dataset_id", "schema_version", "session_id", "frame_id"] {
            assert!(
                v.get(key).is_some(),
                "missing observation binding key: {key}"
            );
        }
        // schema_version фиксируется автоматически.
        assert_eq!(v["schema_version"], serde_json::json!(SCHEMA_VERSION));
        // Уровень сущности.
        let ent = &v["entities"][0];
        for key in ["entity_id", "track_id", "group"] {
            assert!(ent.get(key).is_some(), "missing entity binding key: {key}");
        }
    }

    #[test]
    fn withheld_and_unknown_are_not_negative_class() {
        // Критическое правило §16.0.2.
        assert!(!LabelStatus::Withheld.can_be_negative_class());
        assert!(!LabelStatus::Unknown.can_be_negative_class());
        // Остальные статусы допускают трактовку как класс.
        for s in [
            LabelStatus::Observed,
            LabelStatus::Derived,
            LabelStatus::Temporal,
            LabelStatus::Weak,
        ] {
            assert!(s.can_be_negative_class(), "{s:?} should be a usable class");
            assert!(s.is_asserted(), "{s:?} must count as an assertion");
        }
        assert!(!LabelStatus::Withheld.is_asserted());
        assert!(!LabelStatus::Unknown.is_asserted());
    }

    #[test]
    fn provenance_is_preserved_for_derived_labels() {
        let prov = Provenance {
            model: "fusion:fall".to_string(),
            version: "e2-scene".to_string(),
            source: "fusion".to_string(),
            rule: Some("pelvis_speed>1.1 && on_floor".to_string()),
        };
        let label = Label::derived("fall".to_string(), 0.91, prov.clone());
        let line = serde_json::to_string(&label).unwrap();
        let back: Label<String> = serde_json::from_str(&line).unwrap();
        assert_eq!(back, label);
        assert_eq!(back.provenance.as_ref().unwrap(), &prov);
        assert_eq!(back.status, LabelStatus::Derived);
    }

    #[test]
    fn withheld_label_carries_no_value() {
        let label: Label<String> = Label::withheld();
        assert!(label.value.is_none());
        assert!(!label.status.can_be_negative_class());
        // При сериализации отсутствующее значение не попадает в JSON.
        let v: serde_json::Value = serde_json::to_value(&label).unwrap();
        assert!(
            v.get("value").is_none(),
            "withheld must not emit a value key"
        );
        assert_eq!(v["status"], serde_json::json!("withheld"));
    }

    #[test]
    fn activities_field_defaults_when_absent_in_old_lines() {
        // §16.7 additive bump: a 16.0.1 line (no `activities` key) must still parse,
        // with `activities` defaulting to empty — old JSONL stays readable.
        let mut obs = sample();
        obs.activities
            .push(serde_json::json!({ "label": "typing", "participants": ["p1"] }));
        let line = obs.to_jsonl_line().unwrap();
        assert!(
            line.contains("\"activities\""),
            "new schema emits the activities key"
        );

        // Simulate an old line by stripping the activities key entirely.
        let mut v: serde_json::Value = serde_json::from_str(&line).unwrap();
        v.as_object_mut().unwrap().remove("activities");
        v["schema_version"] = serde_json::json!(LEGACY_SCHEMA_VERSION);
        let old_line = serde_json::to_string(&v).unwrap();
        assert!(!old_line.contains("\"activities\""));
        let back = CanonicalObservation::from_jsonl_line(&old_line).expect("old line parses");
        assert!(
            back.activities.is_empty(),
            "missing activities defaults to empty"
        );
        assert_eq!(back.schema_version, LEGACY_SCHEMA_VERSION);
    }

    #[test]
    fn semantic_group_codes_are_stable() {
        assert_eq!(SemanticGroup::Human.code(), "G1");
        assert_eq!(SemanticGroup::Scene.code(), "G2");
        assert_eq!(SemanticGroup::Interaction.code(), "G3");
        assert_eq!(SemanticGroup::Temporal.code(), "G4");
        assert_eq!(SemanticGroup::Quality.code(), "G5");
        assert_eq!(SemanticGroup::ALL.len(), 5);
        // snake_case-сериализация для стабильности JSONL.
        assert_eq!(
            serde_json::to_string(&SemanticGroup::Interaction).unwrap(),
            "\"interaction\""
        );
    }
}
