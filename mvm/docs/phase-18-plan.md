# Фаза 18 — Visual Cortex Layer: явный когнитивный слой MVM

> **Версия:** 0.4 (27 июля 2026)
> **Статус:** 🚧 реализация начата; 18.0 State Contract + Benchmark Baseline ✅;
> 18.1 Typed Observation Bus + Canonical Fusion — в работе
> **Ветка:** `kaggle`
> **Цель фазы:** сначала завершить канонический facts-слой `FusedSceneFrame`, затем добавить над ним явный, читаемый и проверяемый когнитивный слой `mvm-cortex`, превращающий perception pipeline в visual substrate for agency.
> **Манифест:** `~/wiki/mvm/north-star-visual-cortex-agent.md`
> **Родительский манифест:** `~/wiki/mvm/north-star-conscious-vision.md`
> **Академическое основание:** `~/wiki/machine-vision/deconstructing-consciousness-as-engineering-method.md`

---

## 0. Короткая формулировка

Фаза 18 не добавляет новые детекторы, ONNX-модели, GPU-зависимости или end-to-end learning.

Фаза 18 завершает два явно разделённых слоя:

```text
FusedSceneFrame = что сейчас наблюдается и арифметически выведено в сцене
CortexFrame     = что эти факты значат для агента
```

Новый центральный артефакт:

```text
crates/mvm-cortex/
```

Перед его включением `WorldState`-blackboard заменяется typed observation bus и единственным
immutable `FusedSceneFrame`. Старый `WorldState` остаётся только read-only compatibility projection
до конца v0.4.0.

`mvm-cortex` собирает в одном месте:

- prediction / forecast;
- prediction error;
- интерпретацию canonical scene events;
- runtime episodic memory;
- affordance;
- attention;
- visual self-model;
- uncertainty report;
- cognitive report для CLI/MCP/LLM.

Главная задача фазы — **не распылить когницию по ядру, CLI, dataset и отдельным крейтам**, а сделать её видимой как один красивый слой.

---

## 1. Зачем

Фазы 5–14 построили продуктовый MVM: детекторы, body-primary tracker, fusion, scene-graph, LLM-render, MCP, `setup`, XDG и релизы.

Фазы 15–16 усилили ML-основание и Dataset Factory: MVM стал не только источником live-сцены для LLM, но и производителем семантических/пиксельных сессий для будущего обучения.

Фаза 16.7 свела vision history в единый **Semantic Session Log**: `mvm run` по умолчанию пишет text-only semantic session, а `--record` доращивает её до полного датасета.

Фаза 17 выносит agent know-how в canonical skills.

Но MVM всё ещё в основном отвечает на вопрос:

```text
Что видно сейчас?
```

Полярная звезда требует другого вопроса:

```text
Что происходит в мире, где агент присутствует как наблюдатель?
```

Для этого нужен сначала непротиворечивый canonical facts substrate, а затем когнитивный слой,
который делает поверх `FusedSceneFrame` следующие операции:

- ожидает следующее состояние;
- сравнивает ожидание с наблюдением;
- выделяет события;
- помнит недавние эпизоды;
- понимает возможные действия;
- выбирает, что важно;
- знает свои ограничения как наблюдателя;
- честно сообщает uncertainty.

---

## 2. Управляющие принципы

1. **Один видимый когнитивный слой.** Когнитивная логика живёт в `mvm-cortex`, а не расползается по `mvm-core`, `mvm-cli`, `mvm-dataset` и множеству равноправных новых крейтов.
2. **Один canonical facts authority.** Fusion один раз публикует immutable `FusedSceneFrame`; renderer, Dataset Factory, MCP и Cortex не пересобирают scene graph самостоятельно.
3. **`WorldState` — не authority.** В фазе 18 он становится deprecated read-only projection для совместимости v0.4.0 и удаляется в следующем cleanup, а не живёт параллельной моделью навсегда.
4. **`CortexFrame` — интерпретация.** Forecast, prediction error, memory, attention, affordance и self-model не являются фактами сцены; это cognitive view поверх `FusedSceneFrame`.
5. **Observed и cognitive events разделены.** Fusion владеет доказуемыми temporal transitions; Cortex добавляет unexpectedness, salience, prediction/memory/affordance interpretation.
6. **Ни одной новой модели в фазе 18.** Если требуется обучение, новый ONNX, GPU или VLM — это отдельная будущая фаза, не 18.
7. **Чистая арифметика / FSM / эвристики.** Все слои фазы 18 используют существующие сигналы MVM: bbox, 3D, confidence, velocity, lifecycle, gestures, relations, activities, zones, semantic session.
8. **Benchmark-first.** До feature-слоёв фиксируются replay fixtures, current baseline и метрики; claims без измеримого gate не принимаются.
9. **Honesty-first.** Каждый uncertain scene signal использует `SceneFact<T>`; `withheld`, `unknown`, `weak`, stale и low-confidence claims не скрываются и не превращаются в отрицательные классы.
10. **Один durable store.** `labels.jsonl` остаётся единственной долговременной семантической историей. Latest-state файлы и Cortex memory не становятся вторым архивом.
11. **Обратная совместимость.** `snapshot`, `recap`, `inspect`, `serve`, Dataset Factory и schema 16.7.0 сохраняются; расширения IDs/events только аддитивны.
12. **Camera-driven immutable ticks.** Canonical revision задаёт RGB capture frame. Поздний expert result не переписывает прошлое, а входит в следующий tick со своим source stamp и честным age.
13. **Cortex — consumer/peer, не replacement.** Fusion остаётся арифметическим интерфейсом. Cortex читает результат fusion, но не заменяет его.
14. **Схематическая читаемость важнее микрокрейтной чистоты.** На старте лучше один `mvm-cortex` с модулями, чем пять новых островов.

---

## 3. Архитектурные этажи MVM после фазы 18

```text
┌────────────────────────────────────────────┐
│ L4. Agent Interface                        │
│ CLI / MCP / StateFile / snapshot / recap   │
│ inspect / cortex(view=...)                 │
└────────────────────────────────────────────┘
                    ▲
┌────────────────────────────────────────────┐
│ L3. Visual Cortex / Cognitive Layer        │
│ prediction / events / memory / attention   │
│ affordance / self-model / uncertainty      │
└────────────────────────────────────────────┘
                    ▲
┌────────────────────────────────────────────┐
│ L2. World Model Substrate                  │
│ FusedSceneFrame / SceneFact / EntityKey    │
│ relations / persistent zones / activities  │
└────────────────────────────────────────────┘
                    ▲
┌────────────────────────────────────────────┐
│ L1. Perception Experts                     │
│ body / face / id / emotion / posture       │
│ hands / objects / depth / masks            │
└────────────────────────────────────────────┘
                    ▲
┌────────────────────────────────────────────┐
│ L0. Sensor Stream                          │
│ camera frames / timestamps / config        │
└────────────────────────────────────────────┘
```

Это фиксирует главный разрез:

```text
Experts produce signals.
Fusion produces scene facts.
Cortex produces cognitive interpretation.
Agent Interface exposes it to LLM/humans.
```

---

## 4. Целевая структура репозитория

```text
mvm/
├── crates/
│   ├── mvm-core/          # canonical scene types + deprecated WorldState projection
│   ├── mvm-detect/        # face detector
│   ├── mvm-identity/      # identity expert
│   ├── mvm-emotion/       # emotion expert
│   ├── mvm-posture/       # body/posture expert
│   ├── mvm-hands/         # hands expert
│   ├── mvm-scene/         # objects/masks
│   ├── mvm-depth/         # depth expert
│   ├── mvm-fusion/        # canonical scene assembler + arithmetic fusion + scene events
│   ├── mvm-mesh/          # agent runtime + generic LatestValue<T>
│   ├── mvm-dataset/       # semantic/pixel sessions, dataset CLI backend
│   ├── mvm-cortex/        # NEW: visual cognitive layer
│   │   ├── src/lib.rs
│   │   ├── src/cortex.rs
│   │   ├── src/types.rs
│   │   ├── src/predict.rs
│   │   ├── src/events.rs
│   │   ├── src/memory.rs
│   │   ├── src/affordance.rs
│   │   ├── src/attention.rs
│   │   ├── src/self_model.rs
│   │   ├── src/uncertainty.rs
│   │   ├── src/report.rs
│   │   └── src/prelude.rs
│   └── mvm-cli/           # orchestration + MCP + CLI rendering
└── benchmark/             # Visual QA Benchmark
```

Не создавать на старте отдельные крейты:

```text
mvm-predict
mvm-memory
mvm-attention
mvm-affordance
mvm-self
```

Они возможны позже, если слой станет достаточно большим. В фазе 18 главный приоритет — единая читаемая когнитивная архитектура.

---

## 5. Границы ответственности

| Слой | Ответственность | Не делает |
|---|---|---|
| `mvm-core` | `SceneFact<T>`, `EntityKey`, unified `SceneEntity`, `FusedSceneFrame`, базовые tracker/activity types, compatibility projection | Не владеет orchestration или cognitive state |
| `mvm-fusion` | Единожды собирает entities/relations/zones/activities/scene events, ведёт kinematics и event sequence | Не решает memory/attention/self-model |
| `mvm-mesh` | `LatestValue<T>`, lifecycle threads и typed channels | Не знает detector/fusion/cortex semantics |
| `mvm-dataset` | Read-only проецирует `FusedSceneFrame` в semantic/pixel sessions, validates/exports/prunes data; сам владеет session lifecycle | Не мутирует scene facts и не является runtime cognition |
| `mvm-cortex` | Строит `CortexFrame` из `FusedSceneFrame`, canonical scene events и собственного прошлого | Не вызывает детекторы, не пересобирает scene graph, не заменяет fusion |
| `mvm-cli` | Запускает agents, CLI/MCP UX, render | Не содержит когнитивную бизнес-логику |

Ключевой контракт:

```text
FusedSceneFrame = canonical scene facts
CortexFrame     = cognitive interpretation
WorldState      = deprecated compatibility projection only
```

---

## 5.1. Аудит текущего state-контура

Слово “statefile” сейчас скрывает несколько разных сущностей.

| Категория | Сколько | Что входит |
|---|---:|---|
| Literal `StateFile` | 1 | `current_state.md` |
| Штатные persisted views | 4 | `current_state.md`, `labels.jsonl`, `session.toml`, `journal.md` |
| С legacy views | 6 | + `scene-*.jsonl`, `events-*.jsonl` |
| Межагентные runtime channels | 3 | latest RGB frame, shared `WorldState`, depth/person-mask slots |
| Private temporal state | несколько | person/object trackers, activity journal, anomaly scorer, два независимых zone trackers |

Файлы не публикуют данные друг другу. Текущий обмен идёт через shared memory, после чего разные sinks
материализуют собственные views:

```text
CaptureAgent ───────────────→ LatestFrame<RGB>

SceneAgent ─────────────────→ WorldState.objects/animals
          └─────────────────→ PersonMaskSlot → HumanAgent

HumanAgent ─────────────────→ WorldState.people
          └─────────────────→ WorldState.frame_id/timestamp

DepthAgent ─────────────────→ DepthSlot

FusionAgent ────────────────→ mutates WorldState:
                               object 3D, activities, events, anomaly

WorldState ─┬───────────────→ StateFileWriter
            │                  current_state.md / journal.md /
            │                  legacy scene/events JSONL
            └───────────────→ DatasetWriter
                               labels.jsonl / session.toml
```

Проблемы, которые блокируют честный Cortex:

1. `WorldState` не содержит готовые relations и persistent zones.
2. `build_scene()` независимо вызывают snapshot renderer, StateFile JSON и DatasetWriter.
3. StateFileWriter и DatasetWriter имеют разные `ZoneTracker`; zone id/age зависят от cadence sink.
4. StateFileWriter destructive-очищает `world.events`, конкурируя с DatasetWriter.
5. DatasetWriter меняет `current_session_id`/`last_people_seen` внутри общего blackboard.
6. `DepthSlot` и `PersonMaskSlot` не несут source frame/freshness.
7. HumanAgent единолично двигает глобальный `frame_id`, хотя experts асинхронны.
8. `current_state.md` пишется прямым overwrite, без atomic temp+rename.
9. `--native-camera`/`--native` имеют отдельный sync state pipeline и обходят mesh facts contract.

Вывод: Cortex нельзя безопасно строить непосредственно поверх текущего `WorldState`. Сначала требуется
state refactor 18.0–18.1.

---

## 5.2. Canonical scene facts

### Frame clock и source freshness

Один canonical tick задаёт camera capture frame, а не HumanAgent или самый медленный expert.

```rust
pub struct FrameStamp {
    pub run_id: String,
    pub frame_id: u64,
    pub captured_at: Instant,
    pub published_at: Instant,
}

pub struct SourceStamp {
    pub source: FactSource,
    pub source_frame_id: u64,
    pub produced_at: Instant,
    pub age_frames: u64,
    pub freshness: SourceFreshness,
}

pub enum SourceFreshness {
    Fresh,
    Coasted,
    Stale,
    Unavailable,
}
```

Published `FusedSceneFrame` immutable. Expert result, завершившийся позже соответствующего camera tick,
попадает только в следующий tick, сохраняя исходный `source_frame_id`. Переиздания старого frame и
late mutation запрещены.

Latest-wins runtime не обязан публиковать каждый захваченный RGB frame: при перегрузке `frame_id` может
иметь gaps, но каждый опубликованный fused tick привязан ровно к одному реальному capture frame.

Initial freshness budgets привязаны к уже существующим cadence и затем калибруются benchmark:

```text
Human: age <= 2 camera frames
Scene/masks: age <= 2 * object_cadence
Depth: age <= 2 * depth_cadence
```

Stale source может поддержать явно помеченный `remembered`/`temporal` факт, но не создаёт новую
`observed` relation или свежую metric-3D claim.

### `SceneFact<T>`: единая honesty-семантика

```rust
pub struct SceneFact<T> {
    pub status: FactStatus,
    pub value: Option<T>,
    pub confidence: Option<f32>,
    pub evidence: Vec<FactEvidenceRef>,
    pub reason: Option<FactReason>,
}

pub enum FactStatus {
    Observed,
    Derived,
    Temporal,
    Remembered,
    Weak,
    Withheld,
    Unknown,
}
```

Правила:

- `Observed` — прямой результат expert observation.
- `Derived` — чистая арифметика над доступными facts.
- `Temporal` — FSM/diff/continuity по нескольким ticks.
- `Remembered` — сохранённый tracker/permanence state без свежего наблюдения.
- `Weak` — полезное, но недостаточное supervision.
- `Withheld` — сигнал намеренно не заявлен из-за качества/coverage/OOD.
- `Unknown` — данных нет.
- `FactEvidenceRef` связывает claim с одним или несколькими expert/source frames; у `Unknown` evidence
  может быть пустым.
- `Withheld`/`Unknown` не получают искусственное отрицательное value и не превращаются в confidence=0
  отрицательного класса.

### Unified entity registry и dual identity

```rust
pub struct EntityKey {
    pub run_id: String,
    pub track_domain: TrackDomain,
    pub track_id: u32,
    pub incarnation: u32,
}

pub enum EntityKind {
    Person(PersonFacts),
    Object(ObjectFacts),
    Animal(AnimalFacts),
}

pub struct SceneEntity {
    pub key: EntityKey,
    pub display_id: u32,
    pub kind: EntityKind,
    pub existence: SceneFact<Existence>,
    pub bbox: SceneFact<BBox>,
    pub position_3d: SceneFact<Position3D>,
    pub kinematics: SceneFact<Kinematics>,
}
```

`EntityKey` — run-scoped identity конкретной tracker incarnation. Gallery identity остаётся отдельным
`SceneFact<GalleryIdentity>`: разные появления распознанного человека можно связать семантически, не
притворяясь, что это один непрерывный track.

Person/Object trackers увеличивают `incarnation` при каждом повторном выделении локального id; reuse из
free-list никогда не восстанавливает старый `EntityKey`.

Current display ranges (`0–9`, `10–9999`, `>=10000`) сохраняются только как compatibility/rendering
contract v0.4.0 и больше не используются как единственный binding key.

Animal-классы проходят stable tracking и публикуются как `Animal` entities. Anti-pet IoU gate остаётся
производным guard против ложного human skeleton, а не единственным местом существования animal signal.

Rich typed entity хранится ровно один раз. Старый lossy `Scene.entities` становится вычисляемым
`SceneEntityView`, а не второй сохранённой копией истины.

### Canonical frame

```rust
pub struct FusedSceneFrame {
    pub stamp: FrameStamp,
    pub sources: Vec<SourceStamp>,
    pub entities: Vec<SceneEntity>,
    pub relations: Vec<SceneRelation>,
    pub zones: Vec<SceneZone>,
    pub activities: Vec<SceneActivity>,
    pub events: EventTailView,
    pub anomaly: SceneFact<f32>,
    pub quality: SceneQuality,
}
```

Fusion:

- единожды строит relations;
- единолично владеет persistent `ZoneTracker`;
- выводит 2D/3D kinematics по последовательным `EntityKey`;
- единолично владеет activity FSM, anomaly temporal state и previous fused frame;
- назначает каждому scene event монотонный `event_seq`;
- хранит bounded event tail; каждый consumer имеет собственный cursor;
- при отставании cursor сообщает явный `EventGap`, а не молча теряет данные;
- никогда не делает destructive `events.clear()`.

### Facts, evidence и transport

RGB, raw depth map и instance masks — evidence/artifacts, а не сериализуемые scene facts. Runtime
публикует единый packet:

```rust
pub struct FusedFramePacket {
    pub scene: Arc<FusedSceneFrame>,
    pub evidence: FrameEvidenceHandles,
}
```

`FrameEvidenceHandles` содержит cloneable `Arc`-handles и source stamps. Cortex/StateFile читают только
`scene`; DatasetWriter получает тот же packet и поэтому не соединяет independently latest RGB с другим
world revision.

---

## 5.3. Кто что публикует после refactor

```text
CaptureAgent ──────→ LatestValue<Frame>
                         │ camera clock
HumanAgent ─────────→ LatestValue<HumanObservationFrame>
SceneAgent ─────────→ LatestValue<SceneObservationFrame>
                         ├ objects + tracked animals
                         └ masks (HumanAgent may read latest stamped masks)
DepthAgent ─────────→ LatestValue<DepthObservationFrame>

FusionAgent ────────→ LatestValue<FusedFramePacket>
                         ├ FusedSceneFrame  ← only facts authority
                         └ evidence handles

Compatibility ─────→ WorldState read-only projection (v0.4.0 only)
CortexAgent ────────→ LatestValue<CortexFrame>

StateFileWriter ────→ current_state.md       (atomic latest facts)
CortexWriter ───────→ current_cortex.json    (atomic latest cognition)
DatasetWriter ──────→ labels.jsonl + pixels  (only durable semantic history)
JournalWriter ──────→ journal.md             (derived rolling view)
```

`mvm-mesh` обобщает нынешний `LatestFrame` до latest-wins `LatestValue<T>`. Intermediate camera и
observation values могут отбрасываться без backlog — это текущая CPU-first семантика, но каждый
сохранившийся value stamped и причинно проверяем.

`WorldState` в v0.4.0:

- строится только из `FusedSceneFrame`;
- не содержит session lifecycle;
- не имеет agent write-sites;
- помечен deprecated;
- удаляется после v0.4.0, когда downstream compatibility window закрыта.

Ни один persisted view не публикует обратно в runtime:

- DatasetWriter забирает все `SceneEvent` после своего cursor и прикрепляет их к следующей записанной
  observation, сохраняя реальный source frame/time;
- CortexAgent, StateFileWriter и deprecated flat-event writer имеют независимые cursors;
- `journal.md`, `recap` и `inspect` читают semantic sessions и являются только производными views;
- ошибка одного sink не останавливает и не меняет canonical Fusion state.

---

## 5.4. Scene events против Cortex interpretations

Evidence boundary утверждён:

| Fusion `SceneEvent` | Cortex interpretation |
|---|---|
| entity appeared/disappeared, person entered/exited | novelty, unexpected appearance, identity continuity claim |
| relation started/ended, object touched/held/released | affordance change, interaction importance |
| activity started/ended | episode relevance, memory association |
| zone occupied/freed | context change, attention target |
| posture transition, object moved | prediction error significance |
| anomaly threshold transition | risk salience, uncertainty-qualified warning |

Fusion events — canonical temporal facts и могут попадать в `labels.jsonl`. Cortex events/claims всегда
ссылаются на evidence (`event_seq`, `EntityKey`, prediction error, facts) и не записываются как новая
конкурирующая durable truth.

---

## 6. Публичные типы `mvm-cortex`

Ориентировочный API:

```rust
pub struct VisualCortex {
    predictor: PredictiveLayer,
    events: EventInterpretationLayer,
    memory: EventMemoryLayer,
    affordance: AffordanceLayer,
    attention: AttentionLayer,
    self_model: SelfModelLayer,
    uncertainty: UncertaintyLayer,
}

pub struct CortexInput<'a> {
    pub previous_scene: Option<&'a FusedSceneFrame>,
    pub current_scene: &'a FusedSceneFrame,
    pub scene_events: &'a [SceneEvent],
    pub config: &'a Config,
    pub camera: Option<&'a CameraParams>,
    pub now: SystemTime,
}

pub struct CognitiveClaim<T> {
    pub value: Option<T>,
    pub confidence: Option<f32>,
    pub status: ClaimStatus,
    pub reasons: Vec<ReasonCode>,
    pub evidence: Vec<EvidenceRef>,
}

pub struct CortexFrame {
    pub timestamp: SystemTime,
    pub scene_frame_id: u64,

    pub forecast: CognitiveClaim<SceneForecast>,
    pub prediction_errors: Vec<PredictionError>,

    pub scene_event_refs: Vec<EventRef>,
    pub interpretations: Vec<CortexEvent>,
    pub memory: MemoryContext,

    pub affordances: Vec<AffordanceResult>,
    pub attention: AttentionMap,

    pub self_model: VisualSelfModel,
    pub uncertainty: UncertaintyReport,
}
```

Canonical call:

```rust
let mut cortex = VisualCortex::new(CortexConfig::default());

let cortex_frame = cortex.update(CortexInput {
    previous_scene: prev_scene.as_ref(),
    current_scene: &scene,
    scene_events: scene.event_delta(cursor),
    config: &config,
    camera: camera_params.as_ref(),
    now,
});
```

`CognitiveClaim<T>` обязателен для всех cortex outputs. Он не переиспользует `SceneFact<T>`:
`SceneFact` описывает evidence-backed world substrate, `CognitiveClaim` — интерпретацию этого evidence.
Любой claim обязан перечислить reason codes и evidence refs; без evidence он `withheld`/`unknown`, а не
утверждение по умолчанию.

---

## 7. Внутренний порядок вычисления Cortex

Порядок слоёв не произвольный. Его нужно закрепить контрактом:

```text
1. Scene event ingestion
   consume Fusion SceneEvent[] by cursor

2. Prediction error
   compare(previous_forecast, current_scene) → PredictionError[]

3. Event interpretation
   scene events + prediction error → CortexEvent[]

4. Prediction
   forecast(current_scene, horizon) → SceneForecast

5. Memory
   update runtime memory with scene/cortex events
   retrieve relevant context

6. SelfModel
   camera pose / visible zones / blind zones / calibration / uncertainty context

7. Affordance
   reachable / held / attended / blocked / sittable / approaching

8. Attention
   rank entities/events using novelty, prediction error, uncertainty,
   recent events, affordance, identity, activity and anomaly

9. Report
   produce compact human/LLM-readable cognitive summary
```

Правила зависимости:

```text
Scene events before Cortex interpretation and Memory.
Prediction before Attention.
SelfModel before Report.
Affordance before Attention.
Attention after all salience sources are known.
```

---

## 8. Runtime-интеграция: один `CortexAgent`

В AgentMesh не нужно плодить пять новых равноправных агентов. Добавляется один верхний агент:

```text
CortexAgent
```

Он живёт в:

```text
mvm-cli/src/agents/cortex.rs
```

и делает только orchestration:

```text
read LatestValue<FusedSceneFrame>
call VisualCortex::update(...)
write LatestValue<CortexFrame>
```

Вся когнитивная логика остаётся в `mvm-cortex`.

Целевой AgentMesh:

```text
CaptureAgent → LatestValue<Frame>
    ↓
┌───────────────────┬───────────────────┬───────────────────┐
│ HumanAgent        │ SceneAgent        │ DepthAgent        │
│ body/id/emotion   │ objects/masks     │ depth             │
└───────────────────┴───────────────────┴───────────────────┘
    ↓
FusionAgent → LatestValue<FusedFramePacket>
    ↓
CortexAgent → LatestValue<CortexFrame>
    ↓
StateFileWriter / MCP / CLI render / Dataset consumer
```

Красивый смысл:

```text
Perception → Fusion → Cortex → Agent
```

---

## 9. Semantic Session Log и Event Memory

Фаза 16.7 уже дала durable history. Фаза 18 не должна создавать второй конкурирующий store.

Разделение:

```text
Semantic Session Log = долговременная запись наблюдений на диск
EventMemoryLayer     = рабочая когнитивная память runtime
```

Метафора:

```text
mvm-dataset/session = archive / hippocampus log
mvm-cortex/memory   = working episodic context
```

Правила:

1. `mvm-cortex::memory` хранит recent runtime events и compact context.
2. `mvm-dataset` не зависит от `mvm-cortex`.
3. `mvm-cortex` может читать recent session context через публичные read-only APIs, если это нужно для retrieval.
4. Persistent schema `labels.jsonl` не ломается ради runtime memory и остаётся schema `16.7.0`.
5. `CortexFrame` не добавляется в `labels.jsonl` целиком и не создаёт cortex-history JSONL.
6. `recap`/`inspect` постепенно могут вычислять cognitive view из semantic facts, но semantic session остаётся source of durable observation.

Additive durable compatibility:

- старый frame-local `entity_id` остаётся;
- `EntityRecord` получает optional structured `entity_key`;
- relations/events сохраняют старые participant/end-point поля и аддитивно получают canonical keys;
- event object получает `event_seq`, `source_frame_id`, `source_timestamp`;
- новые поля используют `#[serde(default)]`/`skip_serializing_if`, поэтому старые readers продолжают читать;
- animal entity пишется как G₂ record с `kind="animal"`;
- schema version не bumpается: wire shape расширяется только внутри уже открытых объектов.

Session lifecycle полностью принадлежит DatasetWriter. Он больше не записывает `current_session_id` или
`last_people_seen` в scene substrate.

---

## 10. CLI и MCP-интерфейс

Существующие инструменты остаются:

```text
snapshot
recap
inspect
serve
```

Новый канонический интерфейс — агрегированный:

```text
mvm cortex --view <view>
```

Views:

```text
mvm cortex --view full
mvm cortex --view forecast
mvm cortex --view errors
mvm cortex --view events
mvm cortex --view memory
mvm cortex --view affordance
mvm cortex --view attention
mvm cortex --view self
mvm cortex --view uncertainty
```

`view=events` показывает два явно подписанных раздела: canonical `SceneEvent` delta и Cortex
interpretations. Они не смешиваются в один event enum.

MCP tool:

```json
{
  "name": "cortex",
  "description": "Read MVM visual cortex state: forecast, events, memory, affordance, attention, self-model, uncertainty.",
  "parameters": {
    "view": {
      "type": "string",
      "enum": ["full", "forecast", "errors", "events", "memory", "affordance", "attention", "self", "uncertainty"],
      "default": "full"
    },
    "top_k": { "type": "integer", "default": 5 },
    "horizon_seconds": { "type": "number", "default": 2.0 }
  }
}
```

Не добавлять сразу семь равноправных MCP tools как канон. Допустимы aliases позже:

```text
forecast       → cortex(view="forecast")
events         → cortex(view="events")
affordances    → cortex(view="affordance")
attention      → cortex(view="attention")
self_model     → cortex(view="self")
```

Но основной discoverable интерфейс — один: `cortex`.

Cross-process latest artifacts:

```text
current_state.md     = observed/fused facts only
current_cortex.json  = latest CortexFrame only
labels.jsonl         = only durable semantic history
journal.md           = derived rolling recap
```

`current_state.md` и `current_cortex.json` пишутся atomic temp+rename в той же директории. Snapshot не
смешивает cognition с facts. `mvm cortex` и MCP tool `cortex` читают `current_cortex.json`, валидируют
freshness/scene frame id и честно отвечают `unavailable`/`stale`, если producer не запущен или Cortex
отстал.

Legacy transition:

- отдельный sync state pipeline удаляется;
- `--native-camera` сохраняется как Rust capture backend общего AgentMesh и получает тот же
  Fusion/Cortex contract;
- Python bridge `--native` остаётся deprecated compatibility path до конца v0.4.0, печатает warning и
  не обещает Cortex parity;
- flat `scene-*.jsonl`/`events-*.jsonl` остаются deprecated projections с независимыми event cursors на
  compatibility window; они не являются source of truth.

---

## 11. Visual QA Benchmark

Benchmark остаётся отдельным каталогом:

```text
benchmark/
├── README.md
├── scenarios/
├── questions/
├── baselines/
└── eval/
```

Минимальные сценарии:

| # | Сценарий | Проверяемые claims |
|---|---|---|
| 001 | Человек входит в кадр | entity appeared, temporal continuity |
| 002 | Человек садится / встаёт | posture transition, activity change |
| 003 | Человек берёт объект | hand contact, held, affordance |
| 004 | Объект перекрыт телом | object permanence, occlusion reasoning |
| 005 | Человек выходит и возвращается | identity continuity, re-identification |
| 006 | Человек отворачивается от камеры | body-primary, identity persistence |
| 007 | Два человека пересекаются | multi-person tracking, ID swap |
| 008 | Новый объект появляется | novelty, attention |
| 009 | Собака входит в кадр | anomaly, entity classification |
| 010 | Несколько объектов взаимодействия | multi-object relation, affordance |
| 011 | Человек стоит неподвижно | forecast stability, activity persistence |
| 012 | Резкое движение | prediction error, attention spike |

Метрики:

| Claim | Метрика |
|---|---|
| Canonical determinism | Одинаковые facts/relations/zones/events при разных sink cadence |
| Event delivery | 0 silent loss/duplicates; explicit `EventGap` under forced overflow |
| Source honesty | Correct fresh/coasted/stale/withheld classification |
| Temporal continuity | QA accuracy на temporal questions |
| Identity continuity | IDF1, ID switches, re-id delay |
| Object permanence | Occlusion reasoning accuracy |
| Uncertainty honesty | Honest uncertainty rate |
| Affordance | Affordance classification accuracy |
| Prediction | Forecast IoU @ 1s / 2s |
| Attention | Recall@3: реальные события в top-3 attention |
| Memory | Retrieval precision@5 |
| Self-model | Correct uncertain/blind/calibration claims |

Benchmark — не декоративный. Без него claims visual-cortex manifesto остаются риторикой.

---

## 12. Подфазы

### 18.0 — State Contract + Benchmark Baseline

**Статус:** ✅ закрыта 27 июля 2026 г. Добавлены serializable canonical types в
`mvm-core::canonical`, read-only `WorldStateProjection`, incarnation registry,
frame-based freshness, non-destructive event cursor с явным `EventGap`, девять
contract tests и `benchmark/` с шестью обязательными сценариями. Baseline v0.3.0:
synthetic continuity 2/2, scene-event recall 0/4, honesty 2/2,
last-observation forecast 1/2; неизмеренные live-метрики оставлены
`not measured`. Same-hardware live performance повторно снимается перед merge
runtime-миграции 18.1.

**Цель:** до runtime-рефакторинга зафиксировать executable contract canonical facts и baseline, против
которого будут измеряться все Cortex claims.

**Задачи:**

1. Добавить в `mvm-core` типы `SceneFact<T>`, `EntityKey`, `SceneEntity`, `FusedSceneFrame` и
   read-only `WorldState` projection.
2. Зафиксировать JSON/replay fixture format для `FusedSceneFrame` и `CortexFrame`; fixture не становится
   production history store.
3. Создать benchmark skeleton и минимум сценарии 001, 004, 007, 009, 011, 012.
4. Снять current baseline без Cortex: continuity, event recall, honesty, last-observation forecast.
5. Добавить contract tests на `withheld`/`unknown`, incarnation reuse, stale sources и deterministic
   ordering.
6. Зафиксировать performance budget: не хуже текущего capture throughput более чем на 5%; никакой
   новой модели/зависимости.

**Гейт:**

```bash
cargo test --workspace
cargo test -p mvm-core canonical_scene_contract
```

Baseline report committed; ни один следующий feature gate не принимается без соответствующей метрики.

---

### 18.1 — Typed Observation Bus + Canonical Fusion

**Статус:** 🚧 начата 27 июля 2026 г. Первый проверенный срез обобщил mesh-channel до
`LatestValue<T>`, ввёл runtime-only stamped Human/Scene/Depth observations, заменил unstamped
depth/person-mask slots и убрал прямые `WorldState` write-sites из expert agents. Fusion теперь
camera-driven. Второй срез добавил `mvm-fusion::canonical::CanonicalFusion`: incarnation-safe entities,
единожды вычисленные relations/persistent zones/kinematics, sequenced event tail и публикацию
`LatestValue<FusedFramePacket>` до compatibility projection. Production StateFile/DatasetWriter читают
тот же immutable packet, имеют независимое состояние/cursor и не мутируют scene; DatasetWriter владеет
session lifecycle, `current_state.md` пишется atomic temp+rename. Search-gate оставляет единственный
compatibility `WorldState` write-site в Fusion. Следующий срез — canonical events в semantic labels,
replay determinism/cursor-overflow gates и завершение legacy capture transition.

**Цель:** заменить multi-writer `WorldState` на stamped expert observations и один
`LatestValue<FusedFramePacket>`.

**Задачи:**

1. Обобщить `LatestFrame` в latest-wins `LatestValue<T>`.
2. Human/Scene/Depth agents публикуют immutable stamped observation frames и больше не мутируют общий
   blackboard.
3. SceneAgent tracks objects и animals; stamped masks заменяют unstamped `PersonMaskSlot`.
4. FusionAgent становится camera-driven assembler, владеет relations, `ZoneTracker`, kinematics,
   activity/anomaly temporal state и sequenced event tail.
5. Мигрировать StateFileWriter, DatasetWriter и renderer на read-only `FusedFramePacket`.
6. DatasetWriter переносит session lifecycle внутрь себя.
7. `WorldState` остаётся только deprecated projection; запретить новые write-sites тестом/search gate.
8. Унифицировать `--native-camera` с AgentMesh; `--native` пометить deprecated.
9. Писать latest artifacts atomic temp+rename.

**Гейт:**

- один и тот же replay даёт одинаковые entity keys, relations, zone ids и scene events при разных sink
  cadence;
- два consumers читают один event tail без потерь/дубликатов;
- forced cursor overflow выдаёт `EventGap`;
- late depth/mask result не меняет опубликованный frame;
- DatasetWriter не имеет mutable access к canonical scene;
- `mvm dataset check` проходит на text-only и pixel fixture;
- legacy snapshot/recap/inspect outputs сохраняют honesty semantics.

---

### 18.2 — Cortex Skeleton

**Цель:** создать каркас `mvm-cortex` без изменения поведения runtime.

**Задачи:**

1. Добавить `crates/mvm-cortex` в workspace.
2. Создать публичные типы `VisualCortex`, `CortexInput`, `CortexFrame`, `CortexConfig`,
   `CognitiveClaim<T>`.
3. Создать пустые/минимальные modules: `predict`, `events`, `memory`, `affordance`, `attention`, `self_model`, `uncertainty`, `report`.
4. Добавить `CortexAgent` в `mvm-cli/src/agents/cortex.rs`, но за feature/config flag или no-op режимом.
5. Проверить, что существующие `run`/`snapshot`/`recap`/`inspect` не изменились.

**Гейт:**

```bash
cargo test --workspace
cargo run -p mvm-cli --release -- run --stub --max-frames 100
```

Ожидание: поведение MVM до включения cortex не меняется.

---

### 18.3 — Scene Events + Cortex Interpretation

**Цель:** завершить evidence split: Fusion выделяет наблюдаемые transitions, Cortex интерпретирует их,
но не дублирует scene diff.

**События V1:**

```text
EntityAppeared
EntityDisappeared
PersonEntered
PersonExited
PersonSatDown
PersonStoodUp
ObjectTouched
ObjectHeld
ObjectReleased
ObjectMoved
ActivityStarted
ActivityEnded
AnomalySpike
ZoneOccupied
ZoneFreed
```

Fusion output:

```text
SceneEvent { event_seq, source_frame_id, source_timestamp, entity_keys, fact }
```

Cortex input: canonical event delta + current/previous `FusedSceneFrame` + previous forecast.

Cortex output: evidence-linked `CortexEvent`/`CognitiveClaim`, например `UnexpectedAppearance`,
`PredictionMiss`, `HighSalienceInteraction`.

**Гейт:** unit-тесты на synthetic/replay diff для каждого базового scene event; повторное чтение cursor
не создаёт event, Cortex interpretation всегда ссылается на существующий evidence ref.

---

### 18.4 — Runtime Event Memory

**Цель:** дать Cortex рабочую эпизодическую память без создания второго durable store.

**Состав:**

```rust
pub struct EventMemoryLayer {
    recent_events: VecDeque<MemoryEvent>,
    index: EventIndex,
    config: MemoryConfig,
}

pub struct MemoryContext {
    pub recent: Vec<MemoryEvent>,
    pub relevant: Vec<MemoryEvent>,
    pub summary: String,
}
```

**Retrieval V1:**

- recent by entity;
- recent by type;
- recent by time window;
- contact duration;
- interaction frequency by object class.

**Не делать в 18.4:** vector store, FAISS/Qdrant, embeddings. Это будущий learned memory слой, не фаза 18 V1.

**Гейт:** retrieval precision@5 > 0.8 на размеченных событиях benchmark или synthetic benchmark до появления live-разметки.

---

### 18.5 — Predictive Layer

**Цель:** forecast следующего `FusedSceneFrame` через constant velocity + persistence model.

**Вход:** `current_scene`, previous forecast, canonical kinematics, activity states, relations.

**Выход:**

```rust
SceneForecast
Vec<PredictionError>
```

**Методы V1:**

- bbox extrapolation: `current + velocity * dt`;
- 3D extrapolation, если есть velocity/position;
- activity persistence with decay;
- relation persistence while both entities alive;
- prediction error: position drift, entity appeared/disappeared, activity changed, relation broken, posture changed.
- prediction связывается по `EntityKey`, а gallery identity используется только как отдельный semantic
  continuity hint;
- forecast never upgrades `remembered`/`weak` input до observed claim.

**Не делать:** Transformer/Mamba/world model training. Это направление North Star, но не фаза 18 V1.

**Гейт:** forecast @1s лучше last-observation baseline по IoU на 3+ сценариях benchmark.

---

### 18.6 — Visual Self-Model + Uncertainty

**Цель:** система должна уметь ответить: где камера, что видно, что не видно, в чём она не уверена.

**Состав:**

```rust
VisualSelfModel
SpatialSelf
CameraPose
ZoneVisibility
CalibrationReport
UncertaintyReport
UncertainClaim
```

**Источники uncertainty:**

- posture withheld;
- high emotion entropy;
- low identity confidence;
- occlusion/lost lifecycle;
- stale entity;
- source age / event gap / missing expert;
- unverified torso length / ordinal depth;
- blind/partially occluded zones.

**Гейт:** на вопросы «чего ты не знаешь?» система перечисляет uncertain claims с accuracy > 0.8 на benchmark.

---

### 18.7 — Affordance Layer

**Цель:** перейти от “объект рядом” к “что с объектом можно сделать”.

**Affordance V1:**

```text
Reachable
Held
LikelyUsed
Attended
Blocked
Approaching
Leaving
Sittable
Interactable
```

**Источники:**

- wrist landmarks;
- object bbox / mask / 3D;
- hand-object relations;
- contact duration из EventMemory;
- head yaw / facing;
- zone labels;
- object class.

**Не делать:** learned affordance predictor. Только geometry + scoring + reason strings.

**Гейт:** 5 основных классов (`reachable`, `held`, `attended`, `blocked`, `sittable`) accuracy > 0.85 на размеченных сценариях.

---

### 18.8 — Attention Layer

**Цель:** ранжировать, что важно прямо сейчас.

**Attention score V1:**

```rust
AttentionBreakdown {
    novelty,
    prediction_error,
    uncertainty,
    event_recent,
    activity,
    anomaly,
    identity,
    affordance,
    zone_violation,
}
```

**Вход:**

```text
FusedSceneFrame
PredictionError[]
EventMemory
AffordanceResult[]
UncertaintyReport
VisualSelfModel
```

**Выход:**

```text
AttentionMap / top-k AttentionScore
```

**Гейт:** top-3 attention покрывает >80% реальных событий benchmark.

---

### 18.9 — Cortex Report + CLI/MCP

**Цель:** сделать `CortexFrame` доступным человеку и LLM.

**CLI:**

```bash
mvm cortex --view full
mvm cortex --view attention --top-k 5
mvm cortex --view forecast --horizon-seconds 2
mvm cortex --view self
mvm cortex --view uncertainty
```

**MCP:** один tool `cortex(view=...)`.

**Latest artifact:** писать отдельный atomic `current_cortex.json`; `current_state.md` остаётся чистым
facts snapshot.

**Render contract:**

- human-readable;
- compact;
- honesty markers preserved;
- no raw dump by default;
- full JSON доступен только по явному флагу.

**Гейт:** LLM может через MCP получить current attention, self uncertainty и forecast без чтения raw JSON.

Также обновляются canonical public-surface contracts:

- `mvm-vision` skill — новый `cortex` pull interface, entity key semantics и неизменная honesty policy;
- `mvm-dataset` skill — additive `entity_key`/event provenance и неизменный `mvm dataset check` gate;
- repo validator/CI — drift gate для CLI/MCP names, schema markers и обязательных honesty markers.

---

### 18.10 — Benchmark Evaluation

**Цель:** закрыть фазу не наличием кода, а измерением claims.

**Сравнить:**

1. MVM-current без Cortex.
2. MVM + Cortex.
3. Baselines: single-frame VLM / sampled-frames VLM, если доступны.

**Минимальный отчёт:**

```text
Temporal QA accuracy
Identity continuity
Object permanence QA
Uncertainty honesty
Forecast IoU @1s/@2s
Attention recall@3
Affordance accuracy
Memory precision@5
Self-model claim accuracy
```

**Гейт фазы:** Cortex улучшает минимум 4 из 6 core claims без ухудшения honesty и без нарушения runtime-инвариантов.

---

## 13. Этапы сдачи

| Этап | Суть | Зависит от | Оценка |
|---|---|---|---|
| 18.0 | State contract + benchmark baseline | — | 1 нед. |
| 18.1 | Typed bus + canonical fusion | 18.0 | 2 нед. |
| 18.2 | Cortex skeleton | 18.1 | 1 нед. |
| 18.3 | Scene events + Cortex interpretation | 18.1/18.2 | 1 нед. |
| 18.4 | Runtime Event Memory | 18.3 | 1 нед. |
| 18.5 | Predictive Layer | 18.1/18.3 | 2 нед. |
| 18.6 | SelfModel + Uncertainty | 18.2 | 1 нед. |
| 18.7 | Affordance Layer | 18.3/18.4 | 2 нед. |
| 18.8 | Attention Layer | 18.4/18.5/18.6/18.7 | 1 нед. |
| 18.9 | CLI/MCP/Report + skills | все слои | 1 нед. |
| 18.10 | Benchmark Evaluation | benchmark + все слои | 1 нед. |

Критический путь:

```text
18.0 → 18.1 → 18.2/18.3 → 18.4/18.5/18.6/18.7 → 18.8 → 18.9 → 18.10
```

Оценка: **10–12 недель**.

Оптимистичный путь после закрытия state migration: **8–9 недель**.

---

## 14. Что идёт в v0.4.0

Фаза 18 — кандидат на `v0.4.0` после закрытия всех подфаз и benchmark evaluation.

В v0.4.0 входят:

- canonical `FusedSceneFrame` и typed observation bus;
- deprecated read-only `WorldState` compatibility projection;
- `--native-camera` как общий mesh capture backend;
- deprecated Python `--native` warning/compatibility path;
- Visual Cortex, CLI/MCP `cortex` и atomic `current_cortex.json`;
- additive `entity_key`/event provenance при неизменной schema 16.7.0.

Удаление `WorldState` projection и Python bridge — post-v0.4 cleanup после compatibility window.

```text
v0.1.0 — Product / фаза 14
v0.3.0 — Dataset Factory / фаза 16
v0.4.0 — Visual Cortex / фаза 18
```

`v0.2.0` исторически свёрнута в код v0.3.0.

---

## 15. Что не входит в фазу 18

Не входит:

- новые ONNX-модели;
- GPU runtime;
- VLM-интеграция;
- foundation model;
- MAE/V-JEPA/MVM-SRM training;
- Transformer/Mamba world model;
- vector store / Qdrant / FAISS;
- learned Theory of Mind;
- learned affordance predictor;
- replacement Fusion;
- bump Dataset Factory schema или второй semantic history store ради Cortex;
- ломка `snapshot`/`recap`/`inspect`;
- destructive event drain;
- хранение raw RGB/depth/masks внутри `FusedSceneFrame`;

Эти направления остаются в North Star как будущие ветки, но не входят в архитектурную V1 фазу 18.

---

## 16. Связь с North Star и академической работой

| Компонент Cortex | Способность из North Star | Теоретическая проекция |
|---|---|---|
| PredictiveLayer | continuous prediction | Predictive Processing |
| PredictionError | attention on error | PP / Free Energy |
| EventMemoryLayer | episodic memory | memory-augmented agents |
| AttentionLayer | selective attention | GWT / AST / Consciousness Prior |
| AffordanceLayer | action-oriented perception | Gibson affordance / embodied cognition |
| SelfModelLayer | observer/world separation | HOT / AST / embodied cognition |
| UncertaintyLayer | honesty / precision weighting | PP precision weighting |
| CortexFrame | global workspace content | GWT |

Фаза 18 — не “создать сознание”.

Фаза 18 — построить **операциональный когнитивный слой**, функционально закрывающий часть способностей из Полярной звезды без нарушения MVM-инвариантов.

---

## 17. Финальная схема

```text
             ┌────────────────────────────┐
             │          LLM Agent          │
             └─────────────▲──────────────┘
                           │ MCP / CLI
             ┌─────────────┴──────────────┐
             │      Agent Interface        │
             │ snapshot / recap / cortex   │
             └─────────────▲──────────────┘
                           │
             ┌─────────────┴──────────────┐
             │        CortexFrame          │
             │ forecast                    │
             │ prediction_errors           │
             │ interpretations             │
             │ memory                      │
             │ affordances                 │
             │ attention                   │
             │ self_model                  │
             │ uncertainty                 │
             └─────────────▲──────────────┘
                           │
             ┌─────────────┴──────────────┐
             │       VisualCortex          │
             │    cognitive layer          │
             └─────────────▲──────────────┘
                           │
             ┌─────────────┴──────────────┐
             │      FusedSceneFrame         │
             │ canonical immutable facts   │
             │ entities / relations/zones  │
             │ activities / scene events   │
             └─────────────▲──────────────┘
                           │
             ┌─────────────┴──────────────┐
             │          Fusion             │
             │ one assembler / event seq   │
             └─────────────▲──────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
 HumanExpert          SceneExpert          DepthExpert
 body/id/emotion      objects/masks        geometry
```

---

## 18. Критерий успеха всей фазы

Фаза 18 считается успешной, если после неё MVM можно честно описать так:

> MVM публикует один immutable `FusedSceneFrame` с stamped sources, unified entities, relations,
> persistent zones и sequenced scene events, а поверх него формирует `CortexFrame` — прогноз, ошибки
> ожидания, интерпретации событий, рабочую память, affordances, attention, self-model и uncertainty.
> Оба слоя доступны через стабильные CLI/MCP projections, проверены benchmark'ом и не нарушают
> CPU-first / $0 / интерпретируемость / honesty-first инварианты.

---

*План обновлён 27 июля 2026 г.; 18.0 реализована и проверена полным workspace gate.*
*Основание: фазовый план 18 v0.2, актуальные runtime/dataset contracts, `north-star-visual-cortex-agent.md`,
`north-star-conscious-vision.md`, академическая работа `deconstructing-consciousness-as-engineering-method.md`
и утверждённые решения о canonical `FusedSceneFrame`, evidence split и едином `mvm-cortex`.*
