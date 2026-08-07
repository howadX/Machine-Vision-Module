# Фаза 19 — Visual Spatial Substrate: RGB → agent-relative пространство

> **Версия:** 0.3 (7 августа 2026)
> **Статус:** 19.0 ✅ закрыта; 19.1–19.6 approved как последующие отдельные slices; графовая архитектура утверждена как основа spatial substrate
> **Ветка:** `kaggle`
> **Цель фазы:** расширить MVM из object-centric 3D scene graph в evidence-backed **camera-/agent-relative spatial world model**, не заменяя существующих human-centric экспертов и не смешивая spatial facts с cognitive interpretation.
>
> **Основание:** `~/wiki/autonomous-transport-cortex/06-visual-modalities-single-rgb-to-mvm-spatial-substrate.md`.

---

## 0. Короткая формулировка

Phase 18 отвечает на вопрос:

```text
Что означают устойчивые facts текущей сцены для агента?
```

Phase 19 добавляет недостающий вопрос:

```text
Как устроено пространство между сущностями, что в нём наблюдается,
что свободно или занято, что закрыто окклюзией и что меняется при движении камеры?
```

Целевой путь:

```text
RGB + camera/body metadata
  → visual fields
  → entities
  → spatial world substrate
  → persistent local world model
  → Cortex: forecast / affordance / attention policy
```

Фаза **не строит автомобильный autopilot**, не выпускает actuator commands и не заявляет automotive safety case. Она переносит из AV архитектурную дисциплину: timestamp, calibration, provenance, freshness, uncertainty, bounded authority и replay evidence.

---

## 0.1. Утверждённая графовая архитектура

Фаза 19 развивается как **graph-first visual spatial substrate**, но не заменяет `FusedSceneFrame` мутируемой graph database.

```text
visual evidence
  → immutable FusedSceneFrame (current frame graph / facts authority)
  → GraphDelta (incremental change contract)
  → bounded spatial graph projection
  → persistent temporal graph / visual memory
  → Cortex read-only interpretation
```

### Слои

```text
L0 Evidence Graph       — observations, source/provenance, timestamps
L1 Frame Scene Graph    — entities, relations, zones, events in one tick
L2 Temporal Entity Graph — incarnations, history, re-identification hypotheses
L3 Spatial World Graph  — regions, places, surfaces, visibility, occupancy
L4 Cognitive Graph      — forecast, uncertainty, claims, advisory requests
```

`FusedSceneFrame` остаётся immutable snapshot и единственной authority canonical facts. Graph projections могут иметь revisions, expire, contradict и rebuild, но не переписывают исходное evidence.

### Графовые контракты Phase 19.0

Сначала определяются typed IDs и deltas без graph database:

```text
ObservationId / FrameId / EntityKey / PersistentEntityHypothesisId
RegionId / ZoneId / SurfaceId / SpatialShardId / EventId / ClaimId

GraphDelta:
  NodeAdded / NodeUpdated / NodeRetired
  EdgeAdded / EdgeChanged / EdgeExpired
  ObservationLinked / ContradictionRecorded / ReconfirmationRecorded
```

Графовая реализация вводится поэтапно:

```text
19.0  contracts, IDs, GraphDelta, replay fixtures
19.1  camera-state evidence; no persistent graph yet
19.2  surface/region nodes from semantic evidence
19.3  bounded current spatial graph
19.4  evidence-backed spatial relation edges
19.5  persistent temporal graph and local visual memory
19.6  Cortex read-only affordance/attention integration
```

На старте не добавляются Neo4j/RDF/внешняя graph database, глобальный SLAM, unbounded voxel store или новый graph-specific crate. Первые projections реализуются в существующих `mvm-core`/`mvm-fusion`/`mvm-dataset` boundaries и используют deterministic Rust collections.

---

## 1. Зачем

Current MVM уже хорошо покрывает **entities**:

```text
body / pose / face / identity / emotion / hands
objects / optional instance masks / monocular depth
tracking / kinematics / relations / zones / activities / events
```

Это даёт ответ «кто и какие предметы сейчас в кадре». Но самостоятельному пространственному агенту также нужны facts о фоне и промежутках между объектами:

```text
floor / wall / support surface / opening
occupied / free / unknown
visible / occluded / out-of-view
camera motion / agent-relative local geometry
on / inside / supports / behind / occludes
```

Без них object detector может сказать `chair`, но не может честно обосновать:

```text
chair occupies this volume
floor around it is observed and traversable
unseen region behind the chair remains unknown
agent of this width can or cannot pass through this corridor
```

Главный вывод фазы: **не наращивать сперва object ontology, а сначала построить spatial substrate.**

---

## 2. Граница MVM и внешний sensor contract

### 2.1. MVM — self-sufficient visual subsystem, не весь организм

Один RGB stream не даёт прямых LiDAR/radar/GPS/IMU, бинокулярности, вестибулярного чувства, проприоцепции, тактильной обратной связи или аудио. Он может быть достаточным источником зрительных facts, но не заменяет отдельные sensor streams и body controller.

Для mobile-agent режима внешний runtime обязан явно предоставить либо сознательно не предоставить:

```text
CameraIntrinsics
CameraToBodyExtrinsics
EgoOdometry / actuator state
CaptureHealth
```

Если этого нет, Phase 19 публикует только `camera_local` claims. Глобальная или `agent_local` metric map должна быть `unknown`/`withheld`, а не неявно выведена из одного RGB frame.

### 2.2. Что не входит

Не входит в Phase 19:

- motion planning, trajectory generation, vehicle control или physical actuation;
- RSS braking envelope, MRM, automotive ISO 26262/SOTIF claim;
- глобальная HD map, GPS navigation или обязательный SLAM monolith;
- VLM/API/cloud service/GPU runtime;
- Transformer/Mamba/foundation world model «ради cortex»;
- замена `FusedSceneFrame`, existing Fusion, Dataset Factory или Phase 18 Cortex;
- второй durable semantic history store;
- вывод `free`, `empty`, `closed`, `safe` по одному отсутствующему detector output;
- перенос raw RGB/depth/mask payloads в `FusedSceneFrame`.

Отдельные audio/IMU/tactile sources могут появиться позже как самостоятельные typed streams; они не маскируются под RGB evidence.

---

## 3. Непереговорные инварианты

1. **Facts authority не меняется.** Только Fusion публикует immutable `FusedSceneFrame`; spatial facts входят в него evidence-backed и canonicalized.
2. **Cortex интерпретирует, не наблюдает.** Phase 19 добавляет perception/fusion facts. Cortex может строить forecast/affordance/attention поверх них, но не повышает hypotheses до observations.
3. **Unknown-first free-space.** Unseen, occluded, stale, uncalibrated или unlabeled area = `unknown`, а не `free`.
4. **Positive evidence for traversability.** `free`/`passable` требуют позитивного geometry + visibility evidence и agent-specific clearance.
5. **Stamps everywhere.** Каждый new expert result имеет source frame, produced time, age and freshness. Поздний result входит только в следующий canonical tick.
6. **Calibration gates metric claims.** Без актуальных intrinsics/extrinsics/comparable depth ranges geometry остаётся relative/camera-local или withheld.
7. **No expert replacement.** Body-primary, identity, emotion, hands, objects, masks, depth, tracker и Dataset Factory сохраняются и становятся evidence для spatial layers.
8. **CPU-first / $0 / Rust hot path.** New intelligence is arithmetic/FSM over available facts until a separately approved ONNX expert proves a measurable benefit.
9. **Benchmark-first.** Новый fact type, model или runtime path не принимается без synthetic replay fixture, truth criterion, honesty gate и no-regression measurement.
10. **Bounded map/memory.** Runtime local map has capacity/TTL/provenance; durable semantic history остаётся `labels.jsonl` through Dataset Factory.

---

## 4. Целевая слоистость после Phase 19

```text
L5  Agent Interface
    snapshot / recap / inspect / cortex / spatial view

L4  Visual Cortex
    forecast / prediction error / memory / affordance / attention policy

L3  Persistent spatial world model
    local occupancy / local map / temporal visibility / topology

L2  Canonical scene facts
    entities / spatial surfaces / relations / zones / events / quality

L1  Perception experts
    human / object / masks / depth / flow / semantic-panoptic / camera state

L0  Sensor + body context
    RGB frames / timestamps / calibration / camera-to-body / odometry / health
```

Разделение обязательно:

```text
Visual fields       = measurements and local per-pixel evidence
Spatial substrate   = fused world facts
Cortex              = meaning, forecast and advisory policy
External agent      = accepts/rejects policy and owns physical action
```

---

## 5. Future contracts

Точные Rust types появляются только в 19.0 после проверки current canonical code and fixtures. Следующая семантика обязательна уже на уровне плана.

### 5.1. Spatial facts

```text
SurfaceFact
  kind: floor | wall | ceiling | support | opening
  geometry: optional boundary / plane / normal

CellOccupancyFact
  state: occupied | free | unknown
  clearance_m: optional
  reference_frame: camera_local | agent_local

VisibilityFact
  state: visible | occluded | out_of_view | unknown
  occluders: evidence-linked entity/surface refs

CameraPoseFact
  transform: optional relative camera/body/map transform
  quality/covariance and reference frame

SpatialRelationFact
  kind: on | inside | supports | behind | occludes
  subject / object keys, evidence and freshness
```

`reachable`, `passable`, `graspable`, `inspectable` остаются **Cortex affordance claims**, поскольку зависят от body geometry, configuration and agent capability, а не являются голыми scene facts.

### 5.2. Required statuses

Все типы используют current `SceneFact<T>` semantics:

```text
Observed | Derived | Temporal | Remembered | Weak | Withheld | Unknown
```

Обязательные правила:

- `Withheld`/`Unknown` have no synthetic value/confidence and never mean a negative class;
- remembered map cells and coasted entities are not fresh observations;
- stale depth cannot assert a new metric relation or `free` cell;
- absent segmentation cannot assert `floor` or `free`;
- an accepted re-observation request is not evidence until a new stamped observation reaches Fusion.

---

## 6. Work slices and gates

### 19.0 — Spatial Contract + Replay Baseline

**Статус:** ✅ закрыта 7 августа 2026 г. Коммиты `c54797c7` (canonical contract) и
`933f4930` (replay baseline). Добавлены additive `SpatialScene` facts, typed run-scoped IDs,
reference-frame/calibration validation, bounded `GraphDelta` с append-only evidence links,
deterministic incremental/batched projection replay и 10 contract-only synthetic fixtures.
Runtime availability baseline остаётся `0/10`: Phase 19.0 намеренно не добавляет motion,
semantic/panoptic, occupancy expert или новый ONNX dependency.

**Goal:** define minimum typed spatial facts, graph IDs/GraphDelta, reference-frame rules and benchmark fixtures before adding a new model or changing runtime.

**Likely files:**

```text
crates/mvm-core/src/canonical.rs
crates/mvm-core/tests/canonical_scene_contract.rs
crates/mvm-fusion/src/canonical.rs
benchmark/visual-spatial/...
mvm/docs/phase-19-plan.md
```

**Deliverables:**

- additive typed contract for surfaces, visibility and occupancy with canonical ordering/validation;
- graph-compatible node/edge IDs and bounded `GraphDelta` contract;
- explicit `camera_local` versus `agent_local` reference semantics;
- evidence/source stamps and quality counters for spatial sources;
- six or more deterministic synthetic replay fixtures;
- no new runtime expert and no new ONNX dependency in this slice.

**TDD gates:**

1. `unknown` occupancy cannot serialise as `free` or carry synthetic clearance.
2. graph projections cannot mutate or delete source evidence.
3. GraphDelta replay is deterministic and sink-cadence independent.
4. `free` cell requires positive evidence and valid reference frame.
5. missing intrinsics/extrinsics prevents an agent-local metric claim.
6. stale source cannot create a fresh observed spatial relation.
7. canonical replay remains deterministic regardless of sink cadence.

**Exit gate:** workspace tests plus a spatial benchmark baseline pass; no existing `FusedSceneFrame` consumer observes breaking schema behavior.

---

### 19.1 — Motion Field + Camera-State Spike

**Goal:** prove whether optical flow/static-background consensus can yield useful camera-motion evidence within CPU budget before committing to a production expert.

**Scope:**

```text
RGB frame pair → flow candidates + confidence
              → static/dynamic partition
              → relative camera-motion hypothesis or withheld
```

**Rules:**

- track velocity is not a substitute for dense/background motion; it covers only detected entities;
- a fixed camera must be explicit configuration/health state, not inferred global zero motion;
- low texture, blur, dropped frames, dominant dynamic actors or insufficient parallax produce `withheld` / high uncertainty;
- external odometry, if supplied, remains its own stamped source and cross-check, not a hidden override.

**Required fixtures:** static room + pan, moving person in fixed room, moving object in fixed room, blur, dropped frames, dynamic crowd.

**Metrics:** flow endpoint error where labelled, camera-motion classification, static-background consensus, CPU p50/p95, false ego-motion rate.

**Decision gate:** select the lightest adequate implementation — classical/pyramidal flow, compact approved ONNX expert, or defer. Do not introduce a model merely because flow exists in a paper.

---

### 19.2 — Semantic/Panoptic Surface Expert

**Goal:** add observed scene fields that represent both things and background stuff:

```text
floor / wall / ceiling / table-top/support / door/opening / window/background
```

**Architecture:** a new expert publishes stamped semantic/panoptic measurements. Fusion owns conversion into `SurfaceFact`; it does not expose model labels as unqualified world truth.

**Integration:** existing object masks, person silhouettes and depth stay complementary:

```text
semantic field + instance masks + depth + temporal consistency
→ surface boundary / visibility / support candidates
```

**Honesty gates:** unlabeled pixels, poor lighting, glare, reflective/mirrored regions and out-of-distribution classes remain `unknown`/`withheld`. They must never become floor/free-space by default.

**Metrics:** per-class mIoU, floor/door boundary quality, corruption/OOD robustness, CPU latency and memory footprint.

**Decision gate:** no occupancy/traversability claim consumes the expert until semantic quality and unknown discipline pass replay/live validation.

---

### 19.3 — Surfaces, Visibility and Local Occupancy

**Goal:** build a bounded camera-/agent-local representation from already stamped evidence.

```text
semantic surfaces + depth + masks + flow/camera pose
→ occupied | free | unknown cells
→ clearance and visibility/occlusion facts
```

**Scope:**

- capacity-bounded local grid/region model, not global map;
- observed floor/support/opening boundaries;
- occupied/free/unknown cells and explicit occlusion/out-of-view state;
- agent dimensions/config only when deriving agent-relative clearance;
- `unknown` after TTL rather than persistent false certainty.

**Do not:** implement a global SLAM architecture, map loop closure, navigation policy or actuator control in this slice.

**TDD gates:**

1. a person crossing a corridor cannot turn the occluded region behind them into `free`;
2. spatial graph nodes/edges remain bounded by local-map capacity and TTL;
3. a table occluding the floor yields `unknown/occluded`, not traversable;
4. a moved chair changes future occupancy without rewriting historical facts;
5. uncalibrated depth cannot yield metric clearance;
6. map capacity/TTL bounds memory and marks old cells remembered/unknown according to contract.

**Metrics:** free-space precision/recall, unsafe-free-space false-positive rate, occupancy `unknown` discipline violations (= 0), map consistency and CPU/throughput regression.

---

### 19.4 — Evidence-Backed Spatial Relations

**Goal:** derive relations that require geometry, not just class priors:

```text
on / inside / supports / behind / occludes
```

Existing `near`, `overlapping`, `touching`, `held_by` remain unchanged; new relations are additive and receive evidence for both endpoints plus source frames.

**Rules:**

- `on` needs support-surface/depth/boundary evidence;
- `inside` needs container/opening/containment evidence;
- `behind` and `occludes` depend on camera viewpoint and must name the reference frame;
- no relation is asserted only because a `cup` normally sits on a `table`.

**Metrics:** relation precision/recall by type, counterfactual fixture failures, evidence completeness and false assertion rate.

---

### 19.5 — Persistent Local Spatial Memory

**Goal:** maintain a capacity-bounded local map across camera motion and short occlusions while preserving observation time and uncertainty.

**Scope:**

- camera-local map always available when inputs support it;
- agent-local map only with valid camera-to-body and ego-motion/odometry contract;
- provenance-bearing remembered cells, surfaces and relations;
- revisit/re-observation can confirm, update or invalidate memory; it never retrospectively edits old facts;
- temporal graph projections store revisions/contradictions while immutable frame evidence remains unchanged;
- Dataset Factory captures additive spatial semantic records through the existing session lifecycle only.

**Required tests:** revisit after occlusion, camera rotation, map expiry, calibration version change, entity ID incarnation change, replay determinism.

**Metrics:** map drift/consistency, revisit precision, memory precision@k, bounded memory and dataset validation compatibility.

---

### 19.6 — Cortex Affordance Integration

**Goal:** allow the Phase 18 Cortex to interpret spatial facts as conditional agent-relative affordances without receiving actuator authority.

```text
spatial facts + agent configuration + current Cortex uncertainty
→ passable / reachable / graspable / inspectable / blocked
→ advisory AttentionPolicy / ObservationRequest
```

**Rules:**

- `passable` requires observed free-space, clearance and a declared agent width; otherwise it is withheld;
- `reachable` requires entity/surface geometry and agent pose/capability evidence;
- `blocked` must distinguish observed obstacle from unknown area;
- Cortex may ask for `RefreshScene`, `RefreshDepth` or external odometry/agent state, but scheduler/external agent keeps authority;
- no physical motion command is emitted by MVM.

**Metrics:** affordance accuracy, unsafe-positive rate, attention recall, request resolution latency, fairness/cooldown behavior and no more than 5% same-run capture-throughput regression.

---

## 7. Benchmark suite

Every slice contributes a deterministic fixture and an explicit expected fact status.

| Scenario | Required outcome |
|---|---|
| Static room, camera pans | Background field provides camera-motion evidence; static furniture gets no false world velocity. |
| Fixed camera, person crosses a corridor | Observed person occupies cells; hidden region behind them is `occluded/unknown`, not `free`. |
| Chair moves between frames | New occupancy/event is published in a new frame with provenance; history remains immutable. |
| Table covers floor | Floor beneath/behind table remains `unknown` or `occluded`; traversability is withheld. |
| Door partially visible | Opening/openable claim is withheld until boundary/state evidence is sufficient. |
| Blur, glare, low light | Quality degradation reduces authority of dependent claims and preserves `unknown`; it creates no negative class. |
| Dynamic scene dominates frame | Camera pose is withheld/high-uncertainty, not fabricated from moving people. |
| Revisit after short occlusion | Remembered map/entity is marked temporal; new observation is required for `observed`. |
| Missing camera-to-body transform | camera-local relation allowed; agent-local clearance/passability withheld. |
| Source arrives late | Result can affect next frame only; it cannot mutate prior occupancy/relations. |

Minimum phase metrics:

```text
flow EPE / camera-motion classification
semantic mIoU + boundary quality
free-space precision/recall
unsafe-free-space false-positive rate
occupancy unknown-discipline violations = 0
spatial relation precision/recall
local-map consistency / drift / revisit precision
spatial-affordance accuracy and unsafe-positive rate
capture throughput, Human/Fusion p50/p95, dropped-frame ratio
```

---

## 8. Dependency and delivery order

```text
Phase 18 closed
  ↓
19.0 spatial contract + replay baseline ✅
  ↓
19.1 motion/camera-state spike ─┐
                                ├→ 19.2 semantic/panoptic surfaces
                                ↓
                       19.3 occupancy/visibility
                                ↓
                       19.4 spatial relations
                                ↓
                       19.5 bounded local memory
                                ↓
                       19.6 Cortex affordance integration
                                ↓
                         benchmark + live validation
```

`19.1` is a spike/decision gate, not automatic commitment to an implementation. `19.2` can research/select an expert while 19.1 is measured, but no shared production contract beyond 19.0 is changed without its evidence.

---

## 9. Success criterion

Phase 19 is successful only if MVM can be described honestly as follows:

> MVM consumes a timestamped RGB stream and explicit camera/body context, fuses entities with observed visual fields into one immutable `FusedSceneFrame`, and publishes bounded camera-/agent-relative spatial facts: surfaces, visibility, occupied/free/unknown space, local geometry and evidence-backed relations. It never mistakes unseen space for free space, stale memory for fresh observation, a Cortex hypothesis for a fact, or an advisory affordance for a physical command. Cortex can use those facts to request more observation or describe conditional affordances, while an external agent retains execution authority.

---

## Sources

- `~/wiki/autonomous-transport-cortex/06-visual-modalities-single-rgb-to-mvm-spatial-substrate.md` — matrix of virtual visual modalities, honesty rules and initial benchmark contract.
- `~/wiki/autonomous-transport-cortex/05-av-cortex-to-mvm.md` — AV principles that transfer to MVM and explicit non-transferable automotive components.
- `mvm/docs/phase-18-plan.md` — canonical facts/Cortex separation, advisory re-observation contract and runtime invariants.
- `mvm/crates/mvm-core/src/canonical.rs` — current `FrameStamp`, `SourceStamp`, `SceneFact<T>`, `FusedSceneFrame` authority.
- `mvm/crates/mvm-cortex/src/types.rs` — current evidence-linked cognitive claim contract.

*Plan created 3 August 2026. Approved 5 August 2026 as the post-Phase-18 direction. Approval fixes architecture and order, but each implementation slice still requires its own TDD, replay/live, honesty and performance gates.*
