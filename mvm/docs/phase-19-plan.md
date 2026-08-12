# Фаза 19 — Post-Expert Semantic and Spatial Rust Runtime

> **Версия:** 1.0 (11 августа 2026)
>
> **Статус:** historical foundation 19.0/19.1 committed; semantic model/runtime не
> принят. Все новые runtime slices заблокированы до `ExpertAcceptanceRecord` из
> [`indoor-spatial-expert-plan.md`](indoor-spatial-expert-plan.md).
>
> **Ветка:** `kaggle`.
>
> **Главный порядок:** сначала доказанно создаётся expert; только затем строится
> Rust measurement runtime, semantic compiler, Fusion facts, occupancy, memory и
> Cortex integration.

## 1. Authority двух планов

### Indoor Spatial Expert plan владеет

```text
taxonomy + data + evaluator + candidate experiments
  → calibration/OOD/abstention
  → ONNX parity + standalone CPU feasibility
  → untouched real/MVM holdouts
  → ExpertAcceptanceRecord
```

В нём нет MVM Rust runtime.

### Phase 19 владеет

```text
accepted ExpertAcceptanceRecord
  → Rust semantic measurement implementation
  → SemanticAgent and observation bus
  → controlled runtime validation
  → structural observation compiler
  → Fusion-owned canonical spatial facts
  → durable compact session records
  → visibility/occupancy
  → spatial relations
  → one bounded graph-backed local memory
  → Cortex/read-only product surfaces
```

Phase 19 не может самостоятельно объявить rejected/unmeasured model «shadow
authorized». Synthetic contract tests доказывают только code semantics, не model
quality и не право запуска модели в runtime.

## 2. Исторический committed foundation

Историю не переписываем:

- **19.0 ✅:** additive `SpatialScene`, `SurfaceFact`, `VisibilityFact`,
  `CellOccupancyFact`, typed IDs, calibration/reference-frame validation,
  bounded `GraphDelta`/`GraphProjection` и deterministic replay contracts;
- **19.1 ✅:** dependency-free stamped camera-local motion evidence через
  `MotionAgent`, Fusion-owned publication и fail-closed quality gates;
- **19.2.1 ✅ contract-only:** synthetic surface taxonomy/honesty corpus;
- **19.2.2 ✅ measured reject:** PP-MobileSeg-Tiny rejected; SegFormer-B0 passed
  static ONNX/CPU feasibility but failed measured semantic quality/honesty.

Этот foundation остаётся dormant там, где нет accepted source evidence. Он не
разрешает semantic runtime, surfaces, occupancy или local memory. Следующая
разработка начинается не с «Iteration C runtime authorization», а с expert plan.

## 3. Current clean baseline and worktree quarantine

Committed audit baseline на момент replanning — `b9125c73`. Существующие
незакоммиченные изменения (`mvm-semantic`, `SemanticAgent`, Iteration C,
Cargo/config/docs и локальные artifacts) не являются частью плана и не идут в
production.

Перед `RT-0` обязательно:

1. записать accepted expert record и точный integration-base SHA;
2. создать clean branch/worktree от этого SHA;
3. подтвердить clean `git status`;
4. не cherry-pick существующий shadow runtime целиком;
5. переносить только заново проверенные идеи через TDD и slice path allow-list;
6. сравнивать каждый gate с clean default-off baseline.

Проверки в грязном workspace не считаются release evidence, даже если проходят.

## 4. Непереговорные runtime-инварианты

1. **Expert first.** Без принятого artifact/handoff нет semantic Rust code.
2. **Fusion owns facts.** Expert и compiler публикуют measurements/observations,
   но не canonical truth.
3. **SceneAgent/HumanAgent сохраняют identity authority.** Semantic output не
   создаёт `EntityKey`, не ведёт tracker и не заменяет YOLO/body-primary.
4. **Label и status разделены.** `withheld/unknown` не semantic classes и не
   negative labels.
5. **Unknown-first.** Missing/stale/occluded/OOD/uncalibrated evidence не означает
   floor, free, empty, open, closed, passable или safe.
6. **Stamps everywhere.** Late result может войти только в следующий immutable
   fused tick; прошлый frame не мутируется.
7. **Raw payload boundary.** RGB/depth/masks/dense semantic grids остаются
   runtime evidence handles и не входят в `FusedSceneFrame`/`labels.jsonl`.
8. **One durable history.** Compact accepted facts пишутся только через Dataset
   Factory в `labels.jsonl`; второго semantic/spatial store нет.
9. **One temporal projection.** Local spatial memory использует bounded
   `GraphProjection`; отдельная pre-graph memory не создаётся.
10. **CPU-first / $0 / native Rust.** No GPU/cloud/VLM/Python hot path.
11. **Default behavior unchanged until final rollout.** Runtime integration
    default-off до controlled A/B и release gate.
12. **No physical authority.** MVM не planner/controller/safety authority.

## 5. Accepted expert input contract

`RT-0` принимает immutable copy/reference на reviewed `ExpertAcceptanceRecord`:

```text
artifact_sha256
model/checkpoint/preprocess/postprocess versions
input tensor shape/layout/range
output logits/probability shape and label order
door/window-preserving mapping
confidence/calibration/abstention thresholds
quality + corruption/OOD + selective-risk reports
target CPU p50/p95/RSS/model bytes
licence/provenance/distribution constraints
known limitations
```

Изменение artifact, preprocessing, label mapping или thresholds инвалидирует
handoff и возвращает работу в expert plan. Runtime tuning не может обучать модель
или выбирать thresholds по production/live test.

## 6. Runtime data model

### 6.1. Expert measurement

```text
SemanticMeasurementFrame
├─ ObservationStamp
├─ grid geometry
├─ per-cell visual label
├─ per-cell status
├─ calibrated confidence / uncertainty
├─ model artifact + contract version
└─ health/quality metadata
```

Labels:

```text
floor_candidate | wall_candidate | ceiling_candidate |
door_candidate | window_candidate | other_non_surface
```

Statuses:

```text
observed | weak | withheld | unknown
```

Door/window distinction нельзя silently collapse. Если committed canonical types
не могут сохранить требуемую семантику, соответствующий downstream claim
withheld до одобренного additive compatibility contract.

### 6.2. Structural observation

```text
StructuralObservation
├─ source frame / reference frame
├─ image boundary or geometry candidate
├─ evidence refs
├─ status / confidence / reason
└─ optional compatible depth/entity/motion bindings
```

Это deterministic non-canonical language. Базовые operations:

```text
observe_surface_patch(floor|wall|ceiling, image_boundary, evidence_refs)
observe_opening_candidate(door|window, image_boundary, evidence_refs)
```

Она не утверждает host wall, aperture state, support, free-space, passability или
room topology.

### 6.3. Canonical and durable data

```text
ephemeral RGB/depth/masks/grid
  → measurement
  → structural observation
  → Fusion-owned compact SceneFact
  → FusedSceneFrame.spatial
  → additive compact labels.jsonl record
  → rebuildable bounded GraphProjection
```

External corpora, raw logits/predictions and private GT остаются research data и
никогда не смешиваются с product `labels.jsonl`.

## 7. Строгая последовательность runtime slices

Каждый slice требует отдельного approval. Нельзя начинать следующий до exit gate
предыдущего.

### RT-0 — Expert handoff and clean integration contract

**Goal:** доказать, что Phase 19 получает принятый expert, а не локальный prototype.

**Deliverables:**

- reviewed `ExpertAcceptanceRecord` and artifact hashes;
- clean integration SHA/worktree and path allow-list;
- exact mapping into existing `ObservationStamp`, `FactSource::Semantic`,
  freshness and health contracts;
- required versus optional evidence matrix for every future claim;
- explicit rollback/default-off strategy;
- no code implementation.

**Exit gate:** all names map to current committed code; no dirty-worktree artifact
is an implicit dependency.

### RT-1 — Rust semantic measurement crate and offline parity

**Goal:** implement only the accepted expert envelope in native Rust.

**Scope:**

- fixed accepted preprocessing/inference/postprocessing;
- label/status separation and finite-data validation;
- model/contract version on every frame;
- offline golden parity against pinned expert predictions;
- no AgentMesh, Fusion, DatasetWriter, StateFile, CLI/MCP or Cortex wiring.

**Exit gate:** Rust output matches the accepted ONNX contract and fails closed on
bad shape, non-finite output, hash/version mismatch and withheld/OOD cases.

### RT-2 — Stamped default-off SemanticAgent

**Goal:** add a bounded measurement producer only after RT-1.

**Scope:** latest-wins queue=1, accepted cadence, source stamps, deadline,
stale/future drop, explicit health, fail-soft missing model and a shared
`SemanticObservationBus` with an actual diagnostic consumer.

**Non-goals:** Fusion facts, DatasetWriter, StateFile, durable raw grids, manifest
distribution or default-on behavior.

**Exit gate:** captured replay proves stamp/freshness/health semantics and the
consumer can inspect measurements without a second durable store. A disconnected
bus or compute-without-consumer fails the gate.

### RT-3 — Controlled runtime A/B and shadow closure

**Goal:** validate integration cost and output integrity, not model quality anew.

Measure same-run capture throughput, Human/Fusion p50/p95, dropped frames,
semantic invocation p50/p95, amortized CPU, RSS, deadline/stale drops and agent
errors. Run missing/corrupt model controls and bounded live camera smoke.

**Exit gate:** pre-registered runtime budgets pass, default-off path is bit/JSON
compatible with baseline, and observed outputs match the accepted expert
envelope. Failure returns to RT-1/RT-2; it never weakens expert quality gates.

### RT-4 — Structural Observation Compiler

**Goal:** combine only compatible fresh semantic measurements with optional
Scene/Human masks, Depth and Motion/Camera evidence.

**Rules:**

- each output names source/reference frame, evidence, confidence, status/reason;
- optional missing evidence withholds only claims that require it; basic valid
  image-plane measurements are not discarded accidentally;
- stale/conflicting inputs never form a fresh observation;
- door/window remain distinguishable;
- raw labels never become canonical facts directly.

**Exit gate:** deterministic fixtures cover absent/stale/late/conflicting inputs,
mask ownership, remembered entities, OOD and invalid calibration.

### RT-5 — Fusion-owned structural surface facts

**Goal:** admit accepted structural observations into canonical spatial facts.

**Scope:** deterministic boundary admission, stable IDs/order, evidence/freshness/
reference-frame validation and quality counters. Existing `SurfaceKind::Support`
remains dormant until a separate support expert/evidence gate exists.

**Exit gate:** every asserted fact traces to current source stamps; semantic
absence creates no surface; door/window subtype is preserved by an approved
additive contract or the claim is withheld.

### RT-6 — Durable compact surface session schema

**Goal:** make accepted canonical spatial facts replayable without storing dense
grids or creating another history.

**Scope:**

- additive `labels.jsonl` schema revision with `#[serde(default)]` compatibility;
- DatasetWriter consumes `FusedSceneFrame.spatial`;
- only compact surface facts available after RT-5 and their evidence refs;
- RT-7/RT-8/RT-9 extend the same additive section only after their own canonical
  facts pass; future empty fields are not pre-authorized here;
- reader, validator, recap/inspect and old-session compatibility tests;
- no raw RGB/depth/masks/logits/grids in text-only sessions.

**Exit gate:** old sessions still parse; new facts round-trip; replay reconstructs
the same compact canonical revision; `dataset check` remains green.

### RT-7 — Fusion-owned current visibility

**Goal:** publish current `visible|occluded|unknown` facts for registered targets.

`out_of_view` is allowed only for an already registered target with sufficient
camera-motion/reference evidence. An arbitrary unseen area remains `unknown`.

**Exit gate:** person/table occlusion, stale masks/depth, frame gaps, camera pan
and unknown-target fixtures produce no false visibility/free implication. Only
after this gate does DatasetWriter receive the additive compact visibility
projection and compatibility tests.

### RT-8 — Bounded local occupancy

**Goal:** derive current occupied/free/unknown cells or regions from positive
geometry and visibility evidence.

Before implementation freeze coordinate frame, cell geometry/units, capacity,
TTL and calibration authority:

- image-plane floor is not volumetric free space;
- `free` requires positive visible ray/volume evidence;
- occluded/out-of-view/stale regions are unknown, never inherited free;
- metric clearance requires current scale/intrinsics/extrinsics and declared
  agent geometry;
- moved entities affect only later revisions.

**Exit gate:** free-space precision/recall, unsafe-free false positives,
unknown-discipline violations (=0), latency/RSS and throughput gates pass. Only
then is the compact occupancy projection added to the existing session schema.

### RT-9 — Evidence-backed spatial relations

**Goal:** add only geometry-supported `on|inside|supports|behind|occludes` facts.

No relation comes from class priors alone. Every endpoint, frame and reference
authority is explicit. Existing near/overlap/touch/held semantics remain
unchanged.

**Exit gate:** per-relation precision/recall, counterfactual negatives and
complete evidence checks pass. Only then is the compact spatial-relation
projection added to the existing session schema.

### RT-10 — Single bounded temporal spatial projection

**Goal:** maintain local continuity using the already committed bounded
`GraphDelta`/`GraphProjection`, not a second memory representation.

**Scope:** capacity/TTL, reconfirmation, contradiction, expiry, calibration
version change, entity incarnation and deterministic rebuild from compact
`labels.jsonl` records. New observations create later revisions and never edit
historical frames.

**Exit gate:** revisit/occlusion/camera-rotation/expiry replay proves remembered
state never upgrades itself to fresh observed truth; memory stays bounded.

### RT-11 — Cortex and read-only product surfaces

**Goal:** allow Cortex and users to interpret accepted spatial facts only after
RT-10.

Scope may add conditional affordances, attention/re-observation and read-only
CLI/MCP views. `passable/reachable` require declared agent geometry/capability;
otherwise they are withheld. No actuator command or scene mutation is added.

**Exit gate:** honesty, scheduler fairness/cooldown, CLI/MCP schemas, skills and
same-run performance gates pass.

### RT-12 — Distribution and production rollout

Only after all applicable gates:

- add reviewed model manifest/setup/bundle policy consistent with licence;
- preserve default-off compatibility window, then separately approve default-on;
- document owner, target release, rollback and removal criteria;
- verify clean release build, version metadata and live bounded smoke;
- do not merge unrelated or quarantined worktree changes.

## 8. Что не делается в Phase 19

- expert training/model selection or threshold fitting;
- a second semantic/graph/history store;
- raw dense semantic grids in `FusedSceneFrame` or `labels.jsonl`;
- duplicate object/person tracking or semantic-owned `EntityKey`;
- global SLAM/map/loop closure/navigation/controller;
- `safe/passable/free` from missing detector or segmentation output;
- broad furniture/context ontology without a separate expert plan;
- physical action or safety authority.

## 9. Validation gates for every runtime slice

Start and finish each slice with explicit `git status`; record baseline and tested
SHA. Required common gates:

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
python3 scripts/validate_skills.py
git diff --check
```

Additionally require targeted deterministic replay, captured expert parity,
controlled runtime A/B where applicable, schema backward compatibility and a
bounded live camera smoke before runtime claims.

The following are different evidence and cannot substitute for one another:

```text
synthetic contract fixture
real model-quality evaluation
offline Rust/ONNX parity
controlled runtime A/B
live camera smoke
production release validation
```

## 10. Dependency order and success criterion

```text
historical 19.0/19.1 dormant foundation ✅
  ↓
Indoor Spatial Expert EXP-0…EXP-7
  ↓ only ExpertAcceptanceRecord
RT-0 handoff
  → RT-1 Rust measurement
  → RT-2 SemanticAgent
  → RT-3 runtime A/B
  → RT-4 structural compiler
  → RT-5 Fusion surfaces
  → RT-6 durable compact surface schema
  → RT-7 visibility
  → RT-8 occupancy
  → RT-9 relations
  → RT-10 one graph-backed memory
  → RT-11 Cortex/read-only interfaces
  → RT-12 distribution/release
```

Phase 19 успешна только если MVM принимает уже доказанный expert, сохраняет его
output как stamped measurement, выводит canonical spatial facts исключительно
через Fusion, пишет только compact accepted facts в единственный session store и
никогда не принимает unseen/stale/occluded space за free или temporal memory за
fresh observation.
