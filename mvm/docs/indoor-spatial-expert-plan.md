# Indoor Spatial Expert — model/data acceptance plan

> **Версия:** 1.0 (11 августа 2026)
>
> **Статус:** proposed; expert ещё не принят.
>
> **Ветка:** `kaggle`.
>
> **Граница:** этот документ создаёт и принимает **сам эксперт** как воспроизводимый
> model/data/ONNX artifact. Он не реализует Rust runtime, `SemanticAgent`, Fusion,
> spatial facts, occupancy, memory, DatasetWriter, CLI/MCP или Cortex.
>
> **Следующий документ:** только после финального `ExpertAcceptanceRecord` работу
> продолжает [`phase-19-plan.md`](phase-19-plan.md).

## 1. Цель и единственный результат

Цель — выбрать, обучить или снова честно отклонить RGB semantic expert для узкого
indoor structural visual field:

```text
RGB
  → per-pixel semantic measurement
  → calibrated confidence / abstention
  → accepted fixed-shape ONNX artifact
```

Успешный результат этого плана — не Rust-код и не `SurfaceFact`, а один
версионированный handoff:

```text
ExpertAcceptanceRecord
├─ exact model/checkpoint and SHA-256
├─ code/weights/data licence and provenance
├─ frozen label/status/mapping contract
├─ input/output tensor and preprocessing contract
├─ calibration and abstention thresholds
├─ real indoor and MVM-camera quality reports
├─ corruption/OOD/selective-risk reports
├─ ONNX framework parity report
├─ standalone target-CPU latency/RSS/model-size report
└─ explicit remaining limitations
```

Если любой обязательный gate не пройден, результатом становится
`ExpertRejectionRecord`. В этом случае Phase 19 Rust runtime не начинается.

## 2. Непереговорная последовательность

```text
taxonomy/data contract
  → real data and isolated splits
  → independent evaluator
  → frozen baseline
  → bounded candidate experiments
  → calibration/OOD/abstention
  → ONNX parity and standalone CPU measurement
  → untouched real test + MVM-camera holdout
  → ExpertAcceptanceRecord
  → only then Phase 19 Rust runtime
```

Нельзя параллельно с этим планом заранее создавать runtime crate/agent/config или
подключать модель к AgentMesh. Статическая ONNX-проверка и standalone research
harness допустимы как model-acceptance инструменты, но не добавляются в MVM Rust
workspace и не являются MVM runtime.

## 3. Clean-room boundary

Текущий committed baseline для аудита — `b9125c73`. Существующие незакоммиченные
`mvm-semantic`, `SemanticAgent`, Iteration C, Cargo/config/docs изменения и любые
локальные веса:

- не считаются частью baseline;
- не доказывают прохождение ни одного gate;
- не должны попадать в expert commits;
- могут использоваться только как read-only research evidence после отдельной
  проверки происхождения;
- не переносятся в Phase 19 автоматически.

Работа по этому плану начинается в чистой ветке/worktree от явно записанного
committed SHA. Разрешённые repository outputs до acceptance:

```text
mvm/benchmark/visual-spatial/indoor-expert-*/
mvm/docs/indoor-spatial-expert-plan.md
mvm/docs/research/                       # только reviewed manifests/reports
```

Запрещены до финального handoff:

```text
mvm/Cargo.toml / Cargo.lock
mvm/crates/mvm-semantic/
mvm/crates/mvm-cli/src/agents/semantic.rs
AgentMesh/config/manifest/setup/bundle changes
Fusion/FusedSceneFrame/DatasetWriter/CLI/MCP/Cortex changes
```

## 4. Measurement contract: label, status и ignore не смешиваются

### 4.1. Visual labels

Минимальный expert output:

```text
floor_candidate
wall_candidate
ceiling_candidate
door_candidate
window_candidate
other_non_surface
```

Это visual measurements, а не доказанные поверхности, topology или state.
`door_candidate`/`window_candidate` не означают aperture, `open`, `closed`,
`free` или `passable`. Door и window сохраняются раздельно в native mapping и
prediction artifacts; coarse collapse допускается только поздним явным adapter.

`support`, furniture/landmarks, room layout и broad indoor ontology не входят в
первый эксперт. Они требуют отдельного benefit/data/model decision после
принятия минимального structural expert.

### 4.2. Measurement status

Статус ортогонален label:

```text
observed | weak | withheld | unknown
```

- `withheld` — runtime abstention из-за invalid/non-finite/low-confidence/
  high-entropy/OOD/corruption evidence;
- `unknown` — источник или применимость отсутствуют;
- `withheld` и `unknown` не являются supervised classes и negative labels;
- `other_non_surface` допустим только для известного валидного non-target
  материала, а не как корзина для OOD/void/ignore.

### 4.3. Dataset ignore/void

Source `ignore`/`void` сохраняется отдельной evaluator mask и исключается из
обычного class denominator. Он не преобразуется ни в `other_non_surface`, ни в
`withheld` GT. Runtime abstention оценивается отдельно через selective metrics.

## 5. Data planes без смешения

| Plane | Назначение | Хранение | Запрещено |
|---|---|---|---|
| External research corpus | train/calibration/test | вне product session store; pinned manifest | коммит raw pixels/weights в git |
| Private MVM-camera holdout | untouched camera-domain acceptance | consented restricted storage | identity leakage, split reuse, threshold fitting |
| Prediction/GT artifacts | independent reproducible evaluation | hashed research artifact store | выдавать synthetic fixture за quality evidence |
| Product `labels.jsonl` | будущая durable canonical history | Dataset Factory, только Phase 19 | dense logits/grids и research GT |

Обязательны source URL/revision, acquisition date, archive/file hashes, licence/
terms, annotation version, physical scene/session identity и retention policy.
Train/calibration/test делятся по scene/session/physical space; соседние кадры
одной последовательности не могут пересекать split.

В git входят только manifests, schemas, hashes, evaluators, tiny synthetic
fixtures и aggregate reports. Raw external/private pixels, identity data,
downloaded checkpoints и full prediction tensors в git не входят.

## 6. Expert delivery gates

Каждый gate требует отдельного approval. Прохождение нижнего gate не разрешает
работу следующего автоматически.

### EXP-0 — Freeze task, baseline and authority

**Deliverables:**

- clean committed SHA и path allow-list;
- frozen visual labels/status/ignore contract;
- exact questions, non-goals and failure semantics;
- pre-registered quality, calibration, OOD and CPU metrics;
- explicit `accept | reject` decision template.

**Exit gate:** ни один type/runtime path не добавлен; thresholds не выбраны по
test/holdout результату.

### EXP-1 — Dataset/provenance manifest

**Deliverables:**

- candidate indoor corpora with code/data/weight rights reviewed separately;
- scene-isolated train/calibration/test manifests;
- versioned native-label mapping preserving door/window;
- MVM-camera consent/privacy/annotation/physical-scene protocol;
- hard negatives: cabinet fronts, posters, mirrors, curtains, glass, closed doors,
  rugs/thresholds, clutter and partial occlusions.

**Exit gate:** no unresolved licence/provenance blocker; no split leakage; absent
MVM holdout remains `not_measured`, never “covered by ADE20K”.

### EXP-2 — Independent evaluator

Evaluator reads pinned GT and stored predictions as separate artifacts and
computes:

```text
per-class support / confusion / IoU / precision / recall / F1
floor, door and window boundary F1
ECE / Brier
selective risk–coverage and withheld coverage
false-floor on ignore/OOD
door↔window and opening-lookalike errors
worst-scene / worst-condition results
```

Negative tests must reject hash drift, schema drift, leakage, collapsed output,
fabricated metrics and `ignore|withheld|unknown → floor` promotion.

**Exit gate:** evaluator reproduces a pinned tiny contract fixture but marks it
`contract_fixture_only`; real quality remains `not_measured` until real artifacts
exist.

### EXP-3 — Frozen baseline and error diagnosis

Reproduce the committed SegFormer-B0 rejection without rewriting its measured
values. Run it on the new real-data protocol as a historical control only.

**Exit gate:** baseline artifact, thresholds and error buckets are frozen before
candidate work. A rejected baseline cannot receive runtime authorization.

### EXP-4 — Bounded candidate selection/training

At most three exact candidates per iteration. All arms use the same manifests,
input geometry, augmentation policy, evaluator and CPU host. One experiment
changes one falsifiable hypothesis. Training uses only train; checkpoint,
early-stop and thresholds use only calibration.

**Exit gate:** candidate beats the frozen pre-registered profile, especially
false-floor, door/window precision/boundaries and selective risk—not only mean
IoU. Failed candidates produce rejection records, not more runtime scope.

### EXP-5 — Calibration, abstention, corruption and OOD

Freeze class-specific calibration/abstention exclusively on calibration data.
Evaluate low light, blur, glare, reflections, glass/mirrors, clutter, partial
views, corrupt tensors and out-of-domain images.

**Exit gate:** zero unreviewed `ignore|withheld|unknown|OOD → floor` promotions;
risk–coverage and worst-condition results satisfy the pre-registered contract.

### EXP-6 — Export, parity and standalone CPU feasibility

Export exact fixed-shape FP32 ONNX after quality selection. Pin framework,
opset, preprocessing/postprocessing and artifact SHA-256. Compare raw logits,
probabilities and masks between framework and ONNX Runtime. Measure complete
preprocess + inference + postprocess p50/p95, RSS and model bytes on the target
CPU; sparse amortization is reported as a hypothesis, not runtime proof.

**Exit gate:** parity and standalone CPU budgets pass. No `mvm-semantic`,
AgentMesh, manifest/setup or production model distribution is added here.

### EXP-7 — Untouched final acceptance and handoff

Run the locked artifact and thresholds once on the untouched real indoor test and
once on the untouched MVM-camera holdout. Report confidence intervals,
worst-scene/condition failures and all remaining unknowns.

**Exit gate:** reviewed `ExpertAcceptanceRecord` contains every item from section
1 and says exactly:

```text
accepted_for: stamped semantic measurement runtime integration
not_accepted_for: SurfaceFact | visibility | occupancy | free | passable |
                  door state | support geometry | navigation | safety
```

Only this record opens Phase 19 `RT-0`. If it is absent or rejected, runtime work
remains blocked.

## 7. Explicit non-goals

- any Rust runtime crate/agent/bus/config/manifest/setup implementation;
- Fusion facts, structural compiler, visibility, occupancy or free-space;
- GraphDelta, local map/memory, Cortex, CLI/MCP or durable schema changes;
- global SLAM, navigation, planner, controller or physical safety authority;
- broad furniture/context expert;
- treating semantic absence as a negative or free-space signal;
- a second durable product history store.

## 8. Validation and handoff discipline

Every experiment records exact commands, environment, input/output hashes and
decision. Required repository checks for expert-only changes:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 <targeted evaluator tests>
PYTHONDONTWRITEBYTECODE=1 python3 <targeted evaluator>
git diff --check
git status --short
```

Before handoff, verify that Cargo/workspace/runtime/config/manifest files are
unchanged relative to the clean baseline. Full Rust gates belong to Phase 19,
after expert acceptance.
