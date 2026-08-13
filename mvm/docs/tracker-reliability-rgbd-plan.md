# MVM — Tracker Reliability RGB-D

> **Статус:** APPROVED — production implementation source of truth.
>
> **Approval:** 2026-08-13. Разрешена последовательная реализация всех slices этого документа.
> Внутренние promotion gates обязательны: approval плана не разрешает объявлять непрошедшую policy
> production-ready, подменять live evidence synthetic replay или включать недоказанный cue.
>
> **Ветка при утверждении:** `sensors`, foundation baseline `f67f5eab` (Kinect RGB-D blocks A–H).
>
> **Владельцы контрактов:** `mvm-core::PersonTracker` — person association/lifecycle;
> `mvm-scene::ObjectTracker` — object association/lifecycle; `CanonicalFusion` — единственный владелец
> canonical facts и `EntityKey` incarnations; `labels.jsonl` — единственная durable semantic history.

## 0. Цель и Definition of Done

Tracker — фундамент качественного временного ряда MVM. Этот план доводит **PersonTracker и
ObjectTracker** до законченного honesty-first production состояния: без публикации однокадровых
фантомов, forced matches, неотличимых predictions и stale facts; с устойчивыми ID при crossing,
occlusion, dropout, exit/re-entry и object churn.

```text
stamped RGB/RGB-D detections
    → bounded association measurements
    → PersonTracker / ObjectTracker
    → lifecycle-aware observation handoff
    → CanonicalFusion + EntityKey incarnation
    → StateFile / Dataset Factory / MCP / Cortex
```

План считается завершённым только когда одновременно:

1. deterministic replay исполняет настоящий Rust production tracker, а не упрощённую модель;
2. person и object held-out gates пройдены независимо;
3. `Tentative`/`Lost` не протекают в canonical presence, activities, zones, Dataset, MCP или Cortex;
4. `Occluded`/remembered evidence явно отличается от current observation;
5. RGB и RGB-D profiles имеют отдельные acceptance records; `auto` выбирает только принятый path;
6. appearance ReID имеет финальный `AcceptanceRecord` либо `RejectionRecord`, не `deferred`;
7. конфигурация, diagnostics, schema migration, rollback и compatibility cleanup завершены;
8. controlled runtime A/B, bounded live gates и long-run health подтверждены реальными артефактами;
9. в active plan не осталось TODO, optional follow-up или неподключённого production contract.

Rejection отдельной гипотезы допустим, но только если immutable evidence показывает, что итоговые
обязательные quality gates выполнены без неё.

### Execution status

| Slice | Status | Evidence |
|---|---|---|
| T0 — baseline, manifests, inventory | ✅ complete | active fail-closed registry under `benchmark/tracker-reliability/`; 7 artifacts SHA-256 pinned; 4 contract + 5 historical evaluator tests pass |
| T1 — production Rust replay/telemetry | ✅ complete | exact RGB-D `TrackerFrameBinding`; AgentMesh and frozen replay share bound tracker entrypoints; deterministic bounded typed traces and fail-closed Rust replay |
| T2 — pre-association measurement | pending | blocked on T1 production-parity envelope |
| T3 — lifecycle/publication | pending | blocked on T1/T2 evidence seam |
| T4 — metric association/abstention | pending | blocked on captured calibration split |
| T5 — entity-local occlusion | pending | blocked on T2/T3 contracts |
| T6 — evidence-decided ReID | pending | evaluated only after T3–T5 |
| T7 — ObjectTracker completion | pending | uses T1/T2 shared primitives |
| T8 — canonical/downstream integration | pending | follows accepted person/object lifecycle |
| T9 — profiles/diagnostics/rollback | pending | follows accepted runtime policies |
| T10 — promotion/default-on/cleanup | pending | requires all acceptance and live gates |

---

## 1. Неподвижные границы

1. **Observation ≠ prediction.** Kalman/constant velocity/coast — hypothesis. Новый observed fact
   появляется только от нового stamped expert observation.
2. **Fusion owns facts.** Tracker владеет association, lifecycle и local track ID; sensor, depth и
   tracker не создают `EntityKey` и не переписывают canonical history.
3. **Unknown-first.** Missing/invalid/stale/unregistered depth, holes, skew и calibration mismatch
   дают `Unknown`/`Withheld`, не zero geometry, absence, free space или отрицательный класс.
4. **No forced match.** Неоднозначный candidate разрешено оставить unmatched. False merge хуже
   краткой честной fragmentation, особенно для person identity.
5. **Exact provenance.** Geometry несёт RGB/depth stamps, pairing skew, source, calibration revision,
   registration, sampling policy, quality и typed reason.
6. **CPU-first, +0 mandatory dependencies.** Association и geometry — bounded scalar arithmetic над
   уже доступными bbox/mask/depth. Новый ONNX ReID допускается только после отдельного evidence gate.
7. **Один track table на domain.** Никаких shadow trackers, parallel lifecycle или sensor-owned IDs.
8. **Person/object separate.** Общие primitives разрешены; thresholds, corpora и acceptance decisions
   различны. Person и Object tracker не сливаются в одну state machine.
9. **Одна durable history.** `labels.jsonl` хранит canonical lifecycle/provenance. Candidate matrices,
   embeddings и debug traces туда не попадают и второго semantic store не создают.
10. **Camera-relative truth.** Fixed Kinect работает в `camera_optical`. World/odom association требует
    отдельного внешнего timestamped pose+quality contract; MVM не становится SLAM/odometry owner.

### Не делать

- не заменять MVM tracker ByteTrack/BoT-SORT/DeepSORT/AB3DMOT/ROS stack;
- не добавлять ROS, PCL, Open3D, SLAM, GPU или Python в production hot path;
- не переносить LiDAR/vehicle thresholds на indoor Kinect;
- не строить и не сохранять dense point cloud каждый кадр;
- не использовать full bbox depth, когда доступна пригодная current mask;
- не обновлять attributes окклюдированного track данными foreground/background;
- не выдавать remembered object за observed/current contact;
- не менять past frames или переиспользованную incarnation;
- не включать policy по synthetic-only или adapter-only результату.

---

## 2. Исходная точка и historical evidence

### 2.1. Production baseline

`PersonTracker` уже имеет Kalman prediction, global Hungarian, IoU/OCM/mask cues, ByteTrack low-score
recovery, embedding/identity/center re-link и lifecycle
`Tentative → Confirmed/IdLocked → Occluded → Lost`. Association происходит до enrichment.

`ObjectTracker` пока проще: class-aware greedy IoU, EMA bbox, timeout и long static permanence. Он не
имеет global assignment, явного confirmation/lifecycle, ambiguity abstention или metric gates.

Kinect blocks A–H уже дают strict paired RGB-D, registered `u16 mm`, exact native owner association,
timestamps/skew, calibration revision, sensor health и bounded canonical entity geometry. Текущая
geometry считается **после tracking** в Fusion и потому не может участвовать в association.

Два существующих config fields являются ложными knobs: `tracker_iou_threshold` не управляет
захардкоженным person `IOU_GATE`, а `tracker_timeout_frames` не переносится `with_config()` в tracker.
Они должны быть подключены к validated profile либо удалены; декоративная config не остаётся.

### 2.2. Block I synthetic baseline

Frozen `synthetic_replay` сравнил:

```text
A = current bbox IoU + 2D motion adapter
B = A + impossible metric-Z gate
C = B + bounded XYZ constant-velocity score
```

| Variant | IDF1 proxy | ID switches | Fragmentations | False births | False merges | Adapter runtime |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.750 | 9 | 1 | 2 | 4 | baseline |
| B | 0.750 | 9 | 1 | 2 | 0 | +8.51% |
| C | 0.750 | 9 | 1 | 2 | 0 | +18.10% |

Вердикт остаётся historical rejection: A retained; B/C не разрешены для production. Это adapter
microbenchmark, не end-to-end MVM latency и не acceptance текущего плана.

### 2.3. Exploratory captured baseline

В workspace уже есть consented one-person Kinect capture `2026-08-13_115555`: 39 вручную
просмотренных paired RGB-D samples. Он выявил production fragmentation/phantoms, но недостаточен для
promotion:

- production projection: IDF1 `0.308`, 14 switches, 4 fragmentations, 11 false positives,
  4 stale predictions на absent samples;
- adapter A/B одинаковы; C хуже (7 proxy switches, около +24% adapter runtime);
- один человек, нет captured crossing, object/mask slice и независимого XYZ ground truth;
- 39 Dataset samples на 1800 capture / 375 Human ticks не дают tracker-cadence replay;
- A/B/C adapter не воспроизводит `PersonTracker::associate_core`.

Этот corpus сохраняется как exploratory evidence. Он не является calibration или held-out split.

---

## 3. Evidence system и corpus

### 3.1. Каноническое размещение

Active benchmark root: `benchmark/tracker-reliability/`.

```text
tracker-reliability/
├── contract.json
├── manifests/{development,calibration,held-out}.json
├── replay/                 # image-free typed Rust replay inputs
├── annotations/            # consented review labels, no model predictions as truth
├── reports/                # raw counts + derived metrics + bootstrap intervals
├── decisions/              # immutable Acceptance/Rejection records
└── capture-protocol.md
```

Старый `benchmark/tracker-depth/` мигрируется в этот root либо становится immutable `historical/`
input с одним указателем. Два активных schemas/evaluators не остаются.

### 3.2. Evidence classes

```text
unit / contract test
production-parity frozen synthetic replay
captured annotated RGB and Kinect RGB-D replay
controlled same-corpus runtime A/B
bounded live smoke
multi-entity live gate
long-run health / reconnect / calibration-change gate
```

Ни один дешёвый класс не заменяет следующий. Ignored hardware test не является live validation.

### 3.3. Split discipline

- `development`: debugging и feature design;
- `calibration`: единственное место выбора thresholds/profile constants;
- `held-out`: не читается до frozen candidate; только acceptance;
- `live`: wiring/health evidence, не замена accuracy annotation.

Splits делятся по целым sessions, людям, room setup и object instances. Manifest хранит SHA-256,
source/calibration identity, consent/privacy mode, scenario labels и parent artifacts. Frame leakage,
threshold tuning на held-out и relabel после просмотра candidate result fail closed.

### 3.4. Обязательные captured scenarios

| Domain | Scenario | Обязательная проверка |
|---|---|---|
| person | empty/static room, poster, reflection, animal | false births, tentative expiry, phantom publication |
| person | one person enter/walk/exit/re-entry | birth latency, continuity, expiry/incarnation |
| person | fast motion and frame edges | OCR/relink, fragmentation |
| person | dropout 1/2/5/10/30 frames | bounded coast versus fresh-ID churn |
| person | short and long occlusion | occluded/lost/expiry semantics |
| person | two- and three-person crossing | ID switch, false merge, abstention |
| person | similar clothing, no usable face | geometry limits and unknown discipline |
| person | partial body behind furniture | sampling quality, withheld geometry |
| person | sitting/standing/lying | bbox/extent change must not corrupt identity |
| sensor | depth holes/boundary bleed/skew/loss/mismatch | no fake metric admission |
| object | static fixtures and detector flicker | confirmation and permanence |
| object | movable removal/reappearance | lost/expiry/new incarnation |
| object | two same-class crossing/overlap | global assignment, mask/depth cues |
| object | hand occlusion and object churn | remembered versus observed |
| parity | RGB-only equivalents | accepted fallback behavior |

Minimum annotation: truth identity/instance, bbox while visible, `visible|partial|occluded|absent|unknown`
intervals, entry/exit/re-entry, known detector false positives and ambiguity markers. Dense masks и
surveyed XYZ нужны только для gates, которые заявляют mask/metric accuracy.

### 3.5. Capture contract

Tracker capture идёт на Human/Scene observation cadence, независимо от Dataset snapshot cadence, и
содержит body/object detections, low-score detections, masks/landmarks, admitted geometry, complete
association trace, lifecycle transitions и canonical projection result. Dataset Factory сохраняет
consented RGB/depth artifacts по canonical privacy/session contracts; annotation не меняет production
semantic history.

---

## 4. Public и internal contracts

### 4.1. Tracker-facing stamped observation

Конкретные имена могут быть скорректированы только для соответствия Rust naming, но wire semantics
фиксированы:

```rust
AssociationObservation {
    detection_key: DetectionKey,
    source: ObservationStamp,
    bbox: BBox,
    detector_score: f32,
    mask: Option<StampedMaskRef>,
    geometry: AssociationGeometry,
}

AssociationGeometry {
    centroid_xyz_m: Option<[f32; 3]>,
    median_z_m: Option<f32>,
    extent_m: Option<AxisAlignedExtent>,
    valid_ratio: f32,
    depth_mad_m: Option<f32>,
    boundary_contamination: f32,
    sample_count: u32,
    rgb_stamp: SensorStamp,
    depth_stamp: Option<SensorStamp>,
    pairing_skew_ns: Option<u64>,
    calibration_revision: Option<CalibrationRef>,
    reference_frame: CameraOptical,
    sampling: Mask | ErodedMask | InnerBbox | None,
    status: Observed | Weak | Withheld | Unknown,
    reason: Option<GeometryReason>,
}
```

Measurement создаётся до association из того же `mvm-depth` primitive, который после association
использует Fusion. Он не знает track ID, identity или canonical entity.

Sampling priority:

```text
current instance/person mask
    > eroded current mask/body mask
    > shrinked inner bbox
    > explicit withheld/unknown
```

Full bbox не используется как foreground metric estimate. Ownerless, empty, non-finite, stale,
misregistered, high-contamination или calibration-mismatched sample fail soft с typed reason.

### 4.2. Lifecycle vocabulary

```rust
TrackLifecycle = Tentative | Confirmed | IdLocked | Occluded | Lost
TrackObservation = Observed | Coasted | Remembered | Unknown
```

`IdLocked` применим только к person. Expired — retirement event, не живое состояние.

| Lifecycle | Canonical/downstream policy |
|---|---|
| Tentative | diagnostics/replay only; no canonical entity or current-person/object count |
| Confirmed / IdLocked + fresh match | `Existence::Present`, observed facts with current stamps |
| Occluded | same incarnation may remain; existence temporal/occluded, attributes retain original stamps |
| Lost | tracker-local strict relink candidate; no current entity claim |
| Expired | Fusion retires key; reused local ID receives new incarnation |

Tracker передаёт Fusion четыре раздельных набора: current observations, occluded projections, retained
local IDs и retirements. Это предотвращает преждевременную incarnation смену без публикации Lost как
presence.

### 4.3. Diagnostics

Bounded typed event:

```rust
TrackerDecision {
    frame,
    domain,
    detection_key,
    internal_track_id,
    display_id,
    entity_key: Option<EntityKey>,
    candidate_cues,
    rejected_reason,
    decision,
    lifecycle_before,
    lifecycle_after,
    publication,
}
```

Always-on in-memory counters bounded по cardinality. Operator surface — summary и opt-in bounded JSONL
trace с capacity/rotation. Raw embeddings, pixels, unbounded history и candidate matrices в
`labels.jsonl` запрещены.

### 4.4. Production profiles

```toml
[tracker]
profile = "auto"          # auto | rgb | rgbd
legacy_rollback = false
diagnostics = "summary"  # off | summary | bounded_trace
```

- `auto`: accepted RGB-D policy только при fresh valid metric evidence, иначе accepted RGB policy;
- `rgb`: accepted lifecycle/association без metric cues;
- `rgbd`: fail closed при недоступном RGB-D contract;
- `legacy_rollback`: один аварийный switch на pre-plan association.

Calibrated weights и thresholds являются versioned internal profile, не десятками operator knobs.
`mvm doctor` показывает profile revision, выбранный path, geometry/calibration availability, diagnostics
и rollback. Старые tracker config fields подключаются к profile либо удаляются с migration note.

---

## 5. Реализация по обязательным gates

### T0 — Freeze baseline и migration inventory

**Работа:** классифицировать существующие untracked artifacts; зафиксировать historical synthetic и
exploratory captured reports; создать active contract/manifests; перечислить все tracker callers,
config knobs и downstream projections.

**Gate:** hashes verified; никаких потерянных artifacts; baseline воспроизводим; active и historical
schemas однозначны.

### T1 — Production-parity replay и telemetry

**Работа:** Rust replay вызывает те же association/lifecycle functions, что AgentMesh. Trace включает
admission, prediction, все candidate edges/cues, hard veto, assignment, birth, confirm, relink,
occlude/lost/expire и downstream publication.

**Gate:** повторный replay byte-identical кроме timing; malformed schema/hash/split fail closed;
production behavior не изменён; person/object aggregates разделены.

### T2 — Pre-association measurement seam

**Работа:** сформировать stamped `AssociationObservation` в Human/Scene agents; переиспользовать exact
registered owner unprojection и bounded sampling; передать один measurement дальше в tracker и Fusion.

**Gate:** mask/inner-bbox sampling tests; stale/skew/owner/calibration negative tests; RGB path parity;
zero duplicated geometry math; no entity ID in measurement layer.

### T3 — Confirmation и честная publication

**Работа:** person и object получают explicit confirmation/lifecycle. Tentative не публикуется.
Fusion получает retained IDs/retirements; downstream filters становятся lifecycle-aware.

**Gate:** single-frame false detections produce zero canonical entities; confirmation latency bounded;
incarnation stable through accepted occlusion and changes only after expiry; zero leakage в activities,
zones, Dataset, StateFile, MCP и Cortex.

### T4 — Metric association и ambiguity abstention

Policies измеряются отдельно, затем вместе:

1. accepted temporal RGB confirmation baseline;
2. fresh high-quality depth corroboration для birth, но missing depth neutral;
3. impossible Z/XYZ hard veto до global assignment;
4. normalized bounded geometry residual только tie-breaker;
5. ambiguity margin для 2–4 competing established tracks: unmatched/coast вместо forced merge.

Raw `IoU + metres` sum запрещён. Quality-derived residual не называется physical covariance.

**Gate:** calibration split фиксирует constants; held-out не тронут; target metric improves;
false merges/relinks remain zero; RGB parity и runtime budget соблюдены.

### T5 — Entity-local occlusion evidence

```text
OccludedByObservedForeground
TemporarilyUnobserved
VisibilityContradicted
UnknownVisibility
```

Это tracker lifecycle evidence, не room visibility/occupancy/free-space fact. Foreground occlusion
разрешает bounded coast; contradiction ускоряет только Tentative/weak expiry; unknown нейтрален.

**Gate:** меньше fragmentation/phantom persistence; zero false-free claims; no Phase 19 spatial store,
surface authority или Cortex mutation.

### T6 — Evidence-decided appearance ReID

После T3–T5 измеряется остаточная crossing/re-entry ошибка.

- Если full gate уже пройден — commit `ReIdRejectionRecord` с причиной «не требуется».
- Иначе bounded CPU ONNX candidate обязан пройти provenance/license/digest, source-ground-truth quality,
  unknown discipline, ONNX parity, CPU/RSS и controlled A/B.

Accepted ReID запускается только для shortlist ambiguous confirmed/lost candidates. Он не создаёт track,
не отменяет metric veto, не переписывает history и не merge IDs единолично.

**Gate:** только `Accepted` production integration либо formal `Rejected`; `Deferred` запрещён.

### T7 — ObjectTracker completion

**Работа:** заменить greedy IoU deterministic class-aware global assignment; добавить prediction,
confirmation, lifecycle, abstention, current mask/depth cues и typed diagnostics. Static fixtures и
movable objects имеют отдельные retention policies. Animal остаётся отдельным domain.

Remembered static object не создаёт fresh contact/activity/zone evidence. Removal/reappearance после
expiry создаёт новую incarnation.

**Gate:** same-class overlap/crossing, flicker, hand occlusion, removal/re-entry и object-only scene;
zero false merge, zero remembered-as-observed leakage, bounded ID churn.

### T8 — Canonical, Dataset, MCP и Cortex integration

**Работа:** additive lifecycle/existence provenance в canonical packets и Dataset schema; StateFile,
Scene JSON, recap/inspect/MCP/Cortex читают единую projection. Old sessions читаются без rewrite.

`labels.jsonl` хранит canonical key/incarnation, existence/status и evidence refs, но не Tentative/Lost
observations и не internal trace.

**Gate:** old/new serde fixtures; dataset check + pixel validator; event cursor/replay; no phantom
activities/zones/relations/episodes; MCP/Cortex read-only и incarnation-safe.

### T9 — Configuration, diagnostics и rollback

**Работа:** реализовать validated profiles, `mvm doctor`, summary/trace output и один legacy rollback.
Удалить или подключить старые ложные knobs; документировать effective profile в run/session metadata.

**Gate:** config precedence/validation tests; `rgbd` fail-closed; `auto` deterministic fallback;
rollback не удаляет data и не меняет schema history.

### T10 — Promotion, default-on и cleanup

После acceptance `auto` становится production default. Legacy rollback живёт один compatibility release;
в рамках этого же плана выполняется removal gate: migration notice, отсутствие usage по diagnostics,
rollback rehearsal, удаление dead adapter/config/code/tests. Бессрочного compatibility tail нет.

**Gate:** final acceptance record, release notes, setup/doctor/help/README/skills synchronized; historical
rejected paths archived; active tree не содержит disconnected adapter или TODO.

---

## 6. Metrics и promotion policy

### Person

- phantom births и published phantoms per person-minute;
- confirmed false-positive rate и Tentative→Confirmed false-positive rate;
- IDF1/ID precision/recall; HOTA/AssA только при эквивалентном evaluator;
- ID switches, fragmentations, false merges;
- relink precision/false relink, TID и longest gap duration;
- mean/p95 phantom persistence.

### Object

- phantom births/published phantoms per object-minute;
- same-class ID switches/merges и removal persistence;
- observed↔remembered classification error;
- permanence false retention и reappearance incarnation correctness.

### Geometry/honesty

- median/p95 XYZ/Z error по distance/visibility bins;
- valid ratio, MAD, contamination distributions;
- geometry gate false accept/reject;
- stale/skew/mismatch admissions = 0;
- unknown/withheld→observed/absence/free violations = 0.

### Runtime

- capture/Human/Scene/Fusion p50,p95;
- delivered FPS/drop ratio;
- geometry sampling per entity;
- RSS и bounded allocation/history;
- worst same-run throughput regression ≤5%.

### Statistical acceptance

Person и object считаются раздельно по session-level paired bootstrap: 10,000 resamples, 95% CI.
Promotion требует одновременно:

1. статистически подтверждённого снижения phantom publication в обоих domains;
2. identity/continuity и object churn не хуже baseline;
3. observed false merges = 0 и false relinks = 0;
4. honesty/provenance violations = 0;
5. RGB-only independent held-out gate пройден;
6. throughput/RSS budgets пройдены;
7. deterministic output и raw counts приложены.

Если corpus не содержит достаточного числа critical events для заявления нулевой ошибки, gate остаётся
`insufficient_evidence`, а не pass.

---

## 7. Verification matrix

Каждый code slice:

```bash
cargo fmt --all -- --check
cargo test --workspace --no-fail-fast
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --features kinect-v1 --no-fail-fast
cargo clippy --workspace --all-targets --features kinect-v1 -- -D warnings
python3 -m unittest discover -s benchmark/tracker-reliability -p 'test_*.py'
git diff --check
```

Feature-on команды запускаются только при доступном pinned toolchain/library; отсутствие записывается как
непройденный gate, не успех. Dataset schema slice дополнительно запускает `mvm dataset check --strict`
на old, RGB-only и metric RGB-D fixtures и pixel validator там, где есть pixels.

Final runtime validation:

1. same-corpus legacy/RGB/RGB-D A/B;
2. bounded one-person + empty-room live smoke;
3. multi-person crossing;
4. same-class object overlap/removal;
5. reconnect, packet loss и calibration revision change;
6. long-run capture/agent health;
7. snapshot/recap/inspect/MCP/Cortex audit на отсутствие phantom leakage.

---

## 8. Rollback и compatibility

- RGB-only V4L2 и accepted monocular path сохраняются.
- `auto` не использует metric cues без fresh valid RGB-D evidence.
- `rgbd` не молча откатывается на другой sensor; он сообщает fail-closed.
- Rollback выбирает legacy association, но не удаляет sessions и не переписывает history.
- Schema changes additive; old sessions/readers покрыты fixtures.
- Local ID reuse никогда не означает reuse `EntityKey` incarnation.
- Compatibility removal входит в T10 и имеет owner, release boundary и executable gate.

---

## 9. Research provenance

Алгоритмы ниже — comparators, не dependencies или автоматическое разрешение на порт:

- ByteTrack — low-score observations recover tracks, но не создают identities;
- OC-SORT/BoT-SORT — motion, appearance и camera-motion cues разделены;
- AB3DMOT — comparator для 3D gate/global assignment без переноса LiDAR assumptions;
- Autoware tracker — typed gates и maximum-score association patterns;
- ROS tf2/RTAB-Map — только reference для timestamped external camera motion boundary;
- SPENCER — multimodal detection-to-track seam;
- PCL/Open3D — offline geometry comparators, не runtime dependencies.

Любой новый source, model или threshold фиксируется точным revision/digest/license в соответствующем
decision record. Внешняя публикация не заменяет MVM captured acceptance evidence.
