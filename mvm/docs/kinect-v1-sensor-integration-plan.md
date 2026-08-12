# Kinect v1 — план фундаментальной интеграции сенсора в MVM

> **Статус:** approved foundation tranche A–F выполнен и проверен; дальнейшие блоки требуют отдельного review/approval.
>
> **Ветка:** `sensors`.
>
> **Назначение документа:** заменить RGB-only capture seam на проверенный sensor-first RGB-D pipeline для Kinect v1. План самодостаточен и рассматривает только sensor integration.
>
> **Цель:** MVM должен принимать синхронизированный RGB-D пакет Kinect, сохранять метрическую глубину и calibration provenance, строить из них ограниченные geometry measurements и передавать их существующим RGB-экспертам, tracker, Fusion и Dataset Factory без подмены наблюдений предположениями.
>
> **Approved scope первого tranche:** image-free contracts, typed replay, generic source seam, feature-gated local Kinect backend, metric depth authority и lossless Dataset RGB-D. Canonical entity geometry, tracker cues, surfaces/visibility, IR/accelerometer interpretation и motor executor требуют следующего отдельного approval.

---

## 1. Исходное состояние

Текущий production capture contract принимает один RGB-кадр:

```text
CameraSource::read() → DynamicImage
                     → Frame
                     → Human / Scene / Depth-Anything
                     → Fusion
```

Текущий `mvm-depth`:

1. запускает Depth-Anything-V2-Small над RGB;
2. получает относительную inverse-depth;
3. ищет metric anchor по приблизительной torso-Z человека;
4. переводит относительную глубину объектов в приблизительные метры.

Следствия текущего подхода:

- абсолютная метрика зависит от known-size prior;
- без человека объектам не из чего получить metric Z;
- Depth-Anything работает редко и переиспользует старую карту;
- model inference занимает значительную часть CPU budget;
- карта не является физическим измерением сенсора.

Kinect v1 `kinect-v1-main` уже проверен как источник:

- RGB;
- raw и metric depth;
- registered depth;
- IR;
- accelerometer;
- motor tilt;
- LED;
- четырёх микрофонных каналов.

Локальные источники истины:

```text
~/.config/kinect-v1/calibration.json
~/wiki/studium/01-craft/xbox-360-kinect-sensor.md
~/.local/opt/libfreenect-0.7.5/
```

Идентичность сенсора:

```text
logical name:  kinect-v1-main
camera serial: A00364911518102A
audio serial:  A44884D07719102A
runtime:       libfreenect 0.7.5
```

Рабочая установка стационарна: optical center `1.25 м`, рабочий motor tilt `−15°`; машинный код обязан читать точные intrinsics/extrinsics и revision из calibration profile, а не копировать числа из документации.

Базовый workspace gate перед созданием плана выполнен на ветке `sensors`:

```text
cargo test --workspace: PASS
live-camera test: ignored по существующему контракту
```

### 1.1. Live combined-stream baseline 2026-08-12

Перед approval выполнены bounded direct-`libfreenect` измерения RGB +
`FREENECT_DEPTH_REGISTERED` без motor/LED/audio и без записи изображений. Всего сенсор работал 125 секунд;
для 95 секунд сохранён полный метрический вывод, один 30-секундный прогон завершился успешно, но его stdout
не был возвращён оболочкой.

Измеренный baseline:

| Метрика | Результат |
|---|---:|
| RGB delivery | `29.93–29.97 FPS` |
| registered depth delivery | `30.17–30.47 FPS` |
| duplicate/out-of-order callback timestamps | `0 / 0` |
| nominal RGB device interval | `33.338 ms` |
| nominal depth device interval | `33.369 ms` |
| observed depth gap | один interval `66.739 ms` |
| host-monotonic nearest skew p50/p95/max | `7.142 / 16.527 / 18.715 ms` |
| pair coverage при `15 ms` | `85–88%` |
| pair coverage при `20 ms` | `100%` на bounded baseline |
| registered-depth valid ratio p05/p50/p95 | примерно `0.69 / 0.70–0.71 / 0.72` |

После каждого прогона camera/audio/motor USB functions оставались доступны, serial-bound V4L2 path
восстанавливался в `/dev/video0`, reconnect watcher был active, а last init result — success.

Критическое наблюдение: RGB и depth callback timestamps имеют разные стабильные increments
(`2_000_288` и `2_002_155` raw ticks). Поэтому они сохраняются как raw per-stream device timestamps,
но **не объявляются общим clock domain** и не вычитаются друг из друга для pairing. Общей временной шкалой
первого tranche является host monotonic callback time. Начальный pairing threshold — `20 ms`; это
измеренный bounded default, который должен быть подтверждён 10-минутным live gate, а не вечная аппаратная
константа.

---

## 2. Непереговорные решения

### 2.1. Kinect — sensor bundle, не новая RGB-камера

Нельзя подключить RGB через V4L2, depth отдельным процессом и объявить потоки синхронными без:

- source timestamps;
- frame pairing policy;
- calibration revision;
- registration status;
- stream health;
- явной семантики missing/stale depth.

Основной Kinect path использует direct `libfreenect`, владеет устройством целиком и публикует один typed capture packet.

### 2.2. Metric depth становится главным depth evidence

Authority задаётся по типу результата и reference frame, а не одной глобальной лестницей:

```text
RGB-bound metric geometry:
  fresh metric_registered_sensor > explicit fallback evidence > unknown

native depth-camera geometry:
  fresh metric_unregistered_sensor > unknown

relative ordering/comparator:
  relative_monocular > unknown
```

Правила:

- при свежей registered Kinect depth Depth-Anything не запускается;
- unregistered depth нельзя связывать с RGB bbox/mask без валидного transform;
- unregistered metric depth не конкурирует с registered/monocular evidence за RGB-bound claim;
- stale depth не повышает authority свежего RGB claim;
- invalid depth означает `unknown`, не `free` и не background;
- monocular depth никогда не сериализуется как миллиметры;
- torso/face pinhole остаётся отдельным fallback evidence, пока миграция не доказана;
- разные источники геометрии не усредняются без provenance и явной fusion policy.

### 2.3. Текущий DepthAgent не удаляется сразу

Нужно разделить роль и реализацию:

```text
Depth-Anything estimator → MonocularDepthProvider
Kinect depth stream      → MetricDepthProvider
работа с геометрией       → GeometryAgent / pure geometry modules
canonical facts          → Fusion
```

`MonocularDepthProvider` остаётся:

- для RGB-only камер;
- как явно включаемый fallback при полном отсутствии metric stream;
- как comparator против Kinect ground truth;
- как источник Dataset Factory для обучения/валидации monocular depth.

Он default-off для исправного Kinect source.

### 2.4. Kinect не заменяет MVM tracker

Kinect v1 не выдаёт аппаратный MOT/skeleton tracking. Исторический Microsoft Kinect SDK выполнял skeleton tracking программно. Текущий Linux backend `libfreenect` предоставляет sensor streams, но не body tracker.

Старые OpenNI/NiTE или Microsoft SDK не включаются в production stack. Существующие `PersonTracker` и `ObjectTracker` сохраняют authority. Metric depth позже добавляется к association как ещё одно stamped evidence после отдельного replay comparison.

### 2.5. Point cloud — derived view

Primary runtime representation:

```text
registered metric depth u16
+ invalid mask/sentinel
+ intrinsics
+ calibration revision
+ timestamps and registration status
```

Dense XYZ cloud не материализуется и не сохраняется каждый кадр. XYZ вычисляется лениво для ROI, masks, downsampled diagnostics и geometry measurements.

### 2.6. Границы MVM сохраняются

- Sensor backend владеет capture, pairing, registration metadata и health.
- Geometry layer владеет stamped measurements, не canonical truth.
- Fusion владеет canonical facts.
- Dataset Factory владеет durable session history.
- Cortex только читает facts и может запросить новое наблюдение.
- Motor не вызывается напрямую из Cortex.
- Audio не включается в visual facts MVM.

---

## 3. Целевая архитектура

```text
┌────────────────────────────────────────────────────────────┐
│ Sensor source                                               │
│ KinectV1Source | V4l2RgbSource | ReplaySensorSource         │
└──────────────────────────┬─────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ SensorFrameBundle                                          │
│ RGB + metric depth + stamps + calibration + health          │
│ optional IR + gravity + motor state                         │
└───────────┬───────────────────┬────────────────────────────┘
            │                   │
            ↓                   ↓
   Existing RGB experts     MetricDepthProvider
   Human/Scene/Face/...             │
            │                       ↓
            │                Geometry measurements
            │              ROI XYZ / planes / visibility
            └───────────────────┬─────────────────────────────
                                ↓
                         Canonical Fusion
                                ↓
              StateFile / Dataset Factory / Cortex
```

### 3.1. SensorFrameBundle

Целевой conceptual contract:

```rust
SensorFrameBundle {
    source: SensorSourceId,
    run_id: RunId,
    frame_id: u64,
    rgb: Arc<DynamicImage>,
    rgb_stamp: SensorStamp,
    depth: Option<Arc<MetricDepthFrame>>,
    ir: Option<Arc<IrFrame>>,
    gravity: Option<GravityObservation>,
    motor: Option<MotorStateObservation>,
    calibration: CalibrationRef,
    health: SensorHealth,
}
```

`MetricDepthFrame`:

```rust
MetricDepthFrame {
    width: u32,
    height: u32,
    depth_mm: Vec<u16>,
    invalid_value: u16,
    stamp: SensorStamp,
    geometry: DepthGeometry,
    calibration: CalibrationRef,
    valid_ratio: f32,
}
```

`DepthGeometry` минимум различает:

```text
registered_to_rgb
native_depth_camera
unknown_registration
```

`SensorStamp` содержит:

- device timestamp, если backend его предоставляет;
- host monotonic capture time;
- wall-clock time для durable log;
- clock domain;
- source frame sequence;
- age/freshness policy.

Для Kinect RGB и depth получают разные `clock_domain`; raw device ticks сохраняются для диагностики и
wraparound-aware per-stream cadence, но cross-stream pairing выполняется только на общей host-monotonic
шкале. Wall clock не участвует в pairing.

`CalibrationRef` первого tranche содержит schema, camera serial и стабильный
`calibration_revision = sha256(canonical calibration profile bytes)`. `calibrated_at` остаётся
provenance и не служит identity. Любая смена digest создаёт новую revision и инвалидирует зависимые
world/mount projections.

### 3.2. Размещение типов

Чтобы не создавать циклические зависимости:

- `mvm-core`: image-free sensor stamps, calibration references, metric-depth metadata, units/status enums;
- `mvm-pipeline`: concrete capture packet с `DynamicImage`, Kinect/V4L2/replay backends;
- `mvm-mesh`: immutable runtime `Frame` с RGB handle и optional typed sensor evidence;
- `mvm-cli`: source selection, agents, buses и orchestration;
- `mvm-depth`: relative/metric depth containers, unprojection и geometry primitives;
- `mvm-fusion`: admission stamped measurements в canonical facts;
- `mvm-dataset`: schema, validation и durable encoding metadata.

Новый крейт на старте не создаётся. Если после первых contracts `mvm-depth` начнёт смешивать sensor I/O с geometry, выделение `mvm-geometry` принимается отдельным локальным решением по dependency graph, а не заранее.

---

## 4. Capture и synchronization contract

### 4.1. Владение устройством

Direct Kinect backend является единственным process owner камеры на время работы.

Нельзя одновременно использовать:

```text
V4L2 RGB consumer + direct libfreenect Kinect consumer
```

Backend обязан:

1. открыть только ожидаемый camera serial;
2. проверить pinned runtime/ABI;
3. запустить RGB и depth streams;
4. получать source sequence/timestamps;
5. pair RGB и depth по определённой policy;
6. публиковать packet только с честным registration/health status;
7. корректно остановить streams и вернуть kernel camera binding;
8. не управлять motor/LED вне явно разрешённого lifecycle.

### 4.2. Pairing policy

До реализации фиксируются измерения реального устройства:

- распределение `|t_rgb - t_depth|`;
- dropped RGB/depth frames;
- duplicate source timestamps;
- out-of-order callbacks;
- packet resync errors;
- sustained delivery FPS;
- valid-depth ratio по кадру и зонам.

Pairing contract должен быть capacity-bounded:

- bounded RGB/depth queues;
- nearest compatible source timestamp;
- maximum accepted skew;
- no unbounded backlog;
- explicit `rgb_only`, `depth_only_dropped`, `paired`, `stale_depth` counters;
- ни один старый depth frame не маскируется под синхронный.

Первый approved contract после live measurement:

- packet clock — RGB;
- bounded RGB/depth queues capacity `3` на поток;
- nearest **не использованный** depth по host-monotonic callback time;
- `max_pair_skew = 20 ms`;
- один depth frame не связывается с двумя RGB frames;
- RGB публикуется с paired depth либо с explicit unavailable/stale status, когда допустимая будущая
  пара уже невозможна;
- unmatched depth удаляется с explicit drop counter;
- callback gap больше одного ожидаемого периода публикуется в health counters и не скрывается reuse;
- threshold остаётся конфигурируемым, но default меняется только новым measurement evidence.

Raw Kinect device timestamps дополнительно проверяются внутри каждого потока на duplicate, out-of-order,
wraparound и gaps. Они не считаются взаимно синхронизированными.

### 4.3. Fail-soft behavior

- RGB исправен, depth временно отсутствует → RGB packet + depth status `unknown/unavailable`;
- depth corrupt/out-of-order → packet не получает metric authority;
- calibration отсутствует/не совпадает с sensor serial → Kinect RGB может быть diagnostic-only, metric association fail-closed;
- source полностью умер → CaptureAgent завершает mesh с explicit health/error;
- auto fallback на другую случайную камеру не допускается для явно выбранного Kinect source.

---

## 5. Geometry и point-cloud capabilities

### 5.1. Lazy unprojection

Для валидного registered pixel `(u,v,Z)`:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
```

Реализуются функции:

- single-pixel unprojection;
- iterator over valid ROI/mask pixels;
- stride/downsample iterator;
- camera-to-local-world transform при валидной mount calibration;
- обратная проекция для deterministic tests.

Каждый result несёт reference frame и calibration revision.

В текущем calibration profile сохранены depth-camera intrinsics, но нет отдельного подтверждённого
intrinsics contract для RGB geometry зарегистрированной карты. Поэтому первый approved tranche не
продвигает registered pixels в XYZ. Registered depth в нём разрешена для lossless persistence и
RGB-pixel association; unprojection/canonical entity geometry начинается только после отдельной проверки
camera geometry. Нельзя молча применять depth intrinsics к registered RGB pixels.

### 5.2. Entity geometry

Для RGB bbox/mask:

- valid-depth ratio;
- robust median Z;
- trimmed/median 3D centroid;
- median absolute deviation/covariance proxy;
- observed 3D AABB/extents;
- nearest/farthest observed surface;
- boundary contamination indicator;
- status/reason при недостатке данных.

Порядок качества:

```text
instance mask samples
  > body/person mask samples
  > inner bbox samples
  > no metric claim
```

Empty/invalid sample set не даёт `[0,0,0]` и не наследует чужую глубину.

### 5.3. Body geometry

RGB body landmarks связываются с registered depth только при валидном pixel neighborhood:

```text
observed RGB landmark + valid registered depth → measured 3D landmark
inferred/low-visibility landmark              → not measured
invalid depth                                 → unknown
```

Это должно позволить измерять:

- metric torso/hip/head position;
- body height/extent в наблюдаемой части;
- joint velocities;
- hand-to-object distance;
- floor-relative hip/head height;
- geometry evidence для posture/fall logic.

Ни один inferred landmark не повышается до sensor observation.

### 5.4. Planes и surfaces

Pure geometry layer может публиковать measurements:

- plane normal/distance;
- inlier count/coverage;
- residual RMSE/p95;
- gravity agreement;
- depth discontinuities;
- local normals;
- connected geometric regions.

Он не объявляет самостоятельно:

```text
horizontal plane = table
vertical plane   = wall
depth gap        = door
plane            = free/passable
```

### 5.5. Visibility и occlusion

Валидная depth описывает первую поверхность на луче:

```text
camera → observed ray interval → first surface → occluded/unknown behind
```

Обязательные статусы:

```text
visible | occupied_surface | occluded | unknown | stale | out_of_range
```

`free` не вводится как побочный эффект отсутствия depth. Если позже понадобится observed-free interval, он должен быть отдельным derived measurement с sensor range, ray validity и calibration evidence.

### 5.6. Fixed-camera change model

После стабилизации capture допускается bounded background-depth model:

- стабильный depth baseline;
- changed clusters;
- appeared/disappeared geometry candidates;
- reconfirmation и contradiction;
- invalidation при calibration/mount revision change.

Это не tracker и не durable truth. Dynamic entity authority сохраняют существующие trackers/Fusion.

### 5.7. Diagnostic point cloud

Добавляется только opt-in bounded export/view:

- configurable stride/ROI;
- optional RGB color;
- camera/local-world reference frame;
- calibration revision;
- PLY или другой простой переносимый формат;
- no automatic per-frame persistence.

Dense cloud не входит в `FusedSceneFrame`, `labels.jsonl` или latest-state files.

---

## 6. Depth-assisted tracking

### 6.1. Что не делаем

- не подключаем Microsoft Kinect SDK;
- не подключаем OpenNI/NiTE как production tracker;
- не создаём второй person/object tracker;
- не отдаём sensor backend право назначать `EntityKey`;
- не меняем association до replay baseline.

### 6.2. Candidate evidence для существующего tracker

После принятия entity geometry сравниваются:

```text
A: current IoU + OCM + masks + appearance/SFace
B: A + metric-Z gate
C: A + bounded XYZ/velocity/occlusion evidence
```

Возможные additions:

- metric centroid distance;
- impossible XYZ jump gate;
- depth-order consistency;
- 3D constant-velocity prediction;
- physical extent compatibility;
- foreground occlusion reason;
- depth-supported track birth confirmation;
- metric position as one cue для re-link.

Depth cue никогда не заменяет appearance/identity и не связывает людей только потому, что они находятся рядом.

### 6.3. Acceptance metrics

Person и object tracking измеряются отдельно:

- IDF1/HOTA/AssA;
- ID switches;
- fragmentations per tracked-person-minute;
- false track births;
- false persistence duration;
- false merges;
- re-link precision;
- tentative-to-confirmed false-positive rate;
- metric position/velocity error;
- leakage в activities/zones/Cortex;
- p50/p95 latency, FPS, RSS.

Принимается только минимальный cue set, который улучшает fragmentation/ID switches/phantom births без роста false merges и runtime regression.

---

## 7. Dataset Factory migration

Текущий depth artifact кодирует относительную `f32` карту как min-max normalized 16-bit PNG. Этот contract нельзя переиспользовать для Kinect без новой metadata semantics.

Нужно различать минимум:

```text
relative_model_depth:
  unit: relative
  encoding: normalized_u16
  reversible_range: [min, max]
  source: depth_anything

metric_sensor_depth:
  unit: millimetres
  encoding: raw_u16
  invalid_value/mask
  source: kinect-v1-main
  geometry: registered_to_rgb | native_depth_camera
  calibration_revision
  rgb/depth stamps and skew
```

Требования:

- metric `u16` round-trip exact;
- invalid pixels сохраняются;
- no dangling depth paths;
- text-only session не сохраняет dense depth;
- pixel session сохраняет sensor/source/calibration metadata;
- старые sessions остаются читаемыми через backward-compatible optional fields;
- validator fail-closed проверяет unit/encoding/source consistency;
- exporters не смешивают relative и metric samples без explicit conversion policy;
- RGB-D samples сохраняют session/scene split key и пригодны для holdout evaluation.

Dataset Factory получает новые применения:

- ground truth для Depth-Anything evaluation;
- geometry-supervised samples;
- invalid-depth/corruption hard examples;
- RGB/depth alignment checks;
- floor/plane measurements;
- tracker replay с metric trajectories.

---

## 8. Конфигурация и CLI

Предлагаемый additive config shape:

```toml
[sensor]
backend = "auto"        # auto | v4l2 | kinect_v1 | replay
device = "auto"
logical_name = ""

[sensor.kinect_v1]
serial = ""
calibration = "~/.config/kinect-v1/calibration.json"
library_prefix = "~/.local/opt/libfreenect-0.7.5"
depth_geometry = "registered_to_rgb"
monocular_fallback = false
max_pair_skew_ms = 20
```

Точные имена keys утверждаются после аудита existing config parser. Unknown keys продолжают fail closed.

CLI должен уметь:

- явно выбрать Kinect backend;
- показать source identity и calibration revision;
- вывести stream health/sync counters;
- выполнить bounded RGB-D capture smoke;
- записать replay fixture;
- экспортировать bounded diagnostic cloud;
- проверить dataset session.

`auto` не должен случайно выбирать Kinect RGB V4L2 node как обычную камеру, если пользователь ожидает полный RGB-D source.

`doctor` проверяет read-only:

- expected serial/functions;
- pinned library presence/version;
- calibration schema/serial match;
- device ownership/busy state;
- firmware presence только как operational note;
- supported modes;
- last capture health, если доступен.

`doctor` не устанавливает packages, не меняет udev/systemd и не двигает motor.

Первый Kinect backend является локальной optional Cargo feature. Default и release builds продолжают
собираться без Kinect. Backend проверяет explicit local `library_prefix`, runtime/ABI `0.7.5`, camera
serial и calibration digest; официальный release packaging, setup и CI distribution входят только в
отдельный последующий approval.

Runtime читает tilt/accelerometer только для проверки mount state и не командует motor. При несовпадении
с calibrated mount camera-local metric depth остаётся доступной, а world/mount transform получает
`unknown` с причиной mismatch. `gravity_mean_m_s2` нельзя считать optical-frame vector, пока calibration
schema не задаёт его source axes явно.

---

## 9. Рабочие блоки реализации

Каждый блок начинается с failing test/contract и завершается targeted tests, workspace gates и явным evidence report. Следующий блок не использует результаты предыдущего как доказанные, пока его exit gate не пройден.

### Блок A — image-free sensor contracts

**Цель:** зафиксировать types для stamps, units, source identity, calibration refs, registration и health.

**Изменить:**

```text
mvm/crates/mvm-core/src/
mvm/crates/mvm-core/tests/
```

**Tests:** serde round-trip, unknown-field/backward compatibility policy, non-finite/invalid dimensions, source/calibration mismatch, stale/future stamps.

**Exit gate:** contracts не зависят от `image`, `libfreenect` или CLI и fail closed на semantic mismatch.

### Блок B — RGB-D replay envelope

**Цель:** получить deterministic sensor fixture, построенный на принятых typed contracts, до production
hardware integration.

**Создать:**

```text
mvm/benchmark/rgbd/contract.json
mvm/benchmark/rgbd/fixtures/
mvm/benchmark/rgbd/evaluate.py или Rust evaluator
```

**Fixture cases:**

- paired registered RGB-D;
- RGB-only;
- missing depth;
- stale depth;
- timestamp skew over `20 ms`;
- mismatched dimensions;
- invalid calibration revision;
- unregistered depth;
- depth holes/out-of-range;
- packet gap/resync;
- per-stream timestamp wraparound/future observation.

**Exit gate:** deterministic parsing/replay; invalid evidence не получает metric RGB association; `unknown → free` violations = 0.

### Блок C — sensor packet и generic source seam

**Цель:** расширить RGB-only source без регрессии V4L2.

**Изменить:**

```text
mvm/crates/mvm-pipeline/src/source.rs
mvm/crates/mvm-mesh/src/frame.rs
mvm/crates/mvm-cli/src/agents/capture.rs
relevant stubs/tests
```

**Tests:** V4L2/synthetic source публикует bundle с RGB и explicit unavailable depth; existing RGB consumers получают тот же image; latest-wins semantics сохраняются; payload immutable/Arc-backed; bounded queue не переиспользует depth и честно считает drops/gaps.

**Exit gate:** current RGB path bit/behavior-compatible на controlled fixtures.

### Блок D — native Kinect v1 backend

**Цель:** bounded direct capture RGB + metric registered depth через pinned `libfreenect 0.7.5`.

**Создать/изменить:**

```text
mvm/crates/mvm-pipeline/src/kinect_v1.rs
mvm/crates/mvm-pipeline/src/lib.rs
mvm/crates/mvm-pipeline/Cargo.toml
mvm/crates/mvm-cli config/source selection/doctor
```

FFI feature-gated; build/release без Kinect support остаётся возможным. Не распространять firmware автоматически.

**Live gate:** serial verified; три независимых bounded starts; measured host-monotonic RGB-depth skew;
valid-depth ratio; packet errors; clean shutdown; camera binding restored; no parallel consumer conflict.
Затем записываются реальные frozen replay fixtures. По решению оператора от 2026-08-12 требование
длинного gate сокращено с 30 до 10 минут: 10 минут combined capture,
pairing ratio `≥99%` при `20 ms`, explicit accounting всех gaps/resync и restoration `3/3`.

**Implementation evidence 2026-08-12:** feature-gated runtime-loaded backend открывает только camera
serial `A00364911518102A` через exact local `libfreenect.so.0.7.5`, использует RGB 640×480 +
`FREENECT_DEPTH_REGISTERED`, callback-entry host-monotonic stamps, bounded queues capacity 3 и one-use
nearest-depth pairing. Motor/LED/audio не открываются и не управляются. Calibration revision exact-bytes
SHA-256: `0571de941380b9e9c593abc4509d02e1a7f7085501399bf19e37ce66bc91ce1c`.

Три independent bounded starts завершились cleanly: `10/10`, `30/30`, `29/30` paired. После исправления
момента host timestamp (callback entry до копирования payload) контрольный run дал `900/900`, max skew
`16.983 ms`. Финальный operator-approved 10-minute gate:

```text
frames=18000 elapsed_s=600.120 delivered_fps=29.994
paired=17966 pair_ratio=0.998111 max_skew_ms=19.980 valid_depth_mean=0.694493
health={"paired":17966,"rgb_only":34,"depth_dropped":24,"rgb_gaps":0,"depth_gaps":2,
        "duplicate_timestamps":0,"out_of_order_timestamps":0,"resync_errors":0}
```

Gate `≥99% @ 20 ms` пройден. После Drop штатный `kinect-v1-status` подтвердил serial-bound V4L2
`/dev/video0`, camera/audio/motor `present`, reconnect watcher `active`, last init `success`.

### Блок E — metric depth authority

**Цель:** разделить metric and relative depth representations и отключить model inference на исправном Kinect path.

**Изменить:**

```text
mvm/crates/mvm-depth/src/
mvm/crates/mvm-cli/src/agents/depth.rs
mvm/crates/mvm-cli/src/agents/observations.rs
mvm/crates/mvm-cli/src/agents/fusion.rs
scheduler/config tests
```

**Tests:** registered metric wins; unregistered cannot bind to RGB; stale metric rejected; relative never becomes mm; Kinect mode does not schedule ONNX estimator; RGB-only mode retains existing fallback.

**Exit gate:** default исправный Kinect path показывает zero Depth-Anything invocations; metric и relative
representations нельзя перепутать по типу/units/provenance; metric evidence доступна runtime/Dataset, но
не повышается до новых canonical XYZ facts в этом tranche.

**Implementation evidence (2026-08-12):** `mvm-depth` разделяет `RelativeDepthMap` и
fallible `RegisteredMetricDepth`; admission требует `Paired`, `RegisteredToRgb`, matching dimensions,
calibration revision и measured skew в configured bound. Fusion выбирает fresh registered metric evidence
раньше relative fallback, но только relative map допускается в прежний object anchoring path; metric
остаётся immutable runtime/Dataset evidence без новых canonical XYZ facts. В default Kinect mode
Depth-Anything даже не загружается; `monocular_fallback = true` является explicit opt-in, а V4L2/auto
сохраняют прежний provider.

Targeted и workspace tests покрывают registered-wins, unregistered/stale rejection, typed units,
default-off Kinect scheduling и RGB-only/explicit fallback. Feature-on workspace tests, default workspace
tests, strict clippy, fmt и diff-check зелёные. Bounded startup вывел
`Kinect registered metric provider primary (Depth-Anything disabled)` до native open, то есть zero model
load/invocation gate подтверждён. Полный 30-frame повтор не состоялся из-за текущего external hardware
state: `freenect_init failed`, затем operational status показал все три USB functions `missing`; код/config
не мутировали устройство или system services.

Operator-requested 10-minute live validation затем выполнена на feature-on binary с Kinect снова
доступным. Полный `mvm run --max-frames 18000 --no-vision-log` завершил frame budget без capture,
human, scene, Fusion или Cortex errors:

```text
depth: Kinect registered metric provider primary (Depth-Anything disabled)
capture ticks=18000 errors=0
human ticks=3333 errors=0
scene ticks=11981 errors=0
fusion ticks=17962 errors=0
cortex ticks=17962 errors=0
```

В agent health отсутствует monocular depth provider, поэтому за весь gate было zero Depth-Anything
loads/invocations; scheduler также сообщил `depth_extra=0`. Четыре `statefile` errors соответствовали
явно напечатанным canonical event-gap медленного read-only consumer и не затронули capture/depth authority.
После Drop `kinect-v1-status` подтвердил serial-bound `/dev/video0`, все USB functions `present`, watcher
`active`, last init `success`; stray MVM process отсутствует.

### Блок F — lossless Dataset Factory RGB-D contract

**Цель:** сохранить raw metric depth и provenance без повреждения old relative-depth sessions.

**Изменить:**

```text
mvm/crates/mvm-dataset/src/observation.rs
mvm/crates/mvm-dataset/src/writer.rs
mvm/crates/mvm-dataset/src/validate.rs
mvm/crates/mvm-cli/src/agents/dataset_writer.rs
prototype/pipeline/dataset exporters/validators where applicable
```

**Tests:** exact `u16` round-trip, invalid mask, metadata consistency, no dangling paths, old-session parse, text-only no dense payload, calibration change recorded.

**Exit gate:** `mvm dataset check` проходит на relative legacy fixture и новой metric RGB-D fixture.

**Implementation evidence (2026-08-13):** Canonical Observation schema `16.8.0` аддитивно добавляет
typed depth metadata с encoding/units/dimensions/invalid semantics, registration, RGB/depth stamps,
skew, sensor identity и calibration revision. Старые `16.7.0` строки без typed block продолжают
читаться как legacy relative depth; их `depth_path + depth_range` contract не переписан.

Metric `depth_mm` записывается напрямую в 16-bit PNG без min-max normalization и тестом декодируется
exact `u16`, включая invalid value. `session.toml` содержит observed encodings и полный sorted set
calibration revisions; validator сверяет summary с per-frame labels, rejecting mismatched units,
incomplete provenance, dangling metadata и inconsistent stamps/skew. Text-only recorder не выделяет
frame/depth paths и не сериализует dense depth metadata. Python records/validator/exporters сохраняют
typed metadata; metric loader возвращает exact `uint16 mm`, legacy relative loader — прежний `float32`.

End-to-end DatasetWriterAgent fixture провёл admitted metric evidence через `labels.jsonl`, lossless PNG,
`session.toml` и Rust checker с `0 errors / 0 warnings`. Executable
`mvm dataset check --strict` отдельно прошёл frozen relative `16.7.0` и metric `16.8.0` fixtures:

```text
legacy relative: 1 session · 0 errors · 0 warnings · OK
metric RGB-D:    1 session · 0 errors · 0 warnings · OK
```

Feature-on/default workspace tests, strict clippy, fmt/diff-check и offline Python exporter suite
зелёные. Existing repo venv использован без установки packages или network access.

**Approved tranche stop:** после F проводится отдельный review capture/data evidence. Блоки G–K не
наследуют approval блоков A–F.

**Capture/data evidence review correction (2026-08-13):** изолированный live `mvm run --record`
с production cadence записал 120 реальных RGB-D samples в две finalized sessions: 120 RGB JPEG и
120 raw metric PNG (`640×480 uint16`, `millimetres`, `registered_to_rgb`) с единой calibration revision
`0571de…ce1c`. Максимальный durable skew составил `17.991 ms` при admission limit `20 ms`; Rust
`dataset check --strict` и Python pixel validator завершились с `0 errors / 0 warnings`. Независимый
pixel пересчёт подтвердил `valid_ratio` с max delta `5.84e-8`; decode/missing-artifact failures = 0.
Capture, Dataset и остальные agents завершили bounded runs без errors, shutdown/reconnect прошёл.

Review также устранил обнаруженный durability gap: CLI теперь кодирует RGB/depth во временные sibling
files, атомарно переименовывает каждый artifact и только после успешного commit всего набора разрешает
`DatasetWriter` append в `labels.jsonl`, накопление stats и advance sequence. Ошибка может оставить
только unreferenced artifact, но не durable label с отсутствующим pixel path. Regression test инъецирует
pixel failure и доказывает отсутствие label/seq commit; success test проверяет оба artifacts и отсутствие
`.tmp` residue. Python metric validator получил отсутствовавший NumPy import и отдельный `uint16` decode
regression test; тот же validator реально прошёл сохранённые live sessions.

### Блок G — geometry primitives ✅

**Цель:** реализовать pure arithmetic over depth без canonical promotion.

**Изменить:**

```text
mvm/crates/mvm-depth/src/metric.rs
mvm/crates/mvm-depth/src/unproject.rs
mvm/crates/mvm-depth/src/roi.rs
mvm/crates/mvm-depth/src/plane.rs
mvm/crates/mvm-depth/src/visibility.rs
```

Точные module names могут меняться, responsibilities — нет.

**Tests:** synthetic plane/box/holes, coordinate signs/units, mask vs bbox bleed, invalid propagation, transform revision, deterministic robust fit, bounded allocation.

**Exit gate:** geometry measurements воспроизводимы на replay и не публикуют semantic/canonical claims.

**Implementation evidence (2026-08-13):** `mvm-depth` получил image-free pure geometry modules:
validated explicit `CameraIntrinsics`, project/unproject с millimetres→metres и calibration revision,
right-handed camera→local-world transform с transform revision, bounded ROI/mask sampling, invalid-depth
propagation, robust median/MAD/trimmed centroid/AABB/nearest/farthest/boundary measurements,
deterministic bounded plane fit с inlier coverage/RMSE/p95 и typed ray visibility. Sampling и plane input
жёстко ограничены `4096` points; empty/insufficient evidence возвращает typed error, не `[0,0,0]`.

Synthetic tests закрывают signs/units и project round-trip, holes/outlier plane, mask-vs-bbox background,
invalid pixels, dimension/calibration mismatch, transform revision/reflection, sampling bound и отсутствие
`free` в visibility vocabulary. Ни один G type не подключён к `CanonicalFusion`, tracker, runtime agent,
Dataset schema или Cortex. Текущий Kinect calibration profile автоматически не читается: отдельного
подтверждённого RGB-intrinsics contract для registered pixels всё ещё нет, поэтому live unprojection и
canonical entity geometry остаются выключены до отдельного camera-geometry approval.

#### Camera-geometry gate после G (2026-08-13)

Проверка pinned `libfreenect 0.7.5` и factory EEPROM tables уточнила geometry contract:
`FREENECT_DEPTH_MM` лежит в native IR/depth-camera grid, а `FREENECT_DEPTH_REGISTERED` переносит эти
samples в RGB grid нелинейной registration table и depth-dependent horizontal shift с nearest-depth
z-buffer ownership. Поэтому сохранённые depth-camera intrinsics нельзя применять непосредственно к
registered `(u, v)`.

Image-free device probe на `79 237` factory-derived correspondences показал, что наивное применение
depth intrinsics к registered pixels даёт `26.704 px RMSE` (`p95 40.460 px`, max `49.820 px`). Best-fit
simple RGB pinhole уменьшает ошибку до `2.487 px RMSE`, но всё равно не воспроизводит factory nonlinear
registration и не принимается как authoritative calibration.

Отдельный bounded live probe прочитал `60` кадров `FREENECT_DEPTH_MM`, не сохраняя RGB/depth images, и
воспроизвёл factory registration вместе с source ownership. Получено `15 157 372` valid native samples;
`99.857%` попали в registered bounds, `15.119%` mapped attempts столкнулись с уже занятым target pixel,
после nearest-depth z-buffer осталось в среднем `214 121` registered samples/кадр (`69.701%` grid).
Для `41 861` sampled retained owners native-camera XYZ был finite, depth mismatch = `0`.

Обратный поиск только по доступному registered tuple `(u, v, z)` недостаточен для exact native ray:
в первом real frame из `199 740` retained targets только `166 776` имели единственного native candidate,
а `32 964` (`16.503%`) — от двух до четырёх; истинный z-buffer owner не отсутствовал ни в одном candidate
set, но после выдачи только registered depth его identity потеряна. Следовательно, H нельзя честно
реализовать поверх текущего `FREENECT_DEPTH_REGISTERED` callback посредством одних intrinsics или
inverse lookup.

**Требуется отдельное решение перед H:** рекомендуемый exact path — захватывать native metric depth и
внутри Kinect backend выполнять ту же pinned factory registration, одновременно публикуя bounded
registered-depth source-owner map для unprojection. Альтернативы требуют отдельного approval: публиковать
approximate inverse с измеренной ambiguity/uncertainty либо выполнять физическую RGB calibration и
принимать её residual model. До выбора профиль calibration не дополняется недоказанными RGB intrinsics,
live unprojection остаётся выключен, H не начат.

### Блок G.1 — exact RGB-depth association и strict pairing ✅

**Цель:** сохранить factory-exact связь каждого принятого registered depth sample с native depth ray и
гарантировать, что Kinect frames, видимые vision experts, всегда являются admitted RGB-D pairs.

**Approved implementation:**

- backend принимает `FREENECT_DEPTH_MM`, а не готовый `FREENECT_DEPTH_REGISTERED`;
- один раз копирует calibration/registration tables из pinned `libfreenect 0.7.5`;
- pure Rust registration воспроизводит pinned depth-to-RGB mapping и nearest-depth z-buffer;
- одновременно формируется immutable owner map `registered RGB pixel → native depth pixel`;
- registered metric depth остаётся `640×480 u16 mm`; owner map runtime-only и не становится новым
  durable dataset artifact;
- exact XYZ вычисляется только через owner pixel и подтверждённые native depth intrinsics;
- Kinect expert admission становится strict: packet без depth pair в `max_pair_skew` не публикуется;
  unmatched RGB/depth учитываются и удаляются bounded policy, diagnostic health остаётся явным;
- default V4L2/RGB-only sources не меняют поведение;
- никаких entity/canonical geometry facts и никакого H в этом блоке.

**Tests/gates:** synthetic collisions/out-of-bounds/nearest-depth ownership; C/libfreenect oracle parity;
owner/depth consistency и exact native unprojection; replay serialization сохраняет registered depth без
runtime owner map; strict admission не выпускает RGB-only Kinect frame; bounded queues/drop accounting;
feature-on/default workspace tests, strict clippy/fmt; bounded live overlay/owner gate и 10-minute pairing
gate перед закрытием блока.

**Exit gate:** каждый expert-visible Kinect packet содержит RGB, registered metric depth, source-owner
association, обе stamps/skew и calibration revision; invalid/ambiguous evidence остаётся unknown. H
начинается только после отдельного `go`.

**Implementation evidence (2026-08-13):** backend переключён на `FREENECT_DEPTH_MM`, при open один раз
копирует factory `registration_table` и `depth_to_rgb_shift` из pinned runtime, а затем pure Rust mapper
строит `640×480 u16 mm` registered map и параллельный runtime-only `RegisteredDepthOwners`. Mapping
сохраняет nearest-depth z-buffer semantics, отвергает invalid/out-of-bounds samples и не сериализует owner
map в `MetricDepthFrame`/Dataset. `mvm-depth` получил fail-closed exact unprojection: depth берётся в RGB
grid, ray — только из retained native owner с matching dimensions/calibration revision.

Kinect source теперь strict-paired: просроченный RGB учитывается в `rgb_only`, но не публикуется в mesh;
каждый возвращённый `CapturedSensorFrame` имеет `DepthStatus::Paired`, metric depth и owner association.
V4L2/RGB-only path не изменён. Synthetic tests закрывают collisions, nearest owner, invalid/out-of-bounds,
wait-for-future-depth, expired-RGB drop, exact native-ray unprojection и ownerless fail-closed. C oracle на
том же collision fixture и Rust mapper согласовались: target `12810`, depth `900`, owner `1`, collisions
`1`, out-of-bounds `2`. Feature-on workspace tests и strict clippy/fmt/diff gates зелёные.

Sensor-only smoke `300/300` дал `29.909 FPS`, max skew `15.792 ms`; для `64 369 815` valid registered
samples missing owner = `0`, owner on invalid = `0`. Финальный 10-minute gate:

```text
frames=18000 elapsed_s=600.765 delivered_fps=29.962
paired=18000 pair_ratio=1.000000 max_skew_ms=19.999 valid_depth_mean=0.696883
health={"paired":18000,"rgb_only":19,"depth_dropped":5,"rgb_gaps":0,"depth_gaps":1,
        "duplicate_timestamps":0,"out_of_order_timestamps":0,"resync_errors":0}
owner_consistent=3853486241 owner_missing_for_valid=0 owner_present_for_invalid=0
```

То есть все `18 000` expert-visible packets были полными RGB-D pairs; 19 непарных RGB были явно
посчитаны и отброшены до публикации. После clean Drop operational status подтвердил `/dev/video0`, все
camera/audio/motor USB functions `present`, reconnect watcher `active`, last init `success`. Calibration
profile и его revision `0571de…ce1c` не менялись. Ни один owner/XYZ не повышен до canonical entity fact;
H не начат.

### Блок H — entity metric geometry in Fusion ✅

**Цель:** привязать people/objects/masks к direct metric measurements.

**Изменить:**

```text
mvm/crates/mvm-cli/src/agents/fusion.rs
mvm/crates/mvm-cli/src/pipeline.rs compatibility path
mvm/crates/mvm-core/src/canonical.rs
mvm/crates/mvm-fusion/src/
```

**Tests:** no-person object metric position; mask-aware centroid; invalid boundary confidence; person body geometry; fallback evidence remains separately stamped; reference frame serialized.

**Exit gate:** existing scene facts сохраняют backward compatibility; direct metric source явно виден в provenance.

**Implementation evidence (2026-08-13):** canonical `SceneEntity` получил additive `geometry` contract:
`direct_metric` хранит robust centroid/AABB/median-Z/MAD, sample count, valid ratio, boundary contamination,
`camera_optical` reference frame, calibration revision и признак mask-aware sampling;
`fallback_position` отдельно сохраняет прежнюю Human size-prior или relative-depth estimate. Поле имеет
serde default, поэтому старые serialized frames остаются читаемыми. Legacy `position_3d` сохранил форму и
смысл preferred position: direct centroid выбирается только при fresh exact metric evidence, иначе остаётся
прежний fallback/unknown.

Fusion выполняет bounded sampling (`stride=4`, максимум `4096` samples/entity) непосредственно над paired
registered millimetre map. Каждый sample unprojected через exact native owner ray и factory depth intrinsics,
которые backend валидирует и переносит только в runtime owner association. Objects используют instance mask,
people — сопоставленную person mask, при отсутствии mask используется bbox. Remembered objects не получают
новое direct measurement. Недостаточная выборка fail-closed; высокая invalid boundary contamination или
низкий combined confidence публикуются как `weak`, а не как точное наблюдение.

Direct facts имеют отдельный `metric_depth` source stamp/evidence; fallback Human/Depth evidence не
перемаркируется. Tests закрывают no-person object metric position, mask-aware centroid, boundary confidence
downgrade, person geometry, отдельно stamped fallback и serialized reference frame. Synchronous
`BodyPrimaryPipeline` оставлен compatibility fallback и не заявляет direct RGB-D geometry. Surface,
visibility, occupancy, temporal spatial memory и Cortex не затронуты.

Verification: default и `--features kinect-v1` workspace suites зелёные (live V4L2 test остаётся штатно
ignored); strict `cargo clippy --workspace --all-targets --features kinect-v1 -- -D warnings`, fmt и
`git diff --check` зелёные. Bounded live sensor smoke после H: `90/90` paired, `29.428 FPS`, max skew
`14.686 ms`, valid-depth mean `0.699015`; `19 326 358` valid samples имели exact owner, missing owner `0`,
owner on invalid `0`. Calibration revision осталась
`0571de941380b9e9c593abc4509d02e1a7f7085501399bf19e37ce66bc91ce1c`.

### Блок I — depth-assisted tracker comparison

**Цель:** определить минимальный полезный набор 3D cues.

**Создать:** frozen replay scenarios и evaluator adapter. Production tracker изменяется только после measured acceptance.

**Scenarios:** crossing, fast motion, enter/exit, short/long occlusion, partial body, sitting/lying, false detection, animal, object churn, depth holes, skew и packet loss.

**Exit gate:** выбран A/B/C по tracker metrics и runtime budget; rejection также является допустимым результатом.

### Блок J — surfaces, visibility и fixed-camera change measurements

**Цель:** использовать point cloud beyond entity centroids, сохраняя unknown-first semantics.

**Добавить:** planes, depth boundaries, first-surface visibility, bounded background change model, invalidation on mount/calibration change.

**Exit gate:** plane residual/coverage, visibility accuracy, dynamic-to-static leakage, bounded memory, no false-free claims.

### Блок K — IR, accelerometer и motor executor

**Цель:** подключить вторичные возможности только поверх стабильного sensor contract.

- accelerometer → gravity/pitch/roll and mount-change evidence;
- IR → separate stamped visual modality, не RGB replacement;
- motor → bounded external executor: limits, cooldown, one-shot request и explicit outcome;
- LED → optional operational status only;
- audio → не входит в этот MVM plan.

**Exit gate:** каждый канал имеет отдельные stamps/health/provenance; motor не получает прямую physical authority от Cortex.

---

## 10. Проверки и evidence classes

Нельзя подменять один вид доказательства другим:

```text
synthetic unit/contract test
frozen recorded Kinect replay
bounded live sensor smoke
controlled A/B against RGB baseline
long-run health/reconnect test
```

Общие quality gates после каждого изменяющего код блока:

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
git diff --check
```

Sensor-specific metrics:

- RGB/depth source FPS;
- pairing success ratio;
- timestamp skew distribution;
- dropped/out-of-order/resync counts;
- valid-depth ratio по range/region;
- calibration/registration status;
- capture/Fusion p50/p95;
- delivered FPS;
- CPU/RSS;
- metric plane residual;
- entity XYZ error against measured targets;
- dataset exact round-trip;
- tracker metrics;
- `unknown → free` violations = 0.

Live test не считается пройденным, если hardware test ignored или был заменён synthetic fixture.

---

## 11. Rollback и совместимость

До финального controlled A/B:

- Kinect backend выбирается явно или по строгой verified-device policy;
- V4L2 RGB path остаётся рабочим;
- `MonocularDepthProvider` сохраняется для RGB-only mode;
- new schema fields additive и optional для old readers;
- failed Kinect initialization не мутирует config/dataset;
- source/calibration mismatch fail closed;
- все runtime additions имеют clear disable switch;
- ни одна migration не переписывает старые sessions.

Rollback должен означать выбор прежнего RGB source/provider, а не удаление новых данных или откат schema history.

Первый tranche не меняет canonical entity geometry, tracker association, Phase 19 spatial substrate,
surface/visibility claims или motor state. Это позволяет откатить runtime выбором V4L2/monocular path,
не удаляя уже записанные self-describing metric RGB-D samples.

---

## 12. Definition of Done

Foundation tranche завершён только когда одновременно выполнено:

1. `mvm run` получает typed synchronized RGB-D packets от `kinect-v1-main`.
2. Каждый packet сохраняет source stamps, stream health, registration и calibration revision.
3. Fresh registered metric depth является primary depth evidence.
4. Depth-Anything не запускается в исправном Kinect mode и остаётся рабочим RGB-only fallback/comparator.
5. Metric depth доступна как отдельное stamped evidence без person anchor, но direct canonical geometry
   остаётся за следующим approved slice.
6. Invalid/stale/unregistered depth никогда не превращается в metric RGB association или `free`.
7. Dataset Factory losslessly записывает metric `u16` depth, invalid semantics и provenance; legacy relative-depth sessions читаются.
8. Dense XYZ/point cloud не является runtime или Dataset requirement первого tranche.
9. Existing tracker остаётся неизменённым authority; 3D cues не включены без replay comparison и нового approval.
10. V4L2 RGB-only path, StateFile, Fusion, Dataset Factory и Cortex не имеют необъяснённой регрессии.
11. Workspace format/test/clippy gates зелёные.
12. Три bounded starts, 10-минутный combined run и reconnect/shutdown validation реально выполнены и сохранены как evidence.

---

## 13. Рекомендуемый порядок начала работы

Первый approved implementation tranche ограничен фундаментом:

```text
A. image-free sensor contracts ✅
→ B. typed replay envelope ✅
→ C. generic SensorFrameBundle/source seam ✅
→ D. native Kinect capture ✅
→ E. metric depth authority ✅
→ F. lossless Dataset Factory contract ✅
```

После D немедленно выполняются bounded live capture gates и записываются frozen replay fixtures. После F
выполняется отдельный review качества capture и данных. Geometry/Fusion promotion, tracker cues, surfaces,
visibility/background model, IR/accelerometer interpretation и motor executor строятся только на реально
измеренных RGB-D свойствах этого сенсора и требуют отдельного approval.
