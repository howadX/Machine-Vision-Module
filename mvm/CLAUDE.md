# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*
*Это source-of-truth по Rust-части. Детальные планы закрытых фаз — в `mvm/docs/phase-*.md` и `~/wiki/mvm/`.*

---

## Статус (2026-07-01)

- **Ветка:** `rust-core` (основная). **12 крейтов, 361 тест зелёный**, release чист.
- **Фаза 8 ЯДРО ЗАКРЫТО** (body-primary + Agent-Mesh + fusion), live-validated на 0/1/2 людях.
- **Фаза 9 — Tracker: 9.1–9.10 ЗАКРЫТЫ** (`+0 ONNX`) — **фаза завершена**. Закрыто:
  кинематика B1 (motion ring + `MotionClassifier` + жесты) → OCM/OCR (9.2) → lifecycle FSM (9.3) →
  ByteTrack two-stage по телам (9.4) → инверсия контракта `associate/apply_detection/finalize` (9.5) →
  cadenced SFace + identity-lock (9.6) → emotion cadence (9.7) → лендмарки на треке + motion в Scene JSON
  (9.8) → pose-ROI recovery для locked-треков (9.9) → **Hungarian global assignment (9.10)**: жадный
  порядко-зависимый IoU-матчинг → оптимальный Kuhn–Munkres (`assignment.rs`, zero-deps) над той же
  IoU+OCM cost-матрицей. План — `docs/phase-9-plan.md`. **Фаза 10 — Segmentation** (+1 ONNX YOLO-seg)
  вынесена отдельно — `docs/phase-10-plan.md`.
- **Live-ревалидация 9 (1 чел., камера eye-level ~175см, ~горизонтальна):** idle/сидя — `id=0` lock 100%, motion idle↔walking
  + velocity в JSON. Ходьба — `id=0` держится 91% кадров (9.9), но детектор фрагментирует тело → краткие
  анон-мелькания остаются (**эмпирика для 10.7 person-масок**; см. память `walking-track-fragmentation`).
- **Tier 0 — robustness позы/эмоции ЗАКРЫТ** (`+0 ONNX`, чистая арифметика; коммит `1f9caead`), **live-validated**:
  0.1 `PostureStabilizer` (dwell+conf-гейт) + 0.2 OOD-гейт по VAE-`plausibility` → покадровый флип позы
  снят (live: flip-rate 2%, длинные стабильные прогоны вместо `standing↔lying_side`); 0.3 гейт эмоции по
  энтропии (≥0.90) → near-chance метка подавляется (live: 47/48 подавлено, `emotion_distribution` сохранён,
  редкий уверенный кадр проходит). Лечит **симптом** #1/#2; корни (вход EmotiEffLib / точность позы) открыты.
  Модули `mvm-core/posture.rs`, `emotion.rs::emotion_label_gated`.
- **⚠️ Камера — eye-level (~175см, ~горизонтальна, `rotate_180=false`), НЕ потолок/45°.** Прежняя «потолочная»
  формулировка в доках была ошибочной (исправлено); симптомы #1/#2 — не от ракурса. Bridge всё ещё крутит
  `ROTATE_180` (legacy) → на upright-камере перевернёт кадр (код-хвост для bridge-пути).
- Hot-path `--native-camera` — **чистый Rust** (ноль Python). Bridge (`--native`) сохранён для
  тестов/дебага (снос venv отменён решением владельца).

---

## Три инварианта (любая фича обязана им соответствовать)

1. **Минимализм / чистая арифметика.** Fusion, трекинг, 3D, uncertainty — это скалярная арифметика
   над данными, уже лежащими в памяти. **Ноль новых зависимостей** на каждую интеллектуальную фичу.
   Footprint растёт только когда добавляется новый ONNX-эксперт.
2. **CPU-first, $0.** Ни рантайм, ни модели не требуют GPU. Никаких внешних API. Цель — `<50ms` CPU.
3. **Полный нативный Rust-пайплайн.** Python-модели — прототипы/леса. Новый код не углубляет связь
   с Python в hot-path.

**Что производит MVM:** метрический 3D scene-graph каждый кадр — узлы-сущности с метрической 3D-позицией
(origin = камера, pinhole от known-size априоров, **не** depth-сеть) + рёбра-отношения. Тир — грубый
метрический (±15-30% абсолюта) + сильный относительный субстрат (робастный ординал «кто ближе» + зоны).
Fusion: реляционная (граф близости/пересечений → зоны, главная) + атрибутивная (PoE по атрибуту, частный случай).

**Почему не VLM:** ансамбль специалистов vs одна end-to-end модель. <50ms CPU / $0 / приватно /
per-signal confidence / интерпретируемо / per-agent заменяемо. Покрывает ~80% сценариев (присутствие,
эмоция, поза, позиция) при 0.1% стоимости VLM; открытый мир / zero-shot / QA → VLM как fallback.

---

## Живая цепочка (`--native-camera`, тело-первичная)

```
frame → person_detection (ТЕЛО, первичный) → сущность/трек = ТЕЛО (NativePose::bodies)
          ├─ BlazePose → 33 lm → PoseVAE → posture                 (всегда)
          ├─ торс-3D (плечи 11/12 → бёдра 23/24, torso_to_3d)       (всегда; gated на visibility ≥0.5)
          ├─ кисти: ROI из запястий BlazePose (15/16) → handpose 21 lm/рука (presence-gate)
          └─ YuNet (ЛИЦО) по кадру, матч к телу по НОСУ (lm[0] в face-bbox)
                есть лицо → SFace id + EmotiEffLib emotion;  нет → identity/emotion = None
        + объекты (YOLO26n, cadenced) + глубина (Depth-Anything, cadenced) → метрич. 3D объектов
        tracker по bbox ТЕЛА (Kalman + IoU + cosine/label/center re-link) → build_scene → StateFile/JSON
```

**Body-primary** (FRONTIER, закрыт 8.0.2): человек без фронтального лица трекается (presence + поза +
торс-3D); identity/emotion дотягиваются при появлении лица. A1 анти-flip подавляет мигание body↔face.

---

## Запуск и команды

```bash
# Сборка / тесты / бандл
cargo build --release          # → target/release/mvm-cli (~26 МБ, модели embedded)
cargo test --workspace         # 248 тестов
cargo xtask bundle             # → dist/mvm

# Agent Mesh (фаза 8, основной путь) — Capture→{Human,Scene,Depth}→Fusion→StateFileWriter на std::thread
cargo run --release -- mesh --dump-json --interval-ms 1000   # живой (камера + все модели)
cargo run --release -- mesh --stub --max-frames 100          # model-free CI-smoke

# Синхронные пути (живы)
cargo run --release -- --native-camera --once --dump-interval 1   # pure-Rust hot-path (без mesh)
cargo run --release -- --native-camera --dump-interval 3 --scene-debug  # дебаг: детекторы цветом → /tmp/mvm_scene.jpg
cargo run --release -- --native --once --dump-interval 1          # Python bridge (поза от MediaPipe)
```

**Камера:** `--native-camera` = прямой v4l2 MJPG mmap; bridge = `cv2.CAP_FFMPEG` (обход PipeWire на Ubuntu 24+).
`fps` в StateFile = темп **доставки кадров** (прирост `frame_id`), **не** темп обработки человеко-цепочки.

---

## Крейты (12)

| Крейт | Содержимое |
|-------|------------|
| **mvm-core** | типы, Config, PersonTracker (Kalman 4×скаляр + identity/cosine re-link + reserved-id), Pinhole/space, EMOTION fuse (PoE), **WorldState** (блэкборд, 8.A/8.B), `assignment` (Hungarian/Kuhn–Munkres, 9.10) |
| **mvm-detect** | YuNet ONNX (228KB, embed, multi-stride NMS) — лицо |
| **mvm-identity** | SFace ONNX (37MB, 128d L2-эмбеддинг, gallery, `embed()`) |
| **mvm-emotion** | EmotiEffLib B0 ONNX (16MB, 8-class VGAF), EmotionResult + entropy |
| **mvm-posture** | нативная BlazePose two-stage (детектор+поза) + PoseVAE/Classifier (порт, чистая арифметика); `NativePose` мульти-persona |
| **mvm-fusion** | `build_scene` (union-find граф+зоны) + `fuse()` PoE; activities + anomaly_score (8.F) |
| **mvm-scene** | YOLO26n ONNX (9.5МБ, end-to-end NMS-free), object-узлы (id≥10000) + `ObjectTracker` [8.E] |
| **mvm-depth** | Depth-Anything-V2-Small ONNX (квант 27МБ, ~230мс CPU), depth-anchoring к торс-Z [8.E.2] |
| **mvm-hands** | MediaPipe handpose ONNX (4МБ), 21 lm/рука, ROI из запястий BlazePose [8.E.4] |
| **mvm-mesh** | Agent trait + `LatestFrame` (single-slot latest-wins) + `AgentMesh` оркестратор на std::thread [8.C] |
| **mvm-pipeline** | Camera (Python bridge subprocess) + NativeCamera (v4l2 MJPG) |
| **mvm-cli** | main + `mesh`-подкоманда; `pipeline.rs` (BodyPrimaryPipeline), `agents/` (Capture/Human/Scene/Depth/Fusion/StateFileWriter), `state.rs`, `scene_debug.rs` |

> Модели ONNX (YuNet/SFace/EmotiEffLib/BlazePose/YOLO26n/Depth-Anything/handpose) лежат в `mvm/models/`,
> крупные — gitignored с URL (download-separately). `include_bytes!`-embed грузит их в release + bundle.

---

## Agent Mesh (фаза 8.C–8.D)

`mvm-cli mesh` гоняет цепочку на `std::thread` (ноль tokio — contention опровергнут спайком,
эксперты ~12мс): `Capture → {Human, Scene, Depth} → Fusion → StateFileWriter`.

- **HOL-блокинг снят:** depth (~230мс) и YOLO на своих cadence/потоках не стопорят человеко-цепочку.
- **HumanAgent** владеет human-only pipeline + своим `PersonTracker`. **Depth** публикует в `DepthSlot`
  (slot в cli, не в mesh → mesh detector-agnostic). **FusionAgent** видит людей+объекты+depth, делает
  depth-anchoring. **StateFileWriter** сериализует `WorldState` → StateFile + Scene JSON.
- **FPS (честно, человек в кадре):** движок ~16-17 fps (mesh) vs ~14 синхронно; доставка ~25 fps.

---

## Рабочие факты и константы

- **Identity-модель (reserved-id):** id **0–9** — за пользователями галереи (Вадим=0, липко на присутствие);
  анонимы — эфемерные id **≥10** (переиспользуются при выходе). **Эксклюзивно** (один user-id — одному телу).
  Наружу: `PersonTrack.display_id` + `shown_identity()`. margin-identity отсеивает чужого (#5).
- **3D:** торс-pinhole (`space::torso_to_3d`, `Config.torso_length_m=0.5` — **не откалиброван рулеткой**,
  ординал ок, абсолют приблизителен). Лицо-bbox 3D: `face_height_m_yunet=0.32` (калибр под YuNet-боксы).
- **Cadence:** `object_cadence=3` (YOLO); depth cadence ~30. Object-фильтры: A score≥0.45, B стабильный
  `ObjectTracker` (IoU+free-list, `age`), C allow-list классов (person исключён → G₁), D min bbox area 0.004.
- **Взаимодействие человек↔объект** — по **body-landmarks** (запястье/кисть ∈ bbox объекта), не по body-bbox.
- **Depth-anchoring:** относит. inverse-depth привязывается к метрич. торс-Z (`Z·d≈k`, медиана по людям)
  → объекты получают метрич. 3D в той же системе → реальные 3D-`near` рёбра.

---

## Python bridge (родительский проект)

MVM — гибрид. Rust-крейты в `mvm/`, Python-прототипы — в родительском каталоге (доступны по `../`):

```
Machine Vision Module/           ← корень (cwd при запуске)
├── core/        face_id.py, workspace_3d.py, pose_vae.py, emotion_hse.py, cpu_face_id.py
├── tools/       mvm_bridge.py (камера → JSON lines, subprocess)
├── models/      mediapipe/*.task, emotion/*.pth, references/
├── venv/        Python 3.11+ (нужен только для bridge-пути)
└── mvm/         ← Rust workspace (этот каталог)
```

Запуск bridge напрямую: `../venv/bin/python3 ../tools/mvm_bridge.py --json-only --max-frames 1`.

---

## Известные открытые хвосты

| # | Проблема | Статус |
|---|----------|--------|
| 1 | Эмоция near-chance: распределение почти равномерно (peak ≈0.16–0.22 vs uniform 0.125), метка слабоинформативна. **Причина НЕ в ракурсе** — камера eye-level (~175см, ~горизонтальна), лица фронтальны; под подозрением вход в EmotiEffLib (выравнивание/кроп/размер лица) или подлинная неоднозначность. Tier 0.3 **подавляет** near-chance метку (энтропия ≥0.90); downstream читает `emotion_conf`/entropy/распределение | 🟡 (симптом сглажен; корень открыт) |
| 2 | Выброс позы `standing`↔`sitting_at_desk` (точность модели, #7.5.5; согласие native↔Python 5/7). **НЕ ракурс** (eye-level). Tier 0.1/0.2 **сглаживают** покадровый флип (dwell+conf+plausibility-гейт) | 🟡 (симптом сглажен; точность классификатора открыта) |
| 3 | `torso_length_m=0.5` не откалиброван рулеткой (абсолют 3D приблизителен) | 🟡 |
| 4 | `fps` в StateFile = темп доставки, не обработки | 🟡 (семантика) |
| 5 | Hand-ROI из запястья требует чёткого жеста (vis-gate 0.5: нет руки → нет контакта, консервативно) | 🟡 |
| 6 | Объекты видно с ракурса человеческого роста | 🟡 |
| — | Ре-валидация стабилизации 8.0.4 на 3 людях; 8.E.3 audio (YAMNet) — новая dep против минимализма, вне ядра | 🔜 опц. |

---

## TDD-правила

- TDD обязателен: сначала падающий тест, потом реализация.
- `cargo test --workspace` остаётся зелёным после каждого изменения; коммит после каждой задачи.
- `test_config()`: `Config { models_path: PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")), ..Default::default() }`.

---

## Roadmap

| Фаза | Статус | Суть |
|------|--------|------|
| **5** | ✅ | Стабилизация (YuNet preprocessing, emotion fallback, галерея, FFMPEG backend) |
| **6** | ✅ | Metric 3D Scene-Graph (Kalman bbox, pinhole-3D, entropy, `build_scene` + PoE) — ноль новых deps |
| **7** | ✅ | Python-light: hot-path `--native-camera` = pure-Rust (v4l2→YuNet→ONNX), ~25 fps; SFace-эмбеддинг + margin-identity + cosine-re-link |
| **7.5** | ✅ | Нативная поза: BlazePose+PoseVAE порт (bit-close к Python), мульти-persona (#3 закрыт) |
| **8** | ✅ ядро | Body-primary + Agent-Mesh (`mesh`) + fusion (activities/anomaly); G₁ Human полна (+scene/depth/hands) |
| **9** | ✅ | **Tracker** — кинематика + ByteTrack/OCM/OCR + lifecycle + инверсия + cadenced SFace/emotion + motion-эмит + pose-ROI recovery + Hungarian global assignment, **+0 ONNX** — закрыта |
| **10** | 🔜 | **Segmentation** — YOLO-seg инстанс-маски (взаимодействие), **+1 ONNX** (`docs/phase-10-plan.md`) |
| **11** | 🔜 | Продукт |

Детали закрытых фаз: `mvm/docs/phase-7.5-plan.md`, `phase-8-plan.md`; `~/wiki/mvm/roadmap.md`.

### 🟢 Фаза 9 — Tracker: body-first track-centric рефактор (9.1–9.9 закрыты, осталось 9.10)

> **Архитектура:** **Body-first tracking + Face-as-attribute + ByteTrack для робастности.** Тело видно
> всегда → трекаем **тело**, не лицо; identity/emotion — **атрибуты тела**, переживают окклюзию лица;
> SFace — для долгосрочной ре-идентификации, **не каждый кадр**. ByteTrack по телам — **страховка**
> (пересечение людей / слабый кадр детектора), не главный механизм. **+0 ONNX** — всё чистая арифметика
> и оркестрация. Детальный план — `mvm/docs/phase-9-plan.md`. Разбор — `~/wiki/machine-vision/02-tracking/`.

**Что фаза 8 уже даёт:** тело — несущая сущность; лицо матчится по носу, identity/emotion/3D — уже
атрибуты `PersonTrack`; reserved-id sticky; каскад re-link = грубый OCR. **Дельта 9 — три оси:**
(A) инверсия потока обогащения в track-gated, (B) робастность ассоциации, (C) identity lifecycle + cadenced SFace.

> **⛔ Слоистость (критично):** `PersonTracker` в `mvm-core` (нет `ort`/`image`) — трекер **НЕ вызывает
> модели**. Спина оркестрируется в `HumanAgent` (`mvm-cli`); трекер лишь **экспонирует запросы**
> («какие треки без identity / устаревший emotion / дай предсказанный ROI»).

| Этап | Задача | Реализация / коммит |
|------|--------|-----------|
| **9.1** ✅ | **Кинематика (B1):** ring 33-лендмарков (N=16) → centroid speed (torso/s) → `MotionClassifier` idle↔walking (dead-zone+dwell+гистерезис) + `ArmGesture` reach/wave по запястью | `motion.rs`; b34a7be2 / 6740f9b9 |
| **9.2** ✅ | **OCM/OCR:** `obs_velocity` (EMA, gap-norm) → momentum-терм в cost (анти-swap) + center-relink по **виртуальной траектории** | `tracker.rs`; e87f143d |
| **9.3** ✅ | **Lifecycle FSM** `Tentative→Confirmed→IdLocked→Occluded→Lost` — чистая `next_lifecycle`, аддитивно (churn идентичен) | `types.rs`+`tracker.rs`; 2d266332 |
| **9.4** ✅ | **ByteTrack two-stage:** `BodyPose.score`, `detect_scored`, `bodies_byte` (high posed / low bbox-only) → `tick_byte` low-recovery | posture+cli+core; 2db33c7c |
| **9.5** ✅ | **Инверсия контракта:** `BodyObservation` + `associate`/`apply_detection`/`finalize`; `tick_byte` = тонкий оркестратор, поведение байт-идентично | `tracker.rs`; 5afd2bec |
| **9.6** ✅ | **Cadenced SFace + lock:** `is_id_locked` (reserved/cosine-стаб); `identity_run_flags` — skip SFace только для continuation locked-трека | core+cli; 671e4221 / **05bdcddf** (рефайн) |
| **9.7** ✅ | **Emotion cadence:** `frames_since_emotion`/`needs_emotion`; лейбл заморожен между прогонами (matched → нет decay) | core+cli; 818376b8 |
| **9.8** ✅ | **Motion на треке:** `MotionTrack` (bounded ring), `tick_motion`; `motion_state`/`velocity` → `PersonState` → Scene JSON; live idle↔walking | core+cli; 7d818fc6 |
| **9.9** ✅ | **Pose-ROI recovery:** `bodies_byte_recover` — locked-трек, потерянный детектором, позируется из предсказанного ROI (torso-vis гейт). Live: id-continuity 91%, 0 фантомов на пустом | posture+cli; 326cbc19 |
| **9.10** ✅ | **Hungarian assignment:** hand-rolled Kuhn–Munkres (compact O(n³), dual-potentials, zero-deps) в новом `assignment.rs`; жадный IoU-матчинг Step 3 → глобально-оптимальный над IoU+OCM cost-матрицей (cols sorted by id → детерминизм; `IOU_GATE→∞`; каскад re-link + ByteTrack сохранены как fallback). 158 прежних тестов трекера зелёные (байт-идентичность 1–2 тел) + дискриминирующий тест оптимальности vs greedy | `assignment.rs`+`tracker.rs` |

> **Инварианты 9 (соблюдены):** **+0 ONNX**; всё — чистая арифметика + оркестрация; модели **не** въехали
> в `mvm-core`; body-primary FRONTIER сохранён. Робастность (9.2–9.4) сделана **до** инверсии (9.5).
> **Эмпирика для фазы 10 (live):** при ходьбе детектор фрагментирует тело → краткие
> анон-мелькания даже после 9.9 (recovery держит только locked) → **person-маски 10.7 оправданы**.
> Подробно — память `walking-track-fragmentation`.

### 🔜 Фаза 10 — Segmentation: инстанс-маски (взаимодействие)

> **+1 ONNX (YOLO-seg nano)** и больше ничего. Один forward-pass даёт object-маски (взаимодействие,
> instance-разделение) + опц. person-маски (ассоциация). Mask-декод/контакт/depth-по-маске — чистая
> арифметика. Cadenced на `SceneAgent` (hot-path ≈ 0). Детальный план — `mvm/docs/phase-10-plan.md`.
> **Точка входа — спайк-гейт 10.1** (go/no-go по замеру масок vs bbox + учёт эмпирики фазы 9).

| Этап | Задача |
|------|--------|
| **10.1** | **Спайк-гейт** (первый): выигрыш маски vs bbox/landmark на боевом ракурсе + частота перекрытия людей (с учётом фазы 9) → go/no-go + судьба 10.7 |
| **10.2–10.6** | **Object masks:** YOLO-seg drop-in (декод 32 прото 160²) → mask→`Entity` → `median_in_mask` depth → mask-IoU `ObjectTracker` (переиспользует Hungarian 9.10) → mask-контакт fusion |
| **10.7** | **person-mask ассоциация** (опц., gated 10.1): `person`-маска **только в cost-матрицу фазы 9** (не узел!); только если арифметика 9 не закрыла перекрытие тел |

> **❌ Отложено (фаза 11+):** ST-GCN/MPED-RNN/VideoMAE (нейро-действия). **Caveat:** nano-маски грубы на
> мелких/дальних объектах → спайк-гейт 10.1 первый. SAM/Mask R-CNN отвергнуты ($0/CPU).

### 🔜 Фаза 11 — Продукт

Авто-калибровка fx (zero-config) · drag-and-drop галерея (фото в папку = новый человек) · one-click
install (один бинарник, без pip/venv/Python) · systemd service + cron state dump.

---

## Терминология

- **Микро-агент** — независимый детектор с confidence. **Fusion** — слой, собирающий сигналы в граф/описание.
- **Workspace 3D** — метрическая система координат (метры, origin = камера).
- **Confidence-first** — каждый сигнал несёт числовую уверенность. **Prediction error** —
  `||predicted − observed||` → естественный confidence-сигнал.
- **Body-primary** — несущая сущность цепочки = тело (не лицо). **Reserved-id** — id 0–9 за галереей, ≥10 анонимам.
