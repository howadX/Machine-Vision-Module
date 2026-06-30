# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*
*Это source-of-truth по Rust-части. Детальные планы закрытых фаз — в `mvm/docs/phase-*.md` и `~/wiki/mvm/`.*

---

## Статус (2026-06-30)

- **Ветка:** `rust-core` (основная). **12 крейтов, 248 тестов зелёные**, release чист.
- **Фаза 8 ЯДРО ЗАКРЫТО** (body-primary + Agent-Mesh + fusion), live-validated на 0/1/2 людях.
- **Активная фаза — 9: точное взаимодействие с объектами (Трек A, маски) + трекинг движений Human
  (Трек B, кинематика-арифметика + OC-SORT).** Точка входа — **спайк-гейт 9.1** (go/no-go). План — `docs/phase-9-plan.md`.
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
| **mvm-core** | типы, Config, PersonTracker (Kalman 4×скаляр + identity/cosine re-link + reserved-id), Pinhole/space, EMOTION fuse (PoE), **WorldState** (блэкборд, 8.A/8.B) |
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
| 1 | Эмоция near-chance на потолочном ракурсе (лицо сверху ≠ фронтальные VGAF-кропы) → метка слабоинформативна, downstream обязан читать `emotion_conf`/entropy | 🟡 (лучше при eye-level) |
| 2 | Выброс позы `standing`↔`sitting_at_desk` на потолочном ракурсе (точность модели, #7.5.5; согласие native↔Python 5/7) | 🟡 |
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
| **9** | 🟠 активная | Segmentation + tracker upgrade (см. ниже) |
| **10** | 🔜 | Продукт |

Детали закрытых фаз: `mvm/docs/phase-7.5-plan.md`, `phase-8-plan.md`; `~/wiki/mvm/roadmap.md`.

### 🟠 Фаза 9 — точное взаимодействие с объектами + трекинг движений Human (активная)

> **Цель (две оси):** (A) **точное взаимодействие с объектами** через инстанс-сегментацию и
> (B) **трекинг движений Human**. Объединены **одной моделью**: один forward-pass **YOLO-seg** даёт
> И object-маски (Трек A), И person-маски (Трек B-ассоциация) → **+1 ONNX на обе цели**.
> **Ключ:** маски дают instance-separation и контакт, но **не кинематику** — «движение» = траектории
> лендмарков во времени = **чистая арифметика**. Детальный план — `mvm/docs/phase-9-plan.md`.
> Разбор релевантности — `~/wiki/machine-vision/03-segmentation/`, `02-tracking/`, `04-pose-action/`.

**Каким агентам нужна сегментация:** ✅ объекты (контакт, `median_in_mask`, instance-разделение);
🟡 ассоциация Human-трека (person-mask при перекрытии body-bbox). ❌ posture/hands/identity/emotion и
кинематика движения (лендмарки/арифметика, силуэт не добавляет). **Точка входа — спайк-гейт 9.1.**

**Трек A — object segmentation (взаимодействие):**

| Этап | Задача | Гейт/тесты |
|------|--------|-----------|
| **9.1** | **Спайк-гейт** (обязателен, первый): на боевом ракурсе замерить выигрыш масок vs bbox/landmark для (1) мелких настольных предметов (под вопросом — грубые nano-маски), (2) близкого контакта рука↔предмет (вероятно да); **заодно частоту перекрытия person-bbox** (вход для 9.B2b). **go/no-go по замеру** | кривая выигрыша задокументирована |
| **9.2** | YOLO-seg nano как **drop-in замена** `mvm-scene::YoloDetector` (детект+маски из одной модели; +~4МБ ONNX; ~60-90мс на потоке `SceneAgent`, cadenced). Декод: prototype-маски (32×160²) + per-instance коэффициенты → линейн. комбинация + sigmoid + crop. **Ноль новых deps** (ndarray/ort). SAM/Mask R-CNN отвергнуты ($0/CPU) | golden mask vs Python; декод-тесты |
| **9.3** | `mask` плумбинг: `TrackedObject` → `ObjectState` → `Entity` (low-res bitmask + origin) | сериализация контракта |
| **9.4** | **Depth-anchoring по маске:** `median_in_mask` вместо `median_in_bbox` (убирает фон-bleed → точнее метрич. Z) | quality-тест vs bbox |
| **9.5** | **ObjectTracker по маске:** mask-IoU association (разделение инстансов при наложении bbox) | swap-регрессия на перекрытии |
| **9.6** | **Fusion по маске:** контакт человек↔объект через mask-смежность/IoU (точнее landmark-точек) + окклюзия-порядок (кто держит / кто перед кем) | mask-контакт тесты |

**Трек B — трекинг движений Human:**

| Этап | Задача | Гейт/тесты |
|------|--------|-----------|
| **9.B1** | **Кинематика (чистая арифметика, БЕЗ сегментации):** кольцевой буфер 33-лендмарков per-track (N≈16) → скорости суставов/центроида, переходы sit↔stand / idle↔walking (порог скорости + dwell + гистерезис), жест/reach по запястью; эмит `motion_state`/`velocity`/transitions в Scene JSON. **Ноль новых deps.** ⚠️ сейчас лендмарки в трек **не** персистятся — greenfield | юнит-тесты на синтетике; тест отсутствия роста памяти |
| **9.B2a** | **OCM/OCR апгрейд `PersonTracker`** (чистая арифметика): observation-centric momentum (согласованность направления скорости в cost матчинга) + recovery через виртуальную траекторию сквозь окклюзию. Дополняет Kalman+IoU+cosine+center-relink (OC-SORT-стиль) | регрессия churn (как 8.0.3) |
| **9.B2b** | **person-mask ассоциация** (опц., gated 9.1): класс `person` в YOLO-seg **только для маски** (не узел!) → mask-IoU разделяет перекрытые тела; матч к BlazePose-телу по носу/лендмаркам | только если 9.1 показал перекрытие людей как реальный failure-mode |

> **Инварианты 9:** **+1 ONNX** (YOLO-seg) и не более; всё остальное (кинематика, OCM/OCR, mask-декод,
> контакт) — **чистая арифметика, ноль новых deps**; cadenced на потоке `SceneAgent` (hot-path = 0);
> человеко-сторона остаётся body-primary (BlazePose) — person-mask лишь усиливает ассоциацию, person
> **не** становится узлом графа. **❌ Отложено:** ST-GCN/MPED-RNN/VideoMAE (нейро-действия — ещё +N ONNX,
> вне scope; движение берём арифметикой). **Caveat:** nano-маски грубы на мелких/дальних объектах
> потолочного ракурса → выигрыш неоднороден (потому спайк-гейт 9.1 первый). Landmark-контакт (8.F) — baseline.

### 🔜 Фаза 10 — Продукт

Авто-калибровка fx (zero-config) · drag-and-drop галерея (фото в папку = новый человек) · one-click
install (один бинарник, без pip/venv/Python) · systemd service + cron state dump.

---

## Терминология

- **Микро-агент** — независимый детектор с confidence. **Fusion** — слой, собирающий сигналы в граф/описание.
- **Workspace 3D** — метрическая система координат (метры, origin = камера).
- **Confidence-first** — каждый сигнал несёт числовую уверенность. **Prediction error** —
  `||predicted − observed||` → естественный confidence-сигнал.
- **Body-primary** — несущая сущность цепочки = тело (не лицо). **Reserved-id** — id 0–9 за галереей, ≥10 анонимам.
