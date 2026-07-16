# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*
*Это source-of-truth по Rust-части. Детальные планы закрытых фаз — в `mvm/docs/phase-*.md` и `~/wiki/mvm/`.*
*Как LLM-агенту пользоваться MVM (pull-модель, чтение выходов, честность) — `mvm/AGENT.md`.*

---

## Статус

- **Версия `v0.2.0` подготовлена (ветка `kaggle`, порт 15.6 P0–P8 закрыт по коду)** — `Cargo.toml`
  bump 0.1.0→0.2.0, паспорт/CHANGELOG/контракт-изменения записаны; **релиз ждёт owner-пуша тега `v0.2.0`**
  (CI триггерится тегом, версия из тега). Опубликованный прод пока `v0.1.0`.
- **Ветка `master`** (продакшн после слияния `rust-core`); порт 15.6 идёт на `kaggle`. **12 крейтов, 608 тестов**
  (1 ignored — live-камера).
- **Фаза 14 «Продукт» — ЗАКРЫТА целиком** (2026-07-06, live-провалидирована с нуля: `git clone` → `cargo build`
  → `mvm setup` 9/9 → `serve --with-camera` → MCP-снапшот отдал живой срез). Оболочка «поставил → само нашлось
  и заработало»: XDG-пути + `config.toml`, авто-детект камеры, `mvm setup` (fetch моделей + sha256-verify),
  watched-галерея (`gallery/<Имя>/`), `serve --with-camera` (продюсер+MCP), `mvm mcp install`, `mvm service`
  (systemd user), дистрибуция через GitHub Releases.
- **Что живо:** `mvm run` (Agent-Mesh, чистый Rust, ~16 fps с человеком) пишет метрич. 3D scene-graph +
  `snapshot`/`recap`/`journal` в `~/.local/share/mvm`; `mvm serve --mcp` отдаёт это LLM read-only.
- **История фаз — таблица Roadmap ниже. Детальные runbook'и закрытых фаз — `mvm/docs/phase-*.md`.**
- **Владельцу (вне репо):** репойнт Sight-профиля Hermes на XDG-путь.

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
          ├─ BlazePose → 33 lm → compute_features(14) → RandomForest → posture (всегда; VAE→OOD-гейт)
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
cargo build --release          # → target/release/mvm (~29 МБ; embed только мелкие YuNet/BlazePose/RF,
                               #   крупные ONNX грузятся из models-dir — ./models в dev, [models].dir/mvm setup)
cargo test --workspace         # 597 тестов
cargo xtask bundle             # → dist/mvm

# Agent Mesh (фаза 8, основной путь) — Capture→{Human,Scene,Depth}→Fusion→StateFileWriter на std::thread
# Фаза 14.0: камера авто-детект, state+логи+journal → ~/.local/share/mvm (XDG), конфиг ~/.config/mvm/config.toml
cargo run --release -- run --dump-json                       # живой (камера + все модели); `mesh` = алиас
cargo run --release -- run --stub --max-frames 100           # model-free CI-smoke

# Синхронные пути (живы)
cargo run --release -- --native-camera --once --dump-interval 1   # pure-Rust hot-path (без mesh)
cargo run --release -- --native-camera --dump-interval 3 --scene-debug  # дебаг: детекторы цветом → /tmp/mvm_scene.jpg
cargo run --release -- --native --once --dump-interval 1          # Python bridge (поза от MediaPipe)
```

**Камера:** `--native-camera`/`run` = прямой v4l2 MJPG mmap через шов `CameraSource` (фаза 14.0.3:
`device="auto"` — перебор `/dev/video*`); bridge = `cv2.CAP_FFMPEG` (обход PipeWire на Ubuntu 24+).
`fps` в StateFile = темп **доставки кадров** (прирост `frame_id`), **не** темп обработки человеко-цепочки.

---

## Крейты (12)

| Крейт | Содержимое |
|-------|------------|
| **mvm-core** | типы, Config, PersonTracker (Kalman 4×скаляр + identity/cosine re-link + reserved-id), Pinhole/space (+`ScaleCalibrator` 11.D1), EMOTION fuse (PoE), **WorldState** (блэкборд, 8.A/8.B), `assignment` (Hungarian/Kuhn–Munkres, 9.10), `activity` (FSM+journal 11.C), `anomaly` (v2 scorer 11.B2), `mask` (`InstanceMask` 11.A1) |
| **mvm-detect** | YuNet ONNX (228KB, embed, multi-stride NMS) — лицо |
| **mvm-identity** | SFace ONNX (37MB, 128d L2-эмбеддинг, gallery, `embed()`) |
| **mvm-emotion** | EmotiEffLib B0 ONNX (16MB, 8-class VGAF), EmotionResult + entropy |
| **mvm-posture** | нативная BlazePose two-stage (детектор+поза) + **feature→RandomForest** классификатор (GRIP, 3 класса) + VAE (OOD-only); `compute_features`/`pose_from_face`, `NativePose` мульти-persona |
| **mvm-fusion** | `build_scene` (union-find граф+зоны) + `fuse()` PoE; activities + anomaly_score (8.F); `ZoneTracker` (persistence+семантика зон, 11.B1) |
| **mvm-scene** | YOLO26n ONNX (9.5МБ, end-to-end NMS-free), object-узлы (id≥10000) + `ObjectTracker` [8.E] + YOLO-seg (`--masks`, 10.4) + **object-permanence** статичных фикстур под окклюзией руками (`STATIC_CLASSES`/`static_hold`, 11) |
| **mvm-depth** | Depth-Anything-V2-Small ONNX (квант 27МБ, ~230мс CPU), depth-anchoring к торс-Z [8.E.2] |
| **mvm-hands** | MediaPipe handpose ONNX (4МБ), 21 lm/рука, ROI из запястий BlazePose [8.E.4] |
| **mvm-mesh** | Agent trait + `LatestFrame` (single-slot latest-wins) + `AgentMesh` оркестратор на std::thread [8.C] |
| **mvm-pipeline** | Camera (Python bridge subprocess) + NativeCamera (v4l2 MJPG) |
| **mvm-cli** | main + подкоманды `run`/`snapshot`/`recap`/`inspect`/`serve`/`setup`/`doctor`; `pipeline.rs` (BodyPrimaryPipeline), `agents/`, `state.rs`, product-слой (`paths.rs`/`config_file.rs`/`manifest.rs`/`setup.rs`/`doctor.rs`), `scene_debug.rs` |

> Модели ONNX (YuNet/SFace/EmotiEffLib/BlazePose/YOLO26n/Depth-Anything/handpose) — единый **манифест**
> `mvm-cli/src/manifest.rs` (URL+sha256+size, проверены 2026-07-06). `mvm setup` тянет их в models-dir
> с sha256-verify + HTTP range-resume (`setup.rs`, deps `ureq`+`sha2`, product-слой); 7/9 из публичных
> источников (OpenCV Zoo/HuggingFace), 2 YOLO — project-hosted (без публичного URL, добавляются вручную).
> `mvm run` дотягивает обязательные модели на первом запуске. `include_bytes!`-embed (bundle) — только
> мелкие YuNet/BlazePose/RF.

---

## Agent Mesh (фаза 8.C–8.D)

`mvm run` (алиас `mesh`) гоняет цепочку на `std::thread` (ноль tokio — contention опровергнут спайком,
эксперты ~12мс): `Capture → {Human, Scene, Depth} → Fusion → StateFileWriter`.

- **HOL-блокинг снят:** depth (~230мс) и YOLO на своих cadence/потоках не стопорят человеко-цепочку.
- **HumanAgent** владеет human-only pipeline + своим `PersonTracker`. **Depth** публикует в `DepthSlot`
  (slot в cli, не в mesh → mesh detector-agnostic). **FusionAgent** видит людей+объекты+depth, делает
  depth-anchoring. **StateFileWriter** сериализует `WorldState` → StateFile + Scene JSON.
- **FPS (честно, человек в кадре):** движок ~16-17 fps (mesh) vs ~14 синхронно; доставка ~25 fps.

---

## Рабочие факты и константы

- **⚠️ Камера — eye-level (~175см, ~горизонтальна, `rotate_180=false`), НЕ потолок/45°.** Симптомы поз/эмоций
  — не от ракурса (прежняя «потолочная» формулировка в доках была ошибочной). Bridge всё ещё крутит legacy
  `ROTATE_180` → на upright-камере перевернёт кадр (хвост bridge-пути).
- **Поза (GRIP):** 14 gravity-referenced инвариантных фич (`compute_features`) → RandomForest
  (`pose_rf_forest.bin` 873КБ, soft-vote) → 3 класса `standing/sitting/lying`; VAE понижен до OOD-гейта.
  **Occlusion-honesty:** при видимой только верхней части тела (`BodyCoverage::Upper`) поза **withheld** (не
  угадывается по галлюцинированным ногам); `lying`+upright физически невозможен → отвергается. `mvm-posture/classifier.rs`.
- **Tier-0 сглаживание позы/эмоции:** `PostureStabilizer` (dwell+conf-гейт), VAE OOD-гейт
  (`POSTURE_OOD_LOGPZ=170`), emotion entropy-гейт (≥0.90 подавляет near-chance метку). Чистая арифметика.
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
| 1 | Эмоция near-chance — **КОРЕНЬ НАЙДЕН, ЗАКРЫТ И LIVE-ВАЛИДИРОВАН (10.1)**: препроцесс писал HWC-interleaved в NCHW-тензор → модель видела скремблированные каналы → near-uniform softmax на любом лице (жило с порта фазы 6). Фикс planar-CHW + regression-тест на layout; вдобавок кроп выравнивается по 5 лендмаркам YuNet (Umeyama→ArcFace-шаблон). **Live (mesh, свет, 1 чел., 2026-07-02): Neutral conf 0.82–0.93, энтропия H=0.12–0.24** (было ≈0.96, peak 0.16–0.22) — метка проходит гейт 0.90, distribution резко пиковое (0.93 на Neutral). Прежний симптом «47/48 подавлено» снят по корню | ✅ (live-validated) |
| 2 | Поза: старый `standing↔sitting_at_desk` флип **закрыт по корню** (GRIP, `9308783c`). Остаточный desk-гэп (публичная модель читала desk-sitting как `lying`, ноги под столом) **закрыт архитектурно, occlusion-aware, БЕЗ файн-тюна**: `BodyCoverage`+2D-attitude → метка withheld в Upper-режиме, `lying`+upright отвергается, fusion даёт `seated_at_workstation` реляционно; оракул переписан и включён. | ✅ (occlusion-aware; файн-тюн отменён как ненужный) |
| 3 | `torso_length_m=0.5` не откалиброван рулеткой (абсолют 3D приблизителен) | 🟡 |
| 4 | `fps` в StateFile = темп доставки, не обработки | 🟡 (семантика) |
| 5 | Hand-ROI из запястья требует чёткого жеста — **закрыт (10.2)**: vis-floor 0.5→0.2, score честно downweight'ится ×(vis/0.5), presence-гейт модели сохранён; кисти за столом/у края кадра возвращаются в поток | ✅ |
| 6 | Объекты видно с ракурса человеческого роста | 🟡 |
| — | Ре-валидация стабилизации 8.0.4 на 3 людях; 8.E.3 audio (YAMNet) — новая dep против минимализма, вне ядра | 🔜 опц. |

### Ledger отложенных сигналов (10.5 — явная диспозиция, чтобы «не забыли»)

| Сигнал | Решение | Почему |
|--------|---------|--------|
| **Audio / YAMNet (G₄)** | **deferred** (Фаза 11+, опт-ин модуль) | новая capture-dep + микрофон ломают +0-dep/CPU/$0; ортогонально зрению. `AudioEvent` остаётся стабом контракта в `world.rs` |
| **Age / gender / clothing** | **out** | каждый +1 ONNX без текущего use-case — минимализм |
| **Temporal journal / event log** (длительности поз, start/stop активностей) | **Фаза 11** (продукт) | не producer-работа; строится над Scene JSON |
| **Richer anomaly, зоны/комнаты** | **Fusion-polish фаза** (после 10) | реляционная работа, не producer |
| **3D metric calibration** (`torso_length_m`, fx) | **Фаза 11 auto-calib** | ординал уже ок; абсолют — продуктовая калибровка |
| **Сырые 33 body-lm в JSON** | опт-ин расширение `--dump-landmarks` при необходимости | лендмарки кистей уже за флагом; 33-lm тела на PersonState не хранятся — добавлять только под реальный use-case |

---

## TDD-правила

- TDD обязателен: сначала падающий тест, потом реализация.
- `cargo test --workspace` остаётся зелёным после каждого изменения; коммит после каждой задачи.
- `test_config()`: `Config { models_path: PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")), ..Default::default() }`.

---

## Roadmap

История фаз одним взглядом; детальные runbook'и закрытых фаз — `mvm/docs/phase-*.md`, разбор — `~/wiki/mvm/`.

| Фаза | Статус | Суть |
|------|--------|------|
| **5** | ✅ | Стабилизация (YuNet preprocessing, emotion fallback, галерея, FFMPEG backend) |
| **6** | ✅ | Metric 3D Scene-Graph (Kalman bbox, pinhole-3D, entropy, `build_scene` + PoE), +0 deps |
| **7** | ✅ | Python-light: hot-path `--native-camera` = pure-Rust (v4l2→YuNet→ONNX), ~25 fps; SFace-identity |
| **7.5** | ✅ | Нативная поза: BlazePose+VAE порт (bit-close к Python), мульти-persona |
| **8** | ✅ | Body-primary + Agent-Mesh (`mesh`) + fusion (activities/anomaly), live 0/1/2 чел. |
| **9** | ✅ | Tracker: кинематика + ByteTrack/OCM + lifecycle + cadenced SFace/emotion + Hungarian assignment, +0 ONNX |
| **10** | ✅ | Full Data Collection (SUPREME-validated, 3 чел.+собака) + сегментация YOLO-seg (`--masks`, +1 ONNX) |
| **11** | 🔨 | Fusion-полировка (+0 ONNX): mask-контакт, activity-FSM+journal, зоны-persistence, anomaly v2, self-calib, object-permanence, person-маски |
| **12** | ✅ | LLM-render (+0 ONNX/+0 deps): pull-модель snapshot/recap/inspect, токен-диета, MCP stdio-сервер |
| **13** | ✅ | Agent Tools/Skills Polishing: когерентное CLI-дерево + `--help`, `mvm/AGENT.md`, само-докум. MCP |
| **14** | ✅ | Продукт: XDG/config, `setup`/`doctor`, watched-галерея, `serve --with-camera`, `mcp install`, `service`, дистрибуция v0.1.0 |
| **15** | 🔨 | ML Refactor → v0.2.0 (ветка `kaggle`, lab-first): **15.0 ✅** лаборатория+бенч-протокол (LJ0), **15.1 ✅** поза (2D-GRIP z-free, mocap как train-актив, flexion, Mahalanobis↔VAE; footprint −79%, LJ1), **15.2 ✅** переходы (матрица переходов CMU; prior vs dwell — агрегатный wash не подтверждён, ships гибрид dwell+SL-вето → ложный fall-флип в 0 под стрессом; E2 fall-правило «пол+скорость таза ≳1.1 м/с» AUROC 1.000; LJ2), **15.3 ✅** трекер (world-translation CMU-траектории, Python-порт `associate_core`; walking-фрагментация воспроизведена, корень — структурная re-link dead-zone устаревшей identity = 57%; фикс `center-relink через stale identity` established med Δ−4.70/heavy Δ−8.25, idle 0/0; Kalman-Q-калибровка опровергнута → gate/Kalman без изменений; LJ3), **15.4 ✅** симулятор golden-сценариев (28 CMU-инстансов G1–G6, инжекция после детекторов; FLOOR event-P/R 1.000/1.000; стресс-фрагментацию рвёт FSM exit-dwell → sweep 5/8→12/15 P 0.133→0.667; fall-правило: прод per-id = 3/10 FA + пропуски от id-churn → ships E2 scene-level @≈1.05 15fps = 0 FA + 6/6; VRD-аудит: typing/using_mouse/watching_tv = 0 инстансов → словарь v2 `using_object(class)`; CI-гейт `agents::golden` 597+2 зелёные; LJ4), **15.5 ✅** паспорт внешних экспертов + диета (человеко-цепочка YuNet+SFace+emotion+BlazePose-пара = 16.7 мс all-cores <50 PASS; ONNX 121.7 MiB, доминант SFace 36.9; emotion RAF-DB held-out 0.665 balanced healthy (neg-control взят, баг 10.1 не воспроизводится); ⭐ находка: прод шлёт в SFace НЕвыровненный bbox-кроп (`pipeline.rs:470`) — LFW unaligned 0.7833 vs aligned 0.9783 = align-фикс +19.5 п.п. за +0 вес/latency → топ-порт 15.6; detect кросс-домен OI: YuNet face 0.805, YOLO26n объект-ядро 0.6–0.8; диета: SFace INT8 измеренный НЕГАТИВ (acc lossless, latency ×6) → REJECT, VAE DROP + forest 220КБ = арифметика 1068→222 КБ (−79%), все ONNX KEEP; гейт вес/latency ≤ v0.1.0 MET; LJ5). **15.6 ✅ по коду** — Rust-порт победивших решений (P0 parity-фикстуры + maha-order · P1 поза 2D-GRIP-12/forest-v2/Mahalanobis↔VAE/flexion · P2 tracker stale-relink · P3 SL-вето · P4 E2 scene-level fall + `camera_height_m` · P5 SFace align-fikс · P6 словарь v2 `using_object(class)`/drop watching_tv · P7 FSM exit-dwell sweep = defer · P8 bump 0.2.0 + паспорт/CHANGELOG/контракт-изменения); **608 тестов**; релиз v0.2.0 ждёт owner-пуша тега. План — `docs/phase-15-plan.md`, runbook порта — `docs/phase-15.6-port-plan.md` |

## Терминология

- **Микро-агент** — независимый детектор с confidence. **Fusion** — слой, собирающий сигналы в граф/описание.
- **Workspace 3D** — метрическая система координат (метры, origin = камера).
- **Confidence-first** — каждый сигнал несёт числовую уверенность. **Prediction error** —
  `||predicted − observed||` → естественный confidence-сигнал.
- **Body-primary** — несущая сущность цепочки = тело (не лицо). **Reserved-id** — id 0–9 за галереей, ≥10 анонимам.
