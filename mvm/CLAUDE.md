# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*
*Это source-of-truth по Rust-части. Детальные планы закрытых фаз — в `mvm/docs/phase-*.md` и `~/wiki/mvm/`.*
*Agent know-how: `skills/mvm-vision/SKILL.md` (pull-модель/MCP) и
`skills/mvm-dataset/SKILL.md` (запись/проверка/экспорт сессий).*

---

## Статус

- **Фаза 15 (ML Refactor) ЗАКРЫТА; Фаза 16 (Dataset Factory) 16.0–16.6-A + Фаза 16.7 (Semantic Session Log)
  закрыты и live-провалидированы (ветка `kaggle`). 14 крейтов, 717 Rust-тестов** (1 ignored — live-камера)
  + Python-группы (экспортёры + validate-гейт). **Осталось: 16.6-B** (hard-example mining, зависит от данных).
- **Фаза 16.7 — Semantic Session Log ✅:** свела два стора истории сцены в один. `mvm run` по умолчанию
  пишет **text-only семантическую сессию** (`datasets/sessions/<id>/labels.jsonl`, без пикселей — durable
  vision-лог, SCHEMA 16.7.0 с relations/activities/events); плоские `scene-*.jsonl`/`events-*.jsonl`
  ретайрнуты (legacy `--scene-log`); `--record` доращивает до пиксельного датасета; `--no-vision-log`
  выключает. `recap`/`inspect`/journal читают сессии (shim → те же ридеры). Live: 4-мин прогон, сессия
  199 записей + 45 эпизодов, `dataset check` зелёный. План — `docs/phase-16.7-plan.md`.
- **Фаза 17 — Canonical MVM Agent Skills ✅:** agent know-how разделён на canonical
  `mvm-vision` и `mvm-dataset`; stdlib-only validator и GitHub Actions защищают minimal frontmatter,
  honesty markers, stale-stack запреты и public-surface drift. Официальный `skills-ref` и fresh-agent
  forward-tests прошли; профиль Hermes `sight` читает repo skills напрямую, legacy flow архивирован.
- **Фаза 18 — Visual Cortex Layer 🚧:** 18.0 ✅ — в `mvm-core::canonical` зафиксированы
  `FusedSceneFrame`, `SceneFact<T>`, `EntityKey`+incarnation, stamped sources, event-tail cursor с
  `EventGap` и read-only compatibility projection. `benchmark/` содержит replay-контракт, шесть
  обязательных сценариев и честный pre-Cortex baseline. **18.1 ✅:** добавлен generic
  `LatestValue<T>` и stamped Human/Scene/Depth observation bus; `CanonicalFusion` публикует immutable
  `FusedFramePacket` с incarnation-safe entities, едиными relations/zones/kinematics и event tail.
  StateFile/DatasetWriter читают packet без scene mutation; session lifecycle локален DatasetWriter,
  `current_state.md` atomic. Typed activity episodes попадают в `labels.jsonl` через независимый
  event cursor с source provenance; replay/cursor/late-result gates добавлены. Корневой
  `--native-camera` маршрутизируется в canonical AgentMesh, text-only/pixel fixtures проходят
  dataset check. **18.2 ✅:** новый единый `mvm-cortex` задаёт serializable
  `CortexFrame`/`CognitiveClaim` с отдельной honesty-семантикой и пустыми typed modules;
  `CortexAgent` disabled-by-default и не подключён к production mesh, поэтому текущий runtime
  не меняется. Dataset test roots получают уникальный nonce и full workspace gate снова
  воспроизводим. Далее — 18.3 Scene Events + Cortex Interpretation. План — `docs/phase-18-plan.md`.
- **Версия — `v0.3.0` ОПУБЛИКОВАНА** (2026-07-23, Dataset Factory): `master` = `kaggle`+merge, тег `v0.3.0`
  пушнут → CI `release.yml` собрал x86_64+aarch64. Релизы: **v0.1.0** (Продукт) · **v0.3.0** (Dataset Factory);
  **v0.2.0** (ML Refactor) свёрнута в код v0.3.0 (отдельным тегом не выпускалась). CI триггерится тегом `v*`,
  версия из `Cargo.toml`. Активная разработка продолжается на `kaggle`.
- **Фаза 14 «Продукт» — ЗАКРЫТА целиком** (2026-07-06, live с нуля: `git clone` → `cargo build` →
  `mvm setup` 9/9 → `serve --with-camera` → MCP-снапшот): XDG-пути + `config.toml`, авто-детект камеры,
  `mvm setup` (fetch+sha256), watched-галерея, `serve --with-camera`, `mvm mcp install`, `mvm service`,
  дистрибуция через GitHub Releases.
- **Что живо:** `mvm run` (Agent-Mesh, чистый Rust, ~16 fps с человеком) пишет метрич. 3D scene-graph +
  `snapshot`/`recap`/`journal` в `~/.local/share/mvm` + **default-on text-only семантическую сессию**
  (16.7); `mvm serve --mcp` отдаёт это LLM read-only. **Фаза 16:** `mvm run --record` доращивает сессию до
  пиксельного датасета (RGB+depth+masks+labels, privacy-safe), `mvm dataset list/info/stats/check/prune`,
  Python-экспортёры в `prototype/pipeline/dataset/`.
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

## Живая цепочка (`mvm run`, тело-первичная)

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
cargo test --workspace         # 689 тестов
cargo xtask bundle             # → dist/mvm

# Agent Mesh (фаза 8, основной путь) — Capture→{Human,Scene,Depth}→Fusion→StateFileWriter на std::thread
# Фаза 14.0: камера авто-детект, state+логи+journal → ~/.local/share/mvm (XDG), конфиг ~/.config/mvm/config.toml
cargo run --release -- run --dump-json                       # живой (камера + все модели); `mesh` = алиас
                                                             #   ↑ по умолчанию пишет text-only сессию (16.7)
cargo run --release -- run --stub --max-frames 100           # model-free CI-smoke

# Dataset Factory (фаза 16/16.7): text-only сессия по умолчанию; --record = пиксели
cargo run --release -- run --record                          # доростить до пиксельного датасета (RGB+depth+labels)
cargo run --release -- run --no-vision-log                   # выключить запись целиком
cargo run --release -- dataset list                          # сессии на диске (info/stats/check/prune/export)
# Python-экспортёры (PyTorch/TFRecord/WebDataset) — prototype/pipeline/dataset/

# Синхронные пути (живы)
cargo run --release -- run --max-frames 1 --dump-json          # bounded canonical AgentMesh run
cargo run --release -- run --interval-ms 3000 --dump-json      # pure-Rust hot-path, ноль Python
cargo run --release -- --native --once --dump-interval 1          # Python bridge (поза от MediaPipe)
```

**Камера:** `run` = прямой v4l2 MJPG mmap через шов `CameraSource` (фаза 14.0.3;
корневой `--native-camera` теперь deprecated alias того же AgentMesh):
`device="auto"` — перебор `/dev/video*`); bridge = `cv2.CAP_FFMPEG` (обход PipeWire на Ubuntu 24+).
`fps` в StateFile = темп **доставки кадров** (прирост `frame_id`), **не** темп обработки человеко-цепочки.

---

## Крейты (13)

| Крейт | Содержимое |
|-------|------------|
| **mvm-core** | типы, Config, PersonTracker (Kalman 4×скаляр + identity/cosine re-link + reserved-id), Pinhole/space (+`ScaleCalibrator` 11.D1), EMOTION fuse (PoE), **WorldState** (legacy blackboard), `canonical` (`FusedSceneFrame`/`SceneFact`/`EntityKey`, 18.0), `assignment` (Hungarian/Kuhn–Munkres, 9.10), `activity` (FSM+journal 11.C), `anomaly` (v2 scorer 11.B2), `mask` (`InstanceMask` 11.A1) |
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
| **mvm-dataset** | ★ **Dataset Factory (фаза 16/16.7)** — чистый потребитель WorldState. `observation` (Canonical Observation SCHEMA 16.7.0, статусы `observed/derived/temporal/weak/withheld/unknown`, provenance, ключи связывания, +`activities`), `router` (Semantic Router G₁–G₅ + `route_with_relations`: relations/activities/events из WorldState+`SceneRelations`), `config` (`DatasetConfig`, `vision_log`/`effective_recorder` — 16.7 default-on text-only), `writer` (`DatasetWriter`: сессии+JSONL+session.toml, text-only-гейт), `privacy` (blur-регионы), `session` (ридеры для `mvm dataset`), **`validate` (16.6-A data-quality gate → `mvm dataset check`)**. Deps: только serde/serde_json/toml — **ноль ML-фреймворков** |
| **mvm-cortex** | ★ **Visual Cortex skeleton (18.2)** — cognitive interpretation поверх immutable `FusedSceneFrame`: `VisualCortex`, `CortexInput`, serializable `CortexFrame`, evidence-linked `CognitiveClaim`, typed empty modules (`predict/events/memory/affordance/attention/self_model/uncertainty/report`). Runtime disabled by default; no durable store and no scene mutation. |
| **mvm-cli** | main + подкоманды `run`/`snapshot`/`recap`/`inspect`/`serve`/`setup`/`doctor`/**`dataset`**; `pipeline.rs` (BodyPrimaryPipeline), `agents/` (+`dataset_writer` 16.1.2b), `dataset_cmd.rs` (16.3), `state.rs`, product-слой (`paths.rs`/`config_file.rs`/`manifest.rs`/`setup.rs`/`doctor.rs`), `scene_debug.rs` |

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

**Открытые (🟡):**

| # | Проблема | Статус |
|---|----------|--------|
| 3 | `torso_length_m=0.5` не откалиброван рулеткой (абсолют 3D приблизителен; ординал ок) | 🟡 |
| 4 | `fps` в StateFile = темп доставки, не обработки | 🟡 (семантика) |
| 6 | Объекты видно с ракурса человеческого роста (eye-level камера) | 🟡 |

**Закрытые по корню (историческая справка; детали — в `docs/phase-*.md` и git):** эмоция near-chance
(10.1 — HWC→NCHW layout-баг + landmark-align; live: Neutral 0.82–0.93, H=0.12–0.24) ✅ · поза desk-гэп
(GRIP + occlusion-aware withhold, БЕЗ файн-тюна) ✅ · hand-ROI vis-floor (10.2) ✅ · SFace unaligned-кроп
(15.6 P5 align-фикс +19.5 п.п.) ✅.

> **Фаза 16 (Dataset Factory) хвостов в ядро не внесла** — `mvm-dataset` чистый потребитель WorldState
> (как StateFileWriter), детекторы/fusion/mesh не тронуты. Privacy-blur (16.5): точный YuNet-face-bbox
> когда лицо детектнуто, щедрая fallback-полоса когда нет (privacy-first, геометрическая эвристика).
> Опционально: аудио (YAMNet, G₄) — deferred (новая dep против минимализма, вне ядра).

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
| **11** | ✅ | Fusion-полировка (+0 ONNX): mask-контакт, activity-FSM+journal, зоны-persistence, anomaly v2, self-calib, object-permanence, person-маски |
| **12** | ✅ | LLM-render (+0 ONNX/+0 deps): pull-модель snapshot/recap/inspect, токен-диета, MCP stdio-сервер |
| **13** | ✅ | Agent Tools/Skills Polishing: когерентное CLI-дерево + `--help`, agent how-to, само-докум. MCP |
| **14 → v0.1.0** | ✅ | Продукт: XDG/config, `setup`/`doctor`, watched-галерея, `serve --with-camera`, `mcp install`, `service`, дистрибуция v0.1.0 |
| **15** | ✅ | ML Refactor (`kaggle`, lab-first, журналы LJ0–LJ6): 15.0 лаб+бенч · 15.1 поза (2D-GRIP z-free, flexion, Mahalanobis↔VAE, footprint −79%) · 15.2 переходы (E2 fall «пол+скорость таза», AUROC 1.000) · 15.3 трекер (stale-relink фикс) · 15.4 golden-симулятор + словарь v2 `using_object(class)` · 15.5 паспорт+диета (⭐ SFace align +19.5 п.п.) · 15.6 Rust-порт P0–P8. Свёрнута в код v0.3.0 (v0.2.0 отдельным тегом не выпускалась). Детали — `docs/phase-15-plan.md`, `docs/phase-15.6-port-plan.md` |
| **16 → v0.3.0** | ✅ 16.0–16.6-A | **Dataset Factory** (`kaggle`, +крейт `mvm-dataset`, чистый потребитель WorldState): **16.0 ✅** Semantic Contract (Canonical Observation, статусы, provenance, Router G₁–G₅) · **16.1 ✅** инфра (session-grouper, `DatasetConfig`, `DatasetWriter`+агент, `--record`) · **16.2 ✅** сырьё (RGB/depth-16bit/masks-RLE) + quality-as-data · **16.3 ✅** CLI `mvm dataset list/info/stats/prune` · **16.4 ✅** Python-экспортёры (PyTorch/TFRecord/WebDataset) · **16.5 ✅** точный YuNet-face-bbox privacy (`--privacy` флаг) · **16.6-A ✅** data-quality gates (`mvm dataset check` + Python `validate.py`, CI exit-code). Всё live-провалидировано, релиз v0.3.0. **Осталось:** 16.6-B (hard-example mining). План — `docs/phase-16-plan.md` |
| **16.7** | ✅ live | **Semantic Session Log** (`kaggle`): два стора истории сцены → один. **1 ✅** text-only запись (`save_frames`) · **1.5 ✅** обогащение роутера (relations/activities/events, SCHEMA 16.7.0, `route_with_relations`+`SceneRelations`) · **2 ✅** миграция ридеров (recap/inspect читают сессии через flat-shape shim + `Source`-resolver, golden-паритет) · **3 ✅** default-on text-only (`effective_recorder`, `vision_log`, ретайр плоских логов, `--no-vision-log`). Live-провалидировано (4-мин камера). План — `docs/phase-16.7-plan.md` |
| **17** | ✅ | **Canonical MVM Agent Skills** — repo-canonical `mvm-vision` + `mvm-dataset`, CI contract gate, portability и Hermes integration завершены. Только know-how-слой: ядро/версия не затронуты. План — `docs/phase-17-plan.md` |
| **18 → v0.4.0** | 🚧 18.0–18.2 ✅ | **Visual Cortex Layer:** canonical state contract, benchmark baseline, typed observation bus, canonical Fusion и Cortex skeleton закрыты; далее Scene Events + Cortex Interpretation, затем runtime memory/prediction/self-model/affordance/attention → CLI/MCP + benchmark evaluation. План — `docs/phase-18-plan.md` |

## Терминология

- **Микро-агент** — независимый детектор с confidence. **Fusion** — слой, собирающий сигналы в граф/описание.
- **Workspace 3D** — метрическая система координат (метры, origin = камера).
- **Confidence-first** — каждый сигнал несёт числовую уверенность. **Prediction error** —
  `||predicted − observed||` → естественный confidence-сигнал.
- **Body-primary** — несущая сущность цепочки = тело (не лицо). **Reserved-id** — id 0–9 за галереей, ≥10 анонимам.
