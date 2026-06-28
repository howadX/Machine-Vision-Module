# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*

---

## ✅ FRONTIER ЗАКРЫТ (2026-06-28): тело-первичная цепочка + id-модель + стабилизация

> **Бывший вопрос №1** (несущая сущность цепочки детекторов) — **решён и live-validated**
> в фазе 8.0.x. `--native-camera` теперь **тело-первичен**: человек без фронтального лица
> трекается (presence + поза + торс-3D), identity/emotion дотягиваются, когда лицо появляется.
> Полный разбор — `wiki/mvm/subsystems/detector-hierarchy.md` + `wiki/machine-vision/IMPLEMENTATION_PLAN.md`.

### Живая цепочка (`--native-camera`, тело-первичная)
```
frame → person_detection (ТЕЛО, первичный) → сущность/трек = ТЕЛО (NativePose::bodies)
          ├─ BlazePose → 33 lm → PoseVAE → posture                 (всегда)
          ├─ торс-3D (плечи 11/12 → бёдра 23/24, torso_to_3d)       (всегда; gated на visibility ≥0.5)
          └─ YuNet (ЛИЦО) по кадру, матч к телу по НОСУ (lm[0] в face-bbox)
                есть лицо → SFace id + EmotiEffLib emotion;  нет → identity/emotion = None
        tracker по bbox ТЕЛА (Kalman + IoU + cosine/label/center re-link) → build_scene → StateFile/JSON
```
Bridge-путь (`--native`) не тронут (Python тело→поза→3D как был).

### Как разрешились открытые решения
1. **3D-источник:** ✅ ТОРС (`space::torso_to_3d`, `Config.torso_length_m=0.5`). torso_length не откалиброван рулеткой → абсолют приблизителен, ординал ок.
2. **Модель данных:** ✅ минимальный — `Detection.bbox` = тело; `face_bbox: Option<BBox>` **отложен** (не нужен для закрытия отказа).
3. **Тело без лица:** ✅ трекается, identity/emotion = None.
4. **Лицо без тела:** ✅ фолбэк на лицо-сущность (с A1-подавлением, чтобы не мигать body↔face).
5. **Поиск лица:** ✅ YuNet по кадру + матч **по носу тела** (keypoint-in-bbox, `best_face_for_nose`) — чинит swap при перекрытии person-bbox.
6. **bridge-путь:** ✅ не тронут.

### Identity-модель (решение владельца, 8.0.2)
id **0–9** — резерв за пользователями галереи (Вадим=0, липко на присутствие); тела-анонимы — **эфемерные id ≥10**, освобождаются/переиспользуются при выходе. **Эксклюзивно** (один user-id — одному телу): при #5 (галерея из 1) чужой остаётся анонимным, а не вторым «Вадимом». Наружу идёт `PersonTrack.display_id` + `shown_identity()`.

### Стабилизация (8.0.3–8.0.4, опция A)
- **A1 анти-flip:** подавление face-fallback, когда лицо объяснено телом/активным треком → churn 44–55 → **6–7 треков**, нет мигания body↔face.
- **A3:** торс-3D EMA. **R:** visibility-gate торс-3D (нет фантомных позиций). **A2-lite:** center-relink анонимов под движением (стабильный эфемерный id).

**Вне scope (известное):** выбросы классификатора позы (`standing`↔`sitting`, #7.5.5 — точность модели); остаточный #5 (нужен 2-й человек в галерее); калибровка `torso_length_m`; `--emotion-debug` линковка деградировала.

---

## Философия сборки

MVM проектируется от сборки наружу, а не от кода внутрь.

**Принцип:** если модуль добавить легко — архитектура останется чистой по умолчанию.
**Цель:** один бинарник. Скачал — запустил — работает. Без Python, без pip, без GPU.

## Фундаментальный стек

Три инварианта, которые определяют каждое архитектурное решение. Если фича их нарушает — она не входит в MVM.

1. **Минимализм / чистая арифметика.** Логика fusion, трекинга, 3D, uncertainty — это умножения, деления, скалярные фильтры над данными, которые уже в памяти. **Ноль новых зависимостей** на каждую новую интеллектуальную фичу. Не тянем linalg-крейты, не обучаем MLP там, где работает замкнутая форма. Footprint растёт только когда добавляется новый ONNX-эксперт.
2. **CPU-first, $0.** Ни рантайм, ни модели не требуют GPU. Никаких внешних API. Цель задержки — `<50ms` на CPU.
3. **Полный нативный Rust-пайплайн.** Python-модели (`mvm_bridge.py`, HSEmotion, MediaPipe) — это **прототипы-эксперты, строительные леса**. Боевой кластер целиком на Rust; Python планомерно сносится (фаза 7). Новый код не должен углублять связь с Python.

### Что производит MVM

**Метрический 3D scene-graph каждый кадр.** Не плоский список людей, а граф: узлы-сущности с метрической 3D-позицией (origin = камера), рёбра-отношения между ними.

- **3D-позиция** считается **pinhole-геометрией с known-size априорами** (`z = fx · real_size / pixel_size`), а не depth-estimation сетью. Чистая арифметика. Тир — **грубый метрический + сильный относительный субстрат** (не robotics-grade метрология): метод даёт ±15-30% абсолютной глубины, но робастный ординал (кто ближе) и контекстные зоны. Точность растёт по upgrade-пути (identity-prior → landmark-калибровка → stereo/мульти-камера), наполняя тот же контракт `pos_3d`. Подробный конверт точности — в задаче 6.2.
- **Fusion** имеет две операции, и пространственная — главная:
  - **Реляционная (основная):** N сущностей → граф близости/пересечений → контекстные зоны (связные компоненты). «Голова + тело + рука + чашка в общей зоне; расстояния между объектами = тонкий контекст взаимодействия».
  - **Атрибутивная (частный случай):** слияние нескольких оценок одного атрибута (эмоция во времени / разные модели) через Product-of-Experts.

Текущий тяжёлый стек (оправдан): `ort` (ONNX inference, CPU) + `ndarray` + `image` — мост к ML-моделям без Python. Всё остальное (`anyhow`, `serde`, `chrono`) тривиально. Фундаментальные фичи фазы 6 (Kalman, 3D, entropy, реляционная fusion) **не добавляют ни одной зависимости**.

## Почему MVM, а не VLM?

Vision-Language Models (GPT-4V, Claude Vision, LLaVA) делают всё: изображение → текст.
Но за универсальность платят: GPU, секунды задержки, $0.05/запрос, облако, чёрный ящик.

MVM — альтернативная архитектура: **ансамбль специализированных микро-агентов**.
Каждый агент обучен на одну задачу и выдаёт структурированный сигнал с confidence.

| | MVM | VLM |
|---|-----|-----|
| **Архитектура** | Ансамбль специалистов | Одна end-to-end модель |
| **Задержка** | <50ms CPU | 2-30s GPU |
| **Стоимость** | $0 | $0.05/запрос |
| **Приватность** | Абсолютная (всё локально) | Облако |
| **Confidence** | Per-signal (каждый сигнал с confidence) | Нет |
| **Интерпретируемость** | Полная (каждый сигнал прослеживаем) | Чёрный ящик |
| **Заменяемость** | Per-agent (один крейт) | Переобучение всей модели |
| **Предел качества** | Качество fusion-логики | Размер модели |

### Когда MVM превосходит VLM

- **Real-time мониторинг** (30fps, локально, $0)
- **Структурированный выход** (identity, emotion, posture, 3D-позиция — каждое поле с confidence)
- **Приватные сценарии** (дом, офис, медицина)
- **Промышленная аналитика** (N камер, structured data → Grafana)
- **Робототехника** (control loop требует <100ms)

### Когда VLM сильнее

- **Открытый мир** (незнакомые объекты, нетипичные сцены)
- **Zero-shot** (работает без обучения на новых концептах)
- **QA-режим** («Сколько людей в комнате? Что они делают?»)

### Гибридная стратегия

MVM покрывает 80% практических сценариев (присутствие, эмоции, поза, позиция) при 0.1% стоимости VLM.
Для оставшихся 20% (открытое описание, QA) — MVM + VLM как fallback.

```
Камера → MVM (real-time, 30fps, structured)
              ├── 80% → StateFile (MVM сам)
              └── 20% → VLM по запросу (сложные сцены)
```

---

## Текущий статус (25 июня 2026, фаза 6 закрыта ✅ + 3D откалибровано)

> ⚠️ **Снимок фазы 6 (исторический).** Актуальное состояние на 2026-06-28 — выше в
> «✅ FRONTIER ЗАКРЫТ» + «Фаза 8» (body-primary, id-модель, стабилизация; **151 тест**,
> mvm-posture/mvm-fusion давно production). Таблица ниже отражает состояние на 25 июня.

### Что работает при `mvm-cli --native --once --dump-json`

```
Камера /dev/video0 (CAP_FFMPEG — обход PipeWire)
    │  [смонтирована под потолком, ~45° вниз, перевёрнута → cv2.ROTATE_180 в bridge]
    ▼
Python bridge (mvm_bridge.py, ~12s загрузка моделей)
    ├─ SCRFD           → bbox лица (CPU, ONNX)
    ├─ ArcFace         → identity + confidence (512d, multi-angle gallery match)
    ├─ HSEmotion       → 8 эмоций + confidence (EfficientNet-B0, GPU PyTorch)
    ├─ PoseLandmarker  → 33 COCO landmarks
    ├─ PoseVAE         → 6 поз + confidence + plausibility
    └─ Workspace3D     → (x,y,z)м + ws_confidence (pinhole estimated_distance)
         │
         ▼ JSON line
Rust mvm-cli
    ├─ parse_line      → Vec<Detection>
    ├─ EmotiEffLib B0  → emotion + distribution[8] + entropy + calibrated_conf ← 6.3
    ├─ SFace           → identity (gallery: Вадим, 8 эталонных фото)
    ├─ PersonTracker   → Kalman bbox (6.1) + identity re-link + native 3D pinhole (6.2) + temporal emotion PoE (6.4b, λ=0.7)
    ├─ mvm-fusion      → build_scene() граф близости + зоны (6.4)
    ├─ StateFile       → ~/.hermes/profiles/sight/workspace/current_state.md
    ├─ dump_table()    → ASCII таблица в терминал
    └─ --dump-json     → Scene JSON: faces+emotion_distribution+pos+zones+relations ← 6.4
```

### Бинарники

| Файл | Размер | Команда сборки |
|------|--------|----------------|
| `target/release/mvm-cli` | **~26 МБ** | `cargo build --release` |
| `dist/mvm` | **~26 МБ** | `cargo xtask bundle` |

### Крейты

| Крейт | Статус | Содержимое | Тесты |
|-------|--------|------------|-------|
| **mvm-core** | ✅ production | BBox, Detection, PersonTrack, PersonTracker (Kalman + **identity re-link**), Pinhole, DepthEma, Entity/Scene, **EMOTION_LABELS + temporal fuse (6.4b)** | 48 |
| **mvm-detect** | ✅ production | YuNet ONNX (228 КБ), CHW BGR preprocessing, set_score_threshold() | 5 |
| **mvm-identity** | ✅ production | SFace ONNX (37 МБ), gallery Вадим, cosine match | 5 |
| **mvm-emotion** | ✅ production | EmotiEffLib B0 ONNX (16 МБ), 8 классов VGAF (eng), EmotionResult + entropy | 7 |
| **mvm-posture** | ✅ production (с 7.5) | NativePose: person-det + BlazePose + PoseVAE/classifier (порт), `bodies()` | ~12 |
| **mvm-fusion** | ✅ production | fuse() PoE, build_scene() union-find граф, Entity/Relation/Scene | 9 |
| **mvm-pipeline** | ✅ production | Camera (Python bridge subprocess), parse_line | 5 |
| **mvm-cli** | ✅ production | main, StateFile, Scene JSON, --dump-json, --native/--once, --emotion-debug | 17 |
| **xtask** | ✅ | `cargo xtask bundle` → dist/mvm | 1 |

**Всего: 106 тестов, все зелёные.**

### Известные проблемы

| # | Проблема | Причина | Приоритет |
|---|----------|--------|-----------|
| 1 | **EmotiEffLib near-chance на потолочном ракурсе** | На живых данных (2026-06-25) распределение почти равномерное (peak ≈0.16 vs uniform 0.125) — лицо сверху ≠ фронтальные кропы VGAF. Temporal-fusion (6.4b) джиттер убрал, метка стабильна, но `conf` честно низкая. Нужен A/B vs HSEmotion + переоценка угла/датасета | 🟡 |
| 2 | **emotion near-uniform → метка слабоинформативна** | Следствие #1: argmax (`Fear`) — слабая склонность модели, а не уверенность. Downstream обязан читать `emotion_conf`/entropy, не голую метку | 🟡 |
| 3 | **mvm-posture заглушка; поза только у 1 человека** | Было: поза от Python bridge, MediaPipe `.task` возвращал `poses=1`. **✅ Закрыто (7.5.2–7.5.4):** нативный `NativePose` (person-det → BlazePose → VAE) делает инференс по ROI на каждого человека (мульти-persona по построению), EN-метки. Врезан в `--native-camera`. **Живая валидация на 2 людях (2026-06-26): 2 независимые позы в одном кадре** (5 кадров, напр. Вадим `sitting` + другой `standing`) — `poses=1` больше нет | ✅ |
| 4 | **Детекция только через Python** | YuNet есть, но не подключён как camera source | 🟢 |
| 5 | **identity false-positive при галерее из 1 человека** | `two_people`/`unknown_face` (2026-06-25): чужой опознан «Вадим»@0.63–0.69, порог 0.45 присваивает ближайшему. **Но разделимо:** реальный Вадим=1.00 vs чужой≈0.66 → порог ~0.8 или margin-based решает. **Отложено по решению.** | 🟡 |
| 6 | **track-id хрупок при движении/возврате** | Доминирующая находка live-кампании. При ~1 fps резкое изменение bbox рвёт IoU → новый id. **Частично исправлено:** identity-based re-link (tracker, `relink_window_frames=90`, `relink_min_conf=0.5`) — детекция, не прошедшая IoU, переподцепляется к недавнему missed-треку с той же identity-меткой. ⚠️ **Оговорка:** re-link по метке, не по эмбеддингу → при галерее из 1 человека (#5) два разных «Вадима» в мульти-persona сцене могут слиться в один id. Полный фикс (cosine эмбеддингов) — будущее. Остаточный churn снимает FPS (фаза 7) | 🟡 |

### Live-валидация на камере (2026-06-25)

Прогон через `tools/cam_test.py` (окно от первого кадра, действие непрерывно):

| Сценарий | Итог |
|----------|------|
| `two_people` | ✅ 6.4a: зона размера 2 + `near` (dist_m=1.09, 3D-евклид проверен) |
| `walk_near_far` | ✅ 6.2 3D монотонна 1.1–2.33м; 🟡 2 разрыва (re-link gap на пустом кадре) |
| `occlusion` | ✅ 6.1 Kalman бриджит 4 окклюзии (`missed`-инерция), 0 разрывов id |
| `leave_return` | ✅ identity re-locks каждый возврат; 🟡 track-id не переживает уход |
| `single_sit` | ✅ покой 20/20 стабилен; 📏 2.8м→2.58м (-8%, дальний край за ~2.2м) |
| `unknown_face` | ❌ чужой→Вадим@0.66 (#5); ✅ бонус: 6.4a масштаб на 3 сущности |

**Сквозной вывод:** ядро фазы 6 (3D, граф, Kalman, сглаживание) работает. Две слабости — низкий FPS (Python bridge) и хрупкость track-id (#6); обе закрывает либо фаза 7, либо identity-relink.

### Модели эмоций: архитектурное примечание

**EmotiEffLib B0 (Rust) и HSEmotion (Python) — одна архитектура: EfficientNet-B0.** Оба — CNN на кропе лица 224×224. Разница в датасете (VGAF vs AffectNet) и рантайме (ONNX/CPU vs PyTorch/GPU).

Стратегия выбора: **confidence-based fallback** (реализовано в фазе 5.1) — берём предсказание с более высокой confidence. A/B сравнение (`--emotion-debug`) реализовано в фазе 5.2.2.

### Камера: FFMPEG backend

OpenCV `VideoCapture(0)` (V4L2) не работает когда PipeWire держит камеру эксклюзивно (Ubuntu 24+, Wayland). Решение: `VideoCapture('/dev/video0', cv2.CAP_FFMPEG)` — ffmpeg/avdevice не конфликтует с PipeWire.

Зафиксировано в `tools/mvm_bridge.py`.

### TDD-правила (для Claude Code)

- TDD обязателен: сначала падающий тест, потом реализация
- `cargo test --workspace` должен оставаться зелёным после каждого изменения
- Коммиты после каждой задачи
- `Config { models_path: PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")), ..Default::default() }` — паттерн test_config()

---

---

## Python bridge (родительский проект)

MVM — гибридный проект. Rust-крейты (`mvm/`) используют Python bridge из родительского каталога:

```
Machine Vision Module/           ← корень проекта (cwd при запуске Claude)
├── core/                        ← Python модули
│   ├── face_id.py               # SCRFD + ArcFace + multi-angle gallery
│   ├── workspace_3d.py          # 3D Workspace (pinhole calibration)
│   ├── pose_vae.py              # PoseVAE (48.6K params)
│   ├── emotion_hse.py           # HSEmotion (EfficientNet-B0, GPU)
│   └── cpu_face_id.py           # YuNet + SFace CPU-native identity
├── tools/
│   └── mvm_bridge.py            # Камера → JSON lines (запускается как subprocess)
├── models/
│   ├── mediapipe/               # FaceLandmarker + PoseLandmarker .task файлы
│   └── emotion/                 # vposer_coco33.pth, pose_classifier.pth, ...
├── venv/                        # Python 3.11+ virtualenv
└── mvm/                         ← Rust workspace (ЭТОТ каталог)
    └── crates/...
```

**Важно для Claude Code:** Python-файлы доступны по путям `../core/`, `../tools/`, `../models/`. При чтении Python bridge кода используй эти относительные пути.

**Запуск Python bridge (из mvm/):**
```bash
../venv/bin/python3 ../tools/mvm_bridge.py --json-only --max-frames 1
```

**Запуск mvm-cli (из mvm/):**
```bash
cargo run --release -- --native --once --dump-interval 1
# Camera::spawn() внутри запускает ../venv/bin/python3 ../tools/mvm_bridge.py
```

## Структура (9 крейтов)

```
mvm/
├── Cargo.toml              # [workspace] 9 members, edition 2024
├── .cargo/config.toml      # alias: cargo xtask = run --package xtask --
├── crates/
│   ├── mvm-core/           # ✅ типы, трейты, Config, PersonTracker
│   ├── mvm-detect/         # ✅ YuNet ONNX (228KB, embed-model, multi-stride NMS)
│   ├── mvm-identity/       # ✅ SFace ONNX (37MB, L2-embeddings, gallery)
│   ├── mvm-emotion/        # ✅ EmotiEffLib B0 ONNX (16MB, 8-class VGAF)
│   ├── mvm-posture/        # ✅ нативная BlazePose + PoseVAE/classifier (НЕ заглушка)
│   ├── mvm-fusion/         # ✅ build_scene (union-find граф+зоны) + PoE (НЕ заглушка)
│   ├── mvm-scene/          # ✅ YOLO26n ONNX (80 классов COCO, end-to-end/raw, object-узлы) [8.E]
│   ├── mvm-pipeline/       # ✅ Camera (Python bridge) + NativeCamera (v4l2 MJPG)
│   └── mvm-cli/            # ✅ main, StateFile, dump_table(), --native/--native-camera/--once
├── xtask/                  # ✅ cargo xtask bundle → dist/mvm
├── dist/                   # bundled binary output
└── models/                 # ONNX: YuNet (228KB), SFace (37MB), EmotiEffLib (16MB)
```

## Roadmap

### ✅ Фаза 5.1 — Стабилизация (выполнено)

| Задача | Статус |
|--------|--------|
| YuNet CHW/HWC preprocessing fix + threshold API | ✅ |
| Confidence-based emotion fallback (Rust↔Python) | ✅ |
| Обновить dist/mvm (xtask bundle, 26 МБ) | ✅ |
| Сборка галереи на эталонных фото (Вадим) | ✅ |
| Camera: FFMPEG backend (fix PipeWire) | ✅ |

### ✅ Фаза 5.2 — Качество (выполнено)

| Задача | Статус |
|--------|--------|
| `--dump-json` флаг (machine-readable JSON stdout) — 5.2.1 | ✅ |
| A/B сравнение EmotiEffLib vs HSEmotion (`--emotion-debug`) — 5.2.2 | ✅ |
| PersonTracker: emotion decay при missed > N кадров — 5.2.3 | ✅ |
| Интеграционные тесты: bridge + Rust pipeline — 5.2.4 | ✅ |

**Фаза 5 закрыта. Активная фаза — 6 (Metric 3D Scene-Graph), точка входа 6.2.0.**

### ✅ Фаза 6 — Metric 3D Scene-Graph (закрыта, 25 июня 2026)

| Задача | Статус |
|--------|--------|
| 6.1 SG-KF: Kalman bbox (4× скаляр + State-Gating) | ✅ |
| 6.2 Метрический 3D (pinhole + DepthEma, native Rust) | ✅ |
| 6.2.0 Эмпирическая калибровка face_height_m (рулетка) | ✅ |
| 6.3 Uncertainty: EmotionResult + entropy + calibrated_conf | ✅ |
| 6.4 Fusion: build_scene() граф + fuse() PoE + Scene JSON | ✅ |
| 6.4a live-валидация: 2 чел → зона размера 2 + ребро `near` (dist_m=1.09, 11/11 кадров) | ✅ 2026-06-25 |
| 6.4b live-валидация: temporal-fusion → 0 смен метки за 18 кадров (был джиттер каждый кадр) | ✅ 2026-06-25 |

**Калибровка завершена (2026-06-25, `tools/calib_3d.py`):**
- Камера под потолком (~45° вниз), перевёрнута → `cv2.ROTATE_180` в bridge (коммит `73dc8815`)
- Рулеточный замер 0.3–2.5м (субъект Вадим). Инвариант пинхола `pixel_h·z` = **181 px·м** стабилен в полосе 1.0–2.0м (<1%) → одна константа.
- **`face_height_m = 0.44`** (= 181/fx). Это *эффективная bbox-высота* (≈2× анатомии, абсорбирует bbox-конвенцию SCRFD), не фронтальные 0.19. Фит <3% на 0.5–2.0м.
- Артефакты концов (не дрейф модели): клиппинг лица о край кадра при <0.4м; шум захвата >2.2м.
- `DEPTH_MAX_M` возвращён к 3.0м (физический диапазон комнаты).
- Полная эмпирическая таблица точности — в docstring `space.rs`.
- ⚠️ Константа привязана к SCRFD-bbox → **рекалибровать при нативном детекторе (фаза 7)**.

### 🟢 Фаза 7 — Снос Python из hot-path (Python-light)

> **Цель (scope зафиксирован 2026-06-25):** убрать Python из боевого hot-path — камера → детекция → 3D → identity → fusion целиком в Rust, fps↑. **Поза (7.3) вынесена в фазу 7.5** (нет ONNX-модели лендмарков тела) → bridge ужимается до pose-only, venv остаётся только ради позы. Полный Python-zero — фаза 7.5. Поэтапный план — ниже («Поэтапный план исполнения»).

| Задача | Риск | Статус |
|--------|------|--------|
| 7.1 YuNet первичный детектор (A/B-harness `--detect-ab`, G4) | 🟡 | ✅ harness; YuNet — источник в `--native-camera`; `face_height_m_yunet=0.32` откалиброван (рулетка, наклон ±11%) |
| 7.2 Нативный захват камеры в Rust | 🔴→🟢 риск снят (spike) | ✅ `mvm-pipeline::native`, MJPG mmap, hardware-тест прошёл |
| 7.3 Нативная поза: ONNX-лендмарки + порт PoseVAE, мульти-persona | 🔴 самое дорогое (**нет ONNX-модели лендмарков тела**) | ⏭ вынесено в фазу 7.5 |
| 7.4-light Python-light hot-path (`--native-camera`, Вариант A: без позы) | 🟢 | ✅ **25 fps с лицом, ноль Python**; bridge сохранён как fallback (поза) |
| 7.5 SFace identity в трекер: margin-identity (#5) + cosine-re-link (#6) | 🟢 | ✅ реализовано + unit; **#5 live-validated** (Вадим 0.55–0.59, чужой→unknown); **исправлен фатальный баг SFace** (scalar_op=0 обнулял вход) |

> **Аудит готовности проведён 2026-06-25** — план сверен с фактическим кодом/моделями. 6 расхождений и список пробелов инструментария задокументированы ниже в детальном плане («Аудит готовности»). Риски/статусы в этой таблице уже отражают аудит.
>
> **Фаза 7 (Python-light) реализована и провалидирована на камере 2026-06-25.** Scope: поза вынесена в 7.3/7.5 (Вариант A — без позы в быстром пути). `mvm-cli --native-camera` = pure-Rust hot-path (v4l2 MJPG → YuNet → ONNX-обогащение), **25 fps с лицом в кадре, ноль Python** (vs 1.1 fps bridge baseline); `mvm_bridge.py` сохранён как legacy-fallback с позой (НЕ удалён — по требованию владельца).
>
> **Живая валидация (`tools/cam_test.py --native-camera`):** ✅ fps 25 с лицом; ✅ #5 identity (Вадим 0.55–0.59 опознан, чужой → unknown) — *после фикса фатального бага SFace, где `scalar_op1/2=0` обнуляли вход и любое лицо матчилось 1.0*; ✅ 3D-калибровка `face_height_m_yunet=0.32` (наклон даёт ±11%, в конверте); ✅ ординал 3D (`walk_near_far`). Эмоция — near-chance/Fear (известный #1, потолочный ракурс).
>
> **Фаза 7 ЗАКРЫТА 2026-06-25 (Вариант A).** Дефолт = bridge-путь (`--native`, с позой); `--native-camera` = быстрый opt-in (это текущее состояние кода, менять нечего). Остаточный гейт #6 (cosine-re-link no-merge) закрыт unit-тестом `cosine_relink_rejects_different_person`. Следующая фаза — 7.5.

### 🟢 Фаза 7.5 — Python-zero: нативная поза + удаление bridge

> **Цель:** последняя ответственность Python — поза — становится нативной, затем `mvm_bridge.py` + venv удаляются. Один самодостаточный бинарник, мульти-persona поза (чинит #3). **Детальный план: `mvm/docs/phase-7.5-plan.md`. Итоги спайка: `mvm/docs/phase-7.5-spike-findings.md`.**

| Задача | Статус |
|--------|--------|
| 7.5.0 Spike: загрузка BlazePose ONNX + совпадение топологии 33-лендмарков (главный риск) | ✅ **гейт пройден** (2026-06-26): обе ONNX грузятся+работают в Rust `ort`; recon+xy_diff подтвердили топологию на студийных И боевом потолочном ракурсе; ROI-источник `person_detection_mediapipe` провалидирован на камере (Open decision #1 закрыт) |
| 7.5.1 Достать+забандлить модели OpenCV Zoo (person_detection + pose_estimation mediapipe) | ✅ модели в `mvm/models/` (+gitignore с URL, конвенция download-separately); гейты пройдены: грузятся в release-сборке из канонического каталога + `cargo xtask bundle` собирает `dist/mvm` (26.6МБ). `include_bytes!`-embed вводится в 7.5.2 вместе с загрузчиком (иначе мёртвые байты) |
| 7.5.2 `mvm-posture` production: ROI → 33 лендмарка (нативно) | ✅ two-stage нативный инференс (`detector.rs` + `pose.rs` + `NativePose` мульти-persona); оракул-валидирован vs Python (боксы <20px, 33 лендмарка <6%, топология); 6 тестов; embed-model wiring (`include_bytes!` обеих ONNX + якоря `anchors_person.bin`) |
| 7.5.3 Порт PoseVAE + PoseClassifier (hand-rolled, чистая арифметика; EN-метки готовы) | ✅ `classifier.rs`: `normalize_landmarks` + VAE-энкодер/декодер + классификатор, чистая арифметика, ноль новых зависимостей; веса `pose_weights.bin` (53205 float). **Bit-close к Python** (probs <1e-3, recon <1e-4, logpz <1e-2). `NativePose::poses()` = end-to-end на человека |
| 7.5.4 Мульти-persona + врезка в `--native-camera` (чинит #3) | ✅ `NativePose` врезан в `mvm-cli --native-camera`: person-det ROI → поза → матч к лицу по bbox → поля `posture/posture_confidence/recon_error/plausibility` в StateFile + `--dump-json`. **Живая валидация 2026-06-26:** 1 чел — Вадим, 23 fps, поза реагирует (`sitting_at_desk`/`arms_crossed`@0.99/`standing`@0.93). **2 чел — мульти-persona подтверждена** (20 fps; 5 кадров с 2 РАЗНЫМИ позами одновременно, напр. Вадим `sitting`@0.93 + другой `standing`). ROI = `person_detection` (данные побили YuNet-face) |
| 7.5.5 G6 оракул точности позы (native vs MediaPipe) | ✅ воспроизводимый оракул (`spike_pose/golden_refs.py` + regression-тест `pose_class_oracle_vs_python`) на 7 закоммиченных эталонах. **Согласие классов native↔Python = 5/7 (71%)**, гейт ≥5/7. Расхождения — `standing`→`sitting_at_desk` (слабый класс на потолочном ракурсе, остаточный z-зазор). **Quality:** pose-суперсэмплинг (≈INTER_AREA) снизил z-расхождение vs Python на 33% (0.099→0.067) |
| 7.6 Python-zero: удалить bridge + venv ⚠️ **только с явного согласия владельца** | 🔜 |

### 🟢 Фаза 8 — Body-primary + scene-graph наружу (в работе, 2026-06-28)

> **CHECKPOINT 2026-06-28 (обновлён, 8.E приземлён).** Ветка `rust-core`. **171 тест зелёный**, release чист.
> Коммиты поверх `117b8df1`: `52907ca4` (спайк 8.0.1), `dfdca72c` (body-primary+id 8.0.2), `b50d7288`
> (A1+A3 8.0.3), `4380c1e9` (R+A2-lite 8.0.4), `ebd9e5a9` (docs checkpoint), `0cae2f17` (mvm-scene +
> object-узлы + cadence 8.E), `5eebed53` (YOLO26n end-to-end decode + person-фильтр + модель).
> Решение владельца: **venv остаётся для тестов/дебага** (7.6-снос отменён).
> План фазы: `wiki/machine-vision/IMPLEMENTATION_PLAN.md`; оценка Agent Mesh: `wiki/machine-vision/08-fusion-architecture/agent-mesh-eval-8.0.1.md`.

| Задача | Статус |
|--------|--------|
| 8.0.1 Спайк Agent Mesh (ort contention + latest-frame) → **threads, не tokio; cadence — рычаг** | ✅ |
| 8.0.2 Body-primary иерархия (FRONTIER) + reserved-id identity-модель | ✅ live-validated |
| 8.0.3 Стабилизация A1 (анти-flip) + A3 (торс-3D EMA) → churn 44→6 | ✅ live-validated |
| 8.0.4 R (visibility-gate фантомов) + A2-lite (center-relink анонимов) | ✅ (ре-валидация на 3 людях ожидается) |
| **8.E cadence + YOLO26n object-узлы** (крейт `mvm-scene`, 9-й крейт) | ✅ — модель = **YOLO26n** (ultralytics 8.4.81 export, end-to-end NMS-free `[1,300,6]`, 9.5МБ, gitignored). Object-узлы → `build_scene` (overlap-рёбра человек↔объект), cadence `object_cadence=3` в синхронном цикле, person-класс отфильтрован (G₁ владеет людьми). Валидировано: person 0.93 на референсе, live 21.7fps. **Live-валидация сцены с реальными объектами — за владельцем** |
| 8.E depth (Depth-Anything-V2-Small, low-cadence) / audio (YAMNet) / hands | 🔜 СЛЕДУЮЩЕЕ |
| 8.F cross-group fusion = object/audio-узлы → существующий `build_scene` (НЕ MBT) | 🟡 object-часть готова (8.E); audio/depth-узлы — с их крейтами |
| Agent Mesh (async threads) — отложен до появления тяжёлого детектора (depth ~100мс) | 🔜 |

> **Решения 8.0.1 (зафиксированы спайками):** mesh на `std::thread` (НЕ tokio; contention опровергнут — текущие эксперты ~12мс/~80fps); cross-group fusion = object-узлы в `build_scene` (НЕ MBT; verified-research); cadence врезается вместе с тяжёлым детектором (YOLO/depth). A2 (полный torso-center matching) — не нужен, A1+A2-lite хватило.

### 🟢 Фаза 9 — Продукт

| Задача | Статус |
|--------|--------|
| Авто-калибровка fx (zero-config) | 🔜 |
| Drag-and-drop галерея (добавить человека = кинуть фото в папку) | 🔜 |
| One-click install (один бинарник, без pip/venv/Python) | 🔜 |
| systemd service + cron state dump | 🔜 |

---

## Фаза 6 — Metric 3D Scene-Graph: детальный план

> **Для:** brainstorm с Opus / делегирование Claude Code.
> **Статус базы:** все тесты зелёные. Фаза 5.2 завершена.
> **Принцип фазы:** чистая арифметика, CPU, **ноль новых зависимостей**, всё в нативном Rust (без углубления связи с Python).

### Цель фазы

Превратить плоский список детекций в **метрический 3D scene-graph**: узлы-сущности с 3D-позицией (origin = камера), рёбра-отношения между ними. Это перцептивный субстрат для робототехники.

Несущая абстракция (живёт в `mvm-core/src/types.rs`):

```rust
pub struct Entity {
    pub id: u32,
    pub kind: EntityKind,      // Person | Face | Hand | BodyPart | Object(class)
    pub bbox: BBox,            // 2D, сглажен Kalman (6.1)
    pub pos_3d: [f32; 3],      // метры, origin = камера (pinhole, 6.2)
    pub radius_m: f32,         // габарит из bbox+глубины (для зон/дистанций)
    pub pos_conf: f32,         // confidence калибровки 3D
    pub attrs: Attributes,     // identity, emotion{dist[8]+uncertainty}, ...
}

// mvm-fusion
pub struct Relation { pub a: u32, pub b: u32, pub distance_m: f32, pub overlap: f32, pub kind: RelationKind }
pub struct Scene { pub entities: Vec<Entity>, pub relations: Vec<Relation>, pub zones: Vec<Vec<u32>> }
```

`EntityKind::Object(class)` — точка расширения: чашку/предмет добавит генерик-детектор фазы 7, движок отношений не меняется (object-agnostic).

### Контекст: что есть сейчас после фазы 5.2

```
PersonTracker
  ├─ IoU-матчинг (threshold 0.3)
  ├─ EMA bbox-сглаживание (α = 0.3)              → заменяет 6.1 (Kalman)
  ├─ identity: best-confidence persistence
  ├─ emotion: latest snapshot + decay (×0.8/кадр) → заменяет 6.4 (атрибутивный PoE)
  └─ timeout: 1800 кадров (30s @ 60fps)

mvm-emotion
  └─ classify() → (label: String, conf: f32)      → расширяет 6.3 (EmotionResult+entropy)

Workspace3D (Python prototype, core/workspace_3d.py) → порт в Rust = 6.2
mvm-fusion   🔶 заглушка                           → реляционный движок = 6.4
mvm-posture  🔶 заглушка
```

**Проблема EMA:** реактивен. При occluded-кадрах bbox не движется → IoU с новыми детекциями падает → трек теряется раньше срока.

**Проблема 3D:** метрические координаты считаются только в Python (`Workspace3D`), в нативном пайплайне их нет → fusion нечем мерить расстояния.

**Проблема confidence:** `max(softmax) = 0.8` может означать «уверен» (7 классов ≈ 0) или «не уверен» (7 классов ≈ 0.03 каждый). Без меры uncertainty downstream-агенты не различают эти случаи, и атрибутивная fusion не знает весов экспертов.

**Проблема fusion:** заглушка. Главная её задача — не слияние эмоций (это частный случай), а **реляционная**: понять, как сущности связаны в пространстве (близость, пересечение bbox, контекстные зоны).

---

### Задача 6.1 — SG-KF: Kalman Filter для bbox

**Цель:** заменить EMA на Linear Kalman Filter с constant-velocity motion model. Независима от остальных.

> **Реализация: декомпозированная форма (минимализм).** При диагональных Q, R, P восемь координат не связаны → 8D-фильтр **тождественно распадается на 4 независимых скалярных 2D-фильтра** (по одному на `cx, cy, w, h`, каждый со state `[pos, vel]`, `F=[[1,1],[0,1]]`, `H=[1,0]`, `S` — скаляр). **Никакого обращения матриц, никакого linalg-крейта** — ~50 строк скалярной арифметики. Матричная форма ниже — это спецификация; кодим декомпозицию. Тесты не меняются.
>
> **6.1 даёт 3D бесплатно:** сглаженный размер bbox (`w/h`) — вход для pinhole-глубины (6.2). 6.1 и 6.2 не дублируют работу, а складываются.

#### State vector (8D) — спецификация

```
x = [cx, cy, w, h, vcx, vcy, vw, vh]
      центр      размер     скорости
```

#### Матрицы

```
Transition F (dt=1):          Observation H:
[1 0 0 0 | 1 0 0 0]           [1 0 0 0 | 0 0 0 0]
[0 1 0 0 | 0 1 0 0]           [0 1 0 0 | 0 0 0 0]
[0 0 1 0 | 0 0 1 0]           [0 0 1 0 | 0 0 0 0]
[0 0 0 1 | 0 0 0 1]           [0 0 0 1 | 0 0 0 0]
[0 0 0 0 | 1 0 0 0]
[0 0 0 0 | 0 1 0 0]
[0 0 0 0 | 0 0 1 0]
[0 0 0 0 | 0 0 0 1]

Process noise Q: diag([1,1,1,1, 0.01,0.01,0.01,0.01])
Measurement noise R: diag([1,1,10,10])  // position точнее size
Initial P: diag([10,10,10,10, 100,100,100,100])  // высокая начальная неуверенность
```

#### Алгоритм на каждый тик

```
PREDICT (для каждого трека, до IoU matching):
  x_pred = F * x
  P_pred = F * P * F^T + Q

IoU MATCHING (используем predicted bbox, не последний observed):
  predicted_bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)

UPDATE (для matched треков):
  y = z - H * x_pred            // innovation
  S = H * P_pred * H^T + R      // innovation covariance
  K = P_pred * H^T * S^{-1}     // Kalman gain
  x = x_pred + K * y
  P = (I - K*H) * P_pred

MISSED (для unmatched треков):
  x = x_pred   // bbox продолжает двигаться по инерции
  P = P_pred   // неуверенность растёт
```

#### Что меняется в коде

```rust
// mvm-core/src/kalman.rs  (новый файл)
pub struct KalmanBBox {
    pub x: [f32; 8],   // state vector
    pub p: [[f32; 8]; 8],  // covariance
}

impl KalmanBBox {
    pub fn new(bbox: BBox) -> Self { ... }
    pub fn predict(&mut self) -> BBox { ... }   // возвращает predicted bbox
    pub fn update(&mut self, observed: BBox) { ... }
}

// mvm-core/src/types.rs  (PersonTrack)
pub struct PersonTrack {
    // ... existing fields ...
    pub kalman: KalmanBBox,  // заменяет EMA-логику
}
```

**"SG" = State-Gating:** при IoU-матчинге используем `kalman.predict()` вместо `t.bbox`. Это ключевое: трек «ищет» детекцию там, где предсказывает движение.

#### TDD тесты для 6.1

| Тест | Что проверяет |
|------|--------------|
| `kalman_predict_stationary_stays_put` | Неподвижный объект → predicted ≈ observed |
| `kalman_predict_moving_extrapolates` | Объект со скоростью → predicted смещён |
| `kalman_update_corrects_prediction` | Update tягает state к observation |
| `kalman_missed_frames_grow_uncertainty` | P растёт при missed-кадрах |
| `tracker_uses_predicted_bbox_for_iou` | IoU matching использует predicted, не observed |

---

### Задача 6.2 — Метрический 3D (native bbox-pinhole)

**Цель v1:** каждый трек получает нативную метрическую позицию (`position_x/y/z`, `distance`, `ws_confidence`), считаемую в Rust, без зависимости от Python `Workspace3D`. Метод — **depth-from-known-size** (`z = fx · real_size / pixel_size`).

> **Важно — сужение области (см. анализ ниже).** В native-пути нет MediaPipe-landmark'ов, поэтому это **НЕ порт `workspace_3d.py` 1:1**. Портируется bbox-вариант: масштаб от высоты face-bbox + known-size приор. Landmark-калибровка (торс 0.5м, межзрачковое 63мм) — точнее и строже, но требует нативной позы → **фаза 7**. Формулировка «порт workspace_3d.py» отменена.

#### Что уже готово (готовность 🟢 высокая)

| Компонент | Состояние |
|-----------|-----------|
| Интринзики | `Config.fx_ratio = 0.64` (откалибровано под камеру), `width/height = 640×480`. fx ≈ 410px. |
| Выходной плумбинг | `Detection`/`PersonTrack` уже несут `distance/position_x/y/z/ws_confidence` сквозь `parse_line → tracker → StateFile → JSON`. 6.2 **заполняет существующие поля**, новой магистрали не нужно. |
| Вход | face bbox (нормализован 0..1) в native-пути. |
| Оракул | Python `Workspace3D` работает рядом — эталон для валидации (исчезнет в фазе 7 → **строить и валидировать сейчас**). |

**Решение по типам:** 6.2 пишет в **плоские поля** (минимум плумбинга, немедленно видно в StateFile/JSON). Абстракцию `Entity{pos_3d,radius_m}` вводим на 6.4, где она нужна для рёбер графа.

#### Метод и формулы

```
z = (fx · real_size) / pixel_size          // глубина из известного размера
x = (px − cx) · z / fx,  cx = width/2       // обратная проекция в метры
y = (py − cy) · z / fy,  cy = height/2
pixel_size = (bbox.y2 − bbox.y1) · height   // денормализация в пиксели
real_size  = FACE_HEIGHT_M (= 0.19)         // популяционный fallback-приор
```

EMA-сглаживание `z` (α из Config). Валидация: восстановленная глубина в [0.3, 6.0]м → иначе `uncalibrated`, `ws_confidence = 0`.

#### ⚠️ Конверт точности и ограничения (трезвая оценка)

Метод легитимен (так SLAM бутстрапит масштаб), но это **грубая оценка, не метрология**. На этой камере:

| Дистанция | bbox height | джиттер ±2px | + разброс априора (σ≈12%) |
|-----------|-------------|--------------|--------------------------|
| 0.5 м | 156 px | ±0.6 см | ±6 см |
| 1.0 м | 78 px | ±2.6 см | ±12 см |
| 2.0 м | 39 px | ±10 см | ±24 см |
| 3.0 м | 26 px | ±23 см | ±36 см |

- **Доминирует не пиксельный шум, а разброс размера-априора:** `z ∝ real_size` → ±12% вариации высоты лица в популяции = ±12% глубины, неустранимо. Конверт: **±15% на 1м → ±25-30% к 3м**. Рабочая зона `0.5–3 м`.
- **Слабость метода — bbox как «линейка».** Face-bbox не привязан к анатомии: включает/исключает волосы, челюсть, уши при повороте головы/мимике → «глубина» прыгает при повороте на месте. Это **худшая инстанциация** метода; landmark-вариант (ф.7) строже методологически, а не просто точнее.
- **Тир клейма:** не «robotics-grade metric perception», а **грубый метрический + сильный относительный субстрат**. Робот-манипуляция требует landmark + stereo/мульти-камера (будущее). Тот же контракт `pos_3d` наполнят более точные методы — интерфейс переживёт смену реализации.

#### Рычаги, повышающие вес (заносится в реализацию, не «потом»)

1. **Identity-linked prior — главный рычаг точности.** Если трек опознан (есть identity-модуль + галерея), использовать *персональную* высоту лица вместо популяционной → априор ±12% коллапсирует в ±2-3%. Это межагентная синергия, которую монолитная depth-сеть не умеет — реальное преимущество ансамбля. Популяционный 0.19м — fallback для unknown.
2. **Ординал/относительная глубина — первичный продукт.** «Кто ближе», «дистанция сокращается» робастны там, где абсолют врёт (ошибка априора систематична на класс → в разностях частично сокращается). Абсолют — вторичный, всегда с `ws_confidence`.

#### Что меняется в коде

```rust
// mvm-core/src/space.rs  (новый файл)
pub struct Pinhole { pub fx: f32, pub fy: f32, pub cx: f32, pub cy: f32 }
impl Pinhole {
    pub fn from_config(cfg: &Config) -> Self { ... }           // fx = fx_ratio·width, cx = width/2
    pub fn distance_from_size(&self, pixel_size: f32, real_size: f32) -> f32 { ... } // z = fx·real/pixel
    pub fn pixel_to_3d(&self, px: f32, py: f32, z: f32) -> [f32; 3] { ... }          // обратная проекция
}

// Config: новые поля (с дефолтами)
//   face_height_m: f32 = 0.19      // популяционный приор
//   depth_ema_alpha: f32 = 0.3     // сглаживание z
//   (фаза 7: torso_length_m, interocular_m)
```

`PersonTrack.position_*`, `distance`, `ws_confidence` заполняются после tracker-update; identity-prior подменяет `real_size`, если трек опознан.

#### Строгий план действий 6.2 (порядок исполнения)

| # | Шаг | Гейт |
|---|-----|------|
| **6.2.0** | **Эмпирический замер конверта.** Рулетка на 0.5/1/2/3м, лицо в кадр, записать bbox-height (через `--dump-json`) + сравнить с Python `position_z`. 20 мин. | Реальная кривая ошибки на *этой* камере задокументирована. Без неё не доверяем модели. |
| **6.2.1** | `space.rs`: `Pinhole`, `distance_from_size`, `pixel_to_3d`. TDD-first. | Тесты геометрии зелёные. |
| **6.2.2** | `Config`: `face_height_m`, `depth_ema_alpha`. | Дефолты, test_config обновлён. |
| **6.2.3** | Хук в tracker/cli: после update заполнять `position_*`/`distance`/`ws_confidence` нативно (populated `Pinhole::from_config`). | Native 3D виден в StateFile и `--dump-json`. |
| **6.2.4** | Identity-linked prior: опознанный трек → персональный `real_size`. | Опознанный трек точнее unknown (проверка на замере 6.2.0). |
| **6.2.5** | Сверка с оракулом: Rust face-bbox vs Python torso. | Порядок величины и **монотонность** совпадают (точного равенства НЕТ — разные методы). |

#### TDD тесты для 6.2

| Тест | Что проверяет |
|------|--------------|
| `distance_from_size_pinhole` | известный размер + fx → корректная z (метры) |
| `pixel_to_3d_backprojection` | центр кадра → x≈0,y≈0; угол → знак x,y верный |
| `depth_ema_smooths` | EMA: z стабилизируется за N кадров |
| `depth_rejects_implausible` | z вне [0.3, 6.0]м → uncalibrated, ws_conf=0 |
| `closer_object_larger_bbox_smaller_z` | больше bbox → меньше дистанция (монотонность) |
| `identity_prior_overrides_population` | опознанный трек берёт персональный real_size |

---

### Задача 6.3 — Uncertainty Estimation для emotion

**Цель:** `classify()` возвращает не только `(label, prob)`, но и `entropy` / `uncertainty` — меру сомнения модели.

#### Теория

Текущий выход: `softmax(logits)` → вектор p[8], берём `max(p)` как confidence.

Проблема: softmax всегда суммируется в 1.0, даже если все logits = 0. Нет способа отличить «уверен» от «равномерно не уверен».

**Практическое решение (без переобучения ONNX):**

```
// Из выходного вектора p[8] (softmax probs):
entropy    = -Σ p_i * log2(p_i) / log2(K)   // [0, 1], 0=уверен, 1=равномерно
uncertainty = entropy
calibrated_conf = max(p) * (1 - uncertainty)  // скорректированная уверенность
```

Примеры:
```
p = [0.8, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]  → entropy ≈ 0.32 → cal_conf ≈ 0.54
p = [0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.03]    → entropy ≈ 0.93 → cal_conf ≈ 0.02
```

**True EDL (Sensoy et al. 2018) — для дальнейшего:**
Требует переобучения с loss: `L = MSE(p, y) + λ * KL(Dir(α) || Dir(1))`.
Выходной слой: `softplus(logits) + 1` → Dirichlet α → `uncertainty = K / Σ α_i`.
Это будет в ONNX-файле, не в Rust. Откладывается на фазу 7+.

#### Что меняется в коде

```rust
// mvm-emotion/src/lib.rs
pub struct EmotionResult {
    pub label: String,
    pub prob: f32,           // max(softmax)
    pub distribution: [f32; 8], // полный вектор softmax
    pub entropy: f32,        // нормализованная [0, 1]
    pub uncertainty: f32,    // = entropy (пока)
    pub calibrated_conf: f32, // prob * (1 - uncertainty)
}

// classify() возвращает EmotionResult вместо (String, f32)
pub fn classify(&mut self, face: &DynamicImage) -> Result<EmotionResult> { ... }
```

**Влияние на emotion_wins():**

```rust
// mvm-cli/src/main.rs — было:
fn emotion_wins(rust_conf: f32, current_conf: Option<f32>) -> bool {
    rust_conf > current_conf.unwrap_or(0.0)
}

// станет:
fn emotion_wins(rust: &EmotionResult, py_conf: Option<f32>) -> bool {
    rust.calibrated_conf > py_conf.unwrap_or(0.0)
}
```

#### TDD тесты для 6.3

| Тест | Что проверяет |
|------|--------------|
| `entropy_zero_for_one_hot` | p=[1,0,0,...] → entropy=0.0 |
| `entropy_one_for_uniform` | p=[0.125; 8] → entropy=1.0 |
| `calibrated_conf_penalizes_uncertain` | Высокая prob + высокая entropy → cal_conf < prob |
| `emotion_result_distribution_sums_to_one` | Σ distribution ≈ 1.0 |
| `emotion_wins_uses_calibrated_conf` | При равной raw prob побеждает менее энтропийный |

---

### Задача 6.4 — Fusion: реляционный scene-graph + атрибутивный PoE

**Цель:** `mvm-fusion` из заглушки → ядро кластера. Две операции; **пространственная — главная**.

#### 6.4a — Реляционная fusion (основная): scene-graph

Над `Vec<Entity>` (3D-позиции из 6.2, bbox из 6.1) строим граф. Чистая арифметика, ноль зависимостей:

```
Для каждой пары (i, j):
  distance_m = ‖pos_3d_i − pos_3d_j‖        // 3D евклид
  overlap    = IoU(bbox_i, bbox_j)          // 2D

Ребро есть, если distance_m < d_thresh ИЛИ overlap > 0.
Контекстные зоны = связные компоненты графа рёбер (union-find, ~30 строк).
```

**Семантика:** «голова + тело + рука + чашка» в одной зоне = связная компонента. Дистанции на рёбрах = тонкий контекст взаимодействия (рука близко к чашке → тянется/держит). `RelationKind` — атрибут ребра (`Near | Overlapping | PartOf`).

**Object-agnostic:** движок не знает тип узла. Чашку добавит генерик-детектор фазы 7 как `EntityKind::Object` — рёбра считаются той же математикой, код не меняется. Сейчас обкатываем на person/face/body-part.

#### 6.4b — Атрибутивная fusion (частный случай): Product-of-Experts

Source-agnostic комбинатор распределений. Единственный публичный контракт:

```rust
pub struct Expert { pub distribution: [f32; 8], pub weight: f32 }  // weight = 1 - uncertainty, clamp[0,1]

// Log-linear pooling = PoE. Нормировка = softmax(Σ w·ln p). Численно устойчиво.
pub fn fuse(experts: &[Expert]) -> FusedEmotion;
```

Один `fuse()`, два вызова:

```rust
// ФАЗА 6 — temporal: эксперты = (накопленный приор трека) + (наблюдение кадра).
// Заменяет хак ×0.8/кадр принципиальным забыванием.
track.emotion_dist = fuse(&[
    Expert { distribution: track.emotion_dist, weight: LAMBDA },          // приор + забывание
    Expert { distribution: emo.distribution,  weight: 1.0 - emo.uncertainty }, // кадр
]).distribution;

// ФАЗА 7 — cluster: эксперты = несколько нативных моделей. Та же функция, длиннее слайс.
fuse(&[expert_emotieff, expert_model2, ...]);
```

> **Подвох — схлопывание.** Наивный PoE распределения с самим собой концентрирует массу → за сотни кадров эмоция «застынет» в one-hot. Поэтому temporal-форма **рекурсивна с забыванием**: вес приора `LAMBDA < 1` (экспоненциальное забывание в лог-пространстве). Cluster-вызов забывания не имеет, веса = уверенности экспертов.

**Mono-Confidence** = `max(p_fused)`. **Holo-Confidence** = весь вектор `p_fused[8]` наружу (новый JSON-field `emotion_distribution: [f32;8]` в `to_json_line()`). Деградация: при бинарных весах `fuse()` ≡ текущий `emotion_wins()`.

```rust
// mvm-fusion/src/lib.rs
pub struct FusedEmotion { pub label: String, pub prob: f32, pub distribution: [f32;8], pub uncertainty: f32 }
```

#### TDD тесты для 6.4

| Тест | Что проверяет |
|------|--------------|
| `relation_distance_3d_euclidean` | дистанция между двумя `pos_3d` корректна |
| `zone_connected_components` | пересекающиеся/близкие сущности → одна зона |
| `zone_far_entities_separate` | далёкие сущности → разные зоны |
| `object_agnostic_relations` | граф считается для смешанных `EntityKind` |
| `fuse_identical_distributions_unchanged` | эксперты идентичны → p_fused ≈ вход |
| `fuse_zero_weight_expert_ignored` | weight=0 (uncertainty=1) → эксперт исключён |
| `fuse_distribution_sums_to_one` | Σ p_fused ≈ 1.0 (нормировка) |
| `fuse_disagreement_raises_uncertainty` | расходящиеся эксперты → выше entropy результата |
| `temporal_no_collapse_with_forgetting` | LAMBDA<1 → распределение не схлопывается за N кадров |

---

### Зависимости и порядок реализации

```
6.1 Kalman bbox   ──┬──────────────────── независима (даёт сглаженный bbox)
6.2 Метрический 3D ─┴── использует bbox из 6.1 → pos_3d
6.3 Uncertainty   ──────────────────────── независима (даёт веса экспертов)
6.4 Fusion        ── 6.4a нужны pos_3d (6.2); 6.4b нужен uncertainty (6.3)
```

**Рекомендуемый порядок:** 6.1 → 6.2 (геометрия, складываются) → 6.3 → 6.4.

### Решённые вопросы (зафиксировано)

1. ✅ **Kalman vs MLP** → **Kalman** (декомпозированный, 4× скаляр). MLP требует датасет и второй inference-путь → нарушает «чистую арифметику». Roadmap-формулировка «MLP-предсказатель» отменена.
2. ✅ **Entropy vs True EDL** → **entropy-approximation**. True EDL требует переобучения ONNX (GPU-день) → откладывается на 7+. Достаточно для Hermes.
3. ✅ **Python uncertainty / patch bridge** → **не нужен**. Атрибутивная fusion в фазе 6 — temporal (один нативный эксперт во времени), не cross-source с Python. Bridge не трогаем.
4. ✅ **Holo-Confidence downstream** → **да**, добавляем `emotion_distribution: [f32;8]` в `to_json_line()`. Downstream начинает читать вектор, а не только `emotion: String`.

### Открытые вопросы для brainstorm

1. **Параметры в Config:** Q, R (6.1), known-size приоры + α (6.2), LAMBDA-забывание (6.4). Единое решение — все поля `Config` с дефолтами (статичная сцена-монитор ≠ динамичная). Подтвердить.
2. **3D-позиция сущности — точка или объём?** Старт: центроид `[f32;3]` + `radius_m` (из bbox+глубины). Хватает для дистанций/зон; полноценный объём — позже. Ок?
3. **`Scene` в JSON целиком?** Если да — `to_json_line()` отдаёт entities+relations+zones, и это новый стабильный контракт для Hermes (структурированный 3D-снимок вместо плоского списка людей). Подтвердить формат.
4. **Где живёт KalmanBBox / Calibration:** в `PersonTrack` (чище для трейтов) или внешним map в `PersonTracker` (тестируемость без полного трека)?

---

## Фаза 7 — Снос Python: детальный план

> **Для:** brainstorm / делегирование Claude Code. **Статус:** фаза 6 закрыта, провалидирована на камере (2026-06-25), re-link сделан. 106 тестов зелёные.
> **Принцип фазы:** убрать Python из боевого пути. Каждый шаг снимает одну ответственность Python; деградации функционала нет; бинарник остаётся самодостаточным.

### Цель и мотивация (эмпирическая)

Live-кампания дала точные приоритеты — фаза 7 чинит ровно то, что вскрылось (см. «Известные проблемы» + «Live-валидация»):

| Находка кампании | Что снимает фаза 7 |
|------------------|--------------------|
| FPS ~1–2 (узел — Python bridge + disk round-trip кадров) | нативный захват/инференс → цель 30 fps, `<50ms` |
| track-id churn (#6, частично закрыт label-re-link) | остаток добивает FPS; полный фикс — эмбеддинг-re-link (7.5) |
| posture только у 1 (#3 — MediaPipe `poses=1` несмотря на `num_poses=3`) | нативная мульти-persona поза (7.3) |
| `face_height_m=0.44` привязан к **bbox-конвенции SCRFD** | рекалибровка под YuNet-боксы (7.1) |
| identity false-positive (#5 — чужой→Вадим@0.66) | SFace-эмбеддинг + margin/threshold в трекере (7.5) |

### Что УЖЕ нативно (важно для инкрементальности)

- **YuNet (`mvm-detect`, 228 КБ) загружается в `--native`, но в боевом цикле НЕ вызывается.** ⚠️ *Исправлено аудитом 2026-06-25:* `detector.detect()` зовётся **только** в `build_gallery()` (`main.rs:184`) при сборке галереи. В per-frame цикле детекции **100% из Python SCRFD** (`cam.read()` → JSON → `parse_line` → tracker). YuNet в live-пути спит.
- **Кадр Rust уже читает с диска:** bridge пишет его через `--write-frame` в `/tmp/mvm_frame.jpg`, Rust грузит для SFace/EmotiEffLib кропов по SCRFD-bbox. → нативная детекция = переиспользовать тот же кадр, без новой магистрали.
- **YuNet 5 face-лендмарков не парсятся:** `detect.rs` извлекает только bbox (`cls/obj/bbox` головы), `kps_*` игнорируются. Бесплатный актив (face-alignment для SFace, interocular-3D), но **к позе тела отношения не имеет**.
- **Нативны:** emotion (EmotiEffLib), identity (SFace, **128d** эмбеддинг, `embed()` готов), 3D (pinhole, `space.rs`), трекинг (Kalman + re-link), fusion (`build_scene` + temporal PoE).
- **Остаётся Python:** захват камеры (cv2 FFMPEG + ROTATE_180), первичная детекция (SCRFD), поза (MediaPipe + PoseVAE).

### Аудит готовности (2026-06-25)

> План сверен с фактическим кодом и моделями перед стартом фазы. Вывод: декомпозиция здорова и эмпирически обоснована, но в формулировках было 6 расхождений с кодом, а 2 задачи (7.2, 7.3) несут реальные неизвестные. Ниже — расхождения (исправлены по тексту плана), наличный инструментарий и список пробелов приобретения.

#### Расхождения «план ↔ код» (исправлены)

| # | Было в плане | Факт в коде | Следствие |
|---|--------------|-------------|-----------|
| D1 | 7.5: «SFace **512d**-эмбеддинг» | `mvm-identity/src/lib.rs:3` — **128d** L2-norm | Поправлена цифра; математика re-link та же |
| D2 | 7.5: «пробросить эмбеддинг» (как новую работу) | `embed()` (`lib.rs:92`) **уже возвращает** `Vec<f32>` | Примитив готов → 7.5 меньше; остаётся довести до трекера + cosine-re-link |
| D3 | «YuNet **уже запускается** в native-пути» | `detect()` только в `build_gallery()` (`main.rs:184`); в live-цикле SCRFD | 7.1 = «подключить YuNet в live-цикл», крупнее «повысить выход» |
| D4 | (молчание) | YuNet **5 face-лендмарков не парсятся** (`detect.rs`: только bbox) | Бесплатный актив для SFace-alignment/interocular-3D; к позе тела нерелевантен |
| D5 | 7.3: «ONNX-лендмарки... порт PoseVAE» | **Нет ONNX-модели лендмарков тела в репо.** `pose_landmarker.task` (5.6 МБ) — TFLite, `ort 2.0.0-rc.12` его **не грузит** | Главный пробел приобретения 7.3 (G1) |
| D6 | (неявно: «порт PoseVAE») | Поза — **3 стадии**: landmark-модель → VAE-encode(99→z32) → Classifier(32→64→32→6); метки **русские** (`class_names_ru`) | Порт = модель(G1)+экспорт весов(G2)+RU→EN маппинг(G7) |

#### Аудит доказательств (на чём стоит каждая задача)

| Задача | Нужность доказана | Реализуемость доказана | Вердикт |
|--------|-------------------|-------------------------|---------|
| 7.1 детектор | ✅ измерено (SCRFD/Python = горло FPS) | ⚠️ YuNet работает (5 тестов), но **нет A/B покрытие vs SCRFD** на живых | гейт не оснащён (G4) |
| 7.2 камера | ✅ риск доказан жёстко (cv2 потребовал `CAP_FFMPEG`) | ❌ **ноль доказательств**: нет spike, Cargo.lock пуст | чистый неизвестный (G3) |
| 7.3 поза | ✅ измерено (#3: MediaPipe `poses=1`, vis 0.96–1.0) | ❌ **ключевая модель отсутствует** (D5) | самый дорогой (G1+G2+G6) |
| 7.5 identity | ✅ сильнее всех (Вадим=1.00 vs чужой=0.66 — разделимо) | ✅ **примитив `embed()` готов** (D2) | наиболее готовая |
| 7.4 снос bridge | — (производная) | зависит от 7.1–7.3 | тривиальна последней |

#### Наличный инструментарий (активы)

- **`tools/cam_test.py`** — harness живых сценариев: 6 пресетов (`single_sit`, `walk_near_far`, `leave_return`, `occlusion`, `two_people`, `unknown_face`), GO-сигнал по первому кадру, **replay-режим**, структурированный вердикт (FPS, track-churn, dist-монотонность, zone-size, posture-coverage). → **приёмочный гейт всей фазы 7**.
- **`tools/calib_3d.py`** — рулеточная калибровка `face_height_m` (back-solve `real_eff = dist·px/fx`). → **готов к рекалибровке под YuNet-боксы (7.1)**, запустить заново.
- **`embed()`** (mvm-identity) — примитив 7.5.
- **Оракул жив:** Python `workspace_3d.py` (torso 0.5м, interocular 0.063м) рядом для валидации 3D, пока bridge не снесён.

#### Пробелы приобретения/постройки (gap-list)

| # | Чего нет | Для чего | Тип |
|---|----------|----------|-----|
| **G1** | ONNX-модель лендмарков тела (RTMPose/MoveNet/BlazePose) | 7.3 стадия 1 | приобретение модели |
| **G2** | Скрипт экспорта весов `pth→ONNX/raw` для PoseVAE + PoseClassifier | 7.3 стадии 2–3 | инструмент |
| **G3** | Камера-spike (`nokhwa`/`v4l`/`ffmpeg-next`, чтение `/dev/video0` при живом PipeWire) | 7.2 — снять риск ПЕРВЫМ | PoC |
| **G4** | A/B-гейт детектора: YuNet-primary vs SCRFD по покрытию/стабильности | 7.1 гейт | расширение harness |
| **G5** | Численный FPS-baseline (сейчас только проза «~1–2 fps») | регресс-метрика фазы | замер |
| **G6** | Количественный оракул точности позы (cam_test даёт coverage %, не correctness) | 7.3 гейт | данные/разметка |
| **G7** | Маппинг RU→EN меток позы | 7.3 таксономия | тривиально |

#### Рекомендованный порядок по итогам аудита

- **7.5 — быстрый первый win:** чистый Rust, ноль приобретений, нужность доказана сильнее всех, примитив `embed()` готов.
- **7.2 — spike (G3) параллельно и рано:** риск PipeWire дешёв проверить, но дорог поздно.
- **7.3 — кандидат «вынести за scope фазы»** (открытое решение №1): G1+G2+G6 — самая дорогая ветка; MediaPipe single-pose покрывает основной юзкейс.

### Задачи

#### 7.1 — YuNet как первичный детектор 🟡 *крупнее, чем казалось (D3)*
⚠️ *Уточнено аудитом:* YuNet в live-цикле **не вызывается** — только в `build_gallery()`. Задача = **подключить `detector.detect()` в per-frame цикл** на уже-существующем `/tmp/mvm_frame.jpg` и сделать его выход **источником детекций** для трекера (вместо SCRFD-bbox из JSON). Магистраль (кадр на диске) уже есть, новой не нужно. Камера/поза пока Python.
- **Бонус-актив (D4):** заодно можно распарсить 5 face-лендмарков YuNet (`kps_*` головы сейчас игнорируются) → face-alignment для SFace + interocular-3D. Опционально, не блокирует 7.1.
- **Тонкость — рекалибровка:** `face_height_m=0.44` откалиброван под SCRFD-боксы (≈2× анатомии). YuNet-боксы **другой плотности** → прогнать `tools/calib_3d.py` заново, обновить константу. Без этого нативная 3D собьётся.
- **Гейт:** нативные YuNet-детекции по покрытию/стабильности ≥ SCRFD на живых данных; 3D-дистанция совпадает после рекалибровки.

#### 7.2 — Нативный захват камеры в Rust 🟡 *главный риск*
Заменить cv2-граб на Rust-крейт; убрать disk round-trip (perf-win).
- **Ключевой риск — PipeWire/Wayland (Ubuntu 24+).** Ровно поэтому cv2 потребовал `CAP_FFMPEG` (V4L2 падал при эксклюзивном захвате PipeWire). Rust-путь должен сосуществовать с PipeWire. Кандидаты: `nokhwa`, `v4l`, `ffmpeg-next`. **Сделать spike ПЕРВЫМ** — снять риск до планирования остального.
- **Тонкость:** ROTATE_180 (камера перевёрнута) перенести в Rust.
- **Гейт:** Rust стабильно читает `/dev/video0` на этой машине при работающем PipeWire, fps растёт.

#### 7.3 — Нативная поза 🔴 *самое дорогое (D5, D6)*
MediaPipe в Rust нет; ⚠️ *аудит уточнил — это 3-стадийная цепочка, и стадия 1 отсутствует.*
```
33 body-landmarks (99=33×3) → PoseVAE.encode → z(32) → PoseClassifier(32→64→32→6) → 6 поз
                                     └→ reconstruction_error → plausibility
```
- **Стадия 1 (G1):** ❌ **ONNX-модели лендмарков тела в репо НЕТ.** `pose_landmarker.task` (5.6 МБ) — TFLite, `ort` его не грузит. Достать RTMPose / MoveNet / BlazePose в ONNX.
- **Стадия 2 (G2):** PoseVAE-encoder `99→128→64→μ32`, веса есть (`vposer_coco33.pth`, 206 КБ). Чистый matmul — тривиально в Rust, но **нужен экспорт весов** pth→ONNX/raw.
- **Стадия 3 (G2):** PoseClassifier `32→64→32→6`, веса есть (`pose_classifier.pth`, 21 КБ). Тривиально.
- **Метки RU→EN (G7):** в bridge метки **русские** (`class_names_ru`) — при порте сразу маппить в английские (согласовать с emotion-таксономией).
- **Мульти-persona:** инференс по кропам-на-детекцию (по одному на YuNet-bbox) → чинит #3 (`poses=1`).
- **Гейт:** нативная поза на 2+ людях по качеству ≥ MediaPipe single-pose. ⚠️ количественного оракула точности нет (G6) — cam_test даёт только coverage %.
- **Рекомендация аудита:** кандидат «вынести за scope фазы» (открытое решение №1) — самая дорогая ветка (G1+G2+G6).

#### 7.4 — Удалить bridge 🟢
Камера+детекция+поза нативны → снести `mvm_bridge.py`, убрать зависимость от venv. Обновить `cargo xtask bundle`.
- **Гейт:** `mvm-cli --native` отрабатывает end-to-end без единой строки Python.

#### 7.5 — SFace-эмбеддинг в трекер (полный фикс #5/#6) 🟢 *готовее всех (D1, D2)*
Пробросить SFace **128d-эмбеддинг** в трекер (сейчас доходит только метка+conf). ⚠️ *Аудит: примитив `embed()` (`mvm-identity/src/lib.rs:92`) уже извлекает вектор — задача = довести его до `PersonTrack`, не строить с нуля.*
- **margin-based identity (#5):** метка только если топ-1 заметно выше топ-2 / абсолютного порога, иначе `unknown`. Данные подтверждают разделимость: реальный Вадим=1.00 vs чужой≈0.66.
- **cosine-re-link (#6):** заменить label-only re-link (текущий, с оговоркой о слиянии двух «Вадимов») на cosine эмбеддингов → корректно различает людей.
- **Почему первый:** чистый Rust, ноль приобретений, нужность доказана сильнее всех → быстрый win.

### Порядок и зависимости

```
7.2 камера (spike-риск) ──┐
7.1 детектор (готов)    ──┴── независимы, можно параллельно
7.3 поза ──────────────────── после 7.1 (нужны нативные кропы-bbox)
7.4 снос bridge ───────────── после 7.1+7.2+7.3 (всё нативно)
7.5 эмбеддинг-identity ────── независим; усиливает re-link фазы 6
```

**Рекомендованный старт (уточнён аудитом 2026-06-25):** 7.5 как быстрый Rust-win (примитив `embed()` готов, ноль приобретений) + spike 7.2 (G3, снять главный риск рано) параллельно. 7.1 (детектор + рекалибровка) следом. 7.3 — вынесена за scope (см. ниже).

---

## Поэтапный план исполнения (финал, scope: Python-light)

> **Решение по scope зафиксировано 2026-06-25:** **7.3 (нативная поза) вынесена за scope фазы 7.** Причина — единственная ветка с приобретением модели (G1) + экспортом весов (G2) + отсутствием оракула (G6); блокирует отгрузку. Фаза 7 финиширует как **Python-light**: камера, детекция, 3D, identity, fusion — нативны и быстры (главные боли FPS и track-churn закрыты), поза остаётся на MediaPipe ещё фазу. Полный **Python-zero** (поза + 7.4-full) — следующая короткая фаза 7.5.

**Критический путь:** `Этап 0 → (7.5 ∥ 7.1) → 7.2 → 7.4-light`.

```
Этап 0 (оснащение+риск) ──┬─→ 7.5 identity ─────────────┐
   G5 baseline            ├─→ 7.1 детектор (G4) ─────────┼─→ 7.4-light
   G3 камера-spike ───────┴─→ 7.2 камера (если spike ок)─┘   (bridge → pose-only)
```

### Этап 0 — Оснащение и снятие риска *(всё параллельно, до вложений в остальное)*

| Шаг | Действие | Гейт выхода | Пробел |
|-----|----------|-------------|--------|
| 0.1 | **FPS-baseline**: текущий Python-путь через `cam_test.py` по 6 сценариям → `logs/baseline_phase7.md` | численный fps (не «~1–2»), репер для win | G5 |
| 0.2 | **Камера-spike**: bin (`examples/` или `xtask spike-cam`), кандидат `nokhwa` (фолбэк `v4l`→`ffmpeg-next`), читает `/dev/video0` под живым PipeWire | 🟢 Rust стабильно читает камеру ИЛИ 🔴 провал зафиксирован → 7.2 пересматривается | G3 |

**Выход:** известен baseline и реальность нативной камеры. Без 0.2 этап 3 не планируется дальше.

### Этап 1 — 7.5 SFace-identity в трекер *(быстрый Rust-win, ∥ этапу 0)*

TDD-first, чистый Rust, ноль приобретений. Примитив `embed()` готов.

| Шаг | Действие | Гейт |
|-----|----------|------|
| 1.1 | `PersonTrack.embedding: Option<Vec<f32>>` (128d); enrichment пишет `id.embed(&face)` | тест: трек несёт эмбеддинг |
| 1.2 | **margin-identity (#5):** метка только если cos(top1)−cos(top2) > margin ИЛИ cos > abs_thresh, иначе `unknown` | тест: Вадим=1.00 проходит, чужой=0.66 → `unknown` |
| 1.3 | **cosine-re-link (#6):** label-only re-link → cosine эмбеддингов в `relink_window_frames` | тест: два разных эмбеддинга не сливаются в id |
| 1.4 | Live: `cam_test.py --scenario unknown_face` + `leave_return` | #5 чужой≠Вадим; #6 id переживает возврат без слияния |

**Закрывает:** #5, #6. **Гейт фазы:** `unknown_face` не даёт «Вадим@0.66».

### Этап 2 — 7.1 YuNet как первичный детектор

YuNet НЕ в live-цикле (D3) → «подключить», не «повысить». Магистраль `/tmp/mvm_frame.jpg` уже есть.

| Шаг | Действие | Гейт |
|-----|----------|------|
| 2.0 | **G4: A/B-harness** — режим cam_test: покрытие/стабильность YuNet vs SCRFD на одном логе | есть метрика сравнения |
| 2.1 | Подключить `detector.detect(&frame)` в per-frame цикл `main.rs` | YuNet-bbox в дампе |
| 2.2 | YuNet — **источник детекций** для трекера вместо SCRFD-bbox из JSON | трекинг по YuNet-боксам |
| 2.3 | **Рекалибровка** `face_height_m` под YuNet-боксы: `calib_3d.py` заново (0.5/1/2/3м) | новая константа, 3D-монотонность совпадает с оракулом |
| 2.4 | *(опц., бонус D4)* распарсить 5 face-лендмарков YuNet → SFace-alignment + interocular-3D | отдельный гейт, не блокирует |

**Закрывает:** привязку `face_height_m` к SCRFD; даёт нативную детекцию. **Гейт:** покрытие YuNet ≥ SCRFD (G4), 3D совпадает после рекалибровки.

### Этап 3 — 7.2 Нативная камера *(только если spike 0.2 зелёный)*

Убрать disk round-trip — основной FPS-win.

| Шаг | Действие | Гейт |
|-----|----------|------|
| 3.1 | Промоутить spike в `mvm-pipeline`: `CameraBackend::Native` рядом с Python-bridge | бэкенд переключается флагом |
| 3.2 | `ROTATE_180` в Rust | картинка не перевёрнута |
| 3.3 | Убрать disk round-trip — кадр в память, без `/tmp/mvm_frame.jpg` | enrichment берёт кадр из памяти |
| 3.4 | Live по 6 сценариям | fps↑ vs baseline (0.1), churn не вырос |

**Закрывает:** FPS-горло. **Гейт:** стабильное чтение под PipeWire + fps↑ против baseline.

### Этап 4 — 7.4-light Снос Python из hot-path

Камера+детекция+identity нативны → bridge ужимается до **pose-only** (поза 7.3 за scope).

| Шаг | Действие | Гейт |
|-----|----------|------|
| 4.1 | Ужать `mvm_bridge.py` до pose-only: камеру/детекцию/frame-write убрать | bridge не трогает камеру/детекцию |
| 4.2 | `mvm-cli`: камера+детекция нативны; поза дотягивается из ужатого bridge как опциональный сигнал | hot-path без Python |
| 4.3 | Обновить `cargo xtask bundle` | dist/mvm собирается |
| 4.4 | E2E `cam_test.py` по 6 сценариям | боевой hot-path (камера→детекция→3D→identity→fusion) = 0 строк Python |

**Закрывает фазу 7 как Python-light.** venv остаётся только ради позы. **Гейт фазы:** hot-path нативный, fps↑, #3/#5/#6 в состоянии «закрыто или вынесено».

### Что переносится в фазу 7.5 (Python-zero)

7.3 целиком (G1 модель лендмарков + G2 экспорт весов + G6 оракул точности + RU→EN метки) → затем 7.4-full (полный снос bridge, удаление venv). Триггер старта 7.5 — появление пригодной ONNX-модели лендмарков тела.

### Открытые решения (остаток)

1. ✅ **Scope фазы — РЕШЕНО (2026-06-25):** Python-light, 7.3 вынесена в фазу 7.5. См. выше.
2. **Камера (7.2):** `nokhwa` / `v4l` / `ffmpeg-next` — выбор крейта определяется результатом spike 0.2 (G3).
3. **Модель позы (7.3):** RTMPose / MoveNet / BlazePose ONNX (G1) — решается на старте фазы 7.5.

---

## Команды

```bash
# Сборка и тесты
cargo build --release              # → target/release/mvm-cli (25 МБ)
cargo test --workspace             # 47 тестов

# Запуск (требует Python MVM venv)
target/release/mvm-cli --once                    # Python bridge
target/release/mvm-cli --native --once           # + Rust emotion/identity
target/release/mvm-cli --native --once --dump-interval 1  # быстрый тест

# Bundle
cargo xtask bundle                 # → dist/mvm (модели embedded)
```

## Контракт модуля

```rust
trait Module {
    type Input;
    type Output;
    fn init(config: &Config) -> Result<Self>;
    fn process(&mut self, input: &Self::Input) -> Result<Self::Output>;
    fn metadata(&self) -> ModuleMetadata;
}
```

## Микро-агенты (целевой состав)

| Агент | Модель | Сигнал | Где сейчас |
|-------|--------|--------|------------|
| **Detect** | YuNet ONNX 228KB | bbox лица | Rust ✅ |
| **Identity** | SFace ONNX 37MB | identity + confidence | Rust ⚠️ (gallery пустая) |
| **Emotion** | EmotiEffLib B0 16MB | 8 эмоций + confidence | Rust ⚠️ (затирает HSEmotion) |
| **Posture** | PoseVAE 48.6K | 6 поз + confidence + plausibility | Python bridge |
| **3D** | Workspace3D | (x,y,z)м + ws_confidence | Python bridge |

## Терминология

- **Микро-агент** — независимый детектор с confidence
- **Structured Decomposed PTT** — архитектурная идентичность MVM
- **Fusion** — логический слой, собирающий сигналы агентов в описание
- **Workspace 3D** — метрическая система координат в метрах
- **Confidence-first** — каждый сигнал имеет числовую уверенность
- **Prediction error** — `||predicted − observed||` → естественный confidence-сигнал
