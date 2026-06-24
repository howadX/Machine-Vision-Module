# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*

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

## Текущий статус (24 июня 2026, фаза 5 закрыта → старт фазы 6)

### Что работает при `mvm-cli --native --once --dump-json`

```
Камера /dev/video0 (CAP_FFMPEG — обход PipeWire)
    │
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
    ├─ EmotiEffLib B0  → emotion (confidence-based fallback: Rust > Python только если conf выше)
    ├─ SFace           → identity (gallery: Вадим, 8 эталонных фото)
    ├─ PersonTracker   → temporal fusion (IoU match, identity persistence)
    ├─ StateFile       → ~/.hermes/profiles/sight/workspace/current_state.md
    ├─ dump_table()    → ASCII таблица в терминал
    └─ --dump-json     → одна JSON-строка на stdout каждый тик ← Фаза 5.2.1
```

### Бинарники

| Файл | Размер | Команда сборки |
|------|--------|----------------|
| `target/release/mvm-cli` | **25 МБ** | `cargo build --release` |
| `dist/mvm` | **26 МБ** | `cargo xtask bundle` (актуален) |

### Крейты

| Крейт | Статус | Содержимое | Тесты |
|-------|--------|------------|-------|
| **mvm-core** | ✅ production | BBox, Detection, PersonTrack, PersonTracker, Config, Module trait | 23 |
| **mvm-detect** | ✅ production | YuNet ONNX (228 КБ), CHW BGR preprocessing, set_score_threshold() | 5 |
| **mvm-identity** | ✅ production | SFace ONNX (37 МБ), gallery Вадим, cosine match | 5 |
| **mvm-emotion** | ✅ production | EmotiEffLib B0 ONNX (16 МБ), 8 классов VGAF, confidence-based | 3 |
| **mvm-posture** | 🔶 заглушка | Пустой Module trait | 0 |
| **mvm-fusion** | 🔶 заглушка | Пустой Module trait | 0 |
| **mvm-pipeline** | ✅ production | Camera (Python bridge subprocess), parse_line | 5 |
| **mvm-cli** | ✅ production | main, StateFile, dump_table(), --dump-json, --native/--once | 14 |
| **xtask** | ✅ | `cargo xtask bundle` → dist/mvm | 1 |

**Всего: 56 тестов, все зелёные.**

### Известные проблемы

| # | Проблема | Причина | Приоритет |
|---|----------|--------|-----------|
| 1 | **EmotiEffLib vs HSEmotion — разные датасеты** | Оба EfficientNet-B0, VGAF (Rust) vs AffectNet (Python). Нужен A/B на живых данных | 🟡 |
| 3 | **mvm-posture, mvm-fusion — заглушки** | Поза и 3D только от Python bridge | 🟢 |
| 4 | **Детекция только через Python** | YuNet есть, но не подключён как camera source | 🟢 |

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

## Структура (8 крейтов)

```
mvm/
├── Cargo.toml              # [workspace] 8 members, edition 2024
├── .cargo/config.toml      # alias: cargo xtask = run --package xtask --
├── crates/
│   ├── mvm-core/           # ✅ типы, трейты, Config, PersonTracker
│   ├── mvm-detect/         # ✅ YuNet ONNX (228KB, embed-model, multi-stride NMS)
│   ├── mvm-identity/       # ⚠️ SFace ONNX (37MB, L2-embeddings, gallery)
│   ├── mvm-emotion/        # ⚠️ EmotiEffLib B0 ONNX (16MB, 8-class VGAF)
│   ├── mvm-posture/        # 🔶 заглушка Module
│   ├── mvm-fusion/         # 🔶 заглушка Module
│   ├── mvm-pipeline/       # ✅ Camera (Python bridge), parse_line
│   └── mvm-cli/            # ✅ main, StateFile, dump_table(), --native/--once
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

### 🟢 Фаза 6 — Metric 3D Scene-Graph (чистая арифметика, ноль новых deps)

| Задача | Статус |
|--------|--------|
| 6.1 SG-KF: Kalman bbox (4× скаляр + State-Gating) | 🔜 |
| 6.2 Метрический 3D (порт pinhole-геометрии в Rust) | 🔜 |
| 6.3 Uncertainty: entropy-based confidence для emotion | 🔜 |
| 6.4 Fusion: реляционный scene-graph (осн.) + атрибутивный PoE (доп.) | 🔜 |

### 🟢 Фаза 7 — Нативный детектор

| Задача | Статус |
|--------|--------|
| mvm-posture: портировать PoseVAE (48.6K params) в Rust | 🔜 |
| mvm-fusion: confidence-based слияние эмоций | 🔜 |
| Замена Python SCRFD на нативный (орт или кастомный ONNX) | 🔜 |
| YuNet как camera source (без Python bridge) | 🔜 |

### 🟢 Фаза 8 — Продукт

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
