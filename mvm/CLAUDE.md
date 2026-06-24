# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*

## Философия сборки

MVM проектируется от сборки наружу, а не от кода внутрь.

**Принцип:** если модуль добавить легко — архитектура останется чистой по умолчанию.
**Цель:** один бинарник. Скачал — запустил — работает. Без Python, без pip, без GPU.

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

## Текущий статус (24 июня 2026, фаза 5.2)

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

Стратегия выбора: **confidence-based fallback** (реализовано в фазе 5.1) — берём предсказание с более высокой confidence. A/B сравнение запланировано на фазу 5.2.2.

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

### 🟡 Фаза 5.2 — Качество (в процессе)

| Задача | Статус |
|--------|--------|
| `--dump-json` флаг (machine-readable JSON stdout) | ✅ |
| A/B сравнение EmotiEffLib vs HSEmotion (`--emotion-debug`) | 🔜 |
| PersonTracker: emotion decay при missed > N кадров | 🔜 |
| Интеграционные тесты: bridge + Rust pipeline | 🔜 |

### 🟢 Фаза 6 — Predictive State

| Задача | Статус |
|--------|--------|
| SG-LKF: MLP-предсказатель bbox | 🔜 |
| Evidential DL: softmax→softplus confidence | 🔜 |
| PDF fusion (Mono/Holo-Confidence) | 🔜 |

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

## Фаза 6 — Predictive State: детальный план

> **Для:** brainstorm с Opus / делегирование Claude Code.
> **Статус базы:** 70 тестов, все зелёные. Фаза 5.2 завершена.

### Контекст: что есть сейчас после фазы 5.2

```
PersonTracker
  ├─ IoU-матчинг (threshold 0.3)
  ├─ EMA bbox-сглаживание (α = 0.3)
  ├─ identity: best-confidence persistence
  ├─ emotion: latest snapshot + decay (missed > 5 → ×0.8/кадр → None при conf < 0.05)
  └─ timeout: 1800 кадров (30s @ 60fps)

mvm-emotion
  └─ classify() → (label: String, conf: f32)  // max(softmax), 8 классов VGAF

mvm-fusion   🔶 заглушка
mvm-posture  🔶 заглушка
```

**Проблема EMA:** реактивен. При occluded-кадрах bbox не движется → IoU с новыми детекциями падает → трек теряется раньше срока.

**Проблема confidence:** `max(softmax) = 0.8` может означать «уверен» (7 классов ≈ 0) или «не уверен» (7 классов ≈ 0.03 каждый). Без меры uncertainty downstream-агенты не различают эти случаи.

**Проблема fusion:** два источника emotion (Rust EmotiEffLib + Python HSEmotion) сейчас сравниваются по одному скалярному числу, хотя оба дают вектор из 8 вероятностей.

---

### Задача 6.1 — SG-KF: Kalman Filter для bbox

**Цель:** заменить EMA на Linear Kalman Filter с constant-velocity motion model.

#### State vector (8D)

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

### Задача 6.2 — Uncertainty Estimation для emotion

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

#### TDD тесты для 6.2

| Тест | Что проверяет |
|------|--------------|
| `entropy_zero_for_one_hot` | p=[1,0,0,...] → entropy=0.0 |
| `entropy_one_for_uniform` | p=[0.125; 8] → entropy=1.0 |
| `calibrated_conf_penalizes_uncertain` | Высокая prob + высокая entropy → cal_conf < prob |
| `emotion_result_distribution_sums_to_one` | Σ distribution ≈ 1.0 |
| `emotion_wins_uses_calibrated_conf` | При равной raw prob побеждает менее энтропийный |

---

### Задача 6.3 — PDF Fusion в mvm-fusion

**Цель:** `mvm-fusion` из заглушки → Product-of-Experts fusion двух emotion-распределений.

#### Product of Experts

Если Rust даёт `p_rust[8]` и Python даёт `p_py[8]`:

```
p_fused[i] = (p_rust[i] ^ w_rust) * (p_py[i] ^ w_py) / Z

где:
  w_rust = (1 - rust.uncertainty)   // вес = уверенность
  w_py   = (1 - py_uncertainty)     // аналогично
  Z      = Σ_j (p_rust[j] ^ w_rust * p_py[j] ^ w_py)  // нормировка
```

**Семантика:** эксперт с низкой uncertainty получает больший вес. При uncertainty=1 (равномерное) — фактически исключается из fusion.

**Mono-Confidence** (текущий подход): `max(p_fused)`.
**Holo-Confidence** (новый): весь вектор `p_fused[8]` передаётся downstream.

#### Деградация до текущего поведения

При `w_rust=1, w_py=0`: `p_fused = p_rust` (чистый Rust).
При `w_rust=0, w_py=1`: `p_fused = p_py` (чистый Python).
→ Текущий `emotion_wins()` — частный случай PoE при бинарных весах.

#### Что меняется в коде

```rust
// mvm-fusion/src/lib.rs
pub struct FusedEmotion {
    pub label: String,
    pub prob: f32,                  // max(p_fused)
    pub distribution: [f32; 8],    // Holo-Confidence
    pub uncertainty: f32,           // entropy(p_fused)
    pub rust_weight: f32,           // w_rust использованный при fusion
    pub py_weight: f32,             // w_py
}

pub fn fuse_emotions(
    rust: &EmotionResult,
    py_probs: &[f32; 8],
    py_uncertainty: f32,
) -> FusedEmotion { ... }
```

#### TDD тесты для 6.3

| Тест | Что проверяет |
|------|--------------|
| `fuse_identical_distributions_unchanged` | p_rust == p_py → p_fused ≈ p_rust |
| `fuse_zero_weight_expert_ignored` | uncertainty=1.0 → expert excluded |
| `fuse_confident_expert_dominates` | u_rust=0, u_py=1 → p_fused ≈ p_rust |
| `fuse_distribution_sums_to_one` | Σ p_fused ≈ 1.0 (нормировка корректна) |
| `fuse_disagreement_raises_uncertainty` | Расходящиеся эксперты → выше entropy результата |

---

### Зависимости и порядок реализации

```
6.1 Kalman bbox   ─────────────────────────────────────── независима
6.2 Uncertainty   ────────────── нужна для 6.3
6.3 PDF fusion    ─ зависит от 6.2 (нужен entropy/uncertainty)
```

**Рекомендуемый порядок:** 6.1 → 6.2 → 6.3.

### Открытые архитектурные вопросы для brainstorm

1. **Kalman vs MLP predictor:** roadmap говорит "MLP-предсказатель". MLP обучается на данных движения — нужен dataset. KF работает без обучения. Что выбрать для первой итерации?

2. **Где живёт KalmanBBox:** в `PersonTrack` (всегда) или в `PersonTracker` как внешний map `HashMap<u32, KalmanBBox>`? Первое чище для трейтов, второе — для тестируемости без полного трека.

3. **Матрица Q и R:** hardcoded или из Config? Для 30fps и статичной сцены (монитор) параметры процессного шума отличаются от динамичной.

4. **Entropy vs True EDL:** стоит ли переобучить EmotiEffLib B0 с EDL loss? Нужен датасет AffectNet/VGAF, ~1 GPU-день. Или entropy-approximation достаточна для Hermes?

5. **Python uncertainty:** HSEmotion не возвращает логиты, только top-1 emotion+conf. Для PoE нужен полный вектор p_py[8]. Придётся патчить `mvm_bridge.py` чтобы отдавал `emotion_probs: [f32; 8]`.

6. **Holo-Confidence downstream:** что с ним делает Hermes/Alter? Если downstream читает только `emotion: String`, то Holo ничего не даёт. Нужен ли JSON-field `emotion_distribution: [f32; 8]` в `to_json_line()`?

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
