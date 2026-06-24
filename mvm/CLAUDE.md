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
