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

## Текущий статус (24 июня 2026)

### Что работает при `mvm-cli --native --once`

```
Камера /dev/video0
    │
    ▼
Python bridge (mvm_bridge.py, ~8s загрузка моделей)
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
    ├─ EmotiEffLib B0  → emotion (ПЕРЕзаписывает Python HSEmotion) ← Фаза 5.1
    ├─ SFace           → identity (пропускает: gallery пустая)      ← Фаза 5.1
    ├─ PersonTracker   → temporal fusion (IoU match, identity persistence)
    ├─ StateFile       → ~/.hermes/profiles/sight/workspace/current_state.md
    └─ dump_table()    → таблица в терминал (при --once)
```

### Бинарники

| Файл | Размер | Команда сборки |
|------|--------|----------------|
| `target/release/mvm-cli` | **25 МБ** | `cargo build --release` |
| `dist/mvm` | 24 МБ | `cargo xtask bundle` (устарел: нет dump_table) |

### Крейты

| Крейт | Статус | Содержимое | Тесты |
|-------|--------|------------|-------|
| **mvm-core** | ✅ production | BBox, Detection, PersonTrack, PersonTracker, Config, Module trait | 23 |
| **mvm-detect** | ✅ production | YuNet ONNX (228 КБ), multi-stride NMS | 3 |
| **mvm-identity** | ⚠️ не подключён | SFace ONNX (37 МБ), gallery, cosine match | 5 |
| **mvm-emotion** | ⚠️ безусловная замена | EmotiEffLib B0 ONNX (16 МБ), 8 классов VGAF | 3 |
| **mvm-posture** | 🔶 заглушка | Пустой Module trait | 0 |
| **mvm-fusion** | 🔶 заглушка | Пустой Module trait | 0 |
| **mvm-pipeline** | ✅ production | Camera (Python bridge subprocess), parse_line | 5 |
| **mvm-cli** | ✅ production | main, StateFile, dump_table(), --native/--once | 7 |
| **xtask** | ✅ | `cargo xtask bundle` → dist/mvm | 1 |

**Всего: 47 тестов, все зелёные.**

### Известные проблемы

| # | Проблема | Причина | Приоритет |
|---|----------|--------|-----------|
| 1 | **SFace gallery пустая** | YuNet score_threshold=0.5 не находит лиц на эталонных фото p_V_*.jpg | 🔴 |
| 2 | **EmotiEffLib затирает HSEmotion** | В `--native` моде Rust безусловно перезаписывает emotion от Python (даже если confidence ниже) | 🔴 |
| 3 | **EmotiEffLib vs HSEmotion — разные датасеты** | Оба EfficientNet-B0, но VGAF (Rust) vs AffectNet (Python). Могут давать разные предсказания на одном лице | 🟡 |
| 4 | **dist/mvm устарел** | Старый bundle без dump_table() | 🔴 |
| 5 | **mvm-posture, mvm-fusion — заглушки** | Поза и 3D только от Python bridge | 🟢 |
| 6 | **Детекция только через Python** | YuNet есть, но не подключён как camera source | 🟢 |

### Модели эмоций: архитектурное примечание

**EmotiEffLib B0 (Rust) и HSEmotion (Python) — одна архитектура: EfficientNet-B0.** Оба — CNN на кропе лица 224×224. Разница в датасете (VGAF vs AffectNet) и рантайме (ONNX/CPU vs PyTorch/GPU). Это **не** геометрические признаки (MLP) vs CNN — оба CNN. Расстояние до камеры влияет на них одинаково.

Правильная стратегия выбора: **confidence-based fallback** — брать предсказание с более высокой confidence, а не distance-based.

### Делегирование Claude Code

Фаза 5.1 делегируется Claude Code для экономии токенов основного агента (Hermes).

**Промт:** `docs/phase-5.1-claude-prompt.md` — self-contained, содержит весь необходимый контекст:
- Архитектура проекта и пайплайна
- 4 задачи с конкретными файлами и кодом
- TDD-паттерны и команды проверки
- Критерии успеха

**Правила делегирования:**
- Все задачи выполняются в репозитории `~/repos/mvm/`
- Claude Code читает `AGENTS.md` и `CLAUDE.md` автоматически
- TDD обязателен: сначала тест, потом реализация
- `cargo test --workspace` должен оставаться зелёным после каждого изменения
- Коммиты после каждой задачи

---

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

### 🔴 Фаза 5.1 — Стабилизация (промт готов, делегируется Claude Code)

| Задача | Статус |
|--------|--------|
| YuNet threshold 0.5→0.3 для gallery | 🔜 → Claude Code |
| Confidence-based emotion fallback (Rust↔Python) | 🔜 → Claude Code |
| Обновить dist/mvm (xtask bundle) | 🔜 → Claude Code |
| Починить сборку галереи на эталонных фото | 🔜 → Claude Code |

### 🟡 Фаза 5.2 — Качество

| Задача | Статус |
|--------|--------|
| A/B сравнение EmotiEffLib vs HSEmotion на живых данных | 🔜 |
| PersonTracker: confidence decay, трекинг при поворотах | 🔜 |
| `--dump-json` флаг для machine-readable вывода | 🔜 |
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
