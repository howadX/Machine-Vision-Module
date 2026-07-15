# Фаза 16 — Dataset Factory: фабрика данных для будущих ML-моделей

> **Версия:** 0.2 (перенумерована 15→16 15 июля 2026: фаза 15 отдана ML-рефакторингу v0.2.0,
> см. `phase-15-plan.md` — фабрике нужна доведённая до production-качества ML-логика как фундамент).
> **Цель фазы:** превратить MVM из источника данных «только для LLM» в **двухпотребительскую архитектуру**: LLM (MCP) + Dataset Factory (накопительный слой для будущего обучения).
> Ядро зрения готово (фазы 5–14); эта фаза добавляет **экспортный слой**, не трогая зрение.

## Обещания пользователю (вайб)

1. MVM не просто «видит и забывает» — он **копит данные** для будущих моделей.
2. Одна команда включает запись: `mvm record` — и каждый кадр сохраняется с полной разметкой.
3. Данные самоорганизуются в сессии (по присутствию людей) и паттерны (повторяющиеся активности).
4. Одна команда конвертирует накопленное в датасет: `mvm dataset export --format pytorch`.
5. Всё локально, приватно, $0. Ничего не утекает наружу.

## Управляющие принципы (сквозные для всей фазы)

1. **Инварианты ядра целы.** Детекторы, классификаторы, трекер, fusion — **0 изменений**. Dataset-слой — чистый потребитель WorldState, как и StateFileWriter.
2. **Dataset deps живут ТОЛЬКО в новом крейте `mvm-dataset` + Python-утилитах.** Ни ONNX, ни PyTorch, ни TF не протекают в ядро.
3. **Pull-модель, не push.** DatasetWriter читает WorldState (Arc<RwLock<>>) — не требует изменений в агентах-продюсерах.
4. **Структура на диске = self-describing.** Папка сессии содержит кадры + `labels.jsonl` — самодостаточна, можно скопировать на другую машину и использовать.
5. **Privacy-first.** Сохранение лиц с identity — только для галерейных персон (по умолчанию); анонимы — с маскированными лицами или bbox-only. Настраивается в конфиге.

## Решения (зафиксированы с владельцем)

- **DatasetWriter — агент Agent Mesh.** Работает на своём потоке, читает WorldState с настраиваемым cadence (по умолчанию — каждый N-й кадр, N=30 для 1 fps при 30 fps камере).
- **Формат хранения:**
  - Кадры: JPEG (сжатие 85%) в `sessions/<session_id>/frames/XXXXXX.jpg`
  - Глубина: 16-bit PNG (inverse depth) в `sessions/<session_id>/depth/XXXXXX.png`
  - Разметка: JSONL `sessions/<session_id>/labels.jsonl` — одна строка на кадр
  - Мета-данные сессии: `sessions/<session_id>/session.toml` (время, участники, активности)
- **Session grouper — в mvm-core (WorldState).** Сессия = непрерывный интервал с хотя бы одним tracked человеком. Разрыв >30с без людей → новая сессия. Session-id = `YYYY-MM-DD_HHMMSS`.
- **Экспорт в ML-форматы — Python-скрипты.** PyTorch Dataset, TFRecord, WebDataset. Python = гибкость для research-потребителя; Rust не тащит ML-фреймворки.
- **Конфиг-слой:** `[dataset]` секция в `config.toml`. Флаги CLI переопределяют.
- **Privacy-политика:** `privacy = "safe"` (по умолчанию) — лица анонимов размыты/пропущены, identity только для галерейных. `privacy = "full"` — всё сохраняется (для владельца). `privacy = "anonymous"` — все лица маскированы.

---

## 15.0 — Инфраструктура датасет-слоя

### 15.0.1 — Конфиг `[dataset]` в `config.toml`

```toml
[dataset]
enabled = false                  # включить запись (CLI: --record)
dir = "~/.local/share/mvm/datasets"   # корень хранилища датасетов
frame_cadence = 30               # сохранять каждый N-й кадр (1 fps при 30 fps)
frame_format = "jpg"             # jpg | png
frame_quality = 85               # JPEG quality (0-100)

save_depth = true                # сохранять карты глубины
save_masks = true                # сохранять instance-маски (если --masks активен)
save_raw_embeddings = true       # сохранять SFace 128d эмбеддинги
save_landmarks = true            # сохранять BlazePose 33×3 + Hands 21×2

privacy = "safe"                 # safe | full | anonymous
session_timeout_secs = 30        # разрыв >N секунд без людей → новая сессия

[dataset.export]
default_format = "pytorch"       # pytorch | tfrecord | webdataset
```

### 15.0.2 — DatasetWriter: агент Agent Mesh

Новый агент в `mvm-cli/agents/dataset_writer.rs`:

- **Вход:** `Arc<RwLock<WorldState>>` (читает после FusionAgent)
- **Cadence:** сохраняет каждый N-й кадр (`dataset.frame_cadence`)
- **Логика сессий:** отслеживает `has_people` во времени → закрывает сессию после `session_timeout_secs` без людей
- **На каждый сохраняемый кадр пишет:**
  - RGB-кадр (JPEG/PNG на диск)
  - Depth map (16-bit PNG, опционально)
  - Instance masks (опционально)
  - JSONL-строку в `labels.jsonl` с полной разметкой
- **Session metadata:** при закрытии сессии пишет `session.toml` (сводка: участники, активности, аномалии, длительность)

Формат JSONL-строки (одна строка = один кадр):

```json
{
  "frame_id": 12345,
  "timestamp": "2026-07-14T15:30:00.123",
  "session_id": "2026-07-14_153000",
  "people": [
    {
      "track_id": 1,
      "identity": {"label": "Вадим", "gallery_id": 0, "confidence": 0.92},
      "bbox": [320, 180, 480, 600],
      "position_3d": {"x": -0.3, "y": 0.2, "z": 1.2},
      "posture": {"class": "sitting", "confidence": 0.95, "ood_score": 15.2},
      "emotion": {"class": "Neutral", "confidence": 0.87, "entropy": 0.18, "softmax": {"Anger": 0.01, "Neutral": 0.87, ...}},
      "face_bbox": [380, 190, 460, 260],
      "landmarks_body": [[x,y,z,v], ...33 точек],
      "landmarks_hands_left": [[x,y,z], ...21 точек],
      "landmarks_hands_right": [[x,y,z], ...21 точек],
      "gesture_left": {"class": "open", "confidence": 0.8},
      "gesture_right": {"class": "point", "confidence": 0.6},
      "body_coverage": "Upper"
    }
  ],
  "objects": [
    {"track_id": 10001, "class": "cell phone", "bbox": [...], "position_3d": {...}, "confidence": 0.72}
  ],
  "activities": ["typing", "seated_at_workstation"],
  "anomaly_score": 0.12,
  "zones": {"workstation": [1], "seating": [1]},
  "depth_anchored": true
}
```

### 15.0.3 — Session grouper в WorldState

Добавить в `mvm-core/src/world.rs` (WorldState):

- Поле `current_session_id: Option<String>`
- Поле `last_people_seen: Instant`
- Метод `maybe_close_session()` — вызывается FusionAgent при `has_people == false`
- `session_timeout_secs` — из Config (новое поле, дефолт 30)

**Никаких новых зависимостей.** Чистая арифметика: время последнего присутствия + порог.

---

## 15.1 — Сохранение сырых данных

### 15.1.1 — RGB frames

- Формат: JPEG (по умолчанию, качество 85%) или PNG (lossless)
- Именование: `sessions/<session_id>/frames/<frame_id:06d>.jpg`
- Сжатие через `image` крейт (уже есть в зависимостях) или `turbojpeg` (быстрее, опционально)

### 15.1.2 — Depth maps

- Формат: 16-bit grayscale PNG
- Значение: inverse depth × 10000 (сохраняет точность)
- Именование: `sessions/<session_id>/depth/<frame_id:06d>.png`
- Только если `save_depth = true` и DepthAgent активен

### 15.1.3 — Instance masks (опционально)

- Формат: 8-bit indexed PNG (class id → цвет) ИЛИ RLE в JSONL
- Только если `--masks` активен при `mvm run`/`record` + `save_masks = true`

---

## 15.2 — Сохранение разметки

### 15.2.1 — Per-frame labels (JSONL)

Файл `sessions/<session_id>/labels.jsonl` — одна JSON-строка на сохранённый кадр. Полный дамп WorldState в self-describing формате (см. 15.0.2).

**Ключевое решение:** JSONL, а не один большой JSON — позволяет потоковое чтение и дозапись без перезаписи всего файла.

### 15.2.2 — Session metadata (TOML)

При закрытии сессии — сводный файл `session.toml`:

```toml
[session]
id = "2026-07-14_153000"
started = 2026-07-14T15:30:00.123
ended = 2026-07-14T15:47:22.456
duration_secs = 1042
total_frames = 1042
saved_frames = 34

[participants]
known = ["Вадим"]       # gallery identity
anonymous = 2           # сколько разных анонимов

[activities]
"seated_at_workstation" = {count = 12, total_secs = 980}
"typing" = {count = 5, total_secs = 340}
"holding_drink" = {count = 2, total_secs = 45}

[anomalies]
max_score = 0.87
mean_score = 0.12
spikes = [{frame = 1200, score = 0.87, zone = "entrance"}]

[objects]
top = ["keyboard", "cell phone", "cup", "book", "chair"]
```

---

## 15.3 — Экспорт в ML-форматы

### 15.3.1 — Архитектура экспорта

Экспорт — **Python-скрипты**, не Rust. Причины:
- PyTorch/TF — Python-экосистема; Rust-биндинги хрупкие
- Потребитель датасета — ML-инженер с Python
- Форматы датасетов (HuggingFace `datasets`, PyTorch `DataLoader`) имеют Python API

Скрипты лежат в `prototype/pipeline/dataset/` (расширение ML-лаборатории):

```
prototype/pipeline/dataset/
├── export_pytorch.py    # → PyTorch IterableDataset
├── export_tfrecord.py   # → TFRecord + tf.data
├── export_webdataset.py # → WebDataset (shards)
├── dataset_config.yaml  # mapping классов, фильтры
└── README.md
```

### 15.3.2 — PyTorch Dataset

```python
# prototype/pipeline/dataset/export_pytorch.py

class MVMSessionDataset(torch.utils.data.IterableDataset):
    """
    Стримит кадры + разметку из сессии MVM.
    Поддерживает фильтрацию: только галерейные, только с emotion confidence > X.
    """
    def __init__(self, session_path, filters=None):
        ...
    def __iter__(self):
        # читает labels.jsonl построчно
        # загружает кадры по требованию (lazy)
        # выдаёт dict: {frame, depth, labels}
```

Фильтры в `dataset_config.yaml`:

```yaml
filters:
  min_face_confidence: 0.8
  min_emotion_confidence: 0.7
  emotion_entropy_max: 0.5    # исключить near-chance предсказания
  privacy: safe               # только галерейные лица
  postures: [sitting, standing]  # только эти классы
  activities: [typing]        # только кадры с этой активностью
```

### 15.3.3 — CLI: `mvm dataset export`

```bash
# Экспорт последней сессии в PyTorch
mvm dataset export --format pytorch --session latest

# Экспорт всех сессий за сегодня
mvm dataset export --format pytorch --date 2026-07-14

# Экспорт с фильтрами
mvm dataset export --format pytorch --privacy safe --min-confidence 0.8

# Вывод пути к экспортированному датасету
mvm dataset export --format pytorch
# → Exported to: ~/.local/share/mvm/datasets/exports/mvm_sessions_2026-07-14/
```

Команда `mvm dataset`:
- `list` — список сессий (дата, длительность, участники, активности)
- `info <session_id>` — детальная информация о сессии
- `export` — конвертация в ML-формат
- `prune` — удаление старых сессий (политика хранения)
- `stats` — статистика: всего кадров, людей, объектов, активностей

---

## 15.4 — Интеграция с ML-лабораторией

### 15.4.1 — Связка с prototype/pipeline

Существующие скрипты тренировки в `prototype/pipeline/` тренируются на публичных датасетах. Dataset Factory даёт им **собственные данные**:

| Скрипт тренировки | Что может использовать из Dataset Factory |
|-------------------|-------------------------------------------|
| `train_emotion.py` | Эмоции на реальных лицах (не VGAF/AffectNet) — другой домен, другие ракурсы |
| `train_posture.py` | Позы с occlusion-метками (BodyCoverage) — desk-гэп на своих данных |
| `train_identity.py` | Эмбеддинги лиц в реальных условиях (освещение, ракурсы) |
| `distill_emotion.py` | Knowledge distillation: ансамбль → одна модель, на своих данных |
| `self_supervised.py` | Depth prediction по RGB (есть ground truth от Depth-Anything) |

### 15.4.2 — Active learning loop (фаза 15+, будущее)

```
MVM record → dataset → train model → deploy model в MVM → MVM record с новой моделью → ...
```

Самоулучшающаяся петля: чем больше MVM работает, тем больше данных, тем лучше модели, тем лучше MVM.

---

## 15.5 — Privacy-слой (детально)

### 15.5.1 — Режимы

| Режим | Лица галерейных | Лица анонимов | Identity-метки | Эмбеддинги |
|-------|----------------|---------------|----------------|-----------|
| `full` | Сохраняются | Сохраняются | Все | Все |
| `safe` (default) | Сохраняются | Размыты (Gaussian blur) | Только галерейные | Только галерейные |
| `anonymous` | Размыты | Размыты | Нет | Нет |

### 15.5.2 — Реализация

Размытие лиц — на уровне DatasetWriter при сохранении кадра:
- YuNet bbox → Gaussian blur region (σ = bbox_width / 10)
- `image` крейт (уже есть) или простая свёртка

---

## Критичные файлы

| Файл | Роль |
|------|------|
| `crates/mvm-cli/src/config_file.rs` | + `[dataset]` секция (изменение) |
| `crates/mvm-cli/src/agents/dataset_writer.rs` | **Новый** — агент DatasetWriter |
| `crates/mvm-cli/src/dataset_cmd.rs` | **Новый** — подкоманда `mvm dataset` |
| `crates/mvm-core/src/world.rs` | + session tracking (изменение, минимальное) |
| `crates/mvm-core/src/config.rs` | + `DatasetConfig` (изменение, +1 struct) |
| `crates/mvm-cli/src/main.rs` | + диспетч `dataset`, + `--record` флаг для `run` |
| `prototype/pipeline/dataset/` | **Новая** директория — Python-экспортёры |
| `mvm/docs/phase-15-plan.md` | Этот документ |

---

## Верификация

### Unit-тесты (Rust)

- `DatasetConfig` — парсинг из TOML, CLI-override
- Session grouper — открытие/закрытие сессии по таймауту
- DatasetWriter — запись JSONL (мок WorldState)
- Privacy-режимы — проверка, что лица анонимов не сохраняются в safe/anonymous
- `mvm dataset list/info` — корректный вывод на реальных сессиях

### Интеграционные тесты

- `mvm run --record --stub --max-frames 100` → создана сессия, файлы на диске
- `mvm dataset export --session latest` → PyTorch Dataset загружается без ошибок
- `mvm dataset prune --keep 7d` → старые сессии удалены

### Live-валидация

- `mvm record` (живая камера, 5 минут) → сессия с реальными людьми → `mvm dataset export` → загрузить в Python, визуализировать случайный батч
- Проверка privacy: аноним в кадре → лицо размыто в `safe`-режиме, чёткое в `full`

---

## Roadmap: что в 15.0, что потом

| Шаг | Что | Опирается на |
|-----|-----|--------------|
| **15.0** | Инфраструктура: `[dataset]` конфиг, `DatasetConfig`, session grouper в WorldState, каркас DatasetWriter | Фаза 14 (XDG, config.toml) |
| **15.1** | DatasetWriter: сохранение RGB + depth + masks на диск | 15.0 |
| **15.2** | Сохранение разметки: JSONL labels + session TOML | 15.0, 15.1 |
| **15.3** | CLI `mvm dataset` (list/info/export/prune/stats) + `--record` флаг | 15.1, 15.2 |
| **15.4** | Privacy-слой: размытие лиц, режимы safe/full/anonymous | 15.1 |
| **15.5** | Python-экспортёры: PyTorch Dataset, TFRecord, WebDataset | 15.2, 15.3 |
| **15.6** | Интеграция с prototype/pipeline: скрипты тренировки на своих данных | 15.5 |
| **later** | Active learning loop (MVM → dataset → train → deploy → repeat) | 15.6 |
| **later** | Pattern miner: автоматическое выявление повторяющихся паттернов в journal | 15.2 |
| **later** | Dataset versioning: DVC-совместимые чекпойнты, сравнение версий датасета | 15.2 |

Порядок внутри 15.0: **конфиг → session grouper → DatasetWriter → JSONL → CLI → privacy → экспортёры**.

---

## Что НЕ меняется (подтверждение)

| Крейт | Изменения |
|-------|-----------|
| mvm-detect, mvm-identity, mvm-emotion, mvm-posture, mvm-scene, mvm-depth, mvm-hands | **0** |
| mvm-fusion | **0** |
| mvm-mesh, mvm-pipeline | **0** |
| mvm-core | **+2 поля в WorldState** (session tracking) + **+1 struct** (DatasetConfig) |
| mvm-cli | **+1 агент** (DatasetWriter), **+1 модуль** (dataset_cmd), **+диспетч** |

**Строк кода (оценка):**
- Rust: ~800–1200 строк (DatasetWriter 200, dataset_cmd 300, config 100, privacy 150, session grouper 50, тесты 200)
- Python: ~300–500 строк (3 экспортёра + конфиг)
- Документация: ~200 строк

---

*Ядро зрения (фазы 5–14) не трогается. Инварианты №1–3 целы. Dataset-слой — чистый потребитель WorldState, как и StateFileWriter.*
