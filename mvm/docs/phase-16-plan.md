# Фаза 16 — Dataset Factory: семантически структурированная фабрика данных

> **Версия:** 0.3.1 (Dendritron-inspired semantic routing + фиксация решений: 13-й крейт `mvm-dataset`, старт на `kaggle`/v0.2.0; 17 июля 2026).
> **Основание:** фаза 15 отдана ML-рефакторингу v0.2.0; фабрике нужна доведённая до production-качества ML-логика как фундамент.
> **Цель фазы:** превратить MVM из источника данных «только для LLM» в **двухпотребительскую архитектуру**: LLM (MCP) + Dataset Factory (структурированная накопительная память для будущего обучения).
> Фабрика не сжимает наблюдение преждевременно в непрозрачный общий вектор: она маршрутизирует данные по семантическим ветвям, связывает их общими идентификаторами и сохраняет provenance, quality и интерпретируемость.
> Ядро зрения готово (фазы 5–14); эта фаза добавляет **семантический экспортный слой**, не меняя детекторы и fusion.

## Статус реализации (чекпоинт 2026-07-17)

**16.0–16.4 закрыты по коду и live-провалидированы** (ветка `kaggle`; 657 Rust-тестов + 6 Python-групп):

| Шаг | Статус | Что реализовано |
|-----|--------|-----------------|
| **16.0** | ✅ | Semantic Contract: `mvm-dataset` крейт, Canonical Observation (JSONL, ключи связывания, статусы `observed/derived/temporal/weak/withheld/unknown`, provenance), Semantic Router G₁–G₅ |
| **16.1** | ✅ | Инфра: session-grouper в WorldState (+2 поля, чистая арифметика), `DatasetConfig`, `DatasetWriter` + mesh-агент, флаг `--record`, privacy-blur |
| **16.2** | ✅ | Сырьё: RGB (jpg/png), depth (16-bit PNG, min-max + `depth_range`, обратимо), masks (RLE в JSONL); quality-as-data (`posture_status`, `emotion_entropy`, счётчики, anomaly) |
| **16.3** | ✅ | CLI `mvm dataset list/info/stats/prune` (prune — dry-run по умолчанию); публичный `session.rs` ридер |
| **16.4** | ✅ | Python-экспортёры в `prototype/pipeline/dataset/` (PyTorch `IterableDataset` / TFRecord / WebDataset), фильтры `dataset_config.yaml`, README, тесты. Сквозной путь запись→PyTorch DataLoader(num_workers=2) live-провалиден |
| **16.5** | ⏳ | Privacy: заменить консервативную полосу (верх bbox тела) точным YuNet-face-bbox |
| **16.6** | ⏳ | Интеграция с тренерами `prototype/pipeline` + measured quality gates |

Отклонения от плана (зафиксированы): (1) session-lifecycle драйвит **DatasetWriter** (recording-gated),
а не FusionAgent — ноль стоимости когда запись выключена; (2) пиксельное кодирование/blur — в mesh-агенте
(`mvm-cli`), логика — в `mvm-dataset` (крейт без image-зависимостей, полностью unit-тестируем);
(3) depth-нормализация — min-max + `depth_range` вместо наивного «×10000» (реальные значения до ~9 →
переполнили бы u16; поймано на live-данных); (4) типизированный `Label<T>` group-encoding отложен в 16.4
(Python-экспортёры парсят текущий JSON атрибутов). `mvm dataset export` печатает готовые команды к экспортёрам.

## Обещания пользователю (вайб)

1. MVM не просто «видит и забывает» — он **копит данные** для будущих моделей.
2. Одна команда включает запись: `mvm record` — и каждый кадр сохраняется с полной разметкой.
3. Данные самоорганизуются в сессии (по присутствию людей) и паттерны (повторяющиеся активности).
4. Одна команда конвертирует накопленное в датасет: `mvm dataset export --format pytorch`.
5. Всё локально, приватно, $0. Ничего не утекает наружу.

## Архитектурная идея фазы

Dataset Factory — не архив кадров и не простой dump `WorldState`. Это **компилятор наблюдений в обучающие представления**:

```text
WorldState → Canonical Observation → Semantic Router
    ├── G₁ Human       → pose / identity / emotion / landmarks
    ├── G₂ Scene       → objects / depth / zones / permanence
    ├── G₃ Interaction → hands / gestures / contacts / activities
    ├── G₄ Temporal    → transitions / episodes / prediction targets
    └── G₅ Quality     → confidence / entropy / OOD / provenance
                         ↓
              Canonical records + Dataset Views
```

### Dendritron-inspired principles

1. **Local branches first.** Семантические группы обрабатываются независимо и не теряют локальную структуру.
2. **Selective routing.** Кадр попадает только в релевантные ветви: руки — в Interaction, переходы — в Temporal, низкая уверенность — в Quality/Review.
3. **Compositional binding.** Ветви связываются через `dataset_id`, `session_id`, `frame_id`, `entity_id` и `track_id`.
4. **Memory as operation.** Фабрика выполняет `register → route → encode → validate → bind → index → export`, а не только пишет файлы.
5. **Interpretable encoding.** Любая метка хранит значение, статус, confidence, источник, версию схемы и ссылку на исходный сигнал.
6. **No premature collapse.** Нельзя заменять структурированную сцену одним embedding до сохранения сущностей, отношений, временных связей и provenance.

Dendritron здесь используется как источник design principles, а не как утверждение о доказанной универсальности конкретной модели.

## Управляющие принципы (сквозные для всей фазы)

1. **Инварианты ядра целы.** Детекторы, классификаторы, трекер, fusion — **0 изменений**. Dataset-слой — чистый потребитель WorldState, как и StateFileWriter.
2. **Dataset deps живут ТОЛЬКО в новом 13-м крейте `mvm-dataset` + Python-утилитах.** Ни ONNX, ни PyTorch, ни TF не протекают в ядро. `mvm-dataset` = библиотека (Canonical Observation, Semantic Router, encoding/JSONL, privacy-blur); он же — будущий дом для MVM-SRM. `mvm-cli` держит лишь тонкую обвязку: mesh-агент + CLI-диспетч.
3. **Pull-модель, не push.** DatasetWriter читает WorldState (Arc<RwLock<>>) — не требует изменений в агентах-продюсерах.
4. **Структура на диске = self-describing.** Папка сессии содержит кадры + `labels.jsonl` — самодостаточна, можно скопировать на другую машину и использовать.
5. **Privacy-first.** Сохранение лиц с identity — только для галерейных персон (по умолчанию); анонимы — с маскированными лицами или bbox-only. Настраивается в конфиге.
6. **Semantic-first.** Canonical record сохраняет полное наблюдение, а Dataset Views строятся поверх него по группам и задачам.
7. **Quality is data.** Confidence, entropy, OOD, visibility, missing и withheld — часть разметки, а не служебный шум.
8. **Provenance is mandatory.** Для каждой derived/weak метки известны модель, версия, источник и правило вывода.

## Решения (зафиксированы с владельцем)

- **Новый 13-й крейт `mvm-dataset`** (решение 17.07). Изолирует dataset-логику и будущие ML-deps экспортёров от ядра и `mvm-cli`. Содержит: типы Canonical Observation, Semantic Group Router, encoding/JSONL-сериализацию, privacy-blur, session-логику записи. Зависит от `mvm-core` (типы WorldState), но **не** от detect/identity/emotion/posture/scene/depth/hands. `mvm-cli` получает лишь thin mesh-агент (`agents/dataset_writer.rs`) и CLI-модуль (`dataset_cmd.rs`), делегирующие в `mvm-dataset`.
- **Старт фазы — на ветке `kaggle` поверх незарелиженного v0.2.0** (решение 17.07). Код v0.2.0 закрыт (608 тестов); пуш тега `v0.2.0` — независимое owner-действие, фазу 16 не блокирует.
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

## 16.0 — Semantic Dataset Contract

### 16.0.1 — Canonical Observation

Вводится стабильная внутренняя схема наблюдения, не зависящая от PyTorch/TF:

```text
Observation
 ├── frame
 ├── session
 ├── entities
 ├── relations
 ├── events
 ├── semantic_groups
 ├── derived_targets
 ├── quality
 └── provenance
```

Минимальные ключи связывания: `dataset_id`, `schema_version`, `session_id`, `frame_id`, `entity_id`, `track_id`, `group`.

### 16.0.2 — Статусы разметки

Каждое значение получает явный статус:

| Статус | Значение |
|---|---|
| `observed` | непосредственно получено детектором/экспертом |
| `derived` | вычислено Fusion или правилом над наблюдениями |
| `temporal` | выведено из последовательности кадров |
| `weak` | pseudo-label или эвристика, требующая аудита |
| `withheld` | данных недостаточно, утверждение намеренно не делается |
| `unknown` | свойство не наблюдалось или не включено |

Критическое правило: `withheld` и `unknown` нельзя автоматически превращать в отрицательный класс.

### 16.0.3 — Semantic Group Router

```text
WorldState → Snapshot → SemanticGroupRouter → GroupWriters
```

Router не меняет WorldState, не вызывает модели и не добавляет вычисления в hot path. Он маршрутизирует уже полученные сигналы и сохраняет их provenance.

## 16.1 — Инфраструктура датасет-слоя

### 16.1.1 — Конфиг `[dataset]` в `config.toml`

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

### 16.1.2 — DatasetWriter: агент Agent Mesh

Логика записи (routing, encoding, JSONL, session-close, privacy-blur) живёт в крейте **`mvm-dataset`** как `DatasetWriter`. В `mvm-cli/agents/dataset_writer.rs` — **тонкий mesh-агент-обёртка**: держит `Arc<RwLock<WorldState>>`, на своём потоке дёргает cadence и делегирует в `mvm-dataset`. Обёртка:

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

### 16.1.3 — Session grouper в WorldState

Добавить в `mvm-core/src/world.rs` (WorldState):

- Поле `current_session_id: Option<String>`
- Поле `last_people_seen: Instant`
- Метод `maybe_close_session()` — вызывается FusionAgent при `has_people == false`
- `session_timeout_secs` — из Config (новое поле, дефолт 30)

**Никаких новых зависимостей.** Чистая арифметика: время последнего присутствия + порог.

---

## 16.2 — Сохранение сырых данных

### 16.2.1 — RGB frames

- Формат: JPEG (по умолчанию, качество 85%) или PNG (lossless)
- Именование: `sessions/<session_id>/frames/<frame_id:06d>.jpg`
- Сжатие через `image` крейт (уже есть в зависимостях) или `turbojpeg` (быстрее, опционально)

### 16.2.2 — Depth maps

- Формат: 16-bit grayscale PNG
- Значение: inverse depth × 10000 (сохраняет точность)
- Именование: `sessions/<session_id>/depth/<frame_id:06d>.png`
- Только если `save_depth = true` и DepthAgent активен

### 16.2.3 — Instance masks (опционально)

- Формат: 8-bit indexed PNG (class id → цвет) ИЛИ RLE в JSONL
- Только если `--masks` активен при `mvm run`/`record` + `save_masks = true`

---

## 16.3 — Сохранение разметки и provenance

### 16.3.1 — Per-frame labels (JSONL)

Файл `sessions/<session_id>/labels.jsonl` — одна JSON-строка на сохранённый кадр. Canonical Observation в self-describing формате (см. §16.0 и §16.1.2).

**Ключевое решение:** JSONL, а не один большой JSON — позволяет потоковое чтение и дозапись без перезаписи всего файла.

### 16.3.2 — Session metadata (TOML)

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

## 16.4 — Семантические Dataset Views и экспорт в ML-форматы

### 16.4.1 — Архитектура экспорта

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

### 16.4.2 — PyTorch Dataset

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

### 16.4.3 — CLI: `mvm dataset export`

```bash
# Экспорт последней сессии в PyTorch
mvm dataset export --format pytorch --session latest

# Экспорт всех сессий за сегодня
mvm dataset export --format pytorch --date 2026-07-14

# Экспорт с фильтрами
mvm dataset export --format pytorch --privacy safe --min-confidence 0.8

# Семантические views
mvm dataset export --group human --task posture
mvm dataset export --group interaction --task hand_object_contact
mvm dataset export --group temporal --task next_activity
mvm dataset export --hard-examples

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

## 16.5 — Интеграция с ML-лабораторией

### 16.5.1 — Связка с prototype/pipeline

Существующие скрипты тренировки в `prototype/pipeline/` тренируются на публичных датасетах. Dataset Factory даёт им **собственные данные**:

| Скрипт тренировки | Что может использовать из Dataset Factory |
|-------------------|-------------------------------------------|
| `train_emotion.py` | Эмоции на реальных лицах (не VGAF/AffectNet) — другой домен, другие ракурсы |
| `train_posture.py` | Позы с occlusion-метками (BodyCoverage) — desk-гэп на своих данных |
| `train_identity.py` | Эмбеддинги лиц в реальных условиях (освещение, ракурсы) |
| `distill_emotion.py` | Knowledge distillation: ансамбль → одна модель, на своих данных |
| `self_supervised.py` | Depth prediction по RGB (есть ground truth от Depth-Anything) |

### 16.5.2 — Active learning loop (будущее)

```
MVM record → dataset → train model → deploy model в MVM → MVM record с новой моделью → ...
```

Самоулучшающаяся петля: чем больше MVM работает, тем больше данных, тем лучше модели, тем лучше MVM.

---

## 16.6 — Privacy-слой (детально)

### 16.6.1 — Режимы

| Режим | Лица галерейных | Лица анонимов | Identity-метки | Эмбеддинги |
|-------|----------------|---------------|----------------|-----------|
| `full` | Сохраняются | Сохраняются | Все | Все |
| `safe` (default) | Сохраняются | Размыты (Gaussian blur) | Только галерейные | Только галерейные |
| `anonymous` | Размыты | Размыты | Нет | Нет |

### 16.6.2 — Реализация

Размытие лиц — на уровне DatasetWriter при сохранении кадра:
- YuNet bbox → Gaussian blur region (σ = bbox_width / 10)
- `image` крейт (уже есть) или простая свёртка

---

## Критичные файлы

| Файл | Роль |
|------|------|
| `crates/mvm-dataset/` | **Новый 13-й крейт** — вся dataset-логика (см. ниже) |
| `crates/mvm-dataset/src/observation.rs` | **Новый** — Canonical Observation + статусы разметки + provenance |
| `crates/mvm-dataset/src/router.rs` | **Новый** — Semantic Group Router (G₁–G₅) |
| `crates/mvm-dataset/src/writer.rs` | **Новый** — `DatasetWriter`: routing → encode → JSONL/TOML → privacy-blur |
| `crates/mvm-dataset/src/config.rs` | **Новый** — `DatasetConfig` (парсинг `[dataset]`, CLI-override) |
| `crates/mvm-cli/src/agents/dataset_writer.rs` | **Новый** — тонкий mesh-агент-обёртка над `mvm-dataset` |
| `crates/mvm-cli/src/dataset_cmd.rs` | **Новый** — подкоманда `mvm dataset` (делегирует в `mvm-dataset`) |
| `crates/mvm-cli/src/config_file.rs` | + подключение `[dataset]` секции (изменение) |
| `crates/mvm-core/src/world.rs` | + session tracking (изменение, минимальное) |
| `crates/mvm-cli/src/main.rs` | + диспетч `dataset`, + `--record` флаг для `run` |
| `prototype/pipeline/dataset/` | **Новая** директория — Python-экспортёры |
| `mvm/docs/phase-16-plan.md` | Этот документ и контракт фазы |

---

## Верификация

### Unit-тесты (Rust)

- Canonical Observation — round-trip JSONL и обязательные ключи связывания
- Semantic Group Router — детерминированная маршрутизация по группам
- Label status — `observed/derived/temporal/weak/withheld/unknown`
- Provenance — источник и версия сохраняются для derived-меток
- `DatasetConfig` — парсинг из TOML, CLI-override
- Session grouper — открытие/закрытие сессии по таймауту
- DatasetWriter — запись JSONL (мок WorldState)
- Privacy-режимы — проверка, что лица анонимов не сохраняются в safe/anonymous
- `mvm dataset list/info` — корректный вывод на реальных сессиях

### Интеграционные тесты

- `mvm run --record --stub --max-frames 100` → создана сессия, файлы на диске
- один canonical frame → ожидаемые Human/Scene/Interaction/Quality views
- низкая confidence или occlusion → `withheld`, а не ложный отрицательный label
- `mvm dataset export --session latest` → PyTorch Dataset загружается без ошибок
- `mvm dataset prune --keep 7d` → старые сессии удалены

### Live-валидация

- `mvm record` (живая камера, 5 минут) → сессия с реальными людьми → `mvm dataset export` → загрузить в Python, визуализировать случайный батч
- Проверка privacy: аноним в кадре → лицо размыто в `safe`-режиме, чёткое в `full`

---

## Roadmap фазы 16

| Шаг | Что | Опирается на |
|-----|-----|--------------|
| **16.0** | Semantic Dataset Contract: canonical record, group keys, label status, provenance | Фаза 14, WorldState |
| **16.1** | `[dataset]` конфиг, session grouper, DatasetWriter, запись RGB/depth/masks | 16.0 |
| **16.2** | JSONL labels + session TOML + quality-aware provenance | 16.0, 16.1 |
| **16.3** | CLI `mvm dataset` и Semantic Group Router | 16.1, 16.2 |
| **16.4** | Dataset Views и экспорт PyTorch/TFRecord/WebDataset по группам и задачам | 16.3 |
| **16.5** | Privacy-слой и проверка отсутствия утечек identity | 16.1, 16.2 |
| **16.6** | Интеграция с prototype/pipeline и measured quality gates | 16.4, фаза 15 |
| **later** | Active learning loop (MVM → dataset → train → deploy → repeat) | 16.6 |
| **later** | Pattern miner: автоматическое выявление повторяющихся паттернов в journal | 16.3 |
| **later** | Dataset versioning: DVC-совместимые чекпойнты, сравнение версий датасета | 16.3 |

Порядок внутри фазы 16: **canonical contract → semantic router → конфиг → session grouper → DatasetWriter → JSONL/provenance → CLI → Dataset Views → privacy → экспортёры → quality gates**.

---

## Что НЕ меняется (подтверждение)

| Крейт | Изменения |
|-------|-----------|
| mvm-detect, mvm-identity, mvm-emotion, mvm-posture, mvm-scene, mvm-depth, mvm-hands | **0** |
| mvm-fusion | **0** |
| mvm-mesh, mvm-pipeline | **0** |
| **mvm-dataset** (новый 13-й крейт) | **вся dataset-логика**: Canonical Observation, Semantic Router, `DatasetWriter`, `DatasetConfig`, privacy |
| mvm-core | **+2 поля в WorldState** (session tracking) + **+1 скаляр в Config** (`session_timeout_secs`) |
| mvm-cli | **+1 тонкий агент-обёртка** (dataset_writer), **+1 модуль** (dataset_cmd), **+диспетч** |

**Строк кода (оценка):**
- Rust: ~800–1200 строк, из них ~700–900 в новом крейте `mvm-dataset` (observation/router/writer/config/privacy + тесты), ~150 в `mvm-cli` (тонкий агент + dataset_cmd + диспетч), ~50 в `mvm-core` (session grouper)
- Python: ~300–500 строк (3 экспортёра + конфиг)
- Документация: ~200 строк

---

*Ядро зрения (фазы 5–14) не трогается. Инварианты №1–3 целы. Dataset-слой — чистый потребитель WorldState, как и StateFileWriter. Dendritron используется как источник принципов структурной маршрутизации, compositional binding и сохранения локальной интерпретируемости.*
