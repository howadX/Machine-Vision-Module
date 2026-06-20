# Machine Vision Module

*Structured Decomposed Picture-to-Text — эталонная multi-agent система машинного зрения.*

## Идентичность

MVM — не монолитная PTT-модель. MVM — ансамбль независимых микро-агентов, каждый из которых обучен под одну задачу и отдаёт структурированный сигнал с confidence. Истина рождается не в генерации текста, а в **fusion-логике**, собирающей пазл из мнений агентов.

### MVM vs Holistic PTT

| | Holistic PTT (LLaVA, GPT-4V) | Structured Decomposed PTT (MVM) |
|---|---|---|
| Архитектура | Одна модель end-to-end | Ансамбль специалистов |
| Обучение | Совместное | Независимое per-agent |
| Вывод | Текст (неструктурированный) | Поля + confidence |
| Заменимость | Переобучение всей модели | Замена одного агента |
| Предел качества | Размер модели | Качество fusion-логики |
| Интерпретируемость | Чёрный ящик | Каждый сигнал прослеживаем |

## Архитектура

### Слои

| Слой | Роль | Статус |
|------|------|--------|
| **0 — Ядро** | Identity, Emotion, Posture, 3D-пространство | ✅ production |
| **1 — Gaze + Hands** | Направление взгляда, жесты рук | Фаза 1 |
| **2 — Objects** | Предметы в кадре (YOLO) | Фаза 2 |
| **3 — Activity** | Fusion-правила: поза+объекты+руки→действие | Фаза 3 |
| **4 — Scene** | Описание помещения (Places365) | Фаза 4 |

### Микро-агенты (слой 0)

```
Кадр ──→ [PoseLandmarker]  ──→ 33 COCO landmarks ──→ 3D Workspace ──→ (x,y,z)м
     ├─→ [FaceLandmarker]  ──→ 468 face landmarks ──→ MLP ──→ emotion
     ├─→ [ArcFace]         ──→ 512d embedding ──→ identity + confidence
     ├─→ [HSEmotion]       ──→ 8-class ──→ emotion (fallback)
     └─→ [PoseVAE]         ──→ latent 32d ──→ posture + plausibility
```

Каждый агент:
- Обучается независимо
- Отдаёт confidence
- Ничего не знает о других агентах
- **Осознание за 1-6 мс** — в 20-500× быстрее человеческого зрительного восприятия (150-500ms; Thorpe 1996, Dehaene 2006)

### 3D Workspace

Единое метрическое пространство в метрах. Камера = (0,0,0). Калибровка через pinhole-модель по известному размеру торса (0.5 м).

### Ключевой принцип: Fusion Logic

Fusion — не ML-модель. Fusion — набор правил и геометрических операций:

- **Temporal Fusion** (PersonTracker): связывает identity между кадрами по bbox + 3D-близости
- **Spatial Fusion** (World3D): связывает детекты с разных камер в единой системе координат
- **Logical Fusion** (Activity): поза + руки + объекты → контекст

**Главный вызов:** с ростом числа агентов сложность fusion-логики растёт экспоненциально. Архитектура должна быть устойчивой с самого начала — каждый слой изолирован, каждый fusion-оператор заменяем.

### Архитектурные принципы (из нейронауки сознания)

Проект опирается на пересечение нейронауки сознания и ML-архитектур. Полный анализ: `~/wiki/machine-vision/02-consciousness-ml-research.md` и `~/wiki/machine-vision/03-architecture-principles.md`.

| Принцип | Нейронаучная основа | Реализация в MVM |
|----------|-------------------|------------------|
| **Feedforward = бессознательное** | Feedforward sweep (150ms) — классификация без осознания (Lamme 2000) | MLP-агенты: 1 проход, <1ms |
| **Рекуррентность = осознание** | Recurrent loops (150-500ms) — conscious access (Dehaene 2006) | PersonTracker: связь кадров во времени |
| **Global Workspace = StateFile** | GWT (Baars/Dehaene): broadcast информации | StateFile.write(): момент «осознания» |
| **System 1 / System 2** | Kahneman + Bengio (NeurIPS 2019) | MLP=System 1, CoT/LLM=System 2 |
| **Confidence-first** | Индикаторы сознания (Butlin/Bengio et al. 2023) | Каждый сигнал агента имеет confidence |
| **Attention = conscious access** | Attention bottleneck (VanRullen & Kanai 2021) | Будущее: attention поверх MLP для селекции |
| **Деградация без подтверждения** | Memory decay без recurrent reactivation | PersonTracker: −0.02/с decay confidence |

**Ключевая мантра:** MVM-агент не «осознаёт» в нейронаучном смысле. И это правильно — его задача не требует осознания. Осознание происходит в fusion-слое, когда разрозненные сигналы собираются в целостную картину.

## Проект

```
core/          # Модули: face_id, emotion, pose, workspace_3d
tools/         # Продакшен: sight.py, whoishere.py
pipeline/      # Обучение и сбор данных
models/        # Веса: emotion/, mediapipe/, references/
src/           # Rust-компоненты
```

### Ключевые инструменты

| Инструмент | Назначение | Время |
|-----------|-----------|-------|
| `sight.py` | Полный пайплайн (identity + emotion + posture + 3D) | ~15s |
| `whoishere.py` | Лёгкий (только identity + сцена) | ~3s |

### Окружение

- Python 3.11+ (venv)
- GPU: onnxruntime-gpu (CUDA 12) + ctypes.CDLL preload
- MediaPipe: CPU delegate (llvmpipe не держит GPU)
- HSEmotion: PyTorch GPU (~7ms)

## Терминология

- **Micro-agent** — независимый детектор (PoseLM, FaceLM, ArcFace, etc.)
- **Fusion** — логический слой, собирающий сигналы агентов в описание
- **Structured Decomposed PTT** — архитектурная идентичность MVM
- **3D Workspace** — метрическая система координат в метрах
- **Confidence-first** — каждый сигнал имеет числовую уверенность
