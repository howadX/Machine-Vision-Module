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

### 3D Workspace

Единое метрическое пространство в метрах. Камера = (0,0,0). Калибровка через pinhole-модель по известному размеру торса (0.5 м).

### Ключевой принцип: Fusion Logic

Fusion — не ML-модель. Fusion — набор правил и геометрических операций:

- **Temporal Fusion** (PersonTracker): связывает identity между кадрами по bbox + 3D-близости
- **Spatial Fusion** (World3D): связывает детекты с разных камер в единой системе координат
- **Logical Fusion** (Activity): поза + руки + объекты → контекст

**Главный вызов:** с ростом числа агентов сложность fusion-логики растёт экспоненциально. Архитектура должна быть устойчивой с самого начала — каждый слой изолирован, каждый fusion-оператор заменяем.

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
