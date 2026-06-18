# Machine Vision Module

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-RTX_4070_Ti-76B900.svg)]()

**Инструменты машинного зрения для LLM-агентов.**

Один кадр с веб-камеры → текстовый отчёт `key: value` для парсинга агентом.

---

## Архитектура

```
                ┌─────────────────────┐
                │   Веб-камера        │
                │   /dev/video*       │
                └────────┬────────────┘
                         │
          ┌──────────────▼──────────────┐
          │        SCRFD + ArcFace      │
          │   (детекция + эмбеддинги)   │
          │          core/face_id.py    │
          └──────────────┬──────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│   whoishere.py  │            │    sight.py     │
│   (лёгкий, ~3s) │            │  (полный, ~15s)  │
│                 │            │                 │
│ Face ID + сцена │            │ Face ID         │
│                 │            │ + MLP-эмоции    │
│                 │            │ + LightGBM-поза │
└─────────────────┘            └─────────────────┘
```

---

## Структура проекта

```
Machine Vision Module/
├── tools/                     # Продакшен — точки входа для LLM
│   ├── whoishere.py           # Face ID + сцена (лёгкий, ~3s)
│   ├── sight.py               # Face ID + эмоции + поза (полный, ~15s)
│   └── whois.py               # То же, для человека (HSEmotion)
│
├── core/                      # Общие модули
│   ├── face_id.py             # SCRFD + ArcFace 512d
│   ├── emotion.py             # EfficientNet-B0 (fallback)
│   └── emotion_hse.py         # HSEmotion (fallback)
│
├── pipeline/                  # Обучение и сбор данных
│   ├── rlm.py                 # Realtime Live Monitor (окно с оверлеем)
│   ├── live_monitor.py        # Старый реалтайм-монитор
│   ├── extract_geometric.py   # Извлечение 468 признаков
│   ├── train_mlp.py           # Обучение MLP (основная модель эмоций)
│   ├── train_lightgbm.py      # Обучение LightGBM (бэкап)
│   ├── train_emotion.py       # Обучение EfficientNet
│   ├── train_svm.py           # Обучение SVM (легаси)
│   ├── train_combined.py      # Свой+AffectNet → LightGBM
│   ├── 1_make_dataset.py      # Захват лиц с камеры
│   └── capture/               # Захват данных
│       ├── capture_reference.py
│       └── capture_my_emotions.py
│
├── models/
│   ├── emotion/
│   │   ├── mlp_emotion.pth           # Основная модель эмоций (MLP, 68%)
│   │   ├── mlp_emotion_scaler.pkl    # StandardScaler
│   │   ├── mlp_emotion_config.json
│   │   ├── lightgbm_body_holistic_33.pkl  # Поза тела (LightGBM, 71.7%)
│   │   └── fallback/                 # Цепочка fallback-моделей
│   ├── mediapipe/
│   │   ├── face_landmarker.task
│   │   └── holistic_landmarker.task
│   └── references/            # Галерея Face ID (p_V.jpg, p_D.jpg, ...)
│
├── legacy/                    # Архив (dlib, SVM, чекпоинты, эксперименты)
├── assets/                    # Шрифты
├── docs/                      # Технические заметки
└── tests/                     # Тесты
```

---

## Быстрый старт

```bash
cd ~/repos/Machine\ Vision\ Module

# Кто перед камерой? (Face ID + сцена)
venv/bin/python3 tools/whoishere.py

# Полный анализ (Face ID + эмоции + поза)
venv/bin/python3 tools/sight.py

# Реалтайм-монитор
venv/bin/python3 pipeline/rlm.py
```

Вывод `whoishere.py` и `sight.py` — `key: value`, пригоден для парсинга LLM:

```
camera: /dev/video0
face_count: 1
lighting: normal
brightness: 130
face_1:
  identity: Вадим
  confidence: 0.62
  emotion: Нейтральная
  emotion_confidence: 0.72
  posture: сидит
  distance: medium
```

---

## Технологии

| Компонент | Технология |
|-----------|-----------|
| Детектор лиц | SCRFD (InsightFace) |
| Эмбеддинги | ArcFace 512d |
| Face ID | Косинусное сходство, порог 0.35 |
| Эмоции | MLP на 468 геометрических признаках (MediaPipe FaceLandmarker) |
| Поза тела | LightGBM на 33 Holistic-точках (MediaPipe Holistic) |
| GPU | onnxruntime-gpu через CUDA 12 |

---

## Нюансы

- **GPU**: onnxruntime-gpu требует предзагрузку CUDA-библиотек через `ctypes.CDLL` (автоматически в `face_id.py`)
- **MediaPipe**: только CPU delegate — GPU-делегат падает на llvmpipe
- **Камера**: итерация `/dev/video*` через V4L2, авто-прогрев 3–5 кадров
- **Кириллица**: PIL ImageDraw + DejaVuSans (OpenCV не поддерживает русский)
- **Эталоны**: `models/references/p_X.jpg` → `p_V.jpg` = Вадим, `p_D.jpg` = Даня, `p_N.jpg` = Никита, `p_I.jpg` = Белла

---

## Лицензия

MIT © 2019–2026
