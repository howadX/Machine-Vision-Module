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
┌─────────────────┐            ┌──────────────────────┐
│   whoishere.py  │            │      sight.py        │
│   (лёгкий, ~3s) │            │    (полный, ~15s)     │
│                 │            │                      │
│ Face ID + сцена │            │ Face ID              │
│                 │            │ + MLP-эмоции (68%)   │
│                 │            │ + VAE-поза (98.3%)   │
└─────────────────┘            │ + plausibility       │
                               └──────────────────────┘
```

---

## Поза тела — самообучающаяся VAE-модель

**Главное нововведение (июнь 2026).** Вместо старого LightGBM на 3 классах (71.7%) — VAE (PoseVAE) + классификатор на 6 классах (98.3%).

### Как работает

```
Кадр → PoseLandmarker (33 COCO landmarks, ~20ms CPU)
  → Spatial Encoding (нормализация + инварианты)
  → PoseVAE encoder (MLP: 99→128→64→μ/σ:32, 48K параметров, <1ms)
  → latent z [32-dim]
    ├─ Pose Classifier → 6 классов (98.3% accuracy)
    ├─ Plausibility (-log p(z)) → детектор незнакомых поз
    └─ VAE Decoder → реконструкция (MSE <0.01)
```

### Классы поз

| Класс | Русский | Примеров |
|-------|---------|----------|
| standing | стоит | 215 |
| sitting | сидит | 205 |
| lying_back | лежит на спине | 193 |
| lying_side | лежит на боку | 112 |
| arms_crossed | руки скрещены | 192 |
| sitting_at_desk | сидит за ПК | 220 |

### Самообучение

VAE обучен на многообразии правдоподобных поз. Когда встречается незнакомая поза:
1. `-log p(z) > 14.3` → поза вне известного manifold
2. Пользователь даёт метку (5 примеров)
3. Prototypical classification: новый класс без переобучения
4. Experience Replay: защита от забывания старых классов

### Ключевые файлы

| Файл | Назначение |
|------|-----------|
| `core/pose_vae.py` | VAE модель (Encoder/Decoder) |
| `core/pose_encoder.py` | Spatial Encoding: нормализация + инварианты |
| `core/pose_heads.py` | Классификатор + аналитические головы |
| `pipeline/train_pose_vae.py` | Обучение VAE |
| `pipeline/train_pose_classifier.py` | Обучение классификатора |
| `pipeline/capture_poses.py` | Захват датасета поз |
| `models/emotion/vposer_coco33.pth` | Веса VAE (48K) |
| `models/emotion/pose_classifier.pth` | Веса классификатора (4K) |

---

## Структура проекта

```
Machine Vision Module/
├── tools/                     # Продакшен — точки входа для LLM
│   ├── whoishere.py           # Face ID + сцена (лёгкий, ~3s)
│   ├── sight.py               # Face ID + эмоции + VAE-поза (полный, ~15s)
│   └── whois.py               # То же, для человека (HSEmotion)
│
├── core/                      # Общие модули
│   ├── face_id.py             # SCRFD + ArcFace 512d
│   ├── pose_vae.py            # PoseVAE: Encoder/Decoder
│   ├── pose_encoder.py        # Spatial Encoding для 33 landmarks
│   ├── pose_heads.py          # Классификатор + аналитика
│   ├── emotion.py             # EfficientNet-B0 (fallback)
│   └── emotion_hse.py         # HSEmotion (fallback)
│
├── pipeline/                  # Обучение и сбор данных
│   ├── rlm.py                 # Realtime Live Monitor (A/B: LightGBM vs VAE)
│   ├── train_pose_vae.py      # Обучение PoseVAE
│   ├── train_pose_classifier.py # Обучение классификатора поз
│   ├── capture_poses.py       # Захват датасета поз (OpenCV GUI)
│   ├── test_pose_landmarker.py # Тест PoseLandmarker
│   ├── train_mlp.py           # Обучение MLP эмоций
│   ├── train_lightgbm.py      # Обучение LightGBM (бэкап)
│   ├── train_emotion.py       # Обучение EfficientNet
│   ├── extract_geometric.py   # Извлечение 468 признаков
│   ├── live_monitor.py        # Старый реалтайм-монитор
│   ├── 1_make_dataset.py      # Захват лиц с камеры
│   └── capture/               # Захват данных
│
├── models/
│   ├── emotion/
│   │   ├── mlp_emotion.pth           # Эмоции (MLP, 68%)
│   │   ├── vposer_coco33.pth         # VAE позы (48K параметров)
│   │   ├── vposer_coco33_config.json
│   │   ├── pose_classifier.pth       # Классификатор поз (98.3%)
│   │   ├── pose_classifier_config.json
│   │   ├── lightgbm_body_holistic_33.pkl  # Старая модель (71.7%)
│   │   └── fallback/                 # Цепочка fallback-моделей эмоций
│   ├── mediapipe/
│   │   ├── face_landmarker.task      # 468 landmarks лица
│   │   ├── holistic_landmarker.task  # Face+Body+Hands (legacy)
│   │   └── pose_landmarker.task      # 33 COCO landmarks тела (скачать отдельно)
│   └── references/            # Галерея Face ID
│
├── legacy/                    # Архив (dlib, SVM, чекпоинты)
├── assets/                    # Шрифты
├── docs/                      # Технические заметки
└── tests/                     # Тесты
```

---

## Быстрый старт

```bash
cd ~/repos/Machine\ Vision\ Module

# 1. Скачать PoseLandmarker (один раз)
wget -O models/mediapipe/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

# 2. Кто перед камерой?
venv/bin/python3 tools/whoishere.py

# 3. Полный анализ (Face ID + эмоции + VAE-поза)
venv/bin/python3 tools/sight.py

# 4. Реалтайм-монитор с A/B сравнением
venv/bin/python3 pipeline/rlm.py
```

Вывод `sight.py` — `key: value`, пригоден для парсинга LLM:

```
camera: /dev/video0
face_count: 1
lighting: bright
brightness: 156
face_1:
  identity: Вадим
  confidence: 0.60
  emotion: Нейтральная
  emotion_confidence: 0.42
  posture: сидит
  posture_confidence: 0.63
  recon_error: 0.0258
  plausibility: normal
  det_score: 0.84
  bbox: 344,206,442,340
  size: 97x134
  distance: medium
```

---

## Обучение своей модели поз

```bash
# 1. Захват датасета
venv/bin/python3 pipeline/capture_poses.py
#   Управление: 0-9 — сохранить позу, q — выход
#   Метки: 0=standing, 1=sitting, 2=lying_back, 3=lying_side,
#          4=arms_crossed, 5=sitting_at_desk

# 2. Обучение VAE
venv/bin/python3 pipeline/train_pose_vae.py

# 3. Обучение классификатора
venv/bin/python3 pipeline/train_pose_classifier.py
```

---

## Технологии

| Компонент | Технология |
|-----------|-----------|
| Детектор лиц | SCRFD (InsightFace) |
| Эмбеддинги | ArcFace 512d |
| Face ID | Косинусное сходство, порог 0.35 |
| Эмоции | MLP на 468 геометрических признаках (MediaPipe FaceLandmarker) |
| Поза тела | **PoseVAE** (MLP 48K + классификатор 4K, 98.3% на 6 классах) |
| Landmarks тела | MediaPipe PoseLandmarker (33 COCO, ~20ms CPU) |
| GPU | onnxruntime-gpu через CUDA 12 |

---

## Нюансы

- **GPU**: onnxruntime-gpu требует предзагрузку CUDA-библиотек через `ctypes.CDLL` (автоматически в `face_id.py`)
- **MediaPipe**: только CPU delegate — GPU-делегат падает на llvmpipe
- **PoseLandmarker**: скачивается отдельно (5.6MB). См. Быстрый старт.
- **Камера**: итерация `/dev/video*` через V4L2, авто-прогрев 3–5 кадров
- **Кириллица**: PIL ImageDraw + DejaVuSans (OpenCV не поддерживает русский)
- **Эталоны**: `models/references/p_X.jpg` → `p_V.jpg` = Вадим, `p_D.jpg` = Даня, `p_N.jpg` = Никита, `p_B.jpg` = Белла

---

## Лицензия

MIT © 2019–2026
