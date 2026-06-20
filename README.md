# Machine Vision Module

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Rust](https://img.shields.io/badge/rust-1.93+-orange.svg)](https://rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-RTX_4070_Ti-76B900.svg)]()

**Локальная система восприятия для LLM-агентов.**

Кадр с веб-камеры → многослойный контекст: кто перед камерой, эмоция, поза, направление взгляда, предметы, активность. Всё в текстовом формате `key: value` для парсинга агентом.

---

## Философия

MVM — это не один скрипт, а **стек слоёв восприятия**, расширяющийся от человека наружу:

```
Слой 0: Ядро — кто + эмоция + поза (готово, 98.3% VAE)
Слой 1: Взаимодействие — gaze + руки (MediaPipe, бесплатно)
Слой 2: Предметы — YOLO-детекция (планируется)
Слой 3: Активность — fusion-правила (планируется)
Слой 4: Сцена — тип помещения (опционально)
```

**Скоро:** переход на постоянно работающий Rust-демон (`sightd`) с непрерывным трекингом людей и объектов.

---

## Ветки

| Ветка | Назначение |
|-------|-----------|
| `master` | Python-прод: sight.py, whoishere.py, VAE, Workspace3D |
| `python-prototype` | Полигон: обучение моделей, эксперименты, MediaPipe |
| `rust-core` | Rust-демон: трекер, камера, state-file (чистый лист) |

---

## Архитектура (текущая: Python)

```
Кадр (HD 1280×720 + SD 640×480)
  │
  ├─ PoseLandmarker (33 COCO, CPU, ~20ms)
  │    ├─ body_bbox → head_crop
  │    │    ├─ FaceLM PASS 1 → bbox + 5 kps → ArcFace HD → identity
  │    │    └─ FaceLM PASS 2 → 468 landmarks → MLP → emotion
  │    └─ Spatial Encoding → VAE → posture (6 кл, 98.3%)
  │
  └─ Вывод: key: value → state file → sight-journal → wiki
```

---

## Быстрый старт

```bash
cd ~/repos/Machine\ Vision\ Module

# Кто перед камерой? (Face ID + сцена, ~3s)
venv/bin/python3 tools/whoishere.py

# Полный анализ (Face ID + MLP-эмоции + VAE-поза + plausibility, ~15s)
venv/bin/python3 tools/sight.py
```

Вывод — `key: value`, пригоден для парсинга LLM:

```
camera: /dev/video0
face_count: 1
lighting: normal
brightness: 91
face_1:
  identity: Вадим
  confidence: 0.60
  emotion: Нейтральная
  emotion_confidence: 0.69
  posture: сидит_за_ПК
  posture_confidence: 0.94
  recon_error: 0.0234
  plausibility: normal
```

---

## Технологии

| Компонент | Технология |
|-----------|-----------|
| Детектор | PoseLandmarker (CPU) |
| Face ID | ArcFace 512d (GPU) |
| Эмоции | MLP 468 геом. (68%) + HSEmotion fallback (73%) |
| Поза | PoseVAE 48K + Classifier 4K (98.3%, 6 классов) |
| GPU | onnxruntime-gpu + CUDA 12 |

### 6 классов позы

| стоит | сидит | сидит_за_ПК | лежит_на_спине | лежит_на_боку | руки_скрещены |
|-------|-------|-------------|----------------|---------------|---------------|

---

## Структура

```
Machine Vision Module/
├── tools/              # Точки входа для LLM
│   ├── whoishere.py    # Face ID + сцена (~3s)
│   ├── sight.py        # Полный анализ (~15s)
│   └── whois.py        # Для человека (HSEmotion)
├── core/               # Модели и утилиты
│   ├── face_id.py      # ArcFace
│   ├── pose_vae.py     # PoseVAE
│   ├── pose_heads.py   # Классификатор поз
│   └── workspace_3d.py # 3D-пространство
├── pipeline/           # Обучение и реалтайм
│   ├── rlm.py          # Монитор с оверлеем
│   ├── train_pose_vae.py
│   └── capture/        # Захват датасетов
├── models/             # Веса
│   ├── emotion/        # MLP, VAE, fallback
│   └── references/     # Галерея Face ID
└── legacy/             # Архив
```

---

## Лицензия

MIT © 2019–2026
