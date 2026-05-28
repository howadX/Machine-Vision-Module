# Video Recognition Module

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-RTX_4070_Ti-76B900.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

**Локальное распознавание лиц и классификация эмоций в реальном времени.**  
Проект прошёл путь от школьного прототипа на dlib (2019) до современного ML-стека на ArcFace + HSEmotion (2025).

 **Статус:** активная разработка каталога `src/utils/` — обновление легаси-логики на новые инструменты.

---

## Архитектура

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Веб-камера   │───▶│  ArcFace (SCRFD) │───▶│  512d эмбеддинг     │
│  /dev/video*  │    │  детекция лица   │    │  + bbox + landmarks │
└──────────────┘    └──────────────────┘    └─────────┬───────────┘
                                                      │
                                         ┌────────────▼───────────┐
                                         ▼                        ▼
                              ┌──────────────────┐    ┌──────────────────┐
                              │   Face ID         │    │   Эмоции          │
                              │   косинусное       │    │   HSEmotion       │
                              │   сходство         │    │   EfficientNet-B0 │
                              │   (512d)           │    │   8 классов        │
                              └────────┬─────────┘    └────────┬─────────┘
                                       ▼                       ▼
                              ┌─────────────────────────────────────────┐
                              │           PIL-оверлей (кириллица)        │
                              │      Имя + уверенность + топ-3 эмоций    │
                              └─────────────────────────────────────────┘
```

---

## 📁 Структура проекта

```
Video Recognition Module/
├── src/
│   ├── pipeline/              # Легаси-пайплайн (dlib + SVM) — заморожен
│   │   ├── 1_make_dataset.py
│   │   ├── 2_extract_descriptors.py
│   │   ├── 3_train_svm.py
│   │   ├── 4_identification.py
│   │   └── README.md          # Оригинальная инструкция (~2019)
│   │
│   ├── utils/                 #  Активная разработка
│   │   ├── face_id.py         # ArcFace: детекция + эмбеддинг (512d)
│   │   ├── emotion_hse.py     # HSEmotion: классификация эмоций (ВШЭ)
│   │   ├── emotion.py         # EfficientNet-B0: свой файн-тюн FER2013
│   │   ├── whois.py           # One-shot: кто перед камерой?
│   │   ├── live_monitor.py    # Реалтайм-монитор с оверлеем
│   │   ├── train_emotion.py   # Обучение EfficientNet + свои данные
│   │   ├── capture_reference.py   # Переснять эталонное фото
│   │   └── capture_my_emotions.py # Снять 7 эмоций для дообучения
│   │
│   └── arduino/               # Заглушка: генератор pin-map
│
├── models/                    # Бинарные модели и эталоны
│   ├── face_detection/        # dlib shape predictor + haarcascade
│   ├── face_recognition/      # dlib ResNet (легаси)
│   ├── emotion/               # EfficientNet-B0 .pth + SVM .dat
│   ├── descriptors/           # 68-точечные и 128d дескрипторы
│   └── references/            # Эталонные фото + mp3
│
├── data/my_emotions/          # Пользовательские снимки (7 эмоций)
├── eeg/                       # EEG-подсистема (неполная)
├── archive/                   # 28 исторических скриптов (v1→v3)
├── tests/                     # Тестовые скрипты
├── assets/                    # Шрифты и примеры эмоций
├── requirements.txt
├── run.sh                     # Запуск легаси-пайплайна
└── README.md
```

---

## Быстрый старт

### Требования

- **Python** 3.11+
- **CUDA** 12.x (опционально — работает и на CPU)
- **GPU** с ≥4GB VRAM (RTX 4070 Ti рекомендуется)
- Веб-камера

### Установка

```bash
git clone https://github.com/howadX/Video-Recognition-Module.git
cd "Video Recognition Module"

# Виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Зависимости
pip install -r requirements.txt

# Дополнительно (не в requirements.txt — ставятся по необходимости):
pip install insightface onnxruntime-gpu      # ArcFace
pip install hsemotion timm                    # HSEmotion + EfficientNet
pip install kagglehub                         # для скачивания FER2013
```

### Первый запуск — кто перед камерой?

```bash
venv/bin/python3 src/utils/whois.py
```

Снимет один кадр, определит личность (из галереи эталонов) и эмоцию.

### Реалтайм-монитор

```bash
venv/bin/python3 src/utils/live_monitor.py
```

Окно с камеры: имя, уверенность, эмоция, топ-3, FPS. `Q` — выход.

### Запуск легаси-пайплайна

```bash
bash run.sh
```

Запускает `4_identification.py` (dlib + SVM + аудио-фидбек через pygame).

---

## Утилиты

| Команда | Назначение |
|---------|-----------|
| `venv/bin/python3 src/utils/whois.py` | One-shot: распознать лицо и эмоцию |
| `venv/bin/python3 src/utils/live_monitor.py` | Реалтайм-монитор |
| `venv/bin/python3 src/utils/capture_reference.py <имя>` | Переснять эталонное фото |
| `venv/bin/python3 src/utils/capture_my_emotions.py` | Снять 7 эмоций для дообучения |
| `venv/bin/python3 src/utils/train_emotion.py` | Дообучить EfficientNet-B0 (нужен FER2013) |

Имена для `capture_reference.py`: **Вадим, Даня, Никита, Белла** (эталоны в `models/references/`).

---

## Два поколения кода

### Поколение 1: `src/pipeline/` — dlib + sklearn SVM

Классический CV-пайплайн 2019 года:
- **Детектор:** Haar Cascade (OpenCV)
- **Лица:** dlib HOG → 68 landmarks → ResNet 128d
- **Идентификация:** евклидово расстояние на 128d
- **Эмоции:** SVM (poly kernel) на 68 точках лица
- **Аудио:** pygame mixer

📄 [Подробнее в README пайплайна](src/pipeline/README.md)

### Поколение 2: `src/utils/` — ArcFace + HSEmotion (текущее)

Полная замена стека на SOTA:
- **Детектор:** SCRFD (InsightFace ArcFace) — профиль, маски, очки
- **Эмбеддинги:** ArcFace 512d — 99.83% на LFW
- **Идентификация:** косинусное сходство (порог 0.35)
- **Эмоции:** HSEmotion (ВШЭ, AffectNet + VGAF) — русские лица
- **Альтернатива:** собственный EfficientNet-B0, файн-тюн FER2013
- **Оверлей:** PIL с поддержкой кириллицы

---

## ⚡ Нюансы

### CUDA и GPU
- `face_id.py` автоматически подхватывает nvidia-библиотеки из venv через `ctypes.CDLL`
- Если GPU недоступен — падает на CPU (медленнее, но работает)
- `torch.backends.cudnn.enabled = False` — для совместимости с onnxruntime

### Камера
- Итерация `/dev/video*` вместо хардкода `cv2.VideoCapture(0)`
- Прямой доступ через `cv2.CAP_V4L2` + прогрев 3–5 кадров
- После USB-переподключения индексы не сбиваются

### HSEmotion и timm
- Старая модель HSEmotion использует `conv_s2d` (переименован в `conv_dw` в новых timm)
- `emotion_hse.py` содержит авто-патч `_patch_depthwise_separable_conv`

### Эталоны
- Хранятся в `models/references/` как обычные фото (p_D.jpg, p_V.jpg и т.д.)
- Используются ArcFace-эмбеддинги, а не оригинальные пиксели
- Порог идентификации: **0.35** (косинусное сходство)

---

## 🗺️ Roadmap

- [x] Реорганизация из плоского корня в ML-структуру
- [x] Миграция с dlib на ArcFace (FaceID)
- [x] HSEmotion вместо SVM для эмоций
- [x] Реалтайм-монитор с оверлеем
- [x] Дообучение EfficientNet-B0 на своих эмоциях
- [ ] Обучение HSEmotion на своих данных
- [ ] ONNX-оптимизация для инференса
- [ ] Мультитрекинг лиц
- [ ] EEG-интеграция

---

## 📄 Лицензия

MIT © 2019–2025
