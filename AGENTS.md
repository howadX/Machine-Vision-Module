# Machine Vision Module

## Project Overview
Локальная система компьютерного зрения: захват с камеры → детекция лиц → идентификация → анализ эмоций.
Точка входа для внешних систем: `tools/whoishere.py` — один кадр → текстовый отчёт `key: value`.

## Architecture
- `tools/` — продакшен-инструменты для LLM: whoishere.py (лёгкий), sight.py (полный), whois.py (для человека)
- `core/` — общие модули: face_id.py (SCRFD + ArcFace), emotion.py, emotion_hse.py
- `pipeline/` — обучение и сбор данных: rlm.py (реалтайм-монитор), extract_geometric.py, train_*.py
- `models/` — веса и артефакты: emotion/ (MLP + LightGBM + fallback), mediapipe/, references/ (эталоны)
- `legacy/` — архив: dlib, SVM, чекпоинты, эксперименты с позой, EEG, Arduino
- **Основная модель эмоций:** `models/emotion/mlp_emotion.pth` (MLP, 68%, 3000 эпох)
- **Модель позы тела:** `models/emotion/lightgbm_body_holistic_33.pkl` (LightGBM, COCO, 71.7%)
- **Цепочка fallback:** `models/emotion/fallback/` (LinearSVC → LightGBM Regions → LightGBM AffectNet → LightGBM Geometric)

## Key Constraints
- **GPU**: onnxruntime-gpu через CUDA 12, MediaPipe ТОЛЬКО CPU (нет EGL/OpenGL, llvmpipe крашит)
- **Запуск**: ТОЛЬКО через venv — `venv/bin/python3 tools/whoishere.py`
- **Не создавать новых entry points** — только whoishere.py и sight.py для LLM
- **Формат вывода**: `key: value`, plain text, без эмодзи, пригоден для парсинга
- **Эталоны**: `models/references/p_X.jpg` (где X — буква имени)

## GPU Setup Pattern (face_id.py)
Перед импортом onnxruntime-gpu:
```python
import ctypes, sys
cuda_lib = ctypes.CDLL(f"libcudart.so.{sys.version_info.major}.{sys.version_info.minor}")
```
Без этого падает. MediaPipe — строго CPU delegate.

## Environment
- Python: 3.11+ (через venv)
- Venv: `venv/bin/python3`
- Key packages: opencv-python, insightface, onnxruntime-gpu, mediapipe, lightgbm, dlib, torch

## Code Style
- Type hints обязательно
- Без эмодзи в коде, логах и коммитах
- Подавление отладочного шума (ONNXRUNTIME_LOG_LEVEL=3 и т.д.)

## Active Branch: 3d-workspace (июнь 2026)

Цель: перенос MVM на метрические 3D-координаты как core-представление.

### План фазы 3D-1: Калибровка
- `core/workspace_3d.py` — слой Workspace3D (калибровка MediaPipe z → метры, единая система координат)
- Калибровка по известному размеру torsо (~0.5м)
- Интеграция в `tools/sight.py`

### План фазы 3D-2: Metric PoseVAE
- Обучение VAE на метрических координатах (без scale-нормализации)
- Сохранение метрической информации: дистанция, высота над полом, реальный размер

### План фазы 3D-3: PointNet Face Encoder
- Замена MLP эмоций на PointNet (468×3 → 7 эмоций)
- ~2700 параметров, <0.1ms CPU
- Инвариантность к повороту головы

### План фазы 3D-4: DepthAnything V2 (опционально)
- ONNX-модель для метрической глубины объектов
- Интеграция с YOLO для 3D bounding boxes

### Принципы
- 3D-координаты — нативные, не пост-обработка
- Все модели потребляют и отдают XYZ в метрах
- Единая система координат: камера = (0,0,0), Z = глубина
- Новые архитектуры должны быть легче существующих (PointNet < MLP)
