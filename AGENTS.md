# Machine Vision Module

## Project Overview
Локальная система компьютерного зрения: захват с камеры → детекция лиц → идентификация → анализ эмоций.
Точка входа для внешних систем: `src/utils/whoishere.py` — один кадр → текстовый отчёт `key: value`.
Команда «Кто за ПК?» вызывает именно whoishere.py.

## Architecture
- `src/pipeline/` — этапы пайплайна (1_make → 2_extract → 3_train)
- `src/utils/` — утилиты: whoishere.py (основная), face_id.py (GPU-детекция), emotion.py, emotion_hse.py, capture_reference.py, live_monitor.py
- `models/` — веса и артефакты: face_detection (SCRFD), face_recognition (ArcFace), emotion (LightGBM), references (эталоны)
- `eeg/` — экспериментальный EEG-анализ (не основной код)
- `tests/` — тесты

## Key Constraints
- **GPU**: onnxruntime-gpu через CUDA 12,MediaPipe ТОЛЬКО CPU (нет EGL/OpenGL, llvmpipe крашит)
- **Запуск**: ТОЛЬКО через venv — `venv/bin/python3 src/utils/whoishere.py`
- **Не создавать новых entry points** — только whoishere.py для внешних вызовов
- **Формат вывода**: `key: value`, plain text, без эмодзи, пригоден для парсинга
- **Эталоны**: `models/references/p_X.jpg` (где X — номер)

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
- Key packages: opencv-python, insightface, onnxruntime-gpu, mediapipe, lightgbm, dlib

## Code Style
- Type hints обязательно
- Без эмодзи в коде, логах и коммитах
- Подавление отладочного шума (ONNXRUNTIME_LOG_LEVEL=3 и т.д.)
