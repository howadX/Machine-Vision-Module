# Machine Vision Module — структура `src/`

Архитектура конвейера: **данные → признаки → модель → рантайм**.

## `pipeline/` — 4 этапа конвейера

### Этап 1 — сбор данных

| Скрипт | Что делает |
|--------|-----------|
| `1_make_dataset.py` | Захват лиц с камеры (Haar Cascade). Легаси, но рабочий. |

### Этап 2 — извлечение признаков (изображения → CSV)

Две параллельные ветки:

| Ветка | Технология | Файлы |
|-------|-----------|-------|
| **Эмоции (лицо)** | MediaPipe FaceLandmarker → 468 расстояний от кончика носа, нормировка на межзрачковое | `2_extract_geometric.py`, `2_extract_descriptors.py` (легаси) |
| **Позы (тело)** | COCO keypoints → эвристическая разметка standing/sitting/lying → 17 или 33 расстояний от центра таза | `2e_build_body_dataset.py` (COCO val), `2f_build_body_holistic.py` (MediaPipe Holistic 33 точки), `2f_build_body_fast.py` (упрощённая), `2g_build_body_train2017.py` (COCO train) |

### Этап 3 — обучение моделей

#### Эмоции

| Скрипт | Модель |
|--------|--------|
| `3_train_lightgbm.py` | LightGBM на 468 геометрических признаках → 7 классов эмоций |
| `3_train_svm.py` | SVM (запасной вариант) |

#### Позы тела — эволюция

История проб и ошибок от простого к сложному:

| # | Скрипт | Идея |
|---|--------|------|
| 1 | `3e_train_body_lightgbm.py` | Базовая LightGBM на 17 COCO-точках |
| 2 | `3f_train_body_holistic.py` | Переход на Holistic 33 точки |
| 3 | `3g_tune_body_holistic.py` | Тюнинг гиперпараметров |
| 4 | `3h_train_body_full.py` | Расширенные признаки: +8 relational +3 foot, GridSearch |
| 5 | `3i_force_lying.py` | Попытка форсировать lying через class_weight |
| 6 | `3j_clean_retrain.py` | Откат к честному обучению (без SMOTE), Platt-калибровка |
| 7 | `3k_merge_train.py` | Слияние val2017 + train2017, финальная 3-классовая модель |
| 8 | `3l_upright_vs_lying.py` | Бинарный классификатор upright vs lying (внутри upright — standing/sitting) |
| 9 | `3m_orient_features.py` | **Финал**: ориентационные признаки, lying ×20 вес, dart-бустинг |

Итоговая модель позы: **71.7%** на COCO (LightGBM, 44 признака).

### Этап 4 — рантайм

| Скрипт | Что делает |
|--------|-----------|
| `4_identification.py` | **Главный боевой файл**. Камера → SCRFD+ArcFace (лицо) + MediaPipe Holistic (лицо+тело) + LightGBM_A (эмоции) + LightGBM_Б (поза). Всё в одном цикле, с PIL-оверлеем на кириллице. |

---

## `utils/` — библиотеки и утилиты

| Модуль | Роль |
|--------|------|
| `face_id.py` | Обёртка над InsightFace (buffalo_l): SCRFD-детектор + ArcFace 512d. Косинусное сходство, порог 0.35. CUDA-прелоад. |
| `emotion.py` | EfficientNet-B0 (FER2013) — fallback-классификатор эмоций. Используется, если LightGBM не загрузился. |
| `emotion_hse.py` | HSEmotion — альтернативный fallback, используется в старом `live_monitor.py`. |
| `who_is_at_pc.py` | Однофотонный идентификатор: кадр с камеры → имя. Вызывается по команде «Кто за ПК?». |
| `whois.py` | Вариант who_is_at_pc. |
| `live_monitor.py` | Старая версия реалтайм-монитора (EmoZid), без MediaPipe Holistic и модели позы. |
| `capture_my_emotions.py` | Захват датасета эмоций с камеры. |
| `capture_reference.py` | Захват эталонных фото для галереи Face ID. |
| `train_emotion.py` | Обучение классификатора эмоций. |

---

## Ключевые архитектурные решения

1. **Две системы эмоций.** LightGBM на геометрических признаках (быстро, ~0.05ms на кадр) — основной. EfficientNet на сырых пикселях (точнее, но тяжелее) — fallback.

2. **Модель позы — двухэтапная в рантайме.** Сначала эвристика: если `стопы/голова > 1.25` → сразу `lying`. Иначе ML-модель решает между `standing` и `sitting`. Эвристика отсекает самый неоднозначный класс без нагрузки на модель.

3. **Кириллица через PIL.** OpenCV не поддерживает русский шрифт — весь текстовый оверлей рисуется через `PIL.ImageDraw` с DejaVuSans.

4. **CUDA-прелоад.** Перед импортом onnxruntime принудительно загружаются `.so` из nvidia-* pip-пакетов через `ctypes.CDLL(…, RTLD_GLOBAL)`. Без этого onnxruntime-gpu не видит CUDA.

5. **Стек рантайма:** SCRFD (детекция лиц) → ArcFace 512d (эмбеддинги) → MediaPipe Holistic (face 478 + body 33 + hands) → LightGBM_A (7 эмоций, ~48.5%) → LightGBM_Б (3 позы, 71.7%).

---

