# Карта артефактов для Kaggle-ноутбуков

> Как ссылаться в промтах к Claude Code. Для каждого ноутбука — где код, данные, документация.

**Корень проекта:** `~/repos/Machine Vision Module/`
**Kaggle-портфолио:** `~/repos/Machine Vision Module/kaggle-portfolio/`

---

## NB0 — EmoZid: SVM на 68 точках лица (2019)

### Код (legacy)

| Файл | Что внутри |
|------|-----------|
| `prototype/legacy/dlib/2_extract_descriptors.py` | dlib HOG → 68 landmarks → 128d эмбеддинги → SVM. **Оригинальный код 2019 года.** |
| `prototype/legacy/dlib/haarcascade_frontalface_default.xml` | Haar Cascade (OpenCV) — первичный детектор |
| `prototype/legacy/dlib/Em_1.txt` … `Em_7.txt` | Файлы с геометрическими признаками для каждой эмоции (68 расстояний от носа) |
| `prototype/legacy/dlib/Em_1_128.txt` … `Em_7_128.txt` | Файлы с dlib 128d-эмбеддингами для каждой эмоции |

### Данные

| Файл | Что внутри |
|------|-----------|
| `prototype/legacy/dlib/Em_*.txt` | Приватные данные (я + друзья). **НЕ публиковать.** Использовать CK+ как публичный аналог |

### Документация

| Файл | Что внутри |
|------|-----------|
| `prototype/pipeline/README.md` (первые 40 строк) | История EmoZid: «Этот README — оригинальная инструкция к проекту EmoZid (~2019–2020)» |
| `prototype/run.sh` | Запуск EmoZid: «EmoZid v4: идентификация + тренировка эмоций» |
| `kaggle-portfolio/nb0-emozid-skeleton.md` | Скелет ноутбука |

### Что сказать Claude Code

```
Работаем над NB0 (EmoZid). Исходный код 2019 года в prototype/legacy/dlib/2_extract_descriptors.py.
Оригинальные данные приватные — замени на CK+ (Cohn-Kanade) через kagglehub.
Стек: dlib HOG → 68 landmarks → dlib ResNet 128d → sklearn SVM (poly kernel).
Ноутбук — публичная реконструкция подхода.
```

---

## NB1 — Distillation: CNN → geometric MLP (2023–2025)

### Код (prototype/pipeline)

| Файл | Что внутри |
|------|-----------|
| `prototype/pipeline/train_emotion.py` | **Teacher:** EfficientNet-B0 на FER2013 + свои данные. Label smoothing 0.1, RandAugment, class weights, OneCycleLR. |
| `prototype/pipeline/distill_hse_to_mlp.py` | **Дистилляция:** HSEmotion (8 кл.) → MediaPipe FaceLandmarker 468 точек → софт-метки для student |
| `prototype/pipeline/train_mlp_distilled.py` | **Student:** MLP на дистиллированных данных |
| `prototype/pipeline/extract_geometric.py` | Извлечение 468 геометрических признаков (расстояния от носа) |
| `prototype/pipeline/train_lightgbm.py` | LightGBM-бейзлайн на геометрических признаках |

### Данные

| Файл | Что внутри |
|------|-----------|
| `pipeline/geometric_468.csv` | 468 признаков для своих 350 кадров |
| `pipeline/geometric_affectnet.csv` | 468 признаков для AffectNet (~14-16K) |
| `pipeline/geometric_distilled_train.csv` / `geometric_distilled_test.csv` | **Дистиллированные данные:** 468 признаков + софт-метки HSEmotion |

### Модели (prototype/models/emotion)

| Файл | Что внутри |
|------|-----------|
| `prototype/models/emotion/mlp_emotion_distilled.pth` + `_scaler.pkl` + `_config.json` | Student-MLP (дистиллированный) |
| `prototype/models/emotion/efficientnet_b0_fer2013.pth` | Teacher CNN (через timm) |
| `prototype/models/emotion/lightgbm_combined_14k.pkl` | LightGBM-бейзлайн |
| `prototype/legacy/svm/efficientnet_b0_fer2013.pth` | Ещё одна копия CNN |

### Документация

| Файл | Что внутри |
|------|-----------|
| `kaggle-portfolio/nb1-distillation-skeleton.md` | Скелет ноутбука |
| `prototype/pipeline/README.md` | Контекст пайплайна |

### Production (mvm/)

| Файл | Что внутри |
|------|-----------|
| `mvm/crates/mvm-emotion/src/lib.rs` | EmotiEffLib B0 ONNX — production emotion |
| `mvm/crates/mvm-detect/src/lib.rs` | YuNet — детектор лиц |
| `mvm/CLAUDE.md` | Раздел «Эмоции — EmotiEffLib B0», баг HWC/NCHW, фикс |

### Что сказать Claude Code

```
Работаем над NB1 (Distillation). Teacher: prototype/pipeline/train_emotion.py (EfficientNet-B0, FER2013).
Дистилляция: prototype/pipeline/distill_hse_to_mlp.py (HSEmotion→MLP, 468 признаков).
Данные: geometric_distilled_train.csv (468 фич + софт-метки).
В ноутбуке показать: teacher inference → soft labels → student training → сравнение accuracy/latency/size.
Важно: production (mvm-emotion) использует EmotiEffLib, а не этого student'а — честно объяснить почему.
```

---

## NB2 — PoseVAE: manifold + few-shot + anomaly (2024–2025)

### Код (prototype/pipeline + core)

| Файл | Что внутри |
|------|-----------|
| `prototype/pipeline/train_pose_vae.py` | **VAE:** 99→32 latent, 1000 эпох, KL-дивергенция |
| `prototype/pipeline/train_pose_classifier.py` | Классификатор на latent VAE (linear probe) |
| `prototype/core/pose_vae.py` | Класс PoseVAE — encoder/decoder/reparameterize |
| `prototype/core/pose_heads.py` | PoseClassifier — голова на latent |
| `prototype/core/pose_encoder.py` | `normalize_landmarks()` — нормализация 33 landmarks |

### Данные

| Файл | Что внутри |
|------|-----------|
| `pipeline/poses_dataset.csv` | Ранний датасет поз (свои) |
| `pipeline/poses_dataset_full.csv` | Расширенная версия |
| `pipeline/coco_poses.csv` | COCO позы после `extract_coco_poses.py` |

### Модели

| Файл | Что внутри |
|------|-----------|
| `prototype/models/emotion/vposer_coco33.pth` + `_config.json` | PoseVAE веса |
| `prototype/models/emotion/pose_classifier.pth` + `_config.json` | Классификатор на latent |

### Production (mvm/ + spike_pose)

| Файл | Что внутри |
|------|-----------|
| `spike_pose/train_pose_p1.py` | Переобучение VAE на публичных (URFD+Zenodo) — **актуальная production-версия** |
| `spike_pose/pose_vae_public.bin` | VAE веса для Rust |
| `mvm/crates/mvm-posture/src/classifier.rs` | Rust-реализация VAE-инференса (OOD-gate) |
| `mvm/CLAUDE.md` | Раздел «Поза — method C», VAE-порог 14.3→170 |

### Что сказать Claude Code

```
Работаем над NB2 (PoseVAE). VAE: prototype/pipeline/train_pose_vae.py + core/pose_vae.py.
Данные: poses_dataset.csv (ранний) + coco_poses.csv (COCO). Для ноутбука — использовать публичный COCO.
Показать: обучение VAE → UMAP латента → интерполяция → few-shot (5 примеров) → anomaly (-log p(z)).
Важно: в production VAE = только OOD-gate, классификацию делает method-C (NB3b). Честно указать метрики.
Актуальные веса: spike_pose/pose_vae_public.bin (переобучены на URFD+Zenodo, порог 170).
```

---

## NB3 — Weak Supervision (2024)

### Код

| Файл | Что внутри |
|------|-----------|
| `prototype/pipeline/extract_coco_poses.py` | **Главный файл.** `classify_pose_heuristic()` — 5 правил. Прогон MediaPipe по COCO. |
| `prototype/pipeline/train_combined.py` | LightGBM + Optuna на геометрических признаках (эмоции + позы) |
| `prototype/pipeline/train_lightgbm.py` | LightGBM на COCO-позах |
| `prototype/legacy/body_experiments/2e_build_body_dataset.py` | Ранняя версия: извлечение поз из COCO, другая эвристика |
| `prototype/legacy/body_experiments/3e_train_body_lightgbm.py` | Ранняя версия: обучение LightGBM |

### Данные

| Файл | Что внутри |
|------|-----------|
| `pipeline/coco_poses.csv` | Результат extract_coco_poses.py: 6000 поз с weak-метками |
| `prototype/models/descriptors/body_posture_coco.csv` | Ранний формат данных (legacy) |

### Модели

| Файл | Что внутри |
|------|-----------|
| `prototype/models/emotion/lightgbm_body_holistic_33.pkl` | LightGBM для позы (71.7%) |
| `prototype/legacy/body_experiments/lightgbm_body_posture.pkl` | Ранняя версия LightGBM |

### Документация

| Файл | Что внутри |
|------|-----------|
| `kaggle-portfolio/nb3-weak-supervision-skeleton.md` | Скелет — **там детальный анализ шума (4 типа ошибок)** |

### Production (spike_pose)

| Файл | Что внутри |
|------|-----------|
| `spike_pose/spike_data.py` | Сбор данных URFD + Zenodo + COCO → landmarks.npz |
| `spike_pose/data_probe/landmarks.npz` | **Датасет для method-C** (33×4 landmarks + y + grp + src) |

### Что сказать Claude Code

```
Работаем над NB3 (Weak Supervision). Главный файл: prototype/pipeline/extract_coco_poses.py.
В нём classify_pose_heuristic() — 5 правил с confidence.
Данные: COCO val2017 → MediaPipe PoseLandmarker → 33 landmarks.
В ноутбуке: эвристика → анализ шума (вручную проверить K=100 примеров) → LightGBM + Optuna → 71.7%.
Опубликовать derived-датасет на Kaggle (CC BY 4.0, атрибуция COCO).
Важно: НЕ публиковать изображения — только keypoints + weak-метки + инженерные фичи.
Связать с NB3b (method-C): «эти данные + URFD + Zenodo → 0.959».
```

---

## NB3b — Method-C: 14 features beat VAE (2025–2026)

### Код (spike_pose)

| Файл | Что внутри |
|------|-----------|
| `spike_pose/spike_feats.py` | **14 инженерных фич.** `compute_features()`, `_ang_to_vert()`, `_flex()`, azimuth-тест |
| `spike_pose/train_pose_p1.py` | **Полный прогон.** Grouped 5-fold CV, аугментации, экспорт в Rust, golden-тесты |
| `spike_pose/spike_data.py` | Сбор данных URFD + Zenodo + COCO |
| `spike_pose/export_for_rust.py` | Экспорт моделей в бинарный формат |
| `spike_pose/golden_refs.py` | Golden-тесты для бит-точности Rust↔Python |
| `spike_pose/pose_p1_golden.in` | Golden-векторы для Rust-тестов |

### Данные

| Файл | Что внутри |
|------|-----------|
| `spike_pose/data_probe/landmarks.npz` | **Датасет.** X (33×4), y (0=lying/1=sitting/2=standing), grp (id человека), src |

### Модели

| Файл | Что внутри |
|------|-----------|
| `spike_pose/pose_rf_forest.bin` | **Production-модель.** RandomForest в бинарном формате (12 КБ) |
| `spike_pose/pose_vae_public.bin` | VAE для OOD-gate (публичные веса) |
| `spike_pose/pose_p1_config.json` | Конфиг: class_names, feature_names, n_trees |

### Production (mvm/)

| Файл | Что внутри |
|------|-----------|
| `mvm/crates/mvm-posture/src/classifier.rs` | Rust-реализация: загрузка RF из бинарника, инференс, VAE-gate |
| `mvm/CLAUDE.md` | Раздел «Поза — method C»: метрики, occlusion-honesty, BodyCoverage |

### Что сказать Claude Code

```
Работаем над NB3b (Method-C). Код: spike_pose/spike_feats.py (14 фич) + spike_pose/train_pose_p1.py (полный прогон).
Данные: spike_pose/data_probe/landmarks.npz.
В ноутбуке: 14 фич с объяснением → grouped 5-fold CV → 0.959 ± 0.01 → azimuth-тест (0.959→0.957 на 90°).
Показать сравнение: сырые координаты (99d) → 0.83, VAE latent (32d) → 0.85, фичи (14d) → 0.959.
Экспорт в бинарный формат для Rust (12 КБ).
Бонус: occlusion-honesty (BodyCoverage — mvm-fusion).
```

---

## NB4 — Teardown: identity embeddings ≠ emotion (2023)

### Код

| Файл | Что внутри |
|------|-----------|
| `prototype/core/face_id.py` | SCRFD + ArcFace — identity-эмбеддинги |
| `prototype/core/emotion_hse.py` | HSEmotion — emotion-классификатор |
| `prototype/tools/whoishere.py` | Продакшен-скрипт: identity + emotion в одном проходе |

### Документация

| Файл | Что внутри |
|------|-----------|
| `prototype/pipeline/legacy_pipeline_comparison.md` | Сравнение legacy vs modern |
| `prototype/pipeline/live_monitor.py` | Живой мониторинг, запуск EmoZid |

### Production (mvm/)

| Файл | Что внутри |
|------|-----------|
| `mvm/crates/mvm-detect/src/lib.rs` | YuNet — детектор |
| `mvm/crates/mvm-identity/src/lib.rs` | SFace — identity (128d) |
| `mvm/crates/mvm-emotion/src/lib.rs` | EmotiEffLib B0 — emotion |
| `mvm/CLAUDE.md` | Таблица детекторов, identity lifecycle |

### Что сказать Claude Code

```
Работаем над NB4 (Teardown). Это written notebook — 2 code-ячейки, 10 markdown.
Главная идея: identity-эмбеддинги (ArcFace/SFace) обучены игнорировать эмоции как шум → не подходят для emotion classification.
Эмпирическое доказательство: UMAP любого face-эмбеддинга — раскраска по человеку даёт кластеры, по эмоции — кашу.
Таблицы сравнения детекторов/landmarks/эмбеддингов из mvm/CLAUDE.md.
P0–P3 приоритеты миграции. Кривая масштабирования.
```

---

## Общие ресурсы для всех ноутбуков

### Структура скелетов

| Файл | Для |
|------|-----|
| `kaggle-portfolio/nb0-emozid-skeleton.md` | NB0 |
| `kaggle-portfolio/nb1-distillation-skeleton.md` | NB1 |
| `kaggle-portfolio/nb2-posevae-skeleton.md` | NB2 |
| `kaggle-portfolio/nb3-weak-supervision-skeleton.md` | NB3 |
| `kaggle-portfolio/nb3b-method-c-skeleton.md` | NB3b |
| `kaggle-portfolio/nb4-pipeline-teardown-skeleton.md` | NB4 |
| `kaggle-portfolio/README.md` | Коллекция |

### Production-контекст

| Файл | Что внутри |
|------|-----------|
| `mvm/CLAUDE.md` | **Главный source of truth.** Архитектура, крейты, метрики, roadmap, баги |
| `mvm/AGENT.md` | Как LLM-агенту пользоваться MVM (pull-модель) |
| `AGENTS.md` (корень) | Философия проекта, структура, как работает |

### Python-прототип (prototype)

| Каталог | Что внутри |
|---------|-----------|
| `prototype/core/` | Python-модули: face_id, emotion_hse, pose_vae, pose_encoder, workspace_3d |
| `prototype/pipeline/` | Обучение: train_*.py, extract_*.py, capture_*.py, distill_*.py |
| `prototype/legacy/` | Архив: dlib, SVM, body_experiments |
| `prototype/models/` | Веса: emotion/, mediapipe/, references/ |
| `prototype/data/` | Датасеты: my_emotions/, datasets/ |
| `prototype/tools/` | Продакшен-инструменты: whoishere.py, sight.py |

---

## Порядок работы с Claude Code

Для каждого ноутбука — **три прохода**:

### Проход 1: перенос кода в `.ipynb`
```
Возьми файлы [код-1, код-2, ...] из [каталог].
Адаптируй под Kaggle: замени приватные данные на публичные, добавь kagglehub-загрузку, зафиксируй seeds.
Выдай готовый .ipynb со всеми code-ячейками.
```

### Проход 2: нарратив и визуализации
```
Добавь markdown-ячейки по скелету [nbX-skeleton.md].
Добавь визуализации: [список графиков].
Добавь Production Status из mvm/CLAUDE.md.
```

### Проход 3: верификация
```
Прогони ноутбук на Kaggle. Исправь ошибки.
Проверь: все seeds зафиксированы, все данные публичные, метрики честные.
Добавь секцию What Changed с датой сегодня.
```

---

## Ярлыки для промтов (копировать-ready)

### NB0 EmoZid
```
Код: prototype/legacy/dlib/2_extract_descriptors.py
Скелет: kaggle-portfolio/nb0-emozid-skeleton.md
Данные: CK+ через kagglehub (публичный аналог приватных Em_*.txt)
Продакшн: mvm/CLAUDE.md (раздел «Детекторы — актуальная таблица»)
```

### NB1 Distillation
```
Код: prototype/pipeline/train_emotion.py, distill_hse_to_mlp.py, train_mlp_distilled.py, extract_geometric.py
Скелет: kaggle-portfolio/nb1-distillation-skeleton.md
Данные: FER2013 через kagglehub + geometric_distilled_train.csv
Продакшн: mvm/CLAUDE.md (раздел «Эмоции — EmotiEffLib B0», баг HWC/NCHW)
```

### NB2 PoseVAE
```
Код: prototype/pipeline/train_pose_vae.py, prototype/core/pose_vae.py, prototype/core/pose_encoder.py
Скелет: kaggle-portfolio/nb2-posevae-skeleton.md
Данные: COCO val2017 keypoints + poses_dataset.csv
Продакшн: spike_pose/train_pose_p1.py (VAE-часть), mvm/crates/mvm-posture/src/classifier.rs
```

### NB3 Weak Supervision
```
Код: prototype/pipeline/extract_coco_poses.py (главный), train_combined.py, train_lightgbm.py
Скелет: kaggle-portfolio/nb3-weak-supervision-skeleton.md
Данные: COCO val2017 → MediaPipe PoseLandmarker → coco_poses.csv
Продакшн: spike_pose/spike_data.py, spike_pose/data_probe/landmarks.npz
```

### NB3b Method-C
```
Код: spike_pose/spike_feats.py + spike_pose/train_pose_p1.py
Скелет: kaggle-portfolio/nb3b-method-c-skeleton.md
Данные: spike_pose/data_probe/landmarks.npz (URFD+Zenodo+COCO)
Продакшн: spike_pose/pose_rf_forest.bin, mvm/crates/mvm-posture/src/classifier.rs
```

### NB4 Teardown
```
Код: prototype/core/face_id.py (identity), prototype/core/emotion_hse.py (emotion)
Скелет: kaggle-portfolio/nb4-pipeline-teardown-skeleton.md
Данные: любой публичный face-датасет с метками person+emotion
Продакшн: mvm/crates/mvm-identity/src/lib.rs, mvm/crates/mvm-emotion/src/lib.rs
```
