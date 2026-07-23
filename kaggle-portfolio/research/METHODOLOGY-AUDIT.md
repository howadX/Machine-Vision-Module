# Аудит ML-методологии MVM

> Глубокий анализ подходов к обучению моделей в проекте.
> Ничего не меняем — только диагностика. Для ревизии и улучшения.

---

## A. Данные и разметка

### A1. Сбор данных

| Модель | Источник | Метод сбора | Объём |
|--------|----------|------------|-------|
| SVM (2019) | Приватный | `1_make_dataset.py` — Haar Cascade, 30 кадров за сессию, ESC для выхода | ~200/класс |
| LightGBM эмоции | Приватный + AffectNet | `capture_my_emotions.py` — свои 350 кадров | 350 + 14K |
| CNN эмоции | FER2013 + свои | `kagglehub` автозагрузка + свои с оверсемплингом до 15% | ~35K + 350×repeats |
| LightGBM позы | COCO val2017 | `extract_coco_poses.py` — MediaPipe PoseLandmarker | 6000 |
| GRIP позы | URFD + Zenodo + COCO | `spike_data.py` → `landmarks.npz` | ~14K+ |

**🔴 Ошибка `1_make_dataset.py` (2019):** запись **каждые 300 мс** подряд → соседние кадры почти идентичны. 30 кадров за 9 секунд = один expression, а не вариативность. Train/test split НЕ учитывает temporal proximity → кадры одного сеанса попадают и в train, и в test → **data leakage через temporal correlation**. Метрика 45% на SVM — возможно, завышена именно из-за этого.

### A2. Train/test split

| Скрипт | Метод | Стратификация | Group-aware | Оценка |
|--------|-------|:---:|:---:|--------|
| `train_svm.py` | `shuffle` + `dat[:pp], dat[pp+1:]` | ❌ | ❌ | **🔴 Off-by-one.** `pp+1:` пропускает один элемент. Нет stratification. Нет groups. |
| `train_lightgbm.py` | `train_test_split(stratify=y)` | ✅ | ❌ | **🟡 Приемлемо** для эмоций (один CSV — не видно групп), но нет CV |
| `train_combined.py` | `train_test_split(stratify=y)` | ✅ | ❌ | **🟡** 3-fold CV внутри Optuna — но на TRAIN, не на полном датасете |
| `train_emotion.py` | `np.random.permutation` + ручной split 80/20 | ❌ | ❌ | **🔴 Нет stratification** для 7-классового FER2013 с дисбалансом |
| `train_pose_classifier.py` | `train_test_split(stratify=y)` 70/15/15 | ✅ | ❌ | **🟡** Stratified — хорошо. Но VAE-эмбеддинги посчитаны на ВСЕХ данных → **data leakage** |
| `train_pose_vae.py` | `random_split` 80/20 | ❌ | ❌ | **🔴** VAE видит позы того же человека в train и val → overfitting |
| `train_pose_p1.py` (GRIP) | `StratifiedGroupKFold(n_splits=5)` | ✅ | ✅ | **🟢 Золотой стандарт.** Группы = люди. Все кадры человека либо в train, либо в test. |
| `distill_hse_to_mlp.py` | Файловый уровень: `merged/` vs `merged_test/` | ? | ? | **🟡** Неясно, разделены ли люди между train/test |

**🔴 Критическая находка:** почти все модели до GRIP игнорируют group structure. Для позы это критично: один человек в train и test → модель запоминает человека, а не позу. Реальные метрики на невиданных людях могут быть на **5–15 п.п. ниже** заявленных.

### A3. Качество разметки

| Датасет | Метки | Качество |
|--------|-------|----------|
| Приватный (2019) | Ручные: сам называл эмоцию перед камерой | **Низкое.** Self-reported emotion ≠ expression. Нет inter-annotator agreement. |
| FER2013 | Краудсорсинг | **Низкое.** ~15% ошибок (известная проблема FER2013) |
| AffectNet | Ручная разметка экспертами | **Высокое** |
| COCO позы | `classify_pose_heuristic()` | **Среднее.** ~25-35% шума (см. NB3) |
| URFD | Экспертная разметка падений | **Высокое** для standing/lying |
| Zenodo MultiPosture | Автоматическая + ручная | **Среднее** |

**🟡 Инсайт:** модель учится на шумных метках → её accuracy НЕ может превысить precision разметки. 71.7% на COCO-позах — это близко к precision эвристики (~65-75%). GRIP поднял accuracy не столько сменой модели, сколько **расширением датасета** точными метками URFD и Zenodo.

---

## B. Feature Engineering

### B1. Использованные подходы

| Подход | Где | Что | Оценка |
|--------|-----|-----|--------|
| **Сырые пиксели** | `train_emotion.py` | 224×224×3 → EfficientNet | **🟢 Стандарт.** CNN сама извлекает признаки |
| **Геометрические 68 расстояний** | SVM (2019) | Расстояния от кончика носа до 68 точек, нормировка на точку 27 | **🟡 Нормировка на одну точку нестабильна** (точка 27 — нижняя часть носа, зависит от выражения) |
| **Геометрические 468 расстояний** | LightGBM, MLP, дистилляция | Расстояния от носа до 468 точек, нормировка на interocular (lm33↔263) | **🟢 Хорошо.** Interocular стабильнее одной точки. Но зависимость от FaceLandmarker (падает на профиль) |
| **14 инженерных фич** | GRIP | Наклоны к гравитации, сгибы суставов, пропорции | **🟢 Отлично.** Viewpoint-инвариантны by design. Не зависят от landmark-детектора так сильно, как 468 расстояний |
| **99 сырых координат** | PoseVAE | 33×3 (x,y,z) → нормализация | **🟡 Зависимость от азимута катастрофическая (0.83→0.31 на 90°)** |

### B2. Нормализация

| Скрипт | Метод | Проблема |
|--------|-------|----------|
| SVM | `landmarks_vectorised / landmarks_vectorised[27]` | **🔴 Деление на одну точку.** Если точка 27 смещена (выражение лица), вся нормализация ломается |
| LightGBM эмоции | 468 расстояний / interocular | **🟢** Interocular — стабильный референс |
| PoseVAE | Центрирование в таз + масштаб по длине торса + Rodrigues-поворот | **🟢** Инвариантность к сдвигу/масштабу/наклону |
| GRIP | Признаки самомасштабируются (углы, пропорции) | **🟢** Не требуют нормализации |
| CNN | ImageNet-статистика `mean=[0.485,0.456,0.406]` | **🟢** Стандарт |

---

## C. Модели и обучение

### C1. Выбор архитектур

| Модель | Архитектура | Параметры | Обоснованность |
|--------|------------|-----------|---------------|
| SVM (2019) | `SVC(C=2.0, gamma=2.0, kernel='poly', degree=3)` | — | **🔴 Гиперпараметры вбиты вручную, без поиска.** C=2.0, gamma=2.0 — откуда эти числа? |
| LightGBM (эмоции) | `n_estimators=100, lr=0.1, max_depth=6` | — | **🟡 Дефолты.** Без HPO. Accuracy могла бы быть выше |
| LightGBM (позы) | `n_estimators=100, lr=0.1, max_depth=6` | — | **🟡 Те же дефолты.** Не tuned |
| LightGBM + Optuna | 9-параметрический поиск, 15 trials, `balanced_accuracy` | — | **🟢** HPO — хорошо. Но 15 trials для 9 параметров — мало (рекомендуется 50-100) |
| EfficientNet-B0 | `timm.create_model('efficientnet_b0', pretrained=True)` | ~5.3M | **🟢** Transfer learning со взрослой модели — стандарт |
| PoseVAE | `99→128→64→μ/logvar:32`, decoder симметрично | 48.6K | **🟡** Неплохо для 33×3 данных. Но 1000 эпох на малом датасете — риск overfitting |
| PoseClassifier | `32→64→32→6` | 4.39K | **🟢** Лёгкая голова на замороженном VAE |
| EmotionMLP | `BN(468)→512→512→256→128→7` | ~400K | **🟡** 400K параметров для 468-мерного входа — potential overfitting |
| GRIP RF | `RandomForest(n_estimators=120, max_depth=16)` | — | **🟢** 120 деревьев — достаточно. max_depth=16 — разумно для 14 признаков |

### C2. Функции потерь

| Модель | Loss | Проблема |
|--------|------|----------|
| SVM | Hinge (встроен в SVC) | **🟢** |
| LightGBM | Multi-logloss | **🟢** |
| EfficientNet-B0 | `CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)` | **🟢** Label smoothing + class weights — грамотно |
| PoseVAE | `MSE(recon) + β·KL`, β=0.001 | **🟡** β=0.001 (VPoser) — упор на реконструкцию. Для anomaly detection это ок, для генерации — может давать нереалистичные сэмплы |
| EmotionMLP (дистилляция) | `α·CE(hard) + (1-α)·KL(soft/T)·T²`, α=0.3, T=3.0 | **🟢 Классическая формула Hinton'а.** T² scaling — правильно |
| PoseClassifier | `CrossEntropyLoss(weight=class_weights)` | **🟢** |

### C3. Оптимизация и регуляризация

| Модель | Оптимизатор | LR schedule | Регуляризация | Gradient clipping |
|--------|------------|-------------|---------------|:---:|
| SVM | Встроенный | — | C=2.0 (L2) | — |
| LightGBM | GBDT (встроенный) | — | max_depth, min_child_samples | — |
| EfficientNet-B0 | `AdamW(lr=3e-4, weight_decay=1e-4)` | `OneCycleLR(pct_start=0.1)` | Weight decay + label smoothing + RandAugment | ❌ |
| PoseVAE | `Adam(lr=1e-3)` | ❌ Нет (ручной loop) | β-KL | ❌ |
| PoseClassifier | `AdamW(lr=1e-3, weight_decay=1e-4)` | `CosineAnnealingLR(T_max=200, eta_min=1e-5)` | Weight decay + dropout 0.3 | ❌ |
| EmotionMLP | `AdamW(lr=1e-3, weight_decay=1e-4)` | `CosineAnnealingWarmRestarts(T_0=200, T_mult=2)` | Weight decay + dropout 0.5 + grad clip | ✅ `clip_grad_norm_(1.0)` |
| GRIP RF | Gini impurity | — | max_depth=16, n_estimators=120 | — |

**🔴 Пропущен gradient clipping** в EfficientNet-B0 и PoseVAE. При нестабильных данных (FER2013 с шумом) это может приводить к spike-градиентам.

**🟡 PoseVAE без LR schedule.** 1000 эпох с постоянным lr=1e-3 — к концу обучения loss может осциллировать, а не сходиться.

### C4. Early stopping

| Модель | Механизм | Оценка |
|--------|----------|--------|
| PoseVAE | `PATIENCE=50`, следит за val loss | **🟢** |
| EfficientNet-B0 | Сохраняет чекпоинт по best val_acc | **🟡** Не останавливается, но сохраняет лучший |
| EmotionMLP | Сохраняет чекпоинт по best test_acc + NaN-guard | **🟡** NaN-guard — хорошо. Но 3000 эпох × 512 batch — многовато для ~15K примеров |
| Остальные | ❌ | **🔴** Риск overfitting'а не обнаружен |

---

## D. Оценка и метрики

### D1. Использованные метрики

| Модель | Метрики | Проблемы |
|--------|---------|----------|
| SVM | `clf.score()` = accuracy | **🔴 Только accuracy** для несбалансированных данных |
| LightGBM | Accuracy, confusion matrix, classification_report | **🟢** |
| LightGBM + Optuna | `balanced_accuracy` как objective + accuracy на holdout | **🟢** Balanced accuracy для HPO — зрелое решение |
| EfficientNet-B0 | Accuracy + `mean_conf` (средняя уверенность) | **🟢** Mean confidence как прокси калибровки — нестандартно, но разумно |
| PoseVAE | MSE, KL, accuracy классификатора | **🟡** Только accuracy для классификатора. Нет per-class метрик |
| PoseClassifier | Accuracy, macro F1, classification_report, confusion matrix | **🟢** |
| GRIP | Accuracy, macro F1, confusion matrix, per-fold metrics | **🟢 Золотой стандарт** |
| EmotionMLP | Accuracy от argmax софт-меток | **🟡** Accuracy на софт-метках ≠ accuracy на реальных метках. Софт-метки teacher'а могут быть самоуверенными |

### D2. Validation strategy

| Уровень | Где используется | Оценка |
|---------|-----------------|--------|
| **Single holdout** | SVM, LightGBM, PoseVAE, CNN | **🔴 Недостаточно.** Одна случайная выборка — метрика зависит от удачности разбиения |
| **3-fold CV** | LightGBM + Optuna (внутри HPO) | **🟡** Минимально приемлемо |
| **Stratified 5-fold** | GRIP (не grouped) | **🟡** Stratified — хорошо, но без групп |
| **Grouped 5-fold** | GRIP (`StratifiedGroupKFold`) | **🟢 Индустриальный стандарт** |

**🔴 Ключевой пробел:** ни одна модель кроме GRIP не использует grouped validation. Для позы и эмоций это критично: accuracy на «невиданных кадрах того же человека» может быть на 10–20 п.п. выше, чем на «невиданных людях».

---

## E. Production-переход

### E1. Экспорт моделей

| Модель | Формат | В production? | Проблемы |
|--------|--------|:---:|-----------|
| SVM | `pickle` | ❌ | sklearn-зависимость. Не портируется в Rust |
| LightGBM | `pickle` | ❌ | C++-рантайм. Не портируется без liblightgbm |
| EfficientNet-B0 | `torch.save` (.pth) | ❌ | PyTorch-зависимость. Не конвертирован в ONNX |
| EmotionMLP | `torch.save` + `StandardScaler` pickle | ❌ | PyTorch-зависимость |
| EmotiEffLib B0 | ONNX | ✅ | **🟢** ONNX — стандарт для кросс-платформенного инференса |
| PoseVAE | `.bin` (свой бинарный формат) | ✅ (OOD-gate) | **🟢** Кастомный бинарный формат под `include_bytes!` |
| GRIP RF | `.bin` (свой бинарный формат) | ✅ | **🟢** ~270 КБ, 0 зависимостей |

**🟡 Инсайт:** только 3 из 9+ моделей дошли до production. Остальные — «исследовательский код», который никогда не был предназначен для деплоя. Это нормально для R&D, но в портфолио стоит явно разграничить: «исследование» vs «production».

### E2. Воспроизводимость

| Практика | Статус | Где |
|----------|:------:|-----|
| **Фиксированные seeds** | ✅ | `random_state=42` в большинстве скриптов |
| **Requirements.txt** | ✅ | `prototype/requirements.txt` (но без версий!) |
| **Docker** | ❌ | Нет Dockerfile для тренировки |
| **Конфиги моделей** | 🟡 | JSON-конфиги для GRIP, EmotionMLP, PoseVAE. Но не для SVM, LightGBM |
| **Датасеты версионированы** | ❌ | CSV/NPZ лежат в репозитории без хешей и версий |
| **Experiment tracking** | ❌ | Нет W&B/MLflow/TensorBoard. Только print() и JSON-отчёты |

### E3. Тестирование моделей

| Практика | Статус | Где |
|----------|:------:|-----|
| **Unit-тесты для инференса** | ✅ | 670 тестов в Rust (mvm/) |
| **Golden-тесты (Python↔Rust)** | ✅ | `spike_pose/golden_refs.py` + `pose_p1_golden.in` |
| **Smoke-тесты (модель не падает)** | ❌ | Нет для эмоций |
| **Regression-тесты (accuracy не упала)** | ❌ | Нет ни для одной модели |
| **Тесты на известных данных** | ❌ | **Из-за этого баг HWC/NCHW жил 4 фазы** |

**🔴 Критический инсайт:** баг EmotiEffLib (HWC/NCHW) жил с фазы 6 по фазу 10 **потому что не было численного теста на известном лице**. Модель выдавала near-uniform softmax на любом входе, но accuracy на FER2013 всё равно была ~60% (random guess = 14%) — «достаточно хорошо», чтобы не заподозрить баг. Golden-тест GRIP — **единственная защита от таких багов**. Нужен такой же для эмоций.

---

## F. Сравнение с индустриальными стандартами

### F1. Что сделано правильно (на уровне Middle/Senior)

| Практика | Где |
|----------|-----|
| **Grouped cross-validation** | GRIP — золотой стандарт, редкость даже в индустрии |
| **Label smoothing + class weights** | EfficientNet-B0 — борется и с overconfidence, и с дисбалансом |
| **Knowledge distillation (Hinton)** | EmotionMLP — правильная формула с T² |
| **Weak supervision + анализ шума** | NB3 — data-centric мышление, редкость |
| **Инвариантность by design** | GRIP — признаки спроектированы, а не выучены |
| **Честные метрики** | «Что измерено vs что спроецировано» — редкость |
| **Экспорт в бинарный формат** | GRIP, PoseVAE — кастомный формат для Rust |
| **HPO с balanced_accuracy** | LightGBM + Optuna — правильный выбор метрики |
| **NaN-guard при обучении** | EmotionMLP — защита от взрыва градиентов |
| **Occlusion-honesty** | mvm-fusion — не утверждать то, что не видно |

### F2. Что отсутствует (разрыв с индустрией)

| Практика | Важность | Что теряем |
|----------|:--------:|-----------|
| **Experiment tracking (W&B/MLflow)** | 🔴 Высокая | Невозможно сравнить запуски, воспроизвести результат через месяц |
| **Model registry** | 🟡 Средняя | Модели лежат в файлах, нет версионирования |
| **Data versioning (DVC/Kaggle Datasets)** | 🟡 Средняя | Неизвестно, на какой версии датасета обучена модель |
| **Mixed precision (AMP)** | 🟢 Низкая | Ускорение в 1.5-2× на GPU — не критично для CPU-инференса |
| **Distributed training** | 🟢 Низкая | Не нужно для моделей такого размера |
| **A/B-тесты в production** | 🟡 Средняя | Нельзя измерить, стала ли модель лучше на живых данных |
| **Мониторинг моделей (data drift)** | 🟡 Средняя | Не узнаем, когда распределение входных данных изменилось |
| **CI/CD для тренировки** | 🟢 Низкая | Тренировка ручная — ок для соло-проекта |
| **Docker для тренировки** | 🟡 Средняя | «На моей машине работает» — для соло-проекта ок, для команды нет |

---

## G. Что исправить сейчас (ревизия моделей)

### G1. Критические (влияют на метрики)

1. **Добавить grouped cross-validation во все модели позы и эмоций.**
   - Проблема: accuracy на «невиданных кадрах того же человека» ≠ accuracy на «невиданных людях»
   - Действие: перезапустить LightGBM (71.7%), PoseVAE, PoseClassifier с `StratifiedGroupKFold`
   - Ожидаемый эффект: accuracy упадёт на 5–15 п.п. → **честные метрики для портфолио**

2. **Пересчитать accuracy эмоций на grouped split.**
   - FER2013: нет метаданных о людях → использовать аугментации как прокси
   - AffectNet: есть identity → можно сделать grouped
   - Ожидаемый эффект: 73% → возможно 60-65% на невиданных людях

3. **Проверить data leakage в VAE-пайплайне.**
   - `train_pose_classifier.py`: VAE обучен на ВСЕХ данных → эмбеддинги для train и test «знают» друг друга
   - Действие: обучить VAE только на train, затем вычислить эмбеддинги отдельно
   - Ожидаемый эффект: accuracy классификатора упадёт на 3–5 п.п.

4. **Добавить golden-тесты для эмоций.**
   - Метод: одно известное лицо с известной эмоцией → прогнать через EmotiEffLib → сравнить с эталонным прогоном
   - Защита от багов типа HWC/NCHW в будущем

### G2. Средние (улучшают качество)

5. **HPO для LightGBM-позы (71.7%).**
   - Сейчас: дефолтные параметры `n_estimators=100, lr=0.1, max_depth=6`
   - Действие: Optuna, 50+ trials, grouped CV
   - Ожидаемый эффект: +3–8 п.п. accuracy

6. **HPO для GRIP RF.**
   - Сейчас: `n_estimators=120, max_depth=16` — вбиты вручную
   - Действие: Optuna по `n_estimators`, `max_depth`, `min_samples_split`
   - Ожидаемый эффект: ≈0.93 → возможно 0.965+

7. **Заменить `train_test_split` на `StratifiedKFold` в CNN-тренировке.**
   - 5-fold CV вместо одного holdout'а
   - Отчёт: accuracy μ±σ, а не одно число

### G3. Портфолио (улучшают нарратив)

8. **Замерить latency CPU для ВСЕХ моделей.**
   - Единый бенчмарк: SVM vs LightGBM vs CNN vs MLP vs EmotiEffLib vs RF
   - Таблица accuracy↔latency↔size — для каждого ноутбука

9. **Замерить confidence calibration.**
   - Reliability diagram для EfficientNet-B0 vs EmotiEffLib
   - Expected Calibration Error (ECE)
   - Это покажет, насколько модель «честна» в своих вероятностях

10. **Документировать data leakage в distill_hse_to_mlp.py.**
    - Если HSEmotion обучался на AffectNet → teacher-разметка того же датасета = «учитель знает ответы»
    - Честно указать в NB1: «teacher trained on overlapping data → soft labels are optimistic»

---

## H. Итоговая оценка

**Сильные стороны (на уровне Senior):**
- Grouped CV в GRIP — индустриальный стандарт, редкость
- Label smoothing + class weights + RandAugment — зрелый training pipeline
- Knowledge distillation с правильной формулой Hinton'а
- Feature engineering с физической инвариантностью
- Golden-тесты и bit-точность Python↔Rust

**Средний уровень (норма для Middle):**
- HPO через Optuna (но мало trials)
- Class balancing через веса и оверсемплинг
- Early stopping и gradient clipping (не везде)
- Экспорт ONNX и кастомных бинарных форматов

**Слабые стороны (на уровне Junior):**
- Отсутствие grouped validation почти везде → метрики завышены
- Нет experiment tracking → невоспроизводимость
- Нет тестов для моделей → баг HWC/NCHW как симптом
- Off-by-one и отсутствие stratification в раннем коде (SVM 2019)
- Data leakage в VAE-пайплайне и дистилляции

**Общий вердикт:**

Это **честный Middle+ уровень** с элементами Senior (GRIP) и Junior (ранний код). Самое важное: **методология эволюционирует**. Ранний код (2019–2023) — джуновские ошибки (нет stratification, off-by-one, дефолтные параметры). Поздний код (2025–2026) — зрелые решения (grouped CV, HPO, инвариантные признаки, golden-тесты).

Для портфолио это **плюс**, а не минус: видна траектория роста. Но нужно честно отразить в ноутбуках: «ранние модели (NB0, NB1) имели методологические проблемы X и Y. В GRIP (NB3b) мы их исправили.»

**Главное, что нужно сделать прямо сейчас для ревизии:** перезапустить все модели с grouped validation и честно записать оба числа: «на невиданных кадрах» и «на невиданных людях». Это превратит «возможно завышенные метрики» в «честно измеренные метрики» — что для портфолио важнее абсолютных цифр.
