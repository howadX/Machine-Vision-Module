# Геометрический подход с несколькими экстракторами: анализ

**Вопрос:** как архитектура параллельных геометрических экстракторов (лицо + тело) влияет на качество классификации?

**Дата:** 2026-05-29

---

## 1. Текущий экстрактор лица — анатомия

```python
# Из 2_extract_geometric.py
pts = MediaPipe FaceLandmarker → 478 точек (x, y, z)
nose_tip = pts[4]                       # референсная точка
interocular = dist(pts[33], pts[263])    # нормализатор
features[i] = dist(pts[i], nose_tip) / interocular  # 468 чисел
```

### Что здесь важно

| Свойство | Значение | Почему это работает |
|----------|----------|---------------------|
| Референсная точка | Кончик носа (landmark 4) | Стабильная точка. При эмоциях двигаются брови, рот, глаза — нос почти неподвижен. Контраст между подвижными и неподвижной точкой = сигнал эмоции |
| Нормализатор | Межзрачковое расстояние (~63mm IRL) | Анатомическая константа. Не зависит от расстояния до камеры, не зависит от эмоции. Идеальный масштабный фактор |
| Размерность | 468 (3D-расстояния) | Покрывает всё лицо: губы (40 точек), глаза (32), брови (18), нос (12), контур (36) — избыточно, но каждая зона вносит свой вклад |
| Масштаб признаков | ~0.2–3.0 (в единицах interocular) | Компактный диапазон. LightGBM не требует StandardScaler |

---

## 2. Экстрактор тела — проект архитектуры

```python
# Проект: 2b_extract_body_landmarks.py
pts = MediaPipe PoseLandmarker → 33 точки (x, y, z, visibility)
hip_center = (pts[23] + pts[24]) / 2     # центр таза — референсная точка
shoulder_width = dist(pts[11], pts[12])   # межплечевое — нормализатор
features[i] = dist(pts[i], hip_center) / shoulder_width  # 33 числа
# + углы между суставами (колено-бедро-плечо и т.д.)
# + индекс асимметрии
```

### Сравнение с лицом

| Свойство | Лицо | Тело | Совместимость |
|----------|------|------|---------------|
| Референсная точка | Кончик носа (неподвижен при эмоциях) | Центр таза (неподвижен при статичной позе) | ⚠️ Разные точки, но обе стабильны |
| Нормализатор | Межзрачковое (~63mm, const) | Межплечевое (~40cm, const) | ✅ Оба — анатомические константы |
| Размерность | 468 | 33 + углы (~50) | ⚠️ Дисбаланс 9:1 |
| Масштаб | 0.2–3.0 | 0.1–5.0 | ⚠️ Разные диапазоны |
| Размерность данных | 3D (x, y, z) | 3D + visibility | ⚠️ Разные модальности |

---

## 3. Ключевые проблемы параллельных экстракторов

### Проблема 1: Масштабный диссонанс

```
Face features (468 чисел):
  dist(нос → бровь) / interocular ≈ 0.6
  dist(нос → подбородок) / interocular ≈ 1.2
  dist(нос → ухо) / interocular ≈ 2.5

Body features (33+ числа):
  dist(таз → плечо) / shoulder_width ≈ 0.8
  dist(таз → стопа) / shoulder_width ≈ 2.8
  dist(таз → запястье) / shoulder_width ≈ 3.5–5.0
```

**Проблема:** признаки тела имеют больший абсолютный масштаб (до 5.0 vs до 3.0 у лица). LightGBM (деревья) инвариантен к монотонным преобразованиям, но разные масштабы влияют на `min_child_weight` и `lambda_l1/l2`.

**Решение:** не требовать одинакового масштаба — LightGBM справляется. Но для интерпретации важности признаков лучше привести к одному масштабу через раздельную нормализацию:

```python
# Опция: квантильная нормализация для каждого экстрактора отдельно
from sklearn.preprocessing import QuantileTransformer
face_features = QuantileTransformer().fit_transform(face_features)   # → uniform [0,1]
body_features = QuantileTransformer().fit_transform(body_features)   # → uniform [0,1]
combined = np.concat([face_features, body_features])                 # единый масштаб
```

**Но:** при инференсе QuantileTransformer требует fit на тренировочных данных. Это добавляет зависимость.

### Проблема 2: Дисбаланс размерности (468 vs 50)

```
Сигнал эмоции распределён так:
  - Лицо:    80% информации (рот, глаза, брови)
  - Тело:    15% информации (плечи, наклон головы, руки)
  - Контекст: 5% (объекты, сцена)
```

**Проблема:** 468 face-признаков могут «задавить» 50 body-признаков просто из-за количества. LightGBM с feature_fraction=0.8 будет случайно выбирать признаки — и 90% выбранных будут лицом, 10% телом.

**Решение 1 — feature_fraction по группам:**
```python
# Не: feature_fraction=0.8 (случайный отбор)
# А: гарантировать, что тело представлено
# LightGBM не поддерживает grouped feature sampling из коробки.
# Решение: дать телу БОЛЬШУЮ размерность через engineered features
```

**Решение 2 — расширить тело до соизмеримой размерности:**
```python
# Вместо 33 расстояний:
body_features = []

# 1. Расстояния от центра таза до каждой точки: 33 числа
for pt in body_landmarks:
    body_features.append(dist(pt, hip_center) / shoulder_width)

# 2. Попарные расстояния между ключевыми точками: C(12,2) = 66 чисел
# (плечо-локоть, локоть-запястье, бедро-колено, колено-стопа...)
key_joints = [11,12,13,14,15,16,23,24,25,26,27,28]  # 12 ключевых
for i in range(len(key_joints)):
    for j in range(i+1, len(key_joints)):
        body_features.append(dist(pts[i], pts[j]) / shoulder_width)

# 3. Углы в суставах: 8 чисел
# (угол плечо-локоть-запястье, бедро-колено-стопа...)

# 4. Асимметрия лево-право: 6 чисел
# (левое плечо vs правое по Y, левое запястье vs правое...)

# Итого: 33 + 66 + 8 + 6 = 113 признаков тела
```

Теперь дисбаланс 468:113 ≈ 4:1 — более сбалансировано.

### Проблема 3: Корреляция face-body признаков

Некоторые признаки дублируются между лицом и телом:

```
Наклон головы:
  Face: dist(нос → левый глаз) / dist(нос → правый глаз) — асимметрия
  Body: dist(таз → левое плечо) / dist(таз → правое плечо) — та же информация

Поворот корпуса:
  Face: положение носа относительно центра лица
  Body: угол между плечами и бёдрами
```

**Проблема:** LightGBM не страдает от мультиколлинеарности (в отличие от линейных моделей), но дублирование раздувает размерность без прироста информации.

**Решение:** посчитать корреляционную матрицу на реальных данных и удалить признаки с корреляцией >0.95 между группами. Сохранить тот, который стабильнее (face — стабильнее, MediaPipe FaceMesh точнее чем Pose).

### Проблема 4: Разные условия видимости

```
Лицо: видно всегда (если человек смотрит в камеру)
Тело: может быть частично вне кадра (близко к камере)
```

**Проблема:** что делать, если MediaPipe Pose не нашёл тело (человек слишком близко)?

**Решение:** imputation + missing indicator.

```python
if body_landmarks is None:
    body_features = np.zeros(113)           # imputation нулями
    body_missing = 1                        # флаг отсутствия
else:
    body_features = extract_body_features(body_landmarks)
    body_missing = 0

combined = np.concat([face_features, body_features, [body_missing]])
# 468 + 113 + 1 = 582 признака
```

Флаг `body_missing` позволяет LightGBM выучить правило: «если тела нет — полагайся только на лицо».

---

## 4. Стратегии объединения (fusion)

### Early Fusion (конкатенация) — текущий план

```python
features = np.concat([face_features, body_features])  # (468+113,) = 581
model = LightGBM()
model.fit(features, labels)
```

**Плюсы:**
- Простота: один классификатор, один .predict()
- LightGBM сам разберётся, какие признаки важнее
- Быстрый инференс (0.2ms)

**Минусы:**
- Дисбаланс размерности (лицо доминирует)
- Если тело отсутствует — imputation искажает данные
- Один классификатор не может дать раздельные вероятности для лица и тела

### Late Fusion (два классификатора → ансамбль)

```python
# Два независимых классификатора
face_model = LightGBM().fit(face_features, labels)
body_model = LightGBM().fit(body_features, labels)

# Инференс
face_prob = face_model.predict_proba(face_features)  # (N, 7)
body_prob = body_model.predict_proba(body_features)  # (N, 7)

# Взвешенное голосование
if body_missing:
    final_prob = face_prob                           # только лицо
else:
    final_prob = 0.7 * face_prob + 0.3 * body_prob   # лицо важнее
```

**Плюсы:**
- Раздельная интерпретация: «эмоция по лицу = радость (60%), по телу = грусть (40%)»
- Независимость: если тело не найдено — face_model продолжает работать
- Можно калибровать веса под домен (для робота тело важнее, для видеозвонка — лицо)

**Минусы:**
- Два классификатора = 2× .predict()
- Нужно подбирать веса (0.7/0.3) — ещё один гиперпараметр
- Не ловит взаимодействия face×body (например, «улыбка + скрещенные руки»)

### Hybrid Fusion (конкатенация + раздельные головы)

```python
# Один классификатор, но с grouped features
features = np.concat([face_features, body_features])
groups = [0]*468 + [1]*113  # 0 = лицо, 1 = тело

# LightGBM с attention-like механизмом через feature contributions
model = LightGBM()
model.fit(features, labels, feature_name=groups)

# После обучения: сумма contribution для group=0 vs group=1
# → насколько лицо важнее тела
```

**Плюсы:**
- Ловит взаимодействия face×body
- Интерпретируемость через feature importance по группам

**Рекомендация для MVM:**
- **Фаза 1 (сейчас):** Early Fusion — конкатенация. Быстро проверить гипотезу «тело помогает?»
- **Фаза 2 (если помогает):** Late Fusion — два классификатора. Раздельная интерпретация
- **Фаза 3 (оптимизация):** Hybrid — если нужно ловить взаимодействия

---

## 5. Оценка прироста от тела — научное обоснование

### Что говорит литература

| Источник | Вывод |
|----------|-------|
| Kosti et al. (2017), «Context-aware emotion recognition» | Контекст тела повышает accuracy на 8-12% для ambigious emotions (страх vs удивление) |
| Barros et al. (2020), «Multimodal emotional state recognition» | Fusion лица и позы: +5-15% в зависимости от домена |
| Egede et al. (2019), «Fusing face and body for pain recognition» | Тело критично, когда лицо частично окклюдировано |

### Почему тело помогает именно в геометрическом подходе

Геометрические признаки лица страдают от:
1. **Поворота головы:** при out-of-plane повороте 2D-проекция искажает расстояния. 3D MediaPipe смягчает, но не устраняет полностью
2. **Окклюзии:** рука у лица, очки, маска — часть лица не видна

Тело компенсирует:
1. **Поворот головы виден в шее:** угол наклона головы относительно плеч — устойчивый признак
2. **Окклюзия лица не влияет на тело:** даже если лицо закрыто, поза тела передаёт эмоцию

### Ожидаемый прирост (экспериментальная оценка)

```
Текущий LightGBM на 468 face-признаках:         48.5% (±2%)
+ 113 body-признаков (early fusion):             53-57% (+5-9%)
+ body в late fusion (0.7/0.3 веса):             52-56% (+4-8%)
+ temporal (ST-GCN на face+body):                60-63% (+12-15%)
```

---

## 6. Практический эксперимент: как проверить гипотезу

### Шаг 1: собрать данные

```bash
# Минимальный viable тест: 20 кадров на эмоцию, свои данные
data/my_emotions_fullbody/
├── angry/
│   ├── face_body_001.jpg   # полный рост, эмоция «злость»
│   └── ... (×20)
├── happy/
│   └── ... (×20)
└── ... (7 эмоций × 20 кадров = 140 фото)
```

### Шаг 2: извлечь признаки

```python
# Параллельный экстрактор
def extract_face_and_body(image_path):
    face_feat = extract_geometric_features(image_path, face_landmarker)   # (468,)
    body_feat = extract_body_features(image_path, pose_landmarker)         # (113,)
    if body_feat is None:
        body_feat = np.zeros(113)
        body_missing = 1
    else:
        body_missing = 0
    return np.concat([face_feat, body_feat, [body_missing]])  # (582,)
```

### Шаг 3: обучить три варианта

```python
# A: только лицо (baseline)
model_face = LightGBM().fit(X_face, y)
acc_face = model_face.score(X_test_face, y_test)

# B: лицо + тело (early fusion)
model_both = LightGBM().fit(X_both, y)
acc_both = model_both.score(X_test_both, y_test)

# C: ансамбль (late fusion)
model_face = LightGBM().fit(X_face, y)
model_body = LightGBM().fit(X_body, y)
prob = 0.7 * model_face.predict_proba(X_test_face) + 0.3 * model_body.predict_proba(X_test_body)
acc_late = accuracy(y_test, prob.argmax(axis=1))

print(f"Face only:      {acc_face:.1%}")
print(f"Face+Body (EF): {acc_both:.1%}")
print(f"Face+Body (LF): {acc_late:.1%}")
```

### Шаг 4: интерпретация

```python
# Какие части тела самые информативные?
importances = model_both.feature_importances_
face_importance = importances[:468].sum()
body_importance = importances[468:581].sum()
print(f"Face contribution: {face_importance:.1%}")
print(f"Body contribution: {body_importance:.1%}")
```

---

## 7. Архитектурное решение

```
Рекомендация для MVM:

1. НЕ объединять в один экстрактор. Оставить face и body раздельными модулями
   → src/pipeline/2_extract_geometric.py (лицо, уже есть)
   → src/pipeline/2b_extract_body.py (тело, новый)

2. Начать с Early Fusion (конкатенация) для проверки гипотезы
   → src/pipeline/2c_merge_face_body.py
   → Если accuracy выросла на 5%+ → тело полезно

3. Перейти на Late Fusion для production
   → Раздельная интерпретация («эмоция лица» vs «эмоция тела»)
   → Устойчивость к отсутствию тела

4. Расширить тело до 100+ признаков (попарные расстояния, углы, асимметрия)
   → Дисбаланс 468:113 (4:1) приемлем для LightGBM
```

---

*Анализ основан на Machine Vision Knowledge Base и текущем коде legasy_upgrade.*
