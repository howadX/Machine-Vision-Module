# MVM Holistic Pipeline: единый экстрактор лица + тела + рук

**Дата:** 2026-05-29
**Репозиторий:** `~/repos/Machine Vision Module`, ветка `legasy_upgrade`

---

## 1. Проблема: раздельные экстракторы не связаны

### Текущий подход (v2)

```
Кадр
├── SCRFD → ArcFace 512d → Face ID («Вадим»)
├── MediaPipe FaceLandmarker (478 точек) → геометрия лица → LightGBM → эмоция
└── [пока нет] MediaPipe PoseLandmarker (33 точки) → геометрия тела → ?
```

**Проблема при добавлении тела:** если в кадре два человека, face-ландмарки принадлежат одному, body-ландмарки — другому. Без spatial matching модель обучается на несвязанных признаках → accuracy падает вместо роста.

### Решение: MediaPipe Holistic

```
Кадр → MediaPipe Holistic (~15ms)
├── face_landmarks (478)  ─┐
├── pose_landmarks (33)   ─┤ Все landmarks ОДНОГО человека
├── left_hand (21)        ─┤ Гарантированная связь
└── right_hand (21)       ─┘
```

**Holistic** — это НЕ сумма Face + Pose + Hands. Это **единая модель**, которая трекает все landmarks одновременно и согласованно для одного человека. Face и body связаны на уровне модели, а не постфактум.

---

## 2. Архитектура MediaPipe Holistic

### Внутреннее устройство

```
Кадр (RGB, любое разрешение)
    │
    ▼
BlazePose (детектор тела)
    │
    ├── pose_landmarks (33 × x,y,z,visibility)
    │     └── ROI для лица: расширенный bbox вокруг головы
    │           │
    │           ▼
    │     FaceMesh (468/478 точек лица)
    │           │
    │           └── ROI для рук: bbox вокруг запястий
    │                 │
    │                 ▼
    │           HandMesh (21 × 2 точки рук)
    │
    └── Выход: все landmarks в единой системе координат
```

### Ключевые свойства

| Свойство | Значение |
|----------|----------|
| **Связь компонентов** | Гарантирована моделью (pose → face ROI, pose → hands ROI) |
| **Система координат** | Единая (пиксели изображения) |
| **Число лиц** | 1 (ограничение Holistic) |
| **Latency (GPU)** | ~10-15ms |
| **Latency (CPU)** | ~20-40ms |
| **Модель** | ~15 MB (TFLite) |

### Режимы работы

```python
import mediapipe as mp

# Режим 1: статическое изображение (точнее, медленнее)
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,            # 0=легче, 1=средне, 2=точно
    refine_face_landmarks=True,    # 478 точек лица (а не 468)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Режим 2: видео (быстрее, temporal smoothing)
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,            # баланс точность/скорость
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

---

## 3. Признаки: что извлекаем

### Полный вектор: 553 признака на человека

```python
def extract_holistic_features(results) -> np.ndarray:
    """
    Извлекает единый геометрический вектор из Holistic-результата.
    
    Возвращает:
        np.ndarray формы (553,) — все признаки одного человека
    """
    features = []
    
    # ── 1. ЛИЦО: 478 расстояний от кончика носа ──
    if results.face_landmarks:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        nose = pts[4]                               # landmark 4 = кончик носа
        eye_left = pts[33]                          # внешний угол левого глаза
        eye_right = pts[263]                        # внешний угол правого глаза
        interocular = np.linalg.norm(eye_left - eye_right)
        if interocular < 1e-8: interocular = 1.0
        face_dist = np.linalg.norm(pts - nose, axis=1) / interocular  # (478,)
    else:
        face_dist = np.zeros(478)
    features.extend(face_dist)
    
    # ── 2. ТЕЛО: 33 расстояния от центра таза ──
    if results.pose_landmarks:
        pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                        for lm in results.pose_landmarks.landmark])
        hip_center = (pts[23] + pts[24]) / 2        # центр таза
        shoulder_w = np.linalg.norm(pts[11] - pts[12])  # межплечевое
        if shoulder_w < 1e-8: shoulder_w = 1.0
        body_dist = np.linalg.norm(pts[:, :3] - hip_center[:3], axis=1) / shoulder_w  # (33,)
    else:
        body_dist = np.zeros(33)
    features.extend(body_dist)
    
    # ── 3. РУКИ: 42 расстояния от запястья ──
    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            wrist = pts[0]                          # landmark 0 = запястье
            # Нормировка на длину ладони (запястье → средний палец)
            palm = np.linalg.norm(pts[9] - wrist)   # landmark 9 = основание среднего пальца
            if palm < 1e-8: palm = 1.0
            hand_dist = np.linalg.norm(pts - wrist, axis=1) / palm  # (21,)
        else:
            hand_dist = np.zeros(21)
        features.extend(hand_dist)
    
    return np.array(features)  # (478 + 33 + 21 + 21) = 553
```

### Сводка признаков

| Источник | Точек | Референс | Нормализатор | Признаков |
|----------|-------|----------|-------------|-----------|
| Face (FaceMesh) | 478 | Кончик носа (lm 4) | Межзрачковое (lm 33↔263) | 478 |
| Pose (BlazePose) | 33 | Центр таза (lm 23+24) | Межплечевое (lm 11↔12) | 33 |
| Left Hand | 21 | Запястье (lm 0) | Длина ладони (lm 0↔9) | 21 |
| Right Hand | 21 | Запястье (lm 0) | Длина ладони (lm 0↔9) | 21 |
| **Итого** | **553** | — | — | **553** |

---

## 4. Latency: сравнение подходов

### Раздельный подход (FaceLandmarker + PoseLandmarker)

```
SCRFD detect (GPU):        3-5ms
FaceLandmarker (CPU/GPU):  3-8ms
PoseLandmarker (CPU/GPU):  5-10ms
─────────────────────────────────
Total:                    11-23ms
Проблема:                 2 вызова, landmarks не связаны
```

### Holistic (единый вызов)

```
Holistic.process() (GPU):  10-15ms
Holistic.process() (CPU):  20-40ms
─────────────────────────────────
Total:                     10-15ms (GPU)
Преимущество:              все landmarks ОДНОГО человека, temporal smoothing
```

### Сравнение

| Подход | Latency (GPU) | Связь face-body | Multi-person | Точек всего |
|--------|--------------|-----------------|-------------|-------------|
| Раздельный | 11-23ms | ❌ Нужен matching | ⚠️ Сложно | 511 |
| Holistic | 10-15ms | ✅ Автоматически | ❌ 1 человек | 553 |

**Holistic БЫСТРЕЕ на GPU** (один вызов вместо трёх) и **даёт гарантированную связь** face-body-hands.

---

## 5. Интеграция в MVM

### Что меняется в `4_identification.py`

```python
# БЫЛО:
face_landmarker = vision.FaceLandmarker.create_from_options(...)
res = face_landmarker.detect(mp_img)
if res.face_landmarks:
    X = compute_geometric_features(res.face_landmarks[0])  # 478 признаков

# СТАНЕТ:
import mediapipe as mp
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=True,
)
results = holistic.process(frame_rgb)  # ОДИН вызов
X = extract_holistic_features(results)  # 553 признака
```

### Совместимость с LightGBM

```python
# Модель переучивается на 553 признаках:
clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6)
clf.fit(X_train_553, y_train)
clf.save("models/emotion/lightgbm_holistic.pkl")

# Рантайм — без изменений:
proba = clf.predict_proba(X_553)[0]
emotion = EMOTIONS_RU[proba.argmax()]
```

### Как обучить

```bash
# 1. Снять full-body данные (210 кадров):
#    7 эмоций × 30 кадров с камеры (полный рост)
#    data/my_emotions_fullbody/{angry,disgust,...}/*.jpg

# 2. Извлечь 553 признака через Holistic:
python src/pipeline/2d_extract_holistic.py

# 3. Обучить LightGBM:
python src/pipeline/3d_train_lightgbm_holistic.py
# Ожидаемая точность: 55-60% (против 48.5% для только лица)

# 4. Заменить в 4_identification.py:
#    FaceLandmarker → Holistic
#    compute_geometric_features → extract_holistic_features
```

---

## 6. Ожидаемый прирост точности

```
Текущий LightGBM (478 face, геометрия):    48.5% ±2%
+ Holistic body (33 pose landmarks):        53-57%  (+5-9%)
+ Holistic hands (42 hand landmarks):       55-60%  (+7-12%)
+ Temporal (ST-GCN на последовательностях):  60-65%  (+12-17%)
```

**Почему руки дают прирост:**
- Страх: руки у лица / скрещены на груди
- Злость: сжатые кулаки
- Радость: открытые ладони, руки вверх
- Грусть: руки опущены, безвольные кисти

---

## 7. Ограничения Holistic

| Ограничение | Следствие | Решение |
|-------------|-----------|---------|
| **1 человек** | Не видит второго | Для multi-person: SCRFD + отдельный Holistic на каждом лице |
| **Нет Face ID** | Holistic не даёт эмбеддингов | ArcFace остаётся отдельно (как сейчас) |
| **CPU: 20-40ms** | Медленно на CPU | Требуется GPU (уже есть) |
| **TFLite, не ONNX** | Нет TensorRT | Терпимо (15ms на GPU — приемлемо) |

## 8. MVM Pipeline с Holistic

```
Камера
├── SCRFD + ArcFace (Face ID, каждый 6-й кадр, ~8ms GPU)
├── MediaPipe Holistic (face+body+hands, каждый кадр, ~12ms GPU)
│     └── extract_holistic_features() → 553 признака
│           └── LightGBM → эмоция (0.1ms)
└── Оверлей (PIL, 1ms)
────────────────────────────────────────────
Total per frame: ~15ms → 65 FPS (теоретически)
С учётом камеры (24 FPS предел): ~24 FPS (реально)
```

---

*Основан на Machine Vision Knowledge Base и текущем коде legasy_upgrade.*
