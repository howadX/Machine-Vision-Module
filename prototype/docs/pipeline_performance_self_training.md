# Pipeline v2: производительность, масштабируемость, самообучение

**Дата:** 2026-05-29
**Анализируемый код:** `src/pipeline/4_identification.py` (текущий рантайм)

---

## 1. Профилирование: latency текущего пайплайна

### Фаза 1 — идентификация (per-frame)

```
Кадр (640×480, 30 fps = 33ms бюджет)
│
├── SCRFD detect (каждый кадр)               ~3-5ms   ← InsightFace ONNX
├── ArcFace embedding (каждый 3-й кадр)      ~8-12ms  ← ResNet-100, 512d
├── MediaPipe FaceLandmarker (каждый кадр)    ~3-8ms   ← 478 точек, CPU/GPU
└── Отрисовка (landmarks + bbox)              ~1-2ms
────────────────────────────────────────────────────────
Total (каждый кадр):         ~7-15ms  → 66-140 FPS ✅
Total (каждый 3-й, +ArcFace): ~18-27ms → 37-55 FPS ✅
```

**Бутылочное горлышко:** ArcFace эмбеддинг (8-12ms). Но он нужен только раз в 3 кадра → терпимо.

### Фаза 2 — эмоции (per-frame)

```
Кадр (640×480)
│
├── SCRFD detect                               ~3-5ms
├── MediaPipe FaceLandmarker                    ~3-8ms
├── compute_geometric_features (167/478 точек)  ~0.1ms  ← numpy
├── LightGBM.predict()                          ~0.05ms ← 100 деревьев
├── Отрисовка + cv2.imshow                      ~1-2ms
└── cv2.waitKey(10)                             ~10ms   ← ИСКУССТВЕННАЯ задержка!
────────────────────────────────────────────────────────
Total (без waitKey):  ~7-15ms → 66-140 FPS
Total (с waitKey):    ~17-25ms → 40-58 FPS
```

### Полный цикл эмоций (7 эмоций × 20 кадров = 140 кадров)

```
7 эмоций × (20 кадров + до 5 попыток) = 140-700 кадров
При 40 FPS: 3.5-17.5 секунд на весь цикл
```

**Узкое место:** не модель, а `cv2.waitKey(10)` + окно из 20 кадров + до 5 попыток.

---

## 2. Масштабируемость точности

### Что ограничивает accuracy сейчас

```
Текущая точность: ~48.5% (LightGBM на 167 геометрических признаках)

Факторы, влияющие на accuracy:
┌─────────────────────────────────────────────┐
│ Данные (my_emotions)                         │
│ ├── Размер: ~100 кадров / эмоция            │ ← МАЛО (нужно 200+)  │
│ ├── Разнообразие: один фон, одно освещение   │ ← Domain shift       │
│ ├── Позиции: только фронтальные              │ ← Нет поворотов      │
│ └── Люди: 1-2 человека                       │ ← Person-specific     │
│                                               │                       │
│ Признаки (геометрические)                     │                       │
│ ├── 167 отобранных регионов лица             │ ← 35% от 478 точек   │
│ ├── Только расстояния от носа                │ ← Нет углов, кривизны │
│ ├── Нет текстуры (морщины, напряжение)       │ ← Фундаментально      │
│ └── Нет temporal контекста                   │ ← Статика             │
│                                               │                       │
│ Классификатор (LightGBM)                      │                       │
│ ├── 100 деревьев, max_depth=6                │ ← Недообучен?         │
│ └── 7 классов с дисбалансом                  │ ← Neutral доминирует  │
└─────────────────────────────────────────────┘
```

### Кривая масштабирования точности

```
Точность (%)
   70 ┤                                    ╱─ CNN + face (потолок)
      │                              ╱
   60 ┤                        ╱─── геометрия лица + тела + temporal
      │                  ╱
   55 ┤            ╱───── геометрия лица + тела
      │      ╱
   50 ┤─╱──────── геометрия лица (167 признаков)  ← сейчас
      │╱
   45 ┤ геометрия лица (все 478 признаков)
      │
   40 ┤ SVM на identity-эмбеддингах (легаси v1)
      │
   35 ┤
      └─────┬─────┬─────┬─────┬─────┬─────→ Размер датасета
           100   200   500  1000  5000  10000
```

**Ключевой вывод:** на 100-200 примерах геометрические признаки уже упираются в потолок ~50%. Дальнейший рост — через новые признаки (тело, temporal) и больше данных.

---

## 3. Самообучение (Self-Training) — архитектура

### Концепция

Вместо того чтобы снимать 200 кадров каждой эмоции ВРУЧНУЮ, модель сама размечает новые кадры, а ты только подтверждаешь.

```
Цикл самообучения:
──────────────────────────────────────────
1. Начальная модель M0 обучена на my_emotions (48.5%)
2. M0 работает в рантайме → видит новые кадры
3. Для кадров с confidence > 0.7 → авторазметка
4. Для кадров с confidence 0.4-0.7 → показываем человеку
5. Новые размеченные кадры → датасет D1
6. Переобучение: M1 на D0 + D1
7. Повтор с шага 2
──────────────────────────────────────────
```

### Почему это работает для геометрического подхода

Геометрические признаки имеют важное свойство: **расстояния от носа стабильны при одинаковой эмоции на одном человеке.** Если Вадим показывает «радость» 100 раз, геометрические признаки кластеризуются вокруг одной точки в 167-мерном пространстве.

Это значит:
- **Высокая confidence (>0.7):** модель видит знакомый паттерн → авторазметка с низким риском ошибки
- **Средняя confidence (0.4-0.7):** модель не уверена (новый ракурс? другая интенсивность?) → показываем человеку
- **Низкая confidence (<0.4):** модель не понимает → игнорируем или показываем как «unknown»

### Три стратегии самообучения

#### Стратегия A: Pseudo-Labeling (авторазметка по порогу)

```python
def self_train_pseudo_labeling(model, unlabeled_frames, threshold=0.7):
    """Модель размечает уверенные кадры сама."""
    new_dataset = []
    for frame in unlabeled_frames:
        features = extract_geometric(frame)
        proba = model.predict_proba(features)[0]  # (7,)
        max_prob = proba.max()
        if max_prob > threshold:
            emotion = proba.argmax()
            new_dataset.append((features, emotion))
    return new_dataset
```

**Плюсы:** полностью автоматически
**Минусы:** confirmation bias — модель укрепляет свои ошибки

**Mitigation:** threshold=0.7 — только уверенные. И проверка человеком случайной подвыборки.

#### Стратегия B: Active Learning (показываем неуверенные)

```python
def active_learning_query(model, unlabeled_frames, n_queries=10):
    """Выбираем n самых неуверенных кадров для разметки человеком."""
    uncertainties = []
    for frame in unlabeled_frames:
        features = extract_geometric(frame)
        proba = model.predict_proba(features)[0]
        # Uncertainty = 1 - max_prob (чем меньше уверенность, тем интереснее)
        # ИЛИ entropy: -sum(p * log(p))
        entropy = -np.sum(proba * np.log(proba + 1e-10))
        uncertainties.append(entropy)
    
    # Топ-N самых uncertain
    top_indices = np.argsort(uncertainties)[-n_queries:]
    return [unlabeled_frames[i] for i in top_indices]
```

**Плюсы:** человек размечает только САМЫЕ информативные кадры
**Минусы:** требует участия человека (но сильно меньше, чем ручная разметка всего)

#### Стратегия C: Co-Training (два классификатора учат друг друга)

```python
# Два взгляда на одну эмоцию:
model_face = LightGBM()      # классификатор по геометрии лица
model_body = LightGBM()      # классификатор по геометрии тела

for frame in unlabeled_frames:
    face_feat = extract_face(frame)   # 478 чисел
    body_feat = extract_body(frame)   # 113 чисел
    
    # Если ОБА уверены и согласны → авторазметка
    face_prob = model_face.predict_proba(face_feat)
    body_prob = model_body.predict_proba(body_feat)
    
    if (face_prob.max() > 0.7 and body_prob.max() > 0.7 
        and face_prob.argmax() == body_prob.argmax()):
        emotion = face_prob.argmax()
        # Добавляем в датасет ОБОИХ классификаторов
        new_face_data.append((face_feat, emotion))
        new_body_data.append((body_feat, emotion))
```

**Плюсы:** два классификатора проверяют друг друга — риск confirmation bias минимален
**Минусы:** требует body-экстрактор (ещё не реализован)

**Это ИДЕАЛЬНАЯ стратегия для MVM** — лицо и тело дают независимые сигналы одной эмоции.

---

## 4. Интеграция самообучения в пайплайн

### Новый модуль: `src/pipeline/5_self_train.py`

```python
"""
5_self_train.py — самообучающийся цикл MVM.

Режимы:
  --mode collect   : сбор неразмеченных кадров в фоне
  --mode label     : active learning — показать неуверенные кадры
  --mode retrain   : переобучить модель на расширенном датасете
  --mode auto      : полный цикл (collect → label → retrain)
"""

class SelfTrainingLoop:
    def __init__(self, model, threshold=0.7):
        self.model = model
        self.threshold = threshold
        self.auto_labeled = []   # (features, emotion, confidence)
        self.human_labeled = []  # (features, emotion)
    
    def collect(self, camera, duration_seconds=60):
        """Собирает кадры в фоне, авторазмечает уверенные."""
        unlabeled = []
        for frame in camera.stream(duration_seconds):
            features = extract_geometric(frame)
            proba = self.model.predict_proba(features)
            
            if proba.max() > self.threshold:
                self.auto_labeled.append((features, proba.argmax(), proba.max()))
            elif proba.max() > 0.3:  # неуверенные, но не безнадёжные
                unlabeled.append((frame, features, proba))
        
        return unlabeled  # для active learning
    
    def query_human(self, unlabeled, n=10):
        """Показывает топ-N неуверенных кадров человеку."""
        # Сортировка по entropy
        unlabeled.sort(key=lambda x: -np.sum(x[2] * np.log(x[2] + 1e-10)))
        for frame, features, proba in unlabeled[:n]:
            emotion = show_and_ask(frame, proba)  # GUI: человек кликает
            self.human_labeled.append((features, emotion))
    
    def retrain(self):
        """Переобучает модель на старых + новых данных."""
        X = np.array([f for f, _ in self.auto_labeled + self.human_labeled])
        y = np.array([e for _, e in self.auto_labeled + self.human_labeled])
        self.model.fit(X, y)
        self.model.save("models/emotion/lightgbm_self_trained.pkl")
```

### Как это выглядит для тебя (пользователя)

```bash
# 1. Запускаешь сбор в фоне (60 секунд)
python src/pipeline/5_self_train.py --mode collect --duration 60
# → Модель смотрит на тебя, авторазмечает уверенные кадры

# 2. Показываешь эмоции перед камерой (просто живёшь своей жизнью)
# → Злость, радость, нейтрально — что угодно

# 3. Active learning: модель показывает 10 самых непонятных кадров
python src/pipeline/5_self_train.py --mode label
# → GUI: кадр + топ-3 предсказания → кликаешь правильный

# 4. Переобучение
python src/pipeline/5_self_train.py --mode retrain
# → accuracy: 48.5% → 52-55% (через 100 новых кадров)
```

---

## 5. Предотвращение деградации (catastrophic forgetting)

### Проблема

Если модель обучается ТОЛЬКО на новых авторазмеченных данных, она «забывает» старые паттерны.

### Решение: Replay Buffer

```python
class ReplayBuffer:
    """Хранит старые примеры и подмешивает их при переобучении."""
    def __init__(self, max_size=1000):
        self.buffer = []  # (features, emotion)
        self.max_size = max_size
    
    def add(self, features, emotion):
        self.buffer.append((features, emotion))
        if len(self.buffer) > self.max_size:
            # Удаляем случайный (reservoir sampling)
            idx = random.randint(0, len(self.buffer) - 1)
            self.buffer.pop(idx)
    
    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))

# При переобучении:
old_samples = replay_buffer.sample(n=50)  # 50 старых примеров
new_samples = recently_labeled             # новые примеры
X = np.concat([old_X, new_X])
y = np.concat([old_y, new_y])
model.fit(X, y)
```

### Другой подход: ансамбль моделей

```python
# Вместо перезаписи модели — сохраняем старую и усредняем
model_v1 = load("lightgbm_v1.pkl")    # старая
model_v2 = LightGBM().fit(new_data)   # новая

def predict_proba(features):
    return 0.3 * model_v1.predict_proba(features) + \
           0.7 * model_v2.predict_proba(features)
```

---

## 6. Дорожная карта самообучения

```
Неделя 1: Pseudo-Labeling (авторазметка)
  ├── Добавить confidence-порог в 4_identification.py
  ├── Сохранять кадры с confidence > 0.7 → data/auto_labeled/
  └── Ручная валидация случайных 10% авторазмеченных

Неделя 2: Active Learning
  ├── Uncertainty sampling: entropy-based
  ├── GUI: показать кадр + топ-3 → клик
  └── Интеграция с 5_self_train.py

Неделя 3: Co-Training (после реализации body-экстрактора)
  ├── Face model + Body model → взаимная проверка
  └── Авторазметка только при согласии обоих

Неделя 4: Непрерывное обучение
  ├── Replay Buffer → нет забывания
  └── Мониторинг accuracy на валидационном сете
```

---

## 7. Практический порог для старта

```python
# Минимальный viable self-training — 30 строк:
THRESHOLD = 0.70
SAVE_DIR = "data/auto_labeled"

for frame in camera_stream:
    features = extract_geometric(frame)
    proba = model.predict_proba(features)[0]
    
    if proba.max() > THRESHOLD:
        emotion_idx = proba.argmax()
        emotion_name = EMOTIONS[emotion_idx]
        save_path = f"{SAVE_DIR}/{emotion_name}/{timestamp}.jpg"
        cv2.imwrite(save_path, frame)
        print(f"Авторазметка: {emotion_name} ({proba.max():.0%})")
```

**Это уже можно добавить в `4_identification.py` прямо сейчас.** Модель будет собирать данные, пока ты просто находишься перед камерой.

---

*Анализ основан на коде `src/pipeline/4_identification.py` и Machine Vision Knowledge Base.*
