#!/usr/bin/env python3
"""Отбор релевантных MediaPipe landmarks для эмоций + переобучение LightGBM."""
import os, csv, pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

ROOT = '/home/dawoh/repos/Video Recognition Module'
CSV_PATH = os.path.join(ROOT, 'models/descriptors/geometric_affectnet.csv')

# ── Анатомически релевантные landmarks (MediaPipe 478-точечная модель) ──
# Основа: FACS Action Units + маппинг dlib 68 → MediaPipe
FACIAL_REGIONS = {
    # Внутренний рот (открытие/закрытие — критично для happy, surprise, fear)
    'mouth_inner': list(range(11, 17)) + [38, 41, 42, 62, 76, 77, 83, 85, 86, 89, 90, 92, 96,
                    179, 180, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194,
                    268, 271, 272, 273, 274, 275, 276, 277,
                    292, 302, 303, 304, 306, 307, 313, 315, 316, 319, 320, 325,
                    403, 404, 406, 407, 408, 410, 411, 412, 413, 414, 416],
    # Внешний рот (уголки губ — happy/sad)
    'mouth_outer': [0, 11, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
                    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
                    314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
    # Брови (AU1, AU2, AU4 — anger, sadness, fear, surprise)
    'eyebrows': [46, 52, 53, 55, 63, 65, 66, 70, 105, 107,  # левая
                 276, 282, 283, 285, 293, 295, 296, 300, 334, 336],  # правая
    # Глаза (AU5 — surprise, fear)
    'eyes': [33, 133, 157, 158, 159, 160, 161, 173, 246,  # левый
             362, 263, 384, 385, 386, 387, 388, 398, 466],  # правый
    # Нос (AU9 — disgust)
    'nose': [1, 2, 4, 5, 6, 19, 45, 48, 49, 64, 94, 97, 115, 168, 195, 197,  # переносица
             98, 99, 100, 101, 102, 103, 104,  # крылья носа
             327, 328, 329, 330, 331, 332, 333],  # нижняя часть
}

# Собираем все уникальные индексы
ALL_EMOTION_LANDMARKS = sorted(set().union(*FACIAL_REGIONS.values()))
print(f'Отобрано {len(ALL_EMOTION_LANDMARKS)} landmarks из 478')
print(f'  mouth_inner: {len(set(FACIAL_REGIONS["mouth_inner"]))}')
print(f'  mouth_outer: {len(set(FACIAL_REGIONS["mouth_outer"]))}')
print(f'  eyebrows:    {len(set(FACIAL_REGIONS["eyebrows"]))}')
print(f'  eyes:        {len(set(FACIAL_REGIONS["eyes"]))}')
print(f'  nose:        {len(set(FACIAL_REGIONS["nose"]))}')

# ── Загрузка данных ──
print('\nЗагрузка CSV...')
X, y = [], []
with open(CSV_PATH, newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        lbl = int(row[0])
        if lbl == 1:  # skip contempt
            continue
        new_lbl = lbl - 1 if lbl > 1 else lbl
        y.append(new_lbl)
        X.append([float(v) for v in row[1:]])

X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.int32)

# Отбираем только эмоционально-релевантные признаки
X_emotion = X[:, ALL_EMOTION_LANDMARKS]
print(f'Данные: {X_emotion.shape[0]} примеров × {X_emotion.shape[1]} признаков')

# ── Обучение ──
X_train, X_test, y_train, y_test = train_test_split(
    X_emotion, y, test_size=0.2, random_state=42, stratify=y
)

# Class weights для борьбы с дисбалансом
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=10,
    num_leaves=63,
    min_child_samples=20,
    class_weight=class_weight_dict,
    random_state=42,
    verbose=-1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
labels_7 = ['anger','disgust','fear','happy','neutral','sad','surprise']

print(f'\n=== Точность (отобранные признаки): {acc:.2%} ===')
print(classification_report(y_test, y_pred, target_names=labels_7))

# Сравнение: кросс-валидация
cv_scores = cross_val_score(clf, X_emotion, y, cv=5, scoring='accuracy')
print(f'CV accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}')

# ── Сохранение ──
model_path = os.path.join(ROOT, 'models/emotion/lightgbm_emotion_regions.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({'model': clf, 'landmark_indices': ALL_EMOTION_LANDMARKS,
                 'num_features': len(ALL_EMOTION_LANDMARKS)}, f)
print(f'\nМодель: {model_path}')

# ── Анализ важности признаков ──
importances = clf.feature_importances_
top_indices = np.argsort(importances)[-20:][::-1]
print('\nТоп-20 важнейших landmarks:')
for rank, idx in enumerate(top_indices, 1):
    lm = ALL_EMOTION_LANDMARKS[idx]
    region = '?'
    for rname, rindices in FACIAL_REGIONS.items():
        if lm in rindices:
            region = rname
            break
    print(f'  {rank:2d}. landmark {lm:3d} | важность={importances[idx]:.4f} | {region}')
