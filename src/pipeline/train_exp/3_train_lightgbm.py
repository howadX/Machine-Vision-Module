"""
3_train_lightgbm.py — обучение LightGBM на геометрических признаках (MediaPipe 468 точек).

Вход: models/descriptors/geometric_468.csv
Выход: models/emotion/lightgbm_geometric.pkl

Классы: 7 эмоций.
Параметры LightGBM: n_estimators=100, learning_rate=0.1, max_depth=6.
"""
import os as _os
import csv
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb

# ── Пути ──────────────────────────────────────────────
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_SCRIPT_DIR))

dis_path = _os.path.join(_ROOT, 'models', 'descriptors')
dat_path = _os.path.join(_ROOT, 'models', 'emotion')

# ── 1. Загрузка CSV (через csv.reader — pandas глючит) ─
csv_path = _os.path.join(dis_path, 'geometric_468.csv')
print(f'Загрузка: {csv_path}')

X_rows, y_rows = [], []
with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        y_rows.append(int(row[0]))
        X_rows.append([float(v) for v in row[1:]])

X = np.array(X_rows, dtype=np.float64)
y = np.array(y_rows, dtype=np.int32)

n_samples, n_features = X.shape
n_classes = len(np.unique(y))
print(f'Загружено: {n_samples} примеров, {n_features} признаков, {n_classes} классов')
print(f'Метки классов: {sorted(np.unique(y))}')
print(f'Распределение: {np.bincount(y)}')

# ── 2. Разделение train/test 80/20 ────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train: {len(X_train)}, Test: {len(X_test)}')

# ── 3. Обучение LightGBM ──────────────────────────────
clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbose=-1,
)
clf.fit(X_train, y_train)

# ── 4. Оценка ─────────────────────────────────────────
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'\nТочность LightGBM: {acc:.2%}')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# ── 5. Сохранение модели ──────────────────────────────
_os.makedirs(dat_path, exist_ok=True)
model_path = _os.path.join(dat_path, 'lightgbm_geometric.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f'\nМодель сохранена: {model_path}')
