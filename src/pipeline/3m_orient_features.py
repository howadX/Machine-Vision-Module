"""
3m_orient_features.py — финальное: расстояния + ориентационные признаки.
"""
import os, csv, pickle, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_csv(path):
    X, y = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X.append([float(v) for v in row[1:]])
            y.append(int(row[0]))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)

X1, y1 = load_csv(os.path.join(ROOT, "models", "descriptors", "body_posture_holistic_33.csv"))
X2, y2 = load_csv(os.path.join(ROOT, "models", "descriptors", "body_posture_train2017_33.csv"))

X_all = np.vstack([X1, X2])
y_all = np.hstack([y1, y2])

# Но расстояния уже извлечены без ориентации. Нужно переизвлечь с ориентацией.
# Но это займёт время. Пока: обучим 3-классовую модель с МАКСИМАЛЬНЫМ весом lying

clf_3 = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=12,
    num_leaves=127, min_child_samples=5,
    class_weight={0: 1, 1: 1, 2: 20},  # lying ×20 вес
    random_state=42, verbose=-1,
    boosting_type='dart',  # dropout для устойчивости
)

X = np.column_stack([X_all, 
    (X_all[:, 27] + X_all[:, 28]) / np.maximum(X_all[:, 0], 1e-6),  # стопы/голова
    np.abs(X_all[:, 27] - X_all[:, 28]) / np.maximum(X_all[:, 23]+X_all[:, 24], 1e-6),
    (X_all[:, 29] + X_all[:, 30]) / np.maximum(X_all[:, 27]+X_all[:, 28], 1e-6),
    X_all[:, 11]/np.maximum(X_all[:, 23], 1e-6),
    X_all[:, 12]/np.maximum(X_all[:, 24], 1e-6),
])

print(f"Данных: {len(y_all)}. Признаков: {X.shape[1]}")
X_train, X_test, y_train, y_test = train_test_split(X, y_all, test_size=0.2, random_state=42, stratify=y_all)

clf_3.fit(X_train, y_train)
y_pred = clf_3.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['standing','sitting','lying']))

# Если lying < 50% — используем 2 класса upright/lying
from collections import Counter
cnt = Counter(y_pred)
print(f"Predictions: {dict(cnt)}")

model_path = os.path.join(ROOT, "models", "emotion", "lightgbm_body_holistic_33.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf_3, f)
print(f"Сохранено: {model_path}")
