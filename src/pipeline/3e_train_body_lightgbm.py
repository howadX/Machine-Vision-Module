"""
3e_train_body_lightgbm.py — обучение Модели Б (позы тела) на COCO.

Вход: models/descriptors/body_posture_coco.csv
Выход: models/emotion/lightgbm_body_posture.pkl
Классы: 0=standing, 1=sitting, 2=lying
"""

import os, csv, pickle, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

csv_path = os.path.join(ROOT, "models", "descriptors", "body_posture_coco.csv")
print(f"Загрузка: {csv_path}")

X_rows, y_rows = [], []
with open(csv_path, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        y_rows.append(int(row[0]))
        X_rows.append([float(v) for v in row[1:]])

X = np.array(X_rows, dtype=np.float64)
y = np.array(y_rows, dtype=np.int32)
print(f"Примеров: {len(X)}, признаков: {X.shape[1]}, классов: {len(np.unique(y))}")
print(f"Распределение: {np.bincount(y)}")

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# LightGBM
clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=-1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nТочность: {acc:.2%}")

POSTURE_NAMES = ["standing", "sitting", "lying"]
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("         standing  sitting  lying")
for i, name in enumerate(POSTURE_NAMES):
    print(f"{name:9s} {cm[i][0]:5d}    {cm[i][1]:5d}   {cm[i][2]:5d}")

print("\nFeature importances:")
importances = clf.feature_importances_
KP_NAMES = ["nose","L_eye","R_eye","L_ear","R_ear","L_shoulder","R_shoulder",
            "L_elbow","R_elbow","L_wrist","R_wrist","L_hip","R_hip",
            "L_knee","R_knee","L_ankle","R_ankle"]
for i, (name, imp) in enumerate(zip(KP_NAMES, importances)):
    bar = "█" * int(imp * 100)
    print(f"  {name:12s}: {imp:.4f} {bar}")

# Сохранение
model_dir = os.path.join(ROOT, "models", "emotion")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "lightgbm_body_posture.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"\nМодель сохранена: {model_path}")
