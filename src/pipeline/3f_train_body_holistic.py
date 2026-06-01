"""
3f_train_body_holistic.py — обучение Модели Б на 33 Holistic-признаках.
"""
import os, csv, pickle, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

csv_path = os.path.join(ROOT, "models", "descriptors", "body_posture_holistic_33.csv")
print(f"Загрузка: {csv_path}")

X_rows, y_rows = [], []
with open(csv_path, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        y_rows.append(int(row[0]))
        X_rows.append([float(v) for v in row[1:]])

X = np.array(X_rows, dtype=np.float64)
y = np.array(y_rows, dtype=np.int32)
print(f"Примеров: {len(X)}, признаков: {X.shape[1]}")
print(f"Распределение: standing={np.sum(y==0)}, sitting={np.sum(y==1)}, lying={np.sum(y==2)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=5,
    class_weight="balanced", random_state=42, verbose=-1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nТочность: {acc:.2%}")
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("         standing  sitting  lying")
for i, name in enumerate(["standing", "sitting", "lying"]):
    print(f"{name:9s} {cm[i][0]:5d}    {cm[i][1]:5d}   {cm[i][2]:5d}")

model_path = os.path.join(ROOT, "models", "emotion", "lightgbm_body_holistic_33.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"\nСохранено: {model_path}")
