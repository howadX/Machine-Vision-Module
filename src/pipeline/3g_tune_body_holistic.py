"""
3g_tune_body_holistic.py — тюнинг Модели Б с расширенными признаками.
"""
import os, csv, pickle, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# ── 1. Загрузка ──────────────────────────────────
csv_path = os.path.join(ROOT, "models", "descriptors", "body_posture_holistic_33.csv")
X_raw, y_raw = [], []
with open(csv_path, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        y_raw.append(int(row[0]))
        X_raw.append([float(v) for v in row[1:]])

X_raw = np.array(X_raw, dtype=np.float64)  # (243, 33)
y_raw = np.array(y_raw, dtype=np.int32)

# ── 2. Расширение признаков ─────────────────────
# Из 33 расстояний → добавляем углы между ключевыми суставами
def expand_features(X_33):
    """X: (N, 33) расстояний. Добавляем углы и попарные отношения."""
    N = len(X_33)
    extra = []
    for i in range(N):
        d = X_33[i]  # 33 расстояния
        feats = []
        # Угол плечо-бедро (наклон туловища): d[11]/d[23], d[12]/d[24]
        feats.append(d[11] / max(d[23], 1e-6))  # левое плечо/бедро
        feats.append(d[12] / max(d[24], 1e-6))  # правое
        # Соотношение верх/низ: (d[11]+d[12])/(d[23]+d[24])
        feats.append((d[11]+d[12]) / max(d[23]+d[24], 1e-6))
        # Колени: d[25]/d[26] (асимметрия ног)
        feats.append(d[25] / max(d[26], 1e-6))
        # Стопы: d[27]/d[28]
        feats.append(d[27] / max(d[28], 1e-6))
        # Руки: d[13]/d[14] (локти), d[15]/d[16] (запястья)
        feats.append(d[13] / max(d[14], 1e-6))
        feats.append(d[15] / max(d[16], 1e-6))
        # Высота головы: d[0] (нос) относительно таза
        feats.append(d[0] / max((d[23]+d[24])/2, 1e-6))
        extra.append(feats)
    return np.column_stack([X_33, np.array(extra)])  # (N, 33+8=41)

X = expand_features(X_raw)
print(f"Признаков: было 33, стало {X.shape[1]} (добавлено 8 relational features)")
print(f"Распределение: standing={np.sum(y_raw==0)}, sitting={np.sum(y_raw==1)}, lying={np.sum(y_raw==2)}")

# ── 3. Тюнинг ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

# Grid search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, 10],
    'num_leaves': [31, 63],
    'learning_rate': [0.05, 0.1],
    'min_child_samples': [5, 10],
}

base = lgb.LGBMClassifier(
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

print("\nGrid Search (2×3×2×2×2 = 48 комбинаций)...")
grid = GridSearchCV(base, param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1)
grid.fit(X_train, y_train)

print(f"\nЛучшие параметры: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.2%}")

# ── 4. Финальная модель ──────────────────────────
best = grid.best_estimator_
y_pred = best.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.2%}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("         standing  sitting  lying")
for i, name in enumerate(["standing", "sitting", "lying"]):
    print(f"{name:9s} {cm[i][0]:5d}    {cm[i][1]:5d}   {cm[i][2]:5d}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["standing","sitting","lying"]))

# ── 5. Сохранение ────────────────────────────────
model_path = os.path.join(ROOT, "models", "emotion", "lightgbm_body_holistic_33.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best, f)
print(f"\nМодель сохранена: {model_path}")
print(f"Признаков: {X.shape[1]} (33 расстояния + 8 relational)")
