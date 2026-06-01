"""
3h_train_body_full.py — модель с foot features (стопы) для улучшения lying.
"""
import os, csv, pickle, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

csv_path = os.path.join(ROOT, "models", "descriptors", "body_posture_holistic_33.csv")
X_raw, y_raw = [], []
with open(csv_path, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        y_raw.append(int(row[0]))
        X_raw.append([float(v) for v in row[1:]])

X_raw = np.array(X_raw, dtype=np.float64)
y_raw = np.array(y_raw, dtype=np.int32)

# ── Расширенные признаки: 33 расстояния + FOOT + relational ──
def expand_full(X33):
    N = len(X33); extra = []
    for i in range(N):
        d = X33[i]  # 33 расстояния от центра таза
        feats = []
        # Relational (8)
        feats.extend([d[11]/max(d[23],1e-6), d[12]/max(d[24],1e-6),
            (d[11]+d[12])/max(d[23]+d[24],1e-6),
            d[25]/max(d[26],1e-6), d[27]/max(d[28],1e-6),
            d[13]/max(d[14],1e-6), d[15]/max(d[16],1e-6),
            d[0]/max((d[23]+d[24])/2,1e-6)])
        # Foot features: стопы vs голова (критично для lying!)
        head_y = d[0]  # расстояние нос → таз
        foot_y = (d[27] + d[28]) / 2  # среднее пяток
        feats.append(foot_y / max(head_y, 1e-6))  # стопы/голова: >1 если стопы дальше от таза чем голова
        feats.append(abs(d[27] - d[28]) / max(d[23]+d[24], 1e-6))  # асимметрия стоп
        feats.append((d[29]+d[30]) / max(d[27]+d[28], 1e-6))  # носки/пятки
        extra.append(feats)
    return np.column_stack([X33, np.array(extra)])  # (N, 33+11=44)

X = expand_full(X_raw)
print(f"Признаков: было 33, стало {X.shape[1]} (+8 relational +3 foot)")
print(f"Распределение: standing={np.sum(y_raw==0)}, sitting={np.sum(y_raw==1)}, lying={np.sum(y_raw==2)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [7, 10],
    'num_leaves': [31, 63],
    'min_child_samples': [10, 20],
    'learning_rate': [0.05, 0.1],
}

base = lgb.LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1)
print(f"\nGrid Search ({np.prod([len(v) for v in param_grid.values()])} комбинаций)...")
grid = GridSearchCV(base, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

print(f"Лучшие: {grid.best_params_}")
print(f"CV accuracy: {grid.best_score_:.2%}")

best = grid.best_estimator_
y_pred = best.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.2%}")

print("\nConfusion:           standing sitting lying")
cm = confusion_matrix(y_test, y_pred)
for i, n in enumerate(['standing','sitting ','lying   ']):
    print(f'{n} {cm[i][0]:5d}    {cm[i][1]:5d}   {cm[i][2]:5d}')
print(f"\n{classification_report(y_test, y_pred, target_names=['standing','sitting','lying'])}")

# Сохранение
model_path = os.path.join(ROOT, "models", "emotion", "lightgbm_body_holistic_33.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best, f)
print(f"Сохранено: {model_path}")
