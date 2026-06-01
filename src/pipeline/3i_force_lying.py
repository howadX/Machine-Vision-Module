"""
3i_force_lying.py — агрессивное обучение на lying через SMOTE + weight.
Цель: >70% recall для всех классов.
"""
import os, csv, pickle, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
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

def expand_full(X33):
    N = len(X33); extra = []
    for i in range(N):
        d = X33[i]
        extra.append([
            d[11]/max(d[23],1e-6), d[12]/max(d[24],1e-6),
            (d[11]+d[12])/max(d[23]+d[24],1e-6),
            d[25]/max(d[26],1e-6), d[27]/max(d[28],1e-6),
            d[13]/max(d[14],1e-6), d[15]/max(d[16],1e-6),
            d[0]/max((d[23]+d[24])/2,1e-6),
            (d[27]+d[28])/2 / max(d[0],1e-6),
            abs(d[27]-d[28])/max(d[23]+d[24],1e-6),
            (d[29]+d[30])/max(d[27]+d[28],1e-6),
        ])
    return np.column_stack([X33, np.array(extra)])

X = expand_full(X_raw)
print(f"Примеров: {len(y_raw)}, lying={np.sum(y_raw==2)} ({np.sum(y_raw==2)/len(y_raw):.0%})")

# ── SMOTE oversampling lying до 500 ─────────────────
target_lying = 500
current_lying = np.sum(y_raw == 2)
if current_lying < target_lying:
    smote = SMOTE(
        sampling_strategy={0: len(y_raw[y_raw==0]), 1: len(y_raw[y_raw==1]), 2: target_lying},
        random_state=42, k_neighbors=min(5, current_lying-1))
    X, y = smote.fit_resample(X, y_raw)
    print(f"После SMOTE: standing={np.sum(y==0)}, sitting={np.sum(y==1)}, lying={np.sum(y==2)}")
else:
    X, y = X, y_raw

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Сильный class_weight на lying ──────────────────
# Даём lying в 5× больше веса чем standing
scale_pos = {0: 1.0, 1: 2.0, 2: 5.0}
clf = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=10,
    num_leaves=63, min_child_samples=10,
    class_weight=scale_pos,
    random_state=42, verbose=-1, n_jobs=-1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.2%}")
print(f"\n{classification_report(y_test, y_pred, target_names=['standing','sitting','lying'])}")

# Сохранение
model_path = os.path.join(ROOT, "models", "emotion", "lightgbm_body_holistic_33.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"Сохранено: {model_path}")
