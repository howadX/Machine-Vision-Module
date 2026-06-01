"""
3j_clean_retrain.py — честное обучение без SMOTE. Только class_weight='balanced'.
"""
import os, csv, pickle, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

# Честное обучение: только class_weight='balanced', без SMOTE
clf = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=10,
    num_leaves=63, min_child_samples=10,
    class_weight="balanced",
    random_state=42, verbose=-1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, target_names=['standing','sitting','lying']))

# Калибровка вероятностей (Platt scaling)
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
calibrated.fit(X_train, y_train)
y_pred_cal = calibrated.predict(X_test)
print(f"\nCalibrated accuracy: {accuracy_score(y_test, y_pred_cal):.2%}")
print(classification_report(y_test, y_pred_cal, target_names=['standing','sitting','lying']))

# Сохраняем калиброванную модель
model_path = os.path.join(ROOT, "models", "emotion", "lightgbm_body_holistic_33.pkl")
with open(model_path, "wb") as f:
    pickle.dump(calibrated, f)
print(f"Сохранено: {model_path}")
