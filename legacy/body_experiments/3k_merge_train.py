"""
3k_merge_train.py — объединение val2017 + train2017, финальное обучение.
"""
import os, csv, pickle, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_csv(path):
    X, y = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            y.append(int(row[0]))
            X.append([float(v) for v in row[1:]])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)

X1, y1 = load_csv(os.path.join(ROOT, "models", "descriptors", "body_posture_holistic_33.csv"))
X2, y2 = load_csv(os.path.join(ROOT, "models", "descriptors", "body_posture_train2017_33.csv"))

X_all = np.vstack([X1, X2])
y_all = np.hstack([y1, y2])

# Расширение признаков
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

X = expand_full(X_all)

for name, count in zip(['standing','sitting','lying'], np.bincount(y_all)):
    print(f'{name}: {count}')
print(f'Total: {len(y_all)}, признаков: {X.shape[1]}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y_all, test_size=0.2, random_state=42, stratify=y_all)

clf = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=10,
    num_leaves=63, min_child_samples=20,
    class_weight="balanced",
    random_state=42, verbose=-1,
)
clf.fit(X_train, y_train)

print("\n=== Без калибровки ===")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['standing','sitting','lying']))

cal = CalibratedClassifierCV(clf, method='sigmoid', cv=3, n_jobs=-1)
cal.fit(X_train, y_train)
print("\n=== С калибровкой ===")
y_pred_cal = cal.predict(X_test)
print(classification_report(y_test, y_pred_cal, target_names=['standing','sitting','lying']))

model_path = os.path.join(ROOT, "models", "emotion", "lightgbm_body_holistic_33.pkl")
with open(model_path, "wb") as f:
    pickle.dump(cal, f)
print(f"Сохранено: {model_path}")
