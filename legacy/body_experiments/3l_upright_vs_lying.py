"""
3l_upright_vs_lying.py — 2 класса: upright (standing+sitting) vs lying + эвристика внутри upright.
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

# 0,1 (standing,sitting) → 0 (upright); 2 (lying) → 1 (lying)
y_bin = np.where(y_all == 2, 1, 0).astype(np.int32)

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
print(f"upright: {np.sum(y_bin==0)}, lying: {np.sum(y_bin==1)}, total: {len(y_bin)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

clf = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=10,
    num_leaves=63, class_weight={0: 1, 1: 8},
    random_state=42, verbose=-1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['upright','lying']))

model_path = os.path.join(ROOT, "models", "emotion", "lightgbm_body_upright.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"Сохранено: {model_path}")
