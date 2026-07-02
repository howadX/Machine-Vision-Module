#!/usr/bin/env python3
"""
Шаг 1+2+3: объединить свои 350 + AffectNet (исключив contempt) → обучить LightGBM с Optuna.
"""
import os, csv, json, time, pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb
import optuna

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESC_DIR = os.path.join(ROOT, "pipeline")
MODEL_DIR = os.path.join(ROOT, "models", "emotion")

LABEL_ORDER = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CONTEMPT_LABEL = 7  # класс contempt в geometric_affectnet.csv

# ── 1. Загрузка своих 350 ──────────────────────────────
print("=" * 60)
print("ШАГ 1: Загрузка данных")
print("=" * 60)

X_all, y_all = [], []
for path, name in [
    (os.path.join(DESC_DIR, "geometric_468.csv"), "свои 350"),
    (os.path.join(DESC_DIR, "geometric_affectnet.csv"), "AffectNet 16K"),
]:
    x_rows, y_rows = [], []
    skipped_contempt = 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            label = int(row[0])
            if label == CONTEMPT_LABEL:
                skipped_contempt += 1
                continue
            y_rows.append(label)
            x_rows.append([float(v) for v in row[1:]])
    print(f"  {name}: {len(y_rows)} rows (skipped contempt: {skipped_contempt})")
    X_all.extend(x_rows)
    y_all.extend(y_rows)

X = np.array(X_all, dtype=np.float64)
y = np.array(y_all, dtype=np.int32)
n_samples, n_features = X.shape
print(f"\n  ИТОГО: {n_samples} примеров, {n_features} признаков")
print(f"  Распределение классов: {dict(zip(LABEL_ORDER, np.bincount(y)))}")

# ── 2. Разделение train/test 80/20 ────────────────────
print("\n" + "=" * 60)
print("ШАГ 2: Train/test split (80/20, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ── 3. Optuna tuning ───────────────────────────────────
print("\n" + "=" * 60)
print("ШАГ 3: Optuna tuning LightGBM (30 trials)")
print("=" * 60)

def objective(trial):
    params = {
        "objective": "multiclass",
        "num_class": 7,
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(
        lgb.LGBMClassifier(**params), X_train, y_train,
        cv=skf, scoring="balanced_accuracy", n_jobs=1
    )
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15, show_progress_bar=True)

best_params = study.best_params
print(f"\n  Best trial #{study.best_trial.number}: balanced_accuracy={study.best_value:.4f}")
print(f"  Best params: {best_params}")

# ── 4. Обучение финальной модели ───────────────────────
print("\n" + "=" * 60)
print("ШАГ 4: Обучение финальной модели")
print("=" * 60)

final_params = {
    "objective": "multiclass",
    "num_class": 7,
    "metric": "multi_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "random_state": 42,
    "n_jobs": -1,
    **best_params,
}

start = time.time()
clf = lgb.LGBMClassifier(**final_params)
clf.fit(X_train, y_train)
elapsed = time.time() - start
print(f"  Обучено за {elapsed:.1f} сек")

# ── 5. Оценка ──────────────────────────────────────────
print("\n" + "=" * 60)
print("ШАГ 5: Оценка на holdout")
print("=" * 60)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"  Holdout accuracy: {acc:.4f}")
print(f"  Holdout balanced_accuracy: {bal_acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion matrix:")
print(f"  {cm.tolist()}")

print(f"\n  Classification report:")
print(classification_report(y_test, y_pred, target_names=LABEL_ORDER, digits=4))

# ── 6. Сохранение ──────────────────────────────────────
print("\n" + "=" * 60)
print("ШАГ 6: Сохранение модели")
print("=" * 60)

model_path = os.path.join(MODEL_DIR, "lightgbm_combined_14k.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"  Модель сохранена: {model_path} ({os.path.getsize(model_path) / 1024:.0f} KB)")

# Сохраняем отчёт
report_path = os.path.join(ROOT, "pipeline", "combined_14k_report.json")
report = {
    "dataset": {
        "source": "geometric_468.csv (350 own) + geometric_affectnet.csv (14,496 AffectNet, contempt excluded)",
        "total_samples": n_samples,
        "features": n_features,
        "class_counts": np.bincount(y).tolist(),
        "label_order": LABEL_ORDER,
        "split": "stratified 80/20 holdout, random_state=42",
    },
    "best_trial": {
        "number": study.best_trial.number,
        "cv3_balanced_accuracy": float(study.best_value),
        "params": best_params,
    },
    "holdout": {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "training_seconds": round(elapsed, 1),
    },
    "saved_model": model_path,
}

os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"  Отчёт сохранён: {report_path}")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
