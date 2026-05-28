#!/usr/bin/env python3
"""Hyperparameter tuning LightGBM для геометрических признаков эмоций (Optuna)."""
import os, csv, pickle, json
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = '/home/dawoh/repos/Video Recognition Module'
CSV_PATH = os.path.join(ROOT, 'models/descriptors/geometric_affectnet.csv')

# ── Анатомически релевантные landmarks ──
FACIAL_REGIONS = {
    'mouth_inner': list(range(11, 17)) + [38, 41, 42, 62, 76, 77, 83, 85, 86, 89, 90, 92, 96,
                    179, 180, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194,
                    268, 271, 272, 273, 274, 275, 276, 277,
                    292, 302, 303, 304, 306, 307, 313, 315, 316, 319, 320, 325,
                    403, 404, 406, 407, 408, 410, 411, 412, 413, 414, 416],
    'mouth_outer': [0, 11, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
                    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
                    314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
    'eyebrows': [46, 52, 53, 55, 63, 65, 66, 70, 105, 107,
                 276, 282, 283, 285, 293, 295, 296, 300, 334, 336],
    'eyes': [33, 133, 157, 158, 159, 160, 161, 173, 246,
             362, 263, 384, 385, 386, 387, 388, 398, 466],
    'nose': [1, 2, 4, 5, 6, 19, 45, 48, 49, 64, 94, 97, 115, 168, 195, 197,
             98, 99, 100, 101, 102, 103, 104,
             327, 328, 329, 330, 331, 332, 333],
}
ALL_EMOTION_LANDMARKS = sorted(set().union(*FACIAL_REGIONS.values()))

# ── Загрузка данных ──
print('Загрузка...')
X, y = [], []
with open(CSV_PATH, newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        lbl = int(row[0])
        if lbl == 1:  # skip contempt
            continue
        new_lbl = lbl - 1 if lbl > 1 else lbl
        y.append(new_lbl)
        all_f = [float(v) for v in row[1:]]
        X.append([all_f[idx] for idx in ALL_EMOTION_LANDMARKS])

X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.int32)
print(f'Данные: {X.shape[0]} × {X.shape[1]} признаков')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── Усиленные class weights ──
CLASS_WEIGHT_CUSTOM = {
    0: 3.0,   # anger — ×3
    1: 4.0,   # disgust — ×4
    2: 2.5,   # fear — ×2.5
    3: 0.8,   # happy — ослабить
    4: 0.6,   # neutral — ослабить
    5: 1.0,   # sad
    6: 1.0,   # surprise
}

# ── Optuna objective ──
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'class_weight': CLASS_WEIGHT_CUSTOM,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        scores.append(accuracy_score(y_val, clf.predict(X_val)))
    
    return np.mean(scores)

# ── Запуск ──
print('Optuna tuning (50 trials)...')
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f'\nЛучшая точность (CV): {study.best_value:.2%}')
print(f'Параметры: {json.dumps(study.best_params, indent=2)}')

# ── Обучение финальной модели ──
print('\nОбучение финальной модели...')
best_params = study.best_params
best_params.update({
    'class_weight': CLASS_WEIGHT_CUSTOM,
    'random_state': 42,
    'verbose': -1,
})

clf = lgb.LGBMClassifier(**best_params)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
labels_7 = ['anger','disgust','fear','happy','neutral','sad','surprise']

print(f'\n=== Финальная точность: {acc:.2%} ===')
print(classification_report(y_test, y_pred, target_names=labels_7))

# Сохранение
out = {
    'model': clf,
    'landmark_indices': ALL_EMOTION_LANDMARKS,
    'num_features': len(ALL_EMOTION_LANDMARKS),
    'optuna_params': study.best_params,
    'class_weights': CLASS_WEIGHT_CUSTOM,
}
model_path = os.path.join(ROOT, 'models/emotion/lightgbm_emotion_regions.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(out, f)
print(f'\nМодель: {model_path}')
