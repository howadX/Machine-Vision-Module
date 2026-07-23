#!/usr/bin/env python3
"""Builder for NB3 (Kaggle): Weak supervision at scale — auto-labeling COCO poses
+ Optuna-tuned LightGBM, and publishing a derived Kaggle dataset. Emits
nb3_weak_supervision.ipynb.

Reproducible authoring: run `python _build_nb3.py`. Metrics are placeholders to
be filled from a real Kaggle run (never fabricated).
"""
import json
import os

cells = []


def md(text: str):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------- 1. Title
md(r"""
# Weak Supervision at Scale: Auto-labeling COCO Poses (Optuna-tuned LightGBM)

*MVM ML Journey — Notebook 3 of 4 · [GitHub repo](#) · prev: NB2 (pose manifold) · next: NB4 (teardown)*

**TL;DR.** Zero manual labels. A geometric **labeling function** annotates thousands of COCO poses
as standing / sitting / lying; I class-balance them and train a **LightGBM** classifier with
**Optuna** hyper-parameter search (balanced-accuracy, 3-fold CV). Then I package the result as a
**public Kaggle dataset**, closing the loop *own data → own model → own dataset*.

> - Auto-labeled poses: **[fill after run]** · balanced-accuracy (holdout): **[fill]**
> - Optuna trials: **[fill]** · lift over default params: **[fill] pts**

**Skills shown:** weak supervision · Optuna / HPO · class imbalance · LightGBM · dataset curation.
""")

# ---------------------------------------------------------------- 2. Premise
md(r"""
## The data-centric premise

Models are cheap; **labels** are the bottleneck. Weak supervision replaces hand-labeling with a
*programmatic* labeling function — noisy, but scalable to thousands of examples in seconds. The
question that makes or breaks it: **does a model trained on weak labels generalize beyond the
heuristic, or just memorize it?** I measure both.
""")

# ---------------------------------------------------------------- 3. Setup + load
code(r"""
!pip install -q optuna lightgbm 2>/dev/null

import os, json, random, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# COCO keypoints only (no images downloaded). Set to your attached dataset.
KEYPOINTS_JSON = "/kaggle/input/coco-2017-dataset/coco2017/annotations/person_keypoints_val2017.json"
MAX_POSES = 6000   # cap, as in the project's extract_coco_poses.py
""")

code(r"""
# --- Load COCO poses; keep torso-complete bodies (heuristic needs shoulders+hips) ---
L_SH,R_SH,L_HIP,R_HIP,L_ANK,R_ANK = 5,6,11,12,15,16
SKELETON = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),(0,5),(0,6),(0,1),(0,2),(1,3),(2,4)]

with open(KEYPOINTS_JSON) as f:
    coco = json.load(f)

rows = []
for a in coco["annotations"]:
    if a.get("num_keypoints", 0) < 8:
        continue
    kp = np.array(a["keypoints"], dtype=np.float32).reshape(17, 3)   # (x,y,v)
    if min(kp[[L_SH, R_SH, L_HIP, R_HIP], 2]) == 0:                  # torso must be labeled
        continue
    rows.append((a["image_id"], a["id"], kp))
    if len(rows) >= MAX_POSES:
        break
print("candidate poses:", len(rows))
""")

# ---------------------------------------------------------------- 4-5 labeling
md(r"""
## 4. The labeling function

Pure geometry → standing / sitting / lying:

- **lying** — torso vector far from vertical (angle > 55°),
- else split by **leg extension** (ankle-below-hip distance / torso length): large → **standing**,
  small (bent / tucked legs) → **sitting**.

`label_confidence` = normalized margin from the deciding threshold (kept in the published dataset).
These are **weak** labels, not ground truth.
""")

code(r"""
CLASSES = ["standing", "sitting", "lying"]
def label_pose(kp):
    pts = kp[:, :2]; v = kp[:, 2]
    hip = (pts[L_HIP] + pts[R_HIP]) / 2.0
    sh  = (pts[L_SH]  + pts[R_SH])  / 2.0
    torso = sh - hip; tl = np.linalg.norm(torso) + 1e-6
    cosv = np.dot(torso / tl, np.array([0.0, -1.0]))
    ang = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    if ang > 55:
        conf = min(1.0, (ang - 55) / 35.0)
        return "lying", conf
    if v[L_ANK] == 0 and v[R_ANK] == 0:            # no ankle info -> low-confidence sitting
        return "sitting", 0.2
    ankle = np.nanmean(np.where(v[[L_ANK, R_ANK], None] > 0, pts[[L_ANK, R_ANK]], np.nan), axis=0)
    leg = (ankle[1] - hip[1]) / tl
    lab = "standing" if leg > 1.2 else "sitting"
    conf = min(1.0, abs(leg - 1.2) / 1.2)
    return lab, conf

labels, confs = [], []
for _, _, kp in rows:
    lab, conf = label_pose(kp); labels.append(lab); confs.append(conf)
labels = np.array(labels); confs = np.array(confs, dtype=np.float32)
ax = pd.Series(labels).value_counts().reindex(CLASSES).plot.bar(title="Weak-label distribution (raw)")
ax.set_ylabel("poses"); plt.tight_layout(); plt.show()
""")

code(r"""
# --- Class balancing (cap per class), as in extract_coco_poses.py ---
idx_by_cls = {c: np.where(labels == c)[0] for c in CLASSES}
cap = min(len(v) for v in idx_by_cls.values())      # balance to the smallest class
cap = min(cap, MAX_POSES // len(CLASSES))
rng = np.random.default_rng(SEED)
keep = np.concatenate([rng.choice(idx_by_cls[c], cap, replace=False) for c in CLASSES])
rng.shuffle(keep)
print("balanced set:", len(keep), "| per class:", cap)
""")

# ---------------------------------------------------------------- 6 sanity viz
md(r"""
## 6. Sanity check — look at the auto-labels

Weak labels are only worth training on if they *look* right. A few per class, by eye:
""")

code(r"""
def draw_pose(ax, pts, color, title=None):
    ax.scatter(pts[:, 0], pts[:, 1], s=10, c=color)
    for i, j in SKELETON:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], c=color, lw=1)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    if title: ax.set_title(title, fontsize=9)

fig, axes = plt.subplots(len(CLASSES), 4, figsize=(11, 8))
for r, c in enumerate(CLASSES):
    picks = rng.choice(idx_by_cls[c], 4, replace=False)
    for k, i in enumerate(picks):
        draw_pose(axes[r, k], rows[i][2][:, :2], f"C{r}", c if k == 0 else None)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 7 noise
md(r"""
## 7. Honest label-noise estimate

The model inherits the labeling function's mistakes. Borderline **sitting/lying** and crouches are
the usual failure modes. Below: a random audit sample — annotate agreement to estimate the error
rate, and record it in the dataset card. *(In a headless run this is a visual audit; fill the
estimated error rate after inspecting.)*
""")

code(r"""
audit = rng.choice(keep, 8, replace=False)
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for k, i in enumerate(audit):
    lab, conf = label_pose(rows[i][2])
    draw_pose(axes[k // 4, k % 4], rows[i][2][:, :2], "C7", f"{lab} ({conf:.2f})")
plt.suptitle("Audit sample — judge label correctness by eye"); plt.tight_layout(); plt.show()
# estimated_error_rate = ??? %   # <- fill after visual audit, put in dataset card
""")

# ---------------------------------------------------------------- 8 features
md(r"""
## 8. Feature extraction (LightGBM)

Torso-normalized geometry. **Note (for §13):** I deliberately **exclude** the exact quantities the
labeling function uses (torso angle, leg extension), so the model must generalize from *correlated
but different* features rather than trivially echo the rule. Missing joints → **NaN**, which LightGBM
handles natively.
""")

code(r"""
KP = ["nose","l_eye","r_eye","l_ear","r_ear","l_sh","r_sh","l_elb","r_elb",
      "l_wri","r_wri","l_hip","r_hip","l_knee","r_knee","l_ank","r_ank"]

def features(kp):
    pts = kp[:, :2].copy(); v = kp[:, 2]
    hip = (pts[L_HIP] + pts[R_HIP]) / 2.0
    sh  = (pts[L_SH]  + pts[R_SH])  / 2.0
    tl = np.linalg.norm(sh - hip) + 1e-6
    Q = (pts - hip) / tl
    Q[v == 0] = np.nan                                  # missing -> NaN
    feat = {f"dist_{KP[i]}": np.linalg.norm(Q[i]) for i in range(17)}   # 17 distances
    feat["shoulder_width"] = np.linalg.norm(Q[L_SH] - Q[R_SH])
    feat["hip_width"]      = np.linalg.norm(Q[L_HIP] - Q[R_HIP])
    feat["arm_asym"]       = abs(feat["dist_l_wri"] - feat["dist_r_wri"])
    feat["leg_asym"]       = abs(feat["dist_l_ank"] - feat["dist_r_ank"])
    ys = Q[:, 1][~np.isnan(Q[:, 1])]
    feat["vert_extent"]    = (ys.max() - ys.min()) if len(ys) else np.nan
    return feat

Xdf = pd.DataFrame([features(rows[i][2]) for i in keep])
y = np.array([CLASSES.index(labels[i]) for i in keep])
print("feature matrix:", Xdf.shape, "| features:", list(Xdf.columns))
""")

# ---------------------------------------------------------------- 9-10 optuna
md(r"""
## 9–10. Optuna HPO — balanced accuracy, 3-fold CV

Objective = **balanced_accuracy** (robust to residual imbalance) over StratifiedKFold(3). Search
space: `n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, subsample,
colsample_bytree, reg_alpha, reg_lambda`.
""")

code(r"""
import optuna, lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

Xtr, Xte, ytr, yte = train_test_split(Xdf, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
N_TRIALS = 30      # raise if Kaggle time allows

def objective(t):
    params = dict(
        n_estimators=t.suggest_int("n_estimators", 100, 600),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=t.suggest_int("max_depth", 3, 10),
        num_leaves=t.suggest_int("num_leaves", 15, 255),
        min_child_samples=t.suggest_int("min_child_samples", 5, 80),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        random_state=SEED, n_jobs=-1, verbose=-1)
    clf = lgb.LGBMClassifier(**params)
    return cross_val_score(clf, Xtr, ytr, cv=cv, scoring="balanced_accuracy").mean()

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"best CV balanced-acc: {study.best_value:.3f}  # <- fill into TL;DR")
print("best params:", study.best_params)
""")

code(r"""
# --- Optuna diagnostics ---
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
plot_optimization_history(study); plt.tight_layout(); plt.show()
plot_param_importances(study); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 11 final
md(r"""
## 11. Final model & holdout evaluation (vs default-params baseline)
""")

code(r"""
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
import itertools

def evaluate(clf, tag):
    clf.fit(Xtr, ytr); pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred); bacc = balanced_accuracy_score(yte, pred)
    print(f"[{tag}] acc={acc:.3f} balanced_acc={bacc:.3f}")
    return pred, bacc

pred_base, _ = evaluate(lgb.LGBMClassifier(random_state=SEED, verbose=-1), "default params")
best = lgb.LGBMClassifier(**study.best_params, random_state=SEED, n_jobs=-1, verbose=-1)
pred_tuned, bacc_tuned = evaluate(best, "Optuna-tuned")

cm = confusion_matrix(yte, pred_tuned)
plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xticks(range(3), CLASSES, rotation=45, ha="right"); plt.yticks(range(3), CLASSES)
for i, j in itertools.product(range(3), range(3)):
    plt.text(j, i, cm[i, j], ha="center", va="center")
plt.ylabel("true (weak)"); plt.xlabel("pred"); plt.title("Optuna-tuned LightGBM")
plt.tight_layout(); plt.show()
print(classification_report(yte, pred_tuned, target_names=CLASSES, digits=3))
""")

# ---------------------------------------------------------------- 12 importance
code(r"""
# --- Feature importance ---
imp = pd.Series(best.feature_importances_, index=Xdf.columns).sort_values()
imp.plot.barh(figsize=(6, 6), title="LightGBM feature importance"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 13 discussion
md(r"""
## 13. Does the model generalize beyond the heuristic?

Since the exact deciding quantities (torso angle, leg extension) were **withheld**, the classifier
reconstructs the labels from *correlated* geometry (limb distances, extent, widths). Where model and
heuristic disagree is exactly where to look next:

- agreement high → the rule is geometrically redundant (many features encode it);
- systematic disagreement on borderline sitting/lying → candidates for **active learning** (send
  low-`label_confidence` poses to a human), the natural next step beyond this notebook.
""")

# ---------------------------------------------------------------- 14 publish
md(r"""
## 14. Publish as a Kaggle Dataset — `coco-poses-weakly-labeled`

A **derived** dataset: images belong to COCO; *mine* is the labeling function + geometric features +
balancing. I publish **features + labels + keypoints**, referencing images by `coco_image_id` — so no
COCO images are redistributed.
""")

code(r"""
# --- Assemble the publishable table (features + labels + raw keypoints + provenance) ---
kp_cols = []
kp_arr = []
for i in keep:
    flat = rows[i][2].reshape(-1)   # 51 = 17*(x,y,v)
    kp_arr.append(flat)
kp_arr = np.array(kp_arr)
for i in range(17):
    kp_cols += [f"kp_{KP[i]}_x", f"kp_{KP[i]}_y", f"kp_{KP[i]}_v"]

out = Xdf.reset_index(drop=True).copy()
out.insert(0, "coco_image_id", [rows[i][0] for i in keep])
out.insert(1, "coco_ann_id",   [rows[i][1] for i in keep])
out["label"] = [CLASSES[c] for c in y]
out["label_confidence"] = [confs[i] for i in keep]
kp_df = pd.DataFrame(kp_arr, columns=kp_cols)
out = pd.concat([out, kp_df], axis=1)
# reuse the same split as evaluation
out["split"] = "train"
out.loc[out.index.isin(Xte.index), "split"] = "test"

os.makedirs("/kaggle/working", exist_ok=True)
out.to_csv("/kaggle/working/coco_poses_weakly_labeled.csv", index=False)
print("wrote coco_poses_weakly_labeled.csv:", out.shape)
out.head(3)
""")

code(r"""
# --- Write the dataset card / schema alongside the CSV ---
card = '''# coco-poses-weakly-labeled

Weakly-labeled human-pose dataset **derived from COCO** `person_keypoints_val2017`.
Labels are produced by a geometric labeling function (weak supervision), not human annotation.

## Provenance
Derived from COCO 2017 person keypoints. Images are NOT redistributed; reference them via
`coco_image_id` (COCO image id) and reconstruct from the official COCO release.

## License
CC BY 4.0, with attribution to the COCO Consortium (https://cocodataset.org). Derivative work.

## Labels
Weak / heuristic: standing | sitting | lying. `label_confidence` = normalized margin from the
labeling-function threshold. Estimated label error rate: FILL_AFTER_AUDIT %. Not ground truth.

## Schema
- coco_image_id, coco_ann_id : provenance / image reference
- dist_*, shoulder_width, hip_width, arm_asym, leg_asym, vert_extent : torso-normalized geometric features (NaN = missing joint)
- kp_<name>_{x,y,v} : raw COCO keypoints (v: 0 unlabeled, 1 labeled/occluded, 2 visible)
- label, label_confidence, split

## Reproduce / starter notebook
See NB3 of the MVM ML-journey collection (this notebook).

## Intended use & limitations
3 coarse pose classes; accuracy ceiling bounded by the heuristic. For research/education.
'''
open("/kaggle/working/DATASET_CARD.md", "w").write(card)
print("wrote DATASET_CARD.md")
""")

# ---------------------------------------------------------------- 15 conclusion
md(r"""
## 15. Conclusion

Weak supervision turned unlabeled COCO poses into a balanced, Optuna-tuned classifier with **zero
manual labels**, and the artifact ships as a reusable public dataset. Honest limits: the ceiling is
the labeling function's quality; `label_confidence` is the hook for the obvious next step —
**active learning** on the disagreements.

**In production (MVM):** heuristic auto-labeling bootstrapped the pose dataset before any hand-labels
existed. → **NB4: why identity embeddings are the wrong tool for emotion — a component-level
teardown.**

---
*Data: COCO `person_keypoints_val2017` (CC BY 4.0), keypoints only. Published derivative:
`coco-poses-weakly-labeled`.*
""")

# ---------------------------------------------------------------- write
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb3_weak_supervision.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out_path, "|", len(cells), "cells")
