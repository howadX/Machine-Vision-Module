#!/usr/bin/env python3
"""Builder for NB3b (Kaggle): GRIP (gravity-referenced pose features) — 14 azimuth-invariant
features for standing/sitting/lying, validated honestly
with grouped cross-validation. Emits nb3b_grip.ipynb.

Reproducible authoring: run `python _build_nb3b.py`. The notebook loads the
project's published keypoint dataset (`mvm-pose-landmarks`, keypoints only,
derived from URFD + Zenodo MultiPosture) — no images, fully Run-All on Kaggle.
Metrics printed by the notebook come from a real run; placeholders in prose are
marked [fill].
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
# GRIP: Gravity-Referenced Invariant Pose Features

*MVM ML Journey — Notebook 3b of the collection · prev: NB3 (weak supervision) · NB2 (pose manifold) · next: NB4 (teardown)*

> **GRIP** = **G**ravity-**R**eferenced **I**nvariant **P**ose features: fourteen quantities measured
> against the gravity vector (inclinations, joint flexions, extent ratios) that are invariant to camera
> azimuth *by construction*. (In the project's spike notes this was labelled "method C" — an A/B/C
> enumeration artifact; the descriptive name is used everywhere now.)

**TL;DR.** Fourteen gravity-referenced, azimuth-invariant geometric features + a shallow RandomForest
classify standing / sitting / lying at **0.93 ± 0.05** under **StratifiedGroupKFold** (no person in
both train and test). The honest surprise: on clean, fixed-viewpoint data the **raw 99-D coordinates
score *higher* — 0.98 ± 0.03** — because the hand features are a lossy summary that gives up ~4 points.
So why ship the features? **Rotate the camera and the ranking inverts:** raw coordinates collapse from
**≈0.94 → ≈0.35** at 90°, while GRIP stays flat at **0.91** at every angle. The engineered features
don't win on accuracy — they win on the axis you cannot afford to lose in the field: *the subject turning
relative to a fixed camera.* That trade is why GRIP is MVM's production pose classifier
(dependency-free forest, **< 1 ms** CPU).

> - grouped-CV: GRIP **0.93 ± 0.05** · raw coords **0.98 ± 0.03** · VAE-latent (~PCA32) **0.93 ± 0.06**
> - azimuth 0°→90°: GRIP **0.91 → 0.91** (flat) · raw coords **≈0.94 → ≈0.35** (collapse)

**Skills shown:** feature engineering with physical invariants · grouped cross-validation · honest
trade-off analysis (reporting where my own method *loses*) · production export.

> **Reproducibility note — read this.** Earlier project docs quoted **~0.959** for GRIP and a
> **"12 KB"** forest. *Neither reproduces here.* With the shipped RF params on this public data the honest
> grouped-CV is **≈0.93**, and the serialized forest is **~270 KB** in this notebook (**873 KB** in
> production, which trains on augmented data). The lower, reproducible numbers are the ones this notebook
> stands behind — an honest metric beats a flattering one.
""")

# ---------------------------------------------------------------- 2. Premise
md(r"""
## Invariance: learned vs. designed — and what it costs

A VAE (NB2) must **learn** from data that "torso inclination to gravity" is unchanged when the subject
turns relative to the camera. GRIP **designs** that invariance in: every feature is an angle to the
gravity-vertical, a joint flexion (fully rotation-invariant), or a ratio of extents. Rotating about the
vertical axis leaves them fixed *by construction* — the model spends neither capacity nor data
discovering it.

The catch this notebook measures honestly: **designed invariance is not free.** Compressing 33 landmarks
into 14 physical quantities discards information that helps on clean, fixed-viewpoint data — which is
exactly why raw coordinates out-score the features below. The engineering question is not "which is more
accurate?" but "**is the robustness worth the accuracy I pay for it?**" — and that answer is set by the
deployment, not the dataset. For a camera that people turn away from, it is worth it.

The dataset is the project's own **keypoints-only** pose set (URFD falls + Zenodo MultiPosture), 33×4
MediaPipe-style landmarks, **26 person/sequence groups**, three classes. No images — safe to publish,
fast to Run-All.
""")

# ---------------------------------------------------------------- 3. Load
code(r"""
import os, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
SEED = 0
np.random.seed(SEED)

# Published derivative of URFD + Zenodo MultiPosture — keypoints only, CC BY 4.0.
# Locate the file wherever Kaggle mounts the attached dataset (mount slug can vary).
cands = glob.glob("/kaggle/input/**/landmarks.npz", recursive=True)
if not cands:
    mounted = os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "NONE"
    raise FileNotFoundError(f"landmarks.npz not found. Attach dataset 'howadx/mvm-pose-landmarks'. "
                            f"Currently mounted under /kaggle/input: {mounted}")
NPZ = cands[0]
print("loading:", NPZ)
d = np.load(NPZ, allow_pickle=True)
X, y_str, src, grp = d["X"], d["y"], d["src"], d["grp"]   # X: (N, 33, 4) = x,y,z,vis
CLASSES = ["lying", "sitting", "standing"]
y = np.array([CLASSES.index(s) for s in y_str])
import collections
print("poses:", X.shape, "| classes:", dict(collections.Counter(y_str.tolist())))
print("groups (people/sequences):", len(set(grp.tolist())), "| sources:", dict(collections.Counter(src.tolist())))

METRICS = {"n_poses": int(X.shape[0]), "n_groups": len(set(grp.tolist()))}   # harvested by kctl fetch
""")

# ---------------------------------------------------------------- 4. Features
md(r"""
## The 14 features

Straight from the production spike (`spike_pose/spike_feats.py`). MediaPipe convention: **y grows
downward**, so "up" = `-y`. Every quantity below is invariant to rotation about the vertical axis.

| # | Feature | Measures | Why invariant |
|---|---------|----------|---------------|
| 1 | `torso_inc` | torso angle to vertical | gravity ⟂ azimuth |
| 2–4 | `thigh_inc`, `thigh_asym`, `shank_inc` | limb inclinations | same |
| 5–8 | `knee_flex`, `knee_asym`, `hip_flex`, `elbow_flex` | joint angles | a 3-D angle is absolute |
| 9–11 | `vext`, `hext`, `aspect` | point-cloud shape | ratios of extents |
| 12–14 | `nose_above`, `ank_below`, `kn_below` | vertical placement vs. hips | signed, in torso units |
""")

code(r"""
UP = np.array([0.0, -1.0, 0.0], np.float32)   # MediaPipe: y down -> up is -y
L = dict(nose=0, Lsh=11, Rsh=12, Lel=13, Rel=14, Lwr=15, Rwr=16,
         Lhip=23, Rhip=24, Lkn=25, Rkn=26, Lank=27, Rank=28)

def _ang_to_vert(v):
    n = np.linalg.norm(v)
    if n < 1e-6: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v / n, UP), -1, 1))))

def _flex(a, b, c):                      # angle a-b-c at vertex b
    u, w = a - b, c - b
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-6 or nw < 1e-6: return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u / nu, w / nw), -1, 1))))

FEAT_NAMES = ["torso_inc","thigh_inc","thigh_asym","shank_inc","knee_flex","knee_asym",
              "hip_flex","elbow_flex","vext","hext","aspect","nose_above","ank_below","kn_below"]

def features(lm):
    '''33x4 (x,y,z,vis) -> 14 azimuth-invariant, gravity-preserving features.'''
    p = lm[:, :3].astype(np.float32)
    g = lambda k: p[L[k]]
    hip = (g("Lhip") + g("Rhip")) / 2
    sh  = (g("Lsh")  + g("Rsh"))  / 2
    torso = sh - hip
    tlen = np.linalg.norm(torso)
    if tlen < 1e-6:
        tlen = max(np.linalg.norm(g("Lsh") - g("Rsh")), 1e-3)
    torso_inc = _ang_to_vert(torso)
    thigh_L = _ang_to_vert(g("Lkn") - g("Lhip")); thigh_R = _ang_to_vert(g("Rkn") - g("Rhip"))
    shank_L = _ang_to_vert(g("Lank") - g("Lkn")); shank_R = _ang_to_vert(g("Rank") - g("Rkn"))
    knee_L = _flex(g("Lhip"), g("Lkn"), g("Lank")); knee_R = _flex(g("Rhip"), g("Rkn"), g("Rank"))
    hip_L  = _flex(g("Lsh"),  g("Lhip"), g("Lkn"));  hip_R  = _flex(g("Rsh"),  g("Rhip"), g("Rkn"))
    elb_L  = _flex(g("Lsh"),  g("Lel"),  g("Lwr"));  elb_R  = _flex(g("Rsh"),  g("Rel"),  g("Rwr"))
    ys = p[:, 1]
    vext = (ys.max() - ys.min()) / tlen
    rad = np.sqrt((p[:, 0] - hip[0]) ** 2 + (p[:, 2] - hip[2]) ** 2)
    hext = 2 * rad.max() / tlen
    aspect = vext / (hext + 1e-6)
    nose_above = (hip[1] - g("nose")[1]) / tlen
    ank_below  = ((g("Lank")[1] + g("Rank")[1]) / 2 - hip[1]) / tlen
    kn_below   = ((g("Lkn")[1] + g("Rkn")[1]) / 2 - hip[1]) / tlen
    return np.array([torso_inc, (thigh_L+thigh_R)/2, abs(thigh_L-thigh_R), (shank_L+shank_R)/2,
                     (knee_L+knee_R)/2, abs(knee_L-knee_R), (hip_L+hip_R)/2, (elb_L+elb_R)/2,
                     vext, hext, aspect, nose_above, ank_below, kn_below], np.float32)

F = np.array([features(x) for x in X], np.float32)
Fdf = pd.DataFrame(F, columns=FEAT_NAMES)
print("feature matrix:", Fdf.shape)
Fdf.head(3)
""")

# ---------------------------------------------------------------- 5. Distributions
md(r"""
## The classes separate by eye

Three features carry most of the signal — `torso_inc` (lying), `knee_flex` (sitting), `aspect`
(standing is tall-and-thin). Overlapping-but-distinct histograms are exactly what a shallow model needs.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, feat in zip(axes, ["torso_inc", "knee_flex", "aspect"]):
    for c, name in enumerate(CLASSES):
        ax.hist(Fdf[feat][y == c], bins=30, alpha=0.55, label=name)
    ax.set_title(feat); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 6. Grouped CV
md(r"""
## Grouped cross-validation — the honest number (and a surprise)

A random split lets frames of the *same person* land in both train and test, and the model scores
itself partly on memorized identity. **StratifiedGroupKFold** keeps every person's frames on one side of
the split. This is the number that predicts performance on people the model has never seen — the only one
worth putting on a résumé.

Three inputs, same RF (production params), same folds: **14 features** vs. **raw normalized coordinates
(99-D, the VAE-style input)** vs. **VAE latent (32-D)** (approximated by PCA→32 — `sklearn` doesn't ship
our tiny VAE; the qualitative gap is what matters).

**The result is not what the title implies.** Raw coordinates score **highest** (≈0.98), GRIP and
the VAE-latent tie a few points lower (≈0.93). On clean, single-viewpoint data, throwing 33 landmarks
into a forest simply carries more signal than 14 summaries. If accuracy on fixed-camera data were the
only goal, you would ship the raw coordinates. Hold that thought for the azimuth test — it is the whole
argument.
""")

code(r"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def raw_coords(lm):
    '''VAE-style baseline: center on hip, scale by torso length, flatten 33x3.'''
    p = lm[:, :3].astype(np.float32)
    hip = (p[L["Lhip"]] + p[L["Rhip"]]) / 2
    sh  = (p[L["Lsh"]]  + p[L["Rsh"]])  / 2
    tlen = np.linalg.norm(sh - hip) + 1e-6
    return ((p - hip) / tlen).reshape(-1)

C = np.array([raw_coords(x) for x in X], np.float32)          # (N, 99)
groups = grp
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

def grouped_cv(Xin, estimator):
    s = cross_val_score(estimator, Xin, y, groups=groups, cv=sgkf, scoring="accuracy")
    return s.mean(), s.std()

# Same RF hyper-params as the production export (n_estimators=120, max_depth=16),
# so this number is directly comparable to the shipped forest and the quoted figure.
rf = lambda: RandomForestClassifier(n_estimators=120, max_depth=16, random_state=SEED, n_jobs=-1)
METRICS["grouped_cv"] = {}
for tag, key, Xin, est in [
    ("14 features", "grip", F, rf()),
    ("raw coords (99-D)", "raw_coords", C, rf()),
    ("VAE latent ~PCA32", "vae_pca32", C, make_pipeline(StandardScaler(), PCA(32, random_state=SEED), rf())),
]:
    m, s = grouped_cv(Xin, est)
    METRICS["grouped_cv"][key] = [round(float(m), 4), round(float(s), 4)]
    print(f"  {tag:22} grouped-CV acc = {m:.3f} ± {s:.3f}")   # <- fill the three into TL;DR
""")

# ---------------------------------------------------------------- 7. Azimuth test
md(r"""
## The azimuth test — where the trade-off pays off

Rotate every test pose about the vertical axis (the subject turning relative to a fixed camera) and
re-score **without retraining**. This is the deployment reality a home camera faces every minute.

The result is the entire argument: GRIP holds **0.91 at 0°, 15°, 30°, 45° and 90° — perfectly
flat.** Raw coordinates start *higher* (≈0.94) but decay steadily and then **fall off a cliff to the
1-in-3 chance line by 90°** (≈0.33–0.40 across runs). A 99-D coordinate vector *is* the viewpoint, so
rotating it destroys the model's world; the 14 features never saw a viewpoint to begin with.
""")

code(r"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def azimuth_rotate(lm, deg):
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    out = lm.copy(); x, z = lm[:, 0], lm[:, 2]
    out[:, 0] = x * c + z * s
    out[:, 2] = -x * s + z * c
    return out

# a plain grouped holdout so rotation is applied only to unseen people
gss_tr, gss_te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
ROT = [0, 15, 30, 45, 90]
curves = {}
for tag, feat_fn in [("14 features", features), ("raw coords", raw_coords)]:
    Xtr = np.array([feat_fn(X[i]) for i in gss_tr]); ytr = y[gss_tr]
    m = RandomForestClassifier(n_estimators=120, max_depth=16, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    accs = []
    for deg in ROT:
        Xte = np.array([feat_fn(azimuth_rotate(X[i], deg)) for i in gss_te])
        accs.append(accuracy_score(y[gss_te], m.predict(Xte)))
    curves[tag] = accs
    print(f"  {tag:12}: " + " ".join(f"az{d}={a:.3f}" for d, a in zip(ROT, accs)))
METRICS["azimuth_deg"] = ROT
METRICS["azimuth_acc"] = {t: [round(float(a), 4) for a in accs] for t, accs in curves.items()}

plt.figure(figsize=(6, 4))
for tag, accs in curves.items():
    plt.plot(ROT, accs, marker="o", label=tag)
plt.xlabel("camera azimuth rotation (deg)"); plt.ylabel("accuracy")
plt.title("Robustness to viewpoint — designed vs. learned invariance")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 7b. The trade-off figure
md(r"""
## The trade-off, in one figure

Two panels, one story. **Left:** on clean (fixed-viewpoint) data, raw coordinates are the *more*
accurate input — the 14 hand features are a lossy summary and give up a few points. **Right:** rotate
the camera and that ranking inverts violently — raw coordinates fall off a cliff while GRIP does not
move. Engineering is choosing which axis you refuse to lose.
""")

code(r"""
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.3))
gc = METRICS["grouped_cv"]
order = ["raw_coords", "grip", "vae_pca32"]
names = {"raw_coords": "raw coords\n(99-D)", "grip": "14 features\n(GRIP)",
         "vae_pca32": "VAE latent\n(~PCA32)"}
means = [gc[k][0] for k in order]; stds = [gc[k][1] for k in order]
axL.bar(range(3), means, yerr=stds, capsize=5, color=["#9aa0a6", "#2a9d8f", "#9aa0a6"])
axL.set_xticks(range(3)); axL.set_xticklabels([names[k] for k in order])
axL.set_ylim(0.80, 1.0); axL.set_ylabel("grouped-CV accuracy")
for i, m in enumerate(means):
    axL.text(i, m + stds[i] + 0.006, f"{m:.3f}", ha="center", fontsize=9)
gap = (gc["raw_coords"][0] - gc["grip"][0]) * 100
axL.set_title(f"Clean data: raw wins by ~{gap:.0f} pts")

for tag, accs in METRICS["azimuth_acc"].items():
    col = "#2a9d8f" if "feat" in tag else "#e76f51"
    axR.plot(METRICS["azimuth_deg"], accs, marker="o", label=tag, color=col)
axR.set_xlabel("camera azimuth (deg)"); axR.set_ylabel("accuracy")
axR.set_ylim(0.2, 1.0); axR.legend(); axR.grid(alpha=0.3)
axR.set_title("Under rotation: raw collapses, GRIP holds")
plt.suptitle("The trade-off: a few points of clean accuracy bought total viewpoint invariance",
             fontweight="bold")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 8. Why RF
md(r"""
## Why RandomForest, not LightGBM?

NB3's LightGBM peaked at ~71.7% on noisy COCO weak-labels. The lever here was **better data + invariant
features**, not a fancier booster — and for *this* product RandomForest wins on deployment, not accuracy:

| | LightGBM | RandomForest |
|---|---|---|
| grouped-CV accuracy (features) | comparable | **0.93 ± 0.05** |
| export to Rust | C++ runtime (`liblightgbm`) | plain binary, **no runtime** |
| runtime deps | yes | **zero** |
| CPU inference | ~1–2 ms | **< 1 ms** |

The decisive column is *runtime deps*: a forest is just thresholds and probabilities, so it re-implements
as a ~40-line reader over a flat binary — no `liblightgbm`, no ONNX, nothing to ship but bytes. That is
what makes the next section possible.
""")

# ---------------------------------------------------------------- 9. Export
md(r"""
## Export to a dependency-free binary (what production actually loads)

The forest is serialized as: `u32 n_trees`, then per tree `u32 n_nodes` and nodes of
`<i32 feature, f32 threshold, i32 left, i32 right, f32 prob[3]>`. Rust `include_bytes!` embeds it; a
~40-line reader walks the trees. Below, the same forest is written in that format and its on-disk size
reported.

**Honest sizing.** This notebook's forest is **~270 KB**; the *production* `pose_rf_forest.bin` is
**873 KB** because it is trained on augmented data (more samples → deeper trees). Earlier docs said
"12 KB" — that was never true for a 120-tree, depth-16 forest and is corrected here. Still: a few hundred
KB of pure bytes, zero dependencies, sub-millisecond — which is the property that matters, not the exact
figure.
""")

code(r"""
import struct
rf_final = RandomForestClassifier(n_estimators=120, max_depth=16, random_state=SEED, n_jobs=-1).fit(F, y)

def serialize_forest(forest):
    buf = bytearray(); buf += struct.pack("<I", len(forest.estimators_))
    for est in forest.estimators_:
        t = est.tree_; buf += struct.pack("<I", t.node_count)
        proba = t.value.reshape(t.node_count, -1)
        proba = proba / proba.sum(1, keepdims=True)
        for n in range(t.node_count):
            feat = int(t.feature[n]); thr = float(t.threshold[n])
            l, r = int(t.children_left[n]), int(t.children_right[n])
            p = proba[n]
            buf += struct.pack("<ifii3f", feat, thr, l, r, *[float(p[c]) for c in range(3)])
    return bytes(buf)

blob = serialize_forest(rf_final)
open("/kaggle/working/pose_rf_forest.bin", "wb").write(blob)
METRICS["forest_kb"] = round(len(blob) / 1024, 1)
METRICS["forest_trees"] = len(rf_final.estimators_)
print(f"serialized forest: {len(blob)/1024:.1f} KB, {len(rf_final.estimators_)} trees "
      f"-> pose_rf_forest.bin  (production embeds this via include_bytes!)")
""")

# ---------------------------------------------------------------- metrics dump
code(r"""
# --- Emit all harvested numbers for kctl fetch -> results/metrics.json ---
import json, os
os.makedirs("/kaggle/working", exist_ok=True)
with open("/kaggle/working/metrics.json", "w") as f:
    json.dump(METRICS, f, indent=2)
print(json.dumps(METRICS, indent=2))
""")

# ---------------------------------------------------------------- 10. Production
md(r"""
## Production Status (July 2026)

GRIP is the **primary pose classifier** in MVM (`mvm-posture`) since 2025. The VAE (NB2) was
demoted to an out-of-distribution gate. The forest inference is `< 1 ms` on CPU with no dependencies.

**Occlusion honesty (fusion layer, `mvm-fusion`).** When only the upper body is visible
(`BodyCoverage::Upper`), pose is **withheld** rather than guessed — GRIP supplies the 2-D torso
attitude used to make that call. The classifier does *not* detect falls (a standing→lying transition
over seconds); that is the anomaly detector's job (NB2 + fusion).
""")

# ---------------------------------------------------------------- 11. Limits
md(r"""
## Honest limits

- **Raw coordinates beat GRIP on clean data (≈0.98 vs ≈0.93).** The feature engineering costs ~4
  points of fixed-viewpoint accuracy; it is only worth it because the deployment rotates. On a bolted-down,
  known-viewpoint camera, this trade would be the *wrong* call.
- **The quoted 0.959 does not reproduce** with the shipped RF params on this public data (honest grouped-CV
  = ≈0.93). The gap is most plausibly the production run's train-time augmentation and/or a different data
  snapshot — flagged, not hidden.
- **Only 26 groups.** The ±0.052 std is wide; the mean alone is not trustworthy. Adding public pose data
  (more people) to tighten the interval is the obvious next experiment.
- `torso_length_m = 0.5` is assumed, not tape-measured — absolute 3-D placement is approximate.
- Three coarse classes; "sitting on a chair" vs. "sitting on the floor" is not distinguished.

## Conclusion — the engineering lesson

GRIP is not the most accurate model in this notebook; raw coordinates are. It is the *right* model,
because it is accurate on the one nuisance that a home camera cannot escape — people turning — and it is
right by **construction**, not by luck of the training distribution. The portfolio point is not a big
number; it is the discipline to (1) measure with grouped CV, (2) report where my own method loses, and
(3) pick the input whose failure mode matches the deployment. A model you can defend beats a model that
looks good.

## What changed
- **2026-07 (revision)** — live Kaggle run replaced projected numbers with measured ones. Reframed from
  "features beat a VAE" to an **honest viewpoint-invariance trade-off**: corrected 0.959 → **≈0.93**,
  "12 KB" → **~270 KB (873 KB prod)**, and surfaced that **raw coords win on clean data**.
- **2026-07** — notebook created from `spike_pose/spike_feats.py` + `train_pose_p1.py`.
- **2025** — GRIP replaced LightGBM (NB3) and the VAE (NB2) in production.

*Prev: NB3 (weak supervision) — where the data pipeline began · NB2 (pose manifold) — what GRIP
replaced. Next: NB4 — a component-level teardown on engineering judgment.*
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
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb3b_grip.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out_path, "|", len(cells), "cells")
