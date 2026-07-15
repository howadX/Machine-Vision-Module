#!/usr/bin/env python3
"""Builder for NB1 (Kaggle): Distilling a CNN emotion recognizer into a
sub-millisecond geometric MLP. Emits nb1_distillation.ipynb.

Authoring is kept reproducible: run `python _build_nb1.py` to regenerate the
notebook. Result numbers are intentionally left as placeholders — they are
produced by an actual Kaggle run, never fabricated here.
"""
import json
import os

cells = []


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------- 1. Title
md(r"""
# Distilling a CNN Emotion Recognizer into a Sub-millisecond Geometric MLP

*MVM ML Journey — Notebook 1 of 4 · [GitHub repo](#) · next: NB2 (pose manifold)*

**TL;DR.** I take a heavy CNN facial-emotion recognizer (HSEmotion, EfficientNet-B0)
as a **teacher**, and distill its knowledge into a tiny **student** MLP that works on
**468 scale-invariant geometric features** of the face — not raw pixels. The student is
~7× smaller and runs in **< 1 ms on CPU**.

> - Teacher accuracy (7 emotions): **[fill after run]**
> - Student (distilled) accuracy: **[fill after run]** — gap ≈ **[fill]** pts
> - Student CPU head-latency: **[fill] ms** · size: **[fill]** vs teacher **[fill]**

**Scope, honestly.** This is an *efficiency / edge-inference* study, **not** a SOTA chase.
On 7 emotions these accuracies are modest by design — the story is the *trade-off*
(accuracy ↔ latency ↔ size), and the *engineering decisions* behind it, taken straight
from the real MVM project history.

**Skills shown:** knowledge distillation · feature engineering · model compression · edge inference.
""")

# ---------------------------------------------------------------- 2. Framing
md(r"""
## Why distill into a geometric MLP?

MVM is an on-device perception module for LLM agents (webcam → who / emotion / pose / scene),
running **real-time on CPU** alongside other models. A raw-pixel CNN per face is expensive.

Two ideas do the heavy lifting:

1. **Geometric features instead of pixels.** From 468 MediaPipe face landmarks I compute
   distances normalized by the interocular distance → invariant to face scale / camera distance,
   and only 468 floats instead of a 224×224×3 image.
2. **Distillation instead of training from scratch.** With a small dataset, the student learns
   far more from the teacher's *soft* probability targets than from hard labels alone.

```
teacher (pixels → EfficientNet-B0) ──soft labels──▶ student (468 geom features → MLP)
```
""")

# ---------------------------------------------------------------- 3. Setup
code(r"""
# --- Setup (Kaggle: enable Internet in notebook settings) ---
!pip install -q hsemotion mediapipe lightgbm timm 2>/dev/null

import os, io, time, random, warnings, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["GLOG_minloglevel"] = "3"  # quiet mediapipe/absl

SEED = 42
random.seed(SEED); np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Distillation hyper-parameters (from the MVM project: train_mlp_distilled.py)
T = 3.0          # temperature
ALPHA = 0.3      # weight on the hard-label CE term

EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
N_CLS = len(EMOTIONS)
print("device:", DEVICE, "| torch:", torch.__version__)
""")

# ---------------------------------------------------------------- 4. Data
md(r"""
## 4. Data — the actual project line

MVM's emotion data evolved through several sources. I keep that history because *the choices are
the story*:

- **FER2013** (`msambare/fer2013`) — really pulled via `kagglehub` in `train_emotion.py` for the
  EfficientNet-B0 fine-tune. 7 classes, but **48×48 grayscale**.
- **AffectNet subset** — the real training data of the production LightGBM (`train_combined.py`,
  Optuna-tuned). Higher resolution.
- **`my_emotions`** (350 self-collected frames) — oversampled in the CNN branch. **Not used here**
  (privacy: it contains real people); the public notebook substitutes public data and says so.

> **Attach on Kaggle:** the FER2013 dataset (always), and an AffectNet dataset you have the rights
> to use. Set the paths below. AffectNet is a research-licensed dataset — attach your own copy.
""")

code(r"""
# --- Configure dataset paths (edit to match attached Kaggle datasets) ---
# FER2013: folders <split>/<emotion>/*.jpg  (labels: angry, disgust, fear, happy, neutral, sad, surprise)
FER2013_DIR   = "/kaggle/input/fer2013"
# AffectNet: any layout with per-emotion subfolders of RGB face crops. Optional but recommended.
AFFECTNET_DIR = "/kaggle/input/affectnet"   # <-- set to your attached AffectNet, or leave missing

FER_LABEL_MAP = {  # FER2013 folder name -> our canonical label
    "angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "sad": "sad", "surprise": "surprise",
}

def list_images(root, label_map=None, limit_per_class=None):
    "Return (path, canonical_label) pairs from a <root>/<class>/*.jpg layout."
    rows = []
    if not os.path.isdir(root):
        return rows
    for cls_dir in sorted(glob.glob(os.path.join(root, "**", ""), recursive=False)):
        raw = os.path.basename(os.path.normpath(cls_dir)).lower()
        lab = (label_map or {}).get(raw, raw if raw in EMOTIONS else None)
        if lab is None:
            continue
        paths = sorted(glob.glob(os.path.join(cls_dir, "*")))
        if limit_per_class:
            paths = paths[:limit_per_class]
        rows += [(p, lab) for p in paths]
    return rows

# FER2013 usually splits into train/ and test/ — gather both, tag the split.
fer_rows = []
for split in ("train", "test"):
    fer_rows += [(p, l, split) for (p, l) in
                 list_images(os.path.join(FER2013_DIR, split), FER_LABEL_MAP)]
if not fer_rows:  # flat layout fallback
    fer_rows = [(p, l, "train") for (p, l) in list_images(FER2013_DIR, FER_LABEL_MAP)]

fer_df = pd.DataFrame(fer_rows, columns=["path", "label", "split"])
print("FER2013 images:", len(fer_df))
if len(fer_df):
    ax = fer_df["label"].value_counts().reindex(EMOTIONS).plot.bar(title="FER2013 class distribution")
    ax.set_ylabel("images"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 4b. Discovery
md(r"""
## 4b. A limitation discovered in development → a decision

Before trusting geometric features, I measure whether landmarks are even *detectable* on FER2013's
48×48 grayscale faces. This is the concrete, measured finding that pushed MVM off pixel-res FER2013
for the geometric branch.
""")

code(r"""
import cv2, mediapipe as mp

_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.3)

def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)   # grayscale files load as 3-equal-channel BGR
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_landmarks(rgb):
    "Return (468,2) pixel-space landmarks or None."
    if rgb is None:
        return None
    res = _mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = rgb.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

# Detection-rate probe on a FER2013 sample
probe = fer_df.sample(min(400, len(fer_df)), random_state=SEED) if len(fer_df) else fer_df
hits = sum(detect_landmarks(read_rgb(p)) is not None for p in probe["path"])
rate = hits / max(len(probe), 1)
print(f"FER2013 landmark detection rate: {rate:.1%}  (n={len(probe)})   # <- fill into TL;DR")
""")

md(r"""
**Decision (as taken in MVM).** If landmarks fail on a large share of 48×48 faces, the geometric
pipeline is starved of inputs. So:

- **(a)** run teacher-labeling and train the geometric student on a **higher-resolution source**
  (AffectNet subset — the real production data);
- **(b)** keep **FER2013 as a raw-pixel baseline** only.

If no AffectNet dataset is attached, the notebook falls back to FER2013 so it still runs end-to-end
— with a printed warning that the geometric numbers are then resolution-limited.
""")

# ---------------------------------------------------------------- 5. Teacher
md(r"""
## 5. Teacher — HSEmotion (EfficientNet-B0)

HSEmotion (`enet_b0_8_best_vgaf`, trained on AffectNet + VGAF, ~73% RAF-DB) predicts **8** emotions.
I drop **Contempt** and renormalize to the 7 MVM classes.
""")

code(r"""
from hsemotion.facial_emotions import HSEmotionRecognizer

teacher = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf", device=DEVICE)

# HSEmotion output order (8 classes) and mapping to our 7 (drop 'Contempt')
HSE_CLASSES = ["Anger","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
HSE_TO_OURS = {"Anger":"anger","Disgust":"disgust","Fear":"fear","Happiness":"happy",
               "Neutral":"neutral","Sadness":"sad","Surprise":"surprise"}
KEEP_IDX = [i for i, c in enumerate(HSE_CLASSES) if c != "Contempt"]
OURS_ORDER = [EMOTIONS.index(HSE_TO_OURS[HSE_CLASSES[i]]) for i in KEEP_IDX]  # reorder to EMOTIONS

def teacher_logits7(rgb_face):
    "Return teacher logits reordered to EMOTIONS (7,), Contempt dropped."
    _, scores = teacher.predict_emotions(rgb_face, logits=True)  # (8,) logits
    logits7 = np.asarray(scores, dtype=np.float32)[KEEP_IDX]
    out = np.empty(N_CLS, np.float32)
    out[OURS_ORDER] = logits7
    return out
""")

# ---------------------------------------------------------------- 6-9 build set
md(r"""
## 6–9. Build the distillation set (teacher soft labels + geometric features)

For each image from the high-res source: teacher logits (soft targets) **and** 468 geometric
features. Scale-invariant features = distance from the nose tip to every landmark, divided by the
interocular distance.
""")

code(r"""
NOSE_IDX = 4          # MediaPipe FaceMesh nose tip
EYE_L, EYE_R = 33, 263  # outer eye corners (interocular normalization)

def compute_features_468(pts):
    "pts: (468,2) pixel landmarks -> (468,) scale-invariant distances from the nose."
    nose = pts[NOSE_IDX]
    interocular = np.linalg.norm(pts[EYE_L] - pts[EYE_R]) + 1e-6
    return (np.linalg.norm(pts - nose, axis=1) / interocular).astype(np.float32)

# --- Choose the high-res source; fall back to FER2013 with a warning ---
affect_rows = list_images(AFFECTNET_DIR, limit_per_class=None)
if affect_rows:
    src = pd.DataFrame(affect_rows, columns=["path", "label"])
    print(f"Teacher-labeling source: AffectNet ({len(src)} images)")
else:
    print("WARNING: AffectNet not attached -> falling back to FER2013 (resolution-limited geometry).")
    src = fer_df[["path", "label"]].copy()

MAX_PER_CLASS = 2000  # cap for Kaggle runtime; raise if time allows
src = src.groupby("label", group_keys=False).apply(
    lambda d: d.sample(min(len(d), MAX_PER_CLASS), random_state=SEED))

feats, soft, hard, kept = [], [], [], 0
for path, label in src[["path", "label"]].itertuples(index=False):
    rgb = read_rgb(path)
    lms = detect_landmarks(rgb)
    if lms is None:
        continue
    feats.append(compute_features_468(lms))
    soft.append(teacher_logits7(rgb))
    hard.append(EMOTIONS.index(label))
    kept += 1

X = np.stack(feats); S = np.stack(soft); y = np.array(hard)
print(f"distillation rows: {kept} / {len(src)}  (landmark yield {kept/max(len(src),1):.1%})")
print("feature matrix:", X.shape)
""")

code(r"""
# Standardize features (scaler saved as in the project), stratified split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X).astype(np.float32)

Xtr, Xte, Str, Ste, ytr, yte = train_test_split(
    Xs, S, y, test_size=0.2, stratify=y, random_state=SEED)
print("train/test:", Xtr.shape[0], Xte.shape[0])
""")

# ---------------------------------------------------------------- 10 theory
md(r"""
## 10. Distillation objective

$$\mathcal{L} = \alpha\,\underbrace{\mathrm{CE}(z_s, y)}_{\text{hard label}}
\; + \; (1-\alpha)\,\underbrace{T^2\cdot \mathrm{KL}\!\big(\sigma(z_s/T)\,\|\,\sigma(z_t/T)\big)}_{\text{soft, temperature-scaled}}$$

with temperature **T = 3.0** and **α = 0.3** (from `train_mlp_distilled.py`). The `T²` factor keeps
the soft-gradient magnitude comparable to the hard term. Softened teacher targets carry the
"dark knowledge" — the relative probabilities of the *wrong* classes.
""")

# ---------------------------------------------------------------- 11 model
code(r"""
import torch.nn as nn
import torch.nn.functional as F

class EmotionMLP(nn.Module):
    def __init__(self, in_dim=468, n_cls=N_CLS):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 512),    nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),    nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, n_cls),
        )
    def forward(self, x):
        return self.net(x)

def n_params(m):
    return sum(p.numel() for p in m.parameters())

print("student params:", f"{n_params(EmotionMLP()):,}")
""")

# ---------------------------------------------------------------- 12 train
code(r"""
def distill_loss(z_s, hard, teacher_logits, T=T, alpha=ALPHA):
    ce = F.cross_entropy(z_s, hard)
    t_soft = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(F.log_softmax(z_s / T, dim=1), t_soft, reduction="batchmean") * (T * T)
    return alpha * ce + (1 - alpha) * kl

def train_mlp(Xtr, ytr, Str=None, epochs=300, lr=1e-3, hard_only=False):
    m = EmotionMLP().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2)
    Xt = torch.tensor(Xtr, device=DEVICE)
    yt = torch.tensor(ytr, device=DEVICE)
    St = None if Str is None else torch.tensor(Str, device=DEVICE)
    m.train()
    for ep in range(epochs):
        opt.zero_grad()
        z = m(Xt)
        loss = F.cross_entropy(z, yt) if hard_only else distill_loss(z, yt, St)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
    return m

@torch.no_grad()
def mlp_acc(m, X, y):
    m.eval()
    pred = m(torch.tensor(X, device=DEVICE)).argmax(1).cpu().numpy()
    return (pred == y).mean(), pred

student = train_mlp(Xtr, ytr, Str, epochs=300)          # distilled
acc_student, pred_student = mlp_acc(student, Xte, yte)
print(f"student (distilled) test acc: {acc_student:.3f}   # <- fill into TL;DR")
""")

# ---------------------------------------------------------------- 13 eval
code(r"""
from sklearn.metrics import confusion_matrix, classification_report
import itertools

def show_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(N_CLS))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues"); plt.title(title); plt.colorbar()
    plt.xticks(range(N_CLS), EMOTIONS, rotation=45, ha="right"); plt.yticks(range(N_CLS), EMOTIONS)
    for i, j in itertools.product(range(N_CLS), range(N_CLS)):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    plt.ylabel("true"); plt.xlabel("pred"); plt.tight_layout(); plt.show()

show_confusion(yte, pred_student, "Student (distilled)")
print(classification_report(yte, pred_student, target_names=EMOTIONS, digits=3))
""")

# ---------------------------------------------------------------- 14 baselines
md(r"""
## 14. Baselines — isolate what distillation buys

- **(a)** the same MLP trained on **hard labels only** (measures the distillation gain),
- **(b)** **LightGBM** on the same 468 features,
- **(c)** the **teacher** itself (upper bound),
- **(d)** random (1/7 ≈ 14.3%).
""")

code(r"""
import lightgbm as lgb

# (a) hard-only MLP
mlp_hard = train_mlp(Xtr, ytr, epochs=300, hard_only=True)
acc_hard, _ = mlp_acc(mlp_hard, Xte, yte)

# (b) LightGBM
gbm = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                         subsample=0.9, random_state=SEED)
gbm.fit(Xtr, ytr)
acc_gbm = (gbm.predict(Xte) == yte).mean()

# (c) teacher on the same test rows: argmax of the soft targets we already stored
acc_teacher = (Ste.argmax(1) == yte).mean()

# (d) random
acc_rand = 1.0 / N_CLS

summary = pd.DataFrame({
    "model":    ["teacher (HSEmotion)", "student (distilled)", "MLP (hard only)", "LightGBM", "random"],
    "test_acc": [acc_teacher, acc_student, acc_hard, acc_gbm, acc_rand],
})
print(summary.to_string(index=False))
""")

# ---------------------------------------------------------------- 15 pareto
md(r"""
## 15. The money chart — accuracy ↔ latency ↔ size

CPU inference latency measured per model. **Honesty note:** the student/LightGBM latency below is
the **classifier head** on a 468-vector; the *full* geometric pipeline also pays the MediaPipe
landmark cost (~3–8 ms) once per face — still far below a per-face CNN, and shared across emotion +
pose + gaze heads.
""")

code(r"""
def cpu_latency_ms(fn, reps=200):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3

student_cpu = student.to("cpu").eval()
x1 = torch.tensor(Xte[:1])
feat1 = Xte[:1]

lat_student = cpu_latency_ms(lambda: student_cpu(x1))
lat_gbm     = cpu_latency_ms(lambda: gbm.predict(feat1))
# teacher latency is image-level; measure on one decoded face if available
sample_rgb = read_rgb(src["path"].iloc[0])
lat_teacher = cpu_latency_ms(lambda: teacher.predict_emotions(sample_rgb, logits=True), reps=30) \
              if sample_rgb is not None else np.nan

sizes = {  # params as a size proxy
    "student (distilled)": n_params(student_cpu),
    "MLP (hard only)":     n_params(student_cpu),
    "LightGBM":            gbm.booster_.num_trees(),
    "teacher (HSEmotion)": 5_300_000,  # EfficientNet-B0 approx
}
lat = {"student (distilled)": lat_student, "MLP (hard only)": lat_student,
       "LightGBM": lat_gbm, "teacher (HSEmotion)": lat_teacher}
acc = dict(zip(summary["model"], summary["test_acc"]))

plt.figure(figsize=(7, 5))
for name in ["teacher (HSEmotion)", "student (distilled)", "LightGBM"]:
    plt.scatter(lat[name], acc[name], s=80, label=name)
    plt.annotate(name, (lat[name], acc[name]), textcoords="offset points", xytext=(6, 6), fontsize=8)
plt.xscale("log"); plt.xlabel("CPU latency per inference (ms, log)"); plt.ylabel("test accuracy")
plt.title("Accuracy vs latency — student sits on the efficient frontier")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
print(f"latency ms — student: {lat_student:.3f} | lgbm: {lat_gbm:.3f} | teacher: {lat_teacher:.2f}")
""")

# ---------------------------------------------------------------- 16 discussion
md(r"""
## 16. Results & honest framing

**[fill after run]** Distillation lifted the geometric MLP by **[X] pts** over the hard-only MLP,
landing within **[Y] pts** of the teacher — at **[Z] ms** CPU head-latency and **~7×** fewer
parameters.

The point is **not** that we beat SOTA (raw-pixel CNNs reach ~72–76% on 7 emotions). It is that
teacher-level behavior is transferable into a model cheap enough to run continuously on CPU,
sharing the MediaPipe landmark pass with the rest of MVM's perception heads.
""")

# ---------------------------------------------------------------- 17 ablations
md(r"""
## 17. Limitations & what to probe next

- **Landmark failures** on occlusion / extreme profile starve the geometric head — measured yield
  reported above; a pixel fallback is warranted for hard frames.
- **Temperature / α sensitivity** — a small `T ∈ {2,3,4}` × `α ∈ {0.2,0.3,0.5}` sweep would quantify
  robustness (left as an exercise cell).
- **Small / imbalanced data** — class weights and balanced-accuracy reporting matter (see NB3 for the
  full Optuna + balanced-metric treatment on poses).
- **Domain shift** — teacher trained on AffectNet/VGAF; on-device faces differ. MVM's answer was a
  self-collected set (`my_emotions`) — out of scope here for privacy.
""")

# ---------------------------------------------------------------- 18 conclusion
md(r"""
## 18. Conclusion

A CNN emotion recognizer's knowledge distills cleanly into a **468-feature, sub-millisecond MLP**
suitable for always-on CPU perception. The interesting part was the *decision trail* — measuring the
FER2013 landmark wall, moving the geometric branch to a higher-res source, and letting soft targets
do the teaching.

**In production (MVM):** this student runs real-time inside the Rust pipeline, one head among
identity / pose / gaze, all sharing a single landmark pass. → **NB2: a pose manifold that knows when
it's surprised.**

---
*Data: FER2013 (Kaggle) + a research-licensed AffectNet subset (attach your own). Teacher: HSEmotion
`enet_b0_8_best_vgaf`. Self-collected data excluded for privacy.*
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

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb1_distillation.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "|", len(cells), "cells")
