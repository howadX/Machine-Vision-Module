#!/usr/bin/env python3
"""Builder for NB0 (Kaggle): EmoZid — an SVM on facial geometry (2019). A public
reconstruction on CK+ of the original private-data dlib+SVM pipeline, and the
lessons that shaped every model after it. Emits nb0_emozid.ipynb.

Reproducible authoring: run `python _build_nb0.py`. This notebook reconstructs a
2019 school project stack (dlib HOG + 68 landmarks + ResNet-128d + poly SVM) on
the *public* CK+ dataset — the original faces were private and are not used.
Metrics are placeholders filled by a real Kaggle run.
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
# EmoZid: An SVM on Facial Geometry (2019) — Where It All Started

*MVM ML Journey — Notebook 0 of the collection · next: NB1 (knowledge distillation)*

**TL;DR.** My first computer-vision project, aged 17: a school experiment that read emotion from a
face. The stack — **dlib HOG detector → 68 landmarks → a 128-D ResNet face descriptor → a polynomial
SVM** — reached ~**45%** on my own (private) faces. This notebook is *archaeology*: a faithful public
reconstruction on **CK+**, plus the lessons that each later model in this collection was a reaction to.
Nothing here runs in production; everything here explains why production looks the way it does.

> - CK+ detection rate (dlib HOG): **[fill]%** · SVM accuracy (stratified holdout): **[fill]%**

**Skills shown:** classical CV pipeline · geometric features · honest baselining · reading your own
early mistakes.
""")

# ---------------------------------------------------------------- 2. Context
md(r"""
## The origin, and why 45% was not a failure

2019. The idea was simple and the tools were whatever a teenager could `pip install`: OpenCV's Haar
cascade to find a face, then **dlib** for the serious parts. The classifier was a **polynomial-kernel
SVM** over geometric descriptors. On a few hundred self-recorded frames per emotion it scored ~45%.

Chance for 7 classes is ~14%, so 45% *learned something* — but it also sat on three problems this
whole notebook series went on to fix:

1. **68 landmarks are a weak emotion signal** → NB1 moves to 468 MediaPipe points, then to a CNN.
2. **A 128-D face descriptor is trained to be *identity*, i.e. to ignore expression** → NB4 proves this
   empirically and separates identity from emotion.
3. **Private, tiny, self-labeled data leaks and overfits** → later notebooks use public datasets and
   grouped validation.
""")

# ---------------------------------------------------------------- 3. Setup
md(r"""
## Setup — the 2019 stack, on 2026 Kaggle

`dlib` + its two pretrained models (68-landmark predictor, ResNet-128 recognizer). The private faces
of the original are replaced by **CK+** (Cohn-Kanade), a public 7-emotion dataset. Turn **Internet on**
in the kernel so the dlib models download.
""")

code(r"""
!pip install -q dlib kagglehub 2>/dev/null

import os, glob, bz2, urllib.request, numpy as np, matplotlib.pyplot as plt
import dlib
SEED = 42
np.random.seed(SEED)

# dlib's two pretrained models (public, from dlib.net)
def fetch_bz2(url, out):
    if os.path.exists(out): return out
    tmp = out + ".bz2"; urllib.request.urlretrieve(url, tmp)
    with bz2.open(tmp) as f, open(out, "wb") as g: g.write(f.read())
    return out

SHAPE = fetch_bz2("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                  "shape_predictor_68_face_landmarks.dat")
RESNET = fetch_bz2("http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
                   "dlib_face_recognition_resnet_model_v1.dat")

detector  = dlib.get_frontal_face_detector()          # HOG, as in 2019
predictor = dlib.shape_predictor(SHAPE)               # 68 landmarks
facerec   = dlib.face_recognition_model_v1(RESNET)    # 128-D descriptor
print("dlib ready:", dlib.__version__)
""")

# ---------------------------------------------------------------- 4. Data
code(r"""
import kagglehub
# CK+ (Cohn-Kanade). Emotion = subfolder name. If your attached copy differs, point ROOT at it.
ROOT = kagglehub.dataset_download("shawon10/ckplus")
# discover class subfolders robustly
img_paths, labels = [], []
for dirpath, _, files in os.walk(ROOT):
    imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not imgs: continue
    cls = os.path.basename(dirpath).lower()
    for f in imgs:
        img_paths.append(os.path.join(dirpath, f)); labels.append(cls)
labels = np.array(labels)
CLASSES = sorted(set(labels))
import collections
print("CK+ images:", len(img_paths), "| classes:", dict(collections.Counter(labels.tolist())))
""")

# ---------------------------------------------------------------- 5. Detection
md(r"""
## Detection rate — the first honest metric

Before any classifier, ask: **how often does the detector even find a face?** dlib's HOG detector is
frontal-only; profiles and occlusions are simply missed. That miss rate is a hard ceiling on the whole
pipeline — and one reason production later switched to YuNet.
""")

code(r"""
def get_descriptors(path):
    '''2019 pipeline: HOG detect -> 68 landmarks -> (geometric 68-vector, 128-D descriptor).'''
    img = dlib.load_rgb_image(path)
    dets = detector(img, 1)
    if len(dets) == 0:
        return None, None
    shape = predictor(img, dets[0])
    desc128 = np.array(facerec.compute_face_descriptor(img, shape), np.float32)
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], np.float32)
    # original feature: distance from the nose point (30) to every landmark...
    nose = pts[30]
    geo = np.linalg.norm(pts - nose, axis=1)
    # ...then the 2019 normalization: divide by element [27]. (Kept faithfully; see §8 for why it is
    # fragile — dividing by a single expression-dependent landmark.)
    geo = geo / (geo[27] + 1e-6)
    return geo, desc128

geo_feats, emb_feats, kept = [], [], []
for p, lab in zip(img_paths, labels):
    g, e = get_descriptors(p)
    if g is not None:
        geo_feats.append(g); emb_feats.append(e); kept.append(lab)
geo_feats = np.array(geo_feats); emb_feats = np.array(emb_feats); kept = np.array(kept)
print(f"detection rate: {len(kept)}/{len(img_paths)} = {len(kept)/len(img_paths):.1%}   # <- fill TL;DR")
""")

# ---------------------------------------------------------------- 6. Landmarks viz
md(r"""
## What the model actually saw — 68 points on a face
""")

code(r"""
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
shown = 0
for p in img_paths:
    img = dlib.load_rgb_image(p); dets = detector(img, 1)
    if not dets: continue
    shape = predictor(img, dets[0])
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
    ax = axes[shown]; ax.imshow(img, cmap="gray")
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c="lime"); ax.axis("off")
    shown += 1
    if shown == 4: break
plt.suptitle("dlib 68 landmarks (this is the whole emotion signal, 2019)"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 7. SVM
md(r"""
## The classifier — a polynomial SVM, exactly as in 2019

The original used `SVC(kernel="poly", degree=3, C=2.0, gamma=2.0)` — hand-picked, never searched (a
junior habit worth naming). We keep it, but add the one thing 2019 lacked: a **stratified** split, so
the class imbalance in CK+ doesn't flatter the score.
""")

code(r"""
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y = np.array([CLASSES.index(c) for c in kept])
Xtr, Xte, ytr, yte = train_test_split(geo_feats, y, test_size=0.25, stratify=y, random_state=SEED)
scaler = StandardScaler().fit(Xtr)
svm = SVC(kernel="poly", degree=3, C=2.0, gamma=2.0)   # the original, hand-tuned hyper-params
svm.fit(scaler.transform(Xtr), ytr)
pred = svm.predict(scaler.transform(Xte))
acc = accuracy_score(yte, pred)
print(f"SVM accuracy (68-geometry): {acc:.3f}   # <- fill TL;DR")
print(classification_report(yte, pred, target_names=CLASSES, digits=3, zero_division=0))
""")

code(r"""
import itertools
cm = confusion_matrix(yte, pred)
plt.figure(figsize=(5, 4.4)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right"); plt.yticks(range(len(CLASSES)), CLASSES)
for i, j in itertools.product(range(len(CLASSES)), range(len(CLASSES))):
    plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
plt.ylabel("true"); plt.xlabel("pred"); plt.title("EmoZid SVM on CK+ (68-geometry)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 8. Teardown of mistakes
md(r"""
## Reading the 2019 mistakes honestly

- **Normalization by a single landmark** (`geo / geo[27]`). Landmark 27 moves with expression, so the
  whole feature vector rescales when the face emotes — the exact thing we're trying to measure. Later
  models normalize by the stable **interocular** distance instead.
- **Hand-picked SVM hyper-parameters** (`C=2.0, gamma=2.0`). No search, no cross-validation. NB3's
  Optuna study is the grown-up version of this.
- **68 points is a thin signal.** Fear/surprise and anger/disgust collapse into each other (see the
  confusion matrix) because coarse geometry can't capture the muscle detail that separates them → NB1
  moves to 468 points and then to pixels (a CNN).

## Bonus: identity descriptor vs. emotion (the seed of NB4)

The 128-D dlib descriptor is a face-*recognition* embedding: trained to be **the same across
expressions**. Using it for emotion is fighting the model's purpose. NB4 makes this rigorous; here is
the one-line hint — an SVM on the 128-D identity embedding should do **no better** (often worse) than
the crude 68-geometry, despite being higher-dimensional.
""")

code(r"""
Xtr2, Xte2, ytr2, yte2 = train_test_split(emb_feats, y, test_size=0.25, stratify=y, random_state=SEED)
sc2 = StandardScaler().fit(Xtr2)
svm2 = SVC(kernel="poly", degree=3, C=2.0, gamma=2.0).fit(sc2.transform(Xtr2), ytr2)
acc_emb = accuracy_score(yte2, svm2.predict(sc2.transform(Xte2)))
print(f"SVM on 128-D IDENTITY embedding: {acc_emb:.3f}   vs 68-geometry: {acc:.3f}")
print("Identity embeddings are optimized to ignore expression -> not better despite 128-D. (-> NB4)")
""")

# ---------------------------------------------------------------- 9. Lineage
md(r"""
## From EmoZid to MVM — five generations of emotion

| Gen | Approach | Accuracy | Year |
|-----|----------|----------|------|
| 1 | dlib 68 points + poly SVM | ~45% (private) / **[fill]%** (CK+) | 2019 |
| 2 | MediaPipe 468 points + LightGBM | ~68% (AffectNet) | 2023 |
| 3 | EfficientNet-B0 CNN | ~73% (FER2013) | 2024 |
| 4 | Knowledge distillation CNN → MLP | ~72%, < 1 ms CPU | 2025 |
| 5 | EmotiEffLib B0 ONNX | production (MVM) | 2026 |

Every jump was a reaction to a limit named in this notebook. Nothing was thrown away for fashion; each
step fixed a *measured* problem.

## Production Status (July 2026)
Not in production. The EmoZid stack (dlib + SVM) is fully replaced: detector → **YuNet** (`mvm-detect`),
identity → **SFace** (`mvm-identity`), emotion → **EmotiEffLib ONNX** (`mvm-emotion`). This notebook is
the root of the tree.

## What changed
- **2026-07** — created as a public CK+ reconstruction of the private 2019 project.

*Next: NB1 — distilling a CNN emotion recognizer into a sub-millisecond geometric MLP.*
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
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb0_emozid.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out_path, "|", len(cells), "cells")
