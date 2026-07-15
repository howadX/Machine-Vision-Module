#!/usr/bin/env python3
"""Builder for NB2 (Kaggle): A pose manifold that knows when it's surprised —
beta-VAE + few-shot + anomaly detection. Emits nb2_posevae.ipynb.

Reproducible authoring: run `python _build_nb2.py`. Result numbers are left as
placeholders to be filled from an actual Kaggle run (never fabricated).

Data decision: public COCO person_keypoints_val2017 (17 2D keypoints). The MVM
project used MediaPipe 33 3D landmarks; the adaptation to COCO-17-2D is stated
honestly in-notebook.
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
# A Pose Manifold That Knows When It's Surprised — β-VAE + Few-shot + Anomaly Detection

*MVM ML Journey — Notebook 2 of 4 · [GitHub repo](#) · prev: NB1 (distillation) · next: NB3 (weak supervision)*

**TL;DR.** I learn a low-dimensional **manifold of human poses** with a β-VAE, then reuse the *same*
frozen latent for three jobs:

> - **Classify** pose (linear probe on the latent): test acc **[fill after run]**
> - **Few-shot**: add a brand-new pose class from just **5 examples**, no retraining
> - **Anomaly / OOD**: flag unusual poses via **−log p(z)** under the prior

**Skills shown:** representation learning · VAE · few-shot / continual learning · anomaly detection.

**Data & honesty.** Public **COCO `person_keypoints_val2017`** (17 2D keypoints). The MVM project
used MediaPipe **33 3D** landmarks with a 32-d latent (48.6K params, classifier 97.1% on its own
labeled set); here I adapt the same design to **COCO-17-2D** for full reproducibility — so the
architecture sizes and numbers below are recomputed, not copied.
""")

# ---------------------------------------------------------------- 2. Motivation
md(r"""
## Why a manifold instead of a plain classifier?

A direct pose classifier gives you a label and nothing else. MVM needs more from the same features:
**plausibility** (is this a real human pose?), **generation** (interpolate / sample), and
**out-of-distribution** detection (an unusual pose the agent should notice). A VAE gives all three
from one 32-d latent — the VPoser idea: the latent *is* the space of plausible poses; stepping
outside it = anomaly.
""")

# ---------------------------------------------------------------- 3. Setup + data
code(r"""
# --- Setup (Kaggle: enable Internet for umap-learn) ---
!pip install -q umap-learn 2>/dev/null

import os, json, time, random, warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
import torch; torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# COCO keypoint annotations. Attach a dataset that contains this file and set the path.
# (We only need the keypoints JSON — no images are downloaded.)
KEYPOINTS_JSON = "/kaggle/input/coco-2017-dataset/coco2017/annotations/person_keypoints_val2017.json"
""")

code(r"""
# --- Load COCO-17 keypoints (no images needed) ---
KP_NAMES = ["nose","l_eye","r_eye","l_ear","r_ear","l_sh","r_sh","l_elb","r_elb",
            "l_wri","r_wri","l_hip","r_hip","l_knee","r_knee","l_ank","r_ank"]
L_SH,R_SH,L_HIP,R_HIP,L_ANK,R_ANK = 5,6,11,12,15,16
SKELETON = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),(0,5),(0,6),(0,1),(0,2),(1,3),(2,4)]

with open(KEYPOINTS_JSON) as f:
    coco = json.load(f)

poses = []
for a in coco["annotations"]:
    if a.get("num_keypoints", 0) != 17:      # keep fully-labeled bodies -> no imputation
        continue
    kp = np.array(a["keypoints"], dtype=np.float32).reshape(17, 3)  # (x, y, v)
    poses.append(kp[:, :2])                    # (17,2) image-space
P = np.stack(poses)
print("fully-labeled COCO poses:", P.shape)   # (N,17,2)
""")

# ---------------------------------------------------------------- 4-5 normalize
md(r"""
## 4. Feature normalization — invariance by construction

The MVM encoder made pose features invariant *before* any learning: center on the hip, scale so the
torso length = 1, and rotate the spine to vertical (Rodrigues in 3D). Here, adapted to 2D:

1. translate to the **hip center** (mid of hips 11/12),
2. **scale** by torso length (hip→shoulder) = 1,
3. **rotate** the torso vector onto the vertical axis.

Result: a 34-d vector (17×2) invariant to image position, person scale, and in-plane tilt.
""")

code(r"""
def normalize_pose(pts):
    "pts (17,2) image-space -> (17,2) hip-centered, torso-scaled, spine-vertical."
    hip = (pts[L_HIP] + pts[R_HIP]) / 2.0
    sh  = (pts[L_SH]  + pts[R_SH])  / 2.0
    torso = sh - hip
    tl = np.linalg.norm(torso) + 1e-6
    Q = (pts - hip) / tl
    theta = np.arctan2(torso[1], torso[0])     # current torso angle
    rot = (-np.pi / 2.0) - theta               # rotate torso onto 'up' = (0,-1) in y-down coords
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (Q @ R.T).astype(np.float32)

Pn = np.stack([normalize_pose(p) for p in P])         # (N,17,2)
X = Pn.reshape(len(Pn), -1).astype(np.float32)        # (N,34) VAE input
print("VAE input:", X.shape)
""")

code(r"""
# --- Skeleton plotting helper + before/after normalization ---
def draw_pose(ax, pts, color="C0", title=None):
    ax.scatter(pts[:, 0], pts[:, 1], s=12, c=color)
    for i, j in SKELETON:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], c=color, lw=1)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    if title: ax.set_title(title, fontsize=9)

idx = np.random.choice(len(P), 4, replace=False)
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for k, i in enumerate(idx):
    draw_pose(axes[0, k], P[i],  "C3", "raw" if k == 0 else None)
    draw_pose(axes[1, k], Pn[i], "C0", "normalized" if k == 0 else None)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- weak labels
md(r"""
### Weak pose labels (shared with NB3)

COCO has no pose-class labels, so I derive weak standing/sitting/lying labels with a geometric
heuristic — the same labeling-function idea developed fully in **NB3**. These are *weak* labels; the
classifier head is evaluated against them, so its accuracy measures agreement with the heuristic,
not ground truth.
""")

code(r"""
CLASSES = ["standing", "sitting", "lying"]
def weak_label(pts):
    hip = (pts[L_HIP] + pts[R_HIP]) / 2.0
    sh  = (pts[L_SH]  + pts[R_SH])  / 2.0
    torso = sh - hip; tl = np.linalg.norm(torso) + 1e-6
    cosv = np.dot(torso / tl, np.array([0.0, -1.0]))          # vs vertical (up = -y)
    ang = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    if ang > 55:
        return "lying"
    ankle = (pts[L_ANK] + pts[R_ANK]) / 2.0
    leg = (ankle[1] - hip[1]) / tl                            # y-down: standing -> large positive
    return "standing" if leg > 1.2 else "sitting"

y = np.array([CLASSES.index(weak_label(p)) for p in P])
for c in range(len(CLASSES)):
    print(f"{CLASSES[c]:9s}: {(y == c).sum()}")
""")

# ---------------------------------------------------------------- 6-8 VAE
md(r"""
## 6. β-VAE architecture

Encoder `34 → 128 → 64 → (μ, logσ²):32`, symmetric decoder, reparameterization trick. Loss =
reconstruction MSE + **β·KL**, with **β = 0.001** (VPoser-style: favor reconstruction fidelity;
the KL stays high but the latent remains smooth enough to sample/interpolate).
""")

code(r"""
import torch.nn as nn
import torch.nn.functional as F

LATENT = 32
class PoseVAE(nn.Module):
    def __init__(self, in_dim=34, latent=LATENT):
        super().__init__()
        self.enc = nn.Sequential(nn.BatchNorm1d(in_dim),
                                 nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2))
        self.fc_mu = nn.Linear(64, latent)
        self.fc_lv = nn.Linear(64, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, in_dim))
    def encode(self, x):
        h = self.enc(x); return self.fc_mu(h), self.fc_lv(h)
    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
    def forward(self, x):
        mu, lv = self.encode(x); z = self.reparam(mu, lv)
        return self.dec(z), mu, lv
    def encode_to_z(self, x):                # deterministic latent (mu)
        return self.encode(x)[0]
    def log_prob_z(self, x):                 # -log p(z) surprise under N(0,I)
        mu, lv = self.encode(x)
        return 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(1)

print("VAE params:", f"{sum(p.numel() for p in PoseVAE().parameters()):,}")  # recomputed for 34-d
""")

code(r"""
# --- Train the VAE ---
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

BETA = 0.001
def vae_loss(recon, x, mu, lv):
    rec = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1))
    return rec + BETA * kl, rec, kl

vae = PoseVAE().to(DEVICE)
opt = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
Xt = torch.tensor(Xtr, device=DEVICE)
hist = []
vae.train()
for ep in range(400):
    opt.zero_grad()
    recon, mu, lv = vae(Xt)
    loss, rec, kl = vae_loss(recon, Xt, mu, lv)
    loss.backward(); nn.utils.clip_grad_norm_(vae.parameters(), 1.0); opt.step(); sched.step()
    hist.append((rec.item(), kl.item()))
hist = np.array(hist)
plt.plot(hist[:, 0], label="recon MSE"); plt.plot(hist[:, 1] * BETA, label="β·KL")
plt.legend(); plt.xlabel("epoch"); plt.title("VAE training"); plt.show()
print(f"final recon MSE: {hist[-1,0]:.4f}  # <- fill into TL;DR")
""")

code(r"""
# --- Reconstruction quality: original vs decoded skeletons ---
vae.eval()
with torch.no_grad():
    rec, _, _ = vae(torch.tensor(Xte[:4], device=DEVICE))
rec = rec.cpu().numpy().reshape(-1, 17, 2)
orig = Xte[:4].reshape(-1, 17, 2)
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for k in range(4):
    draw_pose(axes[0, k], orig[k], "C0", "original" if k == 0 else None)
    draw_pose(axes[1, k], rec[k], "C2", "reconstructed" if k == 0 else None)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 11-13 latent
md(r"""
## 11–13. Exploring the latent space

Structure (2-D projection), generation (interpolation), and sampling from the prior.
""")

code(r"""
# --- 2-D projection of the 32-d latent, colored by weak class ---
with torch.no_grad():
    Z = vae.encode_to_z(torch.tensor(X, device=DEVICE)).cpu().numpy()
try:
    import umap
    emb = umap.UMAP(random_state=SEED).fit_transform(Z)
    proj = "UMAP"
except Exception:
    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, random_state=SEED, init="pca").fit_transform(Z)
    proj = "t-SNE"
plt.figure(figsize=(6, 5))
for c in range(len(CLASSES)):
    m = y == c
    plt.scatter(emb[m, 0], emb[m, 1], s=6, label=CLASSES[c], alpha=0.5)
plt.legend(); plt.title(f"Latent ({proj}) colored by weak pose class"); plt.show()
""")

code(r"""
# --- Latent interpolation between two poses (generation) ---
with torch.no_grad():
    z0 = vae.encode_to_z(torch.tensor(Xte[0:1], device=DEVICE))
    z1 = vae.encode_to_z(torch.tensor(Xte[1:2], device=DEVICE))
    steps = 7
    zs = torch.stack([z0[0] * (1 - t) + z1[0] * t for t in np.linspace(0, 1, steps)])
    gen = vae.dec(zs).cpu().numpy().reshape(steps, 17, 2)
fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 3))
for k in range(steps):
    draw_pose(axes[k], gen[k], "C4")
axes[0].set_title("pose A", fontsize=9); axes[-1].set_title("pose B", fontsize=9)
plt.suptitle("Latent interpolation A → B"); plt.tight_layout(); plt.show()
""")

code(r"""
# --- Sampling novel poses from the prior N(0, I) ---
with torch.no_grad():
    zsamp = torch.randn(8, LATENT, device=DEVICE)
    samp = vae.dec(zsamp).cpu().numpy().reshape(8, 17, 2)
fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
for k in range(8):
    draw_pose(axes[k], samp[k], "C5")
plt.suptitle("Poses sampled from the prior"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- head 1
md(r"""
## 14–15. Head 1 — linear probe classifier on the frozen latent

Transfer learning: freeze the encoder, train a tiny head on the 32-d latent (weak labels).
""")

code(r"""
class PoseHead(nn.Module):
    def __init__(self, latent=LATENT, n_cls=len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(32, n_cls))
    def forward(self, z): return self.net(z)

with torch.no_grad():                      # frozen-encoder latents
    Ztr = vae.encode_to_z(torch.tensor(Xtr, device=DEVICE))
    Zte = vae.encode_to_z(torch.tensor(Xte, device=DEVICE))

# class weights (inverse frequency) for imbalance
cw = torch.tensor([len(ytr) / (ytr == c).sum() for c in range(len(CLASSES))],
                  dtype=torch.float32, device=DEVICE)
head = PoseHead().to(DEVICE)
ho = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
yt = torch.tensor(ytr, device=DEVICE)
head.train()
for ep in range(200):
    ho.zero_grad()
    loss = F.cross_entropy(head(Ztr), yt, weight=cw)
    loss.backward(); ho.step()
head.eval()
with torch.no_grad():
    pred = head(Zte).argmax(1).cpu().numpy()
acc = (pred == yte).mean()
print(f"linear-probe head params: {sum(p.numel() for p in head.parameters()):,}")
print(f"test acc vs weak labels: {acc:.3f}  # <- fill into TL;DR (project's own-data number was 0.971)")
from sklearn.metrics import confusion_matrix
print(confusion_matrix(yte, pred))
""")

# ---------------------------------------------------------------- head 2
md(r"""
## 16–17. Head 2 — prototypical few-shot (add a class from 5 examples)

A prototypical classifier stores one **centroid** per class in latent space (mean of examples) and
classifies by nearest centroid — so a *new* class becomes usable from ~5 examples with **no NN
training**. Demo: build prototypes for the existing classes from 5 shots each, then classify the
test set.
""")

code(r"""
def build_prototypes(Z, labels, k_shot=5):
    protos = {}
    for c in np.unique(labels):
        zc = Z[labels == c][:k_shot]
        protos[c] = zc.mean(0)
    return protos

def proto_classify(Z, protos):
    keys = list(protos); C = np.stack([protos[k] for k in keys])
    d = ((Z[:, None, :] - C[None]) ** 2).sum(-1)      # (N, n_proto)
    return np.array(keys)[d.argmin(1)]

Ztr_np, Zte_np = Ztr.cpu().numpy(), Zte.cpu().numpy()
protos = build_prototypes(Ztr_np, ytr, k_shot=5)
pred_fs = proto_classify(Zte_np, protos)
print(f"few-shot (5 shots/class) test acc: {(pred_fs == yte).mean():.3f}")
print("-> a NEW pose class would be added the same way: 5 latents -> centroid -> instantly usable.")
""")

# ---------------------------------------------------------------- head 3
md(r"""
## 18–19. Head 3 — anomaly / OOD via −log p(z)

Surprise = `−log p(z) = ½·Σ(μ² + σ² − 1 − logσ²)`. High value → the pose sits far from the prior →
unusual. The MVM project used a fixed threshold (14.3) tuned for its 33-landmark/32-latent model;
here the scale differs, so I derive a **data-driven** threshold (95th percentile of the training
distribution).
""")

code(r"""
with torch.no_grad():
    lp_tr = vae.log_prob_z(torch.tensor(Xtr, device=DEVICE)).cpu().numpy()
    lp_all = vae.log_prob_z(torch.tensor(X, device=DEVICE)).cpu().numpy()
thr = np.percentile(lp_tr, 95)
plt.hist(lp_all, bins=60); plt.axvline(thr, color="r", ls="--", label=f"thr(95pct)={thr:.1f}")
plt.xlabel("-log p(z)  (surprise)"); plt.legend(); plt.title("Anomaly score distribution"); plt.show()

order = np.argsort(lp_all)
fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
for k in range(5):
    draw_pose(axes[0, k], P[order[k]],       "C2", "most typical" if k == 0 else None)
    draw_pose(axes[1, k], P[order[-1 - k]],  "C3", "most anomalous" if k == 0 else None)
plt.tight_layout(); plt.show()
print(f"flagged unusual (> thr): {(lp_all > thr).mean():.1%} of poses")
""")

# ---------------------------------------------------------------- continual
md(r"""
## 20. Continual learning (note)

When new classes are added over time, naive retraining forgets old ones. MVM pairs the prototypical
head with an **experience replay buffer** (≤50 latents/class) that mixes past examples into each
update — a lightweight guard against catastrophic forgetting (`core/pose_heads.py`,
`pipeline/self_learn.py`). Here the frozen VAE latent makes this cheap: only centroids/buffers move,
the encoder never retrains.
""")

# ---------------------------------------------------------------- conclusion
md(r"""
## 21. Conclusion

One 32-d β-VAE latent serves **classification, few-shot class addition, and anomaly detection** —
representation learning that pays off three times. The honest caveats: COCO-17 is 2D and
single-instance; weak labels cap the classifier's meaning; the anomaly threshold is data-driven, not
the project's tuned 14.3.

**In production (MVM):** the pose manifold underpins posture understanding and unusual-pose alerts in
the real-time Rust pipeline. → **NB3: weak supervision at scale — auto-labeling 6k COCO poses.**

---
*Data: COCO `person_keypoints_val2017` (CC BY 4.0), keypoints only, no images redistributed.*
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
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb2_posevae.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "|", len(cells), "cells")
