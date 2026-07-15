#!/usr/bin/env python3
"""LJ1 experiment 4 — OOD gate: can class-conditional Mahalanobis replace the 195 KB VAE?

Denouden (2018): reconstruction-error OOD detectors have a blind spot — an OOD sample can
reconstruct well and slip through. A distance-in-feature-space score (Mahalanobis) does not.
We test three OOD scorers, all fit on the SAME in-distribution set (clean core), so the
comparison is method-vs-method, not data-vs-data:

  A. Mahalanobis   — class-conditional mean+cov on the 14 GRIP features (2D, ship rep);
                     score = min over classes of Mahalanobis distance. ~a few KB.
  B. AE-recon      — a small autoencoder on the landmarks; score = reconstruction MSE.
                     Stands in for the VAE family (recon-error), trained on the same ID.
  C. vote-entropy  — the shipped RF's soft-vote entropy. Free (already have the forest).

OOD sets (real failure modes): degraded (heavy landmark noise), scrambled (valid coords /
invalid configuration), uniform-random. Metric: AUROC separating held-out ID from each OOD.
Verdict: if Mahalanobis AUROC >= AE-recon across OOD types, drop the VAE (diet win, 15.5).

Run: ../venv/bin/python3 spike_pose/lj1_ood.py
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mocap_sanity_gate import features_13, L, GRIP_IDX
import bench

rng = np.random.default_rng(bench.SEED)


def feat_2d(lm):
    l = lm.copy(); l[:, 2] = 0.0; return features_13(l)
def raw2d(lm):                                   # hip-centered, torso-scaled 13-joint (x,y) -> 26-d
    p = lm[:, :3].astype(np.float32)
    hip = (p[L["Lhip"]] + p[L["Rhip"]]) / 2; sh = (p[L["Lsh"]] + p[L["Rsh"]]) / 2
    q = (p[GRIP_IDX] - hip) / (np.linalg.norm(sh - hip) + 1e-6)
    return q[:, :2].reshape(-1)


# ---------- OOD generators (operate on held-out ID landmark tensors) ----------
def ood_degraded(X):
    out = X.copy().astype(np.float32); out[:, :, :3] += rng.normal(0, 0.15, out[:, :, :3].shape); return out
def ood_scrambled(X):
    out = X.copy()
    for i in range(len(out)): out[i, GRIP_IDX] = out[i, rng.permutation(GRIP_IDX)]
    return out
def ood_random(X):
    out = np.zeros_like(X, np.float32); out[:, GRIP_IDX, :2] = rng.uniform(0, 1, (len(X), 13, 2))
    out[:, GRIP_IDX, 3] = 1.0; return out


# ---------- scorers ----------
class Mahalanobis:
    def fit(self, F, y):
        self.stats = {}
        for c in bench.CLASSES:
            Fc = F[y == c]; mu = Fc.mean(0)
            cov = np.cov(Fc, rowvar=False) + 1e-3 * np.eye(F.shape[1])   # ridge for stability
            self.stats[c] = (mu, np.linalg.inv(cov))
        return self
    def score(self, F):                          # min Mahalanobis dist to any class (high = OOD)
        d = np.stack([np.einsum("ij,jk,ik->i", F - mu, P, F - mu) for mu, P in self.stats.values()], 1)
        return np.sqrt(d.min(1))

def train_ae(Xr, dim, epochs=250):
    import torch, torch.nn as nn
    torch.manual_seed(bench.SEED)
    net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(),
                        nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, dim))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); T = torch.tensor(Xr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(); loss = ((net(T) - T) ** 2).mean(); loss.backward(); opt.step()
    net.eval()
    return lambda A: ((net(torch.tensor(A, dtype=torch.float32)) - torch.tensor(A, dtype=torch.float32))
                      ** 2).mean(1).detach().numpy()


def main():
    d = np.load(os.path.join(HERE, "data_probe", "landmarks_v2.npz"), allow_pickle=True)
    X = d["X"]; y = np.array([str(s) for s in d["y"]]); src = np.array([str(s) for s in d["src"]])
    grp = d["grp"]; core = np.isin(src, ["lab", "gold5"])
    Xc, yc, gc = X[core], y[core], grp[core]
    # group-disjoint ID fit/test split
    ug = np.unique(gc); rng.shuffle(ug); fit_g = set(ug[: int(0.8 * len(ug))])
    fit = np.array([g in fit_g for g in gc]); te = ~fit
    print(f"ID: fit={fit.sum()} test={te.sum()} (group-disjoint)")

    Ffit = np.array([feat_2d(x) for x in Xc[fit]]); Fte = np.array([feat_2d(x) for x in Xc[te]])
    Rfit = np.array([raw2d(x) for x in Xc[fit]]);   Rte = np.array([raw2d(x) for x in Xc[te]])

    maha = Mahalanobis().fit(Ffit, yc[fit])
    ae = train_ae(Rfit, Rfit.shape[1])
    forest = RandomForestClassifier(n_estimators=300, random_state=bench.SEED).fit(Ffit, yc[fit])
    def vote_entropy(F):
        P = forest.predict_proba(F); return -(P * np.log(P + 1e-9)).sum(1)

    id_scores = {"Mahalanobis": maha.score(Fte), "AE-recon": ae(Rte), "vote-entropy": vote_entropy(Fte)}
    print("\nAUROC (ID-test vs OOD; higher = better OOD separation):")
    print(f"  {'OOD type':12} {'Mahalanobis':>12} {'AE-recon':>10} {'vote-entropy':>13}")
    Xte = Xc[te]
    ood_sets = [("degraded", ood_degraded(Xte)), ("scrambled", ood_scrambled(Xte)), ("random", ood_random(Xte))]
    # real near-manifold OOD: MPPose forced onto dog/cat images (Denouden's actual case)
    ap = os.path.join(HERE, "data_probe", "animal_ood_poses.npz")
    if os.path.exists(ap):
        ood_sets.append(("animals (real)", np.load(ap, allow_pickle=True)["X"]))
    for name, Xo in ood_sets:
        Fo = np.array([feat_2d(x) for x in Xo]); Ro = np.array([raw2d(x) for x in Xo])
        oo = {"Mahalanobis": maha.score(Fo), "AE-recon": ae(Ro), "vote-entropy": vote_entropy(Fo)}
        aucs = {k: roc_auc_score(np.r_[np.zeros(len(Fte)), np.ones(len(Fo))], np.r_[id_scores[k], oo[k]])
                for k in id_scores}
        print(f"  {name:14} {aucs['Mahalanobis']:10.3f} {aucs['AE-recon']:10.3f} {aucs['vote-entropy']:13.3f}")


if __name__ == "__main__":
    main()
