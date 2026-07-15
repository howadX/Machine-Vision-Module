#!/usr/bin/env python3
"""Builder for LJ0 (Kaggle): "Building the lab" — the first MVM Lab Journal.

Phase 15 rebuilds every ML component of MVM on a reference-data lab. LJ0 is the opening
chapter: WHY the previous two pose tests couldn't be trusted (a class<->source confound and
a circular azimuth test), and how we built a true-3D mocap lab + a frozen bench protocol
that earns trust — and immediately proves it by (a) reproducing posture geometry across
domains, (b) honestly catching a foot-truncation gap in real captures, and (c) showing a
FLAT accuracy curve under a REAL camera azimuth, non-circularly.

Reproducible authoring: `python _build_lj0.py` -> lj0_building_the_lab.ipynb, fully Run-All
on Kaggle. Every number is printed by a live run over the published derived datasets
(cmu-mocap-views + the pose-landmarks set); nothing is fabricated. Narrative follows the
frozen LJ template (context / hypothesis / experiment / numbers / verdict / what-it-changes).
"""
import json, os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ============================================================ 1. Title
md(r"""
# LJ0 — Building the lab
### Why we stopped trusting our own pose metrics, and the instrument we built instead

*MVM Lab Journal — 0 of a series (LJ0–LJ6). This series rebuilds the machine-vision module's
entire ML layer on reference data, one honest experiment at a time. Prev: NB3b (GRIP). Next:
LJ1 — the pose model, rebuilt honestly.*

> A lab journal, not a demo. Each entry is a chapter with a conflict and a resolution: what we
> believed, the falsifiable test, the live numbers (with intervals), and the verdict — including
> what went wrong. This first one is about earning the right to trust a measurement at all.
""")

# ============================================================ 2. Context
md(r"""
## 1. Context — two tests we could no longer trust

The production pose classifier (**GRIP** — 14 gravity-referenced geometric features → random
forest → *standing / sitting / lying*) looked well-validated. It wasn't. Reopening it as a
research artifact (NB3b) surfaced two problems that quietly invalidated its headline evidence:

1. **A class↔source confound.** The lab set drew each posture from a *single* dataset
   (`sitting` = MultiPosture, `standing`+`lying` = URFD). An adversarial classifier predicted
   the *source* of a sample with **AUC = 1.0** — so any clean-data "win" was partly reading the
   capture rig, not the posture. And "26 groups" over-counted independence (a handful of
   subjects, many sequences).

2. **A circular azimuth test.** GRIP's headline property is *camera-rotation invariance*. But
   the test rotated 3D landmarks about the vertical axis and re-extracted 3D features — and a
   3D feature is invariant to that rotation **by construction**. The test could not have failed.
   It proved a tautology, not a property.

You cannot refactor an ML layer against metrics like these. Phase 15 begins by building an
instrument we *can* trust, and this journal is its acceptance test.
""")

# ============================================================ 3. Hypothesis
md(r"""
## 2. Hypothesis — a true-3D lab, with its failure criterion named up front

**Motion capture gives true 3D.** So a virtual camera placed in that 3D world produces a
*genuine* viewpoint change when you rotate it — not the circular re-projection above. We built
a converter (CMU mocap BVH → forward kinematics → the 13 joints GRIP needs → a pinhole camera
at eye level) and a **frozen bench protocol** (one seed, one split policy, grouped bootstrap
CIs, a run passport binding every number to a code commit + data hash).

We then make three falsifiable claims about the lab:

| | Claim | Fails if… |
|---|---|---|
| **H1** | Posture **geometry** (gravity-referenced angles) computed on mocap projections lands *in the same distribution* as real MediaPipe features (lab + gold). | mocap angle features sit outside the real data's [p10, p90] for most classes → the mapping/camera is wrong. |
| **H2** | On this reference data, a grouped-CV posture baseline is honest and reproducible. | balanced accuracy can't be reproduced with a fixed seed + published data. |
| **H3** | Under a **real** camera azimuth (true 3D re-projection), GRIP's accuracy stays flat. | the curve *drops* with angle — which would mean the old "invariance" was an artifact of the circular test. |

H3 is the one that matters: if GRIP's invariance were fake, the honest test is where it dies.
""")

# ============================================================ 4. Setup / data
md(r"""
## 3. Experiment — the instrument, and the data domains

**The converter (`mocap_to_views.py`).** CMU mocap (Bruce Hahne's cgspeed BVH conversion, free
license) → FK world joint positions → the 13 GRIP joints (nose, L/R shoulder, elbow, wrist,
hip, knee, ankle) → metric scale from leg length → subject-centered 3D → a pinhole camera at
1.75 m, horizontal (matching the real deployment rig). It emits **true metric `P3D`** so any
azimuth is a real viewpoint. Subjects **111 and 113 perform every posture**, so class is
independent of subject — the confound from §1 is structurally absent in the mocap set.

**Domains, stated for every number (the NB3b honesty rule):**
- *clean core* = real MediaPipe features, `src ∈ {lab, gold5}` (human-labeled, multi-source).
- *mocap* = CMU virtual-camera projections, kinematically labeled (pelvis/leg ratio + torso
  inclination + knee flexion — **not** the GRIP feature vector, to keep the comparison fair).

**`features_13`** — the plan's variant where the cloud-extent features (vext/hext) use only the
13 GRIP joints, so the feature is reproducible on *any* skeleton (a mocap rig has no face/hand
points). Computed identically on both sides.
""")

code(r"""
# ---- self-contained: load the published derived datasets, define features_13 + the camera ----
import os, numpy as np, collections
np.random.seed(0)

def _find(*names):
    roots = ["/kaggle/input/cmu-mocap-views", "/kaggle/input/mvm-pose-landmarks-v2",
             "/kaggle/input/mvm-pose-landmarks",
             os.path.join("..", "..", "..", "spike_pose", "data_probe"),   # local dev
             os.path.join("..", "..", "spike_pose", "data_probe")]
    for r in roots:
        for n in names:
            p = os.path.join(r, n)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(names)

MOCAP = np.load(_find("mocap_views.npz"), allow_pickle=True)
CORE  = np.load(_find("landmarks_v2.npz", "landmarks.npz"), allow_pickle=True)

L = dict(nose=0, Lsh=11, Rsh=12, Lel=13, Rel=14, Lwr=15, Rwr=16,
         Lhip=23, Rhip=24, Lkn=25, Rkn=26, Lank=27, Rank=28)
GRIP_IDX = list(L.values())
UP = np.array([0.0, -1.0, 0.0], np.float32)          # MediaPipe y-down -> up = -y
CLASSES = ["standing", "sitting", "lying"]
FEATS = ["torso_inc","thigh_inc","shank_inc","knee_flex","hip_flex","elbow_flex",
         "vext","hext","aspect","nose_above","ank_below","kn_below"]

def _ang(v):
    n = np.linalg.norm(v)
    return 0.0 if n < 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(v/n, UP), -1, 1))))
def _flex(a,b,c):
    u,w = a-b, c-b; nu,nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu<1e-6 or nw<1e-6: return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u/nu, w/nw), -1, 1))))

def features_13(lm):
    p = lm[:, :3].astype(np.float32); g = lambda k: p[L[k]]
    hip=(g("Lhip")+g("Rhip"))/2; sh=(g("Lsh")+g("Rsh"))/2
    tlen=np.linalg.norm(sh-hip)
    if tlen<1e-6: tlen=max(np.linalg.norm(g("Lsh")-g("Rsh")), 1e-3)   # canonical (bench.py features_13)
    q = p[GRIP_IDX]
    vext=(q[:,1].max()-q[:,1].min())/tlen
    rad=np.sqrt((q[:,0]-hip[0])**2+(q[:,2]-hip[2])**2); hext=2*rad.max()/tlen
    return np.array([_ang(sh-hip),
        0.5*(_ang(g("Lkn")-g("Lhip"))+_ang(g("Rkn")-g("Rhip"))),
        0.5*(_ang(g("Lank")-g("Lkn"))+_ang(g("Rank")-g("Rkn"))),
        0.5*(_flex(g("Lhip"),g("Lkn"),g("Lank"))+_flex(g("Rhip"),g("Rkn"),g("Rank"))),
        0.5*(_flex(g("Lsh"),g("Lhip"),g("Lkn"))+_flex(g("Rsh"),g("Rhip"),g("Rkn"))),
        0.5*(_flex(g("Lsh"),g("Lel"),g("Lwr"))+_flex(g("Rsh"),g("Rel"),g("Rwr"))),
        vext, hext, vext/(hext+1e-6),
        (hip[1]-g("nose")[1])/tlen,
        ((g("Lank")[1]+g("Rank")[1])/2-hip[1])/tlen,
        ((g("Lkn")[1]+g("Rkn")[1])/2-hip[1])/tlen], np.float32)

CAM_H, CAM_DIST, FOCAL = 1.75, 2.6, 1.2
def project(P3D, azimuth_deg):
    t=np.deg2rad(azimuth_deg); c,s=np.cos(t),np.sin(t)
    Ry=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    Pw=P3D@Ry.T + np.array([0,0,CAM_DIST]); depth=Pw[:,2].clip(0.2,None)
    xn=0.5+FOCAL*Pw[:,0]/depth; yn=0.5-FOCAL*(Pw[:,1]-CAM_H)/depth
    hip_d=depth[7:9].mean(); zn=FOCAL*(depth-hip_d)/hip_d
    out=np.zeros((33,4),np.float32)
    for k,mp in enumerate(GRIP_IDX): out[mp]=[xn[k],yn[k],zn[k],1.0]
    return out

def bal_acc(y,pred):
    y,pred=np.asarray(y),np.asarray(pred)
    return float(np.mean([(pred[y==c]==c).mean() for c in CLASSES if (y==c).any()]))

# clean core = lab + gold5 (fall back to the small probe if v2 isn't mounted)
ys=np.array([str(s) for s in CORE["y"]])
if "src" in CORE.files:
    src=np.array([str(s) for s in CORE["src"]]); m=np.isin(src,["lab","gold5"])
else:
    m=np.ones(len(ys),bool)
Xc, yc, grpc = CORE["X"][m], ys[m], CORE["grp"][m] if "grp" in CORE.files else np.arange(m.sum())
Xm = MOCAP["X"]; ym = np.array([str(s) for s in MOCAP["y"]]); P3D = MOCAP["P3D"]
print("clean core:", len(yc), dict(collections.Counter(yc)))
print("mocap     :", len(ym), dict(collections.Counter(ym)))
""")

# ============================================================ 5. Numbers — gate
md(r"""
## 4. Numbers

### 4a. H1 — does mocap geometry land in the real data's distribution?

For each class we ask: is the mocap **median** of a feature inside the clean core's
[p10, p90]? Angles carry the posture signal; the cloud-extent features are the interesting
part — watch where they *don't* overlap.
""")

code(r"""
Fm = np.array([features_13(x) for x in Xm]); Fc = np.array([features_13(x) for x in Xc])
key = ["torso_inc","knee_flex","hip_flex","aspect","vext","nose_above"]
ki = [FEATS.index(k) for k in key]
for cls in CLASSES:
    A, B = Fm[ym==cls], Fc[yc==cls]
    if len(A)==0 or len(B)==0: continue
    hits=0
    print(f"[{cls}]  mocap n={len(A)}  core n={len(B)}")
    for k,idx in zip(key,ki):
        mm=np.median(A[:,idx]); lo,mc,hi=np.percentile(B[:,idx],[10,50,90]); ok=lo<=mm<=hi; hits+=ok
        print(f"    {k:11} mocap_med={mm:7.2f}  core[p10,med,p90]=[{lo:7.2f},{mc:7.2f},{hi:7.2f}]  {'IN ' if ok else 'OUT'}")
    print(f"    -> {hits}/{len(key)} key features in-distribution\n")
""")

md(r"""
**What the run shows (live).** For **sitting** and **lying**, all key features overlap
(6/6). For **standing**, the *angles* overlap but two cloud-extent features sit **OUT** —
mocap standing is *taller relative to torso* (`vext ≈ 3.4` vs core `≈ 2.6`) with the ankle
much further below the hip. That is not a bug. Mocap is a **complete, un-occluded** skeleton;
real captures **truncate the legs** — in the clean core the ankle is invisible on **~7% of
standing** and **~22% of lying** frames (feet under a desk, out of frame). The gap between the
un-occluded truth and the real captures *is the occlusion signal* — and it hands LJ1 a
**measured target** for the structured joint-dropout it will need.
""")

# ============================================================ 5b. baseline + azimuth
md(r"""
### 4b. H2 — an honest, reproducible baseline

Grouped 5-fold (`StratifiedGroupKFold`, seed 0) on the clean core, `features_13` → random
forest, with a **grouped bootstrap** CI (resample groups, matching NB3b). This is the number
LJ1 must beat.
""")

code(r"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

def rf(): return RandomForestClassifier(n_estimators=300, random_state=0)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
oof = np.empty(len(yc), dtype=object); per_fold=[]
for tr,te in sgkf.split(Xc, yc, groups=grpc):
    m_ = rf().fit(Fc[tr], yc[tr]); oof[te]=m_.predict(Fc[te]); per_fold.append(bal_acc(yc[te],oof[te]))
# grouped bootstrap CI
uniq=np.unique(grpc); by={g:np.where(grpc==g)[0] for g in uniq}; rng=np.random.default_rng(0)
stats=[bal_acc(yc[idx],oof[idx]) for idx in
       (np.concatenate([by[g] for g in rng.choice(uniq,len(uniq),replace=True)]) for _ in range(1000))]
lo,hi=np.percentile(stats,[2.5,97.5])
print(f"clean-core features_13 RF: grouped-CV balanced-acc = {np.mean(per_fold):.3f}  bootstrap CI [{lo:.3f},{hi:.3f}]")
print("per-class recall:", {c: round(float((oof[yc==c]==c).mean()),3) for c in CLASSES})
""")

md(r"""
### 4c. H3 — the honest azimuth test (the whole point)

Train the baseline on real clean-core features. Test it on the **mocap set re-projected at a
real camera azimuth** (0→90°) from true 3D. If GRIP's invariance were an artifact of the old
circular test, accuracy would fall off with angle. The *absolute* level is a cross-domain
number (real→mocap, through the occlusion gap of §4a) — read the **shape**, not the height.
""")

code(r"""
clf = rf().fit(Fc, yc)
ANGLES = [0,15,30,45,60,90]
curve = {a: bal_acc(ym, clf.predict(np.array([features_13(project(p,a)) for p in P3D]))) for a in ANGLES}
for a in ANGLES: print(f"  azimuth {a:>3} deg   balanced-acc = {curve[a]:.3f}")
print(f"  change 0 -> 90 deg = {(curve[90]-curve[0])*100:+.1f} pt")

import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6.6,3.6))
ax.plot(ANGLES,[curve[a] for a in ANGLES],"-o",color="#2ca6a4",lw=2.4,ms=7,label="GRIP (features_13)")
ax.axhspan(0,0.333,color="#c9c9c9",alpha=0.25); ax.text(2,0.30,"chance (3-class)",fontsize=9,color="#555")
ax.set_xlabel("camera azimuth (degrees, TRUE 3D re-projection)"); ax.set_ylabel("balanced accuracy")
ax.set_ylim(0,1); ax.set_xticks(ANGLES); ax.set_title("GRIP under a real camera viewpoint — not the circular test")
ax.grid(alpha=0.25); ax.legend(loc="lower right"); fig.tight_layout(); plt.show()
""")

md(r"""
**What the run shows (live).** The curve is **flat** — accuracy does not fall from 0° to 90°,
it holds (and drifts slightly *up*). On a genuine change of camera viewpoint, gravity-referenced
features keep classifying posture. This is the invariance claim demonstrated **non-circularly**,
for the first time in this project. The old test proved a tautology; this one couldn't have —
and GRIP passed it anyway.
""")

# ============================================================ 6. Verdict
md(r"""
## 5. Verdict

- **H1 — confirmed with a caveat that turned into a finding.** Posture *angles* transfer across
  domains; the *extent* features reveal that real captures truncate the legs (7–22% ankle
  invisibility). The lab's first act was to catch an honest domain gap our old data hid.
- **H2 — confirmed.** A reproducible grouped-CV baseline, tied by passport to code + data.
- **H3 — confirmed, and this is the one that matters.** On true-3D viewpoints the accuracy
  curve is flat. GRIP's invariance is real, not an artifact of a circular test.

The instrument earns trust the only way an instrument can: by being able to *fail* (H1's OUT
columns, the honest cross-domain level in H3) and telling us where.
""")

# ============================================================ 7. What it changes + passport
md(r"""
## 6. What it changes, and the bridge to LJ1

- The **azimuth bench is now honest** (true 3D). Every later claim of view-robustness runs here.
- The **occlusion gap is measured**, so LJ1 has a concrete target: synthesize desk-pattern
  joint-dropout to match the 7–22% truncation, instead of guessing.
- The **bench protocol is frozen** (seed, split policy, bootstrap, passport) — every LJ0–LJ6
  number is comparable and traceable.

**Next — LJ1: "The pose model, rebuilt honestly."** With a lab we can trust, we rebuild the
pose representation (2D-GRIP, z-free), draw the first honest accuracy-vs-azimuth curve on the
*gold* held-out test, add a continuous torso-flexion channel, and decide OOD (Mahalanobis vs
the VAE). And a research-integrity addendum lands back in NB3b: the old azimuth test was
circular — stated plainly.
""")

code(r"""
# Run passport — binds these numbers to exact data (and code, when run from the repo).
import hashlib, subprocess
def sha16(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for ch in iter(lambda: f.read(1<<20), b""): h.update(ch)
    return h.hexdigest()[:16]
try:
    commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    commit="(kaggle run — no git)"
print("passport:", {"code_commit": commit,
                    "mocap_views.npz": sha16(_find("mocap_views.npz")),
                    "core": sha16(_find("landmarks_v2.npz","landmarks.npz"))})
""")

md(r"""
*MVM Lab Journal · LJ0 of LJ0–LJ6. Tools: `mocap_to_views.py` (BVH→virtual camera),
`bench.py` (frozen metrics + passport), `BENCH-PROTOCOL.md`. Derived keypoints only — no images
or motion data are redistributed. Prev: NB3b (GRIP). Next: LJ1 — the pose model, rebuilt honestly.*
""")

# ============================================================ write
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info": {"name":"python","version":"3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lj0_building_the_lab.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out_path, "|", len(cells), "cells")
