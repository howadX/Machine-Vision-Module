#!/usr/bin/env python3
"""Builder for NB3b (Kaggle): GRIP (Gravity-Referenced Invariant Pose features).

A research notebook, not a demo: a clear question, honest hypotheses (including the
ones the data REFUTED), grounded charts, and honest limits. Reproducible authoring:
run `python _build_nb3b.py` -> nb3b_grip.ipynb. The notebook loads the project's
published keypoint dataset (`mvm-pose-landmarks`, keypoints only, derived from
URFD + Zenodo MultiPosture) and is fully Run-All on Kaggle. All numbers printed and
plotted come from that live run; nothing is fabricated. Charts follow one locked,
colorblind-validated palette (GRIP=aqua, raw=orange, VAE=blue).
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


# ================================================================ 1. Title
md(r"""
# GRIP: Gravity-Referenced Invariant Pose Features
### A pose classifier, and the honest story of what it does and doesn't buy

*MVM ML Journey — Notebook 3b · prev: NB3 (weak supervision) · NB2 (pose manifold) · next: NB4 (teardown)*

> **GRIP** = **G**ravity-**R**eferenced **I**nvariant **P**ose features: fourteen geometric quantities
> measured against the gravity vector — inclinations, joint flexions, extent ratios — that classify a body
> as **standing / sitting / lying**. (In the project's spike notes this was labelled "method C", an A/B/C
> enumeration artifact; the descriptive name is used everywhere now.)

**How this notebook is framed.** This is production code, but I reopened it as a *research question*, not a
victory lap. I reached for gravity-referenced features because they felt like the **more principled
representation** for a fixed home camera — physical priors, interpretable, dependency-free to ship in Rust.
What that choice would *cost in accuracy* was, honestly, an open question. So I measured it under honest
(grouped) cross-validation, and two things happened that this notebook is built around:

1. **On clean, fixed-viewpoint data the features are *not* the most accurate input — the raw 33 landmarks
   are.** The 14 features are a lossy summary and give up a few points. I report that plainly.
2. **The features are invariant to camera rotation, where the raw coordinates collapse to near-chance.**
   That robustness wasn't the goal — it was a property I only appreciated after measuring, and it is exactly
   the failure mode a home camera cannot escape (people turn away from it). *That* is why GRIP ships.

So the arc is: a principled design choice → an honest measurement of its cost → a robustness payoff
discovered along the way that happens to match the deployment. Not "my features beat everything."

> **Reproducibility note — read this.** Earlier project docs quoted **~0.959** for GRIP and a **"12 KB"**
> forest. *Neither reproduces here.* On this public data with the shipped RF params the honest grouped-CV is
> **≈0.93**, and the serialized forest is **~270 KB** (873 KB in production, which trains on augmented data).
> The lower, reproducible numbers are the ones this notebook stands behind. An honest metric beats a
> flattering one — and finding that gap was the point.

**Skills shown:** feature engineering with physical invariants · grouped cross-validation ·
cluster-bootstrap confidence intervals · adversarial validation of a confound · probability calibration (ECE)
· grouped hyperparameter search · honest trade-off analysis (reporting where my method *loses*) ·
dependency-free production export.
""")

# ================================================================ 2. Research framing
md(r"""
## The question, and the hypotheses I set out to test

**Deployment.** One fixed, eye-level home camera. People move *and turn* relative to it. The signal we
need is coarse posture (standing / sitting / lying) for a smart-home / safety context — cheap enough to run
in Rust at sub-millisecond CPU, with no heavy runtime.

**Central question.** *Is a hand-engineered, gravity-referenced representation the right input for this
deployment — and what does choosing it cost against simply throwing the raw landmarks at the model?*

| | Hypothesis | Verdict (measured below) |
|---|---|---|
| **H1** | The 14 features are at least as accurate as raw coordinates / a VAE latent on clean data | ◐ **not separable** — raw leads on the point estimate, but 95% CIs overlap |
| **H2** | Designed invariance holds under camera rotation, where a raw representation breaks | ✅ **confirmed** — GRIP flat; raw collapses toward chance |
| **H3** | Honest grouped CV (unseen *people*) is lower than the ungrouped figure long quoted | ✅ **confirmed** — 0.959 did not reproduce → ≈0.93 |
| **H4** | For a camera people turn away from, paying clean-accuracy for invariance is the right call | ⚖️ an engineering judgement, argued from H1+H2 |

The interesting notebook is the one where the tidy hypothesis (*my* features are the most accurate) **does
not cleanly hold, and I say so** — where I then bootstrap the gap to see whether it is even real — and where
the reason to ship the model turns out to be H2, a property I did not originally reach for.

**Data.** The project's own **keypoints-only** pose set (URFD falls + Zenodo MultiPosture): 33×4
MediaPipe-style landmarks per frame, three classes, grouped by person/sequence. No images — safe to publish,
fast to Run-All. Its structure also hides a confound, which we confront first.
""")

# ================================================================ 3. Load
code(r"""
import os, glob, collections, numpy as np, pandas as pd
SEED = 0
np.random.seed(SEED)

# Published derivative of URFD + Zenodo MultiPosture — keypoints only, CC BY 4.0.
cands = glob.glob("/kaggle/input/**/landmarks.npz", recursive=True)
if not cands:
    mounted = os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "NONE"
    raise FileNotFoundError(f"landmarks.npz not found. Attach dataset 'howadx/mvm-pose-landmarks'. "
                            f"Mounted under /kaggle/input: {mounted}")
NPZ = cands[0]; print("loading:", NPZ)
d = np.load(NPZ, allow_pickle=True)
X, y_str, src, grp = d["X"], d["y"], d["src"], d["grp"]   # X: (N,33,4)=x,y,z,vis
CLASSES = ["lying", "sitting", "standing"]
y = np.array([CLASSES.index(s) for s in y_str])
print("poses:", X.shape, "| classes:", dict(collections.Counter(y_str.tolist())))
print("groups (people/sequences):", len(set(grp.tolist())),
      "| sources:", dict(collections.Counter(src.tolist())))

METRICS = {"n_poses": int(X.shape[0]), "n_groups": len(set(grp.tolist()))}   # harvested by kctl fetch
""")

# ================================================================ 4. viz setup (design system)
code(r"""
# --- One locked visual language for every chart (dataviz design system) ---
# Palette assigned BY ENTITY (same colour everywhere) and CVD-validated:
#   GRIP=aqua  raw coords=orange  VAE=blue.  Pose classes get their own trio.
import matplotlib as mpl, matplotlib.pyplot as plt
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"
GRIP="#1baf7a"; RAW="#eb6834"; VAE="#2a78d6"
CLS={"lying":"#2a78d6","sitting":"#1baf7a","standing":"#eb6834"}
SEQ=["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]
mpl.rcParams.update({
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "font.size":11,"text.color":INK,"axes.labelcolor":INK2,"axes.edgecolor":"#c3c2b7",
    "xtick.color":MUTED,"ytick.color":MUTED,"axes.titlecolor":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False})
def style(ax, ygrid=True):
    ax.grid(axis="y" if ygrid else "x", alpha=1.0)
    ax.grid(axis="x" if ygrid else "y", visible=False)
    ax.set_axisbelow(True)
    for s in ("left","bottom"): ax.spines[s].set_linewidth(1.0)
print("viz ready")
""")

# ================================================================ 5. Confound (md)
md(r"""
## First, an honest look at the data: a structural confound

Before any model, look at *where each class comes from*. The chart below is uncomfortable on purpose:

- every **sitting** frame is from **MultiPosture**,
- every **lying** and **standing** frame is from **URFD**,
- **no pose is seen across both datasets.**

That means class is almost perfectly aligned with *dataset identity*. A flexible model — especially one fed
the raw coordinates — can score partly by learning "which capture rig is this" (camera intrinsics,
normalization quirks) rather than "which posture is this." Grouped CV controls for unseen **people**, but it
**cannot** disentangle pose from dataset here, because a class lives entirely in one dataset. I can't fix
this with the data I have, so I do the next best thing: **name it, and lean on a test that a dataset-identity
shortcut cannot pass** — the azimuth-rotation test later, which destroys exactly the viewpoint-specific cues
a raw model would exploit. Keep this confound in mind when raw coordinates "win" in a few cells.
""")

# ================================================================ 5b. V1 confound (code)
code(r"""
ct = pd.crosstab(pd.Series(src, name="src"), pd.Series(y_str, name="cls"))[CLASSES]
METRICS["class_by_source"] = {s: {c: int(ct.loc[s, c]) for c in CLASSES} for s in ct.index}

fig, ax = plt.subplots(figsize=(7.4, 3.2))
srcs = list(ct.index); ypos = np.arange(len(srcs)); left = np.zeros(len(srcs))
for cls in CLASSES:
    vals = ct[cls].values
    ax.barh(ypos, vals, left=left, color=CLS[cls], height=0.62, label=cls, edgecolor=SURF, linewidth=2)
    for i, (v, l) in enumerate(zip(vals, left)):
        if v > 0:
            ax.text(l + v/2, i, f"{cls}\n{v}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
    left += vals
ax.set_yticks(ypos); ax.set_yticklabels([s.upper() for s in srcs]); ax.invert_yaxis()
ax.set_xlabel("pose samples"); style(ax, ygrid=False); ax.grid(axis="x", alpha=1.0)
ax.set_title("Each pose class lives in exactly one source — a structural confound",
             fontweight="bold", fontsize=12, loc="left")
fig.text(0.01, -0.03, "'sitting' = 100% MultiPosture   ·   'lying' + 'standing' = 100% URFD   ·   "
         "no pose is seen across both datasets.", fontsize=9, color=INK2)
plt.tight_layout(); plt.show()
""")

# ================================================================ 6. Features (md)
md(r"""
## The 14 features

Straight from the production spike (`spike_pose/spike_feats.py`). MediaPipe convention: **y grows
downward**, so "up" = `-y`. Every quantity below is invariant to rotation about the vertical axis — an angle
to gravity, a joint flexion (a 3-D angle is absolute), or a ratio of extents.

| # | Feature | Measures | Why azimuth-invariant |
|---|---------|----------|-----------------------|
| 1 | `torso_inc` | torso angle to vertical | gravity ⟂ azimuth |
| 2–4 | `thigh_inc`, `thigh_asym`, `shank_inc` | limb inclinations | same |
| 5–8 | `knee_flex`, `knee_asym`, `hip_flex`, `elbow_flex` | joint angles | a 3-D angle is absolute |
| 9–11 | `vext`, `hext`, `aspect` | point-cloud shape | ratios of extents |
| 12–14 | `nose_above`, `ank_below`, `kn_below` | vertical placement vs. hips | signed, in torso units |
""")

# ================================================================ 6b. Features (code)
code(r"""
UP = np.array([0.0, -1.0, 0.0], np.float32)   # MediaPipe: y down -> up is -y
L = dict(nose=0, Lsh=11, Rsh=12, Lel=13, Rel=14, Lwr=15, Rwr=16,
         Lhip=23, Rhip=24, Lkn=25, Rkn=26, Lank=27, Rank=28)

def _ang_to_vert(v):
    n = np.linalg.norm(v)
    return 0.0 if n < 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(v/n, UP), -1, 1))))

def _flex(a, b, c):                      # angle a-b-c at vertex b
    u, w = a - b, c - b
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-6 or nw < 1e-6: return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u/nu, w/nw), -1, 1))))

FEAT_NAMES = ["torso_inc","thigh_inc","thigh_asym","shank_inc","knee_flex","knee_asym",
              "hip_flex","elbow_flex","vext","hext","aspect","nose_above","ank_below","kn_below"]

def features(lm):
    '''33x4 (x,y,z,vis) -> 14 azimuth-invariant, gravity-preserving features.'''
    p = lm[:, :3].astype(np.float32); g = lambda k: p[L[k]]
    hip = (g("Lhip") + g("Rhip")) / 2; sh = (g("Lsh") + g("Rsh")) / 2
    torso = sh - hip; tlen = np.linalg.norm(torso)
    if tlen < 1e-6: tlen = max(np.linalg.norm(g("Lsh") - g("Rsh")), 1e-3)
    thigh_L = _ang_to_vert(g("Lkn") - g("Lhip")); thigh_R = _ang_to_vert(g("Rkn") - g("Rhip"))
    shank_L = _ang_to_vert(g("Lank") - g("Lkn")); shank_R = _ang_to_vert(g("Rank") - g("Rkn"))
    knee_L = _flex(g("Lhip"), g("Lkn"), g("Lank")); knee_R = _flex(g("Rhip"), g("Rkn"), g("Rank"))
    hip_L  = _flex(g("Lsh"),  g("Lhip"), g("Lkn"));  hip_R  = _flex(g("Rsh"),  g("Rhip"), g("Rkn"))
    elb_L  = _flex(g("Lsh"),  g("Lel"),  g("Lwr"));  elb_R  = _flex(g("Rsh"),  g("Rel"),  g("Rwr"))
    ys = p[:, 1]; vext = (ys.max() - ys.min()) / tlen
    rad = np.sqrt((p[:, 0] - hip[0])**2 + (p[:, 2] - hip[2])**2)
    hext = 2 * rad.max() / tlen; aspect = vext / (hext + 1e-6)
    nose_above = (hip[1] - g("nose")[1]) / tlen
    ank_below  = ((g("Lank")[1] + g("Rank")[1]) / 2 - hip[1]) / tlen
    kn_below   = ((g("Lkn")[1] + g("Rkn")[1]) / 2 - hip[1]) / tlen
    return np.array([_ang_to_vert(torso), (thigh_L+thigh_R)/2, abs(thigh_L-thigh_R), (shank_L+shank_R)/2,
                     (knee_L+knee_R)/2, abs(knee_L-knee_R), (hip_L+hip_R)/2, (elb_L+elb_R)/2,
                     vext, hext, aspect, nose_above, ank_below, kn_below], np.float32)

def raw_coords(lm):
    '''VAE-style baseline: center on hip, scale by torso length, flatten 33x3 -> 99-D.'''
    p = lm[:, :3].astype(np.float32)
    hip = (p[L["Lhip"]] + p[L["Rhip"]]) / 2; sh = (p[L["Lsh"]] + p[L["Rsh"]]) / 2
    return ((p - hip) / (np.linalg.norm(sh - hip) + 1e-6)).reshape(-1)

F = np.array([features(x) for x in X], np.float32)          # (N, 14)
C = np.array([raw_coords(x) for x in X], np.float32)        # (N, 99)
Fdf = pd.DataFrame(F, columns=FEAT_NAMES)
print("feature matrix:", Fdf.shape, "| raw matrix:", C.shape)
Fdf.head(3)
""")

# ================================================================ 7. Separability (md)
md(r"""
## Do the classes separate by eye?

The three *highest-signal* features (by RandomForest importance, next cell) tell most of the story:
`aspect` (standing is tall-and-thin), `ank_below` (where the ankles sit relative to the hips), and
`torso_inc` (lying tips the torso off vertical). Overlapping-but-distinct histograms are exactly what a
shallow model needs — and a good sign the signal is real, not memorized.
""")

# ================================================================ 7b. V2 separability (code)
code(r"""
feats = ["aspect", "ank_below", "torso_inc"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6)); handles = None
for ax, ft in zip(axes, feats):
    for name in CLASSES:
        ax.hist(Fdf[ft][y == CLASSES.index(name)], bins=28, alpha=0.62, color=CLS[name],
                label=name, edgecolor="none")
    style(ax); ax.set_title(ft, fontsize=11); ax.set_ylabel("count" if ft == feats[0] else "")
    if handles is None: handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, fontsize=10)
fig.suptitle("The classes separate by eye — the three highest-signal features",
             fontweight="bold", fontsize=12, x=0.012, ha="left", y=1.08)
plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.show()
""")

# ================================================================ 7c. V3 importance (md)
md(r"""
## Which of the 14 actually carry signal?

A quick RandomForest importance ranking. Two honest takeaways: (1) the signal concentrates in ~3 features —
`aspect`, `ank_below`, `torso_inc` — so 14 is already generous; (2) `knee_flex`, which older project notes
name as a key discriminator, is in fact near the **bottom**. Measuring beats remembering.
""")

# ================================================================ 7d. V3 importance (code)
code(r"""
from sklearn.ensemble import RandomForestClassifier
imp = RandomForestClassifier(n_estimators=120, max_depth=16, random_state=SEED, n_jobs=-1).fit(F, y).feature_importances_
order = np.argsort(imp); cmax = imp.max()
METRICS["feat_importance"] = {FEAT_NAMES[i]: round(float(imp[i]), 4) for i in order[::-1]}

fig, ax = plt.subplots(figsize=(7.2, 4.6)); yp = np.arange(len(FEAT_NAMES))
ax.barh(yp, imp[order], color=[SEQ[min(6, 2 + int(4*imp[i]/cmax))] for i in order],
        height=0.7, edgecolor=SURF, linewidth=1.5)
for i, v in enumerate(imp[order]): ax.text(v + 0.004, i, f"{v:.02f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(yp); ax.set_yticklabels([FEAT_NAMES[i] for i in order], fontsize=9)
ax.set_xlabel("RandomForest feature importance"); style(ax, ygrid=False); ax.grid(axis="x", alpha=1.0)
ax.set_title("Three features carry most of the signal (14 -> ~3)", fontweight="bold", fontsize=12, loc="left")
plt.tight_layout(); plt.show()
""")

# ================================================================ 8. Grouped CV / Finding A (md)
md(r"""
## Grouped cross-validation — the honest number, and where H1 fails

A random split lets frames of the *same person* fall in both train and test, so the model scores itself
partly on memorized identity. **StratifiedGroupKFold** keeps every person's frames on one side. This is the
number that predicts performance on people the model has never seen — the only one worth putting on a résumé.

Three inputs, same forest (production params), same folds: **14 GRIP features** vs. **raw 99-D coordinates**
(the VAE-style input) vs. **VAE latent (32-D)** approximated by PCA→32 (`sklearn` doesn't ship our tiny VAE;
the qualitative gap is the point).

**On the point estimate, H1 does not hold.** Raw coordinates score **highest**; GRIP and the VAE latent land
a few points lower. On clean, single-viewpoint data, 33 landmarks simply carry more signal than 14 summaries
— *and*, per the confound above, the raw model may be partly rewarded for reading dataset identity. Whether
that few-point gap is *real* or just noise at 26 groups is the very next question — a bootstrap answers it. If
clean, fixed-camera accuracy were the only goal, you would ship the raw coordinates. Hold that thought.
""")

# ================================================================ 8b. grouped CV + V4 (code)
code(r"""
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
rf = lambda: RandomForestClassifier(n_estimators=120, max_depth=16, random_state=SEED, n_jobs=-1)

runs = [("raw coords\n(99-D)",  RAW,  "raw_coords", cross_val_score(rf(), C, y, groups=grp, cv=sgkf, scoring="accuracy")),
        ("GRIP\n(14 feats)",     GRIP, "grip",       cross_val_score(rf(), F, y, groups=grp, cv=sgkf, scoring="accuracy")),
        ("VAE latent\n(~PCA32)", VAE,  "vae_pca32",
         cross_val_score(make_pipeline(StandardScaler(), PCA(32, random_state=SEED), rf()), C, y,
                         groups=grp, cv=sgkf, scoring="accuracy"))]
METRICS["grouped_cv"] = {key: [round(float(sc.mean()), 4), round(float(sc.std()), 4)] for _, _, key, sc in runs}
for nm, _, key, sc in runs:
    print(f"  {nm.splitlines()[0]:16} grouped-CV = {sc.mean():.3f} +/- {sc.std():.3f}")

fig, ax = plt.subplots(figsize=(7.4, 4.4))
for i, (nm, col, key, sc) in enumerate(runs):
    m, s = sc.mean(), sc.std()
    ax.bar(i, m, width=0.6, color=col, alpha=0.9, edgecolor=SURF, linewidth=2, zorder=2)
    ax.errorbar(i, m, yerr=s, color=INK2, capsize=5, lw=1.4, zorder=3)
    jit = (np.random.RandomState(i).rand(len(sc)) - 0.5) * 0.22
    ax.scatter(i + jit, sc, color="white", edgecolor=col, s=42, lw=1.6, zorder=4)
    ax.text(i, 1.012, f"{m:.3f}", ha="center", fontsize=12, fontweight="bold", color=col)
ax.set_xticks(range(len(runs))); ax.set_xticklabels([r[0] for r in runs]); ax.set_ylim(0.80, 1.03)
ax.set_ylabel("grouped-CV accuracy (unseen people)"); style(ax)
gap = (runs[0][3].mean() - runs[1][3].mean()) * 100
ax.set_title(f"The honest surprise: raw coords win on clean data (~{gap:.0f} pts) — dots = 5 folds",
             fontweight="bold", fontsize=11.5, loc="left")
plt.tight_layout(); plt.show()
""")

# ================================================================ 8c. bootstrap CI / Finding A-rigor (md)
md(r"""
## Is that ~4-point win real? A bootstrap over the groups

Four points *sounds* decisive — but with only **26 groups** the fold-to-fold spread is wide, so a mean (or a
std over just five folds) can mislead. To put an honest interval on each score I resample the **groups**
(people / sequences) with replacement 1000× and recompute accuracy on the pooled out-of-fold predictions — a
*cluster* bootstrap that respects the grouped structure instead of trusting five noisy fold estimates. The
forest plot below is each input's accuracy with its 95% CI. Read the **overlap**: where the intervals touch,
"raw wins" is a soft claim — exactly what the confound above warned we should expect.
""")

# ================================================================ 8d. bootstrap CI (code)
code(r"""
from sklearn.model_selection import cross_val_predict
grp_arr = np.array(grp)
idx_by_group = [np.where(grp_arr == g)[0] for g in np.unique(grp_arr)]
G = len(idx_by_group)

def cluster_bootstrap_ci(correct, n=1000, seed=SEED):
    '''Resample GROUPS with replacement; recompute pooled accuracy. Returns (2.5%, 97.5%).'''
    rng = np.random.RandomState(seed); out = np.empty(n)
    for b in range(n):
        sel = np.concatenate([idx_by_group[k] for k in rng.randint(0, G, G)])
        out[b] = correct[sel].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

boot_inputs = [("raw coords", "raw_coords", RAW, rf(), C),
               ("GRIP",       "grip",       GRIP, rf(), F),
               ("VAE latent", "vae_pca32",  VAE, make_pipeline(StandardScaler(), PCA(32, random_state=SEED), rf()), C)]
METRICS["grouped_cv_oof"] = {}; METRICS["grouped_cv_ci"] = {}; boot_rows = []
for nm, key, col, est, Xin in boot_inputs:
    correct = (cross_val_predict(est, Xin, y, groups=grp, cv=sgkf) == y).astype(float)
    acc = float(correct.mean()); lo, hi = cluster_bootstrap_ci(correct)
    METRICS["grouped_cv_oof"][key] = round(acc, 4); METRICS["grouped_cv_ci"][key] = [round(lo, 4), round(hi, 4)]
    boot_rows.append((nm, col, acc, lo, hi))
    print(f"  {nm:12} OOF acc = {acc:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")

raw_lo = METRICS["grouped_cv_ci"]["raw_coords"][0]; grip_hi = METRICS["grouped_cv_ci"]["grip"][1]
gap_pts = (METRICS["grouped_cv_oof"]["raw_coords"] - METRICS["grouped_cv_oof"]["grip"]) * 100
verdict = (f"overlap — the {gap_pts:.0f}-pt gap is within group noise" if raw_lo <= grip_hi
           else f"stay apart — the {gap_pts:.0f}-pt gap survives the bootstrap")
fig, ax = plt.subplots(figsize=(7.4, 3.0)); yp = np.arange(len(boot_rows))[::-1]
for yy, (nm, col, acc, lo, hi) in zip(yp, boot_rows):
    ax.plot([lo, hi], [yy, yy], color=col, lw=3, solid_capstyle="round", zorder=2)
    ax.scatter([acc], [yy], color="white", edgecolor=col, s=90, lw=2.2, zorder=3)
    ax.text((lo + hi) / 2, yy + 0.22, f"{acc:.3f}  [{lo:.3f}, {hi:.3f}]", ha="center", fontsize=8.5, color=INK2)
ax.set_yticks(yp); ax.set_yticklabels([r[0] for r in boot_rows]); ax.set_ylim(-0.6, len(boot_rows) - 0.35)
ax.set_xlim(0.85, 1.0); ax.set_xlabel("grouped-CV accuracy · 95% cluster-bootstrap CI (1000 resamples over groups)")
style(ax, ygrid=False); ax.grid(axis="x", alpha=1.0)
ax.set_title(f"Raw vs GRIP 95% CIs {verdict}", fontweight="bold", fontsize=11.5, loc="left")
plt.tight_layout(); plt.show()
""")

# ================================================================ 8e. adversarial validation (md)
md(r"""
## Is raw's edge posture — or dataset identity? An adversarial check

The confound warned that a flexible model fed raw coordinates could score partly by recognizing *which capture
rig* a pose came from. Here I measure that directly: train a classifier to predict the **source dataset**
(URFD vs MultiPosture) under the same grouped CV, and read its ROC-AUC. Because each class lives in one source,
*some* source-signal is unavoidable — it literally *is* posture. So the informative quantity is not the
absolute AUC but the **gap**: how much better the raw coordinates predict the source than the geometry-only
GRIP features do. That surplus is rig-identity information GRIP discards by construction — the cleanest
one-number read on the shortcut we were worried about.
""")

# ================================================================ 8f. adversarial validation (code)
code(r"""
from sklearn.metrics import roc_auc_score
src_bin = (np.array(src) == np.unique(src)[0]).astype(int)      # binary: is-this-source-A
METRICS["adversarial_source_auc"] = {}
for nm, key, Xin in [("raw coords", "raw_coords", C), ("GRIP", "grip", F)]:
    proba = cross_val_predict(rf(), Xin, src_bin, groups=grp, cv=sgkf, method="predict_proba")[:, 1]
    METRICS["adversarial_source_auc"][key] = round(float(roc_auc_score(src_bin, proba)), 4)
    print(f"  source-AUC from {nm:11}: {METRICS['adversarial_source_auc'][key]:.3f}")
draw = METRICS["adversarial_source_auc"]["raw_coords"]; dgrip = METRICS["adversarial_source_auc"]["grip"]
gapp = (draw - dgrip) * 100
print(f"  surplus (raw - GRIP): {gapp:+.1f} AUC-points of rig identity that GRIP throws away")
tt = (f"Raw reads the capture rig almost perfectly (AUC {draw:.2f}) — {gapp:+.0f} pts above GRIP's {dgrip:.2f}"
      if abs(gapp) >= 1 else
      "Raw and GRIP read the source near-equally — the confound is posture, not a raw-only rig fingerprint")

fig, ax = plt.subplots(figsize=(7.0, 2.6))
for yy, (nm, key, col) in zip([1, 0], [("raw coords", "raw_coords", RAW), ("GRIP", "grip", GRIP)]):
    a = METRICS["adversarial_source_auc"][key]
    ax.barh(yy, a, height=0.55, color=col, edgecolor=SURF, linewidth=2)
    ax.text(a - 0.015, yy, f"{a:.3f}", ha="right", va="center", color="white", fontweight="bold", fontsize=10)
ax.axvline(0.5, ls=(0, (4, 4)), color=MUTED, lw=1)
ax.text(0.5, 0.5, "0.5 = chance", rotation=90, va="center", ha="right", color=MUTED, fontsize=8)
ax.set_yticks([1, 0]); ax.set_yticklabels(["raw coords", "GRIP"]); ax.set_xlim(0.4, 1.02); ax.set_ylim(-0.55, 1.55)
ax.set_xlabel("grouped-CV ROC-AUC for predicting the source dataset")
style(ax, ygrid=False); ax.grid(axis="x", alpha=1.0)
ax.set_title(tt, fontweight="bold", fontsize=10.5, loc="left")
plt.tight_layout(); plt.show()
""")

# ================================================================ 9. Azimuth / Finding B (md)
md(r"""
## The azimuth test — where the discovery pays off (H2)

Now rotate every *test* pose about the vertical axis — the subject turning relative to a fixed camera — and
re-score **without retraining**. This is the deployment reality a home camera faces every minute, and it is
a test a dataset-identity shortcut *cannot* pass: rotation scrambles exactly the viewpoint-specific
coordinates a raw model leans on.

The result is the whole argument. **GRIP stays flat at every angle** (its features never saw a viewpoint to
begin with). **Raw coordinates start higher but decay and then fall off a cliff toward the 1-in-3 chance
line by 90°.** This robustness wasn't what I reached for — it fell out of the physics — but it is precisely
the axis this deployment cannot afford to lose. Each curve is the **mean across all five grouped folds** and
the band is **±1 SD**, so the flatness is a property of the method, not a lucky single split.
""")

# ================================================================ 9b. azimuth + V5 (code)
code(r"""
from sklearn.metrics import accuracy_score

def azimuth_rotate(lm, deg):
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    out = lm.copy(); x, z = lm[:, 0], lm[:, 2]
    out[:, 0] = x*c + z*s; out[:, 2] = -x*s + z*c
    return out

def azimuth_curve_folds(feat_fn, angles):
    '''Per-fold azimuth accuracy: train on each grouped train-split, score its own
    UNSEEN people at every rotation (no retrain per angle). Returns (n_folds, n_angles).'''
    rows = []
    for tr, te in sgkf.split(X, y, groups=grp):
        m = rf().fit(np.array([feat_fn(X[i]) for i in tr]), y[tr])
        rows.append([accuracy_score(y[te], m.predict(np.array([feat_fn(azimuth_rotate(X[i], dg)) for i in te])))
                     for dg in angles])
    return np.array(rows)

ROT = [0, 15, 30, 45, 90]
az = {"GRIP": azimuth_curve_folds(features, ROT), "raw coords": azimuth_curve_folds(raw_coords, ROT)}
METRICS["azimuth_deg"] = ROT
METRICS["azimuth_acc"] = {nm: [round(float(v), 4) for v in a.mean(0)] for nm, a in az.items()}
METRICS["azimuth_std"] = {nm: [round(float(v), 4) for v in a.std(0)] for nm, a in az.items()}
for nm, a in az.items():
    print(f"  {nm:12}: " + " ".join(f"az{d}={m:.3f}+/-{s:.3f}" for d, m, s in zip(ROT, a.mean(0), a.std(0))))

fig, ax = plt.subplots(figsize=(7.4, 4.4))
for nm, a in az.items():
    col = GRIP if nm == "GRIP" else RAW
    m, s = a.mean(0), a.std(0)
    ax.fill_between(ROT, m - s, m + s, color=col, alpha=0.15, lw=0, zorder=1)
    ax.plot(ROT, m, marker="o", color=col, lw=2.4, ms=8, mec=SURF, mew=1.5, zorder=3)
    ax.text(ROT[-1] + 1.5, m[-1], f"{nm}\n{m[-1]:.2f}", color=col, fontsize=10, fontweight="bold", va="center")
ax.axhline(1/3, ls=(0, (4, 4)), color=MUTED, lw=1); ax.text(2, 1/3 + 0.015, "chance (1 of 3)", color=MUTED, fontsize=8)
ax.set_xlim(-3, 108); ax.set_ylim(0.25, 1.02); ax.set_xticks(ROT)
ax.set_xlabel("camera azimuth rotation (deg)"); ax.set_ylabel("accuracy (no retrain)"); style(ax)
ax.set_title("The payoff: rotate the camera and raw collapses; GRIP does not move\n"
             "(mean +/- 1 SD across all 5 grouped folds)",
             fontweight="bold", fontsize=11.5, loc="left")
plt.tight_layout(); plt.show()
""")

# ================================================================ 9a2. per-class azimuth (md)
md(r"""
## Per class: which posture does raw lose first?

The aggregate azimuth curve hides *which* class drives the collapse. Split three ways — one panel per class,
GRIP vs raw, recall against rotation — a non-uniform and slightly counter-intuitive picture appears. **GRIP is
flat in every panel** (its invariance is per-feature, so it cannot depend on class). **Raw is anything but:**
its overall decay is *concentrated*, not spread evenly — the aggregate curve is really the story of one or two
classes falling off a cliff while another stays robust. And it is not the class you might guess: read the
panels rather than trust an intuition about which posture "should" be viewpoint-sensitive. The class whose
single-frame signature is most viewpoint-specific is the one a rotated camera cannot recover — this is the
aggregate azimuth curve, decomposed and explained.
""")

# ================================================================ 9a3. per-class azimuth (code)
code(r"""
def azimuth_perclass(feat_fn, angles):
    '''Per-fold, per-angle, per-class recall (mean over folds). Returns {class_idx: [recall @ each angle]}.'''
    acc = {c: [[] for _ in angles] for c in range(3)}
    for tr, te in sgkf.split(X, y, groups=grp):
        m = rf().fit(np.array([feat_fn(X[i]) for i in tr]), y[tr]); yt = y[te]
        for ai, dg in enumerate(angles):
            pr = m.predict(np.array([feat_fn(azimuth_rotate(X[i], dg)) for i in te]))
            for c in range(3):
                mask = yt == c
                if mask.sum(): acc[c][ai].append(float((pr[mask] == c).mean()))
    return {c: [float(np.mean(v)) if v else float("nan") for v in acc[c]] for c in range(3)}

pc = {"GRIP": azimuth_perclass(features, ROT), "raw coords": azimuth_perclass(raw_coords, ROT)}
METRICS["azimuth_perclass"] = {nm: {CLASSES[c]: [round(a, 4) for a in pc[nm][c]] for c in range(3)} for nm in pc}
for nm in pc:
    for c in range(3):
        print(f"  {nm:11} {CLASSES[c]:9}: " + " ".join(f"{a:.2f}" for a in pc[nm][c]))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
for c, ax in enumerate(axes):
    for nm in ("raw coords", "GRIP"):
        col = GRIP if nm == "GRIP" else RAW
        ax.plot(ROT, pc[nm][c], marker="o", color=col, lw=2.3, ms=7, mec=SURF, mew=1.4, label=nm, zorder=3)
    ax.axhline(1/3, ls=(0, (4, 4)), color=MUTED, lw=1)
    ax.set_title(CLASSES[c], fontsize=11, fontweight="bold"); ax.set_xticks(ROT); ax.set_ylim(0, 1.05)
    ax.set_xlabel("azimuth (deg)"); style(ax)
    if c == 0: ax.set_ylabel("recall (no retrain)")
axes[-1].legend(frameon=False, loc="lower left", fontsize=9)
fig.suptitle("Per class: GRIP holds at every angle; raw's collapse under rotation is not uniform",
             fontweight="bold", fontsize=12, x=0.012, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.show()
""")

# ================================================================ 9c. Ablation (md)
md(r"""
## How many features does it actually need? (ablation)

The importance ranking hinted that ~3 features carry the signal. So does the model actually *need* all 14, or
are the weak ones just noise the forest ignores? I retrain on the **top-3, top-5, top-10 and all-14** features
(by the importance rank above) and report three numbers each: grouped-CV accuracy (clean), and azimuth
accuracy at **0°** and **90°**. Two things to watch: clean accuracy **saturates early** (14 is generous, not
essential — a smaller model would ship almost as well), and **invariance holds at every subset size** —
pruning never reintroduces a viewpoint dependence, because every surviving feature is invariant *by
construction*, not by ensemble averaging. In fact the `az_0` and `az_90` columns below come out **identical to
the digit**: rotating the input about the vertical axis leaves every GRIP feature unchanged, so accuracy
cannot move. That exactness is the mathematical claim, made empirical.
""")

# ================================================================ 9d. Ablation (code)
code(r"""
imp_rank = np.argsort(imp)[::-1]                      # imp from the importance cell (full 14-feature fit)
abl = []
for k in [3, 5, 10, 14]:
    idx = imp_rank[:k]
    gcv = cross_val_score(rf(), F[:, idx], y, groups=grp, cv=sgkf, scoring="accuracy")
    azc = azimuth_curve_folds(lambda x, idx=idx: features(x)[idx], [0, 90]).mean(0)
    abl.append({"n_feats": int(k), "top_feats": ", ".join(FEAT_NAMES[i] for i in idx[:3]) + ("..." if k > 3 else ""),
                "grouped_CV": round(float(gcv.mean()), 4), "az_0": round(float(azc[0]), 4),
                "az_90": round(float(azc[1]), 4)})
    print(f"  top-{k:<2} feats: grouped-CV={gcv.mean():.3f}  az0={azc[0]:.3f}  az90={azc[1]:.3f}")
METRICS["ablation"] = abl
ks = [r["n_feats"] for r in abl]

gc = [r["grouped_CV"] for r in abl]
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.plot(ks, gc, marker="o", color=GRIP, lw=2.6, ms=9, mec=SURF, mew=1.6, zorder=3)
for r in abl:
    ax.annotate(f"{r['grouped_CV']:.3f}", (r["n_feats"], r["grouped_CV"]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8.5, color=INK2)
ax.axhline(gc[-1], ls=(0, (4, 4)), color=MUTED, lw=1)
ax.text(ks[0] + 0.1, gc[-1] + 0.003, "all-14 ceiling", color=MUTED, fontsize=8, va="bottom")
ax.set_xticks(ks); ax.set_ylim(0.86, 0.96); ax.set_xlabel("number of features (top-k by importance)")
ax.set_ylabel("grouped-CV accuracy"); style(ax)
ax.set_title("Clean accuracy saturates by ~10 features — 14 is generous, not essential\n"
             "(azimuth 0 deg == 90 deg == clean, to the digit: invariance is exact)",
             fontweight="bold", fontsize=11, loc="left")
plt.tight_layout(); plt.show()
pd.DataFrame(abl)[["n_feats", "top_feats", "grouped_CV", "az_0", "az_90"]]
""")

# ================================================================ 10. Synthesis (md)
md(r"""
## The trade-off, in one figure

Two panels, one story. **Left:** on clean data the raw coordinates are the *more* accurate input — the 14
features are a lossy summary and give up a few points (H1 refuted). **Right:** rotate the camera and the
ranking inverts violently — raw falls off a cliff while GRIP does not move (H2 confirmed). Engineering is
choosing which axis you refuse to lose; for a camera people turn away from, that axis is viewpoint.
""")

# ================================================================ 10b. V6 (code)
code(r"""
gc = METRICS["grouped_cv"]; order = ["raw_coords", "grip", "vae_pca32"]
names = {"raw_coords": "raw coords\n(99-D)", "grip": "GRIP\n(14 feats)", "vae_pca32": "VAE latent\n(~PCA32)"}
cols  = {"raw_coords": RAW, "grip": GRIP, "vae_pca32": VAE}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.4))
for i, k in enumerate(order):
    m, s = gc[k]
    axL.bar(i, m, width=0.6, color=cols[k], alpha=0.9, edgecolor=SURF, linewidth=2)
    axL.errorbar(i, m, yerr=s, color=INK2, capsize=4, lw=1.2)
    axL.text(i, 1.012, f"{m:.3f}", ha="center", fontweight="bold", color=cols[k])
axL.set_xticks(range(3)); axL.set_xticklabels([names[k] for k in order]); axL.set_ylim(0.80, 1.03)
axL.set_ylabel("grouped-CV accuracy"); style(axL)
axL.set_title("Clean data: raw wins", fontweight="bold", loc="left", pad=26)
for nm, accs in METRICS["azimuth_acc"].items():
    col = GRIP if nm == "GRIP" else RAW
    axR.plot(METRICS["azimuth_deg"], accs, marker="o", color=col, lw=2.4, ms=8, mec=SURF, mew=1.5, label=nm)
axR.axhline(1/3, ls=(0, (4, 4)), color=MUTED, lw=1); axR.set_ylim(0.25, 1.02); axR.set_xticks(METRICS["azimuth_deg"])
axR.set_xlabel("camera azimuth (deg)"); axR.legend(frameon=False); style(axR)
axR.set_title("Under rotation: raw collapses, GRIP holds", fontweight="bold", loc="left")
fig.suptitle("A few points of clean accuracy bought total viewpoint invariance", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.show()
""")

# ================================================================ 11. Where it fails (md)
md(r"""
## Where GRIP fails — a confusion matrix

Honest models report their mistakes. Under grouped CV, GRIP's errors are not uniform: **standing is
sometimes read as sitting** (a partially-bent standing pose looks seated in coarse geometry), while lying is
almost never confused. This is the right failure mode for a smart-home context — a missed *fall* (lying) is
far costlier than a standing/sitting mix-up — but it names exactly where a fourth "transitional" class or a
temporal model would help next.
""")

# ================================================================ 11b. V7 confusion (code)
code(r"""
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
pred = cross_val_predict(rf(), F, y, groups=grp, cv=sgkf)
cm = confusion_matrix(y, pred); cmn = cm / cm.sum(1, keepdims=True)
METRICS["confusion"] = cm.tolist()
METRICS["recall_per_class"] = {CLASSES[i]: round(float(cmn[i, i]), 4) for i in range(3)}

fig, ax = plt.subplots(figsize=(4.9, 4.3)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cmn[i,j]:.2f}\n({cm[i,j]})", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else INK, fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
ax.set_title("GRIP grouped-CV confusion (row-normalized)", fontweight="bold", fontsize=11, loc="left")
plt.tight_layout(); plt.show()
""")

# ================================================================ 11c. calibration (md)
md(r"""
## Can we trust the probabilities? Calibration and ECE

Accuracy says *how often* the model is right; calibration says whether its **confidence means anything**. For a
safety context that matters — a 0.9-confidence call should be wrong about 10% of the time, no more. I take the
grouped-CV out-of-fold probabilities, bin predictions by confidence, and plot each bin's accuracy against its
mean confidence (the reliability diagram). The size-weighted gap between the two is the **Expected Calibration
Error (ECE)**. The per-class print-out is where to watch `standing` — the class the confusion matrix flagged.
""")

# ================================================================ 11d. calibration (code)
code(r"""
oof_proba = cross_val_predict(rf(), F, y, groups=grp, cv=sgkf, method="predict_proba")
conf = oof_proba.max(1); pred_c = oof_proba.argmax(1); hit = (pred_c == y).astype(float)
bins = np.linspace(0, 1, 11); centers = (bins[:-1] + bins[1:]) / 2
bacc = np.full(10, np.nan); bcnt = np.zeros(10, int)
for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
    m = (conf >= lo) & (conf < (hi if b < 9 else 1.0001))
    if m.sum(): bacc[b] = hit[m].mean(); bcnt[b] = int(m.sum())
bconf = np.array([conf[(conf >= lo) & (conf < (hi if b < 9 else 1.0001))].mean() if bcnt[b] else centers[b]
                  for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]))])
ece = float(np.nansum(bcnt * np.abs(bacc - bconf)) / bcnt.sum())
METRICS["ece"] = round(ece, 4)
print(f"  ECE = {ece:.3f}  (mean |confidence - accuracy|, weighted by bin count)")
METRICS["calibration_by_pred_class"] = {}
for ci, cn in enumerate(CLASSES):
    mc = pred_c == ci
    if mc.sum():
        METRICS["calibration_by_pred_class"][cn] = [round(float(conf[mc].mean()), 4), round(float(hit[mc].mean()), 4)]
        print(f"    predicted {cn:9}: mean-conf={conf[mc].mean():.3f}  accuracy={hit[mc].mean():.3f}  (n={int(mc.sum())})")

fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color=MUTED, lw=1.2, zorder=1)
ax.bar(centers, np.nan_to_num(bacc), width=0.09, color=GRIP, alpha=0.85, edgecolor=SURF, linewidth=1.5, zorder=2)
for c, a, n in zip(centers, bacc, bcnt):
    if n: ax.text(c, a + 0.02, str(n), ha="center", fontsize=7, color=MUTED)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.06); ax.set_aspect("equal", adjustable="box"); style(ax)
ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("actual accuracy")
ax.text(0.04, 0.92, f"ECE = {ece:.3f}", fontsize=12, fontweight="bold", color=INK)
ax.set_title("Reliability diagram — bars on the dashed line are well-calibrated",
             fontweight="bold", fontsize=10.5, loc="left")
plt.tight_layout(); plt.show()
""")

# ================================================================ 11e. HPO (md)
md(r"""
## Were 120 trees at depth 16 the right call? A grouped hyperparameter search

The forest params were inherited from the production spike. Rather than trust them, I grid-search
`n_estimators × max_depth × min_samples_split` under the *same* grouped CV and check where the shipped
configuration lands. The honest outcome I expect is boring — a broad plateau where the defaults sit within
noise of the best — and boring is exactly what documents that the headline number wasn't cherry-picked.
""")

# ================================================================ 11f. HPO (code)
code(r"""
from sklearn.model_selection import GridSearchCV
grid = {"n_estimators": [60, 120, 240], "max_depth": [8, 12, 16, 24], "min_samples_split": [2, 5]}
gs = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=-1), grid, cv=sgkf,
                  scoring="accuracy", n_jobs=-1).fit(F, y, groups=grp)
res = gs.cv_results_; shipped = {"n_estimators": 120, "max_depth": 16, "min_samples_split": 2}
ship_s = next((float(s) for p, s in zip(res["params"], res["mean_test_score"])
               if all(p[k] == shipped[k] for k in shipped)), float("nan"))
best_s = float(gs.best_score_)
METRICS["hpo"] = {"best_params": {k: gs.best_params_[k] for k in grid}, "best_score": round(best_s, 4),
                  "shipped_score": round(ship_s, 4), "delta_pts": round((best_s - ship_s) * 100, 2)}
print(f"  best   {gs.best_params_} -> {best_s:.3f}")
print(f"  shipped(120,16,2)          -> {ship_s:.3f}   (gap to best: {(best_s-ship_s)*100:.2f} pts)")

ne = grid["n_estimators"]; mds = grid["max_depth"]; Z = np.full((len(ne), len(mds)), np.nan)
for p, s in zip(res["params"], res["mean_test_score"]):
    if p["min_samples_split"] == 2: Z[ne.index(p["n_estimators"]), mds.index(p["max_depth"])] = s
fig, ax = plt.subplots(figsize=(6.4, 3.5)); ax.imshow(Z, cmap="Greens", aspect="auto",
        vmin=np.nanmin(Z) - 0.004, vmax=np.nanmax(Z))
for i in range(len(ne)):
    for j in range(len(mds)):
        ax.text(j, i, f"{Z[i,j]:.3f}", ha="center", va="center", fontsize=9,
                color="white" if Z[i, j] > np.nanmean(Z) else INK)
si, sj = ne.index(120), mds.index(16)
ax.add_patch(plt.Rectangle((sj - 0.5, si - 0.5), 1, 1, fill=False, edgecolor=RAW, lw=2.5))
ax.text(sj, si - 0.36, "shipped", ha="center", color=RAW, fontsize=8, fontweight="bold")
ax.set_xticks(range(len(mds))); ax.set_xticklabels(mds); ax.set_yticks(range(len(ne))); ax.set_yticklabels(ne)
ax.set_xlabel("max_depth"); ax.set_ylabel("n_estimators"); ax.grid(False)
ax.set_title("Grouped-CV accuracy is a plateau — shipped params sit on it (min_samples_split=2)",
             fontweight="bold", fontsize=10.5, loc="left")
plt.tight_layout(); plt.show()
""")

# ================================================================ 12. Why RF (md)
md(r"""
## Why RandomForest, not LightGBM?

NB3's LightGBM peaked at ~71.7% on noisy COCO weak-labels; the lever here was **better data + invariant
features**, not a fancier booster. And for *this* product RandomForest wins on deployment, not accuracy:

| | LightGBM | RandomForest |
|---|---|---|
| grouped-CV accuracy (features) | comparable | **≈0.93** |
| export to Rust | C++ runtime (`liblightgbm`) | plain binary, **no runtime** |
| runtime deps | yes | **zero** |
| CPU inference | ~1–2 ms | **< 1 ms** |

The decisive column is *runtime deps*: a forest is just thresholds and probabilities, so it re-implements as
a ~40-line reader over a flat binary — no `liblightgbm`, no ONNX, nothing to ship but bytes.
""")

# ================================================================ 13. Export (md)
md(r"""
## Export to a dependency-free binary (what production actually loads)

The forest serializes as `u32 n_trees`, then per tree `u32 n_nodes` and nodes of
`<i32 feature, f32 threshold, i32 left, i32 right, f32 prob[3]>`. Rust `include_bytes!` embeds it; a ~40-line
reader walks the trees.

**Honest sizing.** This notebook's forest is **~270 KB**; the *production* `pose_rf_forest.bin` is **873 KB**
because it trains on augmented data (more samples → deeper trees). Earlier docs said "12 KB" — never true for
a 120-tree, depth-16 forest, and corrected here. A few hundred KB of pure bytes, zero deps, sub-millisecond:
that property is what matters, not the exact figure.
""")

# ================================================================ 13b. export (code)
code(r"""
import struct
rf_final = rf().fit(F, y)
def serialize_forest(forest):
    buf = bytearray(); buf += struct.pack("<I", len(forest.estimators_))
    for est in forest.estimators_:
        t = est.tree_; buf += struct.pack("<I", t.node_count)
        proba = t.value.reshape(t.node_count, -1); proba = proba / proba.sum(1, keepdims=True)
        for n in range(t.node_count):
            buf += struct.pack("<ifii3f", int(t.feature[n]), float(t.threshold[n]),
                               int(t.children_left[n]), int(t.children_right[n]),
                               *[float(proba[n, c]) for c in range(3)])
    return bytes(buf)
blob = serialize_forest(rf_final)
open("/kaggle/working/pose_rf_forest.bin", "wb").write(blob)
METRICS["forest_kb"] = round(len(blob) / 1024, 1); METRICS["forest_trees"] = len(rf_final.estimators_)
print(f"serialized forest: {len(blob)/1024:.1f} KB, {len(rf_final.estimators_)} trees "
      f"-> pose_rf_forest.bin  (production embeds this via include_bytes!)")
""")

# ================================================================ 14. metrics dump (code)
code(r"""
# --- Emit every harvested number for kctl fetch -> results/metrics.json ---
import json, os
os.makedirs("/kaggle/working", exist_ok=True)
with open("/kaggle/working/metrics.json", "w") as f:
    json.dump(METRICS, f, indent=2)
print(json.dumps(METRICS, indent=2))
""")

# ================================================================ 15. Production (md)
md(r"""
## Production status (July 2026)

GRIP is the **primary pose classifier** in MVM (`mvm-posture`) since 2025; the VAE (NB2) was demoted to an
out-of-distribution gate. Inference is `< 1 ms` on CPU with no dependencies.

**Occlusion honesty (fusion layer, `mvm-fusion`).** When only the upper body is visible
(`BodyCoverage::Upper`), pose is **withheld** rather than guessed — GRIP supplies the 2-D torso attitude used
to make that call. The classifier does *not* detect falls (a standing→lying transition over seconds); that
is the anomaly detector's job (NB2 + fusion).
""")

# ================================================================ 15b. SOTA positioning (md)
md(r"""
## Where GRIP sits in the pose-classification field

GRIP lives deliberately at the *small, interpretable* end of the pose literature, and it helps to say so
plainly. Skeleton-based recognition is dominated by graph-convolutional networks — **ST-GCN** (Yan et al.,
2018) and its successors (**2s-AGCN**, **CTR-GCN**) evaluated on **NTU RGB+D** and **Kinetics-Skeleton** —
which learn spatio-temporal filters over the joint graph and reach the high-90s on many-class *action*
benchmarks. Those models target dozens of *dynamic* actions from sequences; GRIP targets three *static*
postures from a single frame, under a constraint they do not carry: sub-millisecond CPU inference and zero
runtime dependencies inside an embedded Rust binary. The contribution here is not a new architecture — it is
the discipline of choosing a representation whose invariances match the deployment and measuring the trade
honestly. The temporal, graph-based direction is exactly where this goes next (NB5), where a *fall* becomes a
trajectory rather than a frame.
""")

# ================================================================ 16. Limits + conclusion (md)
md(r"""
## Honest limits

- **Raw coordinates lead GRIP on clean data (point estimate ≈0.97 vs ≈0.94) — but the 95% CIs overlap.** At
  26 groups that few-point lead is *not statistically established*; the feature engineering plausibly costs a
  little fixed-viewpoint accuracy, and it is only worth it because the deployment rotates. On a bolted-down,
  known-viewpoint camera this trade would be the *wrong* call.
- **The source↔class confound is unfixable with this data.** Each class lives in one dataset, so no
  grouped split can separate "posture" from "capture rig". The azimuth test mitigates but does not erase it;
  a truly clean read needs a dataset where every class appears in multiple capture setups.
- **The quoted 0.959 does not reproduce** with the shipped RF params on this public data (honest grouped-CV
  ≈0.93). Most plausibly the production run's train-time augmentation and/or a different data snapshot —
  flagged, not hidden.
- **Only 26 groups.** The ±0.05 fold std is wide, and the cluster-bootstrap CIs span ~0.06–0.15 and overlap
  across all three inputs — the mean alone is not trustworthy. More public pose data (more people, more rigs)
  is the obvious next experiment — it tightens these intervals *and* attacks the confound.
- `torso_length_m = 0.5` is assumed, not tape-measured — absolute 3-D placement is approximate.
- Three coarse classes; "sitting on a chair" vs. "sitting on the floor" is not distinguished, and there is
  no explicit *transitional* class (see the confusion matrix).

## Breaking the 26-group ceiling — the designated next experiment

The single biggest lever on every number above is **more groups**. Twenty-six is small, and the way this set
was built explains both the size and the confound: it is a deliberately *capped probe* — standing + lying come
only from URFD (fall sequences), sitting only from Zenodo MultiPosture — so class is welded to source and some
"groups" are sequences, not distinct people. Raising the frame caps or augmenting adds rows but neither
diversity nor independent groups, and does nothing for the confound. What actually moves the needle, in order:

1. **Weak supervision on COCO — the lever [NB3](https://www.kaggle.com/code/howadx) already built.** Thousands
   of in-the-wild people, all three postures across many rigs and viewpoints: it *breaks* the source↔class
   weld and multiplies effective groups, tightening the wide CIs seen above. Cost: ~25–35% label noise, and a
   topology bridge — GRIP needs MediaPipe-33 with pseudo-z, so COCO images must be run through PoseLandmarker,
   not read as flat 2-D COCO-17 keypoints. Recipe: weak COCO as the bulk *train* set, this clean lab data as a
   confound-free *gold test*.
2. **A multi-posture, multi-view corpus (e.g. NTU RGB+D).** Every subject performs every posture from several
   real cameras — this breaks the confound *and* validates the azimuth thesis with real viewpoints instead of
   a synthetic rotation. Cost: research-only licensing.
3. **URFD ADL sequences + more MultiPosture subjects.** Cheap, but a partial fix only — still one source per
   class.

Stated as the honest current state, not glossed: the shipped classifier trains on augmented lab data, and the
published grouped-CV rests on 26 groups. Expanding them is the next notebook's job — not a number I will
quietly inflate here.

## Conclusion — the engineering lesson

GRIP is not the most accurate model in this notebook; raw coordinates are. It is the *right* model, because
it is accurate on the one nuisance a home camera cannot escape — people turning — and it is right by
**construction**, not by luck of the training distribution. The portfolio point is not a big number; it is
the discipline to (1) frame the work as a question with falsifiable hypotheses, (2) measure with grouped CV
and confront a confound in my own data, (3) report where my method *loses*, and (4) pick the input whose
failure mode matches the deployment. A model you can defend beats a model that merely looks good.

## If I started this today

Trajectory awareness matters more than any single number, so here is what I would do differently, knowing what
this notebook taught me:

- **Pick the evaluation before the model.** The source↔class confound was baked in the moment each class came
  from one dataset. I would start from a corpus where every posture appears in multiple capture setups (or
  carve out a clean cross-source test split on day one), so "posture" and "rig" are separable by design.
- **Report intervals from the first commit, not the last.** The wide, overlapping bootstrap CIs are the real
  headline; leading with a point estimate (the old 0.959) is *how* the original overclaim happened. CI-first
  framing catches that immediately.
- **Treat invariance as a hypothesis to test, not a property to assert.** I reached for gravity-referenced
  features on physical intuition; the azimuth test is what turned intuition into evidence. Next time I build
  that test *first* and let it steer the feature design.
- **Keep the raw-coordinate baseline.** It is the most useful thing in the notebook — not because it wins, but
  because its collapse under rotation is what makes GRIP's case. A strong baseline you can *lose* to on one
  axis is worth more than a weak one you beat everywhere.

## What changed
- **2026-07 (revision)** — reopened as a research artifact: explicit hypotheses (H1 refuted, H2 confirmed),
  a new **confound** section + chart, feature-importance and confusion charts, and one locked
  colorblind-validated palette across every figure. Numbers from a live Kaggle run: **0.959 → ≈0.93**,
  **"12 KB" → ~270 KB (873 KB prod)**, and the honest finding that **raw coords win on clean data**.
- **2026-07** — notebook created from `spike_pose/spike_feats.py` + `train_pose_p1.py`.
- **2025** — GRIP replaced LightGBM (NB3) and the VAE (NB2) in production.

*Prev: NB3 (weak supervision) — where the data pipeline began · NB2 (pose manifold) — what GRIP replaced.
Next: NB4 — a component-level teardown on engineering judgment.*
""")

# ================================================================ write
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb3b_grip.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out_path, "|", len(cells), "cells")
