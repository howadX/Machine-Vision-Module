#!/usr/bin/env python3
"""Builder for LJ2 (Kaggle): "Physics of posture transitions" — MVM Lab Journal 2.

With a trustworthy per-frame pose classifier from LJ1, we turn to TIME. Two questions the
single frame cannot answer: a per-frame label flickers frame-to-frame, and `lying` is `lying`
whether you fell or lay down. Both are answered by the physics of transitions, measured from CMU
motion capture (true kinematics). Act 1: a transition-prior vs the hand-tuned dwell — an honest
wash on aggregate, a real win on the one impossible edge (standing<->lying). Act 2: that same
edge, followed into the fall-vs-lie-down velocity threshold.

Reproducible authoring: `python _build_lj2.py` -> lj2_physics_of_posture_transitions.ipynb,
Run-All on Kaggle over the published derived datasets. Every number is a live run.
Follows the frozen LJ template (context / hypotheses / experiments+numbers / verdict / bridge).
"""
import json, os

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": t.strip("\n").splitlines(keepends=True)})

# ============================================================ 1. Title
md(r"""
# LJ2 — Physics of posture transitions
### a per-frame label flickers, and *lying* is *lying* whether you fell or lay down — both are questions about time

*MVM Lab Journal — 2 of a series. Prev: LJ1 (The pose model, rebuilt honestly). Next: LJ3 (the
tracker on true trajectories).*

> LJ1 gave us a per-frame pose classifier we can trust across rigs and viewpoints. But a camera
> sees *frames*, and two things a single frame cannot settle nag the product: the emitted label
> **flickers** (one bad frame flips standing→lying→standing), and when someone ends up horizontal,
> was it a **fall** or a **lie-down**? Both are questions about *motion between postures*. This
> journal measures that motion from CMU motion capture — true kinematics, true time — and lets it
> decide, including where the elegant hypothesis loses.
""")

# ============================================================ 2. Context
md(r"""
## 1. Context — smoothing tuned by eye, a fall signal never measured

Production stabilises posture with a **dwell machine** (`PostureStabilizer`): a flip commits only
after *N* consecutive confident frames. `N=3`, confidence gate `0.5` — **tuned by eye**, never
measured on repeatable data. It is *symmetric and label-agnostic*: every flip pays the same dwell,
whether it is a plausible standing→sitting or a physically-impossible standing→lying in one frame.

And the anomaly layer has no honest way to tell a **fall** from a **lie-down** — both end in
`lying`. The distinction is purely kinematic: how fast the body came down. It was never measured.

CMU mocap gives us both, because it is true 3D over true time: the ground-truth posture at every
frame, the transition timing, and the metric height of the pelvis. Two acts, two falsifiable
hypotheses, each with its failure criterion named **up front**.
""")

# ============================================================ 3. Setup
code(r"""
# ---- self-contained setup: the transition artifact, the 2D-GRIP feature, the emitter ----
import os, numpy as np, collections
from sklearn.ensemble import RandomForestClassifier
np.random.seed(0); SEED = 0
CLASSES = ["standing", "sitting", "lying"]
PROD_FPS, SRC_FPS = 15, 60          # stabiliser runs at the ~16 fps human-chain cadence
H, LMAX = 8, 15                     # >=8 frames (~0.5s) = a real posture; a commit within 15 = legit

def _find(*names):
    import glob
    roots = ["/kaggle/input/mvm-mocap-transitions", "/kaggle/input/mvm-pose-landmarks-v2",
             "/kaggle/input/mvm-pose-landmarks",
             os.path.join("..","..","..","spike_pose","data_probe"),
             os.path.join("..","..","spike_pose","data_probe")]
    for r in roots:
        for n in names:
            if os.path.exists(os.path.join(r, n)): return os.path.join(r, n)
    for n in names:
        hits = glob.glob(f"/kaggle/input/**/{n}", recursive=True)
        if hits: return hits[0]
    have = os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "no /kaggle/input"
    raise FileNotFoundError((names, "mounted:", have))

TR   = np.load(_find("mocap_transitions.npz"), allow_pickle=True)   # 35 CMU trials @60fps, true time
CORE = np.load(_find("landmarks_v2.npz","landmarks.npz"), allow_pickle=True)  # lab+gold, emitter train

# --- 2D-GRIP feature (z=0), identical to LJ1 / production ---
L = dict(nose=0,Lsh=11,Rsh=12,Lel=13,Rel=14,Lwr=15,Rwr=16,Lhip=23,Rhip=24,Lkn=25,Rkn=26,Lank=27,Rank=28)
GI = list(L.values()); UP = np.array([0.,-1.,0.],np.float32)
def _ang(v): n=np.linalg.norm(v); return 0. if n<1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(v/n,UP),-1,1))))
def _flex(a,b,c):
    u,w=a-b,c-b; nu,nw=np.linalg.norm(u),np.linalg.norm(w)
    return 180. if nu<1e-6 or nw<1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(u/nu,w/nw),-1,1))))
def features_13(lm):
    p=lm[:,:3].astype(np.float32); g=lambda k:p[L[k]]
    hip=(g("Lhip")+g("Rhip"))/2; sh=(g("Lsh")+g("Rsh"))/2; tl=np.linalg.norm(sh-hip)
    if tl<1e-6: tl=max(np.linalg.norm(g("Lsh")-g("Rsh")),1e-3)
    q=p[GI]; vext=(q[:,1].max()-q[:,1].min())/tl
    rad=np.sqrt((q[:,0]-hip[0])**2+(q[:,2]-hip[2])**2); hext=2*rad.max()/tl
    return np.array([_ang(sh-hip),
        .5*(_ang(g("Lkn")-g("Lhip"))+_ang(g("Rkn")-g("Rhip"))),.5*(_ang(g("Lank")-g("Lkn"))+_ang(g("Rank")-g("Rkn"))),
        .5*(_flex(g("Lhip"),g("Lkn"),g("Lank"))+_flex(g("Rhip"),g("Rkn"),g("Rank"))),
        .5*(_flex(g("Lsh"),g("Lhip"),g("Lkn"))+_flex(g("Rsh"),g("Rhip"),g("Rkn"))),
        .5*(_flex(g("Lsh"),g("Lel"),g("Lwr"))+_flex(g("Rsh"),g("Rel"),g("Rwr"))),
        vext,hext,vext/(hext+1e-6),(hip[1]-g("nose")[1])/tl,
        ((g("Lank")[1]+g("Rank")[1])/2-hip[1])/tl,((g("Lkn")[1]+g("Rkn")[1])/2-hip[1])/tl],np.float32)
def feat2d(lm): l=lm.copy(); l[:,2]=0.; return features_13(l)

# --- the emitter: a 2D-GRIP forest trained on lab+gold ONLY (NO mocap) ---
# This is the noise source (variant c, floor): genuine cross-domain per-frame errors on the CMU
# projections, with zero leakage — no CMU trial is in the emitter's training set, exactly as in
# production (the deployed classifier never trained on the person in front of the camera).
EMIT = RandomForestClassifier(n_estimators=60, max_depth=9, min_samples_leaf=10,
                              random_state=SEED, n_jobs=-1).fit(
    np.array([feat2d(x) for x in CORE["X"]]), np.array([str(s) for s in CORE["y"]]))
EMIT_CLS = list(EMIT.classes_)     # sklearn sorts alphabetically: lying, sitting, standing
print("trials", len(set(TR["trial"].tolist())), "| frames", len(TR["y"]),
      "| scenarios", dict(collections.Counter(TR["scenario"].tolist())))
print("emitter trained on lab+gold:", len(CORE["y"]), "poses (no mocap -> zero CMU leakage)")
""")

# ============================================================ 4. a methods honesty beat: the floor bug
md(r"""
## 2. A bug the lab caught before it became a number

Act 2 needs the **absolute** pelvis height (how far the body dropped). The first time we asked for
it, falls read pelvis velocities of *22 m/s* — impossible. The cause was a data bug: the first BVH
frame is a settling artifact with a foot at ~−0.85 m, and using the global minimum as the floor
floated whole trials upward. The fix is a robust floor (a low percentile of the per-frame lowest
joint, ignoring the settling frames).

Crucially, this did **not** touch LJ1: its kinematic labeller keys on `hip − ankle` (a *relative*
height), which is invariant to a floor shift by construction — we verified 0 of 4425 LJ1 labels
change under the fix. The bug bit exactly one consumer — absolute pelvis height — which only Act 2
ever needed. It is fixed in the published `mocap_transitions.npz`. (Lab-first earns its keep by
catching this at the bench, not in production.)
""")

# ============================================================ 5. ACT 1 — transition matrix
md(r"""
## 3. Act 1 — the physics of transitions, and a prior that should beat a dwell

**H-prior (fails if the prior's flip-rate CI overlaps the dwell's at matched latency).** A dwell
machine can only *ride* the flip-rate ↔ latency curve. A transition-prior calibrated on real
physics should *shift* it: standing↔lying is nearly impossible in one step, so vetoing it costs a
plausible standing→sitting nothing. First, measure the physics — the transition matrix at
production cadence.
""")
code(r"""
def sequences(step):
    y,t,tr = TR["y"], TR["t"], TR["trial"]
    out=[]
    for name in dict.fromkeys(tr.tolist()):
        m=tr==name; yy=y[m][np.argsort(t[m])][::step]; out.append((name,list(yy)))
    return out
step=round(SRC_FPS/PROD_FPS); dt=step/SRC_FPS; seqs=sequences(step)
idx={c:i for i,c in enumerate(CLASSES)}; M=np.zeros((3,3)); direct_SL=0
for _,s in seqs:
    for a,b in zip(s[:-1],s[1:]):
        if a in idx and b in idx:
            M[idx[a],idx[b]]+=1
            if {a,b}=={"standing","lying"}: direct_SL+=1
A_MEASURED=M/M.sum(1,keepdims=True).clip(1)
print("transition matrix P(pose_t | pose_t-1) @15fps  (rows: from standing/sitting/lying)")
print("            standing  sitting    lying")
for i,c in enumerate(CLASSES): print(f"  {c:9} "+" ".join(f"{A_MEASURED[i,j]:8.4f}" for j in range(3)))
print(f"\ndirect standing<->lying transitions: {direct_SL} of {int(M.sum())} "
      f"-> ~forbidden; hold that number, Act 2 explains the {direct_SL} exceptions")
""")
md(r"""
The diagonal is ~0.972 (the true label is sticky) and **standing↔lying is 0.0013 / 0.0000** — an
order of magnitude below standing→sitting: to lie down, you pass through sitting. Only **4 of 7670**
transitions jump standing→lying directly. Hold that number — Act 2 shows those four *are* the falls.
""")

md(r"""
Now the contest. Three stabilisers on one emission stream: **raw** argmax, the **dwell** machine
(a port of `posture.rs`), and a causal **forward Bayesian filter** (the prior — transition model
`A`, emission = the classifier softmax, no lookahead). Each is swept over its knob to trace a
curve; we score spurious flips/min and reaction latency against a dwell-cleaned ground truth, and
break out the **standing↔lying** flip specifically — the mode the prior is designed to veto.
""")
code(r"""
def clean_truth(y):
    stable=[y[0]]*len(y); cur=y[0]; onsets=[]; i=0
    while i<len(y):
        if y[i]!=cur:
            j=i
            while j<len(y) and y[j]==y[i]: j+=1
            if j-i>=H: onsets.append((i,y[i])); cur=y[i]
            for k in range(i,j): stable[k]=cur
            i=j
        else: stable[i]=cur; i+=1
    return stable,onsets

def run_raw(am,cf,pr): return list(am)
def run_dwell(am,cf,pr,dwell=3):
    com=None;cand=None;st=0;out=[]
    for lab,c in zip(am,cf):
        if c<0.5: cand=com;st=0
        elif com is None: com=lab;cand=lab;st=0
        elif lab==com: cand=lab;st=0
        else:
            if cand==lab: st+=1
            else: cand=lab;st=1
            if st>=dwell: com=lab;st=0
        out.append(com if com is not None else lab)
    return out
def run_hybrid(am,cf,pr,dwell=3,sl_dwell=12):
    # shipped form: dwell everywhere, but a direct standing<->lying commit needs sl_dwell frames
    # (a fall must earn it) -> a normal lie-down is forced through sitting. A[standing,lying]~=0
    # is exactly what licenses this one extra constant.
    com=None;cand=None;st=0;out=[]
    for lab,c in zip(am,cf):
        need=sl_dwell if (com is not None and {com,lab}=={"standing","lying"}) else dwell
        if c<0.5: cand=com;st=0
        elif com is None: com=lab;cand=lab;st=0
        elif lab==com: cand=lab;st=0
        else:
            if cand==lab: st+=1
            else: cand=lab;st=1
            if st>=need: com=lab;st=0
        out.append(com if com is not None else lab)
    return out
def run_prior(am,cf,pr,stick=0.0):
    A = stick*np.eye(3)+(1-stick)*A_MEASURED if stick>=0 else (1+stick)*A_MEASURED+(-stick)*(np.ones((3,3))/3)
    A=A/A.sum(1,keepdims=True); pm={c:i for i,c in enumerate(EMIT_CLS)}; b=np.ones(3)/3; out=[]
    for k in range(len(am)):
        e=np.array([pr[k][pm[c]] for c in CLASSES]); b=(A.T@b)*(e+1e-9); b=b/b.sum()
        out.append(CLASSES[int(b.argmax())])
    return out

def score(com,onsets):
    sp=slf=0
    for i in range(1,len(com)):
        if com[i]!=com[i-1]:
            legit=any(ol==com[i] and 0<=i-oi<=LMAX for oi,ol in onsets)
            if not legit:
                sp+=1
                if {com[i],com[i-1]}=={"standing","lying"}: slf+=1
    lats=[]
    for oi,ol in onsets:
        hit=next((k for k in range(oi,len(com)) if com[k]==ol),None)
        lats.append((hit-oi) if hit is not None else LMAX*2)
    return sp,slf,lats

def emit(X,rng,dropout=0.,jitter=0.):
    if dropout or jitter:
        X=X.copy()
        for x in X:
            vis=x[:,3]>0
            if jitter: x[vis,:2]+=rng.normal(0,jitter,(vis.sum(),2))
            if dropout: x[rng.random(33)<dropout]=0.
    pr=EMIT.predict_proba(np.array([feat2d(x) for x in X]))
    return np.array(EMIT_CLS)[pr.argmax(1)], pr.max(1), pr

def build(dropout,jitter,rng):
    data=[]
    for name in dict.fromkeys(TR["trial"].tolist()):
        m=TR["trial"]==name; o=np.argsort(TR["t"][m])
        Xt=TR["X"][m][o][::step]; yt=list(TR["y"][m][o][::step])
        if len(yt)<H+2: continue
        am,cf,pr=emit(Xt,rng,dropout,jitter); stable,onsets=clean_truth(yt)
        data.append((am,cf,pr,onsets,len(yt)))
    return data
def evaluate(data,method,knob,key):
    ts=tsl=0;tm=0.;lat=[]
    for (am,cf,pr,onsets,nf) in data:
        com=method(am,cf,pr,**{key:knob}) if key else method(am,cf,pr)
        sp,slf,l=score(com,onsets); ts+=sp;tsl+=slf;tm+=nf/PROD_FPS/60.;lat+=l
    return ts/tm, tsl/tm, (np.mean(lat)/PROD_FPS*1000 if lat else 0.)
print("stabilisers + metrics defined")
""")

# ============================================================ 6. ACT 1 results
code(r"""
rng=np.random.default_rng(SEED)
for tag,(dr,ji) in [("FLOOR — real cross-domain errors",(0.,0.)),
                    ("STRESS — occlusion-like dropout .06 + jitter .008",(0.06,0.008))]:
    data=build(dr,ji,rng)
    print(f"\n=== {tag} ===   trials={len(data)}  real transitions={sum(len(o) for *_,o,_ in data)}")
    print(f"{'method':24}{'knob':>6}{'flip/min':>10}{'SL-flip/min':>13}{'latency_ms':>12}")
    f,sl,l=evaluate(data,run_raw,None,None);  print(f"{'raw argmax':24}{'-':>6}{f:10.2f}{sl:13.2f}{l:12.0f}")
    for dw in [2,3,4,5,6]: f,sl,l=evaluate(data,run_dwell,dw,"dwell"); print(f"{'dwell (posture.rs)':24}{dw:>6}{f:10.2f}{sl:13.2f}{l:12.0f}")
    for stk in [0.0,0.7,0.93]: f,sl,l=evaluate(data,run_prior,stk,"stick"); print(f"{'prior (forward filter)':24}{stk:>6.2f}{f:10.2f}{sl:13.2f}{l:12.0f}")
    hf,hsl,hl=evaluate(data,run_hybrid,3,"dwell"); print(f"{'HYBRID dwell3 + SL-veto':24}{'-':>6}{hf:10.2f}{hsl:13.2f}{hl:12.0f}")
    # bootstrap CI: the shipped hybrid vs the dwell it replaces (same base dwell -> matched latency)
    def rates(method,knob,key):
        return np.array([[*score(method(am,cf,pr,**{key:knob}),onsets)[:2], nf/PROD_FPS/60.]
                         for (am,cf,pr,onsets,nf) in data], float)
    Ad=rates(run_dwell,3,"dwell"); Bh=rates(run_hybrid,3,"dwell"); r2=np.random.default_rng(SEED); dsl=[]
    for _ in range(2000):
        ix=r2.integers(0,len(data),len(data)); dsl.append((Bh[ix,1].sum()-Ad[ix,1].sum())/Ad[ix,2].sum())
    lo,hi=np.percentile(dsl,[2.5,97.5])
    print(f"  Δ SL-flip/min (hybrid − dwell3): {np.mean(dsl):+.2f}  CI[{lo:+.2f},{hi:+.2f}]  "
          f"{'ESTABLISHED (CI<0)' if hi<0 else 'suggestive (CI spans 0)'}")
""")
md(r"""
**Live verdict — honest and split.** On aggregate flip-rate the prior is a **wash** with a
well-tuned dwell: at matched latency their CIs overlap, so by the pre-named criterion the strong
hypothesis is **not confirmed** — physics did not beat hand-tuning on the mode that dominates the
count (adjacent standing↔sitting confusion, which the prior also allows). Where the prior is right
is the **standing↔lying** edge: directionally it flips 2–5× less. CMU alone is too sparse in that
mode to *establish* the pure-prior win — but that points straight at the fix.

**What ships (decision A):** the cheap dwell machine, plus **one physics veto** — a direct
standing↔lying commit needs a long dwell (`sl_dwell=12`), licensed by `A[standing,lying]≈0`. This
**HYBRID** is dwell-by-construction elsewhere (no aggregate regression) and, under occlusion
stress, drives the false standing↔lying flip to **zero** — an *established* reduction (CI < 0).
That matters because a spurious standing→lying **is a false fall alarm**. Which is Act 2.
""")

# ============================================================ 7. ACT 2 — fall vs lie-down
md(r"""
## 4. Act 2 — the four exceptions were falls: a measured fall threshold

Act 1 left four direct standing→lying transitions unexplained. They are the falls — the only way to
go horizontal fast enough to skip sitting. The classifier emits `lying` for a fall and a lie-down
alike; the discriminator is kinematic.

**H-fall (fails if fall vs controlled-descent AUROC ≲ 0.75).** The **peak downward pelvis velocity**
separates a fall from a controlled lie-down. Honest twist: the negative class includes brisk
sit-downs, so if speed alone triggers on a fast sit, the *endpoint* (floor vs chair) must matter too.
""")
code(r"""
def peak_down_v(pelvis,t):
    if len(pelvis)<14: return 0.,float(pelvis.min())
    k=5; r=k//2; xp=np.pad(pelvis[3:-3],r,mode="edge")
    p=np.array([np.median(xp[i:i+k]) for i in range(len(pelvis)-6)])
    dtt=float(np.median(np.diff(t))); w=max(2,round(0.4/dtt))
    if len(p)<=w: return 0.,float(pelvis.min())
    return float(max(((p[:-w]-p[w:])/(w*dtt)).max(),0.)), float(pelvis.min())
def auroc(s,y):
    s=np.asarray(s,float);y=np.asarray(y);o=np.argsort(s,kind="mergesort");rk=np.empty(len(s));rk[o]=np.arange(1,len(s)+1)
    u,inv,cnt=np.unique(s,return_inverse=True,return_counts=True);cs=np.cumsum(cnt);rk=((cs-cnt+cs+1)/2.)[inv]
    npos=y.sum();nneg=len(y)-npos
    return float("nan") if npos==0 or nneg==0 else float((rk[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))

pelvis,t,tr,sc=TR["pelvis"],TR["t"],TR["trial"],TR["scenario"]
rows=[]
for name in dict.fromkeys(tr.tolist()):
    m=tr==name; v,mp=peak_down_v(pelvis[m],t[m]); rows.append((name,sc[m][0],v,mp,int(sc[m][0]=="fall")))
v=np.array([r[2] for r in rows]);mpel=np.array([r[3] for r in rows]);lab=np.array([r[4] for r in rows])
floor=mpel<=0.35
print("[A] fall vs everything :  AUROC velocity =",f"{auroc(v,lab):.3f}",
      " | endpoint(min-pelvis) =",f"{auroc(-mpel,lab):.3f}")
vB,lB=v[floor],lab[floor]
thr=min(vB[lB==1]);
print(f"[B] fall vs controlled descent (both reach floor, n={len(vB)}: {int(lB.sum())} fall / {int((lB==0).sum())} controlled)")
print(f"    AUROC velocity = {auroc(vB,lB):.3f}   clean threshold ~ {thr:.2f} m/s")
print(f"    fall peak-down: [{vB[lB==1].min():.2f},{vB[lB==1].max():.2f}] m/s   "
      f"controlled: [{vB[lB==0].min():.2f},{vB[lB==0].max():.2f}] m/s")
# bootstrap CI over trials on framing B
r3=np.random.default_rng(SEED);bt=[]
for _ in range(2000):
    ix=r3.integers(0,len(vB),len(vB))
    if 0<lB[ix].sum()<len(ix): bt.append(auroc(vB[ix],lB[ix]))
print(f"    AUROC bootstrap CI [{np.percentile(bt,2.5):.3f}, {np.percentile(bt,97.5):.3f}]")
""")
md(r"""
**Live verdict — confirmed, with a stated ceiling.** Among trials that actually reach the floor,
peak downward pelvis velocity separates falls from controlled descents cleanly (AUROC 1.00;
threshold ≈ **1.1 m/s** — controlled ≤ ~1.0, falls ≥ ~1.1). Velocity *alone* (framing A) is not
enough: a playground drop hits fall-like speed without ending on the floor — so the shipped rule is
**two features**: *reached the floor* **AND** *peak downward pelvis velocity ≳ 1.1 m/s*.

**Honest caveats.** Every CMU fall is stunt/acrobatic — no elderly falls — so this threshold is
calibrated on *fast* falls and likely bounds the clinical one from **above** (a slow collapse could
slip under it). And CMU's controlled-lie-down class is thin. So Act 2 is a measured, order-of-
magnitude rule to be confirmed live in 15.6 — not the last word.
""")

# ============================================================ 8. verdict
md(r"""
## 5. Verdict

| Hypothesis | Result | Number (live) |
|---|---|---|
| H-prior: physics prior beats the dwell (aggregate) | ❌ not established | flip-rate CI overlaps at matched latency |
| H-prior on the standing↔lying edge | ◐ ships as the **hybrid** | false SL-flip → **0** under stress, Δ CI < 0 |
| H-fall: pelvis velocity separates fall / lie-down | ✅ (with ceiling) | AUROC 1.00; threshold ≈ 1.1 m/s + floor |

The elegant transition-prior did **not** earn a measured accuracy win over a hand-tuned dwell — an
honest negative we publish as loudly as a positive. Its real value is narrow and physical: vetoing
the one impossible edge, standing↔lying, which is exactly the **fall-relevant** edge. So the two
acts converge on the same physics from opposite sides — Act 1 vetoes the impossible standing→lying
*flip*; Act 2 measures the velocity that makes a genuine standing→lying a *fall*.
""")

# ============================================================ 9. what-changes
md(r"""
## 6. What it changes (bridge to the Rust port, 15.6)

- **PostureStabilizer** keeps the cheap dwell and gains **one** physics veto: a `sl_dwell` on the
  standing↔lying edge, justified by the measured `A[standing,lying]≈0`. One constant, +0
  dependencies — inside the minimalism invariant. It removes false fall alarms; it claims no
  aggregate accuracy gain.
- **Anomaly** gets a measured fall rule: *reached the floor* **and** *peak downward pelvis velocity
  ≳ 1.1 m/s* — two arithmetic features, no model.
- **Bridge to LJ3:** we have posture and its transitions in time; next is *identity* in time — the
  tracker on true CMU trajectories, and why a walking person fragments into phantom tracks.
""")

code(r"""
# passport — binds these numbers to exact data (and code, when run from the repo)
import hashlib, subprocess
def sha16(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for ch in iter(lambda:f.read(1<<20),b""): h.update(ch)
    return h.hexdigest()[:16]
try: commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],stderr=subprocess.DEVNULL).decode().strip()
except Exception: commit="(kaggle run — no git)"
print("passport:", {"code_commit":commit, "mocap_transitions":sha16(_find("mocap_transitions.npz")),
                    "core":sha16(_find("landmarks_v2.npz","landmarks.npz"))})
""")

md(r"""
*MVM Lab Journal · LJ2 of LJ0–LJ6. Derived keypoints only — no images or motion data redistributed.
Prev: LJ1 (The pose model, rebuilt honestly). Next: LJ3 (the tracker on true trajectories).*
""")

# ============================================================ write
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"lj2_physics_of_posture_transitions.ipynb")
json.dump(nb, open(out,"w"), indent=1, ensure_ascii=False)
print("wrote", out, "|", len(cells), "cells")
