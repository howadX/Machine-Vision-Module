#!/usr/bin/env python3
"""Builder for LJ5 (Kaggle): "Weighing the experts" — MVM Lab Journal 5.

For five phases the product shipped seven third-party ONNX experts (YuNet, SFace, EmotiEffLib,
BlazePose x2, YOLO26n, Depth-Anything, handpose) and NEVER measured either their speed or their
accuracy on an honest bench. We didn't even know our own wiring wasn't scrambling them — until
phase 10.1, when an HWC-into-NCHW preprocessing bug (near-uniform emotion softmax on every face)
was caught by EYE, having shipped since phase 6. This journal is the instrument that would have
caught it on the first run: a latency+weight ledger, and a held-out accuracy passport for each
expert whose first act is a negative control (a number at chance = OUR bug, not a bad model).

Authoring discipline (as LJ4): the heavy code cells are NOT hand-inlined copies — this builder
reads the actual lab passports (spike_pose/lj5_*.py) and embeds them, so published numbers cannot
drift from the repo's. Reproducibility: latency is reference-hardware-bound (stamped with the CPU);
accuracy re-runs live over mounted datasets (ONNX experts + RAF-DB + LFW + OpenImages subset) —
everything mounted because API-pushed Kaggle runs resolve no DNS.

Build: `python _build_lj5.py` -> lj5_weighing_the_experts.ipynb
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.join(HERE, "..", "..", "..", "spike_pose")

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": t.strip("\n").splitlines(keepends=True)})


def src(name, drop_patterns=(), truncate_at=None, replace=()):
    """Read a lab source file, truncate at `truncate_at` (cuts main()/`if __name__` + everything
    after), drop path-constant/cross-import lines, and adapt for the notebook namespace. Keeps the
    reusable function defs intact so the notebook literally runs the repo's code."""
    with open(os.path.join(SPIKE, name)) as fh:
        text = fh.read()
    out = []
    for ln in text.splitlines():
        if truncate_at and re.search(truncate_at, ln):
            break
        if any(re.search(p, ln) for p in drop_patterns):
            continue
        out.append(ln)
    text = "\n".join(out).rstrip() + "\n"
    for a, b in replace:
        text = text.replace(a, b)
    return text


# repo-path constants + cross-module imports to strip (functions kept; bench funcs inlined below)
PATHS = [r"^HERE =", r"^ROOT =", r"^SPIKE =", r"^MODELS =", r"^MODEL =", r"^SFACE =",
         r"^YUNET =", r"^YOLO =", r"^sys\.path", r"^from bench import",
         r"^from openimages_fetch import", r"^from lj5_detect import"]

# ============================================================ 1. Title
md(r"""
# LJ5 — Weighing the experts
### a latency + weight ledger and an honest held-out passport for the models we ship but never trained

*MVM Lab Journal — 5 of a series. Prev: LJ4 (A camera that never lies). Next: LJ6 (Shipping v0.2.0).*

> The logic layers are measured now — pose (LJ1), transitions (LJ2), tracker (LJ3), their
> composition (LJ4). But underneath them sit **seven third-party ONNX experts** the product has
> carried since 2024 without ever measuring their speed or their accuracy on an honest bench. We
> didn't even know our own plumbing wasn't scrambling them: in phase 10.1 an HWC-into-NCHW bug had
> been feeding EmotiEffLib scrambled channels since phase 6 — near-uniform softmax on every face —
> and it was caught **by eye**, not by a number. This journal is the instrument that catches such
> things on the first run. Every accuracy passport opens with a **negative control**: a score at
> chance means OUR wiring is broken, not that the model is bad. Three axes: **how fast, how heavy,
> how accurate** — and one diet: what can the 15.1 results already delete without paying for it.
""")

# ============================================================ 2. Context / hypotheses
md(r"""
## 1. Context — five years of unmeasured experts

MVM is an ensemble of specialists. The *logic* around them is now honest end-to-end. The
*specialists themselves* — YuNet (face), SFace (identity), EmotiEffLib (emotion), BlazePose
(person + pose), YOLO26n (objects), Depth-Anything, handpose — were dropped in, wired up, and
trusted. Two things were never established:

1. **Do they fit the budget?** The invariant is `<50 ms` CPU for the human chain, `$0`, no GPU.
   Nobody had a per-agent latency number; "≈12 ms each" was folklore.
2. **Are they as good as claimed — and does our wiring preserve it?** The 10.1 emotion bug proves
   the danger: a model can be fine and the integration can throw it away silently.

YOLO deserves a special note. It is not periphery. It is the scene's **object channel** — the
persistent substrate fusion reasons over. When no human is in frame the scene is still furnished;
the decorations remain and fusion keeps producing structure from them. So YOLO earns a passport as
weighty as the human stack's.

**Hypotheses, with failure criteria named up front (frozen bench-protocol, Act tables):**

| act | hypothesis | failure criterion (declared before the number) |
|---|---|---|
| A | human chain < 50 ms CPU; per-model attributable | over budget → name the dominant agent as a diet/cadence target |
| B1 | emotion balanced-acc ≫ chance, peaked softmax | chance / near-uniform → OUR preprocessing bug (10.1 redux), stop & fix |
| B2 | SFace LFW ≈ published (~0.99) | LFW ≈ 0.5 → broken embedding (the #5 disaster), stop & fix |
| B3 | detectors recall ≫ 0 cross-domain, decode correct | recall ≈ 0 → broken NMS/letterbox, stop & fix |
| C | diet shrinks footprint at no accuracy or latency cost | any diet that costs accuracy OR latency is rejected, measured |
""")

# ============================================================ 3. Setup / mounts
md(r"""
## 2. Setup — everything mounted (no DNS on API-pushed runs)

The ONNX experts and every bench dataset are mounted (RAF-DB, LFW, an OpenImages detect subset).
Latency is **reference-hardware-bound**: the shipping numbers below were measured on an
i7-12700K; the notebook re-runs the per-model decomposition on Kaggle's CPU as an *echo* (a
different chip → different absolute numbers, same shape).
""")
code(r"""
import os, glob, json, time, sys
import numpy as np

ROOTS = ("/kaggle/input", "/kaggle/working/_ext")
def _find_dir(*names):
    for n in names:
        for root in ROOTS:
            hits = [d for d in glob.glob(f"{root}/**/{n}", recursive=True) if os.path.isdir(d)]
            if hits: return hits[0]
    return None

def _find(*names):
    for n in names:
        for root in ROOTS:
            hits = glob.glob(f"{root}/**/{n}", recursive=True)
            if hits: return hits[0]
    return None

def _unzip_if_needed(keyword, marker):
    # Kaggle usually auto-extracts dataset zips; if not, extract a mounted <keyword>*.zip once.
    import zipfile
    if _find_dir(marker): return
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        if keyword in os.path.basename(z).lower():
            dst = "/kaggle/working/_ext"; os.makedirs(dst, exist_ok=True)
            print("extracting", z); zipfile.ZipFile(z).extractall(dst); return

_unzip_if_needed("lfw", "lfw_home")
MODELS_DIR = _find_dir("mvm-onnx-experts", "models") or "."
RAF_DIR    = _find_dir("rafdb", "raf-db-dataset", "DATASET")
LFW_DIR    = _find_dir("lfw_home", "lfw-verification", "lfw_funneled")
def M(fn):  # resolve a model file from the mounted experts dataset
    p = _find(fn); assert p, f"mount mvm-onnx-experts (missing {fn})"; return p
print("models:", MODELS_DIR, "| raf:", RAF_DIR, "| lfw:", LFW_DIR)
print("inputs:", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "no /kaggle/input")
""")

# frozen bench-protocol metrics, inlined verbatim (the same balanced_accuracy / CI every LJ uses)
code(src("bench.py", drop_patterns=PATHS, truncate_at=r"^if __name__"))

# ============================================================ 4. Act A — latency & weight
md(r"""
## 3. Act A — the latency & weight ledger

Per-model onnxruntime CPU inference (median + p95), and exact file bytes. The human chain runs
its five required experts sequentially per person; the cadenced ones (depth, YOLO) live on their
own threads and are outside the `<50 ms` budget by design (HOL-blocking was removed in phase 8).

**Reference-hardware shipping numbers (i7-12700K, onnxruntime 1.27.0 CPU, 200 timed):**

| agent | role | weight | lat (all cores) | p95 | lat (1 thread) |
|---|---|---:|---:|---:|---:|
| YuNet (face) | chain | 0.23 MiB | 1.88 ms | 2.06 | 7.29 |
| **SFace (identity)** | chain | **36.9 MiB** | **5.20 ms** | 5.41 | 17.52 |
| EmotiEffLib (emotion) | chain | 15.3 MiB | 4.56 ms | 4.87 | 11.44 |
| BlazePose (person det) | chain | 11.4 MiB | 2.19 ms | 2.33 | 8.33 |
| BlazePose (pose) | chain | 5.3 MiB | 2.87 ms | 3.19 | 7.44 |
| YOLO26n (objects) | cadenced | 9.5 MiB | 12.50 ms | 13.22 | 52.72 |
| YOLO-seg (masks) | cadenced | 13.2 MiB | 21.26 ms | 22.24 | 100.0 |
| Depth-Anything | cadenced | 26.0 MiB | 231.8 ms | 240.8 | 939.1 |
| handpose | optional | 3.9 MiB | 0.80 ms | 0.85 | 3.14 |

**Human chain = 16.7 ms all-cores → gate `<50 ms` PASS (3x margin)** (1-thread floor 52.0 ms,
same dominant SFace). Total ONNX = 121.7 MiB. Dominant by latency *and* weight: **SFace (30% of
all ONNX)** — the diet target Act C interrogates. The Kaggle-CPU echo below reproduces the shape.
""")
code(src("lj5_latency.py", drop_patterns=PATHS, truncate_at=r"^def main\("))
code(r"""
# Kaggle-CPU ECHO of the per-model decomposition (chain experts). Absolute ms differ from the
# reference i7-12700K; the point is the shape + that nothing is pathologically slow here.
import onnxruntime as ort
for fn, label, role in ORDER[:5]:
    r = bench_model(M(fn), 20, 100, 0)
    print(f"{label:26s} {os.path.getsize(M(fn))/1024/1024:6.2f} MiB   {r['median_ms']:.2f} ms (Kaggle CPU)")
""")

# ============================================================ 5. Act B1 — emotion
md(r"""
## 4. Act B1 — emotion, on a held-out it never saw

EmotiEffLib `enet_b0_8_best_vgaf` was trained on AffectNet+VGAF, so **RAF-DB is a clean held-out**.
Preprocessing is bit-faithful to prod (`mvm-emotion::preprocess_chw`: NEAREST 224, ImageNet norm,
planar CHW, RGB, softmax over 8). RAF-DB ships pre-aligned crops → this passports the classifier
head (prod adds YuNet+Umeyama alignment upstream). **Negative control first.**
""")
code(src("lj5_emotion.py", drop_patterns=PATHS, truncate_at=r"^def main\("))
code(r"""
from PIL import Image
import onnxruntime as ort
raf = _find_dir("DATASET") or RAF_DIR
data_root = os.path.dirname(raf) if raf and os.path.basename(raf) == "DATASET" else raf
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(M("enet_b0_8_best_vgaf.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
paths, y = load_test(data_root, "test")
probs = np.empty((len(paths), 8), np.float32)
for s in range(0, len(paths), 128):
    ch = paths[s:s+128]; lo = sess.run(None, {iname: preprocess_batch(ch)})[0]
    e = np.exp(lo - lo.max(1, keepdims=True)); probs[s:s+len(ch)] = e / e.sum(1, keepdims=True)
maxp = probs.max(1); ent = (-(probs*np.log2(np.clip(probs,1e-12,1))).sum(1)/3.0)
print(f"[negative control] max-prob {maxp.mean():.3f} (chance .125) | entropy {ent.mean():.3f} (uniform 1) "
      f"-> {'HEALTHY' if maxp.mean()>.25 and ent.mean()<.9 else 'COLLAPSED (bug!)'}")
pred7 = np.array([MODEL_LABELS[KEEP7[i]] for i in probs[:, KEEP7].argmax(1)])
bal = balanced_accuracy(y, pred7, classes=CLASSES7)
print(f"balanced-acc (7-cls) = {bal:.3f} | overall = {(pred7==y).mean():.3f} (chance {1/7:.3f})")
for c in CLASSES7:
    print(f"  {c:10s} recall {per_class_recall(y, pred7, CLASSES7)[c]:.3f}  (n={int((y==c).sum())})")
""")
md(r"""
**Result — hypothesis confirmed.** Negative control HEALTHY (mean max-prob 0.720 vs chance 0.125;
entropy 0.365). **balanced-acc 0.665 [0.643, 0.688], overall 0.726** (chance 0.143) — the crate's
`0.73` claim is not inflated. Strong: Happiness .84 / Sadness .85 / Anger .82 / Fear .74; weak:
Surprise .37 (→Fear) and Disgust .36 (→Anger) — the classic FER confusions, cross-domain, not
wiring. The 10.1 bug does **not** reproduce: our preprocessing is correct. Verdict: **keep.**
""")

# ============================================================ 6. Act B2 — identity
md(r"""
## 5. Act B2 — identity, and a 19.5-point finding

SFace on LFW (10-fold, 6000 pairs), preprocessing bit-faithful to `mvm-identity` (planar CHW, BGR,
raw [0,255], L2). Two paths — because prod does **not** align the face before SFace
(`pipeline.rs:470` feeds the raw YuNet bbox crop, unlike emotion which aligns). So we measure both
the shipped path and the reference-aligned path; the delta is a `+0`-weight fix candidate.
""")
code(src("lj5_identity.py", drop_patterns=PATHS, truncate_at=r"^def main\("))
code(r"""
import cv2, onnxruntime as ort
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(M("face_recognition_sface_2021dec.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
yunet = cv2.FaceDetectorYN.create(M("face_detection_yunet_2023mar.onnx"), "", (250, 250), 0.6, 0.3, 50)
lfw = _find_dir("lfw_home") or LFW_DIR
pairs = parse_pairs(os.path.join(lfw, "pairs.txt"), lfw)
uniq = sorted({p for pr in pairs for p in pr[:2]})
emb_un, emb_al = {}, {}
for path in uniq:
    img = cv2.imread(path); h, w = img.shape[:2]; yunet.setInputSize((w, h))
    _, faces = yunet.detect(img)
    det = faces[np.argmax(faces[:, -1])] if faces is not None and len(faces) else None
    un, al = make_crops(img, det)
    emb_un[path] = sface_embed(sess, un); emb_al[path] = sface_embed(sess, al)
label = np.array([pr[2] for pr in pairs]); fold = np.array([pr[3] for pr in pairs])
for nm, E in [("PROD (unaligned)", emb_un), ("ALIGNED", emb_al)]:
    cos = np.array([float(E[a] @ E[b]) for a, b, _, _ in pairs])
    acc, sd = tenfold_acc(cos, label, fold); tar, _ = tar_at_far(cos, label)
    print(f"[{nm:16s}] 10-fold acc = {acc:.4f} ± {sd:.4f}   TAR@FAR=1e-3 = {tar:.3f}")
""")
md(r"""
**Result — hypothesis confirmed, with a finding.** The **aligned** path hits **0.9783** = the
published SFace-LFW → the model and our SFace wiring are correct (the negative control is taken
hard; the #5 disaster does not reproduce). But the **shipped, unaligned** path is only **0.7833** —
prod feeds SFace a raw bbox crop and leaves **19.5 points** of verification accuracy on the table
(TAR@FAR=1e-3 collapses 0.949 → 0.173, ~5.5x fewer true accepts at fixed false-accept). The fix is
one line — wrap the crop in `align_face(lm5, 112)`, exactly what emotion already does — **+0 weight,
+0 latency, +19.5 points.** (LFW is funneled, so this headroom is a *lower* bound on real faces.)
This is the star of the journal: the same "measure, don't assume" that caught 10.1, now worth 19.5
points for free.
""")

# ============================================================ 7. Act B3 — detect / objects
md(r"""
## 6. Act B3 — the detectors, cross-domain on OpenImages

Recall@IoU0.5 on non-group OI boxes (recall is robust to OI's federated labels; precision is a
lower bound). Cross-domain: YuNet trained on WIDER FACE, YOLO26n on COCO — OI is neither.
Preprocessing bit-faithful to `mvm-scene` (letterbox 640, grey 114, RGB /255; decode xyxy→norm).

**Measured (OpenImages validation):**

- **YuNet (face, in-chain): recall 0.805 prominent / 0.772 all** (543 / 758 GT) — strong, decode
  correct.
- **YOLO26n (person): 0.653 prominent** — nano, cross-domain; prod's person path is BlazePose, not
  YOLO (this is an object-detector proxy).
- **YOLO26n OBJECT passport (18 interaction classes — the scene substrate fusion consumes):**
  MACRO 0.520 / MICRO 0.460 prominent. The **fusion-critical operated objects are strong (0.6–0.8):**
  Mobile phone .81, Remote .78, Laptop .74, Mouse .73, TV .69, Keyboard .60, Clock .68, Vase .67,
  Bowl .66, Wine glass .67. The weak tail is mostly **cross-taxonomy**, not model failure: Book .07
  (OI labels shelf spines; COCO learned held/open books), Table .22 (COCO dining-table ≠ OI Table),
  bags .09–.13. The decode is correct (recall 0.4–0.8, not 0). Verdict: **keep** — YOLO does its
  fusion job (interaction objects) well; the tail is taxonomy + genuine nano limits, a weight↔accuracy
  trade for a future phase, not a wiring bug.

The code (`lj5_detect.py`, `lj5_objects.py`) is in the repo; it runs live over a mounted OI subset.
""")

# ============================================================ 8. Act C — the diet
md(r"""
## 7. Act C — the diet, decided (and one measured negative)

SFace is the heaviest in-chain model (36.9 MiB, 5.2 ms) → the obvious requant target. INT8 dynamic
quantization: **weight 36.9 → 9.4 MiB (−74%), accuracy LOSSLESS** (LFW aligned 0.9790 vs 0.9783,
within CI). It looks free. It is not:
""")
code(r"""
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort, numpy as np, time
src_m = M("face_recognition_sface_2021dec.onnx"); dst_m = "/kaggle/working/sface_int8.onnx"
quantize_dynamic(src_m, dst_m, weight_type=QuantType.QInt8)
def lat(p, thr=0):
    so = ort.SessionOptions(); so.intra_op_num_threads = thr; so.log_severity_level = 3
    s = ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"])
    x = np.random.rand(1,3,112,112).astype(np.float32)
    for _ in range(20): s.run(None, {"data": x})
    t = [time.perf_counter() for _ in range(1)]
    ts = []
    for _ in range(100):
        a = time.perf_counter(); s.run(None, {"data": x}); ts.append((time.perf_counter()-a)*1e3)
    return np.median(ts)
print(f"fp32 {os.path.getsize(src_m)/1024/1024:.1f} MiB  {lat(src_m):.2f} ms")
print(f"int8 {os.path.getsize(dst_m)/1024/1024:.1f} MiB  {lat(dst_m):.2f} ms  <- ~6x SLOWER on CPU")
""")
md(r"""
**On the reference hardware INT8 is ~6x slower (5.10 → 31.63 ms all-cores)** — the classic
dynamic-quant-on-CNN pitfall (DynamicQuantizeLinear + INT8 matmul lose to AVX-tuned fp32 conv).
It would push the human chain 16.7 → ~43 ms and violate the latency gate. **Verdict: REJECT** —
a measured negative. (Static quantization with calibration is a future option, uncertain on CNN CPU.)

The real diet was already banked in 15.1 at the **arithmetic layer**, where footprint is ours to cut:

| layer | v0.1.0 | v0.2.0 candidate | Δ |
|---|---:|---:|---:|
| ONNX experts (9) | 121.7 MiB | 121.7 MiB | 0 (requant rejected) |
| arithmetic (embedded) | 1068 KB (forest 873 + VAE 195) | **222 KB** (forest 220 + Mahalanobis 1.8) | **−79%** |

Total footprint **shrinks**; the human chain stays 16.7 ms. **Weight/latency gate: PASS.**
""")

# ============================================================ 9. Verdict + bridge
md(r"""
## 8. Verdict

| hypothesis | outcome |
|---|---|
| A: chain < 50 ms, attributable | **confirmed** — 16.7 ms all-cores; dominant = SFace (named) |
| B1: emotion ≫ chance, peaked | **confirmed** — 0.665 balanced-acc, wiring HEALTHY (10.1 not reproduced) |
| B2: SFace ≈ published | **confirmed + finding** — aligned 0.978, but prod ships unaligned 0.783 (**+19.5 pts on the table**) |
| B3: detectors recall ≫ 0, decode ok | **confirmed** — YuNet face 0.805; YOLO interaction objects 0.6–0.8 |
| C: diet at no cost | **confirmed** — arithmetic −79%; SFace INT8 **rejected** (measured +6x latency) |

Every negative control was taken: **our wiring breaks no model.** The load-bearing result is that
the biggest quality win in the whole passport costs nothing — **align the identity crop.**

## 9. What it changes — the bridge to LJ6 (shipping v0.2.0)

Into the 15.6 port package, each licensed by a number here:

1. **Align the identity crop** — wrap the SFace input in `align_face(lm5, 112)` (reuse emotion's).
   +0 weight, +0 latency, **+19.5 points** LFW verification. The single best value in the passport.
2. **Diet** — drop the VAE (Mahalanobis won, 15.1), ship the 220 KB forest: arithmetic −79%.
3. **Keep every ONNX expert as-is** — none is both a weight problem and safe to shrink; INT8 SFace
   is a documented reject.
4. **Quality passport in README/AGENT.md** — the numbers above, each traceable to this journal.

What this journal honestly does not do: retrain anything (out of scope); measure the full live
contended chain (per-model isolated + the owner-validated ~16 fps engine is the ceiling); or fix
the cross-taxonomy floor on YOLO's furniture/book classes (a weight↔accuracy trade for later).

Next, **LJ6 — Shipping v0.2.0: the scorecard** — the Rust port of every winning decision from
LJ1–LJ5, and the before/after across all three axes.
""")

# ============================================================ 10. Passport
code(r"""
# ---- passport: bind the numbers to code + data (+ commit when run from the repo) ----
import hashlib, subprocess
def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()[:16]
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    commit = "(kaggle run — no git)"
passport = {"code_commit": commit,
            "sface": sha16(M("face_recognition_sface_2021dec.onnx")),
            "emotion": sha16(M("enet_b0_8_best_vgaf.onnx")),
            "cpu_note": "latency numbers in the tables are reference-hardware (i7-12700K); "
                        "Kaggle echoes above are a different chip"}
print("passport:", passport)
""")

# ---------------------------------------------------------------- write notebook
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "lj5_weighing_the_experts.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "—", len(cells), "cells")
