#!/usr/bin/env python3
"""LJ5 Act B1 — emotion passport: EmotiEffLib B0 on RAF-DB held-out (phase 15.5).

Honest held-out accuracy for the emotion expert we ship but never trained. The model
(`enet_b0_8_best_vgaf.onnx`) was trained on AffectNet+VGAF → RAF-DB is a clean held-out
(no train contamination). This is also the regression guard our phase-10.1 NCHW bug would
have tripped: if our wiring reproduces near-uniform softmax / chance balanced-acc, that's a
PREPROCESSING BUG, not a bad model — so the negative control runs first.

Preprocessing is BIT-FAITHFUL to prod (`mvm-emotion/src/lib.rs::preprocess_chw`):
  resize 224² NEAREST, RGB, (px/255 − ImageNet_mean)/ImageNet_std, planar CHW, softmax over 8.
RAF-DB ships pre-ALIGNED 100² crops → we skip prod's YuNet+Umeyama align stage and feed the
crop directly: this passports the CLASSIFIER HEAD (design caveat; prod adds alignment upstream).

RAF-DB label(1..7) → EmotiEffLib 8-class output idx (Contempt idx 1 has no RAF GT → argmax-over-7
for the headline; free argmax-8 Contempt-rate reported separately as an honesty sink).

Usage: venv/bin/python spike_pose/lj5_emotion.py [--data /home/dawoh/datasets/rafdb] [--json out.json]
"""
import os, sys, json, glob, argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bench import balanced_accuracy, per_class_recall, confusion, bootstrap_ci, passport  # frozen protocol

MODEL = os.path.join(ROOT, "mvm", "models", "enet_b0_8_best_vgaf.onnx")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# EmotiEffLib output order (mvm-core/src/emotion.rs EMOTION_LABELS).
MODEL_LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
# RAF-DB basic label (1..7) → canonical emotion name.
RAF_NAME = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happiness", 5: "Sadness", 6: "Anger", 7: "Neutral"}
# The 7 RAF classes in a fixed reporting order (Contempt excluded — no RAF GT).
CLASSES7 = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
KEEP7 = [MODEL_LABELS.index(c) for c in CLASSES7]   # model indices with RAF correspondence
CONTEMPT = MODEL_LABELS.index("Contempt")           # = 1


def preprocess_batch(paths):
    """Bit-faithful to mvm-emotion::preprocess_chw — NEAREST 224, ImageNet norm, planar CHW, RGB."""
    out = np.empty((len(paths), 3, 224, 224), np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((224, 224), Image.NEAREST)
        a = np.asarray(im, np.float32) / 255.0          # H,W,3 RGB
        a = (a - MEAN) / STD
        out[i] = np.transpose(a, (2, 0, 1))             # → 3,H,W (planar CHW)
    return out


def load_test(data_root, split="test"):
    paths, y = [], []
    for raf_label, name in RAF_NAME.items():
        for p in glob.glob(os.path.join(data_root, "DATASET", split, str(raf_label), "*")):
            paths.append(p); y.append(name)
    return paths, np.array(y)


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/dawoh/datasets/rafdb")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL, sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    paths, y = load_test(a.data, a.split)
    print(f"# LJ5 B1 — emotion (EmotiEffLib B0) on RAF-DB {a.split}: {len(paths)} imgs")

    probs = np.empty((len(paths), 8), np.float32)
    for s in range(0, len(paths), a.batch):
        chunk = paths[s:s + a.batch]
        logits = sess.run(None, {iname: preprocess_batch(chunk)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs[s:s + len(chunk)] = e / e.sum(1, keepdims=True)

    # ── Negative control FIRST: is the wiring healthy or collapsed to chance? ──
    maxp = probs.max(1)
    ent = (-(probs * np.log2(np.clip(probs, 1e-12, 1))).sum(1) / 3.0)   # normalised over 8
    contempt_rate = float((probs.argmax(1) == CONTEMPT).mean())
    print(f"\n[negative control] mean max-prob={maxp.mean():.3f} (chance 1/8=0.125), "
          f"mean norm-entropy={ent.mean():.3f} (uniform=1.0)")
    print(f"[negative control] free argmax-8 → Contempt predicted on {contempt_rate:.1%} of RAF faces (no GT → sink)")
    healthy = maxp.mean() > 0.25 and ent.mean() < 0.9
    print(f"[negative control] wiring: {'HEALTHY (peaked, ≫chance)' if healthy else '⚠ COLLAPSED — preprocessing bug?'}")

    # ── Headline: argmax over the 7 RAF-corresponding classes ──
    pred7 = np.array([MODEL_LABELS[KEEP7[i]] for i in probs[:, KEEP7].argmax(1)])
    bal = balanced_accuracy(y, pred7, classes=CLASSES7)
    pcr = per_class_recall(y, pred7, classes=CLASSES7)
    groups = np.arange(len(y))   # no subject-id in this RAF release → image-level bootstrap
    pt, lo, hi = bootstrap_ci(y, pred7, groups, metric=lambda yy, pp: balanced_accuracy(yy, pp, CLASSES7))
    M = confusion(y, pred7, classes=CLASSES7)
    overall = float((pred7 == y).mean())

    print(f"\nbalanced-acc (mean per-class recall, 7-cls) = {bal:.3f}  bootstrap CI [{lo:.3f},{hi:.3f}]")
    print(f"overall accuracy = {overall:.3f} (chance 1/7={1/7:.3f})")
    print("per-class recall:")
    for c in CLASSES7:
        n = int((y == c).sum())
        print(f"  {c:10s} recall={pcr[c]:.3f}  (n={n})")
    print("\nconfusion (rows=true, cols=pred), order " + ",".join(CLASSES7))
    print(M)

    out = {"passport": passport(), "n": len(paths), "split": a.split,
           "balanced_acc": bal, "ci": [lo, hi], "overall_acc": overall,
           "per_class_recall": pcr, "confusion": M.tolist(), "classes": CLASSES7,
           "neg_control": {"mean_maxprob": float(maxp.mean()), "mean_entropy": float(ent.mean()),
                           "contempt_sink_rate": contempt_rate, "healthy": bool(healthy)},
           "caveats": ["aligned crops bypass prod YuNet+Umeyama (classifier-head passport)",
                       "no subject-id → image-level bootstrap", "Fear n=74 → wide CI",
                       "held-out: model trained on AffectNet+VGAF, RAF-DB unseen"]}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2); print(f"\nwrote {a.json}")
    return out


if __name__ == "__main__":
    main()
