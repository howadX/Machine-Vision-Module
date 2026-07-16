#!/usr/bin/env python3
"""LJ5 Act B2 — identity passport: SFace on LFW verification (phase 15.5).

Honest verification accuracy for the identity expert we ship. SFace (Apache) is the priciest
in-chain model (36.9 MiB, 5.2 ms) → its accuracy headroom decides the Act-C requant verdict.

TWO paths measured, because prod DOES NOT align the face before SFace (pipeline.rs:470 feeds the
raw YuNet bbox crop straight to embed(), unlike emotion which aligns via align_face). So:
  * PROD (unaligned): YuNet bbox → resize 112 → SFace   — what ships today
  * ALIGNED         : YuNet 5-landmark → our Umeyama warp to 112 → SFace   — the 15.6 fix candidate
The delta = the measured cost of not aligning identity (a +0-weight fix, reuses align_face).

SFace preprocessing is BIT-FAITHFUL to prod (`mvm-identity/src/lib.rs::preprocess`):
  planar CHW, **BGR**, **raw [0,255]** (model bakes in (x−127.5)/128), L2-normalise output, cosine match.
Resize is bilinear (≈ prod Triangle). Detection = cv2.FaceDetectorYN on the SAME YuNet ONNX;
its 5 landmarks share order with our ArcFace template (image-left-eye first). Negative control:
LFW acc ≈ 0.5 = broken embedding (the #5 disaster), not a weak model.

Usage: venv/bin/python spike_pose/lj5_identity.py [--lfw .../lfw_home] [--json out.json]
"""
import os, sys, json, argparse
import numpy as np, cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bench import passport

SFACE = os.path.join(ROOT, "mvm", "models", "face_recognition_sface_2021dec.onnx")
YUNET = os.path.join(ROOT, "mvm", "models", "face_detection_yunet_2023mar.onnx")
# ArcFace 5-point template on 112² (mvm-emotion/src/lib.rs ALIGN_TEMPLATE_112), image coords.
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def similarity_matrix(src, dst):
    """Closed-form Umeyama (uniform scale+rot+trans, no reflection) src→dst, as mvm-emotion.
    Returns 2×3 cv2 affine [[a,-b,tx],[b,a,ty]] or None if degenerate."""
    s, d = src.mean(0), dst.mean(0)
    u, v = src - s, dst - d
    num_re = (u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]).sum()
    num_im = (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()
    den = (u * u).sum()
    if den < 1e-9:
        return None
    a, b = num_re / den, num_im / den
    tx = d[0] - a * s[0] + b * s[1]
    ty = d[1] - b * s[0] - a * s[1]
    return np.array([[a, -b, tx], [b, a, ty]], np.float32)


def sface_embed(sess, crop_bgr112):
    """Bit-faithful SFace preprocess: planar CHW, BGR, raw [0,255]; L2-normalise output."""
    x = np.transpose(crop_bgr112.astype(np.float32), (2, 0, 1))[None]  # 1,3,112,112 (already BGR)
    emb = sess.run(None, {"data": x})[0][0]
    n = np.linalg.norm(emb)
    return emb / n if n > 1e-8 else emb


def make_crops(img_bgr, det):
    """Return (unaligned112, aligned112) BGR crops from a YuNet detection row, or center fallback."""
    h, w = img_bgr.shape[:2]
    if det is None:
        c = img_bgr[h // 5:h * 4 // 5, w // 5:w * 4 // 5]          # center fallback
        un = cv2.resize(c, (112, 112), interpolation=cv2.INTER_LINEAR)
        return un, un
    x, y, bw, bh = det[:4]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
    crop = img_bgr[y0:y1, x0:x1]
    un = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR) if crop.size else \
        cv2.resize(img_bgr, (112, 112))
    lm = det[4:14].reshape(5, 2).astype(np.float32)
    M = similarity_matrix(lm, TEMPLATE)
    al = cv2.warpAffine(img_bgr, M, (112, 112), flags=cv2.INTER_LINEAR) if M is not None else un
    return un, al


def parse_pairs(pairs_txt, root):
    """Standard LFW pairs.txt → list of (pathA, pathB, label, fold). 10 folds × (300 same+300 diff)."""
    def p(name, n):
        return os.path.join(root, "lfw_funneled", name, f"{name}_{int(n):04d}.jpg")
    lines = open(pairs_txt).read().splitlines()
    nfold, nper = map(int, lines[0].split())
    out, i = [], 1
    for f in range(nfold):
        for _ in range(nper):                       # matched
            a, n1, n2 = lines[i].split(); i += 1
            out.append((p(a, n1), p(a, n2), 1, f))
        for _ in range(nper):                       # mismatched
            a, n1, b, n2 = lines[i].split(); i += 1
            out.append((p(a, n1), p(b, n2), 0, f))
    return out


def tar_at_far(cos, label, far_target=1e-3):
    same, diff = cos[label == 1], cos[label == 0]
    thr = np.quantile(diff, 1 - far_target)          # threshold giving FAR = far_target
    return float((same > thr).mean()), float(thr)


def tenfold_acc(cos, label, fold):
    """Canonical LFW: threshold from the other 9 folds (max train acc), applied to the held-out fold."""
    accs = []
    grid = np.linspace(-1, 1, 4001)
    for f in range(10):
        tr, te = fold != f, fold == f
        best_t = grid[np.argmax([((cos[tr] > t) == label[tr]).mean() for t in grid])]
        accs.append(((cos[te] > best_t) == label[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", default="/home/dawoh/datasets/lfw_cache/lfw_home")
    ap.add_argument("--sface", default=SFACE, help="SFace ONNX (override for INT8 requant test)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(a.sface, sess_options=so, providers=["CPUExecutionProvider"])
    print(f"# SFace model: {a.sface} ({os.path.getsize(a.sface)/1024:.0f} KiB)")
    yunet = cv2.FaceDetectorYN.create(YUNET, "", (250, 250), 0.6, 0.3, 50)

    pairs = parse_pairs(os.path.join(a.lfw, "pairs.txt"), a.lfw)
    uniq = sorted({p for pr in pairs for p in pr[:2]})
    print(f"# LJ5 B2 — SFace on LFW: {len(pairs)} pairs, {len(uniq)} unique images")

    emb_un, emb_al = {}, {}
    n_nodet = 0
    for k, path in enumerate(uniq):
        img = cv2.imread(path)                       # BGR
        h, w = img.shape[:2]
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(img)
        det = None
        if faces is not None and len(faces):
            det = faces[np.argmax(faces[:, -1])]     # highest score
        else:
            n_nodet += 1
        un, al = make_crops(img, det)
        emb_un[path] = sface_embed(sess, un)
        emb_al[path] = sface_embed(sess, al)
        if (k + 1) % 2000 == 0:
            print(f"  embedded {k+1}/{len(uniq)}")
    print(f"  no-detection fallbacks: {n_nodet}/{len(uniq)}")

    label = np.array([pr[2] for pr in pairs])
    fold = np.array([pr[3] for pr in pairs])
    cos_un = np.array([float(emb_un[pa] @ emb_un[pb]) for pa, pb, _, _ in pairs])
    cos_al = np.array([float(emb_al[pa] @ emb_al[pb]) for pa, pb, _, _ in pairs])

    res = {}
    for name, cos in [("prod_unaligned", cos_un), ("aligned", cos_al)]:
        acc, std = tenfold_acc(cos, label, fold)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(label, cos))
        tar, thr = tar_at_far(cos, label)
        res[name] = {"acc_10fold": acc, "acc_std": std, "auc": auc,
                     "tar_at_far1e-3": tar, "far_thr": thr,
                     "cos_same_mean": float(cos[label == 1].mean()),
                     "cos_diff_mean": float(cos[label == 0].mean())}
        healthy = acc > 0.6
        print(f"\n[{name}] 10-fold acc = {acc:.4f} ± {std:.4f}   AUC = {auc:.4f}   "
              f"TAR@FAR=1e-3 = {tar:.3f}")
        print(f"[{name}] cos same={cos[label==1].mean():.3f} diff={cos[label==0].mean():.3f}  "
              f"wiring {'HEALTHY' if healthy else '⚠ BROKEN (≈chance)'}")

    delta = res["aligned"]["acc_10fold"] - res["prod_unaligned"]["acc_10fold"]
    print(f"\nALIGN HEADROOM (aligned − prod) = {delta:+.4f}  → 15.6 fix candidate (align identity crop)")

    out = {"passport": passport(), "n_pairs": len(pairs), "n_unique": len(uniq),
           "no_detection": n_nodet, "results": res, "align_headroom": delta,
           "caveats": ["prod feeds UNALIGNED bbox crop to SFace (pipeline.rs:470) — 'aligned' is the fix candidate",
                       "LFW-funneled source (congealing-aligned) → align headroom is a LOWER bound vs raw faces",
                       "cv2.FaceDetectorYN postproc (same YuNet ONNX); bilinear resize ≈ prod Triangle"]}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2); print(f"\nwrote {a.json}")
    return out


if __name__ == "__main__":
    main()
