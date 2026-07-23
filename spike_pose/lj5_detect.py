#!/usr/bin/env python3
"""LJ5 Act B3 — detect passport: YuNet (face) + YOLO26n (person) on OpenImages val (phase 15.5).

"Where feasible" per plan. OpenImages detection eval is hard (federated labels → not every object
is annotated; group-of boxes cover crowds). So the HONEST headline is **recall@IoU0.5** on labeled
non-group boxes: a labeled box IS a real object, so a miss is a true miss (recall is robust to
federated labeling). Precision is reported WITH a caveat (an unlabeled-but-real detection scores as
a false positive → precision is a lower bound, not a clean number).

Two detectors, both cross-domain (YuNet trained on WIDER FACE, YOLO26n on COCO — OI is neither):
  * YuNet face   → OI Human-face boxes  (/m/0dzct)   — the in-chain face detector
  * YOLO26n person → OI Person boxes    (/m/01g317)  — object-detector capability proxy
    (note: prod uses BlazePose person_detection for the pose ROI, not YOLO; this passports YOLO's raw
     person recall as a representative object class — cleanest OI class to eval)

YOLO preprocessing is BIT-FAITHFUL to mvm-scene (letterbox 640, grey 114/255, RGB /255 NCHW;
decode [x1,y1,x2,y2,score,class]; canvas_to_norm undoes letterbox). person = class 0.

Usage: venv/bin/python spike_pose/lj5_detect.py [--n 400] [--json out.json]
"""
import os, sys, csv, argparse, urllib.request, concurrent.futures as cf, collections
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bench import passport
from openimages_fetch import cached, BBOX          # reuse GCS-cached CSV + path map

YOLO = os.path.join(ROOT, "mvm", "models", "yolo_object_detect.onnx")
YUNET = os.path.join(ROOT, "mvm", "models", "face_detection_yunet_2023mar.onnx")
IMGDIR = "/home/dawoh/datasets/openimages_val"
S3 = "https://open-images-dataset.s3.amazonaws.com"
PERSON, FACE = "/m/01g317", "/m/0dzct"
PROMINENT_H = 0.10                                   # box height ≥10% of image = "prominent"


def load_gt(split="validation"):
    """Per-image GT boxes (normalized xyxy) for Person and Face, excluding group-of / depiction."""
    path = cached(BBOX[split])
    gt = collections.defaultdict(lambda: {"person": [], "face": []})
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["IsGroupOf"] == "1" or r["IsDepiction"] == "1":
                continue
            cls = "person" if r["LabelName"] == PERSON else "face" if r["LabelName"] == FACE else None
            if cls is None:
                continue
            gt[r["ImageID"]][cls].append(
                [float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"])])
    return gt


def download(ids, split="validation"):
    os.makedirs(IMGDIR, exist_ok=True)
    def one(img):
        out = os.path.join(IMGDIR, f"{img}.jpg")
        if os.path.exists(out):
            return out
        try:
            urllib.request.urlretrieve(f"{S3}/{split}/{img}.jpg", out); return out
        except Exception:
            return None
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        return [p for p in ex.map(one, ids) if p]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(dets, gts, thr=0.5):
    """Greedy score-ordered match. dets = [(box,score)] sorted desc. Returns (tp_det, matched_gt set)."""
    used = set(); tp = 0
    for box, _ in dets:
        best, bj = thr, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(box, g)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj); tp += 1
    return tp, used


# ── YOLO faithful decode (mvm-scene) ──────────────────────────────────────────
def letterbox(w, h, t=640):
    s = min(t / w, t / h)
    return s, (t - w * s) / 2, (t - h * s) / 2


def yolo_person(sess, img_bgr, score_thr=0.35):
    h, w = img_bgr.shape[:2]
    s, px, py = letterbox(w, h)
    nw, nh = round(w * s), round(h * s)
    resized = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    oy, ox = round(py), round(px)
    canvas[oy:oy+nh, ox:ox+nw] = resized
    x = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
    out = sess.run(None, {"images": x})[0][0]           # [300,6]
    dets = []
    for row in out:
        sc, cid = row[4], int(round(row[5]))
        if sc < score_thr or cid != 0:                   # person = class 0
            continue
        b = [((row[0]-px)/s)/w, ((row[1]-py)/s)/h, ((row[2]-px)/s)/w, ((row[3]-py)/s)/h]
        dets.append(([min(max(v, 0), 1) for v in b], float(sc)))
    return sorted(dets, key=lambda d: -d[1])


def yunet_faces(det, img_bgr):
    h, w = img_bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img_bgr)
    out = []
    if faces is not None:
        for f in faces:
            x, y, bw, bh = f[:4]
            out.append(([x/w, y/h, (x+bw)/w, (y+bh)/h], float(f[-1])))
    return sorted(out, key=lambda d: -d[1])


def evaluate(cls, det_fn, images, gt):
    tot_gt = tot_gt_p = tot_det = tp = tp_p = 0
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        iid = os.path.splitext(os.path.basename(path))[0]
        gts = gt[iid][cls]
        if not gts:
            continue
        dets = det_fn(img)
        t, matched = match(dets, gts)
        tp += t; tot_gt += len(gts); tot_det += len(dets)
        # prominent stratum (box height ≥ PROMINENT_H)
        prom_idx = [j for j, g in enumerate(gts) if (g[3]-g[1]) >= PROMINENT_H]
        tot_gt_p += len(prom_idx); tp_p += sum(1 for j in prom_idx if j in matched)
    rec = tp / tot_gt if tot_gt else 0.0
    rec_p = tp_p / tot_gt_p if tot_gt_p else 0.0
    prec = tp / tot_det if tot_det else 0.0
    return {"recall_all": rec, "recall_prominent": rec_p, "precision_caveated": prec,
            "n_gt": tot_gt, "n_gt_prominent": tot_gt_p, "n_det": tot_det}


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    print(f"# LJ5 B3 — detect passport (YuNet face + YOLO26n person) on OpenImages {a.split}")
    gt = load_gt(a.split)
    # Sample each class from images carrying ITS OWN non-group boxes (strided, reproducible),
    # so the face passport isn't starved by incidental faces in person-sampled images.
    def sample(cls):
        imgs = sorted(i for i, g in gt.items() if g[cls])
        step = max(1, len(imgs) // a.n)
        return imgs[::step][:a.n], len(imgs)
    p_pick, n_p = sample("person"); f_pick, n_f = sample("face")
    print(f"  Person: {n_p} val images → sampling {len(p_pick)};  Face: {n_f} → sampling {len(f_pick)}")
    images = download(sorted(set(p_pick) | set(f_pick)), a.split)
    print(f"  downloaded {len(images)} images (union)")

    so = ort.SessionOptions(); so.log_severity_level = 3
    yolo = ort.InferenceSession(YOLO, sess_options=so, providers=["CPUExecutionProvider"])
    yunet = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.6, 0.3, 5000)

    res = {}
    res["yolo_person"] = evaluate("person", lambda im: yolo_person(yolo, im), images, gt)
    res["yunet_face"] = evaluate("face", lambda im: yunet_faces(yunet, im), images, gt)

    for k, r in res.items():
        print(f"\n[{k}] recall@IoU0.5: prominent(h≥{PROMINENT_H}) = {r['recall_prominent']:.3f} "
              f"({r['n_gt_prominent']} gt) | all = {r['recall_all']:.3f} ({r['n_gt']} gt)")
        print(f"[{k}] precision (federated-caveat, lower bound) = {r['precision_caveated']:.3f} "
              f"({r['n_det']} dets)")

    out = {"passport": passport(), "split": a.split, "n_images": len(images), "results": res,
           "caveats": ["recall is the honest metric; precision is a LOWER BOUND (OI federated labels "
                       "→ real-but-unlabeled detections count as FP)",
                       "cross-domain: YuNet=WIDER FACE, YOLO26n=COCO, neither is OpenImages",
                       "prod uses BlazePose person_detection for the pose ROI, not YOLO — this is a "
                       "YOLO object-recall proxy on the cleanest OI class",
                       "YOLO score_thr=0.35 (model default); IoU match 0.5, greedy by score"]}
    if a.json:
        import json; json.dump(out, open(a.json, "w"), indent=2); print(f"\nwrote {a.json}")
    return out


if __name__ == "__main__":
    main()
