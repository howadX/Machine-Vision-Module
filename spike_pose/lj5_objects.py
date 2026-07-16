#!/usr/bin/env python3
"""LJ5 Act B3+ — OBJECT passport for YOLO26n on OpenImages val (phase 15.5).

YOLO is not periphery: it is the scene's OBJECT detection+classification channel, the persistent
substrate fusion reasons over. When no human is in frame the scene is still furnished — the
decorations remain, and fusion keeps producing structure from them. So YOLO earns a passport as
weighty as the human stack's: honest cross-domain recall on the interaction-relevant classes it
actually feeds to fusion (mvm-scene RELEVANT_CLASSES).

Metric = recall@IoU0.5 per class (COCO→OI clean mapping), non-group non-depiction GT. Recall is
the robust number under OI federated labels (a labeled box is real; a miss is a true miss).
Cross-domain: YOLO26n trained on COCO, OI is a different distribution/label policy. Preprocessing
BIT-FAITHFUL to mvm-scene (letterbox 640, grey 114/255, RGB /255; decode xyxy→norm). score_thr
0.35 (model default); prod applies a higher per-class filter (A≥0.45 + HIGH_BAR) downstream that
trades recall for precision — this passports raw model recall.

Usage: venv/bin/python spike_pose/lj5_objects.py [--per-class 100] [--json out.json]
"""
import os, sys, csv, argparse, collections
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bench import passport
from openimages_fetch import cached, BBOX
from lj5_detect import iou, match, letterbox, download, IMGDIR, PROMINENT_H

YOLO = os.path.join(ROOT, "mvm", "models", "yolo_object_detect.onnx")
# (COCO name, COCO id, OI MID, display) — clean interaction-class mappings from RELEVANT_CLASSES.
MAP = [
    ("chair", 56, "/m/01mzpv", "Chair"), ("couch", 57, "/m/02crq1", "Couch"),
    ("potted plant", 58, "/m/03fp41", "Houseplant"), ("dining table", 60, "/m/04bcr3", "Table"),
    ("tv", 62, "/m/07c52", "Television"), ("laptop", 63, "/m/01c648", "Laptop"),
    ("mouse", 64, "/m/020lf", "Computer mouse"), ("remote", 65, "/m/0qjjc", "Remote control"),
    ("keyboard", 66, "/m/01m2v", "Computer keyboard"), ("cell phone", 67, "/m/050k8", "Mobile phone"),
    ("book", 73, "/m/0bt_c3", "Book"), ("clock", 74, "/m/01x3z", "Clock"),
    ("vase", 75, "/m/02s195", "Vase"), ("bottle", 39, "/m/04dr76w", "Bottle"),
    ("wine glass", 40, "/m/09tvcd", "Wine glass"), ("bowl", 45, "/m/04kkgm", "Bowl"),
    ("backpack", 24, "/m/01940j", "Backpack"), ("handbag", 26, "/m/080hkjn", "Handbag"),
]
MIDS = {m[2] for m in MAP}


def load_gt_objects(split="validation"):
    path = cached(BBOX[split])
    gt = collections.defaultdict(lambda: collections.defaultdict(list))   # image -> mid -> [box]
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["LabelName"] not in MIDS or r["IsGroupOf"] == "1" or r["IsDepiction"] == "1":
                continue
            gt[r["ImageID"]][r["LabelName"]].append(
                [float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"])])
    return gt


def yolo_all(sess, img_bgr, score_thr=0.35):
    """All-class YOLO decode → {coco_class_id: sorted[(norm_box, score)]}, mvm-scene-faithful."""
    h, w = img_bgr.shape[:2]
    s, px, py = letterbox(w, h)
    nw, nh = round(w * s), round(h * s)
    resized = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[round(py):round(py)+nh, round(px):round(px)+nw] = resized
    x = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
    out = sess.run(None, {"images": x})[0][0]
    by = collections.defaultdict(list)
    for row in out:
        sc = row[4]
        if sc < score_thr:
            continue
        cid = int(round(row[5]))
        b = [min(max(((row[0]-px)/s)/w, 0), 1), min(max(((row[1]-py)/s)/h, 0), 1),
             min(max(((row[2]-px)/s)/w, 0), 1), min(max(((row[3]-py)/s)/h, 0), 1)]
        by[cid].append((b, float(sc)))
    for cid in by:
        by[cid].sort(key=lambda d: -d[1])
    return by


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    print(f"# LJ5 B3+ — YOLO26n OBJECT passport on OpenImages {a.split} ({len(MAP)} interaction classes)")
    gt = load_gt_objects(a.split)
    by_mid = collections.defaultdict(list)
    for iid, mids in gt.items():
        for mid in mids:
            by_mid[mid].append(iid)
    pick = set()
    for mid in MIDS:
        imgs = sorted(by_mid[mid]); step = max(1, len(imgs) // a.per_class)
        pick.update(imgs[::step][:a.per_class])
    print(f"  sampling {len(pick)} images across classes")
    images = download(sorted(pick), a.split)
    print(f"  downloaded {len(images)} images")

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(YOLO, sess_options=so, providers=["CPUExecutionProvider"])

    # detections cached per image (one YOLO pass, all classes)
    dets_cache = {}
    for path in images:
        img = cv2.imread(path)
        if img is not None:
            dets_cache[os.path.splitext(os.path.basename(path))[0]] = yolo_all(sess, img)

    per = {}
    tp_mi = gt_mi = tp_mi_p = gt_mi_p = 0
    for coco_name, cid, mid, disp in MAP:
        tp = n_gt = tp_p = n_gt_p = 0
        for iid, d in dets_cache.items():
            gts = gt[iid].get(mid, [])
            if not gts:
                continue
            t, matched = match(d.get(cid, []), gts)
            tp += t; n_gt += len(gts)
            prom = [j for j, g in enumerate(gts) if (g[3]-g[1]) >= PROMINENT_H]
            n_gt_p += len(prom); tp_p += sum(1 for j in prom if j in matched)
        per[disp] = {"recall_all": tp/n_gt if n_gt else None,
                     "recall_prominent": tp_p/n_gt_p if n_gt_p else None,
                     "n_gt": n_gt, "n_gt_prominent": n_gt_p}
        tp_mi += tp; gt_mi += n_gt; tp_mi_p += tp_p; gt_mi_p += n_gt_p

    print(f"\n{'class':18s} {'recall_prom':>11s} {'recall_all':>10s} {'n_gt(prom/all)':>16s}")
    print("-" * 60)
    valid = [v for v in per.values() if v["recall_prominent"] is not None]
    for disp, v in per.items():
        rp = f"{v['recall_prominent']:.3f}" if v["recall_prominent"] is not None else "  n/a"
        ra = f"{v['recall_all']:.3f}" if v["recall_all"] is not None else "  n/a"
        print(f"{disp:18s} {rp:>11s} {ra:>10s} {str(v['n_gt_prominent'])+'/'+str(v['n_gt']):>16s}")
    macro_p = float(np.mean([v["recall_prominent"] for v in valid]))
    macro_a = float(np.mean([v["recall_all"] for v in valid if v["recall_all"] is not None]))
    micro_p = tp_mi_p/gt_mi_p if gt_mi_p else 0
    micro_a = tp_mi/gt_mi if gt_mi else 0
    print("-" * 60)
    print(f"MACRO recall  prominent={macro_p:.3f}  all={macro_a:.3f}   (mean over {len(valid)} classes)")
    print(f"MICRO recall  prominent={micro_p:.3f}  all={micro_a:.3f}   (pooled {gt_mi_p}/{gt_mi} gt)")

    out = {"passport": passport(), "split": a.split, "n_images": len(dets_cache),
           "per_class": per, "macro_recall": {"prominent": macro_p, "all": macro_a},
           "micro_recall": {"prominent": micro_p, "all": micro_a},
           "caveats": ["recall@IoU0.5, robust to OI federated labels; cross-domain COCO→OI",
                       "score_thr=0.35 (model default); prod filters A≥0.45+HIGH_BAR downstream "
                       "(trades recall for precision)",
                       "Table/dining-table and cup mappings are fuzzy across taxonomies (cup omitted)"]}
    if a.json:
        import json; json.dump(out, open(a.json, "w"), indent=2); print(f"\nwrote {a.json}")
    return out


if __name__ == "__main__":
    main()
