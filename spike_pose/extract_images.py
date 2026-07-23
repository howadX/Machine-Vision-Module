#!/usr/bin/env python3
"""extract_images.py — run OUR OpenCV-Zoo BlazePose (MPPersonDet + MPPose) over a
folder of images -> 33x4 landmark npz, byte-consistent with spike_data.py's lab
extractor (same models, same [x/w, y/h, z/w, vis] normalization).

This is the shared extraction tool for the data-expansion track (A1): gold datasets,
COCO, Open Images all go through THIS, so a cross-source comparison is not confounded
by a cross-extractor difference.

Usage:
  python extract_images.py --root DIR --src NAME [--out OUT.npz]
        [--label-from parent|arg] [--label LABEL] [--multi] [--max N] [--exts jpg,png]

Label sourcing:
  --label-from parent : label = immediate parent directory name (Roboflow classification layout)
  --label-from arg    : label = --label for every image (single-class folder / smoke test)

Output npz: X(N,33,4) float32 · y(str) · grp(str, image stem) · src(str) · conf(float, pose conf)
"""
import os, sys, glob, argparse
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mp_persondet import MPPersonDet
from mp_pose import MPPose


def _find_model(name):
    for d in (os.path.join(ROOT, "models"), os.path.join(ROOT, "prototype", "models"),
              os.path.join(ROOT, "mvm", "models")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    sys.exit(f"model not found anywhere: {name}")


def build():
    det = MPPersonDet(modelPath=_find_model("person_detection_mediapipe_2023mar.onnx"),
                      nmsThreshold=0.3, scoreThreshold=0.5, topK=5000)
    pose = MPPose(modelPath=_find_model("pose_estimation_mediapipe_2023mar.onnx"), confThreshold=0.5)
    return det, pose


def extract_persons(img, det, pose, multi=False):
    """RGB/BGR frame -> list of (landmarks 33x4 normalized, pose_conf). Same as spike_data.extract."""
    persons = det.infer(img)
    if persons is None or len(persons) == 0:
        return []
    if not multi:
        persons = [max(persons, key=lambda p: float(p[-1]))]
    h, w = img.shape[:2]
    outs = []
    for person in persons:
        res = pose.infer(img, person)
        if res is None:
            continue
        lms = res[1]
        conf = float(res[-1]) if np.ndim(res[-1]) == 0 else float(np.ravel(res[-1])[0])
        out = np.zeros((33, 4), np.float32)
        for k in range(33):
            out[k] = [lms[k][0] / w, lms[k][1] / h, lms[k][2] / w, float(lms[k][3])]
        outs.append((out, conf))
    return outs


def crop_bbox(img, bbox, pad=0.2):
    """COCO bbox [x,y,w,h] -> padded, clamped crop (context for detector + full body)."""
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    px, py = bw * pad, bh * pad
    x0 = max(0, int(x - px)); y0 = max(0, int(y - py))
    x1 = min(w, int(x + bw + px)); y1 = min(h, int(y + bh + py))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return img[y0:y1, x0:x1]


def run_coco(json_path, det, pose, src, valid_labels, no_crop=False):
    """Roboflow-COCO -> best-person MPPose per annotation. grp=image_id.
    no_crop=True: run on the FULL image (matches lab's full-frame extraction context);
    only single-label images are kept (avoids ambiguity), one pose per image."""
    import json, collections
    d = json.load(open(json_path))
    base = os.path.dirname(json_path)
    split = os.path.basename(base) or "x"          # train/valid/test — image_id restarts per split
    src = f"{src}:{split}"                          # namespace groups by split (no cross-split id collision)
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    catname = {c["id"]: c["name"] for c in d["categories"]}
    anns_by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    rows = []; n_img = 0; n_noperson = 0; n_badlabel = 0; n_ambig = 0
    for img_id, anns in anns_by_img.items():
        fn = id2file.get(img_id)
        if not fn:
            continue
        img = cv2.imread(os.path.join(base, fn))
        if img is None:
            continue
        n_img += 1
        if no_crop:
            labs = {catname.get(a["category_id"]) for a in anns} & valid_labels
            if len(labs) != 1:
                n_ambig += 1
                continue
            label = labs.pop()
            got = extract_persons(img, det, pose, multi=False)
            if not got:
                n_noperson += 1
                continue
            lm, c = got[0]
            rows.append((lm, label, f"{src}:{img_id}", c))
        else:
            for a in anns:
                label = catname.get(a["category_id"])
                if label not in valid_labels:
                    n_badlabel += 1
                    continue
                crop = crop_bbox(img, a["bbox"])
                if crop is None:
                    continue
                got = extract_persons(crop, det, pose, multi=False)
                if not got:
                    n_noperson += 1
                    continue
                lm, c = got[0]
                rows.append((lm, label, f"{src}:{img_id}", c))
        if n_img % 200 == 0:
            print(f"  {n_img} imgs  poses={len(rows)}  no_person={n_noperson}")
    if no_crop:
        print(f"  (no-crop) skipped {n_ambig} multi-label images")
    print(f"  {os.path.relpath(json_path)}: {n_img} imgs -> {len(rows)} poses "
          f"(no_person={n_noperson}, skipped_label={n_badlabel})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["folder", "coco"], default="folder")
    ap.add_argument("--label-from", choices=["parent", "arg"], default="parent")
    ap.add_argument("--label", default="unknown")
    ap.add_argument("--valid-labels", default="lying,sitting,standing")
    ap.add_argument("--multi", action="store_true", help="keep all detected persons, not just best")
    ap.add_argument("--no-crop", action="store_true", help="coco mode: full-frame extraction (match lab context)")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--exts", default="jpg,jpeg,png")
    a = ap.parse_args()

    det, pose = build()
    X, y, grp, src, conf = [], [], [], [], []
    n_noperson = 0

    if a.mode == "coco":
        valid = set(a.valid_labels.split(","))
        jsons = sorted(glob.glob(os.path.join(a.root, "**", "_annotations.coco.json"), recursive=True))
        print(f"COCO jsons found: {len(jsons)}  (root={a.root})  valid_labels={valid}")
        for jp in jsons:
            for lm, label, g, c in run_coco(jp, det, pose, a.src, valid, no_crop=a.no_crop):
                X.append(lm); y.append(label); grp.append(g); src.append(a.src); conf.append(c)
    else:
        exts = a.exts.split(",")
        paths = sorted(p for e in exts for p in glob.glob(os.path.join(a.root, "**", f"*.{e}"), recursive=True))
        if a.max:
            paths = paths[:a.max]
        print(f"images found: {len(paths)}  (root={a.root})")
        n_noperson = 0
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                continue
            label = os.path.basename(os.path.dirname(path)) if a.label_from == "parent" else a.label
            got = extract_persons(img, det, pose, multi=a.multi)
            if not got:
                n_noperson += 1
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            for lm, c in got:
                X.append(lm); y.append(label); grp.append(f"{a.src}:{stem}"); src.append(a.src); conf.append(c)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(paths)}  poses={len(X)}  no_person={n_noperson}")

    X = np.asarray(X, np.float32)
    out = a.out or os.path.join(HERE, "data_probe", f"{a.src}_poses.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, X=X, y=np.array(y), grp=np.array(grp), src=np.array(src), conf=np.array(conf, np.float32))
    import collections
    print(f"\nwrote {out}")
    print(f"  poses: {X.shape}  | no-person images: {n_noperson}")
    print(f"  labels: {dict(collections.Counter(y))}")
    print(f"  unique groups: {len(set(grp))}")


if __name__ == "__main__":
    main()
