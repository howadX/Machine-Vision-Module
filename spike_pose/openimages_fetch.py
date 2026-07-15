#!/usr/bin/env python3
"""Fetch clean single-person OpenImages images for the pose pilot (phase 15.0.2 / Track A E3).

Selects images with EXACTLY ONE prominent, non-group, non-depiction Person box (so MPPose
best-person extraction is unambiguous and roughly full-body), then downloads them from the
public S3 bucket (no auth). Feed the folder to extract_images.py (folder mode) -> same
MPPose extractor as the lab, so cross-source is not confounded by cross-extractor.

  validation-annotations-bbox.csv (GCS, 25 MB, cached)   Person MID = /m/01g317
  -> filter: 1 Person/image, IsGroupOf=0, IsDepiction=0, box tall+large (full-body)
  -> download s3://open-images-dataset/validation/<id>.jpg

Usage: openimages_fetch.py [--split validation] [--max 300] [--min-h 0.5] [--min-area 0.12]
Images CC BY 2.0 (publish only DERIVED features, never the images) — matches the pose-track rule.
"""
import os, csv, argparse, urllib.request, concurrent.futures as cf, collections

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "oi_meta")
PERSON = "/m/01g317"
GCS = "https://storage.googleapis.com/openimages"
S3 = "https://open-images-dataset.s3.amazonaws.com"
BBOX = {"validation": "v5/validation-annotations-bbox.csv",
        "test": "v5/test-annotations-bbox.csv"}


def cached(path):
    os.makedirs(META, exist_ok=True)
    dst = os.path.join(META, os.path.basename(path))
    if not os.path.exists(dst):
        print(f"  fetch {GCS}/{path}")
        urllib.request.urlretrieve(f"{GCS}/{path}", dst)
    return dst


def select(split, max_n, min_h, min_area):
    """Return image ids with exactly one prominent standalone Person box."""
    csv_path = cached(BBOX[split])
    per_img = collections.defaultdict(list)   # image -> list of person boxes (as dicts)
    with open(csv_path, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            if row["LabelName"] != PERSON:
                continue
            if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1":
                continue
            w = float(row["XMax"]) - float(row["XMin"]); h = float(row["YMax"]) - float(row["YMin"])
            per_img[row["ImageID"]].append((w, h, w * h))
    picked = []
    for img, boxes in per_img.items():
        if len(boxes) != 1:                    # exactly one person -> unambiguous best-person
            continue
        w, h, area = boxes[0]
        if h >= min_h and area >= min_area:    # tall + large -> roughly full-body
            picked.append(img)
        if len(picked) >= max_n:
            break
    return picked


def download(split, ids, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    def one(img):
        out = os.path.join(dst_dir, f"{img}.jpg")
        if os.path.exists(out):
            return True
        try:
            urllib.request.urlretrieve(f"{S3}/{split}/{img}.jpg", out)
            return True
        except Exception as e:
            return False
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, r in enumerate(ex.map(one, ids), 1):
            ok += bool(r)
            if i % 50 == 0:
                print(f"  downloaded {ok}/{i}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=list(BBOX))
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--min-h", type=float, default=0.5)      # box height >= 50% of image
    ap.add_argument("--min-area", type=float, default=0.12)  # box area >= 12% of image
    ap.add_argument("--dst", default=os.path.join(HERE, "oi_images"))
    a = ap.parse_args()
    ids = select(a.split, a.max, a.min_h, a.min_area)
    print(f"selected {len(ids)} clean single-person {a.split} images "
          f"(min_h={a.min_h}, min_area={a.min_area})")
    dst = os.path.join(a.dst, a.split)
    ok = download(a.split, ids, dst)
    print(f"downloaded {ok}/{len(ids)} -> {dst}")
    print(f"next: extract_images.py --mode folder --root {dst} --src openimages "
          f"--label-from arg --label unknown --out data_probe/openimages_poses.npz")


if __name__ == "__main__":
    main()
