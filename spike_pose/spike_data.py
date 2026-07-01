"""P0 spike — data acquisition.

Builds a PUBLIC 3-class posture landmark set with OUR BlazePose topology:
  - standing / lying  <- UR Fall Detection (URFD), extracted with our OpenCV-Zoo BlazePose
  - sitting           <- Zenodo MultiPosture (real MediaPipe 33-lm 3D, sitting only)

Caches to data_probe/landmarks.npz:  X_raw [N,33,4] (x,y,z,vis, MediaPipe-normalized),
y [N] in {standing,sitting,lying}, src [N] ('urfd'|'multiposture').

Re-run is cheap: skips download if the npz already exists.
Run from spike_pose/ with the project venv.
"""
import os, sys, io, json, base64, zipfile, urllib.request, socket
import numpy as np, cv2

socket.setdefaulttimeout(60)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mp_persondet import MPPersonDet
from mp_pose import MPPose

M = os.path.join(ROOT, "models")
OUT = os.path.join(HERE, "data_probe", "landmarks.npz")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

URFD = "https://fenix.ur.edu.pl/~mkepski/ds/data"
N_SEQ = int(os.environ.get("N_SEQ", "14"))      # fall sequences to pull
PER_CLASS_PER_SEQ = int(os.environ.get("PER_SEQ", "25"))
SITTING_CAP = int(os.environ.get("SIT_CAP", "400"))

det = MPPersonDet(modelPath=f"{M}/person_detection_mediapipe_2023mar.onnx",
                  nmsThreshold=0.3, scoreThreshold=0.5, topK=5000)
pose = MPPose(modelPath=f"{M}/pose_estimation_mediapipe_2023mar.onnx", confThreshold=0.5)


def extract(img):
    """RGB frame -> 33x4 (x/w,y/h,z/w,vis) or None."""
    persons = det.infer(img)
    if persons is None or len(persons) == 0:
        return None
    best = max(persons, key=lambda p: float(p[-1]))
    res = pose.infer(img, best)
    if res is None:
        return None
    lms = res[1]
    h, w = img.shape[:2]
    out = np.zeros((33, 4), np.float32)
    for k in range(33):
        out[k] = [lms[k][0] / w, lms[k][1] / h, lms[k][2] / w, float(lms[k][3])]
    return out


def load_urfd_labels():
    raw = urllib.request.urlopen(f"{URFD}/urfall-cam0-falls.csv").read().decode("utf-8", "replace")
    lab = {}  # (seq, frame) -> label
    for line in raw.splitlines():
        p = line.split(",")
        lab[(p[0], int(p[1]))] = int(p[2])
    return lab


def frame_num(name):
    import re
    m = re.findall(r"(\d+)", os.path.basename(name))
    return int(m[-1]) if m else -1


def pull_urfd(labels):
    X, y, grp = [], [], []
    for i in range(1, N_SEQ + 1):
        seq = f"fall-{i:02d}"
        try:
            blob = urllib.request.urlopen(f"{URFD}/{seq}-cam0-rgb.zip").read()
            z = zipfile.ZipFile(io.BytesIO(blob))
        except Exception as e:
            print(f"  {seq}: download FAIL {str(e)[:50]}"); continue
        pngs = sorted(n for n in z.namelist() if n.lower().endswith((".png", ".jpg")))
        got = {"standing": 0, "lying": 0}
        for n in pngs:
            fn = frame_num(n)
            lb = labels.get((seq, fn))
            cls = {-1: "standing", 1: "lying"}.get(lb)
            if cls is None or got[cls] >= PER_CLASS_PER_SEQ:
                continue
            arr = np.frombuffer(z.open(n).read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            lm = extract(img)
            if lm is None:
                continue
            X.append(lm); y.append(cls); grp.append(f"urfd_{seq}"); got[cls] += 1
        print(f"  {seq}: standing={got['standing']} lying={got['lying']}")
    return X, y, grp


def pull_multiposture():
    """Zenodo MultiPosture: 33-lm 3D, all sitting. Cols: subject,upper,lower, then 33*(x,y,z).
    group = subject id (for cross-subject generalization eval)."""
    raw = urllib.request.urlopen(
        "https://zenodo.org/api/records/14230872/files/data.csv/content").read().decode("utf-8", "replace")
    lines = raw.splitlines()
    rows = lines[1:]
    step = max(1, len(rows) // SITTING_CAP)
    X, y, grp = [], [], []
    for r in rows[::step]:
        p = r.split(",")
        coords = np.array(p[3:3 + 99], np.float32).reshape(33, 3)
        lm = np.concatenate([coords, np.ones((33, 1), np.float32)], axis=1)  # vis=1
        X.append(lm); y.append("sitting"); grp.append(f"mp_subj{p[0]}")
        if len(X) >= SITTING_CAP:
            break
    print(f"  multiposture sitting={len(X)} subjects={len(set(grp))}")
    return X, y, grp


if __name__ == "__main__":
    if os.path.exists(OUT) and "--force" not in sys.argv:
        d = np.load(OUT, allow_pickle=True)
        import collections
        print("cached:", OUT, dict(collections.Counter(d["y"])))
        sys.exit(0)
    print("URFD labels...");  labels = load_urfd_labels()
    print("URFD frames (standing/lying)...");  Xu, yu, gu = pull_urfd(labels)
    print("MultiPosture (sitting)...");  Xs, ys, gs = pull_multiposture()
    X = np.array(Xu + Xs, np.float32); y = np.array(yu + ys)
    src = np.array(["urfd"] * len(yu) + ["multiposture"] * len(ys))
    grp = np.array(gu + gs)
    np.savez_compressed(OUT, X=X, y=y, src=src, grp=grp)
    import collections
    print("SAVED", OUT, X.shape, dict(collections.Counter(y)))
