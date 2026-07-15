#!/usr/bin/env python3
"""CMU mocap (cgspeed BVH) -> 13 GRIP joints -> virtual eye-level camera -> mocap_views.npz.

Phase 15.0.1 lab tool. The point: mocap gives TRUE 3D, so azimuth becomes a real
camera viewpoint (an honest rotation), not the circular y-rotation of MediaPipe's
noisy z that NB3b's azimuth test relied on.

Pipeline
  BVH -> forward kinematics (world joint positions, Y-up)
      -> pick 13 GRIP joints (MediaPipe idx 0,11,12,13,14,15,16,23,24,25,26,27,28)
      -> metric scale (bind-pose height -> 1.70 m; GRIP features are scale-free anyway,
         scale only feeds camera realism)
      -> subject-centered metric 3D  (P3D, N x 13 x 3, meters, Y-up, feet ~ y=0)
      -> project(): virtual pinhole camera at 175 cm, horizontal (no tilt), subject
         rotated about vertical by azimuth -> MediaPipe-style (x/w, y/h, z, vis), y DOWN.

Artifact mocap_views.npz:
  X      (N,33,4)  projected landmarks at azimuth 0, GRIP indices filled, rest 0  (drop-in for features_13)
  P3D    (N,13,3)  TRUE metric 3D, subject-centered  -> re-project at ANY azimuth for the honest bench
  y      (N,)      provisional posture label from pelvis height + torso inclination (kinematic, NOT GRIP)
  azimuth(N,)      azimuth of X (0 here; P3D carries the truth for re-projection)
  t      (N,)      time (s) within trial
  pelvis (N,)      pelvis height (m) -> velocity / transition analysis (15.2)
  subj   (N,)      subject id     trial (N,)  trial id  (== grouping keys)

Joint order for P3D and GRIP indices:
  [nose, Lsh, Rsh, Lel, Rel, Lwr, Rwr, Lhip, Rhip, Lkn, Rkn, Lank, Rank]
Run:  ../venv/bin/python3 spike_pose/mocap_to_views.py
"""
import os, sys, urllib.request, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BVH_DIR = os.path.join(HERE, "mocap_bvh")
OUT = os.path.join(HERE, "data_probe", "mocap_views.npz")
RAW = "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data"

# GRIP 13 joints  ->  CMU BVH joint names
GRIP2BVH = [
    ("nose", "Head"),
    ("Lsh", "LeftArm"),   ("Rsh", "RightArm"),
    ("Lel", "LeftForeArm"), ("Rel", "RightForeArm"),
    ("Lwr", "LeftHand"),  ("Rwr", "RightHand"),
    ("Lhip", "LeftUpLeg"), ("Rhip", "RightUpLeg"),
    ("Lkn", "LeftLeg"),   ("Rkn", "RightLeg"),
    ("Lank", "LeftFoot"), ("Rank", "RightFoot"),
]
GRIP_MP_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # MediaPipe slots
BVH_NAMES = [b for _, b in GRIP2BVH]

NOMINAL_H = 1.70      # meters, for metric scaling
CAM_H = 1.75          # eye-level camera height (matches the real MVM rig)
CAM_DIST = 2.6        # meters subject<->camera
FOCAL = 1.2           # normalized focal (subject ~fills frame)
FPS_OUT = 10          # subsample cadence
MAX_FRAMES = 240      # cap per trial

# Trial selection (15.0.1 starter set; big walk pull deferred to 15.3).
# Subjects 111 & 113 act out EVERY posture -> class independent of subject (breaks the
# class<->source confound within the mocap set itself). Dropped: 77 (martial-arts "ready
# pose", airborne -> bad lying) and 82_05 (ambiguous chair-height "ground sit").
TRIALS = {
    "standing":  ["111_28", "113_21"],                              # standing still
    "sit_stand": ["13_01", "13_02", "13_03", "14_27", "15_10",      # stool sit + stand up
                  "111_09", "111_10", "111_11", "113_15"],          # get up from chair
    "lying":     ["111_12", "111_21", "113_08",                     # lay down / roll over
                  "111_06", "111_07", "111_08"],                    # get up from floor
    "fall":      ["85_15", "90_12", "90_13", "90_16", "90_18"],     # floor-contact falls
    "walk":      ["111_34", "113_25", "07_01", "35_01"],
}


# ----------------------------- BVH parsing + FK -----------------------------
class Joint:
    __slots__ = ("name", "offset", "chans", "parent", "ci")

    def __init__(self, name, parent):
        self.name = name; self.parent = parent
        self.offset = np.zeros(3); self.chans = []; self.ci = -1


def parse_bvh(path):
    with open(path) as f:
        toks = f.read().split("\n")
    joints, stack, ci = [], [], 0
    i = 0
    # HIERARCHY
    while i < len(toks):
        line = toks[i].strip(); i += 1
        if line.startswith("MOTION"):
            break
        w = line.split()
        if not w:
            continue
        if w[0] in ("ROOT", "JOINT"):
            j = Joint(w[1], stack[-1] if stack else -1)
            joints.append(j); stack.append(len(joints) - 1)
        elif w[0] == "End":            # End Site (leaf, no channels)
            stack.append(-2)           # placeholder depth marker
        elif w[0] == "OFFSET":
            if stack and stack[-1] >= 0:
                joints[stack[-1]].offset = np.array(list(map(float, w[1:4])))
        elif w[0] == "CHANNELS":
            j = joints[stack[-1]]
            j.chans = w[2:]; j.ci = ci; ci += int(w[1])
        elif line == "}":
            stack.pop()
    total_ch = ci
    # MOTION
    while i < len(toks) and not toks[i].strip().startswith("Frames:"):
        i += 1
    nframes = int(toks[i].split(":")[1]); i += 1
    dt = float(toks[i].split(":")[1]); i += 1
    data = np.array([list(map(float, toks[i + k].split()))
                     for k in range(nframes) if toks[i + k].strip()], np.float64)
    assert data.shape[1] == total_ch, (data.shape, total_ch)
    return joints, data, dt


def _rot(axis, deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    if axis == "X": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])              # Z


def fk(joints, row):
    """Return dict name -> world position (3,) for one motion row."""
    G = [None] * len(joints)
    pos = {}
    for k, j in enumerate(joints):
        R = np.eye(3); trans = j.offset.copy()
        off = 0
        for ch in j.chans:
            v = row[j.ci + off]; off += 1
            if ch.endswith("position"):
                idx = {"X": 0, "Y": 1, "Z": 2}[ch[0]]
                trans[idx] += v
            else:
                R = R @ _rot(ch[0], v)
        L = np.eye(4); L[:3, :3] = R; L[:3, 3] = trans
        G[k] = L if j.parent < 0 else G[j.parent] @ L
        pos[j.name] = G[k][:3, 3]
    return pos


LEG_M = 0.90   # anthropometric hip-to-ankle length target (m) -> metric scale


def bind_scale(joints):
    """Metric scale from bind-pose leg length (hip->knee->ankle). Anatomically grounded,
    less noisy than head-to-ankle span; makes standing pelvis ~0.9 m and 15.2 velocities metric."""
    p = fk(joints, np.zeros(sum(len(j.chans) for j in joints)))
    legs = []
    for hip, kn, ank in (("LeftUpLeg", "LeftLeg", "LeftFoot"), ("RightUpLeg", "RightLeg", "RightFoot")):
        legs.append(np.linalg.norm(p[hip] - p[kn]) + np.linalg.norm(p[kn] - p[ank]))
    return LEG_M / np.mean(legs)


# ------------------------- virtual camera projection -------------------------
def project(P3D, azimuth_deg):
    """P3D (13,3) metric subject-centered (Y-up, feet~0) -> (33,4) MediaPipe-style, y DOWN."""
    t = np.deg2rad(azimuth_deg); c, s = np.cos(t), np.sin(t)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Pw = P3D @ Ry.T                                   # rotate subject about vertical
    Pw = Pw + np.array([0, 0, CAM_DIST])              # push in front of camera (+Z)
    depth = Pw[:, 2].clip(0.2, None)                  # camera looks along +Z toward subject
    xn = 0.5 + FOCAL * Pw[:, 0] / depth
    yn = 0.5 - FOCAL * (Pw[:, 1] - CAM_H) / depth     # image y DOWN (MediaPipe)
    hip_d = depth[7:9].mean()                         # Lhip,Rhip
    zn = FOCAL * (depth - hip_d) / hip_d
    out = np.zeros((33, 4), np.float32)
    for k, mp in enumerate(GRIP_MP_IDX):
        out[mp] = [xn[k], yn[k], zn[k], 1.0]
    return out


def _flex(a, b, c):
    u, w = a - b, c - b
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-6 or nw < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u / nu, w / nw), -1, 1))))


def label_pose(P3D, pelvis_ref=None):
    """Absolute-metric kinematic label (pelvis height + torso inc + knee flexion).

    Biomechanics, NOT the GRIP feature vector; metric scale (bind-pose->1.70 m) makes
    absolute thresholds robust across trials. pelvis_ref kept for signature compat.
    """
    hip = P3D[7:9].mean(0); sh = P3D[1:3].mean(0); ank = P3D[11:13].mean(0)
    inc = np.degrees(np.arccos(np.clip((sh - hip)[1] / (np.linalg.norm(sh - hip) + 1e-9), -1, 1)))
    leg = 0.5 * (np.linalg.norm(P3D[7] - P3D[9]) + np.linalg.norm(P3D[9] - P3D[11])
                 + np.linalg.norm(P3D[8] - P3D[10]) + np.linalg.norm(P3D[10] - P3D[12]))
    ratio = (hip[1] - ank[1]) / (leg + 1e-9)       # hip height / leg length: scale-free
    knee = 0.5 * (_flex(P3D[7], P3D[9], P3D[11]) + _flex(P3D[8], P3D[10], P3D[12]))
    if inc > 50:                                   # torso horizontal
        return "lying"
    if ratio < 0.70 or knee < 120:                 # hips lowered / knees flexed
        return "sitting"
    return "standing"


# --------------------------------- driver ----------------------------------
def fetch(trial):
    subj = trial.split("_")[0]
    dst = os.path.join(BVH_DIR, f"{trial}.bvh")
    if not os.path.exists(dst):
        os.makedirs(BVH_DIR, exist_ok=True)
        url = f"{RAW}/{int(subj):03d}/{trial}.bvh"
        print(f"  fetch {url}")
        urllib.request.urlretrieve(url, dst)
    return dst


def process(trial):
    path = fetch(trial)
    joints, data, dt = parse_bvh(path)
    scale = bind_scale(joints)
    step = max(1, round((1.0 / dt) / FPS_OUT))
    rows = list(range(0, len(data), step))[:MAX_FRAMES]
    # FK -> metric 13-joint 3D, subject-centered
    P = np.zeros((len(rows), 13, 3), np.float32)
    for n, r in enumerate(rows):
        p = fk(joints, data[r])
        P[n] = np.array([p[b] for b in BVH_NAMES]) * scale
    P[:, :, [0, 2]] -= P[:, 7:9, [0, 2]].mean(1, keepdims=True)   # center hips in x,z
    ymin = P[:, :, 1].min()
    P[:, :, 1] -= ymin                                            # feet to ~0
    pelvis_ref = np.percentile(P[:, 7:9, 1].mean(1), 95)
    return P, np.array([n * step * dt for n in range(len(rows))], np.float32), pelvis_ref


def main():
    Xs, P3Ds, ys, azs, ts, pel, subjs, trials = [], [], [], [], [], [], [], []
    for cls, tl in TRIALS.items():
        for trial in tl:
            try:
                P, tt, pref = process(trial)
            except Exception as e:
                print(f"  !! {trial}: {e}"); continue
            for n in range(len(P)):
                lab = label_pose(P[n], pref)
                Xs.append(project(P[n], 0.0)); P3Ds.append(P[n])
                ys.append(lab); azs.append(0.0); ts.append(tt[n])
                pel.append(P[n, 7:9, 1].mean()); subjs.append(trial.split("_")[0]); trials.append(trial)
            import collections
            pv = P[:, 7:9, 1].mean(1)
            print(f"{trial:8} ({cls:9}) {len(P):3d} fr  pelvis[{pv.min():.2f},{np.median(pv):.2f},{pv.max():.2f}]m  labels="
                  f"{dict(collections.Counter([label_pose(P[n]) for n in range(len(P))]))}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT, X=np.array(Xs, np.float32), P3D=np.array(P3Ds, np.float32),
        y=np.array(ys), azimuth=np.array(azs, np.float32), t=np.array(ts, np.float32),
        pelvis=np.array(pel, np.float32), subj=np.array(subjs), trial=np.array(trials))
    import collections
    print(f"\nsaved {OUT}: {len(ys)} poses, {len(set(trials))} trials, "
          f"dist={dict(collections.Counter(ys))}")


if __name__ == "__main__":
    main()
