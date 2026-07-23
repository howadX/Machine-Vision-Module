#!/usr/bin/env python3
"""CMU mocap -> transition-focused artifact for phase 15.2 (LJ2).

Reuses the 15.0.1 lab machinery (`mocap_to_views`: BVH parse, FK, bind-scale,
virtual eye-level projection, kinematic labeller) but with:
  - a TRANSITION-focused trial set (sit<->stand, lie-down/get-up, floor falls),
  - HIGH temporal resolution (60 fps vs the 10 fps of the frozen 15.1 artifact),
    because a fall lasts ~0.3-0.5 s and its peak pelvis velocity is smeared at 10 fps,
  - a per-trial `scenario` tag (fall vs lie_down vs sit_stand vs standing) so E2 can
    separate the two ways a body ends up horizontal.

It writes a SEPARATE artifact `mocap_transitions.npz` and never touches the frozen
`mocap_views.npz` (15.1 numbers stay bit-stable; sha256 unchanged).

Artifact mocap_transitions.npz:
  X        (N,33,4)  projected landmarks @ azimuth 0 (GRIP slots)      -> feed the 15.1 classifier
  P3D      (N,13,3)  TRUE metric 3D, subject-centered, Y-up, feet~0    -> reproject / inject noise
  y        (N,)      per-frame kinematic ground-truth label            -> flip/latency truth (E1)
  t        (N,)      time (s) within trial (60 fps cadence)            -> velocity / durations
  pelvis   (N,)      pelvis height (m)                                 -> vertical velocity (E2)
  scenario (N,)      per-trial scenario: fall|lie_down|sit_stand|standing
  subj     (N,)      subject id     trial (N,)  trial id  (grouping keys, group-disjoint CV)

Run:  ../venv/bin/python3 spike_pose/lj2_transitions.py
"""
import os, collections, numpy as np
import mocap_to_views as mv   # import, do not copy (bench-protocol discipline)

OUT = os.path.join(mv.HERE, "data_probe", "mocap_transitions.npz")

FPS_OUT = 60          # high cadence: resolves the peak pelvis velocity of a fall
MAX_FRAMES = 1500     # 25 s @ 60 fps — long enough for the multi-transition trials
TRIM_FLOOR = 3        # ignore the BVH settling frames when estimating the ground plane

# Transition-focused set. `scenario` is the TRUE mechanism (E2's fall<->lie_down label);
# per-frame posture `y` still comes from the kinematic labeller (E1's flip/latency truth).
TRIALS = {
    # sit<->stand transitions: stable posture between events -> flip-rate + latency + durations
    "sit_stand": ["13_01", "13_02", "13_03", "13_05", "13_06",
                  "14_27", "14_28", "14_31", "15_10",
                  "111_09", "111_10", "111_11", "113_15", "114_04", "114_06",
                  "86_09", "86_10"],
    # controlled lie-down / get-up-from-floor: the SLOW way to become horizontal (E2 negative)
    "lie_down":  ["111_06", "111_07", "111_08", "111_12", "113_08", "114_16", "139_16"],
    # controlled descent TO the ground (the hard E2 negative — reaches the floor slowly).
    # CMU is genuinely thin here (no clean "lie on bed" rest); these are the closest honest set.
    "controlled_desc": ["01_08", "01_12", "82_05"],
    # floor-contact falls: the FAST way to become horizontal (E2 positive). ALL stunt/acrobatic
    # (no elderly falls in CMU) -> LJ2 states this domain caveat up front.
    "fall":      ["85_15", "88_04", "90_12", "90_13", "90_16", "90_18"],
    # pure stable standing: a clean flip-rate baseline on a genuinely static posture
    "standing":  ["111_28", "113_21"],
}


def process(trial):
    """Metric 13-joint 3D @ FPS_OUT, subject-centered, feet~0, + time axis. Mirrors
    mocap_to_views.process but with this stage's cadence/cap (frozen artifact untouched)."""
    path = mv.fetch(trial)
    joints, data, dt = mv.parse_bvh(path)
    scale = mv.bind_scale(joints)
    step = max(1, round((1.0 / dt) / FPS_OUT))
    rows = list(range(0, len(data), step))[:MAX_FRAMES]
    P = np.zeros((len(rows), 13, 3), np.float32)
    for n, r in enumerate(rows):
        p = mv.fk(joints, data[r])
        P[n] = np.array([p[b] for b in mv.BVH_NAMES]) * scale
    P[:, :, [0, 2]] -= P[:, 7:9, [0, 2]].mean(1, keepdims=True)   # center hips in x,z
    # Robust floor: the BVH first frame is often a settling/bind artifact with a limb far
    # below ground (foot at ~-0.85 m). Using the global min as the floor floats the WHOLE
    # trial up by that amount (broke pelvis height for 82/111/01/139). Take a low percentile
    # of the per-frame lowest joint AFTER trimming the settling frames -> the true ground.
    ymin_per_frame = P[:, :, 1].min(axis=1)
    floor = float(np.percentile(ymin_per_frame[TRIM_FLOOR:len(ymin_per_frame) - TRIM_FLOOR], 2))
    P[:, :, 1] -= floor                                           # feet to ~0 (glitch-robust)
    t = np.array([n * step * dt for n in range(len(rows))], np.float32)
    return P, t


def main():
    Xs, P3Ds, ys, ts, pel, scen, subjs, trials = [], [], [], [], [], [], [], []
    for scenario, tl in TRIALS.items():
        for trial in tl:
            try:
                P, t = process(trial)
            except Exception as e:
                print(f"  !! {trial}: {e}"); continue
            labs = [mv.label_pose(P[n]) for n in range(len(P))]
            for n in range(len(P)):
                Xs.append(mv.project(P[n], 0.0)); P3Ds.append(P[n])
                ys.append(labs[n]); ts.append(t[n]); pel.append(P[n, 7:9, 1].mean())
                scen.append(scenario); subjs.append(trial.split("_")[0]); trials.append(trial)
            pv = P[:, 7:9, 1].mean(1)
            print(f"{trial:8} ({scenario:9}) {len(P):4d} fr @{FPS_OUT}fps  "
                  f"pelvis[{pv.min():.2f},{np.median(pv):.2f},{pv.max():.2f}]m  "
                  f"labels={dict(collections.Counter(labs))}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT, X=np.array(Xs, np.float32), P3D=np.array(P3Ds, np.float32),
        y=np.array(ys), t=np.array(ts, np.float32), pelvis=np.array(pel, np.float32),
        scenario=np.array(scen), subj=np.array(subjs), trial=np.array(trials))
    print(f"\nsaved {OUT}: {len(ys)} poses, {len(set(trials))} trials, "
          f"scenarios={dict(collections.Counter(scen))}, labels={dict(collections.Counter(ys))}")


if __name__ == "__main__":
    main()
