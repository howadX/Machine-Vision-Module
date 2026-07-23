#!/usr/bin/env python3
"""LJ4 bench (S1/S2): drive golden scenarios through the FULL ported temporal stack —
tracker (lj3_tracker) -> posture stabilizer (LJ2 port, online) -> infer_activities ->
Activity FSM -> anomaly (lj4_fusion) — and score journal events against the scenario's
known-by-construction truth.

Frozen bench-protocol component "Simulator/FSM": leading metric = EVENT precision/recall
vs the scenario (label + participants + start/stop within tolerance), plus anomaly
false-alarm rate on quiet scenarios and anomaly hit on falls. Grouped bootstrap CI over
scenario instances (N=1000, seed=0).

Emitters (the LJ3 protocol + one posture knob; comparable only within one protocol):
  FLOOR  — clean injection: exact bbox, true kinematic posture, no identity.
  STRESS — live-like detector defects: bursty occlusion dropout, bbox jitter,
           intermittent early face identity (the 15.3 dead-zone trigger), and
           per-frame posture misclassification (flip to an adjacent class).

Layer ablations at STRESS med attribute the degradation:
  tracker=fix15.3 (stale-identity center re-link), stabilizer=hybrid (LJ2 sl_dwell
  veto), face=0 (no identity stamping), flip=0 (clean posture).

S2 rides the same runs: prod AnomalyScorer (standing->lying <= 3 s window, NO velocity)
vs the LJ2 E2 physics rule (reached floor AND peak downward pelvis velocity) on G3/G4.

Run:  ../venv/bin/python3 spike_pose/lj4_bench.py
"""
import os, sys, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lj3_tracker import PersonTracker, Detection
from lj4_fusion import Person, Obj, FusionStep, _event_from
import bench

GOLD = os.path.join(HERE, "data_probe", "golden_scenarios.npz")
FPS = 15
SEED = 0
N_BOOT = 1000
POSTURE_CLASSES = ["standing", "sitting", "lying"]
POSTURE_CONF_DEFAULT = 0.9      # injected classifier confidence (scenario meta)

# STRESS protocols: (dropout, jitter, face_frames, burst, p_flip). dropout/jitter/face/
# burst are the LJ3 values (same live-detector model); p_flip adds posture noise.
STRESS = {"mild":  (0.06, 0.010, 3, (2, 4), 0.03),
          "med":   (0.10, 0.020, 3, (2, 4), 0.06),
          "heavy": (0.15, 0.030, 3, (2, 5), 0.10)}
FLOOR = (0.0, 0.0, 0, (2, 4), 0.0)

# LJ2 E2 fall rule (results/metrics.json["lj2"]): reached floor AND fast descent.
E2_FLOOR_M = 0.35
E2_VEL_MS = 1.13        # calibrated at 60 fps (LJ2)
E2_VEL_RECAL = 1.05     # 15 fps recalibration candidate: the production cadence smears
                        # the peak (G4_85_15 reads 1.12 < 1.13 while the worst quiet
                        # scenario reads 0.99) — measured here, decided by the table
E2_WIN_S = 0.4


# ── online posture stabilizer (LJ2 port of posture.rs, stateful) ───────────────

class Stabilizer:
    """Per-track online form of lj2_stabilizer.run_dwell / run_hybrid. sl_dwell=None ->
    production dwell machine; sl_dwell=12 -> the 15.2 shipped hybrid (SL veto)."""

    def __init__(self, dwell=3, min_conf=0.5, sl_dwell=None):
        self.dwell, self.min_conf, self.sl_dwell = dwell, min_conf, sl_dwell
        self.committed = None
        self.cand = None
        self.streak = 0

    def step(self, lab, conf):
        need = self.dwell
        if (self.sl_dwell is not None and self.committed is not None
                and {self.committed, lab} == {"standing", "lying"}):
            need = self.sl_dwell
        if conf < self.min_conf:
            self.cand = self.committed
            self.streak = 0
        elif self.committed is None:
            self.committed = lab
            self.cand = lab
            self.streak = 0
        elif lab == self.committed:
            self.cand = lab
            self.streak = 0
        else:
            if self.cand == lab:
                self.streak += 1
            else:
                self.cand = lab
                self.streak = 1
            if self.streak >= need:
                self.committed = lab
                self.streak = 0
        return self.committed if self.committed is not None else lab


# ── data access ────────────────────────────────────────────────────────────────

def load():
    d = np.load(GOLD, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    return d, meta


def frames_of(d, inst, n_frames):
    """frame idx -> list of truth-row dicts for this instance."""
    s = d["inst"] == inst
    by = collections.defaultdict(list)
    for i in np.where(s)[0]:
        by[int(d["frame"][i])].append(dict(
            person=int(d["person"][i]), bbox=d["bbox"][i].astype(float),
            vis=bool(d["vis"][i]), posture=str(d["posture"][i]),
            wrists=d["wrists"][i].astype(float), pelvis3d=d["pelvis3d"][i].astype(float),
            pos3d=tuple(float(v) for v in d["pos3d"][i])))
    return [by.get(f, []) for f in range(n_frames)]


def objects_of(d, inst):
    s = d["obj_inst"] == inst
    return [Obj(int(d["obj_id"][i]), str(d["obj_cls"][i]), tuple(d["obj_bbox"][i]))
            for i in np.where(s)[0]]


# ── emitter (LJ3 semantics + posture flip) ─────────────────────────────────────

def emit(frames, seed_key, dropout=0.0, jitter=0.0, face_frames=0, burst=(2, 4), p_flip=0.0):
    """-> per-frame list of (Detection, truth_row). Bursty occlusion (live misses come
    in runs), bbox jitter, confident face identity on a person's first sightings (the
    dead-zone trigger), posture flipped to another class with p_flip.

    Each noise source draws from its OWN generator (seeded by (seed, instance, source)):
    turning one source off in an ablation must not shift the other sources' streams —
    otherwise 'no posture flip' silently reshuffles the dropout pattern and the ablation
    compares two different worlds (caught on the first run: no-flip scored WORSE)."""
    seed, inst_idx = seed_key
    r_drop, r_jit, r_flip = (np.random.default_rng([seed, inst_idx, salt])
                             for salt in (0, 1, 2))
    seen = collections.Counter()
    occ = collections.defaultdict(int)
    out = []
    for fl in frames:
        cur = []
        for row in fl:
            if not row["vis"]:
                continue
            p = row["person"]
            if occ[p] > 0:
                occ[p] -= 1
                continue
            if dropout and r_drop.random() < dropout:
                occ[p] = int(r_drop.integers(burst[0], burst[1] + 1)) - 1
                continue
            b = row["bbox"].copy()
            if jitter:
                b[:2] += r_jit.normal(0, jitter, 2)
                b[2:] += r_jit.normal(0, jitter, 2)
            ident = conf = None
            seen[p] += 1
            if face_frames and seen[p] <= face_frames:
                ident, conf = f"P{p}", 0.9
            lab = row["posture"]
            if p_flip and r_flip.random() < p_flip:
                lab = r_flip.choice([c for c in POSTURE_CLASSES if c != lab])
            cur.append((Detection(list(b), identity=ident, identity_confidence=conf),
                        {**row, "posture_obs": lab}))
        out.append(cur)
    return out


# ── the full-stack runner ──────────────────────────────────────────────────────

def run_instance(emitted, objects, n_frames, tracker_kw=None, stab_kw=None,
                 journal_kw=None):
    """Drive tracker -> stabilizer -> fusion over one instance. Returns dict with
    journal events (open episodes closed at the end and tagged), anomaly trace,
    tid->person mapping, per-person committed ids, and per-frame pelvis3d truth."""
    tracker = PersonTracker(**(tracker_kw or {}))
    step = FusionStep(journal_kw=journal_kw)
    stabs = {}
    tid_person = collections.Counter()
    ids_of = collections.defaultdict(list)
    events, anomaly = [], []
    for f in range(n_frames):
        dets = [dt for dt, _ in emitted[f]]
        rows = [r for _, r in emitted[f]]
        ids = tracker.tick(dets)
        people = []
        for tid, row in zip(ids, rows):
            tid_person[(tid, row["person"])] += 1
            ids_of[row["person"]].append(tid)
            st = stabs.setdefault(tid, Stabilizer(**(stab_kw or {})))
            lab = st.step(row["posture_obs"], POSTURE_CONF_DEFAULT)
            people.append(Person(
                track_id=tid, bbox=tuple(row["bbox"]), position_3d=row["pos3d"],
                posture=(lab, POSTURE_CONF_DEFAULT),
                wrists=[list(w) for w in row["wrists"]]))
        _, ev, an = step.tick(people, objects, f / FPS, f)
        events.extend([(e, False) for e in ev])
        anomaly.append(an)
    for inst_a in step.journal.active():        # open episodes at scenario end
        events.append((_event_from(inst_a), True))
    tid2p = {}
    for (tid, p), n in tid_person.items():
        if tid not in tid2p or n > tid_person[(tid, tid2p[tid])]:
            tid2p[tid] = p
    return dict(events=events, anomaly=np.array(anomaly), tid2p=tid2p, ids_of=ids_of)


# ── scoring vs expected ────────────────────────────────────────────────────────

def score_instance(res, meta_inst):
    """-> dict(tp, fp, fn, fa (quiet violated), fire_ok (fall fired / None),
    excess, swaps)."""
    fps = meta_inst["fps"]
    t_end = meta_inst["n_frames"] / fps
    tid2p = res["tid2p"]

    def mapped(ev):
        persons = frozenset(tid2p.get(pid, -1) for pid in ev.participants if pid < 10000)
        objs = frozenset(pid for pid in ev.participants if pid >= 10000)
        return persons, objs

    emitted = [(ev, open_, *mapped(ev)) for ev, open_ in res["events"]]
    used = [False] * len(emitted)
    tp = fn = 0
    for exp in meta_inst["expected"]:
        if exp["kind"] != "activity":
            continue
        want_p = frozenset(v for v in (exp.get("person"), exp.get("person2")) if v is not None)
        want_o = frozenset([exp["obj"]] if "obj" in exp else [])
        got = False
        for i, (ev, open_, ps, os_) in enumerate(emitted):
            if used[i] or ev.label != exp["label"] or ps != want_p or os_ != want_o:
                continue
            end = t_end if open_ else ev.ended_at
            if (abs(ev.started_at - exp["t_start"]) <= exp["tol_start"]
                    and abs(end - exp["t_end"]) <= exp["tol_end"]):
                used[i] = True
                got = True
                break
        tp += got
        fn += not got
    # FP breakdown: same key as SOME expected but wrong timing / a split episode
    # ("fragment") vs an event no expected mentions at all ("foreign").
    fp = fp_frag = fp_foreign = 0
    exp_keys = {(e["label"],
                 frozenset(v for v in (e.get("person"), e.get("person2")) if v is not None),
                 frozenset([e["obj"]] if "obj" in e else []))
                for e in meta_inst["expected"] if e["kind"] == "activity"}
    for i, (ev, open_, ps, os_) in enumerate(emitted):
        if used[i]:
            continue
        fp += 1
        if (ev.label, ps, os_) in exp_keys:
            fp_frag += 1
        else:
            fp_foreign += 1

    fa = fire_ok = None
    an = res["anomaly"]
    for exp in meta_inst["expected"]:
        if exp["kind"] == "anomaly_quiet":
            fa = bool((an >= exp["max"]).any())
        elif exp["kind"] == "anomaly_fire":
            a = int(max(0, exp["t_from"] * fps))
            b = int(min(len(an), exp["t_to"] * fps + 1))
            fire_ok = bool((an[a:b] >= exp["min"]).any())

    excess = sum(max(0, len(set(v)) - 1) for v in res["ids_of"].values())
    id2ps = collections.defaultdict(set)
    for p, ids in res["ids_of"].items():
        for i in ids:
            id2ps[i].add(p)
    swaps = sum(1 for ps in id2ps.values() if len(ps) > 1)
    minutes = meta_inst["n_frames"] / fps / 60.0
    return dict(tp=tp, fp=fp, fn=fn, fp_frag=fp_frag, fp_foreign=fp_foreign,
                fa=fa, fire_ok=fire_ok, excess=excess, swaps=swaps, minutes=minutes)


# ── evaluation over all instances ──────────────────────────────────────────────

def evaluate(d, meta, proto, tracker_kw=None, stab_kw=None, journal_kw=None,
             seed=SEED, groups=None):
    """proto = (dropout, jitter, face, burst, p_flip). Fresh RNG per call: paired
    configs with equal emitter params see identical detection streams."""
    dropout, jitter, face, burst, p_flip = proto
    rows = {}
    for idx, (inst, m) in enumerate(meta.items()):
        if groups and m["group"] not in groups:
            continue
        frames = frames_of(d, inst, m["n_frames"])
        emitted = emit(frames, (seed, idx), dropout, jitter, face, burst, p_flip)
        res = run_instance(emitted, objects_of(d, inst), m["n_frames"],
                           tracker_kw, stab_kw, journal_kw)
        rows[inst] = (score_instance(res, m), res)
    return rows


def summarize(rows):
    s = collections.Counter()
    quiet_n = quiet_bad = fire_n = fire_hit = 0
    for sc, _ in rows.values():
        s["tp"] += sc["tp"]; s["fp"] += sc["fp"]; s["fn"] += sc["fn"]
        s["fp_frag"] += sc["fp_frag"]; s["fp_foreign"] += sc["fp_foreign"]
        s["excess"] += sc["excess"]; s["swaps"] += sc["swaps"]
        s["minutes"] += sc["minutes"]
        if sc["fa"] is not None:
            quiet_n += 1; quiet_bad += sc["fa"]
        if sc["fire_ok"] is not None:
            fire_n += 1; fire_hit += sc["fire_ok"]
    prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 1.0
    rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 1.0
    return dict(precision=prec, recall=rec, tp=s["tp"], fp=s["fp"], fn=s["fn"],
                fp_frag=s["fp_frag"], fp_foreign=s["fp_foreign"],
                fa_rate=quiet_bad / quiet_n if quiet_n else 0.0, quiet_n=quiet_n,
                fire_rate=fire_hit / fire_n if fire_n else None, fire_n=fire_n,
                excess_per_min=s["excess"] / s["minutes"], swaps=s["swaps"])


def ci_prf(rows, n_boot=N_BOOT, seed=SEED):
    """Grouped bootstrap over instances: precision & recall CIs."""
    arr = np.array([(sc["tp"], sc["fp"], sc["fn"]) for sc, _ in rows.values()], float)
    rng = np.random.default_rng(seed)
    ps, rs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(arr), len(arr))
        tp, fp, fn = arr[i].sum(0)
        ps.append(tp / (tp + fp) if tp + fp else 1.0)
        rs.append(tp / (tp + fn) if tp + fn else 1.0)
    pct = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return pct(ps), pct(rs)


# ── S2: prod anomaly vs the LJ2 E2 rule on the same runs ───────────────────────

def e2_stats(frames, fps=FPS):
    """LJ2 E2 on the injected world pelvis: reached floor (min pelvis y <= E2_FLOOR_M)
    AND peak windowed descent speed >= E2_VEL_MS. Median-filtered heights, 0.4 s drop
    window — the lj2_fall_velocity robust recipe at the production cadence."""
    y = np.array([fl[0]["pelvis3d"][1] if fl else np.nan for fl in frames], float)
    y = y[~np.isnan(y)]
    if len(y) < 3:
        return False, np.inf, 0.0
    k = 3
    med = np.array([np.median(y[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(y))])
    w = max(1, int(round(E2_WIN_S * fps)))
    drops = med[:-w] - med[w:]
    vmax = float(drops.max() / E2_WIN_S) if len(drops) else 0.0
    fired = bool(med.min() <= E2_FLOOR_M and vmax >= E2_VEL_MS)
    return fired, float(med.min()), vmax


def s1b_exit_dwell_sweep(d, meta):
    """The FLOOR run put P/R at 1.0 and the med ablations put ALL precision loss in
    episode fragments that no tracker/stabilizer knob touches -> the suspect is the
    FSM exit grace (5 frames = 0.33 s) vs realistic multi-frame occlusion bursts.
    Sweep it (young raised in step; prod = 5/8)."""
    print("\n" + "-" * 78)
    print("S1b exit-dwell sweep @ STRESS med/heavy — FSM grace vs occlusion bursts")
    print("-" * 78)
    for name, kw in [("prod 5/8", None), ("8/11", dict(exit_dwell=8, exit_young=11)),
                     ("10/13", dict(exit_dwell=10, exit_young=13)),
                     ("12/15", dict(exit_dwell=12, exit_young=15))]:
        for sname in ("med", "heavy"):
            rows = evaluate(d, meta, STRESS[sname], journal_kw=kw)
            s = summarize(rows)
            pci, rci = ci_prf(rows)
            print(f"exit={name:9} {sname:5} {fmt(s, pci, rci)}")


def s1c_g4_diagnostics(d, meta):
    """Why do falls get missed under stress? Trace what the anomaly actually SAW."""
    print("\n" + "-" * 78)
    print("S1c G4 under STRESS med — what reached the anomaly at the critical moment")
    print("-" * 78)
    idx_of = {inst: i for i, inst in enumerate(meta)}
    for inst, m in meta.items():
        if m["group"] != "G4":
            continue
        frames = frames_of(d, inst, m["n_frames"])
        emitted = emit(frames, (SEED, idx_of[inst]), *STRESS["med"])
        res = run_instance(emitted, objects_of(d, inst), m["n_frames"])
        # Reconstruct the STABILIZED posture stream the anomaly consumed.
        seen = []
        stabs = {}
        for f, fl in enumerate(emitted):
            for _, row in fl:
                st = stabs.setdefault(0, Stabilizer())
                seen.append((f, st.step(row["posture_obs"], POSTURE_CONF_DEFAULT)))
        stand = [f for f, l in seen if l == "standing"]
        lying = [f for f, l in seen if l == "lying"]
        gap = (lying[0] - stand[-1]) / FPS if stand and lying and lying[0] > stand[-1] \
            else ("n/a" if not (stand and lying) else "overlap")
        fired = bool((res["anomaly"] >= 0.5).any())
        print(f"{inst:12} fired={str(fired):5} stand-obs={len(stand):3} "
              f"lying-obs={len(lying):3} last-stand->first-lying="
              f"{gap if isinstance(gap, str) else f'{gap:.2f}s'} "
              f"(window {B_FALL_WINDOW}s)")


B_FALL_WINDOW = 3.0


def s2_run(d, meta, inst, proto, idx):
    """Drive tracker+stabilizer over one G3/G4 instance under `proto`, recording what
    a fall rule could actually consume: per-frame delivered (tid, stabilized label,
    observed pelvis y). Also returns the run's prod anomaly trace."""
    m = meta[inst]
    frames = frames_of(d, inst, m["n_frames"])
    emitted = emit(frames, (SEED, idx), *proto)
    res = run_instance(emitted, objects_of(d, inst), m["n_frames"])
    stabs = {}
    stream = []                          # (t, tid, stab_label, pelvis_y)
    tracker = PersonTracker()
    for f, fl in enumerate(emitted):
        ids = tracker.tick([dt for dt, _ in fl])
        for tid, (_, row) in zip(ids, fl):
            st = stabs.setdefault(tid, Stabilizer())
            stream.append((f / FPS, tid, st.step(row["posture_obs"], POSTURE_CONF_DEFAULT),
                           float(row["pelvis3d"][1])))
    return res, stream


def rule_prod_idfree(stream, window=3.0):
    """The prod fall rule with the track-id dependency removed: the standing memory is
    SCENE-level (any body stood recently -> a body lying now is a fall). Motivated by
    the S1c finding: the fall itself churns the id, and the new id 'never stood'."""
    last_stand = None
    for t, _, lab, _ in stream:
        if lab == "standing":
            last_stand = t
        if lab == "lying" and last_stand is not None and t - last_stand <= window:
            return True
    return False


def rule_e2(stream, thresh, fps=FPS):
    """E2 on the DELIVERED pelvis stream (gaps interpolated onto the frame grid — a
    long occlusion smooths the descent and honestly weakens the rule)."""
    obs = [(t, y) for t, _, _, y in stream]
    if len(obs) < 3:
        return False
    t_obs = np.array([t for t, _ in obs])
    y_obs = np.array([y for _, y in obs])
    grid = np.arange(t_obs[0], t_obs[-1], 1.0 / fps)
    y = np.interp(grid, t_obs, y_obs)
    k = 3
    med = np.array([np.median(y[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(y))])
    w = max(1, int(round(E2_WIN_S * fps)))
    if len(med) <= w:
        return False
    vmax = float((med[:-w] - med[w:]).max() / E2_WIN_S)
    return bool(med.min() <= E2_FLOOR_M and vmax >= thresh)


def s2_table(d, meta):
    print("\n" + "=" * 78)
    print("S2  FALL RULES on G3 (quiet: controlled floor) vs G4 (falls), FLOOR + STRESS med")
    print("    prod        = AnomalyScorer, standing->lying <=3s window, PER-TRACK-ID")
    print("    prod-idfree = same window, scene-level standing memory (churn-robust)")
    print(f"    E2          = floor<={E2_FLOOR_M}m AND descent>={E2_VEL_MS}m/s on delivered pelvis")
    print(f"    E2-recal    = same, threshold {E2_VEL_RECAL} m/s (15 fps recalibration)")
    print("=" * 78)
    idx_of = {inst: i for i, inst in enumerate(meta)}
    for pname, proto in [("FLOOR", FLOOR), ("STRESS med", STRESS["med"])]:
        stats = collections.Counter()
        detail = []
        for inst, m in meta.items():
            if m["group"] not in ("G3", "G4"):
                continue
            g = m["group"]
            res, stream = s2_run(d, meta, inst, proto, idx_of[inst])
            fired = {
                "prod": bool((res["anomaly"] >= 0.5).any()),
                "idfree": rule_prod_idfree(stream),
                "e2": rule_e2(stream, E2_VEL_MS),
                "e2rec": rule_e2(stream, E2_VEL_RECAL),
            }
            want = g == "G4"
            for k, v in fired.items():
                stats[f"{k}_{g}"] += v
            stats[f"{g}_n"] += 1
            miss = [k for k, v in fired.items() if v != want]
            if miss:
                detail.append(f"   {inst}: " + ", ".join(
                    f"{k}={'FALSE ALARM' if fired[k] else 'MISS'}" for k in miss))
        print(f"\n  {pname}:")
        print(f"  {'rule':12} {'G3 false alarms':>16} {'G4 hits':>9}")
        for k, label in [("prod", "prod"), ("idfree", "prod-idfree"),
                         ("e2", "E2 @1.13"), ("e2rec", "E2 @1.05")]:
            print(f"  {label:12} {stats[f'{k}_G3']:>12}/{stats['G3_n']:<3} "
                  f"{stats[f'{k}_G4']:>5}/{stats['G4_n']}")
        for line in detail:
            print(line)


# ── driver ─────────────────────────────────────────────────────────────────────

def fmt(s, pci=None, rci=None):
    p = f"{s['precision']:.3f}" + (f" [{pci[0]:.2f},{pci[1]:.2f}]" if pci else "")
    r = f"{s['recall']:.3f}" + (f" [{rci[0]:.2f},{rci[1]:.2f}]" if rci else "")
    fire = "—" if s["fire_rate"] is None else f"{s['fire_rate']:.2f}"
    return (f"P={p:22} R={r:22} FP(frag/foreign)={s['fp_frag']}/{s['fp_foreign']} "
            f"FA={s['fa_rate']:.2f}({s['quiet_n']}) fall-hit={fire} "
            f"excess/min={s['excess_per_min']:.2f} swaps={s['swaps']}")


def main():
    d, meta = load()
    print(f"instances: {len(meta)}  groups="
          f"{dict(collections.Counter(m['group'] for m in meta.values()))}")

    print("\n" + "=" * 78)
    print("S1  EVENT P/R vs scenario  (leading metric; grouped bootstrap CI over instances)")
    print("=" * 78)
    base = {}
    for name, proto in [("FLOOR", FLOOR)] + [(f"STRESS {k}", v) for k, v in STRESS.items()]:
        rows = evaluate(d, meta, proto)
        s = summarize(rows)
        pci, rci = ci_prf(rows)
        base[name] = rows
        print(f"{name:14} {fmt(s, pci, rci)}")
        if name == "FLOOR":
            for inst, (sc, _) in rows.items():
                if sc["fp"] or sc["fn"] or sc["fa"] or sc["fire_ok"] is False or sc["excess"] or sc["swaps"]:
                    print(f"   !! {inst}: {sc}")

    print("\n" + "-" * 78)
    print("S1 ablations @ STRESS med — attribute the degradation to a layer")
    print("-" * 78)
    med = STRESS["med"]
    for name, kw in [
        ("baseline (prod)", {}),
        ("tracker=fix15.3", dict(tracker_kw=dict(center_relink_lowconf_id=True))),
        ("stab=hybrid sl12", dict(stab_kw=dict(sl_dwell=12))),
        ("no face (no id)", dict(proto=(med[0], med[1], 0, med[3], med[4]))),
        ("no posture flip", dict(proto=(med[0], med[1], med[2], med[3], 0.0))),
    ]:
        proto = kw.pop("proto", med)
        rows = evaluate(d, meta, proto, **kw)
        s = summarize(rows)
        print(f"{name:16} {fmt(s)}")

    s1b_exit_dwell_sweep(d, meta)
    s1c_g4_diagnostics(d, meta)
    s2_table(d, meta)

    print("\npassport:", bench.passport([GOLD]))


if __name__ == "__main__":
    main()
