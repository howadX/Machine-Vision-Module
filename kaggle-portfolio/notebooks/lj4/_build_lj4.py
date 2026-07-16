#!/usr/bin/env python3
"""Builder for LJ4 (Kaggle): "A camera that never lies" — MVM Lab Journal 4.

Every ML layer of the stack has now been measured in isolation (pose LJ1, transitions
LJ2, tracker LJ3) — but their COMPOSITION (tracker -> activities -> FSM -> anomaly ->
journal) was only ever eyeballed on a live camera where the truth is unknowable. This
journal builds a simulator: synthetic golden scenarios injected AFTER the detectors,
with truth known by construction, driven through bit-faithful ports of the production
temporal stack. Any disagreement between the journal and the scenario is a defect of
the stack, not "maybe that's what happened".

Authoring twist vs LJ1-LJ3: the heavy code cells are NOT hand-inlined copies — this
builder reads the actual lab sources (spike_pose/lj4_fusion.py, lj3_tracker.py,
lj4_bench.py, lj4_vrd_audit.py) and embeds them with minimal surgery (repo-local
imports / __file__ paths). The notebook literally runs the lab's code, self-checks
included, so the published numbers cannot drift from the repo's.

Reproducible authoring: `python _build_lj4.py` -> lj4_a_camera_that_never_lies.ipynb,
Run-All on Kaggle over the published `mvm-golden-scenarios` dataset.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.join(HERE, "..", "..", "..", "spike_pose")

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": t.strip("\n").splitlines(keepends=True)})


def src(name, drop_patterns=(), replace=()):
    """Read a lab source file and surgically adapt it for the notebook namespace."""
    with open(os.path.join(SPIKE, name)) as fh:
        text = fh.read()
    lines = []
    for ln in text.splitlines():
        if any(re.search(p, ln) for p in drop_patterns):
            continue
        lines.append(ln)
    text = "\n".join(lines)
    for a, b in replace:
        text = text.replace(a, b)
    return text


# ============================================================ 1. Title
md(r"""
# LJ4 — A camera that never lies
### simulating the world for the fusion layer: golden scenarios vs the production temporal stack

*MVM Lab Journal — 4 of a series. Prev: LJ3 (Why walking people multiply). Next: LJ5 (Weighing
the experts).*

> The pose model is honest now (LJ1), the transitions have physics (LJ2), the tracker no longer
> multiplies walkers (LJ3) — each layer measured **in isolation**. But the product ships their
> *composition*: tracker → activities → FSM → anomaly → journal. That composition has only ever
> been judged on a live camera, where nobody knows what actually happened, so nobody can say
> "the journal lied". This journal builds a camera that cannot lie: scenarios whose truth is
> known **by construction** (from CMU mocap), injected *after* the pixel detectors, driven
> through bit-faithful ports of the production stack. Three questions: does the stack reproduce
> clean scenarios exactly? what breaks it under realistic detector noise — and which layer is
> guilty? and does the fall alarm actually tell a fall from lying down?
""")

# ============================================================ 2. Context
md(r"""
## 1. Context — the layers are measured; the composition is folklore

The live product exhibited two composite failures that no unit test ever caught: a walking person
fragmenting into phantom tracks fired a false `people_together` (two people who never existed), and
the fall alarm silently depended on details nobody had measured. LJ3 fixed the fragmentation at its
cause — but the *event level* (what the journal tells an LLM consumer) remained untested, because
on a live camera there is no ground truth to score against.

**The simulator's honest boundary:** we inject synthetic observations (bbox + posture + wrist
points + metric 3D) *after* the detectors. The pixel layer is explicitly out of scope — LJ1 and
LJ3 measured it. Everything above it runs for real, as byte-faithful Python ports of the Rust:
`infer_activities` (raw cross-group inference), the Activity FSM (enter/exit dwell, cooling,
churn stitching), the anomaly scorer (noisy-OR + decaying peak), the posture stabilizer (LJ2), and
the LJ3 tracker port. Each port self-checks against the Rust unit-test scenarios below — a port
that drifts is a stop-ship.

**Scenarios** (compiled from CMU mocap in world 3D, pelvis-continuous seams, eye-level camera):

| group | story | instances | the contract |
|---|---|---|---|
| G1 | walked in → sat → works at a keyboard | 3 | one `typing` episode, right timing; quiet anomaly |
| G2 | stood up → left the room | 3 | `typing` ends; track disappears; quiet |
| G3 | lay down / got up (controlled floor) | 10 | **anomaly stays silent** |
| G4 | fell | 6 | **anomaly fires** |
| G5 | two people met | 2 | one `people_together`, no id swap |
| G6 | a lone walker (LJ3 regression) | 4 | no events at all, one track |

Two compiler lessons paid for in blood (both were caught by the FLOOR run, not by eyeballs): the
BVH settling frame injects a fake `standing→lying` edge at t=0 (a phantom fall!), and expected
events must be derived from **what the injected data contains**, not what the author intended —
a walking-in person's wrists hang at seated-keyboard height, so 2D contact (production semantics,
no depth) honestly starts `typing` during the final approach steps. The production stack behaves
identically; the scenario truth now says so.
""")

# ============================================================ 3. Hypotheses
md(r"""
## 2. Hypotheses — failure criteria named up front

**S1 (the gate).** On clean injection (FLOOR) the production stack reproduces every scenario's
events with precision = recall = 1.0 within dwell tolerances and zero phantom events. Under
realistic detector stress (bursty occlusion, bbox jitter, intermittent face, posture flips) the
degradation is *measurable and attributable* to a layer via paired ablations.

> **Failure criterion.** If FLOOR already disagrees, it's either a stack defect (publish it — the
> simulator exists for exactly this), an unusable scenario truth (fix the compiler, not the
> tracker), or a broken port (stop-ship). No tuning until attribution.

**S2.** The production fall rule (*was standing within 3 s → lying now*, keyed **per track id**,
no velocity) false-alarms on fast controlled lie-downs and/or loses falls under stress; the LJ2
physics rule E2 (*reached floor AND peak pelvis descent ≥ 1.13 m/s*) separates them cleanly on the
same runs.

> **Failure criterion.** If the production rule behaves on both G3 and G4 — E2 stays in the lab
> and the 15.6 contract change is cancelled. If E2's 60 fps threshold fails at the production
> 15 fps cadence, we say so and recalibrate or retreat. Both endings publishable.

**S3.** Half the raw-activity vocabulary (`typing`, `using_mouse`, `watching_tv`…) is an artifact
of eyeballed model play that **no data can supervise** — measured against OpenImages VRD
*instances* (human-labeled, per-image), not the triplet vocabulary.

> **Failure criterion.** If VRD instances are plentiful for the current labels, the "artifact"
> directive is refuted by data — the vocabulary stays, and we say so.
""")

# ============================================================ 4. Data
md(r"""
## 3. Experiment — the scenario artifact and the ported stack

`golden_scenarios.npz`: 28 instances, 5 277 person-rows @15 fps (the production human-chain
cadence). Per frame: true person id, bbox, kinematic posture (biomechanics labeller, *not* the
GRIP features), wrist points, camera-frame pelvis (3D `people_together` distance) and world pelvis
(fall physics). Expected events + tolerances were emitted by the compiler — the scenario file *is*
the ground truth. The same file (a 6-instance FLOOR subset as JSON) drives a `cargo test` golden
gate inside the Rust workspace, so CI pins today's contract while this journal measures it.
""")

code(r"""
# ---- locate the scenario artifact (Kaggle dataset first, repo fallback) ----
import os, glob, json, collections
import numpy as np

def _find(*names):
    roots = ["/kaggle/input/mvm-golden-scenarios",
             os.path.join("..","..","..","spike_pose","data_probe"),
             os.path.join("..","..","spike_pose","data_probe")]
    for r in roots:
        for n in names:
            if os.path.exists(os.path.join(r, n)): return os.path.join(r, n)
    for n in names:
        hits = glob.glob(f"/kaggle/input/**/{n}", recursive=True)
        if hits: return hits[0]
    have = os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "no /kaggle/input"
    raise FileNotFoundError((names, "mounted:", have))

GOLD = _find("golden_scenarios.npz")
_d = np.load(GOLD, allow_pickle=True)
_meta = json.loads(str(_d["meta_json"]))
print(f"{len(_meta)} instances:",
      dict(collections.Counter(m["group"] for m in _meta.values())),
      f"| {len(_d['inst'])} person-rows @15fps")
""")

# ============================================================ 5. Ports (embedded sources)
md(r"""
### 3.1 The fusion stack, ported byte-for-byte — and self-checked against the Rust unit tests

The next two cells are **the lab's actual source files**, embedded by the notebook builder (not
hand-copied): `lj4_fusion.py` (raw activities + FSM + anomaly, constants quoted at their Rust
definitions) and `lj3_tracker.py` (the LJ3 association port). Each ends by replaying the Rust
unit-test scenarios — 39 for the fusion stack — as a live bit-faithfulness check.
""")

# In Jupyter __name__ == "__main__", so each file's own guard runs its self-check live.
code(src("lj4_fusion.py"))

code(src("lj3_tracker.py"))

md(r"""
### 3.2 The bench: emitter, full-stack runner, event scoring

Also the lab source (`lj4_bench.py`), embedded with its repo-local imports stripped (everything
already lives in this namespace). Key honesty details:

- **FLOOR** = clean injection; **STRESS** = the LJ3 live-detector model (bursty occlusion,
  bbox jitter, intermittent early face — the dead-zone trigger) plus per-frame posture flips.
- Every noise source draws from its **own** seeded generator: switching one off in an ablation
  must not reshuffle the others (the first run caught exactly this — an unpaired "no posture
  flip" ablation scored *worse* than baseline).
- Scoring is event-level: label + participants (mapped through tracker-committed ids) + start/end
  within tolerance. Unmatched emitted events split into **fragments** (right activity, broken
  episode) vs **foreign** (invented activity) — the attribution hinge.
""")

code(src(
    "lj4_bench.py",
    drop_patterns=[
        r"^HERE = os\.path", r"^sys\.path\.insert", r"^import os, sys",
        r"^from lj3_tracker import", r"^from lj4_fusion import", r"^import bench$",
        r"^GOLD = os\.path",
    ],
    # The notebook drives the acts cell by cell; the module's own driver would
    # also call bench.passport (stripped above).
    replace=[('if __name__ == "__main__":\n    main()', "")],
))

# ============================================================ 6. S1 results
md(r"""
## 4. Numbers — S1: the FLOOR gate, then stress with attribution

First the gate: clean injection through the full stack, every instance.
""")

code(r"""
# imports the embedded bench cell needs at the top of this namespace
import os, sys, json, collections
import numpy as np

print("=" * 78)
print("S1  EVENT P/R vs scenario (grouped bootstrap CI over instances, N=1000, seed=0)")
print("=" * 78)
base_rows = {}
for name, proto in [("FLOOR", FLOOR)] + [(f"STRESS {k}", v) for k, v in STRESS.items()]:
    rows = evaluate(_d, _meta, proto)
    s = summarize(rows)
    pci, rci = ci_prf(rows)
    base_rows[name] = rows
    print(f"{name:14} {fmt(s, pci, rci)}")

print("\nFLOOR flags (findings, not scoring failures):")
for inst, (sc, _) in base_rows["FLOOR"].items():
    if sc["fp"] or sc["fn"] or sc["fa"] or sc["fire_ok"] is False or sc["excess"] or sc["swaps"]:
        print(f"  !! {inst}: fp={sc['fp']} fn={sc['fn']} false_alarm={sc['fa']} "
              f"fall_hit={sc['fire_ok']} excess={sc['excess']}")
""")

md(r"""
**The FLOOR gate is taken: event precision = recall = 1.000, zero phantom events, zero swaps** —
the production temporal stack, fed clean post-detector observations, reproduces every scenario.
The flags the clean run *does* raise are findings, not scoring noise: the production anomaly
false-alarms on three fast controlled lie-downs (that's S2's business), and the fastest stunt
fall (`G4_88_04`, bbox centre jumping 0.175/frame — real kinematics) fragments the tracker even
on clean boxes.

Under stress the event level collapses (P 0.20 → 0.13 → 0.03 mild/med/heavy) — and the FP
breakdown says **100% fragments, 0 foreign**: the stack never invents a wrong activity, it
shatters the right ones. Who's guilty? Paired ablations:
""")

code(r"""
print("-" * 78)
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
    rows = evaluate(_d, _meta, proto, **kw)
    print(f"{name:16} {fmt(summarize(rows))}")
""")

md(r"""
Event P/R is **completely invariant** to the tracker fix, the stabilizer, the face and the posture
noise — while `excess/min` swings 5.38 → 1.91 with the LJ3 fix (nicely cross-validating LJ3's
5.43 → 0.78 on an independent harness). Track fragmentation and *event* fragmentation are
decoupled here: typing happens while seated (the tracker holds), and what shatters episodes is the
occlusion bursts themselves outlasting the FSM's exit grace (5 frames = 0.33 s). So the suspect is
a constant nobody had ever measured:
""")

code(r"""
print("-" * 78)
print("S1b exit-dwell sweep — FSM grace vs occlusion bursts (prod = 5/8)")
print("-" * 78)
for name, kw in [("prod 5/8", None), ("8/11", dict(exit_dwell=8, exit_young=11)),
                 ("10/13", dict(exit_dwell=10, exit_young=13)),
                 ("12/15", dict(exit_dwell=12, exit_young=15))]:
    for sname in ("med", "heavy"):
        rows = evaluate(_d, _meta, STRESS[sname], journal_kw=kw)
        s = summarize(rows)
        pci, rci = ci_prf(rows)
        print(f"exit={name:9} {sname:5} {fmt(s, pci, rci)}")
""")

md(r"""
Monotone and large: raising the exit grace from 5/8 to 12/15 frames lifts event precision at the
live-like operating point from **0.133 to 0.667** (heavy: 0.029 → 0.545), by bridging exactly the
gaps that were splitting episodes. The cost is honest and small — an episode's end is committed
~0.5 s later. This is a 15.6 port candidate, licensed by measurement (and it keeps the
`> object_cadence` invariant the constant was originally chosen for).

One more stress casualty deserves its own act: the **fall alarm**.
""")

# ============================================================ 7. S2
md(r"""
## 4b. Numbers — S2: does the alarm know a fall from lying down?

The production rule (`anomaly.rs`): *a body that was standing within the last 3 s and reads lying
now* — keyed per track id, velocity-free. LJ2's E2 rule: *reached the floor* (min pelvis ≤ 0.35 m)
*AND peak downward pelvis velocity ≥ 1.13 m/s* (calibrated at 60 fps). The simulator can finally
race them on identical runs — including two variants the failures below motivate: an **id-free**
production rule (scene-level standing memory) and E2 **recalibrated to the production cadence**.

Under stress, why does the per-id rule lose falls? Trace the critical moment:
""")

code(r"""
s1c_g4_diagnostics(_d, _meta)
print()
s2_table(_d, _meta)
""")

md(r"""
Three verdicts, all measured:

1. **The per-id production rule is broken on both sides.** On clean data it false-alarms on 3/10
   controlled floor scenarios — a person who lies down *briskly* (standing → sitting → lying in
   0.9–2.4 s) is inside the 3 s window; the LJ2 stabilizer veto can't help because these
   transitions legitimately pass through sitting. Under stress it drops to 4/6 hits — and the
   trace shows why: **the fall itself churns the track id** (the impact occludes the detector for
   a few frames; the bbox collapses faster than the re-link's centre gate), and the new id "never
   stood". The standing memory dies with the track exactly when it's needed.
2. **E2 at its 60 fps threshold misses one fall at 15 fps** — `G4_85_15` peaks at 1.12 m/s against
   the 1.13 threshold. The production cadence smears the peak; the threshold does not transfer
   as-is.
3. **E2 recalibrated to the cadence (≥1.05 m/s), scene-level: 0/10 false alarms + 6/6 hits, on
   FLOOR *and* under stress.** The honest caveat: the margin is thin (the worst quiet scenario
   peaks at 0.99, the weakest fall at 1.12), and CMU falls are stunt-performed — the threshold
   bounds a clinical fall from above (LJ2's caveat, still standing). **Ships to 15.6**: the E2
   physics rule, id-robust, cadence-recalibrated; the per-id window rule retires.
""")

# ============================================================ 8. S3
md(r"""
## 4c. Numbers — S3: which activities can data even supervise?

The raw-activity vocabulary (`typing`, `using_mouse`, `using_phone`, `holding_drink`, `reading`,
`watching_tv`) was an artifact of early eyeballed model play. OpenImages VRD is the only large
corpus with *human-labeled* visual relationships — below we count **per-image instances**
(val+test splits, ~110k rows), not the triplet vocabulary (which only says a pair is *possible*).
""")

code(r"""
# fetch the VRD metadata (public GCS, ~12 MB) into the notebook workspace
import urllib.request
os.makedirs("oi_meta", exist_ok=True)
for f in ["oidv6-validation-annotations-vrd.csv", "oidv6-test-annotations-vrd.csv",
          "oidv6-class-descriptions.csv"]:
    dst = os.path.join("oi_meta", f)
    if not os.path.exists(dst):
        urllib.request.urlretrieve(f"https://storage.googleapis.com/openimages/v6/{f}", dst)
        print("fetched", f)
""")

code(src(
    "lj4_vrd_audit.py",
    drop_patterns=[r"^HERE = os\.path", r"^META = os\.path", r"^OUT = os\.path",
                   r"^import os, csv"],
    replace=[("if __name__ == \"__main__\":\n    main()", "")],
) + '\nimport csv\nMETA = "oi_meta"\nOUT = "./LJ4-VRD-AUDIT.md"\nmain()')

md(r"""
The directive survives contact with the data — dramatically. Of the six shipped labels, three
have **zero** human-labeled instances (`typing`, `using_mouse`, `watching_tv`), two are near-zero
(`using_phone` 15, `reading` 24); only `holding_drink` (145) has real support. Meanwhile the
predicates the vocabulary *ignores* carry the mass: generic `holds` (2 153), `at` Table/Desk
(1 360 — corroborating `seated_at_workstation`), `ride` (1 432), `plays` (618). And the `is`
states hold a quiet treasure for other tracks: Standing 39 683 / Sitting 14 529 / **Walking 3 602 /
Running 2 156** (gait supervision for `motion_state`) — but **no lying state at all**: CMU mocap
remains the only fall/lying source, re-confirming the reference-pair complementarity from LJ0.

**Vocabulary v2 verdict** (code change lands in 15.6 as a documented contract change): keep the
*mechanism* (wrist/hand contact, object-major attribution — it's unit-tested geometry), fix the
*names*. Fold `typing`/`using_mouse`/`using_phone`/`reading` into a data-supportable
`using_object(class)` umbrella; **drop** `watching_tv` (a wrist on the remote is not "watching" —
watching is head-facing, a different signal); keep `holding_drink` and the geometric labels
(`people_together`, `seated_at_workstation`, `partially_in_frame`). The sports/riding family is
supervisable but out-of-domain for a room camera — deferred to phase 16.
""")

# ============================================================ 9. Verdict
md(r"""
## 5. Verdict

| hypothesis | outcome |
|---|---|
| S1: FLOOR P/R = 1.0, zero phantoms | **confirmed** — the gate is taken |
| S1: stress degradation attributable | **confirmed** — 100% fragments; invariant to tracker/stabilizer/face/flip; the knob is the FSM exit grace (0.133 → 0.667 @ 12/15) |
| S2: prod fall rule flawed | **confirmed twice** — 3/10 false alarms on clean lie-downs; id-churn loses 2/6 falls under stress |
| S2: E2 transfers | **partially** — 60 fps threshold misses one fall at 15 fps; recalibrated (≥1.05) it is clean on both axes, thin margin stated |
| S3: vocabulary is an artifact | **confirmed by data** — 3 labels at zero instances; verdict table shipped |

And one meta-result: **the simulator caught its own compiler twice** (the settling-frame phantom
fall, the walk-in typing contact) before it ever tested the product — exactly the discipline the
failure criteria demanded.
""")

# ============================================================ 10. What it changes
md(r"""
## 6. What it changes — and the bridge to LJ5

Into the 15.6 port package, each item licensed by a number in this journal:

1. **Fall rule**: retire the per-id 3 s window; ship the E2 physics rule (floor + pelvis descent),
   **scene-level** (id-robust — the fall itself churns ids), threshold recalibrated at the
   production cadence.
2. **FSM exit grace**: raise from 5/8 (a constant chosen for YOLO cadence flicker, never measured
   against occlusion bursts) toward 10–12/13–15 — event precision at live-like noise triples.
3. **Vocabulary v2**: `using_object(class)` umbrella, `watching_tv` dropped — a documented
   StateFile/MCP contract change, alongside LJ1's flexion channel.
4. **Golden scenarios in CI, permanently**: a 6-instance FLOOR subset now runs as a model-free
   `cargo test` inside the Rust workspace (the same compiled file this notebook scores), pinning
   today's contract; the asserts that these findings flip are marked for 15.6.

What the simulator does *not* cover, honestly: the pixel detectors (out of scope by design), the
head-on full-occlusion id swap (inherited from LJ3, characterized here on G5-STRESS, still open),
and clinical fall kinematics (CMU's falls are stunts; the threshold is an upper bound).

Next, **LJ5 — Weighing the experts**: with the logic layers measured, what do the ONNX experts
actually cost — latency, bytes, and honest public-bench accuracy — and what can the 15.1 results
already delete?
""")

# ============================================================ 11. Passport
code(r"""
# ---- passport: bind every number to the exact data (+ code commit when run from the repo) ----
import hashlib, subprocess
def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    commit = "(kaggle run — no git)"
passport = {"code_commit": commit, "golden_scenarios": sha16(GOLD)}
print("passport:", passport)
""")

# ---------------------------------------------------------------- write notebook
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "lj4_a_camera_that_never_lies.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "—", len(cells), "cells")
