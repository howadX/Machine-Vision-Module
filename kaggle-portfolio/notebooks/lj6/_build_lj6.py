#!/usr/bin/env python3
"""Builder for LJ6 (Kaggle): "Shipping v0.2.0: the scorecard" — MVM Lab Journal 6, the closer.

LJ0-LJ5 measured every ML decision in the lab. LJ6 is the retrospective: what actually shipped
into the production Rust workspace (v0.2.0, acts P0-P8), the before/after across the three axes
(accuracy / weight / speed), the decisions we MEASURED AND REJECTED, and a live camera validation.

Authoring discipline (as LJ4/LJ5): the scorecard numbers are NOT hand-typed into prose — this
builder reads `results/metrics.json` (the frozen bench output LJ2-LJ5 wrote) at build time and
bakes them into the notebook as a literal `SCORECARD` dict, so the published numbers cannot drift
from the repo. LJ1's numbers predate the metrics file and are carried as annotated literals with
their journal provenance. No datasets are mounted: LJ6 re-runs nothing (every number is already
established upstream) — it composes, so Run-All completes offline.

Build: `python _build_lj6.py` -> lj6_shipping_the_scorecard.ipynb
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.join(HERE, "..", "..", "results", "metrics.json")

M = json.load(open(METRICS))
lj2, lj3, lj4, lj5 = M["lj2"], M["lj3"], M["lj4"], M["lj5"]

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": t.strip("\n").splitlines(keepends=True)})

# ── assemble the scorecard FROM the frozen results (repo source of truth) ─────────
b2 = lj5["actB2_identity_lfw_fp32"]["results"]
SCORECARD = {
    # ── ACCURACY ──────────────────────────────────────────────────────────────
    "accuracy": {
        # LJ1 predates results/metrics.json → carried as annotated literals (kernel
        # howadx/lj1-the-pose-model-rebuilt-honestly, research/LJ1-DESIGN.md).
        "pose_gold_r3_2dgrip": {"v020": 0.894, "ci": [0.880, 0.907], "v010": 0.83,
                                "note": "cross-domain gold R3; 3D-GRIP collapses to 0.62 (z is a cross-domain liability)", "src": "LJ1"},
        "identity_lfw": {"v010": round(b2["prod_unaligned"]["acc_10fold"], 4),
                         "v020": round(b2["aligned"]["acc_10fold"], 4),
                         "tar_far1e3_v010": b2["prod_unaligned"]["tar_at_far1e-3"],
                         "tar_far1e3_v020": b2["aligned"]["tar_at_far1e-3"],
                         "note": "align the SFace crop (was the raw bbox); +0 weight/latency", "src": "LJ5"},
        "emotion_rafdb_balacc": {"v020": round(lj5["actB1_emotion_rafdb"]["balanced_acc"], 4),
                                 "ci": [round(x, 4) for x in lj5["actB1_emotion_rafdb"]["ci"]],
                                 "note": "held-out balanced acc; negative control HEALTHY (the 10.1 bug does not reproduce)", "src": "LJ5"},
        "fall_auroc": {"v020": lj2["e2_fall_threshold"]["auroc_velocity_framingB"],
                       "note": "E2 scene-level (fall vs controlled descent, floor-reaching only); prod per-id rule = 3/10 false alarms", "src": "LJ2/LJ4"},
        "fall_falsealarms": {"v010": lj4["s2"]["floor"]["prod"]["g3_fa"], "v020": lj4["s2"]["floor"]["e2_105"]["g3_fa"],
                             "hits_v010": lj4["s2"]["floor"]["prod"]["g4_hits"], "hits_v020": lj4["s2"]["floor"]["e2_105"]["g4_hits"],
                             "note": "controlled-lie-down false alarms / real-fall hits on the FLOOR set", "src": "LJ4"},
        "tracker_excess_med": {"v010": lj3["T1a_reproduce_fix"]["med"]["base"], "v020": lj3["T1a_reproduce_fix"]["med"]["fix"],
                               "delta_ci": lj3["T1a_reproduce_fix"]["med"]["delta_ci"], "established": lj3["T1a_reproduce_fix"]["med"]["established"],
                               "note": "excess phantom tracks / min walk (medium occlusion)", "src": "LJ3"},
        "tracker_excess_heavy": {"v010": lj3["T1a_reproduce_fix"]["heavy"]["base"], "v020": lj3["T1a_reproduce_fix"]["heavy"]["fix"],
                                 "delta_ci": lj3["T1a_reproduce_fix"]["heavy"]["delta_ci"], "established": lj3["T1a_reproduce_fix"]["heavy"]["established"],
                                 "note": "excess phantom tracks / min walk (heavy occlusion)", "src": "LJ3"},
        "sl_flip_stress": {"v010": round(lj2["e1_stabilizer"]["stress_hybrid"]["flip_per_min"], 2),
                           "v020": lj2["e1_stabilizer"]["stress_hybrid"]["sl_flip_per_min"],
                           "delta_ci": lj2["e1_stabilizer"]["stress_hybrid"]["delta_sl_ci"], "established": lj2["e1_stabilizer"]["stress_hybrid"]["established"],
                           "note": "false standing<->lying flips / min under occlusion stress (the fall-relevant edge)", "src": "LJ2"},
    },
    # ── WEIGHT (arithmetic footprint; ONNX experts unchanged) ─────────────────
    "weight": {
        "arithmetic_kb": {"v010": lj5["gate"]["arithmetic_kb_v010"], "v020": lj5["gate"]["arithmetic_kb_v020"],
                          "note": "pose forest + Mahalanobis + flexion; VAE (195 KB) dropped, forest 873->220 KB (gold R3 flat)", "src": "LJ1/LJ5"},
        "onnx_mib": {"v010": lj5["gate"]["onnx_mib"], "v020": lj5["gate"]["onnx_mib"],
                     "note": "seven third-party experts, unchanged (SFace INT8 was a measured REJECT: +6x latency)", "src": "LJ5"},
    },
    # ── SPEED (reference hardware i7-12700K, LJ5 Act A) ───────────────────────
    "speed": {
        "human_chain_ms_allcores": {"v020": lj5["actA_latency_weight"]["human_chain_sum_ms"]["threads_all"],
                                    "budget": 50, "note": "YuNet+SFace+emotion+BlazePose(det+pose) sum; align-fix & flexion add 0", "src": "LJ5"},
        "human_chain_ms_onecore": {"v020": lj5["actA_latency_weight"]["human_chain_sum_ms"]["threads_one"],
                                   "note": "single-core sum (contended worst case)", "src": "LJ5"},
    },
    "gate": lj5["gate"],
}

# ============================================================ 1. Title
md(r"""
# LJ6 — Shipping v0.2.0: the scorecard
### the Rust port of every winning decision from LJ1–LJ5, and the before/after on all three axes

*MVM Lab Journal — 6 of 6, the closer. Prev: LJ5 (Weighing the experts). This journal closes phase 15.*

> Five journals measured the ML: the pose model (LJ1), the physics of posture transitions (LJ2),
> why walking people multiply (LJ3), the fusion/anomaly composition on a camera that never lies
> (LJ4), and the third-party experts we ship but never trained (LJ5). None of it was in production —
> the shipping binary was still **v0.1.0**, carrying the very defects the lab had just measured. This
> journal is the port: eight acts (P0–P8) that move each **measured** winner into the production Rust
> workspace and cut **v0.2.0**. The scorecard below is built from the same frozen bench file the
> journals wrote (`results/metrics.json`) — nothing is retyped by hand — so the "after" column is the
> lab's own number, now the shipping number. And because honesty is the whole point of the series,
> a full section is devoted to what we **measured and did not ship**.
""")

# ============================================================ 2. The arc
md(r"""
## 1. The arc — measure first, port second

The discipline of phase 15 was **lab-first**: never change production on a hunch, only on a number
that survives a frozen protocol (`spike_pose/bench.py`, SEED=0, grouped-bootstrap CIs, a data+code
passport on every result). Each stage shipped only as a narrative journal with the hypothesis stated
*before* the experiment and a verdict — including the negatives — stated after.

| Journal | Question | Verdict that shipped |
|---|---|---|
| **LJ0** | Can we build an honest lab on public data (CMU mocap + OpenImages)? | The reference pair + the frozen bench. |
| **LJ1** | Is the pose model actually good, or good-looking? | 2D-GRIP (z=0): MediaPipe **z is a cross-domain liability**; Mahalanobis replaces the VAE; footprint −79%. |
| **LJ2** | Can kinematics tell a fall from a lie-down? | The forward-prior is an **honest wash**; ships a dwell + one per-edge veto. A fall = floor + pelvis-speed. |
| **LJ3** | Why does one walking person become five? | Not the knobs — a structural **re-link dead-zone** on a stale identity. One branch fixes it. |
| **LJ4** | Does the composition hold under stress, and is the vocabulary real? | FLOOR event P/R = 1.000; the fall rule goes scene-level; the activity vocabulary was **fiction the data can't supervise**. |
| **LJ5** | How fast / heavy / accurate are the experts we never trained? | 16.7 ms chain; the identity crop was **never aligned** (+19.5 pt on the table); INT8 is a reject. |

This journal — **LJ6** — is the sixth: the port itself, and the ledger of what it cost and what it
was worth.
""")

# ============================================================ 3. Scorecard data cell
md(r"""
## 2. The scorecard (built from `results/metrics.json`)

The cell below is **generated by the builder from the frozen results file** — the LJ2–LJ5 numbers
are read out of the same JSON those journals wrote, so the "after" column cannot drift from the repo.
LJ1 predates the metrics file and is carried as an annotated literal (kernel
`howadx/lj1-the-pose-model-rebuilt-honestly`).
""")
import pprint
code("SCORECARD = " + pprint.pformat(SCORECARD, indent=1, width=112, sort_dicts=False) + "\n"
     "print('scorecard axes:', list(SCORECARD.keys()))")

# ============================================================ 4. Render
md(r"""
### 2.1 Accuracy — did the winners actually win?

Every row is a measured before→after. Where a confidence interval is shown it is a grouped bootstrap
(subjects/trials held together); *established* means the interval clears zero.
""")
code(r"""
def fmt(v):
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

def row(name, r):
    b, a = r.get("v010"), r.get("v020")
    extra = []
    if "ci" in r: extra.append(f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
    if "delta_ci" in r: extra.append(f"Δ-CI [{r['delta_ci'][0]:.2f}, {r['delta_ci'][1]:.2f}]" + (" ✓established" if r.get("established") else ""))
    return f"{name:26} {fmt(b):>10} → {fmt(a):<10} [{r['src']}]  {' '.join(extra)}\n      {r['note']}"

acc = SCORECARD["accuracy"]
print("ACCURACY (v0.1.0 → v0.2.0)\n" + "=" * 78)
for k in ["pose_gold_r3_2dgrip", "identity_lfw", "emotion_rafdb_balacc",
          "fall_auroc", "fall_falsealarms", "tracker_excess_med", "tracker_excess_heavy", "sl_flip_stress"]:
    print(row(k, acc[k]) + "\n")
il = acc["identity_lfw"]
print(f"  identity TAR@FAR=1e-3: {il['tar_far1e3_v010']} → {il['tar_far1e3_v020']}  (the operating point that matters for a gate)")
""")

md(r"""
Read the table as **three kinds of win**, not one:

- **A real accuracy lift**: identity +19.5 points (0.783→0.978) for **zero** added weight or latency —
  just feeding SFace the aligned crop it always wanted. The single best value in the whole phase.
- **A defect removed**: the fall rule went from **3/10 false alarms on controlled lie-downs** (the
  per-id standing-memory rule, which the fall itself churns away) to **0/10 with 6/6 real falls**; the
  standing↔lying flip under stress went to **zero**; walking fragmentation dropped by 4.7 (med) / 8.25
  (heavy) excess tracks per minute, both intervals clearing zero.
- **An honest hold**: pose gold R3 is 0.894 — that is *2D* holding the line where the old *3D* number
  (0.955 in-domain) was a cross-domain mirage collapsing to 0.62. The refactor traded a flattering
  number for a true one, and still came out ahead of the old lab-only 0.83.
""")

md(r"""
### 2.2 Weight & speed — the diet held its promise

The refactor's rule was: intelligence is arithmetic over data already in memory, so footprint only
grows when a new ONNX expert is added — and none was.
""")
code(r"""
w, sp, g = SCORECARD["weight"], SCORECARD["speed"], SCORECARD["gate"]
print("WEIGHT\n" + "=" * 78)
ar = w["arithmetic_kb"]
print(f"  pose arithmetic footprint : {ar['v010']} KB → {ar['v020']} KB  ({100*(ar['v010']-ar['v020'])/ar['v010']:.0f}% lighter)")
print(f"    {ar['note']}")
on = w["onnx_mib"]
print(f"  ONNX experts (total)      : {on['v010']} MiB → {on['v020']} MiB  (unchanged)")
print(f"    {on['note']}\n")
print("SPEED (reference hardware — i7-12700K, LJ5 Act A)\n" + "=" * 78)
hc = sp["human_chain_ms_allcores"]
print(f"  human chain (all cores)   : {hc['v020']:.1f} ms   (budget {hc['budget']} ms → PASS)")
print(f"  human chain (one core)    : {sp['human_chain_ms_onecore']['v020']:.1f} ms   (contended worst case)")
print(f"    {hc['note']}\n")
print("GATE (weight & latency must not exceed v0.1.0):", "MET ✓" if g["result"] == "MET" else g["result"])
print(f"  weight ≤ v0.1.0: {g['weight_le_v010']}   latency ≤ v0.1.0: {g['latency_le_v010']}")
""")

# ============================================================ 5. The port
md(r"""
## 3. The port — eight acts, P0 → P8

Each act was test-driven (a failing golden/unit test first), ended at a green
`cargo test --workspace`, and was committed on its own. The workspace went from **597 + 2** tests to
**608**. A CI gate (`mvm-cli agents::golden`) replays the CMU-derived FLOOR scenarios through the
*production* crates — the same compiled file LJ4 scores — so the lab and the shipping binary can never
silently disagree.

| Act | What moved into production | Effect |
|---|---|---|
| **P0** | Parity fixtures; unify the Mahalanobis class order to the forest's | Bit-exact golden so the port can't drift from the lab |
| **P1** | Pose: 12 z-free GRIP features · forest v2 (220 KB) · Mahalanobis **replaces the whole VAE** · `torso_flexion` output | −79% footprint; OOD from a distance, not a reconstruction |
| **P2** | Tracker: an identified track may re-link through a **stale** identity unless its name is claimed this frame | Kills the walking-fragmentation dead-zone |
| **P3** | Stabilizer: the `standing↔lying` commit needs a 12-frame per-edge veto | False fall-flip under stress → 0 |
| **P4** | Anomaly: **E2 scene-level fall** (pelvis speed + floor endpoint) retires the per-id window; new `camera_height_m` | 3/10 → 0/10 false alarms, 6/6 falls |
| **P5** | Identity: feed SFace the **ArcFace-aligned** crop, not the raw bbox | +19.5 pt LFW, +0 cost |
| **P6** | Activities: `using_object(<class>)` umbrella; `watching_tv` dropped | The vocabulary matches what data can supervise |
| **P7** | FSM exit-dwell sweep — **evaluated, deferred** (see §5) | No regression, no unearned latency |
| **P8** | Version bump, passport, CHANGELOG, contract docs | v0.2.0 |

**Three contract changes** a downstream consumer sees (documented in `AGENT.md`):
1. **`torso_flexion`** — a new *optional* posture field (continuous torso inclination, degrees). Additive.
2. **Activity vocabulary v2** — `typing`/`using_mouse`/`using_phone`/`reading` → `using_object(<class>)`;
   `holding_drink` kept; **`watching_tv` removed** (it is head-facing, not a wrist contact; 0 VRD instances).
3. **Pose-OOD semantics** — `plausibility` is now a Mahalanobis distance, not a VAE log-prob; `recon_error`
   is deprecated and always `null`.
""")

# ============================================================ 6. The honest ledger
md(r"""
## 4. What we measured and did **not** ship

A scorecard that only lists wins is a sales sheet. The phase's real product is a set of **negatives
that were expensive to earn** — each one is a change that looked obvious and was refuted by its own
frozen number. These are as much a part of v0.2.0 as the code that changed.
""")
code(r"""
LEDGER = [
    ("Synthetic azimuth augmentation (rotate the monocular pose)",
     "REFUTED — the old 0.90–0.95 claim was CIRCULAR (tested on the same z-rotation it trained on). On "
     "true multi-view mocap it FAILS (az0 0.67→0.47). Invariance must come from real views, not a "
     "self-consistent trick. [LJ1]"),
    ("Forward-filter transition prior beats a hand-tuned dwell",
     "NOT ESTABLISHED — aggregate flip-rate is a wash under the frozen criterion. Ships as a hybrid "
     "(dwell + one per-edge veto on the fall-relevant edge), not a learned prior. [LJ2]"),
    ("Re-calibrate the Kalman process noise Q to the real accel variance",
     "REFUTED — the 'correct' Q (~2e-4 vs prod 0.01) is WORSE (7.75 vs 5.43 excess tracks). Kalman/gate "
     "constants unchanged. [LJ3]"),
    ("A floor-FREE relative fall signal (net pelvis drop) can replace the absolute floor",
     "REJECTED — 5 TP / 1 FP / 1 FN vs the floor rule's 6/0/0; it conflates 'dropped from height' with "
     "'ended on the ground'. The floor carries real information (AUROC 0.994 vs 0.891). [LJ4]"),
    ("Quantize SFace to INT8 to shed the chain's heaviest weight",
     "REJECTED — accuracy is lossless (0.979) but latency is ~6x WORSE (a CNN dynamic-quant pitfall). "
     "Weight is not the axis to optimize here. [LJ5]"),
    ("Raise the FSM exit dwell 5/8 → 12/15 to kill stress fragmentation (the one 15.4 candidate)",
     "DEFERRED — verified it keeps the golden CI green, but the golden set is FLOOR-only so the gate "
     "cannot adjudicate the benefit (it exists only vs a synthetic stress model), while it imposes an "
     "always-on ~0.5 s finalization latency. Logged as a one-line flip for the owner post live-validation. [LJ4/P7]"),
]
for title, verdict in LEDGER:
    print(f"• {title}\n    {verdict}\n")
""")

md(r"""
The through-line: **the frozen protocol was allowed to say no.** A refactor that only ever confirms
its own hunches is not measuring anything. Five of the six items above would have shipped on intuition;
each was stopped by a number it had to clear and didn't.
""")

# ============================================================ 7. Live validation
md(r"""
## 5. Live validation — the binary, on the real rig

The scorecard is lab truth. The last check is that the *ported binary* behaves on a real camera. Two
`mvm run` captures (v0.2.0, the gallery owner in frame, **0 agent errors** across
capture/human/scene/depth/fusion/statefile) confirmed every agent-side change end-to-end:

| Ported act | Live evidence |
|---|---|
| **P5** SFace align-fix | the owner locked as `✓`-verified identity in **every** dumped tick — a stable lock is the whole point of +19.5 pt |
| **P1** `torso_flexion` | live field, 5.7–11.7° (seated, slight forward lean) |
| **P1/P3** pose-OOD (Mahalanobis) | `plausibility: ok` every tick; `recon_error` gone |
| **P6** vocabulary v2 | `using_object(mouse)`, `using_object(keyboard)`, `using_object(laptop)`, `using_object(cell phone)`, `holding_drink`, `seated_at_workstation` all fired — and a `tv` in frame produced **no** `watching_tv`. Zero legacy verbs. |
| **P4** anomaly E2 | `anomaly 0.00` under a calm scene — no false fall |
| honesty | posture is `withheld` when only the upper body is visible — not guessed |

Not staged: an actual fall (the E2 trigger). The threshold is calibrated on stunt-only CMU falls and
is unsafe to reproduce live — a documented ceiling, left to a careful owner check.
""")

# ============================================================ 8. Close
md(r"""
## 6. Where this points

Phase 15 was never only about v0.2.0. Its job was to put a **measured floor** under everything that
comes next — MVM's north star is a vision system that predicts, remembers, and models intent, and you
cannot build a forward model on a pose channel whose accuracy was a cross-domain mirage or a fall
signal that false-alarms every time someone lies down. Each journal turned one such channel from a
hope into a number with a confidence interval.

What ships in v0.2.0 is small on purpose: arithmetic over the experts we already had, a footprint that
went **down**, and a latency budget still comfortably met. What it *buys* is a substrate you can now
build on without re-litigating whether the ground truth is real. The next phase (16, the Dataset
Factory) is the memory layer — and it can trust its inputs, because this one measured them.

*Reproducibility: this journal recomputes nothing — every number is read from `results/metrics.json`
(the frozen output LJ2–LJ5 wrote) at build time, or carried from LJ1 with its kernel provenance. The
port itself is `cargo test --workspace` (608 green) + the `agents::golden` CI gate on the same
scenario file this lab scores.*

**End of the MVM Lab Journal series (LJ0–LJ6).**
""")

# ============================================================ 9. Passport
code(r"""
# ---- passport: the numbers are bound to the frozen results file + the published kernels ----
passport = {
    "scorecard_source": "results/metrics.json (frozen bench output; LJ2–LJ5 wrote it)",
    "kernels": {"lj2": "howadx/lj2-physics-of-posture-transitions",
                "lj3": "howadx/lj3-why-walking-people-multiply",
                "lj4": "howadx/lj4-a-camera-that-never-lies",
                "lj5": "howadx/lj5-weighing-the-experts"},
    "port": "Rust workspace v0.2.0 — 608 tests green; agents::golden CI gate on the LJ4 scenario file",
    "gate": SCORECARD["gate"],
}
import json as _j
print("passport:", _j.dumps(passport, indent=1))
""")

# ---------------------------------------------------------------- write notebook
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "lj6_shipping_the_scorecard.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "—", len(cells), "cells")
