# Phase 7 — Operator Runbook (camera calibration + live tests)

> For the human at the camera. Closes the operator-gated gates of Phase 7
> (Python-light): **3D recalibration for YuNet boxes** + the **live validations**
> that unit tests can't cover (someone must actually be in frame).
>
> Run everything **from the project root** (`Machine Vision Module/`). The camera
> is ceiling-mounted, ~45° down, upside-down (the code already rotates 180°).

---

## 0. Prep (once)

```bash
cd mvm && cargo build --release && cd ..
```

Quick sanity that the Python-light path runs (no one needs to be in frame):

```bash
mvm/target/release/mvm-cli --native-camera --dump-interval 1
#   expect: "camera: native v4l2 MJPG 1280x720 (Python-light, no pose)"
#           ticks at ~25 fps, "0 tracks" until someone steps in.  Ctrl-C to stop.
```

---

## Part A — Recalibrate 3D for YuNet boxes  *(gate 7.1)*

**Why:** native-camera depth uses `face_height_m`. The current 0.44 was solved
for **SCRFD** boxes; YuNet boxes have a different size, so native 3D is biased
until we re-solve the constant. Output goes into `Config.face_height_m_yunet`
(the bridge path keeps its own 0.44 — they no longer interfere).

**Setup:** one person, face clearly in view, **hold still** for each run. Use a
tape measure from the camera to the face. Pick 4–5 distances spanning the room.

```bash
# One run per tape-measured distance (metres). Hold still ~5 s each.
venv/bin/python3 tools/calib_3d.py --native-camera --distance 0.5
venv/bin/python3 tools/calib_3d.py --native-camera --distance 1.0
venv/bin/python3 tools/calib_3d.py --native-camera --distance 1.5
venv/bin/python3 tools/calib_3d.py --native-camera --distance 2.0
venv/bin/python3 tools/calib_3d.py --native-camera --distance 3.0

# Summarise + get the recommended constant:
venv/bin/python3 tools/calib_3d.py --native-camera --report
```

Rows land in `models/calib_3d_yunet.csv` (separate from the historical SCRFD
`calib_3d.csv` — they are NOT mixed).

**Read the report:**
- `spread < 15 %` → **CONSISTENT**: it prints `set the constant = <X>`. Good.
- `spread ≥ 15 %` → drifts with distance (tilt non-linearity). Tell me — we
  decide together (e.g. piecewise, or accept a working-range constant).

**Apply the constant** (I can do this step if you paste the report):
1. Edit `mvm/crates/mvm-core/src/config.rs` →
   `face_height_m_yunet: <X>,` (replace the 0.44 PLACEHOLDER).
2. Rebuild: `cd mvm && cargo build --release && cd ..`
3. Re-verify one distance:
   ```bash
   venv/bin/python3 tools/calib_3d.py --native-camera --distance 1.0
   #   "native dist_m" should now read ≈ 1.0 m
   ```

---

## Part B — Live scenario tests  *(gates 7.5 + fps-with-faces)*

Harness drives the **Python-light path** and prints a structured verdict. It
shouts **GO** when the first frame lands — perform the action only after GO.
Logs land in `mvm/logs/cam_<scenario>_nc.jsonl` (replayable).

> Note: **no posture** in this path (dropped until 7.3) — `posture 0/N` in the
> verdict is expected, not a failure.

### B1 — Identity #5: stranger must NOT become "Вадим"
```bash
venv/bin/python3 tools/cam_test.py --native-camera --scenario unknown_face --secs 20
```
A **non-gallery** person in frame. **PASS** = identity `unknown` / low conf,
**never** "Вадим". (This is the margin-identity fix; pure logic already unit-tested.)

### B2 — Identity #6: id survives leave/return without merging
```bash
venv/bin/python3 tools/cam_test.py --native-camera --scenario leave_return --secs 25
```
In frame → leave ~3 s → return. **PASS** = identity re-locks to "Вадим" on
return; a different person returning is NOT labelled the same track (cosine re-link).

### B3 — fps with a face present  *(the real throughput number)*
```bash
venv/bin/python3 tools/cam_test.py --native-camera --scenario single_sit --secs 20
```
Sit still ~1.5 m. **Record the fps** from the verdict — this is the honest
Python-light number *with* per-face ONNX running (the 24.9 fps measurement had
no face). Compare to baseline 1.1 fps.

### B4 — 3D sanity after calibration
```bash
venv/bin/python3 tools/cam_test.py --native-camera --scenario walk_near_far --secs 25
```
Walk 0.5 m → 2.5 m → back. **PASS** = `dist_m` tracks your motion monotonically
and roughly matches the tape (within ±15–25 %).

### B5 — two people (scene-graph + no-merge)
```bash
venv/bin/python3 tools/cam_test.py --native-camera --scenario two_people --secs 25
```
Two people, then stand close. **PASS** = 2 tracks, a `near` relation / zone of
size 2 when close; two distinct ids never merge.

---

## Part C — What to send back

Paste me:
1. The `--report` output from Part A (the recommended constant + spread).
2. The verdict block from each Part B scenario (especially B1, B2, B3 fps).

I'll then: apply the constant, set `--native-camera` as default if 3D checks
out, record the live results in `CLAUDE.md`, and commit.

---

## Expected outcomes (cheat-sheet)

| Test | PASS looks like |
|------|-----------------|
| A calibration | spread < 15 %, one constant; native dist ≈ tape after apply |
| B1 unknown_face | identity = unknown, never "Вадим" |
| B2 leave_return | id re-locks; different person ≠ same id |
| B3 single_sit | stable track; **fps recorded** (the headline number) |
| B4 walk_near_far | dist_m monotonic, ≈ tape ±15–25 % |
| B5 two_people | 2 ids, near/zone when close, no merge |

## Troubleshooting
- *"binary not found"* → run step 0 (build).
- *"no face samples collected"* (calib) → check lighting/framing; the face must
  fill enough of the frame; hold still.
- *camera busy / no frames* → another app (or a stale mvm-cli) holds
  `/dev/video0`. `pkill -f mvm-cli` and retry.
- *fps lower than expected with a face* → expected; per-face ONNX (emotion +
  identity) runs per detection. Still far above the 1.1 fps bridge baseline.
