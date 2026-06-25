# Phase 7 — FPS baseline (G5)

> Numeric reper measured **2026-06-25** before any Phase 7 native work, so the
> FPS-win of native camera (7.2) and de-duplicated inference is measurable, not
> prose. Re-measure after each native stage and compare against this file.

## Method

- Machine: this host, `/dev/video0` (1920×1080), live PipeWire session.
- Camera grab path: `cv2.CAP_FFMPEG` (V4L2 path fails here: `VIDIOC_QBUF: Bad file descriptor`).
- Binary: `target/release/mvm-cli` (release build, commit on `rust-core`).
- fps is content-independent (whatever is in frame doesn't change throughput),
  so these numbers are valid without an operator performing scenarios.

## Numbers

| Path | fps | s/frame | What it includes |
|------|-----|---------|------------------|
| **End-to-end native** (`mvm-cli --native`) | **1.1** | 0.91 | bridge + Rust enrichment (EmotiEffLib + SFace re-inference) + disk round-trip |
| Bridge-only (`mvm_bridge.py --json-only`) | 1.57 | 0.64 | SCRFD + ArcFace + HSEmotion + MediaPipe pose + Workspace3D (Python) |

Raw runs:
- End-to-end: 37 frames in ~35 s, settled at **1.1 fps** (`logs/baseline_native.log`).
- Bridge-only: 25 frames, steady span 14.7 s → **1.57 fps**.

## Decomposition / where the time goes

```
0.91 s/frame end-to-end
 ├─ 0.64 s  Python bridge inference (the bottleneck)
 └─ 0.27 s  Rust enrichment + disk round-trip (~40% overhead from
            re-inferring emotion/identity on a face the bridge already saw,
            plus /tmp/mvm_frame.jpg write+read)
```

## Implications for Phase 7

- **Confirms the "~1–2 fps" prose with a hard number: 1.1 fps end-to-end.**
- The dominant cost is the Python bridge itself (0.64 s/frame) → **7.2 native
  camera + native detection** is the real FPS lever.
- The Rust re-inference adds ~40% on top → eliminating the **double inference**
  (bridge does emotion/identity, then Rust does it again) and the disk
  round-trip is a second, independent win once detection is native (7.1 → 7.4-light).

## Win target (Phase 7 exit gate)

Any native stage must move end-to-end fps **up from 1.1**. The `<50 ms`
(20 fps) invariant is the north star; intermediate gates just require monotonic
improvement vs this baseline.
