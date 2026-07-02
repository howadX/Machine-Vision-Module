# Phase 7.2 camera spike — FINDINGS (G3)

> Decisive risk-removal experiment, **2026-06-25**. Question: can pure-Rust read
> `/dev/video0` under live PipeWire on this host, given cv2 needed `CAP_FFMPEG`
> (raw V4L2 mmap threw `VIDIOC_QBUF: Bad file descriptor` in OpenCV)?

## Verdict: 🟢 GREEN — 7.2 is feasible

Pure-Rust `v4l` crate (v0.14) reads the camera under a live PipeWire session.
No system dev libs, no clang/bindgen, no PipeWire portal needed.

## Evidence

| Format | Resolution | Raw capture fps | Frames |
|--------|-----------|-----------------|--------|
| YUYV (default) | 1920×1080 | **3.4** | 30/30 |
| **MJPG** | **1280×720** | **25.3** | 40/40 |

- mmap streaming (QBUF/DQBUF) **works** here — the OpenCV `VIDIOC_QBUF` error
  was an OpenCV-internal probe artifact, not a hard device/PipeWire block.
- Device is STREAMING-only (`caps=0x84a00001`, no READWRITE) → mmap is the only
  native method, and it works.

## Why this matters

- Current end-to-end native pipeline = **1.1 fps** (baseline_phase7.md).
- The hidden ceiling was the **camera format**: default YUYV-1080 caps raw
  capture at 3.4 fps. Negotiating **MJPG 1280×720 lifts the raw ceiling to
  ~25 fps** — finally in reach of the `<50 ms` / 30 fps north star.
- Even YUYV-raw (3.4) already beats bridge-only steady (1.57), and native
  removes the `/tmp/mvm_frame.jpg` disk round-trip + the double inference.

## Design inputs for Stage 3 (7.2 implementation)

1. **Crate:** `v4l = "0.14"` — pure Rust, builds with zero system/clang deps.
2. **Negotiate MJPG 1280×720** (not default YUYV-1080) for the fps win.
   `Capture::set_format(&dev, Format::new(1280, 720, FourCC::new(b"MJPG")))`.
3. **Decode:** MJPG frame → JPEG bytes (`meta.bytesused`, not `buf.len()` which
   is the mmap buffer capacity). Decode via the `image` crate (already a project
   dep) or `zune-jpeg` for speed. YUYV path would need manual YUV→RGB instead.
4. **Port ROTATE_180** (camera mounted upside-down) into Rust after decode.
5. `MmapStream::with_buffers(&dev, Type::VideoCapture, 4)` — 4 buffers worked.

## Reproduce

```bash
cd mvm/spikes/cam-spike
cargo run --release -- /dev/video0 40 MJPG 1280x720   # 25 fps
cargo run --release -- /dev/video0 30                  # YUYV default, 3.4 fps
```

Spike is detached from the main workspace (empty `[workspace]` in its
Cargo.toml) — it does not touch the 8-crate Cargo.lock. Promote the capture
logic into `mvm-pipeline` in Stage 3.
