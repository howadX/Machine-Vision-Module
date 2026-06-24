# Phase 5.1 — Stabilize Rust-native ONNX pipeline for MVM

> **Target:** Claude Code (copilot, codex, or direct). This is a self-contained prompt.
> **Repo:** `~/repos/mvm/` (Rust workspace, 8 crates, 47 tests).
> **Goal:** Fix the 4 critical issues preventing `mvm-cli --native` from working end-to-end.

---

## What Is MVM

Machine Vision Module — a Structured Decomposed Picture-to-Text system. Instead of one VLM (GPT-4V, LLaVA), MVM is an ensemble of independent micro-agents, each trained for one task and emitting structured signals with confidence.

**Architecture:** crate-per-agent, `Module` trait, CPU-only, ONNX Runtime.

**Current hybrid pipeline (`mvm-cli --native --once`):**

```
Python bridge (mvm_bridge.py, subprocess)
  ├─ SCRFD (face detection, ONNX/CPU)
  ├─ ArcFace (identity, 512d embedding, gallery match)
  ├─ HSEmotion (emotion, EfficientNet-B0, PyTorch/GPU, 8-class AffectNet)
  ├─ PoseLandmarker → PoseVAE (posture, 6 classes)
  └─ Workspace3D → (x,y,z)m
       │
       ▼ JSON lines on stdout
Rust mvm-cli
  ├─ parse_line → Vec<Detection>
  ├─ EmotiEffLib B0 (emotion, ONNX/CPU, 8-class VGAF) ← OVERWRITES Python HSEmotion!
  ├─ SFace (identity, ONNX/CPU, 128d) ← GALLERY IS EMPTY!
  ├─ PersonTracker (temporal fusion)
  ├─ StateFile → current_state.md
  └─ dump_table() → terminal
```

## Repository Layout

```
~/repos/mvm/                          # Rust workspace (THIS IS THE TARGET)
├── Cargo.toml                        # [workspace] 8 members, edition 2024
├── AGENTS.md                         # Full docs, roadmap, known issues
├── models/                           # ONNX files
│   ├── face_detection_yunet_2023mar.onnx   (228 KB)
│   ├── face_recognition_sface_2021dec.onnx (36.9 MB)
│   └── enet_b0_8_best_vgaf.onnx            (15.3 MB)
├── crates/
│   ├── mvm-core/         ✅ types, BBox, Detection, PersonTracker, Config, Module trait
│   ├── mvm-detect/       ✅ YuNet ONNX inference + NMS (score_threshold=0.5)
│   ├── mvm-identity/     ⚠️ SFace ONNX + gallery (gallery is EMPTY → blocked)
│   ├── mvm-emotion/      ⚠️ EmotiEffLib B0 ONNX (unconditional overwrite → blocked)
│   ├── mvm-posture/      🔶 stub (empty Module trait)
│   ├── mvm-fusion/       🔶 stub (empty Module trait)
│   ├── mvm-pipeline/     ✅ Camera (Python bridge subprocess), parse_line
│   └── mvm-cli/          ✅ main, StateFile, dump_table(), --native/--once
├── xtask/                ✅ cargo xtask bundle → dist/mvm
└── dist/                 bundled binary output
```

All ONNX models use the `ort` crate (v2.0.0-rc.12, features: std, download-binaries, tls-rustls, api-22).

## Commands

```bash
# Build & test
cargo build --release          # → target/release/mvm-cli (~25 MB)
cargo test --workspace         # MUST stay green (47 tests)

# Run (requires Python MVM venv at ~/repos/Machine Vision Module/venv)
target/release/mvm-cli --native --once --dump-interval 1

# Bundle
cargo xtask bundle             # → dist/mvm (models embedded)
```

## Task 1: Fix YuNet score_threshold for gallery face detection

**Problem:** `mvm-identity` builds its gallery by running YuNet on reference photos (`models/references/p_V*.jpg`). YuNet's `score_threshold` is 0.5 — too high for the small/low-quality reference images, so NO faces are detected. Result: gallery is empty → `identify()` always returns None.

**Files:**
- `crates/mvm-detect/src/lib.rs` — find the `Detector::detect()` or `new()` function that accepts `score_threshold`
- `crates/mvm-identity/src/lib.rs` — `build_gallery()` calls the detector

**What to do:**

1. Lower YuNet `score_threshold` from `0.5` to `0.3` (or make it configurable in `mvm-detect::Detector`)
2. Verify gallery builds: after `build_gallery()`, `self.gallery.len()` should equal number of unique people (4: Вадим, Даня, Никита, Белла)
3. Add a test: create a test image or use an existing reference photo path, verify `detect()` returns at least 1 detection at the new threshold

**Testing:**
```bash
cargo test -p mvm-identity -- --nocapture
# Watch for: gallery size > 0, identify() returns Some(name)
```

## Task 2: Confidence-based emotion fallback

**Problem:** In `mvm-pipeline` (or `mvm-cli` main loop), Rust unconditionally overwrites `det.emotion` from Python bridge with its own EmotiEffLib prediction. Even if Rust's confidence is lower. The correct behavior: keep the prediction with higher confidence.

**Why it matters:** EmotiEffLib B0 (Rust, VGAF dataset) and HSEmotion (Python, AffectNet dataset) are the SAME architecture (EfficientNet-B0). They just differ in training data and runtime. Confidence comparison is the right strategy — not distance-based, not unconditional replacement.

**Files:**
- `crates/mvm-pipeline/src/lib.rs` — look for the loop where Rust enriches `Detection` objects after parsing JSON from the bridge
- `crates/mvm-cli/src/main.rs` — if the enrichment happens here instead

**What to do:**

```rust
// ❌ CURRENT: unconditional overwrite
if let Ok(rust_emotion) = emotion_module.process(&face_crop) {
    det.emotion = Some(rust_emotion.label);
    det.emotion_confidence = Some(rust_emotion.confidence);
}

// ✅ FIXED: confidence-based
if let Ok(rust_emotion) = emotion_module.process(&face_crop) {
    let current_conf = det.emotion_confidence.unwrap_or(0.0);
    if rust_emotion.confidence > current_conf {
        det.emotion = Some(rust_emotion.label);
        det.emotion_confidence = Some(rust_emotion.confidence);
    }
}
```

**Testing:**
```bash
cargo test --workspace -- --nocapture
# Integration test (manual): run --native --once and check emotion field isn't blindly overwritten
```

## Task 3: Fix gallery build on reference photos

**Problem:** This is the combination of Tasks 1 + gallery loading logic. Even with lowered threshold, check that:
- `build_gallery()` correctly loads multi-angle photos (e.g., `p_V_01.jpg`, `p_V_02.jpg` → both contribute to Вадим's gallery entry)
- The gallery stores EMA-averaged embeddings (α=0.3) for multiple photos of the same person
- `identify()` uses cosine similarity with threshold 0.35

**Files:**
- `crates/mvm-identity/src/lib.rs` — `build_gallery()`, `add_reference()`, `identify()`
- `crates/mvm-identity/src/lib.rs` — tests module

**What to do:**

1. Verify `build_gallery()` correctly reads files matching `p_*.jpg` from `models/references/`
2. Group by prefix: `p_V_01.jpg` → "Вадим", `p_N_01.jpg` → "Никита", etc.
3. For each photo: YuNet detect → crop face → SFace `feature()` → L2-normalize → `add_reference()`
4. `add_reference()` uses EMA: `gallery[name] = α * new_emb + (1-α) * old_emb`
5. `identify()` computes `cosine_similarity(emb, gallery_emb)` for each person, returns max > 0.35

**Name mapping:** Extract from filename: `p_V_01.jpg` → key = "V" → "Вадим". `p_N.jpg` → key = "N" → "Никита".

```rust
const NAME_MAP: &[(&str, &str)] = &[
    ("V", "Вадим"), ("D", "Даня"), ("N", "Никита"), ("B", "Белла"),
];
```

**Testing:**
```bash
cargo test -p mvm-identity -v
# Expected: gallery_build test passes, identify_known_face returns Some
```

## Task 4: Update dist/mvm (xtask bundle)

**Problem:** `dist/mvm` was built with an older version and doesn't include `dump_table()`. Rebuild the bundle with current code.

**Files:**
- `xtask/src/main.rs` — bundle logic (should already be correct)
- Check that embedded models work (YuNet, SFace, EmotiEffLib are included via `include_bytes!`)

**What to do:**

1. Run `cargo xtask bundle`
2. Verify: `./dist/mvm --help` shows `--native`, `--once`, `--dump-interval` flags
3. Verify: `./dist/mvm --version` or similar shows current build

```bash
cargo xtask bundle
ls -lh dist/mvm          # should be ~24-26 MB
./dist/mvm --help        # verify flags
```

## Important: TDD is mandatory

All Rust code changes MUST follow TDD:
1. Write a failing test first
2. Verify it fails
3. Implement the minimal fix
4. Verify the test passes
5. Run `cargo test --workspace` to verify no regressions
6. Commit

Test patterns used in this workspace:
- `#[cfg(test)] mod tests` in the same file as the implementation
- `Config { models_path: PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")), ..Default::default() }`
- `sort_by_key(|t| t.id)` for deterministic tests with HashMaps
- `std::env::temp_dir()` instead of tempfile crate

## Environment

- Rust edition: **2024** (no `ref` in let patterns, use `PathBuf::display()` for formatting)
- ONNX Runtime: `ort = "2.0.0-rc.12"` with features: std, download-binaries, tls-rustls, api-22
- ndarray: must match ort's version (0.17)
- TensorRef construction: `.from_array_view(&owned_array)` not `.from_array_view(array.view())`

## Verification Checklist

After ALL tasks complete:

```bash
# 1. All tests pass
cargo test --workspace
# Expected: 47+ tests, 0 failures

# 2. Gallery builds
cargo test -p mvm-identity -v
# Expected: gallery size assertion passes

# 3. Full pipeline runs (requires camera)
target/release/mvm-cli --native --once --dump-interval 1
# Expected: dumps table with identity, emotion, posture, 3D position

# 4. Bundle works
cargo xtask bundle
./dist/mvm --help
# Expected: all flags present
```

## Success Criteria

1. `mvm-identity` gallery is non-empty after `build_gallery()` — YuNet detects faces on reference photos
2. `mvm-cli --native` does NOT unconditionally overwrite Python bridge emotion — uses confidence comparison
3. `dist/mvm` includes all current features (dump_table, --native)
4. `cargo test --workspace` stays green (47+ tests, zero failures)
5. Full pipeline `--native --once --dump-interval 1` outputs identity ≠ "unknown" when known person is in frame
