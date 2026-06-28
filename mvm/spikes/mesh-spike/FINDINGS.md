# Mesh-spike findings (phase 8.0.1 Agent Mesh prep)

Throwaway spike, detached from the 8-crate workspace. Two questions about the
proposed Agent Mesh, answered by measurement on this host.

Hardware: 20 logical cores (dev box). Also run under `taskset -c 0-3` (4 cores)
and `-c 0-1` (2 cores) to proxy a core-constrained deployment target.

## Spike A — ort thread contention (`src/main.rs`)

Loads the 3 real CPU experts (YuNet, EmotiEffLib, SFace), runs pure inference
(data-independent → zero-input tensors of correct shape), N=40/model.

| Config | 20 cores | 4 cores | 2 cores |
|--------|---------:|--------:|--------:|
| Sequential (1 thread, default intra) — current sync pipeline | 0.50s / **80fps** | 0.52s / 78fps | 0.50s / **80fps** |
| Parallel 3 threads, default intra | 1.39× | 1.32× | 1.31× |
| Parallel 3 threads, intra_op=1 | 0.78× | 0.63× | **0.43×** |
| Parallel 3 threads, intra_op=2 | 1.26× | 0.78× | 0.41× |

Per-inference: YuNet ~2.3ms, Emotion ~5.4ms, SFace ~4.7ms → **~12ms/frame total**.

### Findings (measured, not hypothesised)
1. **Current 3 experts cost ~12ms total (~80fps), nearly core-count-INDEPENDENT.**
   Even on 2 cores, sequential holds ~80fps. ONNX inference is NOT the bottleneck
   of the current pipeline — the real ~25fps is limited by capture / preprocess /
   tracking / Python bridge, not by the experts.
2. **Thread parallelism gives only ~1.3×** for the current light models, on any
   core count. The mesh does not meaningfully speed up the *current* workload.
3. **"ort contention" hypothesis REFUTED** for these models: they are too light
   to contend. Contention was the make-or-break risk; it is not a problem here.
4. **Pinning `intra_op` HURTS** (down to 0.41× on 2 cores) — the opposite of the
   pre-spike guess. Default ort threads are best on this workload. Do NOT pin.

### Consequence for the mesh
The mesh's value is NOT raw-throughput parallelism (1.3× is not worth tokio).
Its only real future value is **removing head-of-line blocking** once HEAVY
detectors land (YOLO26n ~39ms, Depth-Anything-V2-Small ~100ms): a 100ms depth
pass on the critical path would drop the fast human chain to ~9fps; running it
on its own thread at low cadence keeps the human chain smooth. That is
achievable with **plain OS threads**, not an async runtime.

## Spike B — latest-frame channel (`src/bin/latest_frame.rs`)

Producer 30fps, slow consumer ~5fps, single-slot latest-value cell (std
`Mutex<Option<Arc<Frame>>>`; ArcSwap is the lock-free prod version).

Result: consumer processed ids `[0,6,12,18,24,30,36,42,48,54,59]` — avg gap 5.9
(= 30/5 rate ratio), reached the producer's final frame (59), **no backlog**.

**PASS.** Single-slot latest-value is the right primitive. `tokio::broadcast` is
NOT wanted: on lag it catches up from the OLDEST buffered frame (wrong semantic),
whereas we want "always the freshest, drop intermediate".

## Decisions this spike supports

- **Plain `std::thread` + channels, NOT tokio.** Contention isn't the issue;
  async is the wrong fit for CPU-bound inference; std single-slot covers the
  frame plumbing. Respects the project's minimalism invariant (no async runtime).
- **Cadence-scheduling is the primary lever**, not parallelism. Doable in the
  existing synchronous loop today (`if frame_id % N == 0 { … }`).
- **Do not pin `intra_op`** on multi-core dev/desktop. Revisit only if a heavy,
  internally-multi-threaded model is added on a genuinely core-starved target.
- The async Agent Mesh (full tokio design) is **deferred** until ≥1 heavy
  detector exists to justify a dedicated offload thread.

Reproduce: `cargo run --release` (Spike A) · `cargo run --release --bin latest_frame` (Spike B).
Constrained: `taskset -c 0-3 ./target/release/mesh-spike`.
