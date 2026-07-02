//! Spike A — ort thread-contention measurement (phase 8.0.1 Agent Mesh prep).
//!
//! Loads the three real CPU ONNX experts (YuNet, EmotiEffLib, SFace) and runs
//! pure inference (data-independent → zero-input tensors of correct shape) in
//! four configs:
//!   1. SEQUENTIAL  — 1 thread, default ort intra-op threads (proxy for the
//!                    current synchronous `--native-camera` pipeline).
//!   2. PAR default — 3 OS-threads (one expert each), default ort intra-op.
//!   3. PAR intra=1 — 3 OS-threads, each session pinned to 1 intra-op thread.
//!   4. PAR intra=2 — 3 OS-threads, each session pinned to 2 intra-op threads.
//!
//! Reports wall-clock + speedup vs the sequential baseline. The make-or-break
//! question for Agent Mesh: does ANY parallel config beat sequential, and by
//! how much? If best speedup ≈ 1×, the mesh's parallelism premise is broken
//! (cadence-scheduling is the real lever instead). If >2×, the mesh pays off —
//! and config 3/4 shows how to size ort sessions to avoid core contention.

use std::path::PathBuf;
use std::sync::Barrier;
use std::time::Instant;

use anyhow::Result;
use ort::session::Session;
use ort::value::Tensor;

const N: usize = 40; // inference iterations per model per config
const WARMUP: usize = 3;

struct ModelSpec {
    label: &'static str,
    file: &'static str,
    input_name: &'static str,
    chw: [usize; 3], // C, H, W
}

const MODELS: [ModelSpec; 3] = [
    ModelSpec { label: "YuNet  ", file: "face_detection_yunet_2023mar.onnx", input_name: "input", chw: [3, 640, 640] },
    ModelSpec { label: "Emotion", file: "enet_b0_8_best_vgaf.onnx",          input_name: "input", chw: [3, 224, 224] },
    ModelSpec { label: "SFace  ", file: "face_recognition_sface_2021dec.onnx", input_name: "data",  chw: [3, 112, 112] },
];

fn models_dir() -> PathBuf {
    // spike lives at mvm/spikes/mesh-spike → models at mvm/models
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"))
}

fn build_session(file: &str, intra: Option<usize>) -> Result<Session> {
    let path = models_dir().join(file);
    // ort builder methods return `ort::Error<SessionBuilder>` on error, which is
    // NOT Send (carries the builder) → cannot use `?` into anyhow. Stringify.
    let mut b = Session::builder().map_err(|e| anyhow::anyhow!("builder: {e}"))?;
    if let Some(n) = intra {
        b = b
            .with_intra_threads(n)
            .map_err(|e| anyhow::anyhow!("with_intra_threads: {e}"))?;
    }
    b.commit_from_file(&path)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))
}

/// One inference with a fresh zero tensor of the right shape (mirrors the real
/// pipeline, which rebuilds the input tensor every frame).
fn infer_once(session: &mut Session, input_name: &str, chw: [usize; 3]) -> Result<()> {
    let shape = ndarray::IxDyn(&[1, chw[0], chw[1], chw[2]]);
    let arr = ndarray::Array::<f32, _>::zeros(shape);
    let tensor = Tensor::from_array(arr)?;
    let _ = session.run(ort::inputs![input_name => tensor])?;
    Ok(())
}

fn run_n(session: &mut Session, spec: &ModelSpec, n: usize) -> Result<()> {
    for _ in 0..n {
        infer_once(session, spec.input_name, spec.chw)?;
    }
    Ok(())
}

/// Sequential baseline: one thread runs all three experts N times back-to-back.
/// Returns total wall-clock seconds and prints per-model latency.
fn sequential(intra: Option<usize>) -> Result<f64> {
    let mut sessions: Vec<Session> = MODELS
        .iter()
        .map(|m| build_session(m.file, intra))
        .collect::<Result<_>>()?;
    // warmup
    for (s, m) in sessions.iter_mut().zip(MODELS.iter()) {
        run_n(s, m, WARMUP)?;
    }
    let start = Instant::now();
    for (s, m) in sessions.iter_mut().zip(MODELS.iter()) {
        let t = Instant::now();
        run_n(s, m, N)?;
        let per = t.elapsed().as_secs_f64() * 1000.0 / N as f64;
        println!("    {}  {:6.1} ms/inf  ({} inf)", m.label, per, N);
    }
    Ok(start.elapsed().as_secs_f64())
}

/// Parallel: each expert in its own OS thread, all started simultaneously
/// (Barrier). Returns wall-clock seconds = max thread finish.
fn parallel(intra: Option<usize>) -> Result<f64> {
    // Build sessions up front (load time excluded from measurement).
    let sessions: Vec<Session> = MODELS
        .iter()
        .map(|m| build_session(m.file, intra))
        .collect::<Result<_>>()?;

    let barrier = std::sync::Arc::new(Barrier::new(MODELS.len() + 1));
    let mut handles = Vec::new();

    for (mut session, spec) in sessions.into_iter().zip(MODELS.iter()) {
        let b = barrier.clone();
        let label = spec.label;
        let input_name = spec.input_name;
        let chw = spec.chw;
        handles.push(std::thread::spawn(move || -> Result<f64> {
            // warmup before the synchronized start
            for _ in 0..WARMUP {
                infer_once(&mut session, input_name, chw)?;
            }
            b.wait();
            let t = Instant::now();
            for _ in 0..N {
                infer_once(&mut session, input_name, chw)?;
            }
            let secs = t.elapsed().as_secs_f64();
            let _ = label;
            Ok(secs)
        }));
    }

    barrier.wait();
    let wall_start = Instant::now();
    let mut worst = 0.0f64;
    for h in handles {
        let s = h.join().expect("thread panicked")?;
        worst = worst.max(s);
    }
    let wall = wall_start.elapsed().as_secs_f64();
    // wall ≈ worst; report wall (true concurrent finish)
    Ok(wall.max(worst))
}

fn main() -> Result<()> {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    println!("=== Spike A: ort thread contention ===");
    println!("logical cores: {cores}  |  N={N} inferences/model  |  models: YuNet+Emotion+SFace\n");

    println!("[1] SEQUENTIAL (1 thread, default ort intra) — proxy for current sync pipeline:");
    let t_seq = sequential(None)?;
    println!("    → total wall: {:.3}s  ({:.1} 'frames'/s if all 3 = one frame)\n", t_seq, N as f64 / t_seq);

    println!("[2] PARALLEL 3 threads, DEFAULT ort intra (current code reality):");
    let t_par_def = parallel(None)?;
    println!("    → wall: {:.3}s   speedup vs seq: {:.2}x\n", t_par_def, t_seq / t_par_def);

    println!("[3] PARALLEL 3 threads, intra_op=1 each:");
    let t_par_1 = parallel(Some(1))?;
    println!("    → wall: {:.3}s   speedup vs seq: {:.2}x\n", t_par_1, t_seq / t_par_1);

    println!("[4] PARALLEL 3 threads, intra_op=2 each:");
    let t_par_2 = parallel(Some(2))?;
    println!("    → wall: {:.3}s   speedup vs seq: {:.2}x\n", t_par_2, t_seq / t_par_2);

    // Verdict
    let best = [t_par_def, t_par_1, t_par_2]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let best_speedup = t_seq / best;
    println!("=== VERDICT ===");
    println!("best parallel speedup vs sequential: {:.2}x", best_speedup);
    if best_speedup < 1.3 {
        println!("→ CONTENTION DOMINATES. Agent-thread parallelism gives ~nothing on this CPU.");
        println!("  Mesh premise weak → make CADENCE-SCHEDULING the primary lever, keep sync loop.");
    } else if best_speedup < 2.0 {
        println!("→ MODEST parallelism. Mesh helps but cadence-scheduling is the bigger win.");
    } else {
        println!("→ REAL parallelism. Agent Mesh pays off. Pin ort intra_op so N_agents x intra <= cores.");
    }
    let def_speedup = t_seq / t_par_def;
    if best < t_par_def * 0.85 {
        println!("  TUNING MATTERS: pinning intra_op beats default ({:.2}x vs {:.2}x).", best_speedup, def_speedup);
    } else {
        println!("  Default ort threads already near-best ({:.2}x); explicit pinning optional.", def_speedup);
    }
    Ok(())
}
