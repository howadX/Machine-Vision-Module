//! Phase 19.2.2 one-model probe for ONNX compatibility and CPU cost.
//!
//! This is intentionally not a runtime semantic expert. It accepts an explicit
//! model path and reports measurements without adding the model to MVM's
//! manifest or publishing scene facts.

use anyhow::{Context, Result, ensure};
use ndarray::{Array, IxDyn};
use ort::{session::Session, value::TensorRef};
use std::{collections::BTreeSet, env, fs, path::PathBuf, time::Instant};

const INPUT_SIZE: usize = 256;
const WARMUP_RUNS: usize = 5;
const MEASURED_RUNS: usize = 30;

fn percentile(samples: &mut [f64], quantile: f64) -> f64 {
    samples.sort_by(f64::total_cmp);
    let index = ((samples.len() - 1) as f64 * quantile).round() as usize;
    samples[index]
}

fn max_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn main() -> Result<()> {
    let model_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: semantic_onnx_probe <model.onnx>")?;
    let model_bytes = fs::metadata(&model_path)
        .with_context(|| format!("cannot stat {}", model_path.display()))?
        .len();
    let mut session = Session::builder()?
        .with_intra_threads(1)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .commit_from_file(&model_path)
        .with_context(|| format!("current Rust ORT rejected {}", model_path.display()))?;

    let values: Vec<f32> = (0..3 * INPUT_SIZE * INPUT_SIZE)
        .map(|index| -2.0 + 4.0 * index as f32 / (3 * INPUT_SIZE * INPUT_SIZE - 1) as f32)
        .collect();
    let input = Array::from_shape_vec(IxDyn(&[1, 3, INPUT_SIZE, INPUT_SIZE]), values)?;

    for _ in 0..WARMUP_RUNS {
        session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
    }
    let mut samples = Vec::with_capacity(MEASURED_RUNS);
    let mut output_shape = Vec::new();
    let mut labels = BTreeSet::new();
    for run in 0..MEASURED_RUNS {
        let started = Instant::now();
        let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        if run == 0 {
            let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
            output_shape = shape.iter().copied().collect();
            ensure!(
                output_shape == [1, 150, INPUT_SIZE as i64, INPUT_SIZE as i64],
                "unexpected semantic output shape {output_shape:?}"
            );
            let pixels = INPUT_SIZE * INPUT_SIZE;
            for pixel in 0..pixels {
                let mut best_class = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for class in 0..150 {
                    let score = logits[class * pixels + pixel];
                    if score > best_score {
                        best_score = score;
                        best_class = class;
                    }
                }
                labels.insert(best_class);
            }
        }
    }

    let mut p50_samples = samples.clone();
    let p50 = percentile(&mut p50_samples, 0.50);
    let p95 = percentile(&mut samples, 0.95);
    println!("model_bytes={model_bytes}");
    println!("output_shape={output_shape:?}");
    println!("unique_argmax_labels={labels:?}");
    println!("release_single_thread_p50_ms={p50:.3}");
    println!("release_single_thread_p95_ms={p95:.3}");
    println!("process_max_rss_kib={}", max_rss_kib().unwrap_or(0));
    Ok(())
}
