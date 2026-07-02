//! Spike 7.5.0 — Rust `ort` load + inference smoke test for the BlazePose ONNX pair.
//! Proves both OpenCV Zoo models load in the SAME ort runtime as YuNet/SFace and run
//! a forward pass with the expected output shapes.
//!
//! Run: cargo run -p mvm-detect --example spike_pose_ort --release

use ort::{session::Session, value::Tensor};
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    // canonical Rust model dir (same as mvm-cli include_bytes! and test_config)
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"))
}

fn load(name: &str) -> anyhow::Result<Session> {
    let p = models_dir().join(name);
    let s = Session::builder()?.commit_from_file(&p)?;
    println!("LOADED {name}  ({} inputs, {} outputs)", s.inputs().len(), s.outputs().len());
    Ok(s)
}

fn main() -> anyhow::Result<()> {
    // 1) Person detector: input [1,3,224,224] NCHW
    let mut det = load("person_detection_mediapipe_2023mar.onnx")?;
    let din = Tensor::from_array(([1usize, 3, 224, 224], vec![0.0f32; 1 * 3 * 224 * 224]))?;
    let dout = det.run(ort::inputs!["input_1:0" => din])?;
    println!("person_det forward OK, {} outputs", dout.len());

    // 2) Pose estimation: input [1,256,256,3] NHWC
    let mut pose = load("pose_estimation_mediapipe_2023mar.onnx")?;
    let pin = Tensor::from_array(([1usize, 256, 256, 3], vec![0.0f32; 1 * 256 * 256 * 3]))?;
    let pout = pose.run(ort::inputs!["input_1" => pin])?;
    // landmarks output should be [1,195] = 39*5
    let lm = pout["Identity"].try_extract_tensor::<f32>()?;
    println!("pose forward OK, {} outputs; Identity len={}", pout.len(), lm.1.len());
    assert_eq!(lm.1.len(), 195, "expected 39*5=195 landmark values");

    println!("\nGATE: both BlazePose ONNX load + run in Rust ort 2.0.0-rc.12 ✓");
    Ok(())
}
