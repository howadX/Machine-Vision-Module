//! Live native pose driven by the YuNet FACE bbox (7.5.4 ROI source for desk/ceiling).
//! Full Rust path: NativeCamera → YuNet faces → NativePose::poses_from_faces → posture.
//! The face is always visible at a desk even when the body is cut off, so this is
//! robust where person_detection_mediapipe failed.
//!
//! Run: cargo run -p mvm-posture --example live_pose_face --release [seconds]

use mvm_core::{Config, Module};
use mvm_detect::YunetDetector;
use mvm_posture::NativePose;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let secs: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15.0);
    let config = Config {
        models_path: std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")),
        ..Default::default()
    };
    let mut det = YunetDetector::init(&config)?;
    let mut np = NativePose::init(&config)?;
    let mut cam = mvm_pipeline::native::NativeCamera::open("/dev/video0", 1280, 720, true)?;
    for _ in 0..5 {
        let _ = cam.read();
    }

    eprintln!(">>> LIVE pose (YuNet-face ROI) — {secs:.0}s. Sit / stand / cross arms / lean.");
    let t0 = Instant::now();
    let (mut frames, mut with_face, mut with_pose) = (0u32, 0u32, 0u32);
    while t0.elapsed().as_secs_f64() < secs {
        let frame = match cam.read() {
            Ok(f) => f,
            Err(_) => continue,
        };
        frames += 1;
        let faces: Vec<[f32; 4]> = det
            .detect(&frame)?
            .iter()
            .map(|d| [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2])
            .collect();
        if !faces.is_empty() {
            with_face += 1;
        }
        let poses = np.poses_from_faces(&frame, &faces)?;
        let mut line = format!("[f{frames:03}] faces={}", faces.len());
        let mut any = false;
        for (i, p) in poses.iter().enumerate() {
            match p {
                Some(p) => {
                    any = true;
                    line += &format!(
                        " | F{i}: {}@{:.2} maha={:.2}{}",
                        p.label,
                        p.confidence,
                        p.maha_score,
                        if p.ood { " OOD" } else { "" }
                    );
                }
                None => line += &format!(" | F{i}: pose<thr"),
            }
        }
        if any {
            with_pose += 1;
        }
        println!("{line}");
    }
    let fps = frames as f64 / t0.elapsed().as_secs_f64();
    println!("\n{frames} frames, {fps:.1} fps; face in {with_face}, pose in {with_pose}");
    Ok(())
}
