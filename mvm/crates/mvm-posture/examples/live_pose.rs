//! Live native pose on the real ceiling camera — full Rust path:
//!   NativeCamera (v4l2 MJPG → RGB, ROTATE_180) → NativePose (person det → ROI →
//!   BlazePose 33 landmarks → normalize → VAE+classifier) per person, multi-persona.
//!
//! Run: cargo run -p mvm-posture --example live_pose --release [seconds]

use mvm_core::Config;
use mvm_posture::NativePose;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let secs: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15.0);
    // optional 2nd arg: a dir to save each frame (for the 7.5.5 native-vs-Python oracle)
    let save_dir = std::env::args().nth(2);
    if let Some(d) = &save_dir {
        std::fs::create_dir_all(d)?;
    }

    let config = Config {
        models_path: std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")),
        ..Default::default()
    };
    let mut np = NativePose::init(&config)?;
    let mut cam = mvm_pipeline::native::NativeCamera::open("/dev/video0", 1280, 720, true)?;
    for _ in 0..5 {
        let _ = cam.read(); // warm-up
    }

    eprintln!(">>> LIVE native pose — {secs:.0}s. Move / sit / stand / raise arms / lie back.");
    let t0 = Instant::now();
    let (mut frames, mut max_people) = (0u32, 0usize);
    while t0.elapsed().as_secs_f64() < secs {
        let frame = match cam.read() {
            Ok(f) => f,
            Err(_) => continue,
        };
        let (w, h) = (frame.width() as f32, frame.height() as f32);
        frames += 1;
        if let Some(d) = &save_dir {
            frame.save(format!("{d}/f{frames:03}.png"))?;
        }
        let people = np.landmarks_per_person(&frame)?; // landmarks (for positions)
        let poses = np.poses(&frame)?; // classified posture
        max_people = max_people.max(poses.len());
        if save_dir.is_some() {
            let lbl = poses.first().map(|p| p.label).unwrap_or("none");
            eprintln!("RUST f{frames:03} {lbl}");
        }

        let mut line = format!("[f{frames:03}] persons={}", poses.len());
        for (i, p) in poses.iter().enumerate() {
            let lm = &people[i];
            let head = (lm[0][0] / w, lm[0][1] / h);
            let hip = (
                (lm[23][0] + lm[24][0]) / 2.0 / w,
                (lm[23][1] + lm[24][1]) / 2.0 / h,
            );
            line += &format!(
                " | P{i}: {}@{:.2} maha={:.2}{} head=({:.2},{:.2}) hip=({:.2},{:.2})",
                p.label,
                p.confidence,
                p.maha_score,
                if p.ood { " OOD" } else { "" },
                head.0,
                head.1,
                hip.0,
                hip.1,
            );
        }
        println!("{line}");
    }
    let fps = frames as f64 / t0.elapsed().as_secs_f64();
    println!("\n{frames} frames, {fps:.1} fps; max persons simultaneously = {max_people}");
    println!(
        "{}",
        if max_people > 0 {
            "=> native pose live ✓"
        } else {
            "=> no person tracked ✗"
        }
    );
    Ok(())
}
