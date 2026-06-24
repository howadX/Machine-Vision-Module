//! mvm-cli — MVM binary entry point.
//!
//! Modes:
//!   mvm                        continuous, Python bridge camera
//!   mvm --once                 one tick, then exit
//!   mvm --cpu                  CPU-only mode
//!   mvm --bridge-path <DIR>    path to old MVM repo
//!   mvm --native               use Rust-native emotion + identity (FFmpeg capture)

mod state;

use anyhow::{Context, Result};
use mvm_core::Module;
use mvm_core::Detection;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Returns true when Rust emotion should replace the current (Python bridge) prediction.
/// Rust wins only if its confidence is strictly higher.
fn emotion_wins(rust_conf: f32, current_conf: Option<f32>) -> bool {
    rust_conf > current_conf.unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emotion_wins_when_no_python_conf() {
        assert!(emotion_wins(0.6, None));
    }

    #[test]
    fn emotion_wins_when_rust_higher() {
        assert!(emotion_wins(0.8, Some(0.5)));
    }

    #[test]
    fn emotion_loses_when_python_higher() {
        assert!(!emotion_wins(0.5, Some(0.8)));
    }

    #[test]
    fn emotion_loses_on_tie() {
        assert!(!emotion_wins(0.7, Some(0.7)));
    }
}

struct Args {
    use_cpu: bool,
    dump_interval: u64,
    once: bool,
    no_camera: bool,
    native: bool,
    bridge_path: Option<PathBuf>,
    /// --dump-json: print one JSON line per tick to stdout.
    dump_json: bool,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut args = Args {
        use_cpu: false,
        dump_interval: 180,
        once: false,
        no_camera: false,
        native: false,
        bridge_path: None,
        dump_json: false,
    };

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--cpu"       => args.use_cpu = true,
            "--once"      => args.once = true,
            "--no-camera" => args.no_camera = true,
            "--native"    => args.native = true,
            "--dump-json" => args.dump_json = true,
            "--bridge-path" => {
                if i + 1 < raw.len() {
                    args.bridge_path = Some(PathBuf::from(&raw[i + 1]));
                    i += 1;
                }
            }
            "--dump-interval" => {
                if i + 1 < raw.len() {
                    args.dump_interval = raw[i + 1].parse().unwrap_or(180);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    args
}

/// Build identity gallery from reference photos.
/// Supports multi-angle: p_V_01.jpg, p_V_02.jpg, … all map to "Вадим".
/// Uses YuNet to detect and crop faces before SFace embedding.
/// EMA (α=0.3) merges multiple photos of the same person.
fn build_gallery(
    identity: &mut mvm_identity::IdentityModule,
    detector: &mut mvm_detect::YunetDetector,
    _config: &mvm_core::Config,
) -> Result<()> {
    // Search both: Python MVM repo and local mvm repo
    let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_default();
    let candidates = vec![
        home.join("repos/Machine Vision Module/models/references"),
        home.join("repos/mvm/models/references"),
    ];

    let name_map: &[(&[&str], &str)] = &[
        (&["V"], "Вадим"),
    ];

    // Lower threshold for gallery: reference photos can be small/low-quality.
    detector.set_score_threshold(0.3);
    let mut total = 0usize;
    for ref_dir in &candidates {
        if !ref_dir.is_dir() {
            continue;
        }
        let entries = match std::fs::read_dir(ref_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if !fname.ends_with(".jpg") {
                continue;
            }
            // p_V_01.jpg → prefix=V,  p_V.jpg → prefix=V
            let stem = fname.trim_end_matches(".jpg");
            let prefix = if let Some(rest) = stem.strip_prefix("p_") {
                rest.split('_').next().unwrap_or(rest)
            } else {
                continue;
            };

            if let Some(&(_, name)) = name_map.iter().find(|(keys, _)| keys.contains(&prefix)) {
                let path = ref_dir.join(&fname);
                let img = image::open(&path)
                    .with_context(|| format!("Failed to open {}", path.display()))?;

                // Detect and crop face with YuNet before SFace embedding
                if let Ok(dets) = detector.detect(&img) {
                    if let Some(best) = dets.first() {
                        let (w, h) = (img.width(), img.height());
                        let x1 = (best.bbox.x1 * w as f32) as u32;
                        let y1 = (best.bbox.y1 * h as f32) as u32;
                        let x2 = (best.bbox.x2 * w as f32) as u32;
                        let y2 = (best.bbox.y2 * h as f32) as u32;
                        if x2 > x1 && y2 > y1 {
                            let face = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
                            identity.add_reference(name, &face)?;
                            total += 1;
                            continue;
                        }
                    }
                }
                eprintln!("  gallery: no face in {fname}, skipping");
            }
        }
    }

    eprintln!(
        "  gallery: {} people ({} reference photos)",
        identity.gallery_size(),
        total
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args();
    let dump_secs = args.dump_interval;
    let once = args.once;
    let native = args.native;

    let mode_label = if native { "native, " } else { "" };
    let json_label = if args.dump_json { "json, " } else { "" };
    eprintln!(
        "mvm v{} — starting (dump every {dump_secs}s, {mode_label}{json_label}{}mode)",
        env!("CARGO_PKG_VERSION"),
        if once { "once, " } else { "" }
    );

    // Init detector (YuNet)
    #[cfg(feature = "embed-models")]
    let mut _detector = {
        let model_bytes =
            include_bytes!("../../../models/face_detection_yunet_2023mar.onnx");
        eprintln!(
            "  detect: YuNet from embedded bytes ({} bytes)",
            model_bytes.len()
        );
        mvm_detect::YunetDetector::from_bytes(model_bytes)?
    };

    #[cfg(not(feature = "embed-models"))]
    let mut _detector = {
        let config = mvm_core::Config::default();
        eprintln!(
            "  detect: loading YuNet from {}",
            config.models_path.display()
        );
        mvm_detect::YunetDetector::init(&config)?
    };

    // Camera: Python bridge (unless --no-camera)
    let mut camera: Option<mvm_pipeline::Camera> = None;
    if !args.no_camera {
        let mut cfg = mvm_pipeline::CameraConfig::default();
        if let Some(ref bp) = args.bridge_path {
            cfg.mvm_repo = bp.clone();
        }
        if args.use_cpu {
            cfg.cpu = true;
        }

        if native {
            cfg.frame_output = Some(PathBuf::from("/tmp/mvm_frame.jpg"));
        }

        match mvm_pipeline::Camera::open(&cfg) {
            Ok(cam) => {
                eprintln!("  camera: connected via Python bridge");
                camera = Some(cam);
            }
            Err(e) => {
                eprintln!("  camera: {e}");
                eprintln!("  camera: falling back to stub mode (--no-camera)");
            }
        }
    } else {
        eprintln!("  camera: disabled (--no-camera)");
    }

    // Native inference modules
    let mut emotion_mod: Option<mvm_emotion::EmotionModule> = None;
    let mut identity_mod: Option<mvm_identity::IdentityModule> = None;

    if native {
        let config = mvm_core::Config::default();
        eprintln!("  emotion: loading EmotiEffLib B0 ONNX");
        emotion_mod = Some(mvm_emotion::EmotionModule::init(&config)?);
        eprintln!("  identity: loading SFace ONNX");
        let mut id = mvm_identity::IdentityModule::init(&config)?;
        build_gallery(&mut id, &mut _detector, &config)?;
        identity_mod = Some(id);
    }

    let mut tracker = mvm_core::PersonTracker::new();
    let state_file = state::StateFile::open()?;
    let mut last_dump = Instant::now();
    let dump_interval = Duration::from_secs(dump_secs);

    let mut frame_count: u64 = 0;
    let start = Instant::now();

    loop {
        let mut detections: Vec<Detection> = match &mut camera {
            Some(cam) => match cam.read() {
                Ok(dets) => dets,
                Err(e) => {
                    eprintln!("  camera error: {e}");
                    break;
                }
            },
            None => vec![],
        };

        // Native enrichment: emotion + identity per face
        if native && !detections.is_empty() {
            // Frame is written to disk by the Python bridge (--write-frame)
            if let Ok(frame) = image::open("/tmp/mvm_frame.jpg") {
                let (fw, fh) = (frame.width(), frame.height());
                for det in &mut detections {
                    // Crop face
                    let x1 = (det.bbox.x1 * fw as f32) as u32;
                    let y1 = (det.bbox.y1 * fh as f32) as u32;
                    let x2 = (det.bbox.x2 * fw as f32) as u32;
                    let y2 = (det.bbox.y2 * fh as f32) as u32;
                    let face = frame.crop_imm(x1, y1, x2 - x1, y2 - y1);

                    // Emotion: keep higher-confidence prediction (Rust or Python bridge)
                    if let Some(ref mut emo) = emotion_mod {
                        if let Ok((label, conf)) = emo.classify(&face) {
                            if emotion_wins(conf, det.emotion_confidence) {
                                det.emotion = Some(label);
                                det.emotion_confidence = Some(conf);
                            }
                        }
                    }

                    // Identity — skip if Python bridge already set it (ArcFace > SFace)
                    if det.identity.is_none() {
                        if let Some(ref mut id) = identity_mod {
                            if let Ok((name, sim)) = id.identify(&face, 0.35) {
                                if name != "unknown" {
                                    det.identity = Some(name);
                                    det.identity_confidence = Some(sim);
                                }
                            }
                        }
                    }
                }
            }
        }

        frame_count += 1;
        tracker.tick(&detections);

        if last_dump.elapsed() >= dump_interval {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = if elapsed > 0.0 {
                frame_count as f64 / elapsed
            } else {
                0.0
            };
            eprintln!(
                "  tick {frame_count}: {} tracks, {fps:.1} fps",
                tracker.track_count()
            );
            state_file.write(&tracker, fps)?;
            state_file.dump_table(&tracker, fps);
            if args.dump_json {
                println!("{}", state_file.to_json_line(&tracker, fps));
            }
            last_dump = Instant::now();

            if once {
                eprintln!("  once: done after {frame_count} frames");
                break;
            }
        }

        // Camera already paces itself; just a tiny yield
        std::thread::sleep(Duration::from_millis(1));
    }

    Ok(())
}
