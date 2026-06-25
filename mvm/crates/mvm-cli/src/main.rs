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

/// Returns true when the Rust emotion result should replace the current (Python bridge) prediction.
/// Comparison uses `calibrated_conf` (prob × (1 - entropy)) — a high-entropy prediction
/// loses even if its raw probability looks reasonable.
fn emotion_wins(rust: &mvm_emotion::EmotionResult, py_conf: Option<f32>) -> bool {
    rust.calibrated_conf > py_conf.unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_emotion::EmotionResult;

    fn make_result(calibrated_conf: f32) -> EmotionResult {
        EmotionResult {
            label: "test".into(),
            prob: calibrated_conf,
            distribution: [0.0; 8],
            entropy: 0.0,
            uncertainty: 0.0,
            calibrated_conf,
        }
    }

    #[test]
    fn emotion_wins_when_no_python_conf() {
        assert!(emotion_wins(&make_result(0.6), None));
    }

    #[test]
    fn emotion_wins_when_rust_higher() {
        assert!(emotion_wins(&make_result(0.8), Some(0.5)));
    }

    #[test]
    fn emotion_loses_when_python_higher() {
        assert!(!emotion_wins(&make_result(0.5), Some(0.8)));
    }

    #[test]
    fn emotion_loses_on_tie() {
        assert!(!emotion_wins(&make_result(0.7), Some(0.7)));
    }

    /// High-entropy result loses even if raw prob would win.
    #[test]
    fn emotion_wins_uses_calibrated_conf() {
        // Rust prob=0.8, but high entropy → calibrated_conf = 0.8 * (1-0.9) = 0.08
        let uncertain = EmotionResult {
            label: "test".into(),
            prob: 0.8,
            distribution: [0.0; 8],
            entropy: 0.9,
            uncertainty: 0.9,
            calibrated_conf: 0.08,
        };
        // Python only has 0.1 confidence, but uncertain Rust should still lose
        assert!(!emotion_wins(&uncertain, Some(0.1)), "uncertain prediction should lose");
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
    /// --emotion-debug: show A/B table (EmotiEffLib vs HSEmotion) and add emotion_debug to JSON.
    emotion_debug: bool,
    /// --detect-ab: run native YuNet on each frame and log coverage vs Python
    /// SCRFD (phase 7.1, G4). Measurement only — does NOT change the detection
    /// source or 3D; that switch waits on the operator recalibration (7.1 gate).
    detect_ab: bool,
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
        emotion_debug: false,
        detect_ab: false,
    };

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--cpu"       => args.use_cpu = true,
            "--once"      => args.once = true,
            "--no-camera" => args.no_camera = true,
            "--native"    => args.native = true,
            "--dump-json"     => args.dump_json = true,
            "--emotion-debug" => args.emotion_debug = true,
            "--detect-ab"     => args.detect_ab = true,
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
    let config = mvm_core::Config::default();
    let mut emotion_mod: Option<mvm_emotion::EmotionModule> = None;
    let mut identity_mod: Option<mvm_identity::IdentityModule> = None;

    if native {
        eprintln!("  emotion: loading EmotiEffLib B0 ONNX");
        emotion_mod = Some(mvm_emotion::EmotionModule::init(&config)?);
        eprintln!("  identity: loading SFace ONNX");
        let mut id = mvm_identity::IdentityModule::init(&config)?;
        build_gallery(&mut id, &mut _detector, &config)?;
        identity_mod = Some(id);
    }

    let mut tracker = mvm_core::PersonTracker::with_config(&config);
    let state_file = state::StateFile::open()?;
    let mut last_dump = Instant::now();
    let dump_interval = Duration::from_secs(dump_secs);

    let mut frame_count: u64 = 0;
    let start = Instant::now();

    // A/B detector coverage accumulators (phase 7.1, G4).
    let mut ab_cov = mvm_detect::Coverage::default();
    let mut ab_frames: u32 = 0;
    if native && args.detect_ab {
        // Fair comparison: use YuNet's standard live threshold, not the loose
        // gallery-build threshold (0.3) left over from build_gallery.
        _detector.set_score_threshold(0.5);
        eprintln!("  detect-ab: logging native YuNet coverage vs Python SCRFD (measurement only)");
    }

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

        // A/B detector coverage (phase 7.1, G4). Runs even when SCRFD found
        // nothing, so "YuNet finds a face SCRFD missed" is captured. Measurement
        // only — does NOT change the detection source feeding the tracker.
        if native && args.detect_ab {
            if let Ok(frame) = image::open("/tmp/mvm_frame.jpg") {
                if let Ok(yunet) = _detector.detect(&frame) {
                    let cov = mvm_detect::compare_coverage(&yunet, &detections, 0.5);
                    ab_cov.matched += cov.matched;
                    ab_cov.a_only += cov.a_only;
                    ab_cov.b_only += cov.b_only;
                    ab_frames += 1;
                }
            }
        }

        // Native enrichment: emotion + identity per face
        let mut emotion_debug_entries: Vec<state::EmotionDebugEntry> = Vec::new();
        if native && !detections.is_empty() {
            // Frame is written to disk by the Python bridge (--write-frame)
            if let Ok(frame) = image::open("/tmp/mvm_frame.jpg") {
                let (fw, fh) = (frame.width(), frame.height());
                for (det_idx, det) in detections.iter_mut().enumerate() {
                    // Crop face
                    let x1 = (det.bbox.x1 * fw as f32) as u32;
                    let y1 = (det.bbox.y1 * fh as f32) as u32;
                    let x2 = (det.bbox.x2 * fw as f32) as u32;
                    let y2 = (det.bbox.y2 * fh as f32) as u32;
                    let face = frame.crop_imm(x1, y1, x2 - x1, y2 - y1);

                    // Emotion: keep higher calibrated-confidence prediction (Rust or Python bridge)
                    if let Some(ref mut emo) = emotion_mod {
                        if let Ok(result) = emo.classify(&face) {
                            let py_emotion = det.emotion.clone();
                            let py_conf = det.emotion_confidence;
                            // Always store the full distribution for PoE fusion (6.4).
                            det.emotion_distribution = Some(result.distribution);
                            let winner = if emotion_wins(&result, det.emotion_confidence) {
                                det.emotion = Some(result.label.clone());
                                det.emotion_confidence = Some(result.calibrated_conf);
                                "rust"
                            } else {
                                "python"
                            };
                            if args.emotion_debug {
                                emotion_debug_entries.push(state::EmotionDebugEntry {
                                    id: det_idx as u32,
                                    rust_emotion: result.label,
                                    rust_conf: result.calibrated_conf,
                                    py_emotion,
                                    py_conf,
                                    winner,
                                });
                            }
                        }
                    }

                    // Identity + embedding (phase 7.5).
                    // Always embed the face (for cosine re-link #6, even unknown
                    // faces). Apply the SFace label only when the Python bridge
                    // didn't already set one (ArcFace > SFace), and only through
                    // the margin gate (fixes #5: stranger no longer → "Вадим").
                    if let Some(ref mut id) = identity_mod {
                        if let Ok((name, sim, emb)) = id.identify_margin(
                            &face,
                            config.identity_abs_thresh,
                            config.identity_margin_thresh,
                        ) {
                            det.embedding = Some(emb);
                            if det.identity.is_none() && name != "unknown" {
                                det.identity = Some(name);
                                det.identity_confidence = Some(sim);
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
            if args.detect_ab && ab_frames > 0 {
                // YuNet ≥ SCRFD coverage gate: a_only ≥ b_only means YuNet finds
                // at least as many faces as SCRFD. b_only = SCRFD-found YuNet-missed.
                eprintln!(
                    "  detect-ab[{ab_frames}f]: matched={} yunet_only={} scrfd_only={} \
                     → YuNet {} SCRFD",
                    ab_cov.matched, ab_cov.a_only, ab_cov.b_only,
                    if ab_cov.b_only == 0 { "covers" } else { "MISSES some" },
                );
            }
            state_file.write(&tracker, fps)?;
            if args.emotion_debug {
                state_file.dump_table_debug(&tracker, fps, &emotion_debug_entries);
            } else {
                state_file.dump_table(&tracker, fps);
            }
            if args.dump_json {
                let line = if args.emotion_debug {
                    state_file.to_json_line_debug(&tracker, fps, &emotion_debug_entries)
                } else {
                    state_file.to_json_line(&tracker, fps)
                };
                println!("{line}");
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
