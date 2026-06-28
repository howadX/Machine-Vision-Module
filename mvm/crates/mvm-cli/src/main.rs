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

    // ── body-primary face↔body matching (phase 8, FRONTIER) ──
    fn face(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection::bare(x1, y1, x2, y2)
    }

    #[test]
    fn best_face_for_nose_matches_containing() {
        let faces = vec![face(0.20, 0.10, 0.30, 0.20)]; // contains nose (0.25,0.15)
        let used = vec![false];
        assert_eq!(best_face_for_nose(&faces, &used, (0.25, 0.15)), Some(0));
    }

    #[test]
    fn best_face_for_nose_rejects_outside() {
        let faces = vec![face(0.20, 0.10, 0.30, 0.20)];
        let used = vec![false];
        assert_eq!(best_face_for_nose(&faces, &used, (0.80, 0.80)), None);
    }

    #[test]
    fn best_face_for_nose_skips_used() {
        // both faces contain the nose; first already taken → second
        let faces = vec![face(0.10, 0.10, 0.40, 0.40), face(0.10, 0.10, 0.40, 0.40)];
        let used = vec![true, false];
        assert_eq!(best_face_for_nose(&faces, &used, (0.20, 0.20)), Some(1));
    }

    #[test]
    fn best_face_for_nose_picks_smallest() {
        // both contain nose (0.2,0.2); second is smaller (tighter) → preferred
        let faces = vec![face(0.0, 0.0, 0.5, 0.5), face(0.15, 0.15, 0.30, 0.30)];
        let used = vec![false, false];
        assert_eq!(best_face_for_nose(&faces, &used, (0.20, 0.20)), Some(1));
    }

    #[test]
    fn face_center_in_any_suppresses_fallback() {
        let body = mvm_core::BBox { x1: 0.0, y1: 0.0, x2: 0.5, y2: 1.0 };
        let inside = face(0.20, 0.10, 0.30, 0.20); // centre (0.25,0.15) ∈ body
        let outside = face(0.70, 0.10, 0.80, 0.20); // centre (0.75,0.15) ∉ body
        assert!(face_center_in_any(&inside.bbox, &[body]));
        assert!(!face_center_in_any(&outside.bbox, &[body]));
        assert!(!face_center_in_any(&inside.bbox, &[]), "no regions → not suppressed");
    }

    /// The swap-bug regression: two people whose person bboxes OVERLAP. A
    /// body-bbox-center match would assign person B's face to person A; the
    /// nose-based match keeps each face with its own head.
    #[test]
    fn nose_match_survives_overlapping_bodies() {
        // Person A face left, Person B face right; noses inside their own faces.
        let faces = vec![face(0.10, 0.10, 0.25, 0.30), face(0.60, 0.10, 0.75, 0.30)];
        let used = vec![false, false];
        let nose_a = (0.17, 0.20);
        let nose_b = (0.67, 0.20);
        assert_eq!(best_face_for_nose(&faces, &used, nose_a), Some(0));
        assert_eq!(best_face_for_nose(&faces, &used, nose_b), Some(1));
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
    /// --native-camera: full Python-light hot-path (phase 7.4, Variant A).
    /// Pure-Rust v4l2 capture + YuNet detection + native enrichment; ZERO Python,
    /// no pose (until 7.3). Implies --native. Frame stays in memory (no disk
    /// round-trip). NOTE: 3D uses face_height_m calibrated for SCRFD boxes —
    /// provisional until recalibrated for YuNet boxes (operator, 7.1 gate).
    native_camera: bool,
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
        native_camera: false,
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
            "--native-camera" => { args.native_camera = true; args.native = true; }
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
/// Crop a face from `face_bbox` (normalized) and enrich `det` with emotion +
/// identity. Shared by the body-primary path (matched face) and the face-entity
/// fallback. Mirrors the bridge-path enrichment so behaviour is identical.
#[allow(clippy::too_many_arguments)]
fn enrich_face_crop(
    frame: &image::DynamicImage,
    face_bbox: &mvm_core::BBox,
    det: &mut Detection,
    emotion_mod: Option<&mut mvm_emotion::EmotionModule>,
    identity_mod: Option<&mut mvm_identity::IdentityModule>,
    config: &mvm_core::Config,
    emotion_debug: bool,
    det_idx: usize,
    debug_out: &mut Vec<state::EmotionDebugEntry>,
) {
    let (fw, fh) = (frame.width(), frame.height());
    let x1 = (face_bbox.x1 * fw as f32) as u32;
    let y1 = (face_bbox.y1 * fh as f32) as u32;
    let x2 = (face_bbox.x2 * fw as f32) as u32;
    let y2 = (face_bbox.y2 * fh as f32) as u32;
    let face = frame.crop_imm(x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1));

    // Emotion: keep higher calibrated-confidence prediction.
    if let Some(emo) = emotion_mod {
        if let Ok(result) = emo.classify(&face) {
            let py_emotion = det.emotion.clone();
            let py_conf = det.emotion_confidence;
            det.emotion_distribution = Some(result.distribution);
            let winner = if emotion_wins(&result, det.emotion_confidence) {
                det.emotion = Some(result.label.clone());
                det.emotion_confidence = Some(result.calibrated_conf);
                "rust"
            } else {
                "python"
            };
            if emotion_debug {
                debug_out.push(state::EmotionDebugEntry {
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

    // Identity + embedding (margin gate, #5/#6).
    if let Some(id) = identity_mod {
        if let Ok((name, sim, emb)) =
            id.identify_margin(&face, config.identity_abs_thresh, config.identity_margin_thresh)
        {
            det.embedding = Some(emb);
            if det.identity.is_none() && name != "unknown" {
                det.identity = Some(name);
                det.identity_confidence = Some(sim);
            }
        }
    }
}

/// Index of the smallest unused YuNet face whose bbox **contains the body's nose
/// landmark** (`nose` normalized [0..1]). Keypoint-in-bbox association: the face
/// belongs to the body whose head is inside it. Robust to overlapping person
/// bboxes — a coarse body-bbox-center match swaps identities when two people are
/// close (their person boxes overlap); the nose is physically inside its own
/// face, so it never crosses over.
/// True if the face's centre falls inside any of `regions` (body bboxes this
/// frame, or active-track bboxes from last frame). Used to suppress the
/// face-entity fallback so a person isn't represented twice / doesn't flip
/// between a body-entity and a face-entity (A1 anti-flip, phase 8).
fn face_center_in_any(face: &mvm_core::BBox, regions: &[mvm_core::BBox]) -> bool {
    let cx = (face.x1 + face.x2) * 0.5;
    let cy = (face.y1 + face.y2) * 0.5;
    regions
        .iter()
        .any(|r| cx >= r.x1 && cx <= r.x2 && cy >= r.y1 && cy <= r.y2)
}

fn best_face_for_nose(faces: &[Detection], used: &[bool], nose: (f32, f32)) -> Option<usize> {
    let area = |b: &mvm_core::BBox| (b.x2 - b.x1) * (b.y2 - b.y1);
    faces
        .iter()
        .enumerate()
        .filter(|(i, _)| !used[*i])
        .filter(|(_, f)| {
            nose.0 >= f.bbox.x1 && nose.0 <= f.bbox.x2 && nose.1 >= f.bbox.y1 && nose.1 <= f.bbox.y2
        })
        .min_by(|(_, a), (_, b)| area(&a.bbox).partial_cmp(&area(&b.bbox)).unwrap())
        .map(|(i, _)| i)
}

/// Body-primary enrichment (phase 8, FRONTIER). Bodies (person detector) are the
/// carrying entity: each yields a track with posture + torso-3D **always**. A
/// YuNet face matched into the body adds identity + emotion (optional, frontal
/// only). A body without a face still tracks (identity/emotion None) — the core
/// FRONTIER fix. An unmatched face falls back to a face-entity so a blinked
/// body-detection never drops a known person (decision #4).
#[allow(clippy::too_many_arguments)]
fn build_body_primary(
    frame: &image::DynamicImage,
    native_pose: Option<&mut mvm_posture::NativePose>,
    detector: &mut mvm_detect::YunetDetector,
    mut emotion_mod: Option<&mut mvm_emotion::EmotionModule>,
    mut identity_mod: Option<&mut mvm_identity::IdentityModule>,
    config: &mvm_core::Config,
    emotion_debug: bool,
    debug_out: &mut Vec<state::EmotionDebugEntry>,
    prev_tracks: &[mvm_core::BBox],
) -> Vec<Detection> {
    let pinhole = mvm_core::space::Pinhole::from_config(config);
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let bodies = native_pose
        .map(|np| np.bodies(frame).unwrap_or_default())
        .unwrap_or_default();
    let body_bboxes: Vec<mvm_core::BBox> = bodies.iter().map(|b| b.bbox).collect();
    let faces = detector.detect(frame).unwrap_or_default();
    let mut face_used = vec![false; faces.len()];
    let mut dets: Vec<Detection> = Vec::with_capacity(bodies.len() + faces.len());

    for body in &bodies {
        let b = &body.bbox;
        let mut det = Detection::bare(b.x1, b.y1, b.x2, b.y2);
        // Posture (always present for a body).
        det.posture = Some(body.posture.label.to_string());
        det.posture_confidence = Some(body.posture.confidence);
        det.recon_error = Some(body.posture.recon_error);
        det.plausibility =
            Some(if body.posture.plausibility < 14.3 { "normal" } else { "unusual" }.to_string());
        // Torso-3D (anatomically stable; robust to head turn): shoulders 11/12, hips 23/24.
        // R (phantom filter, phase 8): only trust torso-3D when those landmarks are
        // actually visible. BlazePose visibility = sigmoid(logit) ∈ [0,1]; visible
        // joints sit ~0.9+. Hallucinated landmarks (occlusion / lying poses) score
        // low and otherwise produce garbage 3D (negative-y / far phantoms) — gate
        // them out (the body is still tracked, just without a position this frame).
        let lm = &body.landmarks_px;
        const TORSO_VIS_MIN: f32 = 0.5;
        let torso_vis = lm[11][3].min(lm[12][3]).min(lm[23][3]).min(lm[24][3]);
        if torso_vis >= TORSO_VIS_MIN {
            let shoulder = ((lm[11][0] + lm[12][0]) * 0.5, (lm[11][1] + lm[12][1]) * 0.5);
            let hip = ((lm[23][0] + lm[24][0]) * 0.5, (lm[23][1] + lm[24][1]) * 0.5);
            if let Some((pos, z, ws)) =
                mvm_core::space::torso_to_3d(&pinhole, shoulder, hip, config.torso_length_m)
            {
                det.distance = Some(z);
                det.position_x = Some(pos[0]);
                det.position_y = Some(pos[1]);
                det.position_z = Some(pos[2]);
                det.ws_confidence = Some(ws);
            }
        }
        // Match a frontal face by the body's NOSE landmark (lm[0]) → identity +
        // emotion (optional). Keypoint-in-bbox ties the face to the body whose
        // head is in it, so overlapping person bboxes never swap identities.
        let nose = (lm[0][0] / fw, lm[0][1] / fh);
        if let Some(fi) = best_face_for_nose(&faces, &face_used, nose) {
            face_used[fi] = true;
            let fb = faces[fi].bbox;
            let idx = dets.len();
            enrich_face_crop(
                frame, &fb, &mut det,
                emotion_mod.as_deref_mut(), identity_mod.as_deref_mut(),
                config, emotion_debug, idx, debug_out,
            );
        }
        dets.push(det);
    }

    // Fallback: unmatched faces → face-entity (don't lose a person whose body
    // detection blinked). 3D from face height (YuNet-calibrated `face_height_m`).
    //
    // A1 anti-flip (phase 8): suppress the fallback when the face is already
    // explained by a body THIS frame (nose-match just missed a tight YuNet box)
    // or by an active track from LAST frame (the person-detector blinked — let
    // the body track bridge the gap via Kalman inertia instead of swapping in a
    // posture-less, face-3D entity that makes id 0 jump body↔face every blink).
    for (fi, f) in faces.iter().enumerate() {
        if face_used[fi] {
            continue;
        }
        if face_center_in_any(&f.bbox, &body_bboxes) || face_center_in_any(&f.bbox, prev_tracks) {
            continue;
        }
        let fb = f.bbox;
        let mut det = Detection::bare(fb.x1, fb.y1, fb.x2, fb.y2);
        let cx = (fb.x1 + fb.x2) * 0.5;
        let cy = (fb.y1 + fb.y2) * 0.5;
        if let Some((pos, z, ws)) =
            mvm_core::space::face_bbox_to_3d(&pinhole, fb.y1, fb.y2, cx, cy, config.face_height_m)
        {
            det.distance = Some(z);
            det.position_x = Some(pos[0]);
            det.position_y = Some(pos[1]);
            det.position_z = Some(pos[2]);
            det.ws_confidence = Some(ws);
        }
        let idx = dets.len();
        enrich_face_crop(
            frame, &fb, &mut det,
            emotion_mod.as_deref_mut(), identity_mod.as_deref_mut(),
            config, emotion_debug, idx, debug_out,
        );
        dets.push(det);
    }

    dets
}

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

    // Camera. Two mutually-exclusive sources:
    //   --native-camera → pure-Rust v4l2 (phase 7.4-light, Variant A), ZERO Python.
    //   default         → Python bridge (full pipeline incl. pose).
    let mut camera: Option<mvm_pipeline::Camera> = None;
    let mut native_camera: Option<mvm_pipeline::native::NativeCamera> = None;
    if args.no_camera {
        eprintln!("  camera: disabled (--no-camera)");
    } else if args.native_camera {
        match mvm_pipeline::native::NativeCamera::open("/dev/video0", 1280, 720, true) {
            Ok(cam) => {
                let (w, h) = cam.dimensions();
                eprintln!("  camera: native v4l2 MJPG {w}x{h} (Python-light, no pose)");
                native_camera = Some(cam);
            }
            Err(e) => {
                eprintln!("  camera: native open failed: {e}");
                eprintln!("  camera: falling back to stub mode");
            }
        }
    } else {
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
    }

    // Native inference modules
    let mut config = mvm_core::Config::default();
    // Native (YuNet) capture uses YuNet-calibrated face height for 3D — the
    // SCRFD-tuned 0.44 would bias depth (phase 7.1 recalibration target).
    if args.native_camera {
        config.face_height_m = config.face_height_m_yunet;
    }
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

    // Native posture (phase 7.5.4) — only the pure-Rust --native-camera path. The
    // bridge path already gets pose from Python (Variant A). person_detection ROI
    // (the validated source) → BlazePose → VAE/classifier, multi-persona (fixes #3).
    let mut native_pose: Option<mvm_posture::NativePose> = None;
    if args.native_camera {
        eprintln!("  posture: loading native BlazePose (person det + pose + VAE)");
        #[cfg(feature = "embed-models")]
        {
            native_pose = Some(mvm_posture::NativePose::from_embedded()?);
        }
        #[cfg(not(feature = "embed-models"))]
        {
            native_pose = Some(mvm_posture::NativePose::init(&config)?);
        }
    }

    // Scene objects (phase 8.E, G₂ Scene group) — heavy YOLO detector on the
    // pure-Rust --native-camera path only. Gated on the model file: if the
    // bootstrap weights aren't downloaded yet, object nodes are simply disabled
    // (chain still runs, like a missing optional detector).
    let mut scene_detector: Option<mvm_scene::YoloDetector> = None;
    let mut object_tracker = mvm_scene::ObjectTracker::new();
    let mut latest_objects: Vec<mvm_scene::TrackedObject> = Vec::new();
    if args.native_camera {
        let model_path = config.models_path.join("yolo_object_detect.onnx");
        if model_path.exists() {
            match mvm_scene::YoloDetector::from_path(&model_path) {
                Ok(mut d) => {
                    // Filter A: accept only confident objects (above the flicker band).
                    d.set_score_threshold(config.object_score_thresh);
                    eprintln!(
                        "  scene: loading YOLO object detector ({}), cadence every {} frames, score≥{}",
                        model_path.display(),
                        config.object_cadence,
                        config.object_score_thresh
                    );
                    scene_detector = Some(d);
                }
                Err(e) => eprintln!("  scene: YOLO load failed: {e} (object nodes disabled)"),
            }
        } else {
            eprintln!(
                "  scene: no YOLO model at {} — object nodes disabled (download bootstrap weights)",
                model_path.display()
            );
        }
    }

    let mut tracker = mvm_core::PersonTracker::with_config(&config);
    // Body-primary (phase 8): 3D comes from the torso in `build_body_primary`
    // (set on each Detection), so disable the tracker's face-bbox pinhole — it
    // would otherwise overwrite torso-3D by treating the body bbox as a face.
    // The detection-provided 3D then passes through unchanged. Bridge path keeps
    // native pinhole 3D (face-keyed) as before.
    if args.native_camera {
        tracker.pinhole = None;
    }
    // Register gallery users → reserved stable ids (phase 8 identity model): a
    // recognised body adopts a per-person id (Вадим=0, …) while it holds the
    // user; anonymous bodies get ephemeral ids ≥ 10.
    if let Some(id) = &identity_mod {
        let mut names = id.gallery_names();
        names.sort(); // deterministic reserved-id assignment
        for name in names {
            let uid = tracker.register_user(name);
            eprintln!("  identity: user '{name}' → reserved id {uid}");
        }
    }
    let state_file = state::StateFile::open()?;
    let mut last_dump = Instant::now();
    let dump_interval = Duration::from_secs(dump_secs);

    let mut frame_count: u64 = 0;
    let start = Instant::now();

    // A/B detector coverage accumulators (phase 7.1, G4).
    let mut ab_cov = mvm_detect::Coverage::default();
    let mut ab_frames: u32 = 0;
    if native && (args.detect_ab || args.native_camera) {
        // Use YuNet's standard live threshold, not the loose gallery-build
        // threshold (0.3) left over from build_gallery.
        _detector.set_score_threshold(0.5);
    }
    if native && args.detect_ab {
        eprintln!("  detect-ab: logging native YuNet coverage vs Python SCRFD (measurement only)");
    }

    loop {
        // ── Frame + detections acquisition ──────────────────────────────────
        // `tick_frame` is the working image for this tick (enrichment + A/B).
        // Native-camera path (Variant A): capture in memory + YuNet detection,
        // no Python, no pose. Bridge path: detections from JSON, frame on disk.
        let mut emotion_debug_entries: Vec<state::EmotionDebugEntry> = Vec::new();
        let mut tick_frame: Option<image::DynamicImage> = None;
        let mut detections: Vec<Detection> = if let Some(ncam) = &mut native_camera {
            // Body-primary (phase 8): grab the frame here; detections are built
            // below from the person detector (bodies), not from YuNet faces.
            match ncam.read() {
                Ok(frame) => {
                    tick_frame = Some(frame);
                    Vec::new()
                }
                Err(e) => {
                    eprintln!("  native camera error: {e}");
                    break;
                }
            }
        } else {
            match &mut camera {
                Some(cam) => match cam.read() {
                    Ok(dets) => dets,
                    Err(e) => {
                        eprintln!("  camera error: {e}");
                        break;
                    }
                },
                None => vec![],
            }
        };

        // ── BODY-PRIMARY enrichment (phase 8, FRONTIER) — native-camera only ──
        // Bodies (person detector) are the carrying entity, so a person whose face
        // is turned/down still produces a track (posture + torso-3D); identity and
        // emotion attach only when a frontal face is present.
        if native_camera.is_some() {
            if let Some(frame) = &tick_frame {
                // Last frame's active track bboxes — let A1 suppress a face-entity
                // fallback when an existing track already covers that face.
                let prev_tracks: Vec<mvm_core::BBox> =
                    tracker.active().map(|t| t.bbox).collect();
                detections = build_body_primary(
                    frame,
                    native_pose.as_mut(),
                    &mut _detector,
                    emotion_mod.as_mut(),
                    identity_mod.as_mut(),
                    &config,
                    args.emotion_debug,
                    &mut emotion_debug_entries,
                    &prev_tracks,
                );
            }
        }

        // ── Scene objects (phase 8.E) on a cadence ──────────────────────────
        // YOLO is the heaviest detector (~39ms); run it every Nth frame and
        // reuse `latest_objects` in between (decision 8.0.1: cadence in the
        // synchronous loop). On non-cadence frames the previous objects persist.
        if let Some(det) = &mut scene_detector {
            if config.object_cadence <= 1 || frame_count % config.object_cadence as u64 == 0 {
                if let Some(frame) = &tick_frame {
                    match det.detect(frame) {
                        Ok(objs) => {
                            // Filters C + D: keep only interaction-relevant classes
                            // (the allow-list excludes "person" — humans are owned by
                            // the body-primary tracker G₁ — plus outdoor/vehicle/animal
                            // noise) and drop tiny boxes. Then tracker B assigns stable
                            // ids so a node id means the same object across frames.
                            let filtered: Vec<_> = objs
                                .into_iter()
                                .filter(|o| mvm_scene::is_relevant(o.class_name))
                                .filter(|o| {
                                    let area = (o.bbox.x2 - o.bbox.x1) * (o.bbox.y2 - o.bbox.y1);
                                    area >= config.object_min_area
                                })
                                .collect();
                            latest_objects = object_tracker.update(&filtered);
                        }
                        Err(e) => eprintln!("  scene: detect error: {e}"),
                    }
                }
            }
        }

        // Bridge path: the frame lives on disk (Python --write-frame). Load it
        // once for A/B + enrichment. Native path already holds it in memory.
        if tick_frame.is_none() && native && (args.detect_ab || !detections.is_empty()) {
            tick_frame = image::open("/tmp/mvm_frame.jpg").ok();
        }

        // A/B detector coverage (phase 7.1, G4) — bridge path only (the native
        // path already IS YuNet, so there's nothing to compare against).
        if native && args.detect_ab && native_camera.is_none() {
            if let Some(frame) = &tick_frame {
                if let Ok(yunet) = _detector.detect(frame) {
                    let cov = mvm_detect::compare_coverage(&yunet, &detections, 0.5);
                    ab_cov.matched += cov.matched;
                    ab_cov.a_only += cov.a_only;
                    ab_cov.b_only += cov.b_only;
                    ab_frames += 1;
                }
            }
        }

        // Bridge-path enrichment: emotion + identity per face (the native-camera
        // path is handled above by build_body_primary).
        if native && native_camera.is_none() && !detections.is_empty() {
            if let Some(frame) = &tick_frame {
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

        // Native posture for the native-camera path is now produced body-primary
        // (in build_body_primary, per body) — the old face-primary match-by-bbox
        // block was removed in phase 8. Bridge-path posture comes from Python.

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
            state_file.write(&tracker, fps, &latest_objects)?;
            if args.emotion_debug {
                state_file.dump_table_debug(&tracker, fps, &emotion_debug_entries);
            } else {
                state_file.dump_table(&tracker, fps);
            }
            if args.dump_json {
                let line = if args.emotion_debug {
                    state_file.to_json_line_debug(&tracker, fps, &emotion_debug_entries)
                } else {
                    state_file.to_json_line(&tracker, fps, &latest_objects)
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
