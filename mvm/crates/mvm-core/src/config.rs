//! Global MVM configuration.

use std::path::PathBuf;

/// Global configuration for the MVM pipeline.
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the models directory (ONNX files).
    pub models_path: PathBuf,
    /// Camera device path (e.g. "/dev/video0").
    pub camera_path: String,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Minimum bbox IoU for person matching (default: 0.3).
    pub tracker_iou_threshold: f32,
    /// Number of missed frames before a track expires (default: 1800, ~30s at 60fps).
    pub tracker_timeout_frames: u64,
    /// Identity confidence threshold: below this, person is "unknown".
    pub identity_threshold: f32,
    /// Focal length ratio for pinhole calibration (fx / image_width).
    /// Calibrated per-camera. Default: 0.64 (user's USB camera, June 2026).
    pub fx_ratio: f32,
    /// Effective face-bbox height in metres (depth-from-known-size prior).
    /// NOT anatomical face height: absorbs the detector's bbox/anatomy ratio
    /// (SCRFD boxes ≈ 2× the chin-crown span). Empirically calibrated 2026-06-25
    /// on this ceiling mount: pinhole invariant `pixel_h·z` = 181 px·m over
    /// 1.0–2.0 m → 181/409.6 ≈ 0.44 (fit <3% across 0.5–2.0 m). See space.rs.
    /// Override per-identity via `personal_heights` for tighter fit.
    pub face_height_m: f32,
    /// Effective face-bbox height for **YuNet** boxes (phase 7.1/7.4 native path).
    /// YuNet boxes are tighter than SCRFD, so the SCRFD-calibrated `face_height_m`
    /// (0.44) is wrong for `--native-camera`. The cli swaps this in when native
    /// capture is active.
    ///
    /// Calibrated 2026-06-25 (`tools/calib_3d.py --native-camera`, tape sweep
    /// 0.5–2.5 m, `models/calib_3d_yunet.csv`). real_eff rises with distance due
    /// to the ~45° ceiling tilt: ≈0.30 near (0.5–1.0 m) → ≈0.35 far (2.0–2.5 m).
    /// A single balanced constant **0.32** keeps depth bias within ±~11% across
    /// the working range (inside the documented ±15–30% envelope) and leaves the
    /// ordinal/relative signal — the primary product — unaffected. The proper
    /// tilt fix is the landmark-calibration upgrade path (future phase).
    pub face_height_m_yunet: f32,
    /// Effective torso length in metres (shoulder-midpoint → hip-midpoint), the
    /// depth-from-known-size prior for **body-primary** 3D (phase 8, FRONTIER).
    /// Anatomically ≈0.5 m; like `face_height_m_yunet` this is an *effective*
    /// value that absorbs the ~45° ceiling tilt + BlazePose landmark convention
    /// → recalibrate by tape on this mount. Torso is anatomically stabler than
    /// the face bbox (doesn't jump when the head turns), so it is the preferred
    /// 3D source once the entity is the body. Interim anatomical default 0.5.
    pub torso_length_m: f32,
    /// EMA alpha for depth smoothing. Higher = faster response; lower = smoother.
    pub depth_ema_alpha: f32,
    /// Forgetting factor (λ < 1) for temporal emotion PoE fusion. Lower = forgets
    /// the prior faster (more reactive); higher = smoother but slower. Prevents
    /// the per-track distribution from collapsing to one-hot over many frames.
    pub emotion_forget_lambda: f32,
    /// Identity re-link window (frames). A detection that fails IoU matching may
    /// re-attach to a track missed within this many frames if their identity
    /// labels match — keeps the track id stable across motion gaps / brief exits.
    pub relink_window_frames: u64,
    /// Minimum identity confidence required to attempt an identity re-link.
    pub relink_min_conf: f32,
    /// Minimum cosine similarity between SFace embeddings to re-link (phase 7.5).
    /// Replaces label-only re-link → two distinct people never merge into one id.
    pub relink_cos_thresh: f32,
    /// Absolute SFace cosine above which an identity label is accepted (phase 7.5,
    /// fixes #5). Calibrated to the SFace cosine scale, NOT the old ArcFace one
    /// (OpenCV's canonical SFace threshold is 0.363). **Live-validated 2026-06-25
    /// on --native-camera:** the real Вадим lands at 0.55–0.59, a non-gallery
    /// stranger falls below 0.45 (→ unknown). 0.45 sits cleanly between. Was 0.8
    /// (ArcFace era), which would have rejected even the real Вадим on this path.
    pub identity_abs_thresh: f32,
    /// Minimum top1−top2 cosine margin to accept a label when the gallery has ≥2
    /// people (phase 7.5). Guards against ambiguous near-ties.
    pub identity_margin_thresh: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            models_path: PathBuf::from("./models"),
            camera_path: "/dev/video0".into(),
            width: 640,
            height: 480,
            tracker_iou_threshold: 0.3,
            tracker_timeout_frames: 1800,
            identity_threshold: 0.45,
            fx_ratio: 0.64,
            face_height_m: 0.44, // empirical, 2026-06-25 (was 0.19 population frontal)
            face_height_m_yunet: 0.32, // calibrated 2026-06-25 (YuNet, tilt-balanced)
            torso_length_m: 0.5, // anatomical prior (phase 8 body-primary); recalibrate by tape
            depth_ema_alpha: 0.3,
            emotion_forget_lambda: 0.7,
            relink_window_frames: 90,
            relink_min_conf: 0.5,
            relink_cos_thresh: 0.5,
            identity_abs_thresh: 0.45, // SFace cosine scale (was 0.8 ArcFace-era)
            identity_margin_thresh: 0.1,
        }
    }
}
