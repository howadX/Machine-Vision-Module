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
    /// YuNet boxes have a different density than SCRFD, so the SCRFD-calibrated
    /// `face_height_m` (0.44) is wrong for `--native-camera`. The cli swaps this
    /// in when native capture is active. ⚠️ PLACEHOLDER 0.44 until the operator
    /// recalibrates: `venv/bin/python3 tools/calib_3d.py --native-camera …`.
    pub face_height_m_yunet: f32,
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
    /// fixes #5). Stricter than `identity_threshold`: live data shows real match
    /// ≈1.00 vs stranger ≈0.66, so ~0.8 separates them under a single-name gallery.
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
            face_height_m_yunet: 0.44, // PLACEHOLDER — recalibrate for YuNet boxes (7.1)
            depth_ema_alpha: 0.3,
            emotion_forget_lambda: 0.7,
            relink_window_frames: 90,
            relink_min_conf: 0.5,
            relink_cos_thresh: 0.5,
            identity_abs_thresh: 0.8,
            identity_margin_thresh: 0.15,
        }
    }
}
