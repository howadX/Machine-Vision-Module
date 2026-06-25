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
    /// EMA alpha for depth smoothing. Higher = faster response; lower = smoother.
    pub depth_ema_alpha: f32,
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
            depth_ema_alpha: 0.3,
        }
    }
}
