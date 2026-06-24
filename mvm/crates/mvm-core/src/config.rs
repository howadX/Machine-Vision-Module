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
    /// Population-average face height in metres (depth-from-known-size prior).
    /// ±12% population variability → ±12% depth error. Override per-identity for ±2-3%.
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
            face_height_m: 0.19,
            depth_ema_alpha: 0.3,
        }
    }
}
