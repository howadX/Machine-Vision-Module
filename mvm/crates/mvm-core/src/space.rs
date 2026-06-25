//! Metric 3D positioning via pinhole geometry (depth-from-known-size).
//!
//! Formula: `z = fx · real_size / pixel_size`
//! where `pixel_size` is the face bbox height in pixels (denormalised),
//! and `real_size` is the known real-world face height (population prior or
//! identity-linked personal value).
//!
//! ## Accuracy envelope (EMPIRICAL — calibrated 2026-06-25, ceiling mount)
//!
//! Tape-measured sweep (subject = Вадим, native pipeline, `tools/calib_3d.py`).
//! The pinhole invariant `pixel_h · z` is constant when the model holds:
//!
//! | tape z | bbox_h px | `pixel_h·z` | est. z @ 0.44 | note                  |
//! |--------|-----------|-------------|---------------|-----------------------|
//! | 0.30 m | 472.6     | 141.8       | 0.38 m        | bbox clipped by frame |
//! | 0.50 m | 350.0     | 175.0       | 0.51 m        | slight clip           |
//! | 1.00 m | 181.7     | 181.7       | 0.99 m        | ✅ fit                |
//! | 1.50 m | 120.2     | 180.3       | 1.50 m        | ✅ fit                |
//! | 2.00 m |  90.8     | 181.6       | 1.99 m        | ✅ fit                |
//! | 2.50 m |  64.1     | 160.2       | 2.81 m        | subject moved (noisy) |
//!
//! Over the stable band **1.0–2.0 m the invariant is 181 px·m within <1%**, so a
//! single constant `face_height_m = 181/fx ≈ 0.44` fits to **<3%** (0.5–2.0 m) —
//! far better than the theoretical ±15-30%, because this is effectively an
//! identity-locked calibration on a fixed mount.
//!
//! Limits: **near floor ~0.4 m** (face bbox saturates against the frame → depth
//! over-estimates), and capture noise grows past ~2.2 m. `face_height_m` here is
//! the *effective bbox* height (≈2× anatomical), tied to the current SCRFD bbox
//! convention — **recalibrate when detection goes native (phase 7)**.
//!
//! Valid range: 0.3–3.0 m. Outside → `ws_confidence = 0.0` (uncalibrated).
//!
//! ## Tuning
//!
//! | Config field         | Default | Tune when |
//! |----------------------|---------|-----------|
//! | `fx_ratio`           | 0.64    | Different camera / lens |
//! | `face_height_m`      | 0.44    | Different detector bbox convention / deployment |
//! | `depth_ema_alpha`    | 0.3     | More/less temporal smoothing on z |

use crate::config::Config;

const DEPTH_MIN_M: f32 = 0.3;
const DEPTH_MAX_M: f32 = 3.0; // physical room range; calibrated 2026-06-25 (was 25.0 for sweep)

/// Pinhole camera intrinsics derived from `Config`.
#[derive(Debug, Clone, Copy)]
pub struct Pinhole {
    pub fx: f32,
    pub fy: f32,
    pub cx_px: f32, // principal point x (pixels)
    pub cy_px: f32, // principal point y (pixels)
    pub width: f32,
    pub height: f32,
}

impl Pinhole {
    /// Derive intrinsics from config. `fx = fx_ratio · width`, `fy = fx` (square pixels).
    pub fn from_config(cfg: &Config) -> Self {
        let w = cfg.width as f32;
        let h = cfg.height as f32;
        let fx = cfg.fx_ratio * w;
        Self { fx, fy: fx, cx_px: w * 0.5, cy_px: h * 0.5, width: w, height: h }
    }

    /// Estimate depth (metres) from a known real-world size and its pixel footprint.
    /// Returns `None` if `pixel_size` is non-positive or result is outside valid range.
    pub fn distance_from_size(&self, pixel_size: f32, real_size: f32) -> Option<f32> {
        if pixel_size <= 0.0 || real_size <= 0.0 {
            return None;
        }
        let z = self.fx * real_size / pixel_size;
        if z >= DEPTH_MIN_M && z <= DEPTH_MAX_M { Some(z) } else { None }
    }

    /// Back-project a pixel coordinate + depth to metric 3D (origin = camera).
    pub fn pixel_to_3d(&self, px_x: f32, px_y: f32, z: f32) -> [f32; 3] {
        let x = (px_x - self.cx_px) * z / self.fx;
        let y = (px_y - self.cy_px) * z / self.fy;
        [x, y, z]
    }
}

/// EMA smoothing state for a single depth estimate.
#[derive(Debug, Clone, Copy)]
pub struct DepthEma {
    pub z: f32,
    alpha: f32,
}

impl DepthEma {
    pub fn new(z: f32, alpha: f32) -> Self {
        Self { z, alpha }
    }

    /// Update with a new observation; returns smoothed z.
    pub fn update(&mut self, z_new: f32) -> f32 {
        self.z = self.alpha * z_new + (1.0 - self.alpha) * self.z;
        self.z
    }
}

/// Compute native 3D position for a face bbox track.
///
/// Returns `(pos_3d, distance, ws_confidence)` or `None` if depth is implausible.
///
/// - `bbox_y1/y2` — normalised face bbox vertical span [0..1]
/// - `bbox_cx` — normalised horizontal centre [0..1]
/// - `real_face_height_m` — population prior or identity-linked personal height
pub fn face_bbox_to_3d(
    pinhole: &Pinhole,
    bbox_y1: f32,
    bbox_y2: f32,
    bbox_cx: f32,
    bbox_cy: f32,
    real_face_height_m: f32,
) -> Option<([f32; 3], f32, f32)> {
    let pixel_h = (bbox_y2 - bbox_y1).abs() * pinhole.height;
    let pixel_cx = bbox_cx * pinhole.width;
    let pixel_cy = bbox_cy * pinhole.height;

    let z = pinhole.distance_from_size(pixel_h, real_face_height_m)?;
    let pos = pinhole.pixel_to_3d(pixel_cx, pixel_cy, z);

    // Confidence: inversely proportional to depth (larger bbox = more reliable).
    // Capped at 1.0; empirically good below ~2 m.
    let ws_confidence = (1.0 - (z - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)).clamp(0.0, 1.0);

    Some((pos, z, ws_confidence))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pinhole() -> Pinhole {
        Pinhole { fx: 410.0, fy: 410.0, cx_px: 320.0, cy_px: 240.0, width: 640.0, height: 480.0 }
    }

    /// Basic pinhole formula: z = fx · real / pixel.
    #[test]
    fn distance_from_size_pinhole() {
        let ph = test_pinhole();
        // At 1m, face height ~78px: z = 410 * 0.19 / 78 ≈ 1.0
        let pixel_size = 410.0 * 0.19; // ~77.9px → z = 1.0m
        let z = ph.distance_from_size(pixel_size, 0.19).expect("should be in range");
        assert!((z - 1.0).abs() < 0.01, "z={:.4} expected ~1.0m", z);
    }

    /// Centre pixel backprojects to (≈0, ≈0, z).
    #[test]
    fn pixel_to_3d_centre_gives_zero_xy() {
        let ph = test_pinhole();
        let pos = ph.pixel_to_3d(320.0, 240.0, 2.0);
        assert!(pos[0].abs() < 1e-4, "x should be ~0: {}", pos[0]);
        assert!(pos[1].abs() < 1e-4, "y should be ~0: {}", pos[1]);
        assert!((pos[2] - 2.0).abs() < 1e-4, "z should equal input: {}", pos[2]);
    }

    /// Right-of-centre pixel → positive x; below-centre → positive y.
    #[test]
    fn pixel_to_3d_off_centre_signs_correct() {
        let ph = test_pinhole();
        let pos = ph.pixel_to_3d(400.0, 300.0, 1.0);
        assert!(pos[0] > 0.0, "right of centre → positive x");
        assert!(pos[1] > 0.0, "below centre → positive y");
    }

    /// EMA converges z toward new observations over N frames.
    #[test]
    fn depth_ema_smooths() {
        let mut ema = DepthEma::new(2.0, 0.3);
        // Observe constant 1.0 — should converge toward 1.0
        for _ in 0..30 {
            ema.update(1.0);
        }
        assert!((ema.z - 1.0).abs() < 0.05, "EMA should converge to 1.0, got {:.4}", ema.z);
    }

    /// Implausible depth → None (uncalibrated).
    #[test]
    fn depth_rejects_implausible() {
        let ph = test_pinhole();
        // Very small bbox → z > 6m
        assert!(ph.distance_from_size(1.0, 0.19).is_none(), "z > 6m should be None");
        // Zero pixel size
        assert!(ph.distance_from_size(0.0, 0.19).is_none());
        // Very large bbox → z < 0.3m
        assert!(ph.distance_from_size(1000.0, 0.19).is_none(), "z < 0.3m should be None");
    }

    /// Larger bbox (closer object) → smaller z. Monotonicity.
    #[test]
    fn closer_object_larger_bbox_smaller_z() {
        let ph = test_pinhole();
        // close: bbox height = 156px (≈0.5m); far: 78px (≈1m)
        let z_close = ph.distance_from_size(156.0, 0.19).unwrap();
        let z_far   = ph.distance_from_size(78.0,  0.19).unwrap();
        assert!(z_close < z_far, "larger bbox should give smaller z: {:.4} vs {:.4}", z_close, z_far);
    }

    /// Identity-linked prior (personal height) overrides population prior.
    #[test]
    fn identity_prior_overrides_population() {
        let ph = test_pinhole();
        let pixel_h = 78.0; // 78px ≈ 1m with face_height=0.19
        let z_pop = ph.distance_from_size(pixel_h, 0.19).unwrap();
        // Personal height 10% taller
        let z_personal = ph.distance_from_size(pixel_h, 0.19 * 1.1).unwrap();
        assert!((z_personal - z_pop).abs() > 0.05, "personal prior should yield different z");
        // Personal z should be proportionally larger
        assert!(z_personal > z_pop, "taller face → farther estimated depth");
    }
}
