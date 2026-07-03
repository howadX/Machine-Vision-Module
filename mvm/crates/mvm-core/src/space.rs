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

/// Compute native 3D position for a **body** track from torso landmarks
/// (phase 8 body-primary, FRONTIER). Depth-from-known-size on the torso length
/// (shoulder-midpoint → hip-midpoint), which is anatomically stabler than the
/// face bbox — it does not jump when the head turns away.
///
/// Returns `(pos_3d, distance, ws_confidence)` or `None` if depth is implausible.
///
/// - `shoulder_mid` / `hip_mid` — pixel coordinates (x, y) of the shoulder and
///   hip midpoints (BlazePose lm 11/12 and 23/24), in original-image pixels.
/// - `real_torso_m` — effective torso length prior (`Config::torso_length_m`).
///
/// Pixel torso length is the full euclidean distance (robust to torso lean), and
/// the back-projected centre is the midpoint of the two — so a person leaning or
/// turned still gets a stable depth as long as the torso is visible.
pub fn torso_to_3d(
    pinhole: &Pinhole,
    shoulder_mid: (f32, f32),
    hip_mid: (f32, f32),
    real_torso_m: f32,
) -> Option<([f32; 3], f32, f32)> {
    let dx = hip_mid.0 - shoulder_mid.0;
    let dy = hip_mid.1 - shoulder_mid.1;
    let pixel_len = (dx * dx + dy * dy).sqrt();

    let center_x = (shoulder_mid.0 + hip_mid.0) * 0.5;
    let center_y = (shoulder_mid.1 + hip_mid.1) * 0.5;

    let z = pinhole.distance_from_size(pixel_len, real_torso_m)?;
    let pos = pinhole.pixel_to_3d(center_x, center_y, z);

    let ws_confidence = (1.0 - (z - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)).clamp(0.0, 1.0);

    Some((pos, z, ws_confidence))
}

// ── Metric self-calibration (phase 11.D1) ─────────────────────────────────────

/// Minimum observations before the calibrator yields an estimate.
pub const CALIB_MIN_OBS: usize = 5;
/// Bounded observation memory (latest-wins ring).
const CALIB_CAP: usize = 128;

/// Online estimator for the pinhole scale constant `K = fx · real_size` of a
/// known-size class, from `(pixel_size, ground_truth_distance)` observations
/// (phase 11.D1, pure arithmetic, +0 dep).
///
/// The depth model is `d = fx · real / pixel`, so each observation pins
/// `K_i = d_i · pixel_i` — one scalar, independent of how `fx` and `real` split.
/// The estimate is the **median** `K` (robust to the effective-size jitter that
/// makes any single frame unreliable: a face/torso bbox is not a rigid ruler).
/// From `K` either factor follows once the other is fixed:
///   - `fx = K / real` — calibrate focal length against a size prior;
///   - `real = K / fx` — calibrate an unmeasured size (e.g. `torso_length_m`)
///     against a trusted `fx`, self-bootstrapped off the face-derived distance
///     of the *same* stable observation.
#[derive(Debug, Default, Clone)]
pub struct ScaleCalibrator {
    /// `K_i = pixel_i · distance_i`.
    samples: Vec<f32>,
}

impl ScaleCalibrator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one `(pixel_size, ground-truth distance)` observation. Non-positive
    /// inputs are ignored (no fabricated evidence); the ring is bounded.
    pub fn observe(&mut self, pixel_size: f32, true_distance_m: f32) {
        if pixel_size <= 0.0 || true_distance_m <= 0.0 {
            return;
        }
        if self.samples.len() == CALIB_CAP {
            self.samples.remove(0);
        }
        self.samples.push(pixel_size * true_distance_m);
    }

    pub fn observations(&self) -> usize {
        self.samples.len()
    }

    /// Median scale constant `K`, or `None` below [`CALIB_MIN_OBS`].
    pub fn scale(&self) -> Option<f32> {
        if self.samples.len() < CALIB_MIN_OBS {
            return None;
        }
        let mut s = self.samples.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        Some(if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) })
    }

    /// Estimated focal length given a real-size prior (`fx = K / real`).
    pub fn fx_given_real(&self, real_size_m: f32) -> Option<f32> {
        if real_size_m <= 0.0 {
            return None;
        }
        self.scale().map(|k| k / real_size_m)
    }

    /// Estimated real size given a trusted focal length (`real = K / fx`). The
    /// `torso_length_m` self-calibration path.
    pub fn real_given_fx(&self, fx: f32) -> Option<f32> {
        if fx <= 0.0 {
            return None;
        }
        self.scale().map(|k| k / fx)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pinhole() -> Pinhole {
        Pinhole { fx: 410.0, fy: 410.0, cx_px: 320.0, cy_px: 240.0, width: 640.0, height: 480.0 }
    }

    /// Torso depth: vertical torso of the right pixel length → expected z.
    #[test]
    fn torso_to_3d_vertical_gives_expected_depth() {
        let ph = test_pinhole();
        // real torso 0.5 m, want z=1.0 → pixel_len = fx·real/z = 410·0.5/1.0 = 205px
        let shoulder = (320.0, 140.0);
        let hip = (320.0, 345.0); // 205px below → vertical torso length 205
        let (pos, z, conf) = super::torso_to_3d(&ph, shoulder, hip, 0.5)
            .expect("torso depth should be in range");
        assert!((z - 1.0).abs() < 0.02, "z={z:.4} expected ~1.0m");
        assert!(pos[0].abs() < 1e-3, "centred torso → x≈0: {}", pos[0]);
        assert!(conf > 0.0, "ws_confidence should be positive at 1m");
    }

    /// Torso length is full euclidean (lean-robust): a tilted torso of the same
    /// span gives the same depth as a vertical one.
    #[test]
    fn torso_to_3d_lean_uses_euclidean_length() {
        let ph = test_pinhole();
        // diagonal torso, same 205px length: dx=123, dy=164 → sqrt = 205
        let shoulder = (320.0, 140.0);
        let hip = (443.0, 304.0);
        let (_, z, _) = super::torso_to_3d(&ph, shoulder, hip, 0.5)
            .expect("torso depth should be in range");
        assert!((z - 1.0).abs() < 0.02, "leaning torso z={z:.4} expected ~1.0m");
    }

    /// Implausibly small torso (person too far / bad landmarks) → None.
    #[test]
    fn torso_to_3d_rejects_implausible() {
        let ph = test_pinhole();
        // 10px torso → z = 410·0.5/10 = 20.5m, far outside DEPTH_MAX_M → None
        let r = super::torso_to_3d(&ph, (320.0, 235.0), (320.0, 245.0), 0.5);
        assert!(r.is_none(), "20m depth must be rejected as uncalibrated");
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

    // ── metric self-calibration (phase 11.D1) ──

    #[test]
    fn calibrator_below_min_obs_is_none() {
        let mut c = ScaleCalibrator::new();
        for _ in 0..CALIB_MIN_OBS - 1 {
            c.observe(100.0, 1.0);
        }
        assert!(c.scale().is_none(), "needs ≥ CALIB_MIN_OBS observations");
        c.observe(100.0, 1.0);
        assert!(c.scale().is_some());
    }

    #[test]
    fn calibrator_recovers_fx_exactly_from_synthetic_face() {
        // Synthesize a rigid face (real 0.32 m) at several distances through a
        // known fx=410; the calibrator must recover fx to the float.
        let (fx_true, real) = (410.0f32, 0.32f32);
        let mut c = ScaleCalibrator::new();
        for &d in &[0.6f32, 0.9, 1.3, 1.8, 2.4, 3.0] {
            let pixel = fx_true * real / d; // exact projection
            c.observe(pixel, d);
        }
        let fx = c.fx_given_real(real).unwrap();
        assert!((fx - fx_true).abs() < 1e-2, "recovered fx {fx} ≠ {fx_true}");
    }

    #[test]
    fn calibrator_recovers_torso_length_given_fx() {
        // The torso_length_m self-calibration: fx trusted, recover the real size.
        let (fx, real_torso) = (410.0f32, 0.47f32);
        let mut c = ScaleCalibrator::new();
        for &d in &[0.8f32, 1.2, 1.6, 2.0, 2.5] {
            let pixel = fx * real_torso / d;
            c.observe(pixel, d);
        }
        let est = c.real_given_fx(fx).unwrap();
        assert!((est - real_torso).abs() < 1e-3, "recovered torso {est} ≠ {real_torso}");
    }

    #[test]
    fn calibrator_beats_wrong_fx_on_the_calib_sample() {
        // The real 2.5 m YuNet calibration rows (models/calib_3d_yunet.csv):
        // (ground-truth distance, face bbox height px). Face prior 0.32 m.
        let rows: &[(f32, f32)] = &[
            (0.5, 244.91), (1.0, 124.78), (1.5, 76.74), (2.0, 71.4),
            (2.5, 58.82), (2.5, 57.27), (2.5, 59.84),
        ];
        let real = 0.32;
        let mut c = ScaleCalibrator::new();
        for &(d, px) in rows {
            c.observe(px, d);
        }
        let fx_est = c.fx_given_real(real).unwrap();
        let mean_err = |fx: f32| -> f32 {
            let ph = Pinhole { fx, fy: fx, cx_px: 320.0, cy_px: 240.0, width: 640.0, height: 480.0 };
            let s: f32 = rows
                .iter()
                .map(|&(d, px)| (ph.distance_from_size(px, real).unwrap_or(0.0) - d).abs())
                .sum();
            s / rows.len() as f32
        };
        let err_est = mean_err(fx_est);
        let err_bad = mean_err(250.0); // a deliberately wrong focal length
        assert!(err_est < 0.2, "estimated fx should be close to tape, err {err_est}");
        assert!(err_est < err_bad, "estimated {err_est} must beat wrong-fx {err_bad}");
    }
}
