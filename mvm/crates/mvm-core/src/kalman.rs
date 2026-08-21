//! Kalman filter for bounding-box tracking.
//!
//! The canonical 8D state vector `[cx, cy, w, h, vcx, vcy, vw, vh]` with diagonal
//! Q/R/P decomposes into 4 independent scalar 2D filters — one per coordinate.
//! This eliminates matrix inversion entirely (~50 lines, no linalg crate).
//!
//! ## Tuning
//!
//! The filter has three sets of parameters — all exposed as named constants below
//! so they serve as documented tuning points. Future Config fields can replace them.
//!
//! | Constant | Meaning | Tune when |
//! |----------|---------|-----------|
//! | `Q_POS`  | Process noise on position | Object moves erratically |
//! | `Q_VEL`  | Process noise on velocity | Velocity estimate too stiff/loose |
//! | `R_POS`  | Measurement noise for cx/cy | Detector is noisy in position |
//! | `R_SIZE` | Measurement noise for w/h  | Detector is noisy in size (hair/pose) |
//! | `P_POS_INIT` | Initial position uncertainty | Detector is already accurate at birth |
//! | `P_VEL_INIT` | Initial velocity uncertainty | Object already has known velocity |

use crate::types::BBox;

// ── Tuning points ─────────────────────────────────────────────────────────────

/// Position process noise at the legacy nominal tick. Raise for erratic movement.
const Q_POS: f32 = 1.0;
/// Velocity process noise at the legacy nominal tick. Small = smooth velocity estimate.
const Q_VEL: f32 = 0.01;
/// Measurement noise for centroid (cx, cy). Lower = trust detector more.
const R_POS: f32 = 1.0;
/// Measurement noise for size (w, h). Higher than R_POS: size jitters with pose.
const R_SIZE: f32 = 10.0;
/// Initial position covariance (high = uncertain at birth, rapidly corrected).
const P_POS_INIT: f32 = 10.0;
/// Initial velocity covariance (high = velocity is unknown at birth).
const P_VEL_INIT: f32 = 100.0;
/// Reference cadence of the original tuned per-tick filter. State velocity is
/// now units/second, so legacy velocity covariance/noise is converted from this
/// reference rather than silently changing the filter's nominal behaviour.
const NOMINAL_DT_SECS: f32 = 1.0 / 30.0;

// ── Scalar 2D Kalman filter ───────────────────────────────────────────────────

/// Scalar Kalman filter: state = [pos, vel], observation = pos.
/// F = [[1,dt],[0,1]] (constant-velocity), H = [1,0].
#[derive(Debug, Clone, Copy)]
struct ScalarKF {
    pos: f32,
    vel: f32,
    p_pp: f32, // cov(pos, pos)
    p_pv: f32, // cov(pos, vel)  [symmetric with p_vp]
    p_vv: f32, // cov(vel, vel)
    q_pos: f32,
    q_vel: f32,
    r: f32,
}

impl ScalarKF {
    fn new(pos: f32, q_pos: f32, q_vel: f32, r: f32) -> Self {
        Self {
            pos,
            vel: 0.0,
            p_pp: P_POS_INIT,
            p_pv: 0.0,
            p_vv: P_VEL_INIT / (NOMINAL_DT_SECS * NOMINAL_DT_SECS),
            q_pos: q_pos / NOMINAL_DT_SECS,
            q_vel: q_vel / (NOMINAL_DT_SECS * NOMINAL_DT_SECS * NOMINAL_DT_SECS),
            r,
        }
    }

    /// Predict by `dt_secs`. `q_pos` is a continuous position random-walk
    /// density and `q_vel` a white-acceleration density, integrated analytically
    /// over the interval. The caller owns validation/bounding of elapsed time.
    fn predict(&mut self, dt_secs: f32) -> f32 {
        debug_assert!(dt_secs.is_finite() && dt_secs >= 0.0);
        let dt2 = dt_secs * dt_secs;
        let dt3 = dt2 * dt_secs;
        self.pos += self.vel * dt_secs;
        // P_pred = F P F^T + Q(dt), expanded for F = [[1,dt],[0,1]].
        let p_pp = self.p_pp
            + 2.0 * dt_secs * self.p_pv
            + dt2 * self.p_vv
            + self.q_pos * dt_secs
            + self.q_vel * dt3 / 3.0;
        let p_pv = self.p_pv + dt_secs * self.p_vv + self.q_vel * dt2 / 2.0;
        let p_vv = self.p_vv + self.q_vel * dt_secs;
        self.p_pp = p_pp;
        self.p_pv = p_pv;
        self.p_vv = p_vv;
        self.pos
    }

    /// Update with observed position. Mutates state in place.
    fn update(&mut self, observed: f32) {
        let s = self.p_pp + self.r; // innovation covariance (scalar)
        let k_pos = self.p_pp / s; // Kalman gain for pos
        let k_vel = self.p_pv / s; // Kalman gain for vel
        let y = observed - self.pos; // innovation
        self.pos += k_pos * y;
        self.vel += k_vel * y;
        // P = (I - K H) P
        let p_pp = (1.0 - k_pos) * self.p_pp;
        let p_pv = (1.0 - k_pos) * self.p_pv;
        let p_vv = -k_vel * self.p_pv + self.p_vv;
        self.p_pp = p_pp;
        self.p_pv = p_pv;
        self.p_vv = p_vv;
    }

    /// Position variance — proxy for uncertainty (grows on missed frames).
    fn p_pos(&self) -> f32 {
        self.p_pp
    }
}

// ── KalmanBBox ────────────────────────────────────────────────────────────────

/// Kalman filter for a bounding box. Operates in (cx, cy, w, h) space,
/// decomposed into 4 independent scalar filters.
///
/// Usage in the tracker (State-Gating pattern):
/// 1. `predict(dt_secs)` before IoU matching → use predicted bbox for matching.
/// 2. `update(observed)` after a match → smoothed bbox.
/// 3. On missed frames: `predict(dt_secs)` advances the bbox by inertia; no `update`.
#[derive(Debug, Clone, Copy)]
pub struct KalmanBBox {
    cx: ScalarKF,
    cy: ScalarKF,
    w: ScalarKF,
    h: ScalarKF,
}

impl KalmanBBox {
    pub fn new(bbox: BBox) -> Self {
        let (cx, cy, w, h) = to_cwh(bbox);
        Self {
            cx: ScalarKF::new(cx, Q_POS, Q_VEL, R_POS),
            cy: ScalarKF::new(cy, Q_POS, Q_VEL, R_POS),
            w: ScalarKF::new(w, Q_POS, Q_VEL, R_SIZE),
            h: ScalarKF::new(h, Q_POS, Q_VEL, R_SIZE),
        }
    }

    /// Advance the filter by a bounded physical interval. Returns the predicted bbox.
    /// Call this **before** IoU matching (State-Gating).
    pub fn predict(&mut self, dt_secs: f32) -> BBox {
        assert!(dt_secs.is_finite() && dt_secs >= 0.0, "invalid Kalman dt");
        let cx = self.cx.predict(dt_secs);
        let cy = self.cy.predict(dt_secs);
        let w = self.w.predict(dt_secs).max(0.0);
        let h = self.h.predict(dt_secs).max(0.0);
        from_cwh(cx, cy, w, h)
    }

    /// Correct the prediction with an observation. Call after a successful match.
    pub fn update(&mut self, observed: BBox) {
        let (cx, cy, w, h) = to_cwh(observed);
        self.cx.update(cx);
        self.cy.update(cy);
        self.w.update(w);
        self.h.update(h);
    }

    /// Current smoothed bbox (reads state without mutating).
    pub fn smoothed_bbox(&self) -> BBox {
        from_cwh(
            self.cx.pos,
            self.cy.pos,
            self.w.pos.max(0.0),
            self.h.pos.max(0.0),
        )
    }

    /// Sum of cx/cy position variances — proxy for track uncertainty.
    pub fn position_uncertainty(&self) -> f32 {
        self.cx.p_pos() + self.cy.p_pos()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn to_cwh(b: BBox) -> (f32, f32, f32, f32) {
    (
        (b.x1 + b.x2) * 0.5,
        (b.y1 + b.y2) * 0.5,
        b.x2 - b.x1,
        b.y2 - b.y1,
    )
}

fn from_cwh(cx: f32, cy: f32, w: f32, h: f32) -> BBox {
    BBox {
        x1: cx - w * 0.5,
        y1: cy - h * 0.5,
        x2: cx + w * 0.5,
        y2: cy + h * 0.5,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 30.0;

    fn bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { x1, y1, x2, y2 }
    }

    /// Stationary object: after convergence, predicted bbox ≈ observed.
    #[test]
    fn kalman_predict_stationary_stays_put() {
        let b = bbox(0.3, 0.3, 0.7, 0.7);
        let mut kf = KalmanBBox::new(b);
        for _ in 0..20 {
            kf.predict(DT);
            kf.update(b);
        }
        let pred = kf.predict(DT);
        assert!((pred.x1 - 0.3).abs() < 0.01, "x1 drifted: {:.4}", pred.x1);
        assert!((pred.x2 - 0.7).abs() < 0.01, "x2 drifted: {:.4}", pred.x2);
    }

    /// Moving object: predicted bbox extrapolates beyond last observation.
    #[test]
    fn kalman_predict_moving_extrapolates() {
        let mut kf = KalmanBBox::new(bbox(0.1, 0.3, 0.3, 0.7));
        for i in 0..15u32 {
            let x1 = 0.1 + i as f32 * 0.02;
            kf.predict(DT);
            kf.update(bbox(x1, 0.3, x1 + 0.2, 0.7));
        }
        let last_x1 = 0.1 + 14.0 * 0.02; // 0.38
        let pred = kf.predict(DT);
        assert!(
            pred.x1 > last_x1,
            "pred.x1={:.4} not extrapolated past {:.4}",
            pred.x1,
            last_x1
        );
    }

    /// Update pulls state toward the observation.
    #[test]
    fn kalman_update_corrects_prediction() {
        let mut kf = KalmanBBox::new(bbox(0.0, 0.0, 0.2, 0.2));
        kf.predict(DT);
        // Observation far from init
        kf.update(bbox(0.5, 0.5, 0.7, 0.7));
        let smoothed = kf.smoothed_bbox();
        // cx should be between 0.1 (init) and 0.6 (obs), pulled toward obs
        assert!(
            smoothed.x1 > 0.1,
            "not pulled toward observation: x1={:.4}",
            smoothed.x1
        );
    }

    /// Uncertainty grows when no observation arrives (missed frames).
    #[test]
    fn kalman_missed_frames_grow_uncertainty() {
        let b = bbox(0.3, 0.3, 0.7, 0.7);
        let mut kf = KalmanBBox::new(b);
        // Stabilize filter
        for _ in 0..20 {
            kf.predict(DT);
            kf.update(b);
        }
        let u_stable = kf.position_uncertainty();
        // Predict without updates (simulate occluded frames)
        for _ in 0..15 {
            kf.predict(DT);
        }
        let u_missed = kf.position_uncertainty();
        assert!(
            u_missed > u_stable,
            "uncertainty should grow on missed frames: {:.4} vs {:.4}",
            u_missed,
            u_stable
        );
    }

    #[test]
    fn irregular_dt_tracks_constant_velocity_in_seconds() {
        let mut kf = KalmanBBox::new(bbox(0.10, 0.40, 0.30, 0.60));
        let mut elapsed = 0.0_f32;
        for dt in [0.04, 0.11, 0.03, 0.20, 0.07, 0.15, 0.05, 0.10] {
            elapsed += dt;
            kf.predict(dt);
            let x1 = 0.10 + 0.20 * elapsed;
            kf.update(bbox(x1, 0.40, x1 + 0.20, 0.60));
        }
        let before = kf.smoothed_bbox();
        let predicted = kf.predict(0.25);
        let displacement = predicted.x1 - before.x1;
        assert!(
            (displacement - 0.05).abs() < 0.035,
            "0.20 units/s over 0.25 s should move about 0.05, got {displacement}"
        );
    }

    #[test]
    fn delayed_prediction_grows_uncertainty_more_than_short_tick() {
        let b = bbox(0.3, 0.3, 0.7, 0.7);
        let mut short = KalmanBBox::new(b);
        for _ in 0..20 {
            short.predict(DT);
            short.update(b);
        }
        let mut delayed = short;
        short.predict(0.05);
        delayed.predict(0.50);
        assert!(
            delayed.position_uncertainty() > short.position_uncertainty(),
            "longer physical gap must grow uncertainty more"
        );
    }

    /// Tracker uses predicted bbox for IoU — a consistently moving object
    /// should stay as one track, not spawn new ones.
    #[test]
    fn tracker_uses_predicted_bbox_for_iou() {
        use crate::tracker::PersonTracker;
        use crate::types::Detection;

        let mut tracker = PersonTracker::new();
        // Object drifting right across 5 frames
        for i in 0..5u32 {
            let x1 = 0.10 + i as f32 * 0.03;
            tracker.tick(&[Detection::bare(x1, 0.3, x1 + 0.20, 0.7)]);
        }
        assert_eq!(
            tracker.track_count(),
            1,
            "moving object should stay one track"
        );
    }
}
