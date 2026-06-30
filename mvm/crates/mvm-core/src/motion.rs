//! Human movement kinematics (phase 9, Track B1) — **pure arithmetic, zero deps**.
//!
//! The body-primary chain already produces 33 BlazePose landmarks per body each
//! frame, but they live transiently inside the pipeline (only wrists + hands are
//! kept downstream). "Movement" is not a new model — it is the **time-derivative
//! of data we already compute**. This module owns the *bounded* per-track history
//! and the scale-invariant velocity primitives over it.
//!
//! Everything here is scalar arithmetic over a fixed-capacity ring (invariant #1).
//! No image, model, or new dependency. Unit-tested entirely on synthetic landmark
//! sequences with no model file present.
//!
//! ## Scale invariance
//!
//! Landmarks arrive in **image pixels**, so a person far from the camera moves
//! fewer pixels for the same real motion. To stay distance-invariant, velocities
//! are reported in **torso-lengths per second**: the centroid displacement is
//! divided by the body's own torso length (shoulder-centre ↔ hip-centre) that
//! frame. A torso is ≈0.5 m, so 1.0 torso/s ≈ 0.5 m/s regardless of zoom.

use std::collections::VecDeque;

/// 33 BlazePose landmarks `[x, y, z, visibility]`. `x,y` in **image pixels**, `z`
/// relative to the hip, `visibility` in `[0,1]`. Same layout as
/// `mvm_posture::Landmarks33`; defined here too so the motion math stays in
/// `mvm-core` with no posture dependency (the pipeline converts at the boundary).
pub type Landmarks33 = [[f32; 4]; 33];

// BlazePose landmark indices used by the kinematics (the only ones we read).
const L_SHOULDER: usize = 11;
const R_SHOULDER: usize = 12;
const L_HIP: usize = 23;
const R_HIP: usize = 24;

/// A landmark is trusted only above this visibility — below it the point is a
/// guess and would inject phantom velocity. Matches the torso-3D gate (0.5).
pub const VIS_GATE: f32 = 0.5;

/// Midpoint of two landmarks `[x,y]`, but only when **both** are visible enough.
fn midpoint(lm: &Landmarks33, a: usize, b: usize) -> Option<[f32; 2]> {
    if lm[a][3] < VIS_GATE || lm[b][3] < VIS_GATE {
        return None;
    }
    Some([(lm[a][0] + lm[b][0]) * 0.5, (lm[a][1] + lm[b][1]) * 0.5])
}

/// Hip centre (mean of left/right hip) — the body's translational centroid.
pub fn hip_center(lm: &Landmarks33) -> Option<[f32; 2]> {
    midpoint(lm, L_HIP, R_HIP)
}

/// Shoulder centre (mean of left/right shoulder).
pub fn shoulder_center(lm: &Landmarks33) -> Option<[f32; 2]> {
    midpoint(lm, L_SHOULDER, R_SHOULDER)
}

/// Torso length in pixels: distance shoulder-centre ↔ hip-centre. The natural
/// per-body scale that makes pixel velocities distance-invariant. `None` when
/// either centre is occluded.
pub fn torso_scale(lm: &Landmarks33) -> Option<f32> {
    let s = shoulder_center(lm)?;
    let h = hip_center(lm)?;
    let d = ((s[0] - h[0]).powi(2) + (s[1] - h[1]).powi(2)).sqrt();
    if d > 1e-3 {
        Some(d)
    } else {
        None
    }
}

/// Bounded ring of recent landmark frames for one track (B1.1).
///
/// Fixed capacity — `push` past capacity drops the oldest frame, so memory is
/// `O(cap)` and never grows (the no-leak invariant). Each entry carries a
/// monotonic timestamp in **seconds** so velocities are real rates, not
/// per-frame deltas (the chain runs at a variable fps).
#[derive(Debug, Clone)]
pub struct LandmarkHistory {
    cap: usize,
    buf: VecDeque<(f64, Landmarks33)>,
}

impl LandmarkHistory {
    /// New history holding at most `cap` frames (`cap` ≥ 1).
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            buf: VecDeque::with_capacity(cap.max(1)),
        }
    }

    /// Append a frame stamped `t_secs`; evict the oldest if at capacity.
    pub fn push(&mut self, t_secs: f64, lm: Landmarks33) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back((t_secs, lm));
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Most recent (timestamp, frame).
    pub fn latest(&self) -> Option<&(f64, Landmarks33)> {
        self.buf.back()
    }

    /// Oldest retained (timestamp, frame).
    pub fn oldest(&self) -> Option<&(f64, Landmarks33)> {
        self.buf.front()
    }

    /// Centroid (hip-centre) speed over the retained window, in **torso-lengths
    /// per second** (B1.2). Uses the oldest and newest frames that both expose a
    /// visible hip-centre; normalises by the latest valid torso length so the
    /// rate is distance-invariant. `None` when fewer than two usable frames, a
    /// non-positive time span, or no measurable torso.
    pub fn centroid_speed(&self) -> Option<f32> {
        // First and last frames with a usable hip-centre.
        let first = self
            .buf
            .iter()
            .find_map(|(t, lm)| hip_center(lm).map(|c| (*t, c)));
        let last = self
            .buf
            .iter()
            .rev()
            .find_map(|(t, lm)| hip_center(lm).map(|c| (*t, c)));
        let ((t0, c0), (t1, c1)) = (first?, last?);
        let dt = (t1 - t0) as f32;
        if dt <= 0.0 {
            return None;
        }
        // Torso scale from the latest valid frame (fall back to any valid one).
        let scale = self
            .buf
            .iter()
            .rev()
            .find_map(|(_, lm)| torso_scale(lm))?;
        let disp = ((c1[0] - c0[0]).powi(2) + (c1[1] - c0[1]).powi(2)).sqrt();
        Some((disp / scale) / dt)
    }
}

/// Coarse locomotion state of a tracked body (B1.3). The headline "movement"
/// signal: is this person holding still or travelling across the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionState {
    /// Not enough evidence committed yet (startup).
    Unknown,
    /// Centroid effectively still.
    Idle,
    /// Centroid translating across the scene.
    Walking,
}

impl MotionState {
    /// Lowercase tag for serialization / Scene JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            MotionState::Unknown => "unknown",
            MotionState::Idle => "idle",
            MotionState::Walking => "walking",
        }
    }
}

/// Hysteresis + dwell state machine over centroid speed (B1.3) — **pure
/// arithmetic, no allocation**.
///
/// Two guards kill the per-frame chatter a single threshold would produce:
/// - **Hysteresis (dead zone):** `idle_below < walk_above`. A speed *between*
///   the two does not change the committed state — you must clearly cross into
///   the other regime to flip, so a body hovering near one threshold stays put.
/// - **Dwell:** the new regime must hold for `dwell` *consecutive* updates
///   before it commits. A one-frame velocity spike (occlusion jitter, a single
///   bad landmark) never flips the state.
#[derive(Debug, Clone)]
pub struct MotionClassifier {
    state: MotionState,
    idle_below: f32,
    walk_above: f32,
    dwell: u32,
    candidate: MotionState,
    streak: u32,
}

impl Default for MotionClassifier {
    /// Defaults from the phase-9 plan: idle below 0.05 torso/s, walking above
    /// 0.3 torso/s, commit after 3 consecutive frames.
    fn default() -> Self {
        Self::new(0.05, 0.3, 3)
    }
}

impl MotionClassifier {
    /// `idle_below` / `walk_above` are centroid-speed thresholds in torso/s
    /// (must satisfy `idle_below ≤ walk_above` for a sane dead zone); `dwell` is
    /// the consecutive-frame count required to commit a flip (clamped to ≥1).
    pub fn new(idle_below: f32, walk_above: f32, dwell: u32) -> Self {
        Self {
            state: MotionState::Unknown,
            idle_below,
            walk_above: walk_above.max(idle_below),
            dwell: dwell.max(1),
            candidate: MotionState::Unknown,
            streak: 0,
        }
    }

    pub fn state(&self) -> MotionState {
        self.state
    }

    /// Feed one centroid speed (torso/s); returns the committed state.
    ///
    /// A speed in the dead zone (between the thresholds) is *ambiguous*: it
    /// breaks any in-progress streak (so evidence must be consecutive) but never
    /// flips the committed state.
    pub fn update(&mut self, speed: f32) -> MotionState {
        let bucket = if speed >= self.walk_above {
            Some(MotionState::Walking)
        } else if speed <= self.idle_below {
            Some(MotionState::Idle)
        } else {
            None // dead zone → ambiguous
        };

        match bucket {
            // Already in this regime: nothing to prove, clear any pending streak.
            Some(b) if b == self.state => {
                self.candidate = self.state;
                self.streak = 0;
            }
            // A definite, different regime: accumulate consecutive evidence.
            Some(b) => {
                if b == self.candidate {
                    self.streak += 1;
                } else {
                    self.candidate = b;
                    self.streak = 1;
                }
                if self.streak >= self.dwell {
                    self.state = b;
                    self.streak = 0;
                }
            }
            // Ambiguous frame: interrupt the streak, hold the committed state.
            None => {
                self.candidate = self.state;
                self.streak = 0;
            }
        }
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic landmark frame with shoulders, hips (all visible) placed
    /// so that shoulder-centre = (cx, cy_sh) and hip-centre = (cx, cy_hip).
    /// All other landmarks are zeroed/invisible (irrelevant to these tests).
    fn frame(cx: f32, cy_sh: f32, cy_hip: f32) -> Landmarks33 {
        let mut lm = [[0.0f32; 4]; 33];
        // shoulders span ±20px around cx at height cy_sh, fully visible
        lm[L_SHOULDER] = [cx - 20.0, cy_sh, 0.0, 1.0];
        lm[R_SHOULDER] = [cx + 20.0, cy_sh, 0.0, 1.0];
        // hips span ±15px around cx at height cy_hip
        lm[L_HIP] = [cx - 15.0, cy_hip, 0.0, 1.0];
        lm[R_HIP] = [cx + 15.0, cy_hip, 0.0, 1.0];
        lm
    }

    // ── geometry primitives ────────────────────────────────────────────────

    #[test]
    fn hip_center_is_midpoint_of_hips() {
        let f = frame(100.0, 100.0, 200.0);
        let h = hip_center(&f).unwrap();
        assert!((h[0] - 100.0).abs() < 1e-4 && (h[1] - 200.0).abs() < 1e-4);
    }

    #[test]
    fn torso_scale_is_shoulder_to_hip_distance() {
        // shoulder-centre (100,100), hip-centre (100,200) → vertical torso 100px.
        let f = frame(100.0, 100.0, 200.0);
        assert!((torso_scale(&f).unwrap() - 100.0).abs() < 1e-3);
    }

    #[test]
    fn occluded_hip_yields_no_center() {
        let mut f = frame(100.0, 100.0, 200.0);
        f[L_HIP][3] = 0.1; // below VIS_GATE
        assert!(hip_center(&f).is_none());
        assert!(torso_scale(&f).is_none(), "no hip → no torso scale");
    }

    // ── ring buffer (B1.1) ─────────────────────────────────────────────────

    #[test]
    fn ring_is_bounded_and_drops_oldest() {
        let mut h = LandmarkHistory::new(3);
        for i in 0..10 {
            h.push(i as f64, frame(i as f32, 0.0, 100.0));
        }
        assert_eq!(h.len(), 3, "never exceeds capacity (no growth)");
        // oldest retained is frame 7, newest is frame 9
        assert!((h.oldest().unwrap().0 - 7.0).abs() < 1e-9);
        assert!((h.latest().unwrap().0 - 9.0).abs() < 1e-9);
    }

    #[test]
    fn ring_zero_capacity_clamped_to_one() {
        let mut h = LandmarkHistory::new(0);
        h.push(0.0, frame(0.0, 0.0, 100.0));
        h.push(1.0, frame(5.0, 0.0, 100.0));
        assert_eq!(h.len(), 1);
        assert!((h.latest().unwrap().0 - 1.0).abs() < 1e-9);
    }

    // ── centroid speed (B1.2) ──────────────────────────────────────────────

    #[test]
    fn centroid_speed_none_with_single_frame() {
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, frame(100.0, 100.0, 200.0));
        assert!(h.centroid_speed().is_none(), "need ≥2 usable frames");
    }

    #[test]
    fn centroid_speed_is_torso_lengths_per_second() {
        // Torso = 100px. Hip moves 50px in x over 0.5s → 50/100 torso in 0.5s
        // = 1.0 torso/s.
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, frame(100.0, 100.0, 200.0));
        h.push(0.5, frame(150.0, 100.0, 200.0));
        let v = h.centroid_speed().unwrap();
        assert!((v - 1.0).abs() < 1e-3, "expected 1.0 torso/s, got {v}");
    }

    #[test]
    fn centroid_speed_zero_when_still() {
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, frame(100.0, 100.0, 200.0));
        h.push(0.3, frame(100.0, 100.0, 200.0));
        assert!(h.centroid_speed().unwrap().abs() < 1e-4);
    }

    #[test]
    fn centroid_speed_distance_invariant() {
        // A body twice as far (half the pixel size AND half the pixel motion)
        // reports the SAME torso/s — the whole point of torso-normalisation.
        let mut near = LandmarkHistory::new(8);
        near.push(0.0, frame(100.0, 100.0, 200.0)); // torso 100px
        near.push(1.0, frame(150.0, 100.0, 200.0)); // moved 50px
        let mut far = LandmarkHistory::new(8);
        far.push(0.0, frame(100.0, 100.0, 150.0)); // torso 50px
        far.push(1.0, frame(125.0, 100.0, 150.0)); // moved 25px
        let vn = near.centroid_speed().unwrap();
        let vf = far.centroid_speed().unwrap();
        assert!((vn - vf).abs() < 1e-3, "near {vn} vs far {vf} should match");
    }

    #[test]
    fn centroid_speed_none_on_zero_timespan() {
        let mut h = LandmarkHistory::new(8);
        h.push(2.0, frame(100.0, 100.0, 200.0));
        h.push(2.0, frame(150.0, 100.0, 200.0)); // same timestamp
        assert!(h.centroid_speed().is_none());
    }

    #[test]
    fn centroid_speed_skips_occluded_frames() {
        // Middle frame has no visible hips; speed uses the visible endpoints.
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, frame(100.0, 100.0, 200.0));
        let mut blind = frame(999.0, 100.0, 200.0);
        blind[L_HIP][3] = 0.0;
        blind[R_HIP][3] = 0.0;
        h.push(0.5, blind);
        h.push(1.0, frame(200.0, 100.0, 200.0)); // 100px over 1s, torso 100px
        let v = h.centroid_speed().unwrap();
        assert!((v - 1.0).abs() < 1e-3, "endpoints only → 1.0 torso/s, got {v}");
    }

    // ── motion-state machine (B1.3) ────────────────────────────────────────

    #[test]
    fn motion_starts_unknown() {
        let c = MotionClassifier::default();
        assert_eq!(c.state(), MotionState::Unknown);
    }

    #[test]
    fn motion_commits_walking_after_dwell() {
        // dwell=3: a clear walking speed must hold for 3 consecutive frames
        // before it commits.
        let mut c = MotionClassifier::new(0.05, 0.3, 3);
        assert_eq!(c.update(1.0), MotionState::Unknown, "1st frame: not yet");
        assert_eq!(c.update(1.0), MotionState::Unknown, "2nd frame: not yet");
        assert_eq!(c.update(1.0), MotionState::Walking, "3rd frame: commits");
    }

    #[test]
    fn motion_commits_idle_after_dwell() {
        let mut c = MotionClassifier::new(0.05, 0.3, 3);
        assert_eq!(c.update(0.0), MotionState::Unknown);
        assert_eq!(c.update(0.0), MotionState::Unknown);
        assert_eq!(c.update(0.0), MotionState::Idle);
    }

    #[test]
    fn motion_dwell_one_commits_immediately() {
        let mut c = MotionClassifier::new(0.05, 0.3, 1);
        assert_eq!(c.update(1.0), MotionState::Walking, "dwell=1 → instant");
    }

    #[test]
    fn motion_zero_dwell_clamped_to_one() {
        // dwell is clamped to ≥1 so a flip always needs at least one frame.
        let mut c = MotionClassifier::new(0.05, 0.3, 0);
        assert_eq!(c.update(1.0), MotionState::Walking);
    }

    #[test]
    fn motion_single_spike_does_not_flip() {
        // Commit idle, then a one-frame walking spike must NOT flip the state.
        let mut c = MotionClassifier::new(0.05, 0.3, 3);
        for _ in 0..3 {
            c.update(0.0);
        }
        assert_eq!(c.state(), MotionState::Idle);
        assert_eq!(c.update(1.0), MotionState::Idle, "lone spike ignored");
        // and a return to idle keeps it idle (streak was for walking, now reset)
        assert_eq!(c.update(0.0), MotionState::Idle);
    }

    #[test]
    fn motion_dead_zone_holds_committed_state() {
        // Between idle_below(0.05) and walk_above(0.3) is ambiguous: it never
        // changes the committed state.
        let mut c = MotionClassifier::new(0.05, 0.3, 2);
        c.update(1.0);
        assert_eq!(c.update(1.0), MotionState::Walking);
        // now feed dead-zone speeds forever — stays walking
        for _ in 0..10 {
            assert_eq!(c.update(0.15), MotionState::Walking, "dead zone holds");
        }
    }

    #[test]
    fn motion_dead_zone_interrupts_streak() {
        // A dead-zone frame breaks an in-progress flip: evidence must be
        // consecutive.
        let mut c = MotionClassifier::new(0.05, 0.3, 3);
        for _ in 0..3 {
            c.update(0.0);
        }
        assert_eq!(c.state(), MotionState::Idle);
        c.update(1.0); // walking streak = 1
        c.update(1.0); // walking streak = 2
        c.update(0.15); // dead zone → streak reset, still idle
        assert_eq!(c.state(), MotionState::Idle);
        // need a fresh full dwell of walking to flip
        c.update(1.0);
        c.update(1.0);
        assert_eq!(c.update(1.0), MotionState::Walking, "fresh streak commits");
    }

    #[test]
    fn motion_idle_walk_idle_round_trip() {
        let mut c = MotionClassifier::new(0.05, 0.3, 2);
        c.update(0.0);
        assert_eq!(c.update(0.0), MotionState::Idle);
        c.update(1.0);
        assert_eq!(c.update(1.0), MotionState::Walking);
        c.update(0.0);
        assert_eq!(c.update(0.0), MotionState::Idle, "flips back after dwell");
    }

    #[test]
    fn motion_thresholds_are_inclusive() {
        // Speed exactly at walk_above counts as walking; exactly at idle_below
        // counts as idle.
        let mut c = MotionClassifier::new(0.05, 0.3, 1);
        assert_eq!(c.update(0.3), MotionState::Walking, "== walk_above → walk");
        let mut d = MotionClassifier::new(0.05, 0.3, 1);
        assert_eq!(d.update(0.05), MotionState::Idle, "== idle_below → idle");
    }

    #[test]
    fn motion_bad_thresholds_collapse_dead_zone() {
        // walk_above < idle_below is clamped so walk_above ≥ idle_below; the
        // dead zone collapses but the machine stays sane (no panic, commits).
        let mut c = MotionClassifier::new(0.3, 0.1, 1);
        // walk_above was clamped up to 0.3; 0.5 ≥ 0.3 → walking
        assert_eq!(c.update(0.5), MotionState::Walking);
        assert_eq!(c.update(0.0), MotionState::Idle);
    }

    #[test]
    fn motion_state_as_str_round_trips() {
        assert_eq!(MotionState::Unknown.as_str(), "unknown");
        assert_eq!(MotionState::Idle.as_str(), "idle");
        assert_eq!(MotionState::Walking.as_str(), "walking");
    }
}
