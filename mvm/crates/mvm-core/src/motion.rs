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
    Moving,
}

impl MotionState {
    /// Lowercase tag for serialization / Scene JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            MotionState::Unknown => "unknown",
            MotionState::Idle => "idle",
            MotionState::Moving => "moving",
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
    /// Defaults from the phase-9 plan: idle below 0.05 torso/s, moving above
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
            Some(MotionState::Moving)
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

// BlazePose wrist indices — the only arm joints the gesture math reads.
const L_WRIST: usize = 15;
const R_WRIST: usize = 16;

/// Which arm a gesture query is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    fn wrist_idx(self) -> usize {
        match self {
            Side::Left => L_WRIST,
            Side::Right => R_WRIST,
        }
    }
}

/// Wrist position `[x,y]` for one arm, gated on visibility.
pub fn wrist_pos(lm: &Landmarks33, side: Side) -> Option<[f32; 2]> {
    let w = &lm[side.wrist_idx()];
    if w[3] < VIS_GATE {
        return None;
    }
    Some([w[0], w[1]])
}

/// Lateral arm extension (B1.4): how far the wrist sits **off the torso axis**,
/// in torso-lengths. Measured as the perpendicular distance of the wrist from
/// the shoulder-centre↔hip-centre line, normalised by torso length — so it is
/// **orientation-invariant** in the image plane (a body rotated 90° reads the
/// same) and **distance-invariant** (torso-normalised).
///
/// A hanging arm lies *along* the axis → ~0. An outstretched arm sits far to the
/// side/front → ≈ arm length in torso units. This direction-based measure beats
/// raw wrist↔shoulder distance, which barely changes between hanging and
/// reaching (the arm is roughly rigid — only its *direction* changes).
pub fn arm_reach(lm: &Landmarks33, side: Side) -> Option<f32> {
    let sc = shoulder_center(lm)?;
    let hc = hip_center(lm)?;
    let w = wrist_pos(lm, side)?;
    // Torso axis (hip→shoulder) and its length (the normaliser).
    let ax = [sc[0] - hc[0], sc[1] - hc[1]];
    let len = (ax[0] * ax[0] + ax[1] * ax[1]).sqrt();
    if len < 1e-3 {
        return None;
    }
    let unit = [ax[0] / len, ax[1] / len];
    // Wrist relative to the shoulder (the arm's anchor).
    let rel = [w[0] - sc[0], w[1] - sc[1]];
    // Perpendicular component magnitude = |rel − (rel·unit)·unit|.
    let along = rel[0] * unit[0] + rel[1] * unit[1];
    let perp = [rel[0] - along * unit[0], rel[1] - along * unit[1]];
    let perp_len = (perp[0] * perp[0] + perp[1] * perp[1]).sqrt();
    Some(perp_len / len)
}

/// A coarse single-arm gesture event (B1.4) — feeds `activities` (8.F).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmGesture {
    /// Arm resting / not doing anything notable.
    None,
    /// Arm extended outward and steady — reaching toward something.
    Reaching,
    /// Hand moving fast — waving / gesturing.
    Waving,
}

impl ArmGesture {
    pub fn as_str(self) -> &'static str {
        match self {
            ArmGesture::None => "none",
            ArmGesture::Reaching => "reaching",
            ArmGesture::Waving => "waving",
        }
    }
}

/// Thresholds for [`LandmarkHistory::arm_gesture`] (torso-normalised units).
#[derive(Debug, Clone, Copy)]
pub struct GestureThresholds {
    /// Lateral extension (torso) at/above which a steady arm counts as reaching.
    pub reach_above: f32,
    /// Wrist speed (torso/s) at/above which the arm counts as waving.
    pub wave_above: f32,
}

impl Default for GestureThresholds {
    /// Reach ≥ 0.8 torso lateral; wave ≥ 1.5 torso/s wrist speed.
    fn default() -> Self {
        Self {
            reach_above: 0.8,
            wave_above: 1.5,
        }
    }
}

impl LandmarkHistory {
    /// Wrist speed for one arm over the retained window, in **torso-lengths per
    /// second** (B1.4) — same scale-invariant treatment as [`Self::centroid_speed`]
    /// but tracking the wrist. Uses the oldest/newest frames with a visible wrist,
    /// normalised by the latest valid torso length. `None` when fewer than two
    /// usable frames, a non-positive span, or no measurable torso.
    pub fn wrist_speed(&self, side: Side) -> Option<f32> {
        let first = self
            .buf
            .iter()
            .find_map(|(t, lm)| wrist_pos(lm, side).map(|w| (*t, w)));
        let last = self
            .buf
            .iter()
            .rev()
            .find_map(|(t, lm)| wrist_pos(lm, side).map(|w| (*t, w)));
        let ((t0, w0), (t1, w1)) = (first?, last?);
        let dt = (t1 - t0) as f32;
        if dt <= 0.0 {
            return None;
        }
        let scale = self.buf.iter().rev().find_map(|(_, lm)| torso_scale(lm))?;
        let disp = ((w1[0] - w0[0]).powi(2) + (w1[1] - w0[1]).powi(2)).sqrt();
        Some((disp / scale) / dt)
    }

    /// Classify one arm's gesture from the window (B1.4). **Speed wins:** a fast
    /// wrist is *waving* even if also extended; an extended-but-steady arm is
    /// *reaching*; otherwise `None`. Reach is read from the latest frame, speed
    /// from the window. Returns `None` (the variant) when geometry is unavailable.
    pub fn arm_gesture(&self, side: Side, th: &GestureThresholds) -> ArmGesture {
        if let Some(v) = self.wrist_speed(side) {
            if v >= th.wave_above {
                return ArmGesture::Waving;
            }
        }
        if let Some((_, lm)) = self.latest() {
            if let Some(r) = arm_reach(lm, side) {
                if r >= th.reach_above {
                    return ArmGesture::Reaching;
                }
            }
        }
        ArmGesture::None
    }
}

/// Landmark-ring capacity per track (phase 9.8). ≈0.5 s of history at typical
/// fps — enough for a stable centroid-speed window, small enough that the ring
/// (and the track that owns it) stays O(1) in memory.
pub const MOTION_RING_CAP: usize = 16;

/// Per-track motion (phase 9.8): the bounded landmark ring **owned by the track**
/// plus the kinematics derived from it. Persisting this on `PersonTrack` removes
/// the IoU round-trip that previously re-matched pose data to a track after the
/// fact — motion is now read straight off the track it belongs to.
///
/// Each [`observe`](Self::observe) folds one landmark frame in and refreshes the
/// cached outputs (`state`, `speed`, per-arm gestures). Bounded by construction:
/// the ring evicts its oldest frame past capacity, so memory never grows.
#[derive(Debug, Clone)]
pub struct MotionTrack {
    history: LandmarkHistory,
    classifier: MotionClassifier,
    state: MotionState,
    speed: f32,
    left: ArmGesture,
    right: ArmGesture,
}

impl Default for MotionTrack {
    fn default() -> Self {
        Self::new()
    }
}

impl MotionTrack {
    pub fn new() -> Self {
        Self {
            history: LandmarkHistory::new(MOTION_RING_CAP),
            classifier: MotionClassifier::default(),
            state: MotionState::Unknown,
            speed: 0.0,
            left: ArmGesture::None,
            right: ArmGesture::None,
        }
    }

    /// Fold one landmark frame (image **pixels**) stamped `t_secs` (seconds) into
    /// the ring and refresh the derived kinematics. Velocity stays `0` and state
    /// `Unknown` until two usable frames exist.
    pub fn observe(&mut self, lm: Landmarks33, t_secs: f64) {
        self.history.push(t_secs, lm);
        if let Some(v) = self.history.centroid_speed() {
            self.speed = v;
            self.state = self.classifier.update(v);
        }
        let th = GestureThresholds::default();
        self.left = self.history.arm_gesture(Side::Left, &th);
        self.right = self.history.arm_gesture(Side::Right, &th);
    }

    /// Committed locomotion state (idle / moving / unknown).
    pub fn state(&self) -> MotionState {
        self.state
    }

    /// Last centroid speed in torso-lengths per second.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    pub fn left_gesture(&self) -> ArmGesture {
        self.left
    }

    pub fn right_gesture(&self) -> ArmGesture {
        self.right
    }

    /// Frames currently retained (≤ [`MOTION_RING_CAP`]) — the no-growth invariant.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
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
    fn motion_commits_moving_after_dwell() {
        // dwell=3: a clear moving speed must hold for 3 consecutive frames
        // before it commits.
        let mut c = MotionClassifier::new(0.05, 0.3, 3);
        assert_eq!(c.update(1.0), MotionState::Unknown, "1st frame: not yet");
        assert_eq!(c.update(1.0), MotionState::Unknown, "2nd frame: not yet");
        assert_eq!(c.update(1.0), MotionState::Moving, "3rd frame: commits");
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
        assert_eq!(c.update(1.0), MotionState::Moving, "dwell=1 → instant");
    }

    #[test]
    fn motion_zero_dwell_clamped_to_one() {
        // dwell is clamped to ≥1 so a flip always needs at least one frame.
        let mut c = MotionClassifier::new(0.05, 0.3, 0);
        assert_eq!(c.update(1.0), MotionState::Moving);
    }

    #[test]
    fn motion_single_spike_does_not_flip() {
        // Commit idle, then a one-frame moving spike must NOT flip the state.
        let mut c = MotionClassifier::new(0.05, 0.3, 3);
        for _ in 0..3 {
            c.update(0.0);
        }
        assert_eq!(c.state(), MotionState::Idle);
        assert_eq!(c.update(1.0), MotionState::Idle, "lone spike ignored");
        // and a return to idle keeps it idle (streak was for moving, now reset)
        assert_eq!(c.update(0.0), MotionState::Idle);
    }

    #[test]
    fn motion_dead_zone_holds_committed_state() {
        // Between idle_below(0.05) and walk_above(0.3) is ambiguous: it never
        // changes the committed state.
        let mut c = MotionClassifier::new(0.05, 0.3, 2);
        c.update(1.0);
        assert_eq!(c.update(1.0), MotionState::Moving);
        // now feed dead-zone speeds forever — stays moving
        for _ in 0..10 {
            assert_eq!(c.update(0.15), MotionState::Moving, "dead zone holds");
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
        c.update(1.0); // moving streak = 1
        c.update(1.0); // moving streak = 2
        c.update(0.15); // dead zone → streak reset, still idle
        assert_eq!(c.state(), MotionState::Idle);
        // need a fresh full dwell of moving to flip
        c.update(1.0);
        c.update(1.0);
        assert_eq!(c.update(1.0), MotionState::Moving, "fresh streak commits");
    }

    #[test]
    fn motion_idle_walk_idle_round_trip() {
        let mut c = MotionClassifier::new(0.05, 0.3, 2);
        c.update(0.0);
        assert_eq!(c.update(0.0), MotionState::Idle);
        c.update(1.0);
        assert_eq!(c.update(1.0), MotionState::Moving);
        c.update(0.0);
        assert_eq!(c.update(0.0), MotionState::Idle, "flips back after dwell");
    }

    #[test]
    fn motion_thresholds_are_inclusive() {
        // Speed exactly at walk_above counts as moving; exactly at idle_below
        // counts as idle.
        let mut c = MotionClassifier::new(0.05, 0.3, 1);
        assert_eq!(c.update(0.3), MotionState::Moving, "== walk_above → walk");
        let mut d = MotionClassifier::new(0.05, 0.3, 1);
        assert_eq!(d.update(0.05), MotionState::Idle, "== idle_below → idle");
    }

    #[test]
    fn motion_bad_thresholds_collapse_dead_zone() {
        // walk_above < idle_below is clamped so walk_above ≥ idle_below; the
        // dead zone collapses but the machine stays sane (no panic, commits).
        let mut c = MotionClassifier::new(0.3, 0.1, 1);
        // walk_above was clamped up to 0.3; 0.5 ≥ 0.3 → moving
        assert_eq!(c.update(0.5), MotionState::Moving);
        assert_eq!(c.update(0.0), MotionState::Idle);
    }

    #[test]
    fn motion_state_as_str_round_trips() {
        assert_eq!(MotionState::Unknown.as_str(), "unknown");
        assert_eq!(MotionState::Idle.as_str(), "idle");
        assert_eq!(MotionState::Moving.as_str(), "moving");
    }

    // ── arm gesture / reach (B1.4) ─────────────────────────────────────────

    /// Set one wrist visible at `(x, y)` on an existing frame.
    fn with_wrist(mut f: Landmarks33, side: Side, x: f32, y: f32) -> Landmarks33 {
        let idx = match side {
            Side::Left => L_WRIST,
            Side::Right => R_WRIST,
        };
        f[idx] = [x, y, 0.0, 1.0];
        f
    }

    #[test]
    fn reach_zero_for_hanging_arm() {
        // Torso axis vertical (shoulders y=100, hips y=200). A wrist hanging
        // straight below the shoulder-centre sits ON the axis → ~zero lateral
        // reach.
        let f = with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 100.0, 180.0);
        let r = arm_reach(&f, Side::Left).unwrap();
        assert!(r < 1e-3, "hanging arm should read ~0 reach, got {r}");
    }

    #[test]
    fn reach_large_for_outstretched_arm() {
        // Wrist 90px to the side of the shoulder-centre, torso = 100px → 0.9 torso.
        let f = with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 10.0, 110.0);
        let r = arm_reach(&f, Side::Left).unwrap();
        assert!((r - 0.9).abs() < 1e-2, "expected ~0.9 torso reach, got {r}");
    }

    #[test]
    fn reach_is_orientation_invariant() {
        // Same arm pose, body rotated 90° in-image: reach magnitude is unchanged
        // because it's measured perpendicular to the torso axis, not along x.
        // Vertical body: shoulders (100,100), hips (100,200); wrist 90px left.
        let vert = with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 10.0, 100.0);
        // Horizontal body: build a frame with a horizontal torso axis and a
        // wrist 90px perpendicular to it.
        let mut horiz = [[0.0f32; 4]; 33];
        horiz[L_SHOULDER] = [100.0, 100.0, 0.0, 1.0];
        horiz[R_SHOULDER] = [100.0, 100.0, 0.0, 1.0]; // shoulder-centre (100,100)
        horiz[L_HIP] = [200.0, 100.0, 0.0, 1.0];
        horiz[R_HIP] = [200.0, 100.0, 0.0, 1.0]; // hip-centre (200,100) → axis along -x
        horiz[L_WRIST] = [100.0, 10.0, 0.0, 1.0]; // 90px perpendicular (in y)
        let rv = arm_reach(&vert, Side::Left).unwrap();
        let rh = arm_reach(&horiz, Side::Left).unwrap();
        assert!((rv - rh).abs() < 1e-2, "vertical {rv} vs horizontal {rh}");
    }

    #[test]
    fn reach_none_without_wrist() {
        let f = frame(100.0, 100.0, 200.0); // no wrist set
        assert!(arm_reach(&f, Side::Left).is_none());
    }

    #[test]
    fn wrist_speed_is_torso_lengths_per_second() {
        // Wrist moves 50px in x over 0.5s, torso = 100px → 1.0 torso/s.
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, with_wrist(frame(100.0, 100.0, 200.0), Side::Right, 200.0, 100.0));
        h.push(0.5, with_wrist(frame(100.0, 100.0, 200.0), Side::Right, 250.0, 100.0));
        let v = h.wrist_speed(Side::Right).unwrap();
        assert!((v - 1.0).abs() < 1e-3, "expected 1.0 torso/s, got {v}");
    }

    #[test]
    fn wrist_speed_none_with_single_frame() {
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 100.0, 180.0));
        assert!(h.wrist_speed(Side::Left).is_none());
    }

    #[test]
    fn gesture_waving_on_fast_wrist() {
        // Wrist whips 200px in 0.5s, torso 100px → 4.0 torso/s ≥ wave_above.
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 100.0, 110.0));
        h.push(0.5, with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 300.0, 110.0));
        let g = h.arm_gesture(Side::Left, &GestureThresholds::default());
        assert_eq!(g, ArmGesture::Waving, "fast wrist → waving");
    }

    #[test]
    fn gesture_reaching_on_extended_steady_arm() {
        // Arm extended (0.9 torso lateral) but barely moving → reaching, not wave.
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 12.0, 110.0));
        h.push(0.5, with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 10.0, 110.0));
        let g = h.arm_gesture(Side::Left, &GestureThresholds::default());
        assert_eq!(g, ArmGesture::Reaching, "extended + steady → reaching");
    }

    #[test]
    fn gesture_none_for_resting_arm() {
        // Wrist hanging, not moving → no gesture.
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 100.0, 180.0));
        h.push(0.5, with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 100.0, 181.0));
        let g = h.arm_gesture(Side::Left, &GestureThresholds::default());
        assert_eq!(g, ArmGesture::None, "resting arm → no gesture");
    }

    #[test]
    fn gesture_waving_beats_reaching_when_both_fire() {
        // Extended AND fast → waving wins (priority on speed).
        let mut h = LandmarkHistory::new(8);
        h.push(0.0, with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 10.0, 110.0));
        h.push(0.5, with_wrist(frame(100.0, 100.0, 200.0), Side::Left, 210.0, 110.0));
        let g = h.arm_gesture(Side::Left, &GestureThresholds::default());
        assert_eq!(g, ArmGesture::Waving);
    }

    #[test]
    fn gesture_as_str_round_trips() {
        assert_eq!(ArmGesture::None.as_str(), "none");
        assert_eq!(ArmGesture::Reaching.as_str(), "reaching");
        assert_eq!(ArmGesture::Waving.as_str(), "waving");
    }

    // ── per-track motion (B1.5 / phase 9.8) ─────────────────────────────────

    #[test]
    fn motion_track_ring_is_bounded() {
        // Feeding far more frames than capacity never grows the ring (no leak).
        let mut m = MotionTrack::new();
        for i in 0..200 {
            m.observe(frame(i as f32, 100.0, 200.0), i as f64 * 0.05);
        }
        assert_eq!(m.len(), MOTION_RING_CAP, "ring stays at capacity");
    }

    #[test]
    fn motion_track_commits_moving_on_translation() {
        // Hip centre marching across the frame at ~1 torso/s for enough frames.
        let mut m = MotionTrack::new();
        for i in 0..8 {
            // torso 100px; +50px/frame at 0.5s steps → 1.0 torso/s ≥ walk threshold
            m.observe(frame(100.0 + i as f32 * 50.0, 100.0, 200.0), i as f64 * 0.5);
        }
        assert_eq!(m.state(), MotionState::Moving);
        assert!(m.speed() > 0.3, "speed {} should clear the walk threshold", m.speed());
    }

    #[test]
    fn motion_track_commits_idle_when_still() {
        let mut m = MotionTrack::new();
        for i in 0..8 {
            m.observe(frame(100.0, 100.0, 200.0), i as f64 * 0.1); // no translation
        }
        assert_eq!(m.state(), MotionState::Idle);
        assert!(m.speed() < 0.05);
    }

    #[test]
    fn motion_track_starts_unknown() {
        let m = MotionTrack::new();
        assert_eq!(m.state(), MotionState::Unknown);
        assert!(m.is_empty());
    }
}
