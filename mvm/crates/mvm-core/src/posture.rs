//! Posture temporal stabiliser (Tier 0.1) — **pure arithmetic, zero deps**.
//!
//! The posture classifier (`mvm-posture`: BlazePose → geometric features → 3-class
//! RandomForest → standing/sitting/lying; VAE kept for OOD only) is evaluated
//! **per frame** and its argmax label was written straight onto the track (a
//! passthrough "latest snapshot"). Unlike emotion — which is fused with a temporal
//! Product-of-Experts — posture had **no temporal smoothing**, so a single bad
//! frame (partial landmarks from a fragmented detection, a borderline pose the head
//! is overconfident about) flipped the emitted label frame-to-frame
//! (e.g. `standing` → `lying` → `standing`), even at high confidence.
//!
//! This is the categorical analogue of [`crate::motion::MotionClassifier`]'s
//! dwell + hysteresis state machine, adapted to a labelled class instead of a
//! scalar regime:
//!
//! - **Confidence gate (`min_conf`):** a frame whose confidence is below the gate
//!   is *ambiguous* — it never changes the committed label and breaks any pending
//!   streak (so evidence for a flip must be both confident **and** consecutive).
//! - **Trust gate (Tier 0.2):** a frame the pose Mahalanobis OOD detector flags
//!   out-of-distribution (`plausibility` "low" — far from any posture cluster, e.g.
//!   garbage landmarks from a fragmented detection) is *ambiguous* too, regardless of the
//!   classifier's softmax confidence. The signal was already computed upstream but
//!   only reported; now it also guards the committed label. A flip therefore needs
//!   evidence that is confident, **plausible**, and consecutive.
//! - **Dwell:** a *different* confident label must hold for `dwell` consecutive
//!   frames before it commits. A one-frame outlier — even a high-confidence one —
//!   never flips the committed posture. Sustained chatter between two labels never
//!   accumulates a streak, so the committed label simply holds.
//!
//! The first confident observation **bootstraps** immediately (there is nothing to
//! protect yet), preserving the old first-sight behaviour.

/// Trust verdict for one per-frame posture observation, fed as `trusted` to
/// [`PostureStabilizer::observe`]. A frame is distrusted (treated as ambiguous —
/// never flips the committed label) when either:
/// - the pose Mahalanobis OOD detector flagged it out-of-distribution
///   (`plausibility == "low"`), or
/// - it says **`lying` while the body is moving** — `lying` is a static posture, so
///   `moving + lying` is almost always a bad/fragment frame (mitigation, 2026-07).
pub fn posture_trusted(plausibility: Option<&str>, label: &str, moving: bool) -> bool {
    plausibility != Some("low") && !(label == "lying" && moving)
}

/// Per-track posture stabiliser. Feed each frame's `(label, confidence)` through
/// [`Self::observe`]; read the smoothed committed label via [`Self::label`].
#[derive(Debug, Clone)]
pub struct PostureStabilizer {
    label: Option<String>,
    confidence: f32,
    min_conf: f32,
    dwell: u32,
    candidate: Option<String>,
    streak: u32,
    /// Consecutive frames the posture was *withheld* upstream (body coverage
    /// `upper`/`head` — the lower body is occluded, so no full-body claim is
    /// possible). After `dwell` such frames the committed label is cleared:
    /// a person who sat down behind a desk must not keep emitting the stale
    /// `standing` they walked in with. Reset by any [`Self::observe`] call.
    hidden_streak: u32,
}

impl Default for PostureStabilizer {
    /// Commit a flip only after **3** consecutive frames whose confidence clears
    /// **0.5**. The classifier is often overconfident on borderline/OOD frames, so
    /// the *dwell* (not the gate) is the primary guard against single-frame flips.
    fn default() -> Self {
        Self::new(0.5, 3)
    }
}

impl PostureStabilizer {
    /// `min_conf` — below this a frame is ambiguous (never flips, breaks streak).
    /// `dwell` — consecutive confident different-label frames to commit (≥1).
    pub fn new(min_conf: f32, dwell: u32) -> Self {
        Self {
            label: None,
            confidence: 0.0,
            min_conf,
            dwell: dwell.max(1),
            candidate: None,
            streak: 0,
            hidden_streak: 0,
        }
    }

    /// The committed (smoothed) posture label, or `None` before the first
    /// confident observation.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Confidence of the last frame that committed or reinforced the current
    /// label. `None` until a label is committed.
    pub fn confidence(&self) -> Option<f32> {
        self.label.as_ref().map(|_| self.confidence)
    }

    /// Fold one per-frame posture observation into the committed state; returns
    /// the (possibly unchanged) committed label. `trusted` is `false` when the
    /// pose VAE flagged the frame out-of-distribution (`plausibility` "unusual").
    pub fn observe(&mut self, label: &str, conf: f32, trusted: bool) -> Option<&str> {
        self.hidden_streak = 0; // a per-frame label arrived → the body is visible again
        // Ambiguous frame — below the confidence gate OR flagged OOD by the pose
        // VAE. It never changes the committed label and breaks any pending streak,
        // so a flip needs confident, plausible, and consecutive evidence.
        if !trusted || conf < self.min_conf {
            self.candidate = self.label.clone();
            self.streak = 0;
            return self.label.as_deref();
        }

        match self.label.as_deref() {
            // First confident sight → bootstrap immediately (nothing to protect).
            None => {
                self.label = Some(label.to_string());
                self.confidence = conf;
                self.candidate = self.label.clone();
                self.streak = 0;
            }
            // Same as committed → reinforce, refresh confidence, clear streak.
            Some(cur) if cur == label => {
                self.confidence = conf;
                self.candidate = self.label.clone();
                self.streak = 0;
            }
            // A different confident label → accumulate consecutive evidence.
            Some(_) => {
                if self.candidate.as_deref() == Some(label) {
                    self.streak += 1;
                } else {
                    self.candidate = Some(label.to_string());
                    self.streak = 1;
                }
                if self.streak >= self.dwell {
                    self.label = Some(label.to_string());
                    self.confidence = conf;
                    self.streak = 0;
                }
            }
        }
        self.label.as_deref()
    }

    /// Fold one *hidden* frame (occlusion-aware posture): the upstream withheld
    /// the posture because the body coverage was `upper`/`head` — the lower body
    /// is occluded and no honest full-body claim exists. After `dwell`
    /// consecutive hidden frames the committed label is **cleared** (a stale
    /// `standing` must not survive sitting down behind a desk); a transient blip
    /// (a fragment frame) shorter than the dwell leaves the label untouched.
    pub fn observe_hidden(&mut self) -> Option<&str> {
        self.candidate = self.label.clone();
        self.streak = 0;
        self.hidden_streak += 1;
        if self.hidden_streak >= self.dwell {
            self.label = None;
            self.confidence = 0.0;
            self.candidate = None;
        }
        self.label.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_commits_first_confident() {
        let mut s = PostureStabilizer::default();
        s.observe("standing", 0.9, true);
        assert_eq!(s.label(), Some("standing"));
        assert_eq!(s.confidence(), Some(0.9));
    }

    #[test]
    fn low_conf_first_stays_none() {
        let mut s = PostureStabilizer::default();
        s.observe("standing", 0.3, true); // below gate → nothing committed yet
        assert_eq!(s.label(), None);
        assert_eq!(s.confidence(), None);
    }

    #[test]
    fn single_frame_high_conf_outlier_rejected() {
        // The exact live symptom: committed standing, one 0.95 lying_side spike.
        let mut s = PostureStabilizer::default();
        s.observe("standing", 0.9, true);
        s.observe("lying_side", 0.95, true); // outlier — must NOT flip
        assert_eq!(s.label(), Some("standing"));
    }

    #[test]
    fn sustained_change_commits_after_dwell() {
        let mut s = PostureStabilizer::new(0.5, 3);
        s.observe("standing", 0.9, true);
        s.observe("sitting", 0.9, true); // streak 1 → still standing
        assert_eq!(s.label(), Some("standing"));
        s.observe("sitting", 0.9, true); // streak 2 → still standing
        assert_eq!(s.label(), Some("standing"));
        s.observe("sitting", 0.9, true); // streak 3 → commit
        assert_eq!(s.label(), Some("sitting"));
        assert_eq!(s.confidence(), Some(0.9));
    }

    #[test]
    fn chatter_between_two_labels_holds_committed() {
        let mut s = PostureStabilizer::new(0.5, 3);
        s.observe("standing", 0.9, true);
        // Alternating labels never build a consecutive streak of `dwell`.
        for _ in 0..6 {
            s.observe("lying_side", 0.95, true);
            s.observe("sitting", 0.95, true);
        }
        assert_eq!(s.label(), Some("standing"), "chatter must not flip the label");
    }

    #[test]
    fn low_conf_frame_breaks_a_pending_streak() {
        let mut s = PostureStabilizer::new(0.5, 3);
        s.observe("standing", 0.9, true);
        s.observe("sitting", 0.9, true); // streak 1
        s.observe("sitting", 0.3, true); // below gate → ambiguous, resets streak
        s.observe("sitting", 0.9, true); // streak 1 again
        s.observe("sitting", 0.9, true); // streak 2 → NOT yet committed
        assert_eq!(s.label(), Some("standing"), "interrupted streak must not commit");
        s.observe("sitting", 0.9, true); // streak 3 → commit
        assert_eq!(s.label(), Some("sitting"));
    }

    #[test]
    fn same_label_reinforces_confidence() {
        let mut s = PostureStabilizer::default();
        s.observe("standing", 0.9, true);
        s.observe("standing", 0.6, true); // same label → refresh conf, no flip
        assert_eq!(s.label(), Some("standing"));
        assert_eq!(s.confidence(), Some(0.6));
    }

    #[test]
    fn same_label_run_clears_streak_between_outliers() {
        // A lone outlier between long same-label runs never accumulates.
        let mut s = PostureStabilizer::new(0.5, 2);
        s.observe("standing", 0.9, true);
        s.observe("lying_side", 0.9, true); // streak 1
        s.observe("standing", 0.9, true); // same as committed → clears streak
        s.observe("lying_side", 0.9, true); // streak 1 (not 2)
        assert_eq!(s.label(), Some("standing"));
    }

    // ── Tier 0.2: trust gate (pose-VAE plausibility) ────────────────────────

    #[test]
    fn untrusted_frame_does_not_flip() {
        // A high-confidence but OOD-flagged frame must not override the label.
        let mut s = PostureStabilizer::default();
        s.observe("standing", 0.9, true);
        s.observe("lying_side", 0.99, false); // "unusual" → ambiguous
        assert_eq!(s.label(), Some("standing"));
    }

    #[test]
    fn untrusted_frame_does_not_bootstrap() {
        // The first sight being OOD must not seed a posture from garbage.
        let mut s = PostureStabilizer::default();
        s.observe("lying_side", 0.99, false);
        assert_eq!(s.label(), None);
    }

    #[test]
    fn untrusted_frame_breaks_pending_streak() {
        // An OOD frame interrupts an in-progress flip just like a low-conf one.
        let mut s = PostureStabilizer::new(0.5, 3);
        s.observe("standing", 0.9, true);
        s.observe("sitting", 0.9, true); // streak 1
        s.observe("sitting", 0.9, false); // OOD → resets streak
        s.observe("sitting", 0.9, true); // streak 1
        s.observe("sitting", 0.9, true); // streak 2 → not yet
        assert_eq!(s.label(), Some("standing"));
    }

    // ── occlusion-aware hidden frames (coverage upper/head) ─────────────────

    #[test]
    fn sustained_hidden_clears_stale_label() {
        // Walked in standing → sat behind the desk (posture withheld upstream).
        // After `dwell` hidden frames the stale `standing` is cleared.
        let mut s = PostureStabilizer::new(0.5, 3);
        s.observe("standing", 0.9, true);
        s.observe_hidden();
        s.observe_hidden();
        assert_eq!(s.label(), Some("standing"), "below dwell → label holds");
        s.observe_hidden();
        assert_eq!(s.label(), None, "sustained occlusion → stale label cleared");
        assert_eq!(s.confidence(), None);
    }

    #[test]
    fn hidden_blip_does_not_clear() {
        // A 1–2 frame fragment/occlusion blip must not wipe a good posture.
        let mut s = PostureStabilizer::new(0.5, 3);
        s.observe("sitting", 0.9, true);
        s.observe_hidden();
        s.observe_hidden();
        s.observe("sitting", 0.9, true); // visible again → hidden streak resets
        s.observe_hidden();
        s.observe_hidden();
        assert_eq!(s.label(), Some("sitting"), "interrupted hidden streak never clears");
    }

    #[test]
    fn hidden_breaks_pending_flip_streak() {
        // A hidden frame is ambiguous for flips too (like low-conf / OOD).
        let mut s = PostureStabilizer::new(0.5, 3);
        s.observe("standing", 0.9, true);
        s.observe("sitting", 0.9, true); // streak 1
        s.observe_hidden(); // resets the flip streak
        s.observe("sitting", 0.9, true); // streak 1
        s.observe("sitting", 0.9, true); // streak 2 → not yet
        assert_eq!(s.label(), Some("standing"));
    }

    #[test]
    fn rebootstrap_after_hidden_clear() {
        // After a clear, the next confident sight bootstraps immediately.
        let mut s = PostureStabilizer::new(0.5, 2);
        s.observe("standing", 0.9, true);
        s.observe_hidden();
        s.observe_hidden(); // cleared
        assert_eq!(s.label(), None);
        s.observe("sitting", 0.9, true);
        assert_eq!(s.label(), Some("sitting"), "first confident sight re-bootstraps");
    }

    #[test]
    fn posture_trusted_gates_ood_and_moving_lying() {
        // OOD frame → distrusted regardless of label/motion.
        assert!(!posture_trusted(Some("low"), "standing", false));
        // `lying` while moving → distrusted (lying is static).
        assert!(!posture_trusted(Some("ok"), "lying", true));
        assert!(!posture_trusted(None, "lying", true));
        // `lying` while still → trusted; moving non-lying → trusted.
        assert!(posture_trusted(Some("ok"), "lying", false));
        assert!(posture_trusted(None, "standing", true));
        assert!(posture_trusted(None, "sitting", true));
    }
}
