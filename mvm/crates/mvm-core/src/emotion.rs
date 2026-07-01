//! Canonical emotion taxonomy + temporal Product-of-Experts fusion (phase 6.4b).
//!
//! `mvm-core` is the dependency-graph leaf, so the 8-class VGAF taxonomy lives
//! here — shared by `mvm-emotion` (classifier output), `mvm-fusion` (cluster PoE),
//! and the tracker (temporal smoothing). Labels are English: they map 1:1 onto the
//! EmotiEffLib / HSEmotion class order and read cleanly in code and downstream JSON.

/// Number of emotion classes (VGAF).
pub const K: usize = 8;

/// VGAF 8-class emotion taxonomy, in classifier output order.
pub const EMOTION_LABELS: [&str; K] = [
    "Anger",     // 0
    "Contempt",  // 1
    "Disgust",   // 2
    "Fear",      // 3
    "Happiness", // 4
    "Neutral",   // 5
    "Sadness",   // 6
    "Surprise",  // 7
];

/// Normalised Shannon entropy of a distribution. `0.0` = certain (one-hot),
/// `1.0` = uniform. Used as `weight = 1 - entropy` for PoE fusion.
pub fn emotion_entropy(dist: &[f32; K]) -> f32 {
    let mut h = 0.0f32;
    for &p in dist {
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    (h / (K as f32).log2()).clamp(0.0, 1.0)
}

/// argmax label + its probability from a distribution.
pub fn emotion_label(dist: &[f32; K]) -> (&'static str, f32) {
    let (idx, &prob) = dist
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap_or((0, &0.0));
    (EMOTION_LABELS[idx], prob)
}

/// Normalised entropy at/above which an emotion label is too close to chance to
/// emit (Tier 0.3). Uniform = 1.0; a genuine read peaks well below. Empirically a
/// live near-chance distribution (peak ≈0.2 over 8 classes) sits at ≈0.97, while a
/// clearly-read one (peak ≈0.4) sits at ≈0.85 — this threshold separates them.
pub const EMOTION_MAX_ENTROPY: f32 = 0.90;

/// argmax label + prob, but **only when the distribution is peaked enough to be
/// informative** (Tier 0.3): returns `None` when normalised entropy is at/above
/// [`EMOTION_MAX_ENTROPY`] (near-chance), so downstream never sees a meaningless
/// label. The distribution itself is kept by the caller for anyone reading the
/// raw signal.
pub fn emotion_label_gated(dist: &[f32; K]) -> Option<(&'static str, f32)> {
    if emotion_entropy(dist) >= EMOTION_MAX_ENTROPY {
        None
    } else {
        Some(emotion_label(dist))
    }
}

/// Temporal Product-of-Experts: fold a new observation into the accumulated
/// prior via log-linear pooling with forgetting.
///
/// `p_new ∝ prior^λ · obs^{w_obs}` where `λ < 1` decays the prior (so a steady
/// signal never collapses to one-hot) and `w_obs = 1 - entropy(obs)` down-weights
/// high-entropy frames. Returns the normalised fused distribution.
pub fn fuse_emotion_temporal(prior: &[f32; K], obs: &[f32; K], lambda: f32) -> [f32; K] {
    let w_obs = 1.0 - emotion_entropy(obs);
    let mut log_sum = [0.0f32; K];
    for k in 0..K {
        log_sum[k] = lambda * prior[k].max(1e-10).ln() + w_obs * obs[k].max(1e-10).ln();
    }
    // softmax with numerical stability
    let max_ls = log_sum.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: [f32; K] = std::array::from_fn(|k| (log_sum[k] - max_ls).exp());
    let sum: f32 = exps.iter().sum();
    std::array::from_fn(|k| exps[k] / sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_zero_for_one_hot() {
        let mut d = [0.0; K];
        d[3] = 1.0;
        assert!(emotion_entropy(&d) < 1e-6);
    }

    #[test]
    fn entropy_one_for_uniform() {
        let d = [1.0 / K as f32; K];
        assert!((emotion_entropy(&d) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn label_picks_argmax() {
        let mut d = [0.01; K];
        d[7] = 0.93;
        assert_eq!(emotion_label(&d).0, "Surprise");
    }

    #[test]
    fn gated_suppresses_near_chance() {
        // The observed live near-chance: Fear peaks ~0.215, rest ~0.09–0.16.
        // Normalised entropy ≈0.97 ≥ 0.90 → label suppressed.
        let d = [0.085, 0.096, 0.092, 0.215, 0.086, 0.126, 0.128, 0.161];
        assert!(emotion_entropy(&d) >= EMOTION_MAX_ENTROPY);
        assert_eq!(emotion_label_gated(&d), None);
    }

    #[test]
    fn gated_emits_clearly_peaked() {
        // A confident read (peak ~0.93) is well below the entropy gate → emitted.
        let mut d = [0.01; K];
        d[4] = 0.93;
        assert_eq!(emotion_label_gated(&d), Some(("Happiness", 0.93)));
    }

    #[test]
    fn gated_emits_moderately_peaked() {
        // Peak ~0.40 (entropy ≈0.85 < 0.90) is informative enough to keep.
        let d = [0.044, 0.063, 0.074, 0.402, 0.046, 0.088, 0.118, 0.165];
        assert!(emotion_entropy(&d) < EMOTION_MAX_ENTROPY);
        assert_eq!(emotion_label_gated(&d).map(|(l, _)| l), Some("Fear"));
    }

    #[test]
    fn gated_suppresses_uniform() {
        let d = [1.0 / K as f32; K];
        assert_eq!(emotion_label_gated(&d), None);
    }

    #[test]
    fn fused_distribution_sums_to_one() {
        let a = [0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1];
        let b = [0.1, 0.1, 0.5, 0.05, 0.05, 0.05, 0.05, 0.1];
        let f = fuse_emotion_temporal(&a, &b, 0.7);
        let sum: f32 = f.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn temporal_no_collapse_with_forgetting() {
        // Feed the same confident observation many times; with lambda < 1 the
        // distribution must NOT collapse to one-hot.
        let obs = [0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1];
        let mut prior = [1.0 / K as f32; K];
        for _ in 0..200 {
            prior = fuse_emotion_temporal(&prior, &obs, 0.7);
        }
        let max = prior.iter().cloned().fold(0.0f32, f32::max);
        assert!(max < 0.999, "distribution collapsed to one-hot: max={max}");
    }

    #[test]
    fn high_entropy_obs_contributes_less() {
        // An uncertain (high-entropy) observation should move the prior LESS than a
        // confident, conflicting one. Compare the prior's peak mass after each.
        let prior = [0.8, 0.05, 0.03, 0.02, 0.02, 0.03, 0.03, 0.02];
        let uniform_obs = [1.0 / K as f32; K]; // entropy ≈ 1 → weight ≈ 0
        let conflicting_obs = [0.02, 0.03, 0.8, 0.02, 0.03, 0.03, 0.05, 0.02]; // confident on class 2

        let fused_uniform = fuse_emotion_temporal(&prior, &uniform_obs, 0.7);
        let fused_conflict = fuse_emotion_temporal(&prior, &conflicting_obs, 0.7);

        // A single uncertain frame leaves the prior's class on top (anti-jitter
        // stickiness — one frame can't overturn a confident prior).
        assert_eq!(emotion_label(&fused_uniform).0, "Anger");
        // The confident-but-conflicting frame erodes the prior's peak more than the
        // uniform one, and raises the contested class more — i.e. it "contributes more".
        assert!(
            fused_uniform[0] > fused_conflict[0],
            "uniform {} should preserve class-0 mass better than conflict {}",
            fused_uniform[0], fused_conflict[0]
        );
        assert!(
            fused_conflict[2] > fused_uniform[2],
            "conflicting evidence should raise the contested class more: {} vs {}",
            fused_conflict[2], fused_uniform[2]
        );
    }
}
