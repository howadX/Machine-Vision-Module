//! Hand-rolled Rust port of the 2D-GRIP pose classifier (gravity-referenced pose features; formerly 'method C'): raw 33×4 landmarks →
//! `compute_features` (12 azimuth-invariant, gravity-preserving geometric features, **z zeroed**) →
//! RandomForest soft-vote → {lying, sitting, standing}. Plus class-conditional Mahalanobis
//! (replaces the VAE) for the out-of-distribution flag (OOD only, Tier 0.2).
//! Closed-form arithmetic, zero new deps. Weights from `pose_rf_2dgrip_v2.bin` + `pose_maha.bin`
//! (phase 15.6 port of the LJ1 lab decisions; exported by `spike_pose/lj1_build_ship.py`,
//! parity-pinned by `spike_pose/lj6_golden.py` → `pose_v2_golden.in`).

/// EN class labels — MUST match the forest probability order (`rf.classes_`, alphabetical).
/// The Mahalanobis artifact uses this *same* order (unified in 15.6, though its score is a
/// pure min over classes so the order is functionally inert there).
pub const POSE_LABELS: [&str; 3] = ["lying", "sitting", "standing"];
pub const N_CLASS: usize = 3;
/// 12 z-free 2D-GRIP features (was 14 = these + 2 dropped left/right asymmetry terms).
pub const N_FEAT: usize = 12;

static MAHA_BIN: &[u8] = include_bytes!("pose_maha.bin");
static FOREST_BIN: &[u8] = include_bytes!("pose_rf_2dgrip_v2.bin");

#[derive(Debug, Clone)]
pub struct PoseResult {
    pub label: &'static str,
    pub label_idx: usize,
    pub confidence: f32,       // max class probability
    pub probs: [f32; N_CLASS], // [lying, sitting, standing]
    /// Continuous torso flexion = feature[0] `torso_inc` (2D, z-free), degrees to
    /// gravity-up. Pure arithmetic, +0 model weight (15.6). `Some` whenever a torso
    /// is present; the pipeline may still withhold it under `Head` coverage.
    pub torso_flexion: Option<f32>,
    /// Class-conditional Mahalanobis distance to the nearest posture cluster (lower =
    /// more familiar). Replaces the VAE `log_prob_z`. See [`ood`](Self::ood).
    pub maha_score: f32,
    /// Out-of-distribution flag: `maha_score` exceeds the artifact's 99th-pct threshold.
    pub ood: bool,
    /// Body-visibility regime this frame (occlusion-aware posture): which parts
    /// of the body BlazePose actually *saw*. Gates how much of the RF label the
    /// pipeline may claim — see [`BodyCoverage`].
    pub coverage: BodyCoverage,
    /// 2D image-plane torso attitude ("upright"/"inclined"/"horizontal") — the
    /// occlusion-robust orientation signal (see [`torso_attitude_2d`]).
    pub attitude: &'static str,
}

// ── Occlusion-aware body coverage + torso attitude ───────────────────────────
//
// Target deployments (home/lab/robotics) occlude the lower body routinely: a
// desk, a workbench, furniture, the frame edge. BlazePose then *hallucinates*
// knee/ankle coordinates (their visibility collapses), and the leg-dependent
// features feed the RF garbage — measured on the owner's desk frames this read
// desk-sitting as confident `lying` (P(sitting) ≤ 0.12 on all 7 references).
// Rather than fine-tune the classifier to guess from hallucinated joints, the
// pipeline is honest about what was seen: coverage names the regime, and in
// `Upper` the posture label is withheld (attitude still flows).
//
// Thresholds are calibrated on measured distributions (2026-07-02):
//   lower_vis (mean vis of knees 25/26 + ankles 27/28):
//     desk-sitting 0.29–0.40, desk-standing 0.30–0.53 → Upper at < 0.5
//     URFD full-body p50 ≈ 0.8 (standing p5 = 0.47)   → Full at ≥ 0.5
//   2D torso attitude (image-plane angle to image-up; camera ~horizontal so
//   image-y ≈ gravity; the 3D angle is unusable here — BlazePose z on a
//   truncated body read 55–66° even for an upright subject):
//     desk (sitting+standing) 10.1–17.5° → upright < 30°
//     URFD lying p50 = 89°               → horizontal > 60°

/// Which parts of the body were actually visible to BlazePose this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyCoverage {
    /// Lower body visible → the full 14-feature RF label is trustworthy.
    Full,
    /// Hips visible but knees/ankles are not (desk, furniture, frame edge) —
    /// leg features are hallucinated, so the RF label must not be claimed as-is.
    Upper,
    /// Even the hips are invisible — only presence/identity/emotion are safe.
    Head,
}

impl BodyCoverage {
    pub fn as_str(self) -> &'static str {
        match self {
            BodyCoverage::Full => "full",
            BodyCoverage::Upper => "upper",
            BodyCoverage::Head => "head",
        }
    }
}

/// `Full` requires mean knee+ankle visibility at/above this.
pub const COVERAGE_LOWER_VIS: f32 = 0.5;
/// Below this mean hip visibility the coverage degrades to `Head`.
pub const COVERAGE_HIP_VIS: f32 = 0.5;
/// 2D torso angle below this (degrees to image-up) reads "upright".
pub const ATTITUDE_UPRIGHT_DEG: f32 = 30.0;
/// 2D torso angle above this reads "horizontal" (lying / fallen).
pub const ATTITUDE_HORIZONTAL_DEG: f32 = 60.0;

/// Classify the body-visibility regime from landmark visibilities (pure
/// arithmetic): hips 23/24, knees 25/26, ankles 27/28.
pub fn coverage_from_vis(lm: &[[f32; 4]; 33]) -> BodyCoverage {
    let hip_vis = (lm[23][3] + lm[24][3]) / 2.0;
    if hip_vis < COVERAGE_HIP_VIS {
        return BodyCoverage::Head;
    }
    let lower_vis = (lm[25][3] + lm[26][3] + lm[27][3] + lm[28][3]) / 4.0;
    if lower_vis < COVERAGE_LOWER_VIS {
        BodyCoverage::Upper
    } else {
        BodyCoverage::Full
    }
}

/// 2D image-plane torso attitude: the angle of hip-center → shoulder-center
/// against image-up. The camera is ~horizontal (eye-level), so image-y ≈
/// gravity, and — unlike the 3D torso inclination — this ignores BlazePose z,
/// which is unreliable on a truncated (occluded) body. Degenerate torso reads
/// "upright" (same convention as `ang_to_vert` for a zero vector).
pub fn torso_attitude_2d(lm: &[[f32; 4]; 33]) -> &'static str {
    let hip = [(lm[23][0] + lm[24][0]) / 2.0, (lm[23][1] + lm[24][1]) / 2.0];
    let sh = [(lm[11][0] + lm[12][0]) / 2.0, (lm[11][1] + lm[12][1]) / 2.0];
    let v = [sh[0] - hip[0], sh[1] - hip[1]];
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    let deg = if n < 1e-6 {
        0.0
    } else {
        // image-up = (0,-1) → cos = -v_y / |v|
        (-v[1] / n).clamp(-1.0, 1.0).acos().to_degrees()
    };
    if deg < ATTITUDE_UPRIGHT_DEG {
        "upright"
    } else if deg > ATTITUDE_HORIZONTAL_DEG {
        "horizontal"
    } else {
        "inclined"
    }
}

// ── Class-conditional Mahalanobis OOD (replaces the VAE) ─────────────────────

/// Class-conditional Mahalanobis out-of-distribution detector. `score(f)` = the
/// minimum over posture clusters of √((f−μ_c)ᵀ P_c (f−μ_c)); a pose is OOD iff that
/// exceeds `thr` (the ID 99th percentile). All three live in `pose_maha.bin`,
/// computed on the same 12 z-free features the forest consumes.
///
/// The class *order* is inert (the score is a pure min) — the load-bearing invariant
/// is that each cluster's `mean` stays paired with its own `inv_cov`.
struct MahaOod {
    thr: f32,
    means: Vec<[f32; N_FEAT]>,
    inv_cov: Vec<[f32; N_FEAT * N_FEAT]>,
}

impl MahaOod {
    fn load(bytes: &[u8]) -> Self {
        let rd_u32 = |b: &[u8], p: usize| u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
        let rd_f32 = |b: &[u8], p: usize| f32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
        let n_cls = rd_u32(bytes, 0) as usize;
        let n_feat = rd_u32(bytes, 4) as usize;
        assert_eq!(
            n_feat, N_FEAT,
            "pose_maha.bin feature count {} != {}",
            n_feat, N_FEAT
        );
        let thr = rd_f32(bytes, 8);
        let mut p = 12usize;
        let mut means = Vec::with_capacity(n_cls);
        let mut inv_cov = Vec::with_capacity(n_cls);
        for _ in 0..n_cls {
            let mut mu = [0f32; N_FEAT];
            for m in mu.iter_mut() {
                *m = rd_f32(bytes, p);
                p += 4;
            }
            means.push(mu);
            let mut ic = [0f32; N_FEAT * N_FEAT];
            for v in ic.iter_mut() {
                *v = rd_f32(bytes, p);
                p += 4;
            }
            inv_cov.push(ic);
        }
        assert_eq!(
            p,
            bytes.len(),
            "pose_maha.bin length mismatch (used {} of {})",
            p,
            bytes.len()
        );
        MahaOod {
            thr,
            means,
            inv_cov,
        }
    }

    /// Minimum class-conditional Mahalanobis distance (lower = more familiar).
    fn score(&self, feat: &[f32; N_FEAT]) -> f32 {
        let mut best = f32::INFINITY;
        for (mu, ic) in self.means.iter().zip(self.inv_cov.iter()) {
            let mut d = [0f32; N_FEAT];
            for i in 0..N_FEAT {
                d[i] = feat[i] - mu[i];
            }
            // quadratic form dᵀ P d
            let mut q = 0f32;
            for i in 0..N_FEAT {
                let mut row = 0f32;
                for j in 0..N_FEAT {
                    row += ic[i * N_FEAT + j] * d[j];
                }
                q += d[i] * row;
            }
            let dist = q.max(0.0).sqrt();
            if dist < best {
                best = dist;
            }
        }
        best
    }
}

// ── RandomForest (soft-vote over trees) ──────────────────────────────────────

/// One CART node. `left < 0` marks a leaf (then `probs` is the class distribution).
struct Node {
    feature: i32,
    threshold: f32,
    left: i32,
    right: i32,
    probs: [f32; N_CLASS],
}

struct Forest {
    trees: Vec<Vec<Node>>,
}

impl Forest {
    fn load(bytes: &[u8]) -> Self {
        let rd_u32 = |b: &[u8], p: usize| u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
        let rd_i32 = |b: &[u8], p: usize| i32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
        let rd_f32 = |b: &[u8], p: usize| f32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
        let mut p = 0usize;
        let n_trees = rd_u32(bytes, p) as usize;
        p += 4;
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            let n_nodes = rd_u32(bytes, p) as usize;
            p += 4;
            let mut nodes = Vec::with_capacity(n_nodes);
            for _ in 0..n_nodes {
                let feature = rd_i32(bytes, p);
                let threshold = rd_f32(bytes, p + 4);
                let left = rd_i32(bytes, p + 8);
                let right = rd_i32(bytes, p + 12);
                let mut probs = [0f32; N_CLASS];
                for (k, pr) in probs.iter_mut().enumerate() {
                    *pr = rd_f32(bytes, p + 16 + k * 4);
                }
                p += 16 + N_CLASS * 4;
                nodes.push(Node {
                    feature,
                    threshold,
                    left,
                    right,
                    probs,
                });
            }
            trees.push(nodes);
        }
        assert_eq!(
            p,
            bytes.len(),
            "forest.bin length mismatch (used {} of {})",
            p,
            bytes.len()
        );
        Forest { trees }
    }

    /// Soft-vote: average the leaf class-distribution across all trees.
    fn predict(&self, feat: &[f32; N_FEAT]) -> [f32; N_CLASS] {
        let mut acc = [0f32; N_CLASS];
        for tree in &self.trees {
            let mut i = 0usize;
            loop {
                let n = &tree[i];
                if n.left < 0 {
                    for (k, total) in acc.iter_mut().enumerate() {
                        *total += n.probs[k];
                    }
                    break;
                }
                // sklearn: X[feature] <= threshold → left child
                i = if feat[n.feature as usize] <= n.threshold {
                    n.left as usize
                } else {
                    n.right as usize
                };
            }
        }
        let inv = 1.0 / self.trees.len() as f32;
        for a in acc.iter_mut() {
            *a *= inv;
        }
        acc
    }
}

pub struct PoseHead {
    forest: Forest,
    ood: MahaOod,
}

impl PoseHead {
    pub fn load() -> Self {
        Self {
            forest: Forest::load(FOREST_BIN),
            ood: MahaOod::load(MAHA_BIN),
        }
    }

    /// Full chain: landmarks (33×[x,y,z,vis], MediaPipe-normalized) → PoseResult.
    pub fn classify(&self, landmarks: &[[f32; 4]; 33]) -> PoseResult {
        // classification: 12 z-free geometric features → forest soft-vote
        let feat = compute_features(landmarks);
        let probs = self.forest.predict(&feat);
        let mut idx = 0;
        for k in 1..N_CLASS {
            if probs[k] > probs[idx] {
                idx = k;
            }
        }

        // OOD: class-conditional Mahalanobis on the same 12 features.
        let maha_score = self.ood.score(&feat);

        PoseResult {
            label: POSE_LABELS[idx],
            label_idx: idx,
            confidence: probs[idx],
            probs,
            torso_flexion: Some(feat[0]),
            maha_score,
            ood: maha_score > self.ood.thr,
            coverage: coverage_from_vis(landmarks),
            attitude: torso_attitude_2d(landmarks),
        }
    }
}

// ── Geometric features (port of spike_pose/spike_feats.py::features) ──────────

const UP: [f32; 3] = [0.0, -1.0, 0.0]; // MediaPipe: +y is down → up = -y

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Angle (degrees) between `v` and the gravity-up axis.
fn ang_to_vert(v: [f32; 3]) -> f32 {
    let n = norm3(&v);
    if n < 1e-6 {
        return 0.0;
    }
    let d = ((v[0] * UP[0] + v[1] * UP[1] + v[2] * UP[2]) / n).clamp(-1.0, 1.0);
    d.acos().to_degrees()
}

/// Interior angle (degrees) at `b` between `a-b` and `c-b`.
fn flex(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let u = sub(a, b);
    let w = sub(c, b);
    let (nu, nw) = (norm3(&u), norm3(&w));
    if nu < 1e-6 || nw < 1e-6 {
        return 180.0;
    }
    let d = ((u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (nu * nw)).clamp(-1.0, 1.0);
    d.acos().to_degrees()
}

/// The 13 GRIP joints (MediaPipe indices) `vext`/`hext` are measured over — so the
/// cloud-shape features are reproducible on any skeleton (incl. a mocap rig with no
/// face/hand points). Mirrors `mocap_sanity_gate.GRIP_IDX`.
const GRIP_IDX: [usize; 13] = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

/// 33×4 landmarks → 12 z-free features (order matches `mocap_sanity_gate.FEATS`).
///
/// **2D-GRIP (z zeroed):** MediaPipe's z is a cross-domain liability (LJ1), so every
/// coordinate is flattened to the image plane before the geometry is computed. With
/// z=0 the radial extent collapses to |x−hip_x|. Parity-pinned by `pose_v2_golden.in`.
pub fn compute_features(lm: &[[f32; 4]; 33]) -> [f32; N_FEAT] {
    let p = |k: usize| [lm[k][0], lm[k][1], 0.0]; // z zeroed → 2D-GRIP
    let mid = |a: usize, b: usize| {
        let (x, y) = (p(a), p(b));
        [
            (x[0] + y[0]) / 2.0,
            (x[1] + y[1]) / 2.0,
            (x[2] + y[2]) / 2.0,
        ]
    };
    let hip = mid(23, 24);
    let sh = mid(11, 12);
    let torso = sub(sh, hip);
    let mut tlen = norm3(&torso);
    if tlen < 1e-6 {
        tlen = norm3(&sub(p(11), p(12))).max(1e-3);
    }

    let torso_inc = ang_to_vert(torso);
    let thigh_l = ang_to_vert(sub(p(25), p(23)));
    let thigh_r = ang_to_vert(sub(p(26), p(24)));
    let shank_l = ang_to_vert(sub(p(27), p(25)));
    let shank_r = ang_to_vert(sub(p(28), p(26)));
    let knee_l = flex(p(23), p(25), p(27));
    let knee_r = flex(p(24), p(26), p(28));
    let hip_l = flex(p(11), p(23), p(25));
    let hip_r = flex(p(12), p(24), p(26));
    let elb_l = flex(p(11), p(13), p(15));
    let elb_r = flex(p(12), p(14), p(16));

    // cloud shape (azimuth-invariant), over the 13 GRIP joints only: vertical extent
    // vs horizontal radial extent (z=0 ⇒ radial = |x−hip_x|).
    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    let mut rmax = 0f32;
    for &k in &GRIP_IDX {
        let q = p(k);
        ymin = ymin.min(q[1]);
        ymax = ymax.max(q[1]);
        let dx = q[0] - hip[0];
        let dz = q[2] - hip[2];
        rmax = rmax.max((dx * dx + dz * dz).sqrt());
    }
    let vext = (ymax - ymin) / tlen;
    let hext = 2.0 * rmax / tlen;
    let aspect = vext / (hext + 1e-6);

    let nose_above = (hip[1] - p(0)[1]) / tlen;
    let ank_below = ((p(27)[1] + p(28)[1]) / 2.0 - hip[1]) / tlen;
    let kn_below = ((p(25)[1] + p(26)[1]) / 2.0 - hip[1]) / tlen;

    [
        torso_inc,
        (thigh_l + thigh_r) / 2.0,
        (shank_l + shank_r) / 2.0,
        (knee_l + knee_r) / 2.0,
        (hip_l + hip_r) / 2.0,
        (elb_l + elb_r) / 2.0,
        vext,
        hext,
        aspect,
        nose_above,
        ank_below,
        kn_below,
    ]
}

fn norm3(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_load() {
        let h = PoseHead::load(); // asserts exact forest + maha lengths consumed
        assert!(!h.forest.trees.is_empty());
        assert_eq!(h.ood.means.len(), N_CLASS);
        assert_eq!(h.ood.inv_cov.len(), N_CLASS);
        assert!(h.ood.thr > 0.0);
    }

    // Golden parity vectors from spike_pose/lj6_golden.py (computed FROM the shipped
    // pose_rf_2dgrip_v2.bin + pose_maha.bin, so they pin those exact bytes):
    // (input_33x4, features12[z=0], forest_probs3 [lying,sitting,standing], maha_score, flexion, label)
    #[allow(clippy::type_complexity)]
    const GOLDEN: &[([[f32; 4]; 33], [f32; 12], [f32; 3], f32, f32, &str)] =
        include!("pose_v2_golden.in");

    #[test]
    fn features_match_python() {
        for (input, feat_g, ..) in GOLDEN {
            let feat = compute_features(input);
            for k in 0..N_FEAT {
                assert!(
                    (feat[k] - feat_g[k]).abs() < 1e-2,
                    "feat[{k}] {} vs golden {}",
                    feat[k],
                    feat_g[k]
                );
            }
        }
    }

    // ── occlusion-aware coverage + attitude ─────────────────────────────────

    /// A landmark set with a given upright torso and per-group visibilities.
    /// Geometry: shoulders at y=0.35, hips at y=0.60 (upright in image coords).
    fn lm_with_vis(hip_vis: f32, knee_vis: f32, ankle_vis: f32) -> [[f32; 4]; 33] {
        let mut lm = [[0.5f32, 0.5, 0.0, 0.9]; 33];
        lm[11] = [0.45, 0.35, 0.0, 0.95];
        lm[12] = [0.55, 0.35, 0.0, 0.95];
        lm[23] = [0.46, 0.60, 0.0, hip_vis];
        lm[24] = [0.54, 0.60, 0.0, hip_vis];
        lm[25] = [0.46, 0.75, 0.0, knee_vis];
        lm[26] = [0.54, 0.75, 0.0, knee_vis];
        lm[27] = [0.46, 0.90, 0.0, ankle_vis];
        lm[28] = [0.54, 0.90, 0.0, ankle_vis];
        lm
    }

    #[test]
    fn coverage_full_when_legs_visible() {
        // URFD-like full body: lower_vis ≈ 0.8 (measured p50).
        let lm = lm_with_vis(0.99, 0.8, 0.8);
        assert_eq!(coverage_from_vis(&lm), BodyCoverage::Full);
    }

    #[test]
    fn coverage_upper_on_measured_desk_profile() {
        // The owner's desk frames (measured 2026-07-02): hips ~1.0, knees ~0.5,
        // ankles ~0.2 → lower_vis 0.35 → Upper (legs are hallucinated).
        let lm = lm_with_vis(1.0, 0.5, 0.2);
        assert_eq!(coverage_from_vis(&lm), BodyCoverage::Upper);
    }

    #[test]
    fn coverage_head_when_hips_invisible() {
        let lm = lm_with_vis(0.2, 0.1, 0.1);
        assert_eq!(coverage_from_vis(&lm), BodyCoverage::Head);
    }

    #[test]
    fn attitude_upright_for_vertical_torso() {
        // Shoulders directly above hips (0° to image-up) → upright.
        let lm = lm_with_vis(1.0, 0.8, 0.8);
        assert_eq!(torso_attitude_2d(&lm), "upright");
    }

    #[test]
    fn attitude_horizontal_for_lying_torso() {
        // Shoulders level with hips, displaced in x (90°) → horizontal.
        let mut lm = lm_with_vis(1.0, 0.8, 0.8);
        lm[11] = [0.75, 0.59, 0.0, 0.95];
        lm[12] = [0.75, 0.61, 0.0, 0.95];
        lm[23] = [0.30, 0.59, 0.0, 0.9];
        lm[24] = [0.30, 0.61, 0.0, 0.9];
        assert_eq!(torso_attitude_2d(&lm), "horizontal");
    }

    #[test]
    fn attitude_inclined_between_thresholds() {
        // 45° torso: dx = dy → inclined.
        let mut lm = lm_with_vis(1.0, 0.8, 0.8);
        lm[11] = [0.65, 0.40, 0.0, 0.95];
        lm[12] = [0.65, 0.40, 0.0, 0.95];
        lm[23] = [0.45, 0.60, 0.0, 0.9];
        lm[24] = [0.45, 0.60, 0.0, 0.9];
        assert_eq!(torso_attitude_2d(&lm), "inclined");
    }

    #[test]
    fn attitude_degenerate_torso_reads_upright() {
        // Shoulders collapsed onto hips (zero torso) → "upright" by convention.
        let mut lm = lm_with_vis(1.0, 0.8, 0.8);
        for k in [11usize, 12, 23, 24] {
            lm[k] = [0.5, 0.5, 0.0, 0.9];
        }
        assert_eq!(torso_attitude_2d(&lm), "upright");
    }

    #[test]
    fn classify_carries_coverage_and_attitude() {
        // The full chain surfaces the regime alongside the RF label.
        let head = PoseHead::load();
        let (input, ..) = &GOLDEN[0];
        let r = head.classify(input);
        assert!(["full", "upper", "head"].contains(&r.coverage.as_str()));
        assert!(["upright", "inclined", "horizontal"].contains(&r.attitude));
    }

    #[test]
    fn classify_matches_python_oracle() {
        let head = PoseHead::load();
        for (input, _, probs_g, maha_g, flex_g, pred) in GOLDEN {
            let r = head.classify(input);
            assert_eq!(r.label, *pred, "label {} vs golden {}", r.label, pred);
            for (k, &golden_prob) in probs_g.iter().enumerate() {
                assert!(
                    (r.probs[k] - golden_prob).abs() < 5e-3,
                    "prob[{k}] {} vs golden {}",
                    r.probs[k],
                    golden_prob
                );
            }
            assert!(
                (r.maha_score - maha_g).abs() < 1e-2,
                "maha {} vs {}",
                r.maha_score,
                maha_g
            );
            assert!(
                (r.torso_flexion.unwrap() - flex_g).abs() < 1e-2,
                "flexion {:?} vs {}",
                r.torso_flexion,
                flex_g
            );
        }
    }
}
