//! Hand-rolled Rust port of the GRIP pose classifier (gravity-referenced pose features; formerly 'method C'): raw 33×4 landmarks →
//! `compute_features` (14 azimuth-invariant, gravity-preserving geometric features) →
//! RandomForest soft-vote → {lying, sitting, standing}. Plus the VAE (retrained on public
//! data) for `recon_error` and `log_prob_z` → plausibility (OOD only, Tier 0.2).
//! Closed-form arithmetic, zero new deps. Weights from `pose_rf_forest.bin` + `pose_vae_public.bin`
//! (exported by `spike_pose/train_pose_p1.py`, fixed order).

/// EN class labels — MUST match the forest probability order (`rf.classes_`, alphabetical).
pub const POSE_LABELS: [&str; 3] = ["lying", "sitting", "standing"];
pub const N_CLASS: usize = 3;
pub const N_FEAT: usize = 14;

const BN_EPS: f32 = 1e-5;

static VAE_BIN: &[u8] = include_bytes!("pose_vae_public.bin");
static FOREST_BIN: &[u8] = include_bytes!("pose_rf_forest.bin");

#[derive(Debug, Clone)]
pub struct PoseResult {
    pub label: &'static str,
    pub label_idx: usize,
    pub confidence: f32,        // max class probability
    pub probs: [f32; N_CLASS],  // [lying, sitting, standing]
    pub recon_error: f32,       // VAE reconstruction MSE (anomaly)
    pub plausibility: f32,      // log_prob_z surprise; lower = more familiar
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

/// A dense layer y = x·Wᵀ + b. `w` is row-major [out][in] (PyTorch convention).
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    nin: usize,
    nout: usize,
}

impl Linear {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.nin);
        let mut out = vec![0f32; self.nout];
        for o in 0..self.nout {
            let mut acc = self.b[o];
            let base = o * self.nin;
            for i in 0..self.nin {
                acc += self.w[base + i] * x[i];
            }
            out[o] = acc;
        }
        out
    }
}

#[inline]
fn relu(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// Sequential cursor over the flat weight buffer.
struct Cursor<'a> {
    f: &'a [f32],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Vec<f32> {
        let s = self.f[self.pos..self.pos + n].to_vec();
        self.pos += n;
        s
    }
    fn linear(&mut self, nout: usize, nin: usize) -> Linear {
        let w = self.take(nout * nin);
        let b = self.take(nout);
        Linear { w, b, nin, nout }
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
                nodes.push(Node { feature, threshold, left, right, probs });
            }
            trees.push(nodes);
        }
        assert_eq!(p, bytes.len(), "forest.bin length mismatch (used {} of {})", p, bytes.len());
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
                    for k in 0..N_CLASS {
                        acc[k] += n.probs[k];
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
    // VAE (OOD only)
    bn_w: Vec<f32>,
    bn_b: Vec<f32>,
    bn_mean: Vec<f32>,
    bn_var: Vec<f32>,
    enc1: Linear,      // 99→128
    enc2: Linear,      // 128→64
    fc_mu: Linear,     // 64→32
    fc_logvar: Linear, // 64→32
    dec1: Linear,      // 32→64
    dec2: Linear,      // 64→128
    dec3: Linear,      // 128→99
    // classifier
    forest: Forest,
}

impl PoseHead {
    pub fn load() -> Self {
        assert_eq!(VAE_BIN.len() % 4, 0, "pose_vae_public.bin not f32-aligned");
        let f: Vec<f32> = VAE_BIN
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut c = Cursor { f: &f, pos: 0 };
        let bn_w = c.take(99);
        let bn_b = c.take(99);
        let bn_mean = c.take(99);
        let bn_var = c.take(99);
        let enc1 = c.linear(128, 99);
        let enc2 = c.linear(64, 128);
        let fc_mu = c.linear(32, 64);
        let fc_logvar = c.linear(32, 64);
        let dec1 = c.linear(64, 32);
        let dec2 = c.linear(128, 64);
        let dec3 = c.linear(99, 128);
        assert_eq!(c.pos, f.len(), "pose_vae_public.bin length mismatch (got {}, used {})", f.len(), c.pos);
        Self {
            bn_w, bn_b, bn_mean, bn_var, enc1, enc2, fc_mu, fc_logvar, dec1, dec2, dec3,
            forest: Forest::load(FOREST_BIN),
        }
    }

    /// Full chain: landmarks (33×[x,y,z,vis], MediaPipe-normalized) → PoseResult.
    pub fn classify(&self, landmarks: &[[f32; 4]; 33]) -> PoseResult {
        // classification: geometric features → forest soft-vote
        let feat = compute_features(landmarks);
        let probs = self.forest.predict(&feat);
        let mut idx = 0;
        for k in 1..N_CLASS {
            if probs[k] > probs[idx] {
                idx = k;
            }
        }

        // VAE (OOD only): recon MSE + log_prob_z on the normalized-99 vector
        let x99 = normalize_landmarks(landmarks);
        let mut h = vec![0f32; 99];
        for i in 0..99 {
            h[i] = (x99[i] - self.bn_mean[i]) / (self.bn_var[i] + BN_EPS).sqrt() * self.bn_w[i] + self.bn_b[i];
        }
        let mut h1 = self.enc1.forward(&h);
        relu(&mut h1);
        let mut h2 = self.enc2.forward(&h1);
        relu(&mut h2);
        let mu = self.fc_mu.forward(&h2);
        let logvar = self.fc_logvar.forward(&h2);

        let mut d1 = self.dec1.forward(&mu);
        relu(&mut d1);
        let mut d2 = self.dec2.forward(&d1);
        relu(&mut d2);
        let recon = self.dec3.forward(&d2);
        let mut mse = 0f32;
        for i in 0..99 {
            let d = recon[i] - x99[i];
            mse += d * d;
        }
        mse /= 99.0;

        let mut logpz = 0f32;
        for k in 0..32 {
            logpz += mu[k] * mu[k] + logvar[k].exp() - 1.0 - logvar[k];
        }
        logpz *= 0.5;

        PoseResult {
            label: POSE_LABELS[idx],
            label_idx: idx,
            confidence: probs[idx],
            probs,
            recon_error: mse,
            plausibility: logpz,
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

/// 33×4 landmarks → 14 features (order matches `FEATURE_NAMES` in the trainer).
pub fn compute_features(lm: &[[f32; 4]; 33]) -> [f32; N_FEAT] {
    let p = |k: usize| [lm[k][0], lm[k][1], lm[k][2]];
    let mid = |a: usize, b: usize| {
        let (x, y) = (p(a), p(b));
        [(x[0] + y[0]) / 2.0, (x[1] + y[1]) / 2.0, (x[2] + y[2]) / 2.0]
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

    // cloud shape (azimuth-invariant): vertical extent vs horizontal radial extent
    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    let mut rmax = 0f32;
    for k in 0..33 {
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
        (thigh_l - thigh_r).abs(),
        (shank_l + shank_r) / 2.0,
        (knee_l + knee_r) / 2.0,
        (knee_l - knee_r).abs(),
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

/// Port of `core/pose_encoder.normalize_landmarks` + flatten → [99] (for the VAE only).
/// Translate to hip center → scale by torso length → rotate torso to vertical.
pub fn normalize_landmarks(lm: &[[f32; 4]; 33]) -> [f32; 99] {
    let mut pts = [[0f32; 3]; 33];
    for k in 0..33 {
        pts[k] = [lm[k][0], lm[k][1], lm[k][2]];
    }
    let vis = |k: usize| lm[k][3];

    // hip center (23/24), visibility-weighted
    let hip_vis = vis(23) + vis(24);
    let hip = if hip_vis > 0.1 {
        let mut h = [0f32; 3];
        for c in 0..3 {
            h[c] = (pts[23][c] * vis(23) + pts[24][c] * vis(24)) / hip_vis.max(1e-6);
        }
        h
    } else {
        pts[0]
    };
    for k in 0..33 {
        for c in 0..3 {
            pts[k][c] -= hip[c];
        }
    }

    // shoulder center (11/12), already hip-relative
    let sh_vis = vis(11) + vis(12);
    let sh = if sh_vis > 0.1 {
        let mut s = [0f32; 3];
        for c in 0..3 {
            s[c] = (pts[11][c] * vis(11) + pts[12][c] * vis(12)) / sh_vis.max(1e-6);
        }
        s
    } else {
        pts[0]
    };

    // torso length = ‖shoulder_center‖ (hip is origin)
    let torso_vec = sh;
    let mut torso_len = norm3(&torso_vec);
    if torso_len < 0.01 {
        let sw = [pts[11][0] - pts[12][0], pts[11][1] - pts[12][1], pts[11][2] - pts[12][2]];
        torso_len = norm3(&sw).max(0.01);
    }
    let inv = 1.0 / torso_len.max(1e-6);
    for k in 0..33 {
        for c in 0..3 {
            pts[k][c] *= inv;
        }
    }

    // rotate torso to vertical (0,-1,0) via Rodrigues
    if norm3(&torso_vec) > 0.01 {
        let tn = norm3(&torso_vec).max(1e-6);
        let td = [torso_vec[0] / tn, torso_vec[1] / tn, torso_vec[2] / tn];
        let vertical = [0.0f32, -1.0, 0.0];
        let dot = (td[0] * vertical[0] + td[1] * vertical[1] + td[2] * vertical[2]).clamp(-1.0, 1.0);
        let angle = dot.acos();
        if angle > 0.01 {
            // axis = cross(td, vertical)
            let mut axis = [
                td[1] * vertical[2] - td[2] * vertical[1],
                td[2] * vertical[0] - td[0] * vertical[2],
                td[0] * vertical[1] - td[1] * vertical[0],
            ];
            let an = norm3(&axis);
            if an > 1e-6 {
                axis = [axis[0] / an, axis[1] / an, axis[2] / an];
                let r = rodrigues(&axis, angle);
                for k in 0..33 {
                    let p = pts[k];
                    // pts @ Rᵀ  ==  R · p  (row vector times transpose)
                    pts[k] = [
                        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
                        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
                        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
                    ];
                }
            }
        }
    }

    let mut out = [0f32; 99];
    for k in 0..33 {
        out[k * 3] = pts[k][0];
        out[k * 3 + 1] = pts[k][1];
        out[k * 3 + 2] = pts[k][2];
    }
    out
}

fn norm3(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// R = I + sin(a)·K + (1−cos(a))·K², K = skew(axis).
fn rodrigues(axis: &[f32; 3], angle: f32) -> [[f32; 3]; 3] {
    let (ax, ay, az) = (axis[0], axis[1], axis[2]);
    let k = [[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]];
    let mut k2 = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            k2[i][j] = k[i][0] * k[0][j] + k[i][1] * k[1][j] + k[i][2] * k[2][j];
        }
    }
    let (s, c) = (angle.sin(), 1.0 - angle.cos());
    let mut r = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (if i == j { 1.0 } else { 0.0 }) + s * k[i][j] + c * k2[i][j];
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_load() {
        let h = PoseHead::load(); // asserts exact VAE length + forest length consumed
        assert!(!h.forest.trees.is_empty());
    }

    // Golden parity vectors from spike_pose/train_pose_p1.py (same RF that produced the .bin):
    // (input_33x4, features[14], forest_probs[3], pred_label, vae_recon_mse, vae_logpz)
    #[allow(clippy::type_complexity)]
    const GOLDEN: &[([[f32; 4]; 33], [f32; 14], [f32; 3], &str, f32, f32)] =
        include!("pose_p1_golden.in");

    #[test]
    fn features_match_python() {
        for (input, feat_g, _, _, _, _) in GOLDEN {
            let feat = compute_features(input);
            for k in 0..N_FEAT {
                assert!((feat[k] - feat_g[k]).abs() < 1e-2,
                    "feat[{k}] {} vs golden {}", feat[k], feat_g[k]);
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
        for (input, _, probs_g, pred, recon_g, logpz_g) in GOLDEN {
            let r = head.classify(input);
            assert_eq!(r.label, *pred, "label {} vs golden {}", r.label, pred);
            for k in 0..N_CLASS {
                assert!((r.probs[k] - probs_g[k]).abs() < 5e-3,
                    "prob[{k}] {} vs golden {}", r.probs[k], probs_g[k]);
            }
            assert!((r.recon_error - recon_g).abs() < 1e-3, "recon {} vs {}", r.recon_error, recon_g);
            assert!((r.plausibility - logpz_g).abs() < 1e-2, "logpz {} vs {}", r.plausibility, logpz_g);
        }
    }
}
