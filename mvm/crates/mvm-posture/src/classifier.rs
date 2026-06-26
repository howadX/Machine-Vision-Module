//! Hand-rolled Rust port of the Python pose chain: `normalize_landmarks` (33×4 → 99)
//! → PoseVAE encoder (BN + 2 Linears + fc_mu) → z32 → PoseClassifier (3 Linears) → 6 poses,
//! plus decoder → reconstruction MSE and `log_prob_z` → plausibility.
//! Closed-form arithmetic, zero new deps. Weights from `pose_weights.bin`
//! (exported by `spike_pose/export_pose_weights.py`, fixed order).

/// EN class labels (G7 — straight from the checkpoint `class_names`).
pub const POSE_LABELS: [&str; 6] = [
    "standing", "sitting", "lying_back", "lying_side", "arms_crossed", "sitting_at_desk",
];

const BN_EPS: f32 = 1e-5;

static WEIGHTS_BIN: &[u8] = include_bytes!("pose_weights.bin");

#[derive(Debug, Clone)]
pub struct PoseResult {
    pub label: &'static str,
    pub label_idx: usize,
    pub confidence: f32,    // max softmax prob
    pub probs: [f32; 6],
    pub recon_error: f32,   // VAE reconstruction MSE (anomaly)
    pub plausibility: f32,  // log_prob_z surprise; lower = more familiar
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

pub struct PoseHead {
    bn_w: Vec<f32>,
    bn_b: Vec<f32>,
    bn_mean: Vec<f32>,
    bn_var: Vec<f32>,
    enc1: Linear, // 99→128
    enc2: Linear, // 128→64
    fc_mu: Linear, // 64→32
    fc_logvar: Linear, // 64→32
    dec1: Linear, // 32→64
    dec2: Linear, // 64→128
    dec3: Linear, // 128→99
    clf1: Linear, // 32→64
    clf2: Linear, // 64→32
    clf3: Linear, // 32→6
}

impl PoseHead {
    pub fn load() -> Self {
        assert_eq!(WEIGHTS_BIN.len() % 4, 0, "pose_weights.bin not f32-aligned");
        let f: Vec<f32> = WEIGHTS_BIN
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
        let clf1 = c.linear(64, 32);
        let clf2 = c.linear(32, 64);
        let clf3 = c.linear(6, 32);
        assert_eq!(c.pos, f.len(), "pose_weights.bin length mismatch (got {}, used {})", f.len(), c.pos);
        Self { bn_w, bn_b, bn_mean, bn_var, enc1, enc2, fc_mu, fc_logvar, dec1, dec2, dec3, clf1, clf2, clf3 }
    }

    /// Full chain: landmarks (33×[x,y,z,vis], MediaPipe-normalized) → PoseResult.
    pub fn classify(&self, landmarks: &[[f32; 4]; 33]) -> PoseResult {
        let x99 = normalize_landmarks(landmarks);

        // encoder: BN → Lin → ReLU → Lin → ReLU
        let mut h = vec![0f32; 99];
        for i in 0..99 {
            h[i] = (x99[i] - self.bn_mean[i]) / (self.bn_var[i] + BN_EPS).sqrt() * self.bn_w[i] + self.bn_b[i];
        }
        let mut h1 = self.enc1.forward(&h);
        relu(&mut h1);
        let mut h2 = self.enc2.forward(&h1);
        relu(&mut h2);
        let mu = self.fc_mu.forward(&h2); // z (deterministic)
        let logvar = self.fc_logvar.forward(&h2);

        // decoder for reconstruction MSE
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

        // log_prob_z = 0.5 * Σ(mu² + var − 1 − logvar)
        let mut logpz = 0f32;
        for k in 0..32 {
            logpz += mu[k] * mu[k] + logvar[k].exp() - 1.0 - logvar[k];
        }
        logpz *= 0.5;

        // classifier: Lin → ReLU → Lin → ReLU → Lin → softmax
        let mut c1 = self.clf1.forward(&mu);
        relu(&mut c1);
        let mut c2 = self.clf2.forward(&c1);
        relu(&mut c2);
        let logits = self.clf3.forward(&c2);
        let probs = softmax6(&logits);

        let mut idx = 0;
        for k in 1..6 {
            if probs[k] > probs[idx] {
                idx = k;
            }
        }
        PoseResult {
            label: POSE_LABELS[idx],
            label_idx: idx,
            confidence: probs[idx],
            probs,
            recon_error: mse,
            plausibility: logpz,
        }
    }
}

fn softmax6(logits: &[f32]) -> [f32; 6] {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0f32; 6];
    let mut s = 0f32;
    for k in 0..6 {
        e[k] = (logits[k] - m).exp();
        s += e[k];
    }
    for k in 0..6 {
        e[k] /= s;
    }
    e
}

/// Port of `core/pose_encoder.normalize_landmarks` + flatten → [99].
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
        let _ = PoseHead::load(); // asserts exact length consumed
    }

    // Fixed 33×4 landmark input (MediaPipe-normalized x/w,y/h,z/w,vis) from p_V_03.
    const GOLDEN_INPUT: [[f32; 4]; 33] = include!("golden_input.in");

    #[test]
    fn classify_matches_python_oracle() {
        // Golden from spike_pose/export_pose_weights.py on the fixed p_V_03 33×4 input:
        // class=standing, probs[0]=0.848, recon_mse=0.009896, logpz=8.1492
        let head = PoseHead::load();
        let r = head.classify(&GOLDEN_INPUT);

        assert_eq!(r.label, "standing", "expected standing, got {}", r.label);
        let probs_golden = [0.847909, 0.005945, 0.003319, 0.003328, 0.016686, 0.122813];
        for k in 0..6 {
            assert!((r.probs[k] - probs_golden[k]).abs() < 1e-3,
                "prob[{k}] {} vs golden {}", r.probs[k], probs_golden[k]);
        }
        assert!((r.recon_error - 0.009896).abs() < 1e-4, "recon {} vs 0.009896", r.recon_error);
        assert!((r.plausibility - 8.1492).abs() < 1e-2, "logpz {} vs 8.1492", r.plausibility);
    }
}
