//! mvm-emotion — facial emotion recognition via EmotiEffLib B0 ONNX.
//!
//! EfficientNet-B0, 8 emotions (VGAF dataset).
//! Apache 2.0 license. 15.3 MB model, ~15ms CPU inference.
//!
//! Preprocessing: resize 224×224, ImageNet normalisation, CHW.
//! Output: 8-class probabilities → `EmotionResult` with entropy-based uncertainty.

use anyhow::Context;
use image::DynamicImage;
use image::GenericImageView;
use mvm_core::EMOTION_LABELS; // canonical English VGAF taxonomy (mvm-core)
use mvm_core::{Config, Detection, Module, ModuleMetadata};
use ort::{session::Session, value::Tensor};

const K: usize = 8; // number of emotion classes
const LOG2_K: f32 = 3.0; // log2(8)

/// Full emotion classification result.
///
/// `calibrated_conf` is the recommended field for downstream decisions:
/// it discounts raw probability by distributional uncertainty, making
/// "confident in one class" meaningfully different from "spread across all classes".
#[derive(Debug, Clone)]
pub struct EmotionResult {
    /// Predicted emotion label (Russian).
    pub label: String,
    /// `max(softmax)` — raw model confidence, [0, 1].
    pub prob: f32,
    /// Full softmax distribution over 8 classes. Sums to 1.0.
    pub distribution: [f32; K],
    /// Normalised Shannon entropy: 0 = fully certain, 1 = uniform.
    /// `entropy = -Σ p_i * log2(p_i) / log2(K)`
    pub entropy: f32,
    /// Same as `entropy` (placeholder for future EDL / Dirichlet uncertainty).
    pub uncertainty: f32,
    /// `prob * (1 - uncertainty)` — use this for cross-model comparison.
    pub calibrated_conf: f32,
}

/// Compute normalised Shannon entropy over a probability distribution.
/// Returns value in [0, 1]: 0 = one-hot, 1 = uniform.
pub fn entropy_from_probs(probs: &[f32]) -> f32 {
    let h: f32 = probs
        .iter()
        .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
        .sum();
    h / LOG2_K
}

// ── Face alignment (phase 10.1) ──────────────────────────────────────────────

/// ArcFace canonical 5-point template on a 112×112 crop, image coords:
/// image-left eye, image-right eye, nose tip, image-left mouth, image-right mouth.
/// Matches YuNet's landmark order (`mvm-detect::FaceDet::landmarks`).
const ALIGN_TEMPLATE_112: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Least-squares similarity transform (uniform scale + rotation + translation,
/// no reflection) mapping `src` points onto `dst`: returns `[a, -b, tx, b, a, ty]`
/// so `x' = a·x − b·y + tx`, `y' = b·x + a·y + ty`. Closed-form (the 2-DOF
/// complex-regression form of Umeyama). `None` when `src` is degenerate.
pub fn similarity_from_points(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> Option<[f32; 6]> {
    let n = 5.0f32;
    let (mut sx, mut sy, mut dx, mut dy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for k in 0..5 {
        sx += src[k][0];
        sy += src[k][1];
        dx += dst[k][0];
        dy += dst[k][1];
    }
    let (sx, sy, dx, dy) = (sx / n, sy / n, dx / n, dy / n);
    // Centred cross terms: treat points as complex numbers, regress d = (a+ib)·s.
    let (mut num_re, mut num_im, mut den) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..5 {
        let (ux, uy) = (src[k][0] - sx, src[k][1] - sy);
        let (vx, vy) = (dst[k][0] - dx, dst[k][1] - dy);
        num_re += ux * vx + uy * vy;
        num_im += ux * vy - uy * vx;
        den += ux * ux + uy * uy;
    }
    if den < 1e-9 {
        return None;
    }
    let a = num_re / den;
    let b = num_im / den;
    let tx = dx - a * sx + b * sy;
    let ty = dy - b * sx - a * sy;
    Some([a, -b, tx, b, a, ty])
}

/// Affine-align a face out of the **full frame** to the canonical eye/mouth
/// template (phase 10.1, root-#1 companion fix): the 5 YuNet landmarks (pixels)
/// drive a similarity transform onto the ArcFace template scaled to
/// `out_size`², sampled bilinearly. Rotation/offset/scale of the head are
/// normalised away before EmotiEffLib sees the crop. `None` when the landmark
/// geometry is degenerate (fall back to the raw bbox crop).
pub fn align_face(
    frame: &DynamicImage,
    lm5_px: &[[f32; 2]; 5],
    out_size: u32,
) -> Option<DynamicImage> {
    let scale = out_size as f32 / 112.0;
    let mut dst = ALIGN_TEMPLATE_112;
    for p in &mut dst {
        p[0] *= scale;
        p[1] *= scale;
    }
    // Forward transform src→dst; we need the inverse to sample the source.
    let m = similarity_from_points(lm5_px, &dst)?;
    let det = m[0] * m[4] - m[1] * m[3];
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = [
        m[4] / det,
        -m[1] / det,
        (m[1] * m[5] - m[4] * m[2]) / det,
        -m[3] / det,
        m[0] / det,
        (m[3] * m[2] - m[0] * m[5]) / det,
    ];
    let rgb = frame.to_rgb8();
    let (fw, fh) = (rgb.width() as i32, rgb.height() as i32);
    let mut out = image::RgbImage::new(out_size, out_size);
    for y in 0..out_size {
        for x in 0..out_size {
            let (ox, oy) = (x as f32, y as f32);
            let sxf = inv[0] * ox + inv[1] * oy + inv[2];
            let syf = inv[3] * ox + inv[4] * oy + inv[5];
            // Bilinear sample, clamped to the frame.
            let x0 = sxf.floor() as i32;
            let y0 = syf.floor() as i32;
            let (fx, fy) = (sxf - x0 as f32, syf - y0 as f32);
            let cl = |v: i32, hi: i32| v.clamp(0, hi - 1) as u32;
            let p00 = rgb.get_pixel(cl(x0, fw), cl(y0, fh));
            let p10 = rgb.get_pixel(cl(x0 + 1, fw), cl(y0, fh));
            let p01 = rgb.get_pixel(cl(x0, fw), cl(y0 + 1, fh));
            let p11 = rgb.get_pixel(cl(x0 + 1, fw), cl(y0 + 1, fh));
            let mut px = [0u8; 3];
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
                let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                px[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, image::Rgb(px));
        }
    }
    Some(DynamicImage::ImageRgb8(out))
}

/// EmotiEffLib B0 ONNX emotion classifier.
pub struct EmotionModule {
    session: Session,
}

impl EmotionModule {
    const INPUT_NAME: &'static str = "input";
    const IMAGE_SIZE: u32 = 224;

    /// Load from ONNX model bytes (embedded mode).
    #[cfg(feature = "embed-model")]
    pub fn from_bytes(model_bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_memory(model_bytes)
            .context("Failed to load EmotiEffLib from embedded bytes")?;
        Ok(Self { session })
    }

    /// Load from file path.
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load EmotiEffLib from {}", path.display()))?;
        Ok(Self { session })
    }

    /// Classify a face crop. Returns full `EmotionResult` with entropy-based uncertainty.
    pub fn classify(&mut self, face: &DynamicImage) -> anyhow::Result<EmotionResult> {
        let input = self.preprocess(face)?;
        let outputs = self.session.run(ort::inputs![Self::INPUT_NAME => input])?;
        let scores: &[f32] = outputs[0].try_extract_tensor::<f32>()?.1;

        // Softmax (numerically stable)
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let (idx, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let mut distribution = [0.0f32; K];
        for (i, &p) in probs.iter().enumerate().take(K) {
            distribution[i] = p;
        }

        let entropy = entropy_from_probs(&distribution);
        let calibrated_conf = prob * (1.0 - entropy);
        let label = EMOTION_LABELS
            .get(idx)
            .copied()
            .unwrap_or("unknown")
            .to_string();

        Ok(EmotionResult {
            label,
            prob,
            distribution,
            entropy,
            uncertainty: entropy,
            calibrated_conf,
        })
    }

    fn preprocess(&self, image: &DynamicImage) -> anyhow::Result<ort::value::Tensor<f32>> {
        let sz = Self::IMAGE_SIZE as usize;
        let flat = preprocess_chw(image, Self::IMAGE_SIZE);
        let shape = ndarray::IxDyn(&[1, 3, sz, sz]);
        let array = ndarray::Array::from_shape_vec(shape, flat)
            .context("Failed to create emotion input array")?;
        Tensor::from_array(array).context("Failed to create emotion tensor")
    }
}

/// Resize + ImageNet-normalise a crop into **planar CHW** (`[R-plane, G-plane,
/// B-plane]`) — the layout the NCHW ONNX input actually reads.
///
/// Phase 10.1 root-#1 fix: this used to write interleaved HWC into the NCHW
/// buffer, scrambling the channel planes the network saw — every face came out
/// near-uniform (the live "emotion is near-chance" symptom). Kept as a free
/// function so the layout is unit-testable without ONNX.
pub fn preprocess_chw(image: &DynamicImage, size: u32) -> Vec<f32> {
    let sz = size as usize;
    let resized = image.resize_exact(size, size, image::imageops::FilterType::Nearest);
    let mut flat = vec![0.0f32; 3 * sz * sz];
    let plane = sz * sz;
    for y in 0..sz {
        for x in 0..sz {
            let px = resized.get_pixel(x as u32, y as u32);
            let idx = y * sz + x;
            flat[idx] = (px[0] as f32 / 255.0 - 0.485) / 0.229; // R
            flat[plane + idx] = (px[1] as f32 / 255.0 - 0.456) / 0.224; // G
            flat[2 * plane + idx] = (px[2] as f32 / 255.0 - 0.406) / 0.225; // B
        }
    }
    flat
}

impl Module for EmotionModule {
    type Input = Vec<Detection>;
    type Output = Vec<Detection>;

    fn init(config: &Config) -> anyhow::Result<Self> {
        let model_path = config.models_path.join("enet_b0_8_best_vgaf.onnx");
        Self::from_path(&model_path)
    }

    fn process(&mut self, input: &Self::Input) -> anyhow::Result<Self::Output> {
        Ok(input.clone())
    }

    fn metadata(&self) -> ModuleMetadata {
        ModuleMetadata {
            name: "mvm-emotion (EmotiEffLib B0)",
            version: env!("CARGO_PKG_VERSION"),
            accuracy: Some(0.73),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config {
            models_path: std::path::PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../models"
            )),
            ..Default::default()
        }
    }

    // ── Pure math tests (no ONNX) ─────────────────────────────────────────────

    /// One-hot distribution → entropy = 0.
    #[test]
    fn entropy_zero_for_one_hot() {
        let mut p = [0.0f32; K];
        p[4] = 1.0;
        let h = entropy_from_probs(&p);
        assert!(h.abs() < 1e-6, "one-hot entropy should be 0, got {h}");
    }

    /// Uniform distribution → entropy = 1.
    #[test]
    fn entropy_one_for_uniform() {
        let p = [1.0 / K as f32; K];
        let h = entropy_from_probs(&p);
        assert!(
            (h - 1.0).abs() < 1e-5,
            "uniform entropy should be 1.0, got {h}"
        );
    }

    /// High raw prob + high entropy → calibrated_conf < raw prob.
    #[test]
    fn calibrated_conf_penalizes_uncertain() {
        // Nearly uniform — uncertain despite not-terrible max prob
        let p = [0.3_f32, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.03];
        let h = entropy_from_probs(&p);
        let cal = p[0] * (1.0 - h);
        assert!(
            cal < p[0],
            "calibrated_conf should be lower than raw prob when uncertain"
        );
        assert!(
            h > 0.8,
            "nearly-uniform distribution should have high entropy: {h}"
        );
    }

    /// Fully certain prediction: calibrated_conf ≈ prob.
    #[test]
    fn calibrated_conf_near_prob_when_certain() {
        let mut p = [0.0f32; K];
        p[0] = 0.95;
        let tiny = 0.05 / (K - 1) as f32;
        for i in 1..K {
            p[i] = tiny;
        }
        let h = entropy_from_probs(&p);
        let cal = p[0] * (1.0 - h);
        assert!(
            cal > 0.8,
            "calibrated_conf should be high when certain: {cal}"
        );
        assert!(
            h < 0.2,
            "concentrated distribution should have low entropy: {h}"
        );
    }

    /// Root-#1 regression guard: a solid-red image must produce a constant R
    /// plane followed by constant G/B planes (planar CHW), not interleaved HWC.
    #[test]
    fn preprocess_layout_is_planar_chw() {
        let mut img = image::RgbImage::new(8, 8);
        for p in img.pixels_mut() {
            *p = image::Rgb([255, 0, 0]);
        }
        let flat = preprocess_chw(&DynamicImage::ImageRgb8(img), 8);
        let plane = 64;
        let r = (1.0 - 0.485) / 0.229;
        let g = (0.0 - 0.456) / 0.224;
        let b = (0.0 - 0.406) / 0.225;
        assert!(
            flat[..plane].iter().all(|&v| (v - r).abs() < 1e-5),
            "R plane first"
        );
        assert!(
            flat[plane..2 * plane].iter().all(|&v| (v - g).abs() < 1e-5),
            "G plane second"
        );
        assert!(
            flat[2 * plane..].iter().all(|&v| (v - b).abs() < 1e-5),
            "B plane third"
        );
    }

    // ── Face alignment (phase 10.1, pure math) ────────────────────────────────

    #[test]
    fn similarity_identity_when_points_match() {
        let m = similarity_from_points(&ALIGN_TEMPLATE_112, &ALIGN_TEMPLATE_112).unwrap();
        assert!((m[0] - 1.0).abs() < 1e-4 && m[1].abs() < 1e-4 && m[2].abs() < 1e-3);
        assert!(m[3].abs() < 1e-4 && (m[4] - 1.0).abs() < 1e-4 && m[5].abs() < 1e-3);
    }

    #[test]
    fn similarity_recovers_rotation_scale_translation() {
        // Apply a known similarity to the template, then ask the solver for it back.
        let (s, th, tx, ty) = (1.7f32, 0.35f32, 40.0f32, -12.0f32);
        let (a, b) = (s * th.cos(), s * th.sin());
        let mut src = ALIGN_TEMPLATE_112;
        for p in &mut src {
            let (x, y) = (p[0], p[1]);
            p[0] = a * x - b * y + tx;
            p[1] = b * x + a * y + ty;
        }
        // Solver maps src → template; composing with the known forward transform
        // must give identity on every point.
        let m = similarity_from_points(&src, &ALIGN_TEMPLATE_112).unwrap();
        for (p, q) in src.iter().zip(ALIGN_TEMPLATE_112.iter()) {
            let x = m[0] * p[0] + m[1] * p[1] + m[2];
            let y = m[3] * p[0] + m[4] * p[1] + m[5];
            assert!(
                (x - q[0]).abs() < 1e-2 && (y - q[1]).abs() < 1e-2,
                "mapped ({x},{y}) vs template {q:?}"
            );
        }
    }

    #[test]
    fn similarity_degenerate_points_yield_none() {
        let src = [[5.0, 5.0]; 5]; // all identical → no scale information
        assert!(similarity_from_points(&src, &ALIGN_TEMPLATE_112).is_none());
    }

    #[test]
    fn align_face_maps_landmarks_to_canonical_positions() {
        // A synthetic frame with bright dots at the source landmarks: after
        // alignment each dot must sit at its canonical template position.
        let mut img = image::RgbImage::new(320, 240);
        // Landmarks: template shifted +60/+40 and scaled 1.5 (still inside frame).
        let mut src = ALIGN_TEMPLATE_112;
        for p in &mut src {
            p[0] = p[0] * 1.5 + 60.0;
            p[1] = p[1] * 1.5 + 40.0;
        }
        for p in &src {
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let (x, y) = ((p[0] as i32 + dx) as u32, (p[1] as i32 + dy) as u32);
                    img.put_pixel(x, y, image::Rgb([255, 255, 255]));
                }
            }
        }
        let aligned = align_face(&DynamicImage::ImageRgb8(img), &src, 224).unwrap();
        assert_eq!(aligned.width(), 224);
        let rgb = aligned.to_rgb8();
        let scale = 224.0 / 112.0;
        for t in &ALIGN_TEMPLATE_112 {
            let (x, y) = ((t[0] * scale) as u32, (t[1] * scale) as u32);
            let px = rgb.get_pixel(x, y);
            assert!(
                px[0] > 128,
                "expected bright dot at canonical ({x},{y}), got {px:?}"
            );
        }
    }

    // ── ONNX tests ────────────────────────────────────────────────────────────

    #[test]
    fn model_loads_and_runs_on_black_image() {
        let mut emo = EmotionModule::init(&test_config()).unwrap();
        assert!(emo.metadata().name.contains("EmotiEffLib"));
        let img = DynamicImage::new_rgb8(224, 224);
        let result = emo.classify(&img).unwrap();
        eprintln!(
            "Black image → {} (prob={:.3}, entropy={:.3}, cal={:.3})",
            result.label, result.prob, result.entropy, result.calibrated_conf
        );
    }

    #[test]
    fn emotion_result_distribution_sums_to_one() {
        let mut emo = EmotionModule::init(&test_config()).unwrap();
        let img = DynamicImage::new_rgb8(224, 224);
        let result = emo.classify(&img).unwrap();
        let sum: f32 = result.distribution.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "distribution should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn model_runs_on_reference_face() {
        let mut emo = EmotionModule::init(&test_config()).unwrap();
        let ref_path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("repos/Machine Vision Module/models/references/p_V.jpg");
        if !ref_path.exists() {
            eprintln!("Skipping: {} not found", ref_path.display());
            return;
        }
        let img = image::open(&ref_path).unwrap();
        let result = emo.classify(&img).unwrap();
        eprintln!(
            "Reference photo → {} (prob={:.3}, entropy={:.3}, cal={:.3})",
            result.label, result.prob, result.entropy, result.calibrated_conf
        );
        assert!(!result.label.is_empty());
        assert!(result.prob > 0.0);
        assert!(result.entropy >= 0.0 && result.entropy <= 1.0);
        assert!(result.calibrated_conf <= result.prob);
    }
}
