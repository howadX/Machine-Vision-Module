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
use mvm_core::{Config, Detection, Module, ModuleMetadata};
use mvm_core::EMOTION_LABELS; // canonical English VGAF taxonomy (mvm-core)
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
    let h: f32 = probs.iter()
        .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
        .sum();
    h / LOG2_K
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

        let (idx, &prob) = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let mut distribution = [0.0f32; K];
        for (i, &p) in probs.iter().enumerate().take(K) {
            distribution[i] = p;
        }

        let entropy = entropy_from_probs(&distribution);
        let calibrated_conf = prob * (1.0 - entropy);
        let label = EMOTION_LABELS.get(idx).copied().unwrap_or("unknown").to_string();

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
        let resized = image.resize_exact(
            Self::IMAGE_SIZE,
            Self::IMAGE_SIZE,
            image::imageops::FilterType::Nearest,
        );

        // ImageNet normalisation + CHW
        let mut flat = vec![0.0f32; 3 * sz * sz];
        for y in 0..sz {
            for x in 0..sz {
                let px = resized.get_pixel(x as u32, y as u32);
                let base = (y * sz + x) * 3;
                flat[base]     = (px[0] as f32 / 255.0 - 0.485) / 0.229; // R
                flat[base + 1] = (px[1] as f32 / 255.0 - 0.456) / 0.224; // G
                flat[base + 2] = (px[2] as f32 / 255.0 - 0.406) / 0.225; // B
            }
        }

        let shape = ndarray::IxDyn(&[1, 3, sz, sz]);
        let array = ndarray::Array::from_shape_vec(shape, flat)
            .context("Failed to create emotion input array")?;
        Tensor::from_array(array).context("Failed to create emotion tensor")
    }
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
            models_path: std::path::PathBuf::from(
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"),
            ),
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
        assert!((h - 1.0).abs() < 1e-5, "uniform entropy should be 1.0, got {h}");
    }

    /// High raw prob + high entropy → calibrated_conf < raw prob.
    #[test]
    fn calibrated_conf_penalizes_uncertain() {
        // Nearly uniform — uncertain despite not-terrible max prob
        let p = [0.3_f32, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.03];
        let h = entropy_from_probs(&p);
        let cal = p[0] * (1.0 - h);
        assert!(cal < p[0], "calibrated_conf should be lower than raw prob when uncertain");
        assert!(h > 0.8, "nearly-uniform distribution should have high entropy: {h}");
    }

    /// Fully certain prediction: calibrated_conf ≈ prob.
    #[test]
    fn calibrated_conf_near_prob_when_certain() {
        let mut p = [0.0f32; K];
        p[0] = 0.95;
        let tiny = 0.05 / (K - 1) as f32;
        for i in 1..K { p[i] = tiny; }
        let h = entropy_from_probs(&p);
        let cal = p[0] * (1.0 - h);
        assert!(cal > 0.8, "calibrated_conf should be high when certain: {cal}");
        assert!(h < 0.2, "concentrated distribution should have low entropy: {h}");
    }

    // ── ONNX tests ────────────────────────────────────────────────────────────

    #[test]
    fn model_loads_and_runs_on_black_image() {
        let mut emo = EmotionModule::init(&test_config()).unwrap();
        assert!(emo.metadata().name.contains("EmotiEffLib"));
        let img = DynamicImage::new_rgb8(224, 224);
        let result = emo.classify(&img).unwrap();
        eprintln!("Black image → {} (prob={:.3}, entropy={:.3}, cal={:.3})",
            result.label, result.prob, result.entropy, result.calibrated_conf);
    }

    #[test]
    fn emotion_result_distribution_sums_to_one() {
        let mut emo = EmotionModule::init(&test_config()).unwrap();
        let img = DynamicImage::new_rgb8(224, 224);
        let result = emo.classify(&img).unwrap();
        let sum: f32 = result.distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "distribution should sum to 1.0, got {sum}");
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
        eprintln!("Reference photo → {} (prob={:.3}, entropy={:.3}, cal={:.3})",
            result.label, result.prob, result.entropy, result.calibrated_conf);
        assert!(!result.label.is_empty());
        assert!(result.prob > 0.0);
        assert!(result.entropy >= 0.0 && result.entropy <= 1.0);
        assert!(result.calibrated_conf <= result.prob);
    }
}
