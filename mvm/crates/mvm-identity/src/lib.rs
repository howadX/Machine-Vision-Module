//! mvm-identity — face recognition via SFace ONNX (Apache 2.0).
//!
//! OpenCV SFace: 112×112 RGB → 128-dim L2-normalised embedding.
//! Cosine similarity matching against a gallery of reference faces.
//!
//! Model: face_recognition_sface_2021dec.onnx (36.9 MB).

use anyhow::Context;
use image::DynamicImage;
use image::GenericImageView;
use mvm_core::{Config, Detection, Module, ModuleMetadata};
use ort::{session::Session, value::Tensor};
use std::collections::HashMap;

/// SFace ONNX face recogniser with in-memory gallery.
pub struct IdentityModule {
    session: Session,
    /// Gallery: person_name → 128-dim L2-normalised embedding.
    gallery: HashMap<String, Vec<f32>>,
}

impl IdentityModule {
    const INPUT_NAME: &'static str = "data";
    const IMAGE_SIZE: u32 = 112;
    const EMBEDDING_DIM: usize = 128;

    /// Load from ONNX model bytes (embedded mode).
    #[cfg(feature = "embed-model")]
    pub fn from_bytes(model_bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_memory(model_bytes)
            .context("Failed to load SFace from embedded bytes")?;
        Ok(Self { session, gallery: HashMap::new() })
    }

    /// Load from file path.
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load SFace from {}", path.display()))?;
        Ok(Self { session, gallery: HashMap::new() })
    }

    /// Add a reference person to the gallery from a face image.
    /// Returns the number of reference images for this person.
    pub fn add_reference(
        &mut self,
        name: &str,
        face: &DynamicImage,
    ) -> anyhow::Result<usize> {
        let emb = self.embed(face)?;
        // Apply EMA to existing embedding if person already in gallery
        if let Some(existing) = self.gallery.get_mut(name) {
            let alpha = 0.3;
            for i in 0..Self::EMBEDDING_DIM {
                existing[i] = alpha * emb[i] + (1.0 - alpha) * existing[i];
            }
            // Re-normalise
            let n = l2_norm(existing);
            if n > 1e-8 {
                for v in existing.iter_mut() { *v /= n; }
            }
        } else {
            self.gallery.insert(name.to_string(), emb);
        }
        Ok(self.gallery.len())
    }

    /// Identify a face. Returns (name, cosine_similarity).
    /// Returns ("unknown", 0.0) if gallery is empty or no match above threshold.
    pub fn identify(&mut self, face: &DynamicImage, threshold: f32) -> anyhow::Result<(String, f32)> {
        let emb = self.embed(face)?;

        let mut best_name: Option<String> = None;
        let mut best_sim = threshold;

        for (name, ref_emb) in &self.gallery {
            let sim = dot_product(&emb, ref_emb);
            if sim > best_sim {
                best_sim = sim;
                best_name = Some(name.clone());
            }
        }

        match best_name {
            Some(name) => Ok((name, best_sim)),
            None => Ok(("unknown".to_string(), 0.0)),
        }
    }

    /// Extract 128-dim L2-normalised embedding from a face crop.
    pub fn embed(&mut self, face: &DynamicImage) -> anyhow::Result<Vec<f32>> {
        let input = self.preprocess(face)?;

        // SFace model takes 3 inputs: data (image), scalar_op1, scalar_op2
        // The scalar inputs are ignored (normalisation is model-internal).
        let scalar1 = ndarray::Array::from_elem(ndarray::IxDyn(&[1]), 0.0f32);
        let scalar2 = ndarray::Array::from_elem(ndarray::IxDyn(&[1]), 0.0f32);

        let outputs = self.session.run(ort::inputs![
            Self::INPUT_NAME => input,
            "scalar_op1" => Tensor::from_array(scalar1).context("scalar_op1")?,
            "scalar_op2" => Tensor::from_array(scalar2).context("scalar_op2")?,
        ])?;
        let emb: &[f32] = outputs[0].try_extract_tensor::<f32>()?.1;

        // L2-normalise
        let mut vec: Vec<f32> = emb[0..Self::EMBEDDING_DIM].to_vec();
        let n = l2_norm(&vec);
        if n > 1e-8 {
            for v in vec.iter_mut() { *v /= n; }
        }

        Ok(vec)
    }

    /// Number of people in gallery.
    pub fn gallery_size(&self) -> usize {
        self.gallery.len()
    }

    /// List gallery names.
    pub fn gallery_names(&self) -> Vec<&String> {
        self.gallery.keys().collect()
    }

    fn preprocess(&self, image: &DynamicImage) -> anyhow::Result<ort::value::Tensor<f32>> {
        let sz = Self::IMAGE_SIZE as usize;
        let resized = image.resize_exact(
            Self::IMAGE_SIZE,
            Self::IMAGE_SIZE,
            image::imageops::FilterType::Nearest,
        );

        // SFace preprocessing: [0,1] → (x - 0.5) / 0.5 = 2x - 1 → [-1, 1]
        let mut flat = vec![0.0f32; 3 * sz * sz];
        for y in 0..sz {
            for x in 0..sz {
                let px = resized.get_pixel(x as u32, y as u32);
                let base = (y * sz + x) * 3;
                flat[base]     = px[0] as f32 / 127.5 - 1.0; // R
                flat[base + 1] = px[1] as f32 / 127.5 - 1.0; // G
                flat[base + 2] = px[2] as f32 / 127.5 - 1.0; // B
            }
        }

        let shape = ndarray::IxDyn(&[1, 3, sz, sz]);
        let array = ndarray::Array::from_shape_vec(shape, flat)
            .context("Failed to create identity input array")?;
        Tensor::from_array(array).context("Failed to create identity tensor")
    }
}

impl Module for IdentityModule {
    type Input = Vec<Detection>;
    type Output = Vec<Detection>;

    fn init(config: &Config) -> anyhow::Result<Self> {
        let model_path = config.models_path.join("face_recognition_sface_2021dec.onnx");
        Self::from_path(&model_path)
    }

    fn process(&mut self, input: &Self::Input) -> anyhow::Result<Self::Output> {
        // Identity requires face crops — pass-through for now.
        // The pipeline will call identify() directly with face crops.
        Ok(input.clone())
    }

    fn metadata(&self) -> ModuleMetadata {
        ModuleMetadata {
            name: "mvm-identity (SFace)",
            version: env!("CARGO_PKG_VERSION"),
            accuracy: Some(0.995), // LFW accuracy
        }
    }
}

// ── Math helpers ──────────────────────────────────────────────

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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

    #[test]
    fn model_loads_and_embeds() {
        let mut id = IdentityModule::init(&test_config()).unwrap();
        assert!(id.metadata().name.contains("SFace"));

        let img = DynamicImage::new_rgb8(112, 112);
        let emb = id.embed(&img).unwrap();
        assert_eq!(emb.len(), 128);
        // L2 norm should be ≈ 1.0
        let n = l2_norm(&emb);
        assert!((n - 1.0).abs() < 0.01, "norm = {n}");
    }

    #[test]
    fn gallery_add_and_identify() {
        let mut id = IdentityModule::init(&test_config()).unwrap();

        let ref_path = std::path::PathBuf::from(
            std::env::var("HOME").unwrap_or_default(),
        )
        .join("repos/Machine Vision Module/models/references/p_V.jpg");

        if !ref_path.exists() {
            eprintln!("Skipping: {} not found", ref_path.display());
            return;
        }

        let img = image::open(&ref_path).unwrap();
        id.add_reference("Вадим", &img).unwrap();
        assert_eq!(id.gallery_size(), 1);

        let (name, sim) = id.identify(&img, 0.35).unwrap();
        eprintln!("Self-match: {name} ({sim:.4})");
        assert_eq!(name, "Вадим");
        assert!(sim > 0.5, "Self-similarity too low: {sim}");
    }

    #[test]
    fn unknown_when_empty_gallery() {
        let mut id = IdentityModule::init(&test_config()).unwrap();
        let img = DynamicImage::new_rgb8(112, 112);
        let (name, sim) = id.identify(&img, 0.35).unwrap();
        assert_eq!(name, "unknown");
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn embedding_is_reproducible() {
        let mut id = IdentityModule::init(&test_config()).unwrap();
        let img = DynamicImage::new_rgb8(112, 112);
        let emb1 = id.embed(&img).unwrap();
        let emb2 = id.embed(&img).unwrap();
        for i in 0..128 {
            assert!((emb1[i] - emb2[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn dot_product_unit_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        assert!((dot_product(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![0.7071_f32, 0.7071, 0.0];
        let sim = dot_product(&a, &c);
        assert!((sim - 0.7071).abs() < 0.01, "dot = {sim}");
    }
}
