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

    /// Identify with margin gating (phase 7.5, fixes #5).
    ///
    /// Returns `(name | "unknown", best_similarity, embedding)`. The embedding is
    /// **always** returned — even for "unknown" — so the tracker can cosine
    /// re-link any face (#6), not just gallery members. Embeds once; the label
    /// decision is delegated to the pure [`accept_identity`].
    pub fn identify_margin(
        &mut self,
        face: &DynamicImage,
        abs_thresh: f32,
        margin_thresh: f32,
    ) -> anyhow::Result<(String, f32, Vec<f32>)> {
        let emb = self.embed(face)?;

        // top-1 and top-2 similarities across the gallery.
        let mut best_name: Option<String> = None;
        let mut best_sim = f32::MIN;
        let mut second_sim = f32::MIN;
        for (name, ref_emb) in &self.gallery {
            let sim = dot_product(&emb, ref_emb);
            if sim > best_sim {
                second_sim = best_sim;
                best_sim = sim;
                best_name = Some(name.clone());
            } else if sim > second_sim {
                second_sim = sim;
            }
        }
        let best_sim = if best_sim == f32::MIN { 0.0 } else { best_sim };
        let second_sim = if second_sim == f32::MIN { 0.0 } else { second_sim };

        let accepted = best_name.is_some()
            && accept_identity(best_sim, second_sim, self.gallery.len(), abs_thresh, margin_thresh);
        let name = if accepted {
            best_name.unwrap()
        } else {
            "unknown".to_string()
        };
        Ok((name, best_sim, emb))
    }

    /// Extract 128-dim L2-normalised embedding from a face crop.
    pub fn embed(&mut self, face: &DynamicImage) -> anyhow::Result<Vec<f32>> {
        let input = self.preprocess(face)?;

        // The model normalises internally: `(data - scalar_op1) * scalar_op2`,
        // with baked-in initializers scalar_op1=127.5 (mean), scalar_op2=1/128
        // (scale) — i.e. it expects RAW [0,255] input. We must NOT feed the
        // scalars ourselves: the old code passed 0.0 for both, so the graph
        // computed `(data - 0) * 0 = 0`, zeroing every input → a constant
        // embedding that matched any face at cosine ≈ 1.0 (the #5 disaster).
        // Omitting them lets ort use the initializer defaults, exactly as it
        // already does for the conv weights.
        let outputs = self.session.run(ort::inputs![
            Self::INPUT_NAME => input,
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
            image::imageops::FilterType::Triangle,
        );

        // SFace (OpenCV 2021dec) expects a `blobFromImage`-style tensor:
        // **planar CHW**, **BGR** channel order, **raw [0,255]** (scale 1, no mean).
        //
        // The previous code wrote interleaved HWC data into a CHW-declared tensor
        // AND used RGB + [-1,1] normalisation — three mismatches that scrambled the
        // input so badly the network saturated to a near-constant embedding
        // (every face matched the gallery at cosine ≈ 1.0). This is the planar
        // layout mvm-detect already uses for YuNet.
        let plane = sz * sz;
        let mut flat = vec![0.0f32; 3 * plane];
        for y in 0..sz {
            for x in 0..sz {
                let px = resized.get_pixel(x as u32, y as u32);
                let idx = y * sz + x;
                flat[idx]             = px[2] as f32; // B plane
                flat[plane + idx]     = px[1] as f32; // G plane
                flat[2 * plane + idx] = px[0] as f32; // R plane
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

/// Decide whether to accept the top gallery match as an identity label
/// (phase 7.5, fixes #5). Pure — unit-testable without the ONNX model.
///
/// Accept iff `best ≥ abs_thresh` AND (gallery has <2 people OR
/// `best − second ≥ margin_thresh`). The absolute gate rejects strangers under
/// a single-name gallery (live data: real ≈1.00 vs stranger ≈0.66); the margin
/// gate rejects ambiguous near-ties once the gallery holds ≥2 people.
pub fn accept_identity(
    best_sim: f32,
    second_sim: f32,
    gallery_size: usize,
    abs_thresh: f32,
    margin_thresh: f32,
) -> bool {
    if best_sim < abs_thresh {
        return false;
    }
    if gallery_size >= 2 && (best_sim - second_sim) < margin_thresh {
        return false;
    }
    true
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

    // ── Pure margin-identity decision (phase 7.5, #5) — no model needed ──

    #[test]
    fn accept_clear_winner_single_gallery() {
        // Real Вадим ≈1.00 under a single-name gallery → accept.
        assert!(accept_identity(1.00, 0.0, 1, 0.8, 0.15));
    }

    #[test]
    fn reject_stranger_single_gallery() {
        // Stranger ≈0.66 under a single-name gallery → unknown (fixes #5).
        assert!(!accept_identity(0.66, 0.0, 1, 0.8, 0.15));
    }

    #[test]
    fn reject_ambiguous_near_tie_two_gallery() {
        // Two close matches (margin 0.05 < 0.15) → unknown.
        assert!(!accept_identity(0.85, 0.80, 2, 0.8, 0.15));
    }

    #[test]
    fn accept_clear_gap_two_gallery() {
        // Clear winner (margin 0.30 ≥ 0.15) above abs → accept.
        assert!(accept_identity(0.90, 0.60, 2, 0.8, 0.15));
    }

    #[test]
    fn reject_below_abs_even_with_margin() {
        // Big margin but below abs threshold → still unknown.
        assert!(!accept_identity(0.70, 0.10, 2, 0.8, 0.15));
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

    /// Regression for the preprocess scramble bug (phase 7.5 native path): a
    /// degenerate embedding makes *every* input match at cosine ≈ 1.0, which
    /// `gallery_add_and_identify` (self-match only) cannot catch. A real face and
    /// a synthetic non-face must NOT collapse to the same vector.
    #[test]
    fn embeddings_discriminate_face_from_nonface() {
        let mut id = IdentityModule::init(&test_config()).unwrap();
        let ref_path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("repos/Machine Vision Module/models/references/p_V.jpg");
        if !ref_path.exists() {
            eprintln!("Skipping: {} not found", ref_path.display());
            return;
        }
        let face = image::open(&ref_path).unwrap();
        let e_face = id.embed(&face).unwrap();

        // Synthetic non-face: diagonal gradient — structurally unlike a face.
        let mut img = image::RgbImage::new(112, 112);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x as u8).wrapping_mul(2), (y as u8).wrapping_mul(2), 64]);
        }
        let e_non = id.embed(&image::DynamicImage::ImageRgb8(img)).unwrap();

        let cos: f32 = e_face.iter().zip(&e_non).map(|(a, b)| a * b).sum();
        eprintln!("face vs non-face cosine = {cos:.4}");
        assert!(
            cos < 0.6,
            "degenerate embedding: face and non-face match at {cos:.4} (preprocess scramble)"
        );
    }

    /// The Rust port must agree with OpenCV's own SFace: two photos of the same
    /// person (unaligned 112×112) land at cosine ≈ 0.5 (OpenCV reference: 0.49).
    /// Confirms the embedding is correct, not merely non-degenerate.
    #[test]
    fn embeddings_match_same_person() {
        let mut id = IdentityModule::init(&test_config()).unwrap();
        let dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("repos/Machine Vision Module/models/references");
        let (a, b) = (dir.join("p_V.jpg"), dir.join("p_V_01.jpg"));
        if !a.exists() || !b.exists() {
            eprintln!("Skipping: reference photos not found");
            return;
        }
        let ea = id.embed(&image::open(&a).unwrap()).unwrap();
        let eb = id.embed(&image::open(&b).unwrap()).unwrap();
        let cos: f32 = ea.iter().zip(&eb).map(|(x, y)| x * y).sum();
        eprintln!("same-person (different angle) cosine = {cos:.4}");
        // Above OpenCV's canonical SFace threshold (0.363), below self-match.
        assert!(cos > 0.363, "same-person cosine {cos:.4} below SFace threshold");
        assert!(cos < 0.99, "different photos should not be near-identical");
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
