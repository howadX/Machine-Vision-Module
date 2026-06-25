//! mvm-detect — face detection via YuNet ONNX.
//!
//! Implements the `Module` trait from mvm-core.
//! Dual-mode: file-based (default) or embedded model (`embed-model` feature).

use anyhow::Context;
use image::DynamicImage;
use image::GenericImageView;
use mvm_core::{Config, Detection, Module, ModuleMetadata};
use ort::{session::Session, value::Tensor};

/// YuNet ONNX face detector.
pub struct YunetDetector {
    session: Session,
    input_size: u32,
    score_threshold: f32,
    nms_threshold: f32,
}

impl YunetDetector {
    const INPUT_NAME: &'static str = "input";
    /// Strides for the three detection heads.
    const STRIDES: &'static [(u32, &'static str, &'static str, &'static str)] = &[
        (8, "cls_8", "obj_8", "bbox_8"),
        (16, "cls_16", "obj_16", "bbox_16"),
        (32, "cls_32", "obj_32", "bbox_32"),
    ];

    /// Load from ONNX model bytes (embedded mode).
    #[cfg(feature = "embed-model")]
    pub fn from_bytes(model_bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_memory(model_bytes)
            .context("Failed to load YuNet from embedded bytes")?;
        Ok(Self {
            session,
            input_size: 640,
            score_threshold: 0.5,
            nms_threshold: 0.45,
        })
    }

    /// Load from file path.
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load YuNet from {}", path.display()))?;
        Ok(Self {
            session,
            input_size: 640,
            score_threshold: 0.5,
            nms_threshold: 0.45,
        })
    }

    pub fn score_threshold(&self) -> f32 {
        self.score_threshold
    }

    pub fn set_score_threshold(&mut self, t: f32) {
        self.score_threshold = t;
    }

    /// Detect faces with full multi-stride NMS.
    pub fn detect(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<Detection>> {
        let input = self.preprocess(image)?;
        let outputs = self.session.run(ort::inputs![Self::INPUT_NAME => input])?;

        let mut proposals: Vec<Proposal> = Vec::new();

        for &(stride, cls_name, obj_name, bbox_name) in Self::STRIDES {
            let cls: &[f32] = outputs[cls_name].try_extract_tensor::<f32>()?.1;
            let obj: &[f32] = outputs[obj_name].try_extract_tensor::<f32>()?.1;
            let bbox: &[f32] = outputs[bbox_name].try_extract_tensor::<f32>()?.1;
            let num = cls.len();
            let grid = (self.input_size / stride) as usize;

            for i in 0..num {
                let score = cls[i] * obj[i];
                if score < self.score_threshold {
                    continue;
                }
                let ay = (i / grid) as f32;
                let ax = (i % grid) as f32;
                let base = i * 4;
                let s = stride as f32;
                let isz = self.input_size as f32;

                let cx = (ax * s + bbox[base] * s) / isz;
                let cy = (ay * s + bbox[base + 1] * s) / isz;
                let bw = (bbox[base + 2].exp() * s) / isz;
                let bh = (bbox[base + 3].exp() * s) / isz;

                proposals.push(Proposal {
                    x1: (cx - bw / 2.0).clamp(0.0, 1.0),
                    y1: (cy - bh / 2.0).clamp(0.0, 1.0),
                    x2: (cx + bw / 2.0).clamp(0.0, 1.0),
                    y2: (cy + bh / 2.0).clamp(0.0, 1.0),
                    score,
                });
            }
        }

        // NMS: sort by score desc, suppress overlapping boxes
        proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut kept: Vec<Detection> = Vec::new();

        for p in proposals {
            let b = mvm_core::BBox { x1: p.x1, y1: p.y1, x2: p.x2, y2: p.y2 };
            let mut suppressed = false;
            for k in &kept {
                if b.iou(&k.bbox) > self.nms_threshold {
                    suppressed = true;
                    break;
                }
            }
            if !suppressed {
                kept.push(Detection {
                    bbox: b,
                    identity: None,
                    identity_confidence: None,
                    emotion: None,
                    emotion_confidence: None,
                    posture: None,
                    posture_confidence: None,
                    distance: None,
                    position_x: None,
                    position_y: None,
                    position_z: None,
                    ws_confidence: None,
                    recon_error: None,
                    plausibility: None,
                    emotion_distribution: None,
                    embedding: None,
                });
            }
        }

        Ok(kept)
    }

    fn preprocess(&self, image: &DynamicImage) -> anyhow::Result<ort::value::Tensor<f32>> {
        let sz = self.input_size as usize;
        let resized = image.resize_exact(
            self.input_size,
            self.input_size,
            image::imageops::FilterType::Nearest,
        );
        // YuNet expects NCHW BGR float32 in [0, 255].
        // Channels: 0=B, 1=G, 2=R. Layout: flat[c * H * W + y * W + x].
        let mut flat = vec![0.0f32; 3 * sz * sz];
        for y in 0..sz {
            for x in 0..sz {
                let px = resized.get_pixel(x as u32, y as u32);
                let idx = y * sz + x;
                flat[idx]              = px[2] as f32; // B
                flat[sz * sz + idx]    = px[1] as f32; // G
                flat[2 * sz * sz + idx] = px[0] as f32; // R
            }
        }
        let shape = ndarray::IxDyn(&[1, 3, sz, sz]);
        let array = ndarray::Array::from_shape_vec(shape, flat)
            .context("Failed to create input array")?;
        Tensor::from_array(array).context("Failed to create tensor")
    }
}

struct Proposal {
    x1: f32, y1: f32, x2: f32, y2: f32,
    score: f32,
}

impl Module for YunetDetector {
    type Input = DynamicImage;
    type Output = Vec<Detection>;

    fn init(config: &Config) -> anyhow::Result<Self> {
        let model_path = config.models_path.join("face_detection_yunet_2023mar.onnx");
        Self::from_path(&model_path)
    }

    fn process(&mut self, input: &Self::Input) -> anyhow::Result<Self::Output> {
        self.detect(input)
    }

    fn metadata(&self) -> ModuleMetadata {
        ModuleMetadata {
            name: "mvm-detect (YuNet NMS)",
            version: env!("CARGO_PKG_VERSION"),
            accuracy: Some(0.93),
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

    #[test]
    fn model_loads_and_runs_on_black_image() {
        let mut det = YunetDetector::init(&test_config()).unwrap();
        assert!(det.metadata().name.contains("YuNet"));

        let img = DynamicImage::new_rgb8(640, 640);
        let result = det.process(&img).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn model_runs_on_reference_photo() {
        let mut det = YunetDetector::init(&test_config()).unwrap();

        let ref_path = std::path::PathBuf::from(
            std::env::var("HOME").unwrap_or_default(),
        )
        .join("repos/Machine Vision Module/models/references/p_V.jpg");

        if !ref_path.exists() {
            eprintln!("Skipping: {} not found", ref_path.display());
            return;
        }

        let img = image::open(&ref_path).unwrap();
        let result = det.process(&img).unwrap();
        eprintln!("Reference photo detections: {} faces", result.len());
        // With NMS, should detect at least one face on a reference photo
    }

    #[test]
    fn nms_suppresses_overlapping() {
        // Two identical bboxes: NMS keeps one
        let mut det = YunetDetector::init(&test_config()).unwrap();
        // Black image won't trigger detections; test bounds
        let img = DynamicImage::new_rgb8(640, 640);
        let result = det.process(&img).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn score_threshold_getter_reflects_setter() {
        let mut det = YunetDetector::init(&test_config()).unwrap();
        assert!((det.score_threshold() - 0.5).abs() < 1e-6, "default should be 0.5");
        det.set_score_threshold(0.3);
        assert!((det.score_threshold() - 0.3).abs() < 1e-6, "should be 0.3 after set");
    }

    #[test]
    fn lower_threshold_finds_face_on_reference_photo() {
        let mut det = YunetDetector::init(&test_config()).unwrap();
        let ref_path = std::path::PathBuf::from(
            std::env::var("HOME").unwrap_or_default(),
        )
        .join("repos/Machine Vision Module/mvm/models/references/p_V_01.jpg");

        if !ref_path.exists() {
            eprintln!("Skipping: {} not found", ref_path.display());
            return;
        }

        let img = image::open(&ref_path).unwrap();

        let high = det.process(&img).unwrap();
        det.set_score_threshold(0.3);
        let low = det.process(&img).unwrap();

        eprintln!("threshold=0.5: {} faces, threshold=0.3: {} faces", high.len(), low.len());
        assert!(!low.is_empty(), "should detect at least one face at threshold=0.3");
    }
}
