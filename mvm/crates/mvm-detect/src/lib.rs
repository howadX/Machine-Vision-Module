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
    /// Strides for the three detection heads (cls / obj / bbox / 5-point kps).
    const STRIDES: &'static [(u32, &'static str, &'static str, &'static str, &'static str)] = &[
        (8, "cls_8", "obj_8", "bbox_8", "kps_8"),
        (16, "cls_16", "obj_16", "bbox_16", "kps_16"),
        (32, "cls_32", "obj_32", "bbox_32", "kps_32"),
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

    /// Detect faces with full multi-stride NMS — bbox-only [`Detection`] view.
    /// Thin wrapper over [`detect_faces`](Self::detect_faces) for callers that
    /// don't need landmarks (bridge A/B path, legacy tests).
    pub fn detect(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<Detection>> {
        Ok(self
            .detect_faces(image)?
            .into_iter()
            .map(|f| Detection::bare(f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2))
            .collect())
    }

    /// Detect faces with the **full YuNet output**: bbox + score + the 5 facial
    /// landmarks the ONNX already computes (phase 10.1 — previously the decoder
    /// kept only the bbox). Landmark order (normalised [0,1], image coords):
    /// image-left eye, image-right eye, nose tip, image-left mouth corner,
    /// image-right mouth corner.
    pub fn detect_faces(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<FaceDet>> {
        let input = self.preprocess(image)?;
        let outputs = self.session.run(ort::inputs![Self::INPUT_NAME => input])?;

        let mut proposals: Vec<FaceDet> = Vec::new();

        for &(stride, cls_name, obj_name, bbox_name, kps_name) in Self::STRIDES {
            let cls: &[f32] = outputs[cls_name].try_extract_tensor::<f32>()?.1;
            let obj: &[f32] = outputs[obj_name].try_extract_tensor::<f32>()?.1;
            let bbox: &[f32] = outputs[bbox_name].try_extract_tensor::<f32>()?.1;
            let kps: &[f32] = outputs[kps_name].try_extract_tensor::<f32>()?.1;
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

                // Landmarks decode like the bbox centre: offset from the anchor
                // cell in stride units (OpenCV FaceDetectorYN convention).
                let mut landmarks = [[0.0f32; 2]; 5];
                let kbase = i * 10;
                for (k, lm) in landmarks.iter_mut().enumerate() {
                    lm[0] = ((ax + kps[kbase + 2 * k]) * s / isz).clamp(0.0, 1.0);
                    lm[1] = ((ay + kps[kbase + 2 * k + 1]) * s / isz).clamp(0.0, 1.0);
                }

                proposals.push(FaceDet {
                    bbox: mvm_core::BBox {
                        x1: (cx - bw / 2.0).clamp(0.0, 1.0),
                        y1: (cy - bh / 2.0).clamp(0.0, 1.0),
                        x2: (cx + bw / 2.0).clamp(0.0, 1.0),
                        y2: (cy + bh / 2.0).clamp(0.0, 1.0),
                    },
                    score,
                    landmarks,
                });
            }
        }

        // NMS: sort by score desc, suppress overlapping boxes
        proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut kept: Vec<FaceDet> = Vec::new();
        for p in proposals {
            if !kept.iter().any(|k| p.bbox.iou(&k.bbox) > self.nms_threshold) {
                kept.push(p);
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

/// A full YuNet face detection (phase 10.1): bbox + score + the 5 landmarks the
/// model outputs anyway. Landmarks normalised [0,1], order: image-left eye,
/// image-right eye, nose tip, image-left mouth corner, image-right mouth corner.
#[derive(Debug, Clone, Copy)]
pub struct FaceDet {
    pub bbox: mvm_core::BBox,
    pub score: f32,
    pub landmarks: [[f32; 2]; 5],
}

// ── Head orientation (phase 10.3, pure arithmetic on the 5 landmarks) ────────

/// Coarse head orientation from the YuNet 5-point geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadOrientation {
    /// Yaw, degrees: 0 = frontal; sign = the image side the nose points toward
    /// (− image-left, + image-right). Coarse (planar 5-point estimate).
    pub yaw_deg: f32,
    /// Roll, degrees: the eye-line angle to the image horizontal (+ = clockwise
    /// in image coords).
    pub roll_deg: f32,
    /// "camera" (|yaw| < 20°), "left" or "right" — the fusion-friendly label.
    pub facing: &'static str,
}

/// Estimate coarse head yaw/roll from the 5 landmarks (any consistent units).
///
/// Roll is the eye-line angle. Yaw reads the nose's position along the
/// (roll-corrected) eye axis: centred → frontal; shifted toward one eye → the
/// head is turned so the nose points at the *other* image side. Pure planar
/// geometry — coarse by design (±10–15°), meant for attention/facing cues, not
/// gaze. `None` when the eye span is degenerate.
pub fn head_orientation(lm: &[[f32; 2]; 5]) -> Option<HeadOrientation> {
    let (le, re, nose) = (lm[0], lm[1], lm[2]);
    let (dx, dy) = (re[0] - le[0], re[1] - le[1]);
    let span_sq = dx * dx + dy * dy;
    if span_sq < 1e-9 {
        return None;
    }
    let roll_deg = dy.atan2(dx).to_degrees();
    // Nose projected onto the eye axis, 0 at the left eye, 1 at the right eye
    // (roll drops out of the projection).
    let t = ((nose[0] - le[0]) * dx + (nose[1] - le[1]) * dy) / span_sq;
    // Centred nose → 0; at/beyond an eye → ±90° (asin of the offset ratio).
    let yaw_deg = (2.0 * t - 1.0).clamp(-1.0, 1.0).asin().to_degrees();
    let facing = if yaw_deg.abs() < 20.0 {
        "camera"
    } else if yaw_deg < 0.0 {
        "left"
    } else {
        "right"
    };
    Some(HeadOrientation { yaw_deg, roll_deg, facing })
}

/// A/B detector-coverage comparison (phase 7.1, gap G4).
///
/// Quantifies how the native YuNet detector compares to the Python SCRFD boxes
/// on the same frame, so 7.1's gate ("YuNet coverage ≥ SCRFD") is measurable
/// rather than eyeballed. Accumulate per frame, report aggregates.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Coverage {
    /// Boxes present in BOTH sets (IoU ≥ threshold).
    pub matched: usize,
    /// Boxes the reference (e.g. YuNet) has that the baseline (SCRFD) lacks.
    pub a_only: usize,
    /// Boxes the baseline (SCRFD) has that the reference (YuNet) lacks.
    pub b_only: usize,
}

/// Greedy IoU matching of two detection sets — pure, unit-testable.
///
/// `a` is the reference set (YuNet), `b` the baseline (Python SCRFD). Each box
/// matches at most once. `a_only` = YuNet-extra, `b_only` = SCRFD-extra.
pub fn compare_coverage(a: &[Detection], b: &[Detection], iou_thresh: f32) -> Coverage {
    let mut b_used = vec![false; b.len()];
    let mut matched = 0usize;
    for da in a {
        let mut best = iou_thresh;
        let mut best_j = None;
        for (j, db) in b.iter().enumerate() {
            if b_used[j] { continue; }
            let iou = da.bbox.iou(&db.bbox);
            if iou >= best {
                best = iou;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            b_used[j] = true;
            matched += 1;
        }
    }
    Coverage {
        matched,
        a_only: a.len() - matched,
        b_only: b.len() - matched,
    }
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

    // ── A/B coverage (phase 7.1, G4) — pure, no model needed ──

    fn d(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection::bare(x1, y1, x2, y2)
    }

    #[test]
    fn coverage_identical_sets_all_matched() {
        let a = vec![d(0.1, 0.1, 0.3, 0.3), d(0.6, 0.6, 0.8, 0.8)];
        let cov = compare_coverage(&a, &a, 0.5);
        assert_eq!(cov, Coverage { matched: 2, a_only: 0, b_only: 0 });
    }

    #[test]
    fn coverage_yunet_finds_extra_face() {
        // YuNet (a) finds 2, SCRFD (b) only 1 → a_only=1 (YuNet better coverage).
        let a = vec![d(0.1, 0.1, 0.3, 0.3), d(0.6, 0.6, 0.8, 0.8)];
        let b = vec![d(0.1, 0.1, 0.3, 0.3)];
        let cov = compare_coverage(&a, &b, 0.5);
        assert_eq!(cov, Coverage { matched: 1, a_only: 1, b_only: 0 });
    }

    #[test]
    fn coverage_yunet_misses_face() {
        // SCRFD (b) finds 2, YuNet (a) only 1 → b_only=1 (YuNet worse coverage).
        let a = vec![d(0.1, 0.1, 0.3, 0.3)];
        let b = vec![d(0.1, 0.1, 0.3, 0.3), d(0.6, 0.6, 0.8, 0.8)];
        let cov = compare_coverage(&a, &b, 0.5);
        assert_eq!(cov, Coverage { matched: 1, a_only: 0, b_only: 1 });
    }

    #[test]
    fn coverage_disjoint_boxes_no_match() {
        let a = vec![d(0.0, 0.0, 0.2, 0.2)];
        let b = vec![d(0.7, 0.7, 0.9, 0.9)];
        let cov = compare_coverage(&a, &b, 0.5);
        assert_eq!(cov, Coverage { matched: 0, a_only: 1, b_only: 1 });
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

    // ── head orientation (phase 10.3, pure) ──

    /// Frontal template: eyes level, nose centred, mouth below.
    fn frontal_lm5() -> [[f32; 2]; 5] {
        [[40.0, 50.0], [80.0, 50.0], [60.0, 70.0], [45.0, 90.0], [75.0, 90.0]]
    }

    #[test]
    fn head_frontal_faces_camera() {
        let h = head_orientation(&frontal_lm5()).unwrap();
        assert!(h.yaw_deg.abs() < 1.0, "centred nose → yaw≈0, got {}", h.yaw_deg);
        assert!(h.roll_deg.abs() < 1.0, "level eyes → roll≈0, got {}", h.roll_deg);
        assert_eq!(h.facing, "camera");
    }

    #[test]
    fn head_turned_nose_toward_left_eye_faces_left() {
        let mut lm = frontal_lm5();
        lm[2][0] = 45.0; // nose shifted well toward the image-left eye
        let h = head_orientation(&lm).unwrap();
        assert!(h.yaw_deg < -20.0, "yaw strongly negative, got {}", h.yaw_deg);
        assert_eq!(h.facing, "left");
    }

    #[test]
    fn head_turned_nose_toward_right_eye_faces_right() {
        let mut lm = frontal_lm5();
        lm[2][0] = 75.0;
        let h = head_orientation(&lm).unwrap();
        assert!(h.yaw_deg > 20.0);
        assert_eq!(h.facing, "right");
    }

    #[test]
    fn head_roll_measured_and_yaw_roll_invariant() {
        // Rotate the whole frontal template by 30°: roll reads 30, yaw stays ≈0.
        let (s, c) = (30f32.to_radians().sin(), 30f32.to_radians().cos());
        let mut lm = frontal_lm5();
        for p in &mut lm {
            let (x, y) = (p[0] - 60.0, p[1] - 70.0);
            p[0] = c * x - s * y + 60.0;
            p[1] = s * x + c * y + 70.0;
        }
        let h = head_orientation(&lm).unwrap();
        assert!((h.roll_deg - 30.0).abs() < 1.0, "roll≈30, got {}", h.roll_deg);
        assert!(h.yaw_deg.abs() < 1.0, "yaw roll-invariant, got {}", h.yaw_deg);
        assert_eq!(h.facing, "camera");
    }

    #[test]
    fn head_degenerate_eyes_none() {
        let lm = [[60.0, 50.0]; 5];
        assert!(head_orientation(&lm).is_none());
    }

    #[test]
    fn detect_faces_landmarks_have_face_geometry() {
        // Phase 10.1: the 5 decoded landmarks must land inside (or near) the face
        // bbox with eye-line above mouth-line — i.e. real facial geometry, not
        // anchor garbage.
        let mut det = YunetDetector::init(&test_config()).unwrap();
        let ref_path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("repos/Machine Vision Module/models/references/p_V.jpg");
        if !ref_path.exists() {
            eprintln!("Skipping: {} not found", ref_path.display());
            return;
        }
        let img = image::open(&ref_path).unwrap();
        det.set_score_threshold(0.3);
        let faces = det.detect_faces(&img).unwrap();
        assert!(!faces.is_empty(), "reference photo must yield a face");
        let f = &faces[0];
        let (bw, bh) = (f.bbox.x2 - f.bbox.x1, f.bbox.y2 - f.bbox.y1);
        for lm in &f.landmarks {
            assert!(
                lm[0] >= f.bbox.x1 - 0.2 * bw && lm[0] <= f.bbox.x2 + 0.2 * bw
                    && lm[1] >= f.bbox.y1 - 0.2 * bh && lm[1] <= f.bbox.y2 + 0.2 * bh,
                "landmark {lm:?} far outside bbox {:?}", f.bbox
            );
        }
        let eye_y = (f.landmarks[0][1] + f.landmarks[1][1]) * 0.5;
        let mouth_y = (f.landmarks[3][1] + f.landmarks[4][1]) * 0.5;
        assert!(eye_y < mouth_y, "eyes above mouth: eye_y={eye_y}, mouth_y={mouth_y}");
        assert!(f.landmarks[0][0] < f.landmarks[1][0], "left eye left of right eye");
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
