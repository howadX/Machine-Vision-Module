//! mvm-scene — object detection via an ultralytics YOLO ONNX model (80-class COCO).
//!
//! Phase 8.E.1 (G₂ Scene group). Produces `EntityKind::Object(class)` scene-graph
//! nodes so the existing object-agnostic `build_scene` (mvm-fusion) wires
//! person↔object relations — the "context outward from the human" signal.
//!
//! Model-agnostic by design: any ultralytics anchor-free export (YOLOv8 / v11 /
//! **v26**) shares the same output tensor layout `[1, 4+num_classes, num_anchors]`
//! (channel-major, 4 box coords + per-class scores, NO separate objectness). The
//! bootstrap weights are YOLOv8n; swapping in YOLO26n is a file replacement.
//!
//! The decode / letterbox / NMS math is split into pure functions so the chain is
//! fully unit-tested on synthetic tensors with no model file present.

use anyhow::Context;
use image::DynamicImage;
use image::GenericImageView;
use mvm_core::BBox;
use ort::{session::Session, value::Tensor};

/// COCO 80-class names, in the canonical ultralytics order.
pub const COCO_CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
];

/// One detected object. `bbox` is normalized [0,1] in the **original** image frame
/// (letterbox padding already undone), consistent with every other MVM detector.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectDetection {
    pub class_id: usize,
    pub class_name: &'static str,
    pub bbox: BBox,
    pub score: f32,
}

/// Letterbox transform: how the original WxH maps onto a square `target`xtarget
/// canvas (aspect-preserving resize + centred grey padding, the ultralytics
/// convention). `scale` is original→canvas; `pad_x`/`pad_y` are the left/top
/// borders in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

/// Compute the letterbox mapping for an original image onto `target`x`target`.
pub fn compute_letterbox(orig_w: u32, orig_h: u32, target: u32) -> Letterbox {
    let t = target as f32;
    let scale = (t / orig_w as f32).min(t / orig_h as f32);
    let new_w = orig_w as f32 * scale;
    let new_h = orig_h as f32 * scale;
    Letterbox {
        scale,
        pad_x: (t - new_w) * 0.5,
        pad_y: (t - new_h) * 0.5,
    }
}

/// A raw decoded box in **canvas** (letterboxed input) pixel space, corner form.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RawBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: usize,
}

/// IoU of two corner-form boxes (any consistent coordinate space).
fn iou_xyxy(a: &RawBox, b: &RawBox) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    if inter == 0.0 {
        return 0.0;
    }
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    inter / (area_a + area_b - inter).max(1e-10)
}

/// Decode a raw ultralytics output tensor into candidate boxes (canvas pixels).
///
/// `raw` is channel-major flat `[num_attrs * num_anchors]`: attribute `c`, anchor
/// `a` lives at `raw[c * num_anchors + a]`. Attributes 0..4 are `cx,cy,w,h` (in
/// canvas pixels); attributes 4.. are per-class scores (post-sigmoid in [0,1]).
/// For each anchor the best class is taken; below `score_thresh` it is dropped.
fn decode_yolo(raw: &[f32], num_classes: usize, score_thresh: f32) -> Vec<RawBox> {
    let num_attrs = num_classes + 4;
    debug_assert_eq!(raw.len() % num_attrs, 0, "tensor length not divisible by attrs");
    let num_anchors = raw.len() / num_attrs;
    let mut out = Vec::new();

    for a in 0..num_anchors {
        // Best class for this anchor.
        let mut best_score = score_thresh;
        let mut best_class = None;
        for k in 0..num_classes {
            let s = raw[(4 + k) * num_anchors + a];
            if s > best_score {
                best_score = s;
                best_class = Some(k);
            }
        }
        let Some(class_id) = best_class else { continue };

        let cx = raw[a];
        let cy = raw[num_anchors + a];
        let w = raw[2 * num_anchors + a];
        let h = raw[3 * num_anchors + a];
        out.push(RawBox {
            x1: cx - w * 0.5,
            y1: cy - h * 0.5,
            x2: cx + w * 0.5,
            y2: cy + h * 0.5,
            score: best_score,
            class_id,
        });
    }
    out
}

/// Class-aware greedy NMS: sort by score desc, suppress same-class overlaps.
fn nms(mut boxes: Vec<RawBox>, iou_thresh: f32) -> Vec<RawBox> {
    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut kept: Vec<RawBox> = Vec::new();
    for cand in boxes {
        let suppressed = kept.iter().any(|k| {
            k.class_id == cand.class_id && iou_xyxy(k, &cand) > iou_thresh
        });
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

/// Map a canvas-pixel box back to a normalized [0,1] bbox in the original frame.
fn canvas_to_norm(b: &RawBox, lb: &Letterbox, orig_w: u32, orig_h: u32) -> BBox {
    let undo = |v: f32, pad: f32, dim: u32| (((v - pad) / lb.scale) / dim as f32).clamp(0.0, 1.0);
    BBox {
        x1: undo(b.x1, lb.pad_x, orig_w),
        y1: undo(b.y1, lb.pad_y, orig_h),
        x2: undo(b.x2, lb.pad_x, orig_w),
        y2: undo(b.y2, lb.pad_y, orig_h),
    }
}

/// YOLO ONNX object detector (ultralytics anchor-free family).
pub struct YoloDetector {
    session: Session,
    input_size: u32,
    score_threshold: f32,
    nms_threshold: f32,
}

impl YoloDetector {
    const INPUT_NAME: &'static str = "images";

    /// Load from a file path.
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load YOLO from {}", path.display()))?;
        Ok(Self {
            session,
            input_size: 640,
            score_threshold: 0.35,
            nms_threshold: 0.45,
        })
    }

    /// Load from ONNX model bytes (embedded mode).
    #[cfg(feature = "embed-model")]
    pub fn from_bytes(model_bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_memory(model_bytes)
            .context("Failed to load YOLO from embedded bytes")?;
        Ok(Self {
            session,
            input_size: 640,
            score_threshold: 0.35,
            nms_threshold: 0.45,
        })
    }

    pub fn score_threshold(&self) -> f32 {
        self.score_threshold
    }

    pub fn set_score_threshold(&mut self, t: f32) {
        self.score_threshold = t;
    }

    /// Detect objects. Returns normalized [0,1] bboxes in the original frame.
    pub fn detect(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<ObjectDetection>> {
        let (orig_w, orig_h) = image.dimensions();
        let lb = compute_letterbox(orig_w, orig_h, self.input_size);
        let input = self.preprocess(image, &lb)?;
        let outputs = self.session.run(ort::inputs![Self::INPUT_NAME => input])?;

        let raw: &[f32] = outputs[0].try_extract_tensor::<f32>()?.1;
        let proposals = decode_yolo(raw, COCO_CLASSES.len(), self.score_threshold);
        let kept = nms(proposals, self.nms_threshold);

        Ok(kept
            .into_iter()
            .map(|b| ObjectDetection {
                class_id: b.class_id,
                class_name: COCO_CLASSES[b.class_id],
                bbox: canvas_to_norm(&b, &lb, orig_w, orig_h),
                score: b.score,
            })
            .collect())
    }

    /// Letterbox + RGB + NCHW + [0,1] normalization (ultralytics preprocessing).
    fn preprocess(
        &self,
        image: &DynamicImage,
        lb: &Letterbox,
    ) -> anyhow::Result<Tensor<f32>> {
        let sz = self.input_size as usize;
        let (orig_w, orig_h) = image.dimensions();
        let new_w = (orig_w as f32 * lb.scale).round() as u32;
        let new_h = (orig_h as f32 * lb.scale).round() as u32;
        let resized = image.resize_exact(
            new_w.max(1),
            new_h.max(1),
            image::imageops::FilterType::Triangle,
        );
        let resized = resized.to_rgb8();

        // Grey (114) padded canvas, NCHW, channels R,G,B, normalized to [0,1].
        let pad_val = 114.0f32 / 255.0;
        let mut flat = vec![pad_val; 3 * sz * sz];
        let off_x = lb.pad_x.round() as usize;
        let off_y = lb.pad_y.round() as usize;
        for y in 0..(new_h as usize).min(sz.saturating_sub(off_y)) {
            for x in 0..(new_w as usize).min(sz.saturating_sub(off_x)) {
                let px = resized.get_pixel(x as u32, y as u32);
                let cy = off_y + y;
                let cx = off_x + x;
                let idx = cy * sz + cx;
                flat[idx] = px[0] as f32 / 255.0; // R
                flat[sz * sz + idx] = px[1] as f32 / 255.0; // G
                flat[2 * sz * sz + idx] = px[2] as f32 / 255.0; // B
            }
        }
        let shape = ndarray::IxDyn(&[1, 3, sz, sz]);
        let array = ndarray::Array::from_shape_vec(shape, flat)
            .context("Failed to create input array")?;
        Tensor::from_array(array).context("Failed to create tensor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── letterbox (pure) ───────────────────────────────────────────────────

    #[test]
    fn letterbox_square_is_identity_scale() {
        let lb = compute_letterbox(640, 640, 640);
        assert!((lb.scale - 1.0).abs() < 1e-6);
        assert!(lb.pad_x.abs() < 1e-6 && lb.pad_y.abs() < 1e-6);
    }

    #[test]
    fn letterbox_landscape_pads_vertically() {
        // 640x480 → scale 1.0, height 480 → pad top/bottom (640-480)/2 = 80.
        let lb = compute_letterbox(640, 480, 640);
        assert!((lb.scale - 1.0).abs() < 1e-6);
        assert!(lb.pad_x.abs() < 1e-6);
        assert!((lb.pad_y - 80.0).abs() < 1e-6);
    }

    #[test]
    fn letterbox_portrait_pads_horizontally() {
        // 480x640 → scale 1.0 → pad left/right 80.
        let lb = compute_letterbox(480, 640, 640);
        assert!((lb.pad_x - 80.0).abs() < 1e-6);
        assert!(lb.pad_y.abs() < 1e-6);
    }

    // ── canvas_to_norm inverse (pure) ──────────────────────────────────────

    #[test]
    fn canvas_to_norm_undoes_letterbox_padding() {
        // 640x480 letterboxed into 640: a box that in canvas spans y 80..560
        // (the full image band) should map back to the full original height.
        let lb = compute_letterbox(640, 480, 640);
        let b = RawBox { x1: 0.0, y1: 80.0, x2: 640.0, y2: 560.0, score: 0.9, class_id: 0 };
        let n = canvas_to_norm(&b, &lb, 640, 480);
        assert!((n.x1 - 0.0).abs() < 1e-4 && (n.x2 - 1.0).abs() < 1e-4);
        assert!((n.y1 - 0.0).abs() < 1e-4 && (n.y2 - 1.0).abs() < 1e-4, "y={:?}", (n.y1, n.y2));
    }

    // ── decode (pure, synthetic tensor — no model file) ────────────────────

    /// Build a channel-major raw tensor for `num_anchors` anchors, `num_classes`
    /// classes. Anchor 0 gets a strong detection of `hot_class`; the rest are noise.
    fn synth_raw(num_anchors: usize, num_classes: usize, hot_class: usize) -> Vec<f32> {
        let num_attrs = num_classes + 4;
        let mut raw = vec![0.0f32; num_attrs * num_anchors];
        let put = |raw: &mut Vec<f32>, attr: usize, anchor: usize, v: f32| {
            raw[attr * num_anchors + anchor] = v;
        };
        // Anchor 0: box centred at (100,100), 40x40, hot_class score 0.9.
        put(&mut raw, 0, 0, 100.0);
        put(&mut raw, 1, 0, 100.0);
        put(&mut raw, 2, 0, 40.0);
        put(&mut raw, 3, 0, 40.0);
        put(&mut raw, 4 + hot_class, 0, 0.9);
        raw
    }

    #[test]
    fn decode_picks_best_class_above_threshold() {
        let raw = synth_raw(10, 80, 42);
        let boxes = decode_yolo(&raw, 80, 0.35);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].class_id, 42);
        assert!((boxes[0].score - 0.9).abs() < 1e-6);
        // corner form of centre (100,100) size 40 → 80..120.
        assert!((boxes[0].x1 - 80.0).abs() < 1e-4);
        assert!((boxes[0].x2 - 120.0).abs() < 1e-4);
    }

    #[test]
    fn decode_drops_everything_below_threshold() {
        let mut raw = synth_raw(10, 80, 42);
        // Lower the only hot score under threshold.
        raw[(4 + 42) * 10 + 0] = 0.10;
        let boxes = decode_yolo(&raw, 80, 0.35);
        assert!(boxes.is_empty());
    }

    #[test]
    fn decode_maps_class_to_coco_name() {
        let raw = synth_raw(5, 80, 56); // 56 = "chair"
        let boxes = decode_yolo(&raw, 80, 0.35);
        assert_eq!(COCO_CLASSES[boxes[0].class_id], "chair");
    }

    // ── NMS (pure) ─────────────────────────────────────────────────────────

    fn rb(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: usize) -> RawBox {
        RawBox { x1, y1, x2, y2, score, class_id }
    }

    #[test]
    fn nms_suppresses_same_class_overlap() {
        let boxes = vec![
            rb(0.0, 0.0, 40.0, 40.0, 0.9, 1),
            rb(2.0, 2.0, 42.0, 42.0, 0.8, 1), // ~heavy overlap, same class → suppressed
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_overlapping_different_classes() {
        let boxes = vec![
            rb(0.0, 0.0, 40.0, 40.0, 0.9, 1),
            rb(2.0, 2.0, 42.0, 42.0, 0.8, 2), // same box, different class → both kept
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_keeps_disjoint_same_class() {
        let boxes = vec![
            rb(0.0, 0.0, 40.0, 40.0, 0.9, 1),
            rb(100.0, 100.0, 140.0, 140.0, 0.8, 1),
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2);
    }

    // ── coco table sanity ──────────────────────────────────────────────────

    #[test]
    fn coco_table_is_80_classes() {
        assert_eq!(COCO_CLASSES.len(), 80);
        assert_eq!(COCO_CLASSES[0], "person");
        assert_eq!(COCO_CLASSES[79], "toothbrush");
    }

    // ── end-to-end model load (gated on file present, like mvm-detect) ──────

    #[test]
    fn model_loads_and_runs_when_present() {
        let model_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/yolo_object_detect.onnx"
        ));
        if !model_path.exists() {
            eprintln!("Skipping: {} not found (download bootstrap weights)", model_path.display());
            return;
        }
        let mut det = YoloDetector::from_path(&model_path).unwrap();
        let img = DynamicImage::new_rgb8(640, 480);
        let result = det.detect(&img).unwrap();
        eprintln!("black image → {} objects", result.len());
        // Black image: typically zero detections, but the chain must not panic.
    }
}
