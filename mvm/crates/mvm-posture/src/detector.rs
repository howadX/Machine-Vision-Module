//! Native port of OpenCV Zoo `person_detection_mediapipe` (BlazePose person detector).
//! Faithful to `spike_pose/mp_persondet.py`: letterbox 224² NCHW [-1,1], anchor decode,
//! IoU NMS. Produces person boxes + 4 keypoints (kp0 = mid-hip, kp1 = full-body point)
//! that drive the pose ROI.

use image::DynamicImage;
use ort::{session::Session, value::Tensor};

const INPUT: usize = 224;
const N_ANCHORS: usize = 2254;

/// Anchors as little-endian f32 [N,2] (cx,cy normalized). Generated from the
/// reference demo by `spike_pose/export_for_rust.py`.
static ANCHORS_BIN: &[u8] = include_bytes!("anchors_person.bin");

#[derive(Clone, Debug)]
pub struct Person {
    pub bbox: [f32; 4],          // x1,y1,x2,y2 in original-image pixels
    pub keypoints: [[f32; 2]; 4], // kp0=mid-hip, kp1=full-body, kp2=shoulder-center, kp3=upper-body
    pub score: f32,
}

pub struct PersonDetector {
    session: Session,
    anchors: Vec<[f32; 2]>,
    pub score_threshold: f32,
    pub nms_threshold: f32,
}

fn load_anchors() -> Vec<[f32; 2]> {
    assert_eq!(ANCHORS_BIN.len(), N_ANCHORS * 2 * 4, "anchors_person.bin size mismatch");
    let mut v = Vec::with_capacity(N_ANCHORS);
    for i in 0..N_ANCHORS {
        let o = i * 8;
        let ax = f32::from_le_bytes([ANCHORS_BIN[o], ANCHORS_BIN[o + 1], ANCHORS_BIN[o + 2], ANCHORS_BIN[o + 3]]);
        let ay = f32::from_le_bytes([ANCHORS_BIN[o + 4], ANCHORS_BIN[o + 5], ANCHORS_BIN[o + 6], ANCHORS_BIN[o + 7]]);
        v.push([ax, ay]);
    }
    v
}

impl PersonDetector {
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(Self { session, anchors: load_anchors(), score_threshold: 0.5, nms_threshold: 0.3 })
    }

    #[cfg(feature = "embed-model")]
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_memory(bytes)?;
        Ok(Self { session, anchors: load_anchors(), score_threshold: 0.5, nms_threshold: 0.3 })
    }

    /// Letterbox preprocess → CHW [1,3,224,224] in [-1,1]; returns (tensor, pad_x, pad_y, scale).
    fn preprocess(&self, image: &DynamicImage) -> anyhow::Result<(Tensor<f32>, f32, f32, f32)> {
        let rgb = image.to_rgb8();
        let (w, h) = (rgb.width() as f32, rgb.height() as f32);
        let ratio = (INPUT as f32 / h).min(INPUT as f32 / w);
        let rw = (w * ratio) as usize; // astype(int32) truncates
        let rh = (h * ratio) as usize;
        let left = (INPUT - rw) / 2;
        let top = (INPUT - rh) / 2;

        // CHW, init 0.0 (Python pads the normalized image with 0.0). Single bilinear
        // with cv2 INTER_LINEAR half-pixel mapping (supersampling here added cost
        // without net class-agreement gain — the pose stage is where it matters).
        let mut data = vec![0f32; 3 * INPUT * INPUT];
        let (sw, sh) = (rgb.width() as f32 / rw as f32, rgb.height() as f32 / rh as f32);
        for dy in 0..rh {
            let sy = ((dy as f32 + 0.5) * sh - 0.5).clamp(0.0, h - 1.0);
            let y0 = sy.floor() as u32;
            let y1 = (y0 + 1).min(rgb.height() - 1);
            let fy = sy - y0 as f32;
            for dx in 0..rw {
                let sx = ((dx as f32 + 0.5) * sw - 0.5).clamp(0.0, w - 1.0);
                let x0 = sx.floor() as u32;
                let x1 = (x0 + 1).min(rgb.width() - 1);
                let fx = sx - x0 as f32;
                let p00 = rgb.get_pixel(x0, y0).0;
                let p01 = rgb.get_pixel(x1, y0).0;
                let p10 = rgb.get_pixel(x0, y1).0;
                let p11 = rgb.get_pixel(x1, y1).0;
                for c in 0..3 {
                    let top_v = p00[c] as f32 * (1.0 - fx) + p01[c] as f32 * fx;
                    let bot_v = p10[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                    let v = top_v * (1.0 - fy) + bot_v * fy;
                    let norm = (v / 255.0 - 0.5) * 2.0;
                    let idx = c * INPUT * INPUT + (top + dy) * INPUT + (left + dx);
                    data[idx] = norm;
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, 3, INPUT, INPUT], data))?;
        let pad_x = (left as f32 / ratio) as i32 as f32;
        let pad_y = (top as f32 / ratio) as i32 as f32;
        let scale = w.max(h);
        Ok((tensor, pad_x, pad_y, scale))
    }

    /// Detect at the configured `score_threshold` (the phase 6–8 behaviour).
    pub fn detect(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<Person>> {
        let min_score = self.score_threshold;
        self.detect_scored(image, min_score)
    }

    /// Detect at an explicit `min_score` floor (phase 9.4 ByteTrack). Lowering it
    /// below `score_threshold` surfaces the weak boxes BYTE needs for recovery
    /// without touching the validated `detect()` path. NMS is unchanged, so the
    /// high-confidence kept set is identical to `detect()` — extra low boxes only
    /// add non-overlapping survivors.
    pub fn detect_scored(&mut self, image: &DynamicImage, min_score: f32) -> anyhow::Result<Vec<Person>> {
        let (input, pad_x, pad_y, scale) = self.preprocess(image)?;
        let outputs = self.session.run(ort::inputs!["input_1:0" => input])?;
        let boxes = outputs["Identity:0"].try_extract_tensor::<f32>()?.1; // [2254*12]
        let scores = outputs["Identity_1:0"].try_extract_tensor::<f32>()?.1; // [2254]

        let mut cand: Vec<Person> = Vec::new();
        for i in 0..N_ANCHORS {
            let raw = scores[i].clamp(-100.0, 100.0);
            let s = 1.0 / (1.0 + (-raw).exp());
            if s < min_score {
                continue;
            }
            let b = i * 12;
            let [ax, ay] = self.anchors[i];
            let cx = boxes[b] / INPUT as f32;
            let cy = boxes[b + 1] / INPUT as f32;
            let bw = boxes[b + 2] / INPUT as f32;
            let bh = boxes[b + 3] / INPUT as f32;
            let x1 = (cx - bw / 2.0 + ax) * scale - pad_x;
            let y1 = (cy - bh / 2.0 + ay) * scale - pad_y;
            let x2 = (cx + bw / 2.0 + ax) * scale - pad_x;
            let y2 = (cy + bh / 2.0 + ay) * scale - pad_y;
            let mut kp = [[0f32; 2]; 4];
            for k in 0..4 {
                kp[k][0] = (boxes[b + 4 + 2 * k] / INPUT as f32 + ax) * scale - pad_x;
                kp[k][1] = (boxes[b + 4 + 2 * k + 1] / INPUT as f32 + ay) * scale - pad_y;
            }
            cand.push(Person { bbox: [x1, y1, x2, y2], keypoints: kp, score: s });
        }
        Ok(nms(cand, self.nms_threshold))
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn nms(mut cand: Vec<Person>, thr: f32) -> Vec<Person> {
    cand.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep: Vec<Person> = Vec::new();
    for p in cand {
        if keep.iter().all(|k| iou(&k.bbox, &p.bbox) <= thr) {
            keep.push(p);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"))
    }

    fn detector() -> PersonDetector {
        PersonDetector::from_path(&models_dir().join("person_detection_mediapipe_2023mar.onnx")).unwrap()
    }

    #[test]
    fn anchors_load() {
        let a = load_anchors();
        assert_eq!(a.len(), N_ANCHORS);
        // anchors are normalized centers in [0,1]
        assert!(a.iter().all(|[x, y]| *x >= 0.0 && *x <= 1.0 && *y >= 0.0 && *y <= 1.0));
    }

    #[test]
    fn black_image_no_person() {
        let mut det = detector();
        let img = DynamicImage::new_rgb8(640, 480);
        let persons = det.detect(&img).unwrap();
        assert!(persons.is_empty(), "black image should yield no persons, got {}", persons.len());
    }

    #[test]
    fn detects_person_matches_oracle() {
        // Golden from spike_pose/export_for_rust.py on the same fixture (Python reference).
        // best_score 0.92, box [634.1,105.9,890.2,362.0], kp0(mid-hip)=(892,583), kp1(body)=(656,60)
        let mut det = detector();
        let img = image::open(models_dir().join("references/p_V_03.jpg")).unwrap();
        let persons = det.detect(&img).unwrap();
        assert!(!persons.is_empty(), "expected a person in p_V_03");
        let best = persons.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
        assert!(best.score > 0.8, "score {} too low", best.score);

        // Tolerance ~2.5% of the 1280px frame: the Rust JPEG decoder + bilinear resize
        // differ sub-pixel from OpenCV, which the keypoint regression mildly amplifies.
        // The decisive oracle is the 33-landmark pose match (pose_oracle test).
        let gbox = [634.1, 105.9, 890.2, 362.0];
        for i in 0..4 {
            assert!((best.bbox[i] - gbox[i]).abs() < 20.0,
                "bbox[{i}] {} vs golden {} (>20px)", best.bbox[i], gbox[i]);
        }
        // pose ROI keypoints: kp0 mid-hip, kp1 full-body (drive the pose ROI + rotation)
        let (mh, fb) = (best.keypoints[0], best.keypoints[1]);
        assert!((mh[0] - 892.0).abs() < 30.0 && (mh[1] - 583.0).abs() < 30.0,
            "mid-hip {:?} vs (892,583)", mh);
        assert!((fb[0] - 656.0).abs() < 30.0 && (fb[1] - 60.0).abs() < 30.0,
            "full-body {:?} vs (656,60)", fb);
    }
}
