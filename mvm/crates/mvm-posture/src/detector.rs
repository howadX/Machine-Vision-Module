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
    pub bbox: [f32; 4],           // x1,y1,x2,y2 in original-image pixels
    pub keypoints: [[f32; 2]; 4], // kp0=mid-hip, kp1=full-body, kp2=shoulder-center, kp3=upper-body
    pub score: f32,
}

/// Confidence tier is evaluated only after detector-level proposal
/// canonicalization, so one physical body cannot feed both BYTE stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfidenceTier {
    High,
    Low,
}

impl Person {
    /// Axis-aligned full-body proxy from MediaPipe's alignment points.
    ///
    /// Keypoint 0 is the crop centre and the distance to keypoint 1 is the
    /// radius of the full-body crop. This is the unexpanded `POSE_RECTS`
    /// geometry; the landmark model may expand it separately for inference.
    pub fn body_bbox(&self) -> [f32; 4] {
        let center = self.keypoints[0];
        let scale = self.keypoints[1];
        let radius = ((scale[0] - center[0]).powi(2) + (scale[1] - center[1]).powi(2)).sqrt();
        [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ]
    }

    pub fn confidence_tier(&self, high_score: f32) -> ConfidenceTier {
        if self.score >= high_score {
            ConfidenceTier::High
        } else {
            ConfidenceTier::Low
        }
    }
}

pub struct PersonDetector {
    session: Session,
    anchors: Vec<[f32; 2]>,
    pub score_threshold: f32,
    pub nms_threshold: f32,
}

fn load_anchors() -> Vec<[f32; 2]> {
    assert_eq!(
        ANCHORS_BIN.len(),
        N_ANCHORS * 2 * 4,
        "anchors_person.bin size mismatch"
    );
    let mut v = Vec::with_capacity(N_ANCHORS);
    for i in 0..N_ANCHORS {
        let o = i * 8;
        let ax = f32::from_le_bytes([
            ANCHORS_BIN[o],
            ANCHORS_BIN[o + 1],
            ANCHORS_BIN[o + 2],
            ANCHORS_BIN[o + 3],
        ]);
        let ay = f32::from_le_bytes([
            ANCHORS_BIN[o + 4],
            ANCHORS_BIN[o + 5],
            ANCHORS_BIN[o + 6],
            ANCHORS_BIN[o + 7],
        ]);
        v.push([ax, ay]);
    }
    v
}

impl PersonDetector {
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(Self {
            session,
            anchors: load_anchors(),
            score_threshold: 0.5,
            nms_threshold: 0.3,
        })
    }

    #[cfg(feature = "embed-model")]
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_memory(bytes)?;
        Ok(Self {
            session,
            anchors: load_anchors(),
            score_threshold: 0.5,
            nms_threshold: 0.3,
        })
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
        let (sw, sh) = (
            rgb.width() as f32 / rw as f32,
            rgb.height() as f32 / rh as f32,
        );
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

    /// Detect at an explicit `min_score` floor and canonicalize raw proposals
    /// before the BYTE high/low split. MediaPipe-compatible weighted NMS fuses
    /// overlapping detector boxes; the alignment-point pass then collapses
    /// vertically displaced face boxes that encode the same physical body.
    pub fn detect_scored(
        &mut self,
        image: &DynamicImage,
        min_score: f32,
    ) -> anyhow::Result<Vec<Person>> {
        let (input, pad_x, pad_y, scale) = self.preprocess(image)?;
        let outputs = self.session.run(ort::inputs!["input_1:0" => input])?;
        let boxes = outputs["Identity:0"].try_extract_tensor::<f32>()?.1; // [2254*12]
        let scores = outputs["Identity_1:0"].try_extract_tensor::<f32>()?.1; // [2254]

        let mut cand: Vec<Person> = Vec::new();
        for (i, &raw_score) in scores.iter().enumerate().take(N_ANCHORS) {
            let raw = raw_score.clamp(-100.0, 100.0);
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
            cand.push(Person {
                bbox: [x1, y1, x2, y2],
                keypoints: kp,
                score: s,
            });
        }
        Ok(canonicalize_body_proposals(cand, self.nms_threshold))
    }

    /// Frozen pre-canonicalization behaviour for the explicit legacy A/B path.
    /// New production tracking uses [`Self::detect_scored`].
    pub fn detect_scored_legacy(
        &mut self,
        image: &DynamicImage,
        min_score: f32,
    ) -> anyhow::Result<Vec<Person>> {
        let (input, pad_x, pad_y, scale) = self.preprocess(image)?;
        let outputs = self.session.run(ort::inputs!["input_1:0" => input])?;
        let boxes = outputs["Identity:0"].try_extract_tensor::<f32>()?.1;
        let scores = outputs["Identity_1:0"].try_extract_tensor::<f32>()?.1;
        let mut cand = Vec::new();
        for (i, &raw_score) in scores.iter().enumerate().take(N_ANCHORS) {
            let score = 1.0 / (1.0 + (-raw_score.clamp(-100.0, 100.0)).exp());
            if score < min_score {
                continue;
            }
            let b = i * 12;
            let [ax, ay] = self.anchors[i];
            let cx = boxes[b] / INPUT as f32;
            let cy = boxes[b + 1] / INPUT as f32;
            let bw = boxes[b + 2] / INPUT as f32;
            let bh = boxes[b + 3] / INPUT as f32;
            let mut keypoints = [[0.0; 2]; 4];
            for (keypoint, decoded) in keypoints
                .iter_mut()
                .zip(boxes[b + 4..b + 12].chunks_exact(2))
            {
                keypoint[0] = (decoded[0] / INPUT as f32 + ax) * scale - pad_x;
                keypoint[1] = (decoded[1] / INPUT as f32 + ay) * scale - pad_y;
            }
            cand.push(Person {
                bbox: [
                    (cx - bw / 2.0 + ax) * scale - pad_x,
                    (cy - bh / 2.0 + ay) * scale - pad_y,
                    (cx + bw / 2.0 + ax) * scale - pad_x,
                    (cy + bh / 2.0 + ay) * scale - pad_y,
                ],
                keypoints,
                score,
            });
        }
        Ok(hard_nms(cand, self.nms_threshold))
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

fn hard_nms(mut cand: Vec<Person>, thr: f32) -> Vec<Person> {
    cand.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Person> = Vec::new();
    for p in cand {
        if keep.iter().all(|k| iou(&k.bbox, &p.bbox) <= thr) {
            keep.push(p);
        }
    }
    keep
}

fn weighted_person(seed: &Person, group: &[Person]) -> Person {
    let total_score: f32 = group.iter().map(|person| person.score).sum();
    if !total_score.is_finite() || total_score <= f32::EPSILON {
        return seed.clone();
    }
    let mut bbox = [0.0; 4];
    let mut keypoints = [[0.0; 2]; 4];
    for person in group {
        for (value, source) in bbox.iter_mut().zip(person.bbox) {
            *value += source * person.score;
        }
        for (target, source) in keypoints.iter_mut().zip(person.keypoints) {
            target[0] += source[0] * person.score;
            target[1] += source[1] * person.score;
        }
    }
    for value in &mut bbox {
        *value /= total_score;
    }
    for point in &mut keypoints {
        point[0] /= total_score;
        point[1] /= total_score;
    }
    Person {
        bbox,
        keypoints,
        // MediaPipe keeps the highest-scoring seed's confidence while it
        // averages the location and keypoints of the overlap group.
        score: seed.score,
    }
}

fn weighted_nms(mut candidates: Vec<Person>, threshold: f32) -> Vec<Person> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept = Vec::new();
    while !candidates.is_empty() {
        // Remove the seed unconditionally so malformed/degenerate geometry
        // cannot reinsert itself forever when IoU(seed, seed) is 0 or NaN.
        let seed = candidates.remove(0);
        let mut group = vec![seed.clone()];
        let mut remaining = Vec::new();
        for candidate in candidates.drain(..) {
            if iou(&seed.bbox, &candidate.bbox) > threshold {
                group.push(candidate);
            } else {
                remaining.push(candidate);
            }
        }
        kept.push(weighted_person(&seed, &group));
        candidates = remaining;
    }
    kept
}

fn alignment_geometry(person: &Person) -> Option<([f32; 2], f32, [f32; 2])> {
    if !person.bbox.into_iter().all(f32::is_finite) {
        return None;
    }
    let center = person.keypoints[0];
    let vector = [
        person.keypoints[1][0] - center[0],
        person.keypoints[1][1] - center[1],
    ];
    let radius = vector[0].hypot(vector[1]);
    let raw_width = (person.bbox[2] - person.bbox[0]).max(0.0);
    let raw_height = (person.bbox[3] - person.bbox[1]).max(0.0);
    let raw_long_side = raw_width.max(raw_height);
    if !center.into_iter().all(f32::is_finite)
        || !vector.into_iter().all(f32::is_finite)
        || !radius.is_finite()
        || radius < 1.0
        // The full-body crop diameter cannot honestly be smaller than the
        // detector's own raw anchor box. Such keypoints are a regressed
        // fragment, not valid body geometry for a confident observation.
        || radius * 2.0 < raw_long_side
    {
        return None;
    }
    Some((center, radius, [vector[0] / radius, vector[1] / radius]))
}

/// Strict same-body gate. Missing or ambiguous geometry deliberately fails
/// open: a temporary duplicate is safer than merging two real people.
fn same_body_alignment(a: &Person, b: &Person) -> bool {
    const MAX_CENTER_FRACTION: f32 = 0.20;
    const MAX_SCALE_RATIO: f32 = 1.25;
    const MIN_DIRECTION_COSINE: f32 = 0.94; // about 20 degrees
    let (Some((ca, ra, ua)), Some((cb, rb, ub))) = (alignment_geometry(a), alignment_geometry(b))
    else {
        return false;
    };
    let center_distance = (ca[0] - cb[0]).hypot(ca[1] - cb[1]);
    let scale_ratio = ra.max(rb) / ra.min(rb);
    let direction_cosine = ua[0] * ub[0] + ua[1] * ub[1];
    center_distance <= ra.min(rb) * MAX_CENTER_FRACTION
        && scale_ratio <= MAX_SCALE_RATIO
        && direction_cosine >= MIN_DIRECTION_COSINE
}

fn canonicalize_body_proposals(candidates: Vec<Person>, nms_threshold: f32) -> Vec<Person> {
    let mut candidates = weighted_nms(candidates, nms_threshold);
    candidates.retain(|candidate| alignment_geometry(candidate).is_some());
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut canonical = Vec::new();
    while !candidates.is_empty() {
        // As above, progress must not depend on self-comparison geometry.
        let seed = candidates.remove(0);
        let mut group = vec![seed.clone()];
        let mut remaining = Vec::new();
        for candidate in candidates.drain(..) {
            if same_body_alignment(&seed, &candidate) {
                group.push(candidate);
            } else {
                remaining.push(candidate);
            }
        }
        canonical.push(weighted_person(&seed, &group));
        candidates = remaining;
    }
    canonical
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person(bbox: [f32; 4], hip: [f32; 2], scale: [f32; 2], score: f32) -> Person {
        Person {
            bbox,
            keypoints: [hip, scale, [hip[0], hip[1] - 20.0], [hip[0], hip[1] - 40.0]],
            score,
        }
    }

    #[test]
    fn weighted_nms_fuses_box_and_alignment_points_but_keeps_seed_score() {
        let detections = vec![
            person(
                [100.0, 40.0, 180.0, 120.0],
                [140.0, 300.0],
                [140.0, 100.0],
                0.9,
            ),
            person(
                [104.0, 44.0, 184.0, 124.0],
                [144.0, 304.0],
                [144.0, 104.0],
                0.6,
            ),
        ];

        let kept = weighted_nms(detections, 0.3);

        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.9).abs() < f32::EPSILON);
        assert!((kept[0].bbox[0] - 101.6).abs() < 0.01);
        assert!((kept[0].keypoints[0][0] - 141.6).abs() < 0.01);
    }

    #[test]
    fn weighted_nms_always_consumes_a_degenerate_seed() {
        let degenerate = person(
            [100.0, 100.0, 100.0, 100.0],
            [140.0, 300.0],
            [140.0, 100.0],
            0.9,
        );

        let kept = weighted_nms(vec![degenerate.clone()], 0.3);

        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].bbox, degenerate.bbox);
        assert!((kept[0].score - degenerate.score).abs() < f32::EPSILON);
    }

    #[test]
    fn canonicalization_drops_non_finite_bbox_geometry() {
        let malformed = person(
            [f32::NAN, 20.0, 180.0, 100.0],
            [140.0, 300.0],
            [140.0, 100.0],
            0.9,
        );

        assert!(canonicalize_body_proposals(vec![malformed], 0.3).is_empty());
    }

    #[test]
    fn vertical_face_boxes_for_one_body_collapse_before_high_low_split() {
        let detections = vec![
            person(
                [100.0, 20.0, 180.0, 100.0],
                [140.0, 300.0],
                [140.0, 100.0],
                0.72,
            ),
            // Raw face boxes do not overlap enough for the detector IoU NMS,
            // but the independent body alignment points agree.
            person(
                [101.0, 105.0, 181.0, 185.0],
                [143.0, 302.0],
                [143.0, 102.0],
                0.41,
            ),
        ];

        let kept = canonicalize_body_proposals(detections, 0.3);

        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].confidence_tier(0.5), ConfidenceTier::High);
    }

    #[test]
    fn close_but_distinct_people_survive_body_duplicate_gate() {
        let detections = vec![
            person(
                [100.0, 20.0, 180.0, 100.0],
                [140.0, 300.0],
                [140.0, 100.0],
                0.8,
            ),
            person(
                [190.0, 30.0, 270.0, 110.0],
                [230.0, 305.0],
                [230.0, 105.0],
                0.75,
            ),
        ];

        let kept = canonicalize_body_proposals(detections, 0.3);

        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn body_bbox_is_centered_on_hip_and_uses_full_body_radius() {
        let detection = person(
            [100.0, 20.0, 180.0, 100.0],
            [140.0, 300.0],
            [140.0, 100.0],
            0.8,
        );

        assert_eq!(detection.body_bbox(), [-60.0, 100.0, 340.0, 500.0]);
    }

    #[test]
    fn malformed_full_body_scale_cannot_become_a_high_observation() {
        let malformed = person(
            [100.0, 20.0, 180.0, 100.0],
            [140.0, 120.0],
            [150.0, 120.0],
            0.9,
        );

        assert!(canonicalize_body_proposals(vec![malformed], 0.3).is_empty());
    }

    fn models_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"))
    }

    fn detector() -> PersonDetector {
        PersonDetector::from_path(&models_dir().join("person_detection_mediapipe_2023mar.onnx"))
            .unwrap()
    }

    #[test]
    fn anchors_load() {
        let a = load_anchors();
        assert_eq!(a.len(), N_ANCHORS);
        // anchors are normalized centers in [0,1]
        assert!(
            a.iter()
                .all(|[x, y]| *x >= 0.0 && *x <= 1.0 && *y >= 0.0 && *y <= 1.0)
        );
    }

    #[test]
    fn black_image_no_person() {
        let mut det = detector();
        let img = DynamicImage::new_rgb8(640, 480);
        let persons = det.detect(&img).unwrap();
        assert!(
            persons.is_empty(),
            "black image should yield no persons, got {}",
            persons.len()
        );
    }

    #[test]
    fn detects_person_matches_oracle() {
        // Golden from spike_pose/export_for_rust.py on the same fixture (Python reference).
        // best_score 0.92, box [634.1,105.9,890.2,362.0], kp0(mid-hip)=(892,583), kp1(body)=(656,60)
        let mut det = detector();
        let img = image::open(models_dir().join("references/p_V_03.jpg")).unwrap();
        let persons = det.detect(&img).unwrap();
        assert!(!persons.is_empty(), "expected a person in p_V_03");
        let best = persons
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();
        assert!(best.score > 0.8, "score {} too low", best.score);

        // Tolerance ~2.5% of the 1280px frame: the Rust JPEG decoder + bilinear resize
        // differ sub-pixel from OpenCV, which the keypoint regression mildly amplifies.
        // The decisive oracle is the 33-landmark pose match (pose_oracle test).
        let gbox = [634.1, 105.9, 890.2, 362.0];
        for (i, &golden_value) in gbox.iter().enumerate() {
            assert!(
                (best.bbox[i] - golden_value).abs() < 20.0,
                "bbox[{i}] {} vs golden {} (>20px)",
                best.bbox[i],
                golden_value
            );
        }
        // pose ROI keypoints: kp0 mid-hip, kp1 full-body (drive the pose ROI + rotation)
        let (mh, fb) = (best.keypoints[0], best.keypoints[1]);
        assert!(
            (mh[0] - 892.0).abs() < 30.0 && (mh[1] - 583.0).abs() < 30.0,
            "mid-hip {:?} vs (892,583)",
            mh
        );
        assert!(
            (fb[0] - 656.0).abs() < 30.0 && (fb[1] - 60.0).abs() < 30.0,
            "full-body {:?} vs (656,60)",
            fb
        );
    }
}
