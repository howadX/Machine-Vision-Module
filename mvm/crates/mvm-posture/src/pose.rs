//! Native port of OpenCV Zoo `pose_estimation_mediapipe` (BlazePose landmarks).
//! Faithful to `spike_pose/mp_pose.py`: build a square ROI around the person's
//! mid-hip, rotate the torso to vertical, run the 256² model, then invert the
//! transform to recover 33 landmarks in original-image pixel coords.
//!
//! The ROI box is square, so the whole preprocess→recover round trip is a
//! similarity transform (rotation + uniform scale + translation).

use crate::detector::Person;
use image::RgbImage;
use ort::{session::Session, value::Tensor};

const IN: usize = 256;
const N_LM: usize = 39; // model emits 39 landmarks (33 body + 6 aux), 5 each

pub struct PoseEstimator {
    session: Session,
    pub conf_threshold: f32,
}

/// 2x3 affine: getRotationMatrix2D(center, angle_deg, scale=1).
fn rot_mat(cx: f32, cy: f32, angle_deg: f32) -> [[f32; 3]; 2] {
    let a = angle_deg.to_radians();
    let (alpha, beta) = (a.cos(), a.sin());
    [
        [alpha, beta, (1.0 - alpha) * cx - beta * cy],
        [-beta, alpha, beta * cx + (1.0 - alpha) * cy],
    ]
}

#[inline]
fn sample_bilinear(img: &RgbImage, x: f32, y: f32) -> [f32; 3] {
    let (w, h) = (img.width() as i32, img.height() as i32);
    if x < 0.0 || y < 0.0 || x > (w - 1) as f32 || y > (h - 1) as f32 {
        return [0.0, 0.0, 0.0]; // outside original image = black padding
    }
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p = |xx: i32, yy: i32| img.get_pixel(xx as u32, yy as u32).0;
    let (p00, p01, p10, p11) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
    let mut out = [0.0; 3];
    for c in 0..3 {
        let t = p00[c] as f32 * (1.0 - fx) + p01[c] as f32 * fx;
        let b = p10[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
        out[c] = t * (1.0 - fy) + b * fy;
    }
    out
}

impl PoseEstimator {
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(Self { session, conf_threshold: 0.5 })
    }

    #[cfg(feature = "embed-model")]
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_memory(bytes)?;
        Ok(Self { session, conf_threshold: 0.5 })
    }

    /// Run BlazePose using the person-detector ROI (mid-hip + full-body keypoints
    /// define a square ROI rotated so the torso is vertical). Returns 33 landmarks
    /// [x,y,z,visibility] in original-image pixel coords, or None if low conf.
    pub fn infer(&mut self, img: &RgbImage, person: &Person) -> anyhow::Result<Option<[[f32; 4]; 33]>> {
        let mid = person.keypoints[0]; // mid-hip
        let full = person.keypoints[1]; // full-body point
        let full_dist = ((mid[0] - full[0]).powi(2) + (mid[1] - full[1]).powi(2)).sqrt();
        if full_dist < 1.0 {
            return Ok(None);
        }
        // rotation so hip->body becomes vertical
        let mut radians = std::f32::consts::FRAC_PI_2
            - (-(full[1] - mid[1])).atan2(full[0] - mid[0]);
        let two_pi = std::f32::consts::TAU;
        radians -= two_pi * ((radians + std::f32::consts::PI) / two_pi).floor();
        let angle = radians.to_degrees();
        self.infer_roi(img, mid[0], mid[1], 2.0 * full_dist, angle)
    }

    /// Run BlazePose on an explicit square ROI: centered at (roi_cx, roi_cy) in
    /// original-image pixels, `side` wide, rotated `angle` degrees about its center.
    /// This is the generic entry — used both by the person-detector path and the
    /// YuNet-face-bbox path (which has no hip/body keypoints, so angle=0).
    pub fn infer_roi(
        &mut self,
        img: &RgbImage,
        roi_cx: f32,
        roi_cy: f32,
        side: f32,
        angle: f32,
    ) -> anyhow::Result<Option<[[f32; 4]; 33]>> {
        if side < 2.0 {
            return Ok(None);
        }
        let roi0 = [roi_cx - side / 2.0, roi_cy - side / 2.0]; // ROI top-left in original coords
        let cd = side / 2.0; // ROI center in ROI/padded coords

        // sample 256x256 = resize(warpAffine(padded_roi, R)); combined single pass.
        // dst(256) -> rotated coords -> (de-rotate, invM = R^-1 = rot(-angle)) padded
        //          -> original (+roi0).
        let inv = rot_mat(cd, cd, -angle); // inverse of getRotationMatrix2D(center, angle)
        let s = side / IN as f32; // resize scale (rotated_person_bbox / 256)
        let mut data = vec![0f32; IN * IN * 3]; // NHWC
        for dy in 0..IN {
            let ry = (dy as f32 + 0.5) * s - 0.5; // resize map (dst->rotated)
            for dx in 0..IN {
                let rx = (dx as f32 + 0.5) * s - 0.5;
                // de-rotate into padded coords
                let px = inv[0][0] * rx + inv[0][1] * ry + inv[0][2];
                let py = inv[1][0] * rx + inv[1][1] * ry + inv[1][2];
                // padded -> original
                let ox = px + roi0[0];
                let oy = py + roi0[1];
                let rgb = sample_bilinear(img, ox, oy);
                let o = (dy * IN + dx) * 3;
                data[o] = rgb[0] / 255.0;
                data[o + 1] = rgb[1] / 255.0;
                data[o + 2] = rgb[2] / 255.0;
            }
        }

        let input = Tensor::from_array(([1usize, IN, IN, 3], data))?;
        let outputs = self.session.run(ort::inputs!["input_1" => input])?;
        let conf = outputs["Identity_1"].try_extract_tensor::<f32>()?.1[0];
        if conf < self.conf_threshold {
            return Ok(None);
        }
        let lm = outputs["Identity"].try_extract_tensor::<f32>()?.1; // [195] = 39*5

        // recovery (mp_pose._postprocess), similarity transform back to original coords
        let r = rot_mat(cd, cd, angle);
        // inverse_rotation_matrix = [ R2x2^T | -R2x2^T·t ]
        let rc = [[r[0][0], r[1][0]], [r[0][1], r[1][1]]]; // rotation_component (transpose)
        let t = [r[0][2], r[1][2]];
        let inv_t = [
            -(rc[0][0] * t[0] + rc[0][1] * t[1]),
            -(rc[1][0] * t[0] + rc[1][1] * t[1]),
        ];
        // center of rotated_person_bbox = (side/2, side/2)
        let center = [side / 2.0, side / 2.0];
        let original_center = [
            rc[0][0] * center[0] + rc[0][1] * center[1] + inv_t[0],
            rc[1][0] * center[0] + rc[1][1] * center[1] + inv_t[1],
        ];
        // coords_rotation_matrix = getRotationMatrix2D((0,0), angle): rotate landmarks by angle
        let cr = rot_mat(0.0, 0.0, angle);

        let mut out = [[0f32; 4]; 33];
        for k in 0..33 {
            let b = k * 5;
            // (lm - 128) * scale_factor
            let lx = (lm[b] - IN as f32 / 2.0) * s;
            let ly = (lm[b + 1] - IN as f32 / 2.0) * s;
            let lz = lm[b + 2] * s; // z * max(scale_factor); scale_factor square => s
            // rotate by angle around origin: [lx,ly] @ cr[:, :2]
            let rxx = lx * cr[0][0] + ly * cr[1][0];
            let ryy = lx * cr[0][1] + ly * cr[1][1];
            out[k][0] = rxx + original_center[0] + roi0[0];
            out[k][1] = ryy + original_center[1] + roi0[1];
            out[k][2] = lz;
            out[k][3] = 1.0 / (1.0 + (-lm[b + 3]).exp()); // visibility = sigmoid(logit)
        }
        let _ = N_LM;
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::PersonDetector;

    fn models_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"))
    }

    #[test]
    fn pose_landmarks_match_oracle() {
        // Golden (normalized x/w, y/h, z/w) from spike_pose/export_for_rust.py on p_V_03.
        // [idx, x, y]: 0 nose (.552,.326); 11 L-sh (.696,.350); 12 R-sh (.519,.406);
        //              23 L-hip (.724,.668); 24 R-hip (.641,.708); 25 L-knee (.784,.881)
        let mut det = PersonDetector::from_path(
            &models_dir().join("person_detection_mediapipe_2023mar.onnx")).unwrap();
        let mut pe = PoseEstimator::from_path(
            &models_dir().join("pose_estimation_mediapipe_2023mar.onnx")).unwrap();

        let img = image::open(models_dir().join("references/p_V_03.jpg")).unwrap();
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as f32, rgb.height() as f32);
        let persons = det.detect(&img).unwrap();
        let best = persons.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();

        let lm = pe.infer(&rgb, best).unwrap().expect("expected pose for a clear person");
        // normalize like the golden (x/w, y/h, z/w)
        let n = |k: usize| [lm[k][0] / w, lm[k][1] / h, lm[k][2] / w];

        let golden: [(usize, [f32; 2]); 6] = [
            (0, [0.552, 0.326]), (11, [0.696, 0.350]), (12, [0.519, 0.406]),
            (23, [0.724, 0.668]), (24, [0.641, 0.708]), (25, [0.784, 0.881]),
        ];
        for (k, g) in golden {
            let p = n(k);
            assert!((p[0] - g[0]).abs() < 0.06 && (p[1] - g[1]).abs() < 0.06,
                "landmark {k} = ({:.3},{:.3}) vs golden ({:.3},{:.3})", p[0], p[1], g[0], g[1]);
        }

        // topology plausibility: nose above shoulders above hips above knees (image y grows down)
        let (nose, sh, hip, knee) = (n(0)[1], n(11)[1], n(23)[1], n(25)[1]);
        assert!(nose < sh && sh < hip && hip < knee,
            "vertical order broken: nose {nose:.2} sh {sh:.2} hip {hip:.2} knee {knee:.2}");
    }

    #[test]
    fn black_image_low_conf_none() {
        let mut pe = PoseEstimator::from_path(
            &models_dir().join("pose_estimation_mediapipe_2023mar.onnx")).unwrap();
        let black = RgbImage::new(640, 480);
        // a fabricated ROI in the black frame -> model should report low conf
        let fake = Person { bbox: [100.0, 100.0, 300.0, 400.0],
            keypoints: [[200.0, 350.0], [200.0, 100.0], [200.0, 200.0], [200.0, 150.0]], score: 0.9 };
        let out = pe.infer(&black, &fake).unwrap();
        assert!(out.is_none(), "black image should yield no pose (low conf), got Some");
    }
}
