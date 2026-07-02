//! mvm-posture — native BlazePose body landmarks + posture classification.
//!
//! Two-stage native pose (Phase 7.5): `person_detection_mediapipe` → ROI →
//! `pose_estimation_mediapipe` → 33 landmarks, per detected person (multi-persona).
//! Replaces the Python MediaPipe + PoseVAE bridge chain.

use mvm_core::{Config, Detection, Module, ModuleMetadata};

pub mod classifier;
pub mod detector;
pub mod pose;

use classifier::{PoseHead, PoseResult};
use detector::PersonDetector;
use image::DynamicImage;
use pose::PoseEstimator;

/// 33 BlazePose landmarks [x,y,z,visibility] (x,y in original-image pixels, z rel. hip).
pub type Landmarks33 = [[f32; 4]; 33];

/// Face-bbox → body-ROI heuristic (tuned live for the desk/ceiling view).
/// ROI side = SCALE × face_height; ROI center = face_center + DOWN × face_height (down).
pub const FACE_ROI_SCALE: f32 = 4.5;
pub const FACE_ROI_DOWN: f32 = 1.6;

#[cfg(feature = "embed-model")]
static PERSON_DET_BYTES: &[u8] = include_bytes!("../../../models/person_detection_mediapipe_2023mar.onnx");
#[cfg(feature = "embed-model")]
static POSE_EST_BYTES: &[u8] = include_bytes!("../../../models/pose_estimation_mediapipe_2023mar.onnx");

/// One detected **body** (phase 8 body-primary, FRONTIER): normalized body bbox
/// + pixel landmarks (for torso-3D) + posture, all together. Produced by
/// [`NativePose::bodies`] — the carrying-entity source for `--native-camera`.
#[derive(Debug, Clone)]
pub struct BodyPose {
    /// Body bbox, normalized [0..1] — the carrying entity in body-primary.
    pub bbox: mvm_core::BBox,
    /// 33 landmarks in original-image **pixels** (feed `mvm_core::space::torso_to_3d`).
    pub landmarks_px: Landmarks33,
    /// Posture classification (label, confidence, recon_error, plausibility).
    pub posture: PoseResult,
    /// Person-detector confidence (phase 9.4) — threaded through for ByteTrack
    /// high/low gating. Always ≥ [`BYTE_HIGH_SCORE`] for a posed body.
    pub score: f32,
}

/// A weak, **bbox-only** body box (phase 9.4 ByteTrack): the person detector
/// fired between [`BYTE_LOW_SCORE`] and [`BYTE_HIGH_SCORE`], so no pose was run.
/// Feeds the tracker's low-confidence recovery stage and nothing else.
#[derive(Debug, Clone, Copy)]
pub struct LowConfBody {
    pub bbox: mvm_core::BBox,
    pub score: f32,
}

/// Both confidence tiers from one detector pass (phase 9.4). `bodies` are fully
/// posed (high-conf); `low_conf` are bbox-only recovery candidates.
#[derive(Debug, Clone, Default)]
pub struct BodyDetections {
    pub bodies: Vec<BodyPose>,
    pub low_conf: Vec<LowConfBody>,
}

/// High-confidence gate: a body at/above this score is posed + fully enriched.
/// Equals the detector's legacy `score_threshold`, so the posed set is unchanged.
pub const BYTE_HIGH_SCORE: f32 = 0.5;
/// Low-confidence floor (phase 9.4): boxes in `[BYTE_LOW_SCORE, BYTE_HIGH_SCORE)`
/// are kept bbox-only for ByteTrack recovery; below it they are noise.
pub const BYTE_LOW_SCORE: f32 = 0.2;

// ── Pose-ROI recovery (phase 9.9) ───────────────────────────────────────────
/// Score a body recovered from a stable track's predicted ROI enters with — it is
/// a trusted confirmed track, so it counts as high-confidence (≥ `BYTE_HIGH_SCORE`).
const PREDICTED_SCORE: f32 = 0.6;
/// Pose-ROI side = this × the predicted bbox's longer side (pixels). Generous
/// enough to capture limbs a tight track bbox may clip. Tunable; live-validated.
const POSE_RECOVER_ROI_SCALE: f32 = 1.3;
/// A recovered pose is accepted only when the torso is visible at least this much
/// — the guard that keeps an empty ROI from posing into a phantom.
const RECOVER_TORSO_VIS: f32 = 0.5;
/// IoU above which a detector body already covers a predicted ROI (skip recovery).
const RECOVER_COVERED_IOU: f32 = 0.3;

/// Normalised bbox spanning the visible landmarks (phase 9.9) — the recovered
/// body's extent, so it tracks the body's *current* position (not the stale ROI).
fn bbox_from_landmarks(lm: &Landmarks33, w: f32, h: f32) -> mvm_core::BBox {
    let (mut x1, mut y1, mut x2, mut y2) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in lm.iter() {
        if p[3] < 0.3 {
            continue; // ignore low-visibility (hallucinated) joints
        }
        x1 = x1.min(p[0]);
        y1 = y1.min(p[1]);
        x2 = x2.max(p[0]);
        y2 = y2.max(p[1]);
    }
    if x1 > x2 {
        return mvm_core::BBox { x1: 0.0, y1: 0.0, x2: 0.0, y2: 0.0 };
    }
    mvm_core::BBox {
        x1: (x1 / w).clamp(0.0, 1.0),
        y1: (y1 / h).clamp(0.0, 1.0),
        x2: (x2 / w).clamp(0.0, 1.0),
        y2: (y2 / h).clamp(0.0, 1.0),
    }
}

/// Native two-stage pose: person detection → per-person ROI → BlazePose landmarks.
/// Multi-persona by construction (one pose inference per detected person) — this is
/// what fixes #3 (the MediaPipe `.task` only ever returned a single pose).
pub struct NativePose {
    detector: PersonDetector,
    pose: PoseEstimator,
    head: PoseHead,
}

impl NativePose {
    pub fn from_paths(person_det: &std::path::Path, pose_est: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self {
            detector: PersonDetector::from_path(person_det)?,
            pose: PoseEstimator::from_path(pose_est)?,
            head: PoseHead::load(),
        })
    }

    /// Load from `config.models_path` (the canonical mvm/models dir).
    pub fn init(config: &Config) -> anyhow::Result<Self> {
        let dir = &config.models_path;
        Self::from_paths(
            &dir.join("person_detection_mediapipe_2023mar.onnx"),
            &dir.join("pose_estimation_mediapipe_2023mar.onnx"),
        )
    }

    #[cfg(feature = "embed-model")]
    pub fn from_embedded() -> anyhow::Result<Self> {
        Ok(Self {
            detector: PersonDetector::from_bytes(PERSON_DET_BYTES)?,
            pose: PoseEstimator::from_bytes(POSE_EST_BYTES)?,
            head: PoseHead::load(),
        })
    }

    /// One landmark set per detected person (multi-persona). Persons whose pose
    /// inference falls below confidence are skipped.
    pub fn landmarks_per_person(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<Landmarks33>> {
        let rgb = image.to_rgb8();
        let persons = self.detector.detect(image)?;
        let mut out = Vec::with_capacity(persons.len());
        for p in &persons {
            if let Some(lm) = self.pose.infer(&rgb, p)? {
                out.push(lm);
            }
        }
        Ok(out)
    }

    /// Body-primary primitive (phase 8, FRONTIER): one [`BodyPose`] per detected
    /// person — body bbox + pixel landmarks + posture together. The body is
    /// visible from all angles, so a person without a frontal face still yields a
    /// track here (face/identity/emotion attach later as optional attributes).
    pub fn bodies(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<BodyPose>> {
        Ok(self.bodies_byte(image)?.bodies)
    }

    /// ByteTrack two-tier bodies (phase 9.4): one detector pass at the low floor,
    /// split into posed high-conf [`BodyPose`]s and bbox-only [`LowConfBody`]s.
    /// **Pose inference runs only on high-conf boxes** — cost is unchanged vs
    /// [`Self::bodies`]; the weak boxes ride along for tracker recovery only.
    pub fn bodies_byte(&mut self, image: &DynamicImage) -> anyhow::Result<BodyDetections> {
        let rgb = image.to_rgb8();
        let (w, h) = (image.width() as f32, image.height() as f32);
        let persons = self.detector.detect_scored(image, BYTE_LOW_SCORE)?;
        let mut out = BodyDetections::default();
        for p in &persons {
            let bbox = mvm_core::BBox {
                x1: p.bbox[0] / w,
                y1: p.bbox[1] / h,
                x2: p.bbox[2] / w,
                y2: p.bbox[3] / h,
            };
            if p.score < BYTE_HIGH_SCORE {
                // Weak box: keep for recovery only, no pose inference.
                out.low_conf.push(LowConfBody { bbox, score: p.score });
                continue;
            }
            if let Some(lm) = self.pose.infer(&rgb, p)? {
                let mut norm = [[0f32; 4]; 33];
                for k in 0..33 {
                    norm[k] = [lm[k][0] / w, lm[k][1] / h, lm[k][2] / w, lm[k][3]];
                }
                out.bodies.push(BodyPose {
                    bbox,
                    landmarks_px: lm,
                    posture: self.head.classify(&norm),
                    score: p.score,
                });
            }
        }
        Ok(out)
    }

    /// ByteTrack bodies (9.4) **plus pose-ROI recovery** (phase 9.9). Any stable
    /// track whose predicted ROI the person detector **missed** this frame is posed
    /// directly from that ROI — so a confirmed body the detector fragmented or
    /// dropped (e.g. mid-walk under the ceiling cam) is sustained from the track's
    /// own prediction instead of churning a fresh id.
    ///
    /// The detector path is **untouched** ([`Self::bodies_byte`] runs first); this
    /// only ADDS bodies the detector didn't already cover, and each is gated on
    /// **torso visibility** so posing an empty ROI never fabricates a phantom. With
    /// no predicted ROIs it is exactly `bodies_byte`.
    pub fn bodies_byte_recover(
        &mut self,
        image: &DynamicImage,
        predicted: &[mvm_core::BBox],
    ) -> anyhow::Result<BodyDetections> {
        let mut out = self.bodies_byte(image)?;
        if predicted.is_empty() {
            return Ok(out);
        }
        let rgb = image.to_rgb8();
        let (w, h) = (image.width() as f32, image.height() as f32);
        for roi in predicted {
            // Skip if a detector body/box already covers this track's ROI.
            let covered = out.bodies.iter().any(|b| b.bbox.iou(roi) > RECOVER_COVERED_IOU)
                || out.low_conf.iter().any(|l| l.bbox.iou(roi) > RECOVER_COVERED_IOU);
            if covered {
                continue;
            }
            // ROI center + side (pixels) from the predicted bbox; upright (angle 0).
            let cx = (roi.x1 + roi.x2) * 0.5 * w;
            let cy = (roi.y1 + roi.y2) * 0.5 * h;
            let side = ((roi.x2 - roi.x1) * w).max((roi.y2 - roi.y1) * h) * POSE_RECOVER_ROI_SCALE;
            let Some(lm) = self.pose.infer_roi(&rgb, cx, cy, side, 0.0)? else { continue; };
            // Torso visible → a real body is in the ROI (not an empty patch).
            let torso_vis = lm[11][3].min(lm[12][3]).min(lm[23][3]).min(lm[24][3]);
            if torso_vis < RECOVER_TORSO_VIS {
                continue;
            }
            let mut norm = [[0f32; 4]; 33];
            for k in 0..33 {
                norm[k] = [lm[k][0] / w, lm[k][1] / h, lm[k][2] / w, lm[k][3]];
            }
            out.bodies.push(BodyPose {
                bbox: bbox_from_landmarks(&lm, w, h),
                landmarks_px: lm,
                posture: self.head.classify(&norm),
                score: PREDICTED_SCORE,
            });
        }
        Ok(out)
    }

    /// Posture from YuNet **face** bboxes (normalized [x1,y1,x2,y2]) instead of the
    /// person detector. The desk/ceiling deployment cuts off the body → the person
    /// detector fails, but the face is always visible; we derive a body ROI below the
    /// face (no hip/body keypoints → angle 0, person ~upright in the rotated frame).
    /// One `Option<PoseResult>` per face (None = pose below confidence). Fixes #3
    /// (one pose per face = multi-persona).
    pub fn poses_from_faces(
        &mut self,
        image: &DynamicImage,
        faces: &[[f32; 4]],
    ) -> anyhow::Result<Vec<Option<PoseResult>>> {
        let rgb = image.to_rgb8();
        let (w, h) = (image.width() as f32, image.height() as f32);
        let mut out = Vec::with_capacity(faces.len());
        for f in faces {
            // face bbox (normalized) → pixels
            let (x1, y1, x2, y2) = (f[0] * w, f[1] * h, f[2] * w, f[3] * h);
            let fcx = (x1 + x2) / 2.0;
            let fcy = (y1 + y2) / 2.0;
            let fh = (y2 - y1).max(1.0);
            // body ROI: square below the face. In the ROTATE_180 frame the body is
            // *below* the head (head_y < hip_y in all live data).
            let side = FACE_ROI_SCALE * fh;
            let cy = fcy + FACE_ROI_DOWN * fh;
            let res = match self.pose.infer_roi(&rgb, fcx, cy, side, 0.0)? {
                Some(lm) => {
                    let mut norm = [[0f32; 4]; 33];
                    for k in 0..33 {
                        norm[k] = [lm[k][0] / w, lm[k][1] / h, lm[k][2] / w, lm[k][3]];
                    }
                    Some(self.head.classify(&norm))
                }
                None => None,
            };
            out.push(res);
        }
        Ok(out)
    }

    /// Single face → posture **and** pixel landmarks, for the CLI fallback: when the
    /// person detector misses a body but YuNet still sees the face (desk/close view),
    /// derive a body ROI below the face, pose it, and — only if the torso is actually
    /// visible in that ROI (`RECOVER_TORSO_VIS`, so an empty patch fabricates nothing)
    /// — return the classified [`PoseResult`] with its landmarks (fed to the motion
    /// ring). `face` is normalized `[x1,y1,x2,y2]`; caller supplies the RGB frame once.
    pub fn pose_from_face(
        &mut self,
        rgb: &image::RgbImage,
        w: f32,
        h: f32,
        face: [f32; 4],
    ) -> anyhow::Result<Option<(PoseResult, Landmarks33)>> {
        let (x1, y1, x2, y2) = (face[0] * w, face[1] * h, face[2] * w, face[3] * h);
        let fcx = (x1 + x2) * 0.5;
        let fcy = (y1 + y2) * 0.5;
        let fh = (y2 - y1).max(1.0);
        let side = FACE_ROI_SCALE * fh;
        let cy = fcy + FACE_ROI_DOWN * fh;
        let Some(lm) = self.pose.infer_roi(rgb, fcx, cy, side, 0.0)? else { return Ok(None) };
        let torso_vis = lm[11][3].min(lm[12][3]).min(lm[23][3]).min(lm[24][3]);
        if torso_vis < RECOVER_TORSO_VIS {
            return Ok(None);
        }
        let mut norm = [[0f32; 4]; 33];
        for k in 0..33 {
            norm[k] = [lm[k][0] / w, lm[k][1] / h, lm[k][2] / w, lm[k][3]];
        }
        Ok(Some((self.head.classify(&norm), lm)))
    }

    /// Like `poses`, but also returns each person's bbox (normalized [0,1]) so the
    /// caller can match a posture to a face/track by spatial overlap. This is the
    /// CLI integration entry (person_detection ROI — the validated source).
    pub fn poses_located(
        &mut self,
        image: &DynamicImage,
    ) -> anyhow::Result<Vec<(mvm_core::BBox, PoseResult)>> {
        let rgb = image.to_rgb8();
        let (w, h) = (image.width() as f32, image.height() as f32);
        let persons = self.detector.detect(image)?;
        let mut out = Vec::with_capacity(persons.len());
        for p in &persons {
            if let Some(lm) = self.pose.infer(&rgb, p)? {
                let mut norm = [[0f32; 4]; 33];
                for k in 0..33 {
                    norm[k] = [lm[k][0] / w, lm[k][1] / h, lm[k][2] / w, lm[k][3]];
                }
                let bbox = mvm_core::BBox {
                    x1: p.bbox[0] / w,
                    y1: p.bbox[1] / h,
                    x2: p.bbox[2] / w,
                    y2: p.bbox[3] / h,
                };
                out.push((bbox, self.head.classify(&norm)));
            }
        }
        Ok(out)
    }

    /// Full native posture per person: detect → landmarks → normalize → classify.
    /// Multi-persona (fixes #3). Returns one `PoseResult` per confidently-posed person.
    pub fn poses(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<PoseResult>> {
        let (w, h) = (image.width() as f32, image.height() as f32);
        let people = self.landmarks_per_person(image)?;
        Ok(people
            .iter()
            .map(|lm| {
                // MediaPipe-normalized for the VAE: x/w, y/h, z/w, visibility
                let mut norm = [[0f32; 4]; 33];
                for k in 0..33 {
                    norm[k] = [lm[k][0] / w, lm[k][1] / h, lm[k][2] / w, lm[k][3]];
                }
                self.head.classify(&norm)
            })
            .collect())
    }
}

/// Stub kept until the full two-stage pipeline + VAE classifier are wired (7.5.3/7.5.4).
pub struct PostureModule;

impl Module for PostureModule {
    type Input = Vec<Detection>;
    type Output = Vec<Detection>;

    fn init(_config: &Config) -> anyhow::Result<Self> {
        Ok(Self)
    }

    fn process(&mut self, input: &Self::Input) -> anyhow::Result<Self::Output> {
        Ok(input.clone())
    }

    fn metadata(&self) -> ModuleMetadata {
        ModuleMetadata { name: "mvm-posture (stub)", version: env!("CARGO_PKG_VERSION"), accuracy: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config {
            models_path: std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")),
            ..Default::default()
        }
    }

    #[test]
    fn bbox_from_landmarks_spans_visible_joints() {
        let mut lm = [[0.0f32; 4]; 33];
        // three visible joints + one low-vis outlier that must be ignored
        lm[11] = [100.0, 200.0, 0.0, 0.9];
        lm[12] = [300.0, 220.0, 0.0, 0.9];
        lm[24] = [250.0, 600.0, 0.0, 0.9];
        lm[0] = [9999.0, 9999.0, 0.0, 0.1]; // low vis → ignored
        let b = bbox_from_landmarks(&lm, 1000.0, 1000.0);
        assert!((b.x1 - 0.10).abs() < 1e-4 && (b.y1 - 0.20).abs() < 1e-4);
        assert!((b.x2 - 0.30).abs() < 1e-4 && (b.y2 - 0.60).abs() < 1e-4);
    }

    #[test]
    fn bbox_from_landmarks_empty_when_none_visible() {
        let lm = [[0.0f32; 4]; 33]; // all visibility 0
        let b = bbox_from_landmarks(&lm, 1000.0, 1000.0);
        assert_eq!((b.x1, b.y1, b.x2, b.y2), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn native_pose_per_person() {
        let mut np = NativePose::init(&test_config()).unwrap();
        let img = image::open(test_config().models_path.join("references/p_V_03.jpg")).unwrap();
        let people = np.landmarks_per_person(&img).unwrap();
        assert!(!people.is_empty(), "expected ≥1 person's landmarks");
        // each set is 33 landmarks; sanity: hips (23/24) below shoulders (11/12) in image y
        let lm = &people[0];
        let sh_y = (lm[11][1] + lm[12][1]) / 2.0;
        let hip_y = (lm[23][1] + lm[24][1]) / 2.0;
        assert!(hip_y > sh_y, "hips should be below shoulders in image coords");
    }

    #[test]
    fn pose_class_oracle_desk_occlusion_contract() {
        // Occlusion-aware oracle on the owner's desk references. These frames are
        // the live desk framing: legs behind the desk (measured ankle vis
        // 0.09–0.51), where the public-data RF used to read sitting as confident
        // `lying` (the old ≥5/7 label oracle was #[ignore]'d on exactly this).
        // The honest contract replaces label agreement:
        //   - every frame detects + classifies;
        //   - a frame is EITHER Full coverage with the correct human label,
        //     OR Upper coverage with an upright torso (posture withheld);
        //   - the claimable label is NEVER a false `lying` — the desk bug.
        let golden: [(&str, &str); 7] = [
            ("p_V_01.jpg", "sitting"),
            ("p_V_02.jpg", "standing"),
            ("p_V_03.jpg", "standing"),
            ("p_V_04.jpg", "sitting"),
            ("p_V_05.jpg", "standing"),
            ("p_V_06.jpg", "sitting"),
            ("p_V_07.jpg", "sitting"),
        ];
        // Mirror of the pipeline's occlusion-aware claim gate (mvm-cli
        // `posture_claimable`) — mvm-posture cannot depend on mvm-cli.
        let claim = |r: &PoseResult| -> Option<&'static str> {
            use classifier::BodyCoverage as C;
            if r.label == "lying" && r.attitude == "upright" {
                return None; // physically inconsistent — hallucinated-leg artifact
            }
            match r.coverage {
                C::Full => Some(r.label),
                C::Upper if r.label == "lying" && r.attitude == "horizontal" => Some("lying"),
                _ => None,
            }
        };
        let mut np = NativePose::init(&test_config()).unwrap();
        let refdir = test_config().models_path.join("references");
        let mut detected = 0;
        for (file, human) in golden {
            let img = image::open(refdir.join(file)).unwrap();
            let poses = np.poses(&img).unwrap();
            let Some(r) = poses.first() else { continue };
            detected += 1;
            let claimed = claim(r);
            // The bug this architecture closes: no desk frame may claim `lying`
            // (the subject is upright at the desk in all seven).
            assert_ne!(claimed, Some("lying"), "{file}: false lying claim (was the live desk bug)");
            // Honest contract: a claim, if made, must match the human label;
            // withholding (None — legs occluded, sitting-vs-standing unknowable)
            // is never a lie. The desk subject is upright in every frame, so the
            // attitude must corroborate that too.
            assert!(
                claimed.is_none() || claimed == Some(human),
                "{file}: wrong claim {claimed:?} (human {human})"
            );
            assert_eq!(r.attitude, "upright", "{file}: desk subject torso is upright");
        }
        assert_eq!(detected, 7, "native pose should detect+classify all 7 references, got {detected}");
    }

    #[test]
    fn pose_from_face_recovers_or_gates() {
        let mut np = NativePose::init(&test_config()).unwrap();
        let refdir = test_config().models_path.join("references");
        // Desk view (sitting_at_desk) — the framing the face-ROI heuristic targets.
        let img = image::open(refdir.join("p_V_01.jpg")).unwrap();
        let (w, h) = (img.width() as f32, img.height() as f32);
        let rgb = img.to_rgb8();
        // Realistic face bbox = span of the face landmarks (nose/eyes/ears/mouth,
        // indices 0..11), so the ROI-below-face heuristic sees a real-sized head —
        // exercises the full face→ROI→pose→classify path.
        let people = np.landmarks_per_person(&img).unwrap();
        let lm0 = &people[0];
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in lm0.iter().take(11) {
            x0 = x0.min(p[0]); y0 = y0.min(p[1]); x1 = x1.max(p[0]); y1 = y1.max(p[1]);
        }
        let face = [x0 / w, y0 / h, x1 / w, y1 / h];
        let (pose, lm) = np
            .pose_from_face(&rgb, w, h, face)
            .unwrap()
            .expect("torso visible below the face → posture recovered");
        assert!(classifier::POSE_LABELS.contains(&pose.label), "label {} not in taxonomy", pose.label);
        assert!(lm[11][3] >= 0.0 && lm[11][3] <= 1.0);

        // Empty/black frame → torso invisible → gated to None (fabricates nothing).
        let black = image::RgbImage::new(640, 480);
        assert!(np.pose_from_face(&black, 640.0, 480.0, [0.4, 0.1, 0.6, 0.3]).unwrap().is_none());
    }

    #[test]
    fn native_pose_end_to_end() {
        // Full native chain on p_V_03; Python reference classified this frame as "standing".
        let mut np = NativePose::init(&test_config()).unwrap();
        let img = image::open(test_config().models_path.join("references/p_V_03.jpg")).unwrap();
        let poses = np.poses(&img).unwrap();
        assert!(!poses.is_empty(), "expected ≥1 classified pose");
        let p = &poses[0];
        assert!(classifier::POSE_LABELS.contains(&p.label), "label {} not in taxonomy", p.label);
        assert!(p.confidence > 0.0 && p.confidence <= 1.0);
        assert!(p.recon_error >= 0.0 && p.recon_error < 0.2, "recon {} implausible", p.recon_error);
        // native landmarks differ sub-pixel from Python; an upright subject should land
        // on an upright class, not lying.
        assert!(!p.label.starts_with("lying"), "upright subject misclassified as {}", p.label);
    }
}
