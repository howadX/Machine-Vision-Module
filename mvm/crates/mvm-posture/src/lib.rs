//! mvm-posture — native BlazePose body landmarks + posture classification.
//!
//! Two-stage native pose (Phase 7.5): `person_detection_mediapipe` → ROI →
//! `pose_estimation_mediapipe` → 33 landmarks, per detected person (multi-persona).
//! Replaces the Python MediaPipe + PoseVAE bridge chain.

use mvm_core::{Config, Detection, Module, ModuleMetadata};

pub mod detector;
pub mod pose;

use detector::PersonDetector;
use image::DynamicImage;
use pose::PoseEstimator;

/// 33 BlazePose landmarks (x,y in original-image pixels, z relative to hip) for one person.
pub type Landmarks33 = [[f32; 3]; 33];

#[cfg(feature = "embed-model")]
static PERSON_DET_BYTES: &[u8] = include_bytes!("../../../models/person_detection_mediapipe_2023mar.onnx");
#[cfg(feature = "embed-model")]
static POSE_EST_BYTES: &[u8] = include_bytes!("../../../models/pose_estimation_mediapipe_2023mar.onnx");

/// Native two-stage pose: person detection → per-person ROI → BlazePose landmarks.
/// Multi-persona by construction (one pose inference per detected person) — this is
/// what fixes #3 (the MediaPipe `.task` only ever returned a single pose).
pub struct NativePose {
    detector: PersonDetector,
    pose: PoseEstimator,
}

impl NativePose {
    pub fn from_paths(person_det: &std::path::Path, pose_est: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self {
            detector: PersonDetector::from_path(person_det)?,
            pose: PoseEstimator::from_path(pose_est)?,
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
}
