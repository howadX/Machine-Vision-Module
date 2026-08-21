//! Opt-in, calibration-only tracker tick capture.
//!
//! This is not semantic history: `labels.jsonl` remains the sole durable scene
//! store. These bounded per-take files preserve exact production association
//! inputs and decisions so tracker policies can be calibrated and replayed.

use anyhow::{Context, Result};
use mvm_core::{
    AssociationObservation, BBox, Detection, InstanceMask, TrackerDecision,
    TrackerDiagnosticsSnapshot, TrackerDomain, TrackerFrameBinding,
};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const SCHEMA: &str = "mvm-tracker-tick-capture/1.4";
const ENRICHMENT_SCHEMA: &str = "mvm-tracker-enrichment-application/1.0";
const METADATA_SCHEMA: &str = "mvm-tracker-capture-metadata/1.0";
pub(crate) const TRACE_CAPACITY: usize = 16_384;

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrackerCaptureMetadata {
    schema: &'static str,
    tick_schema: &'static str,
    created_at: String,
    mvm_version: &'static str,
    binary_sha256: String,
    privacy: String,
    masks_enabled: bool,
    max_pair_skew_ns: u64,
    gallery_names: Vec<String>,
    tracker_profile: serde_json::Value,
    model_digests: Vec<ModelDigest>,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelDigest {
    file: String,
    sha256: String,
}

impl ModelDigest {
    pub(crate) fn new(file: impl Into<String>, sha256: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            sha256: sha256.into(),
        }
    }
}

impl TrackerCaptureMetadata {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        created_at: String,
        binary_sha256: String,
        privacy: String,
        masks_enabled: bool,
        max_pair_skew_ns: u64,
        gallery_names: Vec<String>,
        tracker_profile: serde_json::Value,
        model_digests: Vec<ModelDigest>,
    ) -> Self {
        Self {
            schema: METADATA_SCHEMA,
            tick_schema: SCHEMA,
            created_at,
            mvm_version: env!("CARGO_PKG_VERSION"),
            binary_sha256,
            privacy,
            masks_enabled,
            max_pair_skew_ns,
            gallery_names,
            tracker_profile,
            model_digests,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TrackerCalibrationCapture {
    root: Arc<PathBuf>,
    human: Arc<Mutex<BufWriter<File>>>,
    scene: Arc<Mutex<BufWriter<File>>>,
    enrichment: Arc<Mutex<BufWriter<File>>>,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct EnrichmentApplication<'a> {
    schema: &'static str,
    source_binding: &'a TrackerFrameBinding,
    applied_before_binding: &'a TrackerFrameBinding,
    handle: mvm_core::PersonTrackHandle,
    outcome: &'static str,
    patch: &'a mvm_core::PersonEnrichment,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct TickRecord<'a> {
    schema: &'static str,
    domain: TrackerDomain,
    binding: &'a TrackerFrameBinding,
    sensor_health: &'a mvm_core::SensorHealth,
    high_observations: &'a [AssociationObservation],
    low_observations: &'a [AssociationObservation],
    detector_inputs: DetectorInputs<'a>,
    decisions: Vec<&'a TrackerDecision>,
    current_ids: &'a [u32],
    occluded_ids: &'a [u32],
    retained_ids: &'a [u32],
    retirements: &'a [u32],
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DetectorInputs<'a> {
    Human {
        high: Vec<HumanDetection<'a>>,
        low: Vec<LowDetection>,
    },
    Object {
        detections: Vec<ObjectDetection<'a>>,
    },
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct HumanDetection<'a> {
    detection_key: u32,
    bbox: BBox,
    detector_score: f32,
    identity: &'a Option<String>,
    identity_confidence: Option<f32>,
    mask: Option<MaskRle>,
    landmarks: Option<&'a [[f32; 4]]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proposal_geometry: Option<ProposalGeometry>,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct LowDetection {
    detection_key: u32,
    bbox: BBox,
    detector_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    proposal_geometry: Option<ProposalGeometry>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(deny_unknown_fields)]
struct ProposalGeometry {
    /// The detector model's raw face box, before conversion to a body ROI.
    raw_bbox_px: [f32; 4],
    /// hip centre, full-body scale/rotation, shoulder centre, upper-body scale.
    keypoints_px: [[f32; 2]; 4],
    association_bbox_kind: &'static str,
}

impl From<&mvm_posture::detector::Person> for ProposalGeometry {
    fn from(person: &mvm_posture::detector::Person) -> Self {
        Self {
            raw_bbox_px: person.bbox,
            keypoints_px: person.keypoints,
            association_bbox_kind: "raw_detector_anchor",
        }
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ObjectDetection<'a> {
    detection_key: u32,
    class_id: usize,
    class_name: &'a str,
    bbox: BBox,
    detector_score: f32,
    mask: Option<MaskRle>,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct MaskRle {
    resolution: usize,
    coverage: f32,
    /// Alternating background/foreground run lengths, starting with background.
    rle: Vec<u32>,
}

impl MaskRle {
    fn from_mask(mask: &InstanceMask) -> Self {
        let mut rle = Vec::new();
        let mut foreground = false;
        let mut run = 0u32;
        for &bit in mask.bits() {
            let next = bit != 0;
            if next == foreground {
                run += 1;
            } else {
                rle.push(run);
                run = 1;
                foreground = next;
            }
        }
        rle.push(run);
        Self {
            resolution: mvm_core::mask::MASK_RES,
            coverage: mask.frame_coverage(),
            rle,
        }
    }
}

impl TrackerCalibrationCapture {
    pub(crate) fn create(root: PathBuf, metadata: &TrackerCaptureMetadata) -> Result<Self> {
        std::fs::create_dir(&root)
            .with_context(|| format!("create fresh tracker capture {}", root.display()))?;
        let mut metadata_file = BufWriter::new(create_new(&root.join("capture.json"))?);
        serde_json::to_writer_pretty(&mut metadata_file, metadata)?;
        metadata_file.write_all(b"\n")?;
        metadata_file.flush()?;
        let human = create_new(&root.join("human-ticks.jsonl"))?;
        let scene = create_new(&root.join("scene-ticks.jsonl"))?;
        let enrichment = create_new(&root.join("human-enrichment-applications.jsonl"))?;
        Ok(Self {
            root: Arc::new(root),
            human: Arc::new(Mutex::new(BufWriter::new(human))),
            scene: Arc::new(Mutex::new(BufWriter::new(scene))),
            enrichment: Arc::new(Mutex::new(BufWriter::new(enrichment))),
        })
    }

    pub(crate) fn root(&self) -> &Path {
        self.root.as_path()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_human(
        &self,
        binding: &TrackerFrameBinding,
        sensor_health: &mvm_core::SensorHealth,
        high_observations: &[AssociationObservation],
        low_observations: &[AssociationObservation],
        high: &[Detection],
        low: &[BBox],
        landmarks: &[Option<mvm_core::motion::Landmarks33>],
        proposals: Option<(
            &[mvm_posture::detector::Person],
            &[mvm_posture::detector::Person],
        )>,
        diagnostics: &TrackerDiagnosticsSnapshot,
        current_ids: &[u32],
        occluded_ids: &[u32],
        retained_ids: &[u32],
        retirements: &[u32],
    ) -> Result<()> {
        anyhow::ensure!(
            high_observations.len() == high.len() && high.len() == landmarks.len(),
            "human capture high observation/detection/landmark alignment mismatch"
        );
        anyhow::ensure!(
            low_observations.len() == low.len(),
            "human capture low observation/detection alignment mismatch"
        );
        if let Some((high_proposals, low_proposals)) = proposals {
            anyhow::ensure!(
                high_proposals.len() == high.len() && low_proposals.len() == low.len(),
                "human capture proposal/detection alignment mismatch"
            );
        }
        let (high_proposals, low_proposals) = proposals
            .map(|(high, low)| (Some(high), Some(low)))
            .unwrap_or((None, None));
        let high_inputs = high
            .iter()
            .zip(high_observations)
            .zip(landmarks)
            .enumerate()
            .map(
                |(index, ((detection, observation), landmarks))| HumanDetection {
                    detection_key: observation.detection_key,
                    bbox: detection.bbox,
                    detector_score: observation.detector_score,
                    identity: &detection.identity,
                    identity_confidence: detection.identity_confidence,
                    mask: detection.mask.as_ref().map(MaskRle::from_mask),
                    landmarks: landmarks.as_ref().map(<[_; 33]>::as_slice),
                    proposal_geometry: high_proposals
                        .and_then(|values| values.get(index))
                        .map(ProposalGeometry::from),
                },
            )
            .collect();
        let low_inputs = low
            .iter()
            .zip(low_observations)
            .enumerate()
            .map(|(index, (bbox, observation))| LowDetection {
                detection_key: observation.detection_key,
                bbox: *bbox,
                detector_score: observation.detector_score,
                proposal_geometry: low_proposals
                    .and_then(|values| values.get(index))
                    .map(ProposalGeometry::from),
            })
            .collect();
        self.write(
            &self.human,
            TickRecord {
                schema: SCHEMA,
                domain: TrackerDomain::Person,
                binding,
                sensor_health,
                high_observations,
                low_observations,
                detector_inputs: DetectorInputs::Human {
                    high: high_inputs,
                    low: low_inputs,
                },
                decisions: decisions_for(binding, TrackerDomain::Person, diagnostics),
                current_ids,
                occluded_ids,
                retained_ids,
                retirements,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_scene(
        &self,
        binding: &TrackerFrameBinding,
        sensor_health: &mvm_core::SensorHealth,
        observations: &[AssociationObservation],
        detections: &[mvm_scene::ObjectDetection],
        masks: &[Option<&InstanceMask>],
        diagnostics: &TrackerDiagnosticsSnapshot,
        current_ids: &[u32],
        occluded_ids: &[u32],
        retained_ids: &[u32],
        retirements: &[u32],
    ) -> Result<()> {
        anyhow::ensure!(
            observations.len() == detections.len() && detections.len() == masks.len(),
            "object capture observation/detection/mask alignment mismatch"
        );
        let detector_inputs = detections
            .iter()
            .zip(observations)
            .zip(masks)
            .map(|((detection, observation), mask)| ObjectDetection {
                detection_key: observation.detection_key,
                class_id: detection.class_id,
                class_name: detection.class_name,
                bbox: detection.bbox,
                detector_score: detection.score,
                mask: mask.map(MaskRle::from_mask),
            })
            .collect();
        self.write(
            &self.scene,
            TickRecord {
                schema: SCHEMA,
                domain: TrackerDomain::Object,
                binding,
                sensor_health,
                high_observations: observations,
                low_observations: &[],
                detector_inputs: DetectorInputs::Object {
                    detections: detector_inputs,
                },
                decisions: decisions_for(binding, TrackerDomain::Object, diagnostics),
                current_ids,
                occluded_ids,
                retained_ids,
                retirements,
            },
        )
    }

    pub(crate) fn write_enrichment_application(
        &self,
        source_binding: &TrackerFrameBinding,
        applied_before_binding: &TrackerFrameBinding,
        handle: mvm_core::PersonTrackHandle,
        outcome: mvm_core::EnrichmentApply,
        patch: &mvm_core::PersonEnrichment,
    ) -> Result<()> {
        let outcome = match outcome {
            mvm_core::EnrichmentApply::Applied => "applied",
            mvm_core::EnrichmentApply::StaleHandle => "stale_handle",
            mvm_core::EnrichmentApply::InvalidTimestamp => "invalid_timestamp",
            mvm_core::EnrichmentApply::StaleTimestamp => "stale_timestamp",
        };
        let mut sink = self
            .enrichment
            .lock()
            .map_err(|_| anyhow::anyhow!("tracker enrichment capture lock poisoned"))?;
        serde_json::to_writer(
            &mut *sink,
            &EnrichmentApplication {
                schema: ENRICHMENT_SCHEMA,
                source_binding,
                applied_before_binding,
                handle,
                outcome,
                patch,
            },
        )?;
        sink.write_all(b"\n")?;
        sink.flush()?;
        Ok(())
    }

    fn write(&self, sink: &Mutex<BufWriter<File>>, record: TickRecord<'_>) -> Result<()> {
        let mut sink = sink
            .lock()
            .map_err(|_| anyhow::anyhow!("tracker capture lock poisoned"))?;
        serde_json::to_writer(&mut *sink, &record)?;
        sink.write_all(b"\n")?;
        sink.flush()?;
        Ok(())
    }
}

fn create_new(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))
}

fn decisions_for<'a>(
    binding: &TrackerFrameBinding,
    domain: TrackerDomain,
    diagnostics: &'a TrackerDiagnosticsSnapshot,
) -> Vec<&'a TrackerDecision> {
    diagnostics
        .events
        .iter()
        .filter(|event| {
            event.domain == domain
                && event.binding.source_frame_id == binding.source_frame_id
                && event.binding.rgb_stamp == binding.rgb_stamp
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{AssociationGeometry, BBox, TrackerDecisionKind};

    #[test]
    fn writes_exact_domain_ticks_to_separate_fail_closed_files() {
        let root = std::env::temp_dir().join(format!(
            "mvm_tracker_capture_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        let metadata = TrackerCaptureMetadata::new(
            "2026-08-13T00:00:00Z".into(),
            "0".repeat(64),
            "full".into(),
            true,
            20_000_000,
            vec!["test-user".into()],
            serde_json::json!({"revision":"test"}),
            vec![ModelDigest::new("test.onnx", "1".repeat(64))],
        );
        let capture = TrackerCalibrationCapture::create(root.clone(), &metadata).unwrap();
        let binding = TrackerFrameBinding::rgb_only(7);
        let observations = [AssociationObservation {
            detection_key: 0,
            source: binding.clone(),
            bbox: BBox {
                x1: 0.1,
                y1: 0.1,
                x2: 0.2,
                y2: 0.3,
            },
            detector_score: 0.8,
            geometry: AssociationGeometry::unavailable(
                &binding,
                mvm_core::GeometryReason::DepthUnavailable,
            ),
            person_birth_conflict: Some(mvm_core::PersonBirthConflict {
                scene_source_frame_id: 6,
                object_track_id: 10_000,
                display_class: mvm_core::StaticDisplayClass::Tv,
                object_evidence: mvm_core::SceneObjectEvidence::Retained,
                person_coverage: 0.4,
            }),
        }];
        let decision = TrackerDecision::new(
            &binding,
            TrackerDomain::Person,
            TrackerDecisionKind::Birth,
            "created",
        );
        let diagnostics = TrackerDiagnosticsSnapshot {
            capacity: 1,
            dropped_events: 0,
            counters: Default::default(),
            events: vec![decision],
        };
        capture
            .write_human(
                &binding,
                &mvm_core::SensorHealth::default(),
                &observations,
                &[],
                &[Detection::bare(0.1, 0.1, 0.2, 0.3)],
                &[],
                &[None],
                None,
                &diagnostics,
                &[],
                &[],
                &[],
                &[],
            )
            .unwrap();
        let line = std::fs::read_to_string(root.join("human-ticks.jsonl")).unwrap();
        let value: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(value["schema"], "mvm-tracker-tick-capture/1.4");
        assert_eq!(value["binding"]["source_frame_id"], 7);
        assert_eq!(value["decisions"][0]["kind"], "birth");
        assert_eq!(value["detector_inputs"]["kind"], "human");
        assert_eq!(value["detector_inputs"]["high"][0]["detection_key"], 0);
        assert_eq!(
            value["high_observations"][0]["person_birth_conflict"]["object_evidence"],
            "retained"
        );
        capture
            .write_enrichment_application(
                &binding,
                &binding,
                mvm_core::PersonTrackHandle {
                    internal_id: 10,
                    generation: 1,
                },
                mvm_core::EnrichmentApply::Applied,
                &mvm_core::PersonEnrichment {
                    identity: Some("test-user".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        let mut bits = vec![0; mvm_core::mask::MASK_RES * mvm_core::mask::MASK_RES];
        bits[1] = 1;
        let mask = InstanceMask::from_bits(bits);
        let object = mvm_scene::ObjectDetection {
            class_id: 73,
            class_name: "book",
            bbox: observations[0].bbox,
            score: 0.8,
        };
        capture
            .write_scene(
                &binding,
                &mvm_core::SensorHealth::default(),
                &observations,
                &[object],
                &[Some(&mask)],
                &diagnostics,
                &[],
                &[],
                &[],
                &[],
            )
            .unwrap();
        drop(capture);
        let scene = std::fs::read_to_string(root.join("scene-ticks.jsonl")).unwrap();
        let scene: serde_json::Value = serde_json::from_str(scene.trim()).unwrap();
        assert_eq!(scene["detector_inputs"]["kind"], "object");
        assert_eq!(
            scene["detector_inputs"]["detections"][0]["class_name"],
            "book"
        );
        assert_eq!(
            scene["detector_inputs"]["detections"][0]["mask"]["rle"],
            serde_json::json!([1, 1, 4094])
        );
        assert!(scene["decisions"].as_array().unwrap().is_empty());
        let enrichment =
            std::fs::read_to_string(root.join("human-enrichment-applications.jsonl")).unwrap();
        let enrichment: serde_json::Value = serde_json::from_str(enrichment.trim()).unwrap();
        assert_eq!(enrichment["source_binding"]["source_frame_id"], 7);
        assert_eq!(enrichment["applied_before_binding"]["source_frame_id"], 7);
        assert_eq!(enrichment["handle"]["generation"], 1);
        assert_eq!(enrichment["outcome"], "applied");
        assert!(TrackerCalibrationCapture::create(root.clone(), &metadata).is_err());
        let _ = std::fs::remove_dir_all(root);
    }
}
