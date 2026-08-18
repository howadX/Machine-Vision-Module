//! Fail-closed replay of exact Human/Scene tracker-tick captures.
//!
//! The capture stays outside `labels.jsonl`; this runner consumes only explicit,
//! SHA-256-pinned artifacts and refuses held-out execution.

use anyhow::{Context, Result, bail, ensure};
use mvm_core::{
    AssociationObservation, BBox, Detection, InstanceMask, SensorHealth, TrackerDecision,
    TrackerDomain, TrackerFrameBinding,
};
use mvm_scene::{COCO_CLASSES, ObjectDetection, ObjectTracker};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::collections::HashSet;

const METADATA_SCHEMA: &str = "mvm-tracker-capture-metadata/1.0";
const TICK_SCHEMA: &str = "mvm-tracker-tick-capture/1.1";
const TRACE_CAPACITY: usize = 16_384;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    schema: String,
    tick_schema: String,
    created_at: String,
    mvm_version: String,
    binary_sha256: String,
    privacy: String,
    masks_enabled: bool,
    max_pair_skew_ns: u64,
    #[serde(default)]
    gallery_names: Vec<String>,
    tracker_profile: serde_json::Value,
    model_digests: Vec<ModelDigest>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelDigest {
    file: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TickRecord {
    schema: String,
    domain: TrackerDomain,
    binding: TrackerFrameBinding,
    sensor_health: SensorHealth,
    high_observations: Vec<AssociationObservation>,
    low_observations: Vec<AssociationObservation>,
    detector_inputs: DetectorInputs,
    decisions: Vec<TrackerDecision>,
    current_ids: Vec<u32>,
    occluded_ids: Vec<u32>,
    retained_ids: Vec<u32>,
    retirements: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DetectorInputs {
    Human {
        high: Vec<HumanDetection>,
        low: Vec<LowDetection>,
    },
    Object {
        detections: Vec<CapturedObjectDetection>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HumanDetection {
    detection_key: u32,
    bbox: BBox,
    detector_score: f32,
    identity: Option<String>,
    identity_confidence: Option<f32>,
    mask: Option<MaskRle>,
    landmarks: Option<Vec<[f32; 4]>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LowDetection {
    detection_key: u32,
    bbox: BBox,
    detector_score: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedObjectDetection {
    detection_key: u32,
    class_id: usize,
    class_name: String,
    bbox: BBox,
    detector_score: f32,
    mask: Option<MaskRle>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MaskRle {
    resolution: usize,
    coverage: f32,
    rle: Vec<u32>,
}

impl MaskRle {
    fn decode(&self) -> Result<InstanceMask> {
        ensure!(
            self.resolution == mvm_core::mask::MASK_RES,
            "unsupported mask resolution {}",
            self.resolution
        );
        ensure!(self.coverage.is_finite() && (0.0..=1.0).contains(&self.coverage));
        let expected = self.resolution * self.resolution;
        let total: usize = self.rle.iter().map(|run| *run as usize).sum();
        ensure!(total == expected, "mask RLE covers {total}, expected {expected}");
        let mut bits = Vec::with_capacity(expected);
        for (index, run) in self.rle.iter().enumerate() {
            bits.extend(std::iter::repeat_n(u8::from(index % 2 == 1), *run as usize));
        }
        let mask = InstanceMask::from_bits(bits);
        ensure!(
            (mask.frame_coverage() - self.coverage).abs() <= 1.0 / expected as f32,
            "mask coverage differs from RLE"
        );
        Ok(mask)
    }
}

#[derive(Serialize)]
struct ReplayReport {
    schema: &'static str,
    split: String,
    capture_created_at: String,
    capture_mvm_version: String,
    capture_binary_sha256: String,
    privacy: String,
    masks_enabled: bool,
    max_pair_skew_ns: u64,
    models: usize,
    person: DomainReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    person_accuracy: Option<PersonAccuracy>,
    object: DomainReport,
}

#[derive(Default, Serialize)]
struct PersonAccuracy {
    truth_present_ticks: usize,
    truth_absent_ticks: usize,
    baseline_phantom_ticks: usize,
    candidate_phantom_ticks: usize,
    baseline_missed_present_ticks: usize,
    candidate_missed_present_ticks: usize,
    baseline_duplicate_present_ticks: usize,
    candidate_duplicate_present_ticks: usize,
    baseline_visible_track_ids: usize,
    candidate_visible_track_ids: usize,
    baseline_birth_latency_source_frames: Option<u64>,
    candidate_birth_latency_source_frames: Option<u64>,
}

#[derive(Clone)]
struct TruthInterval {
    start: u64,
    end: u64,
    present: bool,
}

#[derive(Default, Serialize)]
struct DomainReport {
    ticks: usize,
    input_detections: usize,
    baseline_births: usize,
    candidate_births: usize,
    baseline_observed_publications: usize,
    candidate_observed_publications: usize,
    baseline_current_ticks: usize,
    candidate_current_ticks: usize,
    baseline_max_current: usize,
    candidate_max_current: usize,
    baseline_final_current: Vec<u32>,
    candidate_final_current: Vec<u32>,
    baseline_current_transitions: Vec<CurrentTransition>,
    candidate_current_transitions: Vec<CurrentTransition>,
    output_mismatch_ticks: usize,
    decision_mismatch_ticks: usize,
}

#[derive(Serialize)]
struct CurrentTransition {
    source_frame_id: u64,
    ids: Vec<u32>,
}

struct Args {
    root: PathBuf,
    split: String,
    metadata_sha256: String,
    human_sha256: String,
    scene_sha256: String,
    truth: Option<(PathBuf, String)>,
}

fn sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn checked_file(path: &Path, expected: &str) -> Result<()> {
    ensure!(
        expected.len() == 64
            && expected
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "expected SHA-256 must be 64 lower-case hex characters"
    );
    let actual = sha256(path)?;
    ensure!(actual == expected, "hash mismatch for {}: expected {expected}, got {actual}", path.display());
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut raw = std::env::args().skip(1);
    let root = PathBuf::from(raw.next().context("missing CAPTURE_DIR")?);
    let mut split = None;
    let mut metadata_sha256 = None;
    let mut human_sha256 = None;
    let mut scene_sha256 = None;
    let mut truth_path = None;
    let mut truth_sha256 = None;
    while let Some(flag) = raw.next() {
        let value = raw.next().with_context(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--split" => split = Some(value),
            "--expected-metadata-sha256" => metadata_sha256 = Some(value),
            "--expected-human-sha256" => human_sha256 = Some(value),
            "--expected-scene-sha256" => scene_sha256 = Some(value),
            "--truth" => truth_path = Some(PathBuf::from(value)),
            "--expected-truth-sha256" => truth_sha256 = Some(value),
            _ => bail!("unknown argument {flag}"),
        }
    }
    let split = split.context("missing --split")?;
    ensure!(split == "development" || split == "calibration", "development runner refuses split {split:?}");
    ensure!(truth_path.is_some() == truth_sha256.is_some(), "--truth and --expected-truth-sha256 must be supplied together");
    Ok(Args {
        root,
        split,
        metadata_sha256: metadata_sha256.context("missing --expected-metadata-sha256")?,
        human_sha256: human_sha256.context("missing --expected-human-sha256")?,
        scene_sha256: scene_sha256.context("missing --expected-scene-sha256")?,
        truth: truth_path.zip(truth_sha256),
    })
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let line = line.with_context(|| format!("read {} line {}", path.display(), index + 1))?;
            serde_json::from_str(&line)
                .with_context(|| format!("parse {} line {}", path.display(), index + 1))
        })
        .collect()
}

fn sorted(mut ids: Vec<u32>) -> Vec<u32> {
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn decisions_for(
    diagnostics: &mvm_core::TrackerDiagnosticsSnapshot,
    binding: &TrackerFrameBinding,
    domain: TrackerDomain,
) -> Vec<TrackerDecision> {
    diagnostics
        .events
        .iter()
        .filter(|event| {
            event.domain == domain
                && event.binding.source_frame_id == binding.source_frame_id
                && event.binding.rgb_stamp == binding.rgb_stamp
        })
        .cloned()
        .collect()
}

fn validate_tick(tick: &TickRecord, domain: TrackerDomain, last_frame: &mut Option<u64>) -> Result<()> {
    ensure!(tick.schema == TICK_SCHEMA, "unsupported tick schema {:?}", tick.schema);
    ensure!(tick.domain == domain, "wrong tracker domain");
    ensure!(
        last_frame.is_none_or(|last| tick.binding.source_frame_id > last),
        "source_frame_id is not strictly increasing"
    );
    *last_frame = Some(tick.binding.source_frame_id);
    ensure!(
        tick.sensor_health.out_of_order_timestamps == 0,
        "capture contains out-of-order sensor timestamps"
    );
    for observation in tick.high_observations.iter().chain(&tick.low_observations) {
        observation
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid captured association observation: {error}"))?;
        ensure!(observation.source == tick.binding, "observation binding mismatch");
    }
    Ok(())
}

fn replay_human(
    metadata: &Metadata,
    ticks: Vec<TickRecord>,
    truth: Option<&[TruthInterval]>,
) -> Result<(DomainReport, Option<PersonAccuracy>)> {
    let mut tracker = mvm_core::PersonTracker::new();
    if let Some(value) = metadata.tracker_profile.pointer("/person/timeout_frames").and_then(|v| v.as_u64()) {
        tracker.timeout_frames = value;
    }
    if let Some(value) = metadata.tracker_profile.pointer("/person/relink_window_frames").and_then(|v| v.as_u64()) {
        tracker.relink_window_frames = value;
    }
    if let Some(value) = metadata.tracker_profile.pointer("/person/relink_center_dist").and_then(|v| v.as_f64()) {
        tracker.relink_center_dist = value as f32;
    }
    let mut names = metadata.gallery_names.clone();
    names.sort();
    for name in names {
        tracker.register_user(&name);
    }
    tracker.set_diagnostic_capacity(TRACE_CAPACITY);
    let mut report = DomainReport::default();
    let mut last_frame = None;
    let mut accuracy = truth.map(|_| PersonAccuracy::default());
    let mut baseline_visible_ids: HashSet<u32> = HashSet::new();
    let mut candidate_visible_ids: HashSet<u32> = HashSet::new();
    let entry_frame = truth.and_then(|values| values.iter().find(|value| value.present).map(|value| value.start));
    let mut baseline_first_current = None;
    let mut candidate_first_current = None;
    let mut previous_baseline_current: Option<Vec<u32>> = None;
    let mut previous_candidate_current: Option<Vec<u32>> = None;
    let first_ns = ticks.first().map(|tick| tick.binding.rgb_stamp.host_monotonic_ns).unwrap_or(0);
    for tick in ticks {
        validate_tick(&tick, TrackerDomain::Person, &mut last_frame)?;
        let DetectorInputs::Human { high, low } = &tick.detector_inputs else {
            bail!("person tick contains object inputs")
        };
        ensure!(high.len() == tick.high_observations.len(), "human high alignment mismatch");
        ensure!(low.len() == tick.low_observations.len(), "human low alignment mismatch");
        let mut detections = Vec::with_capacity(high.len());
        let mut landmarks = Vec::with_capacity(high.len());
        for (input, observation) in high.iter().zip(&tick.high_observations) {
            ensure!(input.detection_key == observation.detection_key);
            ensure!(input.bbox == observation.bbox && input.detector_score == observation.detector_score);
            let mut detection = Detection::bare(input.bbox.x1, input.bbox.y1, input.bbox.x2, input.bbox.y2);
            detection.identity = input.identity.clone();
            detection.identity_confidence = input.identity_confidence;
            detection.mask = input.mask.as_ref().map(MaskRle::decode).transpose()?;
            detections.push(detection);
            landmarks.push(match &input.landmarks {
                Some(values) => Some(values.clone().try_into().map_err(|_| anyhow::anyhow!("human landmarks must contain 33 points"))?),
                None => None,
            });
        }
        let low_boxes: Vec<BBox> = low
            .iter()
            .zip(&tick.low_observations)
            .map(|(input, observation)| {
                ensure!(input.detection_key == observation.detection_key);
                ensure!(input.bbox == observation.bbox && input.detector_score == observation.detector_score);
                Ok(input.bbox)
            })
            .collect::<Result<_>>()?;
        let t_secs = tick.binding.rgb_stamp.host_monotonic_ns.saturating_sub(first_ns) as f64 / 1e9;
        tracker
            .tick_motion_observed_bound(
                &tick.binding,
                &tick.high_observations,
                &tick.low_observations,
                &detections,
                &low_boxes,
                &landmarks,
                t_secs,
            )
            .map_err(|error| anyhow::anyhow!("replay Human tick: {error}"))?;
        let current = sorted(tracker.published().map(|track| track.display_id).collect());
        let occluded = sorted(tracker.occluded().map(|track| track.display_id).collect());
        let baseline_current = sorted(tick.current_ids.clone());
        if current != baseline_current
            || occluded != sorted(tick.occluded_ids.clone())
            || tracker.retained_ids() != sorted(tick.retained_ids.clone())
            || sorted(tracker.retirements().to_vec()) != sorted(tick.retirements.clone())
        {
            report.output_mismatch_ticks += 1;
        }
        let actual_decisions = decisions_for(&tracker.diagnostics(), &tick.binding, TrackerDomain::Person);
        if actual_decisions != tick.decisions {
            report.decision_mismatch_ticks += 1;
        }
        report.ticks += 1;
        report.input_detections += high.len() + low.len();
        report.baseline_births += tick.decisions.iter().filter(|event| event.kind == mvm_core::TrackerDecisionKind::Birth).count();
        report.candidate_births += actual_decisions.iter().filter(|event| event.kind == mvm_core::TrackerDecisionKind::Birth).count();
        report.baseline_observed_publications += tick.decisions.iter().filter(|event| event.kind == mvm_core::TrackerDecisionKind::Publication && event.outcome == "observed").count();
        report.candidate_observed_publications += actual_decisions.iter().filter(|event| event.kind == mvm_core::TrackerDecisionKind::Publication && event.outcome == "observed").count();
        report.baseline_current_ticks += usize::from(!baseline_current.is_empty());
        report.candidate_current_ticks += usize::from(!current.is_empty());
        report.baseline_max_current = report.baseline_max_current.max(baseline_current.len());
        report.candidate_max_current = report.candidate_max_current.max(current.len());
        report.baseline_final_current = baseline_current.clone();
        report.candidate_final_current = current.clone();
        if previous_baseline_current.as_ref() != Some(&baseline_current) {
            report.baseline_current_transitions.push(CurrentTransition {
                source_frame_id: tick.binding.source_frame_id,
                ids: baseline_current.clone(),
            });
            previous_baseline_current = Some(baseline_current.clone());
        }
        if previous_candidate_current.as_ref() != Some(&current) {
            report.candidate_current_transitions.push(CurrentTransition {
                source_frame_id: tick.binding.source_frame_id,
                ids: current.clone(),
            });
            previous_candidate_current = Some(current.clone());
        }
        if let (Some(intervals), Some(accuracy)) = (truth, accuracy.as_mut())
            && let Some(interval) = intervals.iter().find(|interval| {
                (interval.start..=interval.end).contains(&tick.binding.source_frame_id)
            })
        {
            if interval.present {
                accuracy.truth_present_ticks += 1;
                accuracy.baseline_missed_present_ticks += usize::from(baseline_current.is_empty());
                accuracy.candidate_missed_present_ticks += usize::from(current.is_empty());
                accuracy.baseline_duplicate_present_ticks += usize::from(baseline_current.len() > 1);
                accuracy.candidate_duplicate_present_ticks += usize::from(current.len() > 1);
                baseline_visible_ids.extend(baseline_current.iter().copied());
                candidate_visible_ids.extend(current.iter().copied());
                if !baseline_current.is_empty() && baseline_first_current.is_none() {
                    baseline_first_current = Some(tick.binding.source_frame_id);
                }
                if !current.is_empty() && candidate_first_current.is_none() {
                    candidate_first_current = Some(tick.binding.source_frame_id);
                }
            } else {
                accuracy.truth_absent_ticks += 1;
                accuracy.baseline_phantom_ticks += usize::from(!baseline_current.is_empty());
                accuracy.candidate_phantom_ticks += usize::from(!current.is_empty());
            }
        }
    }
    if let Some(accuracy) = accuracy.as_mut() {
        accuracy.baseline_visible_track_ids = baseline_visible_ids.len();
        accuracy.candidate_visible_track_ids = candidate_visible_ids.len();
        accuracy.baseline_birth_latency_source_frames = entry_frame.zip(baseline_first_current).map(|(entry, first)| first.saturating_sub(entry));
        accuracy.candidate_birth_latency_source_frames = entry_frame.zip(candidate_first_current).map(|(entry, first)| first.saturating_sub(entry));
    }
    Ok((report, accuracy))
}

fn load_truth(path: &Path, expected_sha256: &str) -> Result<Vec<TruthInterval>> {
    checked_file(path, expected_sha256)?;
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(value["schema"] == "mvm-tracker-truth-annotation/1.0");
    let intervals = value
        .pointer("/truth_entities/0/intervals")
        .and_then(|value| value.as_array())
        .context("truth annotation has no person intervals")?;
    intervals
        .iter()
        .map(|interval| {
            let state = interval["state"].as_str().context("truth interval state")?;
            ensure!(matches!(state, "visible" | "partial" | "occluded" | "absent" | "unknown"));
            Ok(TruthInterval {
                start: interval["start_frame"].as_u64().context("truth interval start")?,
                end: interval["end_frame"].as_u64().context("truth interval end")?,
                present: matches!(state, "visible" | "partial" | "occluded"),
            })
        })
        .collect()
}

fn class(name: &str, class_id: usize) -> Result<(usize, &'static str)> {
    let Some(found) = COCO_CLASSES.get(class_id) else { bail!("invalid COCO class id {class_id}") };
    ensure!(*found == name, "COCO class id/name mismatch");
    Ok((class_id, found))
}

fn replay_object(metadata: &Metadata, ticks: Vec<TickRecord>) -> Result<DomainReport> {
    let timeout = metadata.tracker_profile.pointer("/object/timeout_frames").and_then(|v| v.as_u64()).context("missing object timeout")?;
    let iou = metadata.tracker_profile.pointer("/object/iou_gate").and_then(|v| v.as_f64()).context("missing object IoU gate")?;
    let mut tracker = ObjectTracker::with_params(iou as f32, u32::try_from(timeout)?, 0.5);
    tracker.set_diagnostic_capacity(TRACE_CAPACITY);
    let mut report = DomainReport::default();
    let mut last_frame = None;
    let mut previous_baseline_current: Option<Vec<u32>> = None;
    let mut previous_candidate_current: Option<Vec<u32>> = None;
    for tick in ticks {
        validate_tick(&tick, TrackerDomain::Object, &mut last_frame)?;
        let DetectorInputs::Object { detections } = &tick.detector_inputs else {
            bail!("object tick contains human inputs")
        };
        ensure!(detections.len() == tick.high_observations.len(), "object alignment mismatch");
        let inputs: Vec<ObjectDetection> = detections
            .iter()
            .zip(&tick.high_observations)
            .map(|(input, observation)| {
                ensure!(input.detection_key == observation.detection_key);
                ensure!(input.bbox == observation.bbox && input.detector_score == observation.detector_score);
                if let Some(mask) = &input.mask { let _ = mask.decode()?; }
                let (class_id, class_name) = class(&input.class_name, input.class_id)?;
                Ok(ObjectDetection { class_id, class_name, bbox: input.bbox, score: input.detector_score })
            })
            .collect::<Result<_>>()?;
        let current = tracker
            .update_observed_bound(&tick.binding, &tick.high_observations, &inputs)
            .map_err(|error| anyhow::anyhow!("replay Scene tick: {error}"))?;
        let current = sorted(current.iter().map(|track| track.id).collect());
        let occluded = sorted(tracker.occluded().map(|track| track.id).collect());
        let baseline_current = sorted(tick.current_ids.clone());
        if current != baseline_current
            || occluded != sorted(tick.occluded_ids.clone())
            || tracker.retained_ids() != sorted(tick.retained_ids.clone())
            || sorted(tracker.retirements().to_vec()) != sorted(tick.retirements.clone())
        {
            report.output_mismatch_ticks += 1;
        }
        let actual_decisions = decisions_for(&tracker.diagnostics(), &tick.binding, TrackerDomain::Object);
        if actual_decisions != tick.decisions {
            report.decision_mismatch_ticks += 1;
        }
        report.ticks += 1;
        report.input_detections += detections.len();
        report.baseline_births += tick.decisions.iter().filter(|event| event.kind == mvm_core::TrackerDecisionKind::Birth).count();
        report.candidate_births += actual_decisions.iter().filter(|event| event.kind == mvm_core::TrackerDecisionKind::Birth).count();
        report.baseline_observed_publications += tick.decisions.iter().filter(|event| event.kind == mvm_core::TrackerDecisionKind::Publication && event.outcome == "observed").count();
        report.candidate_observed_publications += actual_decisions.iter().filter(|event| event.kind == mvm_core::TrackerDecisionKind::Publication && event.outcome == "observed").count();
        report.baseline_current_ticks += usize::from(!baseline_current.is_empty());
        report.candidate_current_ticks += usize::from(!current.is_empty());
        report.baseline_max_current = report.baseline_max_current.max(baseline_current.len());
        report.candidate_max_current = report.candidate_max_current.max(current.len());
        report.baseline_final_current = baseline_current;
        report.candidate_final_current = current;
        if previous_baseline_current.as_ref() != Some(&report.baseline_final_current) {
            report.baseline_current_transitions.push(CurrentTransition {
                source_frame_id: tick.binding.source_frame_id,
                ids: report.baseline_final_current.clone(),
            });
            previous_baseline_current = Some(report.baseline_final_current.clone());
        }
        if previous_candidate_current.as_ref() != Some(&report.candidate_final_current) {
            report.candidate_current_transitions.push(CurrentTransition {
                source_frame_id: tick.binding.source_frame_id,
                ids: report.candidate_final_current.clone(),
            });
            previous_candidate_current = Some(report.candidate_final_current.clone());
        }
    }
    Ok(report)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let metadata_path = args.root.join("capture.json");
    let human_path = args.root.join("human-ticks.jsonl");
    let scene_path = args.root.join("scene-ticks.jsonl");
    checked_file(&metadata_path, &args.metadata_sha256)?;
    checked_file(&human_path, &args.human_sha256)?;
    checked_file(&scene_path, &args.scene_sha256)?;
    let metadata: Metadata = serde_json::from_slice(&std::fs::read(&metadata_path)?)?;
    ensure!(metadata.schema == METADATA_SCHEMA);
    ensure!(metadata.tick_schema == TICK_SCHEMA);
    ensure!(metadata.max_pair_skew_ns > 0);
    ensure!(metadata.binary_sha256.len() == 64);
    ensure!(metadata.model_digests.iter().all(|model| model.sha256.len() == 64 && !model.file.is_empty()));
    let truth = args
        .truth
        .as_ref()
        .map(|(path, hash)| load_truth(path, hash))
        .transpose()?;
    let (person, person_accuracy) = replay_human(&metadata, read_jsonl(&human_path)?, truth.as_deref())?;
    let object = replay_object(&metadata, read_jsonl(&scene_path)?)?;
    let report = ReplayReport {
        schema: "mvm-tracker-capture-replay-report/1.0",
        split: args.split,
        capture_created_at: metadata.created_at,
        capture_mvm_version: metadata.mvm_version,
        capture_binary_sha256: metadata.binary_sha256,
        privacy: metadata.privacy,
        masks_enabled: metadata.masks_enabled,
        max_pair_skew_ns: metadata.max_pair_skew_ns,
        models: metadata.model_digests.len(),
        person,
        person_accuracy,
        object,
    };
    serde_json::to_writer_pretty(std::io::stdout(), &report)?;
    println!();
    Ok(())
}
