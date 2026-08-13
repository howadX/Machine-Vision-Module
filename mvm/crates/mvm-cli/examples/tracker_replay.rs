//! Production-parity, image-free tracker replay.
//!
//! Usage:
//! `cargo run -p mvm-cli --example tracker_replay -- INPUT --expected-sha256 HEX`

use anyhow::{Context, Result, bail};
use mvm_core::{
    AssociationObservation, BBox, Detection, PersonTracker, SensorReplayEnvelope,
    TrackedAssociationGeometry, TrackerFrameBinding,
};
use mvm_scene::{COCO_CLASSES, ObjectDetection, ObjectTracker};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const SCHEMA: &str = "mvm-tracker-production-replay/1.0";
const TRACE_CAPACITY: usize = 16_384;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayInput {
    schema: String,
    split: String,
    max_pair_skew_ns: u64,
    frames: Vec<ReplayFrame>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayFrame {
    sensor: SensorReplayEnvelope,
    #[serde(default)]
    people_high: Vec<ReplayPerson>,
    #[serde(default)]
    people_low: Vec<[f32; 4]>,
    #[serde(default)]
    objects: Vec<ReplayObject>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayPerson {
    bbox: [f32; 4],
    #[serde(default = "one")]
    score: f32,
    #[serde(default)]
    identity: Option<String>,
    #[serde(default)]
    identity_confidence: Option<f32>,
}

fn one() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayObject {
    class_name: String,
    bbox: [f32; 4],
    score: f32,
}

#[derive(Debug, Serialize)]
struct PersonSummary {
    id: u32,
    display_id: u32,
    age: u64,
    missed: u64,
    lifecycle: &'static str,
}

#[derive(Debug, Serialize)]
struct ObjectSummary {
    id: u32,
    class_name: &'static str,
    age: u32,
    missed: u32,
    lifecycle: &'static str,
}

#[derive(Debug, Serialize)]
struct ReplayOutput {
    schema: &'static str,
    input_sha256: String,
    split: String,
    frames: usize,
    person: DomainOutput<PersonSummary>,
    object: DomainOutput<ObjectSummary>,
}

#[derive(Debug, Serialize)]
struct DomainOutput<T> {
    final_tracks: Vec<T>,
    measurements: Vec<TrackedAssociationGeometry>,
    diagnostics: mvm_core::TrackerDiagnosticsSnapshot,
}

fn bbox(raw: [f32; 4]) -> Result<BBox> {
    let value = BBox {
        x1: raw[0],
        y1: raw[1],
        x2: raw[2],
        y2: raw[3],
    };
    if raw.iter().all(|v| v.is_finite())
        && value.x1 >= 0.0
        && value.y1 >= 0.0
        && value.x2 <= 1.0
        && value.y2 <= 1.0
        && value.x2 > value.x1
        && value.y2 > value.y1
    {
        Ok(value)
    } else {
        bail!("invalid normalized bbox {raw:?}")
    }
}

fn class(name: &str) -> Result<(usize, &'static str)> {
    COCO_CLASSES
        .iter()
        .position(|candidate| *candidate == name)
        .map(|id| (id, COCO_CLASSES[id]))
        .with_context(|| format!("unknown COCO class {name:?}"))
}

fn parse_args() -> Result<(PathBuf, String)> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(args.next().context("missing INPUT")?);
    if args.next().as_deref() != Some(std::ffi::OsStr::new("--expected-sha256")) {
        bail!("expected --expected-sha256 HEX");
    }
    let expected = args
        .next()
        .context("missing expected SHA-256")?
        .into_string()
        .map_err(|_| anyhow::anyhow!("SHA-256 must be UTF-8"))?;
    if args.next().is_some() {
        bail!("unexpected trailing arguments");
    }
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        bail!("expected SHA-256 must be 64 lower-case hex characters");
    }
    Ok((input, expected))
}

fn validate_header(input: &ReplayInput) -> Result<()> {
    if input.schema != SCHEMA {
        bail!("unsupported replay schema {:?}", input.schema);
    }
    if input.split != "development" && input.split != "calibration" {
        bail!(
            "split {:?} is not executable by this development replay",
            input.split
        );
    }
    if input.frames.is_empty() || input.max_pair_skew_ns == 0 {
        bail!("replay needs frames and a positive max_pair_skew_ns");
    }
    Ok(())
}

fn main() -> Result<()> {
    let (path, expected_sha256) = parse_args()?;
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha256 != expected_sha256 {
        bail!("input hash mismatch: expected {expected_sha256}, got {actual_sha256}");
    }
    let input: ReplayInput = serde_json::from_slice(&bytes).context("parse replay schema")?;
    validate_header(&input)?;

    let mut person = PersonTracker::new();
    person.set_diagnostic_capacity(TRACE_CAPACITY);
    let mut object = ObjectTracker::new();
    object.set_diagnostic_capacity(TRACE_CAPACITY);
    let mut last_frame_id = None;
    let mut person_measurements = Vec::new();
    let mut object_measurements = Vec::new();

    for frame in &input.frames {
        frame
            .sensor
            .validate(input.max_pair_skew_ns)
            .with_context(|| format!("validate sensor envelope {}", frame.sensor.case_id))?;
        let frame_id = frame.sensor.rgb_stamp.source_sequence;
        if last_frame_id.is_some_and(|last| frame_id <= last) {
            bail!("RGB source_sequence must be strictly increasing");
        }
        last_frame_id = Some(frame_id);
        let binding = TrackerFrameBinding::from_sensor_parts(
            frame_id,
            &frame.sensor.source,
            frame.sensor.rgb_width,
            frame.sensor.rgb_height,
            &frame.sensor.rgb_stamp,
            &frame.sensor.depth_status,
            frame.sensor.metric_depth.as_ref(),
            frame.sensor.calibration.as_ref(),
            input.max_pair_skew_ns,
        )?;
        let admitted_depth = match (
            frame.sensor.metric_depth.as_ref(),
            frame.sensor.calibration.as_ref(),
        ) {
            (Some(depth), Some(calibration)) => Some(
                mvm_depth::RegisteredMetricDepth::admit(
                    frame.sensor.rgb_width,
                    frame.sensor.rgb_height,
                    &frame.sensor.rgb_stamp,
                    &frame.sensor.depth_status,
                    std::sync::Arc::new(depth.clone()),
                    calibration,
                    input.max_pair_skew_ns,
                )
                .map_err(|error| anyhow::anyhow!("metric depth admission failed: {error:?}"))?,
            ),
            _ => None,
        };

        let high: Vec<Detection> = frame
            .people_high
            .iter()
            .map(|raw| {
                let b = bbox(raw.bbox)?;
                let mut detection = Detection::bare(b.x1, b.y1, b.x2, b.y2);
                detection.identity = raw.identity.clone();
                detection.identity_confidence = raw.identity_confidence;
                Ok(detection)
            })
            .collect::<Result<_>>()?;
        let low: Vec<BBox> = frame
            .people_low
            .iter()
            .copied()
            .map(bbox)
            .collect::<Result<_>>()?;
        let high_observations: Vec<_> = frame
            .people_high
            .iter()
            .zip(&high)
            .enumerate()
            .map(|(index, (raw, detection))| AssociationObservation {
                detection_key: u32::try_from(index).unwrap_or(u32::MAX),
                source: binding.clone(),
                bbox: detection.bbox,
                detector_score: raw.score,
                geometry: mvm_depth::measure_association_geometry(
                    &binding,
                    admitted_depth.as_ref(),
                    None,
                    detection.bbox,
                    None,
                ),
            })
            .collect();
        let low_observations: Vec<_> = low
            .iter()
            .enumerate()
            .map(|(index, bbox)| AssociationObservation {
                detection_key: u32::try_from(high.len() + index).unwrap_or(u32::MAX),
                source: binding.clone(),
                bbox: *bbox,
                detector_score: 0.25,
                geometry: mvm_depth::measure_association_geometry(
                    &binding,
                    admitted_depth.as_ref(),
                    None,
                    *bbox,
                    None,
                ),
            })
            .collect();
        person
            .tick_motion_observed_bound(
                &binding,
                &high_observations,
                &low_observations,
                &high,
                &low,
                &[],
                frame_id as f64 / 30.0,
            )
            .map_err(|error| anyhow::anyhow!(error))?;
        person_measurements.extend_from_slice(person.association_geometry());

        let objects: Vec<ObjectDetection> = frame
            .objects
            .iter()
            .map(|raw| {
                let (class_id, class_name) = class(&raw.class_name)?;
                if !raw.score.is_finite() || !(0.0..=1.0).contains(&raw.score) {
                    bail!("invalid object score {}", raw.score);
                }
                Ok(ObjectDetection {
                    class_id,
                    class_name,
                    bbox: bbox(raw.bbox)?,
                    score: raw.score,
                })
            })
            .collect::<Result<_>>()?;
        let object_observations: Vec<_> = objects
            .iter()
            .enumerate()
            .map(|(index, detection)| AssociationObservation {
                detection_key: u32::try_from(index).unwrap_or(u32::MAX),
                source: binding.clone(),
                bbox: detection.bbox,
                detector_score: detection.score,
                geometry: mvm_depth::measure_association_geometry(
                    &binding,
                    admitted_depth.as_ref(),
                    None,
                    detection.bbox,
                    None,
                ),
            })
            .collect();
        object
            .update_observed_bound(&binding, &object_observations, &objects)
            .map_err(|error| anyhow::anyhow!(error))?;
        object_measurements.extend_from_slice(object.association_geometry());
    }

    let mut people: Vec<PersonSummary> = person
        .all()
        .values()
        .map(|track| PersonSummary {
            id: track.id,
            display_id: track.display_id,
            age: track.age,
            missed: track.missed,
            lifecycle: track.lifecycle.as_str(),
        })
        .collect();
    people.sort_by_key(|track| track.id);
    let mut objects: Vec<ObjectSummary> = object
        .all()
        .iter()
        .map(|track| ObjectSummary {
            id: track.id,
            class_name: track.class_name,
            age: track.age,
            missed: track.missed,
            lifecycle: track.lifecycle.as_str(),
        })
        .collect();
    objects.sort_by_key(|track| track.id);

    let output = ReplayOutput {
        schema: SCHEMA,
        input_sha256: actual_sha256,
        split: input.split,
        frames: input.frames.len(),
        person: DomainOutput {
            final_tracks: people,
            measurements: person_measurements,
            diagnostics: person.diagnostics(),
        },
        object: DomainOutput {
            final_tracks: objects,
            measurements: object_measurements,
            diagnostics: object.diagnostics(),
        },
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../../benchmark/tracker-reliability/replay/production-baseline.json");

    #[test]
    fn fixture_hash_is_frozen() {
        assert_eq!(
            format!("{:x}", Sha256::digest(FIXTURE.as_bytes())),
            "b99dfab66f9615b7b3119ee06ae7f95f25924a7cc0f96e20b976e9e571fca8b8"
        );
    }

    #[test]
    fn schema_unknown_fields_and_held_out_split_fail_closed() {
        let mut replay: ReplayInput = serde_json::from_str(FIXTURE).unwrap();
        replay.schema = "mvm-tracker-production-replay/0.9".into();
        assert!(validate_header(&replay).is_err());
        replay.schema = SCHEMA.into();
        replay.split = "held-out".into();
        assert!(validate_header(&replay).is_err());

        let malformed = FIXTURE.replacen('{', "{\"unknown\":true,", 1);
        assert!(serde_json::from_str::<ReplayInput>(&malformed).is_err());
    }
}
