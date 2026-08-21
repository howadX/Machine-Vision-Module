//! Fast RGB/RGB-D PersonTracker owner, independent from semantic Human models.

use crate::agents::{
    AssociationMeasurementContext, HumanEnrichmentBus, HumanObservationBus, HumanObservationFrame,
    HumanWorkBus, HumanWorkItem, ObservationStamp, SceneObservationBus, SceneObservationFrame,
    TrackedPersonWork, collect_occluded_people, collect_people,
};
use mvm_core::{
    BBox, BodyObservation, EnrichmentApply, PersonBirthConflict, SceneObjectEvidence,
    StaticDisplayClass, mask::InstanceMask,
};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::sync::Arc;

const PERSON_MASK_IOU: f32 = 0.3;
const STATIC_DISPLAY_PERSON_COVERAGE: f32 = 0.20;
const STATIC_DISPLAY_MIN_CONFIDENCE: f32 = 0.30;

fn normalized_bbox(person: &mvm_posture::detector::Person, width: f32, height: f32) -> BBox {
    BBox {
        x1: (person.bbox[0] / width).clamp(0.0, 1.0),
        y1: (person.bbox[1] / height).clamp(0.0, 1.0),
        x2: (person.bbox[2] / width).clamp(0.0, 1.0),
        y2: (person.bbox[3] / height).clamp(0.0, 1.0),
    }
}

fn attach_person_masks(
    observations: &mut [BodyObservation],
    person_masks: &[(BBox, InstanceMask)],
) {
    let mut used = vec![false; person_masks.len()];
    for observation in observations {
        let best = person_masks
            .iter()
            .enumerate()
            .filter(|(index, _)| !used[*index])
            .map(|(index, (bbox, _))| (index, observation.bbox.iou(bbox)))
            .filter(|(_, iou)| *iou > PERSON_MASK_IOU)
            .max_by(|(_, a), (_, b)| a.total_cmp(b));
        if let Some((index, _)) = best {
            used[index] = true;
            observation.mask = Some(person_masks[index].1.clone());
        }
    }
}

fn intersection_coverage(subject: BBox, other: BBox) -> f32 {
    let intersection = (subject.x2.min(other.x2) - subject.x1.max(other.x1)).max(0.0)
        * (subject.y2.min(other.y2) - subject.y1.max(other.y1)).max(0.0);
    let area = subject.area();
    if area <= f32::EPSILON {
        0.0
    } else {
        (intersection / area).clamp(0.0, 1.0)
    }
}

fn static_display_class(class: &str) -> Option<StaticDisplayClass> {
    match class {
        "tv" => Some(StaticDisplayClass::Tv),
        "laptop" => Some(StaticDisplayClass::Laptop),
        _ => None,
    }
}

fn display_birth_conflict(
    person_bbox: BBox,
    scene: &SceneObservationFrame,
) -> Option<PersonBirthConflict> {
    scene
        .objects
        .iter()
        .map(|object| {
            let evidence = if object.remembered {
                SceneObjectEvidence::Retained
            } else {
                SceneObjectEvidence::Current
            };
            (object, evidence)
        })
        .chain(
            scene
                .occluded_objects
                .iter()
                .map(|object| (object, SceneObjectEvidence::Retained)),
        )
        .filter(|(object, _)| object.age >= 2 && object.confidence >= STATIC_DISPLAY_MIN_CONFIDENCE)
        .filter_map(|(object, object_evidence)| {
            let display_class = static_display_class(&object.class)?;
            let person_coverage = intersection_coverage(person_bbox, object.bbox);
            (person_coverage >= STATIC_DISPLAY_PERSON_COVERAGE).then_some(PersonBirthConflict {
                scene_source_frame_id: scene.stamp.source_frame_id,
                object_track_id: object.id,
                display_class,
                object_evidence,
                person_coverage,
            })
        })
        .max_by(|left, right| {
            left.person_coverage
                .total_cmp(&right.person_coverage)
                .then_with(|| left.object_evidence.cmp(&right.object_evidence))
                .then_with(|| right.object_track_id.cmp(&left.object_track_id))
        })
}

fn attach_display_birth_conflicts(
    observations: &mut [mvm_core::AssociationObservation],
    scene: &SceneObservationFrame,
) {
    for observation in observations {
        observation.person_birth_conflict = display_birth_conflict(observation.bbox, scene);
    }
}

pub(crate) struct TrackingAgent {
    detector: mvm_posture::detector::PersonDetector,
    tracker: mvm_core::PersonTracker,
    scene_observations: SceneObservationBus,
    observations: HumanObservationBus,
    work: Option<HumanWorkBus>,
    enrichment: Option<HumanEnrichmentBus>,
    publish_canonical: bool,
    max_mask_age_frames: u64,
    max_pair_skew_ns: u64,
    metrics: super::SchedulingMetrics,
    calibration_capture: Option<super::TrackerCalibrationCapture>,
    latest_bodies: Vec<crate::pipeline::BodyData>,
    last_emotion_ns: std::collections::HashMap<mvm_core::PersonTrackHandle, u64>,
    pending_enrichment_trace: Vec<(
        mvm_core::TrackerFrameBinding,
        mvm_core::PersonTrackHandle,
        mvm_core::PersonEnrichment,
        EnrichmentApply,
    )>,
}

impl TrackingAgent {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        detector: mvm_posture::detector::PersonDetector,
        tracker: mvm_core::PersonTracker,
        scene_observations: SceneObservationBus,
        observations: HumanObservationBus,
        work: Option<HumanWorkBus>,
        enrichment: Option<HumanEnrichmentBus>,
        publish_canonical: bool,
        object_cadence: u32,
        metrics: super::SchedulingMetrics,
        max_pair_skew_ns: u64,
    ) -> Self {
        Self {
            detector,
            tracker,
            scene_observations,
            observations,
            work,
            enrichment,
            publish_canonical,
            max_mask_age_frames: u64::from(object_cadence.max(1)) * 2,
            max_pair_skew_ns,
            metrics,
            calibration_capture: None,
            latest_bodies: Vec::new(),
            last_emotion_ns: std::collections::HashMap::new(),
            pending_enrichment_trace: Vec::new(),
        }
    }

    pub(crate) fn with_calibration_capture(
        mut self,
        capture: Option<super::TrackerCalibrationCapture>,
    ) -> Self {
        if capture.is_some() {
            self.tracker
                .set_diagnostic_capacity(super::tracker_capture::TRACE_CAPACITY);
        }
        self.calibration_capture = capture;
        self
    }
}

impl Agent for TrackingAgent {
    fn name(&self) -> String {
        if self.publish_canonical {
            "tracking".into()
        } else {
            "tracking-shadow".into()
        }
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0;
        let mut errors = 0;
        let mut last_frame: Option<(mvm_core::SensorSourceId, u64, u64)> = None;
        let mut last_enrichment: Option<(mvm_core::SensorSourceId, u64, u64)> = None;
        while ctx.running() {
            if let Some(result) = self
                .enrichment
                .as_ref()
                .and_then(HumanEnrichmentBus::latest)
            {
                let key = (
                    result.binding.source.clone(),
                    result.source_frame_id,
                    result.binding.rgb_stamp.host_monotonic_ns,
                );
                if last_enrichment.as_ref() != Some(&key) {
                    let mut accepted = false;
                    for item in result.items.iter() {
                        let outcome = self.tracker.apply_enrichment(
                            item.handle,
                            &result.binding,
                            &item.patch,
                        );
                        let applied = outcome == EnrichmentApply::Applied;
                        self.pending_enrichment_trace.push((
                            result.binding.clone(),
                            item.handle,
                            item.patch.clone(),
                            outcome,
                        ));
                        if applied
                            && (item.patch.emotion.is_some()
                                || item.patch.emotion_distribution.is_some())
                        {
                            self.last_emotion_ns
                                .insert(item.handle, result.binding.rgb_stamp.host_monotonic_ns);
                        }
                        accepted |= applied;
                    }
                    if accepted {
                        self.latest_bodies = result.bodies.as_ref().clone();
                    }
                    last_enrichment = Some(key);
                }
            }

            if let Some(frame) = ctx.frames.latest() {
                let frame_key = (
                    frame.sensor.source.clone(),
                    frame.id,
                    frame.sensor.rgb_stamp.host_monotonic_ns,
                );
                if last_frame.as_ref() == Some(&frame_key) {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                last_frame = Some(frame_key);
                let started = std::time::Instant::now();
                let previous_tracks: Vec<_> =
                    self.tracker.active().map(|track| track.bbox).collect();
                let detector_started = std::time::Instant::now();
                let persons = match self
                    .detector
                    .detect_scored(frame.image.as_ref(), mvm_posture::BYTE_LOW_SCORE)
                {
                    Ok(persons) => persons,
                    Err(error) => {
                        eprintln!("  tracking: person detector failed: {error:#}");
                        errors += 1;
                        continue;
                    }
                };
                self.metrics.record_duration(
                    super::MetricAgent::TrackingDetector,
                    detector_started.elapsed(),
                );
                let width = frame.image.width() as f32;
                let height = frame.image.height() as f32;
                let high_persons: Vec<_> = persons
                    .iter()
                    .filter(|person| person.score >= mvm_posture::BYTE_HIGH_SCORE)
                    .cloned()
                    .collect();
                let low_people: Vec<_> = persons
                    .iter()
                    .filter(|person| person.score < mvm_posture::BYTE_HIGH_SCORE)
                    .cloned()
                    .collect();
                let mut high: Vec<_> = high_persons
                    .iter()
                    .map(|person| BodyObservation {
                        bbox: normalized_bbox(person, width, height),
                        score: person.score,
                        landmarks: None,
                        mask: None,
                    })
                    .collect();
                let low: Vec<_> = low_people
                    .iter()
                    .map(|person| normalized_bbox(person, width, height))
                    .collect();
                let recent_scene = self.scene_observations.latest().filter(|scene| {
                    scene
                        .stamp
                        .age_at(frame.id)
                        .is_some_and(|age| age <= self.max_mask_age_frames)
                });
                let mut current_mask_frame = false;
                if let Some(scene) = &recent_scene {
                    current_mask_frame = scene.stamp.source_frame_id == frame.id;
                    attach_person_masks(&mut high, &scene.person_masks);
                }
                let association_started = std::time::Instant::now();
                let measurement = match AssociationMeasurementContext::from_frame(
                    &frame,
                    self.max_pair_skew_ns,
                ) {
                    Ok(measurement) => measurement,
                    Err(error) => {
                        eprintln!("  tracking: rejected sensor binding: {error}");
                        errors += 1;
                        continue;
                    }
                };
                if let Some(capture) = &self.calibration_capture {
                    for (source, handle, patch, outcome) in self.pending_enrichment_trace.drain(..)
                    {
                        if let Err(error) = capture.write_enrichment_application(
                            &source,
                            measurement.binding(),
                            handle,
                            outcome,
                            &patch,
                        ) {
                            eprintln!("  tracking: enrichment capture failed: {error:#}");
                            errors += 1;
                            ctx.request_stop();
                            break;
                        }
                    }
                } else {
                    self.pending_enrichment_trace.clear();
                }
                let mut measurement_inputs: Vec<_> = high
                    .iter()
                    .enumerate()
                    .map(|(index, body)| {
                        (
                            u32::try_from(index).unwrap_or(u32::MAX),
                            body.bbox,
                            body.score,
                            current_mask_frame.then_some(body.mask.as_ref()).flatten(),
                            true,
                        )
                    })
                    .collect();
                let low_metric_candidates = self
                    .tracker
                    .low_metric_candidates(measurement.binding(), &low);
                measurement_inputs.extend(
                    low.iter()
                        .zip(&low_people)
                        .zip(low_metric_candidates)
                        .enumerate()
                        .map(|(index, ((bbox, person), measure))| {
                            (
                                u32::try_from(high.len() + index).unwrap_or(u32::MAX),
                                *bbox,
                                person.score,
                                None,
                                measure,
                            )
                        }),
                );
                let mut all_observations = measurement.observations(&measurement_inputs);
                let (high_observations, low_observations) =
                    all_observations.split_at_mut(high.len());
                if let Some(scene) = &recent_scene {
                    attach_display_birth_conflicts(high_observations, scene);
                }
                let handles = match self.tracker.tick_observations_bound(
                    measurement.binding(),
                    high_observations,
                    low_observations,
                    &high,
                    &low,
                ) {
                    Ok(handles) => handles,
                    Err(error) => {
                        eprintln!("  tracking: rejected association observations: {error}");
                        errors += 1;
                        continue;
                    }
                };
                self.metrics.record_duration(
                    super::MetricAgent::TrackingAssociation,
                    association_started.elapsed(),
                );
                self.last_emotion_ns
                    .retain(|handle, _| self.tracker.track_for_handle(*handle).is_some());

                if let Some(work) = &self.work {
                    let people: Vec<_> = high_persons
                        .iter()
                        .cloned()
                        .zip(handles.iter().copied())
                        .filter_map(|(person, handle)| {
                            handle.map(|handle| TrackedPersonWork { handle, person })
                        })
                        .collect();
                    if !people.is_empty() {
                        let mut locked_tracks = Vec::new();
                        let mut emotion_fresh_tracks = Vec::new();
                        for person in &people {
                            if let Some(track) = self.tracker.track_for_handle(person.handle) {
                                if track.is_id_locked() {
                                    locked_tracks.push(track.bbox);
                                }
                                let moving = !track.motion.is_empty()
                                    && track.motion.state() != mvm_core::motion::MotionState::Idle;
                                let previous_emotion =
                                    self.last_emotion_ns.get(&person.handle).copied();
                                let refresh_ns = u64::from(mvm_core::types::EMOTION_CADENCE)
                                    .saturating_mul(33_333_333);
                                let emotion_due = previous_emotion.is_none_or(|previous| {
                                    measurement
                                        .binding()
                                        .rgb_stamp
                                        .host_monotonic_ns
                                        .saturating_sub(previous)
                                        >= refresh_ns
                                });
                                if !emotion_due || (previous_emotion.is_some() && moving) {
                                    emotion_fresh_tracks.push(track.bbox);
                                }
                            }
                        }
                        work.publish(HumanWorkItem {
                            frame: frame.clone(),
                            binding: measurement.binding().clone(),
                            people: Arc::new(people),
                            previous_tracks: Arc::new(previous_tracks),
                            locked_tracks: Arc::new(locked_tracks),
                            emotion_fresh_tracks: Arc::new(emotion_fresh_tracks),
                        });
                    }
                }

                let stamp = ObservationStamp::from_frame(&frame);
                let people = collect_people(&self.tracker, &self.latest_bodies, stamp.produced_at);
                let published_ids: std::collections::HashSet<_> =
                    people.iter().map(|person| person.tracker_id).collect();
                let current_ids: Vec<_> = people.iter().map(|person| person.tracker_id).collect();
                let occluded_people = collect_occluded_people(&self.tracker, stamp.produced_at);
                let occluded_ids: Vec<_> = occluded_people
                    .iter()
                    .map(|person| person.tracker_id)
                    .collect();
                let retained_ids = self.tracker.retained_ids();
                let retirements = self.tracker.retirements().to_vec();
                if let Some(capture) = &self.calibration_capture {
                    let detections: Vec<_> = high
                        .iter()
                        .map(|body| {
                            let mut detection = mvm_core::Detection::bare(
                                body.bbox.x1,
                                body.bbox.y1,
                                body.bbox.x2,
                                body.bbox.y2,
                            );
                            detection.mask = body.mask.clone();
                            detection
                        })
                        .collect();
                    let landmarks = vec![None; detections.len()];
                    if let Err(error) = capture.write_human(
                        measurement.binding(),
                        &frame.sensor.health,
                        high_observations,
                        low_observations,
                        &detections,
                        &low,
                        &landmarks,
                        Some((&high_persons, &low_people)),
                        &self.tracker.diagnostics(),
                        &current_ids,
                        &occluded_ids,
                        &retained_ids,
                        &retirements,
                    ) {
                        eprintln!("  tracking: calibration capture failed: {error:#}");
                        errors += 1;
                        ctx.request_stop();
                        continue;
                    }
                }
                if self.publish_canonical {
                    self.observations.publish(HumanObservationFrame {
                        stamp,
                        people: Arc::new(people),
                        occluded_people: Arc::new(occluded_people),
                        retained_track_ids: Arc::new(retained_ids),
                        retirements: Arc::new(retirements),
                        metric_geometry: Arc::new(
                            self.tracker
                                .association_geometry()
                                .iter()
                                .filter(|value| published_ids.contains(&value.track_id))
                                .cloned()
                                .collect(),
                        ),
                    });
                }
                self.metrics
                    .record_duration(super::MetricAgent::Tracking, started.elapsed());
                self.metrics.record_tracking_frame(frame.id);
                ticks += 1;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        AgentHealth {
            name: self.name(),
            ticks,
            errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_split_preserves_byte_thresholds() {
        assert_eq!(mvm_posture::BYTE_LOW_SCORE, 0.2);
        assert_eq!(mvm_posture::BYTE_HIGH_SCORE, 0.5);
    }

    #[test]
    fn tracking_uses_stable_raw_anchor_after_body_geometry_validation() {
        let person = mvm_posture::detector::Person {
            bbox: [100.0, 20.0, 180.0, 100.0],
            keypoints: [
                [320.0, 300.0],
                [320.0, 100.0],
                [320.0, 180.0],
                [320.0, 120.0],
            ],
            score: 0.8,
        };

        assert_eq!(
            normalized_bbox(&person, 640.0, 480.0),
            BBox {
                x1: 100.0 / 640.0,
                y1: 20.0 / 480.0,
                x2: 180.0 / 640.0,
                y2: 100.0 / 480.0,
            }
        );
    }

    fn scene_with_object(class: &str, remembered: bool) -> SceneObservationFrame {
        let now = std::time::Instant::now();
        SceneObservationFrame {
            stamp: ObservationStamp {
                source_frame_id: 10,
                captured_at: now,
                produced_at: now,
                captured_at_unix_ms: 0,
                produced_at_unix_ms: 0,
            },
            objects: Arc::new(vec![mvm_core::ObjectState {
                id: 10_000,
                class: class.into(),
                bbox: BBox {
                    x1: 0.42,
                    y1: 0.22,
                    x2: 0.68,
                    y2: 0.42,
                },
                confidence: 0.8,
                age: 3,
                position_3d: None,
                mask: None,
                remembered,
                last_seen: now,
            }]),
            occluded_objects: Arc::new(Vec::new()),
            retained_track_ids: Arc::new(Vec::new()),
            retirements: Arc::new(Vec::new()),
            animals: Arc::new(Vec::new()),
            person_masks: Arc::new(Vec::new()),
            metric_geometry: Arc::new(Vec::new()),
        }
    }

    #[test]
    fn confirmed_current_display_overlap_becomes_typed_birth_conflict() {
        let person = BBox {
            x1: 0.4,
            y1: 0.2,
            x2: 0.7,
            y2: 0.6,
        };
        let conflict = display_birth_conflict(person, &scene_with_object("tv", false)).unwrap();
        assert_eq!(conflict.scene_source_frame_id, 10);
        assert_eq!(conflict.object_track_id, 10_000);
        assert_eq!(conflict.display_class, StaticDisplayClass::Tv);
        assert!(conflict.person_coverage >= STATIC_DISPLAY_PERSON_COVERAGE);
    }

    #[test]
    fn unrelated_objects_do_not_veto_but_retained_displays_do() {
        let person = BBox {
            x1: 0.4,
            y1: 0.2,
            x2: 0.7,
            y2: 0.6,
        };
        assert!(display_birth_conflict(person, &scene_with_object("chair", false)).is_none());
        let retained = display_birth_conflict(person, &scene_with_object("laptop", true)).unwrap();
        assert_eq!(retained.object_evidence, SceneObjectEvidence::Retained);
    }
}
