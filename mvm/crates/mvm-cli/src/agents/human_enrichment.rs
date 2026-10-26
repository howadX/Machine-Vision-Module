//! Asynchronous semantic Human enrichment for the fast TrackingAgent path.
//! It never owns a tracker and never publishes spatial observations.

use crate::agents::{HumanEnrichmentBus, HumanEnrichmentFrame, HumanEnrichmentItem, HumanWorkBus};
use crate::pipeline::BodyPrimaryPipeline;
use mvm_core::{BBox, Detection, PersonEnrichment};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::sync::Arc;

pub(crate) struct HumanAgent {
    pipeline: BodyPrimaryPipeline,
    work: HumanWorkBus,
    results: HumanEnrichmentBus,
    metrics: super::SchedulingMetrics,
}

impl HumanAgent {
    pub(crate) fn new(
        pipeline: BodyPrimaryPipeline,
        work: HumanWorkBus,
        results: HumanEnrichmentBus,
        metrics: super::SchedulingMetrics,
    ) -> Self {
        Self {
            pipeline,
            work,
            results,
            metrics,
        }
    }
}

fn normalized_bbox(person: &mvm_posture::detector::Person, width: f32, height: f32) -> BBox {
    BBox {
        x1: (person.bbox[0] / width).clamp(0.0, 1.0),
        y1: (person.bbox[1] / height).clamp(0.0, 1.0),
        x2: (person.bbox[2] / width).clamp(0.0, 1.0),
        y2: (person.bbox[3] / height).clamp(0.0, 1.0),
    }
}

fn semantic_patch(
    detection: &Detection,
    landmarks: Option<mvm_core::motion::Landmarks33>,
) -> PersonEnrichment {
    PersonEnrichment {
        identity: detection.identity.clone(),
        identity_confidence: detection.identity_confidence,
        emotion: detection.emotion.clone(),
        emotion_confidence: detection.emotion_confidence,
        emotion_distribution: detection.emotion_distribution,
        posture: detection.posture.clone(),
        posture_confidence: detection.posture_confidence,
        recon_error: detection.recon_error,
        plausibility: detection.plausibility.clone(),
        torso_flexion: detection.torso_flexion,
        body_coverage: detection.body_coverage,
        torso_attitude: detection.torso_attitude,
        embedding: detection.embedding.clone(),
        head_yaw: detection.head_yaw,
        head_facing: detection.head_facing,
        landmarks,
    }
}

fn best_detection(target: BBox, detections: &[Detection], used: &[bool]) -> Option<usize> {
    detections
        .iter()
        .enumerate()
        .filter(|(index, _)| !used[*index])
        .map(|(index, detection)| (index, target.iou(&detection.bbox)))
        .filter(|(_, iou)| *iou > 0.1)
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
}

impl Agent for HumanAgent {
    fn name(&self) -> String {
        "human".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0;
        let errors = 0;
        let mut last_work: Option<(mvm_core::SensorSourceId, u64, u64)> = None;
        while ctx.running() {
            if let Some(work) = self.work.latest() {
                let key = (
                    work.binding.source.clone(),
                    work.binding.source_frame_id,
                    work.binding.rgb_stamp.host_monotonic_ns,
                );
                if last_work.as_ref() != Some(&key) {
                    last_work = Some(key);
                    let started = std::time::Instant::now();
                    let persons: Vec<_> = work
                        .people
                        .iter()
                        .map(|person| person.person.clone())
                        .collect();
                    let result = self.pipeline.process_pretracked(
                        work.frame.image.as_ref(),
                        work.frame.id,
                        &work.previous_tracks,
                        &work.locked_tracks,
                        &work.emotion_fresh_tracks,
                        &persons,
                    );
                    let width = work.frame.image.width() as f32;
                    let height = work.frame.image.height() as f32;
                    let mut used = vec![false; result.detections.len()];
                    let mut items = Vec::with_capacity(work.people.len());
                    for person in work.people.iter() {
                        let target = normalized_bbox(&person.person, width, height);
                        let Some(index) = best_detection(target, &result.detections, &used) else {
                            continue;
                        };
                        used[index] = true;
                        items.push(HumanEnrichmentItem {
                            handle: person.handle,
                            patch: semantic_patch(
                                &result.detections[index],
                                result.landmarks.get(index).copied().flatten(),
                            ),
                        });
                    }
                    self.results.publish(HumanEnrichmentFrame {
                        source_frame_id: work.frame.id,
                        binding: work.binding,
                        items: Arc::new(items),
                        bodies: Arc::new(result.bodies),
                    });
                    self.metrics
                        .record_duration(super::MetricAgent::Human, started.elapsed());
                    ticks += 1;
                }
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
    fn semantic_patch_has_no_spatial_fields_to_copy() {
        let mut detection = Detection::bare(0.1, 0.2, 0.4, 0.8);
        detection.identity = Some("alice".into());
        detection.position_z = Some(1.5);
        detection.mask = Some(mvm_core::InstanceMask::empty());
        let patch = semantic_patch(&detection, None);
        assert_eq!(patch.identity.as_deref(), Some("alice"));
        let names = format!("{patch:?}");
        assert!(!names.contains("position_z"));
        assert!(!names.contains("mask"));
        assert!(!names.contains("bbox"));
    }
}
