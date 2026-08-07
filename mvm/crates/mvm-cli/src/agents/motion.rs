//! Phase 19.1 classical motion-field spike.
//!
//! A sparse block matcher estimates the dominant image-plane displacement of
//! textured background patches. It has no model/runtime dependency and is
//! deliberately conservative: frame gaps, weak texture, ambiguous consensus,
//! or an external-odometry contradiction yield `withheld` rather than a camera
//! pose. The result is camera-local evidence, never world odometry.

use crate::agents::{
    HumanObservationBus, MotionObservationBus, MotionObservationFrame, ObservationStamp,
    SceneObservationBus,
};
use image::DynamicImage;
use mvm_core::{
    BBox,
    canonical::{
        FactEvidenceRef, FactReason, FactSource, FactStatus, ReferenceFrame, RelativeCameraMotion,
        RelativeCameraMotionKind, SceneFact,
    },
};
use mvm_mesh::{Agent, AgentHealth, Frame, MeshContext};

const ANALYSIS_WIDTH: u32 = 160;
const ANALYSIS_HEIGHT: u32 = 120;
const GRID_STEP: usize = 8;
const PATCH_RADIUS: i32 = 2;
const SEARCH_RADIUS: i32 = 5;
const MIN_CANDIDATES: usize = 24;
const MIN_TEXTURE: f32 = 12.0;
const MIN_CONSENSUS: f32 = 0.60;
const MOTION_THRESHOLD_PX: f32 = 0.75;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExternalOdometrySample {
    pub source_frame_id: u64,
    /// Optional camera-model projection supplied by the external owner. It is
    /// only a cross-check and never replaces visual displacement.
    pub expected_image_displacement_px: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MotionEstimatorConfig {
    pub fixed_camera: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct MotionEstimator {
    config: MotionEstimatorConfig,
}

impl MotionEstimator {
    pub(crate) fn new(config: MotionEstimatorConfig) -> Self {
        Self { config }
    }

    pub(crate) fn estimate(
        &self,
        previous: &Frame,
        current: &Frame,
        dynamic_regions: &[BBox],
        odometry: Option<ExternalOdometrySample>,
    ) -> SceneFact<RelativeCameraMotion> {
        let motion_evidence = FactEvidenceRef {
            source: FactSource::Motion,
            source_frame_id: current.id,
            entity: None,
            event_seq: None,
        };
        if current.id != previous.id.saturating_add(1) {
            return SceneFact::withheld(FactReason::FrameGap, vec![motion_evidence]);
        }

        let previous = GrayFrame::from_image(previous.image.as_ref());
        let current_gray = GrayFrame::from_image(current.image.as_ref());
        if previous.width != current_gray.width || previous.height != current_gray.height {
            return SceneFact::withheld(FactReason::FrameGap, vec![motion_evidence]);
        }

        let vectors = match_vectors(&previous, &current_gray, dynamic_regions);
        if vectors.len() < MIN_CANDIDATES {
            return SceneFact::withheld(FactReason::LowTexture, vec![motion_evidence]);
        }
        let mut xs: Vec<i32> = vectors.iter().map(|vector| vector.0).collect();
        let mut ys: Vec<i32> = vectors.iter().map(|vector| vector.1).collect();
        xs.sort_unstable();
        ys.sort_unstable();
        let dx = xs[xs.len() / 2];
        let dy = ys[ys.len() / 2];
        let inlier_count = vectors
            .iter()
            .filter(|(x, y)| (x - dx).abs() <= 1 && (y - dy).abs() <= 1)
            .count();
        let ratio = inlier_count as f32 / vectors.len() as f32;
        if ratio < MIN_CONSENSUS {
            return SceneFact::withheld(FactReason::DynamicSceneDominance, vec![motion_evidence]);
        }

        let displacement = [dx as f32, dy as f32];
        let mut evidence = vec![motion_evidence];
        if let Some(odometry) = odometry {
            evidence.push(FactEvidenceRef {
                source: FactSource::CameraState,
                source_frame_id: odometry.source_frame_id,
                entity: None,
                event_seq: None,
            });
            let disagreement =
                squared_distance(displacement, odometry.expected_image_displacement_px);
            if disagreement > 9.0 {
                return SceneFact::withheld(FactReason::OdometryConflict, evidence);
            }
        }

        let magnitude = displacement[0].hypot(displacement[1]);
        let kind = if magnitude >= MOTION_THRESHOLD_PX {
            RelativeCameraMotionKind::ImageTranslation
        } else if self.config.fixed_camera {
            if !evidence
                .iter()
                .any(|item| item.source == FactSource::CameraState)
            {
                evidence.push(FactEvidenceRef {
                    source: FactSource::CameraState,
                    source_frame_id: current.id,
                    entity: None,
                    event_seq: None,
                });
            }
            RelativeCameraMotionKind::StationaryConfigured
        } else {
            return SceneFact::withheld(FactReason::InsufficientParallax, evidence);
        };
        SceneFact::claimed(
            FactStatus::Derived,
            RelativeCameraMotion {
                kind,
                reference_frame: ReferenceFrame::CameraLocal,
                displacement_px: displacement,
                static_background_ratio: ratio,
                candidate_count: u32::try_from(vectors.len()).unwrap_or(u32::MAX),
                inlier_count: u32::try_from(inlier_count).unwrap_or(u32::MAX),
            },
            ratio,
            evidence,
        )
        .expect("bounded motion estimate satisfies canonical fact invariants")
    }
}

pub(crate) struct MotionAgent {
    estimator: MotionEstimator,
    human_observations: HumanObservationBus,
    scene_observations: SceneObservationBus,
    observations: MotionObservationBus,
    previous: Option<Frame>,
    last_frame: Option<u64>,
    metrics: super::SchedulingMetrics,
}

impl MotionAgent {
    pub(crate) fn new(
        fixed_camera: bool,
        human_observations: HumanObservationBus,
        scene_observations: SceneObservationBus,
        observations: MotionObservationBus,
        metrics: super::SchedulingMetrics,
    ) -> Self {
        Self {
            estimator: MotionEstimator::new(MotionEstimatorConfig { fixed_camera }),
            human_observations,
            scene_observations,
            observations,
            previous: None,
            last_frame: None,
            metrics,
        }
    }

    fn dynamic_regions(&self, frame_id: u64) -> Vec<BBox> {
        let mut regions = Vec::new();
        if let Some(human) = self
            .human_observations
            .latest()
            .filter(|observation| observation.stamp.age_at(frame_id).is_some())
        {
            regions.extend(human.people.iter().map(|person| person.bbox));
        }
        if let Some(scene) = self
            .scene_observations
            .latest()
            .filter(|observation| observation.stamp.age_at(frame_id).is_some())
        {
            regions.extend(scene.animals.iter().map(|animal| animal.bbox));
        }
        regions
    }
}

impl Agent for MotionAgent {
    fn name(&self) -> String {
        "motion".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0_u64;
        while ctx.running() {
            if let Some(current) = ctx.frames.latest()
                && self.last_frame != Some(current.id)
            {
                self.last_frame = Some(current.id);
                if let Some(previous) = self.previous.as_ref() {
                    let started = std::time::Instant::now();
                    let fact = self.estimator.estimate(
                        previous,
                        &current,
                        &self.dynamic_regions(current.id),
                        None,
                    );
                    self.observations.publish(MotionObservationFrame {
                        stamp: ObservationStamp::from_frame(&current),
                        camera_motion: fact,
                    });
                    self.metrics
                        .record_duration(super::MetricAgent::Motion, started.elapsed());
                    ticks = ticks.saturating_add(1);
                }
                self.previous = Some(current);
            }
            std::thread::sleep(std::time::Duration::from_millis(3));
        }
        AgentHealth {
            name: self.name(),
            ticks,
            errors: 0,
        }
    }
}

fn squared_distance(left: [f32; 2], right: [f32; 2]) -> f32 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)
}

#[derive(Debug)]
struct GrayFrame {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

impl GrayFrame {
    fn from_image(image: &DynamicImage) -> Self {
        let gray = image.to_luma8();
        let source_width = gray.width().max(1);
        let source_height = gray.height().max(1);
        let scale = (source_width as f32 / ANALYSIS_WIDTH as f32)
            .max(source_height as f32 / ANALYSIS_HEIGHT as f32)
            .max(1.0);
        let width = ((source_width as f32 / scale).round() as u32).max(1);
        let height = ((source_height as f32 / scale).round() as u32).max(1);
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            let source_y = ((y as f32 + 0.5) * scale).floor() as u32;
            for x in 0..width {
                let source_x = ((x as f32 + 0.5) * scale).floor() as u32;
                pixels.push(
                    gray.get_pixel(
                        source_x.min(source_width - 1),
                        source_y.min(source_height - 1),
                    )[0],
                );
            }
        }
        Self {
            pixels,
            width: width as usize,
            height: height as usize,
        }
    }

    fn at(&self, x: i32, y: i32) -> u8 {
        self.pixels[y as usize * self.width + x as usize]
    }
}

fn match_vectors(previous: &GrayFrame, current: &GrayFrame, masks: &[BBox]) -> Vec<(i32, i32)> {
    let margin = PATCH_RADIUS + SEARCH_RADIUS;
    let mut vectors = Vec::new();
    for y in (margin..current.height as i32 - margin).step_by(GRID_STEP) {
        for x in (margin..current.width as i32 - margin).step_by(GRID_STEP) {
            let normalized = [
                x as f32 / current.width as f32,
                y as f32 / current.height as f32,
            ];
            if masks.iter().any(|mask| point_in_bbox(normalized, mask))
                || patch_texture(previous, x, y) < MIN_TEXTURE
            {
                continue;
            }
            let mut best = (i32::MAX, 0_i32, 0_i32);
            for dy in -SEARCH_RADIUS..=SEARCH_RADIUS {
                for dx in -SEARCH_RADIUS..=SEARCH_RADIUS {
                    let error = patch_error(previous, current, x, y, dx, dy);
                    if error < best.0 {
                        best = (error, dx, dy);
                    }
                }
            }
            vectors.push((best.1, best.2));
        }
    }
    vectors
}

fn point_in_bbox(point: [f32; 2], bbox: &BBox) -> bool {
    point[0] >= bbox.x1 && point[0] <= bbox.x2 && point[1] >= bbox.y1 && point[1] <= bbox.y2
}

fn patch_texture(frame: &GrayFrame, x: i32, y: i32) -> f32 {
    let horizontal = (i16::from(frame.at(x + 1, y)) - i16::from(frame.at(x - 1, y))).abs();
    let vertical = (i16::from(frame.at(x, y + 1)) - i16::from(frame.at(x, y - 1))).abs();
    f32::from(horizontal + vertical)
}

fn patch_error(previous: &GrayFrame, current: &GrayFrame, x: i32, y: i32, dx: i32, dy: i32) -> i32 {
    let mut error = 0_i32;
    for py in -PATCH_RADIUS..=PATCH_RADIUS {
        for px in -PATCH_RADIUS..=PATCH_RADIUS {
            error += (i16::from(previous.at(x + px, y + py))
                - i16::from(current.at(x + px + dx, y + py + dy)))
            .abs() as i32;
        }
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};
    use std::time::Instant;

    fn texture(width: u32, height: u32) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| {
            let value = x
                .wrapping_mul(37)
                .wrapping_add(y.wrapping_mul(73))
                .wrapping_add((x ^ y).wrapping_mul(19));
            Luma([(value % 251) as u8])
        })
    }

    fn shifted(source: &GrayImage, dx: i32, dy: i32) -> GrayImage {
        GrayImage::from_fn(source.width(), source.height(), |x, y| {
            let sx = x as i32 - dx;
            let sy = y as i32 - dy;
            if sx >= 0 && sy >= 0 && sx < source.width() as i32 && sy < source.height() as i32 {
                *source.get_pixel(sx as u32, sy as u32)
            } else {
                Luma([0])
            }
        })
    }

    fn frame(id: u64, image: GrayImage) -> Frame {
        Frame::new_stamped(
            id,
            DynamicImage::ImageLuma8(image),
            Instant::now(),
            id as i64 * 40,
        )
    }

    fn estimator(fixed_camera: bool) -> MotionEstimator {
        MotionEstimator::new(MotionEstimatorConfig { fixed_camera })
    }

    #[test]
    fn static_room_pan_yields_relative_translation() {
        let first = texture(160, 120);
        let fact = estimator(false).estimate(
            &frame(1, first.clone()),
            &frame(2, shifted(&first, 3, -2)),
            &[],
            None,
        );
        let motion = fact.value.expect("pan has a motion hypothesis");
        assert_eq!(motion.kind, RelativeCameraMotionKind::ImageTranslation);
        assert!((motion.displacement_px[0] - 3.0).abs() <= 1.0);
        assert!((motion.displacement_px[1] + 2.0).abs() <= 1.0);
        assert!(motion.static_background_ratio >= MIN_CONSENSUS);
    }

    #[test]
    fn fixed_camera_with_moving_person_uses_explicit_fixed_configuration() {
        let first = texture(160, 120);
        let mut second = first.clone();
        for y in 30..90 {
            for x in 20..50 {
                second.put_pixel(x + 18, y, *first.get_pixel(x, y));
                second.put_pixel(x, y, Luma([0]));
            }
        }
        let actor = BBox {
            x1: 0.1,
            y1: 0.2,
            x2: 0.55,
            y2: 0.8,
        };
        let fact = estimator(true).estimate(&frame(3, first), &frame(4, second), &[actor], None);
        assert_eq!(
            fact.value.unwrap().kind,
            RelativeCameraMotionKind::StationaryConfigured
        );
    }

    #[test]
    fn moving_object_in_fixed_room_does_not_become_ego_motion() {
        let first = texture(160, 120);
        let mut second = first.clone();
        for y in 45..65 {
            for x in 35..55 {
                second.put_pixel(x + 35, y, *first.get_pixel(x, y));
                second.put_pixel(x, y, Luma([0]));
            }
        }
        let fact = estimator(true).estimate(&frame(4, first), &frame(5, second), &[], None);
        assert_eq!(
            fact.value.unwrap().kind,
            RelativeCameraMotionKind::StationaryConfigured
        );
    }

    #[test]
    fn zero_flow_without_fixed_configuration_is_withheld() {
        let first = texture(160, 120);
        let fact = estimator(false).estimate(&frame(4, first.clone()), &frame(5, first), &[], None);
        assert_eq!(fact.status, FactStatus::Withheld);
        assert_eq!(fact.reason, Some(FactReason::InsufficientParallax));
    }

    #[test]
    fn blur_and_frame_drop_are_withheld() {
        let blank = GrayImage::from_pixel(160, 120, Luma([128]));
        let blur = estimator(false).estimate(&frame(5, blank.clone()), &frame(6, blank), &[], None);
        assert_eq!(blur.status, FactStatus::Withheld);
        assert_eq!(blur.reason, Some(FactReason::LowTexture));

        let first = texture(160, 120);
        let dropped =
            estimator(false).estimate(&frame(6, first.clone()), &frame(9, first), &[], None);
        assert_eq!(dropped.reason, Some(FactReason::FrameGap));
    }

    #[test]
    fn dynamic_crowd_does_not_fabricate_ego_motion() {
        let first = texture(160, 120);
        let left = shifted(&first, 4, 0);
        let right = shifted(&first, -4, 0);
        let second = GrayImage::from_fn(160, 120, |x, y| {
            if x < 80 {
                *left.get_pixel(x, y)
            } else {
                *right.get_pixel(x, y)
            }
        });
        let fact = estimator(false).estimate(&frame(9, first), &frame(10, second), &[], None);
        assert_eq!(fact.status, FactStatus::Withheld);
        assert_eq!(fact.reason, Some(FactReason::DynamicSceneDominance));
    }

    #[test]
    fn external_odometry_cross_checks_without_overriding_visual_evidence() {
        let first = texture(160, 120);
        let fact = estimator(false).estimate(
            &frame(10, first.clone()),
            &frame(11, shifted(&first, 3, 0)),
            &[],
            Some(ExternalOdometrySample {
                source_frame_id: 11,
                expected_image_displacement_px: [-3.0, 0.0],
            }),
        );
        assert_eq!(fact.status, FactStatus::Withheld);
        assert_eq!(fact.reason, Some(FactReason::OdometryConflict));
        assert_eq!(fact.evidence.len(), 2);
    }

    #[test]
    fn spike_decision_gate_measures_epe_consensus_and_cpu() {
        let first = texture(160, 120);
        let previous = frame(20, first.clone());
        let current = frame(21, shifted(&first, 3, -2));
        let mut elapsed_us = Vec::new();
        let mut last = None;
        for _ in 0..31 {
            let started = Instant::now();
            last = Some(estimator(false).estimate(&previous, &current, &[], None));
            elapsed_us.push(started.elapsed().as_micros());
        }
        elapsed_us.remove(0); // warm-up
        elapsed_us.sort_unstable();
        let p50_us = elapsed_us[elapsed_us.len() / 2];
        let p95_us = elapsed_us[elapsed_us.len() * 95 / 100];
        let motion = last.unwrap().value.unwrap();
        let epe = (motion.displacement_px[0] - 3.0).hypot(motion.displacement_px[1] + 2.0);
        eprintln!(
            "motion-spike epe_px={epe:.3} consensus={:.3} cpu_p50_us={p50_us} cpu_p95_us={p95_us}",
            motion.static_background_ratio
        );
        assert!(epe <= 1.0);
        assert!(motion.static_background_ratio >= MIN_CONSENSUS);
        assert!(
            p95_us <= 50_000,
            "classical flow must remain under the 50ms CPU budget"
        );
    }
}
