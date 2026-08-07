use crate::{
    CalibrationBasis, CalibrationReport, CalibrationValue, CameraPose, CognitiveClaim,
    CoordinateFrame, EvidenceRef, ReasonCode, SpatialSelf, VisibilityState, VisualSelfModel,
    ZoneVisibility,
};
use mvm_core::Config;
use mvm_core::canonical::{Existence, FactStatus, FusedSceneFrame};

pub(crate) struct SelfModelLayer;

impl SelfModelLayer {
    pub(crate) fn build(&self, scene: &FusedSceneFrame, config: &Config) -> VisualSelfModel {
        let frame_evidence = vec![EvidenceRef {
            scene_frame_id: scene.stamp.frame_id,
            entity: None,
            event_seq: None,
        }];
        let camera_relative_pose = CognitiveClaim::asserted(
            CameraPose {
                reference_frame: CoordinateFrame::CameraRelative,
                position_m: [0.0; 3],
                orientation_rpy_degrees: [0.0; 3],
            },
            1.0,
            frame_evidence,
        )
        .expect("camera-relative origin is a definitional coordinate fact");

        VisualSelfModel {
            source_scene_frame_id: scene.stamp.frame_id,
            spatial: SpatialSelf::default(),
            camera_relative_pose,
            world_pose: CognitiveClaim::unknown(ReasonCode::MissingSource),
            zones: scene
                .zones
                .iter()
                .map(|zone| zone_visibility(scene, zone))
                .collect(),
            calibration: CalibrationReport {
                fx_ratio: calibration_value(config.fx_ratio, CalibrationBasis::ConfiguredPrior),
                torso_length_m: calibration_value(
                    config.torso_length_m,
                    CalibrationBasis::UnverifiedPrior,
                ),
                camera_height_m: calibration_value(
                    config.camera_height_m,
                    CalibrationBasis::UnverifiedPrior,
                ),
                world_extrinsics: CalibrationBasis::Unavailable,
            },
        }
    }
}

fn calibration_value(value: f32, basis: CalibrationBasis) -> CalibrationValue {
    if value.is_finite() {
        CalibrationValue {
            value: Some(value),
            basis,
        }
    } else {
        CalibrationValue {
            value: None,
            basis: CalibrationBasis::Unavailable,
        }
    }
}

fn zone_visibility(
    scene: &FusedSceneFrame,
    zone: &mvm_core::canonical::SceneZone,
) -> ZoneVisibility {
    let mut evidence = Vec::new();
    let mut confidence = 1.0_f32;
    let observed = zone
        .members
        .iter()
        .filter(|key| {
            let Some(entity) = scene.entities.iter().find(|entity| &entity.key == *key) else {
                return false;
            };
            let usable = matches!(
                entity.bbox.status,
                FactStatus::Observed | FactStatus::Derived | FactStatus::Temporal
            ) && matches!(entity.existence.value, Some(Existence::Present));
            if usable {
                confidence = confidence.min(entity.bbox.confidence.unwrap_or(0.0));
                evidence.push(EvidenceRef {
                    scene_frame_id: scene.stamp.frame_id,
                    entity: Some(entity.key.clone()),
                    event_seq: None,
                });
            }
            usable
        })
        .count();

    let visibility = if zone.members.is_empty() || observed == 0 {
        CognitiveClaim::unknown(ReasonCode::MissingSource)
    } else {
        let value = if observed == zone.members.len() {
            VisibilityState::ObservedMembers
        } else {
            VisibilityState::PartiallyObserved
        };
        CognitiveClaim::asserted(value, confidence, evidence)
            .expect("zone member evidence is non-empty and bounded")
    };

    ZoneVisibility {
        zone_id: zone.id,
        label: zone.label.clone(),
        visibility,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CalibrationBasis, ClaimStatus, CoordinateFrame};
    use mvm_core::canonical::{
        EventTailView, FactReason, FrameStamp, FusedSceneFrame, SceneQuality, SceneZone,
    };

    fn scene() -> FusedSceneFrame {
        FusedSceneFrame {
            stamp: FrameStamp::new("self-model-test", 7, 1_000, 1_001).unwrap(),
            sources: Vec::new(),
            entities: Vec::new(),
            relations: Vec::new(),
            hand_interactions: Vec::new(),
            zones: vec![SceneZone {
                id: 3,
                label: Some("desk".into()),
                members: Vec::new(),
                age_frames: 4,
            }],
            activities: Vec::new(),
            spatial: Default::default(),
            events: EventTailView::default(),
            anomaly: mvm_core::canonical::SceneFact::unknown(FactReason::MissingSource),
            quality: SceneQuality::default(),
        }
    }

    #[test]
    fn self_model_distinguishes_camera_coordinates_from_unknown_world_pose() {
        let model = SelfModelLayer.build(&scene(), &mvm_core::Config::default());

        assert_eq!(
            model.spatial.coordinate_frame,
            CoordinateFrame::CameraRelative
        );
        assert_eq!(model.camera_relative_pose.status, ClaimStatus::Asserted);
        assert_eq!(
            model
                .camera_relative_pose
                .value
                .as_ref()
                .unwrap()
                .position_m,
            [0.0, 0.0, 0.0]
        );
        assert_eq!(model.world_pose.status, ClaimStatus::Unknown);
        assert_eq!(model.zones[0].visibility.status, ClaimStatus::Unknown);
        assert!(model.zones[0].visibility.value.is_none());
        assert!(model.zones[0].visibility.evidence.is_empty());
    }

    #[test]
    fn calibration_report_labels_runtime_values_as_priors() {
        let model = SelfModelLayer.build(&scene(), &mvm_core::Config::default());

        assert_eq!(
            model.calibration.fx_ratio.basis,
            CalibrationBasis::ConfiguredPrior
        );
        assert_eq!(
            model.calibration.torso_length_m.basis,
            CalibrationBasis::UnverifiedPrior
        );
        assert_eq!(
            model.calibration.camera_height_m.basis,
            CalibrationBasis::UnverifiedPrior
        );
        assert_eq!(
            model.calibration.world_extrinsics,
            CalibrationBasis::Unavailable
        );
    }
}
