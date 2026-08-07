use mvm_core::canonical::{
    CameraIntrinsics, CameraToBodyExtrinsics, CellOccupancyFact, EntityKey, FactEvidenceRef,
    FactReason, FactSource, FactStatus, FrameId, GraphDelta, GraphEdge, GraphEdgeId, GraphEdgeKind,
    GraphNode, GraphNodeId, GraphOperation, GraphProjection, ObservationId, OccupancyCell,
    OccupancyState, ReferenceFrame, RelativeCameraMotion, RelativeCameraMotionKind, SceneFact,
    SourceStamp, SpatialCalibration, SpatialRelationFact, SpatialRelationKind, SpatialScene,
    SpatialShardId, SurfaceFact, SurfaceGeometry, SurfaceId, SurfaceKind, TrackDomain,
    VisibilityFact, VisibilityState,
};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn evidence(source: FactSource, frame_id: u64) -> FactEvidenceRef {
    FactEvidenceRef {
        source,
        source_frame_id: frame_id,
        entity: None,
        event_seq: None,
    }
}

#[test]
fn stationary_camera_claim_requires_fresh_motion_and_camera_state_evidence() {
    let frame_id = 3;
    let camera_motion = SceneFact::claimed(
        FactStatus::Derived,
        RelativeCameraMotion {
            kind: RelativeCameraMotionKind::StationaryConfigured,
            reference_frame: ReferenceFrame::CameraLocal,
            displacement_px: [0.0, 0.0],
            static_background_ratio: 0.9,
            candidate_count: 100,
            inlier_count: 90,
        },
        0.9,
        vec![evidence(FactSource::Motion, frame_id)],
    )
    .unwrap();
    let mut scene = SpatialScene {
        camera_motion,
        ..Default::default()
    };
    let motion = SourceStamp::at_frame(FactSource::Motion, frame_id, 1_000);
    assert!(
        scene
            .validate("run", std::slice::from_ref(&motion))
            .is_err()
    );

    scene
        .camera_motion
        .evidence
        .push(evidence(FactSource::CameraState, frame_id));
    let camera_state = SourceStamp::at_frame(FactSource::CameraState, frame_id, 1_000);
    scene.validate("run", &[motion, camera_state]).unwrap();
}

fn claimed<T>(source: FactSource, frame_id: u64, value: T) -> SceneFact<T> {
    SceneFact::claimed(
        FactStatus::Observed,
        value,
        0.9,
        vec![evidence(source, frame_id)],
    )
    .unwrap()
}

fn calibration(frame_id: u64) -> SpatialCalibration {
    SpatialCalibration {
        intrinsics: claimed(
            FactSource::Capture,
            frame_id,
            CameraIntrinsics {
                fx: 640.0,
                fy: 640.0,
                cx: 320.0,
                cy: 240.0,
                image_width: 640,
                image_height: 480,
                version: "fixture-v1".into(),
            },
        ),
        camera_to_body: claimed(
            FactSource::Capture,
            frame_id,
            CameraToBodyExtrinsics {
                translation_m: [0.0, 0.0, 1.0],
                rotation_rpy_degrees: [0.0; 3],
                version: "fixture-v1".into(),
            },
        ),
    }
}

fn free_cell(run_id: &str, frame_id: u64, reference_frame: ReferenceFrame) -> CellOccupancyFact {
    CellOccupancyFact {
        cell: OccupancyCell {
            shard: SpatialShardId::new(run_id, 1).unwrap(),
            x: 2,
            y: 3,
        },
        reference_frame,
        occupancy: claimed(FactSource::Spatial, frame_id, OccupancyState::Free),
        visibility: claimed(FactSource::Semantic, frame_id, VisibilityState::Visible),
        clearance_m: SceneFact::claimed(
            FactStatus::Derived,
            0.8,
            0.85,
            vec![
                evidence(FactSource::Depth, frame_id),
                evidence(FactSource::Semantic, frame_id),
            ],
        )
        .unwrap(),
    }
}

#[test]
fn unknown_occupancy_cannot_carry_synthetic_clearance_or_serialize_as_free() {
    let frame_id = 7;
    let mut cell = free_cell("run", frame_id, ReferenceFrame::CameraLocal);
    cell.occupancy = SceneFact::unknown(FactReason::Occluded);

    assert!(cell.validate(&SpatialCalibration::default(), &[]).is_err());
    cell.clearance_m = SceneFact::unknown(FactReason::Occluded);
    cell.visibility = SceneFact::withheld(
        FactReason::Occluded,
        vec![evidence(FactSource::Semantic, frame_id)],
    );
    cell.validate(&SpatialCalibration::default(), &[]).unwrap();

    let json = serde_json::to_value(&cell).unwrap();
    assert_eq!(json["occupancy"]["status"], "unknown");
    assert!(json["occupancy"].get("value").is_none());
    assert_ne!(
        json["occupancy"].get("value"),
        Some(&serde_json::json!("free"))
    );
}

#[test]
fn free_cell_requires_positive_visible_evidence() {
    let frame_id = 8;
    let sources = [
        SourceStamp::at_frame(FactSource::Spatial, frame_id, 1_001),
        SourceStamp::at_frame(FactSource::Semantic, frame_id, 1_001),
        SourceStamp::at_frame(FactSource::Depth, frame_id, 1_001),
    ];
    let mut cell = free_cell("run", frame_id, ReferenceFrame::CameraLocal);
    cell.validate(&SpatialCalibration::default(), &sources)
        .unwrap();

    cell.visibility = SceneFact::unknown(FactReason::Occluded);
    assert!(
        cell.validate(&SpatialCalibration::default(), &sources)
            .is_err()
    );
}

#[test]
fn agent_local_metric_claim_requires_intrinsics_and_extrinsics() {
    let frame_id = 9;
    let sources = [
        SourceStamp::at_frame(FactSource::Capture, frame_id, 1_001),
        SourceStamp::at_frame(FactSource::Spatial, frame_id, 1_001),
        SourceStamp::at_frame(FactSource::Semantic, frame_id, 1_001),
        SourceStamp::at_frame(FactSource::Depth, frame_id, 1_001),
    ];
    let cell = free_cell("run", frame_id, ReferenceFrame::AgentLocal);

    assert!(
        cell.validate(&SpatialCalibration::default(), &sources)
            .is_err()
    );
    cell.validate(&calibration(frame_id), &sources).unwrap();
}

#[test]
fn stale_source_cannot_create_fresh_observed_spatial_relation() {
    let frame_id = 10;
    let subject = GraphNodeId::Entity(EntityKey {
        run_id: "run".into(),
        track_domain: TrackDomain::Object,
        track_id: 10_001,
        incarnation: 0,
    });
    let surface = GraphNodeId::Surface(SurfaceId::new("run", 4).unwrap());
    let relation = SpatialRelationFact {
        subject,
        object: surface,
        kind: SpatialRelationKind::On,
        reference_frame: ReferenceFrame::CameraLocal,
        fact: claimed(FactSource::Depth, frame_id - 5, true),
    };
    let stale = SourceStamp::at_frame(FactSource::Depth, frame_id - 5, 900)
        .relative_to(frame_id, 1, 2)
        .unwrap();

    assert!(
        relation
            .validate("run", &SpatialCalibration::default(), &[stale])
            .is_err()
    );
}

#[test]
fn spatial_scene_canonical_order_and_round_trip_are_deterministic() {
    let frame_id = 11;
    let mut scene = SpatialScene {
        calibration: calibration(frame_id),
        camera_motion: SceneFact::unknown(FactReason::MissingSource),
        surfaces: vec![
            SurfaceFact {
                id: SurfaceId::new("run", 2).unwrap(),
                reference_frame: ReferenceFrame::CameraLocal,
                kind: claimed(FactSource::Semantic, frame_id, SurfaceKind::Wall),
                geometry: SceneFact::unknown(FactReason::Uncalibrated),
            },
            SurfaceFact {
                id: SurfaceId::new("run", 1).unwrap(),
                reference_frame: ReferenceFrame::CameraLocal,
                kind: claimed(FactSource::Semantic, frame_id, SurfaceKind::Floor),
                geometry: SceneFact::claimed(
                    FactStatus::Derived,
                    SurfaceGeometry {
                        boundary_normalized: vec![[0.0, 0.5], [1.0, 0.5], [1.0, 1.0]],
                        plane: Some([0.0, 1.0, 0.0, -1.0]),
                    },
                    0.8,
                    vec![evidence(FactSource::Depth, frame_id)],
                )
                .unwrap(),
            },
        ],
        visibility: vec![VisibilityFact {
            target: GraphNodeId::Surface(SurfaceId::new("run", 1).unwrap()),
            reference_frame: ReferenceFrame::CameraLocal,
            state: claimed(FactSource::Semantic, frame_id, VisibilityState::Visible),
            occluders: Vec::new(),
        }],
        occupancy: vec![free_cell("run", frame_id, ReferenceFrame::CameraLocal)],
        relations: Vec::new(),
        quality: Default::default(),
    };
    let sources = vec![
        SourceStamp::at_frame(FactSource::Capture, frame_id, 1_001),
        SourceStamp::at_frame(FactSource::Depth, frame_id, 1_001),
        SourceStamp::at_frame(FactSource::Semantic, frame_id, 1_001),
        SourceStamp::at_frame(FactSource::Spatial, frame_id, 1_001),
    ];
    scene.canonicalize();
    scene.validate("run", &sources).unwrap();
    assert_eq!(scene.surfaces[0].id.local_id, 1);

    let first = serde_json::to_string(&scene).unwrap();
    let mut decoded: SpatialScene = serde_json::from_str(&first).unwrap();
    decoded.canonicalize();
    assert_eq!(first, serde_json::to_string(&decoded).unwrap());
}

fn graph_delta(run_id: &str, from_revision: u64, local_id: u64) -> GraphDelta {
    let surface = GraphNodeId::Surface(SurfaceId::new(run_id, local_id).unwrap());
    let edge_id = GraphEdgeId::new(run_id, local_id).unwrap();
    let observation = ObservationId::new(run_id, FactSource::Semantic, local_id, 0).unwrap();
    let mut delta = GraphDelta {
        run_id: run_id.into(),
        from_revision,
        to_revision: from_revision + 1,
        source_frame: FrameId::new(run_id, local_id).unwrap(),
        operations: vec![
            GraphOperation::ObservationLinked {
                node: surface.clone(),
                observation,
                evidence: vec![evidence(FactSource::Semantic, local_id)],
            },
            GraphOperation::EdgeAdded {
                edge: GraphEdge {
                    id: edge_id,
                    from: GraphNodeId::SpatialShard(SpatialShardId::new(run_id, 1).unwrap()),
                    to: surface.clone(),
                    kind: GraphEdgeKind::Contains,
                },
            },
            GraphOperation::NodeAdded {
                node: GraphNode { id: surface },
            },
        ],
    };
    delta.canonicalize();
    delta
}

#[test]
fn graph_projection_retirement_never_deletes_source_evidence() {
    let mut projection = GraphProjection::new("run");
    let first = graph_delta("run", 0, 1);
    projection.apply(&first).unwrap();
    let observation_count = projection.observations().count();
    let node = GraphNodeId::Surface(SurfaceId::new("run", 1).unwrap());

    let retirement = GraphDelta {
        run_id: "run".into(),
        from_revision: 1,
        to_revision: 2,
        source_frame: FrameId::new("run", 2).unwrap(),
        operations: vec![GraphOperation::NodeRetired { node }],
    };
    projection.apply(&retirement).unwrap();

    assert_eq!(projection.observations().count(), observation_count);
    assert!(projection.retired_nodes().count() == 1);
}

#[test]
fn graph_delta_replay_is_deterministic_for_incremental_or_batched_sinks() {
    let first = graph_delta("run", 0, 1);
    let second = graph_delta("run", 1, 2);

    let mut incremental = GraphProjection::new("run");
    incremental.apply(&first).unwrap();
    incremental.apply(&second).unwrap();

    let batched = GraphDelta::compose(&[first, second]).unwrap();
    let mut batch_sink = GraphProjection::new("run");
    batch_sink.apply(&batched).unwrap();

    assert_eq!(incremental, batch_sink);
    assert_eq!(
        serde_json::to_string(&incremental).unwrap(),
        serde_json::to_string(&batch_sink).unwrap()
    );
}

#[test]
fn graph_delta_is_capacity_bounded_and_rejects_revision_gaps() {
    let mut delta = graph_delta("run", 0, 1);
    delta.operations = (0..=GraphDelta::MAX_OPERATIONS)
        .map(|id| GraphOperation::NodeAdded {
            node: GraphNode {
                id: GraphNodeId::Surface(SurfaceId::new("run", id as u64 + 10).unwrap()),
            },
        })
        .collect();
    assert!(delta.validate().is_err());

    let mut projection = GraphProjection::new("run");
    let mut gap = graph_delta("run", 3, 1);
    gap.from_revision = 3;
    gap.to_revision = 4;
    assert!(projection.apply(&gap).is_err());
}

#[test]
fn visual_spatial_benchmark_has_the_required_replay_contract() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/visual-spatial");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let scenarios = manifest["scenarios"].as_array().unwrap();
    let ids: BTreeSet<&str> = scenarios
        .iter()
        .map(|scenario| scenario["id"].as_str().unwrap())
        .collect();

    for required in [
        "vs001", "vs002", "vs003", "vs004", "vs005", "vs006", "vs007", "vs008", "vs009", "vs010",
    ] {
        assert!(
            ids.contains(required),
            "missing spatial scenario {required}"
        );
        let fixture: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("scenarios/{required}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(fixture["scenario_id"], required);
        assert_eq!(fixture["status"], "synthetic_replay");
        assert_eq!(fixture["measurement_scope"], "contract_only");
        assert!(
            fixture["honesty_assertions"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
        );
    }

    assert!(root.join("schemas/spatial-replay-v1.json").is_file());
    assert!(root.join("evaluate.py").is_file());
    assert!(root.join("baseline-19.0.json").is_file());
}
