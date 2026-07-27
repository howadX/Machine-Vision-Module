use mvm_core::canonical::{
    EventTailView, FactEvidenceRef, FactReason, FactSource, FactStatus, FrameStamp,
    FusedSceneFrame, IncarnationRegistry, SceneEvent, SceneEventKind, SceneFact, SceneQuality,
    SourceFreshness, SourceStamp, TrackDomain,
};
use std::collections::BTreeSet;
use std::path::PathBuf;

#[test]
fn canonical_scene_contract_withheld_and_unknown_are_not_assertions() {
    let withheld = SceneFact::<String>::withheld(FactReason::InsufficientCoverage, Vec::new());
    let unknown = SceneFact::<String>::unknown(FactReason::MissingSource);

    for fact in [withheld, unknown] {
        assert!(!fact.status.is_asserted());
        assert!(!fact.status.can_be_negative_class());
        assert!(fact.value.is_none());
        assert!(fact.confidence.is_none());
        fact.validate().unwrap();
    }
}

#[test]
fn canonical_scene_contract_source_freshness_is_frame_based() {
    let fresh = SourceStamp::at_frame(FactSource::Human, 98, 1_000)
        .relative_to(100, 2, 4)
        .unwrap();
    let coasted = SourceStamp::at_frame(FactSource::Scene, 97, 900)
        .relative_to(100, 2, 4)
        .unwrap();
    let stale = SourceStamp::at_frame(FactSource::Depth, 95, 800)
        .relative_to(100, 2, 4)
        .unwrap();

    assert_eq!(fresh.age_frames, 2);
    assert_eq!(fresh.freshness, SourceFreshness::Fresh);
    assert_eq!(coasted.freshness, SourceFreshness::Coasted);
    assert_eq!(stale.freshness, SourceFreshness::Stale);
    assert!(
        SourceStamp::at_frame(FactSource::Human, 101, 1_100)
            .relative_to(100, 2, 4)
            .is_err()
    );
}

#[test]
fn canonical_scene_contract_reused_local_id_gets_new_incarnation() {
    let mut registry = IncarnationRegistry::new("run-a");
    let first = registry.activate(TrackDomain::Person, 10);
    assert_eq!(first.incarnation, 0);
    assert_eq!(registry.activate(TrackDomain::Person, 10), first);

    assert!(registry.retire(&first));
    let reused = registry.activate(TrackDomain::Person, 10);
    assert_eq!(reused.track_id, first.track_id);
    assert_eq!(reused.incarnation, 1);
    assert_ne!(reused, first);
}

#[test]
fn canonical_scene_contract_frame_stamp_round_trips_as_stable_json() {
    let stamp = FrameStamp::new("run-a", 42, 1_720_000_000_000, 1_720_000_000_012).unwrap();
    let json = serde_json::to_string(&stamp).unwrap();
    let decoded: FrameStamp = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded, stamp);
    assert!(json.contains("\"captured_at_unix_ms\":1720000000000"));
    assert!(json.contains("\"published_at_unix_ms\":1720000000012"));
}

#[test]
fn canonical_scene_contract_asserted_fact_requires_value_and_confidence() {
    let invalid = SceneFact::<String> {
        status: FactStatus::Observed,
        value: None,
        confidence: None,
        evidence: Vec::new(),
        reason: None,
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn canonical_scene_contract_event_cursor_reports_explicit_gap() {
    let evidence = FactEvidenceRef {
        source: FactSource::Fusion,
        source_frame_id: 20,
        entity: None,
        event_seq: Some(10),
    };
    let event = SceneEvent {
        event_seq: 10,
        source_frame_id: 20,
        source_timestamp_unix_ms: 2_000,
        kind: SceneEventKind::EntityAppeared,
        entity_keys: Vec::new(),
        fact: SceneFact::claimed(FactStatus::Temporal, true, 1.0, vec![evidence]).unwrap(),
        payload: None,
    };
    let tail = EventTailView {
        first_seq: 10,
        next_seq: 11,
        events: vec![event],
    };

    let delta = tail.delta_after(Some(4));
    assert_eq!(delta.gap.unwrap().available_from, 10);
    assert_eq!(delta.events.len(), 1);
    assert_eq!(delta.next_cursor, Some(10));

    let replay = tail.delta_after(Some(10));
    assert!(replay.gap.is_none());
    assert!(replay.events.is_empty());
    assert_eq!(replay.next_cursor, Some(10));

    let other_consumer = tail.delta_after(Some(4));
    assert_eq!(
        other_consumer
            .events
            .iter()
            .map(|event| event.event_seq)
            .collect::<Vec<_>>(),
        vec![10],
        "consumer cursors are independent and non-destructive"
    );
}

#[test]
fn canonical_scene_contract_benchmark_baseline_has_required_scenarios() {
    let benchmark = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmark");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(benchmark.join("scenarios/manifest.json")).unwrap())
            .unwrap();
    let ids: BTreeSet<&str> = manifest["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .map(|scenario| scenario["id"].as_str().unwrap())
        .collect();

    for required in ["001", "004", "007", "009", "011", "012"] {
        assert!(
            ids.contains(required),
            "missing benchmark scenario {required}"
        );
        assert!(
            benchmark
                .join(format!("scenarios/{required}.json"))
                .is_file(),
            "missing replay fixture for scenario {required}"
        );
    }
    assert!(benchmark.join("questions/temporal-v1.json").is_file());
    assert!(benchmark.join("baselines/v0.3.0.md").is_file());
}

#[test]
fn canonical_scene_contract_fused_frame_round_trip_is_deterministic() {
    let mut frame = FusedSceneFrame {
        stamp: FrameStamp::new("run-a", 42, 1_000, 1_010).unwrap(),
        sources: vec![
            SourceStamp::at_frame(FactSource::Scene, 41, 999)
                .relative_to(42, 2, 4)
                .unwrap(),
            SourceStamp::at_frame(FactSource::Human, 42, 1_000)
                .relative_to(42, 2, 4)
                .unwrap(),
        ],
        entities: Vec::new(),
        relations: Vec::new(),
        zones: Vec::new(),
        activities: Vec::new(),
        events: EventTailView::default(),
        anomaly: SceneFact::unknown(FactReason::MissingSource),
        quality: SceneQuality::default(),
    };
    frame.canonicalize();
    frame.validate().unwrap();

    assert_eq!(frame.sources[0].source, FactSource::Human);
    let first = serde_json::to_string(&frame).unwrap();
    let decoded: FusedSceneFrame = serde_json::from_str(&first).unwrap();
    let second = serde_json::to_string(&decoded).unwrap();
    assert_eq!(first, second);

    let projection = decoded.compatibility_world_state(std::time::Instant::now());
    assert_eq!(projection.frame_id, 42);
    assert!(projection.current_session_id.is_none());
    assert!(projection.people.is_empty());
}

#[test]
fn canonical_scene_contract_v0_3_0_synthetic_baseline_is_executable() {
    let scenarios = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/scenarios");
    let mut scores = std::collections::BTreeMap::<String, (u64, u64)>::new();

    for id in ["001", "004", "007", "009", "011", "012"] {
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(scenarios.join(format!("{id}.json"))).unwrap())
                .unwrap();
        let baseline = fixture["baseline_v0_3_0"].as_object().unwrap();
        for (metric, value) in baseline {
            if value.is_null() {
                continue;
            }
            let entry = scores.entry(metric.clone()).or_default();
            entry.0 += value["correct"].as_u64().unwrap();
            entry.1 += value["total"].as_u64().unwrap();
        }
    }

    assert_eq!(scores["temporal_continuity"], (2, 2));
    assert_eq!(scores["scene_event_recall"], (0, 4));
    assert_eq!(scores["uncertainty_honesty"], (2, 2));
    assert_eq!(scores["last_observation_forecast"], (1, 2));
}
