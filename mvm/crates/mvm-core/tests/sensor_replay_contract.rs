use mvm_core::{DepthStatus, SensorReplayEnvelope};
use std::collections::BTreeSet;
use std::path::PathBuf;

const MAX_PAIR_SKEW_NS: u64 = 20_000_000;

#[test]
fn rgbd_contract_catalog_covers_required_honesty_cases() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/rgbd");
    let contract: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("contract.json")).expect("RGB-D replay contract exists"),
    )
    .unwrap();
    assert_eq!(contract["schema"], "mvm-rgbd-replay/1.0");
    assert_eq!(contract["max_pair_skew_ns"], MAX_PAIR_SKEW_NS);

    let ids: BTreeSet<String> = std::fs::read_dir(root.join("fixtures"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .map(|path| {
            serde_json::from_slice::<SensorReplayEnvelope>(&std::fs::read(path).unwrap())
                .unwrap()
                .case_id
        })
        .collect();

    for required in [
        "paired_registered",
        "rgb_only",
        "missing_depth",
        "stale_depth",
        "skew_over_limit",
        "mismatched_dimensions",
        "invalid_calibration_revision",
        "calibration_serial_mismatch",
        "unregistered_depth",
        "depth_holes",
        "packet_gap",
        "timestamp_wraparound",
    ] {
        assert!(ids.contains(required), "missing RGB-D fixture {required}");
    }
}

#[test]
fn only_valid_registered_fixture_gets_metric_rgb_authority() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/rgbd/fixtures");
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let parsed = serde_json::from_str::<SensorReplayEnvelope>(&text);
        let expected = matches!(
            path.file_stem().unwrap().to_str(),
            Some("paired_registered" | "depth_holes")
        );
        match parsed {
            Ok(envelope) => {
                assert_eq!(
                    envelope.admits_metric_rgb_association(MAX_PAIR_SKEW_NS),
                    expected,
                    "fixture {} admitted incorrectly",
                    envelope.case_id
                );
                if envelope.case_id == "rgb_only" {
                    assert_eq!(envelope.depth_status, DepthStatus::Unavailable);
                    envelope.validate(MAX_PAIR_SKEW_NS).unwrap();
                }
                if envelope.case_id == "depth_holes" {
                    let depth = envelope.metric_depth.as_ref().unwrap();
                    assert_eq!(depth.invalid_value, 0);
                    assert!(depth.depth_mm.contains(&0));
                    assert!(depth.valid_ratio < 1.0);
                }
            }
            Err(_) => assert!(
                !expected,
                "valid paired fixture must always deserialize: {}",
                path.display()
            ),
        }
    }
}

#[test]
fn replay_serialization_is_deterministic() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/rgbd/fixtures/paired_registered.json");
    let envelope: SensorReplayEnvelope =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    envelope.validate(MAX_PAIR_SKEW_NS).unwrap();
    let first = serde_json::to_string(&envelope).unwrap();
    let decoded: SensorReplayEnvelope = serde_json::from_str(&first).unwrap();
    assert_eq!(serde_json::to_string(&decoded).unwrap(), first);
}
