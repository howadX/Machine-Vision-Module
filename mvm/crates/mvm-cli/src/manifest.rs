//! Model manifest (phase 14.1) — the single source of truth for the ONNX
//! weights `mvm` needs: filename, download URL, SHA-256, size, and whether the
//! human chain requires it.
//!
//! `mvm setup` fetches from here; `mvm doctor` and the friendly "run `mvm
//! setup`" pre-flight both derive their model lists from this table so they
//! can't drift.
//!
//! Every URL + checksum here was verified against the on-disk models: each
//! canonical URL resolves and its bytes hash to the recorded SHA-256 (checked
//! 2026-07-06). Two models (the ultralytics-exported YOLO detector and
//! segmenter, AGPL) have no upstream canonical URL — they are project-hosted
//! on the immutable GitHub Release `models-v1` (14.6; new weights would go to
//! `models-v2`, tags are never rewritten). Both are optional (scene/masks
//! degrade gracefully), so the core chain sets up even without them.
//! `url: None` stays a valid state for future locally-provided weights.

/// One model's provenance and integrity data.
pub struct ModelSpec {
    /// Filename inside the models dir.
    pub file: &'static str,
    /// Canonical download URL, or `None` for project-hosted weights with no
    /// public source yet (must be provided locally / awaited on GH Releases).
    pub url: Option<&'static str>,
    /// Lowercase hex SHA-256 of the exact file `url` serves.
    pub sha256: &'static str,
    /// Expected size in bytes (a cheap pre-hash sanity check).
    pub size: u64,
    /// True if the human chain cannot start without it (`mvm run` pre-flight).
    pub required: bool,
    /// Provenance / license note (shown by `mvm setup`).
    pub note: &'static str,
}

/// All models, verified 2026-07-06. Order = setup/report order (required first).
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        file: "face_detection_yunet_2023mar.onnx",
        url: Some("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
        sha256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size: 232589,
        required: true,
        note: "YuNet face detector — OpenCV Zoo, MIT",
    },
    ModelSpec {
        file: "face_recognition_sface_2021dec.onnx",
        url: Some("https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
        sha256: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        size: 38696353,
        required: true,
        note: "SFace identity embedding — OpenCV Zoo, Apache-2.0",
    },
    ModelSpec {
        file: "enet_b0_8_best_vgaf.onnx",
        url: Some("https://github.com/av-savchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"),
        sha256: "fa07e841fd06c7a67ee651ea4e6e4a3a2bb5695f47b37a7da50492526f59c898",
        size: 16039595,
        required: true,
        note: "EmotiEffLib B0 emotion (VGAF) — av-savchenko/face-emotion-recognition, Apache-2.0",
    },
    ModelSpec {
        file: "person_detection_mediapipe_2023mar.onnx",
        url: Some("https://github.com/opencv/opencv_zoo/raw/main/models/person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx"),
        sha256: "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f",
        size: 11990159,
        required: true,
        note: "BlazePose person detector — OpenCV Zoo, Apache-2.0",
    },
    ModelSpec {
        file: "pose_estimation_mediapipe_2023mar.onnx",
        url: Some("https://github.com/opencv/opencv_zoo/raw/main/models/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx"),
        sha256: "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f",
        size: 5557238,
        required: true,
        note: "BlazePose pose estimator — OpenCV Zoo, Apache-2.0",
    },
    ModelSpec {
        file: "depth_anything_v2_small.onnx",
        url: Some("https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx"),
        sha256: "fcf51f1b230362b28690bb9d1809bf0431f29cad20534e3f589bd7285547f20d",
        size: 27258801,
        required: false,
        note: "Depth-Anything-V2-Small (quantized) — onnx-community, Apache-2.0",
    },
    ModelSpec {
        file: "handpose_estimation_mediapipe_2023feb.onnx",
        url: Some("https://github.com/opencv/opencv_zoo/raw/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx"),
        sha256: "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
        size: 4099621,
        required: false,
        note: "MediaPipe handpose — OpenCV Zoo, Apache-2.0",
    },
    ModelSpec {
        file: "yolo_object_detect.onnx",
        url: Some("https://github.com/howadX/Machine-Vision-Module/releases/download/models-v1/yolo_object_detect.onnx"),
        sha256: "3432b3be1961f28a741ba48c2c161960bf8236fcd3dfc4f7b53a54a50ee58a6b",
        size: 9941992,
        required: false,
        note: "YOLO26n object detector — ultralytics export, AGPL-3.0 (project-hosted, GitHub Release models-v1)",
    },
    ModelSpec {
        file: "yolo_seg.onnx",
        url: Some("https://github.com/howadX/Machine-Vision-Module/releases/download/models-v1/yolo_seg.onnx"),
        sha256: "c7690db549f9297fe4632e618b50e8d8efff81d1c1efa55896b288e58e2db13a",
        size: 13842038,
        required: false,
        note: "YOLO-seg instance masks — ultralytics export, AGPL-3.0 (project-hosted, GitHub Release models-v1)",
    },
];

/// Filenames of the models the human chain requires (`mvm run` pre-flight).
pub fn required_files() -> Vec<&'static str> {
    MODELS.iter().filter(|m| m.required).map(|m| m.file).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_required_model_has_a_public_url() {
        // A required model with no fetch source would make `mvm setup` unable
        // to bootstrap a clean machine — the whole point of 14.1.
        for m in MODELS.iter().filter(|m| m.required) {
            assert!(m.url.is_some(), "required model {} has no URL", m.file);
        }
    }

    #[test]
    fn checksums_are_well_formed_sha256() {
        for m in MODELS {
            assert_eq!(m.sha256.len(), 64, "{}: sha256 must be 64 hex chars", m.file);
            assert!(
                m.sha256.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                "{}: sha256 must be lowercase hex",
                m.file
            );
        }
    }

    #[test]
    fn filenames_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for m in MODELS {
            assert!(seen.insert(m.file), "duplicate model file {}", m.file);
        }
    }

    #[test]
    fn yolo_models_are_fetched_from_project_releases() {
        // 14.6: the two AGPL ultralytics exports are project-hosted on GitHub
        // Releases (tag `models-v1`) — `mvm setup` must be able to fetch them.
        for file in ["yolo_object_detect.onnx", "yolo_seg.onnx"] {
            let m = MODELS.iter().find(|m| m.file == file).unwrap();
            let url = m
                .url
                .unwrap_or_else(|| panic!("{file}: project-hosted YOLO must have a release URL"));
            assert!(
                url.starts_with(
                    "https://github.com/howadX/Machine-Vision-Module/releases/download/models-v1/"
                ),
                "{file}: URL must point at the immutable models-v1 release, got {url}"
            );
            assert!(url.ends_with(file), "{file}: asset name must match filename");
        }
    }

    #[test]
    fn required_files_matches_the_five_core_experts() {
        let req = required_files();
        assert_eq!(req.len(), 5, "core chain = YuNet+SFace+emotion+BlazePose pair");
        assert!(req.contains(&"face_recognition_sface_2021dec.onnx"));
        assert!(!req.contains(&"yolo_object_detect.onnx"), "objects are optional");
    }
}
