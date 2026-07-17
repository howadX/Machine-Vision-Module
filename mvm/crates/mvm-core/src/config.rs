//! Global MVM configuration.

use std::path::PathBuf;

/// Global configuration for the MVM pipeline.
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the models directory (ONNX files).
    pub models_path: PathBuf,
    /// Camera device path (e.g. "/dev/video0").
    pub camera_path: String,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Minimum bbox IoU for person matching (default: 0.3).
    pub tracker_iou_threshold: f32,
    /// Number of missed frames before a track expires (default: 1800, ~30s at 60fps).
    pub tracker_timeout_frames: u64,
    /// Identity confidence threshold: below this, person is "unknown".
    pub identity_threshold: f32,
    /// Focal length ratio for pinhole calibration (fx / image_width).
    /// Calibrated per-camera. Default: 0.64 (user's USB camera, June 2026).
    pub fx_ratio: f32,
    /// Effective face-bbox height in metres (depth-from-known-size prior).
    /// NOT anatomical face height: absorbs the detector's bbox/anatomy ratio
    /// (SCRFD boxes ≈ 2× the chin-crown span). Empirically calibrated 2026-06-25
    /// on this ceiling mount: pinhole invariant `pixel_h·z` = 181 px·m over
    /// 1.0–2.0 m → 181/409.6 ≈ 0.44 (fit <3% across 0.5–2.0 m). See space.rs.
    /// Override per-identity via `personal_heights` for tighter fit.
    pub face_height_m: f32,
    /// Effective face-bbox height for **YuNet** boxes (phase 7.1/7.4 native path).
    /// YuNet boxes are tighter than SCRFD, so the SCRFD-calibrated `face_height_m`
    /// (0.44) is wrong for `--native-camera`. The cli swaps this in when native
    /// capture is active.
    ///
    /// Calibrated 2026-06-25 (`tools/calib_3d.py --native-camera`, tape sweep
    /// 0.5–2.5 m, `models/calib_3d_yunet.csv`). real_eff rises with distance due
    /// to the ~45° ceiling tilt: ≈0.30 near (0.5–1.0 m) → ≈0.35 far (2.0–2.5 m).
    /// A single balanced constant **0.32** keeps depth bias within ±~11% across
    /// the working range (inside the documented ±15–30% envelope) and leaves the
    /// ordinal/relative signal — the primary product — unaffected. The proper
    /// tilt fix is the landmark-calibration upgrade path (future phase).
    pub face_height_m_yunet: f32,
    /// Effective torso length in metres (shoulder-midpoint → hip-midpoint), the
    /// depth-from-known-size prior for **body-primary** 3D (phase 8, FRONTIER).
    /// Anatomically ≈0.5 m; like `face_height_m_yunet` this is an *effective*
    /// value that absorbs the ~45° ceiling tilt + BlazePose landmark convention
    /// → recalibrate by tape on this mount. Torso is anatomically stabler than
    /// the face bbox (doesn't jump when the head turns), so it is the preferred
    /// 3D source once the entity is the body. Interim anatomical default 0.5.
    pub torso_length_m: f32,
    /// Camera mount height above the floor in metres (phase 15.6, E2 fall rule).
    /// The camera frame is y-DOWN (`space::pixel_to_3d`), so a person's height
    /// above the floor is `camera_height_m − position_3d.y`. This is the coarse
    /// ground-endpoint reference the scene-level fall rule (`anomaly.rs`) needs to
    /// tell a fall (descends AND ends on the floor) from a fast crouch (descends,
    /// stays up). It is a *mount* prior, not a surveyed floor plane — north-star
    /// framing: the eye *uses* the floor from body/mount priors, it doesn't survey
    /// it. Inherits the uncalibrated 3D scale (tail #3, ±15–30%) → the 0.35 m gate
    /// is approximate; the eye-level rig sits at ~1.75 m.
    pub camera_height_m: f32,
    /// EMA alpha for depth smoothing. Higher = faster response; lower = smoother.
    pub depth_ema_alpha: f32,
    /// Forgetting factor (λ < 1) for temporal emotion PoE fusion. Lower = forgets
    /// the prior faster (more reactive); higher = smoother but slower. Prevents
    /// the per-track distribution from collapsing to one-hot over many frames.
    pub emotion_forget_lambda: f32,
    /// Identity re-link window (frames). A detection that fails IoU matching may
    /// re-attach to a track missed within this many frames if their identity
    /// labels match — keeps the track id stable across motion gaps / brief exits.
    pub relink_window_frames: u64,
    /// Minimum identity confidence required to attempt an identity re-link.
    pub relink_min_conf: f32,
    /// Minimum cosine similarity between SFace embeddings to re-link (phase 7.5).
    /// Replaces label-only re-link → two distinct people never merge into one id.
    pub relink_cos_thresh: f32,
    /// Max normalized bbox-center distance for the geometric re-link fallback
    /// (A2-lite, phase 8): an ANONYMOUS body (no embedding/identity) whose IoU
    /// dropped under fast motion re-attaches to the nearest recently-missed track
    /// within this distance, keeping its ephemeral id stable instead of spawning
    /// a new one. Centers are normalized [0..1]; 0.15 ≈ 15% of frame.
    pub relink_center_dist: f32,
    /// Absolute SFace cosine above which an identity label is accepted (phase 7.5,
    /// fixes #5). Calibrated to the SFace cosine scale, NOT the old ArcFace one
    /// (OpenCV's canonical SFace threshold is 0.363). **Live-validated 2026-06-25
    /// on --native-camera:** the real Вадим lands at 0.55–0.59, a non-gallery
    /// stranger falls below 0.45 (→ unknown). 0.45 sits cleanly between. Was 0.8
    /// (ArcFace era), which would have rejected even the real Вадим on this path.
    pub identity_abs_thresh: f32,
    /// Minimum top1−top2 cosine margin to accept a label when the gallery has ≥2
    /// people (phase 7.5). Guards against ambiguous near-ties.
    pub identity_margin_thresh: f32,
    /// Cadence for the heavy object detector (phase 8.E, G₂ Scene group): run
    /// YOLO every Nth frame, reusing the last result in between. YOLO (~39ms) is
    /// heavier than the ~12ms human chain, so a cadence keeps the hot loop fast
    /// while still refreshing scene context. 1 = every frame. Decision 8.0.1:
    /// cadence lives in the synchronous loop (the main throughput lever), no mesh.
    pub object_cadence: u32,
    /// Base minimum YOLO score to accept an object (phase 8.E filter A). This is
    /// the *floor* for small interaction objects (cup/keyboard/mouse…), kept low
    /// so a ceiling-angle / partial view still registers. Flicker-prone large
    /// furniture is held to a higher per-class bar — see
    /// `mvm_scene::class_score_threshold`.
    pub object_score_thresh: f32,
    /// Minimum normalized bbox area (fraction of frame) for an object node
    /// (phase 8.E filter D). Cuts tiny spurious boxes. Low (0.0015 ≈ 0.15%) for a
    /// ceiling-mounted camera, where objects subtend a small angle.
    pub object_min_area: f32,
    /// Object tracker miss-timeout in detection (cadence) frames (phase 8.E
    /// filter B). A tracked object survives this many missed detections before its
    /// id is freed — higher = more persistent (a briefly-occluded couch keeps its
    /// id instead of respawning with age 1).
    pub object_track_timeout: u32,
    /// Cadence for the monocular depth estimator (phase 8.E.2, G₄). Depth-Anything
    /// is heavy (~230ms CPU) and the scene depth changes slowly, so run it every
    /// Nth frame and reuse the last map. Much sparser than `object_cadence`.
    pub depth_cadence: u32,
    /// Dataset Factory session timeout (phase 16.1.3): a gap longer than this many
    /// seconds with no tracked people closes the current recording session; the
    /// next person opens a new one. Pure `Instant` arithmetic in [`crate::WorldState`]
    /// — the id string (`YYYY-MM-DD_HHMMSS`) is assigned by the recorder (wall-clock
    /// lives outside core). Default 30 s. Consumed only when recording is enabled.
    pub session_timeout_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            models_path: PathBuf::from("./models"),
            camera_path: "/dev/video0".into(),
            width: 640,
            height: 480,
            tracker_iou_threshold: 0.3,
            tracker_timeout_frames: 1800,
            identity_threshold: 0.45,
            fx_ratio: 0.64,
            face_height_m: 0.44, // empirical, 2026-06-25 (was 0.19 population frontal)
            face_height_m_yunet: 0.32, // calibrated 2026-06-25 (YuNet, tilt-balanced)
            torso_length_m: 0.5, // anatomical prior (phase 8 body-primary); recalibrate by tape
            camera_height_m: 1.75, // eye-level rig (matches the mocap lab CAM_H); E2 floor reference
            depth_ema_alpha: 0.3,
            emotion_forget_lambda: 0.7,
            relink_window_frames: 90,
            relink_min_conf: 0.5,
            relink_cos_thresh: 0.5,
            relink_center_dist: 0.15, // A2-lite: anonymous re-link by torso/bbox center
            identity_abs_thresh: 0.45, // SFace cosine scale (was 0.8 ArcFace-era)
            identity_margin_thresh: 0.1,
            object_cadence: 3, // phase 8.E: YOLO every 3rd frame (sync-loop cadence)
            object_score_thresh: 0.30, // phase 8.E filter A: low floor for small desk items; furniture gated per-class
            object_min_area: 0.0015, // phase 8.E filter D: ceiling-cam → small subtended angle
            object_track_timeout: 15, // phase 8.E filter B: persist ~2s across brief occlusion
            depth_cadence: 30, // phase 8.E.2: heavy depth, run sparsely (~every 1.5s)
            session_timeout_secs: 30, // phase 16.1.3: >30s without people → new session
        }
    }
}
