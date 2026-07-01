//! Body-primary pipeline (phase 8.C-R).
//!
//! Extracts the synchronous `--native-camera` hot-path out of `main.rs` into a
//! self-contained unit that **owns its models** and exposes a single
//! `process(frame)` entry point. This is the precondition for the mesh agents
//! (phase 8.C.5): a `HumanAgent` wraps a `BodyPrimaryPipeline` + its own
//! `PersonTracker` and runs on a thread.
//!
//! Behaviour is identical to the previous free-function `build_body_primary` +
//! the in-loop object/depth cadence + `anchor_objects_depth`; only the *shape*
//! changed (borrowed models → owned fields).

use crate::emotion_wins;
use crate::scene_debug;
use crate::state;
use mvm_core::Detection;

/// Tier 0.2 OOD gate threshold on the pose VAE's `log_prob_z` surprise: a pose
/// above this is flagged `"unusual"` so the `PostureStabilizer` won't commit it.
/// Calibrated to the public-data VAE (`pose_vae_public.bin`): in-distribution
/// `log_prob_z` is p50≈57, p95≈121, p99≈169 — so ~p99 flags only the most
/// surprising poses (and true garbage: empty ROI / hallucinated landmarks).
/// The retired 14.3 belonged to the old user-data VAE and gated *everything* here.
const POSTURE_OOD_LOGPZ: f32 = 170.0;

/// Minimum body-bbox height (normalized) to classify posture at all. A body shorter
/// than this is a fragment / far-partial detection whose landmarks are unreliable —
/// live these bbox_h≈0.08 fragments produced confident false `lying`. Full/seated
/// bodies sit ≥0.35; 0.15 cleanly separates fragments without touching real bodies.
const POSTURE_MIN_BBOX_H: f32 = 0.15;

/// Tier 0.2 plausibility label from a pose's `log_prob_z`.
fn plausibility_flag(logpz: f32) -> &'static str {
    if logpz < POSTURE_OOD_LOGPZ {
        "normal"
    } else {
        "unusual"
    }
}

/// Occlusion-aware posture claim gate. The RF's 3-class label is a *full-body*
/// claim; whether the pipeline may make it depends on what the camera saw:
///
/// - **Full** coverage → the label is trustworthy as-is.
/// - **Upper** (legs occluded — desk/furniture/frame edge): the leg features
///   are hallucinated, so the label is withheld — measured on the owner's desk
///   frames the RF read desk-sitting as confident `lying` (its P(sitting) was
///   ≤0.12 on all 7 references). The one exception is safety-critical: `lying`
///   is still claimed when the **2D torso attitude independently reads
///   horizontal** (a fallen person's legs are often occluded too — the claim
///   then rests on the attitude, not on the hallucinated legs).
/// - **Head** (even hips unseen) → no posture claim at all.
///
/// Sitting-vs-standing behind a desk is geometrically indistinguishable without
/// legs — it is deliberately NOT claimed here; fusion infers the context
/// (`seated_at_workstation`) relationally instead.
///
/// Cross-check on `lying` (all coverages): a `lying` claim additionally requires
/// the 2D attitude to NOT read upright — a lying body cannot have a vertical
/// image-plane torso, so `lying`+upright is a physically impossible combination
/// that only hallucinated legs produce (observed live: a desk frame whose leg
/// visibilities squeaked past the coverage gate still read `lying` at 0.77
/// while the torso was plainly vertical).
fn posture_claimable(coverage: mvm_posture::classifier::BodyCoverage, attitude: &str, label: &str) -> bool {
    use mvm_posture::classifier::BodyCoverage as C;
    if label == "lying" && attitude == "upright" {
        return false; // physically inconsistent — hallucinated-leg artifact
    }
    match coverage {
        C::Full => true,
        C::Upper => label == "lying" && attitude == "horizontal",
        C::Head => false,
    }
}

/// Attach one classified pose to a detection, applying the occlusion-aware
/// claim gate. Coverage/attitude/quality always flow (observability); the
/// `posture` label itself only when [`posture_claimable`] allows the claim.
fn attach_posture(det: &mut Detection, pose: &mvm_posture::classifier::PoseResult) {
    det.body_coverage = Some(pose.coverage.as_str());
    det.torso_attitude = Some(pose.attitude);
    det.recon_error = Some(pose.recon_error);
    det.plausibility = Some(plausibility_flag(pose.plausibility).to_string());
    if posture_claimable(pose.coverage, pose.attitude, pose.label) {
        det.posture = Some(pose.label.to_string());
        det.posture_confidence = Some(pose.confidence);
    }
}

/// Per-body landmark data, kept grouped by body (not flattened) so the mesh can
/// attach each person's wrists + hands to their track (phase 8.F interaction by
/// landmarks). All coordinates normalised [0,1].
pub(crate) struct BodyData {
    pub bbox: mvm_core::BBox,
    /// Visible wrist points (BlazePose 15/16, visibility-gated).
    pub wrists: Vec<[f32; 2]>,
    /// Hand landmark sets for this body (one per detected hand).
    pub hands: Vec<Vec<[f32; 2]>>,
}

/// Per-frame output of [`BodyPrimaryPipeline::process`].
pub(crate) struct FrameResult {
    /// Bodies (+ fallback faces), fully enriched — feed to the tracker as the
    /// ByteTrack high-confidence stage.
    pub detections: Vec<Detection>,
    /// Weak, bbox-only body boxes (phase 9.4) — the ByteTrack low-confidence
    /// recovery stage (`tick_byte`'s `low`). Never enriched, never spawn tracks.
    pub low_conf: Vec<mvm_core::BBox>,
    /// Per-detection pixel landmarks (phase 9.8), aligned 1:1 with `detections`:
    /// `Some` for a body, `None` for a fallback face. Fed to `tick_motion`.
    pub landmarks: Vec<Option<mvm_core::motion::Landmarks33>>,
    /// Per-body landmark data (wrists + hands), grouped by body for per-person
    /// association after tracking.
    pub bodies: Vec<BodyData>,
    /// Per-agent detector overlay for `--scene-debug` (faces / poses / hands).
    pub overlay: scene_debug::DetectorOverlay,
    /// Emotion A/B entries for `--emotion-debug` (empty unless enabled).
    pub emotion_debug_entries: Vec<state::EmotionDebugEntry>,
}

/// The body-primary hot-path: owns every detector/model + the persistent
/// object/depth state that survives across frames on a cadence.
pub(crate) struct BodyPrimaryPipeline {
    config: mvm_core::Config,
    emotion_debug: bool,
    detector: mvm_detect::YunetDetector,
    native_pose: Option<mvm_posture::NativePose>,
    emotion_mod: Option<mvm_emotion::EmotionModule>,
    identity_mod: Option<mvm_identity::IdentityModule>,
    hand_landmarker: Option<mvm_hands::HandLandmarker>,
    scene_detector: Option<mvm_scene::YoloDetector>,
    object_tracker: mvm_scene::ObjectTracker,
    depth_estimator: Option<mvm_depth::DepthEstimator>,
    /// Latest object detections (persist between cadence runs).
    latest_objects: Vec<mvm_scene::TrackedObject>,
    /// Latest relative depth map (persist between cadence runs).
    latest_depth: Option<mvm_depth::DepthMap>,
}

impl BodyPrimaryPipeline {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: mvm_core::Config,
        emotion_debug: bool,
        detector: mvm_detect::YunetDetector,
        native_pose: Option<mvm_posture::NativePose>,
        emotion_mod: Option<mvm_emotion::EmotionModule>,
        identity_mod: Option<mvm_identity::IdentityModule>,
        hand_landmarker: Option<mvm_hands::HandLandmarker>,
        scene_detector: Option<mvm_scene::YoloDetector>,
        object_tracker: mvm_scene::ObjectTracker,
        depth_estimator: Option<mvm_depth::DepthEstimator>,
    ) -> Self {
        Self {
            config,
            emotion_debug,
            detector,
            native_pose,
            emotion_mod,
            identity_mod,
            hand_landmarker,
            scene_detector,
            object_tracker,
            depth_estimator,
            latest_objects: Vec::new(),
            latest_depth: None,
        }
    }

    /// Process one frame: body-primary detection/enrichment + object & depth
    /// cadence. `frame_id` drives the cadences (run every Nth frame). The
    /// detections are returned for the caller to track; depth-anchoring runs
    /// separately (see [`anchor_depth`](Self::anchor_depth)) because it needs
    /// the *post-tick* tracker.
    pub(crate) fn process(
        &mut self,
        frame: &image::DynamicImage,
        frame_id: u64,
        prev_tracks: &[mvm_core::BBox],
        locked_tracks: &[mvm_core::BBox],
        emotion_fresh_tracks: &[mvm_core::BBox],
        predicted_rois: &[mvm_core::BBox],
    ) -> FrameResult {
        let mut overlay = scene_debug::DetectorOverlay::default();
        let mut emotion_debug_entries: Vec<state::EmotionDebugEntry> = Vec::new();
        let mut bodies: Vec<BodyData> = Vec::new();
        let mut low_conf: Vec<mvm_core::BBox> = Vec::new();
        let mut landmarks: Vec<Option<mvm_core::motion::Landmarks33>> = Vec::new();

        // SFace re-check cadence (phase 9.6): every Nth frame re-identify all bodies
        // (defend locked identities); between, locked bodies skip SFace.
        let recheck = SFACE_RECHECK_CADENCE <= 1 || frame_id % SFACE_RECHECK_CADENCE == 0;

        // ── Body-primary enrichment (was build_body_primary) ────────────────
        let detections = build_body_primary(
            frame,
            self.native_pose.as_mut(),
            &mut self.detector,
            self.emotion_mod.as_mut(),
            self.identity_mod.as_mut(),
            self.hand_landmarker.as_mut(),
            &mut overlay,
            &mut bodies,
            &mut low_conf,
            &mut landmarks,
            &self.config,
            self.emotion_debug,
            &mut emotion_debug_entries,
            prev_tracks,
            locked_tracks,
            emotion_fresh_tracks,
            predicted_rois,
            recheck,
        );

        // ── Scene objects (phase 8.E) on a cadence ──────────────────────────
        if let Some(det) = &mut self.scene_detector {
            if self.config.object_cadence <= 1
                || frame_id % self.config.object_cadence as u64 == 0
            {
                match det.detect(frame) {
                    Ok(objs) => {
                        let filtered: Vec<_> = objs
                            .into_iter()
                            .filter(|o| mvm_scene::is_relevant(o.class_name))
                            .filter(|o| {
                                o.score
                                    >= mvm_scene::class_score_threshold(
                                        o.class_name,
                                        self.config.object_score_thresh,
                                    )
                            })
                            .filter(|o| {
                                let area = (o.bbox.x2 - o.bbox.x1) * (o.bbox.y2 - o.bbox.y1);
                                area >= self.config.object_min_area
                            })
                            .collect();
                        self.latest_objects = self.object_tracker.update(&filtered);
                    }
                    Err(e) => eprintln!("  scene: detect error: {e}"),
                }
            }
        }

        // ── Monocular depth (phase 8.E.2) on a sparse cadence ───────────────
        if let Some(est) = &mut self.depth_estimator {
            if self.config.depth_cadence <= 1 || frame_id % self.config.depth_cadence as u64 == 0 {
                match est.estimate(frame) {
                    Ok(dm) => self.latest_depth = Some(dm),
                    Err(e) => eprintln!("  depth: estimate error: {e}"),
                }
            }
        }

        FrameResult { detections, low_conf, landmarks, bodies, overlay, emotion_debug_entries }
    }

    /// Depth-anchor the latest objects to the (post-tick) tracker's people so
    /// objects land in the same metric frame (phase 8.E.2). No-op when no depth
    /// map is available yet.
    pub(crate) fn anchor_depth(
        &mut self,
        tracker: &mvm_core::PersonTracker,
        frame: &image::DynamicImage,
    ) {
        if let Some(depth) = &self.latest_depth {
            anchor_objects_depth(&mut self.latest_objects, depth, tracker, &self.config, frame);
        }
    }

    /// The latest tracked objects (for StateFile / JSON / scene-debug).
    pub(crate) fn objects(&self) -> &[mvm_scene::TrackedObject] {
        &self.latest_objects
    }
}

/// Depth-anchor tracked objects to metric 3D (phase 8.E.2).
///
/// Depth-Anything outputs an affine-relative **inverse** depth (larger = nearer),
/// so for a metric Z and depth value d, `Z·d ≈ k` is roughly constant. People
/// supply the anchor: each has a metric torso-Z (body-primary) and a depth sample,
/// giving `k_i = Z_i·d_i`; we take the median `k`. Each object's metric depth is
/// then `Z = k / d_obj`, back-projected to 3D through the same pinhole people use,
/// so objects land in the **same metric frame** → real person↔object `Near` edges.
///
/// With no person anchor this frame, objects are left depthless (overlap-only) —
/// no fabricated distance.
fn anchor_objects_depth(
    objects: &mut [mvm_scene::TrackedObject],
    depth: &mvm_depth::DepthMap,
    tracker: &mvm_core::PersonTracker,
    config: &mvm_core::Config,
    frame: &image::DynamicImage,
) {
    const OBJ_DEPTH_CONF: f32 = 0.4; // below a person's ws_confidence (anchored, approx)
    const Z_MIN: f32 = 0.3;
    const Z_MAX: f32 = 8.0;

    let mut ks: Vec<f32> = Vec::new();
    for t in tracker.active() {
        if let (Some(z), Some(d)) = (t.position_z, depth.median_in_bbox(&t.bbox)) {
            if d > 1e-3 && z > 0.0 {
                ks.push(z * d);
            }
        }
    }
    if ks.is_empty() {
        for o in objects.iter_mut() {
            o.pos_3d = None;
            o.pos_conf = 0.0;
        }
        return;
    }
    ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ks[ks.len() / 2];

    let pinhole = mvm_core::space::Pinhole::from_config(config);
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    for o in objects.iter_mut() {
        match depth.median_in_bbox(&o.bbox) {
            Some(d) if d > 1e-3 => {
                let z = (k / d).clamp(Z_MIN, Z_MAX);
                let cx = (o.bbox.x1 + o.bbox.x2) * 0.5 * fw;
                let cy = (o.bbox.y1 + o.bbox.y2) * 0.5 * fh;
                o.pos_3d = Some(pinhole.pixel_to_3d(cx, cy, z));
                o.pos_conf = OBJ_DEPTH_CONF;
            }
            _ => {
                o.pos_3d = None;
                o.pos_conf = 0.0;
            }
        }
    }
}

/// Crop a face from `face_bbox` (normalized) and enrich `det` with emotion +
/// identity. Shared by the body-primary path (matched face) and the face-entity
/// fallback. Mirrors the bridge-path enrichment so behaviour is identical.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn enrich_face_crop(
    frame: &image::DynamicImage,
    face_bbox: &mvm_core::BBox,
    det: &mut Detection,
    emotion_mod: Option<&mut mvm_emotion::EmotionModule>,
    identity_mod: Option<&mut mvm_identity::IdentityModule>,
    run_emotion: bool,
    run_identity: bool,
    config: &mvm_core::Config,
    emotion_debug: bool,
    det_idx: usize,
    debug_out: &mut Vec<state::EmotionDebugEntry>,
) {
    let (fw, fh) = (frame.width(), frame.height());
    let x1 = (face_bbox.x1 * fw as f32) as u32;
    let y1 = (face_bbox.y1 * fh as f32) as u32;
    let x2 = (face_bbox.x2 * fw as f32) as u32;
    let y2 = (face_bbox.y2 * fh as f32) as u32;
    let face = frame.crop_imm(x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1));

    // Emotion: keep higher calibrated-confidence prediction — **cadenced** (phase
    // 9.7). Skipped for a body whose track is still coasting on a fresh fused
    // label; the track keeps that label (frozen, no decay while matched).
    if let (true, Some(emo)) = (run_emotion, emotion_mod) {
        if let Ok(result) = emo.classify(&face) {
            let py_emotion = det.emotion.clone();
            let py_conf = det.emotion_confidence;
            det.emotion_distribution = Some(result.distribution);
            let winner = if emotion_wins(&result, det.emotion_confidence) {
                det.emotion = Some(result.label.clone());
                det.emotion_confidence = Some(result.calibrated_conf);
                "rust"
            } else {
                "python"
            };
            if emotion_debug {
                debug_out.push(state::EmotionDebugEntry {
                    id: det_idx as u32,
                    rust_emotion: result.label,
                    rust_conf: result.calibrated_conf,
                    py_emotion,
                    py_conf,
                    winner,
                });
            }
        }
    }

    // Identity + embedding (margin gate, #5/#6) — **cadenced** (phase 9.6).
    // SFace is the priciest model in the chain; `run_identity` is false for a body
    // already covered by an id-locked track (its identity persists on the track),
    // so this whole block — the embed + gallery match — is skipped until the
    // re-check cadence or the track unlocks. Emotion above still runs every frame.
    if run_identity {
        if let Some(id) = identity_mod {
            if let Ok((name, sim, emb)) =
                id.identify_margin(&face, config.identity_abs_thresh, config.identity_margin_thresh)
            {
                det.embedding = Some(emb);
                if det.identity.is_none() && name != "unknown" {
                    det.identity = Some(name);
                    det.identity_confidence = Some(sim);
                }
            }
        }
    }
}

/// Frames between forced SFace re-checks of an id-locked body (phase 9.6 cadence).
/// Between these, a locked body skips SFace entirely; on a re-check frame every
/// body is re-identified to defend against drift / a swapped face. 1 disables the
/// cadence (always run).
const SFACE_RECHECK_CADENCE: u64 = 30;
/// IoU above which a body is considered "the same" as a locked track (so SFace
/// can be skipped this frame). Matches the tracker's association gate.
const IDENTITY_SKIP_IOU: f32 = 0.3;

/// Whether a body overlaps any of `regions` enough to be "the same" entity
/// (shared by the SFace and emotion cadence gates).
fn covered_by_track(body: &mvm_core::BBox, regions: &[mvm_core::BBox]) -> bool {
    regions.iter().any(|r| body.iou(r) > IDENTITY_SKIP_IOU)
}

/// Per-body SFace gate (phase 9.6, refined after the walking-fragmentation
/// finding). Each id-locked track from last frame claims its **single
/// best-overlapping** body as its continuation — that one body skips SFace (its
/// identity persists on the track). **Every other body runs SFace**, so a phantom
/// fragment or a second person that merely overlaps a locked track is still
/// identified instead of lingering anonymous (the old per-body "covered → skip"
/// let all overlapping bodies skip, so phantoms near `id=0` never got re-ID'd).
/// On a re-check frame every body runs. Returns one flag per body, aligned with
/// `bodies`. Pure — unit-tested without models.
fn identity_run_flags(bodies: &[mvm_core::BBox], locked: &[mvm_core::BBox], recheck: bool) -> Vec<bool> {
    let mut run = vec![true; bodies.len()];
    if recheck {
        return run;
    }
    for lb in locked {
        // The locked track's continuation = its single best-IoU body above the gate.
        let mut best_iou = IDENTITY_SKIP_IOU;
        let mut best = None;
        for (i, b) in bodies.iter().enumerate() {
            let iou = b.iou(lb);
            if iou > best_iou {
                best_iou = iou;
                best = Some(i);
            }
        }
        if let Some(i) = best {
            run[i] = false; // only the true continuation skips SFace
        }
    }
    run
}

/// Whether EmotiEffLib should run for one body this frame (phase 9.7 cadence).
/// Skip it when the body is covered by a track whose emotion is still fresh (it
/// coasts on its fused label); run it for a new body or one whose track has aged
/// past the emotion cadence. Pure decision — unit-tested without models.
fn should_run_emotion(body: &mvm_core::BBox, emotion_fresh: &[mvm_core::BBox]) -> bool {
    !covered_by_track(body, emotion_fresh)
}

/// True if the face's centre falls inside any of `regions` (body bboxes this
/// frame, or active-track bboxes from last frame). Used to suppress the
/// face-entity fallback so a person isn't represented twice / doesn't flip
/// between a body-entity and a face-entity (A1 anti-flip, phase 8).
fn face_center_in_any(face: &mvm_core::BBox, regions: &[mvm_core::BBox]) -> bool {
    let cx = (face.x1 + face.x2) * 0.5;
    let cy = (face.y1 + face.y2) * 0.5;
    regions
        .iter()
        .any(|r| cx >= r.x1 && cx <= r.x2 && cy >= r.y1 && cy <= r.y2)
}

/// Index of the smallest unused YuNet face whose bbox **contains the body's nose
/// landmark** (`nose` normalized [0..1]). Keypoint-in-bbox association: the face
/// belongs to the body whose head is inside it. Robust to overlapping person
/// bboxes — a coarse body-bbox-center match swaps identities when two people are
/// close (their person boxes overlap); the nose is physically inside its own
/// face, so it never crosses over.
fn best_face_for_nose(faces: &[Detection], used: &[bool], nose: (f32, f32)) -> Option<usize> {
    let area = |b: &mvm_core::BBox| (b.x2 - b.x1) * (b.y2 - b.y1);
    faces
        .iter()
        .enumerate()
        .filter(|(i, _)| !used[*i])
        .filter(|(_, f)| {
            nose.0 >= f.bbox.x1 && nose.0 <= f.bbox.x2 && nose.1 >= f.bbox.y1 && nose.1 <= f.bbox.y2
        })
        .min_by(|(_, a), (_, b)| area(&a.bbox).partial_cmp(&area(&b.bbox)).unwrap())
        .map(|(i, _)| i)
}

/// Body-primary enrichment (phase 8, FRONTIER). Bodies (person detector) are the
/// carrying entity: each yields a track with posture + torso-3D **always**. A
/// YuNet face matched into the body adds identity + emotion (optional, frontal
/// only). A body without a face still tracks (identity/emotion None) — the core
/// FRONTIER fix. An unmatched face falls back to a face-entity so a blinked
/// body-detection never drops a known person (decision #4).
#[allow(clippy::too_many_arguments)]
fn build_body_primary(
    frame: &image::DynamicImage,
    mut native_pose: Option<&mut mvm_posture::NativePose>,
    detector: &mut mvm_detect::YunetDetector,
    mut emotion_mod: Option<&mut mvm_emotion::EmotionModule>,
    mut identity_mod: Option<&mut mvm_identity::IdentityModule>,
    mut hand_landmarker: Option<&mut mvm_hands::HandLandmarker>,
    overlay: &mut scene_debug::DetectorOverlay,
    bodies_out: &mut Vec<BodyData>,
    low_conf_out: &mut Vec<mvm_core::BBox>,
    landmarks_out: &mut Vec<Option<mvm_core::motion::Landmarks33>>,
    config: &mvm_core::Config,
    emotion_debug: bool,
    debug_out: &mut Vec<state::EmotionDebugEntry>,
    prev_tracks: &[mvm_core::BBox],
    locked_tracks: &[mvm_core::BBox],
    emotion_fresh_tracks: &[mvm_core::BBox],
    predicted_rois: &[mvm_core::BBox],
    recheck: bool,
) -> Vec<Detection> {
    let pinhole = mvm_core::space::Pinhole::from_config(config);
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let body_dets = native_pose
        .as_deref_mut()
        .map(|np| np.bodies_byte_recover(frame, predicted_rois).unwrap_or_default())
        .unwrap_or_default();
    let bodies = body_dets.bodies;
    // Weak boxes ride straight through to the tracker's recovery stage (9.4) —
    // never posed, enriched, or turned into a BodyData.
    low_conf_out.extend(body_dets.low_conf.iter().map(|l| l.bbox));
    // RGB frame for hand ROIs (only materialised when the hand model is loaded).
    let hand_rgb = hand_landmarker.as_ref().map(|_| frame.to_rgb8());
    let body_bboxes: Vec<mvm_core::BBox> = bodies.iter().map(|b| b.bbox).collect();
    let faces = detector.detect(frame).unwrap_or_default();
    // Overlay (scene-debug): every detector's output, drawn per-agent. Faces =
    // all YuNet boxes; poses + hands collected per body below.
    overlay.faces.extend(faces.iter().map(|f| f.bbox));
    let mut face_used = vec![false; faces.len()];
    let mut dets: Vec<Detection> = Vec::with_capacity(bodies.len() + faces.len());
    // SFace cadence (phase 9.6, refined): one flag per body — only each locked
    // track's true continuation skips SFace; phantoms / a second person near it run.
    let id_flags = identity_run_flags(&body_bboxes, locked_tracks, recheck);

    for (bi, body) in bodies.iter().enumerate() {
        let b = &body.bbox;
        let mut det = Detection::bare(b.x1, b.y1, b.x2, b.y2);
        // Posture — gated on body size: a bbox shorter than `POSTURE_MIN_BBOX_H`
        // is a fragment / far-partial detection whose landmarks are unreliable
        // (live: bbox_h≈0.08 fragments produced confident false `lying`). Skip
        // posture there; the body is still tracked, just without a pose this frame.
        if (b.y2 - b.y1) >= POSTURE_MIN_BBOX_H {
            attach_posture(&mut det, &body.posture);
        }
        // Torso-3D (anatomically stable; robust to head turn): shoulders 11/12, hips 23/24.
        // R (phantom filter, phase 8): only trust torso-3D when those landmarks are
        // actually visible. BlazePose visibility = sigmoid(logit) ∈ [0,1]; visible
        // joints sit ~0.9+. Hallucinated landmarks (occlusion / lying poses) score
        // low and otherwise produce garbage 3D (negative-y / far phantoms) — gate
        // them out (the body is still tracked, just without a position this frame).
        let lm = &body.landmarks_px;
        overlay.poses.push(*lm); // scene-debug: BlazePose skeleton (pose agent)
        const TORSO_VIS_MIN: f32 = 0.5;
        let torso_vis = lm[11][3].min(lm[12][3]).min(lm[23][3]).min(lm[24][3]);
        if torso_vis >= TORSO_VIS_MIN {
            let shoulder = ((lm[11][0] + lm[12][0]) * 0.5, (lm[11][1] + lm[12][1]) * 0.5);
            let hip = ((lm[23][0] + lm[24][0]) * 0.5, (lm[23][1] + lm[24][1]) * 0.5);
            if let Some((pos, z, ws)) =
                mvm_core::space::torso_to_3d(&pinhole, shoulder, hip, config.torso_length_m)
            {
                det.distance = Some(z);
                det.position_x = Some(pos[0]);
                det.position_y = Some(pos[1]);
                det.position_z = Some(pos[2]);
                det.ws_confidence = Some(ws);
            }
        }
        // Wrist landmarks (BlazePose 15/16, visibility-gated) — the always-on
        // interaction anchor for this body (normalised).
        let mut wrists: Vec<[f32; 2]> = Vec::new();
        for &idx in &[15usize, 16] {
            if lm[idx][3] >= 0.5 {
                wrists.push([lm[idx][0] / fw, lm[idx][1] / fh]);
            }
        }
        // Hands (phase 8.E.4): ROIs from this body's wrists → handpose (presence-
        // gated). Body-primary — hands belong to the body whose wrist drove them.
        // Keep them grouped per body (phase 8.F: per-person attachment).
        let mut body_hands: Vec<Vec<[f32; 2]>> = Vec::new();
        if let (Some(hl), Some(rgb)) = (hand_landmarker.as_deref_mut(), hand_rgb.as_ref()) {
            let rois = mvm_hands::hand_rois_from_pose(lm, 0.5);
            let hands = hl.infer_rois(rgb, &rois);
            for h in &hands {
                body_hands.push(h.landmarks_px.iter().map(|p| [p[0] / fw, p[1] / fh]).collect());
            }
            overlay.hands.extend(hands); // flat list still feeds --scene-debug
        }
        bodies_out.push(BodyData { bbox: *b, wrists, hands: body_hands });
        // Match a frontal face by the body's NOSE landmark (lm[0]) → identity +
        // emotion (optional). Keypoint-in-bbox ties the face to the body whose
        // head is in it, so overlapping person bboxes never swap identities.
        let nose = (lm[0][0] / fw, lm[0][1] / fh);
        if let Some(fi) = best_face_for_nose(&faces, &face_used, nose) {
            face_used[fi] = true;
            let fb = faces[fi].bbox;
            let idx = dets.len();
            // Cadence (phase 9.6/9.7): skip SFace only for this locked track's own
            // continuation (precomputed in id_flags); skip emotion when covered by
            // a track still coasting on a fresh emotion label.
            let run_identity = id_flags[bi];
            let run_emotion = should_run_emotion(b, emotion_fresh_tracks);
            enrich_face_crop(
                frame, &fb, &mut det,
                emotion_mod.as_deref_mut(), identity_mod.as_deref_mut(),
                run_emotion, run_identity,
                config, emotion_debug, idx, debug_out,
            );
        }
        dets.push(det);
        // Pixel landmarks travel alongside the detection (phase 9.8) → the tracker
        // folds them into this body's track motion ring. Aligned 1:1 with `dets`.
        landmarks_out.push(Some(*lm));
    }

    // Fallback: unmatched faces → face-entity (don't lose a person whose body
    // detection blinked). 3D from face height (YuNet-calibrated `face_height_m`).
    //
    // A1 anti-flip (phase 8): suppress the fallback when the face is already
    // explained by a body THIS frame (nose-match just missed a tight YuNet box)
    // or by an active track from LAST frame (the person-detector blinked — let
    // the body track bridge the gap via Kalman inertia instead of swapping in a
    // posture-less, face-3D entity that makes id 0 jump body↔face every blink).
    // RGB frame for the face→ROI posture recovery below (materialised once, only
    // when a pose model and at least one face are present).
    let face_rgb = (native_pose.is_some() && !faces.is_empty()).then(|| frame.to_rgb8());
    for (fi, f) in faces.iter().enumerate() {
        if face_used[fi] {
            continue;
        }
        if face_center_in_any(&f.bbox, &body_bboxes) || face_center_in_any(&f.bbox, prev_tracks) {
            continue;
        }
        let fb = f.bbox;
        let mut det = Detection::bare(fb.x1, fb.y1, fb.x2, fb.y2);
        let cx = (fb.x1 + fb.x2) * 0.5;
        let cy = (fb.y1 + fb.y2) * 0.5;
        if let Some((pos, z, ws)) =
            mvm_core::space::face_bbox_to_3d(&pinhole, fb.y1, fb.y2, cx, cy, config.face_height_m)
        {
            det.distance = Some(z);
            det.position_x = Some(pos[0]);
            det.position_y = Some(pos[1]);
            det.position_z = Some(pos[2]);
            det.ws_confidence = Some(ws);
        }
        let idx = dets.len();
        // Fallback faces are, by A1, not covered by any body/track this frame — a
        // genuinely uncovered face, so always enrich it fully (nothing to reuse).
        enrich_face_crop(
            frame, &fb, &mut det,
            emotion_mod.as_deref_mut(), identity_mod.as_deref_mut(), true, true,
            config, emotion_debug, idx, debug_out,
        );
        // Detection fix: the person detector missed this body but the face is
        // visible (desk/close view) — derive a body ROI below the face, pose it,
        // and attach the posture (+ landmarks → motion ring) when the torso is
        // actually visible. Stays a face-3D entity (A1); posture is just an added
        // attribute, so this never fabricates a posture from an empty patch.
        let mut face_lm = None;
        if let (Some(np), Some(rgb)) = (native_pose.as_deref_mut(), face_rgb.as_ref()) {
            if let Ok(Some((pose, lm))) =
                np.pose_from_face(rgb, fw, fh, [fb.x1, fb.y1, fb.x2, fb.y2])
            {
                attach_posture(&mut det, &pose);
                overlay.poses.push(lm);
                face_lm = Some(lm);
            }
        }
        dets.push(det);
        landmarks_out.push(face_lm); // Some when posture recovered from the face ROI
    }

    dets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection::bare(x1, y1, x2, y2)
    }

    // ── body-primary face↔body matching (phase 8, FRONTIER) ──

    fn bb(x1: f32, y1: f32, x2: f32, y2: f32) -> mvm_core::BBox {
        mvm_core::BBox { x1, y1, x2, y2 }
    }

    #[test]
    fn identity_runs_for_uncovered_bodies() {
        // No locked track → every body runs SFace.
        let bodies = [bb(0.30, 0.30, 0.50, 0.50), bb(0.10, 0.10, 0.20, 0.20)];
        assert_eq!(identity_run_flags(&bodies, &[], false), vec![true, true]);
        assert_eq!(
            identity_run_flags(&bodies, &[bb(0.80, 0.80, 0.95, 0.95)], false),
            vec![true, true],
            "non-overlapping locked track → all run"
        );
    }

    #[test]
    fn identity_skipped_only_for_locked_continuation() {
        // One locked track covers body 0 → it skips; a lone body runs.
        let bodies = [bb(0.30, 0.30, 0.50, 0.50)];
        let locked = [bb(0.31, 0.31, 0.51, 0.51)]; // high IoU with body 0
        assert_eq!(identity_run_flags(&bodies, &locked, false), vec![false]);
    }

    #[test]
    fn identity_runs_for_phantom_near_locked_track() {
        // THE refinement (walking-fragmentation fix): two bodies both overlap one
        // locked track; only the BEST match (its continuation) skips SFace, the
        // other — a phantom / second person — still runs and gets identified.
        let cont = bb(0.31, 0.31, 0.51, 0.51); // IoU ≈ 0.82 with locked (best)
        let phantom = bb(0.35, 0.35, 0.55, 0.55); // IoU ≈ 0.39 — also ABOVE the gate
        let bodies = [cont, phantom];
        let locked = [bb(0.30, 0.30, 0.50, 0.50)];
        let flags = identity_run_flags(&bodies, &locked, false);
        // Old per-body logic skipped BOTH (both covered). Refined: only the best
        // match skips; the phantom still runs and gets identified.
        assert_eq!(flags, vec![false, true], "continuation skips, phantom runs");
    }

    #[test]
    fn identity_forced_on_recheck_frame() {
        // Even a covered body re-identifies on the re-check cadence (defend drift).
        let bodies = [bb(0.30, 0.30, 0.50, 0.50)];
        let locked = [bb(0.31, 0.31, 0.51, 0.51)];
        assert_eq!(identity_run_flags(&bodies, &locked, true), vec![true]);
    }

    #[test]
    fn emotion_runs_for_uncovered_body() {
        let body = bb(0.30, 0.30, 0.50, 0.50);
        assert!(should_run_emotion(&body, &[]), "no fresh track → run");
        assert!(should_run_emotion(&body, &[bb(0.80, 0.80, 0.95, 0.95)]), "non-overlap → run");
    }

    #[test]
    fn emotion_skipped_for_body_covered_by_fresh_track() {
        let body = bb(0.30, 0.30, 0.50, 0.50);
        let fresh = bb(0.31, 0.31, 0.51, 0.51);
        assert!(!should_run_emotion(&body, std::slice::from_ref(&fresh)), "fresh → coast");
    }

    #[test]
    fn best_face_for_nose_matches_containing() {
        let faces = vec![face(0.20, 0.10, 0.30, 0.20)]; // contains nose (0.25,0.15)
        let used = vec![false];
        assert_eq!(best_face_for_nose(&faces, &used, (0.25, 0.15)), Some(0));
    }

    #[test]
    fn best_face_for_nose_rejects_outside() {
        let faces = vec![face(0.20, 0.10, 0.30, 0.20)];
        let used = vec![false];
        assert_eq!(best_face_for_nose(&faces, &used, (0.80, 0.80)), None);
    }

    #[test]
    fn best_face_for_nose_skips_used() {
        // both faces contain the nose; first already taken → second
        let faces = vec![face(0.10, 0.10, 0.40, 0.40), face(0.10, 0.10, 0.40, 0.40)];
        let used = vec![true, false];
        assert_eq!(best_face_for_nose(&faces, &used, (0.20, 0.20)), Some(1));
    }

    #[test]
    fn best_face_for_nose_picks_smallest() {
        // both contain nose (0.2,0.2); second is smaller (tighter) → preferred
        let faces = vec![face(0.0, 0.0, 0.5, 0.5), face(0.15, 0.15, 0.30, 0.30)];
        let used = vec![false, false];
        assert_eq!(best_face_for_nose(&faces, &used, (0.20, 0.20)), Some(1));
    }

    #[test]
    fn face_center_in_any_suppresses_fallback() {
        let body = mvm_core::BBox { x1: 0.0, y1: 0.0, x2: 0.5, y2: 1.0 };
        let inside = face(0.20, 0.10, 0.30, 0.20); // centre (0.25,0.15) ∈ body
        let outside = face(0.70, 0.10, 0.80, 0.20); // centre (0.75,0.15) ∉ body
        assert!(face_center_in_any(&inside.bbox, &[body]));
        assert!(!face_center_in_any(&outside.bbox, &[body]));
        assert!(!face_center_in_any(&inside.bbox, &[]), "no regions → not suppressed");
    }

    /// The swap-bug regression: two people whose person bboxes OVERLAP. A
    /// body-bbox-center match would assign person B's face to person A; the
    /// nose-based match keeps each face with its own head.
    #[test]
    fn nose_match_survives_overlapping_bodies() {
        // Person A face left, Person B face right; noses inside their own faces.
        let faces = vec![face(0.10, 0.10, 0.25, 0.30), face(0.60, 0.10, 0.75, 0.30)];
        let used = vec![false, false];
        let nose_a = (0.17, 0.20);
        let nose_b = (0.67, 0.20);
        assert_eq!(best_face_for_nose(&faces, &used, nose_a), Some(0));
        assert_eq!(best_face_for_nose(&faces, &used, nose_b), Some(1));
    }

    // ── occlusion-aware posture claim gate ──

    use mvm_posture::classifier::{BodyCoverage, PoseResult};

    fn pose(label: &'static str, coverage: BodyCoverage, attitude: &'static str) -> PoseResult {
        let mut probs = [0.0f32; 3];
        let idx = mvm_posture::classifier::POSE_LABELS.iter().position(|l| *l == label).unwrap();
        probs[idx] = 0.9;
        PoseResult {
            label,
            label_idx: idx,
            confidence: 0.9,
            probs,
            recon_error: 0.01,
            plausibility: 50.0, // in-distribution → "normal"
            coverage,
            attitude,
        }
    }

    #[test]
    fn full_coverage_claims_upright_and_lying() {
        // Full body: standing/sitting claimable with an upright torso; `lying`
        // claimable when the torso attests horizontal (its natural attitude).
        assert!(posture_claimable(BodyCoverage::Full, "upright", "standing"));
        assert!(posture_claimable(BodyCoverage::Full, "upright", "sitting"));
        assert!(posture_claimable(BodyCoverage::Full, "horizontal", "lying"));
    }

    #[test]
    fn lying_with_upright_torso_never_claimed() {
        // Physically impossible combination (a lying body can't have a vertical
        // image-plane torso) — the hallucinated-leg artifact, rejected at every
        // coverage including Full.
        assert!(!posture_claimable(BodyCoverage::Full, "upright", "lying"));
        assert!(!posture_claimable(BodyCoverage::Upper, "upright", "lying"));
    }

    #[test]
    fn upper_coverage_withholds_the_desk_false_lying() {
        // THE live bug this closes: desk-sitting (legs occluded, upright torso)
        // → RF says `lying` → must NOT be claimed.
        assert!(!posture_claimable(BodyCoverage::Upper, "upright", "lying"));
        assert!(!posture_claimable(BodyCoverage::Upper, "upright", "standing"));
        assert!(!posture_claimable(BodyCoverage::Upper, "upright", "sitting"));
        assert!(!posture_claimable(BodyCoverage::Upper, "inclined", "lying"));
    }

    #[test]
    fn upper_coverage_keeps_safety_lying_when_torso_horizontal() {
        // A fallen person's legs are often occluded too — `lying` survives when
        // the 2D attitude independently attests horizontal.
        assert!(posture_claimable(BodyCoverage::Upper, "horizontal", "lying"));
        assert!(!posture_claimable(BodyCoverage::Upper, "horizontal", "standing"));
    }

    #[test]
    fn head_coverage_claims_nothing() {
        for label in ["standing", "sitting", "lying"] {
            assert!(!posture_claimable(BodyCoverage::Head, "horizontal", label));
        }
    }

    #[test]
    fn attach_posture_withholds_label_but_flows_regime() {
        // Upper + upright (the desk case): no posture label, but coverage,
        // attitude and quality signals still reach the detection.
        let mut det = Detection::bare(0.1, 0.1, 0.6, 0.9);
        attach_posture(&mut det, &pose("lying", BodyCoverage::Upper, "upright"));
        assert_eq!(det.posture, None, "hallucinated-leg lying must be withheld");
        assert_eq!(det.body_coverage, Some("upper"));
        assert_eq!(det.torso_attitude, Some("upright"));
        assert_eq!(det.plausibility.as_deref(), Some("normal"));
    }

    #[test]
    fn attach_posture_claims_on_full_coverage() {
        let mut det = Detection::bare(0.1, 0.1, 0.6, 0.9);
        attach_posture(&mut det, &pose("sitting", BodyCoverage::Full, "upright"));
        assert_eq!(det.posture.as_deref(), Some("sitting"));
        assert_eq!(det.posture_confidence, Some(0.9));
        assert_eq!(det.body_coverage, Some("full"));
    }

    // ── depth-anchoring (phase 8.E.2) ──

    /// With no people to supply a metric anchor, objects are left depthless
    /// (overlap-only) — no fabricated distance.
    #[test]
    fn anchor_no_people_leaves_objects_depthless() {
        let config = mvm_core::Config::default();
        let tracker = mvm_core::PersonTracker::with_config(&config); // empty → no active tracks
        let depth = mvm_depth::DepthMap { data: vec![0.5; 16], w: 4, h: 4 };
        let mut objects = vec![mvm_scene::TrackedObject {
            id: 10000,
            class_name: "cup",
            bbox: mvm_core::BBox { x1: 0.1, y1: 0.1, x2: 0.3, y2: 0.3 },
            score: 0.9,
            age: 1,
            missed: 0,
            pos_3d: Some([1.0, 1.0, 1.0]),
            pos_conf: 0.4,
        }];
        let frame = image::DynamicImage::new_rgb8(64, 64);
        anchor_objects_depth(&mut objects, &depth, &tracker, &config, &frame);
        assert_eq!(objects[0].pos_3d, None);
        assert_eq!(objects[0].pos_conf, 0.0);
    }
}
