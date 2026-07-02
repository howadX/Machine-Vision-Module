//! Multi-person tracker with IoU-based identity persistence.
//!
//! Tracks people across frames, propagating identity (best confidence),
//! emotion and posture (latest snapshot), and 3D workspace data.
//!
//! ## Kalman State-Gating (phase 6.1)
//!
//! Each tick:
//! 1. `predict()` all tracks → predicted bboxes used for IoU matching.
//! 2. Matched tracks: `update(observed)` → smoothed bbox.
//! 3. Missed tracks: bbox advances by inertia (predicted position).
//!
//! This replaces the previous EMA smoothing. Unlike EMA, missed-frame bboxes
//! now move rather than stagnate, so IoU with re-appearing detections stays high.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use crate::types::{BBox, BodyObservation, Detection, LifecycleState, PersonTrack, FIRST_EPHEMERAL_ID};
use crate::space::{Pinhole, DepthEma, face_bbox_to_3d};

/// Multi-person tracker — the temporal fusion layer of MVM.
pub struct PersonTracker {
    next_id: u32,
    /// Freed ephemeral ids available for reuse (smallest first) — bodies release
    /// their id on exit so anonymous ids stay compact (phase 8 identity model).
    free_ids: BinaryHeap<Reverse<u32>>,
    /// Reserved id per recognised gallery user (name → id < FIRST_EPHEMERAL_ID).
    user_ids: HashMap<String, u32>,
    tracks: HashMap<u32, PersonTrack>,
    pub timeout_frames: u64,
    /// Frames of no detection before emotion confidence starts decaying.
    pub decay_start_frames: u64,
    /// Per-frame multiplier applied to emotion_confidence during decay.
    pub decay_factor: f32,
    /// When Some, native 3D position is computed each tick via pinhole geometry.
    pub pinhole: Option<Pinhole>,
    /// Population-average face height in metres. Fallback when identity is unknown.
    pub face_height_m: f32,
    /// EMA alpha for depth smoothing (applied per-track).
    pub depth_ema_alpha: f32,
    /// Personal face heights keyed by identity label.
    /// When a track is identified, this overrides `face_height_m` → ±2-3% error vs ±12%.
    pub personal_heights: HashMap<String, f32>,
    /// Forgetting factor for temporal emotion PoE fusion (phase 6.4b).
    pub emotion_forget_lambda: f32,
    /// Identity re-link window (frames) — re-attach to a recently-missed track
    /// with matching identity instead of spawning a new id (fixes #6 churn).
    pub relink_window_frames: u64,
    /// Minimum identity confidence to attempt a re-link.
    pub relink_min_conf: f32,
    /// Minimum cosine similarity between SFace embeddings to re-link (phase 7.5).
    /// Embeddings are L2-normalised, so cosine == dot product.
    pub relink_cos_thresh: f32,
    /// Max normalized bbox-center distance for the geometric re-link fallback
    /// (A2-lite, phase 8) — keeps a moving anonymous body's ephemeral id stable.
    pub relink_center_dist: f32,
}

/// Cosine similarity of two L2-normalised embeddings (== dot product).
/// Returns 0.0 on length mismatch (defensive; embeddings are fixed-dim).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return 0.0; }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// IoU below which a detection cannot match a track (State-Gating threshold).
/// The OCM momentum term only *re-orders* candidates above this gate — it never
/// pulls a non-overlapping detection into a match.
const IOU_GATE: f32 = 0.3;
/// Weight of the OCM (observation-centric momentum) term added to the IoU score
/// during association (OC-SORT, phase 9.2). Small by design: it disambiguates
/// near-ties (crossing bodies, weak frames) without overriding a clear IoU win.
const W_OCM: f32 = 0.05;
/// EMA factor for the observation-centric velocity (phase 9.2). Responsive
/// enough to follow real motion, smoothed enough to reject single-frame jitter.
const OBS_VEL_EMA: f32 = 0.5;
/// Vector magnitudes below this read as "no direction" (avoids 0/0 in cosine).
const DIR_EPS: f32 = 1e-4;

/// Observation-Centric Momentum consistency (OC-SORT OCM/OCA, phase 9.2): the
/// cosine between a track's observation velocity `vel` and the bearing `delta`
/// from its last observation to a candidate detection. `+1` = the detection lies
/// straight ahead on the track's path, `-1` = directly behind, `0` = orthogonal
/// **or** undefined (either vector too short to have a direction). Pure scalar
/// arithmetic — no allocation, no model.
fn ocm_consistency(vel: [f32; 2], delta: [f32; 2]) -> f32 {
    let vlen = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
    let dlen = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    if vlen < DIR_EPS || dlen < DIR_EPS {
        return 0.0;
    }
    (vel[0] * delta[0] + vel[1] * delta[1]) / (vlen * dlen)
}

/// Matched frames a track must accrue before it is `Confirmed` (phase 9.3).
/// `age` increments only on a match, so this is "seen this many times".
const CONFIRM_AGE: u64 = 3;
/// Consecutive missed frames after which an established track is declared `Lost`
/// (a soft pre-expiry marker — actual removal still happens at `timeout_frames`).
const LOST_AFTER_MISSED: u64 = 30;

/// One step of the explicit track lifecycle FSM (phase 9.3). Pure function of the
/// previous state and the track's freshly-updated `age`/`missed` plus whether its
/// identity is `locked` (holds a reserved gallery id). Tested exhaustively at the
/// transition boundaries.
///
/// - matched this frame (`missed == 0`): climb the trust ladder —
///   `Tentative` until `age ≥ CONFIRM_AGE`, then `IdLocked` if locked else
///   `Confirmed` (this also recovers an `Occluded`/`Lost` track on re-match).
/// - missed this frame: an unconfirmed (`Tentative`) track drops straight to
///   `Lost`; an established/occluded track goes `Occluded`, then `Lost` once the
///   gap reaches `LOST_AFTER_MISSED`.
/// Fold one observation centre into a track's observation-centric velocity
/// (OC-SORT, phase 9.2/9.4). MUST be called BEFORE zeroing `missed` — `missed`
/// is the frame gap since the last observation, so dividing by it yields a
/// per-frame rate that survives occlusion. EMA-smoothed; seeds raw on the second
/// observation. Shared by the high- and low-confidence association stages.
fn record_obs(t: &mut PersonTrack, obs_c: [f32; 2]) {
    if let Some(prev) = t.last_obs_center {
        let gap = t.missed.max(1) as f32;
        let inst = [(obs_c[0] - prev[0]) / gap, (obs_c[1] - prev[1]) / gap];
        t.obs_velocity = Some(match t.obs_velocity {
            Some(v) => [
                OBS_VEL_EMA * inst[0] + (1.0 - OBS_VEL_EMA) * v[0],
                OBS_VEL_EMA * inst[1] + (1.0 - OBS_VEL_EMA) * v[1],
            ],
            None => inst,
        });
    }
    t.last_obs_center = Some(obs_c);
}

fn next_lifecycle(prev: LifecycleState, age: u64, missed: u64, locked: bool) -> LifecycleState {
    use LifecycleState::*;
    if missed == 0 {
        if age < CONFIRM_AGE {
            Tentative
        } else if locked {
            IdLocked
        } else {
            Confirmed
        }
    } else {
        match prev {
            Tentative => Lost, // unconfirmed + a miss → almost certainly a flicker
            Lost => Lost,
            _ if missed >= LOST_AFTER_MISSED => Lost,
            _ => Occluded,
        }
    }
}

impl PersonTracker {
    /// Create tracker without 3D capability (tests, headless use).
    pub fn new() -> Self {
        Self {
            next_id: FIRST_EPHEMERAL_ID,
            free_ids: BinaryHeap::new(),
            user_ids: HashMap::new(),
            tracks: HashMap::new(),
            timeout_frames: 1800,
            decay_start_frames: 5,
            decay_factor: 0.8,
            pinhole: None,
            face_height_m: 0.44, // empirical effective bbox-face height (see config.rs)
            depth_ema_alpha: 0.3,
            personal_heights: HashMap::new(),
            emotion_forget_lambda: 0.7,
            relink_window_frames: 90,
            relink_min_conf: 0.5,
            relink_cos_thresh: 0.5,
            relink_center_dist: 0.15,
        }
    }

    /// Create tracker with native 3D enabled (pinhole derived from config).
    pub fn with_config(cfg: &crate::config::Config) -> Self {
        Self {
            pinhole: Some(Pinhole::from_config(cfg)),
            face_height_m: cfg.face_height_m,
            depth_ema_alpha: cfg.depth_ema_alpha,
            emotion_forget_lambda: cfg.emotion_forget_lambda,
            relink_window_frames: cfg.relink_window_frames,
            relink_min_conf: cfg.relink_min_conf,
            relink_cos_thresh: cfg.relink_cos_thresh,
            relink_center_dist: cfg.relink_center_dist,
            ..Self::new()
        }
    }

    /// Register a gallery user, assigning the next reserved id (0, 1, 2 …). A
    /// recognised body adopts this id while it holds the user (phase 8). Idempotent.
    pub fn register_user(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.user_ids.get(name) {
            return id;
        }
        let id = self.user_ids.len() as u32;
        debug_assert!(id < FIRST_EPHEMERAL_ID, "more than {FIRST_EPHEMERAL_ID} gallery users");
        self.user_ids.insert(name.to_string(), id);
        id
    }

    /// Allocate an ephemeral track id: reuse the smallest freed id, else bump.
    fn alloc_id(&mut self) -> u32 {
        match self.free_ids.pop() {
            Some(Reverse(id)) => id,
            None => {
                let id = self.next_id;
                self.next_id += 1;
                id
            }
        }
    }

    /// Process one frame with **only** high-confidence detections (the common
    /// case and the phase 6–8 behaviour). Equivalent to [`Self::tick_byte`] with
    /// no low-confidence boxes.
    pub fn tick(&mut self, detections: &[Detection]) {
        self.tick_byte(detections, &[]);
    }

    /// Process one frame with ByteTrack two-stage association (phase 9.4) under
    /// the **inverted contract** (phase 9.5): associate FIRST on geometry, then
    /// apply each detection's enrichment to its matched track, then finalize.
    ///
    /// `high` — confident detections (matched via IoU + OCM + the re-link cascade;
    /// unmatched ones spawn tracks). `low` — weak bbox-only boxes that only recover
    /// a track that missed this frame (never enriched, never spawn).
    pub fn tick_byte(&mut self, high: &[Detection], low: &[BBox]) {
        self.tick_motion(high, low, &[], 0.0);
    }

    /// Like [`Self::tick_byte`] but also folds each high detection's landmarks into
    /// its matched track's motion ring (phase 9.8). `landmarks[i]` aligns with
    /// `high[i]` (image pixels), stamped `t_secs` seconds; a `None` entry or a
    /// short/empty slice simply skips the motion update (so `tick_byte` is exactly
    /// this with no landmarks). Persisting motion on the track is what removes the
    /// IoU round-trip in `project_people`.
    pub fn tick_motion(
        &mut self,
        high: &[Detection],
        low: &[BBox],
        landmarks: &[Option<crate::motion::Landmarks33>],
        t_secs: f64,
    ) {
        let ids = self.associate_core(high, low);
        for (i, det) in high.iter().enumerate() {
            let id = ids[i];
            self.apply_detection(id, det);
            if let Some(Some(lm)) = landmarks.get(i) {
                if let Some(t) = self.tracks.get_mut(&id) {
                    t.motion.observe(*lm, t_secs);
                }
            }
        }
        self.finalize();
    }

    /// Lean, enrichment-free association (phase 9.5 — the inverted contract).
    /// Matches body observations to tracks on **geometry alone** (IoU + OCM +
    /// observation-centric recovery) and returns the track id chosen for each. An
    /// observation carries no identity/embedding, so the cosine/label re-link tiers
    /// are inert — geometry is all there is until enrichment runs. The caller then
    /// enriches per id via [`Self::apply_detection`] and closes the frame with
    /// [`Self::finalize`]; this is what lets 9.6/9.7 run SFace/emotion only on the
    /// tracks that need it instead of on every body every frame.
    pub fn associate(&mut self, high: &[BodyObservation], low: &[BBox]) -> Vec<u32> {
        let dets: Vec<Detection> = high
            .iter()
            .map(|o| Detection::bare(o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2))
            .collect();
        self.associate_core(&dets, low)
    }

    /// Association core shared by [`Self::tick_byte`] and [`Self::associate`]:
    /// predict → high match cascade → ByteTrack low recovery → inertia → expire,
    /// doing only **association bookkeeping** (Kalman update, observation velocity,
    /// age/missed). No attribute enrichment, no display-id/lifecycle resolution
    /// (callers do those after enriching). Returns the matched or newly created
    /// track id for each `high` detection, in order. A detection bearing
    /// `identity`/`embedding` engages the cosine/label re-link tiers; a bbox-only
    /// one skips them and relies on geometry.
    fn associate_core(&mut self, high: &[Detection], low: &[BBox]) -> Vec<u32> {
        // ── Step 1: predict all tracks (State-Gating) ──────────────────────
        // KF is now in predicted state; predicted[id] = bbox to use for IoU.
        let predicted: HashMap<u32, BBox> = self.tracks.iter_mut()
            .map(|(&id, t)| (id, t.kalman.predict()))
            .collect();

        // ── Step 2: missed-frame bookkeeping + emotion decay ───────────────
        for t in self.tracks.values_mut() {
            t.missed += 1;
            // Emotion-cadence age (phase 9.7): climbs every tick; apply_detection
            // resets it to 0 when an emotion observation lands.
            t.frames_since_emotion = t.frames_since_emotion.saturating_add(1);
            if t.missed > self.decay_start_frames {
                if let Some(conf) = t.emotion_confidence.as_mut() {
                    *conf *= self.decay_factor;
                    if *conf < 0.05 {
                        t.emotion = None;
                        t.emotion_confidence = None;
                    }
                }
            }
        }

        // ── Step 3: match detections to predicted bboxes ───────────────────
        let mut matched_ids: Vec<u32> = Vec::new();
        let mut ids: Vec<u32> = Vec::with_capacity(high.len());

        // ── Step 3a: GLOBAL primary assignment (Hungarian, phase 9.10) ─────
        // Formerly a per-detection greedy IoU pick: each detection independently
        // grabbed the best free track in detector-output order. Greedy is locally
        // optimal but globally wrong on a crossing — the first detection can steal
        // the track a later one needed, forcing an id swap or a phantom id. We now
        // build the whole IoU+OCM cost matrix and solve it once with Kuhn–Munkres,
        // minimising the TOTAL cost across all detections. The per-pair score is
        // unchanged (`iou + W_OCM·momentum`, gated by IOU_GATE → forbidden), so on
        // 1–2 unambiguous bodies the result is identical to the old greedy match;
        // it only diverges where greedy was suboptimal. Columns are the tracks
        // holding a prediction this tick, sorted by id for a deterministic matrix.
        let cand_ids: Vec<u32> = {
            let mut v: Vec<u32> = predicted.keys().copied().collect();
            v.sort_unstable();
            v
        };
        let primary_track: Vec<Option<u32>> = if high.is_empty() || cand_ids.is_empty() {
            vec![None; high.len()]
        } else {
            let cost: Vec<Vec<f32>> = high
                .iter()
                .map(|det| {
                    let det_cx = (det.bbox.x1 + det.bbox.x2) * 0.5;
                    let det_cy = (det.bbox.y1 + det.bbox.y2) * 0.5;
                    cand_ids
                        .iter()
                        .map(|id| {
                            let pred_bbox = predicted[id];
                            let iou = det.bbox.iou(&pred_bbox);
                            if iou <= IOU_GATE {
                                return f32::INFINITY; // gate → forbidden pair
                            }
                            // OCM (phase 9.2): the momentum term re-orders matches
                            // above the gate toward the track whose motion points
                            // at this detection. Inert for a lone candidate; on a
                            // crossing it (with the global solve) prevents the swap.
                            let t = &self.tracks[id];
                            let momentum = match (t.obs_velocity, t.last_obs_center) {
                                (Some(v), Some(c)) => {
                                    ocm_consistency(v, [det_cx - c[0], det_cy - c[1]])
                                }
                                _ => 0.0,
                            };
                            -(iou + W_OCM * momentum) // minimise cost = maximise score
                        })
                        .collect()
                })
                .collect();
            crate::assignment::solve(&cost)
                .into_iter()
                .map(|c| c.map(|j| cand_ids[j]))
                .collect()
        };
        // Reserve every primary-matched track up front so the re-link cascade for
        // an UNMATCHED detection cannot steal a track the global solver already
        // assigned to a different detection (the cascade skips `matched_ids`).
        for &id in primary_track.iter().flatten() {
            matched_ids.push(id);
        }

        for (d, det) in high.iter().enumerate() {
            let mut best_id = primary_track[d];

            // Identity re-link (fixes #6): if IoU found nothing, try to re-attach
            // to a recently-missed track instead of spawning a new id.
            //
            // Preferred path — **cosine of SFace embeddings** (phase 7.5): robust
            // against a single-name gallery, because it compares the actual face
            // vectors, not just the assigned label. Two different "Вадим"s never
            // merge. Falls back to label-equality only when no embedding is present
            // (Python-only signal).
            if best_id.is_none() {
                if let Some(det_emb) = &det.embedding {
                    let mut best_cos = self.relink_cos_thresh;
                    for (&id, t) in &self.tracks {
                        if matched_ids.contains(&id) { continue; }
                        if t.missed == 0 || t.missed > self.relink_window_frames { continue; }
                        if let Some(track_emb) = &t.embedding {
                            let cos = cosine(det_emb, track_emb);
                            if cos > best_cos {
                                best_cos = cos;
                                best_id = Some(id);
                            }
                        }
                    }
                }
            }
            // Fallback: label-based re-link (no embedding available).
            if best_id.is_none() && det.embedding.is_none() {
                if let (Some(name), Some(conf)) = (&det.identity, det.identity_confidence) {
                    if conf >= self.relink_min_conf {
                        let mut best_missed = u64::MAX;
                        for (&id, t) in &self.tracks {
                            if matched_ids.contains(&id) { continue; }
                            // Only recently-missed tracks (a gap), not active ones.
                            if t.missed == 0 || t.missed > self.relink_window_frames { continue; }
                            if t.identity.as_deref() == Some(name.as_str()) && t.missed < best_missed {
                                best_missed = t.missed;
                                best_id = Some(id);
                            }
                        }
                    }
                }
            }

            // Center-distance re-link (A2-lite, phase 8): last-resort geometric
            // fallback for ANONYMOUS bodies (no embedding, no identity) whose IoU
            // dropped under fast motion. Re-attach to the NEAREST recently-missed
            // track within `relink_center_dist` so a moving anonymous person keeps
            // its ephemeral id instead of churning a new one each gap.
            if best_id.is_none() && det.embedding.is_none() && det.identity.is_none() {
                let dcx = (det.bbox.x1 + det.bbox.x2) * 0.5;
                let dcy = (det.bbox.y1 + det.bbox.y2) * 0.5;
                let mut best_dist = self.relink_center_dist;
                for (&id, t) in &self.tracks {
                    if matched_ids.contains(&id) { continue; }
                    if t.missed == 0 || t.missed > self.relink_window_frames { continue; }
                    // Only re-link to anonymous tracks (don't steal a user's track).
                    if t.identity.is_some() { continue; }
                    // OCR (OC-SORT, phase 9.2): compare against the track's VIRTUAL
                    // position — its last *observation* extrapolated along its
                    // observation velocity across the gap — not the drifted Kalman
                    // bbox. A body that kept moving while unseen is sought where its
                    // own trajectory predicts, so a fast occluded body re-links
                    // instead of churning a fresh id. Falls back to the last
                    // observation (then the bbox) when no velocity is known yet.
                    let (vcx, vcy) = match (t.last_obs_center, t.obs_velocity) {
                        (Some(c), Some(v)) => {
                            let gap = t.missed as f32;
                            (c[0] + v[0] * gap, c[1] + v[1] * gap)
                        }
                        (Some(c), None) => (c[0], c[1]),
                        (None, _) => ((t.bbox.x1 + t.bbox.x2) * 0.5, (t.bbox.y1 + t.bbox.y2) * 0.5),
                    };
                    let d = ((dcx - vcx).powi(2) + (dcy - vcy).powi(2)).sqrt();
                    if d < best_dist {
                        best_dist = d;
                        best_id = Some(id);
                    }
                }
            }

            // Association bookkeeping ONLY — enrichment is deferred to
            // apply_detection (phase 9.5 inverted contract).
            let id = match best_id {
                Some(id) => {
                    let t = self.tracks.get_mut(&id).unwrap();
                    t.kalman.update(det.bbox);
                    t.bbox = t.kalman.smoothed_bbox();
                    t.age += 1;
                    // Observation-centric velocity (OC-SORT, 9.2): BEFORE zeroing missed.
                    record_obs(t, [(det.bbox.x1 + det.bbox.x2) * 0.5,
                                   (det.bbox.y1 + det.bbox.y2) * 0.5]);
                    t.missed = 0;
                    // Primary (Hungarian) matches are already reserved above; a
                    // cascade re-link match is not — reserve it now so later
                    // detections in this tick don't double-claim the track.
                    if !matched_ids.contains(&id) {
                        matched_ids.push(id);
                    }
                    id
                }
                None => {
                    // New body → spawn a BARE track (age 1, missed 0, obs anchor
                    // seeded, all attributes empty). apply_detection seeds it below
                    // (its first observation). Not added to matched_ids: it has no
                    // prediction entry, so the inertia step leaves it untouched.
                    let id = self.alloc_id();
                    self.tracks.insert(id, PersonTrack::new(id, det.bbox));
                    id
                }
            };
            ids.push(id);
        }

        // ── Step 3b: ByteTrack low-confidence recovery (phase 9.4) ─────────
        // Weak bbox-only boxes rescue tracks the high stage left unmatched. A
        // candidate track must have missed THIS frame (`missed >= 1`) — that
        // excludes both stage-1 matches and tracks just spawned this tick (those
        // sit at `missed == 0`). Match by IoU + OCM on the predicted bbox; on a
        // hit, advance the filter + observation velocity but inject no metadata.
        for low_bbox in low {
            let lcx = (low_bbox.x1 + low_bbox.x2) * 0.5;
            let lcy = (low_bbox.y1 + low_bbox.y2) * 0.5;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = None;
            for (&id, t) in &self.tracks {
                if t.missed == 0 || matched_ids.contains(&id) { continue; }
                let Some(&pred_bbox) = predicted.get(&id) else { continue; };
                let iou = low_bbox.iou(&pred_bbox);
                if iou <= IOU_GATE { continue; }
                let momentum = match (t.obs_velocity, t.last_obs_center) {
                    (Some(v), Some(c)) => ocm_consistency(v, [lcx - c[0], lcy - c[1]]),
                    _ => 0.0,
                };
                let score = iou + W_OCM * momentum;
                if score > best_score {
                    best_score = score;
                    best_id = Some(id);
                }
            }
            if let Some(id) = best_id {
                let t = self.tracks.get_mut(&id).unwrap();
                t.kalman.update(*low_bbox);
                t.bbox = t.kalman.smoothed_bbox();
                t.age += 1;
                record_obs(t, [lcx, lcy]); // before zeroing missed (gap)
                t.missed = 0;
                // No enrichment: a low-conf box has no trustworthy attributes.
                matched_ids.push(id);
            }
        }

        // ── Step 4: advance missed tracks by inertia ───────────────────────
        // KF predict() was already called in step 1; update bbox to predicted position.
        for (&id, t) in self.tracks.iter_mut() {
            if !matched_ids.contains(&id) {
                if let Some(&pred) = predicted.get(&id) {
                    t.bbox = pred;
                }
            }
        }

        // Expire stale tracks; return their ephemeral ids to the reuse pool.
        let timeout = self.timeout_frames;
        let expired: Vec<u32> = self
            .tracks
            .iter()
            .filter(|(_, t)| t.missed >= timeout)
            .map(|(&id, _)| id)
            .collect();
        for id in expired {
            self.tracks.remove(&id);
            self.free_ids.push(Reverse(id));
        }

        ids
    }

    /// Apply one detection's enrichment to track `id` (phase 9.5) — the
    /// post-association half of the inverted contract. Identity (keep-best),
    /// temporal-PoE emotion, posture, 3D (passthrough or native pinhole) and the
    /// fresh embedding. A bare track just spawned by association is **seeded** here
    /// (its first observation); an existing track is **updated**. Safe to call once
    /// per (id, frame), between [`Self::associate`] and [`Self::finalize`].
    pub fn apply_detection(&mut self, id: u32, det: &Detection) {
        // Pre-capture Copy config to avoid borrowing self alongside self.tracks.
        let pinhole_opt = self.pinhole;
        let face_height_m = self.face_height_m;
        let depth_ema_alpha = self.depth_ema_alpha;
        let forget = self.emotion_forget_lambda;
        let lock_cos = self.relink_cos_thresh;

        // ── Part A: metadata ───────────────────────────────────────────────
        {
            let t = self.tracks.get_mut(&id).unwrap();
            // Identity lock (phase 9.6). A fresh embedding either CONFIRMS the
            // track's face (cosine-stable → climb toward a lock so SFace can stop)
            // or CONTRADICTS it (reset → SFace re-engages: defend / re-check on
            // doubt). Update the counter against the OLD embedding before replacing
            // it. Also keeps the latest vector for cosine re-link after a gap.
            if let Some(new_emb) = &det.embedding {
                t.id_stable_count = match &t.embedding {
                    Some(old) if cosine(new_emb, old) >= lock_cos => {
                        t.id_stable_count.saturating_add(1)
                    }
                    _ => 1, // first sighting or a different face → fresh lock attempt
                };
                t.embedding = Some(new_emb.clone());
            }
            // Propagate metadata: keep best identity, latest emotion/posture.
            if let (Some(name), Some(conf)) = (&det.identity, det.identity_confidence) {
                if t.identity_confidence.map_or(true, |c| conf > c) {
                    t.identity = Some(name.clone());
                    t.identity_confidence = Some(conf);
                }
            }
            // Emotion: temporal Product-of-Experts fusion (phase 6.4b). A full
            // distribution folds into the track's prior with forgetting, then the
            // label is re-derived from the SMOOTHED distribution (kills per-frame
            // argmax jitter); a bare track's `None` prior is seeded by the obs.
            // Without a distribution (Python-only signal), latest-snapshot.
            if let Some(obs) = det.emotion_distribution {
                let fused = match t.emotion_distribution {
                    Some(prior) => crate::emotion::fuse_emotion_temporal(&prior, &obs, forget),
                    None => obs, // first observation seeds the prior
                };
                t.emotion_distribution = Some(fused);
                // Tier 0.3: emit the label only when the smoothed distribution is
                // peaked enough to be informative. A near-chance read (entropy
                // ≥ EMOTION_MAX_ENTROPY) suppresses the label — downstream sees no
                // meaningless "Fear" — while the raw distribution is still kept
                // above for callers that read the full signal.
                match crate::emotion::emotion_label_gated(&fused) {
                    Some((label, conf)) => {
                        t.emotion = Some(label.to_string());
                        t.emotion_confidence = Some(conf);
                    }
                    None => {
                        t.emotion = None;
                        t.emotion_confidence = None;
                    }
                }
            } else if det.emotion.is_some() {
                t.emotion = det.emotion.clone();
                t.emotion_confidence = det.emotion_confidence;
            }
            // Emotion cadence (phase 9.7): an observation landed → reset the age,
            // so this track coasts on its fused label until the cadence elapses.
            if det.emotion_distribution.is_some() || det.emotion.is_some() {
                t.frames_since_emotion = 0;
            }
            // Posture temporal stabiliser (Tier 0.1): route the per-frame label
            // through dwell + confidence gate instead of writing it straight onto
            // the track. A single bad frame (partial landmarks, borderline pose)
            // no longer flips the emitted posture. Missing confidence reads as 0
            // → treated as an ambiguous frame that holds the committed label.
            if let Some(label) = &det.posture {
                let conf = det.posture_confidence.unwrap_or(0.0);
                // Tier 0.2: the pose VAE's own OOD verdict ("unusual") gates the
                // label; additionally a `lying` label while the track is *moving*
                // is distrusted (lying is static — moving+lying = bad/fragment
                // frame). Absent plausibility (bridge/tests) defaults to trusted.
                // "Moving" = committed Moving OR instantaneous centroid speed above
                // a still-line (0.15 torso/s). Speed catches the startup/settling
                // case the committed state misses (ring not yet dwelled to Moving);
                // a truly lying body is ~motionless and passes.
                let speed = if t.motion.is_empty() { 0.0 } else { t.motion.speed() };
                let moving = t.motion.state() == crate::motion::MotionState::Moving || speed > 0.15;
                let trusted =
                    crate::posture::posture_trusted(det.plausibility.as_deref(), label, moving);
                t.posture_stab.observe(label, conf, trusted);
                t.posture = t.posture_stab.label().map(str::to_string);
                t.posture_confidence = t.posture_stab.confidence();
            } else if matches!(det.body_coverage, Some("upper") | Some("head")) {
                // Occlusion-aware posture: the upstream *withheld* the label
                // because the lower body was not seen (desk/furniture/frame
                // edge) — a hidden frame. Sustained occlusion (dwell) clears the
                // committed label so a stale full-body claim (e.g. `standing`
                // from before sitting down at the desk) never outlives what the
                // camera can actually attest. A short blip leaves it untouched.
                t.posture_stab.observe_hidden();
                t.posture = t.posture_stab.label().map(str::to_string);
                t.posture_confidence = t.posture_stab.confidence();
            }
            // Detection-provided 3D passthrough. Body-primary (pinhole = None): this
            // IS the final 3D, EMA-smoothed against the previous position to damp
            // per-frame landmark jitter (A3). Bridge path: overridden by Part B.
            if det.distance.is_some() {
                let a = depth_ema_alpha;
                let ema = |prev: Option<f32>, new: Option<f32>| match (prev, new) {
                    (Some(p), Some(n)) => Some(a * n + (1.0 - a) * p),
                    (None, n) => n,
                    (p, None) => p,
                };
                t.distance = ema(t.distance, det.distance);
                t.position_x = ema(t.position_x, det.position_x);
                t.position_y = ema(t.position_y, det.position_y);
                t.position_z = ema(t.position_z, det.position_z);
                t.ws_confidence = det.ws_confidence;
            }
            if det.recon_error.is_some() { t.recon_error = det.recon_error; }
            if det.plausibility.is_some() { t.plausibility = det.plausibility.clone(); }
            if det.body_coverage.is_some() { t.body_coverage = det.body_coverage; }
            if det.torso_attitude.is_some() { t.torso_attitude = det.torso_attitude; }
            // Head orientation (phase 10.3): latest snapshot, like body_coverage.
            if det.head_yaw.is_some() {
                t.head_yaw = det.head_yaw;
                t.head_facing = det.head_facing;
            }
        } // mutable borrow drops

        // ── Part B: native 3D (phase 6.2) — two-step borrow ────────────────
        if let Some(ph) = pinhole_opt {
            let real_h = {
                let track = self.tracks.get(&id).unwrap();
                track.identity.as_ref()
                    .and_then(|n| self.personal_heights.get(n.as_str()))
                    .copied()
                    .unwrap_or(face_height_m)
            };
            let t = self.tracks.get_mut(&id).unwrap();
            let b = t.bbox;
            let bbox_cx = (b.x1 + b.x2) * 0.5;
            let bbox_cy = (b.y1 + b.y2) * 0.5;
            if let Some((_pos, z, ws_conf)) =
                face_bbox_to_3d(&ph, b.y1, b.y2, bbox_cx, bbox_cy, real_h)
            {
                let smoothed_z = t.depth_ema
                    .get_or_insert_with(|| DepthEma::new(z, depth_ema_alpha))
                    .update(z);
                let pos_s = ph.pixel_to_3d(bbox_cx * ph.width, bbox_cy * ph.height, smoothed_z);
                t.distance = Some(smoothed_z);
                t.position_x = Some(pos_s[0]);
                t.position_y = Some(pos_s[1]);
                t.position_z = Some(pos_s[2]);
                t.ws_confidence = Some(ws_conf);
            }
        }
    }

    /// Close the frame (phase 9.5): resolve reserved display ids, then step the
    /// lifecycle FSM (which reads the just-resolved identity lock). Call after all
    /// [`Self::apply_detection`] write-backs. Matching/expiry are unchanged, so
    /// churn is identical to the pre-inversion tracker.
    pub fn finalize(&mut self) {
        self.resolve_display_ids();
        for t in self.tracks.values_mut() {
            // IdLocked (phase 9.6): a reserved gallery user OR a cosine-stable face.
            let locked = t.is_id_locked();
            t.lifecycle = next_lifecycle(t.lifecycle, t.age, t.missed, locked);
        }
    }

    /// Resolve each track's `display_id` (phase 8 identity model). Every track
    /// defaults to its ephemeral id; then each reserved user id is granted to the
    /// single highest-confidence ACTIVE track carrying that identity (exclusive
    /// binding — one body per user at a time, so the #5 single-gallery stranger
    /// keeps its ephemeral id instead of cloning "Вадим").
    fn resolve_display_ids(&mut self) {
        for t in self.tracks.values_mut() {
            t.display_id = t.id;
        }
        let users: Vec<(String, u32)> =
            self.user_ids.iter().map(|(k, &v)| (k.clone(), v)).collect();
        for (name, uid) in users {
            let best = self
                .tracks
                .values()
                .filter(|t| t.missed == 0 && t.identity.as_deref() == Some(name.as_str()))
                .max_by(|a, b| {
                    a.identity_confidence
                        .unwrap_or(0.0)
                        .partial_cmp(&b.identity_confidence.unwrap_or(0.0))
                        .unwrap()
                })
                .map(|t| t.id);
            if let Some(bid) = best {
                if let Some(t) = self.tracks.get_mut(&bid) {
                    t.display_id = uid;
                }
            }
        }
    }

    pub fn active(&self) -> impl Iterator<Item = &PersonTrack> {
        self.tracks.values().filter(|t| t.missed == 0)
    }

    #[allow(dead_code)]
    pub fn all(&self) -> &HashMap<u32, PersonTrack> {
        &self.tracks
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kalman::KalmanBBox;
    use crate::types::{BBox, Detection};

    fn det(x1: f32, y1: f32, x2: f32, y2: f32) -> Vec<Detection> {
        vec![Detection::bare(x1, y1, x2, y2)]
    }

    fn det_id(x1: f32, y1: f32, x2: f32, y2: f32, name: &str, conf: f32) -> Vec<Detection> {
        vec![Detection {
            bbox: BBox { x1, y1, x2, y2 },
            identity: Some(name.into()),
            identity_confidence: Some(conf),
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            body_coverage: None, torso_attitude: None, emotion_distribution: None,
            embedding: None,
            head_yaw: None, head_facing: None,
        }]
    }

    /// Detection carrying an SFace embedding (and matching identity label).
    fn det_emb(x1: f32, y1: f32, x2: f32, y2: f32, name: &str, emb: Vec<f32>) -> Vec<Detection> {
        let mut d = Detection::bare(x1, y1, x2, y2);
        d.identity = Some(name.into());
        d.identity_confidence = Some(0.9);
        d.embedding = Some(emb);
        vec![d]
    }

    #[test]
    fn cosine_helper_dot_of_normalised() {
        // Orthogonal → 0, identical → 1.
        assert!((cosine(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[1.0, 0.0], &[1.0]), 0.0, "length mismatch → 0");
    }

    #[test]
    fn embedding_propagates_to_track() {
        let mut t = PersonTracker::new();
        t.tick(&det_emb(0.3, 0.3, 0.5, 0.5, "Вадим", vec![1.0, 0.0, 0.0]));
        let tr = t.active().next().unwrap();
        assert_eq!(tr.embedding.as_deref(), Some(&[1.0, 0.0, 0.0][..]));
    }

    #[test]
    fn cosine_relink_recovers_same_person() {
        // Same face vector reappears far away (no IoU) → re-link via cosine, not new id.
        let mut t = PersonTracker::new();
        t.tick(&det_emb(0.1, 0.1, 0.3, 0.3, "Вадим", vec![1.0, 0.0, 0.0]));
        let id0 = t.active().next().unwrap().id;
        t.tick(&det_emb(0.7, 0.7, 0.9, 0.9, "Вадим", vec![0.99, 0.14, 0.0])); // cos≈0.99
        assert_eq!(t.track_count(), 1, "same embedding must re-link");
        assert_eq!(t.active().next().unwrap().id, id0, "id persists via cosine re-link");
    }

    #[test]
    fn cosine_relink_rejects_different_person() {
        // CORE of #5/#6: two different people both labelled "Вадим" (single-name
        // gallery false-positive). Label-only re-link would MERGE them; cosine
        // keeps them apart because the face vectors are orthogonal.
        let mut t = PersonTracker::new();
        t.tick(&det_emb(0.1, 0.1, 0.3, 0.3, "Вадим", vec![1.0, 0.0, 0.0]));
        t.tick(&det_emb(0.7, 0.7, 0.9, 0.9, "Вадим", vec![0.0, 1.0, 0.0])); // cos=0 < thresh
        assert_eq!(t.track_count(), 2, "different face vectors must NOT merge");
    }

    #[test]
    fn relink_recovers_id_after_motion_gap() {
        // Same identity reappears far away (no IoU overlap) → re-attach, not new id.
        let mut t = PersonTracker::new();
        t.tick(&det_id(0.1, 0.1, 0.3, 0.3, "Вадим", 0.9));
        let id0 = t.active().next().unwrap().id;
        t.tick(&det_id(0.7, 0.7, 0.9, 0.9, "Вадим", 0.9)); // disjoint bbox → IoU=0
        assert_eq!(t.track_count(), 1, "should re-link, not spawn a new track");
        assert_eq!(t.active().next().unwrap().id, id0, "id must persist via re-link");
    }

    #[test]
    fn relink_requires_matching_identity() {
        let mut t = PersonTracker::new();
        t.tick(&det_id(0.1, 0.1, 0.3, 0.3, "Вадим", 0.9));
        t.tick(&det_id(0.7, 0.7, 0.9, 0.9, "Stranger", 0.9));
        assert_eq!(t.track_count(), 2, "different identity must NOT re-link");
    }

    #[test]
    fn relink_respects_window() {
        let mut t = PersonTracker::new();
        t.relink_window_frames = 2;
        t.tick(&det_id(0.1, 0.1, 0.3, 0.3, "Вадим", 0.9));
        for _ in 0..3 { t.tick(&[]); }            // missed grows beyond the window
        t.tick(&det_id(0.7, 0.7, 0.9, 0.9, "Вадим", 0.9));
        assert_eq!(t.track_count(), 2, "stale track beyond window must NOT re-link");
    }

    #[test]
    fn relink_skips_low_confidence() {
        let mut t = PersonTracker::new();
        t.tick(&det_id(0.1, 0.1, 0.3, 0.3, "Вадим", 0.9));
        t.tick(&det_id(0.7, 0.7, 0.9, 0.9, "Вадим", 0.4)); // below relink_min_conf
        assert_eq!(t.track_count(), 2, "low-confidence identity must NOT re-link");
    }

    #[test]
    fn tracker_creates_new_track_on_first_detection() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        assert_eq!(t.track_count(), 1);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].id, FIRST_EPHEMERAL_ID);
        assert_eq!(active[0].age, 1);
    }

    #[test]
    fn tracker_propagates_identity() {
        let mut t = PersonTracker::new();
        let enriched = vec![Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: Some("Neutral".into()),
            emotion_confidence: Some(0.65),
            posture: Some("сидит".into()),
            posture_confidence: Some(0.95),
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            body_coverage: None, torso_attitude: None, emotion_distribution: None,
            embedding: None,
            head_yaw: None, head_facing: None,
        }];
        t.tick(&enriched);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].identity.as_deref(), Some("Вадим"));
        assert_eq!(active[0].emotion.as_deref(), Some("Neutral"));
        assert_eq!(active[0].posture.as_deref(), Some("сидит"));
    }

    #[test]
    fn sustained_upper_coverage_clears_stale_posture() {
        // Occlusion-aware posture: walked in standing (Full coverage), then sat
        // behind the desk — the pipeline withholds `posture` and reports
        // `body_coverage: "upper"`. After the stabiliser's dwell the stale
        // `standing` is cleared; coverage/attitude keep flowing as snapshots.
        let mut t = PersonTracker::new();
        let full = Detection {
            posture: Some("standing".into()),
            posture_confidence: Some(0.9),
            body_coverage: Some("full"),
            torso_attitude: Some("upright"),
            ..Detection::bare(0.3, 0.3, 0.7, 0.7)
        };
        t.tick(std::slice::from_ref(&full));
        assert_eq!(t.active().next().unwrap().posture.as_deref(), Some("standing"));

        let upper = Detection {
            body_coverage: Some("upper"),
            torso_attitude: Some("upright"),
            ..Detection::bare(0.3, 0.3, 0.7, 0.7)
        };
        t.tick(std::slice::from_ref(&upper)); // hidden 1
        t.tick(std::slice::from_ref(&upper)); // hidden 2
        assert_eq!(
            t.active().next().unwrap().posture.as_deref(),
            Some("standing"),
            "below dwell the committed label holds (blip protection)"
        );
        t.tick(std::slice::from_ref(&upper)); // hidden 3 → cleared
        let tr = t.active().next().unwrap();
        assert_eq!(tr.posture, None, "sustained occlusion clears the stale claim");
        assert_eq!(tr.body_coverage, Some("upper"));
        assert_eq!(tr.torso_attitude, Some("upright"));
    }

    #[test]
    fn tracker_keeps_best_identity() {
        let mut t = PersonTracker::new();
        t.tick(&[Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 },
            identity: Some("unknown".into()),
            identity_confidence: Some(0.30),
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            body_coverage: None, torso_attitude: None, emotion_distribution: None,
            embedding: None,
            head_yaw: None, head_facing: None,
        }]);
        t.tick(&[Detection {
            bbox: BBox { x1: 0.31, y1: 0.31, x2: 0.71, y2: 0.71 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            body_coverage: None, torso_attitude: None, emotion_distribution: None,
            embedding: None,
            head_yaw: None, head_facing: None,
        }]);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].identity.as_deref(), Some("Вадим"));
        assert!((active[0].identity_confidence.unwrap() - 0.72).abs() < 0.01);
    }

    #[test]
    fn tracker_updates_emotion_on_match() {
        let mut t = PersonTracker::new();
        t.tick(&[Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.5, y2: 0.5 },
            emotion: Some("Anger".into()), emotion_confidence: Some(0.6),
            identity: None, identity_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            body_coverage: None, torso_attitude: None, emotion_distribution: None,
            embedding: None,
            head_yaw: None, head_facing: None,
        }]);
        t.tick(&[Detection {
            bbox: BBox { x1: 0.32, y1: 0.31, x2: 0.52, y2: 0.51 },
            emotion: Some("Happiness".into()), emotion_confidence: Some(0.8),
            identity: None, identity_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            body_coverage: None, torso_attitude: None, emotion_distribution: None,
            embedding: None,
            head_yaw: None, head_facing: None,
        }]);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].emotion.as_deref(), Some("Happiness"));
    }

    #[test]
    fn tracker_matches_by_iou() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        t.tick(&det(0.32, 0.31, 0.72, 0.71));
        assert_eq!(t.track_count(), 1);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].id, FIRST_EPHEMERAL_ID);
        assert_eq!(active[0].age, 2);
    }

    #[test]
    fn tracker_creates_separate_tracks_for_non_overlapping() {
        let mut t = PersonTracker::new();
        t.tick(&[
            Detection::bare(0.1, 0.1, 0.3, 0.5),
            Detection::bare(0.7, 0.1, 0.9, 0.5),
        ]);
        assert_eq!(t.track_count(), 2);
    }

    #[test]
    fn tracker_does_not_match_below_threshold() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.0, 0.0, 0.2, 0.2));
        t.tick(&det(0.7, 0.7, 0.9, 0.9));
        assert_eq!(t.track_count(), 2);
    }

    #[test]
    fn tracker_increments_missed_when_not_seen() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        t.tick(&[]);
        let all: Vec<_> = t.all().values().collect();
        assert_eq!(all[0].missed, 1);
        assert_eq!(all[0].age, 1);
    }

    #[test]
    fn tracker_expires_stale_tracks() {
        let mut t = PersonTracker::new();
        t.timeout_frames = 3;
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        t.tick(&[]);
        t.tick(&[]);
        t.tick(&[]);
        assert_eq!(t.track_count(), 0);
    }

    /// Kalman smooths bbox toward new observations (replaces EMA smoothing test).
    #[test]
    fn tracker_bbox_smooths_toward_observation() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.5, 0.5));
        t.tick(&det(0.35, 0.3, 0.55, 0.5));
        let active: Vec<_> = t.active().collect();
        let b = active[0].bbox;
        // bbox should move toward the new observation without jumping directly
        assert!(b.x1 >= 0.3 && b.x1 <= 0.35, "x1={:.4} should be between initial and observed", b.x1);
        assert!(b.x2 >= 0.5 && b.x2 <= 0.55, "x2={:.4} should be between initial and observed", b.x2);
    }

    #[test]
    fn tracker_active_filters_missed() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.1, 0.1, 0.3, 0.3));
        t.tick(&det(0.7, 0.7, 0.9, 0.9));
        let active: Vec<_> = t.active().collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, FIRST_EPHEMERAL_ID + 1);
    }

    // ── phase 8 identity model: reserved user ids + ephemeral bodies ──

    #[test]
    fn register_user_assigns_sequential_reserved_ids() {
        let mut t = PersonTracker::new();
        assert_eq!(t.register_user("Вадим"), 0);
        assert_eq!(t.register_user("Анна"), 1);
        assert_eq!(t.register_user("Вадим"), 0); // idempotent
    }

    #[test]
    fn recognised_user_gets_reserved_display_id() {
        let mut t = PersonTracker::new();
        t.register_user("Вадим");
        t.tick(&det_id(0.3, 0.3, 0.7, 0.7, "Вадим", 0.9));
        let a = t.active().next().unwrap();
        assert_eq!(a.display_id, 0, "recognised user adopts reserved id");
        assert!(a.id >= FIRST_EPHEMERAL_ID, "internal id stays ephemeral");
        assert_eq!(a.shown_identity().map(|s| s.as_str()), Some("Вадим"));
    }

    #[test]
    fn unregistered_identity_keeps_ephemeral_and_hides_name() {
        let mut t = PersonTracker::new();
        // "Stranger" not registered → no reserved id, name not shown downstream
        t.tick(&det_id(0.3, 0.3, 0.7, 0.7, "Stranger", 0.9));
        let a = t.active().next().unwrap();
        assert!(a.display_id >= FIRST_EPHEMERAL_ID);
        assert!(a.shown_identity().is_none());
    }

    #[test]
    fn exclusive_user_binding_one_body_per_user() {
        // Two bodies both labelled "Вадим" (the #5 single-gallery case): only the
        // higher-confidence one holds id 0; the other stays anonymous.
        let mut t = PersonTracker::new();
        t.register_user("Вадим");
        let mut dets = det_id(0.1, 0.1, 0.3, 0.3, "Вадим", 0.9);
        dets.extend(det_id(0.6, 0.6, 0.8, 0.8, "Вадим", 0.6));
        t.tick(&dets);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active.len(), 2);
        let winner = active.iter().find(|x| x.display_id == 0).expect("one body holds id 0");
        assert_eq!(winner.identity_confidence, Some(0.9));
        assert!(winner.shown_identity().is_some());
        let other = active.iter().find(|x| x.display_id != 0).unwrap();
        assert!(other.display_id >= FIRST_EPHEMERAL_ID);
        assert!(other.shown_identity().is_none(), "loser reads anonymous");
    }

    #[test]
    fn anonymous_body_relinks_by_center_under_fast_motion() {
        // Anonymous body (no identity/embedding) jumps to a disjoint bbox (IoU=0)
        // but its center moved < relink_center_dist → re-link, not a new id (A2-lite).
        let mut t = PersonTracker::new();
        t.tick(&det(0.30, 0.30, 0.38, 0.50)); // center (0.34,0.40), id 10
        let id0 = t.active().next().unwrap().id;
        t.tick(&det(0.42, 0.30, 0.50, 0.50)); // disjoint x (IoU 0), center (0.46,0.40) → 0.12 away
        assert_eq!(t.track_count(), 1, "center re-link should reuse the track");
        assert_eq!(t.active().next().unwrap().id, id0, "ephemeral id stays stable");
    }

    #[test]
    fn center_relink_skips_when_too_far() {
        // Disjoint AND far (> relink_center_dist) → genuinely a new person/track.
        let mut t = PersonTracker::new();
        t.tick(&det(0.10, 0.10, 0.18, 0.30)); // center (0.14,0.20)
        t.tick(&det(0.70, 0.70, 0.78, 0.90)); // center (0.74,0.80) → ~0.85 away
        assert_eq!(t.track_count(), 2, "far jump must spawn a new track, not re-link");
    }

    #[test]
    fn body_3d_is_ema_smoothed_when_pinhole_disabled() {
        // Body-primary: tracker has no pinhole, so detection-provided torso 3D is
        // EMA-smoothed across frames (A3). alpha defaults to 0.3.
        let mut t = PersonTracker::new(); // pinhole = None
        let mut d1 = det(0.3, 0.3, 0.7, 0.7);
        d1[0].distance = Some(1.0);
        d1[0].position_z = Some(1.0);
        t.tick(&d1);
        assert_eq!(t.active().next().unwrap().distance, Some(1.0), "first frame seeds raw");
        let mut d2 = det(0.3, 0.3, 0.7, 0.7); // same bbox → IoU match
        d2[0].distance = Some(2.0);
        d2[0].position_z = Some(2.0);
        t.tick(&d2);
        let sm = t.active().next().unwrap().distance.unwrap();
        // EMA: 0.3*2.0 + 0.7*1.0 = 1.3
        assert!((sm - 1.3).abs() < 0.01, "expected EMA-smoothed 1.3, got {sm}");
    }

    #[test]
    fn ephemeral_id_freed_on_exit_is_reused() {
        let mut t = PersonTracker::new();
        t.timeout_frames = 1; // remove after one missed frame
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        let first = t.active().next().unwrap().id;
        assert_eq!(first, FIRST_EPHEMERAL_ID);
        t.tick(&[]); // no detections → track misses, expires, id freed
        assert_eq!(t.track_count(), 0);
        t.tick(&det(0.1, 0.1, 0.2, 0.2)); // new body reuses the freed id
        assert_eq!(t.active().next().unwrap().id, FIRST_EPHEMERAL_ID);
    }

    #[test]
    fn tracker_id_increments() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.0, 0.0, 0.2, 0.2));
        t.tick(&[]);
        t.tick(&det(0.5, 0.5, 0.7, 0.7));
        assert_eq!(t.track_count(), 2);
    }

    #[test]
    fn tracker_empty_ticks() {
        let mut t = PersonTracker::new();
        for _ in 0..100 { t.tick(&[]); }
        assert_eq!(t.track_count(), 0);
    }

    fn det_with_emotion(emotion: &str, conf: f32) -> Vec<Detection> {
        vec![Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 },
            emotion: Some(emotion.into()),
            emotion_confidence: Some(conf),
            identity: None, identity_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
            body_coverage: None, torso_attitude: None, emotion_distribution: None,
            embedding: None,
            head_yaw: None, head_facing: None,
        }]
    }

    #[test]
    fn emotion_confidence_decays_after_missed_frames() {
        let mut t = PersonTracker::new();
        t.decay_start_frames = 3;
        t.decay_factor = 0.8;
        t.tick(&det_with_emotion("Happiness", 0.9));
        for _ in 0..3 { t.tick(&[]); }
        let tracks: Vec<_> = t.all().values().collect();
        assert!((tracks[0].emotion_confidence.unwrap() - 0.9).abs() < 0.01);
        t.tick(&[]);
        let tracks: Vec<_> = t.all().values().collect();
        assert!(tracks[0].emotion_confidence.unwrap() < 0.9);
    }

    #[test]
    fn emotion_clears_when_confidence_drops_below_threshold() {
        let mut t = PersonTracker::new();
        t.decay_start_frames = 0;
        t.decay_factor = 0.5;
        t.tick(&det_with_emotion("Sadness", 0.8));
        for _ in 0..10 { t.tick(&[]); }
        let tracks: Vec<_> = t.all().values().collect();
        assert!(tracks[0].emotion.is_none());
        assert!(tracks[0].emotion_confidence.is_none());
    }

    #[test]
    fn emotion_restores_on_rematch_after_decay() {
        let mut t = PersonTracker::new();
        t.decay_start_frames = 0;
        t.decay_factor = 0.5;
        t.tick(&det_with_emotion("Sadness", 0.8));
        for _ in 0..10 { t.tick(&[]); }
        let tracks: Vec<_> = t.all().values().collect();
        assert!(tracks[0].emotion.is_none());
        t.tick(&det_with_emotion("Happiness", 0.75));
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].emotion.as_deref(), Some("Happiness"));
        assert!((active[0].emotion_confidence.unwrap() - 0.75).abs() < 0.01);
    }

    // ── OC-SORT: observation-centric momentum + recovery (phase 9.2) ─────────

    #[test]
    fn ocm_consistency_aligned_opposed_perp_and_zero() {
        // Pure cosine of velocity vs bearing-to-detection.
        assert!((ocm_consistency([1.0, 0.0], [2.0, 0.0]) - 1.0).abs() < 1e-6, "aligned → +1");
        assert!((ocm_consistency([1.0, 0.0], [-2.0, 0.0]) + 1.0).abs() < 1e-6, "opposed → -1");
        assert!(ocm_consistency([1.0, 0.0], [0.0, 2.0]).abs() < 1e-6, "perpendicular → 0");
        assert!(ocm_consistency([0.0, 0.0], [1.0, 0.0]).abs() < 1e-6, "no velocity → 0");
        assert!(ocm_consistency([1.0, 0.0], [0.0, 0.0]).abs() < 1e-6, "no displacement → 0");
    }

    #[test]
    fn obs_velocity_none_until_second_observation() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.25, 0.45, 0.35, 0.55));
        let tr = t.active().next().unwrap();
        assert!(tr.last_obs_center.is_some(), "anchor seeded at birth");
        assert!(tr.obs_velocity.is_none(), "no direction from a single observation");
    }

    #[test]
    fn obs_velocity_tracks_rightward_motion() {
        // A body drifting right (overlapping boxes keep one track) builds a
        // positive-x observation velocity.
        let mut t = PersonTracker::new();
        for i in 0..5u32 {
            let x1 = 0.20 + i as f32 * 0.05;
            t.tick(&det(x1, 0.40, x1 + 0.20, 0.60));
        }
        assert_eq!(t.track_count(), 1, "drifting body stays one track");
        let v = t.active().next().unwrap().obs_velocity.expect("velocity after motion");
        assert!(v[0] > 0.0, "observation velocity points right, got {v:?}");
        assert!(v[1].abs() < 1e-3, "no vertical drift, got {}", v[1]);
    }

    #[test]
    fn obs_velocity_is_gap_normalized() {
        // Observe at 0.30, miss one frame, observe at 0.40: the 0.10 displacement
        // spans a 2-frame gap → 0.05 per frame, NOT 0.10. Locks per-frame rate.
        let mut t = PersonTracker::new();
        t.tick(&det(0.25, 0.45, 0.35, 0.55)); // center 0.30
        t.tick(&[]); // occluded one frame
        t.tick(&det(0.35, 0.45, 0.45, 0.55)); // center 0.40, re-linked
        assert_eq!(t.track_count(), 1, "should re-link across the 1-frame gap");
        let v = t.active().next().unwrap().obs_velocity.expect("velocity after gap");
        assert!((v[0] - 0.05).abs() < 0.01, "gap-normalised 0.05/frame, got {}", v[0]);
    }

    #[test]
    fn ocm_resolves_crossing_without_swap() {
        // Two tracks at the SAME predicted box (pure IoU tie) but opposite
        // observation velocities. Two detections continue each motion. OCM must
        // route each detection to the direction-consistent track — no id swap.
        // Without the momentum term the IoU tie is broken by HashMap order (flaky
        // swap); with it the assignment is deterministic and correct.
        let mut t = PersonTracker::new();
        t.tick(&[
            Detection::bare(0.10, 0.45, 0.20, 0.55),
            Detection::bare(0.80, 0.45, 0.90, 0.55),
        ]);
        let mut ids: Vec<u32> = t.all().keys().copied().collect();
        ids.sort();
        let (right_id, left_id) = (ids[0], ids[1]);
        let box5 = BBox { x1: 0.45, y1: 0.45, x2: 0.55, y2: 0.55 };
        for &(id, vx) in [(right_id, 0.10_f32), (left_id, -0.10_f32)].iter() {
            let tr = t.tracks.get_mut(&id).unwrap();
            tr.kalman = KalmanBBox::new(box5);
            tr.bbox = box5;
            tr.last_obs_center = Some([0.5, 0.5]);
            tr.obs_velocity = Some([vx, 0.0]);
            tr.missed = 0;
        }
        // R continues right (center 0.54), L continues left (center 0.46).
        t.tick(&[
            Detection::bare(0.49, 0.45, 0.59, 0.55),
            Detection::bare(0.41, 0.45, 0.51, 0.55),
        ]);
        assert_eq!(t.track_count(), 2, "no churn — both tracks persist");
        let rc = t.tracks.get(&right_id).unwrap().last_obs_center.unwrap();
        let lc = t.tracks.get(&left_id).unwrap().last_obs_center.unwrap();
        assert!(rc[0] > 0.5, "rightward track followed the right detection, got {rc:?}");
        assert!(lc[0] < 0.5, "leftward track followed the left detection, got {lc:?}");
    }

    #[test]
    fn hungarian_global_assignment_beats_greedy_order() {
        // Ambiguity the greedy-by-detector-order matcher gets wrong (phase 9.10).
        // Two existing tracks: T0 centred 0.45, T1 centred 0.52. Two detections:
        //   D0 @0.48 overlaps BOTH (IoU 0.54 to T0, 0.43 to T1) — prefers T0.
        //   D1 @0.43 overlaps ONLY T0 (IoU 0.67 to T0, 0.05 to T1 → below gate).
        //
        // Greedy in detector order: D0 (first) grabs its best, T0. D1's only IoU
        // option (T0) is now taken → it falls to the center re-link cascade and
        // mis-attaches to T1 (0.09 < relink_center_dist), so T0 ends following the
        // 0.48 box and T1 the 0.43 box — a swapped, suboptimal association.
        //
        // Hungarian minimises total cost: D1→T0 is forced (its only feasible cell),
        // so D0→T1. Each track follows its NEAREST detection: T0→0.43, T1→0.48.
        let mut t = PersonTracker::new();
        // Tick 1 — seed the two tracks (centres 0.45 and 0.52).
        t.tick(&vec![
            Detection::bare(0.40, 0.45, 0.50, 0.55), // centre 0.45 → smaller id (T0)
            Detection::bare(0.47, 0.45, 0.57, 0.55), // centre 0.52 → T1
        ]);
        assert_eq!(t.track_count(), 2, "two tracks seeded");
        let mut ids: Vec<u32> = t.all().keys().copied().collect();
        ids.sort_unstable();
        let (t0, t1) = (ids[0], ids[1]);

        // Tick 2 — the ambiguous pair.
        t.tick(&vec![
            Detection::bare(0.43, 0.45, 0.53, 0.55), // D0 centre 0.48
            Detection::bare(0.38, 0.45, 0.48, 0.55), // D1 centre 0.43
        ]);
        assert_eq!(t.track_count(), 2, "no phantom track — both detections matched");
        let c0 = t.tracks.get(&t0).unwrap().last_obs_center.unwrap();
        let c1 = t.tracks.get(&t1).unwrap().last_obs_center.unwrap();
        // Optimal (Hungarian): T0 follows D1 (0.43), T1 follows D0 (0.48).
        assert!((c0[0] - 0.43).abs() < 0.01, "T0 took the nearer detection D1, got {c0:?}");
        assert!((c1[0] - 0.48).abs() < 0.01, "T1 took D0, got {c1:?}");
    }

    #[test]
    fn ocr_virtual_trajectory_beats_stale_filter() {
        // A body seen once at 0.25, then (manually) occluded for several frames
        // while carrying a rightward observation velocity. Its Kalman bbox is
        // stale at 0.25, but its virtual trajectory extrapolates to ~0.65. A new
        // detection at 0.65 is DISJOINT from the stale filter (IoU 0) → only the
        // observation-centric virtual position recovers it. The old stale-bbox
        // re-link (compare to 0.25) would be 0.40 away → spawn a churned track.
        let mut t = PersonTracker::new();
        t.tick(&det(0.20, 0.45, 0.30, 0.55)); // center 0.25, id = FIRST_EPHEMERAL_ID
        let id0 = t.active().next().unwrap().id;
        {
            let tr = t.tracks.get_mut(&id0).unwrap();
            tr.obs_velocity = Some([0.10, 0.0]); // rightward
            tr.last_obs_center = Some([0.25, 0.50]);
            tr.missed = 3; // pretend 3 occluded frames already elapsed
            // kalman left stale at center 0.25 (velocity ~0)
        }
        t.tick(&det(0.60, 0.45, 0.70, 0.55)); // center 0.65 = virtual (0.25 + 0.10*4)
        assert_eq!(t.track_count(), 1, "virtual trajectory recovers the moving body");
        assert_eq!(t.active().next().unwrap().id, id0, "id persists via OCR, no churn");
    }

    #[test]
    fn ocr_recovers_occluded_moving_body() {
        // End-to-end regression guard: a body moving right, briefly occluded,
        // reappearing ahead on its path stays one track.
        let mut t = PersonTracker::new();
        for i in 0..4u32 {
            let x1 = 0.20 + i as f32 * 0.05;
            t.tick(&det(x1, 0.40, x1 + 0.20, 0.60)); // centers 0.30..0.45
        }
        let id0 = t.active().next().unwrap().id;
        for _ in 0..3 { t.tick(&[]); } // occluded
        t.tick(&det(0.52, 0.40, 0.72, 0.60)); // center 0.62, ahead on the trajectory
        assert_eq!(t.track_count(), 1, "occluded moving body recovers as one track");
        assert_eq!(t.active().next().unwrap().id, id0, "id persists across occlusion");
    }

    // ── per-track motion (phase 9.8) ────────────────────────────────────────

    /// Synthetic landmark frame with shoulders (11/12) + hips (23/24) around `cx`.
    fn motion_lm(cx: f32) -> crate::motion::Landmarks33 {
        let mut lm = [[0.0f32; 4]; 33];
        lm[11] = [cx - 20.0, 100.0, 0.0, 1.0];
        lm[12] = [cx + 20.0, 100.0, 0.0, 1.0];
        lm[23] = [cx - 15.0, 200.0, 0.0, 1.0];
        lm[24] = [cx + 15.0, 200.0, 0.0, 1.0];
        lm
    }

    #[test]
    fn tick_motion_feeds_moving_into_track() {
        // A single body whose landmarks translate → the track's own motion reads
        // Moving, with no IoU round-trip.
        let mut t = PersonTracker::new();
        for i in 0..8 {
            let lm = motion_lm(100.0 + i as f32 * 50.0); // +50px/0.5s, torso 100px → 1 torso/s
            t.tick_motion(&det(0.3, 0.3, 0.5, 0.5), &[], &[Some(lm)], i as f64 * 0.5);
        }
        let tr = t.active().next().unwrap();
        assert_eq!(tr.motion.state(), crate::motion::MotionState::Moving);
    }

    #[test]
    fn tick_motion_ring_does_not_grow() {
        // Many frames on one track → the per-track ring stays bounded (no leak).
        let mut t = PersonTracker::new();
        for i in 0..300 {
            let lm = motion_lm(100.0);
            t.tick_motion(&det(0.3, 0.3, 0.5, 0.5), &[], &[Some(lm)], i as f64 * 0.05);
        }
        assert_eq!(t.track_count(), 1);
        assert!(t.active().next().unwrap().motion.len() <= crate::motion::MOTION_RING_CAP);
    }

    #[test]
    fn tick_byte_leaves_motion_unknown() {
        // The no-landmarks path doesn't touch motion (stays Unknown).
        let mut t = PersonTracker::new();
        t.tick_byte(&det(0.3, 0.3, 0.5, 0.5), &[]);
        assert_eq!(t.active().next().unwrap().motion.state(), crate::motion::MotionState::Unknown);
    }

    // ── lifecycle FSM (phase 9.3) ───────────────────────────────────────────

    #[test]
    fn next_lifecycle_transitions() {
        use LifecycleState::*;
        // Climbing the trust ladder on matches (missed == 0).
        assert_eq!(next_lifecycle(Tentative, 2, 0, false), Tentative, "below confirm age");
        assert_eq!(next_lifecycle(Tentative, 3, 0, false), Confirmed, "reaches confirm age");
        assert_eq!(next_lifecycle(Confirmed, 5, 0, true), IdLocked, "locked + confirmed");
        assert_eq!(next_lifecycle(Tentative, 2, 0, true), Tentative, "lock waits for confirm");
        assert_eq!(next_lifecycle(Occluded, 6, 0, false), Confirmed, "re-match recovers");
        assert_eq!(next_lifecycle(Lost, 6, 0, true), IdLocked, "even a Lost track recovers on match");
        // Losing the track on misses (missed >= 1).
        assert_eq!(next_lifecycle(Tentative, 1, 1, false), Lost, "unconfirmed miss → Lost");
        assert_eq!(next_lifecycle(Confirmed, 5, 1, false), Occluded, "established miss → Occluded");
        assert_eq!(next_lifecycle(IdLocked, 5, 1, true), Occluded, "locked miss → Occluded too");
        assert_eq!(next_lifecycle(Occluded, 5, LOST_AFTER_MISSED, false), Lost, "long gap → Lost");
        assert_eq!(next_lifecycle(Lost, 5, 40, false), Lost, "Lost stays Lost while missing");
    }

    #[test]
    fn born_track_is_tentative() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.5, 0.5));
        assert_eq!(t.active().next().unwrap().lifecycle, LifecycleState::Tentative);
    }

    #[test]
    fn track_confirms_after_enough_matches() {
        let mut t = PersonTracker::new();
        for _ in 0..(CONFIRM_AGE - 1) {
            t.tick(&det(0.3, 0.3, 0.5, 0.5));
            assert_eq!(t.active().next().unwrap().lifecycle, LifecycleState::Tentative);
        }
        t.tick(&det(0.3, 0.3, 0.5, 0.5)); // age now == CONFIRM_AGE
        assert_eq!(t.active().next().unwrap().lifecycle, LifecycleState::Confirmed);
    }

    #[test]
    fn recognised_user_becomes_id_locked() {
        let mut t = PersonTracker::new();
        t.register_user("Вадим");
        for _ in 0..CONFIRM_AGE {
            t.tick(&det_id(0.3, 0.3, 0.7, 0.7, "Вадим", 0.9));
        }
        let a = t.active().next().unwrap();
        assert_eq!(a.display_id, 0, "holds the reserved id");
        assert_eq!(a.lifecycle, LifecycleState::IdLocked);
        assert!(a.lifecycle.is_established());
    }

    #[test]
    fn confirmed_track_occludes_then_lost() {
        let mut t = PersonTracker::new();
        for _ in 0..CONFIRM_AGE { t.tick(&det(0.3, 0.3, 0.5, 0.5)); }
        assert_eq!(t.all().values().next().unwrap().lifecycle, LifecycleState::Confirmed);
        t.tick(&[]); // one miss
        assert_eq!(t.all().values().next().unwrap().lifecycle, LifecycleState::Occluded);
        for _ in 0..LOST_AFTER_MISSED { t.tick(&[]); } // long gap
        assert_eq!(t.all().values().next().unwrap().lifecycle, LifecycleState::Lost);
    }

    #[test]
    fn tentative_track_lost_on_first_miss() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.5, 0.5)); // Tentative (age 1)
        t.tick(&[]); // miss before confirming → Lost
        assert_eq!(t.all().values().next().unwrap().lifecycle, LifecycleState::Lost);
    }

    // ── ByteTrack two-stage (phase 9.4) ─────────────────────────────────────

    fn bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { x1, y1, x2, y2 }
    }

    // ── inverted contract: associate / apply / finalize (phase 9.5) ─────────

    #[test]
    fn associate_creates_bare_track_without_enrichment() {
        // Lean association spawns a track on geometry alone — no attributes.
        let mut t = PersonTracker::new();
        let ids = t.associate(&[BodyObservation::bare(bbox(0.30, 0.30, 0.50, 0.50))], &[]);
        assert_eq!(ids.len(), 1, "one id per observation");
        let tr = t.all().get(&ids[0]).unwrap();
        assert!((tr.bbox.x1 - 0.30).abs() < 1e-6, "geometry associated");
        assert!(tr.identity.is_none(), "association injects no identity");
        assert!(tr.emotion.is_none(), "association injects no emotion");
        assert!(tr.posture.is_none(), "association injects no posture");
    }

    #[test]
    fn apply_detection_seeds_a_bare_track() {
        // The post-association half: enrichment lands on the associated id.
        let mut t = PersonTracker::new();
        let ids = t.associate(&[BodyObservation::bare(bbox(0.30, 0.30, 0.50, 0.50))], &[]);
        let det = det_id(0.30, 0.30, 0.50, 0.50, "Вадим", 0.9).remove(0);
        t.apply_detection(ids[0], &det);
        t.finalize();
        let tr = t.all().get(&ids[0]).unwrap();
        assert_eq!(tr.identity.as_deref(), Some("Вадим"), "enrichment seeded post-match");
    }

    #[test]
    fn associate_re_matches_existing_track_on_geometry() {
        // A second overlapping observation re-associates to the same id (no churn).
        let mut t = PersonTracker::new();
        let id0 = t.associate(&[BodyObservation::bare(bbox(0.30, 0.30, 0.50, 0.50))], &[])[0];
        let id1 = t.associate(&[BodyObservation::bare(bbox(0.31, 0.31, 0.51, 0.51))], &[])[0];
        assert_eq!(id0, id1, "overlapping observation keeps the id");
        assert_eq!(t.track_count(), 1);
    }

    #[test]
    fn associate_runs_low_conf_recovery() {
        // The inverted entry honours ByteTrack: a weak box recovers a track even
        // when no high observation arrives, and high-only ids are returned.
        let mut t = PersonTracker::new();
        let id0 = t.associate(&[BodyObservation::bare(bbox(0.30, 0.30, 0.50, 0.50))], &[])[0];
        let ids = t.associate(&[], &[bbox(0.31, 0.31, 0.51, 0.51)]);
        assert!(ids.is_empty(), "no high observation → no ids");
        assert_eq!(t.active().next().unwrap().id, id0, "low box recovered the track");
    }

    #[test]
    fn inverted_flow_equals_tick_byte_for_one_person() {
        // THE contract: associate → apply_detection → finalize must produce the
        // exact same track state as the integrated tick_byte on one person.
        let det = Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: Some("Neutral".into()),
            emotion_confidence: Some(0.65),
            posture: Some("сидит".into()),
            posture_confidence: Some(0.95),
            distance: Some(1.5), position_x: Some(0.1), position_y: Some(0.2), position_z: Some(1.5),
            ws_confidence: Some(0.8), recon_error: Some(0.01), plausibility: Some("normal".into()),
            body_coverage: None, torso_attitude: None,
            head_yaw: None, head_facing: None,
            emotion_distribution: None, embedding: Some(vec![1.0, 0.0, 0.0]),
        };

        let mut integrated = PersonTracker::new();
        integrated.tick_byte(std::slice::from_ref(&det), &[]);

        let mut inverted = PersonTracker::new();
        let ids = inverted.associate(&[BodyObservation::bare(det.bbox)], &[]);
        inverted.apply_detection(ids[0], &det);
        inverted.finalize();

        let a = integrated.active().next().unwrap();
        let b = inverted.active().next().unwrap();
        assert_eq!(a.identity, b.identity);
        assert_eq!(a.identity_confidence, b.identity_confidence);
        assert_eq!(a.emotion, b.emotion);
        assert_eq!(a.posture, b.posture);
        assert_eq!(a.age, b.age);
        assert_eq!(a.lifecycle, b.lifecycle);
        assert_eq!(a.distance, b.distance);
        assert_eq!(a.embedding, b.embedding);
        assert!((a.bbox.x1 - b.bbox.x1).abs() < 1e-6);
    }

    #[test]
    fn low_conf_recovers_unmatched_track() {
        // High stage finds nothing this frame, but a weak bbox overlapping the
        // track's prediction rescues it (missed cleared) instead of dropping it.
        let mut t = PersonTracker::new();
        t.tick(&det(0.30, 0.30, 0.50, 0.50));
        let id0 = t.active().next().unwrap().id;
        t.tick_byte(&[], &[bbox(0.31, 0.31, 0.51, 0.51)]);
        assert_eq!(t.track_count(), 1);
        let a = t.active().next().expect("track recovered, still active");
        assert_eq!(a.id, id0, "recovery keeps the id");
        assert_eq!(a.age, 2, "bbox-only recovery still ages the track");
    }

    #[test]
    fn low_conf_never_spawns_a_track() {
        // A weak box with no track to rescue is discarded — never a new id.
        let mut t = PersonTracker::new();
        t.tick_byte(&[], &[bbox(0.30, 0.30, 0.50, 0.50)]);
        assert_eq!(t.track_count(), 0, "low-conf alone must not create a track");
    }

    #[test]
    fn low_conf_does_not_touch_an_already_matched_track() {
        // The track is matched by a high detection this frame; an overlapping low
        // box must NOT double-update it (it already sits at missed == 0).
        let mut t = PersonTracker::new();
        t.tick(&det(0.30, 0.30, 0.50, 0.50));
        t.tick_byte(&det(0.30, 0.30, 0.50, 0.50), &[bbox(0.30, 0.30, 0.50, 0.50)]);
        assert_eq!(t.active().next().unwrap().age, 2, "aged once, not twice");
    }

    #[test]
    fn low_conf_recovery_preserves_identity() {
        // A bbox-only rescue injects no attributes, so the track's existing
        // identity survives the weak frame untouched.
        let mut t = PersonTracker::new();
        t.tick(&det_id(0.30, 0.30, 0.50, 0.50, "Вадим", 0.9));
        t.tick_byte(&[], &[bbox(0.31, 0.31, 0.51, 0.51)]);
        let a = t.active().next().expect("recovered");
        assert_eq!(a.identity.as_deref(), Some("Вадим"), "identity preserved through recovery");
    }

    #[test]
    fn low_conf_keeps_confirmed_track_alive_across_weak_frame() {
        // ByteTrack's headline: a Confirmed body the detector only half-saw stays
        // Confirmed via the low box, instead of slipping to Occluded.
        let mut t = PersonTracker::new();
        for _ in 0..CONFIRM_AGE { t.tick(&det(0.30, 0.30, 0.50, 0.50)); }
        assert_eq!(t.active().next().unwrap().lifecycle, LifecycleState::Confirmed);
        t.tick_byte(&[], &[bbox(0.31, 0.31, 0.51, 0.51)]);
        assert_eq!(
            t.active().next().unwrap().lifecycle,
            LifecycleState::Confirmed,
            "low-conf recovery keeps it Confirmed, not Occluded"
        );
    }

    // ── identity lock + cadence (phase 9.6) ─────────────────────────────────

    #[test]
    fn id_locks_after_stable_embeddings() {
        // The same face seen ID_LOCK_FRAMES times locks the identity → IdLocked.
        let mut t = PersonTracker::new();
        for _ in 0..crate::types::ID_LOCK_FRAMES {
            t.tick(&det_emb(0.3, 0.3, 0.5, 0.5, "Вадим", vec![1.0, 0.0, 0.0]));
        }
        let tr = t.active().next().unwrap();
        assert!(tr.is_id_locked(), "stable face should lock");
        assert_eq!(tr.lifecycle, LifecycleState::IdLocked);
    }

    #[test]
    fn id_lock_not_reached_before_threshold() {
        // One sighting is not enough to lock.
        let mut t = PersonTracker::new();
        t.tick(&det_emb(0.3, 0.3, 0.5, 0.5, "Вадим", vec![1.0, 0.0, 0.0]));
        let tr = t.active().next().unwrap();
        assert_eq!(tr.id_stable_count, 1);
        assert!(!tr.is_id_locked(), "single observation must not lock");
    }

    #[test]
    fn id_lock_resets_on_different_face() {
        // A locked track that suddenly sees a different face un-locks (re-check on
        // doubt) so SFace re-engages.
        let mut t = PersonTracker::new();
        for _ in 0..crate::types::ID_LOCK_FRAMES {
            t.tick(&det_emb(0.3, 0.3, 0.5, 0.5, "Вадим", vec![1.0, 0.0, 0.0]));
        }
        assert!(t.active().next().unwrap().is_id_locked());
        t.tick(&det_emb(0.3, 0.3, 0.5, 0.5, "Вадим", vec![0.0, 1.0, 0.0])); // orthogonal
        let tr = t.active().next().unwrap();
        assert_eq!(tr.id_stable_count, 1, "different face restarts the lock");
        assert!(!tr.is_id_locked(), "no longer locked → SFace re-engages");
    }

    #[test]
    fn reserved_user_is_locked_without_embeddings() {
        // A recognised gallery user is locked by virtue of its reserved id, even
        // with no face vectors at all.
        let mut t = PersonTracker::new();
        t.register_user("Вадим");
        t.tick(&det_id(0.3, 0.3, 0.7, 0.7, "Вадим", 0.9));
        let tr = t.active().next().unwrap();
        assert_eq!(tr.id_stable_count, 0, "no embeddings");
        assert!(tr.is_id_locked(), "reserved user is locked regardless");
    }

    // ── emotion cadence (phase 9.7) ─────────────────────────────────────────

    #[test]
    fn fresh_track_needs_emotion() {
        // A brand-new body has never been emotion-classified → it needs it.
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        assert!(t.active().next().unwrap().needs_emotion(), "first sight needs emotion");
    }

    #[test]
    fn emotion_observation_resets_cadence() {
        // An emotion observation zeroes the cadence age → coasts (no re-run yet).
        let mut t = PersonTracker::new();
        t.tick(&det_with_emotion("Happiness", 0.9));
        let tr = t.active().next().unwrap();
        assert_eq!(tr.frames_since_emotion, 0);
        assert!(!tr.needs_emotion(), "just classified → coast");
    }

    #[test]
    fn emotion_cadence_ages_until_stale() {
        // After EMOTION_CADENCE frames without an emotion observation, the track
        // is stale and asks to be re-classified.
        let mut t = PersonTracker::new();
        t.tick(&det_with_emotion("Happiness", 0.9)); // resets to 0
        assert!(!t.active().next().unwrap().needs_emotion());
        for _ in 0..crate::types::EMOTION_CADENCE {
            t.tick(&det(0.3, 0.3, 0.7, 0.7)); // matched, but no emotion observed
        }
        assert!(t.active().next().unwrap().needs_emotion(), "coasted past cadence → stale");
    }

    #[test]
    fn emotion_label_frozen_while_coasting() {
        // The headline "не дёргается": skipping emotion frames keeps the last
        // fused label/confidence intact (a matched track does not decay).
        let mut t = PersonTracker::new();
        t.tick(&det_with_emotion("Happiness", 0.9));
        for _ in 0..2 {
            t.tick(&det(0.3, 0.3, 0.7, 0.7)); // no emotion observed
        }
        let tr = t.active().next().unwrap();
        assert_eq!(tr.emotion.as_deref(), Some("Happiness"), "label frozen, not dropped");
    }

    #[test]
    fn lifecycle_as_str_round_trips() {
        assert_eq!(LifecycleState::Tentative.as_str(), "tentative");
        assert_eq!(LifecycleState::Confirmed.as_str(), "confirmed");
        assert_eq!(LifecycleState::IdLocked.as_str(), "id_locked");
        assert_eq!(LifecycleState::Occluded.as_str(), "occluded");
        assert_eq!(LifecycleState::Lost.as_str(), "lost");
        assert!(!LifecycleState::Tentative.is_established());
        assert!(LifecycleState::Confirmed.is_established());
        assert!(LifecycleState::IdLocked.is_established());
        assert!(!LifecycleState::Occluded.is_established());
    }
}
