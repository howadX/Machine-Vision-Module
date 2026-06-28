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
use crate::types::{BBox, Detection, PersonTrack, FIRST_EPHEMERAL_ID};
use crate::kalman::KalmanBBox;
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
}

/// Cosine similarity of two L2-normalised embeddings (== dot product).
/// Returns 0.0 on length mismatch (defensive; embeddings are fixed-dim).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return 0.0; }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

    /// Process one frame's detections. Matches new detections to existing
    /// tracks via IoU on the **Kalman-predicted** bbox (State-Gating),
    /// propagates metadata, expires stale tracks.
    pub fn tick(&mut self, detections: &[Detection]) {
        // ── Step 1: predict all tracks (State-Gating) ──────────────────────
        // KF is now in predicted state; predicted[id] = bbox to use for IoU.
        let predicted: HashMap<u32, BBox> = self.tracks.iter_mut()
            .map(|(&id, t)| (id, t.kalman.predict()))
            .collect();

        // ── Step 2: missed-frame bookkeeping + emotion decay ───────────────
        for t in self.tracks.values_mut() {
            t.missed += 1;
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

        // Pre-capture config values (Copy) to avoid borrow conflicts inside the loop.
        let pinhole_opt = self.pinhole;
        let face_height_m = self.face_height_m;
        let depth_ema_alpha = self.depth_ema_alpha;

        // ── Step 3: match detections to predicted bboxes ───────────────────
        let mut matched_ids: Vec<u32> = Vec::new();

        for det in detections {
            let mut best_iou = 0.3_f32;
            let mut best_id = None;

            for (&id, _) in &self.tracks {
                if matched_ids.contains(&id) { continue; }
                // Tracks created earlier in this same tick have no prediction entry.
                let Some(&pred_bbox) = predicted.get(&id) else { continue; };
                let iou = det.bbox.iou(&pred_bbox);
                if iou > best_iou {
                    best_iou = iou;
                    best_id = Some(id);
                }
            }

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

            if let Some(id) = best_id {
                // ── Part A: Kalman update + metadata ───────────────────────
                {
                    let t = self.tracks.get_mut(&id).unwrap();
                    t.kalman.update(det.bbox);
                    t.bbox = t.kalman.smoothed_bbox();
                    t.age += 1;
                    t.missed = 0;

                    // Keep the latest embedding so cosine re-link compares against
                    // a fresh face vector after a gap.
                    if det.embedding.is_some() {
                        t.embedding = det.embedding.clone();
                    }
                    // Propagate metadata: keep best identity, latest emotion/posture
                    if let (Some(name), Some(conf)) = (&det.identity, det.identity_confidence) {
                        if t.identity_confidence.map_or(true, |c| conf > c) {
                            t.identity = Some(name.clone());
                            t.identity_confidence = Some(conf);
                        }
                    }
                    // Emotion: temporal Product-of-Experts fusion (phase 6.4b).
                    // When the detection carries a full distribution, fold it into
                    // the track's accumulated prior with forgetting, then re-derive
                    // the label from the SMOOTHED distribution (kills per-frame
                    // argmax jitter). Without a distribution (Python-only signal),
                    // fall back to latest-snapshot.
                    if let Some(obs) = det.emotion_distribution {
                        let fused = match t.emotion_distribution {
                            Some(prior) => crate::emotion::fuse_emotion_temporal(
                                &prior, &obs, self.emotion_forget_lambda,
                            ),
                            None => obs, // first observation seeds the prior
                        };
                        let (label, conf) = crate::emotion::emotion_label(&fused);
                        t.emotion_distribution = Some(fused);
                        t.emotion = Some(label.to_string());
                        t.emotion_confidence = Some(conf);
                    } else if det.emotion.is_some() {
                        t.emotion = det.emotion.clone();
                        t.emotion_confidence = det.emotion_confidence;
                    }
                    if det.posture.is_some() {
                        t.posture = det.posture.clone();
                        t.posture_confidence = det.posture_confidence;
                    }
                    // Python 3D passthrough (overridden by native below when pinhole is set)
                    if det.distance.is_some() {
                        t.distance = det.distance;
                        t.position_x = det.position_x;
                        t.position_y = det.position_y;
                        t.position_z = det.position_z;
                        t.ws_confidence = det.ws_confidence;
                    }
                    if det.recon_error.is_some() { t.recon_error = det.recon_error; }
                    if det.plausibility.is_some() { t.plausibility = det.plausibility.clone(); }
                } // t drops here, releasing mutable borrow

                // ── Part B: native 3D (phase 6.2) ──────────────────────────
                // Two-step: immutable lookup first, then mutable write.
                if let Some(ph) = pinhole_opt {
                    // Step 1: read identity, look up personal height (immutable borrow)
                    let real_h = {
                        let track = self.tracks.get(&id).unwrap();
                        track.identity.as_ref()
                            .and_then(|n| self.personal_heights.get(n.as_str()))
                            .copied()
                            .unwrap_or(face_height_m)
                    }; // immutable borrow drops

                    // Step 2: compute and write (mutable borrow)
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
                        let pos_s = ph.pixel_to_3d(
                            bbox_cx * ph.width,
                            bbox_cy * ph.height,
                            smoothed_z,
                        );
                        t.distance = Some(smoothed_z);
                        t.position_x = Some(pos_s[0]);
                        t.position_y = Some(pos_s[1]);
                        t.position_z = Some(pos_s[2]);
                        t.ws_confidence = Some(ws_conf);
                    }
                }

                matched_ids.push(id);
            } else {
                // New track — KF initialized at first observation.
                // Run native 3D immediately so pos is available from frame 1.
                let (nat_dist, nat_x, nat_y, nat_z, nat_ws, nat_ema) =
                    if let Some(ph) = pinhole_opt {
                        let b = det.bbox;
                        let bbox_cx = (b.x1 + b.x2) * 0.5;
                        let bbox_cy = (b.y1 + b.y2) * 0.5;
                        if let Some((_pos, z, ws_conf)) =
                            face_bbox_to_3d(&ph, b.y1, b.y2, bbox_cx, bbox_cy, face_height_m)
                        {
                            let pos_s = ph.pixel_to_3d(
                                bbox_cx * ph.width, bbox_cy * ph.height, z,
                            );
                            let ema = DepthEma::new(z, depth_ema_alpha);
                            (Some(z), Some(pos_s[0]), Some(pos_s[1]), Some(pos_s[2]),
                             Some(ws_conf), Some(ema))
                        } else {
                            (None, None, None, None, None, None)
                        }
                    } else {
                        (None, None, None, None, None, None)
                    };

                let id = self.alloc_id();
                // Seed emotion from the distribution when present, so the label is
                // consistent with the vector the temporal fusion will accumulate.
                let (seed_emotion, seed_emotion_conf) = match det.emotion_distribution {
                    Some(dist) => {
                        let (label, conf) = crate::emotion::emotion_label(&dist);
                        (Some(label.to_string()), Some(conf))
                    }
                    None => (det.emotion.clone(), det.emotion_confidence),
                };
                self.tracks.insert(id, PersonTrack {
                    id,
                    display_id: id, // resolved at end of tick
                    bbox: det.bbox,
                    kalman: KalmanBBox::new(det.bbox),
                    age: 1,
                    missed: 0,
                    identity: det.identity.clone(),
                    identity_confidence: det.identity_confidence,
                    emotion: seed_emotion,
                    emotion_confidence: seed_emotion_conf,
                    posture: det.posture.clone(),
                    posture_confidence: det.posture_confidence,
                    // Native 3D takes priority; Python bridge value as fallback.
                    distance: nat_dist.or(det.distance),
                    position_x: nat_x.or(det.position_x),
                    position_y: nat_y.or(det.position_y),
                    position_z: nat_z.or(det.position_z),
                    ws_confidence: nat_ws.or(det.ws_confidence),
                    recon_error: det.recon_error,
                    plausibility: det.plausibility.clone(),
                    depth_ema: nat_ema,
                    emotion_distribution: det.emotion_distribution,
                    embedding: det.embedding.clone(),
                });
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

        self.resolve_display_ids();
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
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
            embedding: None,
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
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
            embedding: None,
        }];
        t.tick(&enriched);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].identity.as_deref(), Some("Вадим"));
        assert_eq!(active[0].emotion.as_deref(), Some("Neutral"));
        assert_eq!(active[0].posture.as_deref(), Some("сидит"));
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
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
            embedding: None,
        }]);
        t.tick(&[Detection {
            bbox: BBox { x1: 0.31, y1: 0.31, x2: 0.71, y2: 0.71 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
            embedding: None,
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
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
            embedding: None,
        }]);
        t.tick(&[Detection {
            bbox: BBox { x1: 0.32, y1: 0.31, x2: 0.52, y2: 0.51 },
            emotion: Some("Happiness".into()), emotion_confidence: Some(0.8),
            identity: None, identity_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
            embedding: None,
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
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
            embedding: None,
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
}
