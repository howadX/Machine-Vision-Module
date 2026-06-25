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

use std::collections::HashMap;
use crate::types::{BBox, Detection, PersonTrack};
use crate::kalman::KalmanBBox;
use crate::space::{Pinhole, DepthEma, face_bbox_to_3d};

/// Multi-person tracker — the temporal fusion layer of MVM.
pub struct PersonTracker {
    next_id: u32,
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
}

impl PersonTracker {
    /// Create tracker without 3D capability (tests, headless use).
    pub fn new() -> Self {
        Self {
            next_id: 0,
            tracks: HashMap::new(),
            timeout_frames: 1800,
            decay_start_frames: 5,
            decay_factor: 0.8,
            pinhole: None,
            face_height_m: 0.44, // empirical effective bbox-face height (see config.rs)
            depth_ema_alpha: 0.3,
            personal_heights: HashMap::new(),
        }
    }

    /// Create tracker with native 3D enabled (pinhole derived from config).
    pub fn with_config(cfg: &crate::config::Config) -> Self {
        Self {
            pinhole: Some(Pinhole::from_config(cfg)),
            face_height_m: cfg.face_height_m,
            depth_ema_alpha: cfg.depth_ema_alpha,
            ..Self::new()
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

            if let Some(id) = best_id {
                // ── Part A: Kalman update + metadata ───────────────────────
                {
                    let t = self.tracks.get_mut(&id).unwrap();
                    t.kalman.update(det.bbox);
                    t.bbox = t.kalman.smoothed_bbox();
                    t.age += 1;
                    t.missed = 0;

                    // Propagate metadata: keep best identity, latest emotion/posture
                    if let (Some(name), Some(conf)) = (&det.identity, det.identity_confidence) {
                        if t.identity_confidence.map_or(true, |c| conf > c) {
                            t.identity = Some(name.clone());
                            t.identity_confidence = Some(conf);
                        }
                    }
                    if det.emotion.is_some() {
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
                    if det.emotion_distribution.is_some() { t.emotion_distribution = det.emotion_distribution; }
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
                        if let Some((pos, z, ws_conf)) =
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

                let id = self.next_id;
                self.next_id += 1;
                self.tracks.insert(id, PersonTrack {
                    id,
                    bbox: det.bbox,
                    kalman: KalmanBBox::new(det.bbox),
                    age: 1,
                    missed: 0,
                    identity: det.identity.clone(),
                    identity_confidence: det.identity_confidence,
                    emotion: det.emotion.clone(),
                    emotion_confidence: det.emotion_confidence,
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

        self.tracks.retain(|_, t| t.missed < self.timeout_frames);
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

    #[test]
    fn tracker_creates_new_track_on_first_detection() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        assert_eq!(t.track_count(), 1);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].id, 0);
        assert_eq!(active[0].age, 1);
    }

    #[test]
    fn tracker_propagates_identity() {
        let mut t = PersonTracker::new();
        let enriched = vec![Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: Some("Нейтральная".into()),
            emotion_confidence: Some(0.65),
            posture: Some("сидит".into()),
            posture_confidence: Some(0.95),
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
        }];
        t.tick(&enriched);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].identity.as_deref(), Some("Вадим"));
        assert_eq!(active[0].emotion.as_deref(), Some("Нейтральная"));
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
        }]);
        t.tick(&[Detection {
            bbox: BBox { x1: 0.31, y1: 0.31, x2: 0.71, y2: 0.71 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
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
            emotion: Some("Злость".into()), emotion_confidence: Some(0.6),
            identity: None, identity_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
        }]);
        t.tick(&[Detection {
            bbox: BBox { x1: 0.32, y1: 0.31, x2: 0.52, y2: 0.51 },
            emotion: Some("Радость".into()), emotion_confidence: Some(0.8),
            identity: None, identity_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None, emotion_distribution: None,
        }]);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].emotion.as_deref(), Some("Радость"));
    }

    #[test]
    fn tracker_matches_by_iou() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        t.tick(&det(0.32, 0.31, 0.72, 0.71));
        assert_eq!(t.track_count(), 1);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].id, 0);
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
        assert_eq!(active[0].id, 1);
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
        }]
    }

    #[test]
    fn emotion_confidence_decays_after_missed_frames() {
        let mut t = PersonTracker::new();
        t.decay_start_frames = 3;
        t.decay_factor = 0.8;
        t.tick(&det_with_emotion("Радость", 0.9));
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
        t.tick(&det_with_emotion("Грусть", 0.8));
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
        t.tick(&det_with_emotion("Грусть", 0.8));
        for _ in 0..10 { t.tick(&[]); }
        let tracks: Vec<_> = t.all().values().collect();
        assert!(tracks[0].emotion.is_none());
        t.tick(&det_with_emotion("Радость", 0.75));
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].emotion.as_deref(), Some("Радость"));
        assert!((active[0].emotion_confidence.unwrap() - 0.75).abs() < 0.01);
    }
}
