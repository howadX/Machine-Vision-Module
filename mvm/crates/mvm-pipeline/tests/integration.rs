//! Integration tests: Python bridge JSON → Camera::parse_line → PersonTracker.
//!
//! Simulates multi-frame bridge output without spawning a real process.
//! Each test drives `parse_line` + `tracker.tick` and asserts tracker state.

use mvm_core::PersonTracker;
use mvm_pipeline::Camera;

// ── helpers ──────────────────────────────────────────────────────────────────

fn frame(frame_no: u64, detections_json: &str) -> String {
    format!(
        r#"{{"frame":{frame_no},"elapsed":0.033,"detections":[{detections_json}]}}"#
    )
}

fn det(x1: f32, y1: f32, x2: f32, y2: f32) -> String {
    format!(r#"{{"x1":{x1},"y1":{y1},"x2":{x2},"y2":{y2}}}"#)
}

fn det_enriched(x1: f32, y1: f32, x2: f32, y2: f32,
                identity: &str, id_conf: f32,
                emotion: &str, em_conf: f32) -> String {
    format!(
        r#"{{"x1":{x1},"y1":{y1},"x2":{x2},"y2":{y2},"identity":"{identity}","identity_confidence":{id_conf},"emotion":"{emotion}","emotion_confidence":{em_conf}}}"#
    )
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// One person visible for three consecutive frames → age == 3, one track.
#[test]
fn single_person_three_frames_yields_one_track_age_three() {
    let mut tracker = PersonTracker::new();

    for i in 1..=3u64 {
        let line = frame(i, &det(0.3, 0.3, 0.7, 0.7));
        let dets = Camera::parse_line(&line).unwrap();
        tracker.tick(&dets);
    }

    let active: Vec<_> = tracker.active().collect();
    assert_eq!(active.len(), 1, "expected one active track");
    assert_eq!(active[0].age, 3);
    assert_eq!(active[0].missed, 0);
}

/// Identity seen on frame 2 (higher conf) wins over frame 1 (lower conf).
#[test]
fn identity_best_confidence_propagated_across_frames() {
    let mut tracker = PersonTracker::new();

    let f1 = frame(1, &det_enriched(0.3, 0.3, 0.7, 0.7, "unknown", 0.25, "Neutral", 0.6));
    let f2 = frame(2, &det_enriched(0.32, 0.31, 0.72, 0.71, "Вадим", 0.78, "Neutral", 0.7));

    for line in [&f1, &f2] {
        let dets = Camera::parse_line(line).unwrap();
        tracker.tick(&dets);
    }

    let active: Vec<_> = tracker.active().collect();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].identity.as_deref(), Some("Вадим"));
    assert!((active[0].identity_confidence.unwrap() - 0.78).abs() < 0.01);
}

/// Two non-overlapping faces in one frame → two independent tracks.
#[test]
fn two_faces_in_one_frame_produce_two_tracks() {
    let mut tracker = PersonTracker::new();

    let two = format!("{},{}", det(0.05, 0.1, 0.25, 0.5), det(0.7, 0.1, 0.9, 0.5));
    let line = frame(1, &two);
    let dets = Camera::parse_line(&line).unwrap();
    tracker.tick(&dets);

    assert_eq!(tracker.track_count(), 2);
    let active: Vec<_> = tracker.active().collect();
    assert_eq!(active.len(), 2);
}

/// Person disappears: missed counter rises, emotion decays, then track expires.
#[test]
fn person_disappears_emotion_decays_then_track_expires() {
    let mut tracker = PersonTracker::new();
    tracker.decay_start_frames = 0;
    tracker.decay_factor = 0.5;
    tracker.timeout_frames = 10;

    // Appear with emotion
    let appear = frame(1, &det_enriched(0.3, 0.3, 0.7, 0.7, "Вадим", 0.75, "Happiness", 0.9));
    tracker.tick(&Camera::parse_line(&appear).unwrap());

    // 5 empty frames → emotion decays  (0.9 * 0.5^5 = 0.028 < 0.05 → cleared)
    for i in 2..=6u64 {
        let line = frame(i, "");
        tracker.tick(&Camera::parse_line(&line).unwrap());
    }
    {
        let tracks: Vec<_> = tracker.all().values().collect();
        assert_eq!(tracks.len(), 1, "track still alive during timeout");
        assert!(tracks[0].emotion.is_none(), "emotion should have decayed to None");
        assert_eq!(tracks[0].missed, 5);
    }

    // Remaining empty frames → track expires at timeout_frames
    for i in 7..=11u64 {
        let line = frame(i, "");
        tracker.tick(&Camera::parse_line(&line).unwrap());
    }
    assert_eq!(tracker.track_count(), 0, "stale track should have expired");
}

/// Person leaves and comes back beyond timeout → new track ID (not the same track).
#[test]
fn person_reappears_after_timeout_gets_new_track_id() {
    let mut tracker = PersonTracker::new();
    tracker.timeout_frames = 3;

    let appear = frame(1, &det(0.3, 0.3, 0.7, 0.7));
    tracker.tick(&Camera::parse_line(&appear).unwrap());
    let first_id = tracker.active().next().unwrap().id;

    // Expire the track
    for i in 2..=5u64 {
        tracker.tick(&Camera::parse_line(&frame(i, "")).unwrap());
    }
    assert_eq!(tracker.track_count(), 0, "track should have expired");

    // Person reappears
    tracker.tick(&Camera::parse_line(&frame(6, &det(0.3, 0.3, 0.7, 0.7))).unwrap());
    let active: Vec<_> = tracker.active().collect();
    assert_eq!(active.len(), 1);
    assert_ne!(active[0].id, first_id, "reappeared person should get a new track id");
}

/// Empty frames only → tracker stays empty throughout.
#[test]
fn empty_frames_produce_no_tracks() {
    let mut tracker = PersonTracker::new();

    for i in 1..=20u64 {
        let line = frame(i, "");
        tracker.tick(&Camera::parse_line(&line).unwrap());
    }
    assert_eq!(tracker.track_count(), 0);
}

/// 3D workspace fields (distance, position, ws_confidence) round-trip through parse + tracker.
#[test]
fn workspace_3d_fields_propagate_through_tracker() {
    let mut tracker = PersonTracker::new();

    let line = frame(1, r#"{"x1":0.3,"y1":0.3,"x2":0.7,"y2":0.7,"distance":2.5,"position_x":0.1,"position_y":0.0,"position_z":2.5,"ws_confidence":0.88}"#);
    tracker.tick(&Camera::parse_line(&line).unwrap());

    let active: Vec<_> = tracker.active().collect();
    assert!((active[0].distance.unwrap() - 2.5).abs() < 0.01);
    assert!((active[0].ws_confidence.unwrap() - 0.88).abs() < 0.01);
}

/// Malformed JSON returns an Err (not a panic).
#[test]
fn malformed_json_returns_error() {
    let result = Camera::parse_line(r#"{"frame":1,"elapsed":0.0,BROKEN"#);
    assert!(result.is_err());
}

/// Out-of-range coordinates are clamped by parse_line before reaching the tracker.
#[test]
fn oob_coordinates_are_clamped_before_tracker() {
    let mut tracker = PersonTracker::new();

    let line = frame(1, r#"{"x1":-1.0,"y1":-0.5,"x2":2.0,"y2":1.8}"#);
    let dets = Camera::parse_line(&line).unwrap();
    assert_eq!(dets[0].bbox.x1, 0.0);
    assert_eq!(dets[0].bbox.x2, 1.0);

    tracker.tick(&dets);
    assert_eq!(tracker.track_count(), 1);
}
