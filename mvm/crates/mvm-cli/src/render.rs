//! Snapshot v2 renderer (phase 12.B1) — the "what is happening right now" view.
//!
//! A **deterministic** markdown projection of a [`WorldState`] for LLM consumption:
//! an entity table (only the *conclusions* — distance, posture, motion, emotion,
//! head, hands — never raw coordinates), the current zones, and the active
//! activities with duration. This is a pure function of the world (plus a
//! caller-supplied timestamp so it stays testable), so it golden-tests cleanly and
//! carries no state.
//!
//! Design (research-backed, `09-scene-compiler/00-scene-compiler-research.md` §4):
//! markdown + a table read better than deep JSON; confidence is rendered as a word
//! (`high`/`med`/`low`) with the LLM calibrating on the word, not a bare float.
//!
//! Honesty notes (`⚠️` block — withheld posture, phantom suspicion, plausibility)
//! land in phase 12.B2 on top of this table.

use crate::state::{object_state_to_entity, person_to_entity};
use mvm_core::WorldState;
use std::fmt::Write;

/// Confidence tier as a word (research §4.1: LLMs calibrate on words, not floats).
fn conf_word(c: f32) -> &'static str {
    if c >= 0.75 {
        "high"
    } else if c >= 0.5 {
        "med"
    } else {
        "low"
    }
}

/// Human-readable name for a scene-graph id: `Вадим(0)` / `anon(10)` for people,
/// the class string for objects (id ≥ 10000).
fn entity_name(id: u32, world: &WorldState) -> String {
    if id >= mvm_scene::OBJECT_ID_BASE {
        world
            .objects
            .iter()
            .find(|o| o.id == id)
            .map(|o| o.class.clone())
            .unwrap_or_else(|| format!("obj{id}"))
    } else {
        world
            .people
            .iter()
            .find(|p| p.track_id == id)
            .and_then(|p| p.identity.as_ref().map(|(n, _)| format!("{n}({id})")))
            .unwrap_or_else(|| format!("anon({id})"))
    }
}

/// Render the snapshot markdown for a world frame. `ts` is the caller-formatted
/// timestamp string (kept out of the function so the output is deterministic).
pub fn render_snapshot(world: &WorldState, fps: f64, ts: &str) -> String {
    let status = if world.people.is_empty() { "empty" } else { "occupied" };
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# Scene {ts} · {status} · anomaly {:.2} · {:.0} fps",
        world.anomaly_score, fps
    );

    // Zones (current frame, stateless labels reused from fusion).
    let mut entities: Vec<mvm_core::types::Entity> =
        world.people.iter().map(person_to_entity).collect();
    entities.extend(world.objects.iter().map(object_state_to_entity));
    let scene = mvm_fusion::build_scene(entities.clone());
    for comp in &scene.zones {
        if comp.len() < 2 {
            continue; // a lone node is not a "zone" worth narrating
        }
        let label = mvm_fusion::zone_label(comp, &entities);
        let members: Vec<String> = comp.iter().map(|&id| entity_name(id, world)).collect();
        match label {
            Some(l) => {
                let _ = writeln!(out, "Zone «{l}»: {}", members.join(", "));
            }
            None => {
                let _ = writeln!(out, "Zone: {}", members.join(", "));
            }
        }
    }

    if world.people.is_empty() {
        let _ = writeln!(out, "(no people)");
    } else {
        let mut people: Vec<&mvm_core::PersonState> = world.people.iter().collect();
        people.sort_by_key(|p| p.track_id);
        let _ = writeln!(out, "\n| id | who | dist | posture | motion | emotion | head | hands |");
        let _ = writeln!(out, "|----|-----|------|---------|--------|---------|------|-------|");
        for p in people {
            let _ = writeln!(out, "| {} | {} | {} | {} | {} | {} | {} | {} |",
                p.track_id,
                who_cell(p),
                dist_cell(p),
                posture_cell(p),
                motion_cell(p),
                emotion_cell(p),
                p.head_facing.unwrap_or("—"),
                hands_cell(p),
            );
        }
    }

    // Active activities (with duration when the journal is running).
    let acts = activity_lines(world);
    if !acts.is_empty() {
        let _ = writeln!(out, "\nActive: {}", acts.join(" · "));
    }

    // Honesty notes (phase 12.B2): make the frame's uncertainty explicit rather
    // than silent — research §4.4 (LLMs calibrate on stated uncertainty, and the
    // A3 session showed a silent unverified body reads as a confident person).
    let notes = honesty_notes(world);
    if !notes.is_empty() {
        let _ = writeln!(out);
        for n in notes {
            let _ = writeln!(out, "⚠️ {n}");
        }
    }

    out
}

/// Per-frame uncertainty notes — the honest signals fusion already computes
/// (withheld posture, unverified body, OOD pose, unstable activity participants).
/// Deterministic order: people by id, then activities.
fn honesty_notes(world: &WorldState) -> Vec<String> {
    let mut notes = Vec::new();
    let mut people: Vec<&mvm_core::PersonState> = world.people.iter().collect();
    people.sort_by_key(|p| p.track_id);
    for p in &people {
        let face_verified = p.identity.is_some() || p.emotion.is_some();
        if !face_verified {
            notes.push(format!(
                "id{}: тело не подтверждено лицом (животное или упавший — не различить)",
                p.track_id
            ));
        }
        if p.posture.is_none() && matches!(p.body_coverage, Some("upper") | Some("head")) {
            notes.push(format!(
                "id{}: поза withheld (видна только {})",
                p.track_id,
                p.body_coverage.unwrap_or("?")
            ));
        }
        if p.plausibility.as_deref() == Some("low") {
            notes.push(format!("id{}: поза OOD (plausibility low)", p.track_id));
        }
    }
    for i in &world.activity_instances {
        if !i.key.stable_participants() {
            let names: Vec<String> =
                i.key.participants.iter().map(|&id| entity_name(id, world)).collect();
            notes.push(format!(
                "{}({}): участники нестабильны (эфемерный id)",
                i.key.label,
                names.join(",")
            ));
        }
    }
    notes
}

fn who_cell(p: &mvm_core::PersonState) -> String {
    match &p.identity {
        Some((name, _)) => format!("{name} ✓"),
        None if p.emotion.is_some() => "anon ✓face".to_string(),
        None => "anon".to_string(),
    }
}

fn dist_cell(p: &mvm_core::PersonState) -> String {
    match p.position_3d {
        Some(q) => format!("{:.1}m", q.z),
        None => "—".to_string(),
    }
}

fn posture_cell(p: &mvm_core::PersonState) -> String {
    match &p.posture {
        Some((label, c)) => format!("{label} ({})", conf_word(*c)),
        None => "—".to_string(),
    }
}

fn motion_cell(p: &mvm_core::PersonState) -> String {
    match (p.motion_state, p.velocity) {
        (Some("moving"), Some(v)) => format!("moving {v:.1} t/s"),
        (Some(s), _) => s.to_string(),
        (None, _) => "—".to_string(),
    }
}

fn emotion_cell(p: &mvm_core::PersonState) -> String {
    match &p.emotion {
        Some((label, c)) => format!("{label} ({})", conf_word(*c)),
        None => "—".to_string(),
    }
}

fn hands_cell(p: &mvm_core::PersonState) -> String {
    let g: Vec<&str> = p.hands.iter().filter_map(|h| h.gesture).collect();
    if g.is_empty() {
        "—".to_string()
    } else {
        g.join("+")
    }
}

/// One compact line per active activity: `label(names) — Ns (tier)`.
fn activity_lines(world: &WorldState) -> Vec<String> {
    if !world.activity_instances.is_empty() {
        world
            .activity_instances
            .iter()
            .map(|i| {
                let names: Vec<String> =
                    i.key.participants.iter().map(|&id| entity_name(id, world)).collect();
                format!(
                    "{}({}) — {:.1}s ({})",
                    i.key.label,
                    names.join(","),
                    i.last_seen - i.started_at,
                    conf_word(i.confidence)
                )
            })
            .collect()
    } else {
        world
            .activities
            .iter()
            .map(|a| {
                let names: Vec<String> =
                    a.participants.iter().map(|&id| entity_name(id, world)).collect();
                format!("{}({}) ({})", a.label, names.join(","), conf_word(a.confidence))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{BBox, HandState, PersonState, Position3D, WorldState};
    use std::time::Instant;

    fn person(id: u32, bbox: BBox) -> PersonState {
        PersonState {
            track_id: id,
            bbox,
            identity: None,
            emotion: None,
            posture: None,
            position_3d: None,
            emotion_distribution: None,
            wrists: Vec::new(),
            hands: Vec::new(),
            body_coverage: None,
            torso_attitude: None,
            motion_state: None,
            velocity: None,
            left_gesture: None,
            right_gesture: None,
            recon_error: None,
            plausibility: None,
            head_yaw: None,
            head_facing: None,
            last_seen: Instant::now(),
        }
    }

    fn object(id: u32, class: &str, bbox: BBox) -> mvm_core::ObjectState {
        mvm_core::ObjectState {
            id,
            class: class.into(),
            bbox,
            confidence: 0.8,
            age: 5,
            position_3d: Some(Position3D { x: 0.2, y: 0.3, z: 0.9, confidence: 0.8 }),
            mask: None,
            remembered: false,
            last_seen: Instant::now(),
        }
    }

    #[test]
    fn snapshot_empty_room() {
        let world = WorldState::new(Instant::now());
        let md = render_snapshot(&world, 24.0, "17:51:18");
        assert_eq!(
            md,
            "# Scene 17:51:18 · empty · anomaly 0.00 · 24 fps\n(no people)\n"
        );
    }

    #[test]
    fn snapshot_two_people_table_and_zone() {
        let mut world = WorldState::new(Instant::now());
        let mut vadim = person(0, BBox { x1: 0.4, y1: 0.2, x2: 0.6, y2: 0.9 });
        vadim.identity = Some(("Вадим".into(), 0.63));
        vadim.emotion = Some(("Surprise".into(), 0.59));
        vadim.posture = Some(("standing".into(), 0.53));
        vadim.position_3d = Some(Position3D { x: 0.7, y: 0.2, z: 0.97, confidence: 0.8 });
        vadim.motion_state = Some("moving");
        vadim.velocity = Some(0.32);
        vadim.head_facing = Some("camera");
        vadim.hands = vec![
            HandState { landmarks: vec![], presence: 0.9, side: "left", handedness: 0.1, gesture: Some("open") },
            HandState { landmarks: vec![], presence: 0.9, side: "right", handedness: 0.9, gesture: Some("fist") },
        ];
        let mut anon = person(10, BBox { x1: 0.7, y1: 0.2, x2: 0.9, y2: 0.9 });
        anon.emotion = Some(("Neutral".into(), 0.89));
        anon.posture = Some(("standing".into(), 0.62));
        anon.position_3d = Some(Position3D { x: 1.8, y: 0.14, z: 1.17, confidence: 0.8 });
        anon.motion_state = Some("idle");
        anon.head_facing = Some("left");
        world.people.push(vadim);
        world.people.push(anon);
        // A workstation object so the shared zone gets a label.
        world.objects.push(object(mvm_scene::OBJECT_ID_BASE, "mouse", BBox { x1: 0.45, y1: 0.6, x2: 0.55, y2: 0.7 }));

        let md = render_snapshot(&world, 18.0, "17:51:19");
        // Header + status.
        assert!(md.starts_with("# Scene 17:51:19 · occupied · anomaly 0.00 · 18 fps\n"), "got:\n{md}");
        // Identity marks and tiered confidence words (not raw floats).
        assert!(md.contains("| 0 | Вадим ✓ | 1.0m | standing (med) | moving 0.3 t/s | Surprise (med) | camera | open+fist |"), "got:\n{md}");
        assert!(md.contains("| 10 | anon ✓face | 1.2m | standing (med) | idle | Neutral (high) | left | — |"), "got:\n{md}");
        // Zone labelled from the workstation-class object.
        assert!(md.contains("Zone «workstation»:") && md.contains("Вадим(0)") && md.contains("mouse"), "got:\n{md}");
        // No raw coordinates / bbox leaked into the human view.
        assert!(!md.contains("0.97") && !md.contains("bbox"), "raw coords leaked:\n{md}");
    }

    #[test]
    fn snapshot_activity_line_with_duration() {
        use mvm_core::activity::ActivityJournal;
        use mvm_core::Activity;
        let mut world = WorldState::new(Instant::now());
        let mut v = person(0, BBox { x1: 0.4, y1: 0.2, x2: 0.6, y2: 0.9 });
        v.identity = Some(("Вадим".into(), 0.63));
        world.people.push(v);
        world.people.push(person(10, BBox { x1: 0.7, y1: 0.2, x2: 0.9, y2: 0.9 }));
        // Drive the real journal: present every tick from t=0.25 to t=22.75 →
        // confirmed Active with a 22.5 s observed duration.
        let mut j = ActivityJournal::new();
        let raw = vec![Activity {
            label: "people_together".into(),
            confidence: 0.6,
            participants: vec![0, 10],
        }];
        for f in 0..3u64 {
            j.update(&raw, 0.25 + 0.1 * f as f64, f);
        }
        j.update(&raw, 22.75, 3);
        world.activity_instances = j.active().cloned().collect();

        let md = render_snapshot(&world, 18.0, "17:51:40");
        assert!(
            md.contains("Active: people_together(Вадим(0),anon(10)) — 22.5s ("),
            "got:\n{md}"
        );
        // The pair keys on an ephemeral id (10) → honestly unstable.
        assert!(
            md.contains("⚠️ people_together(Вадим(0),anon(10)): участники нестабильны"),
            "got:\n{md}"
        );
    }

    #[test]
    fn snapshot_honesty_notes_for_unverified_and_withheld() {
        let mut world = WorldState::new(Instant::now());
        // A body with neither identity nor emotion → unverified (animal / fallen?).
        let mut ghost = person(12, BBox { x1: 0.5, y1: 0.4, x2: 0.7, y2: 0.9 });
        ghost.position_3d = Some(Position3D { x: 0.9, y: 0.2, z: 0.8, confidence: 0.6 });
        // Seated at desk: only the upper body is visible → posture withheld.
        ghost.body_coverage = Some("upper");
        ghost.plausibility = Some("low".into());
        world.people.push(ghost);
        let md = render_snapshot(&world, 18.0, "17:51:25");
        assert!(md.contains("⚠️ id12: тело не подтверждено лицом"), "got:\n{md}");
        assert!(md.contains("⚠️ id12: поза withheld (видна только upper)"), "got:\n{md}");
        assert!(md.contains("⚠️ id12: поза OOD (plausibility low)"), "got:\n{md}");
    }

    #[test]
    fn snapshot_clean_scene_has_no_notes() {
        let mut world = WorldState::new(Instant::now());
        let mut v = person(0, BBox { x1: 0.4, y1: 0.2, x2: 0.6, y2: 0.9 });
        v.identity = Some(("Вадим".into(), 0.9));
        v.emotion = Some(("Neutral".into(), 0.9));
        v.posture = Some(("standing".into(), 0.9));
        v.body_coverage = Some("full");
        v.plausibility = Some("ok".into());
        v.position_3d = Some(Position3D { x: 0.7, y: 0.2, z: 1.0, confidence: 0.8 });
        world.people.push(v);
        let md = render_snapshot(&world, 18.0, "17:51:19");
        assert!(!md.contains("⚠️"), "clean scene should carry no notes:\n{md}");
    }
}
