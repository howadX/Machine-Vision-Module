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
    let status = if world.people.is_empty() {
        "empty"
    } else {
        "occupied"
    };
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
        let _ = writeln!(
            out,
            "\n| id | who | dist | posture | motion | emotion | head | hands |"
        );
        let _ = writeln!(
            out,
            "|----|-----|------|---------|--------|---------|------|-------|"
        );
        for p in people {
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} |",
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
            let names: Vec<String> = i
                .key
                .participants
                .iter()
                .map(|&id| entity_name(id, world))
                .collect();
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
                let names: Vec<String> = i
                    .key
                    .participants
                    .iter()
                    .map(|&id| entity_name(id, world))
                    .collect();
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
                let names: Vec<String> = a
                    .participants
                    .iter()
                    .map(|&id| entity_name(id, world))
                    .collect();
                format!(
                    "{}({}) ({})",
                    a.label,
                    names.join(","),
                    conf_word(a.confidence)
                )
            })
            .collect()
    }
}

// ── Recap renderer (phase 12.C3) — the "what happened over a period" view ────
// Renders a [`mvm_core::recap::Recap`] (built from the scene + events logs) to
// markdown: people presence, activities rolled up across object-id churn, scene
// objects/zones. Phantom people surface as `⚠️` notes so a brief ephemeral track
// is read as a suspicion, not a third person.

use mvm_core::recap::{PersonSummary, Recap};

/// Display name for a recap person: `Вадим(0)` when recognised, else `id10`.
fn recap_person_name(p: &PersonSummary) -> String {
    match &p.identity {
        Some(name) => format!("{name}({})", p.id),
        None => format!("id{}", p.id),
    }
}

/// Render a session recap to markdown. `title` is a caller-supplied window label
/// (e.g. the ts span) so the output stays deterministic.
pub fn render_recap(recap: &Recap, title: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# Recap {title} — {} frames · anomaly max {:.2}",
        recap.frame_count, recap.anomaly_max
    );

    // People: real presences first, then phantom notes.
    let (phantoms, real): (Vec<&PersonSummary>, Vec<&PersonSummary>) =
        recap.people.iter().partition(|p| p.is_phantom);
    if !real.is_empty() {
        let _ = writeln!(out, "\n## People");
        for p in &real {
            let dist = match (p.dist_min, p.dist_max) {
                (Some(a), Some(b)) if (a - b).abs() > 0.05 => format!(", {a:.1}–{b:.1}m"),
                (Some(a), _) => format!(", {a:.1}m"),
                _ => String::new(),
            };
            let posture = p
                .dominant_posture
                .as_deref()
                .map(|s| format!(", mostly {s}"))
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "- **{}** — {}/{} frames ({:.0}%){dist}{posture}",
                recap_person_name(p),
                p.frames_present,
                recap.frame_count,
                p.presence_pct,
            );
        }
    }

    // Activities, rolled up (object-id churn already collapsed in the recap).
    if !recap.activities.is_empty() {
        let _ = writeln!(out, "\n## Activities (rolled up)");
        for a in &recap.activities {
            let who = a
                .humans
                .iter()
                .map(|&id| {
                    recap
                        .people
                        .iter()
                        .find(|p| p.id == id)
                        .map(recap_person_name)
                        .unwrap_or_else(|| format!("id{id}"))
                })
                .collect::<Vec<_>>()
                .join(",");
            let eps = if a.episode_count > 1 {
                format!(" across {} episodes", a.episode_count)
            } else {
                String::new()
            };
            let stable = if a.all_stable { "" } else { " ⚠unstable" };
            let _ = writeln!(
                out,
                "- {}({}) — {:.1}s total{eps}, longest {:.1}s, conf {}{stable}",
                a.label,
                who,
                a.total_s,
                a.longest_s,
                conf_word(a.peak_confidence),
            );
        }
    }

    // Scene: objects + zones (one line each).
    if !recap.objects.is_empty() {
        let classes: Vec<String> = recap.objects.iter().map(|o| o.class.clone()).collect();
        let _ = writeln!(out, "\n## Scene\nObjects: {}", classes.join(", "));
        if !recap.zones.is_empty() {
            let _ = writeln!(out, "Zones: {}", recap.zones.join(", "));
        }
    }

    // Phantom suspicions last, as explicit notes.
    if !phantoms.is_empty() {
        let _ = writeln!(out);
        for p in &phantoms {
            let _ = writeln!(
                out,
                "⚠️ id{} — вероятный фантом: {} кадра(ов), появился в середине (эфемерный id, нестабилен)",
                p.id, p.frames_present
            );
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{BBox, HandState, PersonState, Position3D, WorldState};
    use std::time::Instant;

    fn person(id: u32, bbox: BBox) -> PersonState {
        PersonState {
            tracker_id: id,
            track_id: id,
            bbox,
            face_bbox: None,
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
            torso_flexion: None,
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
            position_3d: Some(Position3D {
                x: 0.2,
                y: 0.3,
                z: 0.9,
                confidence: 0.8,
            }),
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
        let mut vadim = person(
            0,
            BBox {
                x1: 0.4,
                y1: 0.2,
                x2: 0.6,
                y2: 0.9,
            },
        );
        vadim.identity = Some(("Вадим".into(), 0.63));
        vadim.emotion = Some(("Surprise".into(), 0.59));
        vadim.posture = Some(("standing".into(), 0.53));
        vadim.position_3d = Some(Position3D {
            x: 0.7,
            y: 0.2,
            z: 0.97,
            confidence: 0.8,
        });
        vadim.motion_state = Some("moving");
        vadim.velocity = Some(0.32);
        vadim.head_facing = Some("camera");
        vadim.hands = vec![
            HandState {
                landmarks: vec![],
                presence: 0.9,
                side: "left",
                handedness: 0.1,
                gesture: Some("open"),
            },
            HandState {
                landmarks: vec![],
                presence: 0.9,
                side: "right",
                handedness: 0.9,
                gesture: Some("fist"),
            },
        ];
        let mut anon = person(
            10,
            BBox {
                x1: 0.7,
                y1: 0.2,
                x2: 0.9,
                y2: 0.9,
            },
        );
        anon.emotion = Some(("Neutral".into(), 0.89));
        anon.posture = Some(("standing".into(), 0.62));
        anon.position_3d = Some(Position3D {
            x: 1.8,
            y: 0.14,
            z: 1.17,
            confidence: 0.8,
        });
        anon.motion_state = Some("idle");
        anon.head_facing = Some("left");
        world.people.push(vadim);
        world.people.push(anon);
        // A workstation object so the shared zone gets a label.
        world.objects.push(object(
            mvm_scene::OBJECT_ID_BASE,
            "mouse",
            BBox {
                x1: 0.45,
                y1: 0.6,
                x2: 0.55,
                y2: 0.7,
            },
        ));

        let md = render_snapshot(&world, 18.0, "17:51:19");
        // Header + status.
        assert!(
            md.starts_with("# Scene 17:51:19 · occupied · anomaly 0.00 · 18 fps\n"),
            "got:\n{md}"
        );
        // Identity marks and tiered confidence words (not raw floats).
        assert!(md.contains("| 0 | Вадим ✓ | 1.0m | standing (med) | moving 0.3 t/s | Surprise (med) | camera | open+fist |"), "got:\n{md}");
        assert!(
            md.contains(
                "| 10 | anon ✓face | 1.2m | standing (med) | idle | Neutral (high) | left | — |"
            ),
            "got:\n{md}"
        );
        // Zone labelled from the workstation-class object.
        assert!(
            md.contains("Zone «workstation»:") && md.contains("Вадим(0)") && md.contains("mouse"),
            "got:\n{md}"
        );
        // No raw coordinates / bbox leaked into the human view.
        assert!(
            !md.contains("0.97") && !md.contains("bbox"),
            "raw coords leaked:\n{md}"
        );
    }

    #[test]
    fn snapshot_activity_line_with_duration() {
        use mvm_core::Activity;
        use mvm_core::activity::ActivityJournal;
        let mut world = WorldState::new(Instant::now());
        let mut v = person(
            0,
            BBox {
                x1: 0.4,
                y1: 0.2,
                x2: 0.6,
                y2: 0.9,
            },
        );
        v.identity = Some(("Вадим".into(), 0.63));
        world.people.push(v);
        world.people.push(person(
            10,
            BBox {
                x1: 0.7,
                y1: 0.2,
                x2: 0.9,
                y2: 0.9,
            },
        ));
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
        let mut ghost = person(
            12,
            BBox {
                x1: 0.5,
                y1: 0.4,
                x2: 0.7,
                y2: 0.9,
            },
        );
        ghost.position_3d = Some(Position3D {
            x: 0.9,
            y: 0.2,
            z: 0.8,
            confidence: 0.6,
        });
        // Seated at desk: only the upper body is visible → posture withheld.
        ghost.body_coverage = Some("upper");
        ghost.plausibility = Some("low".into());
        world.people.push(ghost);
        let md = render_snapshot(&world, 18.0, "17:51:25");
        assert!(
            md.contains("⚠️ id12: тело не подтверждено лицом"),
            "got:\n{md}"
        );
        assert!(
            md.contains("⚠️ id12: поза withheld (видна только upper)"),
            "got:\n{md}"
        );
        assert!(
            md.contains("⚠️ id12: поза OOD (plausibility low)"),
            "got:\n{md}"
        );
    }

    #[test]
    fn snapshot_clean_scene_has_no_notes() {
        let mut world = WorldState::new(Instant::now());
        let mut v = person(
            0,
            BBox {
                x1: 0.4,
                y1: 0.2,
                x2: 0.6,
                y2: 0.9,
            },
        );
        v.identity = Some(("Вадим".into(), 0.9));
        v.emotion = Some(("Neutral".into(), 0.9));
        v.posture = Some(("standing".into(), 0.9));
        v.body_coverage = Some("full");
        v.plausibility = Some("ok".into());
        v.position_3d = Some(Position3D {
            x: 0.7,
            y: 0.2,
            z: 1.0,
            confidence: 0.8,
        });
        world.people.push(v);
        let md = render_snapshot(&world, 18.0, "17:51:19");
        assert!(
            !md.contains("⚠️"),
            "clean scene should carry no notes:\n{md}"
        );
    }

    #[test]
    fn render_recap_shape_and_phantom_note() {
        use mvm_core::recap::{EpisodeObs, FrameObs, PersonObs, Recap};
        // 40-frame session: Вадим(0) throughout, phantom id12 mid-session; a book
        // re-ided across two object ids → one rolled-up `reading`.
        let mut frames = vec![FrameObs::default(); 40];
        for (i, f) in frames.iter_mut().enumerate() {
            f.people.push(PersonObs {
                id: 0,
                dist_m: Some(0.8 + i as f32 * 0.01),
                posture: Some("sitting".into()),
                identity: Some("Вадим".into()),
            });
            f.object_classes.push("keyboard".into());
            f.zone_labels.push("workstation".into());
            if (20..24).contains(&i) {
                f.people.push(PersonObs {
                    id: 12,
                    dist_m: Some(0.9),
                    posture: Some("lying".into()),
                    identity: None,
                });
            }
        }
        let eps = vec![
            EpisodeObs {
                label: "reading".into(),
                participants: vec![0, 10002],
                duration_s: 2.0,
                peak_confidence: 0.4,
                stable_participants: true,
            },
            EpisodeObs {
                label: "reading".into(),
                participants: vec![0, 10003],
                duration_s: 3.0,
                peak_confidence: 0.6,
                stable_participants: true,
            },
            EpisodeObs {
                label: "typing".into(),
                participants: vec![0, 10004],
                duration_s: 10.0,
                peak_confidence: 0.5,
                stable_participants: true,
            },
        ];
        let recap = Recap::build(&frames, &eps);
        let md = render_recap(&recap, "18:55:01–18:57:39");

        assert!(
            md.starts_with("# Recap 18:55:01–18:57:39 — 40 frames"),
            "got:\n{md}"
        );
        assert!(
            md.contains("**Вадим(0)** — 40/40 frames (100%)"),
            "got:\n{md}"
        );
        // typing (10s) leads reading (5s, rolled from 2 episodes).
        let typing = md.find("typing").unwrap();
        let reading = md.find("reading").unwrap();
        assert!(typing < reading, "longer activity first:\n{md}");
        assert!(
            md.contains("reading(Вадим(0)) — 5.0s total across 2 episodes"),
            "roll-up:\n{md}"
        );
        // Phantom surfaces as a note, not in the People list body.
        assert!(
            md.contains("⚠️ id12 — вероятный фантом: 4 кадра"),
            "got:\n{md}"
        );
        assert!(
            md.contains("Objects: keyboard") && md.contains("Zones: workstation"),
            "scene:\n{md}"
        );
    }
}
