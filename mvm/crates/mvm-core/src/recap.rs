//! Recap aggregation (phase 12.C2) — turn the noisy per-frame / per-episode logs
//! into a compact session summary for an LLM.
//!
//! **Why:** a real ~2.5 min session produced 159 scene frames and **62 completed
//! episodes**, but only ~5 of those episodes are meaningful — the rest are
//! *object-id churn* (one physical book re-tracked as ids 10002→10007 splits one
//! `reading` into 31 fragments) plus sub-second flicker at the dwell threshold.
//! This module collapses that: episodes group by `(label, human participants)`
//! with the churny **object ids dropped from the key**, and people roll up into a
//! presence timeline. Pure arithmetic over lightweight observations — no
//! serialisation (the `recap` subcommand parses JSONL and renders markdown, C3),
//! no `ort`/`image`. Mirrors the shape of [`crate::activity`].

/// One frame's worth of observations the recap reads (the `recap` subcommand
/// fills these from a `scene-*.jsonl` line).
#[derive(Debug, Clone, Default)]
pub struct FrameObs {
    pub people: Vec<PersonObs>,
    /// Distinct object classes present this frame (background + foreground).
    pub object_classes: Vec<String>,
    /// `stable_zones` labels present this frame (e.g. "workstation").
    pub zone_labels: Vec<String>,
    pub anomaly: f32,
}

/// A person as seen in one frame.
#[derive(Debug, Clone, Default)]
pub struct PersonObs {
    pub id: u32,
    pub dist_m: Option<f32>,
    pub posture: Option<String>,
    /// Gallery identity when the frame matched a face to a known user.
    pub identity: Option<String>,
}

/// One completed activity episode (a `events-*.jsonl` line).
#[derive(Debug, Clone)]
pub struct EpisodeObs {
    pub label: String,
    /// Participants including object ids (≥ 10000) — dropped from the group key.
    pub participants: Vec<u32>,
    pub duration_s: f64,
    pub peak_confidence: f32,
    pub stable_participants: bool,
}

/// A person's session presence roll-up.
#[derive(Debug, Clone, PartialEq)]
pub struct PersonSummary {
    pub id: u32,
    /// Gallery name if this track was ever recognised (first seen wins).
    pub identity: Option<String>,
    pub frames_present: usize,
    pub first_frame: usize,
    pub last_frame: usize,
    pub presence_pct: f32,
    pub dist_min: Option<f32>,
    pub dist_max: Option<f32>,
    pub dominant_posture: Option<String>,
    /// Suspected phantom: an ephemeral id (≥10) that lived briefly and appeared
    /// mid-session — a *marker*, not a verdict (a real brief visitor trips it too).
    pub is_phantom: bool,
}

/// An activity rolled up across object-id churn and dwell-flicker.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivitySummary {
    pub label: String,
    /// Human participants (ids < 10000), sorted; the group key alongside `label`.
    pub humans: Vec<u32>,
    pub episode_count: usize,
    pub total_s: f64,
    pub longest_s: f64,
    pub peak_confidence: f32,
    /// Every constituent episode had `stable_participants` (durable ids only).
    pub all_stable: bool,
}

/// An object class's presence span.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectSummary {
    pub class: String,
    pub frames_present: usize,
    pub first_frame: usize,
    pub last_frame: usize,
}

/// The whole-session recap.
#[derive(Debug, Clone, Default)]
pub struct Recap {
    pub frame_count: usize,
    pub anomaly_max: f32,
    pub people: Vec<PersonSummary>,
    pub activities: Vec<ActivitySummary>,
    pub objects: Vec<ObjectSummary>,
    pub zones: Vec<String>,
}

/// Object-id base — participants at/above this are objects, dropped from the
/// activity group key so churn (10002→10007) collapses into one activity.
const OBJECT_ID_BASE: u32 = 10000;
/// Reserved (gallery) ids 0–9 are durable; ≥10 are ephemeral and phantom-prone.
const EPHEMERAL_ID_BASE: u32 = 10;
/// A session must be at least this many frames before the phantom heuristic runs
/// (short clips have no meaningful "brief + mid-session" signal).
const PHANTOM_MIN_SESSION_FRAMES: usize = 20;
/// An ephemeral id present for less than this fraction of the session, and born
/// after the first frame, is flagged a suspected phantom.
const PHANTOM_MAX_PRESENCE_FRAC: f32 = 0.25;

impl Recap {
    /// Build the recap from parsed frame + episode observations.
    pub fn build(frames: &[FrameObs], episodes: &[EpisodeObs]) -> Recap {
        let frame_count = frames.len();
        let anomaly_max = frames.iter().map(|f| f.anomaly).fold(0.0f32, f32::max);
        Recap {
            frame_count,
            anomaly_max,
            people: summarize_people(frames),
            activities: summarize_activities(episodes),
            objects: summarize_objects(frames),
            zones: summarize_zones(frames),
        }
    }
}

fn summarize_people(frames: &[FrameObs]) -> Vec<PersonSummary> {
    use std::collections::HashMap;
    struct Acc {
        frames: usize,
        first: usize,
        last: usize,
        dist_min: Option<f32>,
        dist_max: Option<f32>,
        posture_counts: HashMap<String, usize>,
        identity: Option<String>,
    }
    let mut accs: HashMap<u32, Acc> = HashMap::new();
    for (i, f) in frames.iter().enumerate() {
        for p in &f.people {
            let a = accs.entry(p.id).or_insert(Acc {
                frames: 0,
                first: i,
                last: i,
                dist_min: None,
                dist_max: None,
                posture_counts: HashMap::new(),
                identity: None,
            });
            a.frames += 1;
            a.last = i;
            if let Some(d) = p.dist_m {
                a.dist_min = Some(a.dist_min.map_or(d, |m| m.min(d)));
                a.dist_max = Some(a.dist_max.map_or(d, |m| m.max(d)));
            }
            if let Some(post) = &p.posture {
                *a.posture_counts.entry(post.clone()).or_insert(0) += 1;
            }
            if a.identity.is_none() {
                if let Some(name) = &p.identity {
                    a.identity = Some(name.clone());
                }
            }
        }
    }
    let total = frames.len().max(1) as f32;
    let mut people: Vec<PersonSummary> = accs
        .into_iter()
        .map(|(id, a)| {
            let presence_frac = a.frames as f32 / total;
            let is_phantom = id >= EPHEMERAL_ID_BASE
                && id < OBJECT_ID_BASE
                && frames.len() >= PHANTOM_MIN_SESSION_FRAMES
                && presence_frac < PHANTOM_MAX_PRESENCE_FRAC
                && a.first > 0;
            let dominant_posture = a
                .posture_counts
                .into_iter()
                .max_by(|x, y| x.1.cmp(&y.1).then_with(|| y.0.cmp(&x.0)))
                .map(|(p, _)| p);
            PersonSummary {
                id,
                identity: a.identity,
                frames_present: a.frames,
                first_frame: a.first,
                last_frame: a.last,
                presence_pct: presence_frac * 100.0,
                dist_min: a.dist_min,
                dist_max: a.dist_max,
                dominant_posture,
                is_phantom,
            }
        })
        .collect();
    // Gallery users first, then by presence desc, then id — deterministic.
    people.sort_by(|a, b| {
        b.frames_present
            .cmp(&a.frames_present)
            .then_with(|| a.id.cmp(&b.id))
    });
    people
}

fn summarize_activities(episodes: &[EpisodeObs]) -> Vec<ActivitySummary> {
    use std::collections::HashMap;
    // Key = (label, sorted human participants). Object ids drop out → churn
    // (same book re-ided) collapses into one activity.
    struct Acc {
        count: usize,
        total: f64,
        longest: f64,
        peak: f32,
        all_stable: bool,
    }
    let mut accs: HashMap<(String, Vec<u32>), Acc> = HashMap::new();
    for e in episodes {
        let mut humans: Vec<u32> = e
            .participants
            .iter()
            .copied()
            .filter(|&p| p < OBJECT_ID_BASE)
            .collect();
        humans.sort_unstable();
        let a = accs.entry((e.label.clone(), humans)).or_insert(Acc {
            count: 0,
            total: 0.0,
            longest: 0.0,
            peak: 0.0,
            all_stable: true,
        });
        a.count += 1;
        a.total += e.duration_s;
        a.longest = a.longest.max(e.duration_s);
        a.peak = a.peak.max(e.peak_confidence);
        a.all_stable &= e.stable_participants;
    }
    let mut acts: Vec<ActivitySummary> = accs
        .into_iter()
        .map(|((label, humans), a)| ActivitySummary {
            label,
            humans,
            episode_count: a.count,
            total_s: a.total,
            longest_s: a.longest,
            peak_confidence: a.peak,
            all_stable: a.all_stable,
        })
        .collect();
    // Longest cumulative first — the session's dominant activities lead.
    acts.sort_by(|a, b| {
        b.total_s
            .partial_cmp(&a.total_s)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.label.cmp(&b.label))
            .then_with(|| a.humans.cmp(&b.humans))
    });
    acts
}

fn summarize_objects(frames: &[FrameObs]) -> Vec<ObjectSummary> {
    use std::collections::HashMap;
    let mut accs: HashMap<String, (usize, usize, usize)> = HashMap::new(); // count, first, last
    for (i, f) in frames.iter().enumerate() {
        for c in &f.object_classes {
            let e = accs.entry(c.clone()).or_insert((0, i, i));
            e.0 += 1;
            e.2 = i;
        }
    }
    let mut objs: Vec<ObjectSummary> = accs
        .into_iter()
        .map(|(class, (count, first, last))| ObjectSummary {
            class,
            frames_present: count,
            first_frame: first,
            last_frame: last,
        })
        .collect();
    objs.sort_by(|a, b| {
        b.frames_present
            .cmp(&a.frames_present)
            .then_with(|| a.class.cmp(&b.class))
    });
    objs
}

fn summarize_zones(frames: &[FrameObs]) -> Vec<String> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in frames {
        for z in &f.zone_labels {
            *counts.entry(z.clone()).or_insert(0) += 1;
        }
    }
    let mut zones: Vec<(String, usize)> = counts.into_iter().collect();
    zones.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    zones.into_iter().map(|(z, _)| z).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person(id: u32, dist: f32, posture: &str) -> PersonObs {
        PersonObs { id, dist_m: Some(dist), posture: Some(posture.into()), identity: None }
    }
    fn ep(label: &str, participants: Vec<u32>, dur: f64, peak: f32, stable: bool) -> EpisodeObs {
        EpisodeObs {
            label: label.into(),
            participants,
            duration_s: dur,
            peak_confidence: peak,
            stable_participants: stable,
        }
    }

    #[test]
    fn activities_collapse_object_id_churn() {
        // The live finding: one physical book re-ided 10002→10004 splits `reading`
        // into 3 episodes — recap must roll them into ONE (label, [person]) row.
        let eps = vec![
            ep("reading", vec![0, 10002], 2.0, 0.4, true),
            ep("reading", vec![0, 10003], 3.0, 0.6, true),
            ep("reading", vec![0, 10004], 1.0, 0.3, true),
            ep("typing", vec![0, 10005], 10.0, 0.5, true),
        ];
        let recap = Recap::build(&[], &eps);
        let reading = recap.activities.iter().find(|a| a.label == "reading").unwrap();
        assert_eq!(reading.humans, vec![0]);
        assert_eq!(reading.episode_count, 3, "3 churn fragments → one activity");
        assert!((reading.total_s - 6.0).abs() < 1e-9);
        assert!((reading.longest_s - 3.0).abs() < 1e-9);
        assert!((reading.peak_confidence - 0.6).abs() < 1e-6);
        // typing (10s) sorts before reading (6s).
        assert_eq!(recap.activities[0].label, "typing");
    }

    #[test]
    fn two_person_activity_keeps_both_humans_separate() {
        let eps = vec![
            ep("people_together", vec![0, 11], 5.0, 0.5, false),
            ep("reading", vec![0, 10002], 2.0, 0.4, true),
        ];
        let recap = Recap::build(&[], &eps);
        let pt = recap.activities.iter().find(|a| a.label == "people_together").unwrap();
        assert_eq!(pt.humans, vec![0, 11]);
        assert!(!pt.all_stable, "ephemeral participant → not stable");
    }

    #[test]
    fn presence_rle_and_phantom_flag() {
        // 40-frame session: id0 throughout (gallery), id12 4 frames mid-session
        // (ephemeral, brief, born late) → phantom; a hypothetical id13 present
        // from frame 0 for many frames is NOT flagged.
        let mut frames = vec![FrameObs::default(); 40];
        for (i, f) in frames.iter_mut().enumerate() {
            f.people.push(person(0, 1.0 + (i as f32) * 0.01, "sitting"));
            if (20..24).contains(&i) {
                f.people.push(person(12, 0.9, "lying"));
            }
        }
        let recap = Recap::build(&frames, &[]);
        let p0 = recap.people.iter().find(|p| p.id == 0).unwrap();
        let p12 = recap.people.iter().find(|p| p.id == 12).unwrap();
        assert_eq!(p0.frames_present, 40);
        assert!(!p0.is_phantom, "gallery id never phantom");
        assert_eq!(p0.dominant_posture.as_deref(), Some("sitting"));
        assert_eq!(p12.frames_present, 4);
        assert_eq!(p12.first_frame, 20);
        assert!(p12.is_phantom, "ephemeral + brief + mid-session → phantom marker");
    }

    #[test]
    fn brief_early_visitor_not_phantom_in_short_session() {
        // Below PHANTOM_MIN_SESSION_FRAMES the heuristic must not fire.
        let mut frames = vec![FrameObs::default(); 10];
        for f in frames.iter_mut() {
            f.people.push(person(11, 1.0, "standing"));
        }
        let recap = Recap::build(&frames, &[]);
        assert!(!recap.people[0].is_phantom, "short session → no phantom claim");
    }

    #[test]
    fn objects_and_zones_and_anomaly_rolled_up() {
        let mut frames = vec![FrameObs::default(); 5];
        for (i, f) in frames.iter_mut().enumerate() {
            f.object_classes.push("bed".into()); // background, every frame
            if i >= 2 {
                f.object_classes.push("cup".into()); // appears frame 2
            }
            f.zone_labels.push("workstation".into());
            f.anomaly = if i == 3 { 0.54 } else { 0.0 };
        }
        let recap = Recap::build(&frames, &[]);
        assert!((recap.anomaly_max - 0.54).abs() < 1e-6);
        assert_eq!(recap.objects[0].class, "bed");
        assert_eq!(recap.objects[0].frames_present, 5);
        let cup = recap.objects.iter().find(|o| o.class == "cup").unwrap();
        assert_eq!(cup.first_frame, 2);
        assert_eq!(cup.frames_present, 3);
        assert_eq!(recap.zones, vec!["workstation".to_string()]);
    }
}
