//! Semantic Group Router (§16.0.3).
//!
//! ```text
//! WorldState → Snapshot → SemanticGroupRouter → GroupWriters
//! ```
//!
//! Роутер **не** меняет [`WorldState`], **не** вызывает модели и **не** добавляет
//! вычислений в hot-path. Он маршрутизирует уже полученные сигналы по семантическим
//! ветвям G₁–G₅ и сохраняет provenance. Отображение чистое и детерминированное:
//! один и тот же снимок + контекст → идентичное наблюдение.
//!
//! Здесь атрибуты сущностей эмитятся как «сырые» наблюдённые значения; типизированная
//! обёртка [`crate::Label`] с per-signal статусом навешивается group-writer'ами в §16.3.

use crate::observation::{
    CanonicalObservation, EntityRecord, Provenance, SemanticGroup, SCHEMA_VERSION,
};
use mvm_core::world::{ObjectState, PersonState, WorldState};

/// Контекст уровня наблюдения. Роутер сам не знает стенных часов (в [`WorldState`]
/// только монотонный [`std::time::Instant`]), поэтому писатель передаёт готовый
/// ISO-8601 timestamp и порядковый индекс кадра в сессии.
#[derive(Debug, Clone)]
pub struct RouteContext<'a> {
    /// Стабильный id датасета (корень хранилища).
    pub dataset_id: &'a str,
    /// Id сессии (`YYYY-MM-DD_HHMMSS`).
    pub session_id: &'a str,
    /// ISO-8601 timestamp кадра (writer конвертирует `Instant` → стенные часы).
    pub timestamp: &'a str,
    /// Порядковый индекс сохранённого кадра внутри сессии.
    pub seq_in_session: u64,
}

/// Маршрутизировать снимок [`WorldState`] в [`CanonicalObservation`].
///
/// Сущности-люди → ветвь G₁ Human, объекты → G₂ Scene. Кросс-секущие ветви
/// (G₃ Interaction / G₄ Temporal / G₅ Quality) активируются на уровне кадра по
/// уже наблюдённым сигналам и перечисляются в `semantic_groups` (selective routing).
pub fn route(world: &WorldState, ctx: &RouteContext) -> CanonicalObservation {
    let mut obs = CanonicalObservation::new(
        ctx.dataset_id,
        ctx.session_id,
        world.frame_id,
        ctx.timestamp,
        ctx.seq_in_session,
    );

    // G₁ Human — детерминированный порядок по track_id.
    let mut people: Vec<&PersonState> = world.people.iter().collect();
    people.sort_by_key(|p| p.track_id);
    for p in people {
        obs.entities.push(EntityRecord {
            entity_id: format!("p{}", p.track_id),
            track_id: Some(p.track_id as u64),
            group: SemanticGroup::Human,
            kind: "person".to_string(),
            attributes: person_attributes(p),
        });
    }

    // G₂ Scene — детерминированный порядок по object id.
    let mut objects: Vec<&ObjectState> = world.objects.iter().collect();
    objects.sort_by_key(|o| o.id);
    for o in objects {
        obs.entities.push(EntityRecord {
            entity_id: format!("o{}", o.id),
            track_id: Some(o.id as u64),
            group: SemanticGroup::Scene,
            kind: "object".to_string(),
            attributes: object_attributes(o),
        });
    }

    obs.semantic_groups = active_groups(world);

    // G₅ Quality — «quality is data»: счётчики + anomaly кадра из наблюдений.
    obs.quality.people_count = world.people.len() as u32;
    obs.quality.object_count = world.objects.len() as u32;
    obs.quality.anomaly_score = world.anomaly_score;

    // Provenance: этот кадр собран роутером (принцип «Provenance is mandatory»).
    obs.provenance.push(Provenance {
        model: "mvm-dataset:router".to_string(),
        version: SCHEMA_VERSION.to_string(),
        source: "router".to_string(),
        rule: None,
    });

    obs
}

/// Какие ветви активирует этот кадр (selective routing). Порядок — канонический
/// [`SemanticGroup::ALL`]. Quality (G₅) активна всегда: quality/provenance — часть
/// каждой записи. Остальные ветви — только при наличии соответствующих сигналов.
fn active_groups(world: &WorldState) -> Vec<SemanticGroup> {
    let has_human = !world.people.is_empty();
    let has_scene = !world.objects.is_empty();
    let has_interaction = !world.activities.is_empty()
        || world
            .people
            .iter()
            .any(|p| !p.hands.is_empty() || p.left_gesture.is_some() || p.right_gesture.is_some());
    let has_temporal = !world.activities.is_empty()
        || !world.events.is_empty()
        || !world.activity_instances.is_empty()
        || world
            .people
            .iter()
            .any(|p| p.motion_state.is_some() || p.velocity.is_some());

    SemanticGroup::ALL
        .into_iter()
        .filter(|g| match g {
            SemanticGroup::Human => has_human,
            SemanticGroup::Scene => has_scene,
            SemanticGroup::Interaction => has_interaction,
            SemanticGroup::Temporal => has_temporal,
            SemanticGroup::Quality => true,
        })
        .collect()
}

/// Наблюдённые сигналы человека как JSON-объект (только присутствующие поля).
/// Типизация в `Label<T>` — §16.3; здесь ничего не схлопывается и не додумывается.
fn person_attributes(p: &PersonState) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    if let Some((name, c)) = &p.identity {
        m.insert("identity".into(), serde_json::json!({ "label": name, "confidence": c }));
    }
    if let Some((label, c)) = &p.emotion {
        m.insert("emotion".into(), serde_json::json!({ "label": label, "confidence": c }));
    }
    if let Some((label, c)) = &p.posture {
        m.insert("posture".into(), serde_json::json!({ "label": label, "confidence": c }));
    }
    // Posture status is data (§16.0.2): a missing posture under upper-body
    // occlusion is *withheld*, not a negative label — mark it explicitly so the
    // exporter never treats absence as a class.
    m.insert("posture_status".into(), serde_json::json!(posture_status(p)));
    // Emotion entropy (G₅ quality): near-chance predictions carry high entropy.
    if let Some(dist) = &p.emotion_distribution {
        m.insert("emotion_entropy".into(), serde_json::json!(mvm_core::emotion_entropy(dist)));
    }
    if let Some(pos) = &p.position_3d {
        m.insert(
            "position_3d".into(),
            serde_json::json!({ "x": pos.x, "y": pos.y, "z": pos.z, "confidence": pos.confidence }),
        );
    }
    insert_opt(&mut m, "motion_state", p.motion_state);
    if let Some(v) = p.velocity {
        m.insert("velocity".into(), serde_json::json!(v));
    }
    insert_opt(&mut m, "body_coverage", p.body_coverage);
    insert_opt(&mut m, "torso_attitude", p.torso_attitude);
    if let Some(f) = p.torso_flexion {
        m.insert("torso_flexion".into(), serde_json::json!(f));
    }
    insert_opt(&mut m, "head_facing", p.head_facing);
    if let Some(y) = p.head_yaw {
        m.insert("head_yaw".into(), serde_json::json!(y));
    }
    if let Some(pl) = &p.plausibility {
        m.insert("plausibility".into(), serde_json::json!(pl));
    }
    insert_opt(&mut m, "left_gesture", p.left_gesture);
    insert_opt(&mut m, "right_gesture", p.right_gesture);
    if !p.hands.is_empty() {
        let sides: Vec<&str> = p.hands.iter().map(|h| h.side).collect();
        m.insert("hands".into(), serde_json::json!({ "count": p.hands.len(), "sides": sides }));
    }
    serde_json::Value::Object(m)
}

/// Наблюдённые сигналы объекта как JSON-объект.
fn object_attributes(o: &ObjectState) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert("class".into(), serde_json::json!(o.class));
    m.insert("confidence".into(), serde_json::json!(o.confidence));
    m.insert("age".into(), serde_json::json!(o.age));
    m.insert("remembered".into(), serde_json::json!(o.remembered));
    if let Some(pos) = &o.position_3d {
        m.insert(
            "position_3d".into(),
            serde_json::json!({ "x": pos.x, "y": pos.y, "z": pos.z, "confidence": pos.confidence }),
        );
    }
    serde_json::Value::Object(m)
}

/// Label-status of a person's posture (§16.0.2). `observed` when present;
/// `withheld` when the lower body is occluded (honest gap, not a negative);
/// `unknown` otherwise.
fn posture_status(p: &PersonState) -> &'static str {
    if p.posture.is_some() {
        "observed"
    } else if matches!(p.body_coverage, Some("upper") | Some("head")) {
        "withheld"
    } else {
        "unknown"
    }
}

fn insert_opt(m: &mut serde_json::Map<String, serde_json::Value>, key: &str, v: Option<&str>) {
    if let Some(v) = v {
        m.insert(key.into(), serde_json::json!(v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::world::WorldState;
    use mvm_core::{Activity, BBox};
    use std::time::Instant;

    fn ctx() -> RouteContext<'static> {
        RouteContext {
            dataset_id: "mvm_sessions",
            session_id: "2026-07-17_153000",
            timestamp: "2026-07-17T15:30:00.123",
            seq_in_session: 7,
        }
    }

    fn bare_person(track_id: u32, now: Instant) -> PersonState {
        PersonState {
            track_id,
            bbox: BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 },
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
            last_seen: now,
        }
    }

    fn bare_object(id: u32, class: &str, now: Instant) -> ObjectState {
        ObjectState {
            id,
            class: class.to_string(),
            bbox: BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 },
            confidence: 0.7,
            age: 3,
            position_3d: None,
            mask: None,
            remembered: false,
            last_seen: now,
        }
    }

    #[test]
    fn empty_world_routes_only_quality() {
        let now = Instant::now();
        let world = WorldState::new(now);
        let obs = route(&world, &ctx());
        assert!(obs.entities.is_empty());
        assert_eq!(obs.semantic_groups, vec![SemanticGroup::Quality]);
    }

    #[test]
    fn sitting_person_routes_human_and_quality_only() {
        let now = Instant::now();
        let mut world = WorldState::new(now);
        let mut p = bare_person(1, now);
        p.posture = Some(("sitting".to_string(), 0.95));
        // No hands, no gesture, no motion, no activity → no Interaction/Temporal.
        world.people.push(p);
        let obs = route(&world, &ctx());
        assert_eq!(obs.semantic_groups, vec![SemanticGroup::Human, SemanticGroup::Quality]);
    }

    #[test]
    fn full_frame_routes_all_five_groups() {
        let now = Instant::now();
        let mut world = WorldState::new(now);
        let mut p = bare_person(1, now);
        p.motion_state = Some("moving"); // → Temporal
        p.left_gesture = Some("reaching"); // → Interaction
        world.people.push(p);
        world.objects.push(bare_object(10001, "cell phone", now)); // → Scene
        world.activities.push(Activity {
            label: "typing".to_string(),
            confidence: 0.8,
            participants: vec![1],
        }); // → Interaction + Temporal
        let obs = route(&world, &ctx());
        assert_eq!(obs.semantic_groups, SemanticGroup::ALL.to_vec());
    }

    #[test]
    fn person_and_object_carry_binding_keys() {
        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.people.push(bare_person(2, now));
        world.objects.push(bare_object(10005, "cup", now));
        let obs = route(&world, &ctx());

        let person = obs.entities.iter().find(|e| e.kind == "person").unwrap();
        assert_eq!(person.entity_id, "p2");
        assert_eq!(person.track_id, Some(2));
        assert_eq!(person.group, SemanticGroup::Human);

        let object = obs.entities.iter().find(|e| e.kind == "object").unwrap();
        assert_eq!(object.entity_id, "o10005");
        assert_eq!(object.track_id, Some(10005));
        assert_eq!(object.group, SemanticGroup::Scene);
        assert_eq!(object.attributes["class"], serde_json::json!("cup"));
    }

    #[test]
    fn routing_is_deterministic_and_ordered() {
        let now = Instant::now();
        let mut world = WorldState::new(now);
        // Insert out of order — router must sort by id.
        world.people.push(bare_person(3, now));
        world.people.push(bare_person(1, now));
        world.objects.push(bare_object(10009, "book", now));
        world.objects.push(bare_object(10002, "keyboard", now));

        let a = route(&world, &ctx());
        let b = route(&world, &ctx());
        assert_eq!(a, b, "same snapshot + ctx must give identical observations");

        let ids: Vec<&str> = a.entities.iter().map(|e| e.entity_id.as_str()).collect();
        assert_eq!(ids, vec!["p1", "p3", "o10002", "o10009"]);
    }

    #[test]
    fn router_records_its_provenance() {
        let world = WorldState::new(Instant::now());
        let obs = route(&world, &ctx());
        assert!(
            obs.provenance.iter().any(|p| p.model == "mvm-dataset:router" && p.source == "router"),
            "router must stamp observation-level provenance"
        );
    }

    #[test]
    fn quality_block_carries_counts_and_anomaly() {
        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.people.push(bare_person(1, now));
        world.objects.push(bare_object(10001, "cup", now));
        world.anomaly_score = 0.42;
        let obs = route(&world, &ctx());
        assert_eq!(obs.quality.people_count, 1);
        assert_eq!(obs.quality.object_count, 1);
        assert!((obs.quality.anomaly_score - 0.42).abs() < 1e-6);
    }

    #[test]
    fn posture_status_reflects_occlusion_honesty() {
        let now = Instant::now();
        // Observed posture.
        let mut seated = bare_person(1, now);
        seated.posture = Some(("sitting".into(), 0.9));
        // Occluded lower body, posture withheld upstream.
        let mut occluded = bare_person(2, now);
        occluded.posture = None;
        occluded.body_coverage = Some("upper");
        // No posture, no occlusion info → unknown.
        let bare = bare_person(3, now);

        let mut world = WorldState::new(now);
        world.people = vec![seated, occluded, bare];
        let obs = route(&world, &ctx());
        let status = |eid: &str| {
            obs.entities.iter().find(|e| e.entity_id == eid).unwrap().attributes["posture_status"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert_eq!(status("p1"), "observed");
        assert_eq!(status("p2"), "withheld");
        assert_eq!(status("p3"), "unknown");
    }

    #[test]
    fn emotion_entropy_emitted_when_distribution_present() {
        let now = Instant::now();
        let mut p = bare_person(1, now);
        // Uniform distribution → maximal entropy (1.0 after normalization).
        p.emotion_distribution = Some([0.125; 8]);
        let mut world = WorldState::new(now);
        world.people.push(p);
        let obs = route(&world, &ctx());
        let ent = obs.entities[0].attributes["emotion_entropy"].as_f64().unwrap();
        assert!(ent > 0.99, "uniform softmax → entropy ≈ 1.0, got {ent}");
    }

    #[test]
    fn withheld_posture_is_not_emitted_as_a_value() {
        // A person under upper-body occlusion has posture == None (withheld upstream).
        // The router must not invent a posture; the attribute is simply absent — the
        // §16.0.2 "withheld ≠ negative class" rule holds structurally here.
        let now = Instant::now();
        let mut world = WorldState::new(now);
        let mut p = bare_person(1, now);
        p.body_coverage = Some("upper");
        p.posture = None;
        world.people.push(p);
        let obs = route(&world, &ctx());
        let person = &obs.entities[0];
        assert!(person.attributes.get("posture").is_none());
        assert_eq!(person.attributes["body_coverage"], serde_json::json!("upper"));
    }
}
