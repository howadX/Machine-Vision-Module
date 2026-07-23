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

/// Реляционный scene-graph одного кадра, посчитанный вызывающим (§16.7).
///
/// Dataset-слой не владеет `build_scene`/`ZoneTracker` (они в `mvm-fusion`, выше
/// по слоям), поэтому зоны и рёбра считает writer в `mvm-cli` и передаёт сюда —
/// роутер остаётся **чистой функцией своих входов**. `SceneRelations::default()`
/// (пусто) — для путей без реляционного контекста (sync-путь, unit-тесты).
#[derive(Debug, Clone, Default)]
pub struct SceneRelations {
    /// Зоны кадра (связные компоненты; персистентные — из `ZoneTracker`).
    pub zones: Vec<ZoneRel>,
    /// Рёбра близости/пересечения между сущностями.
    pub edges: Vec<EdgeRel>,
}

/// Одна зона: набор сущностей-участников + опц. персистентный id и семантика.
#[derive(Debug, Clone)]
pub struct ZoneRel {
    /// Стабильный id из `ZoneTracker`; `None` для чисто per-frame компонента.
    pub persistent_id: Option<u32>,
    /// Семантическая метка (`workstation`/`seating`/…), `None` если не применима.
    pub label: Option<String>,
    /// Scene-graph id'ы участников (люди/объекты).
    pub members: Vec<u32>,
}

/// Одно реляционное ребро (близость/пересечение) между двумя scene-graph id.
#[derive(Debug, Clone)]
pub struct EdgeRel {
    pub a: u32,
    pub b: u32,
    /// Тип ребра (`near`/`overlapping`/…).
    pub rel: String,
}

/// Граница scene-graph id, зеркалит `mvm_scene::OBJECT_ID_BASE`: id ≥ этого —
/// объекты (`o{id}`), ниже — люди (`p{id}`). Держится локально, чтобы dataset-слой
/// не тянул зависимость на `mvm-scene`.
const OBJECT_ID_BASE: u32 = 10_000;

/// Отобразить сырой scene-graph id в стабильный `entity_id`, согласованный с id
/// сущностей, которые эмитит роутер (`p{track_id}` / `o{id}`) — связь relations/
/// activities/events с `entities`.
fn entity_id_of(id: u32) -> String {
    if id >= OBJECT_ID_BASE {
        format!("o{id}")
    } else {
        format!("p{id}")
    }
}

/// Маршрутизировать снимок [`WorldState`] в [`CanonicalObservation`] без внешнего
/// реляционного контекста (relations остаётся пустым). Тонкая обёртка над
/// [`route_with_relations`] — activities/events всё равно берутся из `WorldState`.
pub fn route(world: &WorldState, ctx: &RouteContext) -> CanonicalObservation {
    route_with_relations(world, ctx, &SceneRelations::default())
}

/// Маршрутизировать снимок [`WorldState`] + посчитанные зоны/рёбра в наблюдение.
///
/// Сущности-люди → ветвь G₁ Human, объекты → G₂ Scene. Кросс-секущие ветви
/// (G₃ Interaction / G₄ Temporal / G₅ Quality) активируются на уровне кадра по
/// уже наблюдённым сигналам и перечисляются в `semantic_groups` (selective routing).
/// `relations`/`activities`/`events` (§16.7) заполняются из `world` + `rel`.
pub fn route_with_relations(
    world: &WorldState,
    ctx: &RouteContext,
    rel: &SceneRelations,
) -> CanonicalObservation {
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

    // G₂ relational — зоны + рёбра, посчитанные вызывающим (§16.7).
    obs.relations = relation_records(rel);
    // G₃/G₄ — per-frame активности из журнала (или flat-списка) + завершённые эпизоды.
    obs.activities = activity_records(world);
    obs.events = event_records(world);

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

/// G₂ реляционные записи (§16.7): зоны + рёбра как гетерогенный `{kind:…}` JSON.
/// Участники/концы рёбер маппятся в `entity_id` — связь с `entities`.
fn relation_records(rel: &SceneRelations) -> Vec<serde_json::Value> {
    let mut out = Vec::with_capacity(rel.zones.len() + rel.edges.len());
    for z in &rel.zones {
        let members: Vec<String> = z.members.iter().map(|&id| entity_id_of(id)).collect();
        out.push(serde_json::json!({
            "kind": "zone",
            "label": z.label,
            "members": members,
            "persistent_id": z.persistent_id,
        }));
    }
    for e in &rel.edges {
        out.push(serde_json::json!({
            "kind": "edge",
            "a": entity_id_of(e.a),
            "b": entity_id_of(e.b),
            "rel": e.rel,
        }));
    }
    out
}

/// G₃/G₄ per-frame активности (§16.7). Предпочитаем инстансы журнала (несут
/// временные поля started_at/duration); иначе — плоский `activities` (sync-путь).
/// Участники → `entity_id`; `stable` зеркалит reserved-id правило (эфемерные
/// анонимы 10..10000 нестабильны — honesty-нота «участники нестабильны»).
fn activity_records(world: &WorldState) -> Vec<serde_json::Value> {
    if !world.activity_instances.is_empty() {
        world
            .activity_instances
            .iter()
            .map(|i| {
                let participants: Vec<String> =
                    i.key.participants.iter().map(|&id| entity_id_of(id)).collect();
                serde_json::json!({
                    "label": i.key.label,
                    "participants": participants,
                    "confidence": i.confidence,
                    "state": i.phase.as_str(),
                    "started_at": i.started_at,
                    "duration_s": i.last_seen - i.started_at,
                    "stable": i.key.stable_participants(),
                })
            })
            .collect()
    } else {
        world
            .activities
            .iter()
            .map(|a| {
                let participants: Vec<String> =
                    a.participants.iter().map(|&id| entity_id_of(id)).collect();
                let stable = a.participants.iter().all(|&id| id < 10 || id >= OBJECT_ID_BASE);
                serde_json::json!({
                    "label": a.label,
                    "participants": participants,
                    "confidence": a.confidence,
                    "stable": stable,
                })
            })
            .collect()
    }
}

/// G₄ завершённые эпизоды (§16.7), инлайн на кадре закрытия — зеркалит плоский
/// `events-*.jsonl` внутри единого файла сессии.
fn event_records(world: &WorldState) -> Vec<serde_json::Value> {
    world
        .events
        .iter()
        .map(|e| {
            let participants: Vec<String> =
                e.participants.iter().map(|&id| entity_id_of(id)).collect();
            serde_json::json!({
                "kind": "episode",
                "label": e.label,
                "participants": participants,
                "started_at": e.started_at,
                "ended_at": e.ended_at,
                "duration_s": e.duration_s,
                "peak_confidence": e.peak_confidence,
                "stable": e.stable_participants,
            })
        })
        .collect()
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
    // This frame's matched YuNet face bbox (normalized [x1,y1,x2,y2]) — a face-crop
    // dataset label (§16.5.1 train_identity). Absent when no face matched.
    if let Some(fb) = &p.face_bbox {
        m.insert("face_bbox".into(), serde_json::json!([fb.x1, fb.y1, fb.x2, fb.y2]));
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
    // Instance mask (§16.2.3) — present only when `--masks` ran. RLE inside the
    // JSONL keeps the 64² grid self-describing (no side file). `remembered` object-
    // permanence holds carry no fresh mask.
    if let Some(mask) = &o.mask {
        m.insert("mask".into(), mask_rle_json(mask));
    }
    serde_json::Value::Object(m)
}

/// Encode a low-res [`mvm_core::InstanceMask`] as run-length JSON: alternating run
/// lengths over the row-major grid, **starting with a run of 0s** (first run is 0
/// if the grid opens on a set cell). Reversible; compact for a sparse 64² grid.
fn mask_rle_json(mask: &mvm_core::InstanceMask) -> serde_json::Value {
    let mut rle: Vec<u32> = Vec::new();
    let mut cur: u8 = 0; // runs start counting the 0 symbol
    let mut run: u32 = 0;
    for &b in mask.bits() {
        let v = (b != 0) as u8;
        if v == cur {
            run += 1;
        } else {
            rle.push(run);
            cur = v;
            run = 1;
        }
    }
    rle.push(run);
    serde_json::json!({
        "res": mvm_core::MASK_RES,
        "coverage": mask.frame_coverage(),
        "rle": rle,
    })
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
    fn object_mask_is_rle_encoded_and_reversible() {
        use mvm_core::mask::{InstanceMask, MASK_RES};
        // A mask with the top-left 8×8 block set.
        let mut bits = vec![0u8; MASK_RES * MASK_RES];
        for gy in 0..8 {
            for gx in 0..8 {
                bits[gy * MASK_RES + gx] = 1;
            }
        }
        let mask = InstanceMask::from_bits(bits.clone());

        let now = Instant::now();
        let mut world = WorldState::new(now);
        let mut o = bare_object(10001, "cup", now);
        o.mask = Some(mask);
        world.objects.push(o);
        let obs = route(&world, &ctx());

        let mj = &obs.entities[0].attributes["mask"];
        assert_eq!(mj["res"].as_u64().unwrap() as usize, MASK_RES);
        assert!((mj["coverage"].as_f64().unwrap() - 64.0 / (MASK_RES * MASK_RES) as f64).abs() < 1e-6);

        // Decode the RLE and confirm it reproduces the original grid exactly.
        let rle: Vec<u32> = mj["rle"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let mut decoded = Vec::with_capacity(MASK_RES * MASK_RES);
        let mut sym = 0u8;
        for run in rle {
            decoded.extend(std::iter::repeat(sym).take(run as usize));
            sym ^= 1;
        }
        assert_eq!(decoded, bits, "RLE must round-trip the mask grid");
    }

    #[test]
    fn no_mask_key_when_object_has_no_mask() {
        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.objects.push(bare_object(10001, "cup", now)); // mask: None
        let obs = route(&world, &ctx());
        assert!(obs.entities[0].attributes.get("mask").is_none());
    }

    #[test]
    fn face_bbox_emitted_as_label_when_matched() {
        let now = Instant::now();
        let mut with_face = bare_person(1, now);
        with_face.face_bbox = Some(BBox { x1: 0.4, y1: 0.15, x2: 0.52, y2: 0.30 });
        let without = bare_person(2, now); // face_bbox None
        let mut world = WorldState::new(now);
        world.people = vec![with_face, without];
        let obs = route(&world, &ctx());

        let p1 = obs.entities.iter().find(|e| e.entity_id == "p1").unwrap();
        let fb: Vec<f64> = p1.attributes["face_bbox"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let want = [0.4, 0.15, 0.52, 0.30];
        assert_eq!(fb.len(), 4);
        for (g, w) in fb.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "face_bbox {g} ≉ {w}");
        }
        let p2 = obs.entities.iter().find(|e| e.entity_id == "p2").unwrap();
        assert!(p2.attributes.get("face_bbox").is_none(), "no face → no label");
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

    // --- §16.7 relations / activities / events enrichment ---

    #[test]
    fn relations_emit_zone_and_edge_records_with_entity_ids() {
        let world = WorldState::new(Instant::now());
        let rel = SceneRelations {
            zones: vec![ZoneRel {
                persistent_id: Some(3),
                label: Some("workstation".into()),
                members: vec![0, 10001],
            }],
            edges: vec![EdgeRel { a: 0, b: 10001, rel: "near".into() }],
        };
        let obs = route_with_relations(&world, &ctx(), &rel);

        let zone = obs.relations.iter().find(|r| r["kind"] == "zone").expect("zone record");
        assert_eq!(zone["label"], "workstation");
        assert_eq!(zone["persistent_id"], 3);
        assert_eq!(zone["members"], serde_json::json!(["p0", "o10001"]));

        let edge = obs.relations.iter().find(|r| r["kind"] == "edge").expect("edge record");
        assert_eq!(edge["a"], "p0");
        assert_eq!(edge["b"], "o10001");
        assert_eq!(edge["rel"], "near");
    }

    #[test]
    fn anonymous_zone_carries_null_persistent_id_and_label() {
        let world = WorldState::new(Instant::now());
        let rel = SceneRelations {
            zones: vec![ZoneRel { persistent_id: None, label: None, members: vec![1, 10] }],
            edges: Vec::new(),
        };
        let obs = route_with_relations(&world, &ctx(), &rel);
        let zone = &obs.relations[0];
        assert!(zone["persistent_id"].is_null(), "anonymous component → null id");
        assert!(zone["label"].is_null());
        assert_eq!(zone["members"], serde_json::json!(["p1", "p10"]));
    }

    #[test]
    fn events_emit_inline_episode_records() {
        let mut world = WorldState::new(Instant::now());
        world.events.push(mvm_core::ActivityEvent {
            label: "seated_at_workstation".into(),
            participants: vec![0],
            started_at: 10.0,
            ended_at: 90.0,
            duration_s: 80.0,
            peak_confidence: 0.9,
            stable_participants: true,
        });
        let obs = route(&world, &ctx());
        assert_eq!(obs.events.len(), 1);
        let e = &obs.events[0];
        assert_eq!(e["kind"], "episode");
        assert_eq!(e["label"], "seated_at_workstation");
        assert_eq!(e["participants"], serde_json::json!(["p0"]));
        assert_eq!(e["duration_s"], 80.0);
        assert_eq!(e["stable"], true);
    }

    #[test]
    fn flat_activities_map_participants_and_stability() {
        let mut world = WorldState::new(Instant::now());
        // Reserved gallery id 0 → stable; ephemeral anon id 12 → unstable.
        world.activities.push(Activity { label: "typing".into(), confidence: 0.8, participants: vec![0] });
        world.activities.push(Activity { label: "gesturing".into(), confidence: 0.5, participants: vec![12] });
        let obs = route(&world, &ctx());
        assert_eq!(obs.activities.len(), 2);

        let typing = obs.activities.iter().find(|a| a["label"] == "typing").unwrap();
        assert_eq!(typing["participants"], serde_json::json!(["p0"]));
        assert_eq!(typing["stable"], true);

        let gest = obs.activities.iter().find(|a| a["label"] == "gesturing").unwrap();
        assert_eq!(gest["participants"], serde_json::json!(["p12"]));
        assert_eq!(gest["stable"], false);
    }

    #[test]
    fn journal_instances_win_over_flat_with_temporal_fields() {
        // Drive the FSM to a confirmed instance (mirrors the flat scene-log path).
        let mut j = mvm_core::ActivityJournal::new();
        let raw = [Activity { label: "typing".into(), confidence: 0.8, participants: vec![0, 10001] }];
        for f in 0..4u64 {
            j.update(&raw, f as f64 * 0.1, f);
        }
        let inst = j.active().next().expect("confirmed").clone();

        let mut world = WorldState::new(Instant::now());
        world.activities = raw.to_vec(); // flat list present too
        world.activity_instances.push(inst);
        let obs = route(&world, &ctx());

        assert_eq!(obs.activities.len(), 1, "journal instances preferred over the flat list");
        let a = &obs.activities[0];
        assert_eq!(a["label"], "typing");
        assert_eq!(a["participants"], serde_json::json!(["p0", "o10001"]));
        assert!(a["started_at"].is_number(), "temporal field present from the journal");
        assert!(a["duration_s"].is_number());
        assert_eq!(a["state"], "active");
        assert_eq!(a["stable"], true, "reserved 0 + object 10001 → stable participants");
    }

    #[test]
    fn route_without_relations_leaves_relations_empty_but_fills_temporal() {
        // Back-compat: plain route() has no relational context, but still surfaces
        // activities/events already present in the WorldState.
        let mut world = WorldState::new(Instant::now());
        world.activities.push(Activity { label: "typing".into(), confidence: 0.8, participants: vec![0] });
        let obs = route(&world, &ctx());
        assert!(obs.relations.is_empty(), "no SceneRelations → no zone/edge records");
        assert_eq!(obs.activities.len(), 1, "activities still routed from world");
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
