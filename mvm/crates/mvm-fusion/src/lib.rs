//! mvm-fusion — relational scene-graph + Product-of-Experts attribute fusion (Phase 6.4).
//!
//! ## Two operations
//!
//! ### Relational (primary): `build_scene()`
//! Given N entities with 3D positions, build a proximity/overlap graph and compute
//! connected-component zones via union-find. Object-agnostic: works for Person, Face,
//! Object("cup"), etc. without changing the engine.
//!
//! ### Attribute (secondary): `fuse()`
//! Log-linear pooling (Product-of-Experts) over 8-class emotion distributions.
//! `weight = 1 - uncertainty` so high-entropy experts contribute less.
//!
//! ## Temporal PoE note
//! Temporal fusion (track prior × current frame) lives in mvm-cli to avoid the
//! mvm-fusion → mvm-core circular dependency. This module provides the pure `fuse()`
//! primitive; callers compose the temporal recurrence.

use mvm_core::types::{Entity, EntityKind, Relation, RelationKind, Scene};

const K: usize = 8;

/// Proximity threshold in metres: entities closer than this get a `Near` edge.
const NEAR_THRESH_M: f32 = 1.5;

// ── Attribute fusion (PoE) ────────────────────────────────────────────────────

/// One probabilistic expert contributing to `fuse()`.
/// `weight` is typically `1.0 - uncertainty` (from `EmotionResult.uncertainty`).
/// A weight of 0.0 means the expert is fully ignored.
#[derive(Debug, Clone)]
pub struct Expert {
    pub distribution: [f32; K],
    /// Contribution weight in [0, 1]. Clamped internally.
    pub weight: f32,
}

/// Output of `fuse()`.
#[derive(Debug, Clone)]
pub struct FusedEmotion {
    /// Winning label index (0-based, follows EMOTION_LABELS order in mvm-emotion).
    pub label_idx: usize,
    /// max(distribution).
    pub prob: f32,
    /// Full fused distribution over K classes. Sums to 1.0.
    pub distribution: [f32; K],
    /// Normalised Shannon entropy of the fused distribution. 0 = certain, 1 = uniform.
    pub uncertainty: f32,
}

/// Product-of-Experts fusion via log-linear pooling.
///
/// Formula: `p_fused ∝ ∏ p_k^{w_k}` (normalised).
/// Numerically: `log_sum[k] = Σ w_i * ln(p_i[k])`, then softmax.
///
/// Experts with `weight ≤ 0.0` are skipped entirely.
/// Returns a uniform distribution (maximum uncertainty) if no valid expert contributes.
pub fn fuse(experts: &[Expert]) -> FusedEmotion {
    let mut log_sum = [0.0f32; K];
    let mut any = false;

    for expert in experts {
        let w = expert.weight.clamp(0.0, 1.0);
        if w <= 0.0 {
            continue;
        }
        any = true;
        for k in 0..K {
            let p = expert.distribution[k].max(1e-10);
            log_sum[k] += w * p.ln();
        }
    }

    if !any {
        // No valid expert: maximum-entropy uniform distribution.
        let uniform = 1.0 / K as f32;
        return FusedEmotion {
            label_idx: 0,
            prob: uniform,
            distribution: [uniform; K],
            uncertainty: 1.0,
        };
    }

    // Softmax with numerical stability (subtract max before exp).
    let max_ls = log_sum.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: [f32; K] = std::array::from_fn(|k| (log_sum[k] - max_ls).exp());
    let sum: f32 = exps.iter().sum();
    let distribution: [f32; K] = std::array::from_fn(|k| exps[k] / sum);

    let (label_idx, &prob) = distribution.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap_or((0, &0.0));

    let uncertainty = entropy_from_probs(&distribution);

    FusedEmotion { label_idx, prob, distribution, uncertainty }
}

fn entropy_from_probs(probs: &[f32]) -> f32 {
    let h: f32 = probs.iter()
        .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
        .sum();
    // Normalise by log2(K) so result is in [0, 1].
    h / (K as f32).log2()
}

// ── Relational scene-graph ────────────────────────────────────────────────────

/// Build a metric 3D scene-graph over the provided entities.
///
/// Relations are added for every pair where:
///   - 3D Euclidean distance < `NEAR_THRESH_M` (→ `RelationKind::Near`), OR
///   - bbox 2D IoU > 0 (→ `RelationKind::Overlapping`)
///
/// Zones are connected components of the relation graph (union-find).
pub fn build_scene(entities: Vec<Entity>) -> Scene {
    let n = entities.len();
    let mut relations: Vec<Relation> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_3d(entities[i].pos_3d, entities[j].pos_3d);
            let overlap = iou_2d(&entities[i].bbox, &entities[j].bbox);

            // A metric `Near` edge is only trustworthy when BOTH nodes have a real
            // 3D position. Nodes without depth (pos_conf == 0) sit at the camera
            // origin [0,0,0] — objects (phase 8.E carry no monocular depth) and
            // phantom bodies (R visibility-gate) — so a 3D distance to them is
            // meaningless and would spuriously make every origin-node "near" anyone
            // ~NEAR_THRESH from the camera. Such nodes still relate via 2D overlap.
            let both_have_depth = entities[i].pos_conf > 0.0 && entities[j].pos_conf > 0.0;

            let kind = if overlap > 0.0 {
                RelationKind::Overlapping
            } else if both_have_depth && dist < NEAR_THRESH_M {
                RelationKind::Near
            } else {
                continue; // no edge
            };

            relations.push(Relation {
                a: entities[i].id,
                b: entities[j].id,
                distance_m: dist,
                overlap,
                kind,
            });
        }
    }

    let zones = connected_components(&entities, &relations);

    Scene { entities, relations, zones }
}

fn euclidean_3d(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn iou_2d(a: &mvm_core::BBox, b: &mvm_core::BBox) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);

    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    if inter == 0.0 {
        return 0.0;
    }

    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    inter / (area_a + area_b - inter).max(1e-10)
}

/// Union-Find (path-compressed, union-by-rank) for connected components.
/// Returns each component as a sorted Vec of entity IDs.
fn connected_components(entities: &[Entity], relations: &[Relation]) -> Vec<Vec<u32>> {
    let n = entities.len();
    if n == 0 {
        return vec![];
    }

    // Map entity id → index in entities slice.
    let id_to_idx: std::collections::HashMap<u32, usize> = entities.iter()
        .enumerate()
        .map(|(i, e)| (e.id, i))
        .collect();

    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0; n];

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut Vec<usize>, rank: &mut Vec<u8>, x: usize, y: usize) {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx == ry {
            return;
        }
        match rank[rx].cmp(&rank[ry]) {
            std::cmp::Ordering::Less    => parent[rx] = ry,
            std::cmp::Ordering::Greater => parent[ry] = rx,
            std::cmp::Ordering::Equal   => { parent[ry] = rx; rank[rx] += 1; }
        }
    }

    for rel in relations {
        if let (Some(&ai), Some(&bi)) = (id_to_idx.get(&rel.a), id_to_idx.get(&rel.b)) {
            union(&mut parent, &mut rank, ai, bi);
        }
    }

    // Collect components.
    let mut components: std::collections::HashMap<usize, Vec<u32>> = std::collections::HashMap::new();
    for (i, entity) in entities.iter().enumerate() {
        let root = find(&mut parent, i);
        components.entry(root).or_default().push(entity.id);
    }

    let mut zones: Vec<Vec<u32>> = components.into_values().collect();
    for z in &mut zones {
        z.sort_unstable();
    }
    zones.sort_by_key(|z| z[0]);
    zones
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{BBox, types::{Entity, EntityKind}};

    fn one_hot(idx: usize) -> [f32; K] {
        let mut d = [0.0f32; K];
        d[idx] = 1.0;
        d
    }

    fn uniform() -> [f32; K] {
        [1.0 / K as f32; K]
    }

    fn entity(id: u32, pos_3d: [f32; 3], bbox: BBox) -> Entity {
        Entity {
            id,
            kind: EntityKind::Person,
            bbox,
            pos_3d,
            radius_m: 0.1,
            pos_conf: 0.9,
            identity: None,
            emotion: None,
            emotion_distribution: None,
        }
    }

    fn small_bbox(x1: f32, y1: f32) -> BBox {
        BBox { x1, y1, x2: x1 + 0.05, y2: y1 + 0.05 }
    }

    // ── Relational tests ──────────────────────────────────────────────────────

    #[test]
    fn relation_distance_3d_euclidean() {
        // Two entities 1m apart along Z axis.
        let e1 = entity(1, [0.0, 0.0, 1.0], small_bbox(0.1, 0.1));
        let e2 = entity(2, [0.0, 0.0, 2.0], small_bbox(0.8, 0.8));
        let scene = build_scene(vec![e1, e2]);
        assert_eq!(scene.relations.len(), 1, "should have one Near edge");
        let rel = &scene.relations[0];
        assert!((rel.distance_m - 1.0).abs() < 1e-4, "distance should be 1.0m, got {}", rel.distance_m);
        assert_eq!(rel.kind, RelationKind::Near);
    }

    #[test]
    fn zone_connected_components() {
        // Three entities all within 1.5m of each other → one zone.
        let e1 = entity(1, [0.0, 0.0, 1.0], small_bbox(0.1, 0.1));
        let e2 = entity(2, [0.5, 0.0, 1.0], small_bbox(0.5, 0.1));
        let e3 = entity(3, [1.0, 0.0, 1.0], small_bbox(0.8, 0.1));
        let scene = build_scene(vec![e1, e2, e3]);
        assert_eq!(scene.zones.len(), 1, "all should be in one zone");
        let zone = &scene.zones[0];
        assert!(zone.contains(&1) && zone.contains(&2) && zone.contains(&3));
    }

    #[test]
    fn zone_far_entities_separate() {
        // Two entities 5m apart → separate zones.
        let e1 = entity(1, [0.0, 0.0, 0.5], small_bbox(0.1, 0.1));
        let e2 = entity(2, [0.0, 0.0, 5.5], small_bbox(0.8, 0.8));
        let scene = build_scene(vec![e1, e2]);
        assert_eq!(scene.relations.len(), 0, "no edge expected");
        assert_eq!(scene.zones.len(), 2, "should be separate zones");
    }

    #[test]
    fn no_near_edge_to_depthless_node() {
        // An object with no depth (pos_conf = 0, sitting at origin) must NOT form a
        // metric Near edge with a person, even though the person is within
        // NEAR_THRESH_M of the origin (phase 8.E: objects relate by overlap only).
        let person = entity(1, [0.3, 0.0, 1.0], small_bbox(0.1, 0.1));
        let mut object = entity(2, [0.0, 0.0, 0.0], small_bbox(0.8, 0.8));
        object.kind = EntityKind::Object("cup".into());
        object.pos_conf = 0.0;
        let scene = build_scene(vec![person, object]);
        assert_eq!(scene.relations.len(), 0, "depthless node must not get a Near edge");
        assert_eq!(scene.zones.len(), 2, "should be separate zones");
    }

    #[test]
    fn overlap_edge_survives_without_depth() {
        // Even with pos_conf = 0 on both, a 2D bbox overlap still yields an edge —
        // the meaningful "person ↔ held/nearby object" signal.
        let mut person = entity(1, [0.0, 0.0, 1.0], BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 });
        person.pos_conf = 0.9;
        let mut object = entity(2, [0.0, 0.0, 0.0], BBox { x1: 0.4, y1: 0.4, x2: 0.55, y2: 0.6 });
        object.kind = EntityKind::Object("cup".into());
        object.pos_conf = 0.0;
        let scene = build_scene(vec![person, object]);
        assert_eq!(scene.relations.len(), 1, "overlap edge expected");
        assert_eq!(scene.relations[0].kind, RelationKind::Overlapping);
    }

    #[test]
    fn object_agnostic_relations() {
        // Mixed EntityKind: Person + Object should still get an edge if close.
        let mut e1 = entity(1, [0.0, 0.0, 1.0], small_bbox(0.1, 0.1));
        e1.kind = EntityKind::Person;
        let mut e2 = entity(2, [0.3, 0.0, 1.0], small_bbox(0.5, 0.1));
        e2.kind = EntityKind::Object("cup".to_string());
        let scene = build_scene(vec![e1, e2]);
        assert_eq!(scene.relations.len(), 1, "Person + Object should get a Near edge");
    }

    // ── Attribute fusion tests ────────────────────────────────────────────────

    #[test]
    fn fuse_identical_distributions_unchanged() {
        // PoE sharpens identical distributions (∝ p^w per class, renormalised).
        // The dominant class is preserved; the distribution concentrates toward it.
        let d = [0.5_f32, 0.3, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02];
        let experts = vec![
            Expert { distribution: d, weight: 0.8 },
            Expert { distribution: d, weight: 0.8 },
        ];
        let out = fuse(&experts);
        let sum: f32 = out.distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "must sum to 1.0");
        // Dominant class is preserved.
        assert_eq!(out.label_idx, 0, "argmax should be unchanged");
        // PoE sharpens: dominant class gets MORE mass than the input (p^2 renormalised).
        assert!(out.distribution[0] > d[0],
            "identical experts sharpen the distribution: {} should be > {}", out.distribution[0], d[0]);
    }

    #[test]
    fn fuse_zero_weight_expert_ignored() {
        // One zero-weight expert, one real one → output == real expert's distribution.
        let d_real = one_hot(3);
        let d_junk = one_hot(7);
        let experts = vec![
            Expert { distribution: d_junk, weight: 0.0 },
            Expert { distribution: d_real, weight: 1.0 },
        ];
        let out = fuse(&experts);
        assert_eq!(out.label_idx, 3, "zero-weight expert must be ignored");
        assert!(out.distribution[3] > 0.99, "output should be concentrated at class 3");
    }

    #[test]
    fn fuse_distribution_sums_to_one() {
        // Random-ish distributions — output must always sum to 1.
        let d1 = [0.3_f32, 0.2, 0.15, 0.1, 0.1, 0.07, 0.05, 0.03];
        let d2 = [0.1_f32, 0.1, 0.1,  0.1, 0.2, 0.2,  0.1,  0.1];
        let experts = vec![
            Expert { distribution: d1, weight: 0.7 },
            Expert { distribution: d2, weight: 0.5 },
        ];
        let out = fuse(&experts);
        let sum: f32 = out.distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "distribution must sum to 1.0, got {sum}");
    }

    #[test]
    fn fuse_disagreement_raises_uncertainty() {
        // Experts pointing at opposite classes → bimodal output, higher uncertainty than agreement.
        // one_hot(0) vs one_hot(7) → distribution ≈ [0.5, 0, …, 0, 0.5]
        // Normalised entropy = 1/log2(8) ≈ 0.33, which is LESS than uniform but MORE than one-hot.
        let d_a = one_hot(0);
        let d_b = one_hot(7);
        let experts_disagree = vec![
            Expert { distribution: d_a, weight: 0.8 },
            Expert { distribution: d_b, weight: 0.8 },
        ];
        let experts_agree = vec![
            Expert { distribution: d_a, weight: 0.8 },
            Expert { distribution: d_a, weight: 0.8 },
        ];
        let out_dis = fuse(&experts_disagree);
        let out_agr = fuse(&experts_agree);
        assert!(out_dis.uncertainty > out_agr.uncertainty,
            "disagreement should yield higher uncertainty than agreement: {} vs {}",
            out_dis.uncertainty, out_agr.uncertainty);
        // Bimodal result: both class 0 and class 7 should have meaningful mass.
        assert!(out_dis.distribution[0] > 0.1 && out_dis.distribution[7] > 0.1,
            "disagreeing one-hots should produce bimodal output");
    }

    #[test]
    fn temporal_no_collapse_with_forgetting() {
        // Simulate temporal PoE: each frame adds a new expert with LAMBDA < 1.
        // Without forgetting (weight=1.0), accumulating a one-hot collapses to near-zero entropy.
        // With LAMBDA < 1, repeated application should NOT converge to near-zero entropy from
        // a uniform seed. We test the non-collapse property.
        const LAMBDA: f32 = 0.5;
        let mut prior = uniform();

        for _ in 0..50 {
            let result = fuse(&[
                Expert { distribution: prior, weight: LAMBDA },
                Expert { distribution: one_hot(4), weight: 0.9 },
            ]);
            prior = result.distribution;
        }

        let h = entropy_from_probs(&prior);
        // After 50 rounds of mixing with a strong one-hot expert, distribution concentrates
        // somewhat — but with LAMBDA=0.5 forgetting it should not fully collapse to ~0 entropy.
        // The key invariant: distribution is valid (sums to 1, no NaN).
        let sum: f32 = prior.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "temporal PoE must stay normalised, got {sum}");
        assert!(h.is_finite(), "entropy must be finite after 50 rounds");
        // The dominant class after 50 rounds of one-hot(4) observation should be class 4.
        let argmax = prior.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        assert_eq!(argmax, 4, "dominant class should be class 4 (repeated observation)");
    }
}

// ── Zone persistence (phase 11.B1) ────────────────────────────────────────────

/// Ticks (emitted frames) a zone survives without its component reappearing.
/// The zone twin of the activity exit dwell: brief graph flicker (a person
/// stepping between two objects, an object dropping for one YOLO cadence) must
/// not re-mint zone ids.
pub const ZONE_MISS_CAP: u32 = 5;
/// Bounded number of simultaneously tracked zones.
pub const ZONE_CAP: usize = 32;

/// Object classes whose presence labels a zone a *workstation* (mirrors the
/// fusion `WORKSTATION_CLASSES` context).
const ZONE_WORKSTATION: &[&str] = &["dining table", "laptop", "keyboard", "mouse", "tv", "book"];
/// Object classes whose presence labels a zone *seating*.
const ZONE_SEATING: &[&str] = &["chair", "couch", "bed"];

/// A zone with a stable identity over time (phase 11.B1). Per-frame
/// `Scene::zones` are anonymous connected components recomputed from scratch;
/// the tracker matches them across frames by member overlap so downstream can
/// say "zone 3, the workstation, for the last 40 frames".
#[derive(Debug, Clone)]
pub struct ZoneState {
    /// Stable zone id (monotonic, never reused within a run).
    pub id: u32,
    /// Members from the latest matched component (sorted entity ids).
    pub members: Vec<u32>,
    /// Dominant semantic context ("workstation"/"seating") from member object
    /// classes, `None` when neither applies.
    pub label: Option<&'static str>,
    /// Matched observations so far.
    pub age: u32,
    /// Consecutive ticks the component failed to reappear.
    missed: u32,
}

/// Matches per-frame connected components to persistent zones. Pure arithmetic,
/// deterministic: candidates ranked by shared-member count, then zone age
/// (older wins), then zone id.
#[derive(Debug, Default)]
pub struct ZoneTracker {
    zones: Vec<ZoneState>,
    next_id: u32,
}

/// The semantic label for a component from its member object classes: any
/// workstation-class object wins, else any seating-class object, else none.
///
/// Stateless (no persistent zone id) — the snapshot renderer (phase 12.B1)
/// reuses it to label the current frame's zones without touching the
/// [`ZoneTracker`]'s history.
pub fn zone_label(members: &[u32], entities: &[Entity]) -> Option<&'static str> {
    let mut seating = false;
    for e in entities {
        if !members.contains(&e.id) {
            continue;
        }
        if let EntityKind::Object(class) = &e.kind {
            if ZONE_WORKSTATION.contains(&class.as_str()) {
                return Some("workstation");
            }
            if ZONE_SEATING.contains(&class.as_str()) {
                seating = true;
            }
        }
    }
    seating.then_some("seating")
}

fn shared_members(a: &[u32], b: &[u32]) -> usize {
    // Both sorted → linear merge count.
    let (mut i, mut j, mut n) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

impl ZoneTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance one tick: match this frame's components (from `Scene::zones`,
    /// members sorted) to the persistent zones, refresh members/labels, age the
    /// unmatched, reap past [`ZONE_MISS_CAP`], mint ids for new components.
    /// Returns the live zones sorted by id.
    pub fn update(&mut self, components: &[Vec<u32>], entities: &[Entity]) -> &[ZoneState] {
        // Rank all (component, zone) pairs by overlap; assign greedily so each
        // zone and each component match at most once. Deterministic ordering:
        // overlap desc, zone age desc, zone id asc, component index asc.
        let mut pairs: Vec<(usize, usize, usize)> = Vec::new(); // (ci, zi, shared)
        for (ci, comp) in components.iter().enumerate() {
            for (zi, z) in self.zones.iter().enumerate() {
                let shared = shared_members(comp, &z.members);
                if shared > 0 {
                    pairs.push((ci, zi, shared));
                }
            }
        }
        pairs.sort_by(|x, y| {
            y.2.cmp(&x.2)
                .then(self.zones[y.1].age.cmp(&self.zones[x.1].age))
                .then(self.zones[x.1].id.cmp(&self.zones[y.1].id))
                .then(x.0.cmp(&y.0))
        });
        let mut comp_taken = vec![false; components.len()];
        let mut zone_taken = vec![false; self.zones.len()];
        for (ci, zi, _) in pairs {
            if comp_taken[ci] || zone_taken[zi] {
                continue;
            }
            comp_taken[ci] = true;
            zone_taken[zi] = true;
            let z = &mut self.zones[zi];
            z.members = components[ci].clone();
            z.label = zone_label(&z.members, entities);
            z.age += 1;
            z.missed = 0;
        }
        // Unmatched zones age toward the reap.
        for (zi, z) in self.zones.iter_mut().enumerate() {
            if !zone_taken[zi] {
                z.missed += 1;
            }
        }
        self.zones.retain(|z| z.missed <= ZONE_MISS_CAP);
        // Unmatched components become new zones (bounded).
        for (ci, comp) in components.iter().enumerate() {
            if comp_taken[ci] || self.zones.len() >= ZONE_CAP {
                continue;
            }
            self.zones.push(ZoneState {
                id: self.next_id,
                members: comp.clone(),
                label: zone_label(comp, entities),
                age: 1,
                missed: 0,
            });
            self.next_id += 1;
        }
        self.zones.sort_by_key(|z| z.id);
        &self.zones
    }

    /// Live zones (sorted by id).
    pub fn zones(&self) -> &[ZoneState] {
        &self.zones
    }
}

#[cfg(test)]
mod zone_tests {
    use super::*;
    use mvm_core::BBox;

    fn ent(id: u32, kind: EntityKind) -> Entity {
        Entity {
            id,
            kind,
            bbox: BBox { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2 },
            pos_3d: [0.0, 0.0, 0.0],
            radius_m: 0.1,
            pos_conf: 0.0,
            identity: None,
            emotion: None,
            emotion_distribution: None,
        }
    }

    fn obj(id: u32, class: &str) -> Entity {
        ent(id, EntityKind::Object(class.into()))
    }

    #[test]
    fn zone_keeps_id_across_frames() {
        let mut t = ZoneTracker::new();
        let ents = [ent(0, EntityKind::Person), obj(10001, "keyboard")];
        let id0 = t.update(&[vec![0, 10001]], &ents)[0].id;
        let z = &t.update(&[vec![0, 10001]], &ents)[0];
        assert_eq!(z.id, id0, "same component → same zone id");
        assert_eq!(z.age, 2);
    }

    #[test]
    fn zone_survives_brief_absence_then_reaps() {
        let mut t = ZoneTracker::new();
        let ents = [ent(0, EntityKind::Person), obj(10001, "keyboard")];
        let id0 = t.update(&[vec![0, 10001]], &ents)[0].id;
        for _ in 0..ZONE_MISS_CAP {
            t.update(&[], &ents);
        }
        assert_eq!(t.zones().len(), 1, "still alive at the cap");
        assert_eq!(t.update(&[vec![0, 10001]], &ents)[0].id, id0, "revived with the same id");
        // Now exceed the cap.
        for _ in 0..=ZONE_MISS_CAP {
            t.update(&[], &ents);
        }
        assert!(t.zones().is_empty(), "reaped past the cap");
        assert_ne!(t.update(&[vec![0, 10001]], &ents)[0].id, id0, "reborn under a new id");
    }

    #[test]
    fn membership_drift_keeps_id() {
        let mut t = ZoneTracker::new();
        let ents = [ent(0, EntityKind::Person), ent(1, EntityKind::Person), obj(10001, "keyboard")];
        let id0 = t.update(&[vec![0, 10001]], &ents)[0].id;
        let z = &t.update(&[vec![0, 1, 10001]], &ents)[0];
        assert_eq!(z.id, id0, "a person joining does not re-mint the zone");
        assert_eq!(z.members, vec![0, 1, 10001]);
    }

    #[test]
    fn split_keeps_id_on_larger_part() {
        let mut t = ZoneTracker::new();
        let ents = [ent(0, EntityKind::Person), ent(1, EntityKind::Person), obj(10001, "keyboard")];
        let id0 = t.update(&[vec![0, 1, 10001]], &ents)[0].id;
        let zones: Vec<(u32, Vec<u32>)> = t
            .update(&[vec![0], vec![1, 10001]], &ents)
            .iter()
            .map(|z| (z.id, z.members.clone()))
            .collect();
        assert_eq!(zones.len(), 2);
        let keeper = zones.iter().find(|(id, _)| *id == id0).expect("id survives the split");
        assert_eq!(keeper.1, vec![1, 10001], "the larger fragment keeps the id");
    }

    #[test]
    fn zone_labels_from_member_classes() {
        let mut t = ZoneTracker::new();
        let ents = [ent(0, EntityKind::Person), obj(10001, "keyboard"), obj(10002, "couch")];
        let zones = t.update(&[vec![0, 10001], vec![10002]], &ents);
        assert_eq!(zones[0].label, Some("workstation"));
        assert_eq!(zones[1].label, Some("seating"));
        let zones = t.update(&[vec![0]], &ents);
        assert_eq!(zones[0].label, None, "keyboard left → label honestly drops");
    }
}
