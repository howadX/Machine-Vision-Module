//! FusionAgent (phase 8.C.7) — the cross-group step. It is the only agent that
//! sees **both** people and objects plus the shared depth map, so depth-anchoring
//! lives here (decision 8.C.5): the relative depth map is anchored to people's
//! metric torso-Z, giving objects a metric position in the same frame → real
//! person↔object `Near` edges (built later by the StateFileWriter / 8.F).
//!
//! It reads the depth handle from the shared `DepthSlot`, the frame dimensions
//! from the latest frame (the same pixel space the body-primary 3D uses), and
//! read-modify-writes `WorldState.objects` under a single write lock so a
//! concurrent `SceneAgent` swap can't clobber the anchored positions.

use crate::agents::DepthSlot;
use mvm_core::{ObjectState, PersonState, Position3D, WorldState};
use mvm_mesh::{Agent, AgentHealth, MeshContext};

pub(crate) struct FusionAgent {
    config: mvm_core::Config,
    depth_slot: DepthSlot,
}

impl FusionAgent {
    pub(crate) fn new(config: mvm_core::Config, depth_slot: DepthSlot) -> Self {
        Self { config, depth_slot }
    }
}

impl Agent for FusionAgent {
    fn name(&self) -> String {
        "fusion".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let pinhole = mvm_core::space::Pinhole::from_config(&self.config);
        let mut ticks = 0u64;
        while ctx.running() {
            // Cheaply grab the latest depth handle (clone the inner Arc, release
            // the slot lock immediately) + the frame's pixel dimensions.
            let depth = self.depth_slot.lock().ok().and_then(|g| g.clone());
            let dims = ctx
                .frames
                .latest()
                .map(|f| (f.image.width() as f32, f.image.height() as f32));
            if let (Some(depth), Some((fw, fh))) = (depth, dims) {
                if let Ok(mut w) = ctx.world.write() {
                    let WorldState { people, objects, .. } = &mut *w;
                    anchor_objects(people.as_slice(), objects.as_mut_slice(), &depth, &pinhole, fw, fh);
                    ticks += 1;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        AgentHealth { name: self.name(), ticks, errors: 0 }
    }
}

/// Anchor objects to metric 3D using people as the scale reference (phase 8.E.2,
/// now on the fusion step). For each person with a metric torso-Z and a depth
/// sample, `k_i = Z_i·d_i`; the median `k` converts each object's depth value to
/// metric `Z = k / d_obj`, back-projected through the same pinhole. With no
/// person anchor, objects are left depthless (overlap-only) — no fabricated
/// distance. Mirrors the synchronous `anchor_objects_depth`.
pub(crate) fn anchor_objects(
    people: &[PersonState],
    objects: &mut [ObjectState],
    depth: &mvm_depth::DepthMap,
    pinhole: &mvm_core::space::Pinhole,
    fw: f32,
    fh: f32,
) {
    const OBJ_DEPTH_CONF: f32 = 0.4;
    const Z_MIN: f32 = 0.3;
    const Z_MAX: f32 = 8.0;

    let mut ks: Vec<f32> = Vec::new();
    for p in people {
        if let Some(q) = p.position_3d {
            if q.z > 0.0 {
                if let Some(d) = depth.median_in_bbox(&p.bbox) {
                    if d > 1e-3 {
                        ks.push(q.z * d);
                    }
                }
            }
        }
    }
    if ks.is_empty() {
        for o in objects.iter_mut() {
            o.position_3d = None;
        }
        return;
    }
    ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ks[ks.len() / 2];

    for o in objects.iter_mut() {
        match depth.median_in_bbox(&o.bbox) {
            Some(d) if d > 1e-3 => {
                let z = (k / d).clamp(Z_MIN, Z_MAX);
                let cx = (o.bbox.x1 + o.bbox.x2) * 0.5 * fw;
                let cy = (o.bbox.y1 + o.bbox.y2) * 0.5 * fh;
                let p3 = pinhole.pixel_to_3d(cx, cy, z);
                o.position_3d = Some(Position3D { x: p3[0], y: p3[1], z: p3[2], confidence: OBJ_DEPTH_CONF });
            }
            _ => o.position_3d = None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{BBox, Config};
    use std::time::Instant;

    fn pinhole() -> mvm_core::space::Pinhole {
        mvm_core::space::Pinhole::from_config(&Config::default())
    }

    fn person(z: f32) -> PersonState {
        PersonState {
            track_id: 10,
            bbox: BBox { x1: 0.3, y1: 0.2, x2: 0.5, y2: 0.7 },
            identity: None,
            emotion: None,
            posture: None,
            position_3d: Some(Position3D { x: 0.0, y: 0.0, z, confidence: 0.5 }),
            emotion_distribution: None,
            last_seen: Instant::now(),
        }
    }

    fn object() -> ObjectState {
        ObjectState {
            id: 10000,
            class: "cup".into(),
            bbox: BBox { x1: 0.45, y1: 0.45, x2: 0.6, y2: 0.6 },
            confidence: 0.8,
            age: 2,
            position_3d: Some(Position3D { x: 9.0, y: 9.0, z: 9.0, confidence: 0.4 }), // stale
            last_seen: Instant::now(),
        }
    }

    /// Uniform depth map → every `median_in_bbox` returns the same value.
    fn uniform_depth(v: f32) -> mvm_depth::DepthMap {
        mvm_depth::DepthMap { data: vec![v; 64], w: 8, h: 8 }
    }

    #[test]
    fn no_people_leaves_objects_depthless() {
        let mut objs = vec![object()];
        anchor_objects(&[], &mut objs, &uniform_depth(0.5), &pinhole(), 640.0, 480.0);
        assert!(objs[0].position_3d.is_none(), "no anchor → depthless");
    }

    #[test]
    fn person_anchor_sets_object_metric_z() {
        // person Z=2.0, d_person=0.5 → k=1.0; object d=0.5 → Z = 1.0/0.5 = 2.0.
        let mut objs = vec![object()];
        anchor_objects(&[person(2.0)], &mut objs, &uniform_depth(0.5), &pinhole(), 640.0, 480.0);
        let q = objs[0].position_3d.expect("object anchored");
        assert!((q.z - 2.0).abs() < 1e-3, "metric Z from people anchor, got {}", q.z);
        assert!((q.confidence - 0.4).abs() < 1e-6);
    }

    #[test]
    fn degenerate_object_bbox_yields_no_position() {
        // A zero-area object bbox → median_in_bbox None → depthless even with anchor.
        let mut o = object();
        o.bbox = BBox { x1: 0.5, y1: 0.5, x2: 0.5, y2: 0.5 };
        let mut objs = vec![o];
        anchor_objects(&[person(2.0)], &mut objs, &uniform_depth(0.5), &pinhole(), 640.0, 480.0);
        assert!(objs[0].position_3d.is_none(), "degenerate bbox → None");
    }
}
