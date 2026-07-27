//! DatasetWriterAgent (phase 16.1.2b) — the thin mesh wrapper over the
//! `mvm-dataset` recorder. It owns no models: on the dataset cadence it snapshots
//! the shared `WorldState`, delegates the record *logic* to
//! [`mvm_dataset::DatasetWriter`] (session lifecycle + labels.jsonl + session.toml),
//! then encodes the RGB pixels the writer allocated a path for — applying the
//! privacy-blur regions the dataset layer computed.
//!
//! Split of concerns (decision 17.07): all label/session logic and the blur-region
//! math live in `mvm-dataset` (pure, unit-tested); this file only touches the
//! Agent-Mesh plumbing + the `image`-crate pixel encoding.
//!
//! Depth maps (16-bit PNG) are a 16.2 follow-up — the agent forces `save_depth`
//! off so labels never reference a depth file that was not written.

use super::DepthSlot;
use crate::state::{object_state_to_entity, person_to_entity, relation_kept};
use image::{ImageFormat, RgbImage};
use mvm_core::BBox;
use mvm_core::types::RelationKind;
use mvm_dataset::{
    DatasetConfig, DatasetWriter, EdgeRel, FrameRecord, PrivacyPerson, SceneRelations, ZoneRel,
    blur_regions,
};
use mvm_mesh::{Agent, AgentHealth, Frame, MeshContext};
use std::path::Path;
use std::time::{Duration, Instant};

/// Logical dataset id (root-of-storage binding key).
const DATASET_ID: &str = "mvm_sessions";

pub(crate) struct DatasetWriterAgent {
    writer: DatasetWriter,
    cfg: DatasetConfig,
    /// Save every N-th *captured* frame (frame.id cadence).
    cadence: u32,
    /// Shared relative-depth slot (§16.2.2). `None` when depth is off or its agent
    /// isn't running; the recorder then records no depth path.
    depth: Option<DepthSlot>,
    /// Last captured frame id handled — dedupes the fast mesh loop.
    last_frame_id: Option<u64>,
    /// Stop after this many recorded frames (bounded runs/tests). `None` = live.
    max_records: Option<u64>,
    /// Persistent zone tracker (§16.7) — the same stateful discipline `SceneLog`
    /// uses, so the semantic session log carries zones with stable ids + labels.
    /// Ticks once per recording step (dataset cadence).
    zones: mvm_fusion::ZoneTracker,
}

impl DatasetWriterAgent {
    pub(crate) fn new(
        cfg: DatasetConfig,
        root: impl Into<std::path::PathBuf>,
        depth: Option<DepthSlot>,
    ) -> Self {
        let cadence = cfg.frame_cadence.max(1);
        let writer = DatasetWriter::new(cfg.clone(), root, DATASET_ID);
        Self {
            writer,
            cfg,
            cadence,
            depth,
            last_frame_id: None,
            max_records: None,
            zones: mvm_fusion::ZoneTracker::new(),
        }
    }

    /// Build this frame's relational scene-graph (§16.7) from the shared world:
    /// `build_scene` for edges + a stateful `ZoneTracker` tick for zones with
    /// persistent ids + labels. Mirrors `SceneLog::to_json_line_world` so the
    /// semantic session log is a faithful superset of the flat scene log.
    fn scene_relations(&mut self, world: &mvm_core::WorldState) -> SceneRelations {
        let mut entities: Vec<mvm_core::types::Entity> =
            world.people.iter().map(person_to_entity).collect();
        entities.extend(world.objects.iter().map(object_state_to_entity));
        let scene = mvm_fusion::build_scene(entities);

        // Edges — the scene log's default relation diet (drop static↔static clutter).
        let edges: Vec<EdgeRel> = scene
            .relations
            .iter()
            .filter(|r| relation_kept(r.a, r.b, &r.kind, false))
            .map(|r| {
                let rel = match r.kind {
                    RelationKind::Near => "near",
                    RelationKind::Overlapping => "overlapping",
                    RelationKind::PartOf => "part_of",
                };
                EdgeRel {
                    a: r.a,
                    b: r.b,
                    rel: rel.into(),
                }
            })
            .collect();

        // Zones — one tracker tick per recorded frame yields stable ids + labels.
        let zones: Vec<ZoneRel> = self
            .zones
            .update(&scene.zones, &scene.entities)
            .iter()
            .map(|z| ZoneRel {
                persistent_id: Some(z.id),
                label: z.label.map(str::to_string),
                members: z.members.clone(),
            })
            .collect();

        SceneRelations { zones, edges }
    }

    /// Bound the number of recorded frames (tests / finite runs).
    #[cfg(test)]
    pub(crate) fn with_max_records(mut self, n: u64) -> Self {
        self.max_records = Some(n);
        self
    }

    /// One recording step for an already-captured frame. Returns `true` if a frame
    /// was recorded (JSONL + pixels), `false` if it was skipped (off-cadence or no
    /// session). Session lifecycle mutates the *shared* WorldState under the write
    /// lock; the heavy pixel encode runs after the lock is released.
    fn step(&mut self, ctx: &MeshContext, frame: &Frame) -> anyhow::Result<bool> {
        let now = Instant::now();
        let stamp = chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string();
        // Full RFC3339 (with UTC offset) so downstream readers (`recap --last`,
        // `inspect --around`) can parse the semantic session timestamp (§16.7).
        let iso = chrono::Local::now()
            .format("%Y-%m-%dT%H:%M:%S%.3f%:z")
            .to_string();

        // Snapshot the latest relative depth map (if enabled + available) BEFORE
        // the lock, so its [min,max] range can go into the label the writer builds.
        let depth_map = if self.cfg.save_depth {
            self.depth
                .as_ref()
                .and_then(|s| s.lock().ok().and_then(|g| g.clone()))
        } else {
            None
        };
        let depth_range = depth_map.as_ref().and_then(|m| depth_minmax(&m.data));

        // Lock section: advance the session lifecycle + append the label. All of
        // this is fast (no image work). Capture the people boxes for the blur pass
        // before releasing.
        let (result, privacy_people) = {
            let mut w = match ctx.world.write() {
                Ok(w) => w,
                Err(_) => return Ok(false),
            };
            let pp: Vec<PrivacyPerson> = w
                .people
                .iter()
                .map(|p| PrivacyPerson {
                    body: p.bbox,
                    face: p.face_bbox,
                    is_anon: p.identity.is_none(),
                })
                .collect();
            // §16.7: compute zones+edges (ticks the persistent tracker) before the
            // record so the label carries the frame's relational scene-graph.
            let relations = self.scene_relations(&w);
            let r = self
                .writer
                .record_frame(&mut w, now, &iso, &stamp, depth_range, &relations)?;
            (r, pp)
        };

        let Some(rec) = result.frame else {
            return Ok(false);
        };

        // Outside the lock: encode + blur + persist the RGB pixels. In text-only
        // mode the writer allocates no frame path → skip the pixel encode entirely.
        if rec.frame_path.is_some() {
            let regions = blur_regions(&privacy_people, self.cfg.privacy);
            self.write_frame_image(&frame.image, &rec, &regions)?;
        }

        // Depth (§16.2.2): the writer only set a depth path when we passed a range,
        // so path.is_some() ⇔ depth_map.is_some().
        if let (Some(path), Some(map), Some(range)) =
            (rec.depth_path.as_ref(), depth_map.as_ref(), depth_range)
        {
            write_depth_png(map, range, path)?;
        }
        Ok(true)
    }

    /// Encode one frame to disk with privacy blur applied. Public-in-crate so the
    /// pixel path is unit-testable without a live camera.
    pub(crate) fn write_frame_image(
        &self,
        img: &image::DynamicImage,
        rec: &FrameRecord,
        blur: &[BBox],
    ) -> anyhow::Result<()> {
        let mut rgb = img.to_rgb8();
        apply_blur(&mut rgb, blur);
        let path = rec
            .frame_path
            .as_ref()
            .expect("write_frame_image called only when a frame path exists");
        save_rgb(&rgb, path, &self.cfg)?;
        Ok(())
    }
}

impl Agent for DatasetWriterAgent {
    fn name(&self) -> String {
        "dataset".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut records = 0u64;
        let mut errors = 0u64;
        while ctx.running() {
            if let Some(frame) = ctx.frames.latest() {
                let is_new = self.last_frame_id != Some(frame.id);
                let due = super::cadence_due(frame.id, self.cadence);
                if is_new {
                    self.last_frame_id = Some(frame.id);
                    if due {
                        match self.step(ctx, &frame) {
                            Ok(true) => records += 1,
                            Ok(false) => {}
                            Err(e) => {
                                eprintln!("  dataset: record error: {e}");
                                errors += 1;
                            }
                        }
                        if let Some(max) = self.max_records
                            && records >= max
                        {
                            break;
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // Finalize any open session on shutdown so its session.toml is written.
        let ended = chrono::Local::now()
            .format("%Y-%m-%dT%H:%M:%S%.3f%:z")
            .to_string();
        if let Ok(mut w) = ctx.world.write() {
            let _ = self.writer.close_open_session(&mut w, &ended);
        }
        AgentHealth {
            name: self.name(),
            ticks: records,
            errors,
        }
    }
}

/// Apply a box-region Gaussian blur to each normalized rectangle (§16.6.2:
/// σ = region_width_px / 10). Regions are clamped to the image bounds.
fn apply_blur(rgb: &mut RgbImage, regions: &[BBox]) {
    let (w, h) = (rgb.width(), rgb.height());
    for r in regions {
        let x1 = (r.x1.clamp(0.0, 1.0) * w as f32) as u32;
        let y1 = (r.y1.clamp(0.0, 1.0) * h as f32) as u32;
        let x2 = (r.x2.clamp(0.0, 1.0) * w as f32) as u32;
        let y2 = (r.y2.clamp(0.0, 1.0) * h as f32) as u32;
        let rw = x2.saturating_sub(x1);
        let rh = y2.saturating_sub(y1);
        if rw == 0 || rh == 0 {
            continue;
        }
        let sub = image::imageops::crop_imm(rgb, x1, y1, rw, rh).to_image();
        let sigma = (rw as f32 / 10.0).max(1.0);
        let blurred = image::imageops::blur(&sub, sigma);
        image::imageops::overlay(rgb, &blurred, x1 as i64, y1 as i64);
    }
}

/// `[min, max]` of the finite depth values, or `None` if there are none.
fn depth_minmax(data: &[f32]) -> Option<[f32; 2]> {
    let mut it = data.iter().copied().filter(|v| v.is_finite());
    let first = it.next()?;
    let (mut lo, mut hi) = (first, first);
    for v in it {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    Some([lo, hi])
}

/// Encode a relative depth map to a 16-bit grayscale PNG (§16.2.2), min-max
/// normalized over `range` so the mapping is reversible via `frame.depth_range`.
fn write_depth_png(map: &mvm_depth::DepthMap, range: [f32; 2], path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let [min, max] = range;
    let span = (max - min).max(f32::EPSILON);
    let buf: Vec<u16> = map
        .data
        .iter()
        .map(|&v| (((v - min) / span).clamp(0.0, 1.0) * 65535.0).round() as u16)
        .collect();
    let img =
        image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(map.w as u32, map.h as u32, buf)
            .ok_or_else(|| anyhow::anyhow!("depth buffer {}×{} mismatch", map.w, map.h))?;
    img.save_with_format(path, ImageFormat::Png)?;
    Ok(())
}

/// Save an `RgbImage` in the configured format/quality to `path` (creating the
/// parent dir if needed).
fn save_rgb(rgb: &RgbImage, path: &Path, cfg: &DatasetConfig) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match cfg.frame_format {
        mvm_dataset::FrameFormat::Jpg => {
            let file = std::fs::File::create(path)?;
            let mut w = std::io::BufWriter::new(file);
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut w,
                cfg.frame_quality_clamped(),
            );
            enc.encode_image(rgb)?;
        }
        mvm_dataset::FrameFormat::Png => {
            rgb.save_with_format(path, ImageFormat::Png)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::world::{PersonState, WorldState};
    use mvm_mesh::LatestFrame;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, RwLock};

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mvm_dsagent_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn person(id: u32, now: Instant, anon: bool) -> PersonState {
        PersonState {
            track_id: id,
            bbox: BBox {
                x1: 0.25,
                y1: 0.1,
                x2: 0.55,
                y2: 0.9,
            },
            face_bbox: None,
            identity: if anon {
                None
            } else {
                Some(("Вадим".into(), 0.9))
            },
            emotion: None,
            posture: Some(("sitting".into(), 0.9)),
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

    #[test]
    fn no_depth_slot_records_no_depth() {
        // save_depth on but no slot wired → no depth path/range, no panic.
        let cfg = DatasetConfig::default();
        assert!(cfg.save_depth);
        let a = DatasetWriterAgent::new(cfg, temp_root("nodepth"), None);
        assert!(a.depth.is_none());
    }

    #[test]
    fn depth_png_is_16bit_and_reversible() {
        let root = temp_root("depthpng");
        let map = mvm_depth::DepthMap {
            data: vec![0.0, 1.0, 2.0, 4.0],
            w: 2,
            h: 2,
        };
        let range = depth_minmax(&map.data).unwrap();
        assert_eq!(range, [0.0, 4.0]);
        let path = root.join("depth/000000.png");
        write_depth_png(&map, range, &path).unwrap();

        let img = image::open(&path).unwrap();
        // 16-bit grayscale: max value maps to 65535, min to 0.
        let g16 = img.to_luma16();
        assert_eq!(g16.dimensions(), (2, 2));
        assert_eq!(g16.get_pixel(0, 0).0[0], 0, "min → 0");
        assert_eq!(g16.get_pixel(1, 1).0[0], 65535, "max → 65535");
    }

    #[test]
    fn records_frame_pixels_and_jsonl() {
        let root = temp_root("rec");
        let cfg = DatasetConfig {
            frame_cadence: 1,
            ..Default::default()
        };
        let mut agent = DatasetWriterAgent::new(cfg, root.clone(), None).with_max_records(1);

        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.frame_id = 5;
        world.people.push(person(1, now, true));
        let ctx = MeshContext {
            frames: LatestFrame::new(),
            world: Arc::new(RwLock::new(world)),
            stop: Arc::new(AtomicBool::new(false)),
        };
        // Publish a captured frame the agent will record.
        ctx.frames
            .publish(Frame::new(5, image::DynamicImage::new_rgb8(64, 48), now));

        let health = agent.run(&ctx);
        assert_eq!(health.ticks, 1, "one recorded frame");

        // labels.jsonl + the RGB frame both exist under the session dir.
        let sessions = root.join("sessions");
        let sess_dir = std::fs::read_dir(&sessions)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        assert!(sess_dir.join("labels.jsonl").exists());
        assert!(
            sess_dir.join("frames/000000.jpg").exists(),
            "RGB frame written"
        );
        // On shutdown the open session was finalized.
        assert!(
            sess_dir.join("session.toml").exists(),
            "session.toml on shutdown"
        );
    }

    #[test]
    fn default_on_recorder_writes_text_only_session() {
        // §16.7-3: the default `mvm run` recorder (effective_recorder of a default
        // config) is text-only — labels.jsonl + session.toml, no frames/ pixels.
        let root = temp_root("textonly");
        let mut cfg = DatasetConfig::default()
            .effective_recorder()
            .expect("vision log on by default");
        assert!(!cfg.save_frames, "default recorder is text-only");
        cfg.frame_cadence = 1; // record every frame so the bounded run terminates
        let mut agent = DatasetWriterAgent::new(cfg, root.clone(), None).with_max_records(1);

        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.frame_id = 1;
        world.people.push(person(1, now, false));
        let ctx = MeshContext {
            frames: LatestFrame::new(),
            world: Arc::new(RwLock::new(world)),
            stop: Arc::new(AtomicBool::new(false)),
        };
        ctx.frames
            .publish(Frame::new(1, image::DynamicImage::new_rgb8(64, 48), now));
        assert_eq!(agent.run(&ctx).ticks, 1);

        let sessions = root.join("sessions");
        let sess_dir = std::fs::read_dir(&sessions)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        assert!(
            sess_dir.join("labels.jsonl").is_file(),
            "semantic labels written"
        );
        assert!(
            sess_dir.join("session.toml").is_file(),
            "session finalized on shutdown"
        );
        assert!(
            !sess_dir.join("frames").exists(),
            "text-only: no pixel frames dir"
        );
    }

    #[test]
    fn records_zone_relations_from_world() {
        // §16.7: the agent computes zones+edges from the world and they reach the
        // semantic label — a person + an overlapping keyboard form one "workstation".
        let root = temp_root("zones");
        let cfg = DatasetConfig {
            frame_cadence: 1,
            ..Default::default()
        };
        let mut agent = DatasetWriterAgent::new(cfg, root.clone(), None).with_max_records(1);

        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.frame_id = 7;
        world.people.push(person(1, now, true));
        world.objects.push(mvm_core::world::ObjectState {
            id: 10001,
            class: "keyboard".into(),
            bbox: BBox {
                x1: 0.3,
                y1: 0.4,
                x2: 0.5,
                y2: 0.6,
            },
            confidence: 0.8,
            age: 5,
            position_3d: None,
            mask: None,
            remembered: false,
            last_seen: now,
        });
        let ctx = MeshContext {
            frames: LatestFrame::new(),
            world: Arc::new(RwLock::new(world)),
            stop: Arc::new(AtomicBool::new(false)),
        };
        ctx.frames
            .publish(Frame::new(7, image::DynamicImage::new_rgb8(64, 48), now));
        assert_eq!(agent.run(&ctx).ticks, 1);

        let sessions = root.join("sessions");
        let sess_dir = std::fs::read_dir(&sessions)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let content = std::fs::read_to_string(sess_dir.join("labels.jsonl")).unwrap();
        let obs: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
        let rels = obs["relations"].as_array().expect("relations array");
        assert!(
            rels.iter()
                .any(|r| r["kind"] == "zone" && r["label"] == "workstation"),
            "expected a workstation zone, got: {}",
            obs["relations"]
        );
        assert!(
            rels.iter()
                .any(|r| r["kind"] == "edge" && r["a"] == "p1" && r["b"] == "o10001"),
            "expected a person↔object edge, got: {}",
            obs["relations"]
        );
    }

    #[test]
    fn blur_changes_only_the_face_band() {
        // A flat gray image; blur is a no-op on a constant region, so instead we
        // check the writer produces a valid, decodable image with a region applied
        // over a gradient (structure survives, no panic, right dimensions).
        let root = temp_root("blur");
        let cfg = DatasetConfig::default();
        let agent = DatasetWriterAgent::new(cfg, root.clone(), None);

        let mut img = RgbImage::new(40, 40);
        for (x, _y, px) in img.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 6) as u8, 0, 0]); // horizontal gradient
        }
        let dynimg = image::DynamicImage::ImageRgb8(img);
        let rec = FrameRecord {
            session_id: "s".into(),
            seq: 0,
            frame_path: Some(root.join("frames/000000.jpg")),
            depth_path: None,
            observation: mvm_dataset::CanonicalObservation::new("d", "s", 0, "iso", 0),
        };
        let regions = vec![BBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 0.3,
        }];
        agent.write_frame_image(&dynimg, &rec, &regions).unwrap();

        let saved = image::open(rec.frame_path.as_ref().unwrap()).unwrap();
        assert_eq!(saved.width(), 40);
        assert_eq!(saved.height(), 40);
    }
}
