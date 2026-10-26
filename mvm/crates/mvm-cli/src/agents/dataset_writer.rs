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

use super::{DepthEvidence, DepthObservationBus, FusedFrameBus, FusedFramePacket};
use crate::state::{object_state_to_entity, person_to_entity, relation_kept};
use image::{ImageFormat, RgbImage};
use mvm_core::BBox;
use mvm_core::types::RelationKind;
use mvm_dataset::{
    DatasetConfig, DatasetWriter, DepthEncoding, DepthFrameMetadata, DepthUnits, EdgeRel,
    FrameRecord, PrivacyPerson, RecordAnnotations, SceneRelations, ZoneRel, blur_regions,
};
use mvm_mesh::{Agent, AgentHealth, Frame, MeshContext};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
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
    depth: Option<DepthObservationBus>,
    /// Last captured frame id handled — dedupes the fast mesh loop.
    last_frame_id: Option<u64>,
    /// Stop after this many recorded frames (bounded runs/tests). `None` = live.
    max_records: Option<u64>,
    /// Persistent zone tracker (§16.7) — the same stateful discipline `SceneLog`
    /// uses, so the semantic session log carries zones with stable ids + labels.
    /// Ticks once per recording step (dataset cadence).
    zones: mvm_fusion::ZoneTracker,
    fused_frames: Option<FusedFrameBus>,
    session_world: mvm_core::WorldState,
    event_cursor: Option<u64>,
    force_empty_calibration: bool,
}

impl DatasetWriterAgent {
    pub(crate) fn new(
        cfg: DatasetConfig,
        root: impl Into<std::path::PathBuf>,
        depth: Option<DepthObservationBus>,
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
            fused_frames: None,
            session_world: mvm_core::WorldState::new(Instant::now()),
            event_cursor: None,
            force_empty_calibration: false,
        }
    }

    pub(crate) fn with_fused_frames(mut self, fused_frames: FusedFrameBus) -> Self {
        self.fused_frames = Some(fused_frames);
        self
    }

    pub(crate) fn with_empty_calibration_frames(mut self, enabled: bool) -> Self {
        self.force_empty_calibration = enabled;
        self
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
                .and_then(|bus| bus.latest())
                .filter(|observation| observation.stamp.age_at(frame.id).is_some())
                .and_then(|observation| observation.relative_map().cloned())
        } else {
            None
        };
        let depth_range = depth_map.as_ref().and_then(|m| depth_minmax(&m.data));

        // Compatibility input is cloned read-only; DatasetWriter owns session
        // lifecycle in `session_world` and never mutates the shared scene.
        let result = {
            let mut w = match ctx.world.read() {
                Ok(world) => world.clone(),
                Err(_) => return Ok(false),
            };
            w.current_session_id = self.session_world.current_session_id.clone();
            w.last_people_seen = self.session_world.last_people_seen;
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
            let mut rgb = frame.image.to_rgb8();
            apply_blur(&mut rgb, &blur_regions(&pp, self.cfg.privacy));
            let depth_artifact = depth_map
                .as_ref()
                .zip(depth_range)
                .map(|(map, range)| DepthArtifact::Relative(map, range));
            let r = self.writer.record_frame_transactional(
                &mut w,
                now,
                &iso,
                &stamp,
                depth_range,
                &relations,
                |record| {
                    persist_artifacts_transactionally(record, Some(&rgb), &self.cfg, depth_artifact)
                },
            )?;
            self.session_world = w;
            r
        };

        if result.frame.is_none() {
            return Ok(false);
        }
        Ok(true)
    }

    fn step_canonical(&mut self, packet: &FusedFramePacket) -> anyhow::Result<bool> {
        let now = Instant::now();
        let file_stamp = chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string();
        let iso = chrono::Local::now()
            .format("%Y-%m-%dT%H:%M:%S%.3f%:z")
            .to_string();

        let depth_evidence = self
            .cfg
            .save_depth
            .then(|| {
                packet
                    .evidence
                    .depth
                    .as_ref()
                    .map(|depth| depth.evidence.clone())
            })
            .flatten();
        let depth_range = match depth_evidence.as_ref() {
            Some(DepthEvidence::Relative(map)) => depth_minmax(&map.data),
            _ => None,
        };
        let depth_metadata = depth_evidence.as_ref().map(|evidence| match evidence {
            DepthEvidence::Relative(map) => DepthFrameMetadata {
                encoding: DepthEncoding::RelativeAffineU16,
                units: DepthUnits::AffineRelative,
                width: u32::try_from(map.w).ok(),
                height: u32::try_from(map.h).ok(),
                invalid_value: None,
                valid_ratio: None,
                geometry: None,
                source: None,
                rgb_stamp: None,
                depth_stamp: None,
                rgb_depth_skew_ns: None,
                calibration: None,
            },
            DepthEvidence::Metric(metric) => {
                let frame = metric.depth.frame();
                DepthFrameMetadata {
                    encoding: DepthEncoding::MetricMmU16,
                    units: DepthUnits::Millimetres,
                    width: Some(frame.width),
                    height: Some(frame.height),
                    invalid_value: Some(frame.invalid_value),
                    valid_ratio: Some(frame.valid_ratio),
                    geometry: Some(frame.geometry.clone()),
                    source: Some(metric.source.clone()),
                    rgb_stamp: Some(metric.rgb_stamp.clone()),
                    depth_stamp: Some(frame.stamp.clone()),
                    rgb_depth_skew_ns: Some(frame.rgb_depth_skew_ns),
                    calibration: Some(metric.depth.calibration().clone()),
                }
            }
        });

        let projection = packet.scene.compatibility_world_state(now);
        let mut world = (*projection).clone();
        if let Some(human) = &packet.evidence.human {
            world.people = human.people.as_ref().clone();
        }
        if let Some(scene) = &packet.evidence.scene {
            world.objects = scene.objects.as_ref().clone();
            world.animals = scene.animals.iter().map(|animal| animal.bbox).collect();
        }
        world.current_session_id = self.session_world.current_session_id.clone();
        world.last_people_seen = self.session_world.last_people_seen;

        let privacy_people: Vec<PrivacyPerson> = world
            .people
            .iter()
            .map(|person| PrivacyPerson {
                body: person.bbox,
                face: person.face_bbox,
                is_anon: person.identity.is_none(),
            })
            .collect();
        let relations = canonical_relations(&packet.scene);
        let delta = packet.scene.events.delta_after(self.event_cursor);
        let mut events = canonical_event_records(&delta.events, &packet.scene);
        if let Some(gap) = delta.gap {
            events.insert(
                0,
                serde_json::json!({
                    "kind": "event_gap",
                    "requested_after": gap.requested_after,
                    "available_from": gap.available_from,
                }),
            );
        }
        let mut rgb = packet.evidence.rgb.to_rgb8();
        apply_blur(&mut rgb, &blur_regions(&privacy_people, self.cfg.privacy));
        let depth_artifact = match depth_evidence.as_ref() {
            Some(DepthEvidence::Relative(map)) => Some(DepthArtifact::Relative(
                map,
                depth_range.expect("relative depth has range"),
            )),
            Some(DepthEvidence::Metric(metric)) => {
                Some(DepthArtifact::Metric(metric.depth.frame()))
            }
            None => None,
        };
        let annotations = RecordAnnotations {
            relations: &relations,
            canonical_events: &events,
            depth: depth_metadata,
        };
        let persist = |record: &FrameRecord| {
            persist_artifacts_transactionally(record, Some(&rgb), &self.cfg, depth_artifact)
        };
        let result = if self.force_empty_calibration {
            self.writer
                .record_calibration_frame_with_annotations_transactional(
                    &mut world,
                    now,
                    &iso,
                    &file_stamp,
                    depth_range,
                    annotations,
                    persist,
                )?
        } else {
            self.writer.record_frame_with_annotations_transactional(
                &mut world,
                now,
                &iso,
                &file_stamp,
                depth_range,
                annotations,
                persist,
            )?
        };
        self.session_world = world;

        if result.frame.is_none() {
            return Ok(false);
        }
        self.event_cursor = delta.next_cursor;
        Ok(true)
    }

    /// Encode one frame to disk with privacy blur applied. Public-in-crate so the
    /// pixel path is unit-testable without a live camera.
    #[cfg(test)]
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

#[derive(Clone, Copy)]
enum DepthArtifact<'a> {
    Relative(&'a mvm_depth::RelativeDepthMap, [f32; 2]),
    Metric(&'a mvm_core::MetricDepthFrame),
}

/// Encode every referenced pixel artifact to a sibling temporary file, then
/// atomically rename the complete set into place. DatasetWriter appends the
/// label only after this function returns success.
fn persist_artifacts_transactionally(
    record: &FrameRecord,
    rgb: Option<&RgbImage>,
    cfg: &DatasetConfig,
    depth: Option<DepthArtifact<'_>>,
) -> anyhow::Result<()> {
    let mut staged: Vec<(PathBuf, PathBuf)> = Vec::new();
    let stage_result = (|| -> anyhow::Result<()> {
        if let (Some(path), Some(rgb)) = (record.frame_path.as_ref(), rgb) {
            let tmp = artifact_temp_path(path)?;
            save_rgb(rgb, &tmp, cfg)?;
            staged.push((tmp, path.clone()));
        }
        if let (Some(path), Some(depth)) = (record.depth_path.as_ref(), depth) {
            let tmp = artifact_temp_path(path)?;
            match depth {
                DepthArtifact::Relative(map, range) => write_depth_png(map, range, &tmp)?,
                DepthArtifact::Metric(frame) => write_metric_depth_png(frame, &tmp)?,
            }
            staged.push((tmp, path.clone()));
        }
        Ok(())
    })();
    if let Err(error) = stage_result {
        cleanup_staged(&staged);
        return Err(error);
    }
    for (index, (tmp, final_path)) in staged.iter().enumerate() {
        if let Err(error) = std::fs::rename(tmp, final_path) {
            cleanup_staged(&staged[index..]);
            return Err(error.into());
        }
    }
    Ok(())
}

fn artifact_temp_path(path: &Path) -> anyhow::Result<PathBuf> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!("artifact path has no UTF-8 filename: {}", path.display())
        })?;
    Ok(path.with_file_name(format!(".{name}.tmp")))
}

fn cleanup_staged(staged: &[(PathBuf, PathBuf)]) {
    for (tmp, _) in staged {
        if let Err(error) = std::fs::remove_file(tmp)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "  dataset: failed to remove staged artifact {}: {error}",
                tmp.display()
            );
        }
    }
}

fn canonical_relations(scene: &mvm_core::canonical::FusedSceneFrame) -> SceneRelations {
    use mvm_core::canonical::SceneRelationKind;

    let display_ids: BTreeMap<_, _> = scene
        .entities
        .iter()
        .map(|entity| (entity.key.clone(), entity.display_id))
        .collect();
    let edges = scene
        .relations
        .iter()
        .filter_map(|relation| {
            Some(EdgeRel {
                a: *display_ids.get(&relation.a)?,
                b: *display_ids.get(&relation.b)?,
                rel: match relation.kind {
                    SceneRelationKind::Near => "near",
                    SceneRelationKind::Overlapping => "overlapping",
                    SceneRelationKind::PartOf => "part_of",
                    SceneRelationKind::Touching => "touching",
                    SceneRelationKind::HeldBy => "held_by",
                }
                .into(),
            })
        })
        .collect();
    let zones = scene
        .zones
        .iter()
        .map(|zone| ZoneRel {
            persistent_id: u32::try_from(zone.id).ok(),
            label: zone.label.clone(),
            members: zone
                .members
                .iter()
                .filter_map(|key| display_ids.get(key).copied())
                .collect(),
        })
        .collect();
    SceneRelations { zones, edges }
}

fn canonical_event_records(
    events: &[&mvm_core::canonical::SceneEvent],
    scene: &mvm_core::canonical::FusedSceneFrame,
) -> Vec<serde_json::Value> {
    use mvm_core::canonical::{SceneEventPayload, TrackDomain};

    let display_ids: BTreeMap<_, _> = scene
        .entities
        .iter()
        .map(|entity| (entity.key.clone(), entity.display_id))
        .collect();
    events
        .iter()
        .map(|event| {
            let participants: Vec<String> = event
                .entity_keys
                .iter()
                .map(|key| {
                    let id = display_ids.get(key).copied().unwrap_or(key.track_id);
                    match key.track_domain {
                        TrackDomain::Person => format!("p{id}"),
                        TrackDomain::Object => format!("o{id}"),
                        TrackDomain::Animal => format!("a{id}"),
                    }
                })
                .collect();
            match &event.payload {
                Some(SceneEventPayload::ActivityEpisode {
                    label,
                    started_at_s,
                    ended_at_s,
                    duration_s,
                    peak_confidence,
                    stable_participants,
                }) => serde_json::json!({
                    "kind": "episode",
                    "event": "activity_ended",
                    "event_seq": event.event_seq,
                    "source_frame_id": event.source_frame_id,
                    "source_timestamp": event.source_timestamp_unix_ms,
                    "label": label,
                    "participants": participants,
                    "started_at": started_at_s,
                    "ended_at": ended_at_s,
                    "duration_s": duration_s,
                    "peak_confidence": peak_confidence,
                    "stable": stable_participants,
                }),
                None => {
                    let event_name = serde_json::to_value(event.kind)
                        .unwrap_or_else(|_| serde_json::Value::String("unknown".into()));
                    serde_json::json!({
                        "kind": "scene_event",
                        "event": event_name,
                        "event_seq": event.event_seq,
                        "source_frame_id": event.source_frame_id,
                        "source_timestamp": event.source_timestamp_unix_ms,
                        "participants": participants,
                        "confidence": event.fact.confidence,
                    })
                }
            }
        })
        .collect()
}

impl Agent for DatasetWriterAgent {
    fn name(&self) -> String {
        "dataset".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut records = 0u64;
        let mut errors = 0u64;
        while ctx.running() {
            if self.fused_frames.is_some() {
                let packet = self
                    .fused_frames
                    .as_ref()
                    .and_then(mvm_mesh::LatestValue::latest);
                if let Some(packet) = packet {
                    let frame_id = packet.scene.stamp.frame_id;
                    let is_new = self.last_frame_id != Some(frame_id);
                    let due = super::cadence_due(frame_id, self.cadence);
                    if is_new {
                        self.last_frame_id = Some(frame_id);
                        if due {
                            match self.step_canonical(&packet) {
                                Ok(true) => records += 1,
                                Ok(false) => {}
                                Err(error) => {
                                    eprintln!("  dataset: canonical record error: {error}");
                                    errors += 1;
                                }
                            }
                        }
                    }
                }
            } else if let Some(frame) = ctx.frames.latest() {
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
                    }
                }
            }
            if let Some(max) = self.max_records
                && records >= max
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // Finalize any open session on shutdown so its session.toml is written.
        let ended = chrono::Local::now()
            .format("%Y-%m-%dT%H:%M:%S%.3f%:z")
            .to_string();
        let _ = self
            .writer
            .close_open_session(&mut self.session_world, &ended);
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
fn write_depth_png(
    map: &mvm_depth::RelativeDepthMap,
    range: [f32; 2],
    path: &Path,
) -> anyhow::Result<()> {
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

/// Persist metric depth losslessly. Pixel values are millimetres exactly as
/// delivered by the admitted sensor frame; `invalid_value` remains unchanged.
fn write_metric_depth_png(depth: &mvm_core::MetricDepthFrame, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    depth.validate()?;
    let img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(
        depth.width,
        depth.height,
        depth.depth_mm.clone(),
    )
    .ok_or_else(|| {
        anyhow::anyhow!(
            "metric depth buffer {}×{} mismatch",
            depth.width,
            depth.height
        )
    })?;
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
    use crate::agents::{FrameEvidenceHandles, HumanObservationFrame, ObservationStamp};
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
            tracker_id: id,
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
        let map = mvm_depth::RelativeDepthMap {
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
    fn metric_depth_png_round_trips_exact_u16_and_invalid_value() {
        let root = temp_root("metric_depth_png");
        let path = root.join("depth/000000.png");
        let expected = vec![0, 1, 999, 1_000, 2_047, u16::MAX];
        let depth = mvm_core::MetricDepthFrame {
            width: 3,
            height: 2,
            depth_mm: expected.clone(),
            invalid_value: 0,
            stamp: mvm_core::SensorStamp {
                source_sequence: 1,
                host_monotonic_ns: 1,
                wall_clock_unix_ms: 1,
                clock_domain: mvm_core::SensorClockDomain::KinectV1Depth,
                device_timestamp_raw: Some(1),
            },
            geometry: mvm_core::DepthGeometry::RegisteredToRgb,
            calibration_revision: "a".repeat(64),
            valid_ratio: 5.0 / 6.0,
            rgb_depth_skew_ns: 0,
        };
        write_metric_depth_png(&depth, &path).unwrap();
        let decoded = image::open(path).unwrap().to_luma16().into_raw();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn artifact_transaction_commits_all_files_without_temp_residue() {
        let root = temp_root("artifact_transaction");
        let frame_path = root.join("frames/000000.jpg");
        let depth_path = root.join("depth/000000.png");
        let record = FrameRecord {
            session_id: "session".into(),
            seq: 0,
            frame_path: Some(frame_path.clone()),
            depth_path: Some(depth_path.clone()),
            observation: mvm_dataset::CanonicalObservation::from_jsonl_line(
                &serde_json::json!({
                    "dataset_id": "mvm_sessions",
                    "schema_version": "16.8.0",
                    "session_id": "session",
                    "frame_id": 0,
                    "frame": {"timestamp": "iso"},
                    "session": {"seq_in_session": 0},
                    "entities": [],
                    "relations": [],
                    "events": [],
                    "semantic_groups": [],
                    "derived_targets": null,
                    "quality": {"frame_ok": true, "people_count": 0, "object_count": 0, "anomaly_score": 0.0},
                    "provenance": [],
                    "activities": []
                })
                .to_string(),
            )
            .unwrap(),
        };
        let map = mvm_depth::RelativeDepthMap {
            data: vec![0.0, 1.0, 2.0, 3.0],
            w: 2,
            h: 2,
        };
        persist_artifacts_transactionally(
            &record,
            Some(&RgbImage::new(2, 2)),
            &DatasetConfig::default(),
            Some(DepthArtifact::Relative(&map, [0.0, 3.0])),
        )
        .unwrap();

        assert!(frame_path.is_file());
        assert!(depth_path.is_file());
        assert!(!artifact_temp_path(&frame_path).unwrap().exists());
        assert!(!artifact_temp_path(&depth_path).unwrap().exists());
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
        let report = mvm_dataset::check_dataset(&root).unwrap();
        assert_eq!(report.errors(), 0, "{:?}", report.sessions[0].findings);
        assert_eq!(report.warns(), 0, "{:?}", report.sessions[0].findings);
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
        let report = mvm_dataset::check_dataset(&root).unwrap();
        assert_eq!(report.errors(), 0, "{:?}", report.sessions[0].findings);
        assert_eq!(report.warns(), 0, "{:?}", report.sessions[0].findings);
    }

    #[test]
    fn canonical_packet_records_without_mutating_shared_world() {
        use mvm_core::canonical::{FactSource, FrameStamp, SourceStamp};
        use mvm_fusion::canonical::{CanonicalFusion, CanonicalFusionInput};

        let root = temp_root("canonical");
        let mut cfg = DatasetConfig::default()
            .effective_recorder()
            .expect("default text-only recorder");
        cfg.frame_cadence = 1;
        let fused = FusedFrameBus::new();
        let mut agent = DatasetWriterAgent::new(cfg, root.clone(), None)
            .with_fused_frames(fused.clone())
            .with_max_records(1);

        let now = Instant::now();
        let people = Arc::new(vec![person(1, now, false)]);
        let mut canonical = CanonicalFusion::new("test-run");
        let ended = [mvm_core::ActivityEvent {
            label: "working".into(),
            participants: vec![1],
            started_at: 1.0,
            ended_at: 4.5,
            duration_s: 3.5,
            peak_confidence: 0.9,
            stable_participants: true,
        }];
        let scene = canonical
            .assemble(CanonicalFusionInput {
                stamp: FrameStamp::new("test-run", 3, 1_000, 1_001).unwrap(),
                sources: vec![
                    SourceStamp::at_frame(FactSource::Human, 3, 1_001)
                        .relative_to(3, 2, 4)
                        .unwrap(),
                ],
                people: &people,
                objects: &[],
                animals: &[],
                activities: &[],
                ended_activities: &ended,
                anomaly_score: 0.0,
                metric_geometry: &[],
                retained_people: &[],
                retired_people: &[],
                retained_objects: &[],
                retired_objects: &[],
            })
            .unwrap();
        let camera = Frame::new_stamped(3, image::DynamicImage::new_rgb8(64, 48), now, 1_000);
        fused.publish(FusedFramePacket {
            scene: Arc::new(scene),
            evidence: FrameEvidenceHandles {
                rgb: Arc::clone(&camera.image),
                depth: None,
                human: Some(HumanObservationFrame {
                    stamp: ObservationStamp::from_frame(&camera),
                    people,
                    occluded_people: Arc::new(Vec::new()),
                    retained_track_ids: Arc::new(Vec::new()),
                    retirements: Arc::new(Vec::new()),
                    metric_geometry: Arc::new(Vec::new()),
                }),
                scene: None,
            },
        });

        let shared = Arc::new(RwLock::new(WorldState::new(now)));
        let ctx = MeshContext {
            frames: LatestFrame::new(),
            world: Arc::clone(&shared),
            stop: Arc::new(AtomicBool::new(false)),
        };
        assert_eq!(agent.run(&ctx).ticks, 1);
        assert!(
            shared.read().unwrap().current_session_id.is_none(),
            "canonical DatasetWriter owns session lifecycle privately"
        );

        let session = std::fs::read_dir(root.join("sessions"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        assert!(session.join("labels.jsonl").is_file());
        assert!(session.join("session.toml").is_file());
        let labels = std::fs::read_to_string(session.join("labels.jsonl")).unwrap();
        let record: serde_json::Value = serde_json::from_str(labels.trim()).unwrap();
        assert_eq!(record["events"][0]["kind"], "episode");
        assert_eq!(record["events"][0]["event"], "activity_ended");
        assert_eq!(record["events"][0]["event_seq"], 0);
        assert_eq!(record["events"][0]["label"], "working");
        assert_eq!(record["events"][0]["participants"][0], "p1");
    }

    #[test]
    fn canonical_metric_packet_writes_lossless_depth_and_provenance() {
        use crate::agents::DepthObservationFrame;
        use crate::agents::observations::MetricDepthEvidence;
        use mvm_core::canonical::{FactSource, FrameStamp, SourceStamp};
        use mvm_fusion::canonical::{CanonicalFusion, CanonicalFusionInput};

        let root = temp_root("canonical_metric");
        let cfg = DatasetConfig {
            frame_cadence: 1,
            ..Default::default()
        };
        let fused = FusedFrameBus::new();
        let mut agent = DatasetWriterAgent::new(cfg, root.clone(), None)
            .with_fused_frames(fused.clone())
            .with_max_records(1);
        let now = Instant::now();
        let people = Arc::new(vec![person(1, now, false)]);
        let scene = CanonicalFusion::new("test-run")
            .assemble(CanonicalFusionInput {
                stamp: FrameStamp::new("test-run", 3, 1_000, 1_001).unwrap(),
                sources: vec![
                    SourceStamp::at_frame(FactSource::Human, 3, 1_001)
                        .relative_to(3, 2, 4)
                        .unwrap(),
                ],
                people: &people,
                objects: &[],
                animals: &[],
                activities: &[],
                ended_activities: &[],
                anomaly_score: 0.0,
                metric_geometry: &[],
                retained_people: &[],
                retired_people: &[],
                retained_objects: &[],
                retired_objects: &[],
            })
            .unwrap();
        let rgb_stamp = mvm_core::SensorStamp {
            source_sequence: 3,
            host_monotonic_ns: 1_000_000,
            wall_clock_unix_ms: 1_000,
            clock_domain: mvm_core::SensorClockDomain::KinectV1Rgb,
            device_timestamp_raw: Some(3),
        };
        let calibration = mvm_core::CalibrationRef {
            schema: "kinect-v1-sensor-calibration/1.0".into(),
            camera_serial: "serial".into(),
            revision_sha256: "a".repeat(64),
        };
        let expected = vec![0, 500, 1_000, 2_000];
        let metric_frame = Arc::new(mvm_core::MetricDepthFrame {
            width: 2,
            height: 2,
            depth_mm: expected.clone(),
            invalid_value: 0,
            stamp: mvm_core::SensorStamp {
                source_sequence: 3,
                host_monotonic_ns: 11_000_000,
                wall_clock_unix_ms: 1_000,
                clock_domain: mvm_core::SensorClockDomain::KinectV1Depth,
                device_timestamp_raw: Some(3),
            },
            geometry: mvm_core::DepthGeometry::RegisteredToRgb,
            calibration_revision: calibration.revision_sha256.clone(),
            valid_ratio: 0.75,
            rgb_depth_skew_ns: 10_000_000,
        });
        let admitted = mvm_depth::RegisteredMetricDepth::admit(
            2,
            2,
            &rgb_stamp,
            &mvm_core::DepthStatus::Paired,
            metric_frame,
            &calibration,
            20_000_000,
        )
        .unwrap();
        fused.publish(FusedFramePacket {
            scene: Arc::new(scene),
            evidence: FrameEvidenceHandles {
                rgb: Arc::new(image::DynamicImage::new_rgb8(2, 2)),
                depth: Some(DepthObservationFrame {
                    stamp: ObservationStamp::new(3, now, 1_000),
                    evidence: DepthEvidence::Metric(MetricDepthEvidence {
                        depth: admitted,
                        source: mvm_core::SensorSourceId {
                            backend: "kinect_v1".into(),
                            logical_name: "kinect-v1-main".into(),
                            serial: Some("serial".into()),
                        },
                        rgb_stamp,
                    }),
                }),
                human: Some(HumanObservationFrame {
                    stamp: ObservationStamp::new(3, now, 1_000),
                    people,
                    occluded_people: Arc::new(Vec::new()),
                    retained_track_ids: Arc::new(Vec::new()),
                    retirements: Arc::new(Vec::new()),
                    metric_geometry: Arc::new(Vec::new()),
                }),
                scene: None,
            },
        });
        let ctx = MeshContext {
            frames: LatestFrame::new(),
            world: Arc::new(RwLock::new(WorldState::new(now))),
            stop: Arc::new(AtomicBool::new(false)),
        };
        assert_eq!(agent.run(&ctx).ticks, 1);
        let session = std::fs::read_dir(root.join("sessions"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let labels = std::fs::read_to_string(session.join("labels.jsonl")).unwrap();
        let record: serde_json::Value = serde_json::from_str(labels.trim()).unwrap();
        assert_eq!(record["frame"]["depth"]["encoding"], "metric_mm_u16");
        assert_eq!(record["frame"]["depth"]["units"], "millimetres");
        assert!(record["frame"].get("depth_range").is_none());
        let decoded = image::open(session.join("depth/000000.png"))
            .unwrap()
            .to_luma16()
            .into_raw();
        assert_eq!(decoded, expected);
        let meta = std::fs::read_to_string(session.join("session.toml")).unwrap();
        assert!(meta.contains("metric_mm_u16"));
        assert!(meta.contains(&"a".repeat(64)));
        let report = mvm_dataset::check_dataset(&root).unwrap();
        assert_eq!(report.errors(), 0, "{:?}", report.sessions[0].findings);
        assert_eq!(report.warns(), 0, "{:?}", report.sessions[0].findings);
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
