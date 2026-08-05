//! `DatasetWriter` — оркестратор записи сессии (§16.1.2).
//!
//! Отвечает за **логику** записи, не за пиксели: жизненный цикл сессии (открытие/
//! закрытие через [`WorldState`]), раскладку на диске, маршрутизацию кадра в
//! [`CanonicalObservation`], дозапись `labels.jsonl` и финализацию `session.toml`.
//!
//! Кодирование RGB/depth/масок и privacy-blur применяет вызывающая сторона
//! (mesh-агент в `mvm-cli`, крейт `image`): writer выделяет пути и записывает их в
//! разметку, а также отдаёт [`FrameRecord::frame_path`] — куда положить пиксели.
//! Такое разделение держит `mvm-dataset` без тяжёлых image-зависимостей и делает
//! логику полностью unit-тестируемой (tempdir, без камеры).

use crate::config::DatasetConfig;
use crate::observation::CanonicalObservation;
use crate::router::{RouteContext, SceneRelations, route_with_relations};
use crate::session::{AnomaliesBlock, ObjectsBlock, ParticipantsBlock, SessionBlock, SessionMeta};
use mvm_core::world::WorldState;
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Итог одного вызова [`DatasetWriter::record_frame`].
#[derive(Debug, Default)]
pub struct RecordResult {
    /// Открыта новая сессия (её id) на этом кадре.
    pub opened: Option<String>,
    /// Записан кадр — что и куда положить пиксели.
    pub frame: Option<FrameRecord>,
    /// Закрыта сессия по таймауту (её id); `session.toml` уже финализирован.
    pub closed: Option<String>,
}

/// Semantic additions computed outside the legacy `WorldState` projection.
pub struct RecordAnnotations<'a> {
    pub relations: &'a SceneRelations,
    pub canonical_events: &'a [serde_json::Value],
}

/// Записанный кадр: наблюдение уже в `labels.jsonl`, пиксели пишет вызывающий.
#[derive(Debug)]
pub struct FrameRecord {
    pub session_id: String,
    pub seq: u64,
    /// Абсолютный путь, куда вызывающий должен записать RGB-кадр. `None` в
    /// text-only режиме (`save_frames = false`) — вызывающий пиксели не пишет.
    pub frame_path: Option<PathBuf>,
    /// Абсолютный путь под карту глубины (если `save_depth`).
    pub depth_path: Option<PathBuf>,
    /// Наблюдение, записанное в `labels.jsonl` (для интроспекции/тестов).
    pub observation: CanonicalObservation,
}

/// Аккумулятор сводки сессии → `session.toml`.
#[derive(Debug)]
struct SessionStats {
    id: String,
    started: String,
    saved_frames: u64,
    known: BTreeMap<String, u32>,
    anon_ids: std::collections::BTreeSet<u32>,
    activities: BTreeMap<String, u32>,
    object_classes: BTreeMap<String, u32>,
    anomaly_max: f32,
    anomaly_sum: f64,
    anomaly_n: u64,
}

impl SessionStats {
    fn new(id: String, started: String) -> Self {
        SessionStats {
            id,
            started,
            saved_frames: 0,
            known: BTreeMap::new(),
            anon_ids: std::collections::BTreeSet::new(),
            activities: BTreeMap::new(),
            object_classes: BTreeMap::new(),
            anomaly_max: 0.0,
            anomaly_sum: 0.0,
            anomaly_n: 0,
        }
    }

    fn accumulate(&mut self, world: &WorldState) {
        self.saved_frames += 1;
        for p in &world.people {
            match &p.identity {
                Some((name, _)) => *self.known.entry(name.clone()).or_default() += 1,
                None => {
                    self.anon_ids.insert(p.track_id);
                }
            }
        }
        for a in &world.activities {
            *self.activities.entry(a.label.clone()).or_default() += 1;
        }
        for o in &world.objects {
            *self.object_classes.entry(o.class.clone()).or_default() += 1;
        }
        self.anomaly_max = self.anomaly_max.max(world.anomaly_score);
        self.anomaly_sum += world.anomaly_score as f64;
        self.anomaly_n += 1;
    }

    fn into_meta(self, ended: String) -> SessionMeta {
        let mean = if self.anomaly_n > 0 {
            (self.anomaly_sum / self.anomaly_n as f64) as f32
        } else {
            0.0
        };
        // Top object classes by frequency (desc), ties broken by name for determinism.
        let mut objs: Vec<(String, u32)> = self.object_classes.into_iter().collect();
        objs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let top: Vec<String> = objs.into_iter().take(5).map(|(c, _)| c).collect();
        SessionMeta {
            session: SessionBlock {
                id: self.id,
                started: self.started,
                ended,
                saved_frames: self.saved_frames,
            },
            participants: ParticipantsBlock {
                known: self.known.keys().cloned().collect(),
                anonymous: self.anon_ids.len() as u32,
            },
            activities: self.activities,
            anomalies: AnomaliesBlock {
                max: self.anomaly_max,
                mean,
            },
            objects: ObjectsBlock { top },
        }
    }
}

/// Оркестратор записи. Один writer на процесс записи; держит корень хранилища,
/// счётчик кадров текущей сессии и её накапливаемую сводку.
pub struct DatasetWriter {
    cfg: DatasetConfig,
    root: PathBuf,
    dataset_id: String,
    seq_in_session: u64,
    stats: Option<SessionStats>,
}

impl DatasetWriter {
    /// Новый writer. `root` — уже разрешённый корень хранилища (тильда раскрыта
    /// вызывающим); `dataset_id` — логическое имя датасета (ключ связывания).
    pub fn new(
        cfg: DatasetConfig,
        root: impl Into<PathBuf>,
        dataset_id: impl Into<String>,
    ) -> Self {
        DatasetWriter {
            cfg,
            root: root.into(),
            dataset_id: dataset_id.into(),
            seq_in_session: 0,
            stats: None,
        }
    }

    /// Директория сессии `<root>/sessions/<id>`.
    fn session_dir(&self, id: &str) -> PathBuf {
        self.root.join("sessions").join(id)
    }

    /// Записать один кадр (вызывается только на cadence-кадрах). Драйвит жизненный
    /// цикл сессии, маршрутизирует, дозаписывает JSONL, накапливает сводку.
    ///
    /// - `now` — монотонные часы (таймаут сессии).
    /// - `wallclock_iso` — ISO-8601 timestamp кадра (стенные часы вызывающего).
    /// - `new_session_stamp` — id для НОВОЙ сессии (`YYYY-MM-DD_HHMMSS`), если
    ///   на этом кадре нужно открыть сессию.
    ///
    /// `depth_range` — `Some([min,max])` of the relative depth map the caller is
    /// about to write for this frame, or `None` when no depth is available. The
    /// depth path is recorded **only** when `save_depth` is on *and* a range is
    /// given → labels never point at a depth file that was not written.
    ///
    /// `relations` — the frame's zones+edges (§16.7), computed by the caller (it
    /// owns the `mvm-fusion` `build_scene`/`ZoneTracker`). Pass
    /// `&SceneRelations::default()` when no relational context is available.
    pub fn record_frame(
        &mut self,
        world: &mut WorldState,
        now: Instant,
        wallclock_iso: &str,
        new_session_stamp: &str,
        depth_range: Option<[f32; 2]>,
        relations: &SceneRelations,
    ) -> anyhow::Result<RecordResult> {
        self.record_frame_with_annotations(
            world,
            now,
            wallclock_iso,
            new_session_stamp,
            depth_range,
            RecordAnnotations {
                relations,
                canonical_events: &[],
            },
        )
    }

    /// Canonical-fusion variant: appends this consumer's non-destructive event
    /// cursor delta to the observation written on this cadence tick.
    pub fn record_frame_with_annotations(
        &mut self,
        world: &mut WorldState,
        now: Instant,
        wallclock_iso: &str,
        new_session_stamp: &str,
        depth_range: Option<[f32; 2]>,
        annotations: RecordAnnotations<'_>,
    ) -> anyhow::Result<RecordResult> {
        let mut res = RecordResult::default();
        let has_people = !world.people.is_empty();

        // 1. Presence + open.
        if has_people {
            world.mark_people_seen(now);
            if !world.session_active() {
                let sid = new_session_stamp.to_string();
                let dir = self.session_dir(&sid);
                fs::create_dir_all(&dir)?; // session root always (holds labels.jsonl + session.toml)
                if self.cfg.save_frames {
                    fs::create_dir_all(dir.join("frames"))?;
                }
                if self.cfg.save_depth {
                    fs::create_dir_all(dir.join("depth"))?;
                }
                world.open_session(sid.clone(), now);
                self.seq_in_session = 0;
                self.stats = Some(SessionStats::new(sid.clone(), wallclock_iso.to_string()));
                res.opened = Some(sid);
            }
        }

        // 2. Write while a session is open.
        if let Some(sid) = world.current_session_id.clone() {
            let seq = self.seq_in_session;
            // Text-only mode (`save_frames = false`): no pixel path is allocated, so
            // the label never points at a frame that was not written.
            let frame_rel = self
                .cfg
                .save_frames
                .then(|| format!("frames/{:06}.{}", seq, self.cfg.frame_format.ext()));
            // Record a depth path only when depth is enabled AND actually available
            // this frame (range given) — otherwise the label would dangle.
            let depth_rel = match (self.cfg.save_depth, depth_range) {
                (true, Some(_)) => Some(format!("depth/{seq:06}.png")),
                _ => None,
            };

            let ctx = RouteContext {
                dataset_id: &self.dataset_id,
                session_id: &sid,
                timestamp: wallclock_iso,
                seq_in_session: seq,
            };
            let mut obs = route_with_relations(world, &ctx, annotations.relations);
            obs.events.extend_from_slice(annotations.canonical_events);
            obs.frame.frame_path = frame_rel.clone();
            obs.frame.depth_path = depth_rel.clone();
            obs.frame.depth_range = depth_rel.as_ref().and(depth_range);

            self.append_label(&sid, &obs)?;
            if let Some(stats) = self.stats.as_mut() {
                stats.accumulate(world);
            }
            self.seq_in_session += 1;

            let dir = self.session_dir(&sid);
            res.frame = Some(FrameRecord {
                session_id: sid,
                seq,
                frame_path: frame_rel.map(|r| dir.join(r)),
                depth_path: depth_rel.map(|d| dir.join(d)),
                observation: obs,
            });
        }

        // 3. Close-on-timeout (a people-frame just refreshed the clock, so won't fire).
        let timeout = Duration::from_secs(self.cfg.session_timeout_secs);
        if let Some(closed) = world.maybe_close_session(now, timeout) {
            self.finalize_session(&closed, wallclock_iso)?;
            res.closed = Some(closed);
        }

        Ok(res)
    }

    /// Принудительно закрыть открытую сессию (конец записи / shutdown).
    pub fn close_open_session(
        &mut self,
        world: &mut WorldState,
        ended_iso: &str,
    ) -> anyhow::Result<Option<String>> {
        if let Some(id) = world.current_session_id.take() {
            world.last_people_seen = None;
            self.finalize_session(&id, ended_iso)?;
            Ok(Some(id))
        } else {
            Ok(None)
        }
    }

    fn append_label(&self, sid: &str, obs: &CanonicalObservation) -> anyhow::Result<()> {
        let path = self.session_dir(sid).join("labels.jsonl");
        let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
        f.write_all(obs.to_jsonl_line()?.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    fn finalize_session(&mut self, sid: &str, ended_iso: &str) -> anyhow::Result<()> {
        let stats = match self.stats.take() {
            Some(s) => s,
            // No stats (already finalized / never opened here) — write a minimal stub.
            None => SessionStats::new(sid.to_string(), ended_iso.to_string()),
        };
        let meta = stats.into_meta(ended_iso.to_string());
        let toml_str = toml::to_string_pretty(&meta)?;
        let path = self.session_dir(sid).join("session.toml");
        fs::write(&path, toml_str)?;
        Ok(())
    }

    /// Корень хранилища (для интроспекции/тестов).
    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::world::{ObjectState, PersonState};
    use mvm_core::{Activity, BBox};
    use std::sync::atomic::{AtomicU32, Ordering};

    static UNIQ: AtomicU32 = AtomicU32::new(0);

    fn temp_root() -> PathBuf {
        let n = UNIQ.fetch_add(1, Ordering::Relaxed);
        // PID + a process-local counter is not sufficient in sandboxes that reuse
        // the PID namespace between test runs: an old session directory can leak
        // pixels or labels into a new test. Keep roots unique without deleting
        // anything outside the test currently creating it.
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mvm_ds_test_{}_{}_{}",
            std::process::id(),
            nonce,
            n
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn writer(root: &Path) -> DatasetWriter {
        let cfg = DatasetConfig {
            session_timeout_secs: 30,
            save_depth: false, // keep the layout minimal for tests
            ..Default::default()
        };
        DatasetWriter::new(cfg, root.to_path_buf(), "mvm_sessions")
    }

    fn person(track_id: u32, now: Instant) -> PersonState {
        PersonState {
            track_id,
            bbox: BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
            face_bbox: None,
            identity: None,
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

    fn read_lines(p: &Path) -> Vec<String> {
        fs::read_to_string(p)
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn records_open_session_and_appends_jsonl() {
        let root = temp_root();
        let mut w = writer(&root);
        let t0 = Instant::now();
        let mut world = WorldState::new(t0);
        world.frame_id = 100;
        world.people.push(person(1, t0));

        let r = w
            .record_frame(
                &mut world,
                t0,
                "2026-07-17T15:30:00.000",
                "2026-07-17_153000",
                None,
                &SceneRelations::default(),
            )
            .unwrap();
        assert_eq!(r.opened.as_deref(), Some("2026-07-17_153000"));
        let frame = r.frame.expect("frame recorded");
        assert_eq!(frame.seq, 0);
        assert!(
            frame
                .frame_path
                .as_ref()
                .unwrap()
                .ends_with("frames/000000.jpg")
        );

        let labels = root.join("sessions/2026-07-17_153000/labels.jsonl");
        let lines = read_lines(&labels);
        assert_eq!(lines.len(), 1);
        let obs = CanonicalObservation::from_jsonl_line(&lines[0]).unwrap();
        assert_eq!(obs.frame_id, 100);
        assert_eq!(obs.session_id, "2026-07-17_153000");
        assert_eq!(obs.frame.frame_path.as_deref(), Some("frames/000000.jpg"));
        assert_eq!(obs.entities.len(), 1);
    }

    #[test]
    fn text_only_mode_writes_labels_without_a_frame_path() {
        // save_frames = false → semantic session: labels.jsonl + session.toml, no pixels.
        let root = temp_root();
        let cfg = DatasetConfig {
            save_frames: false,
            save_depth: false,
            ..Default::default()
        };
        let mut w = DatasetWriter::new(cfg, root.clone(), "mvm_sessions");

        let t0 = Instant::now();
        let mut world = WorldState::new(t0);
        world.frame_id = 100;
        world.people.push(person(1, t0));

        let r = w
            .record_frame(
                &mut world,
                t0,
                "2026-07-17T15:30:00.000",
                "2026-07-17_153000",
                None,
                &SceneRelations::default(),
            )
            .unwrap();
        let frame = r.frame.expect("frame recorded");
        // No pixel path handed to the caller, and none in the label.
        assert!(
            frame.frame_path.is_none(),
            "text-only mode allocates no frame path"
        );
        assert!(frame.observation.frame.frame_path.is_none());

        let sess = root.join("sessions/2026-07-17_153000");
        // The semantic log is present; the pixel dir was never created.
        assert!(sess.join("labels.jsonl").is_file(), "labels.jsonl present");
        assert!(
            !sess.join("frames").exists(),
            "no frames/ dir in text-only mode"
        );
        // The label still carries the full semantic observation (entities, quality).
        let obs = CanonicalObservation::from_jsonl_line(&read_lines(&sess.join("labels.jsonl"))[0])
            .unwrap();
        assert_eq!(obs.entities.len(), 1, "person entity still recorded");
    }

    #[test]
    fn record_frame_threads_relations_into_the_label() {
        // §16.7: zones/edges the caller computed reach labels.jsonl as relation records.
        let root = temp_root();
        let mut w = writer(&root);
        let t0 = Instant::now();
        let mut world = WorldState::new(t0);
        world.people.push(person(1, t0));

        let rel = SceneRelations {
            zones: vec![crate::router::ZoneRel {
                persistent_id: Some(3),
                label: Some("workstation".into()),
                members: vec![1, 10001],
            }],
            edges: vec![crate::router::EdgeRel {
                a: 1,
                b: 10001,
                rel: "near".into(),
            }],
        };
        let r = w
            .record_frame(&mut world, t0, "iso", "2026-07-17_153000", None, &rel)
            .unwrap();
        let obs = r.frame.unwrap().observation;
        assert!(
            obs.relations
                .iter()
                .any(|x| x["kind"] == "zone" && x["label"] == "workstation")
        );
        assert!(
            obs.relations
                .iter()
                .any(|x| x["kind"] == "edge" && x["a"] == "p1" && x["b"] == "o10001")
        );
    }

    #[test]
    fn seq_increments_and_session_persists_across_frames() {
        let root = temp_root();
        let mut w = writer(&root);
        let t0 = Instant::now();
        let mut world = WorldState::new(t0);
        world.people.push(person(1, t0));

        for i in 0..3u64 {
            world.frame_id = i;
            let t = t0 + Duration::from_secs(i);
            let r = w
                .record_frame(
                    &mut world,
                    t,
                    "2026-07-17T15:30:00.000",
                    "2026-07-17_153000",
                    None,
                    &SceneRelations::default(),
                )
                .unwrap();
            // Session opened only on the first frame.
            assert_eq!(r.opened.is_some(), i == 0);
            assert_eq!(r.frame.unwrap().seq, i);
        }
        let labels = root.join("sessions/2026-07-17_153000/labels.jsonl");
        assert_eq!(read_lines(&labels).len(), 3);
    }

    #[test]
    fn timeout_closes_session_and_writes_session_toml() {
        let root = temp_root();
        let mut w = writer(&root);
        let t0 = Instant::now();
        let mut world = WorldState::new(t0);

        // Frame 0: a known person + an object + an activity.
        let mut p = person(1, t0);
        p.identity = Some(("Вадим".into(), 0.9));
        world.people.push(p);
        world.objects.push(ObjectState {
            id: 10001,
            class: "keyboard".into(),
            bbox: BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 0.1,
                y2: 0.1,
            },
            confidence: 0.8,
            age: 5,
            position_3d: None,
            mask: None,
            remembered: false,
            last_seen: t0,
        });
        world.activities.push(Activity {
            label: "typing".into(),
            confidence: 0.8,
            participants: vec![1],
        });
        world.anomaly_score = 0.4;
        w.record_frame(
            &mut world,
            t0,
            "2026-07-17T15:30:00.000",
            "2026-07-17_153000",
            None,
            &SceneRelations::default(),
        )
        .unwrap();

        // People leave.
        world.people.clear();
        world.objects.clear();
        world.activities.clear();

        // A frame 20s later — inside the window, no close.
        let r = w
            .record_frame(
                &mut world,
                t0 + Duration::from_secs(20),
                "iso",
                "unused",
                None,
                &SceneRelations::default(),
            )
            .unwrap();
        assert!(r.closed.is_none());

        // A frame 31s after last presence — session closes.
        let r = w
            .record_frame(
                &mut world,
                t0 + Duration::from_secs(31),
                "2026-07-17T15:30:31.000",
                "unused",
                None,
                &SceneRelations::default(),
            )
            .unwrap();
        assert_eq!(r.closed.as_deref(), Some("2026-07-17_153000"));
        assert!(!world.session_active());

        let meta_path = root.join("sessions/2026-07-17_153000/session.toml");
        assert!(meta_path.exists(), "session.toml must be written on close");
        let meta: toml::Value = toml::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
        assert_eq!(meta["session"]["id"].as_str(), Some("2026-07-17_153000"));
        assert_eq!(meta["participants"]["known"][0].as_str(), Some("Вадим"));
        assert_eq!(meta["activities"]["typing"].as_integer(), Some(1));
        assert_eq!(meta["objects"]["top"][0].as_str(), Some("keyboard"));
    }

    #[test]
    fn depth_path_recorded_only_when_available() {
        let root = temp_root();
        let cfg = DatasetConfig {
            session_timeout_secs: 30, // save_depth stays default true
            ..Default::default()
        };
        let mut w = DatasetWriter::new(cfg, root.to_path_buf(), "mvm_sessions");
        let t0 = Instant::now();
        let mut world = WorldState::new(t0);
        world.people.push(person(1, t0));

        // Frame 0: no depth available → no depth path/range even though save_depth=true.
        let r = w
            .record_frame(
                &mut world,
                t0,
                "iso",
                "2026-07-17_170000",
                None,
                &SceneRelations::default(),
            )
            .unwrap();
        let obs0 = &r.frame.unwrap().observation;
        assert!(
            obs0.frame.depth_path.is_none(),
            "no dangling depth path without depth"
        );
        assert!(obs0.frame.depth_range.is_none());

        // Frame 1: depth available → path + range recorded.
        let r = w
            .record_frame(
                &mut world,
                t0 + Duration::from_secs(1),
                "iso",
                "unused",
                Some([0.2, 3.5]),
                &SceneRelations::default(),
            )
            .unwrap();
        let obs1 = &r.frame.unwrap().observation;
        assert_eq!(obs1.frame.depth_path.as_deref(), Some("depth/000001.png"));
        assert_eq!(obs1.frame.depth_range, Some([0.2, 3.5]));
    }

    #[test]
    fn no_people_no_session_writes_nothing() {
        let root = temp_root();
        let mut w = writer(&root);
        let t0 = Instant::now();
        let mut world = WorldState::new(t0);
        let r = w
            .record_frame(
                &mut world,
                t0,
                "iso",
                "stamp",
                None,
                &SceneRelations::default(),
            )
            .unwrap();
        assert!(r.opened.is_none() && r.frame.is_none() && r.closed.is_none());
        assert!(!root.join("sessions/stamp").exists());
    }

    #[test]
    fn close_open_session_finalizes_on_shutdown() {
        let root = temp_root();
        let mut w = writer(&root);
        let t0 = Instant::now();
        let mut world = WorldState::new(t0);
        world.people.push(person(1, t0));
        w.record_frame(
            &mut world,
            t0,
            "iso",
            "2026-07-17_160000",
            None,
            &SceneRelations::default(),
        )
        .unwrap();
        assert!(world.session_active());

        let closed = w
            .close_open_session(&mut world, "2026-07-17_160500")
            .unwrap();
        assert_eq!(closed.as_deref(), Some("2026-07-17_160000"));
        assert!(!world.session_active());
        assert!(
            root.join("sessions/2026-07-17_160000/session.toml")
                .exists()
        );
    }
}
