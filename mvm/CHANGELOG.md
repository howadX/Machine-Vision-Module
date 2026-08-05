# Changelog

All notable changes to MVM. Format loosely follows [Keep a Changelog]; versions
are the `mvm` binary / workspace version.

## [Unreleased]

## [0.4.0] — 2026-08-06 — Visual Cortex (phase 18) + Semantic Session Log (phase 16.7)

MVM now publishes one immutable, stamped `FusedSceneFrame` as the canonical
facts layer and a separate evidence-linked `CortexFrame` as the cognitive view.
The Cortex never rewrites scene facts: forecasts, working memory, affordances,
attention, uncertainty and advisory re-observation remain explicitly bounded by
their evidence. The release also includes the default-on semantic-session-log
work completed after v0.3.0.

### Added

- **`mvm-cortex` crate and canonical facts contract.** Typed observation bus,
  `SceneFact<T>`, incarnation-safe `EntityKey`, source freshness and immutable
  `FusedFramePacket` make Fusion the sole factual authority.
- **Cognitive layers.** Evidence-linked scene-event interpretations, bounded
  runtime Event Memory, constant-velocity/persistence forecasts at 1 s and 2 s,
  prediction errors, camera-relative self-model, explicit uncertainty inventory,
  affordance V1 and deterministic top-3 attention.
- **Bounded active re-observation.** Cortex can only propose a serializable
  advisory `AttentionPolicy`; the CLI-owned scheduler grants temporary Scene or
  Depth permits with TTL, dedupe, cooldown and explicit outcomes. Only a new
  stamped Experts → Fusion observation can resolve a request.
- **Public cognitive pull interface.** Atomic `current_cortex.json`, read-only
  `mvm cortex` views and MCP `mvm_cortex`; `current_state.md` remains facts and
  `labels.jsonl` remains the sole durable semantic history.
- **Closure evidence.** Frozen replay validates 6/6 pinned cognitive claims
  against the no-Cortex comparator with zero honesty regressions. The controlled
  adaptive runtime gate is +0.31% (within the ≤5% budget); final live validation
  confirmed the production path, 1 s/2 s forecasts and semantic-session pull.

### Included semantic-session log

Converges MVM's **two parallel scene-history stores into one**. Until now vision
history lived in flat per-day `scene-*.jsonl`/`events-*.jsonl` logs, while the
Dataset Factory wrote session-organized `labels.jsonl`. Phase 16.7 makes a durable,
**text-only semantic session** the single vision-history store — on by default,
enriched with the full relational + temporal scene-graph, and read by
recap/inspect/journal. The vision core (phases 5–15) is untouched; `mvm-dataset`
stays a pure WorldState consumer.

- **Default-on text-only recording.** `mvm run` (no flags) now writes a semantic
  session (`datasets/sessions/<id>/labels.jsonl` — labels only, no pixels). `--record`
  upgrades to a full pixel dataset; `--no-vision-log` disables it. Resolved by the
  pure `DatasetConfig::effective_recorder()` (`vision_log` field, default true).
- **Flat logs retired.** `scene-*.jsonl`/`events-*.jsonl` are no longer written by
  default; `--scene-log <DIR>` keeps them for back-compat.
- **Router enrichment (SCHEMA 16.0.1 → 16.7.0, additive).** Canonical Observation now
  carries `relations` (zones + edges), a new `activities` field (per-frame, with
  temporal fields), and inline `events` (completed episodes). `route_with_relations`
  fills them from WorldState + a `SceneRelations` computed by the mesh agent's
  stateful `ZoneTracker`; the router stays a pure function of its inputs.
- **Reader migration.** `recap`/`inspect`/journal read semantic sessions by projecting
  each observation to the flat scene/event shape and reusing the existing parsers —
  one code path, golden-parity-tested against the flat emitters. A `Source` resolver
  prefers flat logs when present and falls back to semantic sessions. Retention via
  `mvm dataset prune --keep`.
- Workspace gate is green (one live-camera test ignored). Live-validated on camera (2026-07-24, ~4 min):
  default-on text-only session with relations/activities/events, no flat logs written,
  `dataset check` clean, recap/inspect correct.

## [0.3.0] — 2026-07-23 — Dataset Factory (phase 16)

MVM grows a second consumer. Until now every observation flowed to one place — an
LLM, via the MCP pull-model. This release adds an **accumulating structured memory**:
a semantic export layer that compiles each frame's WorldState into an interpretable,
provenance-tagged training record. The vision core (phases 5–14) and the measured ML
layer (phase 15) are untouched — the factory is a **pure WorldState consumer**, like
StateFileWriter. New 13th crate `mvm-dataset`; zero ML frameworks leak into the core
(exporters live in Python). +0 runtime cost when recording is off.

### Added
- **`mvm-dataset` crate** — the Dataset Factory. Canonical Observation (stable JSONL
  schema, binding keys, provenance) + honesty statuses `observed/derived/temporal/weak/
  withheld/unknown` (a `withheld` value is *never* auto-coerced to a negative class) +
  Semantic Router G₁–G₅ (selective routing, no hot-path compute). Deps: serde/toml only.
- **`mvm run --record`** — opt-in session recording. A mesh agent snapshots WorldState at
  a configurable cadence and writes self-describing sessions: `frames/` (JPEG, privacy-
  blurred) + `depth/` (16-bit reversible PNG) + instance masks (RLE) + `labels.jsonl`
  (one Canonical Observation per frame) + `session.toml` summary. `[dataset]` config
  section; sessions group by human presence (a >N s gap opens a new one).
- **`mvm dataset`** CLI — `list` / `info` / `stats` / `prune` (dry-run unless `--yes`) /
  `export` (prints ready-to-run Python commands) / **`check`** (see below).
- **Privacy layer** — precise YuNet face-bbox blur (matched to the body by nose), emitted
  as a `face_bbox` label; `full` / `safe` (default) / `anonymous` modes via `--privacy`.
  Anonymous faces are blurred in `safe`; gallery identities kept clear.
- **Python exporters** (`prototype/pipeline/dataset/`) — PyTorch `IterableDataset` /
  TFRecord / WebDataset views over recorded sessions, with quality-aware filters
  (`withheld` never satisfies a posture requirement). Own-domain data feeds the trainers.
- **Data-quality gates (16.6-A)** — two complementary CI exit-code gates. `mvm dataset
  check [session] [--strict]` is authoritative for *structure* (binding keys, referenced-
  file existence, mask-RLE cell counts, depth ranges, **privacy embedding-leak**) + folds
  quality stats; no image deps. Python `validate.py` does the pixel side Rust can't:
  decodes every frame / depth / mask the exporter is about to feed a trainer.

### Notes / honest limits
- The factory infrastructure is complete and live-validated (record → PyTorch DataLoader;
  precise blur; both gates exit 0/1) — but **no dataset is accumulated yet**. Filling it
  is owner record-time, not code. **16.6-B** (hard-example mining over quality-as-data →
  curated set for *human* labeling) remains, and depends on that accumulated data.
- Naive supervised fine-tuning on recorded labels is **deliberately descoped**: the labels
  *are* the models' own predictions (pose/emotion/identity/masks), so training on them is
  self-distillation. The honest loop closes through a human, or self-supervision — a
  separate north-star direction, not this phase.
- The privacy blur's fallback (no YuNet face) is a geometric upper-body band — precise
  where a face is detected, best-effort (eye region) where it isn't.

### Tests
- Workspace **670** tests green (was 608 + phase-16 additions), 1 ignored (live camera)
  + Python exporter/validate groups. `mvm dataset check` gates a session in CI.

## [0.2.0] — 2026-07-17 — ML Refactor (phase 15)

The whole ML layer was re-derived in a lab-first refactor (CMU mocap + OpenImages
reference pair, frozen bench protocol, Kaggle journals **LJ0–LJ6**) across three
axes — speed / accuracy / weight — then ported into production (acts P0–P8). Every
number below runs through `spike_pose/bench.py` (SEED=0, grouped bootstrap CI,
data+code passport). Pure Rust, +0 runtime dependencies; hot-path weight went down.

### Added
- **`torso_flexion`** — continuous torso inclination (degrees) on posture output,
  threaded through StateFile / Scene JSON / MCP / snapshot. Pure arithmetic (`torso_inc`),
  MAE ~6° frontal / 3.5° profile → +0 model weight. *Contract change #1 (additive).*
- **`Config.camera_height_m`** (default 1.75, eye-level rig) — the floor reference the
  scene-level fall rule uses (`height_above_floor = camera_height_m − position_3d.y`).

### Changed
- **Pose classifier → 2D-GRIP (z=0)**: MediaPipe z is a cross-domain liability (3D
  collapses gold R3 0.955→0.62, sitting recall 0.01). 2D-GRIP holds **gold R3 0.894**
  [0.880, 0.907] (mocap folded in as a training asset). 12 gravity-referenced features.
- **Pose OOD → Mahalanobis** (replaces the 195 KB VAE): real animal-OOD 0.825 vs recon
  0.702. `plausibility` (`low`/`ok`) is now Mahalanobis-derived. Pose footprint
  **1068 → 222 KB (−79%)**. *Contract change #3.*
- **Posture stabilizer** — direct `standing↔lying` commit now needs `SL_DWELL=12`
  consecutive frames (per-edge veto, licensed by the measured transition matrix
  `A[standing,lying]≈0`): the false fall-flip under occlusion stress → **0** (Δ CI
  [−3.62, −0.43]), no aggregate regression.
- **Tracker** — an identified track may recover an anonymous center re-link through a
  stale identity unless that name is claimed by a confident detection this frame. Kills
  the structural walking-fragmentation dead-zone: Δ med **−4.70** / heavy **−8.25**
  (established); idle 0/0; Kalman/gate constants unchanged (calibration refuted).
- **Anomaly fall rule → E2 scene-level** (replaces the per-id standing→lying window,
  which false-alarmed on brisk lie-downs and lost real falls to id-churn): fires on a
  fast pelvis descent that ends on the floor (peak windowed descent ≥ 1.05 m/s AND floor
  endpoint ≤ 0.35 m). AUROC ≈ 1.000. Velocity-only is a documented fallback.
- **SFace identity crop is now ArcFace-aligned** (was the raw bbox rectangle, which SFace
  only resized). LFW held-out **0.783 → 0.978 (+19.5 pt)**; TAR@FAR=1e-3 0.173 → 0.949.
  +0 weight / latency.
- **Activity vocabulary v2** — `typing`/`using_mouse`/`using_phone`/`reading` fold into
  `using_object(<class>)`; `holding_drink` kept; **`watching_tv` dropped** (head-facing,
  not a wrist contact; 0 supervisable VRD instances). *Contract change #2.*

### Deprecated
- **`recon_error`** — always `null` now (the VAE it came from was removed); the field is
  kept only for schema stability. Read `plausibility` instead.

### Notes / honest limits
- Dropping z loses the seated-thigh depth cue → a *frontal* desk-sitter can read
  `standing` at a near-tie (measured `p_V_06` .42 vs .48). The safety invariant (never a
  false `lying`) holds; the relational layer still yields `seated_at_workstation`.
- The E2 fall threshold is calibrated on stunt-only CMU falls → it bounds the clinical
  threshold from above; `camera_height_m` inherits the uncalibrated 3D scale so the floor
  gate is approximate.
- The FSM exit-dwell sweep (LJ4 candidate 5/8→12/15, a stress-fragmentation knob) was
  **evaluated and deferred** — verified no golden regression, but the benefit is only
  measurable against the synthetic stress model; it is a documented one-line flip.
- Published Kaggle assets (`mvm-golden-scenarios`, LJ4 notebook) stay at the v1 vocabulary
  by design — the contract change lands here, not retroactively.

### Tests
- Workspace **608** tests green (was 597+2). `agents::golden` CI gate replays the CMU-derived
  FLOOR scenarios through the production stack.

## [0.1.0] — 2026-07-06 — Product (phase 14)

First public release. "Install → it finds itself and works": XDG paths + `config.toml`,
camera auto-detect, `mvm setup` (model fetch + sha256), watched gallery, `serve
--with-camera`, `mvm mcp install`, `mvm service` (systemd user unit), distribution via
GitHub Releases + `install.sh`. 12 crates, 597 tests. Metric 3D scene-graph, body-primary
chain, MCP pull-model (snapshot / recap / inspect).

[Keep a Changelog]: https://keepachangelog.com/
