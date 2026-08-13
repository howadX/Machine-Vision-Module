# T0 tracker integration inventory

Baseline: `f67f5eab` on branch `sensors`. This inventory is an implementation
map, not authority to preserve accidental compatibility behavior.

## Person tracker mutation paths

| Surface | Current role | Required treatment |
|---|---|---|
| `mvm-core/src/tracker.rs` | `PersonTracker`, association, lifecycle, re-link, expiry | single person tracking authority; T1 replay calls the same public frame step |
| `mvm-cli/src/agents/human.rs` | canonical AgentMesh owner; `tick_motion` per Human tick | primary production wiring for stamped observations and diagnostics |
| `mvm-cli/src/agents/golden.rs` | model-free production temporal regression | retain as fast scenario gate; route through production API |
| `mvm-cli/src/main.rs` | synchronous/native/bridge compatibility loops | migrate or explicitly retire; must not retain a phantom-prone private policy |
| `mvm-pipeline/tests/integration.rs` | bridge-to-tracker integration tests | preserve RGB compatibility coverage |
| `mvm-cli/src/state.rs` | legacy tracker rendering/Scene JSON helpers | make lifecycle-aware or remove at the T10 compatibility gate |

`PersonTracker::active()` currently means only `missed == 0`; it does not filter
`Tentative`. `collect_people` therefore publishes a first-frame track into
`PersonState`, Fusion, activities, Dataset, StateFile, MCP, and Cortex. T3 must
replace this implicit contract without breaking the tracker's internal need to
inspect tentative/current tracks.

## Object tracker mutation paths

| Surface | Current role | Required treatment |
|---|---|---|
| `mvm-scene/src/lib.rs` | greedy class-aware IoU, EMA, timeout, static hold | single object tracking authority; T7 adds deterministic global assignment and lifecycle |
| `mvm-cli/src/agents/scene.rs` | canonical AgentMesh owner; object cadence | primary stamped observation/diagnostic wiring |
| `mvm-cli/src/pipeline.rs` | synchronous object cadence and latest objects | migrate to the same accepted policy or retire |
| `mvm-cli/src/agents/mod.rs::collect_objects` | `TrackedObject` to `ObjectState` projection | separate observed from remembered/occluded lifecycle |

New object detections are emitted immediately. Static objects with sufficient
age can remain emitted with `missed > 0`; downstream currently uses only a
`remembered` boolean and can still consume the stale bbox in relations.

## RGB-D measurement path

| Surface | Current role | Required treatment |
|---|---|---|
| `mvm-mesh/src/frame.rs` | immutable RGB + stamped sensor bundle | keep as source envelope |
| `mvm-core/src/sensor.rs` | stamps, calibration, metric frame, registered owners | reuse; no tracking IDs |
| `mvm-cli/src/agents/observations.rs` | admits registered metric depth | reuse freshness/skew/calibration admission |
| `mvm-depth/src/{metric,roi,unproject}.rs` | bounded exact-owner geometry | reuse one primitive before and after association |
| `mvm-cli/src/agents/fusion.rs::direct_metric_geometry` | post-track entity sampling | move measurement before association and pass the accepted value forward |
| `mvm-fusion/src/canonical.rs` | publishes `DirectMetricGeometry` facts | remain the only canonical geometry publisher |

There is no pre-association geometry seam at baseline. Recomputing independent
tracker and Fusion geometry would violate T2.

## Downstream authority and leakage points

| Consumer | Current input | T3/T8 requirement |
|---|---|---|
| `HumanObservationFrame` / `SceneObservationFrame` | flat `PersonState` / `ObjectState` vectors | carry lifecycle-aware current/occluded/retained/retired projections |
| `CanonicalFusion` | current vectors only | keep `EntityKey` across accepted gaps and retire explicitly |
| compatibility `WorldState` | overwritten from expert observations | no tentative/lost presence; remembered remains typed |
| activity/anomaly/zone inference | flat people/objects | current observed authority only unless a contract explicitly accepts temporal evidence |
| `DatasetWriterAgent` | canonical packet plus rich expert evidence | `labels.jsonl` remains sole history; no candidate trace |
| StateFile / Scene JSON | compatibility projection | distinguish current, occluded, remembered, and unknown |
| snapshot/recap/inspect MCP | StateFile/session readers | reflect canonical lifecycle and incarnation only |
| Cortex | immutable `FusedSceneFrame` | read-only; never promote tracker hypothesis to fact |

## Configuration inventory

| Field/surface | Baseline behavior | Resolution |
|---|---|---|
| `Config.tracker_iou_threshold` | declared but person `IOU_GATE` is hard-coded | connect to a validated profile or remove |
| `Config.tracker_timeout_frames` | declared but `with_config()` leaves default timeout | connect to a validated profile or remove |
| `relink_*` fields | consumed by `PersonTracker` | fold into versioned internal profile |
| object score/area/timeout/cadence | consumed across CLI/scene | retain only validated operator-level controls; association constants live in profile |
| product `config.toml` | no `[tracker]` section | T9 adds `profile`, diagnostics, and one legacy rollback switch |

## T0 frozen gaps

1. Historical Python A/B/C is not bit-equivalent to production Rust.
2. Existing captured session is sampled at Dataset cadence, not Human cadence.
3. No qualifying calibration or held-out corpus exists.
4. Person `Tentative` leaks downstream; occluded/lost tracks do not have an
   explicit canonical handoff.
5. Object tracking is greedy and lacks confirmation/lifecycle.
6. Metric geometry is post-association only.
7. Internal, display, and canonical IDs are not joined in one diagnostic event.
8. Compatibility loops can diverge from AgentMesh policy.

T1 begins only from this inventory and may not call a simplified tracker model
the production baseline.

## T1 resolution

- `HumanAgent` now calls `PersonTracker::tick_motion_bound`; `SceneAgent` calls
  `ObjectTracker::update_bound`. Both bindings are built from the exact immutable
  `Frame.sensor` packet consumed by their detector tick.
- `TrackerFrameBinding` retains RGB stamp/source/dimensions and, only for admitted paired RGB-D,
  the exact depth stamp, measured skew and calibration revision. Invalid pairing fails before a
  tracker mutation.
- The Rust replay example calls those same bound entrypoints. It replaces the historical Python
  adapter as the production-parity execution seam, while the old A/B/C result remains historical.
- Person and object traces have separate counters and bounded event buffers. Trace snapshots are
  deterministically ordered and never enter `labels.jsonl`.
- The T2 tail is explicit: binding is exact, but metric measurement still happens post-tracking in
  Fusion. T2 must compute and attach Human/Scene measurements on this same packet before association,
  then pass that one admitted measurement forward without duplicate geometry math.

## T2 resolution

- Human and Scene construct `AssociationObservation` on their own detector tick from the exact
  immutable RGB-D packet. Detection keys are local to that tick; measurement has no track or
  canonical entity ID.
- `mvm-depth::measure_association_geometry` is the single bounded production primitive. It samples
  a current mask when available, otherwise a 20%-inset bbox, and records RGB/depth stamps, measured
  skew, calibration, sampling method, quality and a typed honesty status/reason.
- PersonTracker and ObjectTracker perform their existing RGB association, then map the already-made
  measurement from detection key to track ID. The same value is carried in Human/Scene observation
  packets to Fusion; Fusion no longer resamples a later capture frame through stale track boxes.
- Human and Scene geometry retain independent source ticks and timestamps. Fusion rejects future or
  duplicate per-track geometry before mutating canonical state and uses measurement-local provenance
  for `DirectMetricGeometry` evidence.
- Missing depth is `Unknown`; stale/skew, missing owner, empty mask and calibration mismatch are typed
  `Unknown`/`Withheld`. Live RGB detection continues, while malformed frozen replay input remains
  fail-closed before tracker mutation.
- Geometry sampling exists only in `mvm-depth`; tracker trace and measurements remain runtime/replay
  evidence and do not create a second durable semantic history beside `labels.jsonl`.
