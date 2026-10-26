# Scripted RGB-D corpus plan for MVM and tracker validation

> Status: capture runbook. This document does not approve a capture, a truth annotation, or a
> tracker policy by itself.
>
> Primary operator: `P1`, already assigned to the calibration split by C03. A held-out person must
> be a different consenting participant. Pixels are recorded only after explicit consent and an
> explicit privacy choice.

## 1. Purpose and evidence boundary

The corpus is a reusable, frozen source for:

- person presence, lifecycle, internal-ID continuity, display identity, occlusion and re-entry;
- object confirmation, continuity, removal, reappearance, overlap and hand occlusion;
- person-object relations and conservative activity evaluation;
- RGB-D geometry quality, depth holes and boundary contamination;
- deterministic production Rust replay and controlled same-corpus A/B tests.

Every accepted take produces two synchronized evidence products:

1. a canonical Dataset Factory pixel session (`labels.jsonl`, RGB, registered metric depth and
   `session.toml`); and
2. a schema `mvm-tracker-tick-capture/1.4` tracker calibration capture containing exact Human and
   Scene ticks, association decisions, enrichment applications and birth conflicts.

Tracker traces remain evaluation evidence outside `labels.jsonl`. Manual truth is stored under
`benchmark/tracker-reliability/annotations/` and must be created from consented RGB review without
using detector, tracker or canonical output as truth. `withheld` and `unknown` are never negative
labels.

## 2. Split discipline

| Split | People | Room/setup | Objects | Use |
|---|---|---|---|---|
| scratch | any consenting rehearsal participant | any | any | framing and script rehearsal only; never admitted to a manifest |
| calibration | `P1`; multi-person scenes add `P3` | setup `C` | physical instances `C-*` | debugging and selection of profile constants |
| held-out | `P2`; multi-person scenes add `P4`/`P5` | setup `H`, not setup `C` | separate physical instances `H-*` | frozen acceptance only; never viewed while tuning |

Existing C03 binds `P1` to calibration. New `P1` takes therefore cannot enter development or
held-out manifests. A session, person, room setup, physical object instance, or parent take never
crosses calibration and held-out. A failed or rehearsed take is marked rejected; frames from it are
not salvaged into another split.

Before held-out recording, freeze:

- the candidate binary and its SHA-256;
- tracker profile/config and model digests;
- all calibration decisions;
- held-out scene scripts, without looking at candidate output;
- participant consent, privacy mode and object-instance inventory.

## 3. Recording standard for every take

### Physical setup

- Keep camera mount, Kinect serial and calibration revision fixed within one split.
- Mark three floor positions: `L`, `C`, `R`; mark `near`, `mid`, `far` distances from the camera.
- Use stable split-local object IDs, for example `C-bottle-a`, `C-bottle-b`, `C-book-a`,
  `C-remote-a`, `C-laptop-a`. Held-out uses different physical instances with `H-*` IDs.
- Record clothing description and any non-biometric identity marker used for truth review. Do not
  let the marker become an MVM association cue.
- Remove unrelated people, moving screens and reflective clutter unless the scene explicitly tests
  them.

### Take envelope

1. Start with 5 seconds of a completely empty frame.
2. Perform only the scripted actions, at normal safe speed unless the script says `fast`.
3. Pause for about 1 second at each named waypoint so transitions are reviewable.
4. End with at least 10 seconds of a completely empty frame for phantom/expiry evidence.
5. Use one continuous take with no cuts. Repeat the whole take if the script was missed.
6. Target 45-90 seconds and 30 FPS. Record actual frame counts rather than assuming exact timing.

### Capture command template

After explicit consent and privacy selection, use a fresh root for every take:

```bash
target/release/mvm run \
  --sensor-backend kinect_v1 \
  --person-tracking-mode candidate \
  --record --privacy safe \
  --tracker-capture <take-root>/tracker \
  --max-frames <take-frame-budget>
```

`safe` is the default recommendation, but it retains a recognized gallery face. Select
`anonymous` when all faces must be blurred; use `full` only with explicit need and consent. Record
the effective value from session metadata, not from operator memory.

Run a 90-frame sensor smoke before each recording block. Abort the block on RGB-D pairing loss,
calibration revision drift, source errors, timestamp disorder or a dirty shutdown.

## 4. Required scene families

Each calibration scene gets two accepted repeats: the second reverses direction/order where
possible. Held-out records the same scene family with different routes, timings, participants,
setup and physical objects; it is not a frame-for-frame imitation.

### A. Negative and environment scenes

| ID | Duration | Script | Required evidence |
|---|---:|---|---|
| `N01-empty-static` | 60 s | Empty room. Monitor/laptop, bottle and furniture remain fixed. No person enters. | false births, tentative expiry, display conflict, empty-tail phantoms |
| `N02-screen-reflection` | 60 s | Empty room; play moving human imagery on the display for 20 s, then stop it. If a mirror/reflection exists, include it without a real person entering the camera view. | poster/screen/reflection false person; no canonical presence |
| `N03-object-only-change` | 60 s | Begin empty; an object moves into/out of view from outside the camera FOV using a safe non-person-visible method, then the room is static. | object birth/removal without person birth; no invented interaction |
| `N04-lighting-change` | 45 s | Empty room; switch normal room lighting once, wait, restore it. Do not move the camera. | detector flicker, no phantom publication, depth quality honesty |

### B. One-person motion and lifecycle scenes

| ID | Duration | Operator choreography | Required evidence |
|---|---:|---|---|
| `P01-enter-exit-reenter` | 75 s | 5 s empty → enter from `L` → stop at `C` facing camera 5 s → exit `R` 5 s → re-enter `R` → stop at `C` → exit `L` → 10 s empty. | two births only if expiry boundary requires it; continuity within each presence; correct exit/re-entry/incarnation |
| `P02-slow-traverse` | 60 s | Enter `L`; walk `L→C→R→C→L`, pausing at each mark; face camera, then side-on; exit. | stable ID under ordinary translation and direction reversal |
| `P03-fast-edges` | 60 s | Enter quickly from `L`, cross to `R`, briefly leave half the body at each edge, return quickly to `C`, stop, exit. No running near obstacles. | fast-motion fragmentation, edge partials, bounded relink |
| `P04-near-far-turn` | 75 s | At `C`, move `far→mid→near→mid→far`; at each distance face camera, turn 90°, turn back; exit. | scale change, metric-Z ordering, face loss/recovery, same internal ID |
| `P05-posture-transition` | 90 s | At `C`: stand 5 s → sit on fixed chair 8 s → stand → bend to pick a floor marker → stand → safely lie only if the area is prepared → stand → exit. | bbox/extent and posture changes must not create new identity; withheld pose remains honest |
| `P06-partial-furniture` | 75 s | Walk behind chair/desk so lower body is hidden; pause with upper body visible; move laterally behind it; emerge; exit. | partial-body continuity, body-validation freshness, no furniture merge |
| `P07-opaque-occlusion-ladder` | 90 s | At `C`, pass behind an opaque screen for approximately 0.2 s, 0.5 s, 1 s and 3 s, returning to the same mark after each; finally exit for the empty tail. | observed→occluded/lost/return boundaries, same-ID recovery versus honest new incarnation |
| `P08-display-overlap` | 75 s | Start beside the monitor; cross in front of it, pause partly overlapping it, fully cover it briefly, move away, turn face toward camera, exit. | real person must override display birth conflict; monitor must not survive as a person |

### C. Person-object interaction scenes

| ID | Duration | Operator choreography | Required evidence |
|---|---:|---|---|
| `O01-bottle-pick-carry-place` | 75 s | Enter empty-handed → reach for `bottle-a` → pick it up → carry `L→R` → pause → place it at a new marked position → release hands → exit. | person and bottle ID continuity, touching candidate, carry/remove/reappearance, no false physical hold claim |
| `O02-bottle-hand-occlusion` | 75 s | Pick up `bottle-a`; alternately cover it with one/both hands, hold it against torso, reveal it, place it back, exit. | object remembered versus observed, mask/bbox boundary quality, no duplicate object |
| `O03-workstation-tools` | 90 s | Sit at desk → touch keyboard/laptop → pick up `remote-a` → place it down → open/close `book-a` → stand and exit. | conservative `using_object`, reaching/hand evidence, workstation context without identity churn |
| `O04-remove-reappear` | 75 s | Enter → remove `book-a` completely from frame for 5 s → return the same instance to a distant mark → remove it again → exit carrying it. | object expiry/relink/new incarnation decision with truth instance fixed |
| `O05-same-class-crossing` | 90 s | Place `bottle-a` at `L` and `bottle-b` at `R`; pick both up, cross their image paths, overlap briefly, separate, and place them at swapped positions. | same-class object ID switch, false merge, abstention and depth/mask cues |
| `O06-person-object-display` | 75 s | Carry `bottle-a` across the monitor region, set it in front of the display, reach behind it for `remote-a`, then clear the area and exit. | simultaneous person/display/object overlap; no cross-domain ID or birth leakage |

### D. Multi-person scenes

These require additional consenting participants. Calibration uses `P1+P3` (and `P6` for three
people); held-out uses `P2+P4` (and `P5`). A participant is never reused across splits.

| ID | Duration | Choreography | Required evidence |
|---|---:|---|---|
| `M01-two-crossing` | 90 s | Two people enter opposite edges, cross at `C`, separate, reverse and cross again; one faces away on the second pass; both exit separately. | ID switches, false merge, assignment order independence |
| `M02-two-full-occlusion` | 90 s | One person stands at `C`; the second passes fully in front for 0.5 s and 2 s; swap front/back roles; separate and exit. | entity-local occlusion and correct reappearance |
| `M03-similar-clothing-no-face` | 90 s | Participants wear similar outer colors; begin side/back to camera, cross, then reveal faces one at a time without touching. | geometry limits, identity unknown discipline, no gallery cloning |
| `M04-object-handoff` | 90 s | `P1` picks up `bottle-a`, hands it to `P3`, both separate, `P3` places it down, then each exits through a different edge. | person IDs plus one object instance, handoff relation boundaries, no owner invention |
| `M05-three-person-crossing` | 105 s | Three people enter from distinct marks, converge near `C`, pause without touching, leave along different routes, then re-enter in a changed order. | three-person false merge/relink, fragmentation, ambiguity abstention |

### E. Special coverage

| ID | Duration | Script | Required evidence |
|---|---:|---|---|
| `E01-depth-boundary` | 75 s | One person carries matte and depth-problematic/reflective split-local objects across silhouette boundaries and places them at near/mid/far marks. | holes, owner mismatch and contamination must yield typed weak/withheld/unknown geometry |
| `E02-animal-pass` | 60 s | Only if safe and naturally available: a consenting household scene where an animal passes with no person visible, then a person enters separately. Otherwise leave the corpus explicitly incomplete. | animal/person false birth separation; no simulated substitute may count as animal truth |
| `E03-rgb-only-parity` | 60 s | Repeat `P01`, `P07`, `O01` and `M01` through the accepted RGB-only producer/profile. | independent fallback acceptance; RGB-D evidence must not leak into RGB truth |
| `E04-sensor-health` | separate campaign | Empty-room bounded runs covering reconnect, packet loss and calibration revision change. Do not mix induced transport faults into accuracy takes. | fail-closed source health, no stale facts, clean recovery |

## 5. Collection order

Do not record everything in one day-long session. Use separate blocks so a fault invalidates only a
small unit:

1. **Block C-N:** calibration negatives `N01-N04`.
2. **Block C-P1:** `P01-P04`; review framing before continuing.
3. **Block C-P2:** `P05-P08`.
4. **Block C-O:** `O01-O06` with a photographed split-local object inventory.
5. **Block C-M:** `M01-M05` after extra participant consent.
6. **Block C-E:** depth/animal/RGB-only/health special coverage.
7. Validate, annotate and sign calibration; select/freeze the candidate.
8. Prepare setup `H` and objects `H-*`; record the held-out mirror blocks with new participants.
9. Freeze hashes and truth before executing the candidate on held-out.

For calibration, require two clean repeats per ordinary scene and three repeats for `P03`, `P07`,
`O05`, `M01`, `M02` and `M05`. Held-out needs at least two independent takes per critical family;
zero observed failures in a family with no actual critical event is `insufficient_evidence`.

## 6. Truth annotation plan

For each accepted take annotate from RGB/depth review, not MVM output:

- `truth_id` for every person and physical object instance;
- exact `absent|visible|partial|occluded|unknown` intervals;
- bbox keyframes at entry, exit, direction/scale changes, occlusion boundaries and crossings;
- object possession is not a truth label unless contact is visually reviewable;
- entry, exit, re-entry, pick, contact, release, removal and reappearance markers;
- ambiguity intervals where even a human reviewer cannot distinguish instances;
- known false-positive regions such as monitor, poster, reflection or furniture;
- deviations from the script and unusable intervals;
- reviewer identity/time, independent review source and explicit operator signoff.

The approval workflow is:

```text
recorded
  → session closed
  → structural + pixel validation
  → independent manual truth
  → second-pass review
  → operator signoff
  → SHA-256 pin
  → manifest admission
  → frozen replay/evaluation
```

An annotation remains provisional until signoff. Do not repair truth after looking at candidate
errors; append a correction record with provenance or reject and re-record the take.

## 7. Per-take acceptance checklist

- [ ] Consent and privacy mode recorded before capture.
- [ ] Expected Kinect serial and calibration revision.
- [ ] 90-frame smoke passed with acceptable pairing and no timestamp/resync errors.
- [ ] Script ID, split, repeat, participants, setup and object instances recorded.
- [ ] At least 5 s empty lead and 10 s empty tail.
- [ ] Capture closed cleanly; `session.toml` exists.
- [ ] `mvm dataset check <session_id> --strict` passes.
- [ ] `python3 ../prototype/pipeline/dataset/validate.py <session_dir> --strict` passes.
- [ ] Tracker tick capture, enrichment application log and Dataset session frame binding are complete.
- [ ] All referenced files and the full capture tree are SHA-256 pinned.
- [ ] Production Rust replay reconstructs every person and object tick deterministically.
- [ ] Truth coverage/reviewer provenance are complete and operator signoff is non-null.
- [ ] No person, setup, object instance or parent session leaks across splits.

## 8. Evaluation views

Report person and object domains separately. At minimum retain raw counts and per-session metrics for:

- truth-visible recall, missed-visible and false-present/phantom ticks;
- birth latency, fragmentation, ID switches, false merges and false relinks;
- correct exit, empty-tail persistence, re-entry and canonical incarnation behavior;
- object confirmation, removal/reappearance, same-class switch/merge and remembered/current status;
- relation/activity precision and recall only where truth is reviewable;
- geometry `observed|weak|withheld|unknown` rates and honesty/provenance violations;
- capture/Tracking/Human/Scene/Fusion p50/p95, delivered FPS, drop ratio, RSS and agent errors.

Policy promotion still requires the contract gates: zero observed false merges, false relinks and
honesty violations; independent RGB-only held-out acceptance; deterministic output; and at most 5%
controlled same-run throughput regression. A sparse corpus cannot prove a zero-error claim.

## 9. First practical recording tranche

The smallest useful next tranche is calibration-only and does not require extra people:

1. `N01-empty-static`;
2. `P01-enter-exit-reenter`;
3. `P03-fast-edges`;
4. `P07-opaque-occlusion-ladder`;
5. `P08-display-overlap`;
6. `O01-bottle-pick-carry-place`;
7. `O05-same-class-crossing`.

Record two accepted repeats of each, with the second repeat reversing direction/order. This tranche
directly covers the current live failure (fragmented presence/display IDs), the known monitor phantom,
short/long occlusion, object interaction and same-class object identity. It can guide calibration but
cannot promote the tracker without multi-person, held-out, RGB-only and special sensor coverage.
