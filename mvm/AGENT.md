# MVM for agents — how an LLM drives the Machine Vision Module

*Load this file into context to use MVM. It is the single source of truth for the
CLI `--help`, the MCP tool descriptions, and how to read MVM's output. If they
ever disagree, this file wins.*

## What MVM is

A **local, $0, CPU-only vision module**. It watches a camera and emits **facts,
not pixels**: a metric 3D scene graph each frame — who is present, where (metres,
origin = camera), their posture / motion / emotion, and what they're doing —
plus per-signal confidence. No cloud, no API, private. It is an ensemble of small
specialist models (face, identity, emotion, body pose, objects, depth, hands,
segmentation) fused by plain arithmetic — **not** a VLM.

You never look at an image. You call MVM and read structured text.

MVM serves **two consumers**. Most of this file is consumer 1 — **you, an LLM**,
reading the live scene through the pull model. Consumer 2 is the **Dataset Factory**
(phase 16): the same scene graph, recorded to disk as an accumulating, interpretable
training dataset. You rarely need it in a scene-reading loop, but you can start and
inspect it — see [Recording a dataset](#recording-a-dataset-the-second-consumer).

## The pull model — cheap → expensive

Reach for the cheapest tool that answers the question; escalate only when it
points you deeper.

| Tool | Answers | Cost | When |
|------|---------|------|------|
| **snapshot** | What's happening **now** | ~300 tokens | Always start here. |
| **recap** | What **happened** over a window | ~hundreds of tokens (logs → ~190× compressed) | Summarise a period without reading raw frames. |
| **inspect** | The **raw frames** behind a claim | grows with the window | Adjudicate something snapshot/recap flagged. Not for browsing. |

Rule of thumb: `snapshot` → `recap` → `inspect`. Don't `inspect` until a cheaper
call has pointed you at a specific moment or track id.

## Two ways to access it

1. **MCP tools** (when MVM runs as `mvm serve --mcp`): `mvm_snapshot`,
   `mvm_recap`, `mvm_inspect`. Read-only; the server reads the artifacts a running
   producer writes. Prefer this in an agent loop.
2. **CLI / files** (when you have a shell): `mvm snapshot`, `mvm recap`,
   `mvm inspect` — or just read the files a running producer keeps fresh:
   `current_state.md` (the snapshot) and `journal.md` (a rolling recap of the day).

The **producer** must be running for any of this to have data: just `mvm run`
(camera + all models → StateFile + a durable, text-only **semantic session**
under `~/.local/share/mvm/datasets/sessions/`, on by default since phase 16.7).
`snapshot` reads the StateFile; `recap`/`inspect` read the sessions (no flag —
they resolve the default data dir). The old flat `scene-*.jsonl`/`events-*.jsonl`
logs are retired; pass `--scene-log <DIR>` only if you need them for back-compat.

```
mvm run                               # producer (start once, leave running)
mvm snapshot                          # what's happening now
mvm recap   --last 10m                # summary over the default session store
mvm inspect --around 2026-07-05T17:51:23 --window 5
```

## How to read a snapshot

```
# Scene 17:51:23 · occupied · anomaly 0.00 · 18 fps
Zone «workstation»: Вадим(0), laptop(10001), mouse(10002)

| id | who | dist | posture | motion | emotion | head | hands |
|----|-----|------|---------|--------|---------|------|-------|
| 0  | Вадим ✓ | 0.8m | sitting (high) | idle | Neutral (high) | camera | using_object(mouse) |
| 10 | anon | 1.0m | standing (med) | idle | — | — | — |

Active: seated_at_workstation · using_object(mouse)

⚠️ id10: тело не подтверждено лицом (животное или упавший — не различить)
```

- **Header**: timestamp · `occupied`/`empty` · `anomaly` 0–1 (0 = calm) · delivery fps.
- **Zones**: fused groups (people + objects that are together), e.g. `workstation`.
- **Table**, one row per tracked person:
  - **id** — track id. **0–9 = gallery (known people)**, **≥10 = anonymous people**,
    **≥10000 = objects**. Gallery ids are sticky; anon ids are ephemeral and reused.
  - **who** — `Name ✓` (face-verified identity) · `anon ✓face` (face seen, not in
    gallery) · `anon` (body only, no face).
  - **dist** — metres from camera (Z). Ordinal is trustworthy; absolute is ±15–30%.
  - **posture / emotion** — label + confidence **word** in parens: `low`/`med`/`high`.
    Posture may also carry **`torso_flexion`** (continuous torso inclination in degrees,
    v0.2.0) — a bent-forward torso reads high even while the discrete label stays `standing`.
  - **motion** — `idle` / `walking` (+ velocity). **head** — where the face points.
  - **hands** — active hand/object interaction. Labels: **`using_object(<class>)`**
    (`using_object(keyboard)`, `using_object(mouse)`, `using_object(cell phone)`, …) and
    **`holding_drink`**. (v0.2.0 folded the old `typing`/`using_mouse`/`using_phone`/`reading`
    into the honest `using_object` umbrella and dropped `watching_tv` — see below.)
- **Active** — current activities (with durations when the journal is running).
- **⚠️ notes** — the frame's honest uncertainty, made explicit (see below).

## Honesty — do not over-read

MVM states what it *didn't* see rather than guessing. Treat these as first-class:

- A **bare label under a ⚠️ note, or with `low` confidence, is not a fact.** Say
  "possibly", or `inspect` to confirm.
- **`тело не подтверждено лицом`** ("body not confirmed by a face") — could be a
  person, an animal, or a fallen body. Don't assert "a person" from this alone.
- **`поза withheld`** — only the upper body was visible, so posture is *withheld*,
  not guessed. Absence of a posture label ≠ unknown-because-broken.
- **`поза OOD (plausibility low)`** — the pose is out-of-distribution; don't trust it.
- **`участники нестабильны`** — an activity's participants include an ephemeral id;
  the activity may be real but the *who* is uncertain.
- A **phantom** (a brief spurious track, often a walking person fragmenting into
  extra ids) reads like a person in a single frame. It is **not** a person until
  `inspect` shows a coherent track across frames. A high `anomaly` with a lone new
  id is a classic phantom tell.

(Some ⚠️ notes are in Russian — the deployment language. The tokens above are the
stable markers to match on.)

## A cadence caveat for `inspect`

A semantic session records per-frame observations at the **record cadence**, while
completed **episodes** are written inline on the exact frame they close. A very brief
episode can appear in a recap's activity list yet have **no matching per-frame
observation** in the inspected window. Absence in `inspect` doesn't disprove a recap
episode; it means the moment fell between recorded frames. Widen `--window` before
concluding.

## Contract changes since v0.1.0 (the ML-refactor line, shipped in v0.3.0)

The ML refactor (phase 15, journals LJ0–LJ6) changed three things a consumer sees.
It was prepared as the 0.2.0 line but never tagged on its own — the changes ship in
the current release **v0.3.0** (a v0.1.0 consumer upgrading to v0.3.0 sees all three):

1. **`torso_flexion`** — a new *optional* posture field (continuous torso inclination
   in degrees). Present when the torso is visible; absent otherwise. Additive — old
   readers ignore it safely.
2. **Activity vocabulary v2** — the flat verbs `typing` / `using_mouse` / `using_phone`
   / `reading` are gone; a wrist-on-object interaction is now **`using_object(<class>)`**
   (the object class rides as a parameter). `holding_drink` is kept. **`watching_tv`
   was removed** — "watching" is a head-facing signal, not a wrist contact, and the
   supervising corpus had zero real instances of it. If you matched on the old strings,
   switch to the `using_object(` prefix.
3. **Pose-OOD semantics** — `plausibility` (`low`/`ok`) is now derived from a Mahalanobis
   distance, not the old VAE. The `recon_error` field is **deprecated** and always `null`
   (kept only for schema stability). Read `plausibility`, not `recon_error`.

## Recording a dataset (the second consumer)

Beyond answering "what's happening now", MVM **accumulates** what it sees as a
structured, self-describing store — the Dataset Factory (phase 16/16.7). This is memory,
not an archive: each recorded frame is a Canonical Observation with provenance and
honest label statuses, not just pixels. Since phase 16.7 the store is **on by default**:
`mvm run` keeps a durable **text-only** semantic session (labels only, no pixels);
`--record` upgrades it to a full pixel dataset; `--no-vision-log` turns it off.

```
mvm run                     # default: text-only semantic session (durable vision log)
mvm run --record            # upgrade to a pixel dataset (privacy=safe by default)
mvm dataset list            # sessions on disk (frames, participants, activities)
mvm dataset info <id>       # one session in detail
mvm dataset stats           # dataset-wide totals
mvm dataset check [id]      # data-quality gate — exit 1 on any error (CI-gateable)
mvm dataset prune --keep N  # retention (drop old sessions)
mvm dataset export          # → ready-to-run Python export commands (PyTorch/TFRecord/WebDataset)
```

- **Self-describing on disk.** A session is `labels.jsonl` (one observation per frame,
  carrying entities + zones/edges + activities + completed episodes) + `session.toml`;
  under `--record` also `frames/` + `depth/` + masks. Copy the folder to another machine
  and it still works. Details: `prototype/pipeline/dataset/README.md`.
- **Same honesty contract.** A `withheld` posture is recorded as withheld, **never** as a
  negative class — absence is data (§16.0.2). The exporters and `check` enforce this.
- **Privacy-first.** `safe` (default) blurs anonymous faces and keeps only gallery
  identities; `anonymous` blurs everyone; `full` keeps all. Set `--privacy` on `run`.
- **`mvm dataset check`** is the quality gate: it verifies integrity (binding keys,
  referenced files, mask RLE, depth ranges), flags privacy leaks (no raw face embeddings
  in labels), and reports class balance + withheld-rate. Run it before trusting a session.

## What MVM does NOT do

MVM answers a fixed, closed set: presence, identity (gallery), emotion, posture,
motion, position, object interaction, zones, activities, anomalies. It does **not**
do open-vocabulary detection, visual question answering, zero-shot recognition, OCR,
or "describe this arbitrary thing". Those need a **VLM** — MVM is the cheap,
private, interpretable first pass (~80% of everyday scenarios at ~0.1% of VLM cost);
fall back to a VLM for the open-world tail.

---
*Interfaces: `mvm <command> --help` for flags; `mvm serve --mcp` for the MCP tools.
Deep architecture / crates / phases: `mvm/CLAUDE.md`.*
