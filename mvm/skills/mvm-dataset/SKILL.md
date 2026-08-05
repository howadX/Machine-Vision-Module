---
name: mvm-dataset
description: Record, inspect, validate, retain, or export Machine Vision Module (MVM) semantic and pixel sessions. Use when an agent must manage the Dataset Factory, choose recording or privacy modes, run data-quality gates, or interpret Canonical Observation labels and provenance.
---

# MVM Dataset

Use the Dataset Factory as a structured, self-describing memory of MVM scene
observations. Preserve the same confidence and honesty semantics as the live
scene graph.

## Choose a Recording Mode

Use one of these producer modes:

```bash
mvm run                  # default: durable text-only semantic session
mvm run --record         # semantic labels plus RGB, depth, and masks
mvm run --no-vision-log  # disable session recording
```

Use the default text-only mode for durable scene history without pixels. Use
`--record` only when training or pixel-level inspection requires full data.
Do not describe a default `mvm run` session as a pixel dataset.

Treat flat `scene-*.jsonl` / `events-*.jsonl` as retired legacy output. Current
semantic history and Dataset Factory data share the session store.

## Choose Privacy Deliberately

Set `--privacy` on `mvm run --record`:

- `safe` (default): blur anonymous faces and retain gallery faces
- `anonymous`: blur every face
- `full`: retain every face

Prefer `safe` unless the task explicitly requires another mode. Text-only
sessions contain no pixels, but still contain semantic identities and
activities; handle them as potentially sensitive data.

## Inspect and Maintain Sessions

Use the Dataset Factory CLI:

```bash
mvm dataset list
mvm dataset info <session_id>
mvm dataset stats
mvm dataset check [session_id]
mvm dataset check [session_id] --strict
mvm dataset prune --keep 7d
mvm dataset prune --keep 5 --yes
mvm dataset export
```

Run `mvm dataset check` before trusting or exporting recorded data. Treat any
error as a failed quality gate. Use `--strict` when warnings must also fail CI.

Run prune without `--yes` first and review the dry run. Use `--yes` only after
confirming the retention boundary.

Treat `mvm dataset export` as the source of current exporter commands. The
exporters live under `prototype/pipeline/dataset/`; Python is an offline export
tool here, not part of the Rust vision hot path.

## Read the Session Layout

Treat each `sessions/<id>/` directory as the portable unit:

- `labels.jsonl`: one Canonical Observation per recorded frame
- `session.toml`: finalized session metadata and recording configuration
- `frames/`, `depth/`, and masks: present only when pixel recording saved them

An active session may not have its final `session.toml` yet. Do not call that
corruption without checking whether recording is still in progress.

Use `session.toml` and label metadata for exact current schema values rather
than hard-coding a phase or schema number.

## Preserve Label Semantics

Read Canonical Observation status and provenance together. Keep these statuses
distinct:

- `observed`: directly supported by an expert observation
- `derived`: computed from available scene signals
- `temporal`: inferred across time
- `weak`: useful but insufficiently strong supervision
- `withheld`: deliberately not claimed because evidence is inadequate
- `unknown`: not known

Never map `withheld` or `unknown` to a negative class. Absence of a label is
data, not evidence for the opposite label. Preserve binding keys and provenance
through filtering and export so labels remain traceable to their source entity,
frame, and model/router.

Treat relations, activities, and completed episodes as semantic records from the
same session, not as independent flat-log sources. Account for recording cadence
when comparing episodes with per-frame observations.

## Validate Before Use

Use `mvm dataset check` to verify binding keys, referenced files, mask RLE,
depth ranges, privacy constraints, class balance, and withheld rates. For pixel
datasets, also use the repository's Python validator when the workflow requires
pixel decoding and parity checks.

Do not train on or publish a session that fails its applicable gates. Report
warnings, withheld rates, and known coverage limitations with the resulting
dataset.

## Resolve Contract Questions

Run `mvm dataset --help` for the current CLI and read `session.toml` for the
session's effective settings. When prose disagrees with executable behavior,
follow CLI behavior and session metadata, then flag the documentation drift.

For questions about the live scene or historical scene evidence, load
[mvm-vision](../mvm-vision/SKILL.md).
