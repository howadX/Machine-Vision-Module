---
name: mvm-vision
description: Read the live or historical scene through the local Machine Vision Module (MVM). Use when an agent must answer what the camera sees now, what happened over a time window, or inspect evidence behind a scene claim through MVM snapshot, recap, or inspect via MCP or CLI.
---

# MVM Vision

Use MVM as a local, CPU-only source of structured scene facts. It emits a metric
3D scene graph with per-signal confidence; it is an ensemble of specialist
models, not a VLM. Read its text output rather than camera pixels.

## Stay Within MVM's Boundary

Use MVM for presence, gallery identity, emotion, posture, motion, metric position,
object interaction, zones, activities, and anomalies.

Do not use it for open-vocabulary detection, arbitrary visual question
answering, zero-shot recognition, OCR, or general image description. Those
open-world tasks require another vision system.

## Use the Pull Model

Choose the cheapest operation that can answer the question:

1. Call `snapshot` for the current scene. Always start here for a live-scene
   question.
2. Call `recap` for a summarized time window.
3. Call `inspect` only to adjudicate a specific moment or track identified by a
   cheaper call. Do not browse history with it.

Prefer the read-only MCP tools in an agent loop:

- `mvm_snapshot`
- `mvm_recap`
- `mvm_inspect`

Use the equivalent CLI when a shell is available:

```bash
mvm snapshot
mvm recap --last 10m
mvm inspect --around 2026-07-05T17:51:23 --window 5
```

Require a running producer. Start `mvm run` once and leave it running, or use
`mvm serve --mcp --with-camera`. `snapshot` reads `current_state.md`;
`recap` and `inspect` resolve the default semantic-session store automatically.
The producer writes a durable text-only session by default.

Treat `--scene-log` and flat `scene-*.jsonl` / `events-*.jsonl` as legacy
backward compatibility, not the normal workflow.

## Read a Snapshot

Interpret the header as timestamp, occupancy, anomaly score from 0 to 1, and
frame-delivery rate. Interpret zones as fused groups of nearby people and
objects.

Read identifiers by range:

- `0-9`: reserved gallery identities; sticky while present
- `10-9999`: anonymous people; ephemeral and reusable
- `>=10000`: objects

Read `Name ✓` as face-verified gallery identity, `anon ✓face` as a detected face
without a gallery match, and `anon` as body-only evidence.

Treat distance ordering as stronger than absolute distance. Positions are
metric estimates, but absolute depth is typically approximate by 15-30%.

Read posture and emotion with their `low` / `med` / `high` confidence. Read
`torso_flexion` as an optional continuous torso inclination. Read hand-object
activity as `using_object(<class>)` or `holding_drink`; do not expect the retired
labels `typing`, `using_mouse`, `using_phone`, `reading`, or `watching_tv`.

## Preserve Honesty

Never turn an absent, `withheld`, or `unknown` signal into a negative class.
Report uncertainty explicitly.

- Treat a low-confidence label or a label covered by a warning as tentative.
- Treat `тело не подтверждено лицом` as body-shaped evidence that may be a
  person, animal, or fallen body. Do not assert a person from it alone.
- Treat `поза withheld` as an intentional refusal caused by insufficient body
  coverage, not as standing, not-lying, or a detector failure.
- Treat `поза OOD (plausibility low)` as untrustworthy.
- Treat `участники нестабильны` as uncertainty about who participated, even when
  the activity itself may be real.
- Treat a brief new track as a possible phantom until `inspect` shows a coherent
  track across frames. A high anomaly score plus a lone new id is not proof of a
  new person.

Use qualified language such as "possibly" when confidence is insufficient.

## Account for Recording Cadence

Semantic observations are sampled at the recording cadence. Completed episodes
are attached to the exact recorded observation on which they close. A brief
episode can therefore appear in `recap` without a matching per-frame activity in
a narrow `inspect` window.

Widen `--window` before concluding that the recap is wrong. Absence in one
inspected slice does not disprove an episode.

## Resolve Contract Questions

Treat executable interfaces as authoritative when exact flags or current schema
details matter:

- Run `mvm <command> --help` for CLI flags.
- Inspect `mvm serve --mcp` tool definitions for MCP arguments.
- Read `session.toml` and label metadata for a session's exact schema/config.

Do not infer current interfaces from old phase documents or legacy Python
bridge instructions.
