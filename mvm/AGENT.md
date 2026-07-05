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

The **producer** must be running for any of this to have data:
`mvm run --scene-log <DIR>` (camera + all models → StateFile + durable
`scene-*.jsonl` / `events-*.jsonl` logs in `<DIR>`). `snapshot` reads the
StateFile; `recap`/`inspect` read the logs.

```
mvm run --scene-log ~/vision/logs     # producer (start once, leave running)
mvm snapshot                          # what's happening now
mvm recap  --input ~/vision/logs --last 10m
mvm inspect --input ~/vision/logs --around 2026-07-05T17:51:23 --window 5
```

## How to read a snapshot

```
# Scene 17:51:23 · occupied · anomaly 0.00 · 18 fps
Zone «workstation»: Вадим(0), laptop(10001), mouse(10002)

| id | who | dist | posture | motion | emotion | head | hands |
|----|-----|------|---------|--------|---------|------|-------|
| 0  | Вадим ✓ | 0.8m | sitting (high) | idle | Neutral (high) | camera | using_mouse |
| 10 | anon | 1.0m | standing (medium) | idle | — | — | — |

Active: seated_at_workstation · using_mouse

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
  - **posture / emotion** — label + confidence **word** in parens: `low`/`medium`/`high`.
  - **motion** — `idle` / `walking` (+ velocity). **head** — where the face points.
  - **hands** — active hand/object interaction (e.g. `using_mouse`, `holding_drink`).
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

Two logs run at different rates: the **event journal** (`events-*.jsonl`) records
activity start/stop **per frame**, but the **scene log** (`scene-*.jsonl`, what
`inspect` reads) is written at the **dump cadence**. A very brief event can appear
in a recap's episode list yet have **no matching scene frame** in the inspected
window. Absence in `inspect` doesn't disprove a recap episode; it means the moment
fell between scene dumps. Widen `--window` before concluding.

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
