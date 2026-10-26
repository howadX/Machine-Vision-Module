# T3 lifecycle and publication record

- Date: 2026-08-13
- Person confirmation: 3 accepted Human ticks
- Object confirmation: 2 accepted Scene ticks
- Durable-data impact: none; `labels.jsonl` remains the sole semantic history
- Dependencies/models: unchanged

## Contract result

Person and object trackers now expose current published observations separately from tracker-local
hypotheses, occluded evidence, retained outward IDs, and expiry retirements. `Tentative` and `Lost`
tracks never become current `PersonState`/`ObjectState`. Human and Scene packets forward the lifecycle
handoff to Fusion while metric geometry is restricted to current published IDs.

`CanonicalFusion` uses retained IDs only to reserve an already-active incarnation. A retained-only ID
cannot mint an entity, and an occlusion frame contains no entities, relations, activities, or zones.
Transition memory is retained too: an accepted gap produces no false exit/enter or zone-free/occupied
pair. Explicit expiry emits disappearance/exit, retires the key, and reuse advances incarnation.

The compatibility StateFile/Scene JSON path applies the same person/object filters. Relative-depth
object anchoring accepts only published persons. DatasetWriter, MCP, and Cortex require no parallel
filter because they consume the immutable canonical packet after this boundary.

## Deterministic production replay

- Input: `replay/production-baseline.json`
- Input SHA-256: `b99dfab66f9615b7b3119ee06ae7f95f25924a7cc0f96e20b976e9e571fca8b8`
- Output SHA-256: `d19f7f321113b70c9dba1f8cd30d7be74b87c69e5d37594ed0a51ae6b2e284fb`
- Two independent executions: byte-identical
- Final person: age 4, `confirmed`; publication events: 2 `tentative_suppressed`, 2 `observed`
- Final object: age 3, `confirmed`; publication events: 1 `tentative_suppressed`, 2 `observed`,
  1 `occluded_handoff`

## Gates

- single-frame person and object detections remain tracker hypotheses and publish zero downstream
  entities;
- confirmation latency is bounded by domain expert ticks, not camera frames;
- incarnation remains stable through accepted occlusion and changes only after retirement;
- retained gaps produce zero canonical presence, relation, activity, or zone facts and no false
  disappear/appear events;
- legacy StateFile/JSON rejects tentative person and object hypotheses;
- full workspace: all tests pass, with one pre-existing live-camera test intentionally ignored;
- no new dependency, model, tracker table, durable store, or sensor-owned ID.
