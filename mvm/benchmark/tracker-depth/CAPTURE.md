# Kinect tracker-depth capture protocol

This protocol collects a captured, human-reviewable tracker corpus without
changing the production tracker. The temporary run uses the existing Kinect
RGB-D source, canonical Fusion output, Dataset Factory and Scene JSON surface.

## Capture contract

- Kinect packets are strict RGB-D pairs admitted at at most `20 ms` skew.
- Dataset cadence is `1`; latest-only Fusion frames must not also be filtered by
  a modulo cadence.
- `--max-frames 1800` provides a bounded, clean shutdown and finalized
  `session.toml`; an external timeout is only a safety guard.
- `--record --privacy full` retains RGB for operator-authorized annotation.
- `--dump-json --interval-ms 100` records the existing tracker-facing live
  projection independently of `labels.jsonl`.
- `--masks` remains requested so absence of the optional segmentation model is
  explicit in startup evidence. Missing masks are not synthesized.

## One-person motion script (60 seconds nominal)

| Time | Action | Intended slice |
|---:|---|---|
| 0–5 s | sit still, face camera | stable baseline |
| 5–15 s | stand, walk slowly left then right | ordinary motion |
| 15–20 s | approach and remain partly clipped at a frame edge | partial body |
| 20–25 s | leave the frame completely | exit |
| 25–30 s | remain out of frame | empty interval / occlusion |
| 30–35 s | return from the opposite edge | re-entry / relink |
| 35–43 s | faster lateral movement left-right | fast motion |
| 43–50 s | briefly self-occlude behind the chair/desk, then reappear | short occlusion |
| 50–56 s | sit and lean sideways, then upright | posture/bbox shape change |
| 56–60 s | sit still facing camera | stable closing baseline |

The operator is the sole known physical person and receives ground-truth id
`person_1`. This fact does not make every detector node a person: RGB frames are
reviewed and detections without visible operator support are annotated as false
positives/unknown.

## Raw evidence

The run must retain:

- `labels.jsonl`, finalized `session.toml`, RGB JPEG and metric `u16` depth PNG;
- Scene JSON stdout and agent-health stderr;
- per-sample sensor identity, stamps, skew, calibration revision and valid ratio;
- canonical tracker id, identity, position and available semantic attributes;
- file digests and the exact invocation/configuration.

## Annotation and benchmark derivation

Human review assigns per saved frame: visibility (`visible`, `partial`,
`occluded`, `absent`, `uncertain`), normalized body bbox where visible,
truth id, review status and notes. Production track ids are copied as predictions,
not as truth. Metric XYZ is retained only when current MVM output supplies it;
missing/weak geometry remains null/weak.

The captured replay is immutable after its digest is recorded. A/B/C evaluation
must report captured results separately from the synthetic scenario corpus and
must not call proxy values TrackEval metrics. Missing multi-person crossing,
object or mask coverage is an explicit limitation, not synthetic evidence
relabeled as captured ground truth.

## Gates

1. clean bounded shutdown and `session.toml` present;
2. strict Rust dataset check and pixel validator pass;
3. RGB/depth artifact counts match labels, all paths resolve;
4. every saved sample is paired, registered metric `u16 mm`, calibration-pinned,
   skew at most `20 ms`;
5. annotations cover all retained frames or explicitly mark unreviewed frames;
6. replay/evaluator contract tests and frozen digest pass;
7. report coverage gaps separately from measured tracker failures.
