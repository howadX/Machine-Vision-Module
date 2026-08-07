# Phase 19.2.2 Semantic Model Spike

This directory contains two measured **reject/defer** records: Iteration A for
PP-MobileSeg-Tiny ADE20K and Iteration B for the research-allowed SegFormer-B0
SceneParse150 model. Neither is a runtime
expert, and no downloaded or converted model bytes are committed.

## Why this candidate

PP-MobileSeg-Tiny is the lightest official PaddleSeg ADE20K variant with the
indoor classes needed by the 19.2.1 taxonomy. The implementation is Apache-2.0,
but the official pretrained model is trained on ADE20K and the upstream scene
parsing project limits pretrained models to research use. Production
redistribution is therefore not considered clear.

## Measured result

The official 512x512 VIM export converted to ONNX but failed Paddle/ONNX parity.
A second export used PaddleSeg's ordinary interpolation path at 256x256, emitted
logits before argmax, and was converted with PaddlePaddle 2.6.2 plus
Paddle2ONNX 1.3.1 at opset 16. ONNX checker and the current Rust ORT accepted the
graph, but the result still collapsed:

```text
Paddle argmax labels: {0, 8, 14}
ONNX argmax labels:   {2}
argmax pixel agreement: 0.0
max absolute logit error: 18.2565
```

The current Rust `ort 2.0.0-rc.12/api-22` probe on an Intel i7-12700K,
single-threaded, five warmups and thirty measured runs reported:

```text
p50: 39.793 ms
p95: 55.624 ms
process VmHWM: 132300 KiB
model: 6179545 bytes
```

Thus parity, CPU p95 and provenance gates all fail. Accuracy against the 19.2.1
corpus is deliberately unreported: there are no parity-valid predictions to
score. The candidate is not added to `mvm setup`, the model manifest or runtime.

## Re-run repository gates

The model-free decision record is executable with stdlib Python:

```bash
python3 benchmark/visual-spatial/semantic-19.2.2/evaluate.py
python3 benchmark/visual-spatial/semantic-19.2.2/test_evaluate.py
```

The optional Rust probe accepts an explicit locally converted ONNX path:

```bash
cargo run --release -p mvm-scene --example semantic_onnx_probe -- /path/to/model.onnx
```

The exact artifact URLs, SHA-256 values, conversion versions and measurements
are pinned in `candidate.json`.

## Iteration B — SegFormer-B0

The reproducible contract and measurement decision live under
`iteration-b/`:

```bash
python3 benchmark/visual-spatial/semantic-19.2.2/iteration-b/evaluate.py
python3 benchmark/visual-spatial/semantic-19.2.2/iteration-b/test_evaluate.py
python3 benchmark/visual-spatial/semantic-19.2.2/iteration-b/evaluate_measurement.py
python3 benchmark/visual-spatial/semantic-19.2.2/iteration-b/test_measurement.py
python3 benchmark/visual-spatial/semantic-19.2.2/iteration-b/evaluate_quality.py
python3 benchmark/visual-spatial/semantic-19.2.2/iteration-b/test_quality.py
```

SegFormer-B0 revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da` exported to
valid static ONNX at 256×256 and 320×320. PyTorch↔ONNX argmax agreement was
1.0. Under the revised sparse-expert budget (invocation p95 ≤250 ms,
amortized semantic cost ≤10 ms per capture frame at cadence 30, aggregate with
depth ≤20 ms/frame), FP32 passes: p95 164.979/231.357 ms, amortized
5.499/7.714 ms/frame; depth+semantic aggregate is 13.166/15.711 ms/frame.
INT8 reduced the artifact to 4.20 MB but made p95 worse: 312.644/494.692 ms.
On a fixed 200-frame ADEChallenge validation subset, the best 150→5 reducer
reached mean surface IoU 0.537, floor IoU 0.642 and opening IoU 0.428. Unknown
recall was 0.909, but 9,946 unknown pixels became floor. A 1,000-step decoder
fine-tune and a 500-step classifier-only fine-tune both scored about 0.502 mean
IoU and did not improve the baseline. The candidate is therefore rejected; no
runtime integration is authorized.

The deterministic scheduler harness is executable with:

```bash
python3 benchmark/visual-spatial/semantic-19.2.2/iteration-b/controlled_ab.py
python3 -m unittest benchmark/visual-spatial/semantic-19.2.2/iteration-b/test_controlled_ab.py
```

It verifies the cadence-30, queue-1, non-blocking capture contract and stamped
stale-result withholding using the measured p95 envelope. This is a contract
harness only; it does not replace the required integrated runtime A/B wall-time
measurement. The three audited Hugging Face ScanNet mirrors are not semantic-GT
substitutes: one is mesh-only, one contains derived model outputs, and one has
RGB/depth/pose without paired semantic masks. Their exact revisions and roles
are recorded in `scannet_mirror_audit.json` and `datasets.json`.
