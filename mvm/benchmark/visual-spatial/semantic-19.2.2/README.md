# Phase 19.2.2 Semantic Model Spike

This is a measured **reject/defer** record for PP-MobileSeg-Tiny ADE20K. It is
not a runtime expert, and no downloaded or converted model bytes are committed.

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
