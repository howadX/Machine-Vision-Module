# Phase 19.1 Motion Field Spike

The six deterministic fixtures exercise the production classical sparse block
matcher in `mvm-cli::agents::motion`. The committed baseline records a release
run after one warm-up sample and is guarded by `evaluate.py`.

The decision is intentionally narrow: use the dependency-free classical expert
for bounded camera-local image translation evidence. It does not claim world
pose, replace external odometry, or infer that a zero-flow camera is fixed.

```bash
cargo test --release -p mvm-cli \
  agents::motion::tests::spike_decision_gate_measures_epe_consensus_and_cpu -- --nocapture
python3 benchmark/visual-spatial/motion-19.1/evaluate.py
python3 benchmark/visual-spatial/motion-19.1/test_evaluate.py
```
