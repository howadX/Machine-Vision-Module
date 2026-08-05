# Evaluator contract

The evaluator measures MVM's visual evidence and Cortex contracts only. It
does not claim robot control, safety authority, or physical-world correctness.

The executable entry point is `python3 benchmark/eval/evaluate.py`. It is
stdlib-only and consumes the frozen scenario fixtures under `benchmark/`; it
never reads `current_state.md`, `current_cortex.json`, or mutable live state.
Use `--output` to write a machine-readable report:

```bash
cd mvm
python3 benchmark/eval/evaluate.py --output benchmark/eval/18.10-results.json
```

`skeleton` scenarios are excluded from denominators. Synthetic replay scores
remain explicitly labelled and are not camera claims. The evaluator's six
phase-gate core claims are uncertainty honesty, forecast IoU, attention
recall@3, affordance accuracy, memory precision@5, and advisory
re-observation. Each numeric result is backed by its named Rust replay gate in
the full verification command; the evaluator never pretends to execute live
camera data.
The advisory loop gate requires at least three independent replay scenarios
with an uncertainty or prediction-error trigger; they are 019–022 in the
current fixture set.

The phase-18 evaluator consumes replay fixtures, never live mutable state.

Each evaluator result must include:

- code commit and fixture version;
- scenario IDs included and excluded;
- metric definition and denominator;
- baseline and candidate values;
- withheld/unknown counts;
- runtime hardware/config when timing is reported.

An evaluator must fail closed on malformed facts, future source stamps,
duplicate/non-monotonic event sequences, or an unreported event gap.

Initial metrics:

- canonical determinism;
- event delivery;
- source honesty;
- temporal QA accuracy;
- identity continuity;
- object permanence QA;
- uncertainty honesty;
- forecast IoU;
- attention recall@3;
- affordance accuracy;
- memory precision@5;
- self-model claim accuracy.

Forecast results contain separate `at_1s` and `at_2s` rows. Advisory results
contain only loops that complete `Accepted → Dispatched → Resolved` with a newer
stamped source result.
