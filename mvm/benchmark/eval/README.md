# Evaluator contract

The executable entry point is `python3 benchmark/eval/evaluate.py`. It is
stdlib-only and consumes the frozen scenario fixtures under `benchmark/`; it
never reads `current_state.md`, `current_cortex.json`, or mutable live state.
Use `--output` to write a machine-readable report:

```bash
cd mvm
python3 benchmark/eval/evaluate.py --output benchmark/eval/18.10-results.json
```

`skeleton` scenarios are excluded from denominators. Synthetic replay scores
are labelled as such; Rust contract tests are referenced as evidence rather
than silently being presented as camera measurements. The six core claims are
explicitly pinned by the evaluator: temporal QA, identity continuity, object
permanence, uncertainty honesty, forecast IoU, and attention recall@3.
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
