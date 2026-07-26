# Evaluator contract

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
