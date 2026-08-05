# Phase 18 — final live validation

Date: 2026-08-06 (Europe/Moscow)  
Code: `a202a2dd feat(cortex): close phase 18 benchmark gate`  
Producer: `mvm serve --mcp --with-camera`, camera `/dev/video0`, started through
`hermes-gateway-sight.service`

## Scope and method

This is a production-path smoke validation, not a live accuracy benchmark.  The
deterministic replay corpus remains the accuracy and advisory-loop evidence;
the controlled 18.8b.1 A/B/C run remains the throughput evidence.  This run
checks that the closing binary publishes the public facts and cognitive
projections through the live producer without relaxing the honesty boundary.

The producer found before the run pre-dated the closing commit, so the current
release binary was rebuilt and the `hermes-gateway-sight.service` producer was
restarted.  The Hermes gateway and the camera service were restored through
their systemd user unit; no parallel camera process was introduced.

Read-only pull checks used:

```text
mvm snapshot
mvm cortex --view forecast --horizon-seconds 1
mvm cortex --view forecast --horizon-seconds 2
mvm cortex --view full --json
mvm recap --last 1m
```

## Observations

| Contract | Live evidence | Verdict |
|---|---|---|
| Canonical scene producer | `snapshot` reported an occupied scene at 25 FPS and a persistent seating zone. | Pass |
| Observation honesty | The visible person had upper-only coverage and `posture withheld`; it was not converted into a posture class. | Pass |
| Cortex projection | `current_cortex.json` schema `18.9.0` advanced to scene frame 554; attention returned three evidence-linked entity targets. | Pass |
| Bounded forecast | The closing producer exposed two real projections: 1,000 ms and 2,000 ms, each containing three entity forecasts. | Pass |
| Uncertainty and self-model boundary | Missing identity/emotion stayed `unknown`; coasted source, low metric confidence, calibration, and zone visibility stayed `withheld` with reasons/evidence. | Pass |
| Advisory control boundary | Live policy output showed explicit `accepted`, `dispatched`, `deferred` (`dedupe_cooldown`/`expert_budget`), and `expired` outcomes. No request was silently presented as a confirmed fact. | Pass |
| Durable pull history | `recap --last 1m` read the new semantic session: 45 frames over 55 seconds; it retained unstable-episode and likely-phantom warnings. | Pass |
| Public CLI boundary | `recap --help` and `cortex --help` described the durable-history and read-only cognitive boundaries, including evidence-bounded forecast horizons. | Pass |

## Interpretation

The live run validates the updated release binary, camera-to-Fusion-to-Cortex
path, atomic report projection, both published forecast horizons, durable
semantic-session reading, and honesty-preserving CLI output.  It does **not**
claim live forecast IoU or infer that a single visible person proves benchmark
accuracy.  Those claims remain bounded by the frozen 18.10 replay evaluation:
6/6 core cognitive claims improve, zero honesty regressions, and the three
advisory resolution loops are verified there.  The separate same-hardware
18.8b.1 run remains the `+0.31% <= 5%` throughput result.

## Operational note

Sending `SIGTERM` directly to the old MVM child ended its stdio watchdog rather
than causing an in-place re-exec.  For a deliberate binary refresh, use the
owning `systemctl --user restart hermes-gateway-sight.service` unit and then
repeat the read-only pull checks above.
