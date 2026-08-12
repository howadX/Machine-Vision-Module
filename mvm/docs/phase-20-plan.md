# Фаза 20 — External Expert Runtime: подключаемые Python-модели при Rust authority

> **Версия:** 0.1 (7 августа 2026)
> **Статус:** proposed; реализация не начата
> **Ветка для будущей работы:** `kaggle`
> **Предусловие:** Phase 19 закрыта benchmark + live evidence либо пользователь явно утвердил изменение порядка фаз.
> **Цель фазы:** дать MVM возможность подключать локальные Python-модели как bounded perception experts, сохраняя Rust владельцем capture, scheduling, canonical facts, Fusion, Cortex, memory, Dataset Factory и MCP.

---

## 0. Короткая формулировка

MVM не переписывается на Python и не превращает Python в новый runtime authority. Фаза добавляет контролируемую границу, через которую Rust runtime может вызывать долгоживущий локальный Python worker и получать от него typed, stamped visual observations.

```text
camera / frame source
  → Rust capture + latest-frame/backpressure + deadline
  → expert backend
      ├─ Native ONNX expert (in-process, текущий preferred путь)
      └─ Python worker expert (optional external process)
  → validated typed observation
  → Rust tracking + Canonical Fusion
  → immutable FusedSceneFrame
  → Cortex / Dataset / StateFile / MCP
```

Python — вычислительный исполнитель отдельной модели; Rust остаётся мозгом и единственным владельцем facts authority.

---

## 1. Зачем

Текущий native путь MVM использует Rust + ONNX Runtime и специально не имеет Python в production hot path. Это остаётся оптимальным default для стабильных CPU-first экспертов. Но часть перспективных моделей разумнее сначала подключать в их родной среде:

- собственные PyTorch/CV research-модели;
- результаты Kaggle/ML-lab, ещё не готовые к ONNX export;
- модели с необходимым Python preprocessing/postprocessing;
- optional accelerator profile, если он не становится обязательным условием работы MVM.

Фаза должна позволить сравнить такой эксперт с native baseline без дублирования или переписывания Fusion, tracker, Cortex и semantic history.

---

## 2. Границы и непереговорные инварианты

1. **Rust facts authority.** Только существующая Canonical Fusion публикует immutable `FusedSceneFrame`. Python worker никогда не пишет `WorldState`, `FusedSceneFrame`, `labels.jsonl`, StateFile или Cortex state напрямую.
2. **Python возвращает measurement, не решение.** Worker публикует только typed observation: payload конкретного expert kind, confidence/quality, source stamp, model provenance и явный outcome. Entity binding, tracking, Fusion и meaning остаются в Rust.
3. **No blocking.** Медленный, недоступный или аварийно завершившийся worker не блокирует Capture, Human/body-primary, Fusion, StateFile или Cortex. Его результат становится `unavailable`/`failed`/`withheld` для конкретного expert path.
4. **Latest-wins and deadlines.** Rust выдаёт bounded job для конкретного source frame/ROI, TTL и budget. Просроченный результат не меняет прошлый frame и может быть принят только по явно заданной next-tick policy.
5. **No raw pixels in JSON/pickle.** Control plane передаёт только metadata и compact typed results. Pixel payload передаётся через bounded local shared-memory/ring-buffer transport или другой отдельно доказанный zero-copy/low-copy механизм.
6. **Local, private, opt-in.** V1 — только локальный worker под владельцем процесса MVM; без сети, cloud API или передачи пикселей вовне. Python runtime и model environment не обязательны для базового `mvm run`.
7. **Native path stays default.** Existing Rust ONNX experts не заменяются массово. Python adapter начинается с одного отдельно утверждённого кандидата и имеет native fallback или честный withheld path.
8. **CPU-first remains the baseline.** GPU/PyTorch/CUDA допустимы только как optional named profile и не являются предпосылкой для MVM, correctness или required tests.
9. **Honesty-first.** `failed`, `timeout`, `stale`, `invalid`, `unavailable`, `weak`, `withheld` и `unknown` не означают отрицательный класс. Worker output обязан нести model/version/config provenance и source freshness.
10. **No generic plugin platform in V1.** Не вводятся arbitrary code plugins, remote workers, distributed inference, model marketplace, automatic model downloads или dynamic untrusted loading.

### Явные non-goals

- переписать `AgentMesh` на Python, Tokio или общий executor;
- заменить `mvm run` Python bridge-путём;
- сделать Python authority для tracker/Fusion/Cortex;
- создать второй persistent store или второй scene graph;
- поддержать все типы моделей/фреймворков в первой реализации;
- принять GPU-only expert как обязательный production dependency;
- обещать zero-copy до измерения конкретного transport implementation.

---

## 3. Целевая слоистость и контракты

### 3.1. Разделение control/data plane

```text
Rust runtime
  ├─ control plane: spawn, hello, manifest, health, request, cancel, shutdown
  ├─ data plane: bounded frame/ROI slots with source-frame metadata
  ├─ admission: deadline, cadence, token budget, latest-wins replacement
  ├─ validation: schema, frame/run identity, payload bounds, provenance, freshness
  └─ canonicalization: existing typed bus → Fusion

Python worker
  ├─ loads one declared model once and performs warm-up
  ├─ reads only assigned input slots
  ├─ executes model-specific preprocessing/inference/postprocessing
  ├─ returns bounded typed result envelope
  └─ reports lifecycle/health; does not mutate MVM durable state
```

V1 control plane uses a local Unix-domain socket with explicit versioned messages. Pixel transport is not specified by convenience: its implementation is selected only after a benchmark spike proves bounded memory, cleanup and latency behavior. JSON is allowed only for small control/result envelopes if schema and byte limits are explicit; it is forbidden for RGB/depth/mask tensors.

### 3.2. External expert contract

Exact Rust names are decided only after reading current code at implementation time. The semantic contract must contain:

```text
ExternalExpertManifest
  protocol_version
  expert_id / expert_kind
  input_kind: frame | ROI | declared tensor view
  input geometry / pixel format / max bytes
  output schema version / max result bytes
  model name + version + immutable artifact digest
  declared device policy: cpu | optional accelerator
  supported status values

InferenceJob
  run_id / job_id / source_frame_id
  capture and dispatch timestamps
  source ROI / input slot handle
  deadline / cancellation token / requested reason

InferenceResult
  run_id / job_id / source_frame_id
  model provenance + produced timestamp
  outcome: observed | weak | withheld | unavailable | failed | timeout | invalid
  typed payload or no payload
  confidence / quality / diagnostic code
```

Acceptance conditions in Rust:

- matching protocol/schema/run/job/source-frame identity;
- output size, counts, coordinates and finite-number bounds;
- known model manifest/digest and expected expert kind;
- result arrives before its policy deadline;
- source age is admissible for the current canonical tick;
- result passes model-specific semantic validation.

A rejected result is observable in metrics and diagnostics but is never converted into a fact.

### 3.3. Authority map

| Concern | Owner |
|---|---|
| camera, frame IDs, timestamps, frame retention | Rust Capture / runtime |
| cadence, permits, deadlines, backpressure, cancellation | Rust scheduler |
| model execution and model-local preprocessing | Python worker |
| validation, source stamps, entity association | Rust expert adapter / tracker |
| facts, relations, events, history | Rust Canonical Fusion / Dataset |
| interpretation and re-observation policy | Rust Cortex |
| worker lifecycle, restart budget, health reporting | Rust supervisor |

---

## 4. Delivery slices

Каждый slice требует отдельного approval перед кодом, TDD, replay fixture и performance evidence. Успешный предыдущий slice не означает, что следующий автоматически разрешён.

### 20.0 — Boundary contract and replay baseline

**Goal:** зафиксировать минимальный typed contract external expert без запуска Python.

**Scope:**

- определить Rust-level `ExternalExpert`/adapter boundary поверх current `AgentMesh`, не меняя generic `mvm-mesh` cognitive semantics;
- типизированные envelopes job/result, status taxonomy, provenance and validation errors;
- deterministic fake worker/test harness без process/IPC;
- canonical bus integration только для одного existing observation type;
- replay fixtures: valid, stale, wrong frame, malformed payload, oversized result, unknown schema, timeout and late result.

**Required gates:**

- result for frame N cannot mutate `FusedSceneFrame` N after canonicalization;
- invalid/late/foreign result does not create negative class or fact;
- same ordered inputs produce deterministic output;
- absent external backend leaves existing native path behavior unchanged;
- no external runtime dependency or Python required.

**Decision gate:** approve a minimal contract only if it reuses current stamps, typed buses and Fusion authority rather than introducing a parallel observation universe.

### 20.1 — Local worker lifecycle and control-plane spike

**Goal:** prove that Rust can supervise a long-lived local Python worker safely.

**Scope:**

- Rust-owned spawn/readiness handshake/warm-up/shutdown for an explicitly configured local executable;
- versioned Unix-socket control messages and bounded response sizes;
- health states: starting, ready, busy, degraded, failed, stopped;
- restart budget with cooldown; no restart loop;
- test Python fixture worker only, with no real model and no production CLI surface yet.

**Required gates:**

- failed start, bad handshake, protocol mismatch and worker crash fail closed without stopping native mesh;
- graceful runtime shutdown reaps child worker and socket artifacts;
- restart cap prevents resource exhaustion;
- no socket endpoint is network-exposed or outside MVM-private runtime directory;
- native `mvm run` without external configuration does not spawn Python.

**Metrics:** startup/warm-up p50/p95, restart time, leaked process/socket count (= 0), control-message latency and RSS overhead.

**Decision gate:** abandon process integration if lifecycle isolation/reaping cannot be made reliable with the standard-library-first dependency budget.

### 20.2 — Bounded frame/ROI transport spike

**Goal:** select a transport for frame or ROI input that avoids per-frame Python serialization/copy amplification.

**Candidates:** bounded shared-memory slots/ring buffer with Unix-socket descriptors; another local OS primitive only if it has a simpler audited lifecycle and measured equal-or-better behavior. The final choice must be explicitly documented.

**Scope:**

- fixed-capacity slot ownership and release semantics;
- source frame/ROI geometry and pixel-format metadata;
- latest-wins replacement under a slow consumer;
- cleanup on timeout, worker crash and normal shutdown;
- synthetic image fixture; no model inference yet.

**Required gates:**

- slow worker cannot make Capture wait or grow memory unboundedly;
- slot from wrong run/job/frame is rejected;
- no use-after-release/cross-job read in deterministic tests;
- RGB payload is never serialized as JSON/pickle;
- worker crash returns all slots and leaves no durable pixel artifact.

**Metrics:** end-to-end dispatch→read latency p50/p95, copies/allocation evidence, RSS peak, dropped/replaced jobs, capture throughput and Human/Fusion p95 compared with the no-worker baseline.

**Decision gate:** accept only a transport whose same-run capture throughput regression is ≤5%, whose bounded-resource behavior is demonstrated, and whose integration does not regress Human/body-primary p95 beyond a separately documented budget.

### 20.3 — First typed Python expert adapter

**Goal:** connect one narrow Python model as an evidence producer while retaining a native fallback.

**Candidate policy:** select one model after a separate short selection note. Favor a low-cadence heavy research model with measurable expected value; do not begin with capture, Fusion, identity authority or body-primary. Candidate selection must compare task quality, CPU/GPU profile, licensing, model size and failure behavior.

**Scope:**

- model-specific input/output adapter and strict result validation;
- manifest with immutable model digest, preprocessing version and declared device policy;
- one opt-in CLI/config profile; default remains native;
- fallback policy: current native expert if compatible, otherwise explicit withheld/unavailable;
- synthetic replay plus a captured, privacy-safe fixed corpus for comparison.

**Required gates:**

- Python result receives existing source/provenance stamps and reaches Fusion only through current bus/contracts;
- worker cannot assign `EntityKey`, alter tracker lifecycle or write durable datasets directly;
- Python/native output is labelled by backend/model version in diagnostics and dataset provenance where applicable;
- timeout/invalid output makes no false assertion;
- turning the profile off returns byte/contract-compatible native behavior.

**Metrics:** task-specific quality against held-out corpus, end-to-end result latency p50/p95, capture FPS, Human/Fusion p50/p95, dropped-frame ratio, RSS/VRAM if used, worker crash recovery and fallback rate.

**Decision gate:** a Python expert is retained only if it provides a declared quality/capability gain or equivalent quality with an explicitly acceptable operational trade-off. “It runs” is not sufficient evidence.

### 20.4 — Configuration, admission and observability hardening

**Goal:** make the proven single adapter operable without making external workers implicit or opaque.

**Scope:**

- explicit product config for enabled adapter, executable path, manifest path, device policy, queue/deadline/restart limits;
- `mvm` diagnostics/status view that exposes selected backend, readiness, model provenance, counters and last non-sensitive failure;
- config validation before spawn; no auto-download/install/venv creation;
- privacy contract: only local temporary transport, deterministic cleanup, no raw pixel logging by default;
- documentation for setup, model compatibility, fallback and removal.

**Required gates:**

- invalid config fails before camera capture with actionable error and leaves no child process;
- default config preserves no-Python production path;
- diagnostic surface does not expose raw frames, secrets or unbounded worker logs;
- config/model digest mismatch is visible and fails closed.

### 20.5 — Comparative release evidence and extension decision

**Goal:** decide whether External Expert Runtime becomes a maintained MVM extension point.

**Required evidence:**

- native ONNX baseline vs Python-worker candidate on the same captured corpus and hardware profile;
- normal load, constrained CPU, slow worker, worker crash/restart, malformed output, deadline expiry and shutdown tests;
- all workspace unit/contract/replay tests plus dedicated Python adapter validation;
- live camera smoke run is reported separately from replay/corpus results;
- source/metrics artifacts record repetitions, variance, hardware, model digests, configuration and no-regression verdict.

**Acceptance gates:**

```text
correctness: all protocol, replay, validation and fallback tests pass
honesty: invalid/stale/unavailable results create no false facts
isolation: Python failure does not stop Capture/Human/Fusion/Cortex
resources: bounded queues/slots/restarts; no leaked child/socket artifacts
performance: same-run capture throughput regression ≤5% versus native baseline
latency: Human/Fusion p95 stays within the declared per-profile budget
quality: selected Python model meets its predeclared task metric and trade-off rationale
privacy: no network egress, no implicit pixel persistence, no secret-bearing logs
```

The final decision may be **accept**, **accept only as research adapter**, or **reject/remove**. A generic multi-framework plugin ecosystem is not implied by acceptance.

---

## 5. Benchmark scenarios

| Scenario | Required result |
|---|---|
| Native-only default run | No Python process, no behavior regression. |
| Valid result before deadline | Rust validates and admits one stamped observation on a future/current permitted canonical path. |
| Result for wrong run/job/frame | Rejected; no fact, explicit counter. |
| Slow worker / queue saturation | Latest-wins replacement or expiry; Capture and Human continue. |
| Worker crash during inference | Bounded restart/fallback; no leaked shared slot/process. |
| Protocol/manifest mismatch | Fail closed before result admission. |
| Malformed/NaN/out-of-bounds payload | Rejected; no synthetic negative class. |
| Late result after frame canonicalized | Never rewrites past scene/history. |
| Python unavailable at startup | Native default/fallback works, or requested adapter reports actionable unavailable state. |
| Optional accelerator absent | CPU profile or explicit unavailable state; MVM core remains usable. |
| Privacy inspection | No pixel content in control messages/logs; temporary slots reclaimed. |
| Graceful shutdown | Worker terminates/reaps and runtime artifacts disappear. |

---

## 6. Dependency order

```text
Phase 19 complete (or explicit reprioritization)
  ↓
20.0 contracts + fake-worker replay baseline
  ↓
20.1 lifecycle/control-plane spike
  ↓
20.2 bounded frame/ROI transport spike
  ↓
20.3 one typed Python expert + native comparison
  ↓
20.4 explicit configuration/diagnostics/privacy hardening
  ↓
20.5 replay + corpus + live evidence → extension decision
```

The existing `--native` Python bridge is historical/debug compatibility evidence, not the Phase 20 architecture and not an approved transport. Phase 20 must not silently extend that path into production authority.

---

## 7. Success criterion

Phase 20 is successful only if MVM can be described honestly as follows:

> MVM remains a Rust-owned visual runtime: it captures frames, schedules experts, validates provenance and freshness, canonicalizes evidence into immutable facts, and owns Fusion, Cortex, history and MCP. A locally supervised Python worker may execute one explicitly configured perception model and return a bounded typed measurement, but it cannot block the primary pipeline, rewrite past facts, own entity identity, mutate durable state or make physical decisions. Native Rust/ONNX remains the default; Python is optional, observable, fail-closed and accepted only with replay, live and performance evidence.

---

## 8. Sources and implementation-time checks

Before implementation, re-read live code and contracts rather than treating this plan as an API specification:

- `mvm/CLAUDE.md` — current architecture, phase status and runtime invariants;
- `mvm/docs/phase-19-plan.md` — predecessor spatial contract and phase ordering;
- `mvm/crates/mvm-mesh/src/agent.rs` and `frame.rs` — current `Agent`, latest-frame/value semantics;
- `mvm/crates/mvm-cli/src/main.rs` and `agents/` — current live mesh wiring, typed buses and scheduler;
- `mvm/crates/mvm-core/src/canonical.rs` — current frame/source stamps and facts authority;
- `mvm/crates/mvm-fusion/` — canonical admission/assembly boundary;
- `mvm/skills/mvm-vision/SKILL.md` and `mvm/skills/mvm-dataset/SKILL.md` — current public and durable-data contracts.

*Plan created 7 August 2026 at user request. It fixes a proposed architectural direction only. It does not authorize implementation, a Python dependency, model installation, config changes, GPU requirements or replacement of any native expert without a separately approved slice and measured evidence.*
