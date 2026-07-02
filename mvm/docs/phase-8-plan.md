# Фаза 8 — детальный план (Agent Mesh)

> **Версия:** 3.0 (29 июня 2026) — переписано от актуального состояния кода.
> **Состояние:** ветка `rust-core`, **12 крейтов, 212 тестов зелёных**, release чист.
> **Решения 8.0.1 (зафиксированы спайками `spikes/mesh-spike/`):** `std::thread` **не** tokio;
> latest-frame single-slot канал (не broadcast); cross-group fusion = `build_scene` **не** MBT;
> cadence — главный рычаг throughput; per-entity freshness в `WorldState`.
> **Дублёр:** `~/wiki/machine-vision/IMPLEMENTATION_PLAN.md` (та же версия 3.0).

Канон текущего состояния — `mvm/CLAUDE.md`. Этот документ — только план **оставшейся** работы фазы 8.

---

## Что уже сделано ✅

| Блок | Коммит | Статус |
|------|--------|--------|
| 8.0.1 спайки (threads-not-tokio, latest-frame, cadence) | `52907ca4` | ✅ |
| 8.0.2 body-primary иерархия (FRONTIER) + reserved-id identity | `dfdca72c` | ✅ live |
| 8.0.3 / 8.0.4 стабилизация трека (A1 анти-flip / A3 EMA / R / A2-lite) | `b50d7288`,`4380c1e9` | ✅ live |
| 8.E `mvm-scene` — YOLO26n object-узлы + cadence + фильтры A/C/D + `ObjectTracker` | `0cae2f17`,`5eebed53`,`413436cd` | ✅ live |
| 8.E.2 `mvm-depth` — Depth-Anything-V2 + depth-anchoring → метрич. 3D объектов | `21363ab3`,`b832f7bf` | ✅ live |
| 8.E.4 `mvm-hands` — handpose → **G₁ Human полна** | `a48f517c` | ✅ live |
| `--scene-debug` — все детекторы цветом своего агента (задел под mesh) | `faa6b990`,`4cfaadf1` | ✅ |
| 8.A/8.B `mvm-core::world` — WorldState blackboard + проекции + per-entity freshness | `4efedf08` | ✅ |
| 8.C **скелет** `mvm-mesh` — `Agent`/`Frame`/`LatestFrame`/`AgentMesh` + stub-агенты | `64f0f295` | ✅ (8 тестов) |

**G₁ Human, G₂ Scene, G₄ depth — production. Mesh — скелет стоит, обёрток реальных детекторов нет.**

---

## Сквозная задача фазы 8: «реальные агенты в mesh»

**Цель:** `mvm-cli mesh` запускает всю цепочку на потоках; тяжёлые детекторы (depth ~230мс, hands)
не блокируют hot-path → FPS восстановлен (сейчас синхронно 20→13); `WorldState` заполняется
агентами; scene-graph строится из `WorldState`. Синхронный `--native-camera` остаётся рабочим,
затем **поглощается** mesh (тонкая обёртка с одним HumanAgent).

**Главный измеримый выигрыш:** снятие head-of-line blocking — depth/YOLO в своих потоках на своей
cadence не тормозят человеко-цепочку.

---

## Этап 8.C-R — Рефактор `BodyPrimaryPipeline` (предпосылка, ⚠️ риск-шаг)

**Проблема:** `build_body_primary` — свободная функция в `mvm-cli/src/main.rs` с ~11 параметрами и
заёмными моделями. Чтобы стать агентом-потоком, цепочка должна **владеть своими моделями** и иметь
метод `process(frame)`.

**Решение размещения агентов:** обёртки живут в **`mvm-cli`** (он уже зависит от всех детектор-крейтов
и будет зависеть от `mvm-mesh` ради трейта `Agent`). `mvm-mesh` НЕ зависит от детекторов → нет циклов.
Новый крейт не нужен. Подкоманда `mesh` (8.D) собирает `AgentMesh` из этих cli-агентов.

| Задача | Файл | Тесты |
|--------|------|-------|
| `struct BodyPrimaryPipeline` — владеет `detector, identity, emotion, native_pose, hand_landmarker, scene_detector, object_tracker, depth_estimator, config` | `mvm-cli/src/pipeline.rs` (новый) | — |
| `fn process(&mut self, frame, frame_id) -> FrameResult { detections, objects, hands, faces, poses }` — перенос тела `build_body_primary` + cadence объектов/глубины + `anchor_objects_depth` | там же | ≥3 (детерминир. части) |
| Переписать синхронный `--native-camera` цикл поверх `BodyPrimaryPipeline::process` (логика та же, реорганизована) | `mvm-cli/src/main.rs` | существующие зелёные |

**Риск:** трогает живой hot-path. **Митигация:** чисто механический вынос; `cargo test` + живой
`--native-camera --scene-debug` прогон **до/после** на паритет результата; всё под git (обратимо).
**Acceptance:** `--native-camera` идентичен по поведению; весь workspace зелёный.

---

## Этап 8.C.5 — Агенты-обёртки (`Capture` / `Human` / `Scene` / `Depth`)

После 8.C-R обёртки тривиальны — каждый `impl mvm_mesh::Agent` в `mvm-cli/src/agents/`.

| Агент | Что делает | Cadence | Тесты |
|-------|-----------|---------|-------|
| `CaptureAgent` | обёртка `mvm_pipeline::NativeCamera` → `ctx.frames.publish` | темп камеры | ≥2 (стаб) |
| `HumanAgent` | `BodyPrimaryPipeline::process` + собственный `PersonTracker` → `WorldState.people` (`PersonState::from_track`) | каждый кадр | ≥2 |
| `SceneAgent` | YOLO26n + `ObjectTracker` → `WorldState.objects` (`ObjectState`) | свой (~/3) | ≥2 |
| `DepthAgent` | Depth-Anything → публикует `DepthMap` в общий слот (анкор делает fusion) | свой (~/30) | ≥2 |

> **Решение по depth-anchoring:** анкор требует и людей, и объектов → выполняется на **fusion-шаге**
> (там видны обе группы из `WorldState`), а не отдельным агентом. `DepthAgent` лишь публикует сырую
> карту в общий `Arc<Mutex<Option<DepthMap>>>`.
> **Решение по трекерам:** `PersonTracker` живёт в `HumanAgent`, `ObjectTracker` — в `SceneAgent`
> (каждый владеет своей временной сущностью).

**Acceptance:** агенты юнит-тестируемы по отдельности; `AgentHealth.ticks` растут; `WorldState`
заполняется соответствующими слайсами.

---

## Этап 8.C.7 — Fusion-шаг + `StateFileWriter` в mesh

| Задача | Файл | Тесты |
|--------|------|-------|
| `FusionAgent`: читает `WorldState` (только свежие по `last_seen`) → строит `Vec<Entity>` (люди+объекты) → `mvm_fusion::build_scene` → пишет `zones`/`relations`/`activities`/`anomaly_score` | `mvm-cli/src/agents/fusion.rs` | ≥3 |
| Depth-anchoring перенести сюда (видны и люди, и объекты + общая `DepthMap`) | там же | ≥1 |
| `StateFileWriter`: периодически (~100мс) сериализует `WorldState` → StateFile + Scene JSON | `mvm-cli/src/agents/statefile.rs` | ≥2 |

**Решение 8.0.1:** fusion = `build_scene` (object-agnostic), `near`-рёбра только при `pos_conf>0`.
**Acceptance:** `WorldState` → тот же Scene JSON контракт (`faces[]`+`objects[]`+zones+relations), что и
синхронный путь.

---

## Этап 8.D — Подкоманда `mvm-cli mesh` (главная веха)

| Задача | Файл | Тесты |
|--------|------|-------|
| Парсер подкоманды `mesh` (отдельный codepath; `--native-camera`/`--native` не трогаем) | `mvm-cli/src/main.rs` | ≥2 |
| Сборка `AgentMesh`: Capture+Human+Scene+Depth+Fusion+StateFileWriter; `--stub` (StubCapture) для CI | там же | `mesh_cli_stub`, `mesh_cli_help` |
| Завершение: Ctrl-C / `--max-frames` → `stop_flag` | там же | — |

**Acceptance:**
```bash
cargo run -- mesh --stub --max-frames 100   # не падает, StateFile валиден
cargo run -- --native-camera --once          # старый путь жив
cargo run -- mesh                             # живой: FPS НЕ проседает на depth (HOL снят)
```

---

## Этап 8.E.3 — Audio (YAMNet), группа G₄ (опционально, после ядра)

| Задача | Файл | Тесты |
|--------|------|-------|
| Крейт `mvm-audio` (13-й): YAMNet ONNX → классы звуков | `crates/mvm-audio` | ≥3 |
| `AudioAgent` → `WorldState.audio_events` (независимый поток, своя частота) | `mvm-cli/src/agents/audio.rs` | ≥2 |

> ⚠️ **Открытые решения:** источник аудио (`cpal`/`alsa` — **новая зависимость**, против инварианта
> минимализма — обсудить) + модель YAMNet ONNX (скачать). Кандидат вынести за пределы «ядра» фазы 8.

---

## Этап 8.F — Cross-group fusion (формализация, чистая арифметика)

Object-часть уже работает (объекты → `build_scene`). Осталось дёшево:
- `Activity`-инференс: «печатает» = рука ∩ клавиатура (overlap); «двое рядом» = person↔person near;
  → `WorldState.activities`. Чистые функции.
- `anomaly_score`: простая эвристика (нежданный объект/звук). Тесты на чистых функциях.

---

## Граф зависимостей и порядок

```
8.C-R (BodyPrimaryPipeline)  ← риск-шаг, делать осознанно
   └─→ 8.C.5 (Capture/Human/Scene/Depth агенты)
          └─→ 8.C.7 (Fusion-шаг + StateFileWriter)
                 └─→ 8.D (mesh subcommand)  ★ ГЛАВНАЯ ВЕХА: FPS восстановлен
                        ├─→ 8.F activities/anomaly (чистая арифметика, дёшево)
                        └─→ 8.E.3 audio (опц., новая dep + модель)
```

**Рекомендуемая последовательность:** 8.C-R → 8.C.5 → 8.C.7 → 8.D (это и есть «полноценный mesh»;
здесь меряем FPS-выигрыш) → 8.F → (опц.) 8.E.3.

---

## Сквозные инварианты / решения (не нарушать)

- `std::thread` + `LatestFrame` (single-slot) + `Arc<RwLock<WorldState>>` + `AtomicBool` — **ноль tokio**.
- Fusion = `build_scene` (object-agnostic); `near`-рёбра только при `pos_conf>0` (depth-gated).
- Per-entity freshness (`last_seen`) — fusion не строит рёбра по протухшим узлам.
- Mesh в итоге **поглощает** `--native-camera`, не остаётся вечным третьим codepath.
- TDD: тест → код; `cargo test --workspace` зелёный после каждой задачи; коммит по задаче.
- Большие ONNX — gitignored (скачивать отдельно); venv остаётся для тестов/дебага.

## Критерии приёмки фазы 8

- [ ] `cargo test --workspace` — ≥ ~230 тестов зелёные
- [ ] `mvm-cli mesh --stub --max-frames 100` — не падает, StateFile валиден
- [ ] `mvm-cli mesh` (живой) — FPS **не проседает** при depth (HOL-блокинг снят — главный выигрыш)
- [ ] `mvm-cli --native-camera --once` — не сломан
- [ ] `WorldState` заполняется агентами; Scene JSON контракт сохранён
- [ ] `cargo clippy --workspace` без предупреждений (если доступен)
- [ ] (опц.) audio events; activities/anomaly

## Отложенные хвосты (вне ядра фазы 8)

8.0.4 ре-валидация на 3 людях · калибровка `torso_length_m` рулеткой · hands ROI rotation + cadence ·
per-track привязка рук к людям · `--emotion-debug` link · авто-калибровка `fx` (это уже Фаза 9 — продукт).
