# Machine Vision Module

*Structured Decomposed Agentic Vision — гибридный Python+Rust проект: камера → **метрический 3D
scene-graph** каждый кадр (не VLM). Прод — **v0.4.0** (GitHub Releases). Активная разработка — ветка `kaggle`.*

## Ориентир: что это и по каким принципам

MVM — **ансамбль независимых микро-агентов зрения** (лицо, identity, эмоция, поза, объекты, глубина,
кисти), сшитых **арифметическим fusion** в метрический 3D scene-graph (узлы-сущности + рёбра-отношения).
Не одна end-to-end модель, не VLM: <50мс CPU / $0 / приватно / per-signal confidence / интерпретируемо.

**Три инварианта** (любая фича обязана им соответствовать):
1. **Минимализм / чистая арифметика** — fusion, трекинг, 3D, uncertainty = скалярная арифметика над
   данными, уже лежащими в памяти. **+0 зависимостей** на интеллектуальную фичу; footprint растёт только
   когда добавляется новый ONNX-эксперт.
2. **CPU-first, $0** — ни рантайм, ни модели не требуют GPU; никаких внешних API.
3. **Полный нативный Rust-пайплайн** — Python-модели = прототипы/леса; новый код не углубляет Python в hot-path.

**Честность (honesty-first):** каждый сигнал несёт числовую уверенность, а неопределённость выражается
**явно** (`withheld` / ⚠️), а не замалчивается — модуль скорее скажет «поза withheld, видна только верхняя
часть», чем угадает. `withheld`/`unknown` **никогда** не превращаются в отрицательный класс.

**Два потребителя** (фаза 16): (1) **LLM / MCP** — pull-модель `snapshot`/`recap`/`inspect`, read-only,
поверх durable text-only семантической сессии (default-on, фаза 16.7); (2) **Dataset Factory** —
`mvm run --record` доращивает ту же сессию до полного пиксельного датасета (обучающая память).

## Структура

> **После слияния `rust-core`→`master` (2026-07-02):** корень = **Rust-продакшн** + рабочий гибрид-bridge;
> автономный **Python-прототип** вынесен в `prototype/`. Историю Python-линии см. на ветках
> `python-prototype`/`3d-workspace` и теге `backup/master-pre-merge`.

```
Machine Vision Module/
├── mvm/            # ★ Rust workspace — ПРОДАКШН (14 крейтов; фазы 5–18 закрыты)
│   ├── crates/     #   core, detect, identity, emotion, posture, scene, depth, hands, fusion, mesh, pipeline, dataset, cli
│   ├── docs/       #   phase-* prompts/runbooks
│   ├── CHANGELOG.md#   версии бинаря (0.1.0 → 0.3.0)
│   └── CLAUDE.md   #   ← детальный контекст Rust-части (source of truth)
├── install.sh      # curl|sh установщик релизного бинаря (GitHub Releases, x86_64/aarch64)
├── core/           # Python гибрид-bridge: face_id, workspace_3d, … (нужен пути `--native`)
├── tools/          # Python bridge-инструменты: mvm_bridge.py (JSON-мост), calib_3d.py
├── models/         # Веса: ONNX/MediaPipe (.task), references/ (крупные gitignored, тянутся `mvm setup`)
├── prototype/      # ⚙ Python-ПРОТОТИП (архив) + pipeline/dataset/ — Python-экспортёры + validate.py (фаза 16)
├── kaggle-portfolio/ # ML-портфолио (журналы LJ0–LJ6 фазы 15, kctl-loop) — отдельный трек
├── spike_pose/     # Питон-спайки тренировки позы (GRIP)
└── src/            # sightd (legacy Rust daemon, заменён на mvm/)
```

## Как работает — два пути захвата

- **`run`** (основной, фаза 7+): чистый Rust v4l2 MJPG → body-primary цепочка
  (BlazePose-тело + YuNet-лицо + SFace + EmotiEffLib + YOLO26n-объекты + Depth-Anything + handpose)
  → метрический 3D scene-graph. **Ноль Python.** Корневой `--native-camera` — deprecated alias,
  который маршрутизируется в тот же AgentMesh.
- **`--native`** (bridge, для тестов/дебага): Python `tools/mvm_bridge.py` захватывает камеру и выдаёт
  JSON lines, Rust обогащает нативными ONNX-моделями. venv остаётся только для этого пути.

```bash
cd mvm && target/release/mvm run --dump-json                 # основной: живой mesh, чистый Rust
cd mvm && target/release/mvm run --record                    # + Dataset Factory: копить сессии на диск
cd mvm && target/release/mvm dataset check                   # data-quality гейт накопленного (CI exit-code)
cd mvm && target/release/mvm run --max-frames 1 --dump-json  # bounded canonical AgentMesh run
cd mvm && target/release/mvm --native --once --dump-json     # bridge-путь (Python мост)
```

**Камера:** `run` (и deprecated alias `--native-camera`) использует прямой v4l2 MJPG mmap; bridge использует `cv2.CAP_FFMPEG`
(не V4L2) — обход PipeWire на Ubuntu 24+.

## Ветки и релизы

`master` — опубликованный прод (**v0.4.0**, после мёржа `kaggle`→`master` 2026-08-06). `kaggle` — активная
разработка (следующая фаза требует отдельного approval). Релиз триггерится пушем тега `v*` → CI `release.yml` собирает x86_64+aarch64 бинари.
История: **v0.1.0** (Продукт, фаза 14) · **v0.3.0** (Dataset Factory, фаза 16) · **v0.4.0** (Visual Cortex, фаза 18); **v0.2.0** (ML Refactor,
фаза 15) свёрнута в код v0.3.0, отдельным тегом не выпускалась.

## Текущая фаза

**Фазы 5–14 ✅** (детекторы, 3D scene-graph, трекер, fusion-полировка, LLM-render/MCP, продукт-оболочка
v0.1.0). **Фаза 15 (ML Refactor) ✅** — lab-first пересмотр всей ML-логики (журналы LJ0–LJ6): 2D-GRIP-поза,
E2 scene-level fall, stale-relink трекер, словарь v2 `using_object(class)`, ⭐ SFace align +19.5 п.п.;
footprint позы −79% (в коде v0.3.0).

**Фаза 16 — Dataset Factory (v0.3.0, `kaggle`):** MVM из «источника для LLM» → двухпотребительская
архитектура. Новый крейт `mvm-dataset` — чистый потребитель WorldState. **16.0–16.6-A ✅ и
live-провалидированы:** Semantic Contract (Canonical Observation + статусы + Router G₁–G₅) → инфра
(`mvm run --record`, session-grouper, privacy-blur) → сырьё (RGB+depth-16bit+masks-RLE) + quality-as-data →
CLI `mvm dataset` → Python-экспортёры → точный YuNet-face-bbox privacy → **16.6-A data-quality гейты**
(`mvm dataset check` + `validate.py`, CI exit-code). **Осталось:** 16.6-B (hard-example mining, зависит от
накопленных данных). Детали — `mvm/CLAUDE.md`, план — `mvm/docs/phase-16-plan.md`.

**Фаза 16.7 — Semantic Session Log ✅ (live-провалидирована, `kaggle`):** свела ДВА параллельных стора
истории сцены в ОДИН. Раньше vision-история жила в плоских `scene-*.jsonl`/`events-*.jsonl`, а датасет —
в session-организованных `labels.jsonl`. Теперь **`mvm run` по умолчанию пишет text-only семантическую
сессию** (`labels.jsonl`, без пикселей — durable vision-лог), обогащённую relations/activities/events
(SCHEMA 16.7.0); плоские логи ретайрнуты (legacy `--scene-log`); `--record` — полный пиксельный датасет;
`recap`/`inspect`/journal читают сессии. План — `mvm/docs/phase-16.7-plan.md`.

**Фаза 17 — Canonical MVM Agent Skills ✅:** repo-canonical know-how разделён на два
переносимых skills `mvm/skills/{mvm-vision,mvm-dataset}`; прежний монолитный agent how-to удалён.
Repo-local validator + CI contract gate защищают frontmatter, honesty markers и public-surface drift.
Официальный `skills-ref` и fresh-agent forward-tests подтвердили portability; профиль Hermes `sight`
читает skills напрямую из repo, а legacy Python/wiki flow архивирован. Ядро/версия не затронуты. План —
`mvm/docs/phase-17-plan.md`.

**Фаза 18 — Visual Cortex Layer ✅ (закрыта):** 18.0 ✅ — canonical facts contract
(`FusedSceneFrame`, `SceneFact<T>`, `EntityKey`+incarnation, stamped sources,
non-destructive event cursor/`EventGap`) и Visual QA benchmark baseline. 18.1 ✅: generic
`LatestValue<T>` + stamped Human/Scene/Depth observations, `CanonicalFusion` публикует immutable
`FusedFramePacket`; StateFile/DatasetWriter читают его без scene mutation, session lifecycle локален,
а `current_state.md` пишется atomic. Canonical activity episodes пишутся в `labels.jsonl` через
независимый DatasetWriter cursor с event/source provenance; replay/cursor/late-result gates добавлены.
Корневой `--native-camera` маршрутизируется в canonical AgentMesh; text-only/pixel dataset fixtures
проходят check gate. **18.2–18.8b ✅:** единый `mvm-cortex` теперь включает serializable
honesty-first `CortexFrame`/`CognitiveClaim`, canonical Scene Events + evidence-linked interpretation,
bounded advisory re-observation, runtime-only Event Memory и evidence-linked constant-velocity /
persistence forecast с typed prediction errors, camera-relative `VisualSelfModel` и детерминированный
`UncertaintyReport`, а также evidence-linked Affordance V1. Canonical `PersonFacts` сохраняет
body-coverage, visual-only head orientation и bounded hand summary; Fusion публикует только
hand-to-object `touching_candidate`, не подменяя его physical hold. Evidence-backed `AttentionMap`
детерминированно ранжирует top-3 focus из novelty, prediction error, uncertainty, event recency,
activity, anomaly, identity и affordance; без canonical zone-violation fact этот компонент остаётся
нулевым. World camera pose, полная видимость
зон, emotion entropy и metric calibration явно остаются unknown/withheld, когда canonical evidence
недостаточно. `CortexAgent` включён в AgentMesh,
публикует отдельный cognitive frame и не мутирует canonical facts; scheduler исполняет только
ограниченные Scene/Depth one-shot permits с TTL, semantic dedupe, cooldown, budget/fairness и
explicit outcomes; внешний request без executor становится `Unavailable`, а `labels.jsonl` остаётся
единственным durable semantic store. 18.8b.1 closed controlled A/B/C runtime gate: worst wall-time
delta +0.31% ≤5%, no agent errors. 18.9 adds atomic `current_cortex.json`, read-only `mvm cortex`
views and MCP `mvm_cortex`; 18.10 frozen replay passes 6/6 pinned claims with zero honesty
regressions and three `Accepted → Dispatched → Resolved` advisory loops. Final live validation on the
closing binary confirmed 25 FPS, 1s/2s forecasts, evidence-linked Cortex output, preserved
withheld/unknown facts and semantic-session recap. Phase 19.0 provides the additive spatial
contract/replay baseline; Phase 19.1 adds dependency-free stamped camera-local motion evidence with
explicit fixed-camera authority and fail-closed quality gates. Phase 19.2.1 fixes the semantic surface
taxonomy and executable synthetic honesty corpus without claiming a runtime expert. Phase 19.2.2
rejects/defers PP-MobileSeg-Tiny after failed parity, CPU and pretrained-provenance gates; no weights
or runtime path were added, and 19.2.3 requires a new passing candidate. Plan —
`mvm/docs/phase-19-plan.md`.

## Для Agents

При работе с Rust-кодом — читай `mvm/CLAUDE.md` (полный контекст: архитектура, крейты, фазы, TDD-паттерны).
Для vision pull-модели/MCP загружай `mvm/skills/mvm-vision/SKILL.md`; для записи, проверки и экспорта
сессий — `mvm/skills/mvm-dataset/SKILL.md`. Python-модули: `core/face_id.py`, `tools/mvm_bridge.py`,
`core/workspace_3d.py`.
