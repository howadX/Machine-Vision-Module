# Machine Vision Module

*Structured Decomposed Agentic Vision — гибридный Python+Rust проект: камера → **метрический 3D
scene-graph** каждый кадр (не VLM). Прод — **v0.3.0** (GitHub Releases). Активная разработка — ветка `kaggle`.*

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
├── mvm/            # ★ Rust workspace — ПРОДАКШН (13 крейтов, 689 тестов; фазы 5–16.7 закрыты, 16.6-B осталось)
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

- **`run` / `--native-camera`** (основной, фаза 7+): чистый Rust v4l2 MJPG → body-primary цепочка
  (BlazePose-тело + YuNet-лицо + SFace + EmotiEffLib + YOLO26n-объекты + Depth-Anything + handpose)
  → метрический 3D scene-graph. **Ноль Python.**
- **`--native`** (bridge, для тестов/дебага): Python `tools/mvm_bridge.py` захватывает камеру и выдаёт
  JSON lines, Rust обогащает нативными ONNX-моделями. venv остаётся только для этого пути.

```bash
cd mvm && target/release/mvm run --dump-json                 # основной: живой mesh, чистый Rust
cd mvm && target/release/mvm run --record                    # + Dataset Factory: копить сессии на диск
cd mvm && target/release/mvm dataset check                   # data-quality гейт накопленного (CI exit-code)
cd mvm && target/release/mvm --native-camera --scene-debug   # sync-путь + визуальный дебаг → /tmp/mvm_scene.jpg
cd mvm && target/release/mvm --native --once --dump-json     # bridge-путь (Python мост)
```

**Камера:** `run`/`--native-camera` использует прямой v4l2 MJPG mmap; bridge использует `cv2.CAP_FFMPEG`
(не V4L2) — обход PipeWire на Ubuntu 24+.

## Ветки и релизы

`master` — опубликованный прод (**v0.3.0**, после мёржа `kaggle`→`master` 2026-07-23). `kaggle` — активная
разработка (фаза 16). Релиз триггерится пушем тега `v*` → CI `release.yml` собирает x86_64+aarch64 бинари.
История: **v0.1.0** (Продукт, фаза 14) · **v0.3.0** (Dataset Factory, фаза 16); **v0.2.0** (ML Refactor,
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

**Фаза 17 — Canonical MVM Agent Skills (17.0 ✅):** repo-canonical know-how разделён на два переносимых
skills `mvm/skills/{mvm-vision,mvm-dataset}`; прежний монолитный agent how-to удалён. Впереди 17.1
contract gate, 17.2 portability и 17.3 Hermes integration. Только know-how-слой — ядро/версия не
трогаются; Hermes — по явному согласию. План — `mvm/docs/phase-17-plan.md`.

## Для Agents

При работе с Rust-кодом — читай `mvm/CLAUDE.md` (полный контекст: архитектура, крейты, фазы, TDD-паттерны).
Для vision pull-модели/MCP загружай `mvm/skills/mvm-vision/SKILL.md`; для записи, проверки и экспорта
сессий — `mvm/skills/mvm-dataset/SKILL.md`. Python-модули: `core/face_id.py`, `tools/mvm_bridge.py`,
`core/workspace_3d.py`.
