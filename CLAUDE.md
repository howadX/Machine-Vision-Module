# Machine Vision Module

*Structured Decomposed Picture-to-Text — гибридный Python+Rust проект. Опубликованный прод v0.1.0
(GitHub Releases); v0.2.0 подготовлена по коду (ждёт owner-тега). Активная разработка — ветка `kaggle`.*

## Структура

> **После слияния `rust-core`→`master` (2026-07-02):** корень = **Rust-продакшн** + рабочий гибрид-bridge;
> автономный **Python-прототип** вынесен в `prototype/`. Историю Python-линии см. на ветках
> `python-prototype`/`3d-workspace` и теге `backup/master-pre-merge`.

```
Machine Vision Module/
├── mvm/            # ★ Rust workspace — ПРОДАКШН (13 крейтов, 657 тестов; фазы 5–15 закрыты, 16 в работе)
│   ├── crates/     #   core, detect, identity, emotion, posture, scene, depth, hands, fusion, mesh, pipeline, dataset, cli
│   ├── docs/       #   phase-* prompts/runbooks
│   └── CLAUDE.md   #   ← детальный контекст Rust-части (source of truth)
├── install.sh      # curl|sh установщик релизного бинаря (GitHub Releases, x86_64/aarch64)
├── core/           # Python гибрид-bridge: face_id, workspace_3d, … (нужен пути `--native`)
├── tools/          # Python bridge-инструменты: mvm_bridge.py (JSON-мост), calib_3d.py
├── models/         # Веса: ONNX/MediaPipe (.task), references/ (крупные gitignored, тянутся `mvm setup`)
├── src/            # sightd (legacy Rust daemon, заменён на mvm/)
├── spike_pose/     # Питон-спайки тренировки позы (GRIP)
└── prototype/      # ⚙ Python-ПРОТОТИП (архив) + pipeline/dataset/ — экспортёры Dataset Factory (фаза 16)
```

## Как работает

Два пути захвата:

- **`--native-camera`** (основной, фаза 7+): чистый Rust v4l2 MJPG → body-primary цепочка
  (BlazePose-тело + YuNet-лицо + SFace + EmotiEffLib + YOLO26n-объекты + Depth-Anything + handpose)
  → метрический 3D scene-graph. **Ноль Python.**
- **`--native`** (bridge, для тестов/дебага): Python `tools/mvm_bridge.py` захватывает камеру и выдаёт
  JSON lines, Rust обогащает нативными ONNX-моделями. venv остаётся для этого пути.

```bash
# Основной путь — живой mesh, чистый Rust, всё нативно (фаза 14: бинарь называется `mvm`,
# камера авто-детект, state+логи → ~/.local/share/mvm)
cd mvm && target/release/mvm run --dump-json

# Sync-путь (legacy, скрытый) + визуальный дебаг: детекторы цветом своего агента → /tmp/mvm_scene.jpg
cd mvm && target/release/mvm --native-camera --dump-interval 3 --scene-debug

# Bridge-путь (Python мост, для дебага)
cd mvm && target/release/mvm --native --once --dump-json
```

**Камера:** `run`/`--native-camera` использует прямой v4l2 MJPG mmap; bridge использует `cv2.CAP_FFMPEG`
(не V4L2) — обход PipeWire на Ubuntu 24+.

## Актуальная ветка

`kaggle` — активная разработка (фаза 15.6-порт + фаза 16). `master` — опубликованный прод v0.1.0
(после слияния `rust-core`→`master` 2026-07-02). Мёрж `kaggle`→`master` — вместе с релизом v0.2.0.

## Текущая фаза

**Фазы 5–14 ✅** (детекторы, 3D scene-graph, трекер, fusion-полировка, LLM-render/MCP, продукт-оболочка
v0.1.0). **Фаза 15 (ML Refactor → v0.2.0) ✅ по коду** — lab-first пересмотр всей ML-логики (журналы
LJ0–LJ6): 2D-GRIP-поза, E2 scene-level fall, stale-relink трекер, golden-симулятор + словарь v2
`using_object(class)`, ⭐ SFace align-фикс +19.5 п.п.; footprint позы −79%. v0.2.0 ждёт owner-тега.

**Фаза 16 — Dataset Factory (В РАБОТЕ, `kaggle`):** MVM из «источника для LLM» → двухпотребительская
архитектура (LLM/MCP + накопительная память-датасет). Новый крейт `mvm-dataset` — чистый потребитель
WorldState. **16.0–16.4 ✅ и live-провалидированы:** Semantic Contract (Canonical Observation +
статусы + Router G₁–G₅) → инфра (`mvm run --record`, session-grouper, privacy-blur) → сырьё
(RGB+depth-16bit+masks-RLE) + quality-as-data → CLI `mvm dataset` → Python-экспортёры
(`prototype/pipeline/dataset/`, PyTorch/TFRecord/WebDataset). Сквозной путь запись→PyTorch DataLoader
проверен на живой камере. **Осталось:** 16.5 (точный YuNet-face-bbox privacy), 16.6 (интеграция с
тренерами + quality gates). Детали — `mvm/CLAUDE.md`, план — `mvm/docs/phase-16-plan.md`.

## Для Agents

При работе с Rust-кодом — читай `mvm/CLAUDE.md` (полный контекст: архитектура, крейты, фазы, TDD-паттерны)
и `mvm/AGENT.md` (как LLM-агенту пользоваться MVM как модулем зрения: pull-модель, MCP, honesty-семантика).

Python-модули доступны по путям: `core/face_id.py`, `tools/mvm_bridge.py`, `core/workspace_3d.py`.
