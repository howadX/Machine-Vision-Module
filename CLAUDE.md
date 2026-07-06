# Machine Vision Module

*Structured Decomposed Picture-to-Text — гибридный Python+Rust проект.*

## Структура

> **После слияния `rust-core`→`master` (2026-07-02):** корень = **Rust-продакшн** + рабочий гибрид-bridge;
> автономный **Python-прототип** вынесен в `prototype/`. Историю Python-линии см. на ветках
> `python-prototype`/`3d-workspace` и теге `backup/master-pre-merge`.

```
Machine Vision Module/
├── mvm/            # ★ Rust workspace — ПРОДАКШН (12 крейтов, 501 тест; фаза 11 Fusion в работе)
│   ├── crates/     #   core, detect, identity, emotion, posture, scene, depth, hands, fusion, mesh, pipeline, cli
│   ├── docs/       #   phase-* prompts/runbooks
│   └── CLAUDE.md   #   ← детальный контекст Rust-части (source of truth)
├── core/           # Python гибрид-bridge: face_id, workspace_3d, … (нужен пути `--native`)
├── tools/          # Python bridge-инструменты: mvm_bridge.py (JSON-мост), calib_3d.py
├── models/         # Веса: ONNX/MediaPipe (.task), references/ (крупные gitignored)
├── src/            # sightd (legacy Rust daemon, заменён на mvm/)
├── spike_pose/     # Питон-спайки тренировки позы (method-C)
└── prototype/      # ⚙ Python-ПРОТОТИП (архив): sight.py, pipeline/ (обучение), data/, legacy/, tests/, assets/
```

## Как работает

Два пути захвата:

- **`--native-camera`** (основной, фаза 7+): чистый Rust v4l2 MJPG → body-primary цепочка
  (BlazePose-тело + YuNet-лицо + SFace + EmotiEffLib + YOLO26n-объекты + Depth-Anything + handpose)
  → метрический 3D scene-graph. **Ноль Python.**
- **`--native`** (bridge, для тестов/дебага): Python `tools/mvm_bridge.py` захватывает камеру и выдаёт
  JSON lines, Rust обогащает нативными ONNX-моделями. venv остаётся для этого пути.

```bash
# Основной путь — живой mesh, чистый Rust (фаза 14.0: бинарь называется `mvm`,
# камера авто-детект, state+логи → ~/.local/share/mvm)
cd mvm && target/release/mvm run --dump-json

# Sync-путь (legacy, скрытый) + визуальный дебаг: детекторы цветом → /tmp/mvm_scene.jpg
cd mvm && target/release/mvm --native-camera --dump-interval 3 --scene-debug

# Bridge-путь (Python мост, для дебага)
cd mvm && target/release/mvm --native --once --dump-json
```

**Камера:** `--native-camera` использует прямой v4l2 MJPG mmap; bridge использует `cv2.CAP_FFMPEG`
(не V4L2) — обход PipeWire на Ubuntu 24+.

## Актуальная ветка

`rust-core` — основная ветка разработки.

## Текущая фаза

**Фаза 11 — Fusion-полировка** (реляц.-темпоральная арифметика над Scene JSON, **+0 ONNX**): mask-контакт
10.6 + person-маски 10.7 + фикс мис-атрибуции · зоны/anomaly v2 · **activity-FSM** (конечный автомат +
event-journal start/stop/duration) · метрич. калибровка. План — `mvm/docs/phase-11-plan.md`.
**Фаза 12 — LLM-render ЗАКРЫТА** (MVM как модуль зрения для LLM: snapshot v2 + recap + inspect +
токен-диета + scene/events-логи + rolling journal.md + MCP stdio-сервер `serve --mcp`;
все шаги live-провалидированы — `mvm/docs/phase-12-llm-render-plan.md`). Продукт (упаковка) — следующая.

## Для Claude Code

При работе с Rust-кодом — читай `mvm/CLAUDE.md` (полный контекст: архитектура, крейты, фазы, TDD-паттерны).

Python-модули доступны по путям: `core/face_id.py`, `tools/mvm_bridge.py`, `core/workspace_3d.py`.
