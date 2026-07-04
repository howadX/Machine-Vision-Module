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
# Основной путь — чистый Rust, всё нативно
cd mvm && target/release/mvm-cli --native-camera --dump-interval 3 --dump-json

# Визуальный дебаг: все детекторы цветом своего агента → /tmp/mvm_scene.jpg
cd mvm && target/release/mvm-cli --native-camera --dump-interval 3 --scene-debug

# Bridge-путь (Python мост, для дебага)
cd mvm && target/release/mvm-cli --native --once --dump-json
```

**Камера:** `--native-camera` использует прямой v4l2 MJPG mmap; bridge использует `cv2.CAP_FFMPEG`
(не V4L2) — обход PipeWire на Ubuntu 24+.

## Актуальная ветка

`rust-core` — основная ветка разработки.

## Текущая фаза

**Фаза 11 — Fusion-полировка** (реляц.-темпоральная арифметика над Scene JSON, **+0 ONNX**): mask-контакт
10.6 + person-маски 10.7 + фикс мис-атрибуции · зоны/anomaly v2 · **activity-FSM** (конечный автомат +
event-journal start/stop/duration) · метрич. калибровка. Продукт (упаковка) — следующая, пока не
определённая фаза. План — `mvm/docs/phase-11-plan.md`.

## Для Claude Code

При работе с Rust-кодом — читай `mvm/CLAUDE.md` (полный контекст: архитектура, крейты, фазы, TDD-паттерны).

Python-модули доступны по путям: `core/face_id.py`, `tools/mvm_bridge.py`, `core/workspace_3d.py`.
