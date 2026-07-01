# Machine Vision Module

*Structured Decomposed Picture-to-Text — гибридный Python+Rust проект.*

## Структура

```
Machine Vision Module/
├── core/           # Python модули: face_id, workspace_3d, pose_vae, emotion_hse
├── tools/          # Python инструменты: sight.py, mvm_bridge.py (JSON-мост)
├── pipeline/       # Обучение моделей
├── models/         # Веса: PyTorch (.pth), ONNX, MediaPipe (.task), references/
├── venv/           # Python 3.11+ virtualenv
├── src/            # sightd (legacy Rust daemon, заменён на mvm/)
└── mvm/            # Rust workspace (12 крейтов, 346 тестов, фаза 9 Tracker закрыта)
    ├── crates/     # core, detect, identity, emotion, posture, scene, depth, hands, fusion, mesh, pipeline, cli
    ├── docs/       # phase-* prompts/runbooks
    └── CLAUDE.md   # ← детальный контекст Rust-части (source of truth)
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

## Для Claude Code

При работе с Rust-кодом — читай `mvm/CLAUDE.md` (полный контекст: архитектура, крейты, фазы, TDD-паттерны).

Python-модули доступны по путям: `core/face_id.py`, `tools/mvm_bridge.py`, `core/workspace_3d.py`.
