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
└── mvm/            # 🆕 Rust workspace (8 крейтов, 47 тестов, фаза 5.1)
    ├── crates/     # mvm-core, detect, identity, emotion, posture, fusion, pipeline, cli
    ├── docs/       # phase-5.1-claude-prompt.md
    └── CLAUDE.md   # ← детальный контекст Rust-части
```

## Как работает

Гибридный пайплайн: Python bridge (`tools/mvm_bridge.py`) захватывает камеру и выдаёт JSON lines, Rust `mvm-cli` парсит их и обогащает нативными ONNX-моделями.

```bash
# Python bridge (камера → JSON)
venv/bin/python3 tools/mvm_bridge.py --json-only --max-frames 1

# Rust mvm-cli (запускает bridge как subprocess)
cd mvm && cargo run --release -- --native --once --dump-interval 1
```

## Актуальная ветка

`rust-core` — основная ветка разработки.

## Для Claude Code

При работе с Rust-кодом — читай `mvm/CLAUDE.md` (полный контекст: архитектура, крейты, фазы, TDD-паттерны).

Python-модули доступны по путям: `core/face_id.py`, `tools/mvm_bridge.py`, `core/workspace_3d.py`.
