# Machine Vision Module

*Structured Decomposed Picture-to-Text — гибридный Python+Rust проект.*

## Структура

> **После слияния `rust-core`→`master` (2026-07-02):** корень = **Rust-продакшн** + рабочий гибрид-bridge;
> автономный **Python-прототип** вынесен в `prototype/`. Историю Python-линии см. на ветках
> `python-prototype`/`3d-workspace` и теге `backup/master-pre-merge`.

```
Machine Vision Module/
├── mvm/            # ★ Rust workspace — ПРОДАКШН (12 крейтов, 597 тестов; фаза 14 Продукт закрыта по коду)
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

`master` — продакшн и основная ветка разработки (после слияния `rust-core`→`master` 2026-07-02;
`rust-core` — предок слияния, отстал).

## Текущая фаза

**Фаза 14 — Продукт (ЗАКРЫТА по коду 2026-07-06):** оболочка «поставил → само нашлось и заработало». Закрыто и
live-провалидировано (`mvm/docs/phase-14-plan.md`): **14.0** фундамент (XDG-пути `~/.local/share/mvm`,
`config.toml` с приоритетом CLI>env>file>default, авто-детект камеры через шов `CameraSource`, бинарь
называется `mvm`, `mvm doctor`), **14.1** `mvm setup` (fetch моделей с sha256-verify + HTTP range-resume,
единый манифест, авто-триггер обязательных на первом `run`), **14.2** watched-галерея (`gallery/<Имя>/*.jpg`,
`mvm gallery add/list` — убран hardcoded name-map), **14.3** `mvm serve --with-camera` (единый процесс:
продюсер + MCP над своими свежими артефактами), **14.4** `mvm mcp install` (регистрация MCP-сервером у
claude-desktop/claude-code/hermes; dry-run + безопасный `--write`), **14.5** `mvm service` (systemd
user-юнит продюсера, без root), **14.6** дистрибуция (GitHub Release `models-v1` для 2 project-hosted
YOLO → `mvm setup` 9/9; release-workflow тег `v*` → бинарь x86_64/aarch64; `install.sh` = `curl|sh`).
12 крейтов, 597 тестов. **Владельцу (вне репо):** репойнт Sight-профиля Hermes на XDG-путь; пуш
master+тега `v0.1.0` (CI-релиз) и public-flip репо (анонимные `curl|sh`/fetch YOLO).

Ранее закрыто: **Фаза 11 — Fusion-полировка** (activity-FSM + event-journal, зоны/anomaly v2,
mask-контакт, object-permanence, self-calib; `+0 ONNX`). **Фаза 12 — LLM-render** (snapshot v2 + recap +
inspect + токен-диета + scene/events-логи + rolling journal.md + MCP stdio-сервер). **Фаза 13 —
Agent Tools/Skills Polishing** (когерентное CLI-дерево, `--help`, `AGENT.md`, само-документирующий MCP).

## Для Claude Code

При работе с Rust-кодом — читай `mvm/CLAUDE.md` (полный контекст: архитектура, крейты, фазы, TDD-паттерны).

Python-модули доступны по путям: `core/face_id.py`, `tools/mvm_bridge.py`, `core/workspace_3d.py`.
