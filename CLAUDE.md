# Machine Vision Module

*Structured Decomposed Picture-to-Text — гибридный Python+Rust проект. Продакшн v0.1.0 (GitHub Releases).*

## Структура

> **После слияния `rust-core`→`master` (2026-07-02):** корень = **Rust-продакшн** + рабочий гибрид-bridge;
> автономный **Python-прототип** вынесен в `prototype/`. Историю Python-линии см. на ветках
> `python-prototype`/`3d-workspace` и теге `backup/master-pre-merge`.

```
Machine Vision Module/
├── mvm/            # ★ Rust workspace — ПРОДАКШН (12 крейтов, 597 тестов; фаза 14 «Продукт» закрыта целиком)
│   ├── crates/     #   core, detect, identity, emotion, posture, scene, depth, hands, fusion, mesh, pipeline, cli
│   ├── docs/       #   phase-* prompts/runbooks
│   └── CLAUDE.md   #   ← детальный контекст Rust-части (source of truth)
├── install.sh      # curl|sh установщик релизного бинаря (GitHub Releases, x86_64/aarch64)
├── core/           # Python гибрид-bridge: face_id, workspace_3d, … (нужен пути `--native`)
├── tools/          # Python bridge-инструменты: mvm_bridge.py (JSON-мост), calib_3d.py
├── models/         # Веса: ONNX/MediaPipe (.task), references/ (крупные gitignored, тянутся `mvm setup`)
├── src/            # sightd (legacy Rust daemon, заменён на mvm/)
├── spike_pose/     # Питон-спайки тренировки позы (GRIP)
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

`master` — продакшн и основная ветка разработки (после слияния `rust-core`→`master` 2026-07-02).

## Текущая фаза

**Фаза 14 — Продукт (ЗАКРЫТА по коду 2026-07-06):** оболочка «поставил → само нашлось и заработало» —
XDG-пути/`config.toml`, авто-детект камеры, `mvm setup` (fetch моделей + sha256), watched-галерея
(`gallery/<Имя>/`), `mvm serve --with-camera`, `mvm mcp install`, `mvm service` (systemd user-юнит),
дистрибуция через GitHub Releases + `install.sh`. Репо публичное, релиз `v0.1.0` собран CI, `install.sh`
анонимно ставит `mvm 0.1.0` — live-провалидировано. 12 крейтов, 597 тестов. Детальный статус —
`mvm/CLAUDE.md`, план — `mvm/docs/phase-14-plan.md`.

**Фаза 15 — ML Refactor → v0.2.0 (В РАБОТЕ, ветка `kaggle`):** лабораторный пересмотр всей ML-логики
на эталонной паре (CMU mocap + OpenImages), три оси — скорость/точность/вес. **15.0 ✅** (лаборатория +
замороженный бенч-протокол) и **15.1 ✅** (поза: 2D-GRIP z-free, mocap как тренировочный актив,
flexion-канал, Mahalanobis вместо VAE; footprint позы −79%, gold R3 0.894) закрыты и опубликованы как
Kaggle-журналы LJ0/LJ1 (Run-All COMPLETE). Каждый этап гейтится нарративным журналом (LJ0–LJ6).
**Прод-Rust пока v0.1.0** — порт победивших решений будет в 15.6. План — `mvm/docs/phase-15-plan.md`.

## Для Agents

При работе с Rust-кодом — читай `mvm/CLAUDE.md` (полный контекст: архитектура, крейты, фазы, TDD-паттерны)
и `mvm/AGENT.md` (как LLM-агенту пользоваться MVM как модулем зрения: pull-модель, MCP, honesty-семантика).

Python-модули доступны по путям: `core/face_id.py`, `tools/mvm_bridge.py`, `core/workspace_3d.py`.
