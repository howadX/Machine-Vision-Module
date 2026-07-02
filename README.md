# Machine Vision Module (MVM)

**Structured Decomposed Picture-to-Text** — ансамбль независимых микро-агентов зрения, собирающий
**метрический 3D scene-graph каждый кадр** (не VLM). CPU-first, $0, приватно, интерпретируемо.

## Раскладка репозитория

После слияния `rust-core` → `master` (2026-07-02):

| Путь | Что | Статус |
|------|-----|--------|
| **`mvm/`** | **Rust-продакшн** — 12 крейтов, ансамбль агентов, чистый `--native-camera` hot-path | ★ основной, фаза 10 (Data-Collection) закрыта, 443 теста |
| `core/`, `tools/` | Python гибрид-**bridge** (`mvm_bridge.py` JSON-мост для пути `--native`) | рабочий |
| `models/`, `src/`, `spike_pose/` | веса (крупные gitignored), legacy `sightd`, питон-спайки позы | вспом. |
| **`prototype/`** | **Python-прототип (архив)**: `sight.py`, `pipeline/` (обучение), `data/`, `legacy/`, `tests/` | заморожен |

## Запуск (продакшн)

```bash
cd mvm && cargo build --release
target/release/mvm-cli --native-camera --dump-interval 3 --dump-json   # чистый Rust hot-path
target/release/mvm-cli --native-camera --masks --dump-json             # + инстанс-маски (opt-in)
```

Детальный контекст — **`mvm/CLAUDE.md`** (source of truth Rust-части) и `mvm/docs/phase-*.md`.

## История

Проект прошёл путь Python-прототип → полная Rust-переписка. Python-линия сохранена на ветках
`python-prototype` / `3d-workspace` и теге `backup/master-pre-merge`; её актуальный срез — в `prototype/`.
