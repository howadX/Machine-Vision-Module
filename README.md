<div align="center">

# MVM — Machine Vision Module

**Structured Decomposed Agentic Vision** — локальный модуль зрения: камера →
**метрический 3D scene-graph** каждый кадр (не VLM). Ансамбль независимых микро-агентов вместо
одной end-to-end модели. **Два потребителя:** LLM-агент (pull-модель/MCP) и накопительная
**Dataset Factory** (структурированная память для обучения). CPU-first, $0, приватно, интерпретируемо.

![release](https://img.shields.io/github/v/release/howadX/Machine-Vision-Module?label=release&color=8A2BE2)
![phase](https://img.shields.io/badge/phase-16.6--A%20(Dataset%20Factory)-brightgreen)
![tests](https://img.shields.io/badge/tests-670%20green-brightgreen)
![crates](https://img.shields.io/badge/crates-13-informational)
![runtime](https://img.shields.io/badge/runtime-CPU%20%2F%20%240-success)
![rust](https://img.shields.io/badge/rust-edition%202024-orange)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

</div>

---

## Установка

```bash
# одна команда (GitHub Releases, linux x86_64/aarch64)
curl -fsSL https://raw.githubusercontent.com/howadX/Machine-Vision-Module/master/install.sh | sh

mvm setup           # дотянет модели (sha256-verified, HTTP range-resume)
mvm run             # живой mesh: камера + все агенты → снимок сцены каждый кадр
mvm snapshot        # что происходит прямо сейчас — одной командой

# Dataset Factory (фаза 16) — накопить свой датасет и проверить его
mvm run --record    # копить сессии (RGB+depth+masks+labels, privacy-safe)
mvm dataset list    # что накоплено (info / stats / prune / export)
mvm dataset check   # data-quality гейт (целостность/privacy/баланс, CI exit-code)
```

Полная инструкция, CLI-дерево, выходной JSON-контракт и honesty-семантика — **[`mvm/README.md`](mvm/README.md)**.
Как LLM-агенту пользоваться MVM (pull-модель, MCP) — **[`mvm/AGENT.md`](mvm/AGENT.md)**.

## Что это

MVM превращает видеопоток в граф: **узлы-сущности** с метрической 3D-позицией (origin = камера,
pinhole-геометрия от known-size априоров — не depth-сеть) и **рёбра-отношения** (близость,
пересечение, зоны). Несущая сущность — **тело** (body-primary): человек трекается по телу, а
лицо, identity и эмоция — это его *атрибуты*, которые переживают отворот головы и окклюзию.

Каждый сигнал несёт числовую уверенность, а неопределённость выражается **явно** (⚠️), а не
замалчивается — модуль скорее скажет «поза withheld, видна только верхняя часть», чем угадает.

Этот же граф питает **двух потребителей**. Первый — **LLM-агент**: pull-модель `snapshot`/`recap`/
`inspect` через MCP, read-only. Второй (фаза 16) — **Dataset Factory**: `mvm run --record` компилирует
каждый кадр в интерпретируемую обучающую запись (Canonical Observation + provenance + статусы честности,
`withheld` ≠ negative), копит самодостаточные сессии на диск и отдаёт их Python-экспортёрам
(PyTorch/TFRecord/WebDataset). Это **память системы** — не архив кадров, а компилятор наблюдений.

## Архитектура: агенты и потоки данных

```mermaid
flowchart TD
    CAM(["📷 Камера /dev/video0<br/>нативный v4l2 MJPG mmap, eye-level ~175см<br/>ноль Python"])

    subgraph MESH["Agent Mesh — std::thread, mvm run"]
        direction LR
        subgraph HUMAN["HumanAgent · тело (body-primary)"]
            direction TB
            POSE["BlazePose 33lm → RandomForest<br/>standing / sitting / lying + torso-3D"]
            FACE["YuNet → лицо, матч по носу тела"]
            ID["SFace → identity (галерея)"]
            EMO["EmotiEffLib → emotion×8 + entropy"]
            HANDS["Handpose 21lm/рука → жесты"]
        end
        SCENE["SceneAgent<br/>YOLO26n объекты + YOLO-seg маски (opt-in)"]
        DEPTH["DepthAgent<br/>Depth-Anything-V2 → относит. карта"]
    end

    FUSION["FusionAgent<br/>build_scene: 3D-граф + зоны · activity-FSM · anomaly v2"]

    subgraph C1["Потребитель 1 — LLM (pull-модель)"]
        OUT1["current_state.md · snapshot"]
        OUT3["journal.md · rolling recap"]
        MCP["MCP stdio · serve --mcp"]
    end
    subgraph C2["Потребитель 2 — Dataset Factory (фаза 16, mvm-dataset)"]
        REC["mvm run --record<br/>sessions/: frames + depth + masks + labels.jsonl"]
        GATE["mvm dataset check<br/>+ validate.py · quality-гейт"]
        EXP["Python-экспортёры<br/>PyTorch / TFRecord / WebDataset"]
    end
    LLM(["🤖 LLM-агент<br/>Hermes / Claude Code / Alter"])

    CAM --> HUMAN
    CAM --> SCENE
    CAM --> DEPTH
    HUMAN --> FUSION
    SCENE --> FUSION
    DEPTH --> FUSION
    FUSION --> OUT1 & OUT3 & MCP
    FUSION -->|WorldState-снимок| REC
    REC --> GATE --> EXP
    MCP -->|snapshot → recap → inspect| LLM
```

Пословный разбор каждого агента, крейтов, кадансов и Scene-JSON контракта — **[`mvm/README.md`](mvm/README.md)**.

## Раскладка репозитория

| Путь | Что | Статус |
|------|-----|--------|
| **`mvm/`** | **Rust-продакшн** — 13 крейтов (+`mvm-dataset`), ансамбль агентов, бинарь `mvm` | ★ основной; фаза 16 (Dataset Factory) в работе, 16.0–16.6-A закрыты |
| `install.sh` | `curl \| sh`-установщик (детект ОС/арх, sha256-verify) | рабочий |
| `core/`, `tools/` | Python гибрид-**bridge** (`mvm_bridge.py`) для пути `--native` (тесты/дебаг) | вспом. |
| `models/` | Веса ONNX/MediaPipe; крупные — gitignored, тянутся `mvm setup` | вспом. |
| **`prototype/pipeline/dataset/`** | Python-экспортёры Dataset Factory (PyTorch/TFRecord/WebDataset) + `validate.py` | фаза 16 |
| **`prototype/`** | Python-прототип (архив): `sight.py`, `pipeline/`, `data/`, `legacy/` | заморожен |

**Релизы:** `v0.1.0` (фаза 14, Продукт) опубликован на GitHub Releases. `v0.2.0` (фаза 15, ML Refactor) и
`v0.3.0` (фаза 16, Dataset Factory) подготовлены по коду — ждут owner-тега (CI триггерится тегом).

История: проект прошёл путь Python-прототип → полная Rust-переписка. Python-линия сохранена на
ветках `python-prototype`/`3d-workspace` и теге `backup/master-pre-merge`.

## Roadmap (кратко)

Фазы 1–14 закрыты: стабилизация → metric 3D scene-graph → pure-Rust hot-path → body-primary
Agent Mesh → tracker (Hungarian) → full data collection + segmentation → fusion-полировка
(activity-FSM, anomaly v2) → LLM-render (snapshot/recap/inspect, MCP) → agent-tools polish →
**Продукт** (v0.1.0: auto-setup моделей, watched-галерея, `mvm service`, `mvm mcp install`, дистрибуция).
**Фаза 15 (v0.2.0) — ML Refactor:** lab-first пересмотр всей ML-логики (frozen bench, паспорт данных/кода) —
2D-GRIP поза, E2 fall (AUROC≈1.0), ⭐ SFace align +19.5 п.п., footprint −79%. **Фаза 16 (v0.3.0) — Dataset
Factory:** двухпотребительская архитектура, крейт `mvm-dataset`, `mvm run --record`, экспортёры, quality-гейты
(16.6-A ✅); осталось 16.6-B (hard-example mining). Детальный статус — **[`mvm/CLAUDE.md`](mvm/CLAUDE.md)**.

## Лицензия

Apache-2.0 (см. [`LICENSE`](LICENSE)). Модели — под своими лицензиями (YOLO26n/YOLO-seg — AGPL-3.0),
подробности — таблица агентов в `mvm/README.md`.
