<div align="center">

# MVM — Machine Vision Module

### Зрение, на которое может опираться остальная система

**MVM превращает поток с камеры в честную, долговременную и проверяемую модель видимого мира.**

Локально · CPU-first · Rust hot path · $0 runtime · privacy-first · evidence-backed

![release](https://img.shields.io/github/v/release/howadX/Machine-Vision-Module?label=release&color=8A2BE2)
![status](https://img.shields.io/badge/status-v0.4.0%20%7C%20Visual%20Cortex%20closed-brightgreen)
![runtime](https://img.shields.io/badge/runtime-local%20CPU%20%2F%20%240-success)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

</div>

---

## Зачем нужен MVM

Роботу или агенту недостаточно получать отдельные ответы на вопрос «что сейчас в кадре». Ему нужна **устойчивая зрительная картина мира**, к которой можно вернуться:

- кто и какие объекты были видны;
- где они находились и как были связаны;
- что менялось со временем;
- что система действительно наблюдала, а что лишь предполагает;
- каких данных всё ещё не хватает.

**Цель MVM** — стать настолько глубоким и контекстным зрением, чтобы остальные модули могли использовать его как свою **долговременную визуальную память** и **evidence-backed модель окружающего мира**.

```text
камера + доступный sensor/body context
        ↓
специализированные visual experts
        ↓
canonical visual facts о сцене
        ↓
долговременная semantic history + bounded visual world model
        ↓
Visual Cortex: память, прогноз, неопределённость, внимание, affordance
        ↓
внешний агент / robot runtime принимает решения и действует
```

MVM не пытается быть всем роботом. Он **не** planner, controller, actuator interface, safety authority или универсальный VLM. Планирование задач, моторика, физическое действие, звук, тактильность и проприоцепция принадлежат внешним модулям. Зона ответственности MVM — зрительные факты и их честная интерпретация.

## Что MVM делает сейчас

MVM — ансамбль небольших независимых vision experts, а не одна непрозрачная end-to-end модель.

| Слой | Что даёт |
|---|---|
| **Human** | тело как главная сущность, pose, лицо, identity, emotion, hands |
| **Scene** | объекты, tracking, optional instance masks |
| **Depth / geometry** | camera-relative depth и метрический 3D context, когда его поддерживают evidence и calibration |
| **Fusion** | immutable scene facts: entities, relations, zones, activities, events |
| **Semantic Session Log** | durable `labels.jsonl`: текстовая история наблюдений по умолчанию |
| **Dataset Factory** | `--record` добавляет RGB/depth/masks для обучения и data-quality gates |
| **Visual Cortex** | evidence-linked memory, 1s/2s forecast, uncertainty, attention, affordance и advisory re-observation |

Главное правило MVM: **неизвестное остаётся неизвестным**. `unknown`, `withheld`, stale result или отсутствие детекции никогда не трактуются как отрицательный факт.

## Быстрый старт

### Установить release-бинарь

Linux `x86_64` и `aarch64`:

```bash
curl -fsSL https://raw.githubusercontent.com/howadX/Machine-Vision-Module/master/install.sh | sh

mvm setup    # скачать ONNX-модели с SHA-256 проверкой
mvm doctor   # проверить камеру, модели, конфиг и data directory
```

Если репозиторий закрыт и анонимная загрузка недоступна, установщик автоматически использует аутентифицированный `gh release download`, если GitHub CLI уже настроен.

### Или собрать из исходников

```bash
git clone https://github.com/howadX/Machine-Vision-Module.git
cd Machine-Vision-Module/mvm
cargo build --release

./target/release/mvm setup
./target/release/mvm doctor
```

## Первый запуск

```bash
# запустить production vision pipeline; работает до Ctrl-C
mvm run

# в другом терминале — прочитать canonical facts о сцене сейчас
mvm snapshot

# посмотреть отдельный cognitive view: что требует внимания
mvm cortex --view attention
```

`mvm run` автоматически ищет камеру, запускает Rust AgentMesh и пишет артефакты в `~/.local/share/mvm` (путь настраивается через `~/.config/mvm/config.toml`).

## Как пользоваться зрительной памятью

Начинайте с самого дешёвого и надёжного слоя, затем углубляйтесь только когда это нужно:

```bash
mvm snapshot                         # Что наблюдается сейчас: canonical facts
mvm cortex --view uncertainty         # Чего MVM не знает и почему
mvm cortex --view forecast            # Короткий evidence-backed прогноз
mvm recap --last 10m                  # Что происходило за последние 10 минут
mvm inspect --id 12                   # Проверить evidence вокруг конкретного track
```

### Что является источником истины

| Артефакт | Роль |
|---|---|
| `current_state.md` | атомарный снимок **canonical facts** текущей сцены |
| `labels.jsonl` | единственная durable text-only semantic history |
| `current_cortex.json` | текущая cognitive интерпретация: не facts и не долговременная история |

Cortex не переписывает `FusedSceneFrame`: прогноз, attention, affordance и advisory request всегда остаются отдельными от наблюдённых фактов.

## Запись данных для обучения

По умолчанию `mvm run` сохраняет только text-only semantic session — без кадров. Это и есть безопасная долговременная визуальная память.

Если нужны пиксели для обучения или отладки, включите запись:

```bash
mvm run --record                      # RGB + depth + masks + labels, privacy-safe default
mvm dataset list                      # список сессий
mvm dataset check --strict            # integrity/privacy/quality gate
```

Privacy modes:

- `safe` — по умолчанию, размывает лица анонимных людей;
- `anonymous` — размывает все лица;
- `full` — не размывает лица.

```bash
mvm run --record --privacy anonymous
```

Python exporters для PyTorch, TFRecord и WebDataset находятся в [`prototype/pipeline/dataset/`](prototype/pipeline/dataset/), но production hot path остаётся полностью Rust.

## Подключить LLM через MCP

MVM предоставляет read-only инструменты для LLM-host:

```bash
mvm serve --mcp                  # MCP над уже работающим producer
mvm serve --mcp --with-camera    # producer и MCP в одном процессе
```

Инструменты MCP: `mvm_snapshot`, `mvm_cortex`, `mvm_recap`, `mvm_inspect`.

Рекомендуемый порядок для агента:

```text
snapshot → cortex / recap → inspect только для проверки конкретного утверждения
```

## Архитектурные принципы

1. **Honesty-first.** Facts, memories, forecasts и hypotheses не смешиваются. Неизвестность видима пользователю и downstream-модулям.
2. **Evidence before interpretation.** У каждого существенного утверждения есть source, timestamp, freshness и confidence.
3. **Immutable canonical facts.** Fusion публикует facts; Dataset Factory, memory и Cortex читают их, но не переписывают.
4. **CPU-first, local-first.** Без обязательных GPU, cloud API или утечки пикселей/embeddings.
5. **Минимализм.** Fusion, tracking, uncertainty и spatial reasoning строятся на простой арифметике и bounded state поверх уже доступных evidence.
6. **Rust в hot path.** Python сохранён для прототипов, bridge и offline ML/export workflows.

## Куда развивается проект

- **Phase 18 — Visual Cortex:** закрыта в `v0.4.0`. MVM уже умеет строить evidence-linked cognitive view поверх canonical facts.
- **Phase 19 — Visual Spatial Substrate:** approved следующий этап. Цель — развить object-centric scene graph в graph-first camera-/agent-relative spatial world model: regions, surfaces, visibility, occupancy и bounded temporal visual memory.

При этом MVM не будет выдавать `free`, `passable` или `safe` из одного отсутствующего detector output. Для spatial/robotic claims нужны положительные evidence, calibration и явно заданный reference frame.

## Репозиторий

| Путь | Назначение |
|---|---|
| [`mvm/`](mvm/) | Production Rust workspace, бинарь `mvm`, полный CLI и архитектурные детали |
| [`mvm/skills/`](mvm/skills/) | Canonical agent skills: visual pull/MCP и Dataset Factory |
| [`mvm/benchmark/`](mvm/benchmark/) | Replay и Visual QA contracts для Cortex |
| [`mvm/docs/phase-19-plan.md`](mvm/docs/phase-19-plan.md) | План следующего spatial substrate этапа |
| [`prototype/`](prototype/) | Исторический Python prototype и offline research/export tools |
| [`kaggle-portfolio/`](kaggle-portfolio/) | Воспроизводимая ML laboratory и связь experiments ↔ production |

## Подробнее

- Полный CLI, модельный стек и технические контракты: [`mvm/README.md`](mvm/README.md)
- Visual pull-model / MCP: [`mvm/skills/mvm-vision/SKILL.md`](mvm/skills/mvm-vision/SKILL.md)
- Запись, privacy, quality gates и export: [`mvm/skills/mvm-dataset/SKILL.md`](mvm/skills/mvm-dataset/SKILL.md)
- Текущие архитектурные правила: [`AGENTS.md`](AGENTS.md)

## Лицензия

Apache-2.0. ONNX-модели распространяются по собственным лицензиям.
