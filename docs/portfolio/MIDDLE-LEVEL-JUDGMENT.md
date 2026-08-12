# MVM: middle-level judgment, а не витрина стека

> MVM будет демонстрировать не только широту стека, а **middle-level judgment** — измерение, качество, эксплуатацию, безопасность, ограничения и осмысленные trade-offs.

## Зачем этот документ

MVM — не учебная коллекция моделей и не перечень технологий для резюме. Это локальная CV-система, в которой инженер принимает решения о качестве, latency, ресурсах, данных, приватности и границах достоверности.

Ценность проекта для CV/ML Engineer определяется не числом крейтов, моделей или интеграций. Она определяется тем, можно ли доказать:

1. **что именно система делает;**
2. **на каких данных и в каких условиях это проверено;**
3. **какие компромиссы выбраны и почему;**
4. **где система должна честно отказаться от утверждения;**
5. **как её запустить, измерить, наблюдать и безопасно эксплуатировать.**

## Пять измерений middle-level engineering

### 1. Измерение вместо впечатлений

Любое числовое утверждение должно быть воспроизводимо.

| Утверждение | Что требуется как доказательство |
|---|---|
| FPS / latency | сценарий, hardware, версия MVM и моделей, warm-up, длительность, raw output, различение delivery FPS и processing throughput |
| Качество модели | dataset/source, split, метрика, baseline, версия модели, known failure modes |
| Потребление ресурсов | RAM/VRAM, CPU/GPU, cold start, число потоков, конфигурация |
| Надёжность | конкретный bounded test, replay/golden fixture или воспроизводимый failure/recovery сценарий |

Если измерения нет, допустимые статусы: **design target**, **tested behavior** или **planned experiment**. Это не слабость, а инженерная честность.

### 2. Качество — это контекст, а не одна accuracy

Для MVM нельзя публиковать единую «accuracy системы». У неё разные типы сигналов и разные уровни проверки:

- component-level: detection, identity, posture, emotion, objects, depth;
- pipeline-level: tracking, fusion, occlusion, relation/activity inference;
- product-level: CLI, state contracts, Dataset Factory, MCP, privacy paths;
- scenario-level: пустая комната, один/два человека, crossing, частичная окклюзия, лицо отсутствует, плохой свет, неожиданный объект.

Каждый уровень должен иметь собственную метрику или явную отметку `not measured`.

`withheld` и `unknown` — корректный результат, если входных данных недостаточно. Они не превращаются в отрицательный класс и не маскируются уверенной, но ложной меткой.

### 3. Эксплуатация — часть архитектуры

Работающая модель не равна работающему продукту. Для MVM эксплуатационный контур включает:

- воспроизводимую установку и проверку моделей (`setup`, SHA-256, `doctor`);
- bounded demo без камеры и отдельный live-camera путь;
- конфигурацию и XDG-пути;
- health/failure signals: camera, model loading, stale state, disk pressure, Dataset Factory;
- понятный runbook: симптом → evidence command → безопасное действие → условие остановки;
- release surface: CLI, systemd user service, MCP read-only interface.

Observability не добавляется ради buzzword. Метрики, dashboard и alerts нужны только если их семантика и действие оператора определены заранее.

### 4. Безопасность и приватность — инженерные ограничения

MVM обрабатывает чувствительный визуальный контекст. Следовательно, privacy и security не являются косметическим разделом документации.

Нужно явно описывать:

- активы: frames, записи, gallery, embeddings, labels, модели, конфигурация, MCP state;
- trust boundaries: camera, local user, service, filesystem, MCP client, release/model sources;
- существующие controls: local/offline runtime, SHA-256, privacy blur, read-only MCP, XDG boundaries, explicit uncertainty;
- реальные пробелы: encryption at rest, access control, retention automation, audit trail — пока они не реализованы и не проверены;
- правила портфолио: не коммитить private footage, gallery или identity embeddings; использовать public/synthetic/consented fixtures.

Бакалавриат ИБ проявляется здесь через threat modeling, data lifecycle и проверяемые границы, а не через формальную строку в резюме.

### 5. Trade-offs должны быть видимы

MVM намеренно делает выборы, которые необходимо уметь объяснить:

| Решение | Выигрыш | Цена / ограничение |
|---|---|---|
| CPU-first, $0 | локальность, приватность, доступность, edge-fit | ограниченный throughput и ceiling качества относительно GPU/cloud решений |
| Нативный Rust hot path | контроль latency, deployment и ресурсов | выше стоимость интеграции и меньше готовых ML-абстракций |
| Модульные experts + fusion | интерпретируемость, заменяемость, per-signal confidence | нужно поддерживать contracts и согласовывать ошибки нескольких моделей |
| Body-primary | трекинг человека без фронтального лица | identity/emotion остаются unavailable без face evidence |
| `withheld` вместо догадки | безопасность утверждений и предсказуемое поведение | меньше «красивых» ответов в сложных/окклюдированных сценах |
| Dataset Factory | data lifecycle, provenance, будущие модели | нужны retention, consent и quality gates, а не бессрочное накопление кадров |

## Портфельный стандарт MVM

Перед любым публичным claim, README-блоком или строкой резюме нужно ответить:

```text
1. Какой конкретный сценарий и пользовательская ценность?
2. Чем это измерено или протестировано?
3. Какой artefact позволяет другому инженеру это проверить?
4. Где граница применимости и известный failure mode?
5. Какой trade-off сделан и почему он рационален?
```

Если на вопрос нет ответа, claim откладывается либо получает статус `planned`.

## Практические артефакты

Эта декларация реализуется через следующие документы и evidence:

- `docs/portfolio/EVIDENCE-CONTRACT.md` — реестр доказуемых claims;
- `docs/portfolio/DEMO.md` — bounded reproducible demo;
- `mvm/benchmark/performance/` — performance passport и raw measurements;
- `mvm/benchmark/quality/` — quality passport и failure taxonomy;
- `docs/operations/METRICS-CONTRACT.md` и `RUNBOOK.md` — operational slice;
- `docs/security/THREAT-MODEL.md` и `DATA-LIFECYCLE.md` — security/privacy case;
- `docs/portfolio/CASE-STUDY.md` — десятиминутный narrative для работодателя;
- `docs/portfolio/INTERVIEW-BRIEF.md` — interview deep dives и vacancy-to-evidence mapping.

Полный порядок работ: `~/wiki/studium/.hermes/plans/2026-07-27-mvm-middle-track-portfolio.md`.

## Итог

MVM не должен пытаться казаться большой корпоративной платформой. Он должен быть лучше: **честно ограниченной, измеримой и воспроизводимой CV/ML-системой, в которой видны инженерные решения владельца.**
