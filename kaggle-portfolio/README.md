# MVM — ML Engineering Laboratory

> Kaggle-портфолио — исследовательская лаборатория MVM, а не отдельный runtime.

**Production:** [Machine Vision Module](https://github.com/howadX/Machine-Vision-Module) — Rust, `v0.4.0`, Phase 18 Visual Cortex closed; next approved track is Phase 19 Visual Spatial Substrate.

MVM — evidence-backed **visual subsystem будущего робота**. Лаборатория проверяет perception experts и их data/quality trade-offs; production собирает их в Rust-пайплайн:

```text
visual evidence → canonical facts → spatial/temporal visual memory → Cortex interpretation → external robot runtime
```

MVM не является planner/controller/actuator/safety authority. Ноутбуки не должны расширять проект в эту область и не должны превращать `unknown`/`withheld` в отрицательный класс.

## Как поддерживается

[WORKFLOW.md](./WORKFLOW.md) описывает build → Kaggle push/pull через `kctl.py`. Ноутбуки генерируются и проверяются, а не редактируются вручную без обновления источника.

## Структура

```text
kctl.py                 — build/sync driver и notebook manifest
notebooks/<nb>/         — design docs, builders, notebooks, skeletons
research/               — исследования, аудиты и ML design docs
results/metrics.json    — фактические experiment results
```

## Связь research ↔ production

Каждый эксперимент должен явно отвечать:

1. какой visual signal он улучшает;
2. какой production crate его потребляет (`mvm-posture`, `mvm-identity`, `mvm-emotion` и т. п.);
3. какие данные и domain limits измерены;
4. как сохраняется provenance и honesty semantics;
5. есть ли CPU/latency/footprint trade-off.

Production source of truth — [`../mvm/CLAUDE.md`](../mvm/CLAUDE.md), а не исторический notebook.

## Линия экспериментов

История идёт от classical face/emotion и geometric models к production ONNX/Rust experts, GRIP posture и Dataset Factory. Эти материалы объясняют инженерные решения и ограничения; числа остаются экспериментальными и не являются универсальными claims без указанного benchmark protocol.

| Область | Production boundary |
|---|---|
| face / identity / emotion | visual evidence для canonical Human facts |
| posture / hands | body-primary facts с confidence и coverage |
| objects / depth / masks | Scene/Depth evidence для Fusion и будущего spatial substrate |
| dataset / active learning | `mvm-dataset`, semantic + pixel sessions |
| Cortex | consumer/interpreter; не perception ground truth |

## Лицензия

См. лицензию корневого production-проекта и лицензии конкретных моделей/датасетов.
