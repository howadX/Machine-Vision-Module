# MVM Python Prototype — архив

> **Исторический каталог, не production MVM.**
>
> Актуальный MVM — Rust-based evidence-backed visual subsystem для будущего робота: visual facts → bounded visual memory/world model → advisory Visual Cortex. Смотрите [корневой README](../README.md) и [`mvm/README.md`](../mvm/README.md).

Этот каталог сохраняет раннюю Python-линию экспериментов: face recognition, emotion, posture, 3D spikes и Dataset Factory exporters. Он нужен для воспроизводимости исследований и offline ML-работ, но не определяет текущие runtime-контракты и не является source of truth.

## Что здесь находится

- `pipeline/` — старые training/data scripts и research spikes;
- `pipeline/dataset/` — Python-consumer для pixel sessions, экспорт PyTorch/TFRecord/WebDataset и validation;
- `core/`, `tools/` в корне репозитория — legacy/compatibility bridge;
- `legacy/` — ещё более ранние эксперименты.

## Граница с production

| Слой | Актуальный статус |
|---|---|
| Production capture, experts, Fusion, facts, Cortex | [`mvm/`](../mvm/) — Rust workspace |
| Semantic Session Log и pixel recording | `mvm-dataset`, `labels.jsonl`, `mvm run --record` |
| Python ML/export ecosystem | этот архив, только как consumer/research layer |
| Robot planning, control, actuators, safety authority | вне MVM |

Не переносите старые GPU/CUDA, `key: value`, `sightd`, Python-daemon или early posture assumptions в новые production-инструкции. Для текущих команд используйте `mvm --help` и canonical skills в [`mvm/skills/`](../mvm/skills/).

## Dataset exporter

Pixel exporters требуют записанную pixel session:

```bash
mvm run --record
cd prototype/pipeline/dataset
python validate.py ~/.local/share/mvm/datasets --strict
python test_dataset.py
```

Подробности: [`pipeline/dataset/README.md`](pipeline/dataset/README.md).

## История

Python-прототип предшествовал переходу на полноценный Rust AgentMesh. История сохранена намеренно: она показывает путь от локального perception script к evidence-backed visual subsystem, но не должна восприниматься как текущий API.

Лицензия production-проекта: Apache-2.0; исторические артефакты могут иметь собственные условия.
