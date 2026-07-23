# MVM — ML Engineering Laboratory

> **Kaggle-профиль как живая лаборатория.** Каждый ноутбук = design doc с воспроизводимыми экспериментами.
> Обновляется при изменениях в production. Не архив — платформа прототипирования.

**Production:** [github.com/dawoh/mvm](https://github.com/dawoh/mvm) — Rust, 597 тестов, GitHub Releases
**Автор:** Вадим Ховрин · ML / Computer Vision Engineer
**Как это поддерживается:** [WORKFLOW.md](./WORKFLOW.md) — build → Kaggle push/pull через `kctl.py` (ноутбуки генерируются, не редактируются вручную)

## Структура каталога

```
kctl.py                 — драйвер сборки/синка + манифест ноутбуков
README · WORKFLOW · CHECKPOINT   — обзор, процесс, точка возобновления
notebooks/<nb>/         — по папке на ноутбук: _build_nbX.py + .ipynb + skeleton.md
research/               — исследования и заметки по ML-разработке (design docs, аудиты, планы)
results/metrics.json    — реальные числа, стянутые с прогонов Kaggle
```

---

## Временная шкала: от EmoZid к MVM

```
2019 ─ EmoZid: Haar + dlib + SVM (45%)
 │
2023 ─ LightGBM на 468 точках MediaPipe (68%)
 │      COCO + эвристика для поз (71.7%)
 │
2024 ─ CNN EfficientNet-B0 на FER2013 (73%)
 │      PoseVAE: manifold + few-shot + anomaly
 │
2025 ─ Knowledge distillation: CNN → MLP (72%, 7× легче)
 │      GRIP: 14 фич → RF → 0.959
 │
2026 ─ Production: EmotiEffLib ONNX + GRIP в Rust
        MVM v0.1.0: 597 тестов, MCP, systemd
```

---

## Коллекция ноутбуков

| # | Ноутбук | ML-задача | Период | Production-крейт |
|---|---------|-----------|--------|-----------------|
| **0** | EmoZid: SVM на 68 точках лица | Face emotion, classical ML | 2019–2020 | (исторический) |
| **1** | Distilling CNN → geometric MLP | Knowledge distillation, edge inference | 2023–2025 | `mvm-emotion` |
| **2** | PoseVAE: manifold that knows when it's surprised | Representation learning, few-shot, anomaly | 2024–2025 | `mvm-posture` (OOD-gate) |
| **3** | Weak supervision at scale | Data-centric ML, HPO, свой датасет | 2024 | `mvm-posture` (data pipeline) |
| **3b** | GRIP: 14 features beat VAE | Feature engineering, production classifier | 2025–2026 | `mvm-posture` (core) |
| **4** | Why identity embeddings are the wrong tool for emotion | Engineering judgment, component teardown | 2023 | `mvm-identity` + `mvm-emotion` |

---

## Формат «живого документа»

Каждый ноутбук содержит:

1. **Заголовок + TL;DR + временная метка** (когда последний раз обновлялся)
2. **Исторический контекст** (как пришли к этому решению)
3. **Воспроизводимый эксперимент** (публичные данные, фиксированные seeds)
4. **Метрики и выводы** (честно: что измерено, что спроецировано)
5. **Production Status** — как это живёт в MVM сегодня
6. **What Changed** — хронология обновлений ноутбука
7. **Связи:** ← предыдущий ноутбук | → следующий ноутбук

### Принципы

- **Честность.** Приватные данные → публичный аналог с явной пометкой. Проекции помечены как проекции.
- **Воспроизводимость.** Run All на Kaggle без внешних файлов. Seeds зафиксированы.
- **Живость.** При изменении production-модели → обновляется ноутбук (новая секция «What Changed»).
- **Связность.** Каждый ноутбук ссылается на предыдущий/следующий и на GitHub-крейт.
