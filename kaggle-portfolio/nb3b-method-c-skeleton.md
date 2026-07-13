# NB3b — Method-C: 14 Engineered Features Beat a VAE (2025–2026)

**Живой документ** · Последнее обновление: июль 2026 · Версия 1.0
← [NB3: Weak Supervision](./nb3-weak-supervision-skeleton.md) | → [NB4: Teardown](./nb4-pipeline-teardown-skeleton.md)

---

## Production Status (июль 2026)

**В production — основной классификатор позы.** `mvm-posture` использует method-C:
- 14 viewpoint-инвариантных фич → RandomForest (120 деревьев, max_depth=16)
- Экспорт: `pose_rf_forest.bin` (12 КБ) → `include_bytes!` в Rust
- Инференс: <1ms CPU
- Accuracy: **0.959** (grouped 5-fold CV на невиданных людях, азимут-ровно)

**Связанные крейты:** `mvm-posture` (core classifier), `mvm-fusion` (occlusion-honesty gate)

---

## 1. `[MD]` TL;DR

> 14 физически-инвариантных признаков + RandomForest = 0.959 accuracy.
> В 4 раза меньше параметров, чем VAE. В 100 раз быстрее CNN. Не зависит от поворота камеры.

- **Признаки:** наклоны к гравитации, сгибы суставов, пропорции тела — **всё инвариантно к азимуту**
- **Модель:** RandomForest (120 деревьев) — 12 КБ, <1ms CPU
- **Валидация:** grouped 5-fold CV (по людям, не по кадрам!) — 0.959 ± 0.01
- **Azimuth-тест:** поворот камеры на 90° → падение accuracy <0.5 п.п.

## 2. `[MD]` Исторический контекст: 7 итераций позы

```
2023 ─ Свои данные (capture_poses.py)
       Проблема: 1 человек, модель запоминает меня
2024 ─ COCO + эвристика (NB3)
       LightGBM 71.7% — модель сглаживает шум, но потолок ограничен
2024 ─ PoseVAE (NB2)
       ~85% на своих, ниже на публичных
2025 ─ Method-C spike (spike_feats.py)
       14 фич vs 99 сырых координат → фичи победили (0.959 vs ~0.83)
2025 ─ Полный прогон (train_pose_p1.py)
       URFD + Zenodo + COCO → публичный VAE + RF → экспорт в Rust
2026 ─ Production
       pose_rf_forest.bin в mvm-posture, occlusion-honesty gate в mvm-fusion
```

## 3. `[MD]` Ключевой инсайт: инвариантность by design

VAE не знает, что наклон торса к гравитации — инвариант при повороте камеры. Он должен **выучить** это из данных. Method-C **проектирует** инвариантность.

**14 признаков:**

| # | Признак | Что измеряет | Почему инвариантен |
|---|---------|-------------|-------------------|
| 1 | `torso_inc` | Наклон торса к вертикали | Гравитация не зависит от азимута |
| 2-4 | `thigh_inc`, `asym`, `shank_inc` | Наклоны конечностей | То же |
| 5-8 | `knee_flex`, `hip_flex`, `elbow_flex` | Углы сгиба суставов | Угол в 3D — абсолютный инвариант |
| 9-11 | `vext`, `hext`, `aspect` | Форма облака точек | Пропорции не зависят от поворота |
| 12-14 | `nose_above`, `ank_below`, `kn_below` | Вертикальные позиции | Относительно бёдер, в единицах длины торса |

## 4. `[CODE]` Функция compute_features()

(Из `spike_feats.py::features()` — с комментариями к каждой строчке)

## 5. `[CODE]` Распределение признаков по классам

Гистограммы: `torso_inc`, `knee_flex`, `aspect` — три класса видны невооружённым глазом.

## 6. `[CODE]` Grouped 5-fold CV

**Критически важно:** StratifiedGroupKFold — все кадры одного человека в train ИЛИ в test, никогда в обоих.

- RF: 0.959 ± 0.01
- Сырые координаты + RF: ~0.83
- VAE latent + RF: ~0.85

## 7. `[CODE]` Azimuth-тест

Поворот камеры на 15°/30°/45°/90°. Method-C: падение <0.5 п.п. Сырые координаты: катастрофа (0.83 → 0.31 на 90°).

## 8. `[MD]` Почему RandomForest, а не LightGBM?

| Фактор | LightGBM | RandomForest |
|--------|----------|-------------|
| Accuracy | 71.7% | **0.959** |
| Экспорт в Rust | C++ библиотека | 12 КБ бинарный формат |
| Зависимости | liblightgbm | **0 зависимостей** |
| Инференс | ~1-2ms | <1ms |
| Интерпретируемость | Важность признаков | Важность + дерево |

## 9. `[CODE]` Экспорт в production

Бинарный формат: `u32 n_trees` → per tree `{u32 n_nodes, nodes[...]}`, node = `<i32 feature, f32 threshold, i32 left, i32 right, f32 prob[3]>`.

12 КБ. `include_bytes!("pose_rf_forest.bin")` → десериализация → инференс.

## 10. `[MD]` Production Status

Method-C — **основной классификатор позы в MVM** с 2025 года. VAE (NB2) оставлен как OOD-gate.

**Occlusion-honesty (бонус):** если видны только верхняя часть тела (BodyCoverage::Upper) → поза **withheld** (не утверждается). Это не часть method-C, а fusion-слой (`mvm-fusion`), но method-C поставляет для этого `torso_attitude_2d`.

## 11. `[MD]` Открытые вопросы / что можно улучшить

- `torso_length_m=0.5` — не откалиброван рулеткой. Абсолютные 3D-позиции приблизительны.
- 3 класса (standing/sitting/lying) — достаточно для умного дома, но не различает «сидит на стуле» vs «сидит на полу».
- Классификатор **не ловит падения** (переход standing→lying за 3 секунды) — это делает anomaly detector (NB2 + mvm-fusion).

## 12. `[MD]` What Changed

- **2026-07** — ноутбук создан из spike-экспериментов (spike_feats.py + train_pose_p1.py)
- **2025-XX** — method-C заменяет LightGBM (NB3) и VAE (NB2)

## 13. `[MD]` Связи

← [NB3: Weak Supervision](./nb3-weak-supervision-skeleton.md) — откуда взялись данные
← [NB2: PoseVAE](./nb2-posevae-skeleton.md) — что method-C заменил
→ [NB4: Teardown](./nb4-pipeline-teardown-skeleton.md) — инженерное суждение
