# Бенч-протокол фазы 15 (15.0.3) — заморожен

> Версия 1.0 — 15 июля 2026. Единый воспроизводимый протокол измерений для всей серии
> лабораторных журналов LJ0–LJ6. Реализация метрик — `spike_pose/bench.py` (импортируется,
> не копируется). Правило: **ни одно число в журнале не считается вне этого протокола.**

## 0. Зачем заморозка

Фазы 5–14 меряли ML «на глаз» и невоспроизводимо (NB3b вскрыл confound + циркулярный азимут).
Чтобы дуга LJ0→LJ6 была честной и складывалась в статью, все стадии обязаны мерить **одинаково**
и **трассируемо**: тот же сплит, та же метрика, тот же bootstrap, с записью commit-hash кода и
sha256 данных в каждый прогон. Меняется гипотеза — не меняется линейка.

## 1. Реестр метрик (по компонентам)

| Компонент | Ведущая метрика | Доп. метрики | Домены (train→test) |
|---|---|---|---|
| **Поза (15.1)** | balanced-acc (mean per-class recall) | per-class recall, confusion, **azimuth-curve**, OOD-AUROC | lab+gold5 grouped-CV; **gold = held-out (R3)**; mocap = azimuth-бенч |
| **Переходы (15.2)** | flip-rate ↓ vs reaction-latency | transition-matrix, fall-vs-liedown порог (м/с) | mocap-секвенции (истинное время) |
| **Трекер (15.3)** | лишних треков / мин ходьбы | ID-switches, fragmentation | mocap-проекции (истинная идентичность) |
| **Симулятор/FSM (15.4)** | event-precision/recall vs сценарий | anomaly false-alarm rate | golden-сценарии (инъекция) |
| **Эксперты (15.5)** | latency (мс), вес (КБ), acc на публичном бенче | — | публичные бенчи, эталонное железо |

Каждое число несёт **домены train→test** явно (правило честности NB3b). CI — где уместно (см. §3).

## 2. Политика сплитов (ЗАМОРОЖЕНО)

- **Clean core** = `landmarks_v2.npz[src ∈ {lab, gold5}]` (2633 позы). Оценка: `StratifiedGroupKFold(n_splits=5, random_state=0)`, группировка по `grp`. Это базлайн-линейка.
- **Gold = несущий held-out тест (R3).** Человеческие метки, мульти-сорс. Никогда не участвует в обучении того, что на нём отчитывается. Confound-free обобщение raw↔GRIP меряется здесь.
- **Azimuth-бенч** = `mocap_views.npz` `P3D`, переигранный `mocap_to_views.project(P3D, angle)` на `ANGLES=[0,15,30,45,60,90]`. Это **истинное 3D** реальной камеры, а не y-поворот шумного MediaPipe-z (старый тест был циркулярным by construction).
- **Воспроизводимость = фикс.seed + фикс.контент.** Сплиты не сериализуются списками индексов (ломаются при регенерации данных); идентичность гарантируется `random_state=0` + записью sha256 данных в паспорт. Регенерировал данные → sha256 изменился → это видно в журнале.

Baseline на текущем контенте (для сверки регрессий): **clean-core `features_13` RF = 0.955 balanced-acc, bootstrap CI [0.933, 0.971]**, per-class recall standing .95 / sitting .96 / lying .95 (`bench.py` smoke, commit d7560661).

## 3. Bootstrap CI

`bench.bootstrap_ci` — **групповой** bootstrap (ресэмпл ГРУПП с возвращением, N=1000, seed=0), как в NB3b. CI на пересекающихся интервалах = «разница не установлена» (не выдаём за победу). Точечная оценка + [2.5%, 97.5%].

## 4. Паспорт прогона (обязателен в каждом журнале)

`bench.passport([data_files])` возвращает и журнал **печатает**:
```
{code_commit, working_tree: clean|dirty, data_sha256: {file: hash16}}
```
- `working_tree: dirty` в опубликованном журнале = **красный флаг** (числа не привязаны к коммиту). Публикуем только с `clean` (или явно оговариваем).
- `data_sha256` привязывает числа к точному содержимому датасета.
- Паспорт кладётся в `results/metrics.json` под ключом журнала рядом с числами.

## 5. Шаблон журнала (нарратив, EN — стандарт портфолио)

Каждый LJ — глава с конфликтом и развязкой, не дамп метрик. Скелет ячеек:

```
# LJ<N> — <Title>              e.g. "LJ1 — The pose model, rebuilt honestly"

## 1. Context      what stood before us and why it couldn't be trusted (link prior LJ)
## 2. Hypothesis   falsifiable, with the failure criterion named UP FRONT
## 3. Experiment   setup, DATA (train→test domains!), protocol (cite this bench-protocol)
## 4. Numbers      live runs only, with CI; minimal tables, prose the takeaways
## 5. Verdict      confirmed / refuted / inconclusive — incl. what went wrong
## 6. What it changes   the production decision + bridge to the next journal

<last cell prints bench.passport(...) — code commit + data sha256>
```

Правило на всю серию: **пиши так, чтобы главы Habr-статьи собирались монтажом**, а не переписыванием.
Отрицательный результат публикуется наравне с положительным.

## 6. Что уже заморожено этим документом

- Метрики → `spike_pose/bench.py` (`balanced_accuracy`, `per_class_recall`, `grouped_cv`,
  `bootstrap_ci`, `azimuth_curve`, `confusion`, `passport`; `ANGLES`, `SEED=0`, `N_SPLITS=5`, `N_BOOT=1000`).
- Сплит-политика (§2), CI-метод (§3), паспорт (§4), шаблон журнала (§5).
- Азимут-бенч на истинном 3D (`mocap_to_views.project`), заменяющий циркулярный тест NB3b.
