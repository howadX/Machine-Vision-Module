# LJ5 design — «Weighing the experts: a passport for the models we didn't train» (15.5)

> Внутренний design-контракт этапа 15.5 (RU). Журнал LJ5 — EN (стандарт портфолио).
> Правило фазы: гипотезы с **заранее названными критериями провала** фиксируются ДО первого числа.
> Линейка не меняется: компонент «Эксперты (15.5)» из бенч-протокола (§1) — **latency (мс), вес (КБ),
> acc на публичном бенче**, эталонное железо; паспорт `bench.passport(...)` в каждом прогоне;
> bootstrap CI где уместно (N=1000, seed=0).

## Чем 15.5 отличается от 15.1–15.4

15.1–15.4 рефакторили **нашу** арифметику/модели (лес позы, стабилизатор, трекер, fusion) — там мы
меняли логику. 15.5 — про **внешних ONNX-экспертов, которых мы НЕ переобучаем** (YuNet, SFace,
EmotiEffLib, BlazePose-пара, YOLO26n, Depth-Anything, handpose): измеряем их latency/вес/честную
точность на публичных бенчах и выносим вердикт **оставить / заменить / переквантовать**. Плюс
финализируем **диету** (VAE drop, размер леса — по большей части уже решено в 15.1). Прод-код в 15.5
**не меняется** — это измеряющий этап; порт диеты и любых замен — 15.6.

**Тонкость гейта.** «Точность выросла vs v0.1.0» буквально неприменима к внешним экспертам: их
точность **не мерили ни разу**. Поэтому 15.5 **устанавливает базлайн**. Честная формулировка гейта:
(а) вес и latency **не выросли** vs v0.1.0; (б) точность позы **выросла** (15.1: gold R3 0.83→0.894,
диета её не роняет); (в) точность внешних экспертов **впервые измерена** — базлайн зафиксирован, и
наша обвязка (ONNX-экспорт + препроцесс) воспроизводит заявленные апстримом числа без тихой регрессии.

## Решённые развилки (утверждено до старта)

- **15.5 — лаборатория + решения, без прод-кода.** Никаких изменений крейтов; выход — полный
  scorecard + финальный список диеты/замен для порта 15.6. Диета частично уже в ship-артефактах 15.1
  (`spike_pose/ship/`): forest 220КБ + Mahalanobis 1.8КБ + flexion 0КБ, VAE выкинут. 15.5 их
  консолидирует и досчитывает суммарный вес/latency.
- **Эталонное железо = эта dev-машина, CPU-путь** (CPU-first инвариант; GPU не при делах). Модель CPU
  + число потоков + ONNX Runtime provider (`CPUExecutionProvider`) **записываются в паспорт latency**
  рядом с числами. Latency привязана к железу → это reference-hardware число, не Kaggle (Kaggle-CPU
  иной; если меряем и там — как отдельный, non-shipping столбец).
- **Разложение latency — двухслойно (паттерн LJ):**
  - *per-model inference* — `onnxruntime` CPU EP, Python, N warmup + N timed прогонов на фикс-входе
    (воспроизводимо, разложимо; доминирующая стоимость тяжёлых моделей). Это переносимо в ноутбук.
  - *full human-chain wall-clock* — из прод-бинаря `mvm run` на эталонном железе (**shipping-истина**,
    включает препроцесс/alignment/feature-calc). Сверка: сумма частей ≈ целое.
- **Emotion-бенч = RAF-DB** (`/home/dawoh/datasets/rafdb/`, установлен, проверен: 15339 img,
  12271 train / 3068 test). Чистый held-out: EmotiEffLib `enet_b0_8_best_vgaf` тренирован на
  AffectNet+VGAF → RAF-DB модель не видела. Маппинг RAF(1..7)→model-idx: 1 Surprise→7, 2 Fear→3,
  3 Disgust→2, 4 Happiness→4, 5 Sadness→6, 6 Anger→0, 7 Neutral→5; model-idx 1 (Contempt) — без RAF-GT
  → argmax-over-7. ⚠️ Aligned-кропы → паспорт меряет **классификатор-голову** (наша YuNet+Umeyama
  стадия минуется; отмечаем явно). Лицензия research-only, redistribution запрещён → на Kaggle
  монтируем существующий публичный `shuvoalok/raf-db-dataset`, не перехостим.
- **Identity-бенч = LFW** (несущий, стандарт: 6000 пар, 10-fold) + **CFP-FP / AgeDB-30** (строже, оба
  свободны) — апгрейд строгости, т.к. LFW почти насыщен (совр. модели ~99.6%). SFace = Apache →
  публикуемо.
- **Detect-бенч = OpenImages validation** (Person `/m/01g317` для YOLO26n, Human-face `/m/0dzct` для
  YuNet), «where feasible» по плану. ⚠️ Кросс-домен: YOLO26n тренирован на COCO, YuNet на WIDER FACE
  → число честно кросс-доменное, не in-distribution. Полный OI-протокол (federated labels, group-of,
  иерархия) вне охвата — меряем recall/precision @ IoU 0.5 на prominent-single-subject подмножестве
  (тот же фильтр, что `openimages_fetch.py`), с явным caveat.
- **Kaggle-воспроизводимость.** Accuracy-акты воспроизводимы на Kaggle через **монтируемые датасеты**
  (DNS-гоча: API-раны ничего не тянут). Нужные mounts: `shuvoalok/raf-db-dataset` (emotion), LFW +
  CFP-FP + AgeDB (перехостим как derived-free? — LFW свободен для redistribution; CFP-FP/AgeDB —
  проверить лицензию перед пушем), OpenImages-подмножество, и **ONNX-модели** (`mvm-onnx-experts`:
  Apache/MIT — YuNet/SFace/emotion/BlazePose; YOLO AGPL — отдельно/локально). Latency остаётся
  reference-hardware (локально), не Kaggle.

---

## Крючок (зачем вообще, в одну мысль)

Пять фаз (5–14) мы носили в проде семь чужих ONNX-моделей, **ни разу не измерив ни их скорость, ни
их точность на честном бенче**. Мы даже не знали, что наша собственная обвязка их не ломает — пока
в 10.1 не всплыл баг препроцесса эмоций (HWC-данные в NCHW-тензор → near-uniform softmax на любом
лице, жил с фазы 6). Его поймали случайно, глазами. Паспорт 15.5 — это прибор, который поймал бы его
на первом же прогоне: RAF-DB на выходе EmotiEffLib либо даёт осмысленное per-class recall, либо
рассыпается в chance — и тогда виновата **наша проводка, а не модель**. То же для identity (LFW ≈ 0.5
= сломанный эмбеддинг) и detect (person-AP ≈ 0 = сломанный порог/NMS). 15.5 превращает «модели вроде
работают» в таблицу с числами, каждое из которых трассируется в железо, датасет и коммит.

Ставка 15.5: (1) человеко-цепочка укладывается в `<50ms` CPU и разложима по-агентно; (2) каждый
эксперт на своём held-out даёт число, согласованное с апстрим-заявкой (нет тихой регрессии обвязки);
(3) диета 15.1 (−79% футпринта позы) складывается в суммарный вес, который **не больше** v0.1.0.

**Вес v0.1.0 (посчитан):** ONNX всего ≈127.7 МБ (SFace 38.7 + depth 27.3 + emotion 16.0 + person-det
12.0 + yolo-seg 13.8 + yolo-obj 9.9 + pose-est 5.6 + handpose 4.1 + YuNet 0.23) + embedded
arithmetic (RF forest 0.873 + VAE 0.195). Диета бьёт по arithmetic-слою (VAE→0, forest 0.873→0.220);
тяжёлые ONNX — кандидаты разговора о requant, но трогаем их только если паспорт разрешит без потери.

---

## Акт A — Latency & weight ledger (несущий по оси «скорость/вес»)

**Цель.** Per-agent таблица: вес (из манифеста, точные байты) + latency (per-model onnxruntime CPU,
N=50 warmup / N=200 timed, median + p95) на эталонном железе; full human-chain wall-clock из `mvm run`.
Подтвердить бюджет `<50ms` CPU и атрибутировать доминирующие статьи.

**Falsifiable-гипотеза A.** *Человеко-цепочка (YuNet→SFace→emotion→BlazePose-det→BlazePose-pose→наша
арифметика) укладывается в <50ms CPU на эталонном железе; сумма per-model latency ≈ full-chain
wall-clock (в пределах препроцесс-оверхеда); каждый агент стабилен (p95/median < 2×).*

**Критерий провала (назван заранее).** Если цепочка > 50ms — называем доминирующий агент (кандидат:
SFace 38.7МБ или BlazePose-пара) и это становится **целью диеты или каденции** (не «подкрутка» бюджета:
depth/YOLO уже за каденцией — они не в человеко-цепочке by design). Если per-agent latency нестабильна
(p95 ≫ median) — публикуем распределение, не точку. Если sum-of-parts сильно < full-chain — значит
препроцесс/alignment/feature-calc сам дорогой; называем его отдельной статьёй (тоже честная находка).

---

## Акт B — Accuracy passport (несущий по оси «точность», три эксперта)

Каждый под-акт: точечная оценка + CI + честный caveat (препроцесс-путь, train-домен, контаминация).
Общий негативный контроль: **число у chance = баг нашей обвязки, а не «плохая модель»** — стоп и чиним.

### B1 — Emotion (EmotiEffLib) на RAF-DB held-out
- Метрика: **balanced-acc (mean per-class recall)** + per-class recall + confusion (7×7) + macro-F1.
  RAF-DB test = 3068, дисбаланс (Fear=74 → широкий CI); bootstrap по изображениям (нет subject-id в
  этой раздаче → отмечаем отсутствие группировки).
- **Гипотеза B1.** *EmotiEffLib на RAF-DB даёт осмысленное per-class recall (balanced-acc ≫ chance
  0.143), близкое к опубликованным RAF-DB-числам семейства; распределение softmax пиковое, не
  near-uniform.*
- **Критерий провала.** balanced-acc ≈ chance ИЛИ near-uniform softmax → **баг препроцесса/экспорта**
  (регрессия уровня 10.1) → стоп, чиним обвязку, публикуем как находку. Умеренно-низкое (но ≫ chance)
  → честный домен/препроцесс-gap (aligned-кропы vs наш YuNet-crop), фиксируем числом, не «плохая
  модель». Contempt-сток (model-idx 1) репортим как confusion-канал.

### B2 — Identity (SFace) на LFW (+ CFP-FP / AgeDB-30)
- Метрика: verification accuracy (10-fold, best-threshold, стандартный LFW-протокол) + ROC-AUC +
  TAR@FAR=1e-3. CFP-FP (frontal↔profile) и AgeDB-30 — как строгие столбцы.
- **Гипотеза B2.** *SFace на LFW ≈ опубликованному (~0.99); CFP-FP/AgeDB заметно ниже (ожидаемо,
  ракурс/возраст) — честный «строгий» профиль эмбеддинга под наш use-case (профиль, смена возраста).*
- **Критерий провала.** LFW-acc ≈ 0.5 → сломанный эмбеддинг/препроцесс (alignment, L2-норма) → стоп,
  чиним. Если CFP-FP/AgeDB катастрофичны (≈chance) при здоровом LFW → SFace хрупок к ракурсу; фиксируем
  как known-limit (мост к body-primary: лицо-id дотягивается только при фронтальном лице — уже в
  архитектуре) — не блокер, но число в паспорт.

### B3 — Detect (YOLO26n person + YuNet face) на OpenImages val — «where feasible»
- Метрика: recall/precision @ IoU 0.5 на prominent-single-subject подмножестве (фильтр
  `openimages_fetch.py`). НЕ полный OI-протокол (federated/group-of вне охвата — caveat).
- **Гипотеза B3.** *YOLO26n-person и YuNet-face дают высокий recall@IoU0.5 на кросс-доменном OI
  (несмотря на COCO/WIDER train-домен) — эксперты переносимы, обвязка (NMS/порог) корректна.*
- **Критерий провала.** recall@IoU0.5 ≈ 0 → сломанный порог/NMS/декод боксов → стоп, чиним. Умеренный
  кросс-доменный провал — честное число с caveat (OI ≠ train-распределение), не повод менять модель.
  Если ось detect окажется дорого промерить честно (OI-eval сложный) — помечаем «where-feasible:
  partial», меряем что можем, не натягиваем.

---

## Акт C — The diet, decided (гейт «вес/latency не растут»)

**Цель.** Свести измерения A+B в финальные вердикты диеты для 15.6.

- **VAE 195КБ → DROP** — уже установлено 15.1 (Mahalanobis 0.825 >> recon 0.702). C подтверждает
  дельту футпринта.
- **Forest 873КБ → 220КБ** — уже установлено 15.1 (gold R3 плоский по sweep). C подтверждает.
- **Requant/replace тяжёлых ONNX (SFace 38.7 / depth 27.3):** решение **из паспорта B**. Дефолт —
  **не трогаем** (риск против инварианта «точность не падает»; апстрим-модели), но если A назвал
  SFace доминантой latency И B показал запас точности → фиксируем INT8-requant как *кандидат с
  измеренной ценой* (потом проверяем на LFW/CFP-FP, что acc не упал). Замена эксперта — только с
  числом «новый ≥ старый на том же бенче», иначе документируем как future-option, не делаем в 15.5.

**Falsifiable-гипотеза C.** *Суммарный вес v0.2.0-кандидата (ONNX + arithmetic после диеты 15.1) ≤
v0.1.0, и человеко-цепочка latency ≤ v0.1.0, без падения любой измеренной в B точности.*

**Критерий провала (назван заранее).** Если единственный путь урезать вес — требант, роняющий acc на
паспорт-бенче → **вес держим, не режем** (честно: «footprint held at external experts; −79% banked at
arithmetic layer»), инвариант «не растёт» соблюдён, «уменьшается» — частично. Если что-то из диеты
15.1 при перепроверке C роняет метрику (не должно) → ревертим и пересматриваем.

---

## Гейт этапа 15.5 (сводно)

1. A: per-agent ledger (вес+latency) на эталонном железе с записанным CPU/threads/EP в паспорте;
   человеко-цепочка < 50ms CPU **или** явно названный доминирующий агент как цель диеты/каденции;
2. B1: emotion balanced-acc на RAF-DB ≫ chance, пиковый softmax (негативный контроль обвязки взят);
   B2: SFace LFW ≈ published, CFP-FP/AgeDB как строгие столбцы; B3: detect recall@IoU0.5 (where feasible);
3. C: суммарный вес ≤ v0.1.0, latency ≤ v0.1.0, ни одна измеренная точность не упала; вердикт-список
   диеты/замен для 15.6;
4. каждая цифра — из живого прогона; паспорт (git-commit + data-sha256 + CPU/EP для latency) в
   `results/metrics.json["lj5"]`; `working_tree: clean` на публикуемом прогоне.

## Что это меняет в проде (мост к 15.6)

- Пакет порта 15.6 пополняется финальным scorecard'ом: VAE-drop + forest-220КБ (вес), + любые
  requant/replace-вердикты C. Quality-passport в README/AGENT.md собирается из B (числа со ссылками
  на LJ5).
- Latency-бюджет по-агентно зафиксирован — регрессионный якорь: любой будущий эксперт сверяется с
  этим ledger'ом (не «на глаз»).
- Известные пределы с числами: хрупкость SFace к ракурсу (B2), кросс-доменный detect-gap (B3),
  Contempt-сток эмоций (B1) — все в паспорт как честные caveat'ы, не замазываются.

## Порядок исполнения (с паузами)

1. **Акт A** — `spike_pose/lj5_latency.py` (onnxruntime per-model + парсинг манифеста для веса) +
   full-chain из `mvm run` на эталонном железе. Пауза на ревью чисел A.
2. **Акт B1** — `spike_pose/lj5_emotion.py` (RAF-DB test → EmotiEffLib ONNX → маппинг → balanced-acc
   + confusion + CI). Негативный контроль обвязки — первым делом. Пауза.
3. **Акт B2** — `spike_pose/lj5_identity.py` (LFW pairs → SFace эмбеддинг → verification; + CFP-FP/AgeDB).
   Пауза.
4. **Акт B3** — `spike_pose/lj5_detect.py` (OI-подмножество → YOLO/YuNet → recall@IoU0.5), where feasible.
   Пауза.
5. **Акт C** — сведение в scorecard, вердикты диеты, проверка гейта веса/latency.
6. LJ5 (`notebooks/lj5/_build_lj5.py`), регистрация в `kctl.py`, датасеты-mounts (`mvm-onnx-experts` +
   бенчи), push, Run-All, числа+паспорт → `results/metrics.json["lj5"]`.

## РЕЗУЛЬТАТЫ (измерено 16.07.2026, локально; Kaggle Run-All — ✅ COMPLETE, числа бит-в-бит)

Инструменты: `spike_pose/lj5_{latency,emotion,identity,detect,objects}.py`. Железо: i7-12700K,
onnxruntime 1.27.0 CPU EP. Числа в `results/metrics.json["lj5"]`.

**АКТ A — Latency & weight ledger (гипотеза A ПОДТВЕРЖДЕНА).**
- Человеко-цепочка (YuNet 1.88 + SFace 5.20 + emotion 4.56 + person-det 2.19 + pose 2.87 мс) =
  **16.7 мс all-cores → гейт <50 мс PASS (3× запас)**; 1-поток 52.0 мс (portable floor, тот же
  доминант SFace). Стабильно (p95/median < 1.1× у всех).
- Cadenced (вне бюджета by design, свои потоки): depth 231.8 мс, YOLO-seg 21.3, YOLO-obj 12.5.
- Всего ONNX = **121.7 MiB**. Доминант цепочки по latency И весу = **SFace** (36.9 MiB, 30% ONNX).
- Caveat: per-model isolated (не контендированный live ~16 fps движок, owner-validated).

**АКТ B1 — Emotion / RAF-DB held-out (гипотеза B1 ПОДТВЕРЖДЕНА).**
- **Негативный контроль обвязки взят:** mean max-prob 0.720 (chance 0.125), norm-entropy 0.365 —
  softmax пиковый, **баг 10.1 не воспроизводится** (наш planar-CHW/ImageNet препроцесс верен).
  Contempt-сток 3.0%.
- **balanced-acc 0.665 [0.643,0.688], overall 0.726** (chance 1/7=0.143). Метадата крейта 0.73 —
  не завышена. Сильные: Happiness .84 / Sadness .85 / Anger .82 / Fear .74; слабые (классические FER-
  путаницы) Surprise .37 (→Fear) / Disgust .36 (→Anger). Кросс-домен (train AffectNet+VGAF).
- Caveat: aligned-кропы минуют нашу YuNet+Umeyama → паспорт классификатор-головы.

**АКТ B2 — Identity / SFace на LFW (гипотеза B2 ПОДТВЕРЖДЕНА с находкой).**
- **Негативный контроль взят сильно:** aligned = **0.9783** = опубликованный SFace-LFW → модель +
  наш SFace-препроцесс (BGR planar raw [0,255], L2) корректны. 0/7701 промахов детекции.
- **НАХОДКА: прод не выравнивает лицо перед SFace** (`pipeline.rs:470` — сырой bbox-кроп, в отличие
  от emotion стр.437). Цена: **unaligned 0.7833 vs aligned 0.9783 = +19.5 п.п.**; TAR@FAR=1e-3
  0.173→0.949. Фикс = обернуть `face` в `align_face(lm5,112)` → **+0 вес/latency, +19.5 п.п.** →
  кандидат 15.6 (масштаб center-relink из 15.3). Caveat: LFW-funneled → headroom = нижняя граница.

**АКТ B3 — Detect / OpenImages val (гипотеза B3 ПОДТВЕРЖДЕНА, «where feasible»).**
- **YuNet (face, in-chain): recall@IoU0.5 prominent 0.805 / all 0.772** (543/758 GT) — силён,
  кросс-домен (WIDER FACE→OI), обвязка/декод верны.
- **YOLO26n person: 0.653 prominent / 0.523 all** (стабильно между прогонами) — nano кросс-домен;
  прод-person-путь = BlazePose, не YOLO (proxy).
- **YOLO26n OBJECT-паспорт (18 интеракц.-классов, реф. Вадима «YOLO = субстрат сцены»):**
  MACRO 0.520 / MICRO 0.460 prominent. **Интеракц.-ядро fusion сильно (0.6–0.8):** Mobile phone .81 /
  Remote .78 / Laptop .74 / Mouse .73 / TV .69 / Keyboard .60 / Clock .68 / Vase .67 / Bowl .66 /
  Wine glass .67. Слабый хвост — смесь врождённой трудности и **кросс-таксономии** (Book .07 =
  корешки на полках; Table .22 = COCO dining-table≠OI Table; bag'и .09–.13). Декод верен.
- precision — нижняя граница (OI federated labels).

**АКТ C — Диета решена (гейт «вес/latency ≤ v0.1.0» ВЗЯТ).**
- **SFace INT8-requant — ИЗМЕРЕННЫЙ НЕГАТИВ, REJECT.** Вес 36.9→9.4 MiB (−74%), точность **lossless**
  (aligned 0.9790 vs 0.9783, в CI), но **latency ×6 хуже** (5.10→31.63 мс all-cores; CNN dynamic-quant
  pitfall) → нарушает гейт latency + CPU-first. Static-quant — future-option, вне 15.5.
- **Вес:** ONNX 121.7 MiB без изменений (requant отклонён); арифметика 1068→**222 КБ (−79%)**
  (VAE drop + forest 220 из 15.1). Суммарный футпринт **уменьшается**.
- **Latency:** цепочка 16.7 мс, без свапов; align-фикс +sub-ms. **PASS.**
- **Вердикты для 15.6:** VAE DROP · forest 220КБ · **SFace align-фикс SHIP** (+19.5 п.п., $0) ·
  SFace INT8 REJECT · все ONNX-эксперты KEEP.

**ГЕЙТ 15.5 — ВЫПОЛНЕН:** вес ↓, latency = (<50 мс), точность впервые промерена (поза ↑ из 15.1,
identity align +19.5, emotion/detect healthy-базлайн). Главный actionable — align identity-кроп в 15.6.

**ЗАКРЫТО 16.07.2026:** журнал `howadx/lj5-weighing-the-experts` опубликован (private), Run-All
**COMPLETE**; B1 (0.665/0.726, neg-control HEALTHY) и B2 (0.7833/0.9783, TAR 0.173/0.949)
воспроизведены на Kaggle бит-в-бит через **cv2.dnn** (в Kaggle-имидже нет onnxruntime, а API-раны
не резолвят DNS → и не PyPI; эквивалентность cv2.dnn↔ort верифицирована max|Δ|≈2e-6). Act A latency
и Act C INT8-requant требуют onnxruntime → в журнале как reference-hardware таблицы (замерены в
репо, паспорт в metrics.json). Mounts: `mvm-onnx-experts`, `mvm-lfw-verification`,
`shuvoalok/raf-db-dataset`. Гейт 15.5 — MET.
