# Фаза 11 — Fusion-полировка

> **Версия:** 1.1 (3 июля 2026) — выделено по **модульной границе**: Фаза 10 = **сбор данных**
> (все агенты парсят все сигналы), Фаза 11 = **fusion над готовыми данными** (реляционно-темпоральная
> арифметика). **Продукт (упаковка) вынесен в следующую, пока не определённую фазу** — к продукту рано,
> фазы движутся вперёд по fusion. **Предпосылка:** фаза 10 закрыта, SUPREME-валидирована.
> **+0 ONNX** для всего fusion-ядра (масок из фазы 10 достаточно). **Исследовательская база:**
> `~/wiki/machine-vision/{03-architecture-principles,05-predictive-state-modeling,07-fusion-quality-analysis}.md`,
> `~/wiki/mvm/subsystems/{scene-graph,detectors-research}.md`.

Канон текущего состояния — `mvm/CLAUDE.md`. Здесь — *как делаем* фазу 11.

> **Прогресс (2026-07-04, 501 тест зелёный, LIVE-VALIDATED на риге):** ✅ **A1** mask-контакт · ✅ **A2** фикс
> мис-атрибуции · ✅ **C1–C3, C5, C6** activity-FSM + event-journal + контракт + churn-стичинг (**C6 обобщён на
> object-churn, включён на mesh**) · ✅ **B1** зоны-persistence · ✅ **B2** anomaly v2 · ✅ **D1** метрич.
> self-calib · ✅ **object-permanence** статичных фикстур (не в исходном плане; вскрыт live — клавиатура под
> печатающими руками терялась → typing пропадал; `STATIC_CLASSES`+`static_hold`+`remembered`, `20fb4037`).
> **Live-валидация (`--native-camera --masks`, Вадим):** mask-контакт снял ложный `using_phone` (~50с→2с);
> object-permanence провалидирован end-to-end — seed→hold→`typing` 23с (было 0).
>
> **Разворот на mesh-first (2026-07-04):** владелец сделал **mesh главным продакшн-путём**. Seg заведён в
> `AgentMesh` (был только на sync) → A1/A2 mask-контакт + mask-median depth теперь работают в продакшене
> (`SceneAgent` один seg-forward: object-маски `attach_masks`, person-маски в `PersonMaskSlot`). **✅ A3
> person-маски ЗАКРЫТ по коду** (mask-IoU-cue в cost-матрицу трекера, `W_MASK=0.10`, gated ≥2 established-людей;
> `HumanAgent` матчит person-маски к телам bbox-IoU перед `tick_motion`; терм=0 при <2/без масок → байт-идентично
> pre-A3; 256 тестов трекера целы). **✅ LIVE-ВАЛИДИРОВАН (2026-07-04, `mesh --masks`, 2 чел., 52 кадра):** seg
> публикует 2 person-маски/каденс; оба человека держат стабильный id весь сеанс — **аноним id10 = 1 id в 50/50
> кадрах** (раньше анон-ходьба → 5–7 id), 1 транзиент-фантом id12 в окне ~0.5с (кадры 6–19) рассосался и не
> вернулся; **anomaly=0 весь сеанс → сломанный fall-anomaly из pre-A3 НЕ воспроизвёлся**; реконструкция сцены
> когерентна (2 чел. с метрич. 3D, identity/emotion/posture, зона-близости). **Осталось:** **C4** sync-journal →
> **dev-only** (sync не продакшн, не блокер) · auto-`fx` live-wiring (→ продукт). Detector-хвост: YOLO26n не
> берёт тёмную клавиатуру без hands-off seed (recall).

---

## 0. Тезис

Данные готовы (фаза 10). Фаза 11 работает **НАД Scene JSON**, объединяя покадровые сигналы в
**логические связки, обобщения и активности**. Это fusion/consumer-работа, не producer. Смысловой
центр — **конечный автомат активности**: превратить мерцающие покадровые детекции в стабильные
состояния с началом/длительностью/концом (temporal journal).

> **Мандат чистой арифметики** (`detectors-research.md:89-92`, adversarially verified): *«fusion-math и
> scene-graph/HOI — слабее всего доказаны и вероятнее всего требуют GPU/VLM → держать их как lightweight
> pure-arithmetic proximity/zone-логику, а не обученные модели.»* SGG (REACT/RTX3080), HOI, VLM
> (Llama-11B/GPT-4o), нейро-действия (ST-GCN/VideoMAE) — **отвергнуты или отложены за фазу 11**.

### Слоистый fusion-«торт» (`03-architecture-principles.md:105-119`)

```
Слой 0: Temporal Fusion (PersonTracker)   — identity persistence, decay, bbox+3D
Слой 1: Spatial  Fusion (World3D)         — единые метрические XYZ (масштаб/калибровка)
Слой 2: Logical  Fusion (Activity)        — поза + руки + объекты → контекст/активности   ← ЦЕНТР ФАЗЫ 11
Слой 3: Narrative Fusion (LLM)            — описание на естественном языке (внешний, System 2)
```

---

## 1. Backlog по слоям (что объединяем)

| Группа | Слой | Пункты | В scope 11 | Инварианты |
|--------|------|--------|-----------|------------|
| **A. Реляц./атрибуция** | 2 | mask-контакт **10.6**; person-маски **10.7** (только в cost-матрицу 9, не узел); мис-атрибуция контакта («своя» кисть) | ✅ ядро | ✅ +0 ONNX |
| **B. Зоны/комнаты + anomaly v2** | 2 | семантич. зоны над union-find (сейчас плоские компоненты, `NEAR_THRESH_M=1.5`, без persistence); anomaly v2 (prediction-error + plausibility) | ✅ | ✅ если геометрия/арифметика |
| **C. Temporal activity FSM** | 2 | **конечный автомат** активности + event-journal (start/stop/duration) | ✅ ядро | ✅ +0 dep, детерминизм |
| **D. 3D metric-calib** | 1 | оценка `torso_length_m`/`fx` по стабильным наблюдениям (`=0.5`, не откалиброван; абсолют 3D ±) — Спатиальная fusion, точность метрики для активностей/зон | ✅ | ✅ арифметика |
| **E. Predictive-state fusion** | 0 | SG-LKF (`2508.00358`), EDL (`2409.04720`), PDF (`2406.04802`), PKF/Mamba | ❌ research | ⚠️ каждый = +1 ONNX |
| **F. Narrative / MLP** | 3 | Level-2 MLP-fusion (+1 ONNX, датасет); Level-3 LLM-рендер (TinyLLM, недетерминизм) | ❌ внешний | ⚠️ ломает детерминизм/+0-dep |

**Scope фазы 11:** делаем **A, B, C, D** (реляц.-темпоральная арифметика). **E, F** — исследовательский
хвост (обученные модели против +0-dep); их *идеи* (prediction-error→confidence, дебаунснутые сегменты как
датасет) готовят почву, но модели не въезжают.

> **Продукт вынесен в следующую, пока не определённую фазу** (zero-config auto-`fx` UX, drag-and-drop
> галерея, one-click install / 1 бинарник, systemd service + cron state dump). К упаковке рано — сначала
> fusion-ядро (A–D). Калибровочная **арифметика** `torso_length_m`/`fx` (D) остаётся во Фазе 11 как
> Спатиальная fusion; её *zero-config-обёртка* (UX) — уже в продуктовой фазе.

---

## 2. Research: почему конечный автомат (FSM), а не HMM/HSMM/CRF/GCN

**Вывод:** для малого фиксированного набора активностей под $0/CPU/+0-dep/детерминизм — **правила + FSM
(dwell + гистерезис) — правильный минимальный выбор сейчас.** Прямой сигнальный аналог триггера Шмитта /
debounce: не нужен датасет, бит-точная воспроизводимость, ns/кадр, аудируемо. Уже так сделаны
`MotionClassifier`, `PostureStabilizer`, lifecycle FSM.

| Метод | Duration-модель | Датасет | Детерминизм | CPU-realtime | Интерпр. | Вердикт |
|-------|-----------------|---------|-------------|--------------|----------|---------|
| **Правила + FSM (dwell+гистерезис)** | Явная, руками | **Нет** | **Да** | **Да (ns)** | **Да** | ✅ **Сейчас** |
| HMM | Геометрич. (`a_ii^{d-1}`, memoryless) | Да | Нет | Да | Частично | ❌ |
| HSMM / explicit-duration | Явная, **выученная** | Да | Нет | Маргинально | Частично | 🔶 цель на потом |
| CRF (linear-chain) | Геометрич. (semi-CRF чинит) | Да | Нет | Да | Частично | ❌ сейчас |
| Behavior Tree | — (арбитраж, нет dwell) | Нет | Да | Да | Да | ➖ не наша проблема |
| Skeleton-GCN (ST-GCN/CTR-GCN) | Неявная (окно) | Да (NTU) | Нет | Маргинально | Низкая | 🔶 Level-2, gated |

Ссылки (проверены): Rabiner 1989; Yu 2010 AI 174(2); Lafferty 2001 / Vail 2007; Colledanchise & Ögren
`1709.00084`; ST-GCN `1801.07455`, 2s-AGCN `1805.07694`, CTR-GCN `2107.12213`, PoseC3D `2104.13586`.

- **HMM** — dwell только через `a_ii` → геометрические memoryless длительности; нужен обученный A,B. Мимо.
- **HSMM** — чинит явной `p_j(d)`; это *выученная* версия наших ручных dwell-таймеров (миграция FSM→HSMM
  чистая). Цена: датасет, forward-backward `O(T·N²·D)`, недетерминизм.
- **CRF** — дискриминативный, ест коррелированные фичи, но vanilla имеет ту же duration-проблему + разметку.
- **Behavior Tree** ⊃ FSM (Colledanchise & Ögren), выигрывает на масштабе авторинга; у нас ~4 **независимых**
  4-состояночных автомата → blow-up не возникает; BT — арбитраж, dwell/гистерезиса не даёт. FSM — верная высота.
- **Skeleton-GCN** — Level-2 (ест BlazePose-скелеты), но всем нужен NTU-класс датасет → +0-dep нарушен.

### «Poor man's HSMM» (+0-dep, детерминизм)
Свернуть semi-Markov duration-awareness в сам FSM **без обучаемой модели**: **min-dwell** (enter) убивает
1-кадровую массу; **max-dwell cap** — суррогат ограниченной длительности (не застрять); **duration-aware
exit-порог** `θ_lo(d)` — труднее выйти пока `d<d_min`, легче после `d>d_typ` (переход зависит от длительности
= semi-Markov как скалярная арифметика). Константы — из офлайн-гистограмм длительностей (data-informed, не
data-trained). ~80% пользы HSMM за 0% цены.

### Путь миграции (FSM — предпосылка, не тупик)
(1) отгрузить FSM, эмитить `{activity, t_start, t_end, duration, mean_conf}`; (2) лог сегментов = бесплатная
interval-разметка; (3) гистограмма → `p_j(d)` → умные caps назад в FSM; (4) HSMM/semi-CRF когда разметка
есть; (5) CTR-GCN (lean/ONNX) на FSM-авторазмеченных pose-клипах.

---

## 3. Ядро — модель activity-FSM (группа C)

Инстанс активности = ключ `(label, отсортированный set участников)` во времени. Аналог `MotionClassifier`:

```
                 present ≥ ENTER_DWELL подряд
  Candidate ───────────────────────────────────▶ Active   (started_at, first_seen_frame)
     │  absent → reap (не эмитится)                 │  ▲
     │                                       absent │  │ present (absent_streak=0, conf EMA)
     │                                              ▼  │
     │                                           Cooling  (ещё эмитится — grace/гистерезис)
     │            absent ≥ EXIT_DWELL(d) подряд        │   ← duration-aware (poor man's HSMM)
     └──────────────────────────────────────────  ▼
                                                  Ended  (ended_at, duration → journal; reap)
                          d ≥ MAX_DWELL → форс-Ended (bounded-duration cap)
```

Драйвер на тик = `present` = «содержал ли `infer_activities()` кадра детекцию с этим ключом».

```
ACTIVITY_ENTER_DWELL = 3   // ~0.2с @ 15fps; = ID_LOCK_FRAMES/EMOTION_CADENCE; убивает 1-кадровый typing
ACTIVITY_EXIT_DWELL  = 5   // > object_cadence(3): переживает межкаденс-мигание YOLO; sticky-to-persist
ACTIVITY_CONF_LAMBDA = 0.7 // conf = λ·prev + (1−λ)·obs — семейство emotion temporal-PoE λ
ACTIVITY_MAX_DWELL   = <per-label>  // cap длительности
```

### Структуры (`mvm-core/src/activity.rs`, новый — зеркало `motion.rs`/`posture.rs`)

```rust
enum ActivityPhase { Candidate, Active, Cooling, Ended }
struct ActivityInstance { key, phase, enter_streak, absent_streak, started_at, ended_at,
                          first_seen_frame, last_seen, confidence /*EMA*/, peak_confidence }
struct ActivityJournal  { instances: Vec<ActivityInstance>, /* sorted by key → детерминизм */
                          completed: Vec<ActivityEvent> }
// update(&mut self, raw: &[Activity], t_secs, frame_id) → продвинуть FSM, родить кандидатов,
//   слить Ended→completed, reap; no-growth cap (как MOTION_RING_CAP).
// active() → {Active,Cooling};  drain_events() → Vec<ActivityEvent>
```

### Ключевание / идентичность

`ActivityKey=(label, отсортированные u32)` в id-пространстве (reserved 0–9, ephemeral ≥10, объекты ≥10000).
Матч = точное равенство. Reserved-id липкие (live: id0 91–100%), object-id стабильны → инстанс переживает
провалы; cooling мостит окклюзию. Track churn (walking-fragmentation) → set сменился → старый Ends, новый
born (**честно**: реально потеряли человека; сшивать = риск слить двух). `stable_participants:bool` = все
reserved/object. Опц. **11.C6** (за 10.7): сшивка ephemeral-churn на стабильном object-участнике.

### Размещение и оба пути

`mvm-core/src/activity.rs` (нет `ort`/`image`, моделей не зовёт). `infer_activities` остаётся в
`mvm-cli/agents/fusion.rs`; journal — тонкая обёртка над `&[Activity]` → идентична на обоих путях.
- **Mesh:** поле `journal` в `FusionAgent`, апдейт под gate по `world.frame_id`, запись `active()` +
  `drain_events()` в `WorldState` (+новое поле `events`).
- **Sync:** `ActivityJournal` в цикле `main.rs`, драйв **каждый кадр** (не только на `dump_interval`);
  фусион-инпут вынести из `to_json_line` в `build_fusion_inputs(...)`.

### Контракт вывода (без слома схемы)

`Activity`/struct-литералы не трогаем. `JsonActivity` сохраняет `label/confidence/participants`, добавляет
`#[serde(skip_serializing_if="Option::is_none")]` `state`(«active»/«cooling»), `started_at`, `duration_s`.
Новый `events[]` на `JsonFrame` (`skip_serializing_if="Vec::is_empty"`): `JsonEvent{label, participants,
started_at, ended_at, duration_s, peak_confidence, stable_participants}` — запись при завершении инстанса.
Всё аддитивно. Markdown `write`/`write_world` — `label (duration_s)` + строка `Events:`.

---

## 4. Порядок работ (11.x, TDD, smallest-first)

**Жёсткое правило:** фикс мис-атрибуции (11.A2) — **ДО** FSM. FSM стабилизирует; стабилизировать
мис-атрибуцию = уверенное долгое ложное событие.

### Трек A — реляц./атрибуция

| Шаг | Задача | Тест |
|-----|--------|------|
| **11.A1** ✅ `92e804aa` | **mask-контакт 10.6:** `InstanceMask::contains_norm` вместо `point_near_bbox` в `infer_activities` (под `--masks`; иначе fallback bbox) | контакт по маске; фон срезан |
| **11.A2** ✅ `718cdd83` | **фикс мис-атрибуции:** при попадании точек нескольких людей в один объект — атрибуция ближайшему к центру/маске; опц. не считать вытянутую руку без hand-лендмарка (`motion::arm_reach`) | 2 чел.: запястье A тянется к телефону B → атрибуция B |
| **11.A3** ✅ LIVE-ВАЛИДИРОВАН (2 чел.) | **person-маски 10.7 (mesh-first):** seg-в-mesh (`SceneAgent`→`PersonMaskSlot`), `HumanAgent` матчит к телам bbox-IoU; `person`-маска **только в cost-матрицу трекера** (`-(iou+W_OCM·mom+W_MASK·mask_iou)`, `W_MASK=0.10`), не узел, gated ≥2 established-людей | unit: mask-IoU флипает near-tie crossing, что IoU свапает; <2 людей → cue инертен (байт-идентично). **Live 2 чел.:** оба стабильный id весь сеанс, аноним 50/50 кадров, 1 транзиент-фантом ~0.5с, anomaly=0 |

### Трек C — temporal activity FSM (после 11.A2)

| Шаг | Задача | Тест |
|-----|--------|------|
| **11.C1** ✅ `627ba66a` | **ядро FSM** `mvm-core/src/activity.rs`: Journal/Instance/Phase, enter/exit dwell, Candidate→Active→Cooling→Ended, duration, conf-EMA, bounded reaping. Синтетика | 1-спайк не входит; 3 подряд→Active; 1–2-провал переживается; ≥EXIT→Ended с duration; cap |
| **11.C2** ✅ `638add4b` | **ключевание/матч** + poor-man's-HSMM: `ActivityKey`, exact-set, cooling-мост, churn ends+restart, `stable_participants`, `MAX_DWELL` + `θ_lo(d)` | churn завершает+рестартит; reserved-провал мостится; cap форсит Ended |
| **11.C3** ✅ `3aeeb1ca` | **проводка mesh:** `journal` в `FusionAgent`, frame_id-gate, `events` в `WorldState` | mesh-stub: повторяющийся raw→дебаунс + event |
| **11.C4** ⏸ dev-only | **проводка sync:** journal в `main.rs`, драйв каждый кадр; вынести `build_fusion_inputs`. **Пере-классифицирован**: sync больше не продакшн (mesh — главный путь), journal уже живёт на mesh → не блокер | `to_json_line`: 1-кадровый typing не появляется; 3+ появляется со `state`/`duration` |
| **11.C5** ✅ `9c9c4422` | **контракт:** опц. поля `JsonActivity` + `events[]`; assert обратной совместимости | serde round-trip: старые поля идентичны; новые опускаются |
| **11.C6** ✅ `b1eee6a7` (opt-in, off) | ephemeral-churn стичинг на стабильном object-участнике | churn в N кадров на том же object-id → продолжение |

### Трек B — зоны/anomaly (после C)

| Шаг | Задача | Тест |
|-----|--------|------|
| **11.B1** ✅ `c260b4cf` | **зоны с persistence/семантикой:** стабильные zone-id во времени; опц. подпись зоны по доминирующему контексту (workstation/seating) | зона держит id через кадры |
| **11.B2** ✅ `ee036bc5` | **anomaly v2:** prediction-error (motion/posture) + `plausibility` + «нежданный объект» → калиброванный скор | knife+резкое падение позы → скор выше ровной сцены |

### Трек D — метрическая калибровка (после fusion-ядра)

| Шаг | Задача | Тест |
|-----|--------|------|
| **11.D1** ✅ `b1eee6a7` | оценка `torso_length_m`/`fx` по стабильным наблюдениям (арифметика; абсолют 3D перестаёт зависеть от ручного конфига). Zero-config UX-обёртка — в продуктовой фазе | оценённый `fx` даёт метрику ближе к рулетке на калибр-сэмпле |

**Глобальный порядок:** `11.A1 → 11.A2 → 11.C1 → 11.C2 → (11.A3 при ≥2 людях) → 11.C3 → 11.C4 → 11.C5 →
11.B1 → 11.B2 → 11.D1`. Минимум-правило: **11.A2 до любой проводки FSM**.

> **Следующая фаза (не определена) — Продукт.** zero-config auto-`fx` UX, drag-and-drop галерея
> (фото→папка = новый человек), one-click install (1 бинарник, без pip/venv/Python), systemd service +
> cron state dump. Выносится сюда сознательно: к упаковке рано, фазы движутся вперёд по fusion.

---

## 5. Критические файлы

- `mvm-core/src/activity.rs` — **новый**: `ActivityJournal`/`Instance`/`Phase`/`Event`/`Key`.
- `mvm-core/src/world.rs` — `Activity` без изменений; `+ events: Vec<ActivityEvent>` в `WorldState`.
- `mvm-cli/src/agents/fusion.rs` — 11.A1/11.A2 в `infer_activities`; `journal` + frame_id-gate в `FusionAgent`.
- `mvm-cli/src/state.rs` — темпоральные поля `JsonActivity` + `events[]`; драйв journal в `to_json_line`/`to_json_line_world`; зоны с persistence.
- `mvm-cli/src/main.rs` — `ActivityJournal` в sync-цикле, драйв каждый кадр.
- `mvm-scene/src/{lib.rs,mask.rs}` — 11.A3 person-маска в cost; `InstanceMask::contains_norm` для 11.A1.
- `mvm-core/src/{space.rs,tracker.rs}` — 11.D1 auto-`fx`/`torso_length_m`; 11.A3 в cost.
- `mvm-fusion/src/lib.rs` — 11.B1 стабильные zone-id над union-find.
- Доки: `~/wiki/mvm/outputs/scene-json.md`, `~/wiki/mvm/roadmap.md`, `mvm/CLAUDE.md`.

---

## 6. Верификация (end-to-end)

1. **Юнит:** `cargo test --workspace` зелёный после каждого шага (TDD). Ядро FSM — синтетические потоки
   `Vec<Activity>` (спайк/дропаут/дребезг/churn/cap), как `motion.rs`.
2. **Sync live:** `cargo run --release -- --native-camera --masks --dump-interval 3 --dump-json` — сесть
   за стол: `typing`/`seated_at_workstation` с `state:"active"` + растущим `duration_s`; уход → один
   `events[]` с корректной `duration`. Короткое касание (<3 кадров) → активность **не** появляется.
3. **Мис-атрибуция (11.A2):** 2 чел., один тянет руку к телефону соседа → `using_phone` держащему.
4. **Person-маски (11.A3):** 2 чел. на перекрытии → фрагментация ниже baseline фазы 9.
5. **Mesh:** `mesh --dump-json` — активности стабильны; `mesh --stub --max-frames 100` — CI-smoke зелёный.
6. **Обратная совместимость JSON:** старый парсер не падает (прежние ключи на месте, новые опциональны).

---

## 7. Что НЕ делаем

- **Predictive-state (E):** SG-LKF/EDL/PDF — каждый +1 ONNX, вне `mvm-core`. Отложено; идея
  prediction-error→confidence уже частично арифметически есть (PoE-веса, plausibility) и питает anomaly v2.
- **MLP/LLM (F):** MLP = +1 ONNX + датасет; LLM = недетерминизм + 500МБ. MVM = System 1, LLM = внешняя
  System 2 (Hermes). Дебаунснутые сегменты FSM станут датасетом для MLP/HSMM.
- **Нейро-действия (ST-GCN/CTR-GCN/VideoMAE):** отложены за фазу 11 (+1 ONNX); активности = rule+FSM.
- **SGG/HOI/VLM:** отвергнуты ($0/CPU/детерминизм) — `detectors-research.md:163`.
- **Сшивка track-фрагментов внутри FSM** — это 10.7 person-масок (не слить разных людей).
- **Audio/YAMNet:** новая capture-dep против минимализма — отложено.
