# Фаза 12 — LLM-render: MVM как модуль зрения для LLM

> **Версия:** 1.0 (5 июля 2026). **Предпосылка:** фаза 11 (fusion-ядро) закрыта или в хвостах —
> activity-FSM + event-journal, stable_zones, anomaly v2, object-permanence уже живут на mesh.
> **+0 ONNX, +0 deps** — вся фаза = детерминированная сериализация и арифметика над уже готовыми
> данными. **Ключевое различение:** это рендер **ДЛЯ** LLM (детерминированный, тестируемый Rust),
> а не рендер **СИЛАМИ** LLM (Level-3 из фазы 11 остаётся отложенным — недетерминизм в ядро не въезжает).
> **Исследовательская база:** `~/wiki/machine-vision/09-scene-compiler/00-scene-compiler-research.md`
> (40+ источников: KeySG, EmbodiedRAG, SSG, UQ-survey), эталон ручной реконструкции —
> `~/wiki/mvm/live-2026-07-04-a3-2person.md`.

Канон текущего состояния — `mvm/CLAUDE.md`. Здесь — *как делаем* фазу 12.

---

## 0. Тезис

**Цель продукта:** LLM вызывает MVM как модуль зрения и из его данных **дёшево и быстро** делает
подробную реконструкцию сцены. Сегодня это невозможно экономически: один тик Scene JSON ≈ 3.3 КБ
≈ **~1000 токенов** (замер: A3-сессия, 52 дампа @1с = 174 КБ ≈ 50k токенов за 50 секунд; минута
потока ≈ 60k токенов, час ≈ 3.6M). Проблема не «LLM не понимает JSON» — объём в принципе непередаваем.

**Ключевой инсайт:** research-док scene-compiler предлагает трёхуровневый пайплайн
(keyframes → temporal aggregation → LLM serializer), но **уровень temporal aggregation фаза 11 уже
построила внутри MVM** — `ActivityJournal` (start/stop/duration), `events[]`, `ZoneTracker`
(stable_zones), anomaly v2, object-permanence, churn-стичинг. Это чистая арифметика в `mvm-core`,
дублировать её снаружи (Python/LLM) — ошибка. Недостающее звено ровно одно: **рендерер + интерфейс
подачи**. Слоистый торт из фазы 11 получает завершение:

```
Слой 0: Temporal Fusion (PersonTracker)   — identity persistence         ✅ фазы 6–9
Слой 1: Spatial  Fusion (World3D)         — метрические XYZ              ✅ фазы 6, 11.D
Слой 2: Logical  Fusion (Activity/Zones)  — FSM, journal, зоны, anomaly  ✅ фаза 11
Слой 3: Narrative Fusion                  — подача наружу                ← ФАЗА 12 (детерминир. рендер),
                                                                            LLM-нарратив поверх — внешний
```

**Модель подачи — pull, не push.** LLM не получает поток; она запрашивает три уровня зума,
каждый следующий на порядок дороже и вызывается только при необходимости:

| Уровень | Вопрос | Источник | Бюджет | Сжатие vs сырьё |
|---------|--------|----------|--------|-----------------|
| **1. snapshot** | «что сейчас» | последний `WorldState` | ~150–300 ток. | ~10× на тик |
| **2. recap** | «что произошло за период» | event-journal + scene-log | ~300–800 ток. | ~100–150× |
| **3. inspect** | «покажи сырьё вокруг X» | scene-log, полная точность | по требованию, узко | 1× но редко |

Реконструкция сцены (уровня рукописной `live-2026-07-04-a3-2person.md`) = LLM собирает её из
`snapshot + recap + 1–2 inspect` за ≤2k токенов входа вместо 50k.

---

## 1. Research-опора (что уже доказано, берём как правила)

Из `00-scene-compiler-research.md` (§4–5), проверено на литературе 2023–2026:

1. **Markdown + таблицы** — LLM читают структуру лучше глубокого JSON; entity-таблица + event-список
   = оптимум (SayPlan, CLEAR, KeySG).
2. **Event-centric верхний уровень** — внимание LLM на «что происходит», не на «поля сущностей».
3. **Confidence словами, число в скобках** — `neutral (high, 0.89)`: LLM калибруются по словам
   значительно лучше, чем по голым float (UQ Survey 2025). Тиры: `high ≥0.75`, `med ≥0.5`, `low <0.5`.
4. **Uncertainty — явными нотами, не молчанием.** Фаза 11 уже честна (`stable_participants=false`,
   `withheld`, `remembered`, `face_verified`, `plausibility`) — рендер обязан превращать это в строки
   вида `⚠️ id12: вероятный фантом (родился lying в 0.25м от Вадима, жил 13с)`. A3-сессия показала цену
   молчания: фантом id12 без такой ноты читается как третий человек.
5. **Иерархическая подача** (summary → details по запросу) бьёт плоский поток — наша pull-модель.
6. **RAG/embeddings НЕ делаем** — retrieval по истории = работа GBrain (brain/vision/), не MVM.

---

## 2. Диагноз: куда уходят токены в Scene JSON (замер на A3-дампе)

**Замер на A3-дампе (52 тика, 173948 B, ~43k токенов/сессия):**

| Источник веса | Пример | Выигрыш (замер) | Лечение |
|---------------|--------|-----------------|---------|
| Полная точность f32 | `"fps":17.99596292964407`, `"score":0.83020747` | **−15.1%** (→147.6 КБ) | 12.A1 округление |
| `emotion_distribution[8]` | 8 float × человек × тик (16 near-zero чисел на 2 чел.) | **+4.1%** (→19.2%) | 12.A2 opt-in |
| `relations` O(n²) | 7 рёбер на 5 сущностей, из них 4 — object↔object `near` (статика стола) | **+4.9%** (→24.1%) | 12.A3 диета графа |
| Повтор статики | mouse/bottle/book с теми же pos 50 тиков подряд | смысловой 0 после 1-го тика | recap-дельты (12.C) |
| Сырые координаты | `wrists`, `pos`, `bbox` покомпонентно | остаётся | в JSON (робот/Grafana); в markdown-рендер не идут |

**Важная поправка к первичной оценке:** трек A даёт ~**24%** суммарно (A1+A2+A3), а не 35–45% —
ключи и структура JSON доминируют над float-шумом, и сырой тик остаётся ~2.5–2.8 КБ. **Порядковый
выигрыш (100–150×) даёт НЕ диета тика, а уровни 2/1 (recap/snapshot).** Трек A — дешёвая гигиена
контракта (−¼ объёма даром, ноль риска), но не главный рычаг; рычаг — перейти от «поток тиков» к
«snapshot + recap».

Scene JSON **остаётся стабильным машинным контрактом** (Grafana/робот/дебаг) — фаза 12 его не ломает,
только (а) худеет по значениям, (б) добавляет opt-in/opt-out флаги, (в) строит **новые** артефакты рядом.

---

## 3. Целевые артефакты и контракты

### 3.1 Snapshot v2 (`current_state.md`, перезапись каждый тик)

Рендер последнего `WorldState` → markdown. Из реального тика A3:

```markdown
# Scene 17:51:19 · occupied · anomaly 0.0 · 18 fps
Zone «workstation»: Вадим(0), аноним(10) + mouse, bottle, book

| id | who         | dist | posture  | motion         | emotion          | head   | hands |
|----|-------------|------|----------|----------------|------------------|--------|-------|
| 0  | Вадим ✓     | 1.0m | standing | moving 0.3 t/s | surprise (med)   | camera | open+fist |
| 10 | anon ✓face  | 1.1m | standing | moving 0.5 t/s | neutral (high)   | camera | point |

Active: people_together(0,10) — 0.6s (conf med)
```

Правила: зоны — из `stable_zones` (label+members, имена вместо id где есть identity); строка
сущности — только выводы (dist, posture, motion, emotion tiered, head_facing, gestures), **без**
сырых координат/bbox/distribution; активности — с `state`/`duration_s`; ноты честности отдельным
блоком `⚠️` (см. §1.4): `withheld` (Upper coverage), `remembered` (объект под окклюзией),
`plausibility != normal`, не-`face_verified` тело, `stable_participants=false`. Бюджет ≤300 токенов
при 2–3 людях. Это v2 существующего StateFile (`state.rs::write_world`) — потребитель Sight-профиля
обновляется вместе (наш).

### 3.2 Recap (журнал периода → markdown, ~300–800 токенов на минуты/часы)

Строится **из событий и дельт, не из тиков**. Три чисто арифметических агрегата:

```markdown
# Recap 17:51:18–17:52:09 (51s, 52 тика, anomaly max 0.0)

## Люди
- **Вадим (id0)** — весь сеанс (100% тиков), зона workstation, 0.8–1.9m.
  Поза: standing 8s → sitting 34s → standing 9s.
- **Аноним (id10)** — весь сеанс, лицо есть (не в галерее), стоит справа ~1.1m, преим. idle.
- ⚠️ **id12 — вероятный фантом**: жил 13s (17:51:23–36), родился в 0.25m от Вадима сразу «lying»,
  исчез без выхода из кадра; дал 4 эпизода people_together (все stable=false).

## Активности (эпизоды из journal)
- people_together(Вадим, id10) — непрерывно 22.5s (peak conf 0.51)
- seated_at_workstation(Вадим) — 12.6s (0.91) · using_mouse — до 8s · holding_drink · reading

## Объекты и сцена
Workstation: mouse, bottle, book — весь сеанс, без появлений/исчезновений. Фон справа: bed, remote.
```

Составные части:
- **Presence-таймлайн** per person: run-length-encoding по тикам (`first..last ts`, % присутствия,
  RLE позы, доминирующая зона, диапазон dist, motion-сводка).
- **Эпизоды активностей**: `events[]` journal как есть (label, participants→имена, duration, peak_conf)
  + активные незакрытые на конец периода.
- **Дельты сцены**: объект appeared/disappeared/`remembered`-интервалы, зона born/died, пики anomaly
  (ts + слагаемые noisy-OR).
- **Фантом-эвристика** (арифметика, не suppress — только маркировка): ephemeral id (≥10), время жизни
  < порога, рождение в <0.5м от существующего человека, `lying` с первого тика, смерть без выхода к
  краю кадра → нота `⚠️ вероятный фантом`. Данные не выкидываются — LLM решает сама.

### 3.3 Inspect (узкий полноточный срез, по требованию)

`± N секунд вокруг ts` или `история track-id` из scene-log — полные тики (включая distribution,
wrists, bbox), JSON или markdown. Дорого, но редко и узко.

### 3.4 Интерфейс для LLM (три ступени зрелости)

1. **Файлы** (уже сегодня): `current_state.md` (v2) + `journal.md` (rolling-recap за сегодня) в
   Sight-workspace — читает любой агент. Это же вход для «Sight Aggregator → brain/vision/daily/»
   из плана GBrain (Фаза 5 родительского проекта).
2. **CLI как инструмент**: `mvm-cli recap|inspect` — вызываемо из Claude Code/Hermes через Bash,
   без сервера.
3. **MCP stdio-сервер**: `mvm-cli serve --mcp` — инструменты `mvm_snapshot` / `mvm_recap` /
   `mvm_inspect`. JSON-RPC 2.0 поверх stdin/stdout **hand-rolled на serde_json** (+0 deps — по
   образцу zero-deps Hungarian 9.10). Сервер **не владеет камерой**: читает артефакты
   (StateFile, scene-log, events-log) — producer и подача развязаны, mesh не трогаем.

---

## 4. Порядок работ (12.x, TDD, smallest-first)

### Трек A — токен-диета producer (Scene JSON, контракт не ломаем)

| Шаг | Задача | Тест |
|-----|--------|------|
| **12.A1** ✅ | **Округление f32 в сериализации**: `serialize_with`-хелперы (3 dp для conf/score/pos/dist/bbox/wrists/distribution, 1 dp для fps) на всех `Json*`. Схема не меняется, только значения | serde round-trip валиден; прежние ключи на месте; A3-фикстура: **−15.1%** (замер); downstream-парсер тестов не падает. **Сделано:** `state.rs` хелперы `r3`/`ser_*`, тест `json_rounds_floats_for_compact_stream`, 118 тестов mvm-cli + весь воркспейс зелёные |
| **12.A2** | **`emotion_distribution` → opt-in** `--dump-distributions` (default off; сводка `emotion+conf+entropy` остаётся всегда). Изменение по согласованию — потребители (Sight/Grafana) наши | без флага ключ отсутствует; с флагом — байт-в-байт прежнее поведение |
| **12.A3** | **Диета relations**: по умолчанию только рёбра с участием человека + все `overlapping`/`part_of`; полный граф (object↔object `near`) — за `--dump-full-graph` | 5 сущностей A3: 7 рёбер → 4; с флагом — прежние 7 |

### Трек B — snapshot v2 (рендерер)

| Шаг | Задача | Тест |
|-----|--------|------|
| **12.B0** ✅ | **Mesh↔sync JSON-parity** (вскрыт при B2): mesh терял `remembered` (object-permanence), `mask_coverage` объектов и `likely_animal` лиц; хуже — **поведенчески** mesh-fusion НЕ исключал животных из `infer_activities` → собака могла стать участником `people_together`. `ObjectState.remembered` + `WorldState.animals` (bbox-детекции животных из `SceneAgent` до `is_relevant`-фильтра) → `FusionAgent` дропает pet-overlapping тело (общий `mvm_scene::ANIMAL_IOU`), JSON эмитит `remembered`/`mask_coverage`/`likely_animal` как на sync | **Сделано:** 3 теста (mesh remembered+mask_coverage; likely_animal-флаг; **анти-pet-гейт с позитивным контролем** — пара срабатывает без животных, гасится с ними); 25 крейтов зелёные |
| **12.B1** ✅ | **`mvm-cli/src/render.rs` (новый)**: `render_snapshot(&WorldState, fps, ts) -> String` — markdown по контракту §3.1; `conf_word()` тиры; таблица сущностей (только выводы, без сырых координат); зоны по `build_scene`+`zone_label`; активности с duration из journal | **Сделано:** 3 golden-теста (empty / 2 чел.+объект+зона / journal-driven duration); ts вынесен в параметр (детерминизм); переиспользует `fusion::zone_label` (→pub) + entity-билдеры (→pub(crate)); snapshot ≈350 байт vs ~2.8 КБ тик |
| **12.B2** ✅ | **Ноты честности**: `⚠️`-строки из non-`face_verified`, withheld-позы (`body_coverage` upper/head + posture None), `plausibility=low`, `stable_participants=false` активностей. `remembered` на mesh недоступен (ObjectState без missed-count) — честно не рендерится | **Сделано:** 3 теста (unverified+withheld+OOD; unstable-activity; чистая сцена → блока нет); детерминир. порядок (люди по id, затем активности) |
| **12.B3** ✅ | **Проводка**: `StateFile::write_world` пишет snapshot v2 (`render_snapshot`), sync `write()` не тронут. **Контракт `current_state.md` сменился** (markdown-таблица вместо `key: value`) — ⚠️ Sight-consumer/sight-journal skill надо обновить (owner-approved). Ended-events ушли из snapshot (это recap, 12.C) | **Сделано:** e2e-тесты (mesh-stub → v2 empty-header; StateFileWriter agent → v2 occupied); тесты старого формата переписаны; 0 dead_code warnings (render подключён); 25 крейтов зелёные |

### Трек C — persistence + recap (ядро фазы)

| Шаг | Задача | Тест |
|-----|--------|------|
| **12.C0** ✅ | **Scene-log**: `--scene-log <dir>` — append JSON-строк в `scene-YYYY-MM-DD.jsonl` с ротацией по локальной дате, независимо от `--dump-json`; проведён в mesh (`StateFileWriter`) и sync (`run_native_loop`) | **Сделано:** `SceneLog` (`append_at` date-injected → детерм. тест ротации); mesh-интеграция (`writer_appends_to_scene_log`, dump_json=off); **live-прогон** (`mesh --scene-log`, 4 строки, валидный JSON с A1-округлением); 25 крейтов зелёные |
| **12.C1** ✅ | **Events-log**: `StateFileWriter` дренирует `snapshot.events` → `events-YYYY-MM-DD.jsonl` в том же `--scene-log <dir>` (ts + `JsonEventLogged` flatten: label/participants/started_at/ended_at/duration_s/peak_confidence/stable_participants, A1-округление). `SceneLog` обобщён префиксом; sync journal не гоняет → events-log только mesh | **Сделано:** интеграц. тест `writer_drains_events_to_events_log` (реальный агент, эпизод в файле ровно раз, эксклюзивный дрен из 11.C3 цел); файл ленивый (создаётся на 1-м эпизоде); 129 тестов mvm-cli |
| **12.C2** ✅ | **Агрегаты recap** (`mvm-core/src/recap.rs`, новый — чистая арифметика): `Recap::build(frames, episodes)` — presence per id (frames/first-last/dist-range/dominant-posture), **агрегация эпизодов со схлопыванием churn object-id** (ключ `(label, humans)`, object-id ≥10000 выпадают → 62 сырых эпизода живого прогона сворачиваются в ~5 строк), object/zone spans, anomaly-max, фантом-эвристика (ephemeral + brief + mid-session). **Живой прогон подтвердил алгоритм** (31 «reading»-фрагмента от churn → 1) | **Сделано:** 5 тестов (churn-collapse, 2-чел. ключ, presence+phantom id12, short-session не флажит, objects/zones/anomaly); 25 крейтов зелёные |
| **12.C3** ✅ | **`mvm-cli recap`** подкоманда (`recap_cmd.rs`): `--input <dir\|file> [--events f] [--from/--to iso \| --last 30s/10m/2h]` → парсит scene+events JSONL → `Recap::build` → `render.rs::render_recap` markdown (people presence, rolled-up activities, scene, `⚠️` фантом-ноты). `--last` якорится к последнему ts (детерм., offline) | **Сделано:** golden `render_recap_shape_and_phantom_note`; **живой прогон на реальной сессии** (`/tmp/mvm_ctrack`): 159 кадров+62 эпизода → **~192× сжатие** (71k→368 ток.), `--last 30s` окно работает; 25 крейтов зелёные |

### Трек D — интерфейс подачи

| Шаг | Задача | Тест |
|-----|--------|------|
| **12.D1** ✅ | **`journal.md` rolling-recap**: `StateFileWriter` раз в `cadence=60` записей перерендеривает recap дня (`recap_markdown_from_dir` над scene/events-логами) в `journal.md` рядом с `current_state.md` (Sight-workspace, `StateFile::journal_path`). Включается при `--scene-log`; `recap_cmd` отрефакторен (`render_recap_string` общий для CLI и journal) | **Сделано:** интеграц. тест `writer_renders_rolling_journal` (реальный агент → journal.md = «# Recap …**Вадим(0)**»); **live**: journal.md в Sight-workspace после 60 записей, валидный рекап |
| **12.D2** | **MCP stdio-сервер** `mvm-cli serve --mcp`: hand-rolled JSON-RPC 2.0 (initialize, tools/list, tools/call; serde_json, +0 deps). Инструменты: `mvm_snapshot()` (читает StateFile), `mvm_recap({last?/from?/to?})`, `mvm_inspect({ts?/track_id?, window_s?})`. Read-only, камерой не владеет | интеграционный тест: скриптовый клиент шлёт initialize/tools-call в stdin, ответы валидны; реальная проверка — регистрация в `~/.hermes/config.yaml` и Claude Code `.mcp.json` |

### Трек E — inspect (после C0)

| Шаг | Задача | Тест |
|-----|--------|------|
| **12.E1** ✅ | **`mvm-cli inspect`** (`recap_cmd.rs`): `--input <dir\|file> [--around <ts> --window 5] [--id N] [--md]` — полноточный срез сырых scene-строк; raw JSONL по умолч., `--md` = компактная строка/кадр. `resolve_files` обобщён на inspect | **Сделано:** 3 теста ядра `inspect_select` (окно ±5s, фильтр по id, md-строка); **live** на реальных логах: `--around ±2s` и `--id 12` (история фантома) — raw даёт `posture_conf`/`torso_attitude`/coords для adjudication |

**Глобальный порядок:** `12.A1 → 12.B1 → 12.B2 → 12.B3 → 12.C0 → 12.C1 → 12.C2 → 12.C3 → 12.D1 →
12.E1 → 12.A2 → 12.A3 → 12.D2`. Логика: сначала дешёвый выигрыш (A1), затем ценность (snapshot,
recap), диета контракта (A2/A3) — после того как новые артефакты покрывают потребителей, MCP —
последним (тонкая обёртка над готовым).

---

## 5. Критические файлы

- `mvm-cli/src/render.rs` — **новый**: snapshot v2 + markdown-хелперы (`conf_word`, таблица сущностей, ноты).
- `mvm-core/src/recap.rs` — **новый**: presence-RLE, дельты, фантом-эвристика (без `ort`/`image`, как `activity.rs`).
- `mvm-cli/src/state.rs` — 12.A1 округление в `Json*`; `write_world` → v2; scene/events-log.
- `mvm-cli/src/main.rs` — подкоманды `recap`/`inspect`/`serve --mcp`; флаги `--scene-log`, `--dump-distributions`, `--dump-full-graph`.
- `mvm-cli/src/agents/` — `StateFileWriter`: дрен events → `events.jsonl`, cadence `journal.md`.
- `mvm-cli/tests/fixtures/` — подрезанная A3-фикстура (jsonl) для golden recap.
- Доки: `~/wiki/mvm/outputs/{scene-json,state-file}.md` (+ новый `recap.md`), `mvm/CLAUDE.md`, `~/wiki/mvm/roadmap.md`.

---

## 6. Верификация (end-to-end)

1. **Юнит/golden:** `cargo test --workspace` зелёный после каждого шага; golden-файлы snapshot/recap
   на синтетике и A3-фикстуре.
2. **Live mesh:** `mesh --masks --dump-json --scene-log /tmp/s.jsonl` 2–3 минуты за столом →
   `current_state.md` v2 читаем; `recap --last 3m` даёт эпизоды с корректными длительностями;
   короткие касания (<enter-dwell) в recap не появляются.
3. **LLM-acceptance (главный тест фазы):** свежей LLM-сессии (Claude/Hermes) дать **только**
   snapshot + recap A3-сессии (без сырого JSONL) и попросить подробную реконструкцию сцены.
   Эталон — рукописная `live-2026-07-04-a3-2person.md`. Критерий: восстановлены сущности
   (2 человека + фантом как *подозрение*, не как третий человек), 3D-раскладка, зоны, фазы
   сеанса, объектные активности; вход ≤2k токенов. Фантом-нота обязана предотвратить
   галлюцинацию «третьего человека».
4. **Токен-бюджеты (замер, не вера):** тик после A1 ~2.8 КБ (−15%), после A1+A2+A3 ~2.5 КБ (−24%);
   snapshot ≤ 300 ток.; recap 50с-сессии ≤ 800 ток.; recap часа фоновой сцены ≤ 1.5k ток. Главный
   рычаг — recap/snapshot (100–150×), не диета тика.
5. **Обратная совместимость:** прежние ключи Scene JSON на месте (A1 меняет значения, не схему);
   sync-путь `write()` байт-идентичен; `--dump-distributions`/`--dump-full-graph` возвращают
   прежнюю полноту.
6. **MCP:** инструменты видны из Claude Code (`.mcp.json`) и Hermes (`hermes tools list | grep mvm`);
   `mvm_snapshot` из живой сессии отвечает <100мс (чтение файла, не инференс).

---

## 7. Что НЕ делаем

- **LLM/VLM внутри MVM** — рендер детерминированный; нарратив поверх recap — работа внешней LLM
  (System 2). Level-3 из фазы 11 остаётся снаружи.
- **Embeddings/RAG по истории** — это GBrain (`brain/vision/`), MVM отдаёт markdown-агрегаты,
  индексирует их GBrain-пайплайн (Sight Aggregator, Фаза 5 родительского плана).
- **Подавление сомнительных данных** — фантомы/low-conf маркируются нотами, не вырезаются
  (уроки SUPREME: сырые сигналы точнее осторожных ручных трактовок).
- **Keyframe-экспорт JPEG для VLM** — возможный опт-ин хвост продуктовой фазы, не ядро.
- **Изменения fusion/tracker/моделей** — фаза 12 строго читает то, что производит фаза 11.
- **Веб-сервер/HTTP/tokio** — только stdio (JSON-RPC) и файлы; +0 deps.
