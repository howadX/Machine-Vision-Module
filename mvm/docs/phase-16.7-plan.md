# Фаза 16.7 — Semantic Session Log: конвергенция vision-логов

> **Версия:** 0.2 (2026-07-23; Фазы 1 + 1.5 закрыты). Ветка `kaggle`. Родитель — фаза 16 (Dataset Factory, v0.3.0).
> **Цель фазы:** свести два параллельных стора истории сцены в **один семантический session-лог**.
> Сегодня MVM пишет vision-историю как плоские `scene-YYYY-MM-DD.jsonl` / `events-YYYY-MM-DD.jsonl`
> (по дню, без session-дисциплины, без prune), а Dataset Factory пишет session-организованные
> `labels.jsonl` (Canonical Observation) с pixel-ами. Фаза 16.7 делает **text-only семантические сессии**
> единым durable-стором vision-истории: та же session-дисциплина (grouper, `mvm dataset check`/`prune`),
> но только текст — подробный семантический лог сцены без фотографий.
>
> Мотивация владельца (дословно): «vision-логам нужен такой же тип хранения как и в dataset, но без
> сохранения фотографий, только текстовые подробные/семантические логи».

## Управляющие принципы (сквозные)

1. **Инварианты ядра целы.** Детекторы, классификаторы, трекер, fusion — **0 изменений**. Роутер и ридеры —
   чистые потребители `WorldState` / записанных сессий.
2. **Ноль дублирования (north-star владельца).** Целевое состояние — **один** durable текст-стор истории
   сцены. Плоские `scene-*.jsonl` / `events-*.jsonl` ретайрятся, когда ридеры мигрированы.
3. **Семантический лог = супермножество плоских логов.** До миграции ридеров Canonical Observation обязан
   нести всё, что несут `scene-*.jsonl` (зоны, рёбра, per-frame активности) и `events-*.jsonl` (эпизоды).
   Это достройка **отложенных §16.3 relations/events**, а не новая сущность.
4. **Честность (§16.0.2).** `withheld`/`unknown` не превращаются в отрицательный класс. Статусы и
   provenance сохраняются в семантическом логе.
5. **Text-only ≠ приватность-регресс.** Плоские `scene-*.jsonl` уже пишут identities/активности по
   умолчанию; text-only сессия несёт то же, без пикселей → экспозиция не растёт. Privacy-режимы касаются
   только пиксельного слоя (в text-only его нет).

## Аудит-основание (что читают ридеры vs что несёт Canonical Observation)

| Поле | Кто читает | Источник | В Canonical Observation (16.0.1)? |
|---|---|---|---|
| people (+ атрибуты) | `recap_cmd`, `render` | scene-*.jsonl | ✅ entities kind=person |
| objects (class) | `recap_cmd` | scene-*.jsonl | ✅ entities kind=object |
| anomaly_score | `recap_cmd`, `render` | scene-*.jsonl | ✅ quality.anomaly_score |
| **stable_zones (label)** | `recap_cmd` | scene-*.jsonl | ❌ `relations` пуст → **пробел** |
| **zones (per-frame компоненты)** | `render` | scene-*.jsonl | ❌ **пробел** |
| **activities (per-frame + duration)** | `render` | scene-*.jsonl | ❌ не эмитится → **пробел** |
| **episodes (completed)** | `recap_cmd` | events-*.jsonl | ❌ `events` пуст → **пробел** |

Вывод: миграция ридеров невозможна, пока роутер не заполнит зоны/рёбра/активности/эпизоды. Сигнал уже есть
в `WorldState` (`activity_instances`, `scene.zones`, stable-zones, завершённые эпизоды), схема имеет открытые
слоты (`relations`/`events`) → это заполнение, а не ломка контракта.

## Схема (фиксируется здесь)

`SCHEMA_VERSION` бампается `16.0.1 → 16.7.0`. Все добавления аддитивны (`#[serde(default)]`), старые ридеры
не ломаются. Расширяется `CanonicalObservation`:

### `relations: Vec<Value>` — зоны + рёбра (группа G₂/реляционная)

Гетерогенный список записей, каждая с полем `kind`:

```json
{ "kind": "zone", "label": "workstation", "members": ["p0", "o10001"], "persistent_id": 3 }
{ "kind": "zone", "label": null, "members": ["p1", "p10"], "persistent_id": null }   // аноним. per-frame компонент
{ "kind": "edge", "a": "p0", "b": "o10001", "rel": "near" }
```

- `label` — семантическая метка зоны (`workstation`/`seating`/…) или `null` для анонимного компонента.
- `members` — `entity_id` участников (связь с entities).
- `persistent_id` — id устойчивой зоны (`stable_zones`) или `null` для чисто per-frame компонента.
- `rel` — тип ребра (`near`/`overlap`/…).

### `activities: Vec<Value>` — **новое поле**, per-frame активные активности (G₃/G₄)

```json
{ "label": "using_object(cell phone)", "participants": ["p0"], "started_at": 1721740200.1,
  "duration_s": 4.2, "stable": true }
```

- `stable` — участники устойчивы (`false` при эфемерном id → honesty-нота «участники нестабильны»).
- `started_at`/`duration_s` — временные поля (могут быть `null` без журнала).

### `events: Vec<Value>` — завершённые эпизоды (G₄), инлайн на кадре завершения

```json
{ "kind": "episode", "label": "seated_at_workstation", "started_at": 1721740180.0,
  "ended_at": 1721740260.0, "duration_s": 80.0, "participants": ["p0"] }
```

Эпизод пишется в `labels.jsonl` того кадра, на котором он закрылся (зеркалит текущее поведение
`events-*.jsonl`, но внутри единого файла сессии). **Решение:** эпизоды инлайн в `labels.jsonl`, а не
отдельным `events.jsonl` — единый файл на сессию.

> `derived_targets` (prediction targets) — **вне 16.7** (north-star 7.A/B forward-model).

## Порядок и критерии верификации

### Фаза 1 — text-only recording (`save_frames`) ✅ ЗАКРЫТА (commit `8403e4c0`)

`DatasetConfig.save_frames` (default true). `save_frames=false` → сессия `labels.jsonl` + `session.toml`
без пикселей; writer не аллоцирует frame-путь (`FrameRecord.frame_path: Option`), агент пропускает энкод.
`mvm dataset check`/`prune` работают из коробки (гейт 16.6-A уже толерантен к `frame_path=None`). 671 тест.

### Фаза 1.5 — обогащение роутера (достройка §16.3) ✅ ЗАКРЫТА

`SCHEMA_VERSION → 16.7.0`; `observation.rs` +поле `activities` (аддитивно, `#[serde(default)]`). Роутер
(`router.rs`) заполняет `activities` (per-frame из `activity_instances`, иначе flat `activities`) и `events`
(эпизоды из `world.events`) **напрямую из `WorldState`**; участники маппятся в `entity_id` (`p{id}`/`o{id}`),
`stable` зеркалит reserved-id правило.

**Уточнение по итогам аудита (вариант B):** зоны/рёбра **не лежат в `WorldState`** (их строит
`build_scene`+stateful `ZoneTracker` в fusion/cli-слое). Чтобы роутер остался **чистой функцией входов**,
введён `SceneRelations{zones,edges}` — новый аргумент `route_with_relations()` (старый `route()` = тонкая
обёртка с пустыми relations). Зоны считает stateful **агент** `mvm-cli` (`DatasetWriterAgent` теперь владеет
своим `ZoneTracker`, тикает раз на записанный кадр — как `SceneLog`), передаёт в `record_frame`.
`mvm-dataset` остаётся dep-чистым (serde + mvm-core); зависимость на `mvm-fusion` — только в cli-агенте.

**Верификация (TDD, Rust) — сделано, 680 тестов зелёные (+9):**
- ✅ `route_with_relations` с зонами/рёбрами → зоны `{kind:"zone",label,members,persistent_id}` + рёбра
  `{kind:"edge",a,b,rel}`; активности из журнала несут `started_at`/`duration_s`/`state`/`stable`; эпизоды
  из `world.events` → `{kind:"episode",…}`; flat-ветка activities при пустых инстансах (router unit-тесты).
- ✅ Round-trip JSONL + back-compat: 16.0.1-строка без `activities` парсится (default пусто).
- ✅ Golden-паритет (агент): person + перекрывающая клавиатура → зона `workstation` + ребро `p1↔o10001` в
  `labels.jsonl` — зеркалит `SceneLog::to_json_line_world`.
- ✅ `mvm dataset check` зелёный: `schema_version==SCHEMA_VERSION` (нет drift-warn), обогащённые записи
  проходят валидатор.

### Фаза 2 — миграция ридеров ⏳

`recap_cmd.rs` / `render.rs` / journal читают `labels.jsonl` семантической сессии вместо
`scene-*.jsonl`/`events-*.jsonl` (эпизоды — из инлайн `events`).

**Верификация:**
- Golden: `recap` над семантической сессией даёт тот же episode/zone/activity-свод, что и над плоскими
  логами (фикстура-сессия, сравнение вывода).
- `snapshot`/`inspect` рендерят из семантической записи без потери полей (зоны, активности, ⚠️-ноты).
- MCP-инструменты (`serve --mcp`) работают поверх нового пути.

### Фаза 3 — ретайр плоских логов + default-on ⏳

Text-only семантическая запись становится дефолтным vision-логом `mvm run` (always-on, лёгкий);
запись `scene-*.jsonl`/`events-*.jsonl` убирается; ретеншен — через `mvm dataset prune`.

**Верификация:**
- `mvm run` без флагов → создаётся семантическая сессия; `scene-*.jsonl` больше не пишется.
- `recap`/`inspect`/`snapshot` работают на дефолтном пути (в т.ч. Sight-потребитель).
- Прогон `mvm run --stub --max-frames` → сессия на диске, `check` зелёный.
- Live-валидация: живая камера 3–5 мин → сессия → `recap`/`snapshot` корректны.

## Открытые решения (зафиксировать по ходу)

- **Единый session-store.** Text-only vision-сессии и pixel-датасет живут в одном `datasets/sessions/`
  (сессия может быть text-only или с пикселями; `mvm dataset` инспектирует обе). Зонтичное имя `dataset`
  для always-on vision-лога слегка неточно — переименование зонтика **вне 16.7** (по желанию позже).
- **Default-on vs opt-in (Фаза 3).** Рекомендация — default-on text-only (соответствует текущему
  поведению плоских логов; privacy-экспозиция не растёт). Финализируется в Фазе 3.
- **Back-compat старых `scene-*.jsonl` на диске.** Решить в Фазе 2: `recap` читает только новые сессии,
  либо временный dual-read. Рекомендация — clean cut + нота (старые плоские логи — ручная чистка).

## Что НЕ меняется

| Слой | Изменения |
|------|-----------|
| детекторы / fusion / mesh / трекер | **0** |
| `mvm-core` | **0** (роутер читает существующий `WorldState`) |
| `mvm-dataset::observation` | +поле `activities`, bump `SCHEMA_VERSION` (аддитивно) |
| `mvm-dataset::router` | заполнение relations/activities/events |
| `mvm-cli` (`recap_cmd`/`render`/journal) | смена источника чтения (Фаза 2), ретайр плоских логов (Фаза 3) |

## Критичные файлы

| Файл | Роль |
|------|------|
| `crates/mvm-dataset/src/observation.rs` | ✅ +`activities`, `SCHEMA_VERSION` 16.7.0 (Фаза 1.5) |
| `crates/mvm-dataset/src/router.rs` | ✅ `route_with_relations` + заполнение relations/activities/events (Фаза 1.5) |
| `crates/mvm-dataset/src/config.rs` | ✅ `save_frames` (Фаза 1) |
| `crates/mvm-dataset/src/writer.rs` | ✅ text-only frame-gating (Ф1) + `relations`-аргумент `record_frame` (Ф1.5) |
| `crates/mvm-cli/src/agents/dataset_writer.rs` | ✅ владеет `ZoneTracker`, считает `SceneRelations` (Ф1.5); default-on wiring — Фаза 3 |
| `crates/mvm-cli/src/state.rs` | ✅ `relation_kept` → `pub(crate)` (Ф1.5); ретайр `SceneLog` scene/events — Фаза 3 |
| `crates/mvm-cli/src/recap_cmd.rs` | миграция чтения (Фаза 2) |
| `crates/mvm-cli/src/render.rs` | рендер из семантической записи (Фаза 2) |

---

*Ядро зрения (фазы 5–14) не трогается. mvm-dataset остаётся чистым потребителем WorldState. 16.7 достраивает
отложенные §16.3 relations/events и сводит vision-историю в единый семантический session-лог.*
