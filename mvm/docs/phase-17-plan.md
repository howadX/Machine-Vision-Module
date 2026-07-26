# Фаза 17 — Canonical MVM Agent Skills

> **Версия:** 0.2 (2026-07-25). Ветка `kaggle`. Родитель — фазы 12–13
> (LLM-render / Agent Tools) и 16.7 (Semantic Session Log).
> **Цель:** перенести know-how «как LLM-агенту водить MVM» в два переносимых
> load-on-demand артефакта `SKILL.md`, сделать их единственным живым источником
> MVM-инструкций и защитить от рассинхрона с публичным контрактом.
>
> **Это не релиз и не изменение vision runtime.** Бинарь `mvm`, детекторы,
> fusion, MCP-поведение и версия не меняются. Фаза затрагивает documentation,
> skills и их validation/integration.

## Мотивация / аудит-основание

- `mvm/AGENT.md` (188 строк в `HEAD`) уже содержит правильные правила: pull-модель
  `snapshot -> recap -> inspect`, honesty (`withheld` != negative, `⚠️`-ноты),
  чтение snapshot и text-only semantic sessions 16.7. Это skill по содержанию,
  но не по форме и без progressive disclosure.
- `~/.hermes/profiles/sight/skills/vision/sight/SKILL.md` содержит стейл Python/GPU
  стек: SCRFD, ArcFace 512d, HSEmotion, MediaPipe Holistic и `mvm_bridge.py` как
  основной путь. Он противоречит актуальному Rust-MVM (YuNet / SFace 128d /
  EmotiEffLib / BlazePose).
- `sight-journal` и связанная Python/wiki cron-цепочка retired: Semantic Session
  Log 16.7 уже ведёт durable историю сцены в MVM.

**Почему SKILL.md:** frontmatter `name` + `description` доступен для discovery
дёшево, тело загружается только по trigger, а один portable формат читается
Codex, Claude Code, Gemini, Cursor и Hermes.

## Управляющие принципы

1. **Канон в репозитории.** `mvm/skills/` — единственный редактируемый источник
   MVM know-how; потребители обнаруживают его напрямую, не копируют.
2. **Honesty неизменна.** `withheld`/`unknown` не становятся отрицательным классом;
   `⚠️` и ограничения MVM переносятся без смягчения.
3. **Portable core.** В frontmatter используются только стандартные `name` и
   `description`; Hermes-специфичные `category`, `tags` и другие поля не нужны.
4. **Проверяемый контракт выше prose.** Skills объясняют контракт, но не
   переопределяют CLI, MCP или semantic schema.
5. **Hermes migration обратима.** Legacy artifacts архивируются до окончательной
   очистки; исторические logs/sessions не удаляются.

## Карта источников публичного контракта

| Поверхность | Канонический источник |
|---|---|
| CLI flags, команды и help | `mvm/crates/mvm-cli/src/help.rs` и parser в `main.rs` |
| MCP tool names, arguments и descriptions | `mvm/crates/mvm-cli/src/mcp.rs` |
| `recap` / `inspect` semantics | `mvm/crates/mvm-cli/src/recap_cmd.rs` |
| Sessions, privacy, labels и statuses | `mvm/crates/mvm-dataset/` |
| Наблюдаемое поведение | релевантные Rust tests и live-validated phase docs |

Skills не фиксируют изменчивые номера схемы или версии без необходимости: для
точного текущего значения агент направляется к CLI и `session.toml`/labels metadata.

## Целевая структура

```
mvm/skills/
├── mvm-vision/SKILL.md
└── mvm-dataset/SKILL.md
```

### `mvm-vision`

Trigger: запросы «что камера видит сейчас», «что произошло», анализ живой сцены
или semantic history. Тело содержит:

- назначение и closed-world границы MVM;
- cheapest-first workflow `snapshot -> recap -> inspect`;
- MCP и CLI access, prerequisite работающего producer;
- чтение snapshot (id ranges, confidence, zones, activities);
- honesty markers, phantom/face/pose caveats и inspect cadence caveat.

### `mvm-dataset`

Trigger: запись, проверка, retention или export накопленных vision sessions. Тело
содержит:

- default-on text-only semantic session, `--record` и `--no-vision-log`;
- privacy modes и self-describing session layout;
- `dataset list/info/stats/check/prune/export`;
- Canonical Observation, provenance и правило `withheld`/`unknown` != negative.

Оба файла самодостаточны в своём workflow; дублирование между ними минимально.
Исходным содержимым миграции послужил `HEAD:mvm/AGENT.md`; промежуточный
монолитный черновик `mvm/SKILL.md` удалён после разбиения.

## Фаза 17.0 — Repo Skills ✅

1. ✅ Создать два canonical `SKILL.md` по целевой структуре с frontmatter `name` и
   `description`; descriptions описывают точные triggers и остаются короткими.
2. ✅ Разнести действующее content `AGENT.md` по vision/dataset границе, удалить
   legacy Python claims и не копировать stale flat-log diagrams.
3. ✅ Удалить `mvm/AGENT.md`; перенацелить все repo-ссылки на новый canonical tree.
4. ✅ Исправить известные legacy references в `mvm/README.md`: retired
   `scene-*.jsonl`/`events-*.jsonl` outputs и ссылки на `AGENT.md`. После миграции
   в repo не должно остаться конкурирующего how-to.

**Верификация:** прочтение каждого skill отдельно позволяет корректно выполнить
его workflow; `rg` не находит активных ссылок на `AGENT.md` или legacy flat logs;
honesty statements сохранены.

## Фаза 17.1 — CI Contract Gate ✅

1. ✅ Добавить repo-local validator без зависимости от `~/.codex`, `~/.hermes` или
   внешних Python packages. Он проверяет структуру tree, minimal frontmatter,
   имена/descriptions, межskill-ссылки, обязательные contract markers и отсутствие
   stale Python-stack markers.
2. ✅ Добавить GitHub Actions workflow на pull request и push. Он запускает validator
   и релевантные существующие CLI/MCP tests.
3. ✅ Добавить public-surface audit: изменения согласованных файлов CLI help/parser,
   MCP definitions, recap/inspect, dataset config/schema/validation требуют
   изменения skills или явного audit-артефакта в том же PR.

Gate сообщает, что нужен human audit skills; он не заявляет, что способен
автоматически доказать смысловую полноту prose.

**Верификация:** workflow проходит на текущем контракте и предсказуемо падает на
невалидном frontmatter, stale marker либо public-surface change без skill audit.

## Фаза 17.2 — Portability

- Проверить SKILL.md внешним совместимым validator/discovery mechanism.
- Проверить, что body остаётся компактным и содержит только action-guiding
  knowledge; большие справочные сведения остаются в существующих repo docs.
- Подтвердить, что skills discoverable по frontmatter и загружаются без
  agent-specific assumptions.

## Фаза 17.3 — Hermes Integration

> Изменения в `~/.hermes/` выполняются только с подтверждением владельца. Оно
> дано для этой фазы 2026-07-25.

1. Добавить абсолютный путь checkout `.../Machine Vision Module/mvm/skills` в
   `skills.external_dirs` профиля `sight`; Hermes обнаруживает канон напрямую,
   без симлинков и копий.
2. Добавить profile-level diagnostic для отсутствующего external directory:
   агент сообщает, что MVM skills недоступны, и никогда не откатывается к stale
   local fallback.
3. Архивировать local `sight`/`sight-journal`, paused jobs `sight-capture` и
   `sight-journal` и скрипт `sight-capture.sh`. Они обслуживают только retired
   Python/wiki flow; historical logs/sessions остаются нетронутыми.
4. Заменить устаревший profile `AGENTS.md` короткой навигацией к внешним MVM
   skills, без Python/GPU/wiki workflow.
5. В свежей Hermes session или через supported reload подтвердить discovery и
   загрузку `mvm-vision`/`mvm-dataset` из external directory. Только после
   этого окончательно удалить активные legacy artifacts; archive остаётся
   rollback-опцией до принятия migration.

**Верификация:** профиль не обнаруживает stale skills; оба canonical skills
видны и загружаются из repo path; отсутствие path даёт явную diagnostic; cron не
ссылается на `sight-journal` или `sight.py`.

## Что не меняется

| Слой | Изменение |
|---|---|
| Rust vision runtime, модели, fusion, tracker | 0 |
| CLI/MCP semantics и schema | 0 |
| Версия бинаря и GitHub release | 0 |
| Исторические Hermes logs/sessions | не удаляются |
| Несвязанные worktree changes | не затрагиваются |

---

*Phase 17 переводит MVM agent know-how в portable skills и ставит вокруг него
контрактный audit. Репозиторий остаётся единственным каноном; Hermes потребляет
его напрямую.*
