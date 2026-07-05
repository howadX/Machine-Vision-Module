# Фаза 13 — Agent Tools/Skills Polishing (ЗАКРЫТА)

> **Версия:** 1.0 (2026-07-05). `+0 ONNX / +0 deps` (hand-rolled, без `clap`).
> **Тезис:** инструменты фазы 12 (snapshot/recap/inspect + MCP) есть, но CLI вырос органически
> (hand-rolled парсинг, дубли флагов, нет `--help`), а «как агенту этим пользоваться» нигде не описано.
> Фаза 13 не добавляет возможностей — делает существующее **пригодным для агента**.

## Что сделано

### A. Рефактор CLI (`+0-dep`)
- **A0 (спайк-гейт):** решение — hand-rolled `--help`, `+0-dep` (инвариант №1 против `clap`; справка = статические строки).
- **A1 — дедуп флагов:** `CommonFlags` + `parse_common()` (`main.rs`) — единый парс 6 флагов, дублировавшихся
  между root-sync и mesh (`--dump-json`/`--emotion-debug`/`--masks`/`--scene-log`/`--dump-distributions`/
  `--dump-full-graph`). `Args`/`MeshArgs` встраивают `common: CommonFlags`; downstream читает `args.common.*`.
  `parse_args` теперь берёт `raw: &[String]` (тестируем). Тест `common_flags_parse_identically_on_both_paths`.
- **A2 — дерево подкоманд:** `run` = каноничное имя live-mesh (продакшн), `mesh` — **скрытый алиас** (тот же
  `run_mesh`). Диспетч в `main()` перечисляет `run`/`snapshot`/`recap`/`inspect`/`serve`; неизвестный
  argv[1]-флаг по-прежнему проваливается в legacy sync-путь (`--native-camera`/`--native` живы).
- **A3 — `snapshot`:** новая тонкая подкоманда (`run_snapshot`) — читает StateFile (`--state`, деф
  `default_state_path()`) и печатает `current_state.md`, как MCP `mvm_snapshot`. Разовое «что сейчас» без `serve`.
- **A4 — нормализация:** `inspect --md` теперь **дефолт true** (компактный markdown, как в MCP-инструменте);
  `--raw`/`--json` возвращают сырой JSONL. Тест `inspect_md_defaults_true_and_raw_opts_out`.

### B. `--help` / `-h` / `--version` (`+0-dep`)
- Новый модуль `help.rs`: `MAIN` + `RUN`/`SNAPSHOT`/`RECAP`/`INSPECT`/`SERVE` (константы = golden-строки),
  `wants_help`/`wants_version`/`subcommand`. `main()` перехватывает help/version **до** hand-rolled парсеров.
- `mvm --help` / бэйр `mvm` / `-h` → обзор (pull-модель + список подкоманд + ссылка на `AGENT.md`).
  `mvm --version` / `-V`. `mvm <cmd> --help` → флаги с типами/дефолтами/единицами (сек vs мс) и примерами.
- Golden-тесты в `help.rs` (все подкоманды имеют экран, обзор перечисляет все команды, `mesh`≡`run`).

### C. Agent-манифест `mvm/AGENT.md` (центр фазы)
- Один документ, который агент грузит в контекст: что такое MVM · pull-модель (snapshot→recap→inspect) ·
  два способа доступа (MCP / CLI+файлы) · **как читать snapshot** (таблица `id|who|dist|posture|motion|
  emotion|head|hands`, conf словами, семантика id 0–9/≥10/≥10000) · **честность** (⚠️-ноты, фантом ≠ человек) ·
  каденс-нюанс (event-journal по-кадрово vs scene-log по dump-каденсу) · чего MVM НЕ делает (→ VLM-fallback).
  Линк из `mvm/CLAUDE.md`.

### D. MCP tool-descriptions (`mcp.rs::tool_defs`)
- Описания `mvm_snapshot`/`mvm_recap`/`mvm_inspect` переписаны как «манифест в миниатюре» (когда звать /
  что вернётся / как трактовать), + `title`, + примеры значений параметров, + семантика id в snapshot/inspect.
  Дословно сверены с `AGENT.md`. Golden-тест `tool_descriptions_teach_the_pull_model`.

## Верификация
- `cargo test --workspace` — зелёно (mvm-cli 147→**150**; +A1-parity, +inspect-md-default, +help goldens,
  +MCP-descr golden). Release чист, 0 warnings.
- Live smoke: `mvm --help`/`--version`/`<cmd> --help` читаемы; `mvm snapshot` печатает state (и честно
  ошибается на отсутствии); `mvm run --stub` (алиас→mesh) работает; `serve --mcp` отдаёт обогащённый `tools/list`.
- **Осталось (владелец):** acceptance-прогон — свежая LLM-сессия с одним `AGENT.md` + доступом к CLI/MCP
  сама выбирает инструмент и реконструирует сцену (планка фазы 12, теперь без подсказок).

## Вне scope (→ фаза 14+, продукт)
Новые сигналы/модели; смена форматов выходов (фаза 12 зафиксировала); упаковка: авто-калибровка fx,
drag-drop галерея, one-click install (один бинарник), systemd + cron state-dump.
