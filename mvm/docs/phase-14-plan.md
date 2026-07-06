# Фаза 14 — Продукт: нативная установка и использование

> **Версия:** 1.1 (2026-07-06, после код-ревизии). Детальный план **14.0 (фундамент)** + roadmap последующих шагов.
> **Цель фазы:** «поставил → само нашлось и заработало → пользуешься, не думая о путях, venv и JSON-RPC».
> Ядро зрения готово (фазы 8–13); фаза 14 строит **оболочку продукта**, не трогая зрение.

## Обещания пользователю (вайб)

1. Установка = одна команда. 2. Первый запуск = просто `mvm run`, без флагов (само нашло камеру и папки).
3. Новый человек = кинул фото в папку. 4. Подключение к LLM = одна команда.

## Управляющие принципы (сквозные для всей фазы)

1. **Инварианты ядра целы.** Fusion/трекинг/3D/anomaly остаются `+0 ONNX`, `+0 deps`. Продуктовая
   обвязка (конфиг, XDG, позже HTTP-fetch) — **отдельный слой**.
2. **Продуктовые deps живут ТОЛЬКО в `mvm-cli`.** Ни `toml`, ни XDG, ни fetch не протекают в
   `mvm-core`/`mvm-fusion`/детекторы. Ядро зрения чистое.
3. **Платформо-зависимое — за один шов.** Linux-бэкенды сейчас; macOS, затем Windows — доклеиваются в
   шов (`CameraSource`, `paths`), не переписывая вызовы. **Linux-first, но код кросс-готов.**

## Решения (зафиксированы с владельцем)

- **Модели** — **fetch-on-setup** (мелкий бинарь + `mvm setup` тянет в кэш; не embed-всё).
- **Демон** — **единый `mvm serve --with-camera`** поднимает продюсер + MCP одним процессом.
- **Скоуп первого захода** — **только 14.0 (расхардкод + конфиг)**; галерея/MCP-install/service/fetch — отдельные шаги.
- **Платформа** — **Linux-first**, цель — кросс (Windows в последнюю очередь).
- **Config dep** — **`toml`** (единственный новый dep фундамента; product-слой, только в `mvm-cli`; `serde` уже есть).
- **Hermes** — **сразу переезд на XDG**, без dual-write и без авто-детекта старого пути. Sight-профиль Hermes
  репойнтится на новый XDG-путь (owner, вне репо). Опция `output.state_path` остаётся для ручного override.

---

## 14.0 — Фундамент (детально) — ✅ ЗАКРЫТ (2026-07-06)

> Реализовано и live-провалидировано: 14.0.1 `paths.rs` (XDG, 0700) · 14.0.2 `config_file.rs`
> (`ProductConfig`, env>file>default, models_path во все 3 точки `Config`, friendly «run `mvm setup`») ·
> 14.0.3 шов `CameraSource` (`mvm-pipeline/source.rs`, auto-детект `/dev/video*`) · 14.0.4 XDG-выход
> (Hermes-хардкод снят, `--logs`-ловушка закрыта, `run` пишет state+логи+journal в `data_dir` по
> умолчанию) · 14.0.5 бинарь `mvm` + docs-sweep · 14.0.6 `mvm doctor`. 572 теста зелёные.
> Live: `mvm run` без флагов (авто-камера, data 0700), `snapshot`/`recap`/`serve --mcp` из
> произвольного cwd без path-флагов; без моделей падает дружелюбно. **Hermes-репойнт — owner, вне репо.**

### 14.0.1 — Модуль `paths` (XDG-шов)
Единая точка правды по путям. Hand-rolled XDG (`+0 dep`): `$XDG_DATA_HOME`/`$XDG_CONFIG_HOME`
с фолбэками `~/.local/share`, `~/.config`.
- `data_dir()` → `~/.local/share/mvm/` — **state + scene/events-логи + journal в одном месте**.
  Создаётся с правами **0700** (журнал содержит идентичности и распорядок — приватные данные).
- `config_dir()` → `~/.config/mvm/`.
- `models_dir()` → `~/.local/share/mvm/models/` (сюда потом `mvm setup`). Модели ре-фетчабельны,
  но живут под data (не `$XDG_CACHE_HOME`): кэш ОС вправе чистить, а без моделей продукт мёртв.

**Шов под кросс:** позже здесь ветка macOS (`~/Library/…`) / Windows (`%APPDATA%`) — вызовы не меняются.
Новый файл: `crates/mvm-cli/src/paths.rs`.

### 14.0.2 — Слой конфига `config.toml`
`~/.config/mvm/config.toml`, приоритет **CLI > env (`MVM_*`) > файл > дефолты**. Отсутствие файла — не ошибка
(всё из дефолтов). `~`-раскрытие в путях.

```toml
[camera]
device = "auto"        # "auto" | "/dev/video2"
width  = 1280
height = 720

[output]
data_dir  = "~/.local/share/mvm"    # state + logs + journal вместе
# state_path = "…"                  # опц. override (напр. для внешнего потребителя)

[models]
dir = "~/.local/share/mvm/models"

[gallery]
dir = "~/.config/mvm/gallery"       # структуру наполнит шаг 14.2

[cadence]
snapshot_ms = 1000
```

- Dep: **`toml`** + уже-имеющийся `serde` (derive на структуру `FileConfig`).
- `run`/`snapshot`/`serve` без флагов читают конфиг; существующие флаги (`--scene-log`, `--interval-ms`,
  `--state`, `--logs`, `--masks`, …) **переопределяют** соответствующие поля (не ломаем текущий CLI).
- Новый файл: `crates/mvm-cli/src/config_file.rs` (не путать с рантайм-`mvm_core::Config` — это CLI-слой).
- **⚠️ Резолв втекает в существующие поля рантайм-конфига, не параллельно им** (код-ревизия):
  в `mvm_core::Config` **уже есть** `camera_path` (дефолт `/dev/video0`, CLI его игнорирует) и
  `models_path` (дефолт `./models` — **cwd-относительный!** SFace/EmotiEffLib/YOLO/seg/depth/handpose
  всегда грузятся с диска; embed покрывает только YuNet+BlazePose+RF/VAE). `Config::default()`
  конструируется в **трёх** местах `main.rs` (~672/~843/~898) — все три должны получать
  `models_path`/`camera_path` из резолва `FileConfig`. Без этого `mvm run` из произвольного cwd
  сломан даже со скачанными моделями — это ключевой расхардкод шага, не деталь.

### 14.0.3 — Авто-детект камеры + шов `CameraSource`
Убрать хардкод `NativeCamera::open("/dev/video0", 1280, 720, false)` — он в **двух** местах
(`main.rs::run_mesh` ~726 и sync-путь ~861). Sweep заодно: легаси sync-`write()` пишет литерал
`"- Camera: /dev/video0"` в StateFile (`state.rs:~690`) — подставить реальное устройство.
- `device = "auto"` → перебрать `/dev/video*`, взять первую живую MJPG; иначе — устройство из конфига.
- Ввести **шов** в `mvm-pipeline`: трейт/enum `CameraSource` (open/dimensions/next-frame). Единственный
  бэкенд сейчас — `V4l2Source` (нынешний `NativeCamera`). Сюда потом сядут `AvFoundationSource`/
  `MediaFoundationSource`. CLI работает через шов, не через конкретный v4l2.
- Дружелюбный старт: `▶ camera /dev/video2 (1280×720 MJPG) · data ~/.local/share/mvm · snapshot 1s`.

### 14.0.4 — Отвязать выходные пути от Hermes (переезд на XDG)
`state.rs::default_state_path()` сейчас жёстко возвращает `~/.hermes/profiles/sight/workspace/current_state.md`
(`STATE_PATH` const). Развязка:
- Дефолт = **XDG `data_dir`**: `current_state.md` + `scene-/events-*.jsonl` + `journal.md` **в одной папке**
  → **ловушка `--logs` исчезает** (`serve` по умолчанию читает ту же `data_dir`, что пишет `run`).
- Убрать Hermes-хардкод целиком; `output.state_path` — опциональный override. Sweep: `help.rs:~89`
  и AGENT.md упоминают Hermes-путь как дефолт — обновить на XDG.
- **Hermes** репойнтится на новый XDG-путь на своей стороне (owner, вне репо) — dual-write и авто-детекта
  старого пути **нет** (решение владельца: чистый переезд).
- Одна `data_dir` кормит и `run`, и `serve` → подготовка к единому `serve --with-camera` (14.3).

### 14.0.5 — Команда называется `mvm`
`Cargo.toml` крейта `mvm-cli`: `[[bin]] name = "mvm"`. Пользователь пишет `mvm run`, не `mvm-cli run`.
`--help`/`AGENT.md` уже говорят `mvm`; README — **частично** (`./target/release/mvm-cli run` в
примерах) — sweep. `cargo xtask bundle` копирует `target/release/mvm-cli` → путь обновить.
Заодно поправить фразу «модели embedded» в README/CLAUDE.md: embed покрывает только
YuNet+BlazePose+RF/VAE, крупные модели всегда с диска.

### 14.0.6 — `mvm doctor` (минимальный)
Ответ на «почему не работает» + доказательство, что расхардкод резолвится верно:
```
mvm doctor
  camera     ✅ /dev/video2 (1280×720 MJPG)
  models     ⚠️  3/7 present (run `mvm setup`)     ← заглушка под 14.1
  data dir   ✅ ~/.local/share/mvm (writable)
  config     ✅ ~/.config/mvm/config.toml
```
Новая подкоманда (диспетч + help-экран, как в фазе 13).

### Критичные файлы 14.0
`crates/mvm-cli/src/paths.rs` (new) · `config_file.rs` (new) · `main.rs` (диспетч + оба camera-open +
резолв конфига) · `state.rs` (`default_state_path` → XDG) · `mvm-pipeline` (`CameraSource` шов) ·
`crates/mvm-cli/Cargo.toml` (`toml` dep + bin rename) · `help.rs` (экран `doctor`).

### Верификация 14.0
- `cargo test --workspace` зелёный; новые тесты: приоритет конфига (CLI>env>file>default), XDG-фолбэки,
  `~`-раскрытие, авто-детект камеры (мок-перебор устройств), резолв `data_dir`→state/logs/journal в одной папке.
- **Live (чистое окружение — нет `~/.hermes`, cwd произвольный, модели вручную лежат в `[models] dir`):**
  `mvm run` **без флагов** — сам находит камеру, создаёт `~/.local/share/mvm`, пишет туда; `mvm snapshot`
  и `mvm serve --mcp` работают на дефолтах, **без ручного `--logs`**.
- Без моделей `mvm run` падает **понятно**: «models missing → run `mvm setup`» (не паника ONNX).
- `mvm doctor` даёт корректный отчёт. Команда зовётся `mvm`.
- **Честно:** «поставил → само заработало» целиком наступает только после **14.1** (fetch моделей);
  14.0 гарантирует «правильно резолвится из любого cwd и понятно падает без моделей».

---

## Последующие шаги (roadmap, отдельными заходами)

| Шаг | Что | Опирается на |
|-----|-----|--------------|
| **14.1** ✅ | `mvm setup` — fetch моделей в `models_dir` с sha256-verify + HTTP range-resume; манифест (`manifest.rs`, 7/9 публичных URL, 2 YOLO project-hosted); авто-триггер обязательных на первом `run`. Live: fetch/idempotency/перекачка-битого/resume/auto-trigger ✓. **ЗАКРЫТ 2026-07-06** | `paths::models_dir` + `[models]` |
| **14.2** ✅ | Watched-галерея `gallery/<Имя>/*.jpg` (папка=имя, Unicode) + `mvm gallery add/list`; `build_gallery`→`gallery::load_into`, hardcoded name-map `[("V","Вадим")]` и пути убраны; YuNet-crop на загрузке. Live: Вадим мигрирован (7 фото) → распознан ✓. **ЗАКРЫТ 2026-07-06** | `[gallery]` |
| **14.3** ✅ | **`mvm serve --with-camera`** — единый процесс: `build_live_mesh` (общий с `run`) в фоновом std::thread + MCP на главном потоке над своей свежей `data_dir`; EOF stdin → stop-флаг + join. Live: snapshot-tool вернул свежий срез живого mesh (Вадим ✓, typing/reading). **ЗАКРЫТ 2026-07-06** | единая `data_dir` (14.0.4) |
| **14.4** | `mvm mcp install --host claude-desktop\|claude-code\|hermes` — пишет верный блок конфига (знает `data_dir`→`--logs`) | 14.0 + 14.3 |
| **14.5** | `mvm service install` — systemd **user**-юнит (`~/.config/systemd/user/mvm.service`), без root | 14.3 |
| **14.6** | Дистрибуция: GitHub Releases (linux x86_64/aarch64) + `curl … \| sh` installer | 14.1 |
| **later** | Кросс-бэкенды камеры (macOS → Windows) в шов `CameraSource`; авто-`fx` (zero-config 3D); **retention** scene/events-логов (`data_dir` растёт бесконечно — политика хранения или `mvm doctor`-warning + `mvm clean`) | швы `CameraSource`/`paths` |

Порядок: **фундамент → setup → галерея → единый демон → mcp-install → service → дистрибуция**;
кросс-платформа доклеивается в швы на любом этапе.

---
*Ядро зрения (фазы 8–13) не трогается. Продуктовые deps — только в `mvm-cli`. Инварианты №1–3 целы.*
