# Фаза 14 — Продукт: нативная установка и использование

> **Версия:** 1.0 (2026-07-06). Детальный план **14.0 (фундамент)** + roadmap последующих шагов.
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

## 14.0 — Фундамент (детально)

### 14.0.1 — Модуль `paths` (XDG-шов)
Единая точка правды по путям. Hand-rolled XDG (`+0 dep`): `$XDG_DATA_HOME`/`$XDG_CONFIG_HOME`/
`$XDG_CACHE_HOME` с фолбэками `~/.local/share`, `~/.config`, `~/.cache`.
- `data_dir()` → `~/.local/share/mvm/` — **state + scene/events-логи + journal в одном месте**.
- `config_dir()` → `~/.config/mvm/`.
- `cache_dir()` → `~/.local/share/mvm/models/` (сюда потом `mvm setup`).

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

### 14.0.3 — Авто-детект камеры + шов `CameraSource`
Убрать хардкод `NativeCamera::open("/dev/video0", 1280, 720, false)` — он в **двух** местах
(`main.rs::run_mesh` и sync-путь ~`main.rs:796`).
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
- Убрать Hermes-хардкод целиком; `output.state_path` — опциональный override.
- **Hermes** репойнтится на новый XDG-путь на своей стороне (owner, вне репо) — dual-write и авто-детекта
  старого пути **нет** (решение владельца: чистый переезд).
- Одна `data_dir` кормит и `run`, и `serve` → подготовка к единому `serve --with-camera` (14.3).

### 14.0.5 — Команда называется `mvm`
`Cargo.toml` крейта `mvm-cli`: `[[bin]] name = "mvm"`. Пользователь пишет `mvm run`, не `mvm-cli run`.
`--help`/`AGENT.md`/README уже говорят `mvm` — приводим реальность в соответствие. `cargo xtask bundle`
и пути в доках сверить.

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
- **Live (чистая машина, нет `~/.hermes`):** `mvm run` **без флагов** — сам находит камеру, создаёт
  `~/.local/share/mvm`, пишет туда; `mvm snapshot` и `mvm serve --mcp` работают на дефолтах, **без ручного `--logs`**.
- `mvm doctor` даёт корректный отчёт. Команда зовётся `mvm`.

---

## Последующие шаги (roadmap, отдельными заходами)

| Шаг | Что | Опирается на |
|-----|-----|--------------|
| **14.1** | `mvm setup` — fetch-on-setup моделей в `cache_dir` (checksum, resume); авто-триггер на первом `run` при отсутствии | `paths::cache_dir` + `[models]` |
| **14.2** | Watched-галерея `gallery/<Имя>/*.jpg` + `mvm gallery add/list` — убрать hardcoded name-map/пути из `build_gallery` | `[gallery]` |
| **14.3** | **`mvm serve --with-camera`** — единый процесс: продюсер (mesh) + MCP-сервер над своими свежими артефактами | единая `data_dir` (14.0.4) |
| **14.4** | `mvm mcp install --host claude-desktop\|claude-code\|hermes` — пишет верный блок конфига (знает `data_dir`→`--logs`) | 14.0 + 14.3 |
| **14.5** | `mvm service install` — systemd **user**-юнит (`~/.config/systemd/user/mvm.service`), без root | 14.3 |
| **14.6** | Дистрибуция: GitHub Releases (linux x86_64/aarch64) + `curl … \| sh` installer | 14.1 |
| **later** | Кросс-бэкенды камеры (macOS → Windows) в шов `CameraSource`; авто-`fx` (zero-config 3D) | швы `CameraSource`/`paths` |

Порядок: **фундамент → setup → галерея → единый демон → mcp-install → service → дистрибуция**;
кросс-платформа доклеивается в швы на любом этапе.

---
*Ядро зрения (фазы 8–13) не трогается. Продуктовые deps — только в `mvm-cli`. Инварианты №1–3 целы.*
