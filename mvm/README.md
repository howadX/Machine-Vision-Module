<div align="center">

# MVM — Machine Vision Module

**Structured Decomposed Picture-to-Text** — локальный модуль зрения, который отдаёт **факты, а не пиксели.**

Ансамбль маленьких специалистов вместо одной end-to-end модели: камера → **метрический 3D scene-graph каждый кадр**, с confidence на каждый сигнал. CPU, $0, приватно, интерпретируемо.

![phase](https://img.shields.io/badge/phase-13%20closed-8A2BE2)
![tests](https://img.shields.io/badge/tests-548%20green-brightgreen)
![crates](https://img.shields.io/badge/crates-12-informational)
![runtime](https://img.shields.io/badge/runtime-CPU%20%2F%20%240-success)
![fusion](https://img.shields.io/badge/fusion-%2B0%20deps-success)
![rust](https://img.shields.io/badge/rust-edition%202024-orange)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

</div>

---

```bash
cd mvm && cargo build --release

# продакшн: живой mesh (камера + все модели), авто-детект камеры,
# state + durable-логи для recap/inspect → ~/.local/share/mvm
./target/release/mvm run

# что происходит прямо сейчас — одной командой
./target/release/mvm snapshot
```

```text
# Scene 17:51:23 · occupied · anomaly 0.00 · 18 fps
Zone «workstation»: Вадим(0), laptop(10001), mouse(10002)

| id | who     | dist | posture        | motion | emotion        | head   | hands       |
|----|---------|------|----------------|--------|----------------|--------|-------------|
| 0  | Вадим ✓ | 0.8m | sitting (high) | idle   | Neutral (high) | camera | using_mouse |
| 10 | anon    | 1.0m | standing (med) | idle   | —              | —      | —           |

Active: seated_at_workstation · using_mouse

⚠️ id10: тело не подтверждено лицом (животное или упавший — не различить)
```

Это не картинка и не свободный текст от VLM — это **детерминированный рендер** графа сцены: кто, в каком состоянии, где (в метрах), что делает, и **что модуль честно не смог подтвердить** (`⚠️`).

---

## Что это

MVM превращает видеопоток в граф: **узлы-сущности** с метрической 3D-позицией (origin = камера, pinhole от known-size априоров — не depth-сеть) и **рёбра-отношения** (близость, пересечение, зоны). Несущая сущность цепочки — **тело** (body-primary): человек трекается по телу; лицо, identity и эмоция — это *атрибуты* тела, которые переживают отворот головы и окклюзию.

Каждый сигнал несёт числовую уверенность, а неопределённость выражается **явно**, а не замалчивается. Модуль скорее скажет «позу withheld — видна только верхняя часть», чем угадает по невидимым ногам.

- **Латентность** — цель `<50 мс` на CPU, движок ~16–17 fps с человеком в кадре.
- **$0 и приватно** — ни рантайм, ни модели не требуют GPU, сети или внешних API.
- **Интерпретируемо** — любой вывод прослеживается до конкретного агента; агент заменяется = один крейт.

---

## 🤖 MVM как модуль зрения для LLM (pull-модель)

Главная фишка фаз 12–13: **агент (Hermes / Alter / Claude) берёт MVM в руки и сам решает, что спросить.** Три уровня, дёшево → дорого:

| Инструмент | Отвечает на | Стоимость | Когда |
|------------|-------------|-----------|-------|
| **`snapshot`** | что происходит **сейчас** | ~300 токенов | всегда начинай отсюда |
| **`recap`** | что **было** за период | логи → ~**190× сжатие** | сводка окна без чтения сырья |
| **`inspect`** | **сырые кадры** за утверждением | растёт с окном | адъюдикация того, что flagged snapshot/recap |

Два способа доступа:

```bash
# как файлы/CLI
mvm snapshot                                   # current_state.md
mvm recap   --input ~/vision/logs --last 10m
mvm inspect --input ~/vision/logs --around 2026-07-05T17:51:23 --window 5

# как MCP-инструменты (stdio JSON-RPC, read-only, +0 deps)
mvm serve --mcp        # → mvm_snapshot / mvm_recap / mvm_inspect
```

Всё, что нужно агенту, чтобы пользоваться MVM без чтения исходников, лежит в **[`AGENT.md`](AGENT.md)**: pull-модель, как читать таблицу snapshot, семантика id, `⚠️`-ноты честности, каденс-нюансы и чего MVM **не** делает.

---

## Почему не VLM (GPT-4V, Claude Vision, LLaVA)

VLM универсальна, но платит за это скоростью, ценой, приватностью и интерпретируемостью. MVM — не замена, а **дешёвый, приватный, детерминированный первый проход**, который закрывает ~80% бытовых сценариев за ~0.1% стоимости VLM. Открытый мир / zero-shot / QA → VLM как fallback.

| | MVM | VLM |
|---|-----|-----|
| **Латентность** | `<50 мс` (цель), CPU | 2–30 с, GPU |
| **Локальность** | всё на CPU, без сети | GPU + облако |
| **Стоимость** | $0 | $0.01–0.05 / запрос |
| **Приватность** | абсолютная (ничего не уходит) | кадр уходит на сервер |
| **Confidence** | каждый сигнал: identity 0.92, emotion 0.78 | нет |
| **Интерпретируемость** | любой сигнал прослеживаем до агента | чёрный ящик |
| **Заменяемость** | замена агента = один крейт | переобучение всей модели |
| **Детерминизм** | один вход → один выход | разный текст на один кадр |

**Где MVM на месте:** умный дом / безопасность (тот же CPU, что Home Assistant), промышленная аналитика (N камер × $0 → Grafana), робототехника (control-loop `<100 мс`), медицина / accessibility (приватность absolute), edge/embedded (один бинарник, Raspberry Pi 5 / Jetson / x86_64).

---

## Три инварианта

Каждое решение проверяется этими правилами. Нарушает — не входит в MVM.

1. **Минимализм / чистая арифметика.** Fusion, трекинг, 3D, uncertainty — скалярная арифметика над данными, уже лежащими в памяти. **Ноль новых зависимостей** на интеллектуальную фичу; footprint растёт только с новым ONNX-экспертом. Вся полировка fusion (фазы 9–13) — `+0 deps`.
2. **CPU-first, $0.** Ни рантайм, ни модели не требуют GPU или внешних API.
3. **Полный нативный Rust-путь.** Боевой hot-path — целиком Rust, ноль Python. Python-bridge остаётся отдельным путём для тестов/дебага.

Тяжёлый стек (оправдан): `ort` (ONNX inference, CPU) + `ndarray` + `image`. Остальное (`anyhow`, `serde`, `chrono`, `serde_json`) тривиально.

---

## Архитектура (`mvm run` — Agent Mesh, body-primary, чистый Rust)

```
Камера /dev/video0  (нативный v4l2 MJPG mmap — ноль Python; eye-level ~175см, ~горизонтально)
      │  RGB-кадр в памяти
      ▼  Agent Mesh на std::thread:  Capture → {Human, Scene, Depth} → Fusion → StateFileWriter
 HumanAgent (несущая сущность = ТЕЛО):
      ├─ BlazePose → 33 lm → 14 вьюпойнт-инвариантных фич → RandomForest → поза
      │                        {standing | sitting | lying} + torso-3D   (VAE → OOD-гейт)
      ├─ YuNet → лицо по nose-landmark тела → SFace identity + EmotiEffLib emotion[8]+entropy
      └─ handpose → 21 lm/рука (ROI из запястий) → жесты + взаимодействие с объектами
 SceneAgent (cadence): YOLO26n → объекты (80 кл COCO) + стабильный трекер
      └─ YOLO-seg (opt-in --masks) → инстанс-маски → mask-контакт + mask-median depth
 DepthAgent (cadence): Depth-Anything-V2 → относит. карта → depth-anchoring к торс-Z
      │
      ▼  FusionAgent (реляционная, object-agnostic):
      ├─ build_scene() → метрический 3D граф: узлы + рёбра + зоны (union-find, persistence)
      ├─ activity-FSM  → стабильные активности + event-journal (start/stop/duration)
      └─ anomaly v2    → noisy-OR (нежданный объект + поза-OOD + падение standing→lying)
      │
      ▼  Выходы:
         ├─ current_state.md      (snapshot v2 — таблица + ноты честности)
         ├─ scene-/events-*.jsonl (durable-логи → recap/inspect)
         ├─ journal.md            (rolling-рекап дня)
         ├─ Scene JSON            (--dump-json — машинный контракт)
         └─ MCP stdio             (serve --mcp — snapshot/recap/inspect как инструменты)
```

**Body-primary** (FRONTIER): человек без фронтального лица трекается по телу; identity/emotion дотягиваются при появлении лица. `mesh` = продакшн-путь; синхронный `--native-camera` и Python-bridge `--native` живы для тестов/истории.

---

## CLI

Когерентное дерево подкоманд с `--help` на каждой (фаза 13):

```bash
mvm run       [--scene-log <DIR> --masks --dump-json --interval-ms N …]   # живой mesh (продакшн)
mvm snapshot  [--state <PATH>]                                            # что сейчас
mvm recap     [--input <PATH> --last <DUR> | --from/--to <ISO>]           # что было за период
mvm inspect   [--input <PATH> --around <ISO> --window <SECS> --id N --raw] # сырые кадры
mvm serve     --mcp [--state --logs]                                      # MCP stdio-сервер

mvm --help          # обзор + pull-модель
mvm <cmd> --help    # флаги с типами, единицами и примерами
mvm --version
```

> `mvm mesh` — скрытый алиас `mvm run`. `--native-camera` / `--native` — скрытый legacy-путь (живой).

---

## Выходной контракт — Scene JSON

`--dump-json` отдаёт одну JSON-строку на кадр — стабильный контракт для downstream (Hermes / Grafana / робот). F32 округляются (токен-диета); тяжёлые поля (`emotion_distribution`, object↔object `near`, сырые лендмарки) — за опт-ин флагами.

```json
{
  "ts": "2026-07-05T17:51:23+03:00",
  "fps": 18.0,
  "status": "occupied",
  "faces": [
    {
      "id": 0,
      "identity": "Вадим", "identity_conf": 1.0, "face_verified": true,
      "emotion": "Neutral", "emotion_conf": 0.91, "emotion_entropy": 0.18,
      "posture": "sitting", "posture_conf": 0.88, "body_coverage": "full",
      "motion_state": "idle", "head_facing": "camera",
      "dist_m": 0.82, "pos": [0.31, 0.12, 0.82]
    }
  ],
  "objects": [
    { "id": 10001, "class": "laptop", "score": 0.79, "age": 40, "pos": [0.2, 0.5, 0.9], "mask_coverage": 0.16 },
    { "id": 10002, "class": "mouse",  "score": 0.62, "age": 58, "pos": [0.3, 0.6, 0.8], "remembered": true }
  ],
  "zones": [[0, 10001, 10002]],
  "relations": [ { "a": 0, "b": 10002, "dist_m": 0.24, "overlap": 0.0, "kind": "near" } ],
  "activities": [
    { "label": "seated_at_workstation", "state": "active", "conf": 0.91, "duration_s": 12.6,
      "participants": [0], "stable_participants": true }
  ],
  "anomaly_score": 0.0
}
```

- **`faces`** — люди (несущая сущность = тело). `id` **0–9** = пользователи галереи, **≥10** = анонимные тела. `face_verified` отличает подтверждённое лицом тело от «животное/упавший». `emotion_distribution[8]` и сырые лендмарки — за `--dump-distributions` / `--dump-landmarks`.
- **`objects`** — YOLO26n узлы (`id` **≥10000**, стабильный трекер). `pos` — depth-anchoring'ом к позе; `mask_coverage` при `--masks`; `remembered` = статичная фикстура, удержанная под окклюзией руками (object-permanence).
- **`pos`** — `[x, y, z]` в метрах, origin = камера. Ординал надёжен, абсолют ±15–30%.
- **`zones`** — связные компоненты графа («человек + объекты на столе»).
- **`activities`** — активности с temporal-полями (`state`/`duration_s`) из activity-FSM; `stable_participants=false` честно маркирует эфемерных участников.

---

## Крейты (12)

| Крейт | Содержимое |
|-------|------------|
| **mvm-core** | типы, Config, PersonTracker (Kalman + identity/cosine re-link + reserved-id), Pinhole/space (+`ScaleCalibrator`), WorldState (блэкборд), `assignment` (Hungarian), `activity` (FSM+journal), `anomaly` (v2), `mask`, `recap` |
| **mvm-detect** | YuNet ONNX (228 КБ) — лицо + 5 лендмарков |
| **mvm-identity** | SFace ONNX (37 МБ), 128d L2-эмбеддинг, галерея + margin |
| **mvm-emotion** | EmotiEffLib B0 ONNX (16 МБ), 8 классов VGAF + entropy |
| **mvm-posture** | нативная BlazePose (two-stage) + feature→RandomForest (method-C, 3 класса) + VAE (OOD-only), мульти-persona |
| **mvm-scene** | YOLO26n ONNX (объекты) + ObjectTracker + YOLO-seg (`--masks`) + object-permanence статичных фикстур |
| **mvm-depth** | Depth-Anything-V2-Small ONNX → depth-anchoring к торс-Z |
| **mvm-hands** | MediaPipe handpose ONNX (21 lm/рука), ROI из запястий |
| **mvm-fusion** | `build_scene` (union-find граф + зоны) + PoE; `ZoneTracker` (persistence) |
| **mvm-mesh** | Agent trait + `LatestFrame` + `AgentMesh` оркестратор на std::thread |
| **mvm-pipeline** | NativeCamera (v4l2 MJPG) + Camera (Python bridge subprocess) |
| **mvm-cli** | подкоманды `run`/`snapshot`/`recap`/`inspect`/`serve`, рендер snapshot/recap, MCP-сервер, StateFile |

**580 тестов, все зелёные.** Бинарник `target/release/mvm` (~29 МБ; embed — только мелкие
модели YuNet/BlazePose/RF, крупные ONNX грузятся из models-dir — см. «Сборка и запуск»).

### Микро-агенты и модели

| Группа | Агент | Модель | Сигнал | Лицензия |
|--------|-------|--------|--------|----------|
| **Human** | Pose | BlazePose + RandomForest (нативный) | 33 lm → {standing/sitting/lying} + torso-3D | Apache-2.0 |
| | Face | YuNet ONNX 228 КБ | bbox + 5 лендмарков лица | MIT |
| | Identity | SFace ONNX 37 МБ | identity + confidence + margin | Apache-2.0 |
| | Emotion | EmotiEffLib B0 16 МБ | 8 эмоций + distribution + entropy | Apache-2.0 |
| | Hands | MediaPipe handpose 4 МБ | 21 lm/рука + жесты | Apache-2.0 |
| **Scene** | Object | YOLO26n ONNX | 80 классов COCO + стабильный id | AGPL-3.0 |
| | Seg | YOLO-seg (opt-in `--masks`) | инстанс-маски → контакт + depth | AGPL-3.0 |
| **Env** | Depth | Depth-Anything-V2-Small 27 МБ | относит. карта → метрич. объекты | Apache-2.0 |
| **3D** | Pinhole | нативный | (x,y,z) м + ws_confidence | — (чистая геометрия) |

---

## Сборка и запуск

```bash
cargo build --release          # → target/release/mvm
cargo test --workspace         # 580 тестов
cargo xtask bundle             # → dist/mvm (embed мелких моделей: YuNet + BlazePose + RF/VAE)

# Модели: sha256-проверенный fetch в models-dir (7/9 из публичных источников)
./target/release/mvm setup

# ОСНОВНОЙ путь — живой mesh, чистый Rust, ноль Python. Камера — авто-детект;
# state + логи + journal → ~/.local/share/mvm. Обязательные модели дотянутся сами.
./target/release/mvm run

# живой JSON с инстанс-масками
./target/release/mvm run --masks --dump-json

# model-free CI-smoke (без камеры/моделей)
./target/release/mvm run --stub --max-frames 100

# Python-bridge (для тестов/дебага)
./target/release/mvm --native --once --dump-json
```

**Конфиг:** `~/.config/mvm/config.toml` (`[camera] device`, `[output] data_dir`, `[models] dir`,
`[cadence] snapshot_ms`) + env `MVM_*`; флаги CLI переопределяют (фаза 14.0).

**Камера:** `run` использует прямой v4l2 MJPG mmap через шов `CameraSource` (`device = "auto"` —
перебор `/dev/video*`). **Модели:** `mvm setup` тянет их в models-dir с проверкой SHA-256 (7 из 9
из публичных источников — OpenCV Zoo / HuggingFace; 2 YOLO пока project-hosted, добавляются вручную).
`mvm run` при первом запуске сам дотягивает обязательные модели. Dev-чекаут читает `./models`;
embed (`cargo xtask bundle`) покрывает только мелкие YuNet/BlazePose/RF.

---

## 3D-калибровка

Глубина — pinhole-геометрией: `z = fx · real_size / pixel_size`, чистая арифметика без depth-сети. `Depth-Anything` даёт относительную карту, привязанную к метрической торс-Z (`z·d ≈ k`) — так объекты получают метрику в той же системе.

- **Камера** — eye-level (~175 см, ~горизонтальна), **не потолок**. Абсолют 3D ±15–30% (`torso_length_m` не откалиброван рулеткой); **ординал и относительное — надёжны** («кто ближе» робастно).
- `ScaleCalibrator` (self-calib, фаза 11) восстанавливает масштаб по known-size референсу; авто-`fx` — продуктовая фаза.

---

## Roadmap

| Фаза | | Суть |
|------|---|------|
| **5** | ✅ | Стабилизация (YuNet preprocessing, emotion fallback, галерея, FFmpeg backend) |
| **6** | ✅ | Metric 3D Scene-Graph (Kalman bbox, pinhole-3D, entropy, `build_scene` + PoE) |
| **7 / 7.5** | ✅ | Python-light hot-path (`--native-camera`, ноль Python) + нативная мульти-persona поза |
| **8** | ✅ | Body-primary + Agent-Mesh (`mesh`) + fusion (activities / anomaly) |
| **9** | ✅ | Tracker — кинематика + ByteTrack/OCM/OCR + lifecycle + cadenced SFace + **Hungarian** assignment |
| **10** | ✅ | Full Data Collection (эмоция-корень, кисти, head-orientation) + **Segmentation** (+1 ONNX YOLO-seg) |
| **11** | ✅ | Fusion-полировка — activity-FSM + journal, зоны-persistence, anomaly v2, mask-контакт, object-permanence, self-calib |
| **12** | ✅ | **LLM-render** — pull-модель (snapshot / recap ~190× / inspect), токен-диета, MCP stdio-сервер |
| **13** | ✅ | **Agent Tools/Skills Polishing** — когерентный CLI-дерево, `--help`, [`AGENT.md`](AGENT.md), самодокументирующие MCP-описания |
| **14+** | 🔜 | **Продукт** — авто-калибровка fx, drag-and-drop галерея, one-click install, systemd + cron |

---

## Честные ограничения

- **Walking-фрагментация** — ходящий человек может дробиться на фантомные треки → ложные `people_together` (честно `stable=false`). Person-маски (11.A3, `--masks`, ≥2 чел.) лечат основную часть; остаточный фантом ужимается гейтами.
- **Абсолют 3D приблизителен** (`torso_length_m`/`fx` не откалиброваны рулеткой) — ординал/относительное надёжны, метрика ±15–30%.
- **Объекты под нестандартным ракурсом** — COCO out-of-distribution; лучший рычаг — угол камеры, не пороги детектора.
- **`fps` в StateFile** = темп **доставки** кадров, не темп обработки человеко-цепочки.
- **Фантом ≠ человек**, пока `inspect` не подтвердит когерентный трек — snapshot честно ставит `⚠️`.

---

## Контракт модуля

```rust
trait Module {
    type Input;
    type Output;
    fn init(config: &Config) -> Result<Self>;
    fn process(&mut self, input: &Self::Input) -> Result<Self::Output>;
    fn metadata(&self) -> ModuleMetadata;
}
```

## Терминология

- **Микро-агент** — независимый детектор с confidence. **Fusion** — реляционная (граф/зоны, основная) + атрибутивная (PoE, частный случай).
- **Body-primary** — несущая сущность цепочки = тело (не лицо). **Reserved-id** — id 0–9 за галереей, ≥10 анонимам, ≥10000 объектам.
- **Pull-модель** — snapshot (сейчас) → recap (период) → inspect (сырьё); дёшево-раньше.
- **Confidence-first** — каждый сигнал несёт числовую уверенность; неопределённость выражается явно (`⚠️`).

## Лицензия

Apache-2.0. Модели — под своими лицензиями (см. таблицу выше; YOLO26n/YOLO-seg — AGPL-3.0).
