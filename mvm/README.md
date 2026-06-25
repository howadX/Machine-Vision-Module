# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*

MVM превращает видеопоток в **метрический 3D scene-graph каждый кадр**: узлы-сущности
с 3D-позицией (origin = камера) и рёбра-отношения между ними. Не плоский список людей,
а граф «кто, в каком состоянии, где, и как они связаны в пространстве».

```bash
cd mvm
cargo build --release
./target/release/mvm-cli --native --once --dump-json   # один кадр → Scene JSON
```

> **Статус:** фаза 6 (Metric 3D Scene-Graph) закрыта и провалидирована на живой
> камере (июнь 2026). 106 тестов зелёные. Боевой инференс (emotion, identity, 3D,
> трекинг, fusion) — нативный Rust. Захват камеры, первичная детекция и поза пока
> приходят из Python-моста — их снос в нативный Rust это **фаза 7** (см. roadmap).

---

## Почему MVM, а не VLM (GPT-4V, Claude Vision, LLaVA)?

Vision-Language Models делают всё: фото → текст. Но за универсальность платят
скоростью, ценой, приватностью и интерпретируемостью.

**MVM — ансамбль специализированных микро-агентов.** Каждый обучен на одну задачу
и выдаёт структурированный сигнал с confidence. Итоговое описание собирается
fusion-логикой — не генерацией, а арифметической сборкой.

| | MVM | VLM |
|---|-----|-----|
| **Латентность** | <50 мс (цель), CPU | 2–30 с, GPU |
| **Локальность** | всё на CPU, без сети | GPU + облако |
| **Стоимость** | $0 | $0.01–0.05 / запрос |
| **Приватность** | абсолютная (ничего не уходит) | фото на сервер |
| **Confidence** | каждый сигнал: identity 0.92, emotion 0.78 | нет |
| **Интерпретируемость** | любой сигнал прослеживаем до агента | чёрный ящик |
| **Заменяемость** | замена агента = один крейт | переобучение всей модели |
| **Детерминизм** | один вход → один выход | разный текст на одно фото |

**MVM не замена VLM — это инструмент для других задач.** Для открытого описания
сцены («опиши комнату», незнакомые предметы) сильнее VLM. Для structured-мониторинга
real-time (кто / в каком состоянии / где) — MVM. Гибрид: MVM на 80% кадров, VLM
по запросу на 20% сложных сцен.

**Где MVM на месте:** умный дом / безопасность (тот же CPU, что Home Assistant),
промышленная аналитика (N камер × $0 → Grafana), робототехника (control loop
требует <100 мс), медицина/accessibility (приватность absolute), edge/embedded
(один бинарник, Raspberry Pi 5 / Jetson / x86_64).

---

## Фундаментальный стек (3 инварианта)

Каждое архитектурное решение проверяется этими тремя правилами. Нарушает — не входит в MVM.

1. **Минимализм / чистая арифметика.** Fusion, трекинг, 3D, uncertainty — это
   умножения/деления/скалярные фильтры над данными, уже лежащими в памяти.
   **Ноль новых зависимостей** на каждую новую интеллектуальную фичу. Footprint
   растёт только когда добавляется новый ONNX-эксперт.
2. **CPU-first, $0.** Ни рантайм, ни модели не требуют GPU и внешних API.
3. **Полный нативный Rust-путь.** Python-модели — прототипы-леса; боевой кластер
   целиком на Rust. Python планомерно сносится (фаза 7).

Тяжёлый стек (оправдан): `ort` (ONNX inference, CPU) + `ndarray` + `image`.
Всё остальное (`anyhow`, `serde`, `chrono`) тривиально. Фундаментальные фичи
фазы 6 (Kalman, pinhole-3D, entropy, реляционная fusion, temporal PoE) **не
добавили ни одной зависимости**.

---

## Архитектура (текущий пайплайн)

```
Камера /dev/video0  (cv2 CAP_FFMPEG — обход PipeWire на Ubuntu 24+; ROTATE_180)
      │
      ▼  Python bridge (mvm_bridge.py)        ← прототип-леса, сносится в фазе 7
      ├─ SCRFD          → bbox лица            (первичная детекция)
      ├─ MediaPipe Pose → 33 лендмарка → PoseVAE → поза (6 кл)
      └─ кадр на диск (--write-frame)
      │  JSON line + кадр
      ▼  Rust mvm-cli  (--native)             ← БОЕВОЙ нативный кластер
      ├─ YuNet          → детекция/кроп лица   (mvm-detect, ONNX)
      ├─ SFace          → identity 512d + cosine match (галерея)
      ├─ EmotiEffLib B0 → emotion[8] + entropy + calibrated_conf
      ├─ Pinhole 3D     → (x,y,z)м из bbox + known-size приор  (space.rs)
      ├─ PersonTracker  → Kalman bbox (6.1) + identity re-link + temporal emotion PoE (6.4b)
      ├─ build_scene()  → граф близости + зоны + рёбра (6.4a)
      └─ выход:
           ├─ Scene JSON  (--dump-json)        ← машинный контракт для Hermes
           ├─ ASCII-таблица (stderr)
           └─ StateFile  ~/.hermes/profiles/sight/workspace/current_state.md
```

**Что уже нативно:** emotion, identity, 3D, трекинг (Kalman + re-link), fusion.
**Что ещё Python (фаза 7):** захват камеры, первичный bbox (SCRFD), поза.

---

## Выходной контракт — Scene JSON

`--dump-json` отдаёт одну JSON-строку на кадр. Это стабильный контракт для downstream
(Hermes / Grafana / робот):

```json
{
  "ts": "2026-06-25T14:34:43+03:00",
  "fps": 1.56,
  "status": "occupied",
  "faces": [
    {
      "id": 0,
      "identity": "Вадим", "identity_conf": 1.0,
      "emotion": "Surprise", "emotion_conf": 0.50,
      "emotion_distribution": [0.05,0.13,0.10,0.26,0.07,0.13,0.09,0.18],
      "posture": "сидит_за_ПК", "posture_conf": 0.44,
      "dist_m": 2.34, "pos": [-0.08, 0.39, 2.34],
      "bbox_h": 0.12
    }
  ],
  "zones": [[0]],
  "relations": []
}
```

- **`emotion_distribution[8]`** — полный вектор (Holo-Confidence). Классы (порядок):
  `Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise`.
  Downstream обязан читать `emotion_conf` / энтропию, а не голую метку (на низком
  conf модель не уверена).
- **`pos`** — `[x, y, z]` в метрах, origin = камера, pinhole-геометрия.
- **`zones`** — связные компоненты графа близости (контекстные зоны).
- **`relations`** — рёбра `{a, b, dist_m, overlap, kind}`, kind ∈ `near | overlapping | part_of`.

---

## Крейты

| Крейт | Статус | Содержимое | Тесты |
|-------|--------|------------|-------|
| **mvm-core** | ✅ | типы, Config, PersonTracker (Kalman + identity re-link), Pinhole/DepthEma, EMOTION_LABELS + temporal PoE, Entity/Scene | 48 |
| **mvm-detect** | ✅ | YuNet ONNX (228 КБ), CHW BGR preprocessing, multi-stride NMS | 5 |
| **mvm-identity** | ✅ | SFace ONNX (37 МБ), L2-эмбеддинги, gallery cosine match | 5 |
| **mvm-emotion** | ✅ | EmotiEffLib B0 ONNX (16 МБ), 8 классов VGAF, EmotionResult + entropy | 7 |
| **mvm-posture** | 🔶 заглушка | пустой Module trait (поза от Python, нативная — фаза 7) | 0 |
| **mvm-fusion** | ✅ | `build_scene()` union-find граф, `fuse()` PoE, Entity/Relation/Scene | 9 |
| **mvm-pipeline** | ✅ | Camera (Python bridge subprocess), `parse_line` (+ интеграционные) | 5 (+9) |
| **mvm-cli** | ✅ | main, StateFile, Scene JSON, `--native`/`--once`/`--dump-json`/`--emotion-debug` | 17 |
| **xtask** | ✅ | `cargo xtask bundle` → `dist/mvm` (модели embedded) | 1 |

**Всего: 106 тестов, все зелёные.** Бинарник `target/release/mvm-cli` ≈ 26 МБ.

### Микро-агенты

| Агент | Модель | Сигнал | Лицензия |
|-------|--------|--------|----------|
| **Detect** | YuNet ONNX 228 КБ | bbox лица | MIT |
| **Identity** | SFace ONNX 37 МБ | identity + confidence | Apache 2.0 |
| **Emotion** | EmotiEffLib B0 16 МБ | 8 эмоций + distribution + entropy | Apache 2.0 |
| **Posture** | PoseVAE (Python) | 6 поз + confidence + plausibility | собственная |
| **3D** | Pinhole (нативный) | (x,y,z)м + ws_confidence | — (чистая геометрия) |

---

## Сборка и запуск

```bash
# Сборка
cargo build --release

# Полный пайплайн через Python bridge (детекция + identity + emotion + поза + 3D)
./target/release/mvm-cli --once

# Нативный Rust-инференс (emotion/identity/3D в Rust; камера/детекция/поза — bridge)
./target/release/mvm-cli --native --once

# Machine-readable Scene JSON, по кадру
./target/release/mvm-cli --native --dump-json --dump-interval 1

# A/B эмоций: EmotiEffLib (Rust) vs HSEmotion (Python)
./target/release/mvm-cli --native --once --emotion-debug

# Все тесты
cargo test --workspace

# Один бинарник со встроенными моделями
cargo xtask bundle && ./dist/mvm --once
```

Гибридный запуск требует Python-venv родительского проекта
(`../venv`, `../tools/mvm_bridge.py`) — `Camera::spawn()` поднимает его как subprocess.

---

## 3D-калибровка

Глубина считается pinhole-геометрией: `z = fx · real_size / pixel_size` — чистая
арифметика, без depth-сети. `real_size` (`Config.face_height_m`) — **эффективная
bbox-высота лица**, эмпирически откалиброванная под камеру и детектор.

```bash
# Снять замеры на известных дистанциях (рулетка) и вычислить константу
venv/bin/python3 ../tools/calib_3d.py --distance 1.0
venv/bin/python3 ../tools/calib_3d.py --distance 2.0
venv/bin/python3 ../tools/calib_3d.py --report
```

**Конверт точности** (этой камеры, потолочный монтаж, калибровка 2026-06-25):
`<3%` ошибки на 0.5–2.0 м; на дальнем краю (2.8 м) недооценка ~8%. Тир — **грубый
метрический + сильный относительный субстрат** (кто ближе — робастно). Константа
привязана к bbox-конвенции текущего детектора → **рекалибровать при смене детектора**.
Полный разбор — в docstring `crates/mvm-core/src/space.rs`.

---

## Тестирование на камере

`tools/cam_test.py` — сценарный харнесс: снимает именованную сессию в `mvm/logs/`
и печатает вердикт (FPS, стабильность трека, identity, 3D-дистанция, scene-graph).

```bash
venv/bin/python3 ../tools/cam_test.py --list                 # сценарии
venv/bin/python3 ../tools/cam_test.py --scenario two_people  # 6.4a граф
venv/bin/python3 ../tools/cam_test.py --replay logs/cam_two_people.jsonl
```

Окно записи стартует от первого реального кадра (≈12 с молчит — грузятся модели),
поэтому действие выполняется непрерывно весь интервал.

---

## Roadmap

- **Фаза 5** ✅ — стабилизация, A/B эмоций, dump-json, интеграционные тесты.
- **Фаза 6** ✅ — Metric 3D Scene-Graph: Kalman bbox (6.1), нативная pinhole-3D
  (6.2) + калибровка, entropy/uncertainty эмоций (6.3), реляционный граф +
  temporal PoE (6.4). Провалидирована на камере. + identity re-link.
- **Фаза 7** 🔜 — **снос Python**: YuNet как первичный детектор (+ рекалибровка),
  нативный захват камеры (риск — PipeWire), нативная мульти-persona поза (ONNX +
  порт PoseVAE), удаление bridge → один бинарник, SFace-эмбеддинг в трекер
  (margin-identity + cosine-re-link). Детальный план — в `CLAUDE.md`.
- **Фаза 8** 🔜 — продукт: авто-калибровка fx, drag-and-drop галерея,
  one-click install, systemd service.

### Известные ограничения (честно)

- **FPS ~1–2** в гибридном режиме — узел Python bridge (фаза 7 снимает).
- **Эмоция near-chance на потолочном ракурсе** — распределение почти равномерное;
  temporal-fusion убрал джиттер, метка стабильна, но `conf` честно низкая.
- **Поза только у 1 человека** — MediaPipe возвращает один pose; мульти-persona =
  фаза 7.
- **identity false-positive при галерее из 1 человека** — чужой матчится как
  ближайший; нужен margin/threshold (фаза 7).

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

- **Микро-агент** — независимый детектор с confidence.
- **Structured Decomposed PTT** — архитектурная идентичность MVM.
- **Fusion** — реляционная (граф/зоны, основная) + атрибутивная (PoE эмоций, частный случай).
- **Workspace 3D** — метрическая система координат в метрах, origin = камера.
- **Confidence-first** — каждый сигнал имеет числовую уверенность; Holo-Confidence — весь вектор наружу.

## Лицензия

Apache 2.0
