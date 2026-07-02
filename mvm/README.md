# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*

MVM превращает видеопоток в **метрический 3D scene-graph каждый кадр**: узлы-сущности
с 3D-позицией (origin = камера) и рёбра-отношения между ними. Не плоский список людей,
а граф «кто, в каком состоянии, где, и как они связаны в пространстве».

```bash
cd mvm
cargo build --release
./target/release/mvm-cli --native-camera --dump-json --dump-interval 3   # чистый Rust → Scene JSON
```

> **Статус (июнь 2026): фаза 8 — Agent-Mesh-ready. 197 тестов зелёные, 11 крейтов.**
> Основной путь `--native-camera` — **чистый Rust, ноль Python**: v4l2-захват →
> **body-primary** цепочка (тело — несущая сущность) → метрический 3D scene-graph.
> Нативны все детекторы: лицо (YuNet), identity (SFace), эмоция (EmotiEffLib),
> поза (BlazePose+PoseVAE), объекты (YOLO26n), глубина (Depth-Anything-V2), кисти
> (MediaPipe handpose). Группа детекторов **G₁ Human полна**; объекты получают
> метрическую 3D через depth-anchoring к позе человека. Следующее — **Agent Mesh**
> (вынос тяжёлых детекторов в потоки). Python-bridge (`--native`) остаётся для
> тестов/дебага. См. roadmap.

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
3. **Полный нативный Rust-путь.** Боевой hot-path (`--native-camera`) — целиком Rust,
   ноль Python (снято в фазе 7). Python-bridge остаётся отдельным путём для тестов/дебага.

Тяжёлый стек (оправдан): `ort` (ONNX inference, CPU) + `ndarray` + `image`.
Всё остальное (`anyhow`, `serde`, `chrono`) тривиально. Фундаментальные фичи
фазы 6 (Kalman, pinhole-3D, entropy, реляционная fusion, temporal PoE) **не
добавили ни одной зависимости**.

---

## Архитектура (`--native-camera`, body-primary, чистый Rust)

```
Камера /dev/video0  (нативный v4l2 MJPG mmap — ноль Python)
      │  RGB-кадр в памяти
      ▼  Rust mvm-cli  (--native-camera)            ← ВЕСЬ кластер нативный
 G₁ Human (несущая сущность = ТЕЛО, body-primary):
      ├─ BlazePose+PoseVAE → 33 landmarks → поза (6 кл) + torso-3D   (mvm-posture)
      ├─ YuNet         → лицо по nose-landmark тела → identity/emotion (mvm-detect)
      ├─ SFace         → identity 512d + cosine match + margin (галерея) (mvm-identity)
      ├─ EmotiEffLib B0 → emotion[8] + entropy + calibrated_conf       (mvm-emotion)
      └─ handpose      → 21 landmark/рука (ROI из запястий)            (mvm-hands)
 G₂ Scene (cadence /3):
      └─ YOLO26n       → объекты (80 кл COCO) + стабильный трекер       (mvm-scene)
 G₄ Environment (cadence /30):
      └─ Depth-Anything-V2 → относит. карта → depth-anchoring к позе    (mvm-depth)
      │
      ▼  Fusion (mvm-fusion, object-agnostic):
      ├─ PersonTracker  → Kalman bbox + identity re-link + reserved-id модель + temporal PoE
      ├─ build_scene()  → метрический 3D граф: узлы (люди+объекты) + рёбра + зоны (union-find)
      └─ выход:
           ├─ Scene JSON  (--dump-json)        ← машинный контракт для Hermes
           ├─ --scene-debug → /tmp/mvm_scene.jpg (все детекторы цветом своего агента)
           └─ StateFile  ~/.hermes/profiles/sight/workspace/current_state.md
```

**Всё нативно** в `--native-camera`. Bridge-путь (`--native`, Python мост) остаётся
для тестов/дебага (решение владельца — venv не сносится).

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
      "dist_m": 1.10, "pos": [0.93, 0.43, 1.10],
      "bbox_h": 0.35
    }
  ],
  "objects": [
    { "id": 10000, "class": "keyboard", "score": 0.61, "age": 40, "pos": [0.23, 0.56, 0.87], "bbox": [0.15,0.60,0.26,0.75] },
    { "id": 10001, "class": "cup",      "score": 0.82, "age": 58, "pos": [0.18, 0.65, 0.68], "bbox": [0.44,0.75,0.50,0.89] }
  ],
  "zones": [[0, 10000, 10001]],
  "relations": [
    { "a": 0, "b": 10000, "dist_m": 0.74, "overlap": 0.0, "kind": "near" }
  ]
}
```

- **`faces`** — люди (несущая сущность = тело; `id` 0–9 = пользователи галереи,
  ≥10 = анонимные тела). `emotion_distribution[8]` — полный вектор (Holo-Confidence):
  `Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise`. Downstream
  читает `emotion_conf`/энтропию, а не голую метку.
- **`objects`** — YOLO26n узлы (`id` ≥10000, стабильный трекер). `pos` ставится
  depth-anchoring'ом к позе человека; без depth — overlap-only.
- **`pos`** — `[x, y, z]` в метрах, origin = камера.
- **`zones`** — связные компоненты графа (контекстные зоны: «человек + объекты на столе»).
- **`relations`** — рёбра `{a, b, dist_m, overlap, kind}`, kind ∈ `near | overlapping | part_of`.
  Метрическое `near` строится только между узлами с реальной 3D (`pos_conf>0`).

---

## Крейты

| Крейт | Статус | Содержимое |
|-------|--------|------------|
| **mvm-core** | ✅ | типы, Config, PersonTracker (Kalman + identity re-link + reserved-id), Pinhole/DepthEma, EMOTION_LABELS + temporal PoE, Entity/Scene |
| **mvm-detect** | ✅ | YuNet ONNX (228 КБ), CHW BGR preprocessing, multi-stride NMS |
| **mvm-identity** | ✅ | SFace ONNX (37 МБ), L2-эмбеддинги, gallery cosine match + margin |
| **mvm-emotion** | ✅ | EmotiEffLib B0 ONNX (16 МБ), 8 классов VGAF, EmotionResult + entropy |
| **mvm-posture** | ✅ | нативный two-stage: person-det + BlazePose ONNX + hand-rolled PoseVAE/Classifier, мульти-persona |
| **mvm-scene** | ✅ | YOLO26n ONNX (80 кл COCO, end-to-end/raw decode) + class-aware NMS + стабильный ObjectTracker |
| **mvm-depth** | ✅ | Depth-Anything-V2-Small ONNX (относит. карта) + median-sample → depth-anchoring |
| **mvm-hands** | ✅ | MediaPipe handpose ONNX (21 landmark/рука) + ROI из запястий BlazePose |
| **mvm-fusion** | ✅ | `build_scene()` union-find граф (depth-gated near), `fuse()` PoE, Entity/Relation/Scene |
| **mvm-pipeline** | ✅ | NativeCamera (v4l2 MJPG mmap) + Camera (Python bridge subprocess) |
| **mvm-cli** | ✅ | main, body-primary chain, StateFile, Scene JSON, `--native-camera`/`--native`/`--dump-json`/`--scene-debug` |
| **xtask** | ✅ | `cargo xtask bundle` → `dist/mvm` (модели embedded) |

**Всего: 197 тестов, все зелёные.** Бинарник `target/release/mvm-cli`.

### Микро-агенты (группы детекторов)

| Группа | Агент | Модель | Сигнал | Лицензия |
|--------|-------|--------|--------|----------|
| **G₁ Human** | Pose | BlazePose+PoseVAE (нативный) | 33 landmarks → 6 поз + torso-3D | Apache 2.0 |
| | Face | YuNet ONNX 228 КБ | bbox лица (по nose-landmark тела) | MIT |
| | Identity | SFace ONNX 37 МБ | identity + confidence + margin | Apache 2.0 |
| | Emotion | EmotiEffLib B0 16 МБ | 8 эмоций + distribution + entropy | Apache 2.0 |
| | Hands | MediaPipe handpose 4 МБ | 21 landmark/рука | Apache 2.0 |
| **G₂ Scene** | Object | YOLO26n ONNX | 80 классов COCO + стабильный id | AGPL-3.0 |
| **G₄ Env** | Depth | Depth-Anything-V2-Small 27 МБ | относит. карта → метрич. объекты | Apache 2.0 |
| **3D** | Pinhole | нативный | (x,y,z)м + ws_confidence | — (чистая геометрия) |

---

## Сборка и запуск

```bash
# Сборка
cargo build --release

# ОСНОВНОЙ путь — чистый Rust (body-primary, объекты, глубина, кисти), ноль Python
./target/release/mvm-cli --native-camera --dump-json --dump-interval 3

# Визуальный дебаг — все детекторы цветом своего агента → /tmp/mvm_scene.jpg
./target/release/mvm-cli --native-camera --scene-debug --dump-interval 3

# Bridge-путь (Python мост) — для тестов/дебага
./target/release/mvm-cli --native --once --dump-json

# Все тесты
cargo test --workspace

# Один бинарник со встроенными моделями
cargo xtask bundle && ./dist/mvm --native-camera --once
```

`--native-camera` не требует Python. Bridge-путь (`--native`) поднимает
`../tools/mvm_bridge.py` из venv родительского проекта как subprocess (тесты/дебаг).
Большие ONNX-модели качаются отдельно (см. комментарии в `.gitignore`); только
мелкий YuNet (228 КБ) в репозитории.

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
- **Фаза 6** ✅ — Metric 3D Scene-Graph: Kalman bbox, нативная pinhole-3D + калибровка,
  entropy/uncertainty эмоций, реляционный граф + temporal PoE. + identity re-link.
- **Фаза 7 / 7.5** ✅ — снос Python с hot-path: `--native-camera` (v4l2 MJPG, ноль
  Python, ~25 fps), YuNet-детекция, SFace в трекер (margin + cosine re-link), нативная
  мульти-persona поза (BlazePose ONNX + hand-rolled PoseVAE).
- **Фаза 8** 🟡 (в работе) — **Agent-Mesh + новые группы детекторов + cross-group fusion**:
  - **8.0.1–8.0.4** ✅ — спайки mesh (threads-not-tokio), **body-primary** иерархия
    (тело — несущая сущность, FRONTIER закрыт), reserved-id identity-модель, стабилизация трека.
  - **8.E** ✅ — `mvm-scene`: YOLO26n объект-узлы → `build_scene`, cadence, фильтры + стабильный трекер.
  - **8.E.2** ✅ — `mvm-depth`: Depth-Anything-V2 → depth-anchoring объектов к позе → 3D-зоны.
  - **8.E.4** ✅ — `mvm-hands`: handpose из запястий → **G₁ Human полна (HumanAgent ready)**.
  - **Agent Mesh (8.A–8.D)** 🔜 — WorldState-типы, `mvm-mesh` (std::thread), HumanAgent,
    подкоманда `mesh`. Тяжёлые детекторы (depth ~230мс, hands) доказали head-of-line blocking.
  - **8.E.3 audio (YAMNet)** 🔜 — группа G₄.
- **Фаза 9** 🔜 — продукт: авто-калибровка fx, drag-and-drop галерея, one-click install, systemd.

### Известные ограничения (честно)

- **FPS 20→13 в `--native-camera`** при включённых depth+hands (синхронный цикл, тяжёлые
  детекторы блокируют) — это и есть head-of-line blocking, который снимет Agent Mesh (8.C).
- **Эмоция near-chance на потолочном ракурсе** — распределение почти равномерное; метка
  стабильна (temporal PoE), но `conf` честно низкая. Лучше при виде с уровня глаз.
- **Hand-ROI из запястья** неидеален (BlazePose hand-точки грубые) — кисти ловятся надёжно
  на чётком фронтальном жесте; rotation-from-pose ROI — будущий апгрейд.
- **Объекты под нестандартным ракурсом** (потолок/вверх ногами) — COCO out-of-distribution;
  лучший рычаг — угол камеры (вид с уровня глаз), не настройки детектора.
- **`torso_length_m` не откалиброван рулеткой** — ординальная 3D-глубина корректна, абсолютная
  приблизительна.

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
