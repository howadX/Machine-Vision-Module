# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*

Один бинарник. Скачал — запустил — работает. Без Python, без pip, без GPU.

```bash
git clone .../mvm && cd mvm
cargo build --release
./target/release/mvm-cli --native --once
```

## Почему MVM, а не VLM (GPT-4V, Claude, LLaVA)?

Vision-Language Models делают всё: фото → текст. Но платят универсальностью за скорость, цену и приватность.

**MVM — ансамбль из 5 микро-агентов.** Каждый специализирован на одной задаче и выдаёт сигнал с confidence. Итоговое описание собирается fusion-логикой — не генерацией, а сборкой.

### В чём MVM опережает конкурентов

| Преимущество | MVM | VLM |
|---|---|---|
| **Латентность** | <50ms (30fps на CPU) | 2-30s |
| **Локальность** | Всё на CPU, без сети | GPU + облако |
| **Стоимость** | $0 | $0.01-0.05/запрос |
| **Приватность** | Абсолютная | Фото уходит на сервер |
| **Confidence** | Каждый сигнал: identity (0.92), emotion (0.78) | Нет |
| **Интерпретируемость** | Любой сигнал прослеживаем до агента | Чёрный ящик |
| **Заменяемость** | Замена одного агента — один крейт | Переобучение всей модели |
| **Детерминизм** | Один вход → один выход | Разный текст на одно фото |

### Для каких стеков MVM подходит лучше всего

#### 1. Умный дом / безопасность
`mvm` работает на том же CPU что и Home Assistant. 30fps, локально, без интернета.
Идентифицирует жильцов, определяет эмоции, позу, положение в комнате.
Всё в structured-формате → прямой фид в автоматизацию.

#### 2. Промышленная аналитика
N камер × $0/мес × structured data → Grafana / ClickHouse.
С VLM: N × $0.05 × 60 × 24 × 30 = **$2,160/мес за одну камеру**.

#### 3. Робототехника
Control loop требует <100ms. VLM отвечает 2-30 секунд.
MVM выдаёт position (x,y,z)м с confidence — робот реагирует мгновенно.

#### 4. Медицина / Accessibility
Локальное определение эмоций и позы для пациентов.
Приватность absolute — ни одно фото не покидает устройство.

#### 5. Edge / Embedded
Один бинарник (`cargo xtask bundle`) — все модели внутри.
Raspberry Pi 5, Jetson Nano, любой x86_64 Linux.

### Когда VLM всё же лучше

MVM не замена VLM. Это инструмент для **других задач**:
- MVM: **знает 5 вещей, но мгновенно и точно**
- VLM: **знает всё, но медленно и дорого**

Для открытого описания сцены (незнакомые предметы, «опиши комнату») — VLM.
Для structured мониторинга (кто, в каком состоянии, где) — **MVM**.

### Стратегия: гибрид

```
Камера → MVM (real-time, 30fps, $0)
              ├── 80% сценариев → StateFile (MVM сам)
              └── 20% сложных → VLM по запросу
```

## Архитектура

```
Кадр ──→ [YuNet]        → bbox лица
     ├─→ [SFace]        → identity + confidence
     ├─→ [EmotiEffLib]  → emotion (8 кл) + confidence
     ├─→ [PoseVAE]      → posture (6 кл) + confidence
     └─→ [Workspace3D]  → position (x,y,z)m
                         ↓
              [PersonTracker] → temporal fusion
                         ↓
              [StateFile] → current_state.md
```

## Микро-агенты

| Агент | Модель | Сигнал | Лицензия |
|-------|--------|--------|----------|
| **Detect** | YuNet ONNX 228KB | bbox лица | MIT |
| **Identity** | SFace ONNX 37MB | identity + confidence | Apache 2.0 |
| **Emotion** | EmotiEffLib B0 16MB | 8 эмоций + confidence | Apache 2.0 |
| **Posture** | PoseVAE 210KB | 6 поз + confidence + plausibility | Собственная |
| **3D** | Workspace3D | (x,y,z)m + ws_confidence | Собственная |

## Метрики

| Метрика | Значение |
|---------|----------|
| Крейтов | 8 + xtask |
| Тестов | 48 |
| ONNX-моделей | 3 (YuNet 228KB + SFace 37MB + EmotiEffLib 16MB) |
| Инференс (CPU) | <50ms |
| Бинарник | 717K (модели отдельно) / ~80MB (bundle) |

## Быстрый старт

```bash
# Сборка
cargo build --release

# Python bridge (полный пайплайн: детекция + identity + emotion + posture + 3D)
./target/release/mvm-cli --once

# Rust-native inference (детекция + identity + emotion — без Python)
./target/release/mvm-cli --native --once

# Все тесты
cargo test --workspace

# Один бинарник со встроенными моделями
cargo xtask bundle
dist/mvm --once
```

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

- **Микро-агент** — независимый детектор с confidence
- **Structured Decomposed PTT** — архитектурная идентичность MVM
- **Fusion** — логический слой, собирающий сигналы агентов
- **Workspace 3D** — метрическая система координат в метрах
- **Confidence-first** — каждый сигнал имеет числовую уверенность

## Лицензия

Apache 2.0
