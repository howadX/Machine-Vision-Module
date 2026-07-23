# NB1 — Distilling a CNN Emotion Recognizer into a Sub-millisecond Geometric MLP (2023–2025)

**Живой документ** · Последнее обновление: июль 2026 · Версия 1.0
← [NB0: EmoZid](./nb0-emozid-skeleton.md) | → [NB2: PoseVAE](./nb2-posevae-skeleton.md)

---

## Production Status (июль 2026)

**В production НЕ используется напрямую.** Student-MLP заменён на EmotiEffLib B0 ONNX (`mvm-emotion`), который показал лучший trade-off accuracy/latency после исправления бага HWC/NCHW (фаза 10.1). Однако:
- Методология蒸馏 (teacher→student, soft labels, temperature) **использована** при дистилляции HSEmotion→MLP для геометрического fallback'а
- Геометрические признаки (468 расстояний от носа) **живут** в `mvm-detect` как fallback при отсутствии лица для CNN
- Главный урок NB1 — «дистилляция работает» — применён в production

**Связанные крейты:** `mvm-emotion` (EmotiEffLib ONNX), `mvm-detect` (YuNet → face crop)

---

## 1. `[MD]` TL;DR

> Сжимаем EfficientNet-B0 (эмоции) в геометрический MLP через knowledge distillation.
> Student в 7× легче teacher, <1ms CPU, падение accuracy ~1 п.п.

- **Teacher:** EfficientNet-B0, FER2013 + свои данные, ~73%, 224×224 RGB → ~200ms GPU
- **Student:** MLP на 468 геометрических признаках (MediaPipe FaceLandmarker), ~72%, <1ms CPU
- **Метод:** Knowledge distillation с temperature=3.0, α=0.3
- **Ключевой инсайт:** Soft labels от teacher несут больше информации, чем hard-метки — student учится *распределению вероятностей*, а не одному классу

## 2. `[MD]` Исторический контекст: почему蒸馏, а не просто CNN?

После NB0 (SVM ~45%) мы перешли к CNN (EfficientNet-B0 ~73%). Но:
- CNN требует GPU (200ms) — слишком медленно для real-time CPU
- Нужна модель <1ms на CPU
- Геометрический подход из NB0 (расстояния от носа) — быстрый, но слабый (~45-68%)

**Идея:** использовать CNN как *учителя* для быстрой геометрической модели. CNN видит пиксели → генерирует soft labels → MLP учится на геометрии, но с CNN-качеством.

## 3. `[MD]` + `[CODE]` Обнаруженное ограничение: FER2013 48×48

Исторический факт из разработки: когда мы прогнали MediaPipe FaceLandmarker по FER2013 (48×48 grayscale), **landmark detection rate оказался низким** — лицо слишком маленькое для 468 точек.

→ **Решение:** (а) teacher-разметку вести на higher-res источнике (AffectNet), (б) FER2013 использовать как raw-pixel baseline для teacher, (в) геометрические признаки извлекать из higher-res данных

Это инженерное решение, обнаруженное в разработке — не «причёсанная» история.

## 4–8. `[CODE]` Данные, Teacher, Features, Дистилляция, Student

(Существующие секции из оригинального скелета — без изменений)

## 9. `[MD]` Production Status (актуальное)

**Почему NB1-студент не в продакшене, а EmotiEffLib — да:**

| Фактор | NB1 Student (MLP) | EmotiEffLib B0 (prod) |
|--------|-------------------|----------------------|
| Вход | 468 геом. признаков | 224×224 RGB (через YuNet crop) |
| Качество признаков | Зависит от FaceLandmarker (падает на профиль) | CNN извлекает сама |
| Accuracy | ~72% | ~73-75% (после фикса HWC/NCHW) |
| Latency | <1ms | ~10ms CPU |
| Размер | ~50KB | ~16MB ONNX |
| Решение | **Отклонён** — accuracy ниже, зависимость от landmark'ов | **Принят** — лучше trade-off для production |

**Но:** сама методология蒸馏 использована для fallback-модели (геометрический MLP как backup, когда лицо не найдено для CNN).

## 10. `[MD]` Баг HWC/NCHW — урок о production

(Связь с NB1 косвенная, но важная для нарратива)

EmotiEffLib в production **жил с багом с фазы 6 по фазу 10**: препроцесс писал HWC-interleaved в NCHW-буфер → near-uniform softmax на любом лице. Энтропия 0.88–0.96, модель **казалась** сломанной.

Фикс (фаза 10.1): planar-CHW + YuNet-лендмарки + ArcFace-выравнивание кропа → Neutral conf 0.82–0.93, H=0.12–0.24.

**Урок:** баг жил 4 фазы, потому что не было **численного теста на известном лице**. С тех пор: каждый новый детектор — golden-тест на фиксированном изображении.

## 11. `[MD]` What Changed

- **2026-07** — ноутбук обновлён: добавлен Production Status, баг HWC/NCHW, решение об отказе от MLP в пользу EmotiEffLib
- **2025-XX** — оригинальная версия:蒸馏 CNN→MLP

## 12. `[MD]` Связи

← [NB0: EmoZid](./nb0-emozid-skeleton.md) — откуда вырос геометрический подход
→ [NB2: PoseVAE](./nb2-posevae-skeleton.md) — manifold learning для позы
→ [NB4: Teardown](./nb4-pipeline-teardown-skeleton.md) — почему identity-эмбеддинги ≠ эмоции
