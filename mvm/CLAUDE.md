# MVM — Machine Vision Module

*Structured Decomposed Picture-to-Text. Модульный ансамбль микро-агентов зрения.*
*Это source-of-truth по Rust-части. Детальные планы закрытых фаз — в `mvm/docs/phase-*.md` и `~/wiki/mvm/`.*

---

## Статус (2026-07-04)

- **Ветка:** `master` (продакшн после слияния `rust-core`; `rust-core` — предок слияния, отстал). **12 крейтов,
  510 тестов зелёных** (1 ignored — hardware live-camera), release чист.
- **Mesh = продакшн-путь; seg заведён в `AgentMesh` (2026-07-04, `--masks`):** решение владельца — `mesh` теперь
  главный продакшн-путь (sync `--native-camera` остаётся для тестов/истории). YOLO-seg был подключён только на
  sync → на mesh `ObjectState.mask` всегда `None`, и A1/A2 mask-контакт **молча падал в bbox**. Теперь
  `SceneAgent` гоняет один seg-forward и разветвляет: object-маски → `attach_masks` (A1/A2 mask-контакт +
  mask-median depth в `anchor_objects` работают в продакшене), person-маски → `PersonMaskSlot` (шаблон
  `DepthSlot`) для трекера. `--masks` opt-in (seg — реальный forward на object-cadence). **11.A3 ЗАКРЫТ И
  LIVE-ВАЛИДИРОВАН (2026-07-04, `mesh --masks`, 2 чел.):** person-маска в cost-матрицу трекера
  (`-(iou + W_OCM·momentum + W_MASK·mask_iou)`, `W_MASK=0.10`), gated ≥2 established-людей → лечит walking-
  фрагментацию. Терм = 0 при <2 людей / без масок → association **байт-идентична** pre-A3 (256 тестов трекера
  целы). **Live (~50с, 52 дампа @1с, seg публикует 2 person-маски/каденс):** оба держат стабильный id —
  **аноним id10 = 1 id в 50/50 дампах** (раньше анон-ходьба фрагментировала на 5–7 id); **частично**: сустейнед-
  фантом id12 ~13с в первой трети сеанса (17:51:23–36, near Вадим 0.25м) → дал 4 ложных `people_together`-эпизода
  ([0,12]/[10,12], честно `stable_participants=false`), затем рассосался, ~34с чисто `[0,10]`. **anomaly=0 весь
  сеанс** → сломанный fall-anomaly из pre-A3 НЕ воспроизвёлся. **Activity-journal работает на mesh:** объектные
  активности Вадима атрибутированы `stable=true` (`seated_at_workstation` 12.6с conf0.91, `using_mouse` до 8с,
  `holding_drink`, `reading`) — A1/A2 mask-контакт + FSM в продакшене; real `people_together(0,10)` непрерывен до
  22.5с. **Nota:** `stable_participants=false` и для реальной пары (аноним не галерейный id) → это консерв.
  маркер, не дискриминатор real-vs-фантом. **C4 sync-journal → dev-only** (sync не продакшн).
- **Live-валидация на риге (2026-07-04, `--native-camera --masks`, Вадим, 1 чел.):** producer/fusion-ядро
  фазы 11 отработало вживую. **Подтверждено:** (1) **11.A1 mask-контакт** снял ложную атрибуцию —
  перекрывающиеся на столе bbox (телефон над клавиатурой) больше не крадут контакт: ложный `using_phone`
  упал с ~50с (bbox) до ~2с (маски). (2) **Activity-FSM+journal** дают стабильные эпизоды с
  duration/events; стичинг (ниже) собрал непрерывные `reading`/`typing` вместо фрагментов. (3)
  **Object-permanence** (ниже) провалидирован end-to-end на клавиатуре: seed (руки прочь ~3с → YOLO видит) →
  hold под печатающими руками (`remembered` 17с) → `typing` атрибутирован **23с** (было 0 во всех прогонах
  без permanence). (4) B1-зоны держат id весь сеанс (`workstation`), B2-anomaly=0 (спокойно). **Вскрытые
  хвосты (не блокеры):** фантомная фрагментация одного человека в 7 track-id → ложные `people_together`
  (честно `stable=False`) + сломанный fall-anomaly (standing на id0, lying на id11 — память не совпала) →
  **корень = walking-fragmentation, лечится A3 person-масками**; YOLO26n не берёт конкретную клавиатуру без
  hands-off seed (detector-recall, не permanence).
- **Object-permanence для статичных фикстур (`20fb4037`, `+0 ONNX`):** `ObjectTracker` держал пропавший
  объект внутри (`timeout=15`), но эмитил только `missed==0` → клавиатура под руками исчезала из графа, а с
  ней `typing`. Фикс: `STATIC_CLASSES` (keyboard/mouse/laptop/tv/dining table/remote — оперируются на месте,
  перекрываются руками) при `age≥2` едут на длинном hold (`DEFAULT_STATIC_HOLD=75` ≈15с) и **эмитятся под
  окклюзией** с флагом `remembered` (sync-JSON); не-статичные истекают на timeout. + LOW_BAR-порог keyboard/
  mouse (base−0.08, мин 0.20) для засева. Caveat: нужен ≥1 seed-детект (hands-off), иначе держать нечего.
- **11.C6 обобщён на object-churn + включён на mesh (`252ff7ef`):** live-mesh показал спам журнала от churn
  object-id (7 id на один стол → 91 событие). Стичинг обобщён с «эфемерный человек на стабильном объекте» до
  «ровно один churn-участник в стороне» (`churn_class`: reserved 0–9 sticky, эфемерный человек и объект оба
  churn'абельны) → сшивает и person-churn (12→15), и object-churn (10000→10007 под person 0), но **не сливает
  двух gallery-людей**. Включён на mesh `FusionAgent` (`with_stitching(true)`).
- **Фаза 11 — Fusion-полировка (В РАБОТЕ, `+0 ONNX`, чистая арифметика, 2026-07-03):** реляц.-темпоральная
  арифметика над Scene JSON. Закрыто: **11.A1** mask-контакт (`InstanceMask` → `mvm-core`, `ObjectState.mask`
  плоско на обоих путях, контакт по силуэту с 2-cell дилатацией + честный bbox-fallback, `92e804aa`);
  **11.A2** фикс мис-атрибуции (object-major: спорный объект → одному человеку по силе претензии
  hand>wrist>reaching-wrist, затем близость к центру силуэта, затем id; `718cdd83`); **11.C1–C2** ядро
  activity-FSM `Candidate→Active→Cooling→Ended` + event-journal (`mvm-core/src/activity.rs`, enter-dwell 3 /
  exit-dwell 5, conf-EMA λ0.7, ключ=(label, sorted participants), poor-man's-HSMM: θ_lo(d)-exit + per-label
  MAX_DWELL-cap + `stable_participants`; `627ba66a`/`638add4b`); **11.C3** проводка mesh (`FusionAgent` владеет
  journal, frame_id-гейт, `WorldState.events`, writer дренирует эксклюзивно; `3aeeb1ca`); **11.C5** контракт
  (`JsonActivity` +аддитивно `state`/`started_at`/`duration_s`, новый `events[]`, `stable_zones[]`;
  обратная совместимость байт-в-байт; `9c9c4422`); **11.C6** опц. ephemeral-churn стичинг (за 10.7, off по
  умолчанию; `b1eee6a7`); **11.B1** зоны с persistence (`ZoneTracker`: стабильные id над union-find,
  workstation/seating-семантика, miss-cap 5; `c260b4cf`); **11.B2** anomaly v2 (noisy-OR: нежданный объект +
  pose-OOD `plausibility` + падение standing→lying за 3с; затухающий пик λ0.9; `ee036bc5`); **11.D1** метрич.
  self-calib `ScaleCalibrator` (`K=fx·real=d·pixel`, медиана; `fx_given_real`/`real_given_fx`; провалидирован
  на 2.5м YuNet-сэмпле <0.2м; `b1eee6a7`). **11.A3 person-маски — ЗАКРЫТ по коду** (mesh-first; см. блок про
  seg-в-mesh выше): mask-IoU-cue в cost-матрицу трекера, gated ≥2 людей; live ≥2 людей предстоит.
  **Отложено/gated:** **11.C4** (sync-journal → **dev-only**: sync больше не продакшн); auto-`fx` live-wiring
  (нужен метрич. референс в кадре → продуктовая фаза). **Live-валидация на риге** (1 чел.) **выполнена**
  (см. выше). План — `docs/phase-11-plan.md`.
- **Data-collection ЗАКРЫТ (SUPREME-валидация 2026-07-02, 3 чел.+собака, 120с/116 тиков):** основной sync-путь
  `--native-camera` собирает и парсит ВСЕ сигналы всех агентов (покрытие: id/face_verified 100%, поза/3D 98%,
  motion 99%, эмоция+8dist+энтропия 88%, head 88%, **кисти/запястья 62/68%** — были 0%, дроп починен, hands
  presence/side/handedness/gesture; объекты+маски). Правки сессии (SUPREME-driven, `+0 ONNX`):
  (1) **sync дропал вычисленные wrists/hands** → прокинуто+IoU-матч (`11f2fe58`);
  (2) **`face_verified`** — честный сигнал «тело подтверждено лицом» для неоднозначности животное/упавший (`c78b282e`);
  (3) **fusion-активности на sync-пути** (`people_together`/`typing`/`holding_drink`/`seated_at_workstation` — было
  mesh-only) + **model-corroborated анти-животное-гейт** (`is_animal`+`likely_animal`, тело поверх YOLO-`dog`
  исключается из реляционного fusion) (`f4632b80`). SUPREME-вывод: сырые выходы агентов **точнее** осторожных
  ручных трактовок (эмоция Anger, `bowl` — реальны). Остаточные баги — **не парсинг**: (a) fusion-мис-атрибуция
  контакта при вытянутой руке (телефон→не тот человек), (b) собака-на-руках→YOLO `handbag`, (c) `cell phone`/`bowl`
  пороги YOLO26n. Все три — fusion/детектор-уровень (next phase), не сбор данных.
- **Фаза 10 — Full Data Collection: 10.0–10.3 ЗАКРЫТЫ** (`+0 ONNX`, чистая арифметика): 10.0 эмит
  вычисленного-но-выброшенного (жесты рук reaching/waving, hands[] c presence/side, emotion_entropy,
  recon_error/plausibility → Scene JSON; сырые 21-lm кистей за опт-ин `mesh --dump-landmarks`);
  **10.1 корень эмоции #1 НАЙДЕН И ЗАКРЫТ** — препроцесс EmotiEffLib писал HWC в NCHW-буфер
  (скремблированные каналы → near-uniform на любом лице); planar-CHW фикс: на 6 референс-кропах
  энтропия 0.88–0.91 → 0.48–0.66, константный ложный «Fear» → осмысленный «Neutral». Плюс декод
  **5 лендмарков YuNet** (новый `FaceDet`/`detect_faces`) + `align_face` (Umeyama → ArcFace-шаблон)
  перед эмоцией (спайк 10.1e: p50 H 0.549→0.536, все 8/8 кадров проходят гейт 0.90 — GO, свап модели
  не нужен); 10.2 кисти: `classify_gesture` fist/open/point/pinch (чистая арифметика), handedness в
  JSON, vis-гейт запястья 0.5→0.2 с честным downweight score·(vis/0.5) (#5 закрыт); 10.3 голова:
  `head_orientation` (yaw/roll/facing camera-left-right) из 5 лендмарков → `head_yaw`/`head_facing`
  в Scene JSON — вход для gaze-aware fusion. **Всё live-validated (mesh, свет, 1 чел., 2026-07-02):**
  эмоция Neutral conf 0.82–0.93 / H=0.12–0.24 (гейт проходит), жесты рук reaching, hands presence
  0.69–0.8 + vis-relax, head_yaw≈0 фронтально / до −61° при повороте головы. Сегментация (10.4):
  **спайк замерен на чистом дневном кадре — conditional GO** (`docs/phase-10-plan.md` §3.1–3.2):
  latency ~47мс ок, mask-плотность 0.5–0.8 (⇒ `median_in_mask` depth = однозначный GO), person-силуэт
  чистый (⇒ 10.7 GO по качеству, ждёт ≥2 людей). YOLO26n остаётся детектором, YOLO-seg добавляется
  **поверх** для масок — не drop-in. **Ядро+детектор seg РЕАЛИЗОВАНЫ И ПРОВАЛИДИРОВАНЫ** (коммиты
  `f9c18a75`/`0275d970`): `InstanceMask` (assemble/iou/contains_norm/set_points), `median_at_points`
  (mask-aware depth), `YoloSegDetector` (+1 ONNX, model-gated live-тест: 9 инстансов, person mask-cov
  0.092 плотнее bbox). **Live-интеграция в sync-hot-path (`--native-camera`) ЗАКРЫТА И LIVE-ВАЛИДИРОВАНА**
  (2026-07-02, риг, 1 чел.): `TrackedObject.mask` (маска живёт прямо на объекте — mvm-scene, без слоёв,
  `ObjectState` не тронут), `attach_masks` (жадный class-aware bbox-IoU + mask-IoU тайбрейк = 10.5
  instance-separation, маска сбрасывается каждый тик) навешивает YOLO-seg-маски **поверх** YOLO26n-треков;
  **opt-in `--masks`** (дефолт off, нулевой перф-хит; live ~12 fps со всеми моделями + seg); **depth-по-маске**
  (`median_at_points` по силуэту вместо `median_in_bbox` — sync `anchor_objects_depth`); `mask_coverage` в
  Scene JSON. Live: seg даёт 4–5 дет/тик, 3/5 объектов маскируются (мелкие настольные cup seg иногда не
  берёт → честный fallback на bbox-depth, §3.2), **каждая маска плотнее своего bbox** (chair 0.12 vs 0.22 =
  45% фона срезано; cup ~22%) ⇒ depth-bleed реально уменьшается. Env-гейт `MVM_SEG_DEBUG` — live A/B счётчик.
  **Осталось (§3.3):** mesh-путь (маска плоскими данными в `ObjectState`), mask-контакт fusion (10.6,
  `contains_norm`), person-маски 10.7 (ждут ≥2 людей).
- **Фаза 8 ЯДРО ЗАКРЫТО** (body-primary + Agent-Mesh + fusion), live-validated на 0/1/2 людях.
- **Фаза 9 — Tracker: 9.1–9.10 ЗАКРЫТЫ** (`+0 ONNX`) — **фаза завершена**. Закрыто:
  кинематика B1 (motion ring + `MotionClassifier` + жесты) → OCM/OCR (9.2) → lifecycle FSM (9.3) →
  ByteTrack two-stage по телам (9.4) → инверсия контракта `associate/apply_detection/finalize` (9.5) →
  cadenced SFace + identity-lock (9.6) → emotion cadence (9.7) → лендмарки на треке + motion в Scene JSON
  (9.8) → pose-ROI recovery для locked-треков (9.9) → **Hungarian global assignment (9.10)**: жадный
  порядко-зависимый IoU-матчинг → оптимальный Kuhn–Munkres (`assignment.rs`, zero-deps) над той же
  IoU+OCM cost-матрицей. План — `docs/phase-9-plan.md`. **Фаза 10 — Segmentation** (+1 ONNX YOLO-seg)
  вынесена отдельно — `docs/phase-10-plan.md`.
- **Live-ревалидация 9 (1 чел., камера eye-level ~175см, ~горизонтальна):** idle/сидя — `id=0` lock 100%, motion idle↔walking
  + velocity в JSON. Ходьба — `id=0` держится 91% кадров (9.9), но детектор фрагментирует тело → краткие
  анон-мелькания остаются (**эмпирика для 10.7 person-масок**; см. память `walking-track-fragmentation`).
- **Pose-классификатор ПЕРЕПИСАН (method C, коммит `9308783c`)** — корень хвоста #2 закрыт по данным/методу.
  Старый `PoseVAE→head` (переобучен на ~1137 самозаснятых кадрах) заменён на **инженерные вьюпойнт-инвариантные
  фичи (14: наклоны торса/конечностей к гравитации, флексии суставов, extent/aspect) → RandomForest** (soft-vote,
  `pose_rf_forest.bin` 873КБ) → **3 класса `standing/sitting/lying`** (экзотика `sitting_at_desk/lying_*/arms_crossed`
  = артефакты мусорного файн-тюна, удалены). BlazePose-лендмарки (публичные) сохранены. VAE понижен до OOD-only
  (`pose_vae_public.bin`, переобучен на публичных, seeded). Обучение — **только публичные данные** (URFD через наш
  BlazePose + Zenodo MultiPosture); grouped 5-fold **0.959** на невиданных людях, азимут-ровно (сырые коорд.
  рушились до 0.31 @az90). Спайк/тренер — `spike_pose/{spike_data,spike_feats,train_pose_p1}.py`, findings —
  `spike_pose/spike_feature_findings.md`. Golden Rust↔Python бит-точен (`pose_p1_golden.in`).
- **Occlusion-aware posture — desk-гэп ЗАКРЫТ АРХИТЕКТУРНО, БЕЗ файн-тюна** (`+0 ONNX`, чистая арифметика).
  Диагноз (замер 2026-07-02 на 7 боевых desk-кадрах `p_V`): при сидении за столом ноги реально не видны
  BlazePose'у (ankle vis 0.09–0.51 vs full-body p50≈0.8), лег-фичи галлюцинируются → RF читает desk-sitting
  как уверенный `lying` (P(sitting)≤0.12). Вывод: не файн-тюнить угадывание по мусорным суставам, а **не
  утверждать то, что камера не видела**. Реализация: `mvm-posture/classifier.rs` — `BodyCoverage{Full,Upper,Head}`
  (`coverage_from_vis`, порог lower_vis 0.5) + `torso_attitude_2d` (2D-наклон торса в плоскости кадра — image-y≈
  гравитация; 3D `torso_inc` тут мусор, даёт 55–66° и стоя, и сидя на обрезанном теле). Гейт claim'а —
  `mvm-cli/pipeline.rs::posture_claimable`: Full→как раньше; Upper→метка **withheld** (кроме safety-`lying` при
  `attitude=horizontal`); Head→ничего; кросс-чек: `lying`+upright **невозможен** физически → отвергается на всех
  coverage. Сохранена честность: `body_coverage`/`torso_attitude` + recon/plausibility **всегда** в Scene JSON.
  `PostureStabilizer::observe_hidden` (dwell) **чистит stale-метку** при устойчивом Upper (сел за стол — `standing`
  не переживает). Fusion (`agents/fusion.rs`) даёт контекст реляционно: `seated_at_workstation` (Upper+upright+
  не-moving рядом с desk-объектом) и `partially_in_frame` (bbox у края кадра). sitting-vs-standing без ног
  **сознательно не утверждается** — это вывод fusion, не класс. Оракул `pose_class_oracle_desk_occlusion_contract`
  **переписан и включён** (не `#[ignore]`): 7/7 detect, ни одного ложного `lying`, claim⇒верен-или-withheld.
  Файн-тюн со своей камеры **отменён как ненужный** (опционален для тонкого sitting/standing за столом — отдельный
  спайк). См. память `pose-classifier-method-c-spike`.
- **Tier 0 — robustness позы/эмоции** (`+0 ONNX`, чистая арифметика; коммит `1f9caead`), **live-validated**:
  0.1 `PostureStabilizer` (dwell+conf-гейт) + 0.2 OOD-гейт по VAE-`plausibility` (порог `POSTURE_OOD_LOGPZ`
  **перекалиброван 14.3→170** под новый публичный VAE, коммит `9308783c` — старый порог глушил ВСЮ постуру live);
  0.3 гейт эмоции по энтропии (≥0.90) → near-chance метка подавляется (live: 47/48 подавлено). Позу теперь лечит
  **по корню** method C (см. выше); Tier 0 остаётся темпоральным сглаживанием. Корень эмоции (#1, вход EmotiEffLib)
  открыт. Модули `mvm-core/posture.rs`, `emotion.rs::emotion_label_gated`, `mvm-cli/pipeline.rs::plausibility_flag`.
- **⚠️ Камера — eye-level (~175см, ~горизонтальна, `rotate_180=false`), НЕ потолок/45°.** Прежняя «потолочная»
  формулировка в доках была ошибочной (исправлено); симптомы #1/#2 — не от ракурса. Bridge всё ещё крутит
  `ROTATE_180` (legacy) → на upright-камере перевернёт кадр (код-хвост для bridge-пути).
- Hot-path `--native-camera` — **чистый Rust** (ноль Python). Bridge (`--native`) сохранён для
  тестов/дебага (снос venv отменён решением владельца).

---

## Три инварианта (любая фича обязана им соответствовать)

1. **Минимализм / чистая арифметика.** Fusion, трекинг, 3D, uncertainty — это скалярная арифметика
   над данными, уже лежащими в памяти. **Ноль новых зависимостей** на каждую интеллектуальную фичу.
   Footprint растёт только когда добавляется новый ONNX-эксперт.
2. **CPU-first, $0.** Ни рантайм, ни модели не требуют GPU. Никаких внешних API. Цель — `<50ms` CPU.
3. **Полный нативный Rust-пайплайн.** Python-модели — прототипы/леса. Новый код не углубляет связь
   с Python в hot-path.

**Что производит MVM:** метрический 3D scene-graph каждый кадр — узлы-сущности с метрической 3D-позицией
(origin = камера, pinhole от known-size априоров, **не** depth-сеть) + рёбра-отношения. Тир — грубый
метрический (±15-30% абсолюта) + сильный относительный субстрат (робастный ординал «кто ближе» + зоны).
Fusion: реляционная (граф близости/пересечений → зоны, главная) + атрибутивная (PoE по атрибуту, частный случай).

**Почему не VLM:** ансамбль специалистов vs одна end-to-end модель. <50ms CPU / $0 / приватно /
per-signal confidence / интерпретируемо / per-agent заменяемо. Покрывает ~80% сценариев (присутствие,
эмоция, поза, позиция) при 0.1% стоимости VLM; открытый мир / zero-shot / QA → VLM как fallback.

---

## Живая цепочка (`--native-camera`, тело-первичная)

```
frame → person_detection (ТЕЛО, первичный) → сущность/трек = ТЕЛО (NativePose::bodies)
          ├─ BlazePose → 33 lm → compute_features(14) → RandomForest → posture (всегда; VAE→OOD-гейт)
          ├─ торс-3D (плечи 11/12 → бёдра 23/24, torso_to_3d)       (всегда; gated на visibility ≥0.5)
          ├─ кисти: ROI из запястий BlazePose (15/16) → handpose 21 lm/рука (presence-gate)
          └─ YuNet (ЛИЦО) по кадру, матч к телу по НОСУ (lm[0] в face-bbox)
                есть лицо → SFace id + EmotiEffLib emotion;  нет → identity/emotion = None
        + объекты (YOLO26n, cadenced) + глубина (Depth-Anything, cadenced) → метрич. 3D объектов
        tracker по bbox ТЕЛА (Kalman + IoU + cosine/label/center re-link) → build_scene → StateFile/JSON
```

**Body-primary** (FRONTIER, закрыт 8.0.2): человек без фронтального лица трекается (presence + поза +
торс-3D); identity/emotion дотягиваются при появлении лица. A1 анти-flip подавляет мигание body↔face.

---

## Запуск и команды

```bash
# Сборка / тесты / бандл
cargo build --release          # → target/release/mvm-cli (~26 МБ, модели embedded)
cargo test --workspace         # 501 тест
cargo xtask bundle             # → dist/mvm

# Agent Mesh (фаза 8, основной путь) — Capture→{Human,Scene,Depth}→Fusion→StateFileWriter на std::thread
cargo run --release -- mesh --dump-json --interval-ms 1000   # живой (камера + все модели)
cargo run --release -- mesh --stub --max-frames 100          # model-free CI-smoke

# Синхронные пути (живы)
cargo run --release -- --native-camera --once --dump-interval 1   # pure-Rust hot-path (без mesh)
cargo run --release -- --native-camera --dump-interval 3 --scene-debug  # дебаг: детекторы цветом → /tmp/mvm_scene.jpg
cargo run --release -- --native --once --dump-interval 1          # Python bridge (поза от MediaPipe)
```

**Камера:** `--native-camera` = прямой v4l2 MJPG mmap; bridge = `cv2.CAP_FFMPEG` (обход PipeWire на Ubuntu 24+).
`fps` в StateFile = темп **доставки кадров** (прирост `frame_id`), **не** темп обработки человеко-цепочки.

---

## Крейты (12)

| Крейт | Содержимое |
|-------|------------|
| **mvm-core** | типы, Config, PersonTracker (Kalman 4×скаляр + identity/cosine re-link + reserved-id), Pinhole/space (+`ScaleCalibrator` 11.D1), EMOTION fuse (PoE), **WorldState** (блэкборд, 8.A/8.B), `assignment` (Hungarian/Kuhn–Munkres, 9.10), `activity` (FSM+journal 11.C), `anomaly` (v2 scorer 11.B2), `mask` (`InstanceMask` 11.A1) |
| **mvm-detect** | YuNet ONNX (228KB, embed, multi-stride NMS) — лицо |
| **mvm-identity** | SFace ONNX (37MB, 128d L2-эмбеддинг, gallery, `embed()`) |
| **mvm-emotion** | EmotiEffLib B0 ONNX (16MB, 8-class VGAF), EmotionResult + entropy |
| **mvm-posture** | нативная BlazePose two-stage (детектор+поза) + **feature→RandomForest** классификатор (method C, 3 класса) + VAE (OOD-only); `compute_features`/`pose_from_face`, `NativePose` мульти-persona |
| **mvm-fusion** | `build_scene` (union-find граф+зоны) + `fuse()` PoE; activities + anomaly_score (8.F); `ZoneTracker` (persistence+семантика зон, 11.B1) |
| **mvm-scene** | YOLO26n ONNX (9.5МБ, end-to-end NMS-free), object-узлы (id≥10000) + `ObjectTracker` [8.E] + YOLO-seg (`--masks`, 10.4) + **object-permanence** статичных фикстур под окклюзией руками (`STATIC_CLASSES`/`static_hold`, 11) |
| **mvm-depth** | Depth-Anything-V2-Small ONNX (квант 27МБ, ~230мс CPU), depth-anchoring к торс-Z [8.E.2] |
| **mvm-hands** | MediaPipe handpose ONNX (4МБ), 21 lm/рука, ROI из запястий BlazePose [8.E.4] |
| **mvm-mesh** | Agent trait + `LatestFrame` (single-slot latest-wins) + `AgentMesh` оркестратор на std::thread [8.C] |
| **mvm-pipeline** | Camera (Python bridge subprocess) + NativeCamera (v4l2 MJPG) |
| **mvm-cli** | main + `mesh`-подкоманда; `pipeline.rs` (BodyPrimaryPipeline), `agents/` (Capture/Human/Scene/Depth/Fusion/StateFileWriter), `state.rs`, `scene_debug.rs` |

> Модели ONNX (YuNet/SFace/EmotiEffLib/BlazePose/YOLO26n/Depth-Anything/handpose) лежат в `mvm/models/`,
> крупные — gitignored с URL (download-separately). `include_bytes!`-embed грузит их в release + bundle.

---

## Agent Mesh (фаза 8.C–8.D)

`mvm-cli mesh` гоняет цепочку на `std::thread` (ноль tokio — contention опровергнут спайком,
эксперты ~12мс): `Capture → {Human, Scene, Depth} → Fusion → StateFileWriter`.

- **HOL-блокинг снят:** depth (~230мс) и YOLO на своих cadence/потоках не стопорят человеко-цепочку.
- **HumanAgent** владеет human-only pipeline + своим `PersonTracker`. **Depth** публикует в `DepthSlot`
  (slot в cli, не в mesh → mesh detector-agnostic). **FusionAgent** видит людей+объекты+depth, делает
  depth-anchoring. **StateFileWriter** сериализует `WorldState` → StateFile + Scene JSON.
- **FPS (честно, человек в кадре):** движок ~16-17 fps (mesh) vs ~14 синхронно; доставка ~25 fps.

---

## Рабочие факты и константы

- **Identity-модель (reserved-id):** id **0–9** — за пользователями галереи (Вадим=0, липко на присутствие);
  анонимы — эфемерные id **≥10** (переиспользуются при выходе). **Эксклюзивно** (один user-id — одному телу).
  Наружу: `PersonTrack.display_id` + `shown_identity()`. margin-identity отсеивает чужого (#5).
- **3D:** торс-pinhole (`space::torso_to_3d`, `Config.torso_length_m=0.5` — **не откалиброван рулеткой**,
  ординал ок, абсолют приблизителен). Лицо-bbox 3D: `face_height_m_yunet=0.32` (калибр под YuNet-боксы).
- **Cadence:** `object_cadence=3` (YOLO); depth cadence ~30. Object-фильтры: A score≥0.45, B стабильный
  `ObjectTracker` (IoU+free-list, `age`), C allow-list классов (person исключён → G₁), D min bbox area 0.004.
- **Взаимодействие человек↔объект** — по **body-landmarks** (запястье/кисть ∈ bbox объекта), не по body-bbox.
- **Depth-anchoring:** относит. inverse-depth привязывается к метрич. торс-Z (`Z·d≈k`, медиана по людям)
  → объекты получают метрич. 3D в той же системе → реальные 3D-`near` рёбра.

---

## Python bridge (родительский проект)

MVM — гибрид. Rust-крейты в `mvm/`, Python-прототипы — в родительском каталоге (доступны по `../`):

```
Machine Vision Module/           ← корень (cwd при запуске)
├── core/        face_id.py, workspace_3d.py, pose_vae.py, emotion_hse.py, cpu_face_id.py
├── tools/       mvm_bridge.py (камера → JSON lines, subprocess)
├── models/      mediapipe/*.task, emotion/*.pth, references/
├── venv/        Python 3.11+ (нужен только для bridge-пути)
└── mvm/         ← Rust workspace (этот каталог)
```

Запуск bridge напрямую: `../venv/bin/python3 ../tools/mvm_bridge.py --json-only --max-frames 1`.

---

## Известные открытые хвосты

| # | Проблема | Статус |
|---|----------|--------|
| 1 | Эмоция near-chance — **КОРЕНЬ НАЙДЕН, ЗАКРЫТ И LIVE-ВАЛИДИРОВАН (10.1)**: препроцесс писал HWC-interleaved в NCHW-тензор → модель видела скремблированные каналы → near-uniform softmax на любом лице (жило с порта фазы 6). Фикс planar-CHW + regression-тест на layout; вдобавок кроп выравнивается по 5 лендмаркам YuNet (Umeyama→ArcFace-шаблон). **Live (mesh, свет, 1 чел., 2026-07-02): Neutral conf 0.82–0.93, энтропия H=0.12–0.24** (было ≈0.96, peak 0.16–0.22) — метка проходит гейт 0.90, distribution резко пиковое (0.93 на Neutral). Прежний симптом «47/48 подавлено» снят по корню | ✅ (live-validated) |
| 2 | Поза: старый `standing↔sitting_at_desk` флип **закрыт по корню** (method C, `9308783c`). Остаточный desk-гэп (публичная модель читала desk-sitting как `lying`, ноги под столом) **закрыт архитектурно, occlusion-aware, БЕЗ файн-тюна**: `BodyCoverage`+2D-attitude → метка withheld в Upper-режиме, `lying`+upright отвергается, fusion даёт `seated_at_workstation` реляционно; оракул переписан и включён. | ✅ (occlusion-aware; файн-тюн отменён как ненужный) |
| 3 | `torso_length_m=0.5` не откалиброван рулеткой (абсолют 3D приблизителен) | 🟡 |
| 4 | `fps` в StateFile = темп доставки, не обработки | 🟡 (семантика) |
| 5 | Hand-ROI из запястья требует чёткого жеста — **закрыт (10.2)**: vis-floor 0.5→0.2, score честно downweight'ится ×(vis/0.5), presence-гейт модели сохранён; кисти за столом/у края кадра возвращаются в поток | ✅ |
| 6 | Объекты видно с ракурса человеческого роста | 🟡 |
| — | Ре-валидация стабилизации 8.0.4 на 3 людях; 8.E.3 audio (YAMNet) — новая dep против минимализма, вне ядра | 🔜 опц. |

### Ledger отложенных сигналов (10.5 — явная диспозиция, чтобы «не забыли»)

| Сигнал | Решение | Почему |
|--------|---------|--------|
| **Audio / YAMNet (G₄)** | **deferred** (Фаза 11+, опт-ин модуль) | новая capture-dep + микрофон ломают +0-dep/CPU/$0; ортогонально зрению. `AudioEvent` остаётся стабом контракта в `world.rs` |
| **Age / gender / clothing** | **out** | каждый +1 ONNX без текущего use-case — минимализм |
| **Temporal journal / event log** (длительности поз, start/stop активностей) | **Фаза 11** (продукт) | не producer-работа; строится над Scene JSON |
| **Richer anomaly, зоны/комнаты** | **Fusion-polish фаза** (после 10) | реляционная работа, не producer |
| **3D metric calibration** (`torso_length_m`, fx) | **Фаза 11 auto-calib** | ординал уже ок; абсолют — продуктовая калибровка |
| **Сырые 33 body-lm в JSON** | опт-ин расширение `--dump-landmarks` при необходимости | лендмарки кистей уже за флагом; 33-lm тела на PersonState не хранятся — добавлять только под реальный use-case |

---

## TDD-правила

- TDD обязателен: сначала падающий тест, потом реализация.
- `cargo test --workspace` остаётся зелёным после каждого изменения; коммит после каждой задачи.
- `test_config()`: `Config { models_path: PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models")), ..Default::default() }`.

---

## Roadmap

| Фаза | Статус | Суть |
|------|--------|------|
| **5** | ✅ | Стабилизация (YuNet preprocessing, emotion fallback, галерея, FFMPEG backend) |
| **6** | ✅ | Metric 3D Scene-Graph (Kalman bbox, pinhole-3D, entropy, `build_scene` + PoE) — ноль новых deps |
| **7** | ✅ | Python-light: hot-path `--native-camera` = pure-Rust (v4l2→YuNet→ONNX), ~25 fps; SFace-эмбеддинг + margin-identity + cosine-re-link |
| **7.5** | ✅ | Нативная поза: BlazePose+PoseVAE порт (bit-close к Python), мульти-persona (#3 закрыт) |
| **8** | ✅ ядро | Body-primary + Agent-Mesh (`mesh`) + fusion (activities/anomaly); G₁ Human полна (+scene/depth/hands) |
| **9** | ✅ | **Tracker** — кинематика + ByteTrack/OCM/OCR + lifecycle + инверсия + cadenced SFace/emotion + motion-эмит + pose-ROI recovery + Hungarian global assignment, **+0 ONNX** — закрыта |
| **10** | ✅ | **Full Data Collection — ЗАКРЫТА** (SUPREME-validated, 3 чел.+собака, 120с): 10.0 эмит вычисленного, 10.1 эмоция (CHW-корень + YuNet-лендмарки + align), 10.2 кисти (жесты/handedness/vis-relax), 10.3 head-orientation, 10.5 ledger — `+0 ONNX`; 10.4 Segmentation (**+1 ONNX**): спайк GO + ядро + `YoloSegDetector` + sync-live-интеграция (`--masks`, `attach_masks`, depth-по-маске, `mask_coverage`); **+ фиксы сессии:** sync-дроп кистей починен, `face_verified`, sync-fusion-активности, model-corroborated анти-животное-гейт. **Все агенты собирают+парсят все параметры на основном пути.** Осталось = fusion (реляц. атрибуция, next phase) + детектор-пороги YOLO (телефон/bowl/собака-как-handbag) + опц. mesh-зеркалирование. person-маски 10.7 — есть эмпирика ≥2 чел., реализация в fusion-фазе |
| **11** | 🔨 в работе | **Fusion-полировка** (`+0 ONNX` для fusion; seg-модель уже была) — реляц.-темпоральная арифметика над Scene JSON. **Закрыто+live-validated (2026-07-04, 1 чел.):** A1 mask-контакт, A2 фикс мис-атрибуции, C1–C3+C5–C6 activity-FSM+journal+контракт (C6 обобщён на object-churn, вкл. на mesh), B1 зоны-persistence, B2 anomaly v2, D1 метрич. self-calib, **object-permanence** статичных фикстур (клавиатура под руками → typing). **Mesh→продакшн + seg-в-mesh:** A1/A2 mask-контакт + mask-median depth теперь работают на mesh; **A3 person-маски ЗАКРЫТ И LIVE-ВАЛИДИРОВАН** (2 чел.: оба стабильный id, аноним 50/50 дампов; **частично** — сустейнед-фантом ~13с в первой трети дал ложные people_together stable=false, затем чисто; anomaly=0, fall-anomaly не воспроизвёлся; объектные активности Вадима атрибутированы stable=true). **Осталось:** ужать остаточный фантом (гейт-хвост?); C4 sync-journal → dev-only. План — `mvm/docs/phase-11-plan.md` |

Детали закрытых фаз: `mvm/docs/phase-7.5-plan.md`, `phase-8-plan.md`; `~/wiki/mvm/roadmap.md`.

### 🟢 Фаза 9 — Tracker: body-first track-centric рефактор (9.1–9.9 закрыты, осталось 9.10)

> **Архитектура:** **Body-first tracking + Face-as-attribute + ByteTrack для робастности.** Тело видно
> всегда → трекаем **тело**, не лицо; identity/emotion — **атрибуты тела**, переживают окклюзию лица;
> SFace — для долгосрочной ре-идентификации, **не каждый кадр**. ByteTrack по телам — **страховка**
> (пересечение людей / слабый кадр детектора), не главный механизм. **+0 ONNX** — всё чистая арифметика
> и оркестрация. Детальный план — `mvm/docs/phase-9-plan.md`. Разбор — `~/wiki/machine-vision/02-tracking/`.

**Что фаза 8 уже даёт:** тело — несущая сущность; лицо матчится по носу, identity/emotion/3D — уже
атрибуты `PersonTrack`; reserved-id sticky; каскад re-link = грубый OCR. **Дельта 9 — три оси:**
(A) инверсия потока обогащения в track-gated, (B) робастность ассоциации, (C) identity lifecycle + cadenced SFace.

> **⛔ Слоистость (критично):** `PersonTracker` в `mvm-core` (нет `ort`/`image`) — трекер **НЕ вызывает
> модели**. Спина оркестрируется в `HumanAgent` (`mvm-cli`); трекер лишь **экспонирует запросы**
> («какие треки без identity / устаревший emotion / дай предсказанный ROI»).

| Этап | Задача | Реализация / коммит |
|------|--------|-----------|
| **9.1** ✅ | **Кинематика (B1):** ring 33-лендмарков (N=16) → centroid speed (torso/s) → `MotionClassifier` idle↔walking (dead-zone+dwell+гистерезис) + `ArmGesture` reach/wave по запястью | `motion.rs`; b34a7be2 / 6740f9b9 |
| **9.2** ✅ | **OCM/OCR:** `obs_velocity` (EMA, gap-norm) → momentum-терм в cost (анти-swap) + center-relink по **виртуальной траектории** | `tracker.rs`; e87f143d |
| **9.3** ✅ | **Lifecycle FSM** `Tentative→Confirmed→IdLocked→Occluded→Lost` — чистая `next_lifecycle`, аддитивно (churn идентичен) | `types.rs`+`tracker.rs`; 2d266332 |
| **9.4** ✅ | **ByteTrack two-stage:** `BodyPose.score`, `detect_scored`, `bodies_byte` (high posed / low bbox-only) → `tick_byte` low-recovery | posture+cli+core; 2db33c7c |
| **9.5** ✅ | **Инверсия контракта:** `BodyObservation` + `associate`/`apply_detection`/`finalize`; `tick_byte` = тонкий оркестратор, поведение байт-идентично | `tracker.rs`; 5afd2bec |
| **9.6** ✅ | **Cadenced SFace + lock:** `is_id_locked` (reserved/cosine-стаб); `identity_run_flags` — skip SFace только для continuation locked-трека | core+cli; 671e4221 / **05bdcddf** (рефайн) |
| **9.7** ✅ | **Emotion cadence:** `frames_since_emotion`/`needs_emotion`; лейбл заморожен между прогонами (matched → нет decay) | core+cli; 818376b8 |
| **9.8** ✅ | **Motion на треке:** `MotionTrack` (bounded ring), `tick_motion`; `motion_state`/`velocity` → `PersonState` → Scene JSON; live idle↔walking | core+cli; 7d818fc6 |
| **9.9** ✅ | **Pose-ROI recovery:** `bodies_byte_recover` — locked-трек, потерянный детектором, позируется из предсказанного ROI (torso-vis гейт). Live: id-continuity 91%, 0 фантомов на пустом | posture+cli; 326cbc19 |
| **9.10** ✅ | **Hungarian assignment:** hand-rolled Kuhn–Munkres (compact O(n³), dual-potentials, zero-deps) в новом `assignment.rs`; жадный IoU-матчинг Step 3 → глобально-оптимальный над IoU+OCM cost-матрицей (cols sorted by id → детерминизм; `IOU_GATE→∞`; каскад re-link + ByteTrack сохранены как fallback). 158 прежних тестов трекера зелёные (байт-идентичность 1–2 тел) + дискриминирующий тест оптимальности vs greedy | `assignment.rs`+`tracker.rs` |

> **Инварианты 9 (соблюдены):** **+0 ONNX**; всё — чистая арифметика + оркестрация; модели **не** въехали
> в `mvm-core`; body-primary FRONTIER сохранён. Робастность (9.2–9.4) сделана **до** инверсии (9.5).
> **Эмпирика для фазы 10 (live):** при ходьбе детектор фрагментирует тело → краткие
> анон-мелькания даже после 9.9 (recovery держит только locked) → **person-маски 10.7 оправданы**.
> Подробно — память `walking-track-fragmentation`.

### 🔜 Фаза 10 — Segmentation: инстанс-маски (взаимодействие)

> **+1 ONNX (YOLO-seg nano)** и больше ничего. Один forward-pass даёт object-маски (взаимодействие,
> instance-разделение) + опц. person-маски (ассоциация). Mask-декод/контакт/depth-по-маске — чистая
> арифметика. Cadenced на `SceneAgent` (hot-path ≈ 0). Детальный план — `mvm/docs/phase-10-plan.md`.
> **Точка входа — спайк-гейт 10.1** (go/no-go по замеру масок vs bbox + учёт эмпирики фазы 9).

| Этап | Задача |
|------|--------|
| **10.1** | **Спайк-гейт** (первый): выигрыш маски vs bbox/landmark на боевом ракурсе + частота перекрытия людей (с учётом фазы 9) → go/no-go + судьба 10.7 |
| **10.2–10.6** | **Object masks:** YOLO-seg drop-in (декод 32 прото 160²) → mask→`Entity` → `median_in_mask` depth → mask-IoU `ObjectTracker` (переиспользует Hungarian 9.10) → mask-контакт fusion |
| **10.7** | **person-mask ассоциация** (опц., gated 10.1): `person`-маска **только в cost-матрицу фазы 9** (не узел!); только если арифметика 9 не закрыла перекрытие тел |

> **❌ Отложено (фаза 11+):** ST-GCN/MPED-RNN/VideoMAE (нейро-действия). **Caveat:** nano-маски грубы на
> мелких/дальних объектах → спайк-гейт 10.1 первый. SAM/Mask R-CNN отвергнуты ($0/CPU).

### 🔜 Фаза 11 — Fusion-полировка

Данные готовы (фаза 10) → Фаза 11 работает **НАД Scene JSON**: реляц.-темпоральная арифметика,
объединяющая покадровые сигналы в связки/обобщения/активности. **+0 ONNX** для всего ядра. Детальный
план и research (FSM vs HMM/HSMM/CRF/GCN, «poor man's HSMM») — `mvm/docs/phase-11-plan.md`.

- **A. Реляц./атрибуция:** ✅ A1 mask-контакт (`InstanceMask::contains_norm_within` вместо bbox под
  `--masks`, силуэт срезает фон), ✅ A2 фикс мис-атрибуции (object-major, спорный объект → одному человеку);
  ✅ A3 person-маски 10.7 (LIVE-ВАЛИДИРОВАН, 2 чел.): mask-IoU-cue в cost-матрицу трекера
  (`W_MASK=0.10`, gated ≥2 established-людей, не узел). Требовало **seg-в-mesh** (mesh=продакшн): `SceneAgent`
  один seg-forward → object-маски `attach_masks` + person-маски в `PersonMaskSlot`; `HumanAgent` матчит их к
  телам по bbox-IoU перед `tick_motion`. Терм = 0 при <2 людей / без масок → байт-идентично pre-A3.
- **B. Зоны/anomaly v2:** ✅ B1 семантич. зоны с persistence над union-find (`ZoneTracker`: стабильные id,
  workstation/seating-подпись, miss-cap); ✅ B2 anomaly v2 (`AnomalyScorer` noisy-OR: нежданный объект +
  pose-OOD `plausibility` + падение standing→lying; затухающий пик).
- **C. Activity-FSM (ядро):** ✅ конечный автомат `Candidate→Active→Cooling→Ended` (dwell+гистерезис, аналог
  `MotionClassifier`/`PostureStabilizer`) → стабильные активности + **event-journal** (start/stop/duration),
  `mvm-core/src/activity.rs`; `Activity` не тронут, `JsonActivity` аддитивно получил `state`/`started_at`/
  `duration_s`, новый `events[]`. poor-man's-HSMM (θ_lo(d), MAX_DWELL-cap), C6 опц. churn-стичинг. **C4
  (sync-проводка) отложена** — sync эмитит сырые активности без temporal-полей.
- **D. 3D метрич. калибровка:** ✅ D1 `ScaleCalibrator` (`K=fx·real=d·pixel`, медиана; провалидирован на
  2.5м-сэмпле). Live auto-`fx` (нужен метрич. референс) → продуктовая фаза.

**Порядок (факт):** `A1 → A2 → C1–C2 → C3 → C5 → C6 → B1 → B2 → D1`; A3 (≥2 людей) и C4 (sync-journal) —
осталось. Live-валидация на риге — предстоит.
**Отложено за фазу** (обученные модели против +0-dep): predictive-state (SG-LKF/EDL/PDF), MLP-fusion
(Level-2), LLM-рендер (Level-3), нейро-действия (ST-GCN/VideoMAE), audio/YAMNet.

### 🔜 Следующая фаза (не определена) — Продукт

Авто-калибровка fx (zero-config UX) · drag-and-drop галерея (фото в папку = новый человек) · one-click
install (один бинарник, без pip/venv/Python) · systemd service + cron state dump. К упаковке рано —
сначала fusion-ядро.

---

## Терминология

- **Микро-агент** — независимый детектор с confidence. **Fusion** — слой, собирающий сигналы в граф/описание.
- **Workspace 3D** — метрическая система координат (метры, origin = камера).
- **Confidence-first** — каждый сигнал несёт числовую уверенность. **Prediction error** —
  `||predicted − observed||` → естественный confidence-сигнал.
- **Body-primary** — несущая сущность цепочки = тело (не лицо). **Reserved-id** — id 0–9 за галереей, ≥10 анонимам.
