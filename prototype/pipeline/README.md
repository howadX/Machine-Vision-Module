# Легаси-пайплайн (dlib + SVM)

> ⚠️ **Исторический документ.** Этот README — оригинальная инструкция к проекту EmoZid (~2019–2020).
> Нижеприведённые инструкции описывают установку через Visual Studio 2019 + Anaconda и **неактуальны** для текущей версии.
> Современный MVM — Rust-based evidence-backed visual subsystem будущего робота: visual evidence → canonical facts → bounded visual memory/world model → advisory Cortex. Он не является planner/controller/safety authority. Актуальный стек и инструкции — в [корневом README](../../README.md) и [`mvm/README.md`](../../mvm/README.md).

---

В папке scripts находятся скрипты, с помощью которых работало раннее приложение.
В папке Final находились основные файлы и скрипты для рабочего состояния АПК.

## Оригинальная среда

1. Python 3.6+.
2. Anaconda.
3. Visual Studio 2019.
4. Зависимости: `scipy`, `pygame`, `keyboard`, `glob`, `cv2`, `dlib`, `PIL`, `pincle`, `numpy`, `sklearn`, `csv`, `CMake`, `opencv-contrib-python`, `opencv-python`, `Pillow`, `chardet`.
5. Историческая точка запуска: `Indefication_4.py` в Visual Studio, без отладки.

Эта среда сохранена только для археологии/воспроизводимости. Не используйте её для production MVM.

---

## Пайплайн v1 (dlib)

| Шаг | Скрипт | Назначение |
|-----|--------|-----------|
| 1 | `1_make_dataset.py` | Захват лица через Haar Cascade |
| 2 | `2_extract_descriptors.py` | dlib HOG → 68 landmarks → 128d-эмбеддинги |
| 3 | `3_train_svm.py` | sklearn SVM (poly kernel) на 7 эмоций |
| 4 | `4_identification.py` | Рантайм: Face ID + эмоции + аудио-фидбек |

### Технологии v1

- **Детектор:** Haar Cascade (OpenCV)
- **Лица:** dlib HOG + ResNet 128d
- **Эмоции:** sklearn SVM на 68 точках лица
- **Аудио:** pygame mixer

Исторический запуск: `bash run.sh` (требует окружение с зависимостями из `requirements.txt`).
