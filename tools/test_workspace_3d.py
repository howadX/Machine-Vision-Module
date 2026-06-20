#!/usr/bin/env python3
"""
tools/test_workspace_3d.py — тесты для Workspace3D.

Проверяет:
  1. Синтетическая калибровка: landmarks с torsо 0.5м → z_scale ~1.0
  2. Масштабирование: landmarks ×2 меньше → z_scale ~2.0
  3. Реальные данные из poses_dataset.csv
  4. Низкая visibility → fallback или False
  5. EMA-сглаживание
  6. Fallback на лицо (FaceLandmarker)
  7. pixel_to_3d
  8. Сброс калибровки
"""

import os, sys, csv
import numpy as np

MVM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MVM_ROOT, 'core'))

from workspace_3d import (
    Workspace3D, CalibrationState,
    TORSO_LENGTH_M, INTEROCULAR_M, FACE_HEIGHT_M,
    SHOULDERS, HIPS,
)

PASSED = 0
FAILED = 0

def check(condition, msg):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {msg}")
    else:
        FAILED += 1
        print(f"  FAIL  {msg}")


# ══════════════════════════════════════════════════════════════
# ТЕСТ 1: синтетическая калибровка (torso ровно 0.5м)
# ══════════════════════════════════════════════════════════════

def make_landmarks(torso_y_px=150, torso_z=0.0, vis=1.0, width=640, height=480):
    """
    Создаёт синтетические landmarks с torsо заданного размера.
    Плечи: (0.5, 0.3), Бёдра: (0.5, 0.3 + torso_y_px/height).
    """
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = vis  # все видны

    # Плечи
    lm[11, :3] = [0.5, 0.3, torso_z]
    lm[12, :3] = [0.52, 0.3, torso_z]

    # Бёдра
    hip_y = 0.3 + torso_y_px / height
    lm[23, :3] = [0.5, hip_y, 0.0]
    lm[24, :3] = [0.52, hip_y, 0.0]

    return lm

print("=== Тест 1: синтетическая калибровка (torsо = 0.5m) ===")
ws = Workspace3D()
lm = make_landmarks(torso_y_px=150)  # 150px torsо на 480px кадре
result = ws.calibrate(lm)
check(result, "калибровка успешна")
check(ws.is_calibrated, "is_calibrated = True")
check(ws.confidence > 0.9, f"confidence высокая: {ws.confidence:.3f}")
check(abs(ws.last_torso_m - TORSO_LENGTH_M) < 0.01,
      f"torsо восстановлен: {ws.last_torso_m:.3f}м (ожидалось {TORSO_LENGTH_M}м)")

# Проверка to_metric
metric = ws.to_metric(lm)
check(metric.shape == (33, 3), "to_metric возвращает [33, 3]")

# Плечи и бёдра должны быть на расстоянии ~0.5м
sh_center_m = (metric[11] + metric[12]) / 2
hip_center_m = (metric[23] + metric[24]) / 2
torso_reconstructed = float(np.linalg.norm(sh_center_m - hip_center_m))
check(abs(torso_reconstructed - TORSO_LENGTH_M) < 0.02,
      f"torsо в метрике: {torso_reconstructed:.3f}м")

print(f"  z_scale={ws.z_scale:.6f}, confidence={ws.confidence:.3f}")
print()

# ══════════════════════════════════════════════════════════════
# ТЕСТ 2: масштабирование (torso в 2 раза меньше → z_scale ~2×)
# ══════════════════════════════════════════════════════════════

print("=== Тест 2: масштабирование (torso 75px вместо 150px) ===")
ws2 = Workspace3D(smoothing_alpha=0)  # без сглаживания
lm_small = make_landmarks(torso_y_px=75)
result = ws2.calibrate(lm_small)
check(result, "калибровка успешна")
# Масштаб должен быть ровно в 2 раза больше
expected_ratio = 2.0
actual_ratio = ws2.z_scale / ws.z_scale
check(abs(actual_ratio - expected_ratio) < 0.01,
      f"z_scale удвоился: {ws.z_scale:.6f} → {ws2.z_scale:.6f} (ratio={actual_ratio:.3f})")
print()

# ══════════════════════════════════════════════════════════════
# ТЕСТ 3: реальные данные из poses_dataset.csv
# ══════════════════════════════════════════════════════════════

print("=== Тест 3: реальные данные (poses_dataset.csv) ===")
dataset_path = os.path.join(MVM_ROOT, 'pipeline', 'poses_dataset.csv')
if os.path.exists(dataset_path):
    ws_real = Workspace3D(smoothing_alpha=0)
    calibrated_count = 0
    torsos_m = []

    with open(dataset_path) as f:
        reader = csv.reader(f)
        header = next(reader)  # пропускаем заголовок
        for row_idx, row in enumerate(reader):
            if row_idx >= 100:  # первые 100 примеров достаточно
                break
            label = row[0]
            # Парсим f0_x..f32_v
            lm = np.zeros((33, 4), dtype=np.float32)
            for i in range(33):
                base = 1 + i * 4
                lm[i, 0] = float(row[base])     # x
                lm[i, 1] = float(row[base + 1]) # y
                lm[i, 2] = float(row[base + 2]) # z
                lm[i, 3] = float(row[base + 3]) # visibility

            if ws_real.calibrate(lm):
                calibrated_count += 1
                torsos_m.append(ws_real.last_torso_m)

    check(calibrated_count > 0, f"откалибровано {calibrated_count} из 100 примеров")
    if torsos_m:
        avg_torso = np.mean(torsos_m)
        std_torso = np.std(torsos_m)
        check(0.30 < avg_torso < 0.70,
              f"средний torsо: {avg_torso:.3f}м (std={std_torso:.3f}м)")
        print(f"  Диапазон torsо: [{min(torsos_m):.3f}, {max(torsos_m):.3f}]м")
    print()
else:
    print("  SKIP: poses_dataset.csv не найден")
    print()

# ══════════════════════════════════════════════════════════════
# ТЕСТ 4: низкая visibility
# ══════════════════════════════════════════════════════════════

print("=== Тест 4: низкая visibility ===")
ws_low = Workspace3D(smoothing_alpha=0)
lm_low = make_landmarks(torso_y_px=150, vis=0.1)  # почти не видны
result = ws_low.calibrate(lm_low)
check(not result, f"калибровка отклонена при vis=0.1 (result={result})")
check(not ws_low.is_calibrated, "is_calibrated = False при плохой видимости")
print()

# ══════════════════════════════════════════════════════════════
# ТЕСТ 5: EMA-сглаживание
# ══════════════════════════════════════════════════════════════

print("=== Тест 5: EMA-сглаживание ===")
ws_ema = Workspace3D(smoothing_alpha=0.5)

# Первая калибровка: torsо = 0.5м → scale ~0.00333
lm1 = make_landmarks(torso_y_px=150)
ws_ema.calibrate(lm1)
z1 = ws_ema.z_scale

# Вторая калибровка с torsо в 2 раза МЕНЬШЕ (75px → raw_scale ~0.00667)
lm2 = make_landmarks(torso_y_px=75)
ws_ema.calibrate(lm2)
z2 = ws_ema.z_scale

# EMA: z2 = 0.5*z1 + 0.5*raw_scale
raw_z2 = TORSO_LENGTH_M / 75  # ожидаемый raw_scale (75px torsо)
expected_z2 = 0.5 * z1 + 0.5 * raw_z2
check(abs(z2 - expected_z2) < 1e-6,
      f"EMA сработал: z1={z1:.6f}, z2={z2:.6f}, expected={expected_z2:.6f}")
check(z2 > z1, "z_scale вырос (torso меньше → scale больше)")
print()

# ══════════════════════════════════════════════════════════════
# ТЕСТ 6: fallback на лицо
# ══════════════════════════════════════════════════════════════

print("=== Тест 6: fallback на лицо (межзрачковое 63мм) ===")
ws_face = Workspace3D(smoothing_alpha=0)

# Создаём landmarks torsо с плохой видимостью
lm_no_torso = make_landmarks(torso_y_px=150)
lm_no_torso[11:13, 3] = 0.0   # плечи не видны
lm_no_torso[23:25, 3] = 0.0   # бёдра не видны

# А лицо видно: создаём синтетические 478 landmarks (468 + 10 iris)
face_lm = np.zeros((478, 4), dtype=np.float32)
face_lm[:, 3] = 1.0

# Глаза на известном расстоянии: 50px в кадре 640px
interocular_px = 50
interocular_norm = interocular_px / 640  # в нормализованных координатах [0,1]
face_lm[33, :3] = [0.45, 0.35, 0.0]    # левый глаз
face_lm[263, :3] = [0.45 + interocular_norm, 0.35, 0.0]  # правый глаз

result = ws_face._calibrate_from_face(face_lm)
check(result, "калибровка по лицу успешна")

# z_scale = INTEROCULAR_M / interocular_px (после перевода в пиксели)
expected_scale = INTEROCULAR_M / interocular_px
check(abs(ws_face.z_scale - expected_scale) < 1e-6,
      f"z_scale по лицу: {ws_face.z_scale:.6f}, expected={expected_scale:.6f}")
check(ws_face.confidence == 0.7, f"confidence по лицу = 0.7 (получено {ws_face.confidence})")
print()

# ══════════════════════════════════════════════════════════════
# ТЕСТ 7: pixel_to_3d
# ══════════════════════════════════════════════════════════════

print("=== Тест 7: pixel_to_3d ===")
ws_px = Workspace3D()
ws_px._state.z_scale = 1.0
ws_px._state.calibrated = True

# Центр кадра на глубине 2.0 MP-единиц → (0, 0, 2.0) метров
pt = ws_px.pixel_to_3d(320, 240, 2.0)
check(abs(pt[0]) < 0.01, f"X центра ~0: {pt[0]:.3f}")
check(abs(pt[1]) < 0.01, f"Y центра ~0: {pt[1]:.3f}")
check(abs(pt[2] - 2.0) < 0.01, f"Z = 2.0: {pt[2]:.3f}")

# Точка справа-внизу: пиксель (480, 360) на глубине 1.0
pt2 = ws_px.pixel_to_3d(480, 360, 1.0)
check(pt2[0] > 0, f"X > 0 (справа): {pt2[0]:.3f}")
check(pt2[1] > 0, f"Y > 0 (вниз): {pt2[1]:.3f}")
print()

# ══════════════════════════════════════════════════════════════
# ТЕСТ 8: сброс калибровки
# ══════════════════════════════════════════════════════════════

print("=== Тест 8: сброс калибровки ===")
ws_rst = Workspace3D()
ws_rst.calibrate(make_landmarks())
check(ws_rst.is_calibrated, "калиброван до сброса")
ws_rst.reset()
check(not ws_rst.is_calibrated, "не калиброван после сброса")
check(ws_rst.frames_calibrated == 0, "frames_calibrated обнулён")
print()

# ══════════════════════════════════════════════════════════════
# ИТОГ
# ══════════════════════════════════════════════════════════════

total = PASSED + FAILED
print(f"{'='*60}")
print(f"ИТОГО: {PASSED}/{total} пройдено, {FAILED} провалено")
if FAILED == 0:
    print("Все тесты пройдены.")
else:
    print(f"ВНИМАНИЕ: {FAILED} тестов провалено!")
    sys.exit(1)
