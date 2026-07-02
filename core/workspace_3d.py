"""
core/workspace_3d.py — единое метрическое 3D-пространство MVM.

Назначение:
  Калибрует относительные MediaPipe-координаты (x,y в пикселях, z — безразмерная)
  в метрические XYZ с началом в камере.

Принцип калибровки:
  torso (shoulders → hips) ≈ 0.5 м у взрослого человека.
  Измеряем torsо в MediaPipe-единицах → вычисляем коэффициент перевода в метры.

Система координат:
  origin  = камера (0, 0, 0)
  X       = вправо (как в изображении)
  Y       = вниз (как в изображении)
  Z       = вперёд (глубина от камеры)

Устойчивость:
  - Проверка visibility ключевых точек торса (>= 0.35)
  - Smooth-калибровка: exponential moving average для z_scale
  - Валидация: torsо в разумных пределах (0.3—0.7 м)
  - Fallback: если калибровка невозможна — флаг uncalibrated, сырые координаты
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field


# Индексы COCO-landmarks для калибровки
SHOULDERS = (11, 12)
HIPS = (23, 24)

# Физические константы
TORSO_LENGTH_M = 0.5        # средняя длина torsо (shoulders→hips), метры
FACE_HEIGHT_M = 0.19        # средняя высота лица (hairline→chin), метры
INTEROCULAR_M = 0.063       # среднее межзрачковое расстояние, метры

# Пороги
MIN_VISIBILITY = 0.35       # минимальная visibility для точки
MIN_TORSO_MEDIA_UNITS = 0.01  # минимальная длина torsо в MP-единицах
VALID_TORSO_MIN_M = 0.30    # минимальная реалистичная длина torsо
VALID_TORSO_MAX_M = 0.75    # максимальная реалистичная длина torsо

# Сглаживание
DEFAULT_SMOOTHING = 0.3     # alpha для EMA (0 — нет сглаживания, 1 — без обновления)


@dataclass
class CalibrationState:
    """Состояние калибровки."""
    z_scale: float = 1.0          # метров на MediaPipe-единицу
    calibrated: bool = False
    frames_calibrated: int = 0
    last_torso_mp: float = 0.0
    last_torso_m: float = 0.0
    confidence: float = 0.0       # 0—1, качество калибровки
    estimated_distance: float = 0.0  # pinhole-оценка расстояния (метры)


class Workspace3D:
    """
    Единое метрическое 3D-пространство для MVM.

    Использование:
        ws = Workspace3D(image_width=640, image_height=480)
        ws.calibrate(pose_landmarks_33x4)
        body_xyz = ws.to_metric(pose_landmarks_33x4)
        hand_xyz = ws.to_metric(hand_landmarks_21x4)
        dist = ws.distance(hand_xyz[8], cup_xyz)
    """

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        smoothing_alpha: float = DEFAULT_SMOOTHING,
    ):
        """
        Args:
            image_width, image_height: размер кадра в пикселях
            fx, fy: фокусное расстояние (пиксели). Если None — оценка 0.8 × width
            cx, cy: оптический центр. Если None — centre кадра
            smoothing_alpha: alpha для EMA-сглаживания z_scale (0 — без сглаживания)
        """
        self.w = image_width
        self.h = image_height
        self.fx = fx or (image_width * 0.8)
        self.fy = fy or (image_height * 0.8)
        self.cx = cx or (image_width / 2.0)
        self.cy = cy or (image_height / 2.0)
        self._alpha = max(0.0, min(1.0, smoothing_alpha))

        self._state = CalibrationState()

    # ══════════════════════════════════════════════════════════════
    # КАЛИБРОВКА
    # ══════════════════════════════════════════════════════════════

    def calibrate(self, landmarks_33x4: np.ndarray) -> bool:
        """
        Калибрует scale по torsо (shoulders → hips).

        Переводит нормализованные x,y в пиксельные координаты и вычисляет
        3D-вектор torsо в «пиксельных» единицах (z остаётся как есть —
        MediaPipe калибрует z примерно сравнимым с x,y).

        Args:
            landmarks_33x4: [33, 4] — x, y, z, visibility от PoseLandmarker
                            (x,y в [0,1], z — относительная глубина)

        Returns:
            True если калибровка обновлена, False если недостаточно данных
        """
        if landmarks_33x4.shape != (33, 4):
            return False

        vis = landmarks_33x4[:, 3]

        # Проверка visibility ключевых точек
        sh_vis = vis[11] + vis[12]
        hip_vis = vis[23] + vis[24]

        if sh_vis < MIN_VISIBILITY * 2 or hip_vis < MIN_VISIBILITY * 2:
            # Пробуем fallback: калибровка по лицу
            return self._calibrate_from_face(landmarks_33x4)

        # Центры плеч и бёдер — переводим x,y в пиксели
        pts = landmarks_33x4[:, :3].copy()
        pts[:, 0] *= self.w   # x: [0,1] → пиксели
        pts[:, 1] *= self.h   # y: [0,1] → пиксели
        # z остаётся как есть (MediaPipe-единицы глубины)

        sh_center = (pts[11] * vis[11] + pts[12] * vis[12]) / max(sh_vis, 1e-6)
        hip_center = (pts[23] * vis[23] + pts[24] * vis[24]) / max(hip_vis, 1e-6)

        torso_vec = sh_center - hip_center
        torso_mp = float(np.linalg.norm(torso_vec))

        if torso_mp < MIN_TORSO_MEDIA_UNITS:
            return False

        # scale = метров на «пиксельную» единицу (включая z)
        raw_scale = TORSO_LENGTH_M / torso_mp

        # Pinhole distance estimation: z = (fx × TORSO_LENGTH_M) / torso_px
        torso_2d = float(np.linalg.norm(torso_vec[:2]))
        if torso_2d > 0:
            raw_dist = (self.fx * TORSO_LENGTH_M) / torso_2d
            if self._state.estimated_distance > 0:
                self._state.estimated_distance = (
                    self._alpha * self._state.estimated_distance +
                    (1 - self._alpha) * raw_dist)
            else:
                self._state.estimated_distance = raw_dist

        # Валидация: восстановленный torsо должен быть в разумных пределах
        torso_m_check = torso_mp * raw_scale
        if torso_m_check < VALID_TORSO_MIN_M or torso_m_check > VALID_TORSO_MAX_M:
            return False

        # EMA-сглаживание
        if self._state.calibrated and self._alpha > 0:
            self._state.z_scale = (self._alpha * self._state.z_scale +
                                   (1 - self._alpha) * raw_scale)
        else:
            self._state.z_scale = raw_scale

        self._state.calibrated = True
        self._state.frames_calibrated += 1
        self._state.last_torso_mp = torso_mp
        self._state.last_torso_m = torso_mp * self._state.z_scale

        # Confidence: насколько torsо близок к ожидаемым 0.5м
        deviation = abs(self._state.last_torso_m - TORSO_LENGTH_M) / TORSO_LENGTH_M
        self._state.confidence = max(0.0, 1.0 - deviation)

        return True

    def _calibrate_from_face(self, face_landmarks_468x4: np.ndarray) -> bool:
        """
        Fallback-калибровка по лицу (межзрачковое расстояние ≈ 63мм).
        Используется когда torsо не виден.

        Args:
            face_landmarks_468x4: от FaceLandmarker (468 точек × 4)
                                 (x,y в [0,1], z — относительная)
        """
        if face_landmarks_468x4.shape[0] < 468:
            return False

        vis = face_landmarks_468x4[:, 3]

        # Глаза: индексы 33 (левый) и 263 (правый) в 468-точечной модели
        left_eye = face_landmarks_468x4[33, :3].copy()
        right_eye = face_landmarks_468x4[263, :3].copy()

        eye_vis = vis[33] + vis[263]
        if eye_vis < MIN_VISIBILITY * 2:
            return False

        # Переводим x,y в пиксели
        left_eye[0] *= self.w
        left_eye[1] *= self.h
        right_eye[0] *= self.w
        right_eye[1] *= self.h

        interocular_mp = float(np.linalg.norm(right_eye - left_eye))

        if interocular_mp < MIN_TORSO_MEDIA_UNITS:
            return False

        raw_scale = INTEROCULAR_M / interocular_mp

        # Мягкая валидация (лицо менее надёжно чем torsо)
        if raw_scale <= 0:
            return False

        if self._state.calibrated and self._alpha > 0:
            self._state.z_scale = (self._alpha * self._state.z_scale +
                                   (1 - self._alpha) * raw_scale)
        else:
            self._state.z_scale = raw_scale

        self._state.calibrated = True
        self._state.frames_calibrated += 1
        self._state.confidence = 0.7  # ниже чем torsо

        return True

    # ══════════════════════════════════════════════════════════════
    # КОНВЕРТЕРЫ
    # ══════════════════════════════════════════════════════════════

    def to_metric(self, landmarks_nx4: np.ndarray) -> np.ndarray:
        """
        Конвертирует landmarks в метрические XYZ.

        Args:
            landmarks_nx4: [N, 4] — x, y, z, visibility в MediaPipe-формате
                           (x,y в пикселях [0,1], z — относительная, vis — [0,1])

        Returns:
            [N, 3] — XYZ в метрах. Камера в (0,0,0).
        """
        pts = landmarks_nx4[:, :3].copy().astype(np.float64)

        if self._state.calibrated:
            scale = self._state.z_scale
        else:
            scale = 1.0

        # x,y: нормализованные → пиксели → метры
        # (делим на кадр-координату, потом умножаем на z_scale)
        pts[:, 0] = (pts[:, 0] - 0.5) * self.w * scale
        pts[:, 1] = (pts[:, 1] - 0.5) * self.h * scale
        # Z: pinhole distance (м) — всё тело примерно на одном расстоянии
        if self._state.estimated_distance > 0:
            pts[:, 2] = self._state.estimated_distance
        else:
            pts[:, 2] = pts[:, 2] * scale

        return pts.astype(np.float32)

    def pixel_to_3d(self, px: float, py: float, depth_mp: float) -> np.ndarray:
        """
        Конвертирует 2D-пиксель + MediaPipe-глубину → 3D-точку в метрах.

        Используется для объектов: YOLO bbox центр + depth map значение.

        Args:
            px, py: координаты пикселя
            depth_mp: глубина в MediaPipe-единицах (или относительная)

        Returns:
            [3] — XYZ в метрах
        """
        if not self._state.calibrated:
            return np.array([px, py, depth_mp], dtype=np.float32)

        z = depth_mp * self._state.z_scale
        x = (px - self.cx) * z / self.fx
        y = (py - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════
    # ЗАПРОСЫ
    # ══════════════════════════════════════════════════════════════

    def distance(self, a_xyz: np.ndarray, b_xyz: np.ndarray) -> float:
        """Евклидово расстояние между двумя 3D-точками в метрах."""
        return float(np.linalg.norm(a_xyz[:3] - b_xyz[:3]))

    # ══════════════════════════════════════════════════════════════
    # СВОЙСТВА
    # ══════════════════════════════════════════════════════════════

    @property
    def is_calibrated(self) -> bool:
        return self._state.calibrated

    @property
    def z_scale(self) -> float:
        return self._state.z_scale

    @property
    def confidence(self) -> float:
        return self._state.confidence

    @property
    def frames_calibrated(self) -> int:
        return self._state.frames_calibrated

    @property
    def estimated_distance(self) -> float:
        return self._state.estimated_distance

    @property
    def last_torso_m(self) -> float:
        return self._state.last_torso_m

    @property
    def state(self) -> CalibrationState:
        return self._state

    def reset(self):
        """Сбрасывает калибровку."""
        self._state = CalibrationState()

    def __repr__(self) -> str:
        if not self._state.calibrated:
            return "Workspace3D(uncalibrated)"
        return (f"Workspace3D(calibrated, z_scale={self._state.z_scale:.4f} m/mp, "
                f"confidence={self._state.confidence:.2f}, "
                f"frames={self._state.frames_calibrated})")
