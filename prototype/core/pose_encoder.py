"""
core/pose_encoder.py — Spatial Encoding для 33 COCO landmarks.
Превращает сырые landmarks в нормализованный вектор признаков для VAE.

Pipeline:
  33 landmarks [33×(x,y,z,visibility)]
  → нормализация (translate/scale/rotate)
  → Joint Type Embeddings [33×16]
  → Инварианты: pairwise distances [528] + bone angles [12]
  → Выход: нормализованные координаты [99] + embeddings [528] + инварианты [528+12]
  → Итоговый вектор для VAE: flat вектор или структурированный тензор
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# COCO 33 keypoints — семантические группы
JOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky', 'right_pinky',
    'left_index', 'right_index',
    'left_thumb', 'right_thumb',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]

# Кости скелета (рёбра графа) — COCO skeleton
BONES = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso
    (23, 24),  # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]

# Группы для нормализации
UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # face
TORSO = [11, 12, 23, 24]  # shoulders + hips
ARMS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
LEGS = [25, 26, 27, 28, 29, 30, 31, 32]


def normalize_landmarks(
    landmarks: np.ndarray,  # [33, 4] — x, y, z, visibility
    ref_hips: bool = True,
    ref_shoulders: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Нормализация: translate to hip center → scale by torso length → rotate to vertical.

    Returns:
        normalized: [33, 3] — x, y, z
        meta: dict с параметрами нормализации (для отладки)
    """
    pts = landmarks[:, :3].copy().astype(np.float32)

    # Видимость
    vis = landmarks[:, 3]

    # Центр бёдер (landmarks 23, 24 — left/right hip)
    hip_vis = vis[23] + vis[24]
    if hip_vis > 0.1:
        hip_center = (pts[23] * vis[23] + pts[24] * vis[24]) / max(hip_vis, 1e-6)
    else:
        hip_center = pts[0].copy()  # fallback: nose

    # Translate
    pts -= hip_center

    # Плечи для масштаба и ориентации
    sh_vis = vis[11] + vis[12]
    if sh_vis > 0.1:
        sh_center = (pts[11] * vis[11] + pts[12] * vis[12]) / max(sh_vis, 1e-6)
    else:
        sh_center = pts[0].copy()

    # Масштаб: torso length (hips → shoulders) = 1.0
    torso_vec = sh_center  # уже относительно hip_center (=0)
    torso_len = float(np.linalg.norm(torso_vec))
    if torso_len < 0.01:
        # Fallback: shoulder width
        shoulder_width = float(np.linalg.norm(pts[11] - pts[12]))
        torso_len = max(shoulder_width, 0.01)

    pts /= max(torso_len, 1e-6)

    # Поворот: позвоночник вдоль оси Y (vertical)
    # В MediaPipe: x — горизонталь, y — вертикаль вниз, z — глубина
    # После translate hip_center=0: shoulders должны быть вверх (y < 0 в image coords)
    # Но мы хотим Y вдоль позвоночника вверх
    if np.linalg.norm(torso_vec) > 0.01:
        vertical_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # вверх в image coords
        torso_dir = torso_vec / max(np.linalg.norm(torso_vec), 1e-6)

        # Угол между torso и вертикалью
        dot = np.clip(np.dot(torso_dir, vertical_axis), -1.0, 1.0)
        angle = np.arccos(dot)

        if angle > 0.01:
            # Ось поворота: cross product
            axis = np.cross(torso_dir, vertical_axis)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis /= axis_norm
                # Rodrigues rotation
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ], dtype=np.float32)
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                pts = pts @ R.T

    meta = {
        'hip_center': hip_center.tolist(),
        'torso_len': float(torso_len),
        'hip_vis': float(hip_vis),
        'sh_vis': float(sh_vis),
    }
    return pts.astype(np.float32), meta


def compute_invariants(normalized_pts: np.ndarray) -> np.ndarray:
    """
    Инвариантные признаки: попарные расстояния + углы костей.

    Args:
        normalized_pts: [33, 3] — уже нормализованные координаты

    Returns:
        features: [540] — 528 pairwise distances + 12 bone angles
    """
    # Попарные расстояния (верхний треугольник без диагонали)
    diffs = normalized_pts[:, None, :] - normalized_pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)  # [33, 33]
    triu_idx = np.triu_indices(33, k=1)
    pairwise = dists[triu_idx]  # [528]

    # Углы костей
    angles = []
    for parent, child in BONES:
        bone_vec = normalized_pts[child] - normalized_pts[parent]
        bone_len = np.linalg.norm(bone_vec)
        if bone_len > 1e-6:
            # Угол с вертикалью
            vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            cos_angle = np.clip(np.dot(bone_vec, vertical) / bone_len, -1.0, 1.0)
            angles.append(float(np.arccos(cos_angle)))
        else:
            angles.append(0.0)

    return np.concatenate([pairwise, np.array(angles, dtype=np.float32)])


def compute_movement_features(
    landmarks_history: list,  # list of [33, 4] arrays (последние T кадров)
) -> np.ndarray:
    """
    Temporal признаки: дисперсия и скорость для каждой точки.

    Args:
        landmarks_history: список из T массивов [33, 4]

    Returns:
        features: [33] — variance per joint (magnitude)
    """
    if len(landmarks_history) < 2:
        return np.zeros(33, dtype=np.float32)

    pts_stack = np.stack([lm[:, :3] for lm in landmarks_history], axis=0)  # [T, 33, 3]
    variance = np.var(pts_stack, axis=0)  # [33, 3]
    variance_magnitude = np.linalg.norm(variance, axis=1)  # [33]

    return variance_magnitude.astype(np.float32)


def extract_features(
    landmarks: list,  # list of NormalizedLandmark (33 шт.)
    history: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Полный пайплайн извлечения признаков из PoseLandmarker result.

    Args:
        landmarks: список из 33 NormalizedLandmark объектов
        history: опционально — список предыдущих landmarks для temporal

    Returns:
        features: [99] — нормализованные координаты (для VAE encoder)
        meta: dict — параметры нормализации + инварианты
    """
    pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=np.float32)

    normalized, norm_meta = normalize_landmarks(pts)
    invariants = compute_invariants(normalized)

    # Движение
    if history:
        movement = compute_movement_features(history)
    else:
        movement = np.zeros(33, dtype=np.float32)

    # Плоский вектор нормализованных координат
    flat_coords = normalized.reshape(-1)  # [99]

    meta = {
        **norm_meta,
        'invariants_mean': float(np.mean(invariants[:528])),
        'movement_mean': float(np.mean(movement)),
    }

    return flat_coords, invariants, meta


# ── Для обучения VAE: подготовка входа ───────────────────
def prepare_vae_input(
    landmarks: list,
    use_invariants: bool = True,
) -> np.ndarray:
    """
    Готовит входной вектор для VAE.

    Args:
        landmarks: 33 NormalizedLandmark
        use_invariants: добавлять ли инвариантные признаки

    Returns:
        vector: [99] или [99 + 540]
    """
    flat_coords, invariants, _meta = extract_features(landmarks)

    if use_invariants:
        return np.concatenate([flat_coords, invariants]).astype(np.float32)

    return flat_coords.astype(np.float32)
