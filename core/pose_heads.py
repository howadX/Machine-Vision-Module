"""
core/pose_heads.py — Multi-task головы на основе PoseVAE latent space.

Классификатор позы + аналитические heads (ориентация, дистанция, высота).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class PoseClassifier(nn.Module):
    """Классификатор позы: VAE latent z → N классов."""

    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)


class MultiTaskPoseHeads(nn.Module):
    """
    Полный набор голов для позы.

    Классификатор (обучаемый) + аналитические головы (без параметров).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pose_classifier = PoseClassifier(latent_dim, num_classes, hidden_dim, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты классификатора."""
        return self.pose_classifier(z)

    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает (класс, confidence)."""
        logits = self.forward(z)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        return pred, conf


# ── Аналитические головы (без обучения) ───────────────────

def compute_orientation(landmarks_normalized: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет ориентацию тела из нормализованных landmarks [33, 3].

    Returns:
        yaw, pitch, roll в градусах
    """
    pts = landmarks_normalized  # [33, 3], hips=origin, torso along Y

    # Плечи и бёдра
    l_sh = pts[11]; r_sh = pts[12]
    l_hip = pts[23]; r_hip = pts[24]

    # Pitch: угол torso (плечи→бёдра) от вертикали (Y)
    shoulder_center = (l_sh + r_sh) / 2
    hip_center = (l_hip + r_hip) / 2
    torso_vec = shoulder_center - hip_center
    vertical = np.array([0.0, -1.0, 0.0])  # вверх
    torso_norm = np.linalg.norm(torso_vec) or 1.0
    cos_pitch = np.clip(np.dot(torso_vec, vertical) / torso_norm, -1.0, 1.0)
    pitch = float(np.degrees(np.arccos(cos_pitch)))

    # Roll: угол линии плеч от горизонтали
    shoulder_line = r_sh - l_sh
    horizontal = np.array([1.0, 0.0, 0.0])
    sh_norm = np.linalg.norm(shoulder_line) or 1.0
    cos_roll = np.clip(np.dot(shoulder_line, horizontal) / sh_norm, -1.0, 1.0)
    # Проверяем сторону наклона
    roll = float(np.degrees(np.arccos(abs(cos_roll))))
    if shoulder_line[1] != 0:
        roll = roll if r_sh[1] > l_sh[1] else -roll

    # Yaw: приблизительно из соотношения ширины плеч
    shoulder_width = np.linalg.norm(shoulder_line)
    # В нормализованных координатах ширина плеч ≈ 0.7-0.9
    # Если человек повёрнут — ширина меньше
    expected_width = 0.8
    width_ratio = shoulder_width / expected_width
    yaw = float(np.degrees(np.arccos(np.clip(width_ratio, 0.0, 1.0))))
    # Знак: кто ближе к камере
    if l_sh[2] > r_sh[2]:
        yaw = -yaw  # левое плечо ближе = поворот вправо

    return {'yaw': yaw, 'pitch': pitch, 'roll': roll}


def compute_head_height(landmarks_normalized: np.ndarray) -> float:
    """Высота головы над полом в единицах torso."""
    nose = landmarks_normalized[0]
    # В нормализованной системе: hips=0, torso=1, Y вверх = -Y
    # Голова выше бёдер → nose_y отрицательный
    return float(-nose[1])


def compute_body_lean(landmarks_normalized: np.ndarray) -> float:
    """Отклонение тела от вертикали (градусы)."""
    l_sh = landmarks_normalized[11]; r_sh = landmarks_normalized[12]
    l_hip = landmarks_normalized[23]; r_hip = landmarks_normalized[24]
    shoulder_center = (l_sh + r_sh) / 2
    hip_center = (l_hip + r_hip) / 2
    torso_vec = shoulder_center - hip_center
    vertical = np.array([0.0, -1.0, 0.0])
    cos_lean = np.clip(np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) or 1.0), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_lean)))


def compute_movement_magnitude(
    landmarks_history: list,  # list of [33, 4] arrays
) -> float:
    """Среднее смещение landmarks за последние N кадров."""
    if len(landmarks_history) < 2:
        return 0.0
    pts = np.stack([lm[:, :3] for lm in landmarks_history], axis=0)  # [T, 33, 3]
    displacements = np.linalg.norm(pts[1:] - pts[:-1], axis=2)  # [T-1, 33]
    return float(displacements.mean())


def compute_symmetry_score(landmarks_normalized: np.ndarray) -> float:
    """0 = идеально симметрично, >0 = асимметрия."""
    left_parts = landmarks_normalized[[11, 13, 15, 23, 25, 27]]
    right_parts = landmarks_normalized[[12, 14, 16, 24, 26, 28]]
    right_mirrored = right_parts.copy()
    right_mirrored[:, 0] *= -1  # отражаем X
    asymmetry = np.linalg.norm(left_parts - right_mirrored, axis=1).mean()
    return float(asymmetry)


# ── Полный пайплайн ──────────────────────────────────────

def analyze_pose(
    landmarks: list,  # 33 NormalizedLandmark
    vae_model: nn.Module,
    classifier: nn.Module,
    class_names: list,
    device: torch.device,
    history: Optional[list] = None,
) -> Dict:
    """
    Полный анализ позы: классификация + ориентация + контекст.

    Returns:
        dict с ключами: pose, confidence, orientation_*, head_height,
                        body_lean, movement, symmetry, plausibility
    """
    import numpy as np

    # Нормализация
    pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=np.float32)
    from pose_encoder import normalize_landmarks
    norm_pts, meta = normalize_landmarks(pts)

    flat = norm_pts.reshape(1, -1).astype(np.float32)
    x = torch.from_numpy(flat).to(device)

    # VAE → z
    vae_model.eval()
    with torch.no_grad():
        z = vae_model.encode_to_z(x)
        recon, _, _ = vae_model(x)
        recon_error = torch.nn.functional.mse_loss(recon, x).item()
        logpz = vae_model.log_prob_z(x).item()

    # Классификация
    classifier.eval()
    with torch.no_grad():
        logits = classifier(z)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        confidence = float(probs.max())

    # Аналитические признаки
    orientation = compute_orientation(norm_pts)
    head_height = compute_head_height(norm_pts)
    body_lean = compute_body_lean(norm_pts)
    symmetry = compute_symmetry_score(norm_pts)
    movement = compute_movement_magnitude(history) if history else 0.0

    return {
        'pose': class_names[pred_idx] if pred_idx < len(class_names) else f'class_{pred_idx}',
        'pose_confidence': round(confidence, 2),
        'orientation_yaw': round(orientation['yaw'], 1),
        'orientation_pitch': round(orientation['pitch'], 1),
        'orientation_roll': round(orientation['roll'], 1),
        'head_height': round(head_height, 2),
        'body_lean': round(body_lean, 1),
        'movement': round(movement, 4),
        'symmetry': round(symmetry, 3),
        'recon_error': round(recon_error, 4),
        'plausibility_logpz': round(logpz, 2),
        'plausibility': 'normal' if logpz < 14.3 else 'unusual',
    }
