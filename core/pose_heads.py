"""
core/pose_heads.py — Multi-task головы на основе PoseVAE latent space.

Классификатор позы + Prototypical (few-shot) + Experience Replay (continual).
"""

import torch
import torch.nn as nn
import numpy as np
import pickle, os
from typing import Tuple, Optional, Dict, List
from collections import defaultdict


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


class PrototypicalClassifier:
    """
    Few-shot классификатор: centroids в latent space.
    Новый класс = 5 примеров → вычислить центроид → готово.
    Без переобучения нейросети.
    """

    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        self.prototypes: Dict[str, np.ndarray] = {}  # class_name → centroid [latent_dim]
        self.class_names: List[str] = []

    @property
    def num_classes(self) -> int:
        return len(self.prototypes)

    def add_class(self, name: str, z_vectors: np.ndarray):
        """Добавляет новый класс: name + примеры z-векторов → вычисляет центроид."""
        if z_vectors.ndim == 1:
            z_vectors = z_vectors.reshape(1, -1)
        centroid = z_vectors.mean(axis=0)
        self.prototypes[name] = centroid
        if name not in self.class_names:
            self.class_names.append(name)

    def classify(self, z: np.ndarray) -> Tuple[str, float]:
        """
        Классифицирует z-вектор по ближайшему центроиду.

        Returns:
            (class_name, confidence) — confidence = 1 / (1 + distance)
        """
        if z.ndim == 1:
            z = z.reshape(1, -1)

        best_name = "_unknown"
        best_dist = float('inf')

        for name, proto in self.prototypes.items():
            dist = float(np.linalg.norm(z - proto))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        confidence = 1.0 / (1.0 + best_dist)
        return best_name, confidence

    def classify_batch(self, Z: np.ndarray) -> List[Tuple[str, float]]:
        """Классифицирует батч z-векторов."""
        return [self.classify(z) for z in Z]

    def save(self, path: str):
        """Сохраняет прототипы."""
        data = {
            'prototypes': {k: v.tolist() for k, v in self.prototypes.items()},
            'class_names': self.class_names,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Загружает прототипы."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.prototypes = {k: np.array(v, dtype=np.float32) for k, v in data['prototypes'].items()}
        self.class_names = data['class_names']

    @classmethod
    def from_classifier(cls, clf: PoseClassifier, z_samples: Dict[int, np.ndarray],
                         class_names: List[str]) -> 'PrototypicalClassifier':
        """
        Создаёт PrototypicalClassifier из обученного PoseClassifier.

        Args:
            clf: обученный классификатор
            z_samples: {label: [N, latent_dim]} — эмбеддинги для каждого класса
            class_names: имена классов
        """
        proto = cls(latent_dim=z_samples[0].shape[1])
        for label, name in enumerate(class_names):
            if label in z_samples and len(z_samples[label]) > 0:
                proto.add_class(name, z_samples[label])
        return proto


class ExperienceReplayBuffer:
    """
    Буфер для continual learning: хранит ограниченное число z-векторов на класс.
    При обучении нового класса подмешивает старые примеры → нет забывания.
    """

    def __init__(self, max_per_class: int = 50):
        self.max_per_class = max_per_class
        self.buffer: Dict[str, List[np.ndarray]] = defaultdict(list)

    def add(self, class_name: str, z_vector: np.ndarray):
        """Добавляет z-вектор в буфер."""
        self.buffer[class_name].append(z_vector.flatten())
        if len(self.buffer[class_name]) > self.max_per_class:
            self.buffer[class_name] = self.buffer[class_name][-self.max_per_class:]

    def add_batch(self, class_name: str, z_vectors: np.ndarray):
        """Добавляет батч z-векторов."""
        for z in z_vectors:
            self.add(class_name, z)

    def get_all(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Возвращает все данные буфера: (Z, labels, class_names).
        Z: [N, latent_dim], labels: [N]
        """
        Z_list, y_list = [], []
        for label_idx, (name, vectors) in enumerate(self.buffer.items()):
            for z in vectors:
                Z_list.append(z)
                y_list.append(label_idx)

        if not Z_list:
            return np.array([]), np.array([]), []

        return np.array(Z_list), np.array(y_list), list(self.buffer.keys())

    def save(self, path: str):
        """Сохраняет буфер."""
        data = {
            'max_per_class': self.max_per_class,
            'buffer': {k: [v.tolist() for v in vecs] for k, vecs in self.buffer.items()},
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Загружает буфер."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.max_per_class = data['max_per_class']
        self.buffer = defaultdict(list)
        for k, vecs in data['buffer'].items():
            self.buffer[k] = [np.array(v, dtype=np.float32) for v in vecs]

    def __len__(self):
        return sum(len(v) for v in self.buffer.values())

    def stats(self) -> str:
        """Статистика буфера."""
        parts = [f"{k}: {len(v)}" for k, v in sorted(self.buffer.items())]
        return f"ReplayBuffer({len(self)} total): " + ", ".join(parts)


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
    roll = float(np.degrees(np.arccos(abs(cos_roll))))
    if shoulder_line[1] != 0:
        roll = roll if r_sh[1] > l_sh[1] else -roll

    # Yaw: приблизительно из соотношения ширины плеч
    shoulder_width = np.linalg.norm(shoulder_line)
    expected_width = 0.8
    width_ratio = shoulder_width / expected_width
    yaw = float(np.degrees(np.arccos(np.clip(width_ratio, 0.0, 1.0))))
    if l_sh[2] > r_sh[2]:
        yaw = -yaw

    return {'yaw': yaw, 'pitch': pitch, 'roll': roll}


def compute_head_height(landmarks_normalized: np.ndarray) -> float:
    """Высота головы над полом в единицах torso."""
    nose = landmarks_normalized[0]
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
    landmarks_history: list,
) -> float:
    """Среднее смещение landmarks за последние N кадров."""
    if len(landmarks_history) < 2:
        return 0.0
    pts = np.stack([lm[:, :3] for lm in landmarks_history], axis=0)
    displacements = np.linalg.norm(pts[1:] - pts[:-1], axis=2)
    return float(displacements.mean())


def compute_symmetry_score(landmarks_normalized: np.ndarray) -> float:
    """0 = идеально симметрично, >0 = асимметрия."""
    left_parts = landmarks_normalized[[11, 13, 15, 23, 25, 27]]
    right_parts = landmarks_normalized[[12, 14, 16, 24, 26, 28]]
    right_mirrored = right_parts.copy()
    right_mirrored[:, 0] *= -1
    asymmetry = np.linalg.norm(left_parts - right_mirrored, axis=1).mean()
    return float(asymmetry)
