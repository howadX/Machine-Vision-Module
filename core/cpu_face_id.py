"""
core/cpu_face_id.py — CPU-native face identity pipeline.

Детектор: YuNet (228 KB ONNX, Apache 2.0)
Identity: SFace MobileFaceNet (37 MB ONNX, Apache 2.0)

Архитектура: CPU-only. Ноль зависимостей от CUDA/GPU.
Совместимость: 128d L2-нормализованные эмбеддинги.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# ── Пороги ──
DEFAULT_DET_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_IDENTITY_THRESHOLD = 0.45  # cosine similarity для SFace (ниже чем ArcFace 0.35)

# ── Пути к моделям ──
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "cpu"
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"


class CPUFaceID:
    """CPU-native face detection + recognition pipeline."""

    def __init__(
        self,
        det_threshold: float = DEFAULT_DET_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
        image_size: Tuple[int, int] = (640, 480),
    ):
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.identity_threshold = identity_threshold
        self.image_size = image_size

        yunet_path = str(MODEL_DIR / YUNET_MODEL)
        sface_path = str(MODEL_DIR / SFACE_MODEL)

        if not os.path.exists(yunet_path):
            raise FileNotFoundError(f"YuNet model not found: {yunet_path}")
        if not os.path.exists(sface_path):
            raise FileNotFoundError(f"SFace model not found: {sface_path}")

        # ── YuNet детектор ──
        self._detector = cv2.FaceDetectorYN.create(
            yunet_path, "",
            image_size,
            score_threshold=det_threshold,
            nms_threshold=nms_threshold,
            top_k=5000,
        )

        # ── SFace распознаватель ──
        self._recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

        # ── Прогрев ──
        dummy = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        self._detector.setInputSize(image_size)
        self._detector.detect(dummy)
        dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
        self._recognizer.feature(dummy_face)

    # ══════════════════════════════════════════════════════════════
    # ДЕТЕКЦИЯ
    # ══════════════════════════════════════════════════════════════

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Детекция лиц на BGR-изображении.

        Returns:
            [{bbox, landmarks, score}, ...]
            bbox: (x, y, w, h) int
            landmarks: (5, 2) float32 — глаза, нос, уголки рта
            score: float
        """
        h, w = image.shape[:2]
        if (w, h) != self.image_size:
            # Создаём новый детектор под размер
            detector = cv2.FaceDetectorYN.create(
                str(MODEL_DIR / YUNET_MODEL), "",
                (w, h),
                score_threshold=self.det_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000,
            )
            detector.setInputSize((w, h))
        else:
            detector = self._detector
            detector.setInputSize((w, h))

        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            bbox = face[:4].astype(int)
            landmarks = face[4:14].reshape(5, 2)
            score = float(face[14])
            results.append({
                "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                "landmarks": landmarks,
                "score": score,
            })
        return results

    # ══════════════════════════════════════════════════════════════
    # ЭМБЕДДИНГ
    # ══════════════════════════════════════════════════════════════

    def get_embedding(
        self, image: np.ndarray, landmarks: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Извлекает 128d SFace эмбеддинг.

        Если landmarks не указаны — сначала детектирует лицо.
        Эмбеддинг L2-нормализован.

        Args:
            image: BGR-изображение
            landmarks: (5, 2) ключевые точки (опционально)

        Returns:
            128d float32 вектор, L2-нормализованный, или None
        """
        if landmarks is None:
            faces = self.detect(image)
            if not faces:
                return None
            landmarks = faces[0]["landmarks"]

        aligned = self._recognizer.alignCrop(image, landmarks)
        if aligned is None or aligned.size == 0:
            return None

        embedding = self._recognizer.feature(aligned)
        if embedding is None:
            return None

        # L2-нормализация
        emb = embedding[0].astype(np.float64)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb.astype(np.float32)

    # ══════════════════════════════════════════════════════════════
    # ИДЕНТИФИКАЦИЯ
    # ══════════════════════════════════════════════════════════════

    def identify(
        self,
        embedding: np.ndarray,
        gallery: Dict[str, np.ndarray],
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Идентифицирует человека по эмбеддингу.

        Args:
            embedding: 128d L2-нормализованный вектор
            gallery: {name: 128d_embedding}
            threshold: порог косинусного сходства (default: self.identity_threshold)

        Returns:
            (name, confidence) или (None, 0.0)
        """
        if threshold is None:
            threshold = self.identity_threshold

        best_name = None
        best_sim = -1.0

        for name, ref_emb in gallery.items():
            sim = float(np.dot(embedding, ref_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= threshold:
            return best_name, round(best_sim, 4)
        return None, round(best_sim, 4)

    # ══════════════════════════════════════════════════════════════
    # ГАЛЕРЕЯ
    # ══════════════════════════════════════════════════════════════

    def build_gallery(
        self, reference_dir: str, name_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Строит галерею эмбеддингов из эталонных фото.

        Args:
            reference_dir: директория с .jpg файлами (формат p_X.jpg)
            name_map: {key: human_name} (опционально)

        Returns:
            {name: 128d_embedding}
        """
        gallery = {}
        for fname in sorted(os.listdir(reference_dir)):
            if not fname.endswith(".jpg"):
                continue
            path = os.path.join(reference_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            emb = self.get_embedding(img)
            if emb is None:
                continue

            key = fname[2:-4]  # p_X.jpg → X
            name = name_map.get(key, key) if name_map else key
            gallery[name] = emb

        return gallery
