"""
emotion_hse.py — Классификатор эмоций на базе HSEmotion (ВШЭ, русские лица).

HSEmotion: EfficientNet-B0, обучен на AffectNet + VGAF (русские лица).
Точность: ~73% на RAF-DB, обучен на 8 эмоциях (включая нейтральную).

Использование:
    from emotion_hse import HSEmotionClassifier
    ec = HSEmotionClassifier()
    name, conf = ec.predict(face_bgr)  # → ("Happiness", 0.87)
"""
import torch
import timm
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Optional


# ── Фикс для PyTorch 2.6+ (weights_only=True по умолчанию) ──
# HSEmotion-модель загружается через torch.load с weights_only=False
import torch as _torch
_orig_load = _torch.load
def _load_with_weights_only_false(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
_torch.load = _load_with_weights_only_false

torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])

# ── Фикс совместимости старой HSEmotion-модели с новым timm ──
# В старых версиях timm conv_s2d → в новых conv_dw
def _patch_depthwise_separable_conv(module):
    """Рекурсивно добавляет conv_s2d как alias для conv_dw."""
    from timm.models._efficientnet_blocks import DepthwiseSeparableConv
    if isinstance(module, DepthwiseSeparableConv):
        if not hasattr(module, 'conv_s2d') and hasattr(module, 'conv_dw'):
            object.__setattr__(module, 'conv_s2d', module.conv_dw)
    for child in module.children():
        _patch_depthwise_separable_conv(child)


class HSEmotionClassifier:
    """Обёртка над HSEmotionRecognizer с авто-фиксом совместимости."""

    def __init__(self, model_name: str = "enet_b0_8_best_vgaf", use_gpu: bool = True):
        from hsemotion.facial_emotions import HSEmotionRecognizer

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cudnn.enabled = False

        self.recognizer = HSEmotionRecognizer(model_name=model_name, device=device)
        self.class_names = [self.recognizer.idx_to_class[i] for i in sorted(self.recognizer.idx_to_class)]
        self.device = device

        # Патчим модель после загрузки
        _patch_depthwise_separable_conv(self.recognizer.model)

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        """Предсказывает эмоцию по BGR-изображению лица."""
        face_rgb = face_bgr[:, :, ::-1]
        _, scores = self.recognizer.predict_emotions(face_rgb, logits=False)
        idx = scores.argmax()
        return self.class_names[idx], float(scores[idx])

    def predict_top_k(self, face_bgr: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Топ-k эмоций с вероятностями."""
        face_rgb = face_bgr[:, :, ::-1]
        _, scores = self.recognizer.predict_emotions(face_rgb, logits=False)
        top_idx = scores.argsort()[-k:][::-1]
        return [(self.class_names[i], float(scores[i])) for i in top_idx]

    def __repr__(self):
        return f"HSEmotionClassifier(model=enet_b0_8, classes={len(self.class_names)}, {self.device})"
