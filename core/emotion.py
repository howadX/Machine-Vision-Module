"""
emotion.py — Классификатор эмоций на базе EfficientNet-B0 (файн-тюн FER2013).
Заменяет dlib 128d + SVM.

Использование:
    from emotion import EmotionClassifier
    ec = EmotionClassifier()
    name, confidence = ec.predict(face_image_bgr)  # → ("Радость", 0.87)
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
import numpy as np
import os
from pathlib import Path
from typing import Optional


class EmotionClassifier:
    """EfficientNet-B0, файн-тюн на FER2013. 7 эмоций + нейтральная."""

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        Args:
            model_path: путь к .pth. По умолчанию — models/emotion/efficientnet_b0_fer2013.pth
            use_gpu: использовать CUDA.
        """
        if model_path is None:
            # Ищем модель относительно этого файла
            src = Path(__file__).resolve().parent
            root = src.parent
            model_path = root / "models" / "emotion" / "efficientnet_b0_fer2013.pth"

        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.enabled = False  # совместимость с onnxruntime

        # Загружаем чекпоинт
        ckpt = torch.load(str(model_path), map_location=self.device, weights_only=True)
        config = ckpt["config"]

        self.emotions_ru = config["emotions_ru"]
        self.emotions_en = config["emotions_en"]
        self.image_size = config["image_size"]
        self.num_classes = config["num_classes"]
        self.val_accuracy = config.get("val_accuracy", None)

        # Модель
        self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=self.num_classes)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        """Общий препроцессинг: BGR→RGB, resize, normalize."""
        face_rgb = face_bgr[:, :, ::-1]  # BGR → RGB
        return self.transform(face_rgb).unsqueeze(0).to(self.device)

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        """Предсказывает эмоцию по изображению лица (BGR)."""
        tensor = self._preprocess(face_bgr)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)
        return self.emotions_ru[idx.item()], float(conf.item())

    def predict_top_k(self, face_bgr: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Возвращает топ-k эмоций с уверенностями."""
        tensor = self._preprocess(face_bgr)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        top_probs, top_idx = probs[0].topk(min(k, self.num_classes))
        return [(self.emotions_ru[i.item()], float(p.item())) for i, p in zip(top_idx, top_probs)]

    def __repr__(self):
        gpu_info = f"GPU={torch.cuda.get_device_name(0)}" if self.device.type == "cuda" else "CPU"
        acc_info = f", val_acc={self.val_accuracy:.1f}%" if self.val_accuracy else ""
        return f"EmotionClassifier(model=efficientnet_b0, classes={self.num_classes}, {gpu_info}{acc_info})"
