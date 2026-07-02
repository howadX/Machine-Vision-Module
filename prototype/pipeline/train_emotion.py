#!/usr/bin/env python3
"""
train_emotion.py v2 — Исправленное обучение EfficientNet-B0 на FER2013.

Исправления:
  - Label Smoothing (0.1) — против overconfidence
  - Class weights — балансировка дисбаланса
  - Сильные аугментации (RandAugment)
  - 25 эпох + warmup + cosine annealing
  - Weight decay 1e-4
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import timm
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path

# ── Конфигурация ──────────────────────────────────────
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
MODELS_DIR = ROOT / "models" / "emotion"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "efficientnet_b0_fer2013.pth"

BATCH_SIZE = 64
EPOCHS = 25
LR = 3e-4                    # ниже LR — стабильнее
IMAGE_SIZE = 224
NUM_CLASSES = 7
LABEL_SMOOTHING = 0.1        # ключевое: модель не смеет быть уверена на 100%
WEIGHT_DECAY = 1e-4

EMOTIONS_RU = ["Злость", "Отвращение", "Страх", "Радость", "Грусть", "Удивление", "Нейтральная"]
EMOTIONS_EN = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}")
if DEVICE.type == "cuda":
    torch.backends.cudnn.enabled = False
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Загрузка FER2013 ──────────────────────────────────
def download_fer2013():
    import kagglehub
    print("Скачивание FER2013 через kagglehub...")
    path = kagglehub.dataset_download("msambare/fer2013")
    print(f"  Скачано: {path}")

    images, labels = [], []
    for split in ["train", "test"]:
        split_dir = Path(path) / split
        if not split_dir.exists():
            continue
        for label_idx, emotion in enumerate(EMOTIONS_EN):
            em_dir = split_dir / emotion
            if not em_dir.exists():
                continue
            for img_path in sorted(em_dir.glob("*.jpg"))[:5000]:
                import PIL.Image as PILImage
                pil_img = PILImage.open(img_path).convert("L")
                img = np.array(T.functional.pil_to_tensor(pil_img).squeeze(0))
                images.append(img)
                labels.append(label_idx)

    images, labels = np.array(images), np.array(labels)
    print(f"  FER2013: {len(images)} изображений")

    # ── Добавляем пользовательские эмоции (с оверсемплингом) ──
    my_emotions_dir = ROOT / "data" / "my_emotions"
    if my_emotions_dir.exists():
        my_count = 0
        # Сначала собираем пользовательские изображения
        my_images = []
        my_labels_list = []
        for label_idx, emotion in enumerate(EMOTIONS_EN):
            em_dir = my_emotions_dir / emotion
            if not em_dir.exists():
                continue
            for img_path in sorted(em_dir.glob("*.jpg")):
                import PIL.Image as PILImage
                pil_img = PILImage.open(img_path).convert("L").resize((48, 48))
                my_images.append(np.array(pil_img))
                my_labels_list.append(label_idx)

        if my_images:
            # Оверсемплинг: повторяем, пока пользовательских данных не станет ~15% от общего
            target_ratio = 0.15
            total_fer = len(images)
            target_my = int(total_fer * target_ratio)
            repeats = max(1, target_my // len(my_images))
            my_images = my_images * repeats
            my_labels_list = my_labels_list * repeats
            my_count = len(my_images)

            for img, lbl in zip(my_images, my_labels_list):
                images = np.append(images, [img], axis=0)
                labels = np.append(labels, lbl)

        print(f"  Мои эмоции: {len(set(my_labels_list))} классов, {len(my_images)} кадров "
              f"(×{repeats} оверсемплинг, ~{target_ratio:.0%} датасета)")

    # Пересчитываем class weights с учётом новых данных
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"  Всего: {len(images)} изображений")
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights]}")
    return images, labels, torch.tensor(class_weights, dtype=torch.float32)


# ── Датасет ───────────────────────────────────────────
class FER2013Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        import PIL.Image as PILImage
        pil_img = PILImage.fromarray(img, mode="L").convert("RGB")
        img_tensor = T.functional.to_tensor(pil_img)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, self.labels[idx]


# ── Аугментации ───────────────────────────────────────
train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    T.RandomPerspective(distortion_scale=0.15, p=0.3),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Обучение ──────────────────────────────────────────
def train():
    print("\n" + "=" * 50)
    print("   Обучение EfficientNet-B0 на FER2013 (v2)")
    print(f"   Label Smoothing: {LABEL_SMOOTHING}")
    print(f"   LR: {LR}, Weight Decay: {WEIGHT_DECAY}")
    print("=" * 50)

    images, labels, class_weights = download_fer2013()

    np.random.seed(42)
    idx = np.random.permutation(len(images))
    split = int(len(images) * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = FER2013Dataset(images[train_idx], labels[train_idx], train_transform)
    val_ds = FER2013Dataset(images[val_idx], labels[val_idx], val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    class_weights = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing с линейным warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% на warmup
    )

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*correct/total:.1f}%"})

        model.eval()
        val_correct, val_total = 0, 0
        all_probs = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, pred = out.max(1)
                val_correct += pred.eq(y).sum().item()
                val_total += y.size(0)
                all_probs.append(torch.softmax(out, dim=1).cpu())

        val_acc = 100 * val_correct / val_total

        # Средняя максимальная уверенность (должна быть < 0.9)
        all_probs = torch.cat(all_probs)
        mean_conf = all_probs.max(dim=1).values.mean().item()

        print(f"  Epoch {epoch+1}: train_acc={100*correct/total:.1f}%, val_acc={val_acc:.1f}%, mean_conf={mean_conf:.2f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "model_name": "efficientnet_b0",
                    "num_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "emotions_ru": EMOTIONS_RU,
                    "emotions_en": EMOTIONS_EN,
                    "val_accuracy": val_acc,
                    "label_smoothing": LABEL_SMOOTHING,
                }
            }, MODEL_PATH)
            print(f"  ✅ Сохранена: {MODEL_PATH} ({val_acc:.1f}%)")

    print(f"\n🏆 Лучшая точность: {best_acc:.1f}%")
    print(f"📦 Модель: {MODEL_PATH}")
    return best_acc


if __name__ == "__main__":
    train()
