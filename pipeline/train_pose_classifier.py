#!/usr/bin/env python3
"""
train_pose_classifier.py — обучение классификатора позы на latent-эмбеддингах VAE.

Вход:  pipeline/poses_dataset.csv + models/emotion/vposer_coco33.pth
Выход: models/emotion/pose_classifier.pth

Запуск: venv/bin/python3 pipeline/train_pose_classifier.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'core'))
from pose_encoder import normalize_landmarks
from pose_vae import PoseVAE
from pose_heads import PoseClassifier

# ── Конфигурация ──────────────────────────────────────────
CSV_PATH = os.path.join(ROOT, 'pipeline', 'poses_dataset.csv')
VAE_PATH = os.path.join(ROOT, 'models', 'emotion', 'vposer_coco33.pth')
CLF_PATH = os.path.join(ROOT, 'models', 'emotion', 'pose_classifier.pth')
CLF_CONFIG = os.path.join(ROOT, 'models', 'emotion', 'pose_classifier_config.json')

BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
DROPOUT = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['standing', 'sitting', 'lying_back', 'lying_side', 'arms_crossed', 'sitting_at_desk']
CLASS_NAMES_RU = ['стоит', 'сидит', 'лежит_на_спине', 'лежит_на_боку', 'руки_скрещены', 'сидит_за_ПК']

# ── Датасет (эмбеддинги VAE) ─────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, z_data, labels):
        self.z = torch.from_numpy(z_data.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]

# ── Обучение ──────────────────────────────────────────────
def train():
    print('=' * 60)
    print('  Обучение PoseClassifier на VAE эмбеддингах')
    print(f'  Устройство: {DEVICE}')
    print('=' * 60)

    # Загрузка VAE
    checkpoint = torch.load(VAE_PATH, map_location=DEVICE, weights_only=True)
    vae_config = checkpoint['config']
    vae = PoseVAE(
        input_dim=vae_config['input_dim'],
        latent_dim=vae_config['latent_dim'],
        hidden_dims=tuple(vae_config['hidden_dims']),
        dropout=vae_config['dropout'],
    ).to(DEVICE)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f'  VAE загружен: latent_dim={vae_config["latent_dim"]}')

    # Загрузка датасета → эмбеддинги
    X_raw, y_raw = [], []
    with open(CSV_PATH, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            lbl = int(row[0])
            pts = np.array([
                [float(row[1 + i*4 + j]) for j in range(4)]
                for i in range(33)
            ], dtype=np.float32)
            norm, _ = normalize_landmarks(pts)
            X_raw.append(norm.reshape(-1).astype(np.float32))
            y_raw.append(lbl)

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw, dtype=np.int32)

    # Эмбеддинги через VAE encoder
    print(f'  Вычисление эмбеддингов для {len(X_raw)} примеров...')
    z_all = []
    with torch.no_grad():
        for i in range(0, len(X_raw), BATCH_SIZE * 4):
            batch = torch.from_numpy(X_raw[i:i + BATCH_SIZE * 4]).to(DEVICE)
            z_batch = vae.encode_to_z(batch).cpu().numpy()
            z_all.append(z_batch)
    z_all = np.concatenate(z_all, axis=0)
    print(f'  Эмбеддинги: {z_all.shape}')

    # Train/val/test split: 70/15/15 stratified
    z_train, z_temp, y_train, y_temp = train_test_split(
        z_all, y_raw, test_size=0.3, random_state=42, stratify=y_raw
    )
    z_val, z_test, y_val, y_test = train_test_split(
        z_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f'  Train: {len(z_train)}, Val: {len(z_val)}, Test: {len(z_test)}')

    train_ds = EmbeddingDataset(z_train, y_train)
    val_ds = EmbeddingDataset(z_val, y_val)
    test_ds = EmbeddingDataset(z_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Модель
    num_classes = len(set(y_raw))
    model = PoseClassifier(
        latent_dim=vae_config['latent_dim'],
        num_classes=num_classes,
        hidden_dim=64,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f'  Классов: {num_classes}')
    print(f'  Параметров классификатора: {sum(p.numel() for p in model.parameters()):,}')
    print()

    # Class weights для балансировки
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        for z, y in train_loader:
            z, y = z.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(z)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for z, y in val_loader:
                z, y = z.to(DEVICE), y.to(DEVICE)
                logits = model(z)
                loss = criterion(logits, y)
                val_loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_true.extend(y.cpu().tolist())

        val_acc = accuracy_score(val_true, val_preds)
        train_loss = train_loss_sum / len(train_loader)
        val_loss = val_loss_sum / len(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'latent_dim': vae_config['latent_dim'],
                    'num_classes': num_classes,
                    'hidden_dim': 64,
                    'dropout': DROPOUT,
                    'class_names': CLASS_NAMES[:num_classes],
                    'class_names_ru': CLASS_NAMES_RU[:num_classes],
                    'val_accuracy': val_acc,
                },
            }, CLF_PATH)

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f'  Epoch {epoch:3d}/{EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.2%} | {elapsed:.0f}s')

    elapsed = time.time() - t_start

    # ── Тест ──────────────────────────────────────────────
    print(f'\n  Обучение завершено за {elapsed:.0f}s')
    print(f'  Лучшая val accuracy: {best_val_acc:.2%}')

    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for z, y in test_loader:
            z, y = z.to(DEVICE), y.to(DEVICE)
            logits = model(z)
            preds = logits.argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_true.extend(y.cpu().tolist())

    test_acc = accuracy_score(test_true, test_preds)
    print(f'  Test accuracy: {test_acc:.2%}')

    print(f'\n  Classification Report (Test):')
    print(classification_report(
        test_true, test_preds,
        target_names=CLASS_NAMES[:num_classes],
        digits=4,
    ))

    cm = confusion_matrix(test_true, test_preds)
    print(f'  Confusion Matrix:')
    print(f'  {cm.tolist()}')

    # ── Конфиг ───────────────────────────────────────────
    config = {
        'model': 'PoseClassifier',
        'latent_dim': vae_config['latent_dim'],
        'num_classes': num_classes,
        'hidden_dim': 64,
        'dropout': DROPOUT,
        'epochs_trained': EPOCHS,
        'val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'class_names': CLASS_NAMES[:num_classes],
        'class_names_ru': CLASS_NAMES_RU[:num_classes],
        'parameters': sum(p.numel() for p in model.parameters()),
    }

    os.makedirs(os.path.dirname(CLF_CONFIG), exist_ok=True)
    with open(CLF_CONFIG, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f'\n  Классификатор: {CLF_PATH}')
    print(f'  Конфиг: {CLF_CONFIG}')
    print(f'\n  ГОТОВО. Accuracy: {test_acc:.2%}')

if __name__ == '__main__':
    train()
