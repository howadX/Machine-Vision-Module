#!/usr/bin/env python3
"""
train_pose_vae.py — обучение PoseVAE на датасете поз.

Вход:  pipeline/poses_dataset.csv
Выход: models/emotion/vposer_coco33.pth + vposer_coco33_config.json

Запуск: venv/bin/python3 pipeline/train_pose_vae.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'core'))
from pose_encoder import normalize_landmarks
from pose_vae import PoseVAE, vae_loss

# ── Конфигурация ──────────────────────────────────────────
CSV_PATH = os.path.join(ROOT, 'pipeline', 'poses_dataset.csv')
MODEL_DIR = os.path.join(ROOT, 'models', 'emotion')
MODEL_PATH = os.path.join(MODEL_DIR, 'vposer_coco33.pth')
CONFIG_PATH = os.path.join(MODEL_DIR, 'vposer_coco33_config.json')

INPUT_DIM = 99        # 33 landmarks × 3 (x,y,z)
LATENT_DIM = 32
HIDDEN_DIMS = (128, 64)
BATCH_SIZE = 64
EPOCHS = 1000
LR = 1e-3
BETA = 0.001         # KL weight
DROPOUT = 0.2
VAL_SPLIT = 0.2
PATIENCE = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Датасет ───────────────────────────────────────────────
class PoseDataset(Dataset):
    def __init__(self, csv_path: str):
        self.X = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                pts = np.array([
                    [float(row[1 + i*4 + j]) for j in range(4)]
                    for i in range(33)
                ], dtype=np.float32)
                norm, _meta = normalize_landmarks(pts)
                self.X.append(norm.reshape(-1).astype(np.float32))

        self.X = np.array(self.X)
        print(f'  Загружено: {len(self.X)} примеров, dim={self.X.shape[1]}')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])

# ── Обучение ──────────────────────────────────────────────
def train():
    print('=' * 60)
    print('  Обучение PoseVAE на датасете поз')
    print(f'  Устройство: {DEVICE}')
    print(f'  Архитектура: {INPUT_DIM}→{HIDDEN_DIMS}→μ/σ:{LATENT_DIM}→{tuple(reversed(HIDDEN_DIMS))}→{INPUT_DIM}')
    print(f'  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}, β: {BETA}')
    print('=' * 60)

    # Данные
    dataset = PoseDataset(CSV_PATH)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Модель
    model = PoseVAE(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f'  Параметров: {sum(p.numel() for p in model.parameters()):,}')
    print()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []}

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        recon_sum, kl_sum = 0.0, 0.0

        for x in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=BETA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            recon_sum += recon_l.item()
            kl_sum += kl_l.item()

        scheduler.step()

        n_batches = len(train_loader)
        train_loss = train_loss_sum / n_batches
        recon_avg = recon_sum / n_batches
        kl_avg = kl_sum / n_batches

        # ── Validation ───────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                recon, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar, beta=BETA)
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)

        # ── Логи ─────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['recon_loss'].append(recon_avg)
        history['kl_loss'].append(kl_avg)

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(
                f'  Epoch {epoch:4d}/{EPOCHS} | '
                f'train: {train_loss:.4f} | val: {val_loss:.4f} | '
                f'recon: {recon_avg:.4f} | KL: {kl_avg:.4f} | '
                f'{elapsed:.0f}s'
            )

        # ── Early stopping ───────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Сохраняем лучшую модель
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_dim': INPUT_DIM,
                    'latent_dim': LATENT_DIM,
                    'hidden_dims': list(HIDDEN_DIMS),
                    'dropout': DROPOUT,
                    'beta': BETA,
                    'best_val_loss': best_val_loss,
                },
                'history': history,
            }, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'\n  Early stopping на эпохе {epoch} (patience={PATIENCE})')
                break

    elapsed = time.time() - t_start
    print(f'\n  Обучение завершено за {elapsed:.0f}с')
    print(f'  Лучший val loss: {best_val_loss:.4f}')

    # ── Конфиг ───────────────────────────────────────────
    config = {
        'model': 'PoseVAE',
        'input_dim': INPUT_DIM,
        'latent_dim': LATENT_DIM,
        'hidden_dims': list(HIDDEN_DIMS),
        'dropout': DROPOUT,
        'beta': BETA,
        'epochs_trained': epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'final_recon_loss': recon_avg,
        'final_kl_loss': kl_avg,
        'parameters': sum(p.numel() for p in model.parameters()),
        'device': str(DEVICE),
    }

    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f'  Модель: {MODEL_PATH}')
    print(f'  Конфиг: {CONFIG_PATH}')

    # ── Быстрая проверка на нескольких примерах ──────────
    print('\n  Проверка реконструкции:')
    model.eval()
    with torch.no_grad():
        for i, (name, loader) in enumerate([('train', train_loader), ('val', val_loader)]):
            x_sample = next(iter(loader))[:5].to(DEVICE)
            recon, _, _ = model(x_sample)
            mse = nn.functional.mse_loss(recon, x_sample).item()
            print(f'    {name}: MSE={mse:.6f} (первые 5 примеров)')

    print('\n  ГОТОВО.')

if __name__ == '__main__':
    train()
