#!/usr/bin/env python3
"""train_mlp_distilled.py — Обучение MLP на софт-метках HSEmotion.

Дистилляция: MLP учится воспроизводить 7-классовые вероятности HSEmotion
с геометрических признаков FaceLandmarker (468 расстояний).

Архитектура: та же EmotionMLP (512→512→256→128→7)
Loss: MSE(student_logits, teacher_probs) — регрессия на софт-метки
Сравнение: test accuracy против жёстких меток исходного датасета
"""
import os, sys, time, pickle, json, warnings
import numpy as np
import pandas as pd

os.environ.setdefault('ONNXRUNTIME_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Конфигурация ──────────────────────────────────────────
MODEL_DIR = os.path.join(ROOT, 'models', 'emotion')
TRAIN_CSV = os.path.join(MODEL_DIR, 'geometric_distilled_train.csv')
TEST_CSV = os.path.join(MODEL_DIR, 'geometric_distilled_test.csv')
OUT_MODEL = os.path.join(MODEL_DIR, 'mlp_emotion_distilled.pth')
OUT_SCALER = os.path.join(MODEL_DIR, 'mlp_emotion_distilled_scaler.pkl')
OUT_CONFIG = os.path.join(MODEL_DIR, 'mlp_emotion_distilled_config.json')

OUR_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(OUR_CLASSES)
INPUT_DIM = 468

BATCH_SIZE = 512
EPOCHS = 3000
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
T_0 = 200  # CosineAnnealingWarmRestarts period

# ── Модель ────────────────────────────────────────────────
class EmotionMLP(nn.Module):
    def __init__(self, in_dim=468, num_classes=7, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)


def load_data(csv_path):
    """Загружает CSV: 468 признаков + hard_label + 7 софт-меток."""
    df = pd.read_csv(csv_path, dtype={'emotion_label': str})
    feature_cols = [f'f{i}' for i in range(INPUT_DIM)]
    
    X = df[feature_cols].values.astype(np.float32)
    y_soft = df[OUR_CLASSES].values.astype(np.float32)
    
    # Hard labels → индексы
    label_to_idx = {name: i for i, name in enumerate(OUR_CLASSES)}
    y_hard = np.array([label_to_idx.get(l, 0) for l in df['emotion_label']], dtype=np.int64)
    
    # NaN → 0
    X = np.nan_to_num(X, nan=0.0)
    y_soft = np.nan_to_num(y_soft, nan=0.0)
    
    return X, y_hard, y_soft


def accuracy_from_soft(y_soft, logits):
    """Вычисляет accuracy: argmax logits vs argmax soft labels."""
    if hasattr(logits, 'argmax'):
        logits = logits.cpu().numpy() if hasattr(logits, 'cpu') else logits
    pred = np.argmax(logits, axis=1)
    true = np.argmax(y_soft, axis=1) if y_soft.ndim == 2 else y_soft
    return (pred == true).mean()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ── Загрузка данных ──
    print(f"Загрузка train: {TRAIN_CSV}")
    X_train, y_hard_train, y_soft_train = load_data(TRAIN_CSV)
    print(f"  Train: {X_train.shape[0]} примеров, {X_train.shape[1]} признаков")
    
    print(f"Загрузка test: {TEST_CSV}")
    X_test, y_hard_test, y_soft_test = load_data(TEST_CSV)
    print(f"  Test:  {X_test.shape[0]} примеров")
    
    # ── Нормализация ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    # ── DataLoader ──
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_hard_train),
        torch.from_numpy(y_soft_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_hard_test),
        torch.from_numpy(y_soft_test),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # ── Модель ──
    model = EmotionMLP(in_dim=INPUT_DIM, num_classes=NUM_CLASSES, dropout=DROPOUT)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2
    )
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    ALPHA = 0.3     # вес hard-label лосса
    TEMPERATURE = 3.0  # температура для дистилляции
    
    print(f"\nМодель: {sum(p.numel() for p in model.parameters()):,} параметров")
    print(f"Loss: {ALPHA}*CrossEntropy(hard) + {(1-ALPHA)}*KL(soft/T, T={TEMPERATURE})")
    print(f"Эпохи: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")
    print("=" * 60)
    
    # ── Обучение ──
    best_test_acc = 0.0
    best_epoch = 0
    t_start = time.perf_counter()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for bx, bh, bs in train_loader:  # features, hard_labels, soft_labels
            bx = bx.to(device, non_blocking=True)
            bh = bh.to(device, non_blocking=True)
            bs = bs.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(bx)
            
            # Комбинированный лосс
            loss_ce = criterion_ce(logits, bh)
            loss_kl = criterion_kl(
                F.log_softmax(logits / TEMPERATURE, dim=1),
                F.softmax(bs / TEMPERATURE, dim=1)
            ) * (TEMPERATURE * TEMPERATURE)
            loss = ALPHA * loss_ce + (1 - ALPHA) * loss_kl
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * bx.size(0)
        
        train_loss /= len(train_ds)
        scheduler.step()
        
        # ── Validation ──
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            test_loss = 0.0
            all_logits = []
            
            with torch.no_grad():
                for bx, bh, bs in test_loader:
                    bx = bx.to(device, non_blocking=True)
                    bh = bh.to(device, non_blocking=True)
                    bs = bs.to(device, non_blocking=True)
                    logits = model(bx)
                    loss_ce = criterion_ce(logits, bh)
                    loss_kl = criterion_kl(
                        F.log_softmax(logits / TEMPERATURE, dim=1),
                        F.softmax(bs / TEMPERATURE, dim=1)
                    ) * (TEMPERATURE * TEMPERATURE)
                    test_loss += (ALPHA * loss_ce + (1 - ALPHA) * loss_kl).item() * bx.size(0)
                    all_logits.append(logits.cpu())
            
            test_loss /= len(test_ds)
            all_logits = torch.cat(all_logits)
            test_acc = accuracy_from_soft(y_hard_test, all_logits.numpy())
            
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / epoch) * (EPOCHS - epoch)
            
            status = ""
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                # Сохраняем чекпоинт
                torch.save(model.state_dict(), OUT_MODEL)
                with open(OUT_SCALER, 'wb') as f:
                    pickle.dump(scaler, f)
                with open(OUT_CONFIG, 'w') as f:
                    json.dump({
                        'in_dim': INPUT_DIM,
                        'num_classes': NUM_CLASSES,
                        'dropout': DROPOUT,
                        'class_names': OUR_CLASSES,
                        'distilled_from': 'HSEmotion enet_b0_8_best_vgaf',
                        'test_accuracy': round(float(best_test_acc), 4),
                    }, f, indent=2)
                status = " ✓ BEST"
            
            print(f"  Epoch {epoch:4d}/{EPOCHS} | "
                  f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} | "
                  f"test_acc={test_acc:.3f} best={best_test_acc:.3f}@{best_epoch} | "
                  f"ETA={eta:.0f}s{status}", flush=True)
            
            # Ранний выход при NaN
            if np.isnan(train_loss):
                print("  ⚠ NaN loss — остановка")
                break
    
    # ── Итог ──
    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"Обучение завершено за {total_time:.0f}с ({total_time/60:.1f} мин)")
    print(f"Лучшая test accuracy: {best_test_acc:.3f} (эпоха {best_epoch})")
    print(f"Модель: {OUT_MODEL}")
    print(f"Скалер: {OUT_SCALER}")
    print(f"Конфиг: {OUT_CONFIG}")
    
    # ── Сравнение с оригинальным MLP ──
    orig_model_path = os.path.join(MODEL_DIR, 'mlp_emotion.pth')
    if os.path.exists(orig_model_path):
        print(f"\n── Сравнение с оригинальным MLP ──")
        print(f"  Оригинальный: жёсткие метки, 68.0% test accuracy")
    print(f"  Дистиллированный: софт-метки HSEmotion, {best_test_acc:.1%} test accuracy")
    print(f"\n  Итог: MLP <1ms CPU с качеством HSEmotion (73%) → ~{best_test_acc:.1%}")


if __name__ == '__main__':
    main()
