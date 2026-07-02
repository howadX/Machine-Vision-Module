#!/usr/bin/env python3
"""
self_learn.py — Self-Learning цикл для поз.

Режимы:
  1. init      — инициализировать прототипы из обученного классификатора
  2. classify  — классифицировать CSV-файл (из capture_poses.py)
  3. add       — добавить новую позу: 5 примеров → прототип + replay buffer
  4. retrain   — переобучить классификатор с replay buffer
  5. status    — показать состояние прототипов и буфера

Запуск:
  venv/bin/python3 pipeline/self_learn.py init
  venv/bin/python3 pipeline/self_learn.py classify pipeline/poses_dataset.csv
  venv/bin/python3 pipeline/self_learn.py add <class_name> <csv_path>
  venv/bin/python3 pipeline/self_learn.py status
"""

import os, sys, csv, json, pickle
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'core'))

from pose_encoder import normalize_landmarks
from pose_vae import PoseVAE
from pose_heads import PrototypicalClassifier, ExperienceReplayBuffer

VAE_PATH = os.path.join(ROOT, 'models', 'emotion', 'vposer_coco33.pth')
CLF_CONFIG = os.path.join(ROOT, 'models', 'emotion', 'pose_classifier_config.json')
PROTO_PATH = os.path.join(ROOT, 'models', 'emotion', 'pose_prototypes.pkl')
REPLAY_PATH = os.path.join(ROOT, 'models', 'emotion', 'pose_replay_buffer.pkl')

ANOMALY_THRESHOLD = 14.3  # -log p(z) порог
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_vae():
    ckpt = torch.load(VAE_PATH, map_location=DEVICE, weights_only=True)
    cfg = ckpt['config']
    vae = PoseVAE(
        input_dim=cfg['input_dim'], latent_dim=cfg['latent_dim'],
        hidden_dims=tuple(cfg['hidden_dims']), dropout=cfg['dropout'],
    ).to(DEVICE)
    vae.load_state_dict(ckpt['model_state_dict'])
    vae.eval()
    return vae


def csv_to_z(vae, csv_path):
    """Конвертирует CSV с landmarks в z-векторы + logpz."""
    z_list, logpz_list, labels_list = [], [], []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            lbl = int(row[0]) if row[0].strip() else -1
            pts = np.array([[float(row[1 + i*4 + j]) for j in range(4)] for i in range(33)], dtype=np.float32)
            norm, _ = normalize_landmarks(pts)
            flat = norm.reshape(1, -1).astype(np.float32)
            x = torch.from_numpy(flat).to(DEVICE)
            with torch.no_grad():
                z = vae.encode_to_z(x).cpu().numpy().flatten()
                logpz = vae.log_prob_z(x).item()
            z_list.append(z)
            logpz_list.append(logpz)
            labels_list.append(lbl)
    return np.array(z_list), np.array(logpz_list), np.array(labels_list)


def cmd_init():
    """Инициализирует прототипы из обученного классификатора."""
    print("Инициализация прототипов из классификатора...")

    vae = load_vae()

    with open(CLF_CONFIG) as f:
        cfg = json.load(f)
    class_names = cfg['class_names_ru']
    num_classes = cfg['num_classes']

    # Загружаем наши данные и вычисляем центроиды
    csv_path = os.path.join(ROOT, 'pipeline', 'poses_dataset.csv')
    z_all, _, y_all = csv_to_z(vae, csv_path)

    proto = PrototypicalClassifier(latent_dim=32)
    replay = ExperienceReplayBuffer(max_per_class=50)

    for label in range(num_classes):
        mask = y_all == label
        if mask.sum() == 0:
            print(f"  Пропущен класс {label}: нет примеров")
            continue
        z_class = z_all[mask]
        name = class_names[label] if label < len(class_names) else f'class_{label}'
        proto.add_class(name, z_class)
        replay.add_batch(name, z_class)
        print(f"  {name}: {len(z_class)} примеров → центроид")

    proto.save(PROTO_PATH)
    replay.save(REPLAY_PATH)
    print(f"\nПрототипы: {proto.num_classes} классов → {PROTO_PATH}")
    print(f"Replay buffer: {replay.stats()}")
    print("ГОТОВО.")


def cmd_classify(csv_path=None):
    """Классифицирует примеры через прототипы + VAE anomaly detection."""
    if csv_path is None:
        csv_path = os.path.join(ROOT, 'pipeline', 'poses_dataset.csv')

    vae = load_vae()
    proto = PrototypicalClassifier(latent_dim=32)

    if os.path.exists(PROTO_PATH):
        proto.load(PROTO_PATH)
        print(f"Прототипы загружены: {proto.num_classes} классов")
    else:
        print("Прототипы не найдены. Запусти 'init' сначала.")
        return

    z_all, logpz_all, y_all = csv_to_z(vae, csv_path)

    print(f"\nКлассификация {len(z_all)} примеров:\n")
    print(f"{'Истинный':<20} {'Прототип':<20} {'Dist':>8} {'Conf':>8} {'-log p(z)':>10} {'Статус'}")
    print("-" * 80)

    correct = 0
    total = 0
    anomaly_count = 0

    for i, (z, logpz, true_label) in enumerate(zip(z_all, logpz_all, y_all)):
        name, conf = proto.classify(z)
        status = 'unusual' if logpz > ANOMALY_THRESHOLD else 'normal'
        if status == 'unusual':
            anomaly_count += 1

        true_name = proto.class_names[true_label] if true_label < len(proto.class_names) else f'cls_{true_label}'
        if name == true_name:
            correct += 1
        total += 1

        marker = '✓' if name == true_name else '✗'
        if i < 20 or status == 'unusual':
            print(f"{marker} {true_name:<18} {name:<20} {np.linalg.norm(z - proto.prototypes[name]):>8.3f} {conf:>8.3f} {logpz:>10.2f} {status}")

    if total > 0:
        print(f"\nТочность прототипов: {correct}/{total} = {correct/total:.1%}")
    print(f"Аномалий (-log p(z) > {ANOMALY_THRESHOLD}): {anomaly_count}/{total}")


def cmd_add(class_name, csv_path):
    """Добавляет новую позу: извлекает z-векторы, создаёт прототип, добавляет в replay."""
    print(f"Добавление класса '{class_name}' из {csv_path}...")

    vae = load_vae()

    # Загружаем существующие прототипы и буфер
    proto = PrototypicalClassifier(latent_dim=32)
    if os.path.exists(PROTO_PATH):
        proto.load(PROTO_PATH)
        print(f"  Существующие прототипы: {proto.num_classes} классов")
        if class_name in proto.prototypes:
            print(f"  ⚠ Класс '{class_name}' уже существует! Добавляем примеры к существующему.")

    replay = ExperienceReplayBuffer(max_per_class=50)
    if os.path.exists(REPLAY_PATH):
        replay.load(REPLAY_PATH)

    # Извлекаем z-векторы из CSV
    z_all, _, y_all = csv_to_z(vae, csv_path)
    print(f"  Извлечено {len(z_all)} z-векторов")

    # Добавляем в прототипы и буфер
    proto.add_class(class_name, z_all)
    replay.add_batch(class_name, z_all)

    # Сохраняем
    proto.save(PROTO_PATH)
    replay.save(REPLAY_PATH)

    print(f"\n  Класс '{class_name}': {len(z_all)} примеров")
    print(f"  Прототипы: {proto.num_classes} классов → {PROTO_PATH}")
    print(f"  Replay buffer: {replay}")
    print("ГОТОВО. Новая поза доступна для классификации немедленно.")


def cmd_status():
    """Показывает состояние прототипов и буфера."""
    print("=" * 60)
    print("  Self-Learning Status")
    print("=" * 60)

    if os.path.exists(PROTO_PATH):
        proto = PrototypicalClassifier(latent_dim=32)
        proto.load(PROTO_PATH)
        print(f"\nПрототипы ({proto.num_classes} классов):")
        for name in proto.class_names:
            centroid_norm = np.linalg.norm(proto.prototypes[name])
            print(f"  {name:<20} |centroid|={centroid_norm:.3f}")
    else:
        print("\nПрототипы: не инициализированы")

    if os.path.exists(REPLAY_PATH):
        replay = ExperienceReplayBuffer(max_per_class=50)
        replay.load(REPLAY_PATH)
        print(f"\n{replay.stats()}")
    else:
        print("\nReplay buffer: пуст")

    # Пороги anomaly detection
    print(f"\nAnomaly threshold: -log p(z) > {ANOMALY_THRESHOLD}")


# ── Главное ───────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'init':
        cmd_init()
    elif cmd == 'classify':
        csv_path = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_classify(csv_path)
    elif cmd == 'add':
        if len(sys.argv) < 4:
            print("Использование: self_learn.py add <class_name> <csv_path>")
            sys.exit(1)
        cmd_add(sys.argv[2], sys.argv[3])
    elif cmd == 'status':
        cmd_status()
    else:
        print(f"Неизвестная команда: {cmd}")
        print(__doc__)
