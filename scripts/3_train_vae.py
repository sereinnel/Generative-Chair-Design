# 3_train_vae.py
import os
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from model_improved import PointNetVAE

# -----------------------
# Конфигурация по умолчанию
# -----------------------
class Cfg:
    DATA_DIR = "../data/normalized_npy"   # относительный к scripts/
    TRAIN_SUBDIR = "train"
    TEST_SUBDIR = "test"

    # Точки: данные храним 16384, но для обучения достаточно N_TRAIN (подвыборка)
    RAW_POINTS = 16384
    TRAIN_POINTS = 4096     # количество точек, подаваемых в сеть при обучении (сэмплируется)
    LOSS_POINTS = 4096      # количество точек для расчета Chamfer (может равняться TRAIN_POINTS)

    BATCH_SIZE = 8
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    LATENT_DIM = 256
    EPOCHS = 150
    MODEL_SAVE_DIR = "../models"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    BETA = 1e-4      # коэффициент для KLD
    GRAD_CLIP = 1.0
    PRINT_EVERY = 20

# -----------------------
# Dataset
# -----------------------
class PointCloudDataset(Dataset):
    def __init__(self, folder, num_points=Cfg.TRAIN_POINTS):
        self.folder = folder
        self.files = [p for p in os.listdir(folder) if p.endswith('.npy')]
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        pts = np.load(path).astype(np.float32)  # (RAW_POINTS, 3)
        # Если RAW_POINTS > num_points — случайная подвыборка без замены
        if pts.shape[0] >= self.num_points:
            idxs = np.random.choice(pts.shape[0], self.num_points, replace=False)
        else:
            idxs = np.random.choice(pts.shape[0], self.num_points, replace=True)
        pts = pts[idxs, :]
        # Normalize to zero mean on XY (optional): keep Z floor at 0 (already normalized)
        return torch.from_numpy(pts)  # (num_points, 3)

# -----------------------
# Chamfer loss (GPU)
# -----------------------
def chamfer_loss_gpu(p1, p2):
    # p1, p2: (B, N, 3)
    # using torch.cdist
    dists = torch.cdist(p1, p2)  # (B, N, N)
    # for each point in p1 find nearest in p2
    loss1 = torch.min(dists, dim=2)[0].mean()
    # for each point in p2 find nearest in p1
    loss2 = torch.min(dists, dim=1)[0].mean()
    return loss1 + loss2

# -----------------------
# Train / Eval
# -----------------------
def train_epoch(model, dataloader, optimizer, scaler, cfg, device):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
        # batch: (B, N, 3) -> transpose for encoder: (B, 3, N)
        batch = batch.to(device)  # (B, N, 3)
        batch_t = batch.permute(0, 2, 1).contiguous()

        optimizer.zero_grad()
        with autocast():
            recon, mu, logvar = model(batch_t)
            # recon: (B, RAW_POINTS, 3) — возможно генерируем больше точек than input
            # Для loss: подвыборка LOSS_POINTS from both recon and batch
            B = batch.shape[0]
            N_recon = recon.shape[1]
            N_gt = batch.shape[1]
            # Подвыборка индексов
            idx_recon = torch.randperm(N_recon, device=device)[:cfg.LOSS_POINTS]
            idx_gt = torch.randperm(N_gt, device=device)[:cfg.LOSS_POINTS]
            recon_sub = recon[:, idx_recon, :]
            gt_sub = batch[:, idx_gt, :]

            cd = chamfer_loss_gpu(recon_sub, gt_sub)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            loss = cd + cfg.BETA * kld

        scaler.scale(loss).backward()
        # gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * B

        if (i + 1) % cfg.PRINT_EVERY == 0:
            avg_loss = total_loss / ( (i+1) * dataloader.batch_size )
            print(f"[train] iter {i+1}/{len(dataloader)} avg_loss: {avg_loss:.6f}")

    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, cfg, device, max_batches=50):
    model.eval()
    cds = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            batch_t = batch.permute(0, 2, 1).contiguous()
            recon, mu, logvar = model(batch_t)
            # Compute chamfer on LOSS_POINTS (subsample)
            N_recon = recon.shape[1]
            N_gt = batch.shape[1]
            idx_recon = torch.randperm(N_recon, device=device)[:cfg.LOSS_POINTS]
            idx_gt = torch.randperm(N_gt, device=device)[:cfg.LOSS_POINTS]
            recon_sub = recon[:, idx_recon, :].cpu().numpy()
            gt_sub = batch[:, idx_gt, :].cpu().numpy()
            # Convert to numpy and compute per-sample cd with KDTree
            from scipy.spatial import cKDTree
            for b in range(recon_sub.shape[0]):
                tree1 = cKDTree(recon_sub[b])
                tree2 = cKDTree(gt_sub[b])
                d1, _ = tree1.query(gt_sub[b])
                d2, _ = tree2.query(recon_sub[b])
                cds.append(d1.mean() + d2.mean())
    import numpy as np
    return float(np.mean(cds)), float(np.std(cds))

# -----------------------
# Main
# -----------------------
def main(args):
    cfg = Cfg
    # apply CLI overrides
    if args.epochs: cfg.EPOCHS = args.epochs
    if args.batch_size: cfg.BATCH_SIZE = args.batch_size
    if args.latent_dim: cfg.LATENT_DIM = args.latent_dim
    if args.train_points: cfg.TRAIN_POINTS = args.train_points
    if args.loss_points: cfg.LOSS_POINTS = args.loss_points

    device = torch.device(cfg.DEVICE)
    print("Device:", device)

    train_folder = os.path.join(os.path.dirname(__file__), cfg.DATA_DIR, cfg.TRAIN_SUBDIR)
    test_folder = os.path.join(os.path.dirname(__file__), cfg.DATA_DIR, cfg.TEST_SUBDIR)
    os.makedirs(os.path.join(os.path.dirname(__file__), cfg.MODEL_SAVE_DIR), exist_ok=True)

    train_ds = PointCloudDataset(train_folder, num_points=cfg.TRAIN_POINTS)
    test_ds = PointCloudDataset(test_folder, num_points=cfg.TRAIN_POINTS)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = PointNetVAE(num_points=cfg.RAW_POINTS, latent_dim=cfg.LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = GradScaler()

    best_cd = float('inf')
    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, cfg, device)
        cd_mean, cd_std = evaluate(model, test_loader, cfg, device, max_batches=50)

        t1 = time.time()
        print(f"Epoch {epoch}/{cfg.EPOCHS} | train_loss: {train_loss:.6f} | cd_mean: {cd_mean:.6f} ± {cd_std:.6f} | time: {(t1-t0):.1f}s")

        # checkpoint
        save_path = os.path.join(os.path.dirname(__file__), cfg.MODEL_SAVE_DIR, f"vae_epoch_{epoch:03d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, save_path)

        # keep best
        if cd_mean < best_cd:
            best_cd = cd_mean
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), cfg.MODEL_SAVE_DIR, "model_improved_best.pth"))
            print("  ✅ New best model saved (best cd = {:.6f})".format(best_cd))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument('--train_points', type=int, default=None)
    parser.add_argument('--loss_points', type=int, default=None)
    args = parser.parse_args()
    main(args)
