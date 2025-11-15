# scripts/train_unet.py

import argparse
import csv
import os
import sys
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

# 保证能 import 到 src
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.unet_model import UNet
from src.utils_unet import load_pair_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-train", type=str, required=True)
    ap.add_argument("--manifest-val",   type=str, required=True)
    ap.add_argument("--save-dir",       type=str, required=True)
    ap.add_argument("--epochs",         type=int, default=50)
    ap.add_argument("--batch",          type=int, default=8)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--device",         type=str, default="cuda")
    ap.add_argument("--workers",        type=int, default=0)
    ap.add_argument("--patience",       type=int, default=5,
                    help="若 val_loss 连续多少 epoch 未提升，则提前停止")
    return ap.parse_args()


def save_checkpoint(state, save_dir: Path, name: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / name
    torch.save(state, path)
    print(f"[Info] Saved checkpoint to {path}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batch = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        enhanced = model(noisy)  # [B, F, T']

        # 裁剪时间维到一致
        T_min = min(enhanced.size(2), clean.size(2))
        enhanced = enhanced[:, :, :T_min]
        clean = clean[:, :, :T_min]

        loss = criterion(enhanced, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batch += 1

    return total_loss / max(n_batch, 1)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batch = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            enhanced = model(noisy)

            T_min = min(enhanced.size(2), clean.size(2))
            enhanced = enhanced[:, :, :T_min]
            clean = clean[:, :, :T_min]

            loss = criterion(enhanced, clean)

            total_loss += loss.item()
            n_batch += 1

    return total_loss / max(n_batch, 1)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    train_loader, val_loader = load_pair_dataset(
        manifest_train=args.manifest_train,
        manifest_val=args.manifest_val,
        batch=args.batch,
        workers=args.workers,
    )

    noisy_example, clean_example = next(iter(train_loader))
    freq_bins = noisy_example.shape[1]
    print(f"[Info] freq_bins (channels) = {freq_bins}")

    model = UNet(n_channels=freq_bins, n_classes=freq_bins).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 日志写入
    log_path = save_dir / "train_log_unet.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    best_epoch = -1
    patience_cnt = 0  # 早停计数器

    print("[Info] Start training UNet ...")

    for epoch in range(1, args.epochs + 1):
        t0 = time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        dt = time() - t0

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"time={dt:.1f}s"
        )

        # 写日志
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, val_loss])

        # 保存 last
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "freq_bins": freq_bins,
            },
            save_dir,
            "unet_last.pt",
        )

        # 判断是否更新 best
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_cnt = 0  # 重置早停计数
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "freq_bins": freq_bins,
                },
                save_dir,
                "unet_best.pt",
            )
            print(f"[Info] best model updated: epoch={epoch}, val_loss={val_loss:.4f}")
        else:
            patience_cnt += 1
            print(f"[Info] no improvement, patience = {patience_cnt}/{args.patience}")

            if patience_cnt >= args.patience:
                print(f"[Early Stop] val_loss 连续 {args.patience} 次未提升，提前停止训练。")
                break

    print(f"[Done] best epoch={best_epoch}, 最小 val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
