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
    ap.add_argument("--lr",             type=float, default=2e-3, help="Learning rate (default: 2e-3 for faster convergence)")
    ap.add_argument("--warmup-epochs",  type=int, default=3, help="Warmup epochs for learning rate (default: 3)")
    ap.add_argument("--base-channels",  type=int, default=64, help="Base channels for UNet (default: 64, larger = more capacity)")
    ap.add_argument("--device",         type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--workers",        type=int, default=0)
    ap.add_argument("--patience",       type=int, default=10,
                    help="Early stopping patience (number of epochs without improvement)")
    ap.add_argument("--min-delta",      type=float, default=0.001,
                    help="Minimum improvement threshold for early stopping (default: 0.001 for L1 loss)")
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

        T_min = min(enhanced.size(2), clean.size(2))
        enhanced = enhanced[:, :, :T_min]
        clean = clean[:, :, :T_min]

        loss = criterion(enhanced, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    args = parse_args()

    device = resolve_device(args.device)
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

    try:
        model = UNet(n_channels=freq_bins, n_classes=freq_bins, base_channels=args.base_channels).to(device)
        print(f"[Info] Model created with base_channels={args.base_channels}")
    except TypeError:
        print(f"[Warning] UNet does not support base_channels parameter, using default value")
        model = UNet(n_channels=freq_bins, n_classes=freq_bins).to(device)
        print(f"[Info] Model created with default base_channels")
    
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    def combined_loss(pred, target):
        return 0.7 * l1_loss(pred, target) + 0.3 * mse_loss(pred, target)
    criterion = combined_loss
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / args.warmup_epochs)
        )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = save_dir / "train_log_unet.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val = float("inf")
    best_epoch = -1
    patience_cnt = 0
    min_delta = args.min_delta

    print("[Info] Start training UNet ...")
    print(f"[Info] Early stopping: patience={args.patience}, min_delta={min_delta}")

    for epoch in range(1, args.epochs + 1):
        t0 = time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        dt = time() - t0

        if warmup_scheduler is not None and epoch <= args.warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if epoch == args.warmup_epochs:
                print(f"[Info] Warmup finished, switching to ReduceLROnPlateau scheduler")
        else:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"lr={current_lr:.2e}  "
            f"time={dt:.1f}s"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, val_loss, current_lr])

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

        if best_val - val_loss > min_delta:
            best_val = val_loss
            best_epoch = epoch
            patience_cnt = 0
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
            print(f"[Best] Saved best model with val_loss={val_loss:.4f}")
        else:
            patience_cnt += 1
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
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
            print(f"[Info] No significant improvement, patience = {patience_cnt}/{args.patience}")

            if patience_cnt >= args.patience:
                print(f"[EarlyStop] Patience reached. Best val_loss={best_val:.4f} at epoch {best_epoch}.")
                print("[Done] Training finished (early stopped).")
                break

    print(f"[Done] best epoch={best_epoch}, min val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
