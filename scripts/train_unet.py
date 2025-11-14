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
    ap.add_argument("--manifest-train", type=str, required=True,
                    help=" manifest csv")
    ap.add_argument("--manifest-val", type=str, required=True,
                    help=" manifest csv")
    ap.add_argument("--save-dir", type=str, required=True,
                    help="save log")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda or cpu")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader num_workers")
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
        enhanced = model(noisy)
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
            loss = criterion(enhanced, clean)

            total_loss += loss.item()
            n_batch += 1

    return total_loss / max(n_batch, 1)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[Info] : {device}")

    manifest_train = Path(args.manifest_train)
    manifest_val = Path(args.manifest_val)
    save_dir = Path(args.save_dir)

   
    train_loader, val_loader = load_pair_dataset(
        manifest_train=str(manifest_train),
        manifest_val=str(manifest_val),
        batch=args.batch,
        workers=args.workers,
    )

   
    model = UNet(n_channels=1, n_classes=1)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

  
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log_unet.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    best_epoch = -1

    print("[Info]  UNet ")

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

        
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            save_dir,
            "unet_last.pt",
        )

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
                },
                save_dir,
                "unet_best.pt",
            )
            print(f"[Info] best model: epoch={epoch}, val_loss={val_loss:.4f}")

    print(f"[Done] best epoch={best_epoch}, 最小 val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
