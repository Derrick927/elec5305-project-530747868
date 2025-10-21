#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mask.py
Train a simple mask-based enhancement model (IRM target) with explicit,
reproducible "challenge condition" controls (SNR, noise filtering, seed),
and save run metadata alongside checkpoints.

Usage examples:
1) PowerShell-friendly (space separated SNR list)  <-- RECOMMENDED
   python scripts/train_mask.py `
     --manifest-train manifests/train.csv `
     --manifest-val manifests/val.csv `
     --mode on_the_fly `
     --snr-list -5 0 5 10 `
     --noise-filter 'office|street|metro' `
     --batch 8 --epochs 2 --lr 1e-3 `
     --save-dir checkpoints/demo --seed 1337

2) Comma-separated (must be quoted in PowerShell)
   python scripts/train_mask.py --manifest-train manifests/train.csv \
     --mode on_the_fly --snr-list '-5,0,5,10'
"""

import os
import math
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Make src importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import PairDataset
from src.dnn_mask import MaskNet
from src.stft import N_FFT

# ----------------------------
# Utilities
# ----------------------------

def seed_all(seed: int = 1337):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dirs(path: str):
    """Create directory if not exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def collate_pad(batch):
    """
    Pad variable-length (T, F) to the longest T in the batch.
    Returns:
      feats:  (B, T_max, F)
      labels: (B, T_max, F)
      mask:   (B, T_max) -> 1.0 for valid frames, 0.0 for padding
    """
    Ts = [x[0].shape[0] for x in batch]
    Fdim = batch[0][0].shape[1]
    T_max = max(Ts)
    B = len(batch)

    feats = torch.zeros((B, T_max, Fdim), dtype=torch.float32)
    labels = torch.zeros((B, T_max, Fdim), dtype=torch.float32)
    mask = torch.zeros((B, T_max), dtype=torch.float32)

    for i, (noisy_T, irm_T) in enumerate(batch):
        t = noisy_T.shape[0]
        feats[i, :t, :] = torch.from_numpy(noisy_T)
        labels[i, :t, :] = torch.from_numpy(irm_T)
        mask[i, :t] = 1.0

    return feats, labels, mask

def masked_bce_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Compute BCE loss only on valid frames."""
    bce = nn.BCELoss(reduction="none")
    loss = bce(pred, target)  # (B, T, F)
    mask_exp = mask.unsqueeze(-1)  # (B, T, 1)
    loss = loss * mask_exp
    denom = mask_exp.sum() + 1e-8
    return loss.sum() / denom

@torch.no_grad()
def run_eval(model, loader, device):
    """Average masked BCE on validation set."""
    model.eval()
    total_loss = 0.0
    total_frames = 0.0
    for feats, labels, mask in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        preds = model(feats)
        loss = masked_bce_loss(preds, labels, mask)
        frames = mask.sum().item()
        total_loss += loss.item() * frames
        total_frames += frames
    return total_loss / max(total_frames, 1.0)

def normalize_snr_list(val):
    """
    Accept both:
      - list of strings: ['-5', '0', '5', '10']  (space separated)
      - single string with commas: '-5,0,5,10'
    Return: list[float] or None
    """
    if val is None:
        return None
    # If argparse gave us a list (nargs='+')
    if isinstance(val, list):
        if len(val) == 1 and ("," in val[0]):
            parts = [p.strip() for p in val[0].split(",") if p.strip()]
        else:
            parts = val
    else:
        # Fallback: a plain string
        parts = [p.strip() for p in str(val).split(",") if p.strip()]
    if not parts:
        return None
    out = []
    for p in parts:
        out.append(float(p))
    return out

# ----------------------------
# Main
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Train mask-based enhancement (IRM) with explicit challenge conditions.")
    ap.add_argument("--manifest-train", type=str, required=True,
                    help="CSV for training. on_the_fly => clean,noise | pre_mixed => clean,noisy")
    ap.add_argument("--manifest-val", type=str, default="",
                    help="CSV for validation (optional). If empty, skip validation.")
    ap.add_argument("--mode", type=str, default="on_the_fly",
                    choices=["on_the_fly", "pre_mixed"],
                    help="Dataset mode.")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    ap.add_argument("--epochs", type=int, default=20, help="Epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader num_workers (0 on Windows)")
    ap.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints & logs")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")

    # Challenge condition controls (passed into PairDataset)
    ap.add_argument("--snr-min", type=float, default=None, help="Min SNR (dB) for uniform sampling")
    ap.add_argument("--snr-max", type=float, default=None, help="Max SNR (dB) for uniform sampling")
    # Accept space-separated list (PowerShell-friendly) or a single comma-separated string
    ap.add_argument("--snr-list", nargs="+", default=None,
                    help="SNR list (space separated), e.g., --snr-list -5 0 5 10; "
                         "or a single comma-separated string in quotes.")
    ap.add_argument("--noise-filter", type=str, default="",
                    help="Substring or regex for selecting noise files (on_the_fly).")

    return ap.parse_args()

def main():
    args = parse_args()

    # Prepare
    seed_all(args.seed)
    make_dirs(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # Parse SNR list robustly
    snr_list = normalize_snr_list(args.snr_list)

    # Frequency dimension
    in_dim = N_FFT // 2 + 1

    # Build datasets/loaders with explicit challenge controls
    ds_train = PairDataset(
        args.manifest_train,
        mode=args.mode,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        snr_list=snr_list,
        noise_filter=(args.noise_filter or None),
        seed=args.seed
    )
    dl_train = DataLoader(
        ds_train, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate_pad, drop_last=False
    )

    if args.manifest_val:
        ds_val = PairDataset(
            args.manifest_val,
            mode=args.mode,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            snr_list=snr_list,
            noise_filter=(args.noise_filter or None),
            seed=args.seed + 1
        )
        dl_val = DataLoader(
            ds_val, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, collate_fn=collate_pad, drop_last=False
        )
    else:
        ds_val, dl_val = None, None

    # Model / optimizer
    model = MaskNet(in_dim=in_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- Save run metadata (challenge conditions) ----
    run_meta = {
        "mode": args.mode,
        "manifest_train": args.manifest_train,
        "manifest_val": args.manifest_val or None,
        "save_dir": args.save_dir,
        "seed": args.seed,
        "snr_min": args.snr_min,
        "snr_max": args.snr_max,
        "snr_list": snr_list,
        "noise_filter": args.noise_filter or None,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "workers": args.workers,
        "in_dim": in_dim,
        "train_pairs": len(ds_train),
        "val_pairs": (len(ds_val) if ds_val is not None else 0),
        "device": str(device),
    }
    meta_path = Path(args.save_dir) / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved run metadata -> {meta_path}")

    # ---- Training loop ----
    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        # Train one epoch
        model.train()
        running = 0.0
        seen_frames = 0.0
        for feats, labels, mask in dl_train:
            feats = feats.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            preds = model(feats)
            loss = masked_bce_loss(preds, labels, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            frames = mask.sum().item()
            running += loss.item() * frames
            seen_frames += frames

        tr_loss = running / max(seen_frames, 1.0)

        # Validation
        if dl_val is not None:
            val_loss = run_eval(model, dl_val, device)
            print(f"[Epoch {epoch:03d}] train_bce={tr_loss:.6f}  val_bce={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_path = Path(args.save_dir) / "masknet_best.pt"
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_bce": best_val,
                    "run_meta": run_meta
                }, best_path)
        else:
            val_loss = None
            print(f"[Epoch {epoch:03d}] train_bce={tr_loss:.6f}")

        # Always save last
        last_path = Path(args.save_dir) / "masknet_last.pt"
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_bce": (val_loss if val_loss is not None else None),
            "run_meta": run_meta
        }, last_path)

    print("[Done] Training finished.")

if __name__ == "__main__":
    main()
