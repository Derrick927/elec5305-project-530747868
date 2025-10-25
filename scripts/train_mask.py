#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, json, argparse, random
from pathlib import Path
from typing import Optional, List

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

def seed_all(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_dirs(path: str): Path(path).mkdir(parents=True, exist_ok=True)

def collate_pad(batch):
    Ts = [x[0].shape[0] for x in batch]
    Fdim = batch[0][0].shape[1]; T_max = max(Ts); B = len(batch)
    feats = torch.zeros((B, T_max, Fdim), dtype=torch.float32)
    labels = torch.zeros((B, T_max, Fdim), dtype=torch.float32)
    mask = torch.zeros((B, T_max), dtype=torch.float32)
    for i, (noisy_T, irm_T) in enumerate(batch):
        t = noisy_T.shape[0]
        feats[i, :t, :] = torch.from_numpy(noisy_T)
        labels[i, :t, :] = torch.from_numpy(irm_T)
        mask[i, :t] = 1.0
    return feats, labels, mask

def masked_bce_loss(pred, target, mask):
    bce = nn.BCELoss(reduction="none")
    loss = bce(pred, target) * mask.unsqueeze(-1)
    return loss.sum() / (mask.sum() + 1e-8)

@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); total_loss = 0.0; total_frames = 0.0
    for feats, labels, mask in loader:
        feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
        preds = model(feats)
        loss = masked_bce_loss(preds, labels, mask)
        frames = mask.sum().item()
        total_loss += loss.item() * frames; total_frames += frames
    return total_loss / max(total_frames, 1.0)

def normalize_snr_list(val) -> Optional[List[float]]:
    if val is None: return None
    if isinstance(val, list):
        parts = [p.strip() for p in (val[0].split(",") if (len(val)==1 and "," in val[0]) else val)]
    else:
        parts = [p.strip() for p in str(val).split(",")]
    parts = [p for p in parts if p]; 
    return [float(p) for p in parts] if parts else None

class EarlyStopper:
    def __init__(self, patience=8, min_delta=1e-3):
        self.patience = int(patience); self.min_delta=float(min_delta)
        self.best=None; self.bad=0
    def step(self, current: float) -> bool:
        if (self.best is None) or (self.best - current > self.min_delta):
            self.best = current; self.bad = 0; return False
        self.bad += 1; return self.bad >= self.patience

def parse_args():
    ap = argparse.ArgumentParser(description="Train mask-based enhancement (IRM) with challenge controls.")
    ap.add_argument("--manifest-train", required=True)
    ap.add_argument("--manifest-val", default="")
    ap.add_argument("--mode", default="on_the_fly", choices=["on_the_fly","pre_mixed"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--early-stop-delta", type=float, default=1e-3)
    ap.add_argument("--snr-min", type=float, default=None)
    ap.add_argument("--snr-max", type=float, default=None)
    ap.add_argument("--snr-list", nargs="+", default=None)
    ap.add_argument("--noise-filter", type=str, default="")
    return ap.parse_args()

def resolve_device(flag: str) -> torch.device:
    if flag == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    args = parse_args()
    seed_all(args.seed); make_dirs(args.save_dir)
    device = resolve_device(args.device)
    if device.type == "cuda": torch.backends.cudnn.benchmark = True
    print(f"[Info] Device: {device}")

    snr_list = normalize_snr_list(args.snr_list)
    in_dim = N_FFT // 2 + 1

    ds_train = PairDataset(args.manifest_train, mode=args.mode,
                           snr_min=args.snr_min, snr_max=args.snr_max,
                           snr_list=snr_list, noise_filter=(args.noise_filter or None),
                           seed=args.seed)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate_pad)

    dl_val = None
    if args.manifest_val:
        ds_val = PairDataset(args.manifest_val, mode=args.mode,
                             snr_min=args.snr_min, snr_max=args.snr_max,
                             snr_list=snr_list, noise_filter=(args.noise_filter or None),
                             seed=args.seed+1)
        dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_pad)

    model = MaskNet(in_dim=in_dim).to(device)
    optimiz = optim.Adam(model.parameters(), lr=args.lr)

    run_meta = {
        "mode": args.mode, "manifest_train": args.manifest_train,
        "manifest_val": args.manifest_val or None, "save_dir": args.save_dir,
        "seed": args.seed, "snr_min": args.snr_min, "snr_max": args.snr_max,
        "snr_list": snr_list, "noise_filter": args.noise_filter or None,
        "batch": args.batch, "epochs": args.epochs, "lr": args.lr,
        "workers": args.workers, "in_dim": in_dim,
        "train_pairs": len(ds_train), "val_pairs": (len(ds_val) if dl_val else 0),
        "device": str(device),
        "early_stop_patience": (args.early_stop_patience if dl_val else None),
        "early_stop_delta": (args.early_stop_delta if dl_val else None),
    }
    meta_path = Path(args.save_dir) / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f: json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved run metadata -> {meta_path}")

    # CSV logger
    log_path = Path(args.save_dir) / "train_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_bce,val_bce\n")

    stopper = EarlyStopper(args.early_stop_patience, args.early_stop_delta) if dl_val else None
    best_val = math.inf
    best_path = Path(args.save_dir) / "masknet_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0; seen = 0.0
        for feats, labels, mask in dl_train:
            feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
            preds = model(feats)
            loss = masked_bce_loss(preds, labels, mask)
            optimiz.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiz.step()
            frames = mask.sum().item(); running += loss.item()*frames; seen += frames
        tr_loss = running / max(seen, 1.0)

        if dl_val:
            val_loss = run_eval(model, dl_val, device)
            print(f"[Epoch {epoch:03d}] train_bce={tr_loss:.6f}  val_bce={val_loss:.6f}")
            # save best
            if val_loss < best_val - 1e-12:
                best_val = val_loss
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "val_bce": best_val, "run_meta": run_meta}, best_path)
            # early stop?
            if stopper and stopper.step(val_loss):
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch},{tr_loss},{val_loss}\n")
                last_path = Path(args.save_dir) / "masknet_last.pt"
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "val_bce": val_loss, "run_meta": run_meta}, last_path)
                print(f"[EarlyStop] Patience reached. Best val_bce={best_val:.6f}.")
                print("[Done] Training finished (early stopped).")
                return
        else:
            val_loss = None
            print(f"[Epoch {epoch:03d}] train_bce={tr_loss:.6f}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss},{'' if val_loss is None else val_loss}\n")

        last_path = Path(args.save_dir) / "masknet_last.pt"
        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                    "val_bce": val_loss, "run_meta": run_meta}, last_path)

    print("[Done] Training finished.")

if __name__ == "__main__":
    main()
