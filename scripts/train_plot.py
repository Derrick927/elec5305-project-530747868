#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_plot.py — Plot training/validation BCE curves from <save-dir>/train_log.csv

Usage (PowerShell):
  python scripts/train_plot.py --log checkpoints/demo/train_log.csv --out checkpoints/demo/train_curves.png
"""

import os, sys, csv
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser(description="Plot training curves from CSV log.")
    ap.add_argument("--log", type=str, required=True, help="Path to train_log.csv")
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default: alongside log)")
    return ap.parse_args()

def main():
    args = parse_args()
    log_path = Path(args.log)
    assert log_path.exists(), f"CSV not found: {log_path}"

    epochs, tr, va = [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_bce"]))
            v = row.get("val_bce", "")
            va.append(float(v) if v not in ("", None) else float("nan"))

    plt.figure(figsize=(8,4))
    plt.plot(epochs, tr, label="train_bce")
    if any(x == x for x in va):  # any not-NaN
        plt.plot(epochs, [x if x==x else None for x in va], label="val_bce")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title("Training Curves")
    plt.legend(); plt.tight_layout()

    out = Path(args.out) if args.out else (log_path.parent / "train_curves.png")
    plt.savefig(out); plt.close()
    print(f"[Saved] {out.resolve()}")

if __name__ == "__main__":
    main()
