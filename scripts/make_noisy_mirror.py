#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_noisy_mirror.py — Create a mirrored noisy set with identical filenames.

It reads all .wav files from --clean_dir, adds white noise at a target SNR,
and writes them into --noisy_dir using the exact same filenames. This ensures
eval_dnn_batch.py can pair files by stem (basename).

Usage (PowerShell):
  python scripts/make_noisy_mirror.py `
    --clean_dir data/public/val_clean `
    --noisy_dir data/public/val_noise `
    --snr_db 0 `
    --sr 16000
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_wav, save_wav
from src.add_noise import add_white_noise

def parse_args():
    ap = argparse.ArgumentParser(description="Mirror clean -> noisy with identical filenames.")
    ap.add_argument("--clean_dir", type=str, required=True, help="Directory of clean wavs")
    ap.add_argument("--noisy_dir", type=str, required=True, help="Output directory for noisy wavs")
    ap.add_argument("--snr_db", type=float, default=0.0, help="Target SNR in dB (default: 0)")
    ap.add_argument("--sr", type=int, default=16000, help="Sampling rate for I/O")
    return ap.parse_args()

def main():
    args = parse_args()
    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted([p for p in clean_dir.glob("*.wav") if p.is_file()])
    assert wavs, f"No wav files found under {clean_dir}"

    for p in wavs:
        x = load_wav(str(p), sr=args.sr)
        y = add_white_noise(x, snr_db=args.snr_db)
        out_path = noisy_dir / p.name  # keep identical filename
        save_wav(y, str(out_path), sr=args.sr)

    print(f"[Done] Generated {len(wavs)} noisy files into: {noisy_dir.resolve()}")
    print("[Hint] Now eval_dnn_batch.py can pair by identical stems.")

if __name__ == "__main__":
    main()

