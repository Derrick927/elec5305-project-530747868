#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dnn.py  —  Full replacement (fixed device handling)

Enhance a noisy wav using a trained MaskNet checkpoint, then evaluate
SNR/PESQ/STOI and save a report together with the run's challenge conditions.

Outputs:
  - results/enhanced_from_ckpt.wav
  - results/report.json
  - results/report.csv

Examples (PowerShell-safe):
  python .\scripts\eval_dnn.py `
    --ckpt "checkpoints\demo\masknet_best.pt" `
    --clean "data\clean\example.wav" `
    --noisy "data\noisy\example_noisy.wav" `
    --outdir "results" `
    --device auto
"""

import os
import json
import csv
from pathlib import Path
import argparse
from typing import Tuple, Dict

import numpy as np
import torch

# Make src importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.dnn_mask import MaskNet
from src.eval_metrics import eval_pair  # returns dict with snr_db / pesq_wb / stoi
from src.utils import load_wav, save_wav
from src.stft import stft, istft, N_FFT


# -----------------------------
# Model helpers
# -----------------------------
def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[Dict, Dict]:
    """Load checkpoint and return (state_dict, run_meta)."""
    obj = torch.load(ckpt_path, map_location=device)
    state_dict = obj.get("state_dict", obj)
    run_meta = obj.get("run_meta", {})
    return state_dict, run_meta


def build_model_from_meta(run_meta: Dict, device: torch.device) -> torch.nn.Module:
    """Build MaskNet using frequency dimension from run_meta if available."""
    in_dim = run_meta.get("in_dim", N_FFT // 2 + 1)
    model = MaskNet(in_dim=in_dim).to(device)
    model.eval()
    return model


@torch.no_grad()
def enhance_with_model(model: torch.nn.Module, noisy_wav: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Enhance a noisy waveform using the predicted ratio mask and noisy phase.
    Ensures the forward pass happens on the same device as the model.
    """
    # STFT on CPU (numpy) is fine; we only move the NN features to the model device.
    Y = stft(noisy_wav)                                  # (F, T) complex
    mag_noisy = np.abs(Y).T.astype(np.float32)           # (T, F) float32
    feats = torch.from_numpy(mag_noisy)[None, ...].to(device)  # (1, T, F) on device

    # Forward on device -> mask in [0,1]
    preds = model(feats)[0]                               # (T, F) tensor
    M = preds.transpose(0, 1).detach().cpu().numpy()      # -> (F, T) on CPU for ISTFT

    # Apply mask on magnitude and reuse noisy phase
    S_mag_hat = np.abs(Y) * M                             # (F, T)
    phase = np.exp(1j * np.angle(Y))
    S_hat = S_mag_hat * phase

    enhanced = istft(S_hat).astype(np.float32)

    # Soft clipping for safety
    mx = float(np.max(np.abs(enhanced)) + 1e-12)
    if mx > 1.0:
        enhanced = enhanced / mx
    return enhanced


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a trained MaskNet checkpoint.")
    ap.add_argument("--ckpt",  type=str, required=True, help="Path to checkpoint *.pt")
    ap.add_argument("--clean", type=str, required=True, help="Path to clean wav")
    ap.add_argument("--noisy", type=str, required=True, help="Path to noisy wav")
    ap.add_argument("--outdir", type=str, default="results", help="Output directory")
    ap.add_argument("--sr", type=int, default=16000, help="Target sampling rate for I/O")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"], help="Inference device")
    return ap.parse_args()


def main():
    args = parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Info] Device: {device}")

    # Prepare outputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    enh_path = outdir / "enhanced_from_ckpt.wav"
    report_json = outdir / "report.json"
    report_csv  = outdir / "report.csv"

    # Load model + meta
    state_dict, run_meta = load_checkpoint(args.ckpt, device)
    print(f"[Info] Loaded checkpoint: {args.ckpt}")
    print(f"[Info] Run meta keys: {list(run_meta.keys())}")

    model = build_model_from_meta(run_meta, device)
    model.load_state_dict(state_dict)

    # I/O wavs
    clean = load_wav(args.clean, sr=args.sr)
    noisy = load_wav(args.noisy, sr=args.sr)

    # Enhance
    enhanced = enhance_with_model(model, noisy, device=device)
    save_wav(enhanced, str(enh_path), sr=args.sr)
    print(f"[Info] Saved enhanced wav -> {enh_path}")

    # Evaluate metrics (explicitly pass sr for consistency)
    m_noisy = eval_pair(args.clean, args.noisy, sr=args.sr)
    m_enh   = eval_pair(args.clean, str(enh_path), sr=args.sr)

    # Build report (metrics + challenge conditions)
    report = {
        "checkpoint": args.ckpt,
        "sr": args.sr,
        "metrics": {
            "noisy":     m_noisy,   # snr_db / pesq_wb / stoi
            "enhanced":  m_enh
        },
        "challenge_conditions": {
            "mode":            run_meta.get("mode"),
            "manifest_train":  run_meta.get("manifest_train"),
            "manifest_val":    run_meta.get("manifest_val"),
            "seed":            run_meta.get("seed"),
            "snr_min":         run_meta.get("snr_min"),
            "snr_max":         run_meta.get("snr_max"),
            "snr_list":        run_meta.get("snr_list"),
            "noise_filter":    run_meta.get("noise_filter"),
            "batch":           run_meta.get("batch"),
            "epochs":          run_meta.get("epochs"),
            "lr":              run_meta.get("lr"),
            "workers":         run_meta.get("workers"),
            "in_dim":          run_meta.get("in_dim"),
            "train_pairs":     run_meta.get("train_pairs"),
            "val_pairs":       run_meta.get("val_pairs"),
            "device_train":    run_meta.get("device"),
        }
    }

    # Save JSON
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved report json -> {report_json}")

    # Save CSV (flat)
    with open(report_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "snr_db", "pesq_wb", "stoi"])
        w.writerow(["noisy", m_noisy.get("snr_db"), m_noisy.get("pesq_wb"), m_noisy.get("stoi")])
        w.writerow(["enhanced", m_enh.get("snr_db"), m_enh.get("pesq_wb"), m_enh.get("stoi")])
    print(f"[Info] Saved report csv  -> {report_csv}")
    print("[Done] Evaluation finished.")

if __name__ == "__main__":
    main()

