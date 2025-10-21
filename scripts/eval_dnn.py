#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dnn.py
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
    --outdir "results"

  # If you don't have those example wavs, you can first run:
  #   python .\scripts\noise_test.py
"""

import os
import json
import csv
from pathlib import Path
import argparse
import numpy as np
import torch

# Make src importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dnn_mask import MaskNet
from src.eval_metrics import eval_pair
from src.utils import load_wav, save_wav
from src.stft import stft, istft, N_FFT


def enhance_with_model(model: torch.nn.Module, noisy_wav: np.ndarray) -> np.ndarray:
    """
    Enhance a noisy waveform using the predicted ratio mask and noisy phase.
    """
    model.eval()
    with torch.no_grad():
        Y = stft(noisy_wav)                 # (F, T) complex
        mag_noisy = np.abs(Y).T.astype(np.float32)  # (T, F)
        feats = torch.from_numpy(mag_noisy)[None, ...]  # (1, T, F)
        preds = model(feats).cpu().numpy()[0]          # (T, F) in [0,1]
        # Apply mask on magnitude, reuse noisy phase
        M = preds.T                                 # (F, T)
        S_mag_hat = np.abs(Y) * M                   # (F, T)
        # Reconstruct with noisy phase
        phase = np.exp(1j * np.angle(Y))
        S_hat = S_mag_hat * phase
        enhanced = istft(S_hat).astype(np.float32)
        # Soft clipping
        mx = np.max(np.abs(enhanced)) + 1e-12
        if mx > 1.0:
            enhanced = enhanced / mx
        return enhanced


def load_checkpoint(ckpt_path: str, device: torch.device):
    """
    Load checkpoint and return (state_dict, run_meta).
    """
    obj = torch.load(ckpt_path, map_location=device)
    state_dict = obj.get("state_dict", obj)
    run_meta = obj.get("run_meta", {})
    return state_dict, run_meta


def build_model_from_meta(run_meta: dict, device: torch.device):
    """
    Build MaskNet using frequency dimension from run_meta if available.
    """
    in_dim = run_meta.get("in_dim", N_FFT // 2 + 1)
    model = MaskNet(in_dim=in_dim).to(device)
    return model


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained MaskNet checkpoint.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint *.pt")
    ap.add_argument("--clean", type=str, required=True, help="Path to clean wav")
    ap.add_argument("--noisy", type=str, required=True, help="Path to noisy wav")
    ap.add_argument("--outdir", type=str, default="results", help="Output directory")
    ap.add_argument("--sr", type=int, default=16000, help="Target sampling rate for I/O")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Info] Device: {device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    enh_path = outdir / "enhanced_from_ckpt.wav"
    report_json = outdir / "report.json"
    report_csv = outdir / "report.csv"

    # Load run meta and model
    state_dict, run_meta = load_checkpoint(args.ckpt, device)
    print(f"[Info] Loaded checkpoint: {args.ckpt}")
    print(f"[Info] Run meta keys: {list(run_meta.keys())}")

    model = build_model_from_meta(run_meta, device)
    model.load_state_dict(state_dict)

    # I/O wavs
    clean = load_wav(args.clean, sr=args.sr)
    noisy = load_wav(args.noisy, sr=args.sr)

    # Enhance
    enhanced = enhance_with_model(model, noisy)
    save_wav(enhanced, str(enh_path), sr=args.sr)
    print(f"[Info] Saved enhanced wav -> {enh_path}")

    # Evaluate metrics
    m_noisy = eval_pair(args.clean, args.noisy)
    m_enh = eval_pair(args.clean, str(enh_path))

    # Build report (metrics + challenge conditions)
    report = {
        "checkpoint": args.ckpt,
        "sr": args.sr,
        "metrics": {
            "noisy": m_noisy,       # snr_db / pesq_wb / stoi
            "enhanced": m_enh
        },
        "challenge_conditions": {
            "mode": run_meta.get("mode"),
            "manifest_train": run_meta.get("manifest_train"),
            "manifest_val": run_meta.get("manifest_val"),
            "seed": run_meta.get("seed"),
            "snr_min": run_meta.get("snr_min"),
            "snr_max": run_meta.get("snr_max"),
            "snr_list": run_meta.get("snr_list"),
            "noise_filter": run_meta.get("noise_filter"),
            "batch": run_meta.get("batch"),
            "epochs": run_meta.get("epochs"),
            "lr": run_meta.get("lr"),
            "workers": run_meta.get("workers"),
            "device_train": run_meta.get("device"),
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
    print(f"[Info] Saved report csv -> {report_csv}")

    print("[Done] Evaluation finished.")


if __name__ == "__main__":
    main()

