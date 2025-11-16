import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import soundfile as sf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.stft import stft, istft, N_FFT, SR
from src.utils import load_wav, save_wav, snr_db
from src.dnn_mask_improved import MaskNet, ImprovedMaskNet, DeepMaskNet

try:
    from pesq import pesq as pesq_api
except Exception:
    pesq_api = None

try:
    from pystoi.stoi import stoi as stoi_api
except Exception:
    stoi_api = None


def _align_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = min(len(a), len(b))
    return a[:L], b[:L]


def eval_pair(clean_path: str, test_path: str, sr: int = SR) -> Dict[str, float]:
    """
    Returns:
        {
          "snr_db": float,
          "pesq_wb": float or nan,
          "stoi": float or nan
        }
    """
    c = load_wav(clean_path, sr=sr)
    t = load_wav(test_path, sr=sr)
    c, t = _align_length(c, t)

    # SNR
    try:
        snr = float(snr_db(c, t))
    except Exception:
        snr = float("nan")

    # PESQ (wb)
    pesq_wb = float("nan")
    if pesq_api is not None:
        try:
            pesq_wb = float(pesq_api(sr, c, t, "wb"))
        except Exception:
            try:
                pesq_wb = float(pesq_api(c, t, sr))
            except Exception:
                pesq_wb = float("nan")

    # STOI
    stoi_val = float("nan")
    if stoi_api is not None:
        try:
            stoi_val = float(stoi_api(c, t, sr, extended=False))
        except Exception:
            stoi_val = float("nan")

    return {"snr_db": snr, "pesq_wb": pesq_wb, "stoi": stoi_val}


def enhance_with_model(model: torch.nn.Module,
                       noisy_wav: np.ndarray,
                       device: torch.device = torch.device("cpu"),
                       use_log: bool = False) -> np.ndarray:
    """
    Args:
        model: MaskNet, ImprovedMaskNet, or DeepMaskNet
        noisy_wav: Noisy audio waveform
        device: Inference device
        use_log: Whether to use log magnitude spectrogram (must match training)
    """
    Y = stft(noisy_wav)
    mag_noisy = np.abs(Y)
    phase_noisy = np.angle(Y)

    feats_TF = mag_noisy.T.astype(np.float32)
    
    T, Fdim = feats_TF.shape

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(feats_TF).unsqueeze(0).to(device)
        pred_TF = model(inp).squeeze(0).cpu().numpy()

    mask_FT = pred_TF.T
    mask_FT = np.clip(mask_FT, 0.0, 1.0)
    
    print(f"[Debug] Raw mask stats: min={mask_FT.min():.4f}, max={mask_FT.max():.4f}, mean={mask_FT.mean():.4f}")

    mask_FT = 0.7 * mask_FT + 0.3
    mask_FT = np.clip(mask_FT, 0.3, 1.0)
    
    print(f"[Debug] Processed mask stats: min={mask_FT.min():.4f}, max={mask_FT.max():.4f}, mean={mask_FT.mean():.4f}")

    enh_mag = mag_noisy * mask_FT

    enh_S = enh_mag * np.exp(1j * phase_noisy)
    enh_wav = istft(enh_S).astype(np.float32)

    mx = float(np.max(np.abs(enh_wav)) + 1e-12)
    if mx > 1.0:
        enh_wav = enh_wav / mx

    return enh_wav


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DNN MaskNet: enhance noisy wav and compute SNR/PESQ/STOI."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to masknet_best.pt")
    parser.add_argument("--clean", type=str, required=True, help="Path to clean audio")
    parser.add_argument("--noisy", type=str, required=True, help="Path to noisy audio")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--sr", type=int, default=SR, help="Sample rate (default: stft.SR)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("[Info] Using device:", device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    enh_path = outdir / "enhanced_dnn.wav"
    report_json = outdir / "report_dnn.json"
    report_csv  = outdir / "report_dnn.csv"

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        run_meta = ckpt.get("run_meta", {})
    else:
        state_dict = ckpt
        run_meta = {}

    model_type = run_meta.get("model_type", "original")
    use_log = run_meta.get("use_log", False)
    in_dim = run_meta.get("in_dim", N_FFT // 2 + 1)
    
    print(f"[Info] Model type: {model_type}, use_log: {use_log}, in_dim: {in_dim}")
    
    if model_type == "improved":
        hidden_dim = run_meta.get("hidden_dim", 512)
        num_layers = run_meta.get("num_layers", 2)
        dropout = run_meta.get("dropout", 0.2)
        model = ImprovedMaskNet(in_dim=in_dim, hidden_dim=hidden_dim,
                               num_layers=num_layers, use_log=use_log,
                               dropout=dropout)
    elif model_type == "deep":
        hidden_dims = run_meta.get("hidden_dims", [512, 512, 256, 256])
        dropout = run_meta.get("dropout", 0.3)
        model = DeepMaskNet(in_dim=in_dim, hidden_dims=hidden_dims,
                           dropout=dropout, use_log=use_log)
    else:  # original
        model = MaskNet(in_dim=in_dim)
    
    model.load_state_dict(state_dict, strict=True)
    print(f"[Info] Loaded checkpoint: {args.ckpt}")

    clean = load_wav(args.clean, sr=args.sr)
    noisy = load_wav(args.noisy, sr=args.sr)

    enhanced = enhance_with_model(model, noisy, device=device, use_log=use_log)
    save_wav(enhanced, str(enh_path), sr=args.sr)
    print(f"[Info] Saved enhanced wav -> {enh_path}")

    m_noisy = eval_pair(args.clean, args.noisy, sr=args.sr)
    m_enh   = eval_pair(args.clean, str(enh_path), sr=args.sr)

    report = {
        "checkpoint": args.ckpt,
        "sr": args.sr,
        "metrics": {
            "noisy": m_noisy,
            "enhanced": m_enh,
        }
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved report json -> {report_json}")

    with open(report_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "snr_db", "pesq_wb", "stoi"])
        w.writerow(["noisy", m_noisy.get("snr_db"), m_noisy.get("pesq_wb"), m_noisy.get("stoi")])
        w.writerow(["enhanced", m_enh.get("snr_db"), m_enh.get("pesq_wb"), m_enh.get("stoi")])
    print(f"[Info] Saved report csv  -> {report_csv}")
    print("[Done] Evaluation finished.")


if __name__ == "__main__":
    main()
