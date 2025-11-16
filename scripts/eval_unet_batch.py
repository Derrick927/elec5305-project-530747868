"""
eval_unet_batch.py
Batch-evaluate a trained UNet checkpoint on many (clean,noisy) pairs.

Inputs (choose ONE):
  1) --pairs-csv <csv>     # each line: clean_path,noisy_path
  2) --clean-dir <dir> and --noisy-dir <dir>
     # files are paired by matching filename stem (case-insensitive)

Outputs:
  - results/enhanced/<stem>_enh.wav
  - results/all_reports.csv
  - results/all_reports.json
"""

import os
import csv
import json
from pathlib import Path
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch

# project imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from src.unet_model import UNet
from src.eval_metrics import eval_pair
from src.utils import load_wav, save_wav
from src.stft import stft, istft, SR, N_FFT


# -----------------------------
# Model helpers
# -----------------------------
def load_checkpoint(ckpt_path: str, device: torch.device):
    obj = torch.load(ckpt_path, map_location=device)
    state_dict = obj.get("state_dict", obj.get("model", obj))
    run_meta = obj.get("run_meta", {})
    return state_dict, run_meta

@torch.no_grad()
def enhance_with_model(model: torch.nn.Module, noisy_wav: np.ndarray, device: torch.device = None) -> np.ndarray:
    """
    Enhance noisy audio using UNet model
    
    Args:
        model: UNet model
        noisy_wav: Noisy audio waveform
        device: Inference device (if None, use model's current device)
    """
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()
    
    Y = stft(noisy_wav)
    mag_noisy = np.abs(Y)
    phase_noisy = np.angle(Y)
    
    inp = mag_noisy[np.newaxis, :, :]
    inp = torch.from_numpy(inp).float().to(device)
    
    enh_mag_raw = model(inp).cpu().numpy()[0]
    
    T_stft = mag_noisy.shape[1]
    T_model = enh_mag_raw.shape[1]
    
    if T_model < T_stft:
        pad = T_stft - T_model
        enh_mag = np.pad(enh_mag_raw, ((0, 0), (0, pad)), mode="constant")
    elif T_model > T_stft:
        enh_mag = enh_mag_raw[:, :T_stft]
    else:
        enh_mag = enh_mag_raw
    
    eps = 1e-8
    mask_FT = enh_mag / (mag_noisy + eps)
    
    mask_FT = np.clip(mask_FT, 0.0, 2.0)
    mask_FT = 0.7 * mask_FT + 0.3
    mask_FT = np.clip(mask_FT, 0.3, 1.2)
    
    final_mag = mag_noisy * mask_FT
    
    enh_complex = final_mag * np.exp(1j * phase_noisy)
    enh_wav = istft(enh_complex).astype(np.float32)
    
    mx = np.max(np.abs(enh_wav))
    if mx > 1.0:
        enh_wav /= mx
    
    return enh_wav


# -----------------------------
# Pair loading
# -----------------------------
def read_pairs_from_csv(csv_path: str) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 2:
                continue
            c, n = row[0].strip(), row[1].strip()
            if c and n:
                pairs.append((Path(c), Path(n)))
    if not pairs:
        raise RuntimeError(f"No valid pairs in CSV: {csv_path}")
    return pairs

def pair_by_stem(clean_dir: str, noisy_dir: str) -> List[Tuple[Path, Path]]:
    cdir = Path(clean_dir)
    ndir = Path(noisy_dir)
    cfiles = {p.stem.lower(): p for p in cdir.rglob("*") if p.suffix.lower() in {".wav", ".flac"}}
    nfiles = {p.stem.lower(): p for p in ndir.rglob("*") if p.suffix.lower() in {".wav", ".flac"}}
    common = sorted(set(cfiles.keys()) & set(nfiles.keys()))
    pairs = [(cfiles[s], nfiles[s]) for s in common]
    if not pairs:
        raise RuntimeError("No (clean,noisy) stem matches between folders.")
    return pairs


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Batch evaluation for UNet.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint *.pt")
    # input options
    ap.add_argument("--pairs-csv", type=str, default="", help="CSV with clean,noisy per line")
    ap.add_argument("--manifest", type=str, default="", dest="manifest", help="Alias for --pairs-csv (CSV with clean,noisy per line)")
    ap.add_argument("--clean-dir", type=str, default="", help="Folder of clean wavs")
    ap.add_argument("--noisy-dir", type=str, default="", help="Folder of noisy wavs")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate at most N pairs (0 = all)")
    # model options
    ap.add_argument("--base-channels", type=int, default=64, help="UNet base_channels (default: 64)")
    # common options
    ap.add_argument("--outdir", type=str, default="results", help="Output dir for reports and enhanced wavs")
    ap.add_argument("--sr", type=int, default=16000, help="I/O sampling rate")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    return ap.parse_args()

def main():
    args = parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Info] Device: {device}")

    outdir = Path(args.outdir)
    enh_dir = outdir / "enhanced"
    outdir.mkdir(parents=True, exist_ok=True)
    enh_dir.mkdir(parents=True, exist_ok=True)

    # gather pairs
    pairs_csv = args.manifest or args.pairs_csv
    if pairs_csv:
        pairs = read_pairs_from_csv(pairs_csv)
    else:
        assert args.clean_dir and args.noisy_dir, "Use --manifest/--pairs-csv or (--clean-dir AND --noisy-dir)."
        pairs = pair_by_stem(args.clean_dir, args.noisy_dir)

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"[Info] Num pairs: {len(pairs)}")

    # model
    print("[Info] Building UNet model...")
    model = UNet(n_channels=N_FFT // 2 + 1, n_classes=N_FFT // 2 + 1, base_channels=args.base_channels).to(device)
    
    print("[Info] Loading checkpoint:", args.ckpt)
    state_dict, run_meta = load_checkpoint(args.ckpt, device)
    
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("[Info] Model loaded successfully.")

    # loop
    rows: List[Dict] = []
    for i, (cp, np_) in enumerate(pairs, 1):
        try:
            clean = load_wav(str(cp), sr=args.sr)
            noisy = load_wav(str(np_), sr=args.sr)
            # enhance
            enhanced = enhance_with_model(model, noisy, device=device)
            enh_path = enh_dir / f"{cp.stem}_enh.wav"
            save_wav(enhanced, str(enh_path), sr=args.sr)

            # metrics
            m_noisy = eval_pair(str(cp), str(np_), sr=args.sr)
            m_enh = eval_pair(str(cp), str(enh_path), sr=args.sr)

            row = {
                "index": i,
                "clean": str(cp),
                "noisy": str(np_),
                "enhanced": str(enh_path),
                "noisy_snr_db": m_noisy.get("snr_db"),
                "noisy_pesq_wb": m_noisy.get("pesq_wb"),
                "noisy_stoi": m_noisy.get("stoi"),
                "enh_snr_db": m_enh.get("snr_db"),
                "enh_pesq_wb": m_enh.get("pesq_wb"),
                "enh_stoi": m_enh.get("stoi"),
            }
            rows.append(row)
            print(f"[{i}/{len(pairs)}] {cp.name} -> DONE")
        except Exception as e:
            print(f"[{i}/{len(pairs)}] {cp.name} -> ERROR: {e}")

    # write CSV
    csv_path = outdir / "all_reports.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "index","clean","noisy","enhanced",
            "noisy_snr_db","noisy_pesq_wb","noisy_stoi",
            "enh_snr_db","enh_pesq_wb","enh_stoi"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Info] Saved CSV -> {csv_path}")

    # write JSON with run_meta + per-file + averages
    def _avg(key: str):
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int,float)) and not np.isnan(r[key])]
        return (float(np.mean(vals)) if vals else float("nan"))

    summary = {
        "checkpoint": args.ckpt,
        "sr": args.sr,
        "num_pairs": len(rows),
        "run_meta": run_meta,
        "averages": {
            "noisy_snr_db": _avg("noisy_snr_db"),
            "noisy_pesq_wb": _avg("noisy_pesq_wb"),
            "noisy_stoi": _avg("noisy_stoi"),
            "enh_snr_db": _avg("enh_snr_db"),
            "enh_pesq_wb": _avg("enh_pesq_wb"),
            "enh_stoi": _avg("enh_stoi"),
        },
        "items": rows
    }
    json_path = outdir / "all_reports.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved JSON -> {json_path}")

    print("[Done] Batch evaluation finished.")

if __name__ == "__main__":
    main()

