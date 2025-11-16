
"""
eval_dnn_batch.py
Batch-evaluate a trained MaskNet checkpoint on many (clean,noisy) pairs.

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
current_dir = os.path.dirname(os.path.abspath(__file__))   # ...\speech-enhance-mask\scripts
project_root = os.path.dirname(current_dir)                # ...\speech-enhance-mask
sys.path.insert(0, project_root)
from src.dnn_mask_improved import MaskNet, ImprovedMaskNet, DeepMaskNet
from src.eval_metrics import eval_pair
from src.utils import load_wav, save_wav
from src.stft import stft, istft, N_FFT


# -----------------------------
# Model helpers
# -----------------------------
def load_checkpoint(ckpt_path: str, device: torch.device):
    obj = torch.load(ckpt_path, map_location=device)
    state_dict = obj.get("state_dict", obj)
    run_meta = obj.get("run_meta", {})
    return state_dict, run_meta

def build_model_from_meta(run_meta: dict, device: torch.device):
    """Build model based on run_meta"""
    model_type = run_meta.get("model_type", "original")
    use_log = run_meta.get("use_log", False)
    in_dim = run_meta.get("in_dim", N_FFT // 2 + 1)
    
    if model_type == "improved":
        hidden_dim = run_meta.get("hidden_dim", 512)
        num_layers = run_meta.get("num_layers", 2)
        dropout = run_meta.get("dropout", 0.2)
        model = ImprovedMaskNet(in_dim=in_dim, hidden_dim=hidden_dim,
                               num_layers=num_layers, use_log=use_log,
                               dropout=dropout).to(device)
    elif model_type == "deep":
        hidden_dims = run_meta.get("hidden_dims", [512, 512, 256, 256])
        dropout = run_meta.get("dropout", 0.3)
        model = DeepMaskNet(in_dim=in_dim, hidden_dims=hidden_dims,
                           dropout=dropout, use_log=use_log).to(device)
    else:  # original
        model = MaskNet(in_dim=in_dim).to(device)
    
    return model

@torch.no_grad()
def enhance_with_model(model: torch.nn.Module, noisy_wav: np.ndarray, device: torch.device = None) -> np.ndarray:
    """
    Enhance noisy audio using model
    
    Args:
        model: MaskNet, ImprovedMaskNet, or DeepMaskNet
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
    
    feats_TF = mag_noisy.T.astype(np.float32)
    
    inp = torch.from_numpy(feats_TF).unsqueeze(0).to(device)
    pred_TF = model(inp).squeeze(0).cpu().numpy()
    
    mask_FT = pred_TF.T
    mask_FT = np.clip(mask_FT, 0.0, 1.0)
    
    mask_FT = 0.7 * mask_FT + 0.3
    mask_FT = np.clip(mask_FT, 0.3, 1.0)
    
    enh_mag = mag_noisy * mask_FT
    
    enh_S = enh_mag * np.exp(1j * phase_noisy)
    enh_wav = istft(enh_S).astype(np.float32)
    
    mx = float(np.max(np.abs(enh_wav)) + 1e-12)
    if mx > 1.0:
        enh_wav = enh_wav / mx
    
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
    ap = argparse.ArgumentParser(description="Batch evaluation for MaskNet.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint *.pt")
    # input options
    ap.add_argument("--pairs-csv", type=str, default="", help="CSV with clean,noisy per line")
    ap.add_argument("--manifest", type=str, default="", dest="manifest", help="Alias for --pairs-csv (CSV with clean,noisy per line)")
    ap.add_argument("--clean-dir", type=str, default="", help="Folder of clean wavs")
    ap.add_argument("--noisy-dir", type=str, default="", help="Folder of noisy wavs")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate at most N pairs (0 = all)")
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
    state_dict, run_meta = load_checkpoint(args.ckpt, device)
    model = build_model_from_meta(run_meta, device)
    model.load_state_dict(state_dict)

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
