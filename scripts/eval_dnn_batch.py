#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dnn_mask import MaskNet
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
    in_dim = run_meta.get("in_dim", N_FFT // 2 + 1)
    model = MaskNet(in_dim=in_dim).to(device)
    return model

@torch.no_grad()
def enhance_with_model(model: torch.nn.Module, noisy_wav: np.ndarray) -> np.ndarray:
    model.eval()
    Y = stft(noisy_wav)                        # (F,T) complex
    mag_noisy = np.abs(Y).T.astype(np.float32) # (T,F)
    feats = torch.from_numpy(mag_noisy)[None, ...]  # (1,T,F)
    preds = model(feats).cpu().numpy()[0]           # (T,F) in [0,1]
    M = preds.T                                     # (F,T)
    S_mag_hat = np.abs(Y) * M
    phase = np.exp(1j * np.angle(Y))
    S_hat = S_mag_hat * phase
    enh = istft(S_hat).astype(np.float32)
    mx = float(np.max(np.abs(enh)) + 1e-12)
    if mx > 1.0:
        enh = enh / mx
    return enh


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
    if args.pairs_csv:
        pairs = read_pairs_from_csv(args.pairs_csv)
    else:
        assert args.clean_dir and args.noisy_dir, "Use --pairs-csv or (--clean-dir AND --noisy-dir)."
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
            enhanced = enhance_with_model(model, noisy)
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
