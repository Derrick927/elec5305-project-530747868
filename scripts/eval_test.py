#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_test.py
Evaluate clean vs noisy and denoised wavs with SNR, PESQ (wb), and STOI.

Usage:
    python scripts/eval_test.py
"""

import sys
import os
import csv
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval_metrics import eval_pair  # Calculates snr_db, pesq_wb, stoi

# Default paths (consistent with other scripts)
CLEAN = "data/clean/example.wav"
NOISY = "data/noisy/example_noisy.wav"
DENOISED = "results/example_denoised.wav"

def ensure_exists(paths):
    """Raise AssertionError if any of the required files is missing."""
    for p in paths:
        assert Path(p).exists(), (
            f"Missing file: {p}\n"
            f"Hint: run `python scripts/noise_test.py` to generate noisy and denoised results."
        )

def r(d):
    """Round float metrics for pretty printing; keep NaN unchanged."""
    return {k: (round(v, 4) if v == v else v) for k, v in d.items()}

def main():
    # Check required files
    ensure_exists([CLEAN, NOISY, DENOISED])

    print(">>> Evaluating...")
    m_noisy = eval_pair(CLEAN, NOISY)
    m_deno = eval_pair(CLEAN, DENOISED)

    # Pretty print
    print("\n[NOISY]")
    print(r(m_noisy))
    print("\n[DENOISED]")
    print(r(m_deno))

    # Write CSV
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "snr_db", "pesq_wb", "stoi"])
        w.writerow(["noisy", m_noisy["snr_db"], m_noisy["pesq_wb"], m_noisy["stoi"]])
        w.writerow(["denoised", m_deno["snr_db"], m_deno["pesq_wb"], m_deno["stoi"]])
    print("\nSaved metrics to:", csv_path)

if __name__ == "__main__":
    main()

