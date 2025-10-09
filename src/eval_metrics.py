# src/eval_metrics.py
import numpy as np
from pathlib import Path
from src.utils import load_wav, snr_db

# PESQ / STOI
from pesq import pesq        # pip install pesq
from pystoi.stoi import stoi # pip install pystoi

SR = 16000

def _align(a: np.ndarray, b: np.ndarray):
    """Align two signals to the same length"""
    L = min(len(a), len(b))
    return a[:L], b[:L]

def eval_pair(clean_path: str, test_path: str, sr: int, init_sr = SR):
    """
    Pairwise evaluation (clean vs. test signal): compute SNR / PESQ / STOI
    """
    c, _ = load_wav(clean_path, sr)
    t, _ = load_wav(test_path, sr)
    c, t = _align(c, t)

    # SNR
    snr = snr_db(c, t)

    # PESQ (wideband 16 kHz). May sometimes throw errors due to amplitude/length mismatch.
    try:
        pesq_wb = pesq(sr, c, t, 'wb')
    except Exception:
        pesq_wb = float('nan')

    # STOI (0~1)
    try:
        stoi_val = float(stoi(c, t, sr, extended=False))
    except Exception:
        stoi_val = float('nan')

    return {"snr_db": float(snr), "pesq_wb": float(pesq_wb), "stoi": stoi_val}
