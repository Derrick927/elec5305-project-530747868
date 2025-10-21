# -*- coding: utf-8 -*-
"""
Robust evaluation utilities for speech enhancement:
- SNR (dB) using reference clean vs. test
- PESQ (wideband, 16 kHz) with graceful fallback to NaN
- STOI (0~1) with graceful fallback to NaN

All audio is loaded via utils.load_wav(path, sr=target_sr) which returns
a mono float waveform. No sample-rate value is returned.
"""

from __future__ import annotations
from typing import Dict
import numpy as np

from .utils import load_wav, snr_db

# Optional deps
try:
    from pesq import pesq as pesq_api
except Exception:
    pesq_api = None

try:
    from pystoi.stoi import stoi as stoi_api
except Exception:
    stoi_api = None


def _align_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim both arrays to the same (minimum) length."""
    L = min(len(a), len(b))
    return a[:L], b[:L]


def eval_pair(clean_path: str, test_path: str, sr: int = 16000) -> Dict[str, float]:
    """
    Evaluate a pair of waveforms (clean vs. test).
    Returns a dict with keys: snr_db, pesq_wb, stoi
    """
    # Load and align
    c = load_wav(clean_path, sr=sr)
    t = load_wav(test_path, sr=sr)
    c, t = _align_length(c, t)

    # SNR
    try:
        snr = float(snr_db(c, t))
    except Exception:
        snr = float("nan")

    # PESQ (wideband)
    pesq_wb = float("nan")
    if pesq_api is not None:
        try:
            # pesq package signature (0.0.4+): pesq(sr, ref, deg, mode)
            pesq_wb = float(pesq_api(sr, c, t, "wb"))
        except TypeError:
            # In case of older/other signatures, try alternative call styles safely
            try:
                pesq_wb = float(pesq_api(c, t, sr))
            except Exception:
                pesq_wb = float("nan")
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
