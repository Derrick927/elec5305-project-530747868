# -*- coding: utf-8 -*-
"""
PairDataset with configurable on-the-fly mixing options:
- snr_min/snr_max or snr_list: control SNR of mixing
- noise_filter: include only noise files matching substring or regex
- seed: control randomness for reproducibility

CSV formats:
  on_the_fly: clean_path,noise_path
  pre_mixed : clean_path,noisy_path
"""

from __future__ import annotations
import csv
import re
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Local imports
from .utils import load_wav
from .stft import stft, istft, N_FFT

def _read_csv_pairs(csv_path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            if a and b:
                pairs.append((a, b))
    if not pairs:
        raise RuntimeError(f"No valid pairs found in: {csv_path}")
    return pairs

def _choose_snr(snr_min: Optional[float], snr_max: Optional[float],
                snr_list: Optional[List[float]], rng: random.Random) -> float:
    """Pick an SNR (dB) according to config."""
    if snr_list and len(snr_list) > 0:
        return float(rng.choice(snr_list))
    if snr_min is not None and snr_max is not None:
        lo, hi = float(snr_min), float(snr_max)
        if lo > hi:
            lo, hi = hi, lo
        return float(rng.uniform(lo, hi))
    # Default fallback: 0 dB
    return 0.0

def _apply_noise_filter(noise_path: str,
                        noise_filter: Optional[str]) -> bool:
    """
    Return True if the noise file passes the filter.
    - If noise_filter is None/empty: always True
    - If noise_filter looks like a regex (contains special chars), use regex
    - Else: substring match (case-insensitive)
    """
    if not noise_filter:
        return True
    pat = noise_filter.strip()
    if any(ch in pat for ch in ".*+?[](){}|^$"):
        return re.search(pat, noise_path, flags=re.IGNORECASE) is not None
    return pat.lower() in noise_path.lower()

def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix noise into clean to reach target SNR (dB).
    SNR = 10 * log10( sum(clean^2) / sum(noise^2) )
    We scale noise to meet target SNR.
    """
    L = len(clean)
    if len(noise) < L:
        reps = int(np.ceil(L / max(len(noise), 1)))
        noise = np.tile(noise, reps)[:L]
    else:
        noise = noise[:L]

    p_clean = np.sum(clean.astype(np.float64)**2) + 1e-12
    p_noise = np.sum(noise.astype(np.float64)**2) + 1e-12
    target_ratio = 10.0 ** (-snr_db / 10.0)  # Pn/Ps
    scale = np.sqrt((p_clean * target_ratio) / p_noise)
    noisy = clean + scale * noise
    max_abs = np.max(np.abs(noisy)) + 1e-12
    if max_abs > 1.0:
        noisy = noisy / max_abs
    return noisy.astype(np.float32)

def _compute_irm(mag_clean: np.ndarray, mag_noise: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Ideal Ratio Mask: |S| / (|S| + |N|)
    Input shapes: (F, T)
    Return: (T, F)
    """
    irm = mag_clean / (mag_clean + mag_noise + eps)
    irm = np.clip(irm, 0.0, 1.0)
    return irm.T.astype(np.float32)

class PairDataset:
    def __init__(self,
                 manifest_csv: str,
                 mode: str = "on_the_fly",
                 sr: int = 16000,
                 # New options for challenge condition control:
                 snr_min: Optional[float] = None,
                 snr_max: Optional[float] = None,
                 snr_list: Optional[List[float]] = None,
                 noise_filter: Optional[str] = None,
                 seed: int = 1337):
        """
        Args:
            manifest_csv: CSV file path.
            mode: "on_the_fly" (clean,noise) or "pre_mixed" (clean,noisy).
            sr: target sample rate for loading.
            snr_min/snr_max: if both provided, choose uniform SNR in [min,max].
            snr_list: if provided, choose SNR from this list (overrides min/max).
            noise_filter: substring or regex to select noise files (on_the_fly).
            seed: random seed for reproducible on_the_fly mixing.
        """
        self.mode = mode
        self.sr = int(sr)
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.snr_list = list(snr_list) if snr_list else None
        self.noise_filter = noise_filter
        self.rng = random.Random(int(seed))

        self.pairs = _read_csv_pairs(manifest_csv)
        if self.mode == "on_the_fly" and self.noise_filter:
            before = len(self.pairs)
            self.pairs = [(c, n) for (c, n) in self.pairs if _apply_noise_filter(n, self.noise_filter)]
            after = len(self.pairs)
            if after == 0:
                raise RuntimeError(f"No pairs left after noise_filter='{self.noise_filter}'.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """
        Returns:
            noisy_T: (T, F) float32
            label_T: (T, F) float32 (IRM)
        """
        a, b = self.pairs[idx]
        if self.mode == "on_the_fly":
            clean = load_wav(a, sr=self.sr)  # (L,)
            noise = load_wav(b, sr=self.sr)  # (L2,)

            snr = _choose_snr(self.snr_min, self.snr_max, self.snr_list, self.rng)
            x = _mix_at_snr(clean, noise, snr_db=snr)

            Y = stft(x)           # (F, T) complex
            S = stft(clean)       # (F, T)

            mag_noisy = np.abs(Y)   # (F, T)
            mag_clean = np.abs(S)   # (F, T)
            mag_noise = np.maximum(mag_noisy - mag_clean, 0.0)

            irm_TF = _compute_irm(mag_clean, mag_noise)   # (T, F)
            feats_TF = mag_noisy.T.astype(np.float32)     # (T, F)

            return feats_TF, irm_TF

        elif self.mode == "pre_mixed":
            clean = load_wav(a, sr=self.sr)
            noisy = load_wav(b, sr=self.sr)
            L = min(len(clean), len(noisy))
            clean = clean[:L]
            noisy = noisy[:L]

            Y = stft(noisy)
            S = stft(clean)
            mag_noisy = np.abs(Y)
            mag_clean = np.abs(S)
            mag_noise = np.maximum(mag_noisy - mag_clean, 0.0)

            irm_TF = _compute_irm(mag_clean, mag_noise)
            feats_TF = mag_noisy.T.astype(np.float32)

            return feats_TF, irm_TF

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

