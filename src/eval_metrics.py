from __future__ import annotations
from typing import Dict
import numpy as np

# 修复导入
from utils import load_wav, snr_db

# Optional deps
try:
    from pesq import pesq as pesq_api
except Exception:
    pesq_api = None

try:
    from pystoi.stoi import stoi as stoi_api
except Exception:
    stoi_api = None


# -------------------------------
# 对齐音频长度
# -------------------------------
def _align_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = min(len(a), len(b))
    return a[:L], b[:L]


# -------------------------------
# 单独的指标计算函数（eval_unet.py 需要）
# -------------------------------
def compute_snr(clean: np.ndarray, test: np.ndarray) -> float:
    """Compute SNR between clean and test signals"""
    try:
        return float(snr_db(clean, test))
    except Exception:
        return float("nan")


def compute_pesq_wb(clean: np.ndarray, test: np.ndarray, sr: int = 16000) -> float:
    """Compute PESQ-WB"""
    if pesq_api is None:
        return float("nan")
    try:
        return float(pesq_api(sr, clean, test, "wb"))
    except TypeError:
        try:
            return float(pesq_api(clean, test, sr))
        except Exception:
            return float("nan")
    except Exception:
        return float("nan")


def compute_stoi(clean: np.ndarray, test: np.ndarray, sr: int = 16000) -> float:
    """Compute STOI"""
    if stoi_api is None:
        return float("nan")
    try:
        return float(stoi_api(clean, test, sr, extended=False))
    except Exception:
        return float("nan")


# -------------------------------
# 原版复合评估接口（保持不变）
# -------------------------------
def eval_pair(clean_path: str, test_path: str, sr: int = 16000) -> Dict[str, float]:
    """
    Evaluate a pair of waveforms (clean vs. test).
    Returns a dict with:
        snr_db, pesq_wb, stoi
    """
    # Load
    c = load_wav(clean_path, sr=sr)
    t = load_wav(test_path, sr=sr)
    c, t = _align_length(c, t)

    snr = compute_snr(c, t)
    pesq_wb = compute_pesq_wb(c, t, sr)
    stoi_val = compute_stoi(c, t, sr)

    return {"snr_db": snr, "pesq_wb": pesq_wb, "stoi": stoi_val}
