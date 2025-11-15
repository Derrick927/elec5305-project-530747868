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

# ===================== 把 SRC 加进 sys.path =====================
current_dir = os.path.dirname(os.path.abspath(__file__))   # ...\speech-enhance-mask\scripts
project_root = os.path.dirname(current_dir)                # ...\speech-enhance-mask
src_dir = os.path.join(project_root, "SRC")

sys.path.insert(0, src_dir)
print("Added SRC to sys.path:", src_dir)

# ===================== 导入你自己的模块 =========================
from stft import stft, istft, N_FFT, SR
from utils import load_wav, save_wav, snr_db
from dnn_mask import MaskNet

# 可选依赖：PESQ / STOI
try:
    from pesq import pesq as pesq_api
except Exception:
    pesq_api = None

try:
    from pystoi.stoi import stoi as stoi_api
except Exception:
    stoi_api = None


# ===================== 工具函数：对齐长度 ========================
def _align_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = min(len(a), len(b))
    return a[:L], b[:L]


# ===================== 评价一对语音 ============================
def eval_pair(clean_path: str, test_path: str, sr: int = SR) -> Dict[str, float]:
    """
    返回:
        {
          "snr_db": float,
          "pesq_wb": float or nan,
          "stoi": float or nan
        }
    """
    # 加载并对齐长度
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
            # 常见调用：pesq_api(sr, ref, deg, mode="wb")
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


# ===================== 用模型做增强 ===========================
def enhance_with_model(model: MaskNet,
                       noisy_wav: np.ndarray,
                       device: torch.device = torch.device("cpu")) -> np.ndarray:
    """
    输入：时域 noisy 波形
    输出：时域 enhanced 波形
    """
    # STFT：返回复数谱 Y (F, T)
    Y = stft(noisy_wav)
    mag_noisy = np.abs(Y)      # (F, T)
    phase_noisy = np.angle(Y)  # (F, T)

    # 转成 (T, F)，和训练时 PairDataset 一致
    feats_TF = mag_noisy.T.astype(np.float32)  # (T, F)
    T, Fdim = feats_TF.shape

    # 前向推理
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(feats_TF).unsqueeze(0).to(device)  # (1, T, F)
        pred_TF = model(inp).squeeze(0).cpu().numpy()             # (T, F)

    # 转回 (F, T)
    mask_FT = pred_TF.T

    # 先裁剪到 [0, 1]，防止异常值
    mask_FT = np.clip(mask_FT, 0.0, 1.0)

    # ====== 关键修改：让掩蔽更“保守”，避免过度削弱语音 ======
    # 原始 mask_FT 在 [0,1]，直接用时有可能把语音成分压太狠，
    # 这里做一个线性平滑 + 下限裁剪：
    #   1) 0.7 * M + 0.3   -> 范围变成 [0.3,1.0]
    #   2) 再 clip 一次，完全限定在 [0.3,1.0]
    mask_FT = 0.7 * mask_FT + 0.3
    mask_FT = np.clip(mask_FT, 0.3, 1.0)
    # ========================================================

    # 应用掩蔽
    enh_mag = mag_noisy * mask_FT  # (F, T)

    # 复数谱重建 + ISTFT
    enh_S = enh_mag * np.exp(1j * phase_noisy)
    enh_wav = istft(enh_S).astype(np.float32)

    # 安全归一化，避免溢出
    mx = float(np.max(np.abs(enh_wav)) + 1e-12)
    if mx > 1.0:
        enh_wav = enh_wav / mx

    return enh_wav


# ===================== 主函数 ===========================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DNN MaskNet: enhance noisy wav and compute SNR/PESQ/STOI."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="masknet_best.pt 路径")
    parser.add_argument("--clean", type=str, required=True, help="干净语音路径")
    parser.add_argument("--noisy", type=str, required=True, help="带噪语音路径")
    parser.add_argument("--outdir", type=str, default="results", help="输出目录")
    parser.add_argument("--sr", type=int, default=SR, help="采样率 (默认用 stft.SR)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # 设备
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

    # ----------------- 加载模型 -----------------
    ckpt = torch.load(args.ckpt, map_location=device)
    # 兼容两种格式：直接 state_dict 或 {"state_dict": ..., ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model = MaskNet(N_FFT // 2 + 1)
    model.load_state_dict(state_dict, strict=True)
    print(f"[Info] Loaded checkpoint: {args.ckpt}")

    # ----------------- I/O 波形 -----------------
    clean = load_wav(args.clean, sr=args.sr)
    noisy = load_wav(args.noisy, sr=args.sr)

    # ----------------- 增强 ---------------------
    enhanced = enhance_with_model(model, noisy, device=device)
    save_wav(enhanced, str(enh_path), sr=args.sr)
    print(f"[Info] Saved enhanced wav -> {enh_path}")

    # ----------------- 评价指标 -----------------
    m_noisy = eval_pair(args.clean, args.noisy, sr=args.sr)
    m_enh   = eval_pair(args.clean, str(enh_path), sr=args.sr)

    # ----------------- 写 JSON ------------------
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

    # ----------------- 写 CSV -------------------
    with open(report_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "snr_db", "pesq_wb", "stoi"])
        w.writerow(["noisy", m_noisy.get("snr_db"), m_noisy.get("pesq_wb"), m_noisy.get("stoi")])
        w.writerow(["enhanced", m_enh.get("snr_db"), m_enh.get("pesq_wb"), m_enh.get("stoi")])
    print(f"[Info] Saved report csv  -> {report_csv}")
    print("[Done] Evaluation finished.")


if __name__ == "__main__":
    main()
