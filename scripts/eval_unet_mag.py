import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch

# 保证能 import 到 src
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.unet_model import UNet
from src.utils import load_wav, save_wav
from src.stft import stft, istft, N_FFT


def parse_args():
    ap = argparse.ArgumentParser(description="Use UNet (mag->mag) to enhance a noisy wav.")
    ap.add_argument("--ckpt", type=str, required=True, help="unet_best.pt 路径")
    ap.add_argument("--noisy", type=str, required=True, help="noisy wav 路径")
    ap.add_argument("--out", type=str, required=True, help="输出增强后 wav 路径")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return ap.parse_args()


def main():
    args = parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Info] Device: {device}")

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # 读取 checkpoint
    obj = torch.load(ckpt_path, map_location=device)
    state_dict = obj["model"] if "model" in obj else obj
    freq_bins = obj.get("freq_bins", N_FFT // 2 + 1)
    print(f"[Info] freq_bins from ckpt = {freq_bins}")

    # 构建 UNet
    model = UNet(n_channels=freq_bins, n_classes=freq_bins).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 读取 noisy 波形
    noisy = load_wav(args.noisy, sr=args.sr)   # np.ndarray, [-1,1]

    # STFT
    Y = stft(noisy)                # (F, T) complex
    mag = np.abs(Y)                # (F, T)
    phase = np.angle(Y)            # (F, T)

    # 送入 UNet：变成 [B, C, T]，这里 C=F
    mag_tensor = torch.from_numpy(mag).unsqueeze(0).to(device)  # [1, F, T]
    mag_tensor = mag_tensor.float()

    with torch.no_grad():
        enh_mag = model(mag_tensor)        # [1, F, T']
        enh_mag = enh_mag.squeeze(0).cpu().numpy()  # [F, T']

    # 防止负数
    enh_mag = np.maximum(enh_mag, 0.0)

    # 为了跟原始相位对齐，按时间维裁到最短长度
    T_min = min(enh_mag.shape[1], phase.shape[1])
    enh_mag = enh_mag[:, :T_min]
    phase = phase[:, :T_min]

    # 用 noisy 相位重构
    S_hat = enh_mag * np.exp(1j * phase)
    enhanced = istft(S_hat).astype(np.float32)

    # 简单防 clipping
    mx = float(np.max(np.abs(enhanced)) + 1e-12)
    if mx > 1.0:
        enhanced = enhanced / mx

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(enhanced, str(out_path), sr=args.sr)
    print(f"[Done] Enhanced wav saved to: {out_path}")


if __name__ == "__main__":
    main()
