import os
import argparse
import numpy as np
import soundfile as sf
import torch
import sys

# ---------------------------------------------------------------------
# 把 SRC 加进 sys.path
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "SRC")
sys.path.insert(0, src_dir)

from stft import stft, istft, SR
from utils import load_wav
from unet_model import UNet
from eval_metrics import compute_snr, compute_stoi, compute_pesq_wb


# =====================================================================
#   MAIN
# =====================================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--clean", type=str, required=True)
    parser.add_argument("--noisy", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results_unet")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------
    # 构建模型
    # ------------------------------------------------------------
    print("[Info] Building UNet model...")
    model = UNet(n_channels=513, n_classes=513).to(args.device)

    # ------------------------------------------------------------
    # 加载权重
    # ------------------------------------------------------------
    print("[Info] Loading checkpoint:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=args.device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        print("[Info] Detected full checkpoint dict.")
        state_dict = ckpt["model"]
    else:
        print("[Info] Detected pure state_dict.")
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("[Info] Model loaded successfully.")

    # ------------------------------------------------------------
    # 读取音频
    # ------------------------------------------------------------
    clean_wav = load_wav(args.clean, sr=SR)
    noisy_wav = load_wav(args.noisy, sr=SR)

    # ------------------------------------------------------------
    # STFT
    # ------------------------------------------------------------
    Y = stft(noisy_wav)           # (F=513, T)
    mag_noisy = np.abs(Y)         # (F, T)
    phase_noisy = np.angle(Y)     # (F, T)

    # ------------------------------------------------------------
    # 准备 UNet 输入：形状 (1, 513, T)
    # ------------------------------------------------------------
    inp = mag_noisy[np.newaxis, :, :]           # (1, 513, T)
    inp = torch.from_numpy(inp).float().to(args.device)

    # ------------------------------------------------------------
    # UNet 推理，得到“估计幅度谱”
    # ------------------------------------------------------------
    with torch.no_grad():
        enh_mag_raw = model(inp).cpu().numpy()[0]    # (513, T_model)

    # ------------------------------------------------------------
    # 对齐 UNet 输出时间维到 STFT 帧数
    # ------------------------------------------------------------
    T_stft = mag_noisy.shape[1]
    T_model = enh_mag_raw.shape[1]

    if T_model < T_stft:
        pad = T_stft - T_model
        enh_mag = np.pad(enh_mag_raw, ((0, 0), (0, pad)), mode="constant")
    elif T_model > T_stft:
        enh_mag = enh_mag_raw[:, :T_stft]
    else:
        enh_mag = enh_mag_raw

    # ------------------------------------------------------------
    # 将 UNet 输出转成“隐式 mask”，并做保守平滑
    #   mask = enh_mag / mag_noisy
    #   然后像 DNN 那样做下限裁剪和平滑，避免过度削弱语音
    # ------------------------------------------------------------
    eps = 1e-8
    mask_FT = enh_mag / (mag_noisy + eps)     # 可能 <0 或 >1

    # 先简单裁剪到 [0, 2]，避免爆炸
    mask_FT = np.clip(mask_FT, 0.0, 2.0)

    # 类似 DNN 的“保守掩蔽”：0.7 * M + 0.3，把范围往 [0.3, 1.7] 压
    mask_FT = 0.7 * mask_FT + 0.3

    # 最终再限制在 [0.3, 1.2]，既能适度降噪，又不会把语音削太狠
    mask_FT = np.clip(mask_FT, 0.3, 1.2)

    # 最终的增强幅度谱
    final_mag = mag_noisy * mask_FT           # (F, T)

    # ------------------------------------------------------------
    # ISTFT
    # ------------------------------------------------------------
    enh_complex = final_mag * np.exp(1j * phase_noisy)
    enh_wav = istft(enh_complex).astype(np.float32)

    # 归一化，防止溢出
    mx = np.max(np.abs(enh_wav))
    if mx > 1.0:
        enh_wav /= mx

    # 保存增强后的语音
    out_wav_path = os.path.join(args.outdir, "enhanced_unet.wav")
    sf.write(out_wav_path, enh_wav, SR)
    print("[Info] Enhanced saved to:", out_wav_path)

    # ------------------------------------------------------------
    # 对齐三段波形长度（防止 STOI 报 nan）
    # ------------------------------------------------------------
    L = min(len(clean_wav), len(enh_wav), len(noisy_wav))
    clean_wav = clean_wav[:L]
    noisy_wav = noisy_wav[:L]
    enh_wav   = enh_wav[:L]

    # ------------------------------------------------------------
    # 计算指标
    # ------------------------------------------------------------
    snr_noisy   = compute_snr(clean_wav, noisy_wav)
    stoi_noisy  = compute_stoi(clean_wav, noisy_wav, SR)
    pesq_noisy  = compute_pesq_wb(clean_wav, noisy_wav, SR)

    snr_enh     = compute_snr(clean_wav, enh_wav)
    stoi_enh    = compute_stoi(clean_wav, enh_wav, SR)
    pesq_enh    = compute_pesq_wb(clean_wav, enh_wav, SR)

    print("\n========== UNet Evaluation ==========")
    print(f"Noisy     | SNR={snr_noisy:.4f}, PESQ={pesq_noisy:.4f}, STOI={stoi_noisy:.4f}")
    print(f"Enhanced  | SNR={snr_enh:.4f}, PESQ={pesq_enh:.4f}, STOI={stoi_enh:.4f}")

    # 保存 CSV
    csv_path = os.path.join(args.outdir, "metrics_unet.csv")
    with open(csv_path, "w") as f:
        f.write("type,snr_db,pesq_wb,stoi\n")
        f.write(f"noisy,{snr_noisy},{pesq_noisy},{stoi_noisy}\n")
        f.write(f"enhanced,{snr_enh},{pesq_enh},{stoi_enh}\n")

    print("[Info] Metrics saved:", csv_path)


if __name__ == "__main__":
    main()
