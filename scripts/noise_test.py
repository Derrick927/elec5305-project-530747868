import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import numpy as np
from src.utils import load_wav, save_wav, snr_db
from src.stft import stft, istft
from src.add_noise import add_white_noise

# 输入干净语音
in_path = Path("data/clean/example.wav")
assert in_path.exists(), f"File not found: {in_path}"

x, sr = load_wav(str(in_path))

# Step1: 生成带噪语音
x_noisy = add_white_noise(x, snr_db=0)  # SNR=0dB，强噪声
save_wav("data/noisy/example_noisy.wav", x_noisy, sr)

# Step2: 进行谱减法降噪
X = stft(x_noisy)
mag, phase = np.abs(X), np.angle(X)

# 假设噪声谱为前10帧的平均（简化版）
noise_mag = np.mean(mag[:, :10], axis=1, keepdims=True)
mag_denoised = np.maximum(mag - noise_mag, 0.0)

X_denoised = mag_denoised * np.exp(1j * phase)
x_denoised = istft(X_denoised)

# 保存结果
out_path = Path("results/example_denoised.wav")
out_path.parent.mkdir(parents=True, exist_ok=True)
save_wav(str(out_path), x_denoised, sr)

print("Noisy saved to: data/noisy/example_noisy.wav")
print("Denoised saved to:", out_path)
print("Noisy SNR (dB):", snr_db(x, x_noisy))
print("Denoised SNR (dB):", snr_db(x[:len(x_denoised)], x_denoised))
