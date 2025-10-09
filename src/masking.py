# src/masking.py
import numpy as np
from src.stft import stft, istft

def ideal_ratio_mask(clean, noisy, eps=1e-12):
    """
    计算 IRM = |S| / (|S| + |N|)
    参数:
      clean: 干净语音波形
      noisy: 带噪语音波形 (clean + noise)
    返回:
      掩蔽后的增强语音时域信号
    """
    # STFT
    S = stft(clean)
    Y = stft(noisy)
    # 估计噪声谱: N = Y - S
    N = Y - S

    mag_S = np.abs(S)
    mag_N = np.abs(N)

    # IRM (0~1)
    M = mag_S / (mag_S + mag_N + eps)

    # 应用在带噪幅度上（用带噪相位）
    mag_hat = M * np.abs(Y)
    X_hat = mag_hat * np.exp(1j * np.angle(Y))
    x_hat = istft(X_hat)
    return x_hat

def ideal_binary_mask(clean, noisy, thresh=0.5, eps=1e-12):
    """
    可选：理想二值掩蔽 IBM（更激进，听感可能更脆，但对比用）
    """
    S = stft(clean); Y = stft(noisy); N = Y - S
    M = (np.abs(S) > (np.abs(N) + eps)).astype(np.float32)  # True=1, False=0
    mag_hat = M * np.abs(Y)
    X_hat = mag_hat * np.exp(1j * np.angle(Y))
    return istft(X_hat)
