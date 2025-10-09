# src/wiener.py
import numpy as np
from src.stft import stft, istft

def wiener_enhance(x, sr=16000, n_init_frames=10, alpha=0.98):
    """
    经典 Wiener 滤波 (Decision-Directed)
    x: 带噪语音信号
    n_init_frames: 用前N帧估计噪声功率谱
    alpha: 平滑系数 (0.95~0.99 常用)
    """
    Y = stft(x)                    # 复数谱: F x T
    Y_mag2 = np.abs(Y) ** 2
    F, T = Y.shape

    # 初始噪声估计
    n0 = min(n_init_frames, T)
    noise_psd = np.mean(Y_mag2[:, :n0], axis=1, keepdims=True) + 1e-12

    # 保存增强结果
    X_hat_spec = np.zeros_like(Y, dtype=np.complex64)
    X_hat_prev_mag2 = Y_mag2[:, [0]].copy()

    for t in range(T):
        gamma = Y_mag2[:, [t]] / noise_psd                   # 后验 SNR
        xi = alpha * (X_hat_prev_mag2 / noise_psd) + (1 - alpha) * np.maximum(gamma - 1.0, 0.0)
        G = xi / (1.0 + xi)                                  # Wiener 增益
        X_hat_mag = G * np.abs(Y[:, [t]])
        X_hat_prev_mag2 = X_hat_mag ** 2
        X_hat_spec[:, t] = X_hat_mag.flatten() * np.exp(1j * np.angle(Y[:, t]))

    return istft(X_hat_spec)
