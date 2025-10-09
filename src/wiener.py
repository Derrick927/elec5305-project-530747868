# src/wiener.py
import numpy as np
from src.stft import stft, istft

def wiener_enhance(x, sr=16000, n_init_frames=10, alpha=0.98):
    """
    Classic Wiener filtering (Decision-Directed method)

    Args:
        x: noisy speech signal (time-domain)
        sr: sample rate (default 16 kHz)
        n_init_frames: number of initial frames used to estimate noise power spectrum
        alpha: smoothing factor for decision-directed update (commonly 0.95–0.99)

    Returns:
        Enhanced speech signal (time-domain)
    """
    Y = stft(x)                  # STFT spectrum, complex matrix: F × T
    Y_mag2 = np.abs(Y) ** 2
    F, T = Y.shape

    # Initial noise power spectrum estimation
    n0 = min(n_init_frames, T)
    noise_psd = np.mean(Y_mag2[:, :n0], axis=1, keepdims=True) + 1e-12

    # Initialize enhanced spectrum
    X_hat_spec = np.zeros_like(Y, dtype=np.complex64)
    X_hat_prev_mag2 = Y_mag2[:, [0]].copy()

    for t in range(T):
        # Posterior SNR
        gamma = Y_mag2[:, [t]] / noise_psd

        # Prior SNR (decision-directed update)
        xi = alpha * (X_hat_prev_mag2 / noise_psd) + (1 - alpha) * np.maximum(gamma - 1.0, 0.0)

        # Wiener gain
        G = xi / (1.0 + xi)

        # Apply gain
        X_hat_mag = G * np.abs(Y[:, [t]])
        X_hat_prev_mag2 = X_hat_mag ** 2
        X_hat_spec[:, t] = X_hat_mag.flatten() * np.exp(1j * np.angle(Y[:, t]))

    return istft(X_hat_spec)
