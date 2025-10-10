# src/wiener.py
import numpy as np
from src.stft import stft, istft

def wiener_enhance(x, sr=16000, n_init_frames=10, alpha=0.98,
                   noise_percentile=20, g_min=0.05, noise_update_beta=0.98):
    """
    Classic Wiener filtering (Decision-Directed) with robust noise estimation.

    Args:
        x: time-domain noisy speech.
        sr: sample rate.
        n_init_frames: kept for compatibility (not used as "noise only").
        alpha: decision-directed smoothing (0.95~0.99).
        noise_percentile: % of quietest frames used to estimate noise PSD (e.g., 20).
        g_min: floor for Wiener gain to avoid musical noise (e.g., 0.05).
        noise_update_beta: noise PSD slow update on low-SNR frames (0.95~0.995).

    Returns:
        Enhanced time-domain signal.
    """
    eps = 1e-12

    # STFT
    Y = stft(x)                     # complex F x T
    Y_mag2 = (np.abs(Y) ** 2).astype(np.float64)  # power spectrum
    F, T = Y.shape

    # Robust noise PSD estimation 
    # Use the frame power of the entire segment to pick the "quietest" frames as noise statistics
    frame_power = np.mean(Y_mag2, axis=0)               # (T,)
    thr = np.percentile(frame_power, noise_percentile)  # e.g., 20% 
    idx_noise = frame_power <= thr
    if np.sum(idx_noise) < 3:  
        idx_noise[:max(3, min(T, 5))] = True

    noise_psd = np.mean(Y_mag2[:, idx_noise], axis=1, keepdims=True) + eps  # (F,1)

    # initialization
    X_hat_spec = np.zeros_like(Y, dtype=np.complex128)
    X_prev_mag2 = Y_mag2[:, [0]].copy()  # Previous frame |X|^2 estimation in decision-directed methods

    # Adaptively updated threshold (used to determine "noise-like" frames)
    low_snr_thr = np.percentile(frame_power, noise_percentile + 5)

    for t in range(T):
        Yt = Y[:, t]                        # (F,)
        Y_mag2_t = Y_mag2[:, [t]]           # (F,1)

        # Posterior SNR
        gamma = Y_mag2_t / (noise_psd + eps)

        # Prior SNR (decision-directed)
        xi = alpha * (X_prev_mag2 / (noise_psd + eps)) + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, 0.0)

        # Wiener gain with floor
        G = xi / (1.0 + xi)
        G = np.clip(G, g_min, 1.0)

        # Amplitude and Phase Synthesis
        X_mag = G * np.sqrt(Y_mag2_t)
        X_prev_mag2 = X_mag ** 2  # Update to next frame
        X_hat_spec[:, t] = (X_mag.flatten() * np.exp(1j * np.angle(Yt)))

        # Adaptive noise update
        if frame_power[t] <= low_snr_thr:
            noise_psd = noise_update_beta * noise_psd + (1.0 - noise_update_beta) * Y_mag2_t

    return istft(X_hat_spec)
