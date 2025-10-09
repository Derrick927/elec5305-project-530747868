# src/masking.py
import numpy as np
from src.stft import stft, istft

def ideal_ratio_mask(clean, noisy, eps=1e-12):
    """
    Calculate IRM = |S| / (|S| + |N|)
    Args:
        clean: clean speech waveform
        noisy: noisy speech waveform (clean + noise)
    Returns:
        Enhanced speech signal in time domain
    """
    # STFT
    S = stft(clean)
    Y = stft(noisy)

    # Estimate noise: N = Y - S
    N = Y - S

    mag_S = np.abs(S)
    mag_N = np.abs(N)

    # IRM (values in 0–1)
    M = mag_S / (mag_S + mag_N + eps)

    # Apply IRM mask with noisy phase
    mag_hat = M * np.abs(Y)
    X_hat = mag_hat * np.exp(1j * np.angle(Y))
    X_hat = istft(X_hat)
    return X_hat


def ideal_binary_mask(clean, noisy, thresh=0.5, eps=1e-12):
    """
    Calculate IBM (binary mask)
    Args:
        clean: clean speech waveform
        noisy: noisy speech waveform (clean + noise)
        thresh: threshold (default=0.5)
    Returns:
        Enhanced speech signal in time domain
    """
    S = stft(clean)
    Y = stft(noisy)
    N = Y - S

    # IBM: 1 if speech > noise*threshold, else 0
    M = (np.abs(S) > (np.abs(N) + eps) * thresh).astype(np.float32)

    mag_hat = M * np.abs(Y)
    X_hat = mag_hat * np.exp(1j * np.angle(Y))
    return istft(X_hat)

