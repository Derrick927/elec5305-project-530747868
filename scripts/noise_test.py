import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import numpy as np
from src.utils import load_wav, save_wav, snr_db
from src.stft import stft, istft
from src.add_noise import add_white_noise

SR = 16000

def main():
    # (1) Read clean speech
    in_path = Path("data/clean/example.wav")
    assert in_path.exists(), f"File not found: {in_path}"

    x = load_wav(str(in_path), sr=SR)  # load_wav returns waveform only

    # (2) Generate noisy speech at 0 dB SNR
    noisy = add_white_noise(x, snr_db=0)
    noisy_path = Path("data/noisy/example_noisy.wav")
    noisy_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(noisy, str(noisy_path), sr=SR)  # (wav, path, sr)

    # (3) Simple spectral subtraction (mean of first 10 frames as noise estimate)
    X = stft(noisy)                  # (F, T) complex
    mag, phase = np.abs(X), np.angle(X)
    n_init = min(10, X.shape[1])     # guard for very short files
    noise_mag = np.mean(mag[:, :n_init], axis=1, keepdims=True)
    mag_denoised = np.maximum(mag - noise_mag, 0.0)

    X_denoised = mag_denoised * np.exp(1j * phase)
    x_denoised = istft(X_denoised)

    # (4) Save results
    out_path = Path("results/example_denoised.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(x_denoised, str(out_path), sr=SR)

    # (5) Print metrics
    print("Noisy saved to:", noisy_path)
    print("Denoised saved to:", out_path)
    print("Noisy SNR (dB):", snr_db(x[:len(noisy)], noisy))
    print("Denoised SNR (dB):", snr_db(x[:len(x_denoised)], x_denoised))

if __name__ == "__main__":
    main()
