import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from src.stft import stft, istft, SR
from src.utils import snr_db

def _read_mono_resample(path: str, target_sr: int) -> np.ndarray:
    """
    Read wav from 'path', convert to mono, and resample to target_sr if needed.
    Returns waveform only (np.ndarray).
    """
    x, sr = sf.read(path, always_2d=False)
    if x.ndim == 2:  # stereo -> mono
        x = np.mean(x, axis=1)
    x = x.astype(np.float32)
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    return x

def main():
    """
    Smoke test:
      - Read data/clean/example.wav (any sr/channels)
      - Resample to SR, STFT -> iSTFT
      - Align lengths, compute SNR
      - Save to results/smoke_recon.wav
    """
    in_path = Path("data/clean/example.wav")
    assert in_path.exists(), (
        f"Missing file: {in_path}\n"
        "Please put a clean wav at data/clean/example.wav (any sr; will be resampled)."
    )

    # Read & normalize
    x = _read_mono_resample(str(in_path), SR)

    # STFT -> iSTFT
    X = stft(x)
    x_rec = istft(X)

    # Align lengths
    if len(x_rec) > len(x):
        x_rec = x_rec[:len(x)]
    elif len(x_rec) < len(x):
        pad = np.zeros(len(x) - len(x_rec), dtype=x_rec.dtype)
        x_rec = np.concatenate([x_rec, pad], axis=0)

    # SNR
    snr = snr_db(x, x_rec)
    print(f"[SmokeTest] Reconstruct SNR: {snr:.4f} dB")

    # Save
    out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "smoke_recon.wav"
    x_rec = np.clip(x_rec, -1.0, 1.0).astype(np.float32)
    sf.write(str(out_wav), x_rec, SR)
    print(f"[SmokeTest] Saved: {out_wav.resolve()}")

if __name__ == "__main__":
    main()
