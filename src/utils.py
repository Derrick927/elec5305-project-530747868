import numpy as np
import soundfile as sf
import librosa

def load_wav(path: str, sr: int = None):
    """
    Load audio file.
    Args:
        path: file path
        sr: if not None, resample to this sampling rate
    Returns:
        np.ndarray, shape (T,)
    """
    wav, file_sr = sf.read(path)
    # Convert stereo to mono
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr is not None and file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return wav

def save_wav(wav: np.ndarray, path: str, sr: int = 16000):
    """
    Save waveform to disk.
    """
    sf.write(path, wav, sr)

def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Compute SNR (dB) between clean and noisy signals.
    Args:
        clean: np.ndarray
        noisy: np.ndarray
    Returns:
        float (dB)
    """
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    L = min(len(clean), len(noisy))
    clean = clean[:L]
    noisy = noisy[:L]
    noise = noisy - clean
    p_clean = np.sum(clean ** 2) + 1e-12
    p_noise = np.sum(noise ** 2) + 1e-12
    return 10.0 * np.log10(p_clean / p_noise)
