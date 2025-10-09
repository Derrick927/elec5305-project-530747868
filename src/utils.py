import numpy as np
import soundfile as sf
import librosa

SR = 16000

def load_wav(path: str, sr: int = SR):
    x, orig_sr = sf.read(path, always_2d=False)
    if x.ndim > 1:  # Stereo to mono
        x = np.mean(x, axis=1)
    if orig_sr != sr:
        x = librosa.resample(x, orig_sr=orig_sr, target_sr=sr)
    return x.astype(np.float32), sr

def save_wav(path: str, x: np.ndarray, sr: int = SR):
    x = np.clip(x, -1.0, 1.0)
    sf.write(path, x, sr)

def snr_db(ref: np.ndarray, est: np.ndarray):
    num = np.sum(ref**2) + 1e-12
    den = np.sum((ref - est)**2) + 1e-12
    return 10.0 * np.log10(num / den)
