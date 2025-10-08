import numpy as np
import librosa

SR = 16000
WIN_MS = 25
HOP_MS = 10
N_FFT = 1024
WIN_LEN = int(SR * WIN_MS / 1000)
HOP_LEN = int(SR * HOP_MS / 1000)
WINDOW = "hann"

def stft(x: np.ndarray):
    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP_LEN,
                        win_length=WIN_LEN, window=WINDOW, center=True)

def istft(S: np.ndarray):
    return librosa.istft(S, hop_length=HOP_LEN,
                         win_length=WIN_LEN, window=WINDOW, center=True, length=None)
