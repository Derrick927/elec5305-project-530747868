import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_wav
from src.stft import stft

# === 1. 读取音频文件 ===
clean, sr = load_wav("data/clean/example.wav")         # 原始干净语音
noisy, _  = load_wav("data/noisy/example.wav")         # 带噪语音
enhanced, _ = load_wav("results/example_dnn.wav")      # 增强后的语音

# === 2. 画波形图 (Waveform) ===
plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
plt.plot(clean, color="blue")
plt.title("Clean Speech (Waveform)")
plt.subplot(3,1,2)
plt.plot(noisy, color="red")
plt.title("Noisy Speech (Waveform)")
plt.subplot(3,1,3)
plt.plot(enhanced, color="green")
plt.title("Enhanced Speech (Waveform)")
plt.tight_layout()
plt.show()

# === 3. 画频谱图 (Spectrogram) ===
def plot_spectrogram(signal, sr, title):
    X = stft(signal)
    plt.figure(figsize=(10,4))
    plt.imshow(20*np.log10(np.abs(X)+1e-8), 
               aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="dB")
    plt.title(title)
    plt.xlabel("Time frames")
    plt.ylabel("Frequency bins")
    plt.show()

plot_spectrogram(clean, sr, "Clean Speech (Spectrogram)")
plot_spectrogram(noisy, sr, "Noisy Speech (Spectrogram)")
plot_spectrogram(enhanced, sr, "Enhanced Speech (Spectrogram)")
