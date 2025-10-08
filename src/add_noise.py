import numpy as np
from pathlib import Path
from src.utils import load_wav, save_wav

def add_white_noise(x, snr_db=5):
    # 计算信号功率
    power_signal = np.mean(x**2)
    # 计算噪声功率
    power_noise = power_signal / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(power_noise), size=len(x))
    return x + noise

if __name__ == "__main__":
    in_path = Path("data/clean/example.wav")
    out_path = Path("data/noisy/example_noisy.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x, sr = load_wav(str(in_path))
    x_noisy = add_white_noise(x, snr_db=0)   # 这里 0 dB 噪声，噪声很大
    save_wav(str(out_path), x_noisy, sr)

    print("Noisy file saved to:", out_path)
