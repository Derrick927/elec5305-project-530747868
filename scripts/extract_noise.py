import os
from pathlib import Path
import numpy as np
import soundfile as sf

SR = 16000

def load_wav(path):
    wav, sr = sf.read(path)
    if sr != SR:
        raise ValueError(f"Sample rate mismatch: {sr}")
    return wav

def save_wav(wav, path):
    sf.write(path, wav, SR)

def main():
    clean_dir = Path("data/public/train_clean")
    noisy_dir = Path("data/public/train_noisy")
    noise_out_dir = Path("data/public/noise")
    noise_out_dir.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(clean_dir.glob("*.wav"))
    noisy_files = sorted(noisy_dir.glob("*.wav"))

    assert len(clean_files) == len(noisy_files), "clean/noisy 数量不一致！"

    print(f"[Info] clean 数量: {len(clean_files)}")
    print(f"[Info] noisy 数量: {len(noisy_files)}")

    for c_path, n_path in zip(clean_files, noisy_files):
        clean = load_wav(str(c_path))
        noisy = load_wav(str(n_path))

        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]

        noise = noisy - clean

        out_path = noise_out_dir / c_path.name
        save_wav(noise, str(out_path))

    print(f"[Done] 已提取噪声，共 {len(clean_files)} 条 → 保存到 data/public/noise")

if __name__ == "__main__":
    main()
