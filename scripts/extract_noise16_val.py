import os
from pathlib import Path
import soundfile as sf
import numpy as np

SR = 16000

def load_wav(path):
    wav, sr = sf.read(path)
    if sr != SR:
        raise ValueError(f"Sample rate mismatch: {sr}")
    return wav

def save_wav(wav, path):
    sf.write(path, wav, SR)

def main():
    clean_dir = Path("data16/val_clean")
    noisy_dir = Path("data16/val_noisy")
    out_dir = Path("data16/val_noise")
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(clean_dir.glob("*.wav"))
    noisy_files = sorted(noisy_dir.glob("*.wav"))

    assert len(clean_files) == len(noisy_files), "val clean/noisy 数量不一致！"

    for c_path, n_path in zip(clean_files, noisy_files):
        clean = load_wav(str(c_path))
        noisy = load_wav(str(n_path))

        L = min(len(clean), len(noisy))
        noise = noisy[:L] - clean[:L]

        save_wav(noise, str(out_dir / c_path.name))

    print(f"[Done] 已生成验证集噪声 → data16/val_noise")

if __name__ == "__main__":
    main()
