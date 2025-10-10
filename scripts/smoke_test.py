import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from src.utils import load_wav, save_wav, snr_db
from src.stft import stft, istft

in_path = Path("data/clean/example.wav")
assert in_path.exists(), f"File not found: {in_path}"

x, sr = load_wav(str(in_path))
X = stft(x)
x_hat = istft(X)

out_path = Path("results/smoke_recon.wav")
out_path.parent.mkdir(parents=True, exist_ok=True)
save_wav(str(out_path), x_hat, sr)

print("Input length:", len(x), "Output length:", len(x_hat))
print("Reconstruction SNR (dB):", snr_db(x[:len(x_hat)], x_hat))
print("Saved to:", out_path)
