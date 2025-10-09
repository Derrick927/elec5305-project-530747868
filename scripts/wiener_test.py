# scripts/wiener_test.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from src.utils import load_wav, save_wav, snr_db
from src.add_noise import add_white_noise
from src.wiener import wiener_enhance

# （1) Read clean speech
clean_path = Path("data/clean/example.wav")
assert clean_path.exists(), f"File not found: {clean_path}"
x, sr = load_wav(str(clean_path))

# （2) Generate noisy speech 
noisy_dir = Path("data/noisy"); noisy_dir.mkdir(parents=True, exist_ok=True)
x_noisy = add_white_noise(x, snr_db=0)
noisy_path = noisy_dir / "example_noisy.wav"
save_wav(str(noisy_path), x_noisy, sr)

# （3) Wiener filter Enhancement
x_wiener = wiener_enhance(x_noisy, sr=sr, n_init_frames=10, alpha=0.98)

# （4) save results
out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
wiener_path = out_dir / "example_wiener.wav"
save_wav(str(wiener_path), x_wiener, sr)

# （5) print metrics
print("Noisy saved to:", noisy_path)
print("Wiener saved to:", wiener_path)
print("Noisy SNR (dB):", snr_db(x, x_noisy))
print("Wiener SNR (dB):", snr_db(x[:len(x_wiener)], x_wiener))
