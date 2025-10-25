import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from src.utils import load_wav, save_wav, snr_db
from src.add_noise import add_white_noise
from src.wiener import wiener_enhance

SR = 16000

def main():
    # (1) Read clean speech
    clean_path = Path("data/clean/example.wav")
    assert clean_path.exists(), f"File not found: {clean_path}"
    x = load_wav(str(clean_path), sr=SR)

    # (2) Generate noisy speech
    noisy_dir = Path("data/noisy"); noisy_dir.mkdir(parents=True, exist_ok=True)
    x_noisy = add_white_noise(x, snr_db=0)
    noisy_path = noisy_dir / "example_noisy.wav"
    save_wav(x_noisy, str(noisy_path), sr=SR)

    # (3) Wiener enhancement
    x_wiener = wiener_enhance(x_noisy, sr=SR, n_init_frames=10, alpha=0.98)

    # (4) Save results
    out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
    wiener_path = out_dir / "example_wiener.wav"
    save_wav(x_wiener, str(wiener_path), sr=SR)

    # (5) Print metrics
    print("Noisy saved to:", noisy_path)
    print("Wiener saved to:", wiener_path)
    print("Noisy SNR (dB):", snr_db(x[:len(x_noisy)], x_noisy))
    print("Wiener SNR (dB):", snr_db(x[:len(x_wiener)], x_wiener))

if __name__ == "__main__":
    main()

