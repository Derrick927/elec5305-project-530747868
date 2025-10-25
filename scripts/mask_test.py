import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from src.utils import load_wav, save_wav, snr_db
from src.add_noise import add_white_noise
from src.masking import ideal_ratio_mask

SR = 16000

def main():
    # (1) Read clean speech
    clean_path = Path("data/clean/example.wav")
    assert clean_path.exists(), f"File not found: {clean_path}"
    clean = load_wav(str(clean_path), sr=SR)

    # (2) Ensure a noisy file exists
    noisy_dir = Path("data/noisy"); noisy_dir.mkdir(parents=True, exist_ok=True)
    noisy_path = noisy_dir / "example_noisy.wav"
    if not noisy_path.exists():
        noisy = add_white_noise(clean, snr_db=0)
        save_wav(noisy, str(noisy_path), sr=SR)
    noisy = load_wav(str(noisy_path), sr=SR)

    # (3) IRM enhancement (use noisy phase internally)
    enh = ideal_ratio_mask(clean, noisy)

    # (4) Save
    out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "example_mask_irm.wav"
    save_wav(enh, str(out_path), sr=SR)

    # (5) Metrics
    print("Noisy SNR (dB):", snr_db(clean[:len(noisy)], noisy))
    print("IRM   SNR (dB):", snr_db(clean[:len(enh)], enh))
    print("Saved IRM to  :", out_path)

if __name__ == "__main__":
    main()
