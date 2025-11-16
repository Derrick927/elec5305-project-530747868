import os
import argparse
import numpy as np
import soundfile as sf
import torch
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.stft import stft, istft, SR
from src.utils import load_wav
from src.unet_model import UNet
from src.eval_metrics import compute_snr, compute_stoi, compute_pesq_wb


def resolve_device(flag: str) -> torch.device:
    """Resolve 'auto', 'cuda', 'cpu' to torch.device"""
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--clean", type=str, required=True)
    parser.add_argument("--noisy", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results_unet")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[Info] Device: {device}")

    print("[Info] Building UNet model...")
    model = UNet(n_channels=513, n_classes=513).to(device)

    print("[Info] Loading checkpoint:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        print("[Info] Detected full checkpoint dict.")
        state_dict = ckpt["model"]
    else:
        print("[Info] Detected pure state_dict.")
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("[Info] Model loaded successfully.")

    clean_wav = load_wav(args.clean, sr=SR)
    noisy_wav = load_wav(args.noisy, sr=SR)

    Y = stft(noisy_wav)
    mag_noisy = np.abs(Y)
    phase_noisy = np.angle(Y)

    inp = mag_noisy[np.newaxis, :, :]
    inp = torch.from_numpy(inp).float().to(device)

    with torch.no_grad():
        enh_mag_raw = model(inp).cpu().numpy()[0]

    T_stft = mag_noisy.shape[1]
    T_model = enh_mag_raw.shape[1]

    if T_model < T_stft:
        pad = T_stft - T_model
        enh_mag = np.pad(enh_mag_raw, ((0, 0), (0, pad)), mode="constant")
    elif T_model > T_stft:
        enh_mag = enh_mag_raw[:, :T_stft]
    else:
        enh_mag = enh_mag_raw

    eps = 1e-8
    mask_FT = enh_mag / (mag_noisy + eps)

    mask_FT = np.clip(mask_FT, 0.0, 2.0)
    mask_FT = 0.7 * mask_FT + 0.3
    mask_FT = np.clip(mask_FT, 0.3, 1.2)

    final_mag = mag_noisy * mask_FT

    enh_complex = final_mag * np.exp(1j * phase_noisy)
    enh_wav = istft(enh_complex).astype(np.float32)

    mx = np.max(np.abs(enh_wav))
    if mx > 1.0:
        enh_wav /= mx

    out_wav_path = os.path.join(args.outdir, "enhanced_unet.wav")
    sf.write(out_wav_path, enh_wav, SR)
    print("[Info] Enhanced saved to:", out_wav_path)

    L = min(len(clean_wav), len(enh_wav), len(noisy_wav))
    clean_wav = clean_wav[:L]
    noisy_wav = noisy_wav[:L]
    enh_wav   = enh_wav[:L]

    snr_noisy   = compute_snr(clean_wav, noisy_wav)
    stoi_noisy  = compute_stoi(clean_wav, noisy_wav, SR)
    pesq_noisy  = compute_pesq_wb(clean_wav, noisy_wav, SR)

    snr_enh     = compute_snr(clean_wav, enh_wav)
    stoi_enh    = compute_stoi(clean_wav, enh_wav, SR)
    pesq_enh    = compute_pesq_wb(clean_wav, enh_wav, SR)

    print("\n========== UNet Evaluation ==========")
    print(f"Noisy     | SNR={snr_noisy:.4f}, PESQ={pesq_noisy:.4f}, STOI={stoi_noisy:.4f}")
    print(f"Enhanced  | SNR={snr_enh:.4f}, PESQ={pesq_enh:.4f}, STOI={stoi_enh:.4f}")

    csv_path = os.path.join(args.outdir, "metrics_unet.csv")
    with open(csv_path, "w") as f:
        f.write("type,snr_db,pesq_wb,stoi\n")
        f.write(f"noisy,{snr_noisy},{pesq_noisy},{stoi_noisy}\n")
        f.write(f"enhanced,{snr_enh},{pesq_enh},{stoi_enh}\n")

    print("[Info] Metrics saved:", csv_path)


if __name__ == "__main__":
    main()
