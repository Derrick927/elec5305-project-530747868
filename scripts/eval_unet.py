import torch
import soundfile as sf
from unet_model import UNetSeparator
from utils import stft, istft


def enhance(model_path, noisy_path, out_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = UNetSeparator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    noisy, sr = sf.read(noisy_path)

    mag, phase = stft(noisy)
    mag_tensor = torch.tensor(mag).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        enhanced_mag = model(mag_tensor).squeeze().cpu().numpy()

    enhanced = istft(enhanced_mag, phase)
    sf.write(out_path, enhanced, sr)

    print(f"completed -> {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    enhance(args.ckpt, args.noisy, args.out, device=args.device)
