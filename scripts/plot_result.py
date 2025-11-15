import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
#  PATH & IMPORT
# ==========================================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_dir = project_root / "SRC"

sys.path.insert(0, str(src_dir))

from utils import load_wav
from stft import stft, SR


# ==========================================================
# CONFIG
# ==========================================================
PLOTS_DIR = project_root / "results" / "plots"
DISPLAY_SEC = 2  # Waveform仅显示前2秒，更清晰
DB_CLIP = (-80, 0)  # Spectrogram dB范围


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Util: Align audio lengths
# ==========================================================
def align_length(*waves):
    """Cut all waves to the minimum length among them."""
    L = min(len(w) for w in waves)
    return [w[:L] for w in waves]


# ==========================================================
# Waveform plot
# ==========================================================
def plot_wave(x, title, save_path):
    x = x[: DISPLAY_SEC * SR]  # 截取前2秒
    plt.figure(figsize=(12, 3))
    plt.plot(x, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==========================================================
# Spectrogram plot
# ==========================================================
def plot_spec(x, title, save_path):
    X = stft(x)
    mag = np.abs(X) + 1e-8
    db = 20 * np.log10(mag)
    db = np.clip(db, DB_CLIP[0], DB_CLIP[1])

    plt.figure(figsize=(12, 3))
    plt.imshow(db, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="dB")
    plt.title(title)
    plt.xlabel("Time frames")
    plt.ylabel("Frequency bins")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==========================================================
# Candidate list
# ==========================================================
def candidate_list():
    return [
        ("clean",  project_root / "data" / "clean" / "example.wav"),
        ("noisy",  project_root / "data" / "noisy" / "example_noisy.wav"),
        ("subtract", project_root / "results" / "example_denoised.wav"),
        ("wiener",  project_root / "results" / "example_wiener.wav"),
        ("irm",     project_root / "results" / "example_mask_irm.wav"),
        ("dnn_single", project_root / "results" / "enhanced_from_ckpt.wav"),
        ("unet",   project_root / "results_unet_new" / "enhanced_unet.wav"),
    ]


# ==========================================================
# Main
# ==========================================================
def main():
    ensure_dir(PLOTS_DIR)

    loaded = []
    for tag, fp in candidate_list():
        if fp.exists():
            wav = load_wav(str(fp), sr=SR)
            loaded.append((tag, fp, wav))

    # 扫描 batch DNN outputs
    enh_dir = project_root / "results" / "enhanced"
    if enh_dir.exists():
        for fp in sorted(enh_dir.glob("*.wav")):
            wav = load_wav(str(fp), sr=SR)
            loaded.append((f"dnn_batch:{fp.stem}", fp, wav))

    if not loaded:
        print("[WARN] No audio found.")
        return

    # ====================================================
    # 1) Per-file Wave + Spec
    # ====================================================
    for tag, fp, wav in loaded:
        wave_png = PLOTS_DIR / f"{tag}_wave.png"
        spec_png = PLOTS_DIR / f"{tag}_spec.png"
        plot_wave(wav, f"{tag} — Waveform", wave_png)
        plot_spec(wav, f"{tag} — Spectrogram", spec_png)
        print(f"[Saved] {wave_png.name}, {spec_png.name}")

    # ====================================================
    # 2) Clean / Noisy / DNN 对比
    # ====================================================
    clean = next((x for x in loaded if x[0] == "clean"), None)
    noisy = next((x for x in loaded if x[0] == "noisy"), None)
    dnn = next((x for x in loaded if x[0] == "dnn_single"), None)

    if clean and noisy and dnn:
        _, _, c = clean
        _, _, n = noisy
        _, _, d = dnn

        c, n, d = align_length(c, n, d)

        # waveform
        plt.figure(figsize=(12, 6))
        for i, (tag, wav) in enumerate([("clean", c), ("noisy", n), ("dnn", d)], start=1):
            plt.subplot(3, 1, i)
            plt.plot(wav[:DISPLAY_SEC * SR])
            plt.title(f"{tag} — Waveform")
        plt.tight_layout()
        out = PLOTS_DIR / "compare_wave_clean_noisy_dnn.png"
        plt.savefig(out); plt.close()
        print(f"[Saved] {out.name}")

        # spec
        plt.figure(figsize=(12, 6))
        for i, (tag, wav) in enumerate([("clean", c), ("noisy", n), ("dnn", d)], start=1):
            X = stft(wav)
            mag = np.abs(X)+1e-8
            db = 20*np.log10(mag)
            db = np.clip(db, DB_CLIP[0], DB_CLIP[1])
            plt.subplot(3, 1, i)
            plt.imshow(db, aspect="auto", origin="lower", cmap="magma")
            plt.title(f"{tag} — Spectrogram")
        plt.tight_layout()
        out = PLOTS_DIR / "compare_spec_clean_noisy_dnn.png"
        plt.savefig(out); plt.close()
        print(f"[Saved] {out.name}")

    # ====================================================
    # 3) Clean / Noisy / UNet 对比
    # ====================================================
    unet = next((x for x in loaded if x[0] == "unet"), None)

    if clean and noisy and unet:
        _, _, c = clean
        _, _, n = noisy
        _, _, u = unet

        c, n, u = align_length(c, n, u)

        # waveform
        plt.figure(figsize=(12, 6))
        for i, (tag, wav) in enumerate([("clean", c), ("noisy", n), ("unet", u)], start=1):
            plt.subplot(3, 1, i)
            plt.plot(wav[:DISPLAY_SEC * SR])
            plt.title(f"{tag} — Waveform")
        plt.tight_layout()
        out = PLOTS_DIR / "compare_wave_clean_noisy_unet.png"
        plt.savefig(out); plt.close()
        print(f"[Saved] {out.name}")

        # spectrogram
        plt.figure(figsize=(12, 6))
        for i, (tag, wav) in enumerate([("clean", c), ("noisy", n), ("unet", u)], start=1):
            X = stft(wav)
            db = 20*np.log10(np.abs(X)+1e-8)
            db = np.clip(db, DB_CLIP[0], DB_CLIP[1])
            plt.subplot(3, 1, i)
            plt.imshow(db, aspect="auto", origin="lower", cmap="magma")
            plt.title(f"{tag} — Spectrogram")
        plt.tight_layout()
        out = PLOTS_DIR / "compare_spec_clean_noisy_unet.png"
        plt.savefig(out); plt.close()
        print(f"[Saved] {out.name}")

    print(f"\n[Done] All figures saved to: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
