#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_result.py — Auto-save all figures (waveform + spectrogram) for available audios.

It looks for:
  - data/clean/example.wav
  - data/noisy/example_noisy.wav
  - results/example_wiener.wav
  - results/example_mask_irm.wav
  - results/example_denoised.wav        (spectral subtraction)
  - results/enhanced_from_ckpt.wav      (DNN single-file eval)
  - results/enhanced/*.wav              (batch DNN eval outputs)

Figures are saved under: results/plots/
"""

import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

# Make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_wav
from src.stft import stft

SR = 16000
PLOTS_DIR = Path("results/plots")

CANDIDATES = [
    ("clean",   Path("data/clean/example.wav")),
    ("noisy",   Path("data/noisy/example_noisy.wav")),
    ("wiener",  Path("results/example_wiener.wav")),
    ("mask_irm",Path("results/example_mask_irm.wav")),
    ("subtract",Path("results/example_denoised.wav")),
    ("dnn_single", Path("results/enhanced_from_ckpt.wav")),
]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_wave(x: np.ndarray, title: str, save_path: Path):
    plt.figure(figsize=(12, 3))
    plt.plot(x)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spec(x: np.ndarray, title: str, save_path: Path):
    X = stft(x)
    db = 20 * np.log10(np.abs(X) + 1e-8)
    plt.figure(figsize=(12, 3))
    plt.imshow(db, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="dB")
    plt.title(title)
    plt.xlabel("Time frames")
    plt.ylabel("Frequency bins")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    ensure_dir(PLOTS_DIR)

    # 1) load known candidates
    loaded = []
    for tag, p in CANDIDATES:
        if p.exists():
            wav = load_wav(str(p), sr=SR)
            loaded.append((tag, p, wav))

    # 2) load batch DNN outputs under results/enhanced/*.wav
    enh_dir = Path("results/enhanced")
    if enh_dir.exists():
        for fp in sorted(enh_dir.glob("*.wav")):
            wav = load_wav(str(fp), sr=SR)
            loaded.append((f"dnn_batch:{fp.stem}", fp, wav))

    if not loaded:
        print("[Warn] No audio found to plot. Please run demos/evaluations first.")
        return

    # 3) save per-file plots
    for tag, p, x in loaded:
        wave_png = PLOTS_DIR / f"{tag}_wave.png"
        spec_png = PLOTS_DIR / f"{tag}_spec.png"
        plot_wave(x, f"{tag} — Waveform", wave_png)
        plot_spec(x, f"{tag} — Spectrogram", spec_png)
        print(f"[Saved] {wave_png.name}, {spec_png.name}")

    # 4) if both clean & noisy exist, draw a compact comparison panel
    clean = next((x for x in loaded if x[0] == "clean"), None)
    noisy = next((x for x in loaded if x[0] == "noisy"), None)
    dnn_single = next((x for x in loaded if x[0] == "dnn_single"), None)
    if clean and noisy and dnn_single:
        # waveform comparison
        plt.figure(figsize=(12, 6))
        for i, (tag, _, wav) in enumerate([clean, noisy, dnn_single], 1):
            plt.subplot(3, 1, i); plt.plot(wav); plt.title(f"{tag} — Waveform")
        plt.tight_layout()
        out = PLOTS_DIR / "compare_wave_clean_noisy_dnn.png"
        plt.savefig(out); plt.close(); print(f"[Saved] {out.name}")

        # spectrogram comparison
        plt.figure(figsize=(12, 6))
        for i, (tag, _, wav) in enumerate([clean, noisy, dnn_single], 1):
            X = stft(wav); db = 20*np.log10(np.abs(X)+1e-8)
            plt.subplot(3, 1, i); plt.imshow(db, aspect="auto", origin="lower", cmap="magma")
            plt.title(f"{tag} — Spectrogram")
        plt.tight_layout()
        out = PLOTS_DIR / "compare_spec_clean_noisy_dnn.png"
        plt.savefig(out); plt.close(); print(f"[Saved] {out.name}")

    print(f"[Done] All figures saved under: {PLOTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
