# scripts/train_mask.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.utils import load_wav, save_wav
from src.stft import stft, istft
from src.add_noise import add_white_noise
from src.dnn_mask import MaskNet

# equipment
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# （1) Read clean speech and generate 0 dB noisy speech
clean, sr = load_wav("data/clean/example.wav")
noisy = add_white_noise(clean, snr_db=0)

# （2) STFT -> Amplitude and Phase
def wav_to_mag_phase(x: np.ndarray):
    X = stft(x)                              # (F, T)
    return np.abs(X).astype(np.float32), np.angle(X)

mag_c, _     = wav_to_mag_phase(clean)       # (F, T)
mag_n, pha_n = wav_to_mag_phase(noisy)       # (F, T)

# （3) IRM supervision target (robust, no negative/NaN)
mag_noise = np.maximum(mag_n - mag_c, 0.0)   # (F, T)
den = mag_c + mag_noise + 1e-8               # (F, T)
irm = mag_c / den                            # (F, T) in (0,1]

# 4) Assemble as (B=1, T, F) and explicitly create a Tensor
# (F, T) to (T, F)
mag_n_t = torch.tensor(mag_n.T[None, :, :], dtype=torch.float32, device=device)  # (1, T, F)
irm_t   = torch.tensor(irm.T[None, :, :],   dtype=torch.float32, device=device)  # (1, T, F)

# Shape self-check, if it is wrong, it will report an error immediately
assert mag_n_t is not None and irm_t is not None, "Tensors must not be None"
assert mag_n_t.ndim == 3 and irm_t.ndim == 3, f"ndim wrong: {mag_n_t.ndim}, {irm_t.ndim}"
assert mag_n_t.shape == irm_t.shape, f"shape mismatch: {mag_n_t.shape} vs {irm_t.shape}"

# （5) Define the model (non-lazy build: pass in frequency dimension)
in_dim = mag_n_t.shape[-1]   # example 513
model = MaskNet(in_dim=in_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# （6) training
model.train()
for epoch in range(15):
    optimizer.zero_grad()
    pred = model(mag_n_t)               # (1, T, F)
    # Self-check again to prevent None
    assert pred is not None, "pred is None"
    loss = criterion(pred, irm_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d}  Loss={loss.item():.6f}")

# （7) Inference and reconstruction
model.eval()
with torch.no_grad():
    pred_mask = model(mag_n_t).cpu().numpy()[0].T  # (F, T)
    mag_hat = pred_mask * mag_n
    X_hat = mag_hat * np.exp(1j * pha_n)
    enh = istft(X_hat)

out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
out_path = out_dir / "example_dnn.wav"
save_wav(str(out_path), enh.astype(np.float32), sr)
print("Saved enhanced speech to", out_path)
