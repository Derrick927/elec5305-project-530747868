import os
import sys
import numpy as np
import torch
import soundfile as sf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "SRC")

sys.path.insert(0, src_dir)
print("Added SRC to sys.path:", src_dir)

from stft import stft, istft, N_FFT, SR
from utils import load_wav
from src.dnn_mask_improved import MaskNet

ckpt_path = os.path.join(project_root, "checkpoints", "dnn", "masknet_best.pt")
noisy_path = os.path.join(project_root, "data", "noisy", "example_noisy.wav")

print("CKPT:", ckpt_path)
print("Noisy:", noisy_path)

noisy_wav = load_wav(noisy_path, sr=SR)
print("noisy_wav shape:", noisy_wav.shape)

Y = stft(noisy_wav)
mag_noisy = np.abs(Y)
phase_noisy = np.angle(Y)

print("mag_noisy min/max:", mag_noisy.min(), mag_noisy.max())

feats_TF = mag_noisy.T.astype(np.float32)

model = MaskNet(N_FFT // 2 + 1)

ckpt = torch.load(ckpt_path, map_location="cpu")

if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

model.load_state_dict(state_dict, strict=True)
model.eval()
print("Model weights loaded.")

with torch.no_grad():
    inp = torch.from_numpy(feats_TF).unsqueeze(0)
    pred_TF = model(inp).squeeze(0).numpy()

print("pred_TF min/max:", pred_TF.min(), pred_TF.max())

mask_FT = pred_TF.T
enh_mag = mag_noisy * mask_FT

print("enh_mag min/max:", enh_mag.min(), enh_mag.max())

enh_S = enh_mag * np.exp(1j * phase_noisy)
enh_wav = istft(enh_S).astype(np.float32)

print("enh_wav min/max:", enh_wav.min(), enh_wav.max())

mx = np.max(np.abs(enh_wav))
if mx > 1.0:
    enh_wav /= mx
    print("Clipped to [-1, 1]")

out_path = os.path.join(project_root, "debug_enh.wav")
sf.write(out_path, enh_wav, SR)
print("Saved:", out_path)
