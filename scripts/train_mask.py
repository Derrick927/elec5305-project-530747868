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

# 设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1) 读入干净语音并生成 0 dB 带噪
clean, sr = load_wav("data/clean/example.wav")
noisy = add_white_noise(clean, snr_db=0)

# 2) STFT -> 幅度与相位
def wav_to_mag_phase(x: np.ndarray):
    X = stft(x)                              # (F, T)
    return np.abs(X).astype(np.float32), np.angle(X)

mag_c, _     = wav_to_mag_phase(clean)       # (F, T)
mag_n, pha_n = wav_to_mag_phase(noisy)       # (F, T)

# 3) IRM 监督目标（稳健，不出现负数/NaN）
mag_noise = np.maximum(mag_n - mag_c, 0.0)   # (F, T)
den = mag_c + mag_noise + 1e-8               # (F, T)
irm = mag_c / den                            # (F, T) in (0,1]

# 4) 组装为 (B=1, T, F) 并**显式创建 Tensor**（避免 None）
# 注意把 (F, T) 转置到 (T, F)
mag_n_t = torch.tensor(mag_n.T[None, :, :], dtype=torch.float32, device=device)  # (1, T, F)
irm_t   = torch.tensor(irm.T[None, :, :],   dtype=torch.float32, device=device)  # (1, T, F)

# —— 形状自检，若不对会立刻报错 —— #
assert mag_n_t is not None and irm_t is not None, "Tensors must not be None"
assert mag_n_t.ndim == 3 and irm_t.ndim == 3, f"ndim wrong: {mag_n_t.ndim}, {irm_t.ndim}"
assert mag_n_t.shape == irm_t.shape, f"shape mismatch: {mag_n_t.shape} vs {irm_t.shape}"

# 5) 定义模型（非延迟构建：传入频率维）
in_dim = mag_n_t.shape[-1]   # 例如 513
model = MaskNet(in_dim=in_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6) 训练
model.train()
for epoch in range(15):
    optimizer.zero_grad()
    pred = model(mag_n_t)               # (1, T, F)
    # 再次自检，防止 None
    assert pred is not None, "pred is None"
    loss = criterion(pred, irm_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d}  Loss={loss.item():.6f}")

# 7) 推理与重建
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
