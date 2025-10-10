# scripts/train_mask.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import load_wav, save_wav
from src.stft import stft, istft
from src.add_noise import add_white_noise
from src.dnn_mask import MaskNet

# ----------------------------
# 0) 设备与随机种子
# ----------------------------
# 优先用 CUDA，其次 Apple MPS，最后 CPU
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
torch.manual_seed(0)
np.random.seed(0)

# ----------------------------
# 1) 读取干净语音，并合成 0 dB 噪声语音
# ----------------------------
clean_path = "data/clean/example.wav"
assert Path(clean_path).exists(), f"File not found: {clean_path}"
clean, sr = load_wav(clean_path)
noisy = add_white_noise(clean, snr_db=0)

# ----------------------------
# 2) STFT -> 幅度与相位
# ----------------------------
def wav_to_mag_phase(x: np.ndarray):
    X = stft(x)                                  # 复谱: (F, T)
    return np.abs(X).astype(np.float32), np.angle(X)

mag_c, _     = wav_to_mag_phase(clean)           # (F, T)
mag_n, pha_n = wav_to_mag_phase(noisy)           # (F, T)

# ----------------------------
# 3) IRM 监督目标（稳健、无负值/NaN）
#     IRM = |S| / (|S| + |N|)
# ----------------------------
mag_noise = np.maximum(mag_n - mag_c, 0.0)       # (F, T)
den = mag_c + mag_noise + 1e-8                   # (F, T)
irm = mag_c / den                                # (F, T) ∈ (0,1]

# ----------------------------
# 4) 组装为 Tensor： (B=1, T, F)
# ----------------------------
mag_n_t = torch.tensor(mag_n.T[None, :, :], dtype=torch.float32, device=device)  # (1, T, F)
irm_t   = torch.tensor(irm.T[None, :, :],   dtype=torch.float32, device=device)  # (1, T, F)

# 形状自检
assert mag_n_t is not None and irm_t is not None, "Tensors must not be None"
assert mag_n_t.ndim == 3 and irm_t.ndim == 3, f"ndim wrong: {mag_n_t.ndim}, {irm_t.ndim}"
assert mag_n_t.shape == irm_t.shape, f"shape mismatch: {mag_n_t.shape} vs {irm_t.shape}"

# ----------------------------
# 5) 定义模型（非惰性构建：显式传入频率维）
# ----------------------------
in_dim = mag_n_t.shape[-1]    # 例如 513
model = MaskNet(in_dim=in_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
grad_clip = 1.0  # 梯度裁剪，训练更稳

# ----------------------------
# 6) 训练
# ----------------------------
EPOCHS = 60  # 如需更久训练，可改为 100
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    pred = model(mag_n_t)                       # (1, T, F)
    assert pred is not None, "pred is None"
    loss = criterion(pred, irm_t)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()
    print(f"Epoch {epoch+1:02d}  Loss={loss.item():.6f}")

# 可选：保存模型权重，便于后续评测复用
Path("results").mkdir(exist_ok=True, parents=True)
model_path = Path("results") / "dnn_mask.pth"
torch.save(model.state_dict(), model_path)

# ----------------------------
# 7) 推理并重建时域语音
# ----------------------------
model.eval()
with torch.no_grad():
    pred_mask = model(mag_n_t).detach().cpu().numpy()[0].T  # (F, T)
    pred_mask = np.clip(pred_mask, 0.0, 1.0)                # 安全裁剪
    mag_hat = pred_mask * mag_n
    X_hat = mag_hat * np.exp(1j * pha_n)
    enh = istft(X_hat)

# 保存增强语音
out_wav = Path("results") / "example_dnn.wav"
save_wav(str(out_wav), enh.astype(np.float32), sr)

print(f"Model saved to: {model_path}")
print(f"Saved enhanced speech to: {out_wav}")
print(f"Device: {device}")
