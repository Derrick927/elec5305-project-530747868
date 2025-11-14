# src/utils_unet.py
# 专门给 UNet 用的工具：读 manifest、做 STFT、构建 DataLoader，并在 batch 里做 padding

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 基础函数：读 wav
# =========================

def load_wav(path: str) -> Tuple[torch.Tensor, int]:
    """
    读入 wav，转成 float32 的 mono Tensor，返回 (waveform, sr)
    """
    data, sr = sf.read(path)
    # 立体声转单声道
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data.astype(np.float32))
    return wav, sr


# =========================
# 数据集：从 manifest 读 (clean, noisy) 对
# 输出频谱幅度 (noisy_mag, clean_mag)
# 形状：[freq_bins=513, time_frames]
# =========================

class PairSpectrogramDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        """
        manifest_path: CSV 文件路径，每行：
            clean_path,noisy_path
        """
        self.items: List[Tuple[str, str]] = []

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                clean_path, noisy_path = parts[0], parts[1]
                self.items.append((clean_path, noisy_path))

        if len(self.items) == 0:
            raise RuntimeError(f"Empty manifest: {manifest_path}")

        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.items)

    def _wav_to_mag(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [T]，输出 magnitude STFT: [freq_bins, time_frames]
        """
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft),
            center=True,
            return_complex=True,
        )
        mag = stft.abs()  # [F, T]
        return mag

    def __getitem__(self, idx: int):
        clean_path, noisy_path = self.items[idx]

        clean_wav, sr_c = load_wav(clean_path)
        noisy_wav, sr_n = load_wav(noisy_path)

        # 一般来说 sr_c == sr_n == 16000，这里不强制检查，只要一致就行
        # if sr_c != sr_n:
        #     print(f"Warning: sr mismatch: {sr_c} vs {sr_n} at index {idx}")

        clean_mag = self._wav_to_mag(clean_wav)
        noisy_mag = self._wav_to_mag(noisy_wav)

        # 有时候因为长度差一点点，STFT 帧数会不一样，这里裁成相同长度
        T = min(clean_mag.shape[1], noisy_mag.shape[1])
        clean_mag = clean_mag[:, :T]
        noisy_mag = noisy_mag[:, :T]

        # 返回顺序：(noisy, clean)，方便网络学“去噪”
        return noisy_mag, clean_mag


# =========================
# collate_fn：在时间维做 padding
# =========================

def collate_pad_spectrogram(batch):
    """
    batch: 列表，里面是 (noisy_mag, clean_mag)
    每个元素形状为 [F, T_i]，T_i 可以不同

    输出：
        noisy: [B, F, T_max]
        clean: [B, F, T_max]
    """
    noisy_list, clean_list = zip(*batch)  # 拆成两个 tuple

    # 当前 batch 中最大的时间长度
    max_T = max(x.shape[1] for x in noisy_list)

    def pad_to_max(x: torch.Tensor) -> torch.Tensor:
        # x: [F, T]
        if x.shape[1] == max_T:
            return x
        pad_T = max_T - x.shape[1]
        # 在时间维右侧补零：(left, right)
        return F.pad(x, (0, pad_T))

    noisy_pad = torch.stack([pad_to_max(x) for x in noisy_list], dim=0)   # [B, F, T_max]
    clean_pad = torch.stack([pad_to_max(x) for x in clean_list], dim=0)   # [B, F, T_max]

    return noisy_pad, clean_pad


# =========================
# 对外接口：给 train_unet.py 用
# =========================

def load_pair_dataset(
    manifest_train: str,
    manifest_val: str,
    batch: int = 8,
    workers: int = 0,
):
    """
    读取 train / val 的 manifest，返回两个 DataLoader：
        train_loader, val_loader

    train_unet.py 里面直接：
        train_loader, val_loader = load_pair_dataset(
            args.manifest_train, args.manifest_val,
            batch=args.batch, workers=args.workers
        )
    """
    train_ds = PairSpectrogramDataset(manifest_train)
    val_ds = PairSpectrogramDataset(manifest_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_pad_spectrogram,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_pad_spectrogram,
    )

    return train_loader, val_loader
