

from pathlib import Path
from typing import List, Tuple

import warnings
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




def load_wav(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate (default: 16kHz)
    
    Args:
        path: Audio file path
        target_sr: Target sample rate (default: 16000)
    
    Returns:
        wav: torch.Tensor, shape (L,)
        sr: int, actual sample rate (should be target_sr)
    """
    import librosa
    
    data, file_sr = sf.read(path)

    if data.ndim > 1:
        data = data.mean(axis=1)
    
    if file_sr != target_sr:
        data = librosa.resample(data.astype(np.float32), orig_sr=file_sr, target_sr=target_sr)
        file_sr = target_sr
    
    wav = torch.from_numpy(data.astype(np.float32))
    return wav, file_sr

class PairSpectrogramDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        n_fft: int = 1024,
        hop_length: int = None,
        win_length: int = None,
        target_sr: int = 16000,
    ):
        """
        Args:
            manifest_path: CSV file with clean_path,noisy_path pairs
            target_sr: Target sample rate (default: 16000)
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

        if hop_length is None:
            from src.stft import HOP_LEN
            hop_length = HOP_LEN
        if win_length is None:
            from src.stft import WIN_LEN
            win_length = WIN_LEN
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.target_sr = target_sr

    def __len__(self):
        return len(self.items)

    def _wav_to_mag(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav:  magnitude STFT: [freq_bins, time_frames]
        """
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length if self.win_length else self.n_fft),
            center=True,
            return_complex=True,
        )
        mag = stft.abs()  # [F, T]
        return mag

    def __getitem__(self, idx: int):
        clean_path, noisy_path = self.items[idx]

        clean_wav, sr_c = load_wav(clean_path, target_sr=self.target_sr)
        noisy_wav, sr_n = load_wav(noisy_path, target_sr=self.target_sr)
        
        assert sr_c == self.target_sr and sr_n == self.target_sr, \
            f"Sample rate mismatch: clean={sr_c}, noisy={sr_n}, target={self.target_sr}"


        clean_mag = self._wav_to_mag(clean_wav)
        noisy_mag = self._wav_to_mag(noisy_wav)

        T = min(clean_mag.shape[1], noisy_mag.shape[1])
        clean_mag = clean_mag[:, :T]
        noisy_mag = noisy_mag[:, :T]

      
        return noisy_mag, clean_mag



def collate_pad_spectrogram(batch):
    """
    
        noisy: [B, F, T_max]
        clean: [B, F, T_max]
    """
    noisy_list, clean_list = zip(*batch)  

    max_T = max(x.shape[1] for x in noisy_list)

    def pad_to_max(x: torch.Tensor) -> torch.Tensor:
        # x: [F, T]
        if x.shape[1] == max_T:
            return x
        pad_T = max_T - x.shape[1]
      
        return F.pad(x, (0, pad_T))

    noisy_pad = torch.stack([pad_to_max(x) for x in noisy_list], dim=0)   # [B, F, T_max]
    clean_pad = torch.stack([pad_to_max(x) for x in clean_list], dim=0)   # [B, F, T_max]

    return noisy_pad, clean_pad



def load_pair_dataset(
    manifest_train: str,
    manifest_val: str,
    batch: int = 8,
    workers: int = 0,
):
    """
    

    train_unet.py 
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
