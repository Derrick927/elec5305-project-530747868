# src/dnn_mask.py
import torch
import torch.nn as nn
import torch.nn.functional as F  # avoid overriding variable names with F

class MaskNet(nn.Module):
    """
    Simple fully connected masking network (non-causal version).
    - Input: magnitude spectrogram with frequency dimension `in_dim` (e.g., 513)
    - Output: same dimension mask in range [0, 1]
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, in_dim)

    def forward(self, mag: torch.Tensor):
        """
        Args:
            mag: Tensor of shape (B, T, Fdim)
        Returns:
            Tensor of shape (B, T, Fdim), mask in [0, 1]
        """
        B, T, Fdim = mag.shape
        assert Fdim == self.in_dim, f"Frequency dimension mismatch: got {Fdim}, expected {self.in_dim}"

        x = mag.reshape(-1, Fdim)   # (B*T, Fdim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.reshape(B, T, Fdim)   # reshape back to (B, T, Fdim)

