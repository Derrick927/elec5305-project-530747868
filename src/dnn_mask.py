# src/dnn_mask.py
import torch
import torch.nn as nn
import torch.nn.functional as F  # 不要用变量名覆盖 F

class MaskNet(nn.Module):
    """
    极简全连接掩蔽网络（非延迟构建版）：
    - 构造时显式传入频率维 in_dim（例如 513）
    - 输入一帧幅度谱，输出同维度 [0,1] 掩蔽
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, in_dim)

    def forward(self, mag: torch.Tensor):
        """
        mag: (B, T, Fdim)
        return: (B, T, Fdim) 掩蔽 [0,1]
        """
        B, T, Fdim = mag.shape
        assert Fdim == self.in_dim, f"频率维不一致: got {Fdim}, expect {self.in_dim}"
        x = mag.reshape(-1, Fdim)       # (B*T, Fdim)
        x = F.relu(self.fc1(x))
