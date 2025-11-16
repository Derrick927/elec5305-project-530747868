# src/dnn_mask_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ImprovedMaskNet(nn.Module):
    """
    Improved MaskNet with:
    1. LSTM for temporal context
    2. LayerNorm and Dropout
    3. Deeper network structure
    4. Support for log magnitude spectrogram input
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_layers: int = 2, 
                 use_log: bool = True, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.use_log = use_log
        self._return_logits = False
        
        self.input_norm = nn.LayerNorm(in_dim)
        
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        lstm_out_dim = hidden_dim * 2
        
        self.fc1 = nn.Linear(lstm_out_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, in_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming/He initialization"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name_param, param in m.named_parameters():
                    if 'weight_ih' in name_param:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name_param:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name_param:
                        param.data.fill_(0)
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1)
        
    def forward(self, mag: torch.Tensor):
        """
        Args:
            mag: Tensor of shape (B, T, Fdim) - magnitude or log magnitude spectrogram
        Returns:
            Tensor of shape (B, T, Fdim), mask in [0, 1]
        """
        B, T, Fdim = mag.shape
        assert Fdim == self.in_dim, f"Frequency dimension mismatch: got {Fdim}, expected {self.in_dim}"
        
        if self.use_log:
            x = torch.log1p(mag)
        else:
            x = mag
        
        x = self.input_norm(x)
        
        lstm_out, _ = self.lstm(x)
        
        x = lstm_out.reshape(-1, lstm_out.size(-1))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        if not self._return_logits:
            x = torch.sigmoid(x)
        
        return x.reshape(B, T, Fdim)


class DeepMaskNet(nn.Module):
    """
    Deeper MaskNet with multiple fully connected layers
    """
    def __init__(self, in_dim: int, hidden_dims: list = [512, 512, 256, 256], 
                 dropout: float = 0.3, use_log: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.use_log = use_log
        self._return_logits = False
        
        self.input_norm = nn.LayerNorm(in_dim)
        self.fc_layers = nn.ModuleList()
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.fc_layers.append(nn.LayerNorm(hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.fc_out = nn.Linear(prev_dim, in_dim)
        
    def forward(self, mag: torch.Tensor):
        B, T, Fdim = mag.shape
        
        if self.use_log:
            x = torch.log1p(mag)
        else:
            x = mag
        
        x = self.input_norm(x)
        
        x = x.reshape(-1, Fdim)
        
        for layer in self.fc_layers:
            x = layer(x)
        x = self.fc_out(x)
        if not self._return_logits:
            x = torch.sigmoid(x)
        
        return x.reshape(B, T, Fdim)

