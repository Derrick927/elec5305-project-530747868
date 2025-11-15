# src/unet_model.py
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    一维 U-Net：
    - 输入形状: [B, C, T]，这里 C = 频率 bin 数 (freq_bins)
    - 输出形状: [B, C, T]，学习的是 “noisy 幅度谱 -> clean 幅度谱”
    """

    def __init__(self, n_channels=513, n_classes=513):
        super().__init__()
        # 编码器
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256))

        # 解码器
        self.up1_trans = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.up1 = DoubleConv(256, 128)

        self.up2_trans = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.up2 = DoubleConv(128, 64)

        self.up3_trans = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.up3 = DoubleConv(64, 32)

        self.outc = nn.Conv1d(32, n_classes, kernel_size=1)

    @staticmethod
    def _match_time(a: torch.Tensor, b: torch.Tensor):
        """
        把 a、b 在时间维上裁到同一个长度，返回裁剪后的 (a, b)。
        只做简单的右侧裁剪，保证不会越界。
        """
        t1 = a.size(-1)
        t2 = b.size(-1)
        if t1 == t2:
            return a, b
        t_min = min(t1, t2)
        return a[..., :t_min], b[..., :t_min]

    def forward(self, x):
        # 编码
        x1 = self.inc(x)     # [B, 32, T]
        x2 = self.down1(x1)  # [B, 64, T/2]
        x3 = self.down2(x2)  # [B, 128, T/4]
        x4 = self.down3(x3)  # [B, 256, T/8]

        # 解码 + skip 连接（注意时间维裁剪对齐）
        u1 = self.up1_trans(x4)     # [B, 128, ~T/4]
        u1, x3_ = self._match_time(u1, x3)
        u1 = torch.cat([u1, x3_], dim=1)  # [B, 256, T1]
        u1 = self.up1(u1)

        u2 = self.up2_trans(u1)     # [B, 64, ~T/2]
        u2, x2_ = self._match_time(u2, x2)
        u2 = torch.cat([u2, x2_], dim=1)  # [B, 128, T2]
        u2 = self.up2(u2)

        u3 = self.up3_trans(u2)     # [B, 32, ~T]
        u3, x1_ = self._match_time(u3, x1)
        u3 = torch.cat([u3, x1_], dim=1)  # [B, 64, T3]
        u3 = self.up3(u3)

        out = self.outc(u3)         # [B, n_classes, T3]
        return out
