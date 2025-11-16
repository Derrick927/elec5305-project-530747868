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
    - 输出形状: [B, C, T]，学习的是 "noisy 幅度谱 -> clean 幅度谱"
    """

    def __init__(self, n_channels=513, n_classes=513, base_channels=64):
        """
        Args:
            n_channels: 输入频率 bin 数（默认 513）
            n_classes: 输出频率 bin 数（默认 513）
            base_channels: 基础通道数，控制模型容量（默认 64，原版是 32）
        """
        super().__init__()
        # 编码器：增加通道数以提升模型容量
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(base_channels * 4, base_channels * 8))

        # 解码器
        self.up1_trans = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.up1 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2_trans = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up3_trans = nn.ConvTranspose1d(base_channels * 2, base_channels, 2, stride=2)
        self.up3 = DoubleConv(base_channels * 2, base_channels)

        self.outc = nn.Conv1d(base_channels, n_classes, kernel_size=1)
        
        # 改进的权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用 Kaiming 初始化提升训练效率"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x1 = self.inc(x)     # [B, base_channels, T]
        x2 = self.down1(x1)  # [B, base_channels*2, T/2]
        x3 = self.down2(x2)  # [B, base_channels*4, T/4]
        x4 = self.down3(x3)  # [B, base_channels*8, T/8]

        # 解码 + skip 连接（注意时间维裁剪对齐）
        u1 = self.up1_trans(x4)     # [B, base_channels*4, ~T/4]
        u1, x3_ = self._match_time(u1, x3)
        u1 = torch.cat([u1, x3_], dim=1)  # [B, base_channels*8, T1]
        u1 = self.up1(u1)

        u2 = self.up2_trans(u1)     # [B, base_channels*2, ~T/2]
        u2, x2_ = self._match_time(u2, x2)
        u2 = torch.cat([u2, x2_], dim=1)  # [B, base_channels*4, T2]
        u2 = self.up2(u2)

        u3 = self.up3_trans(u2)     # [B, base_channels, ~T]
        u3, x1_ = self._match_time(u3, x1)
        u3 = torch.cat([u3, x1_], dim=1)  # [B, base_channels*2, T3]
        u3 = self.up3(u3)

        out = self.outc(u3)         # [B, n_classes, T3]
        return out
