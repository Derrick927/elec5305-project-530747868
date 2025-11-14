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
    def __init__(self, n_channels=513, n_classes=513):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256))

        self.up1_trans = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.up1 = DoubleConv(256, 128)

        self.up2_trans = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.up2 = DoubleConv(128, 64)

        self.up3_trans = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.up3 = DoubleConv(64, 32)

        self.outc = nn.Conv1d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1_trans(x4)
        u1 = torch.cat([u1, x3], dim=1)

        u1 = self.up1(u1)

        u2 = self.up2_trans(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2(u2)

        u3 = self.up3_trans(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.up3(u3)

        return self.outc(u3)

