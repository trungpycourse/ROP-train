import torch
import torch.nn as nn
from typing import Dict, Any


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """Simple attention gate that produces attention coefficients for skip connections."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        # g: gating signal (from decoder), x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class AttentionUNetTeacher(nn.Module):
    """Lightweight Attention U-Net teacher compatible with the classification KD mapping.

    It exposes `extract_features(x)` returning a dict with keys:
      - 'enc2_att' (128 channels)
      - 'enc3' (256 channels)
      - 'dec3' (256 channels)
      - 'enc4' (512 channels)
      - 'dec4' (512 channels)

    These sizes are chosen to match the expected fused sizes used by the student connectors
    in the classification training script (enc3+dec3 -> 512, enc4+dec4 -> 1024 when fused).
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 4):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(input_channels, 64)
        self.pool = nn.MaxPool2d(2, 2)

        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Decoder
        self.up3 = UpConv(512, 256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = UpConv(256, 128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256, 128)

        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder forward
        x1 = self.enc1(x)        # [B,64,H,W]
        p1 = self.pool(x1)       # [B,64,H/2,W/2]
        x2 = self.enc2(p1)       # [B,128,H/2,W/2]  -- enc2
        p2 = self.pool(x2)       # [B,128,H/4,W/4]
        x3 = self.enc3(p2)       # [B,256,H/4,W/4]  -- enc3
        p3 = self.pool(x3)       # [B,256,H/8,W/8]
        x4 = self.enc4(p3)       # [B,512,H/8,W/8]  -- enc4

        # Decoder steps to produce dec3 and dec4 features for fusion
        u3 = self.up3(x4)        # [B,256,H/4,W/4]
        x3_att = self.att3(u3, x3)
        d3 = torch.cat([u3, x3_att], dim=1)
        d3 = self.dec3(d3)       # [B,256,H/4,W/4]  -- dec3

        u2 = self.up2(d3)        # [B,128,H/2,W/2]
        x2_att = self.att2(u2, x2)
        d2 = torch.cat([u2, x2_att], dim=1)
        d2 = self.dec2(d2)       # [B,128,H/2,W/2]  -- dec2 (not used by KD)

        # For compatibility with existing mapping, provide enc2_att as attended enc2 representation
        enc2_att = x2_att

        # dec4 as alias to x4 (no separate upsample for dec4 in lightweight teacher)
        dec4 = x4

        return {
            'enc2_att': enc2_att,
            'enc3': x3,
            'dec3': d3,
            'enc4': x4,
            'dec4': dec4,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        p1 = self.pool(x1)
        x2 = self.enc2(p1)
        p2 = self.pool(x2)
        x3 = self.enc3(p2)
        p3 = self.pool(x3)
        x4 = self.enc4(p3)

        u3 = self.up3(x4)
        x3_att = self.att3(u3, x3)
        d3 = torch.cat([u3, x3_att], dim=1)
        d3 = self.dec3(d3)

        u2 = self.up2(d3)
        x2_att = self.att2(u2, x2)
        d2 = torch.cat([u2, x2_att], dim=1)
        d2 = self.dec2(d2)

        out = self.head(d2)
        return out


def get_model(config: Dict[str, Any]) -> nn.Module:
    in_ch = int(config.get('input_channels', 3))
    num_classes = int(config.get('num_classes', 4))
    return AttentionUNetTeacher(input_channels=in_ch, num_classes=num_classes)


__all__ = ['get_model', 'AttentionUNetTeacher']
