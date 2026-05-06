"""
Attention U-Net implementation for segmentation tasks.
Based on: "Attention U-Net: Learning Where to Look for the Pancreas" by Oktay et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with two conv layers, batch norm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate module for focusing on relevant feature regions"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        # Gating signal processing
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Skip connection processing  
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention coefficient generation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: gating signal from deeper layer
            x: skip connection from encoder
        Returns:
            attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UpConv(nn.Module):
    """Upsampling convolution block"""
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB)
        n_classes (int): Number of output classes 
        features (list): List of feature dimensions for each encoder level
    """
    def __init__(self, in_channels=3, n_classes=5, features=[64, 128, 256, 512, 1024]):
        super(AttentionUNet, self).__init__()
        
        self.n_classes = n_classes
        self.features = features
        
        # Encoder (Contracting Path)
        self.encoder1 = ConvBlock(in_channels, features[0])
        self.encoder2 = ConvBlock(features[0], features[1])
        self.encoder3 = ConvBlock(features[1], features[2])
        self.encoder4 = ConvBlock(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[4])
        
        # Decoder (Expanding Path)
        self.upconv4 = UpConv(features[4], features[3])
        self.att4 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[2])
        self.decoder4 = ConvBlock(features[4], features[3])
        
        self.upconv3 = UpConv(features[3], features[2])
        self.att3 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[1])
        self.decoder3 = ConvBlock(features[3], features[2])
        
        self.upconv2 = UpConv(features[2], features[1])
        self.att2 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[0])
        self.decoder2 = ConvBlock(features[2], features[1])
        
        self.upconv1 = UpConv(features[1], features[0])
        self.att1 = AttentionGate(F_g=features[0], F_l=features[0], F_int=32)
        self.decoder1 = ConvBlock(features[1], features[0])
        
        # Final classification layer
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)
        
        # Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))
        e4 = self.encoder4(self.maxpool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.maxpool(e4))
        
        # Decoder with attention gates
        d4 = self.upconv4(b)
        # Apply attention gate
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat((e4_att, d4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat((e3_att, d3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat((e2_att, d2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat((e1_att, d1), dim=1)
        d1 = self.decoder1(d1)
        
        # Final segmentation map
        outputs = self.final_conv(d1)
        
        return outputs

    def extract_features(self, x):
        """
        Extract encoder features for knowledge distillation
        Returns dict with encoder feature maps at each stage
        """
        # Encoder
        e1 = self.encoder1(x)          # 64 channels
        e2 = self.encoder2(self.maxpool(e1))    # 128 channels
        e3 = self.encoder3(self.maxpool(e2))    # 256 channels
        e4 = self.encoder4(self.maxpool(e3))    # 512 channels
        
        # Bottleneck
        b = self.bottleneck(self.maxpool(e4))   # 1024 channels
        
        return {
            'e1': e1,
            'e2': e2,
            'e3': e3,
            'e4': e4,
            'bottleneck': b
        }

    def freeze_bn(self):
        """Freeze batch normalization layers"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def test_attunet():
    """Test function for AttentionUNet"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(in_channels=3, n_classes=5).to(device)
    
    # Test with random input
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    test_attunet()
