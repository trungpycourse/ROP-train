"""
CBAM (Convolutional Block Attention Module)
Adapted from AttnFD implementation for knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module using avg and max pooling"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling branch
        avg_pool = self.avg_pool(x).view(b, c)
        avg_out = self.mlp(avg_pool)
        
        # Max pooling branch
        max_pool = self.max_pool(x).view(b, c)
        max_out = self.mlp(max_pool)
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        out = torch.sigmoid(out).view(b, c, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention module using channel pooling"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        # Channel-wise average and max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = torch.sigmoid(out)
        
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for channel attention MLP
        kernel_size: Kernel size for spatial attention conv
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then spatial attention
        x = self.spatial_attention(x)
        return x
