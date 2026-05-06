import torch
import torch.nn as nn
from typing import Dict, Any


class AttUNetWrapper(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 5):
        super().__init__()
        try:
            from .networks.AttUnet import AttentionUNet as BackboneAtt
        except Exception as e:
            raise ImportError(f"AttentionUNet backbone not available: {e}")

        # Hardcoded configuration - AttentionUNet expects 5-level features (including bottleneck)
        self.model = BackboneAtt(in_channels=in_channels, n_classes=num_classes, features=[64, 128, 256, 512, 1024])
        self.features_dim = [64, 128, 256, 512, 1024]

    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        """Extract intermediate features for knowledge distillation
        Extracts all encoder + decoder stages for comprehensive distillation
        """
        # Encoder
        e1 = self.model.encoder1(x)
        e2 = self.model.encoder2(self.model.maxpool(e1))
        e3 = self.model.encoder3(self.model.maxpool(e2))
        e4 = self.model.encoder4(self.model.maxpool(e3))
        
        # Bottleneck
        b = self.model.bottleneck(self.model.maxpool(e4))
        
        # Decoder with attention gates (all stages)
        d4 = self.model.upconv4(b)
        e4_att = self.model.att4(g=d4, x=e4)
        d4 = torch.cat((e4_att, d4), dim=1)
        d4 = self.model.decoder4(d4)
        
        d3 = self.model.upconv3(d4)
        e3_att = self.model.att3(g=d3, x=e3)
        d3 = torch.cat((e3_att, d3), dim=1)
        d3 = self.model.decoder3(d3)
        
        d2 = self.model.upconv2(d3)
        e2_att = self.model.att2(g=d2, x=e2)
        d2 = torch.cat((e2_att, d2), dim=1)
        d2 = self.model.decoder2(d2)
        
        d1 = self.model.upconv1(d2)
        e1_att = self.model.att1(g=d1, x=e1)
        d1 = torch.cat((e1_att, d1), dim=1)
        d1 = self.model.decoder1(d1)
        
        features = {
            'enc1': e1,
            'enc2': e2,
            'enc3': e3,
            'enc4': e4,
            'bottleneck': b,
            'enc1_att': e1_att,
            'enc2_att': e2_att,
            'enc3_att': e3_att,
            'enc4_att': e4_att,
            'dec1': d1,
            'dec2': d2,
            'dec3': d3,
            'dec4': d4,
        }
        return None, features

    def get_feature_channels(self):
        return self.features_dim


def get_model(config: Dict[str, Any]) -> nn.Module:
    architecture = config['architecture'] if 'architecture' in config else config.get('model_name', 'AttentionUNet')
    arch = architecture.strip()
    if arch in ['AttentionUNet', 'AttUNet', 'AttUnet']:
        in_ch = config.get('input_channels', 3)
        out_ch = config.get('num_classes', 5)
        return AttUNetWrapper(in_ch, out_ch)
    else:
        raise ValueError(f"Unsupported architecture for local model_implements: {arch}")


__all__ = ['get_model', 'AttUNetWrapper']
