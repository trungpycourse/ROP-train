"""
EfficientNet-B0 model for classification.
Provides an API compatible with the ResNet50 wrapper used in this project.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 wrapper with optional custom dropout classifier.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate before final classifier
        freeze_backbone: Whether to freeze backbone parameters
        freeze_until_stage: Freeze backbone stages up to this feature index (0-8)
    """

    def __init__(
        self,
        num_classes=3,
        pretrained=True,
        dropout_rate=0.0,
        freeze_backbone=False,
        freeze_until_stage=None,
    ):
        super(EfficientNetB0, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        if dropout_rate > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)

        # Channel adapters for distillation scripts expecting ResNet-style channels.
        self.proj_enc2 = nn.Conv2d(24, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_enc3 = nn.Conv2d(80, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_enc4 = nn.Conv2d(192, 1024, kernel_size=1, stride=1, padding=0, bias=False)

        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_until_stage is not None:
            self._freeze_until_stage(freeze_until_stage)

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _freeze_until_stage(self, stage_idx):
        """Freeze backbone feature stages up to and including stage_idx (0-8)."""
        max_stage = min(int(stage_idx), len(self.backbone.features) - 1)
        for idx in range(max_stage + 1):
            for param in self.backbone.features[idx].parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def _forward_backbone(self, x):
        # features[0] -> 32 channels
        x = self.backbone.features[0](x)
        enc1 = x

        x = self.backbone.features[1](x)
        x = self.backbone.features[2](x)
        raw_enc2 = x  # 24 channels

        x = self.backbone.features[3](x)
        x = self.backbone.features[4](x)
        raw_enc3 = x  # 80 channels

        x = self.backbone.features[5](x)
        x = self.backbone.features[6](x)
        raw_enc4 = x  # 192 channels

        x = self.backbone.features[7](x)
        x = self.backbone.features[8](x)
        bottleneck = x  # 1280 channels

        return enc1, raw_enc2, raw_enc3, raw_enc4, bottleneck

    def forward_features(self, x, return_dict=False):
        """
        Extract features before classifier.

        Args:
            x: Input tensor
            return_dict: If True, return intermediate features for distillation

        Returns:
            If return_dict=False: pooled feature vector
            If return_dict=True: dict with enc1/enc2/enc3/enc4/bottleneck
        """
        enc1, raw_enc2, raw_enc3, raw_enc4, bottleneck = self._forward_backbone(x)

        if return_dict:
            return {
                "enc1": enc1,
                "enc2": self.proj_enc2(raw_enc2),
                "enc3": self.proj_enc3(raw_enc3),
                "enc4": self.proj_enc4(raw_enc4),
                "bottleneck": bottleneck,
            }

        x = self.backbone.avgpool(bottleneck)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        return self.classifier(features)

    def get_embedding(self, x):
        return self.forward_features(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class EfficientNetB0_CustomHead(nn.Module):
    """
    EfficientNet-B0 with custom multi-layer head.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        hidden_dims: Hidden dimensions for custom head
        dropout_rate: Dropout rate between head layers
        freeze_backbone: Whether to freeze backbone parameters
    """

    def __init__(
        self,
        num_classes=3,
        pretrained=True,
        hidden_dims=None,
        dropout_rate=0.3,
        freeze_backbone=False,
    ):
        super(EfficientNetB0_CustomHead, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.num_classes = num_classes

        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        layers = []
        current_dim = in_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_rate),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        # Channel adapters for distillation scripts expecting ResNet-style channels.
        self.proj_enc2 = nn.Conv2d(24, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_enc3 = nn.Conv2d(80, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_enc4 = nn.Conv2d(192, 1024, kernel_size=1, stride=1, padding=0, bias=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _forward_backbone(self, x):
        x = self.backbone.features[0](x)
        enc1 = x

        x = self.backbone.features[1](x)
        x = self.backbone.features[2](x)
        raw_enc2 = x

        x = self.backbone.features[3](x)
        x = self.backbone.features[4](x)
        raw_enc3 = x

        x = self.backbone.features[5](x)
        x = self.backbone.features[6](x)
        raw_enc4 = x

        x = self.backbone.features[7](x)
        x = self.backbone.features[8](x)
        bottleneck = x

        return enc1, raw_enc2, raw_enc3, raw_enc4, bottleneck

    def forward_features(self, x, return_dict=False):
        enc1, raw_enc2, raw_enc3, raw_enc4, bottleneck = self._forward_backbone(x)

        if return_dict:
            return {
                "enc1": enc1,
                "enc2": self.proj_enc2(raw_enc2),
                "enc3": self.proj_enc3(raw_enc3),
                "enc4": self.proj_enc4(raw_enc4),
                "bottleneck": bottleneck,
            }

        x = self.backbone.avgpool(bottleneck)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        return self.classifier(features)

    def get_embedding(self, x):
        backbone_features = self.forward_features(x)
        embedding = backbone_features
        for layer in self.classifier[:-1]:
            embedding = layer(embedding)
        return embedding

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def create_efficientnet_b0(
    num_classes,
    pretrained=True,
    dropout_rate=0.0,
    freeze_backbone=False,
    custom_head=None,
):
    """
    Factory function to create EfficientNet-B0 models.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate
        freeze_backbone: Whether to freeze backbone
        custom_head: List of hidden dims for custom head, or None
    """
    if custom_head is not None:
        return EfficientNetB0_CustomHead(
            num_classes=num_classes,
            pretrained=pretrained,
            hidden_dims=custom_head,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
        )

    return EfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
