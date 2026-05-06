"""
ResNet50 Model for Classification
Based on torchvision's ResNet implementation with customizations
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    """
    ResNet50 model wrapper with customizable head
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate before final classifier (default: 0.0)
        freeze_backbone: Whether to freeze backbone layers (default: False)
        freeze_until_layer: Freeze layers up to this layer number (0-4, default: None)
    """
    
    def __init__(self, num_classes=3, pretrained=True, dropout_rate=0.0, 
                 freeze_backbone=False, freeze_until_layer=None):
        super(ResNet50, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the input features for the final FC layer
        in_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()  # Remove original FC layer
        
        # Custom classifier head
        if dropout_rate > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_until_layer is not None:
            self._freeze_until_layer(freeze_until_layer)
        
        # Store layer names for feature extraction
        self.layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        
    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _freeze_until_layer(self, layer_num):
        """
        Freeze layers up to specified layer
        layer_num: 0 (conv1), 1 (layer1), 2 (layer2), 3 (layer3), 4 (layer4)
        """
        layers_to_freeze = self.layer_names[:layer_num + 1]
        for layer_name in layers_to_freeze:
            layer = getattr(self.backbone, layer_name)
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
    
    def forward_features(self, x, return_dict=False):
        """
        Extract features before classification head
        
        Args:
            x: Input tensor
            return_dict: If True, return dict with intermediate features for distillation
        
        Returns:
            If return_dict=False: Final feature vector (2048-dim)
            If return_dict=True: Dict with enc1-enc4 intermediate features
        """
        # Initial conv
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        enc1 = x  # After initial conv, before maxpool
        
        x = self.backbone.maxpool(x)
        
        # ResNet layers
        enc2 = self.backbone.layer1(x)  # 256 channels
        enc3 = self.backbone.layer2(enc2)  # 512 channels
        enc4 = self.backbone.layer3(enc3)  # 1024 channels
        bottleneck = self.backbone.layer4(enc4)  # 2048 channels
        
        if return_dict:
            return {
                'enc1': enc1,
                'enc2': enc2,
                'enc3': enc3,
                'enc4': enc4,
                'bottleneck': bottleneck
            }
        
        # Global average pooling and flatten
        x = self.backbone.avgpool(bottleneck)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x):
        """Forward pass through the entire network"""
        features = self.forward_features(x)
        output = self.classifier(features)
        return output
    
    def get_embedding(self, x):
        """Get feature embeddings (same as forward_features)"""
        return self.forward_features(x)
    
    def predict_proba(self, x):
        """Get probability predictions"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x):
        """Get class predictions"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class ResNet50_CustomHead(nn.Module):
    """
    ResNet50 with custom multi-layer head
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        hidden_dims: List of hidden layer dimensions (e.g., [512, 256])
        dropout_rate: Dropout rate between layers (default: 0.3)
        freeze_backbone: Whether to freeze backbone layers (default: False)
    """
    
    def __init__(self, num_classes=3, pretrained=True, hidden_dims=[512, 256], 
                 dropout_rate=0.3, freeze_backbone=False):
        super(ResNet50_CustomHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the input features for the custom head
        in_features = self.backbone.fc.in_features
        
        # Remove original FC layer
        self.backbone.fc = nn.Identity()
        
        # Build custom head
        layers = []
        current_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward_features(self, x, return_dict=False):
        """
        Extract features before classification head
        
        Args:
            x: Input tensor
            return_dict: If True, return dict with intermediate features for distillation
        
        Returns:
            If return_dict=False: Final feature vector (2048-dim)
            If return_dict=True: Dict with enc1-enc4 intermediate features
        """
        # Initial conv
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        enc1 = x  # After initial conv, before maxpool
        
        x = self.backbone.maxpool(x)
        
        # ResNet layers
        enc2 = self.backbone.layer1(x)  # 256 channels
        enc3 = self.backbone.layer2(enc2)  # 512 channels
        enc4 = self.backbone.layer3(enc3)  # 1024 channels
        bottleneck = self.backbone.layer4(enc4)  # 2048 channels
        
        if return_dict:
            return {
                'enc1': enc1,
                'enc2': enc2,
                'enc3': enc3,
                'enc4': enc4,
                'bottleneck': bottleneck
            }
        
        # Global average pooling and flatten
        x = self.backbone.avgpool(bottleneck)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x):
        """Forward pass through the entire network"""
        features = self.forward_features(x)
        output = self.classifier(features)
        return output
    
    def get_embedding(self, x):
        """
        Get feature embeddings from custom head (e.g., 512-dim)
        Extracts features after custom head layers but before final classification
        """
        # Get backbone features (2048-dim)
        backbone_features = self.forward_features(x)
        
        # Pass through custom head layers except final classification layer
        # classifier structure: [Linear, BN, ReLU, Dropout] * n + [Linear(final)]
        embedding = backbone_features
        for layer in self.classifier[:-1]:  # All layers except final Linear
            embedding = layer(embedding)
        
        return embedding
    
    def predict_proba(self, x):
        """Get probability predictions"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x):
        """Get class predictions"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def create_resnet50(num_classes, pretrained=True, dropout_rate=0.0, 
                    freeze_backbone=False, custom_head=None):
    """
    Factory function to create ResNet50 models
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate
        freeze_backbone: Whether to freeze backbone
        custom_head: List of hidden dims for custom head (e.g., [512, 256])
                     If None, uses simple head with dropout
    
    Returns:
        ResNet50 model instance
    """
    if custom_head is not None:
        return ResNet50_CustomHead(
            num_classes=num_classes,
            pretrained=pretrained,
            hidden_dims=custom_head,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
    else:
        return ResNet50(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )


# Test function
if __name__ == "__main__":
    # Test simple ResNet50
    model = ResNet50(num_classes=3, pretrained=False, dropout_rate=0.2)
    x = torch.randn(2, 3, 224, 224)
    
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (2, 3)
    
    features = model.get_embedding(x)
    print(f"Feature shape: {features.shape}")  # Should be (2, 2048)
    
    # Test custom head ResNet50
    model_custom = ResNet50_CustomHead(
        num_classes=3, 
        pretrained=False, 
        hidden_dims=[512, 256],
        dropout_rate=0.3
    )
    
    output = model_custom(x)
    print(f"Custom head output shape: {output.shape}")  # Should be (2, 3)
    
    print("\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
