import torch
import torch.nn as nn
import timm
from abc import ABC, abstractmethod  # For abstract base class

class BaseTimmWrapper(nn.Module, ABC):
    """Abstract base wrapper for TIMM models. Subclasses must implement architecture-specific methods."""
    
    def __init__(self, timm_model_name, num_classes, pretrained=True, dropout_rate=0.0):
        super().__init__()
        self.timm_model_name = timm_model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Create the base TIMM model
        self.model = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Store activation maps and gradients
        self.activation = {}
        self.gradient = {}
        
        # Add dropout if specified
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Register hooks (implemented in subclasses)
        self._register_hooks()
    
    @abstractmethod
    def _register_hooks(self):
        """Architecture-specific hook registration for features/gradients."""
        pass
    
    @abstractmethod
    def get_last_layer(self):
        """Get the last feature layer for GradCAM or similar (architecture-specific)."""
        pass
    
    def get_features(self, x, layer_name='features'):
        """Extract features from a registered layer."""
        _ = self(x)  # Run forward to populate activations
        return self.activation.get(layer_name, None)
    
    def forward_features(self, x):
        """Forward to get features before head (to be overridden if needed)."""
        return self.model.forward_features(x)
    
    def forward_head(self, features):
        """Forward through the head (to be overridden if custom)."""
        return self.model.forward_head(features)
    
    def forward(self, x):
        features = self.forward_features(x)
        features = self.dropout(features)
        return self.forward_head(features)
    
    def predict_proba(self, x):
        return torch.softmax(self(x), dim=1)
    
    def predict(self, x):
        return self(x).argmax(dim=1)
    
    def get_embedding(self, x):
        return self.forward_features(x)
    
    def freeze_layers(self, freeze_stages):
        """Freeze specific stages/layers (to be implemented in subclasses for transfer learning)."""
        raise NotImplementedError("Implement in subclass for architecture-specific freezing.")
    
    def unfreeze_layers(self, unfreeze_stages):
        """Unfreeze specific stages/layers."""
        raise NotImplementedError("Implement in subclass.")

# ResNet-specific wrapper
class CLS_ResNet(BaseTimmWrapper):
    def __init__(self, timm_model_name, num_classes, pretrained=True, dropout_rate=0.0, custom_head_layers=None):
        if 'resnet' not in timm_model_name:
            raise ValueError(f"CLS_ResNet only supports ResNet variants, got {timm_model_name}")
        super().__init__(timm_model_name, num_classes, pretrained, dropout_rate)
        
        # Optional custom head (e.g., add extra layers)
        if custom_head_layers:
            in_features = self.model.fc.in_features
            self.model.fc = self._build_custom_head(in_features, num_classes, custom_head_layers)
    
    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.activation[name] = output.detach()
            return hook
        
        feature_layer = self.model.layer4[-1]  # ResNet-specific last conv layer
        feature_layer.register_forward_hook(get_activation('features'))
    
    def get_last_layer(self):
        return self.model.layer4[-1]
    
    def freeze_layers(self, freeze_stages):
        # ResNet stages: conv1, layer1, layer2, layer3, layer4
        stages = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        for stage_idx in freeze_stages:
            if stage_idx < len(stages):
                for param in getattr(self.model, stages[stage_idx]).parameters():
                    param.requires_grad = False
    
    def unfreeze_layers(self, unfreeze_stages):
        # Similar logic, set requires_grad = True
        stages = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        for stage_idx in unfreeze_stages:
            if stage_idx < len(stages):
                for param in getattr(self.model, stages[stage_idx]).parameters():
                    param.requires_grad = True
    
    def _build_custom_head(self, in_features, num_classes, layers_config):
        # Example: layers_config = [{'type': 'Linear', 'out': 512}, {'type': 'ReLU'}, {'type': 'Dropout', 'p': 0.5}]
        head = nn.Sequential()
        current_in = in_features
        for cfg in layers_config:
            if cfg['type'] == 'Linear':
                head.add_module('linear', nn.Linear(current_in, cfg['out']))
                current_in = cfg['out']
            elif cfg['type'] == 'ReLU':
                head.add_module('relu', nn.ReLU())
            elif cfg['type'] == 'Dropout':
                head.add_module('dropout', nn.Dropout(cfg['p']))
            # Add more types as needed (e.g., BatchNorm)
        head.add_module('output', nn.Linear(current_in, num_classes))
        return head

# Example: ViT-specific wrapper
class CLS_ViT(BaseTimmWrapper):
    def __init__(self, timm_model_name, num_classes, pretrained=True, dropout_rate=0.0, custom_head_layers=None):
        if 'vit' not in timm_model_name:
            raise ValueError(f"CLS_ViT only supports ViT variants, got {timm_model_name}")
        super().__init__(timm_model_name, num_classes, pretrained, dropout_rate)
        
        if custom_head_layers:
            in_features = self.model.head.in_features
            self.model.head = self._build_custom_head(in_features, num_classes, custom_head_layers)
    
    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.activation[name] = output.detach()
            return hook
        
        feature_layer = self.model.blocks[-1]  # ViT-specific last transformer block
        feature_layer.register_forward_hook(get_activation('features'))
    
    def get_last_layer(self):
        return self.model.blocks[-1]
    
    def freeze_layers(self, freeze_stages):
        # ViT stages: patch_embed, blocks[0:N]
        for i in freeze_stages:
            if i == 0:
                for param in self.model.patch_embed.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.blocks[i-1].parameters():
                    param.requires_grad = False
    
    def unfreeze_layers(self, unfreeze_stages):
        # Similar, set requires_grad = True
        pass  # Implement similarly
    
    def _build_custom_head(self, in_features, num_classes, layers_config):
        # Same as above, reusable custom head builder
        # ...
        pass  # Copy from CLS_ResNet or make a shared utility function

# Add more classes similarly...
# For Swin:
# class CLS_Swin(BaseTimmWrapper):
    # Check: 'swin' in timm_model_name
    # Hooks: self.model.layers[-1]
    # Freeze: layers[0:N]
    # Custom head: self.model.head

# For ConvNext:
# class CLS_ConvNext(BaseTimmWrapper):
    # Check: 'convnext' in timm_model_name
    # Hooks: self.model.stages[-1]
    # Freeze: stem, stages[0:N]

# For RegNet:
# class CLS_RegNet(BaseTimmWrapper):
    # Check: 'regnet' in timm_model_name
    # Hooks: self.model.trunk_output
    # Freeze: s0, s1, s2, s3, s4

# For VGG:
# class CLS_VGG(BaseTimmWrapper):
    # Check: 'vgg' in timm_model_name
    # Hooks: self.model.features[-1]
    # Freeze: features[0:N]