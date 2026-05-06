import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Args:
        alpha (float or list): Weighting factor for each class. Can be a float or list of class weights.
        gamma (float): Exponent of the modulating factor to reduce the loss for well-classified examples.
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (float, int)): 
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for handling class imbalance.
    
    Args:
        weights (torch.Tensor): Class weights tensor of shape (C,) where C is the number of classes.
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, weights=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, 
                             weight=self.weights,
                             reduction=self.reduction)

class DiceLoss(nn.Module):
    """Dice Loss for image segmentation, also useful for imbalanced classification."""
    def __init__(self, smooth=1e-8, square=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.square = square

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        if self.square:
            intersection = (inputs * targets).sum()
            union = (inputs * inputs).sum() + (targets * targets).sum()
        else:
            intersection = (inputs * targets).sum()
            union = inputs.sum() + targets.sum()
            
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class TverskyLoss(nn.Module):
    """Tversky Loss for handling imbalanced data."""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-8):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - Tversky

class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss combines focal loss concept with Tversky index."""
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-8):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return (1 - Tversky)**self.gamma

class CrossEntropyLoss(nn.Module):
    """Standard cross entropy loss with optional label smoothing."""
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)

class BCEWithLogitsLoss(nn.Module):
    """Binary cross entropy loss with logits and optional class weights."""
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, inputs, targets):
        targets = targets.float()
        return self.loss(inputs, targets)

class CombinedLoss(nn.Module):
    """Combines multiple loss functions with optional weights."""
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)

    def forward(self, inputs, targets):
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        return total_loss

def get_loss_fn(loss_type, **kwargs):
    """Factory function to get the appropriate loss function.
    
    Args:
        loss_type (str): Type of loss function
        **kwargs: Additional arguments to pass to the loss function
    
    Returns:
        nn.Module: The requested loss function
    """
    loss_functions = {
        'ce': CrossEntropyLoss,
        'weighted_ce': WeightedCrossEntropyLoss,
        'focal': FocalLoss,
        'dice': DiceLoss,
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
        'bce': BCEWithLogitsLoss,
        'combined': CombinedLoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss_functions[loss_type](**kwargs)
