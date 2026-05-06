"""
Classifier factory for architecture selection.
Keeps ResNet50 default behavior while adding EfficientNet-B0 as an option.
"""

from models.efficientnet_b0 import create_efficientnet_b0
from models.resnet50 import create_resnet50


SUPPORTED_CLASSIFIERS = ("resnet50", "efficientnet_b0")


def _normalize_model_name(model_name: str) -> str:
    normalized = (model_name or "resnet50").strip().lower().replace("-", "_")
    if normalized in ("resnet", "resnet_50"):
        return "resnet50"
    if normalized in ("efficientnetb0", "efficientnet_b0"):
        return "efficientnet_b0"
    return normalized


def create_classifier(
    model_name,
    num_classes,
    pretrained=True,
    dropout_rate=0.0,
    freeze_backbone=False,
    custom_head=None,
):
    """
    Create a classifier model by name.

    Args:
        model_name: 'resnet50' (default) or 'efficientnet_b0'
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights
        dropout_rate: Dropout rate for classifier head
        freeze_backbone: Whether to freeze backbone parameters
        custom_head: Optional custom MLP head dims, e.g. [512, 256]
    """
    name = _normalize_model_name(model_name)

    if name == "resnet50":
        return create_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            custom_head=custom_head,
        )

    if name == "efficientnet_b0":
        return create_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            custom_head=custom_head,
        )

    raise ValueError(
        f"Unsupported model_name '{model_name}'. "
        f"Supported: {', '.join(SUPPORTED_CLASSIFIERS)}"
    )
