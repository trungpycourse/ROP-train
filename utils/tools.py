
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Union, Optional


# Import here to avoid circular imports
def _import_wrappers():
    from ..models.wrappers import ClassificationModel, TimmModelWrapper
    return ClassificationModel, TimmModelWrapper

class TrainerCallBack:

    def train_callback(self):
        pass

    def iteration_callback(self):
        pass

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"


def load_config(config_path: Union[str, Path], model_type: Optional[str] = None) -> Dict[str, Any]:
    """Load and process configuration from a YAML file.
    
    Args:
        config_path (Union[str, Path]): Path to the YAML configuration file
        model_type (Optional[str]): Type of model configuration to extract (e.g., 'train', 'inference')
                                    If None, returns the entire config
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If model_type is specified but not found in config
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type is not None:
        if model_type not in config:
            raise KeyError(f"Configuration section '{model_type}' not found in {config_path}")
        return config[model_type]
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of configuration dictionaries
    
    Returns:
        Dict[str, Any]: Merged configuration dictionary
    """
    merged = {}
    for config in configs:
        _deep_update(merged, config)
    return merged


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save
        save_path (Union[str, Path]): Path to save the configuration file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_default_config(config_name: str) -> Dict[str, Any]:
    """Get default configuration from the configs directory.
    
    Args:
        config_name (str): Name of the configuration file (without .yaml extension)
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    
    Raises:
        FileNotFoundError: If default config file doesn't exist
    """
    config_dir = Path(__file__).parent.parent / 'configs'
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Default configuration not found: {config_path}")
    
    return load_config(config_path)


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """Recursively update a dictionary.
    
    Args:
        base_dict (Dict[str, Any]): Dictionary to update
        update_dict (Dict[str, Any]): Dictionary with updates
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Create a model based on configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing model parameters
            Required keys:
            - model_name: Name of the timm model
            - num_classes: Number of output classes
            Optional keys:
            - pretrained: Whether to use pretrained weights
            - dropout_rate: Dropout rate for the classifier
            - features_only: Whether to return features instead of logits
    
    Returns:
        torch.nn.Module: The created model
    """
    ClassificationModel, TimmModelWrapper = _import_wrappers()
    
    model_name = config['model_name']
    num_classes = config['num_classes']
    pretrained = config.get('pretrained', True)
    dropout_rate = config.get('dropout_rate', 0.0)
    features_only = config.get('features_only', False)
    
    if features_only:
        return TimmModelWrapper(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            features_only=True
        )
    else:
        return ClassificationModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )


def save_checkpoint(model: torch.nn.Module,
                   save_path: str,
                   metadata: Dict[str, Any] = None) -> None:
    """Save a model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to save
        save_path (str): Path to save the checkpoint to
        metadata (Dict[str, Any], optional): Additional metadata to save
    """
    if hasattr(model, 'module'):  # Handle DataParallel
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    checkpoint = {
        'state_dict': state_dict,
    }
    
    if metadata is not None:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, save_path)


def load_checkpoint(model: torch.nn.Module, 
                   checkpoint_path: str,
                   strict: bool = True) -> Dict[str, Any]:
    """Load a checkpoint into a model.
    
    Args:
        model (torch.nn.Module): Model to load weights into
        checkpoint_path (str): Path to the checkpoint file
        strict (bool): Whether to strictly enforce that the keys in state_dict match
    
    Returns:
        Dict[str, Any]: Checkpoint dictionary containing metadata
    """
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel/DistributedDataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


# Visualization functions for training
def plot_confusion_matrix_fig(y_true, y_pred, class_names):
    """Plot confusion matrix and return figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    return fig


def plot_roc_curve_fig(y_true, y_probs, class_names):
    """Plot ROC curves for each class and return figure and macro AUC."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    
    num_classes = len(class_names)
    fig, ax = plt.subplots()
    roc_auc_macro = 0
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(np.array(y_true) == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_macro += roc_auc
        ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.4f})')
    roc_auc_macro /= num_classes
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    return fig, roc_auc_macro


def plot_pr_curve_fig(y_true, y_probs, class_names):
    """Plot Precision-Recall curves for each class and return figure and macro AP."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    num_classes = len(class_names)
    fig, ax = plt.subplots()
    pr_auc_macro = 0
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(np.array(y_true) == i, y_probs[:, i])
        pr_auc = average_precision_score(np.array(y_true) == i, y_probs[:, i])
        pr_auc_macro += pr_auc
        ax.plot(recall, precision, label=f'{class_names[i]} (AP={pr_auc:.4f})')
    pr_auc_macro /= num_classes
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    return fig, pr_auc_macro


def plot_tsne_fig(features, labels, class_names):
    """Plot t-SNE visualization of features and return figure."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    
    if len(features) < 2:
        print("Warning: Not enough samples for t-SNE. Skipping.")
        return None
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print("Warning: Features contain NaN or Inf. Skipping t-SNE.")
        return None

    try:
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(features)
        
        fig, ax = plt.subplots()
        for i, cls in enumerate(class_names):
            idx = np.array(labels) == i
            if np.sum(idx) > 0:
                ax.scatter(reduced[idx, 0], reduced[idx, 1], label=cls, alpha=0.6)
        
        if ax.has_data():
            ax.legend()
            ax.set_title("t-SNE Feature Embedding")
            return fig
        else:
            plt.close(fig)
            return None
    except Exception as e:
        print(f"Error in t-SNE: {e}")
        return None


