import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, auc
)
from sklearn.manifold import TSNE

class MetricsTracker:
    """Class to track and compute various metrics during training/evaluation."""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.y_true = []
        self.y_pred = []
        self.y_probs = []
    
    def update(self, y_true, y_pred, y_probs=None):
        """Update metrics with new predictions."""
        self.y_true.extend(y_true.cpu().numpy())
        self.y_pred.extend(y_pred.cpu().numpy())
        if y_probs is not None:
            self.y_probs.extend(y_probs.cpu().numpy())
    
    def compute(self):
        """Compute all metrics."""
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, average='macro'),
            'recall': recall_score(self.y_true, self.y_pred, average='macro'),
            'f1': f1_score(self.y_true, self.y_pred, average='macro')
        }
        
        # Per-class metrics
        metrics['per_class'] = {
            'precision': precision_score(self.y_true, self.y_pred, average=None),
            'recall': recall_score(self.y_true, self.y_pred, average=None),
            'f1': f1_score(self.y_true, self.y_pred, average=None)
        }
        
        return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    return plt.gcf()

def plot_roc_curves(y_true, y_probs, class_names):
    """Plot ROC curves for each class."""
    plt.figure(figsize=(10, 8))
    
    # One-hot encode true labels
    y_true_bin = np.eye(len(class_names))[y_true]
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    return plt.gcf()

def plot_pr_curves(y_true, y_probs, class_names):
    """Plot Precision-Recall curves for each class."""
    plt.figure(figsize=(10, 8))
    
    # One-hot encode true labels
    y_true_bin = np.eye(len(class_names))[y_true]
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    return plt.gcf()

def plot_tsne(features, labels, class_names, perplexity=30):
    """Plot t-SNE visualization of features."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    try:
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       label=class_name, alpha=0.6)
        
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.title('t-SNE visualization of features')
        plt.legend()
        return plt.gcf()
    except Exception as e:
        print(f"Error in t-SNE visualization: {e}")
        return None

def metrics_np(y_pred, y_true, b_auc=True):
    """Compute metrics using numpy arrays.
    
    Args:
        y_pred (np.ndarray): Predicted labels
        y_true (np.ndarray): True labels
        b_auc (bool): Whether to compute AUC (requires probability scores)
    
    Returns:
        dict: Dictionary containing the computed metrics
    """
    metrics_dict = {}
    
    # Ensure inputs are numpy arrays
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    # Basic metrics
    metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
    metrics_dict['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics_dict['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics_dict['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return metrics_dict


def compute_per_class_metrics(y_true, y_pred, class_names):
    """Compute per-class metrics and return as table data.
    
    Returns:
        list: Table data with columns [Class, Precision, Recall, F1]
    """
    f1 = f1_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    table = [[cls, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"] for cls, p, r, f in zip(class_names, precision, recall, f1)]
    return table


def compute_recall_at_k(embeddings, labels, k=1):
    """
    Compute Recall@K for given embeddings and labels using cosine similarity.
    
    Args:
        embeddings (np.ndarray): Feature embeddings (N, D)
        labels (np.ndarray): Ground truth labels (N,)
        k (int): Number of nearest neighbors to consider
    
    Returns:
        float: Recall@K score
    """
    from sklearn.neighbors import NearestNeighbors
    
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    # Fit Nearest Neighbors with cosine distance (exclude self-match)
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings)

    # Count correct labels among nearest neighbors (exclude self-match at idx 0)
    correct = 0
    for i in range(len(labels)):
        neighbor_idxs = indices[i][1:]  # exclude self
        neighbor_labels = labels[neighbor_idxs]
        if labels[i] in neighbor_labels:
            correct += 1

    recall_at_k = correct / len(labels)
    return recall_at_k
