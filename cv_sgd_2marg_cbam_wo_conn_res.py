"""
ResNet50 Classification Training Script with Cross-Validation
CSV-based fold loading, local logging, SGD optimizer
Teacher: No CBAM, no connector
"""

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Sampler
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                              accuracy_score, roc_auc_score)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.amp import autocast, GradScaler

# PyTorch Metric Learning
from pytorch_metric_learning import losses, miners, distances, reducers

# Local imports
from models.classifier_factory import create_classifier
from utils.tools import load_config, plot_confusion_matrix_fig
from utils.metrics import compute_per_class_metrics
from utils.logger import ExperimentLogger
from utils.cbam import CBAM

#####################################################

#####################################################
# Feature Connector for Distillation
#####################################################

def build_feature_connector(t_channel, s_channel):
    """Build 1x1 conv + BN connector to adapt student features to teacher channel dimensions"""
    import math
    
    connector = nn.Sequential(
        nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(t_channel)
    )
    
    # Initialize weights (Kaiming normal for conv, 1/0 for BN)
    for m in connector:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    
    return connector


def generate_fundus_mask_batch(images, target_sizes=None):
    """
    Generate fundus region masks using FundusCrop morphological approach.
    Matches the preprocessing used for training data.
    
    Args:
        images: Input tensor [B, 3, H, W] (normalized ImageNet)
        target_sizes: List of (H, W) tuples for each feature layer, e.g., [(128, 128), (64, 64), (32, 32)]
                     If None, returns mask at input resolution
    
    Returns:
        If target_sizes is None: Single mask tensor [B, 1, H, W]
        Otherwise: List of mask tensors at each target size
    """
    device = images.device
    batch_size, _, H, W = images.shape
    
    # Denormalize to [0, 255]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1) * 255.0
    
    # Process each image in batch
    masks_list = []
    for i in range(batch_size):
        # Extract green channel and convert to uint8
        img_np = images_denorm[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        green_channel = img_np[:, :, 1]
        
        # Apply morphological closing to fill gaps (same as FundusCrop)
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(green_channel, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to remove small noise
        dilated = cv2.dilate(closed, kernel, iterations=3)
        
        # Threshold to binarize
        _, thresh = cv2.threshold(dilated, 20, 255, cv2.THRESH_BINARY)
        
        # Remove noise with opening and erosion
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.erode(cleaned, kernel, iterations=4)
        
        # Find contours to identify fundus region
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask by filling the largest contour
        mask = np.zeros((H, W), dtype=np.uint8)
        if contours:
            # Filter out small contours (noise)
            min_area = 2000
            contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if contours:
                # Fill the largest contour region
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # If no valid contour found, fill entire image (fallback)
        if mask.sum() == 0:
            mask = np.ones((H, W), dtype=np.uint8) * 255
        
        masks_list.append(mask)
    
    # Convert to tensor [B, 1, H, W] and normalize to [0, 1]
    masks = torch.from_numpy(np.stack(masks_list)).unsqueeze(1).float().to(device) / 255.0
    
    if target_sizes is None:
        return masks
    
    # Generate masks at multiple resolutions
    multi_res_masks = []
    for target_h, target_w in target_sizes:
        mask_resized = torch.nn.functional.interpolate(
            masks, size=(target_h, target_w), mode='nearest'
        )
        multi_res_masks.append(mask_resized)
    
    return multi_res_masks


#####################################################
# Load Configuration
#####################################################

config = load_config('./configs/cv_sgd_2marg_res.yaml')

# Initialize logger
logger = ExperimentLogger(
    run_name=config['logging']['run_name'], 
    log_dir=config['logging']['log_dir'], 
    config=config
)

print(f"Configuration loaded successfully")
print(f"Data directory: {config['data']['data_dir']}")
print(f"CSV path: {config['data']['csv_path']}")
print(f"Number of folds: {config['data']['n_folds']}")
print(f"Epochs: {config['training']['epochs']}")
print(f"Batch size: {config['training']['batch_size']}")
print(f"Learning rate: {config['training']['lr']}")
print(f"Optimizer: SGD (momentum={config['training']['momentum']})")

# Data transforms using Albumentations
train_transform = A.Compose([
    A.Resize(*config['data']['input_size']),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(*config['data']['input_size']),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# CSV Dataset class
class CSVDataset(Dataset):
    """Dataset that loads images from CSV with columns: path, label, fold"""
    
    def __init__(self, df, data_dir, transform=None, return_fundus_mask=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.return_fundus_mask = return_fundus_mask
        
        # Precompute fundus masks if needed (before transform, at original size)
        self.fundus_masks = None
        if self.return_fundus_mask:
            print(f"  Precomputing fundus masks for {len(self.df)} images...")
            self.fundus_masks = self._precompute_fundus_masks()
            print(f"  ✅ Fundus masks precomputed")
    
    def _precompute_fundus_masks(self):
        """Precompute fundus masks using FundusCrop morphological approach"""
        masks = []
        kernel = np.ones((7, 7), np.uint8)
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            ubuntu_path = row['path'].replace('\\', '/')
            img_path = os.path.join(self.data_dir, ubuntu_path)
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                # Create full mask as fallback
                masks.append(np.ones((512, 512), dtype=np.float32))
                continue
            
            # Extract green channel
            green_channel = image[:, :, 1]
            
            # Apply morphological operations (FundusCrop approach)
            closed = cv2.morphologyEx(green_channel, cv2.MORPH_CLOSE, kernel)
            dilated = cv2.dilate(closed, kernel, iterations=3)
            _, thresh = cv2.threshold(dilated, 20, 255, cv2.THRESH_BINARY)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.erode(cleaned, kernel, iterations=4)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask by filling largest contour
            mask = np.zeros(green_channel.shape, dtype=np.uint8)
            if contours:
                contours = [c for c in contours if cv2.contourArea(c) > 2000]
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Fallback: fill entire image if no contour found
            if mask.sum() == 0:
                mask.fill(255)
            
            # Normalize to [0, 1] and store as float32
            masks.append(mask.astype(np.float32) / 255.0)
        
        return masks
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ubuntu_path = row['path'].replace('\\', '/')
        img_path = os.path.join(self.data_dir, ubuntu_path)
        label = int(row['label'])
        
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get precomputed mask (before transform)
        fundus_mask = None
        if self.return_fundus_mask and self.fundus_masks is not None:
            fundus_mask = self.fundus_masks[idx]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
            # Apply same transform to mask
            if fundus_mask is not None:
                # Resize mask to match transformed image size
                mask_resized = cv2.resize(fundus_mask, 
                                        (image.shape[2], image.shape[1]), 
                                        interpolation=cv2.INTER_NEAREST)
                fundus_mask = torch.from_numpy(mask_resized).unsqueeze(0)  # [1, H, W]
        
        if self.return_fundus_mask:
            return image, label, fundus_mask
        else:
            return image, label

# Load data from CSV
df = pd.read_csv(config['data']['csv_path'])
print(f"\n{'='*60}")
print(f"Loaded {len(df)} samples from CSV")
print(f"Fold distribution:")
print(df['fold'].value_counts().sort_index())
print(f"Label distribution:")
print(df['label'].value_counts().sort_index())
print(f"{'='*60}\n")

# Get class information
class_names = sorted(df['label'].unique())
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}\n")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

#####################################################
# Loss Functions
#####################################################

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class OrdinalTripletLoss:
    """Ordinal-aware triplet loss using pytorch_metric_learning"""
    
    def __init__(self, margin_diff1, margin_diff2,
                 mining_strategy='semihard', verbose=False, print_freq=10):
        # Dynamic margins based on ordinal gap |anchor - negative|
        self.margin_diff1 = float(margin_diff1)
        self.margin_diff2 = float(margin_diff2)
        self.mining_strategy = mining_strategy
        self.verbose = verbose
        self.print_freq = print_freq
        self.call_count = 0

        if self.margin_diff1 <= 0 or self.margin_diff2 <= 0:
            raise ValueError("ordinal_margin_diff1 and ordinal_margin_diff2 must be > 0")

        if self.margin_diff2 < self.margin_diff1:
            print(f"⚠️  margin_diff2 ({self.margin_diff2:.3f}) < margin_diff1 ({self.margin_diff1:.3f}).")
            print("    For ordinal consistency, consider margin_diff2 >= margin_diff1.")
        
        # Use CosineSimilarity distance
        self.distance = distances.CosineSimilarity()
        
        # Setup miner based on strategy
        if mining_strategy == 'hard':
            self.miner = miners.BatchEasyHardMiner(
                pos_strategy="hard",  # Use easy positives (far from anchor)
                neg_strategy="hard",  # Use semi-hard negatives
                distance=self.distance
            )
            # Mine hardest positives and hardest negatives
            # self.miner = miners.BatchHardMiner(distance=self.distance)
        elif mining_strategy == 'semihard':
            # Mine easy positives and semi-hard negatives (recommended)
            self.miner = miners.BatchEasyHardMiner(
                pos_strategy="easy",  # Use easy positives (far from anchor)
                neg_strategy="semihard",  # Use semi-hard negatives
                distance=self.distance
            )
        elif mining_strategy == 'all':
            # Mine all valid triplets
            self.miner = miners.BatchAllMiner()
        else:
            raise ValueError(f"Unknown mining_strategy: {mining_strategy}")

    def _margin_from_gap(self, label_gap):
        """Map ordinal label gap to margin value for 3-class setting."""
        return torch.where(
            label_gap >= 2,
            torch.tensor(self.margin_diff2, device=label_gap.device, dtype=torch.float32),
            torch.tensor(self.margin_diff1, device=label_gap.device, dtype=torch.float32)
        )
    
    def __call__(self, embeddings, labels):
        """
        Compute ordinal triplet loss
        
        Args:
            embeddings: [batch_size, embedding_dim] tensor
            labels: [batch_size] tensor of class labels
        
        Returns:
            triplet_loss: scalar tensor
        """
        # Mine hard triplets
        hard_pairs = self.miner(embeddings, labels)
        
        # Check if valid triplets were found
        num_triplets = len(hard_pairs[0])
        
        if num_triplets == 0:
            # No valid triplets found, return zero loss
            if self.verbose:
                print("⚠️  No valid triplets found in this batch")
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Print distance statistics periodically
        self.call_count += 1
        if self.verbose and (self.call_count % self.print_freq == 0):
            self._print_triplet_stats(embeddings, labels, hard_pairs)

        # Unpack miner output: (anchor1, positive, anchor2, negative)
        a1_idx, p_idx, a2_idx, n_idx = hard_pairs
        anchor_idx = a1_idx
        positive_idx = p_idx
        negative_idx = n_idx

        # Gather embeddings
        a_emb = embeddings[anchor_idx]
        p_emb = embeddings[positive_idx]
        n_emb = embeddings[negative_idx]

        # Cosine similarities (higher means closer)
        sim_ap = torch.nn.functional.cosine_similarity(a_emb, p_emb, dim=1)
        sim_an = torch.nn.functional.cosine_similarity(a_emb, n_emb, dim=1)

        # Dynamic margin by ordinal label difference: |a-n|=1 -> m1, |a-n|=2 -> m2
        label_gap = torch.abs(labels[anchor_idx] - labels[negative_idx]).float()
        dynamic_margin = self._margin_from_gap(label_gap)

        # Ordinal triplet hinge in similarity space
        # Constraint: sim(a,p) >= sim(a,n) + margin(gap)
        triplet_losses = torch.relu(sim_an - sim_ap + dynamic_margin)

        # Match AvgNonZeroReducer behavior: average only non-zero losses
        valid_mask = triplet_losses > 0
        if valid_mask.any():
            loss = triplet_losses[valid_mask].mean()
        else:
            loss = triplet_losses.sum() * 0.0
        
        return loss
    
    def _print_triplet_stats(self, embeddings, labels, triplets):
        """Print statistics about mined triplets and distances"""
        # Unpack miner output: (anchor1, positive, anchor2, negative)
        # BatchEasyHardMiner returns 4 values, not 3!
        # anchor1 is for (a,p) pairs, anchor2 is for (a,n) pairs
        # For semihard strategy, they're usually the same
        try:
            a1_idx, p_idx, a2_idx, n_idx = triplets
            # Use a1_idx as the anchor (in semihard, a1_idx == a2_idx)
            anchor_idx = a1_idx
            positive_idx = p_idx
            negative_idx = n_idx
        except Exception as e:
            print(f"⚠️  Error unpacking triplets: {e}")
            print(f"    Triplets type: {type(triplets)}, length: {len(triplets) if hasattr(triplets, '__len__') else 'N/A'}")
            return
        
        # Compute all similarities
        ap_similarities = []  # anchor-positive similarities
        an_similarities = []  # anchor-negative similarities
        violations = 0
        
        for i in range(len(anchor_idx)):
            a_idx = anchor_idx[i].item()
            p_idx = positive_idx[i].item()
            n_idx = negative_idx[i].item()
            
            # Compute cosine similarity (range [-1, 1])
            sim_ap = self.distance(embeddings[a_idx].unsqueeze(0), 
                                   embeddings[p_idx].unsqueeze(0)).item()
            sim_an = self.distance(embeddings[a_idx].unsqueeze(0), 
                                   embeddings[n_idx].unsqueeze(0)).item()
            
            ap_similarities.append(sim_ap)
            an_similarities.append(sim_an)

            # Check violation with margin based on ordinal label gap
            label_gap = abs(labels[a_idx].item() - labels[n_idx].item())
            margin_i = self.margin_diff2 if label_gap >= 2 else self.margin_diff1
            if sim_ap < sim_an + margin_i:
                violations += 1
        
        ap_similarities = np.array(ap_similarities)
        an_similarities = np.array(an_similarities)
        gap = ap_similarities - an_similarities
        
        print(f"\n{'='*70}")
        print(f"Ordinal Triplet Loss Statistics (Batch {self.call_count})")
        print(f"{'='*70}")
        print(f"Margins: diff1={self.margin_diff1:.3f}, diff2={self.margin_diff2:.3f} | Mining: {self.mining_strategy} | Triplets: {len(anchor_idx)}")
        print(f"\n📊 Similarity Statistics (CosineSimilarity, range [-1, 1]):")
        print(f"  sim(anchor, positive) - SAME class (should be HIGH):")
        print(f"    Min: {ap_similarities.min():.4f} | Mean: {ap_similarities.mean():.4f} | Max: {ap_similarities.max():.4f}")
        print(f"  sim(anchor, negative) - DIFF class (should be LOW):")
        print(f"    Min: {an_similarities.min():.4f} | Mean: {an_similarities.mean():.4f} | Max: {an_similarities.max():.4f}")
        print(f"  Gap [sim(a,p) - sim(a,n)]:")
        print(f"    Min: {gap.min():.4f} | Mean: {gap.mean():.4f} | Max: {gap.max():.4f}")
        print(f"\n🎯 Constraint Check [sim(a,p) >= sim(a,n) + margin(|a-n|)]:")
        print(f"  Satisfied: {len(anchor_idx) - violations}/{len(anchor_idx)}")
        print(f"  Violations: {violations}/{len(anchor_idx)} ({100*violations/len(anchor_idx):.1f}%)")
        
        # Show first 3 triplets in detail
        print(f"\n📋 Sample Triplets (first 3):")
        for i in range(min(3, len(anchor_idx))):
            a_idx = anchor_idx[i].item()
            p_idx = positive_idx[i].item()
            n_idx = negative_idx[i].item()
            
            a_label = labels[a_idx].item()
            p_label = labels[p_idx].item()
            n_label = labels[n_idx].item()
            gap_label = abs(a_label - n_label)
            margin_i = self.margin_diff2 if gap_label >= 2 else self.margin_diff1
            
            sim_ap = ap_similarities[i]
            sim_an = an_similarities[i]
            satisfied = "✅" if sim_ap >= sim_an + margin_i else "❌"
            
            print(f"  {i+1}. Indices({a_idx},{p_idx},{n_idx}) | Labels({a_label},{p_label},{n_label})")
            print(f"     sim(a,p)={sim_ap:.4f}, sim(a,n)={sim_an:.4f}, gap={sim_ap-sim_an:.4f}, m={margin_i:.3f} {satisfied}")
        
        # Margin recommendations
        print(f"\n💡 Margin Tuning Recommendations:")
        effective_margin = 0.5 * (self.margin_diff1 + self.margin_diff2)
        if gap.mean() > effective_margin * 2:
            print(f"  ⬆️  Gap mean ({gap.mean():.3f}) >> avg margin ({effective_margin:.3f})")
            print(f"     Consider INCREASING margin to {gap.mean()*0.6:.3f}-{gap.mean()*0.8:.3f}")
        elif gap.mean() < effective_margin * 0.5:
            print(f"  ⬇️  Gap mean ({gap.mean():.3f}) << avg margin ({effective_margin:.3f})")
            print(f"     Consider DECREASING margin to {gap.mean()*1.2:.3f}-{gap.mean()*1.5:.3f}")
        else:
            print(f"  ✅ Current margins (m1={self.margin_diff1:.3f}, m2={self.margin_diff2:.3f}) look reasonable!")
            print(f"     Gap mean = {gap.mean():.3f} (within 0.5-2x avg margin range)")
        
        print(f"{'='*70}\n")


def distillation_loss_encoder_decoder_fusion(student_features, teacher_features, student_connectors,
                                             teacher_connectors,
                                             teacher_cbam_modules, student_cbam_modules,
                                             layer_mapping, device, fundus_masks=None,
                                             save_qualitative=False,
                                             qualitative_dir=None,
                                             qualitative_tag=None):
    """
    Novel encoder-decoder fusion distillation loss:
    Concatenates teacher's encoder + decoder features, applies CBAM, then distills to student
    
    Args:
        student_features: Dict of student feature maps
        teacher_features: Dict of teacher feature maps (contains both encoder and decoder)
        student_connectors: nn.ModuleList of 1x1 conv adapters for student features
        teacher_connectors: nn.ModuleList of 1x1 conv adapters for teacher features
        teacher_cbam_modules: Dict of CBAM modules for teacher fused features
        student_cbam_modules: Dict of CBAM modules for student features
        layer_mapping: List of (student_key, teacher_enc_key, teacher_dec_key, connector_idx) tuples.
            If teacher_dec_key is None, distill against teacher_enc_key only (single-target mode).
        device: torch device
        fundus_masks: Optional dict of masks {layer_key: mask_tensor} to filter background attention
        save_qualitative: If True, save teacher flow visualizations (raw/masked/after_cbam)
        qualitative_dir: Directory to save qualitative visualizations
        qualitative_tag: Prefix tag for saved filenames (e.g., epoch-batch)
    
    Returns:
        AttnFD-style normalized MSE loss summed across mapped layers
    """
    def _feature_to_vis_map(feature_tensor):
        """Convert [B, C, H, W] to a normalized 2D map for qualitative visualization."""
        fmap = feature_tensor[0].detach().float().abs().mean(dim=0).cpu().numpy()
        if fmap.max() > fmap.min():
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
        else:
            fmap = np.zeros_like(fmap)
        return fmap

    def _save_teacher_flow_qualitative(raw_feat, masked_feat, cbam_feat, layer_key):
        if not save_qualitative or qualitative_dir is None:
            return

        os.makedirs(qualitative_dir, exist_ok=True)

        raw_map = _feature_to_vis_map(raw_feat)
        masked_map = _feature_to_vis_map(masked_feat)
        cbam_map = _feature_to_vis_map(cbam_feat)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_map, cmap='jet')
        axes[0].set_title('Teacher Raw')
        axes[0].axis('off')

        axes[1].imshow(masked_map, cmap='jet')
        axes[1].set_title('Teacher Masked')
        axes[1].axis('off')

        axes[2].imshow(cbam_map, cmap='jet')
        axes[2].set_title('Teacher After CBAM')
        axes[2].axis('off')

        plt.tight_layout()
        tag = qualitative_tag if qualitative_tag is not None else 'step'
        save_path = os.path.join(qualitative_dir, f"{tag}_{layer_key}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    total_loss = torch.tensor(0.0, device=device)
    
    for s_key, t_enc_key, t_dec_key, conn_idx in layer_mapping:
        has_teacher_target = t_enc_key in teacher_features and (t_dec_key is None or t_dec_key in teacher_features)
        if s_key in student_features and has_teacher_target:
            # Get features
            s_feat = student_features[s_key]
            t_enc_feat = teacher_features[t_enc_key]
            if t_dec_key is None:
                t_target = t_enc_feat
                cbam_key = t_enc_key
            else:
                t_dec_feat = teacher_features[t_dec_key]
                # Concatenate teacher encoder + decoder features
                # Both should have same spatial dimensions
                t_target = torch.cat([t_enc_feat, t_dec_feat], dim=1)  # [B, C_enc+C_dec, H, W]
                cbam_key = f"{t_enc_key}_{t_dec_key}"

            t_target_raw = t_target
            
            # Apply fundus mask to filter out corner/background attention (if provided)
            if fundus_masks is not None and s_key in fundus_masks:
                mask = fundus_masks[s_key]  # [B, 1, H, W]
                # Resize mask to match feature spatial size
                if mask.shape[2:] != t_target.shape[2:]:
                    mask = torch.nn.functional.interpolate(
                        mask, size=t_target.shape[2:], mode='nearest'
                    )
                t_target = t_target * mask  # Zero out features outside fundus region

            t_target_masked = t_target
            
            # Apply CBAM to teacher target features (optional)
            if teacher_cbam_modules is None:
                t_fused_attn = t_target
            elif cbam_key in teacher_cbam_modules:
                t_fused_attn = teacher_cbam_modules[cbam_key](t_target)
            else:
                raise ValueError(f"CBAM module for teacher key '{cbam_key}' not found in teacher_cbam_modules")

            # Apply connector to teacher branch after attention (optional)
            if teacher_connectors is not None:
                t_fused_attn = teacher_connectors[conn_idx](t_fused_attn)

            _save_teacher_flow_qualitative(
                t_target_raw,
                t_target_masked,
                t_fused_attn,
                s_key
            )
            
            # Student branch order matches AttnFD-style adaptation path: CBAM -> connector
            if student_cbam_modules is not None and s_key in student_cbam_modules:
                s_feat_attn = student_cbam_modules[s_key](s_feat)
            else:
                s_feat_attn = s_feat
            
            # Apply connector to adapt student channels to match fused teacher channels
            s_feat_adapted = student_connectors[conn_idx](s_feat_attn)
            
            # Resize spatial dimensions if needed (bilinear interpolation like AttnFD)
            if s_feat_adapted.shape[2:] != t_fused_attn.shape[2:]:
                s_feat_adapted = torch.nn.functional.interpolate(
                    s_feat_adapted, 
                    size=t_fused_attn.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            batch_size = t_fused_attn.shape[0]
            
            # AttnFD loss: global L2 normalization for each feature tensor
            s_norm = s_feat_adapted / torch.norm(s_feat_adapted, p=2)
            t_norm = t_fused_attn / torch.norm(t_fused_attn, p=2)
            
            # Compute MSE-like loss (sum of squared differences / batch_size)
            layer_loss = (s_norm - t_norm).pow(2).sum() / batch_size
            total_loss += layer_loss
            
            # Clean up intermediate tensors
            del s_feat_adapted, t_fused_attn, s_norm, t_norm
    
    # Match AttnFD aggregation: sum over selected layers (no averaging across layer count)
    return total_loss


# Balanced Batch Sampler
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = len(set(labels))
        self.samples_per_class = batch_size // self.num_classes

    def __iter__(self):
        label_to_indices = {label: [] for label in set(self.labels)}
        for idx, label in enumerate(self.labels):
            label_to_indices[label].append(idx)
        
        for label in label_to_indices:
            np.random.shuffle(label_to_indices[label])
        
        batches = []
        while any(len(indices) >= self.samples_per_class for indices in label_to_indices.values()):
            batch = []
            for label in label_to_indices:
                if len(label_to_indices[label]) >= self.samples_per_class:
                    batch.extend(label_to_indices[label][:self.samples_per_class])
                    label_to_indices[label] = label_to_indices[label][self.samples_per_class:]
            if len(batch) > 0:
                batches.append(batch)
        
        for batch in batches:
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size


#####################################################
# Training Function for Single Fold
#####################################################

def train_fold(fold, train_dataset, val_dataset, config, logger, device):
    """Train one fold of cross-validation"""
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}/{config['data']['n_folds']-1}")
    print(f"{'='*60}")
    
    fold_start_time = time.time()
    
    # Get labels for samplers
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    train_class_counts = Counter(train_labels)
    
    print(f"\nFold {fold} - Train: {len(train_labels)} samples")
    for label in sorted(train_class_counts.keys()):
        print(f"  Class {label}: {train_class_counts[label]} samples")
    
    # Create data loaders
    sampler_type = config['sampler']['sampler_type']
    
    if sampler_type == "weighted_random":
        class_weights = {label: len(train_labels) / (num_classes * count) 
                        for label, count in train_class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                                 sampler=sampler, num_workers=config['training']['num_workers'],
                                 pin_memory=config['training']['pin_memory'])
    elif sampler_type == "balanced_batch":
        sampler = BalancedBatchSampler(train_labels, config['training']['batch_size'])
        train_loader = DataLoader(train_dataset, batch_sampler=sampler,
                                 num_workers=config['training']['num_workers'],
                                 pin_memory=config['training']['pin_memory'])
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                                 shuffle=True, num_workers=config['training']['num_workers'],
                                 pin_memory=config['training']['pin_memory'])
    
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                           shuffle=False, num_workers=config['training']['num_workers'],
                           pin_memory=config['training']['pin_memory'])
    
    # Create model
    model_name = config['model'].get('model_name', 'resnet50')
    model = create_classifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=config['model']['dropout_rate'],
        freeze_backbone=config['model']['freeze_backbone'],
        custom_head=config['model']['custom_head']
    ).to(device)
    
    # Loss criterion
    loss_type = config['loss']['loss_type']
    if loss_type == "weighted_ce":
        weights = torch.tensor([len(train_labels) / (num_classes * train_class_counts[l]) 
                               for l in range(num_classes)], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_type == "focal":
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optional: Ordinal loss
    ordinal_criterion = None
    if config['ordinal']['use_ordinal_loss']:
        ordinal_criterion = OrdinalTripletLoss(
            margin_diff1=config['ordinal']['ordinal_margin_diff1'],
            margin_diff2=config['ordinal']['ordinal_margin_diff2'],
            mining_strategy=config['ordinal']['ordinal_mining_strategy'],
            verbose=config['ordinal'].get('verbose', False),
            print_freq=config['ordinal'].get('print_freq', 10)
        )
    
    # Optional: Teacher model for distillation
    teacher_model = None
    student_connectors = None
    teacher_connectors = None
    layer_mapping = None
    
    if config['distillation']['use_distillation']:
        print(f"Loading teacher model for knowledge distillation...")
        print(f"  Teacher: {config['distillation']['teacher_model_name']}")
        print(f"  Checkpoint: {config['distillation']['teacher_checkpoint']}")
        
        import sys
        avseg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AV-seg'))
        if avseg_path not in sys.path:
            sys.path.insert(0, avseg_path)
        
        try:
            from models.model_implements import get_model  #type: ignore
            
            # Create teacher model with its original architecture (4 classes from segmentation)
            # We only use encoder-decoder features, not the final classifier
            teacher_num_classes = config['distillation'].get('teacher_num_classes', 4)
            teacher_model_config = {
                'architecture': config['distillation']['teacher_model_name'],
                'input_channels': 3,
                'num_classes': teacher_num_classes,
                'input_size': config['distillation']['teacher_input_size']
            }
            teacher_model = get_model(teacher_model_config)
            
            if os.path.exists(config['distillation']['teacher_checkpoint']):
                checkpoint = torch.load(config['distillation']['teacher_checkpoint'], map_location=device)
                
                # Get state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Filter out profiling keys (total_ops, total_params) from profiling tools
                state_dict_cleaned = {k: v for k, v in state_dict.items() 
                                     if not ('total_ops' in k or 'total_params' in k)}
                
                # Load weights with strict=False to handle architecture mismatches
                missing_keys, unexpected_keys = teacher_model.load_state_dict(state_dict_cleaned, strict=False)
                
                if missing_keys:
                    print(f"  ⚠️ Missing keys in teacher checkpoint: {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"  ⚠️ Unexpected keys in teacher checkpoint: {len(unexpected_keys)} keys")
                
                teacher_model = teacher_model.to(device)
                teacher_model.eval()
                for param in teacher_model.parameters():
                    param.requires_grad = False
                print(f"  ✅ Teacher model loaded successfully (using hybrid KD features)")
                
                # Initialize encoder-decoder fusion distillation
                # Layer mapping: (student_key, teacher_enc_key, teacher_dec_key, connector_idx)
                # Teacher (AttentionUNet): att2=128, enc3+dec3=512, enc4+dec4=1024
                # Student (ResNet50): enc2=256, enc3=512, enc4=1024
                layer_mapping = [
                    ('enc2', 'enc2_att', None, 0),        # Student 256 ← Teacher att2 (128)
                    ('enc3', 'enc3', 'dec3', 1),          # Student 512 ← Teacher (256+256=512)
                    ('enc4', 'enc4', 'dec4', 2),          # Student 1024 ← Teacher (512+512=1024)
                ]
                
                # Connector configs: adapt student → fused teacher
                student_connector_configs = [
                    (128, 256),    # att2: teacher=128, student enc2=256
                    (512, 512),    # enc3+dec3: fused_teacher=512, student=512
                    (1024, 1024),  # enc4+dec4: fused_teacher=1024, student=1024
                ]
                
                student_connectors = nn.ModuleList([
                    build_feature_connector(t_ch, s_ch) for t_ch, s_ch in student_connector_configs
                ]).to(device)

                # Teacher connectors: adapt teacher features after CBAM (same channel sizes by default)
                # teacher_connector_configs = [
                #     (128, 128),
                #     (512, 512),
                #     (1024, 1024),
                # ]
                # teacher_connectors = nn.ModuleList([
                #     build_feature_connector(t_ch, s_ch) for t_ch, s_ch in teacher_connector_configs
                # ]).to(device)
                # teacher_connectors = None
                # Initialize CBAM modules for teacher (fused features)
                # teacher_cbam_modules = nn.ModuleDict({
                #     'enc2_att': CBAM(128),      # For teacher att2 (single-target)
                #     'enc3_dec3': CBAM(512),    # For concatenated enc3+dec3
                #     'enc4_dec4': CBAM(1024),   # For concatenated enc4+dec4
                # }).to(device)
                

                teacher_cbam_modules = None  # Disable teacher CBAM 
                
                # Initialize CBAM modules for student features
                student_cbam_modules = nn.ModuleDict({
                    'enc2': CBAM(256),    # For student enc2
                    'enc3': CBAM(512),    # For student enc3
                    'enc4': CBAM(1024),   # For student enc4
                }).to(device)
                
                print(f"  ✅ Novel encoder-decoder fusion distillation enabled")
                print(f"  ✅ Initialized {len(student_connectors)} student feature connectors")
                print(f"  ✅ Teacher feature connectors disabled")
                # print(f"  ✅ Initialized {len(teacher_cbam_modules)} teacher CBAM modules")
                print(f"  ✅ Initialized {len(student_cbam_modules)} student CBAM modules")
                mapping_desc = [f"{s}←{te}" if td is None else f"{s}←({te}+{td})" for s, te, td, _ in layer_mapping]
                print(f"  Layer mapping: {mapping_desc}")
                
            else:
                print(f"  ⚠️ Teacher checkpoint not found, disabling distillation")
                teacher_model = None
        except Exception as e:
            print(f"  ⚠️ Failed to load teacher model: {e}")
            print(f"  Disabling distillation...")
            teacher_model = None
    
    # Optimizer - SGD with momentum (created after connectors to include them if needed)
    optimizer_params = [{'params': model.parameters(), 'lr': config['training']['lr']}]
    
    # Add connectors and CBAM modules with higher learning rate if distillation is enabled
    if student_connectors is not None:
        optimizer_params.append({
            'params': student_connectors.parameters(),
            'lr': config['training']['lr'] * 10  # Higher LR for connectors like AttnFD
        })

    if teacher_connectors is not None:
        optimizer_params.append({
            'params': teacher_connectors.parameters(),
            'lr': config['training']['lr'] * 10
        })
    
    if config['distillation']['use_distillation'] and teacher_model is not None:
        # Add teacher CBAM modules (for fusion)
        if 'teacher_cbam_modules' in locals() and teacher_cbam_modules is not None:
            optimizer_params.append({
                'params': teacher_cbam_modules.parameters(),
                'lr': config['training']['lr'] * 10
            })
        # Add student CBAM modules
        if 'student_cbam_modules' in locals() and student_cbam_modules is not None:
            optimizer_params.append({
                'params': student_cbam_modules.parameters(),
                'lr': config['training']['lr'] * 10
            })
    
    optimizer = optim.SGD(
        optimizer_params,
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning Rate Scheduler
    scheduler = None
    if config['training'].get('use_scheduler', False):
        scheduler_type = config['training'].get('scheduler_type', 'cosine')
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['training'].get('scheduler_t_max', config['training']['epochs']),
                eta_min=config['training'].get('scheduler_eta_min', 1e-5)
            )
            print(f"  ✅ Using CosineAnnealingLR scheduler (T_max={config['training'].get('scheduler_t_max', config['training']['epochs'])}, eta_min={config['training'].get('scheduler_eta_min', 1e-5)})")
        elif scheduler_type == 'step':
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            print(f"  ✅ Using StepLR scheduler (step_size=30, gamma=0.1)")
    
    # AMP scaler
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    # Tracking
    best_val_f1 = -1
    best_epoch = 0
    
    # Get checkpoint directory
    ckpt_dir = logger.get_checkpoint_dir(fold)
    
    #####################################################
    # Training Loop
    #####################################################
    
    for epoch in range(config['training']['epochs']):
        model.train()
        
        # Metrics tracking
        train_total_loss = 0.0
        train_ce_loss_sum = 0.0
        train_ordinal_loss_sum = 0.0
        train_distill_loss_sum = 0.0
        y_true_train, y_pred_train, y_probs_train = [], [], []
        
        loop = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{config['training']['epochs']}")

        # Optional qualitative checks for teacher flow (raw -> masked -> CBAM)
        qualitative_check = config['distillation']['qualitative_check']
        qualitative_every_n_batches = max(1, config['distillation']['qualitative_every_n_batches'])
        qualitative_root = os.path.join(logger.run_dir, 'qualitative_distill', f'fold_{fold}')
        
        # Check if dataset returns masks
        use_fundus_mask = config['distillation'].get('use_distillation', False) and config['distillation'].get('use_fundus_mask', False)
        
        for batch_idx, batch_data in enumerate(loop):
            # Unpack batch (with or without mask)
            if use_fundus_mask:
                inputs, labels, precomputed_masks = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                precomputed_masks = precomputed_masks.to(device)  # [B, 1, H, W]
            else:
                inputs, labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                precomputed_masks = None
            
            optimizer.zero_grad()
            
            with autocast(device_type=device.type, enabled=config['training']['use_amp']):
                outputs = model(inputs)
                ce_loss = criterion(outputs, labels)
                
                # Ordinal loss
                ordinal_loss = torch.tensor(0.0).to(device)
                if ordinal_criterion is not None:
                    # Get 512-dim embeddings from custom head for ordinal learning
                    embeddings = model.get_embedding(inputs)
                    ordinal_loss = ordinal_criterion(embeddings, labels)
                
                # Distillation loss (Encoder-Decoder Fusion with CBAM)
                distill_loss = torch.tensor(0.0).to(device)
                distill_start_epoch = config['distillation']['start_epoch']
                if teacher_model is not None and student_connectors is not None and epoch >= distill_start_epoch:
                    # Get student features as dict
                    student_features_dict = model.forward_features(inputs, return_dict=True)
                    
                    # Get teacher features (optimized: only computes up to dec3)
                    with torch.no_grad():
                        if hasattr(teacher_model, 'extract_features'):
                            _, teacher_features = teacher_model.extract_features(inputs)
                        else:
                            teacher_features = {}
                    
                    # Use precomputed fundus masks (if available)
                    fundus_masks = None
                    if config['distillation'].get('use_fundus_mask', False) and precomputed_masks is not None:
                        # Resize precomputed masks to each distillation layer resolution
                        fundus_masks = {}
                        for s_key, _, _, _ in layer_mapping:
                            if s_key in student_features_dict:
                                h, w = student_features_dict[s_key].shape[2:]
                                # Resize mask to match feature spatial size
                                mask_resized = torch.nn.functional.interpolate(
                                    precomputed_masks, size=(h, w), mode='nearest'
                                )
                                fundus_masks[s_key] = mask_resized
                    
                    # Compute encoder-decoder fusion distillation loss
                    save_qualitative_now = qualitative_check and (batch_idx % qualitative_every_n_batches == 0) and (epoch > 149)
                    qualitative_tag = f"epoch{epoch+1:03d}_batch{batch_idx:04d}"
                    distill_loss = distillation_loss_encoder_decoder_fusion(
                        student_features_dict, teacher_features,
                        student_connectors, teacher_connectors,
                        teacher_cbam_modules, student_cbam_modules,
                        layer_mapping, device, fundus_masks,
                        save_qualitative=save_qualitative_now,
                        qualitative_dir=qualitative_root,
                        qualitative_tag=qualitative_tag
                    )

                    if save_qualitative_now:
                        print(f"[Qualitative] Saved teacher flow maps to: {qualitative_root} ({qualitative_tag})")
                
                # Combined loss
                total_loss = (config['loss_weights']['lambda1'] * ce_loss +
                            config['loss_weights']['lambda2'] * ordinal_loss +
                            config['loss_weights']['lambda3'] * distill_loss)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            train_total_loss += total_loss.item() * inputs.size(0)
            train_ce_loss_sum += ce_loss.item() * inputs.size(0)
            if ordinal_criterion is not None:
                train_ordinal_loss_sum += ordinal_loss.item() * inputs.size(0)
            if teacher_model is not None:
                train_distill_loss_sum += distill_loss.item() * inputs.size(0)
            
            with torch.no_grad():
                probs = torch.nn.functional.softmax(outputs.float(), dim=1).cpu().numpy()
                _, preds = outputs.max(1)
                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(preds.cpu().numpy())
                y_probs_train.extend(probs)
            
            loop.set_postfix({'loss': total_loss.item()})
        
        # Compute train metrics
        train_total_loss /= len(train_dataset)
        train_ce_loss_sum /= len(train_dataset)
        train_ordinal_loss_sum /= len(train_dataset)
        train_distill_loss_sum /= len(train_dataset)
        
        y_true_train = np.array(y_true_train)
        y_pred_train = np.array(y_pred_train)
        y_probs_train = np.array(y_probs_train)
        
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_prec = precision_score(y_true_train, y_pred_train, average='macro', zero_division=0)
        train_rec = recall_score(y_true_train, y_pred_train, average='macro', zero_division=0)
        train_f1 = f1_score(y_true_train, y_pred_train, average='macro', zero_division=0)
        train_auc = roc_auc_score(y_true_train, y_probs_train, multi_class='ovr')
        
        #####################################################
        # Validation
        #####################################################
        
        model.eval()
        val_total_loss = 0.0
        val_ce_loss_sum = 0.0
        y_true_val, y_pred_val, y_probs_val = [], [], []
        embeddings_val = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast(device_type=device.type, enabled=config['training']['use_amp']):
                    outputs = model(inputs)
                    ce_loss = criterion(outputs, labels)
                    total_loss = ce_loss
                
                val_total_loss += total_loss.item() * inputs.size(0)
                val_ce_loss_sum += ce_loss.item() * inputs.size(0)
                
                probs = torch.nn.functional.softmax(outputs.float(), dim=1).cpu().numpy()
                _, preds = outputs.max(1)
                
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())
                y_probs_val.extend(probs)
                
        
        val_total_loss /= len(val_dataset)
        val_ce_loss_sum /= len(val_dataset)
        
        y_true_val = np.array(y_true_val)
        y_pred_val = np.array(y_pred_val)
        y_probs_val = np.array(y_probs_val)
        
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_prec = precision_score(y_true_val, y_pred_val, average='macro', zero_division=0)
        val_rec = recall_score(y_true_val, y_pred_val, average='macro', zero_division=0)
        val_f1 = f1_score(y_true_val, y_pred_val, average='macro', zero_division=0)
        val_auc = roc_auc_score(y_true_val, y_probs_val, multi_class='ovr')
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}:")
        print(f"  Train - Loss: {train_total_loss:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"  Val   - Loss: {val_total_loss:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Log to CSV
        logger.log_epoch({
            'epoch': epoch + 1,
            'fold': fold,
            'train_total_loss': train_total_loss,
            'train_ce_loss': train_ce_loss_sum,
            'train_ordinal_loss': train_ordinal_loss_sum if ordinal_criterion else None,
            'train_distill_loss': train_distill_loss_sum if teacher_model else None,
            'train_acc': train_acc,
            'train_prec': train_prec,
            'train_rec': train_rec,
            'train_f1': train_f1,
            'train_auc': train_auc,
            'val_total_loss': val_total_loss,
            'val_ce_loss': val_ce_loss_sum,
            'val_acc': val_acc,
            'val_prec': val_prec,
            'val_rec': val_rec,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_val_acc = val_acc
            best_val_prec = val_prec
            best_val_rec = val_rec
            best_val_auc = val_auc
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_auc': val_auc
            }, ckpt_dir / 'best.pt')
            print(f"  ✅ Best model saved (F1: {val_f1:.4f})")
        
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Save last checkpoint
    torch.save({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
        'val_auc': val_auc
    }, ckpt_dir / 'last.pt')
    
    # Generate and save confusion matrix
    cm_fig = plot_confusion_matrix_fig(y_true_val, y_pred_val, [str(c) for c in class_names])
    logger.save_confusion_matrix(cm_fig, fold)
    plt.close(cm_fig)
    
    # Save artifacts
    artifacts_dir = logger.get_artifacts_dir(fold)
    np.save(artifacts_dir / 'val_labels.npy', y_true_val)
    np.save(artifacts_dir / 'val_scores.npy', y_probs_val)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true_val, y_pred_val, average=None, zero_division=0)
    
    fold_train_time = time.time() - fold_start_time
    
    # Log fold summary
    logger.log_fold_summary({
        'fold': fold,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'best_val_auc': best_val_auc,
        'best_val_acc': best_val_acc,
        'best_val_prec': best_val_prec,
        'best_val_rec': best_val_rec,
        'train_time_seconds': fold_train_time,
        'normal_f1': per_class_f1[0] if len(per_class_f1) > 0 else 0,
        'preplus_f1': per_class_f1[1] if len(per_class_f1) > 1 else 0,
        'plus_f1': per_class_f1[2] if len(per_class_f1) > 2 else 0
    })
    
    # Clean up GPU memory before returning
    del model, optimizer, criterion, scaler
    del train_loader, val_loader
    if ordinal_criterion is not None:
        del ordinal_criterion
    if teacher_model is not None:
        del teacher_model, student_connectors, teacher_connectors, teacher_cbam_modules, student_cbam_modules
    torch.cuda.empty_cache()
    
    return {
        'fold': fold,
        'best_val_f1': best_val_f1,
        'best_val_auc': best_val_auc,
        'best_val_acc': best_val_acc,
        'best_val_prec': best_val_prec,
        'best_val_rec': best_val_rec
    }


#####################################################
# Main Cross-Validation Loop
#####################################################

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Cross-Validation Training (SGD Optimizer)")
    print("="*60 + "\n")
    
    fold_results = []
    
    for fold in range(config['data']['n_folds']):
        # Split data based on fold column
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        
        # Create datasets
        use_fundus_mask = config['distillation'].get('use_distillation', False) and config['distillation'].get('use_fundus_mask', False)
        train_dataset = CSVDataset(train_df, config['data']['data_dir'], train_transform, return_fundus_mask=use_fundus_mask)
        val_dataset = CSVDataset(val_df, config['data']['data_dir'], val_transform, return_fundus_mask=False)  # No mask for validation
        
        # Train this fold
        fold_result = train_fold(fold, train_dataset, val_dataset, config, logger, device)
        fold_results.append(fold_result)
        
        print(f"\n✅ Fold {fold} completed - Best Val F1: {fold_result['best_val_f1']:.4f}\n")
        
        # Clean up memory after fold completes
        # del train_dataset, val_dataset
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"🧹 GPU memory cleaned after fold {fold}")
        
        # break # Remove this break to run all folds in actual training
    
    # Compute CV summary
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)
    
    fold_f1s = [r['best_val_f1'] for r in fold_results]
    fold_aucs = [r['best_val_auc'] for r in fold_results]
    fold_accs = [r['best_val_acc'] for r in fold_results]
    
    print(f"\nPer-fold Best Val F1:")
    for i, f1 in enumerate(fold_f1s):
        print(f"  Fold {i}: {f1:.4f}")
    print(f"\nMean F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Mean Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    
    # Save CV summary
    logger.save_cv_summary({
        'mean_val_f1': float(np.mean(fold_f1s)),
        'std_val_f1': float(np.std(fold_f1s)),
        'mean_val_auc': float(np.mean(fold_aucs)),
        'std_val_auc': float(np.std(fold_aucs)),
        'mean_val_acc': float(np.mean(fold_accs)),
        'std_val_acc': float(np.std(fold_accs)),
        'per_fold_results': fold_results
    })
    
    print(f"\n✅ Training completed!")
    print(f"Results saved to: {logger.run_dir}\n")
