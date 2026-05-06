"""
Local logging utilities for training experiments.
Replaces wandb with file-based logging.
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
import numpy as np


class ExperimentLogger:
    """Logger for local experiment tracking with CSV and JSON."""
    
    def __init__(self, run_name, log_dir='./runs', config=None, is_cv=True):
        """
        Initialize experiment logger.
        
        Args:
            run_name (str): Name for this run
            log_dir (str): Base directory for all runs
            config (dict): Configuration dictionary to save
            is_cv (bool): True for cross-validation, False for full training
        """
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.is_cv = is_cv
        
        # Create unique run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = f"{timestamp}_{run_name}"
        self.run_dir = self.log_dir / self.run_id
        
        # Create directory structure
        self.logs_dir = self.run_dir / 'logs'
        self.checkpoints_dir = self.run_dir / 'checkpoints'
        self.cm_dir = self.run_dir / 'cm'
        self.artifacts_dir = self.run_dir / 'artifacts'
        
        for dir_path in [self.run_dir, self.logs_dir, self.checkpoints_dir, self.cm_dir, self.artifacts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.fold_summary_path = self.logs_dir / 'fold_summary.csv'
        self.cv_summary_path = self.logs_dir / 'cv_summary.json'
        
        # Train log fields (per-fold files will be created on demand)
        self.train_log_fields = [
            'epoch', 'fold', 
            'train_total_loss', 'train_ce_loss', 'train_ordinal_loss', 'train_distill_loss',
            'train_acc', 'train_prec', 'train_rec', 'train_f1', 'train_auc',
            'val_total_loss', 'val_ce_loss',
            'val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_auc', 'lr'
        ]
        
        # Track initialized fold log files
        self._initialized_folds = set()
        
        # Initialize fold summary CSV only for CV mode
        if self.is_cv:
            self.fold_summary_fields = [
                'fold', 'best_epoch', 'best_val_f1', 'best_val_auc', 
                'best_val_acc', 'best_val_prec', 'best_val_rec',
                'train_time_seconds',
                'normal_f1', 'preplus_f1', 'plus_f1'
            ]
            
            with open(self.fold_summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fold_summary_fields)
                writer.writeheader()
        
        # Save config
        if config is not None:
            config_path = self.run_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"Experiment: {self.run_id}")
        print(f"Run directory: {self.run_dir}")
        print(f"{'='*60}\n")
    
    def _init_fold_log(self, fold):
        """Initialize train log CSV for a specific fold."""
        if fold not in self._initialized_folds:
            train_log_path = self.logs_dir / f'train_log_fold{fold}.csv'
            with open(train_log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.train_log_fields)
                writer.writeheader()
            self._initialized_folds.add(fold)
    
    def log_epoch(self, epoch_data):
        """
        Log per-epoch metrics to train_log_fold{fold}.csv
        
        Args:
            epoch_data (dict): Dictionary with keys matching train_log_fields
        """
        fold = epoch_data.get('fold', 1)
        self._init_fold_log(fold)
        
        train_log_path = self.logs_dir / f'train_log_fold{fold}.csv'
        with open(train_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.train_log_fields)
            # Fill missing fields with None
            row = {field: epoch_data.get(field, None) for field in self.train_log_fields}
            writer.writerow(row)
    
    def log_fold_summary(self, fold_data):
        """
        Log end-of-fold summary to fold_summary.csv
        
        Args:
            fold_data (dict): Dictionary with keys matching fold_summary_fields
        """
        with open(self.fold_summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fold_summary_fields)
            row = {field: fold_data.get(field, None) for field in self.fold_summary_fields}
            writer.writerow(row)
    
    def save_cv_summary(self, cv_results):
        """
        Save cross-validation summary to cv_summary.json
        
        Args:
            cv_results (dict): Dictionary with mean/std metrics across folds
        """
        with open(self.cv_summary_path, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        print(f"\nCV summary saved to: {self.cv_summary_path}")
    
    def get_checkpoint_dir(self, fold):
        """Get checkpoint directory for a specific fold or full training."""
        if fold is None:
            # Full training (no CV) - use flat structure
            return self.checkpoints_dir
        else:
            # Cross-validation - use fold subdirectories
            fold_ckpt_dir = self.checkpoints_dir / f'fold_{fold}'
            fold_ckpt_dir.mkdir(exist_ok=True)
            return fold_ckpt_dir
    
    def get_artifacts_dir(self, fold):
        """Get artifacts directory for a specific fold or full training."""
        if fold is None:
            # Full training (no CV) - use flat structure
            return self.artifacts_dir
        else:
            # Cross-validation - use fold subdirectories
            fold_artifacts_dir = self.artifacts_dir / f'fold_{fold}'
            fold_artifacts_dir.mkdir(exist_ok=True)
            return fold_artifacts_dir
    
    def save_confusion_matrix(self, fig, fold):
        """Save confusion matrix figure for a fold or full training."""
        if fold is None:
            cm_path = self.cm_dir / 'confusion_matrix.png'
        else:
            cm_path = self.cm_dir / f'fold_{fold}_cm.png'
        fig.savefig(cm_path, dpi=150, bbox_inches='tight')
        return cm_path
    
    def save_training_summary(self, summary_dict):
        """Save single training run summary to training_summary.json."""
        summary_path = self.logs_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_dict, f, indent=2, default=str)
        print(f"Training summary saved to: {summary_path}")
    
    def __str__(self):
        return f"ExperimentLogger(run_id={self.run_id})"
