import numpy as np
import pandas as pd
import os

def calculate_cv_metrics(saved_model_directory, train_fold, wandb_enabled=False, class_names=None):
    """Read fold metrics CSVs, find best per fold based on val_f1, calculate avg and std across folds."""
    fold_best_metrics = []
    metric_columns = ['val_loss', 'val_f1_score_macro', 'val_precision_macro', 'val_recall_macro', 
                      'val_accuracy', 'val_auc_roc_macro', 'val_auc_pr_macro']  # Adjust based on actual column names

    for fold in range(train_fold):
        csv_path = os.path.join(saved_model_directory, f'fold_{fold}_metrics.csv')
        if not os.path.exists(csv_path):
            print(f"CSV for fold {fold} not found: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        
        # Find the row with the maximum val_f1_score_macro (or adjust metric)
        best_row = df.loc[df['val_f1_score_macro'].idxmax()]
        
        # Extract relevant metrics
        best_metrics = {col: best_row[col] for col in metric_columns if col in df.columns}
        fold_best_metrics.append(best_metrics)
    
    if not fold_best_metrics:
        print("No fold metrics found to calculate CV.")
        return

    # Calculate avg and std for each metric
    metrics_summary = {}
    for metric in metric_columns:
        values = [fold[metric] for fold in fold_best_metrics if metric in fold]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            metrics_summary[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
            print(f"CV {metric}: {metrics_summary[metric]}")

    # Log to WandB if enabled (assume wandb is initialized elsewhere)
    if wandb_enabled:
        import wandb
        wandb.log({
            f'CV Mean {metric}': float(metrics_summary[metric].split(' ± ')[0])
            for metric in metrics_summary
        })
        wandb.log({
            f'CV Std {metric}': float(metrics_summary[metric].split(' ± ')[1])
            for metric in metrics_summary
        })

        # Log best per fold if needed
        wandb.log({
            'CV Best Metrics per Fold': fold_best_metrics
        })

        # Average per-class would require storing per-class in CSV, but assuming not for now

# Example usage (run as script)
if __name__ == "__main__":
    # Replace with actual paths and args
    saved_model_directory = "path/to/saved_model_directory"
    train_fold = 5  # Number of folds
    wandb_enabled = True
    class_names = ['class1', 'class2']  # If needed for per-class
    calculate_cv_metrics(saved_model_directory, train_fold, wandb_enabled, class_names)