"""
Model evaluation utilities for abuse detection.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wandb
from sklearn.metrics import (
    classification_report, multilabel_confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    precision_score, recall_score, confusion_matrix
)
from typing import Dict, Any, Optional

from features import FeatureExtractor
from models.baseline import BaselineModel

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluator with optional W&B integration."""

    def __init__(self, use_wandb: bool = True):
        self.model = None
        self.use_wandb = use_wandb

        if self.use_wandb:
            logger.info("W&B integration enabled for evaluation tracking")
        else:
            logger.info("W&B integration disabled")

    def load_model(self, model):
        self.model = model
        logger.info("Model loaded successfully")

        return self

    def evaluate(self, X, y, target_names: Optional[list] = None) -> Dict[str, Any]:
        """Evaluate model and return metrics for multitarget classification."""
        logger.info("Evaluating multitarget model performance...")

        self.X = X
        self.y_true = y.to_numpy()
        self.y_pred = self.model.predict(X)
        self.y_pred_proba = self.model.predict_proba(X)
        self.target_names = target_names

        # Overall multitarget metrics
        auc_score = roc_auc_score(self.y_true, self.y_pred_proba, multi_class='ovr', average='macro')
        ap_score = average_precision_score(self.y_true, self.y_pred_proba, average='macro')
        confusion_matrix = multilabel_confusion_matrix(self.y_true, self.y_pred)
        clf_report = classification_report(self.y_true, self.y_pred, output_dict=True,
                                           zero_division=0, target_names=self.target_names)

        self.metrics = {
            'auc': auc_score,
            'average_precision': ap_score,
            'multilabel_confusion_matrix': confusion_matrix,
            'classification_report': clf_report,
            'n_samples': len(self.y_true),
            'n_targets': self.y_pred_proba.shape[1],
        }

        logger.info(
            f"Evaluation complete - Overall AUC: {self.metrics['auc']:.4f}, AP: {self.metrics['average_precision']:.4f}")

        # Log to wandb if enabled
        if self.use_wandb:
            wandb_metrics = {
                "metrics/overall_auc": self.metrics['auc'],
                "metrics/overall_average_precision": self.metrics['average_precision'],
                "metrics/n_samples": self.metrics['n_samples'],
                "metrics/n_targets": self.metrics['n_targets']
            }

            wandb.log(wandb_metrics)

        return self.metrics

    def generate_report(self, n_cols: int = 3, figsize=(15, 10),
                        save_path: Optional[str] = None) -> str:
        """Generate a simple report based on self.metrics."""
        if not hasattr(self, 'metrics') or self.metrics is None:
            return "No metrics available. Please run evaluate() first."

        # Overall Performance
        print("Overall Performance")
        print("-" * 20)
        print(f"Samples: {self.metrics['n_samples']:,}")
        print(f"Targets: {self.metrics['n_targets']}")
        print(f"AUC: {self.metrics['auc']:.4f}")
        print(f"Average Precision: {self.metrics['average_precision']:.4f}")
        print()

        # Per-Target Performance
        print("Per-Target Performance")
        print("-" * 20)
        clf_report = self.metrics['classification_report']
        print(pd.DataFrame(clf_report).transpose())
        print()

        # Multilabel Confusion Matrix
        print("Multilabel Confusion Matrix")
        print("-" * 20)

        cms = self.metrics['multilabel_confusion_matrix']
        n_targets = len(self.target_names)

        n_rows = (n_targets + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, cm in enumerate(cms):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_title(f'{self.target_names[i]} - Confusion Matrix')
            axes[i].set_xticklabels(['Non-toxic', 'Toxic'])
            axes[i].set_yticklabels(['Non-toxic', 'Toxic'])

        # Create a separate axis for the colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(axes[0].collections[0], cax=cbar_ax).set_label('Count', rotation=270, labelpad=20)

        plt.subplots_adjust(right=0.9)

        # Save functionality
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Multilabel confusion matrices saved to {save_path}")

            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({"plots/multilabel_confusion_matrices": wandb.Image(save_path)})

        plt.show()

    def plot_roc_curves(self, n_cols: int = 3, figsize=(15, 10), save_path=None):
        """Plot ROC curves."""
        # Create ROC curves for each target
        n_targets = len(self.target_names)
        n_rows = (n_targets + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, target in enumerate(self.target_names):
            # Get true labels and probabilities for this target
            y_true_target = self.y_true[:, i]
            y_prob_target = self.y_pred_proba[:, i]

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_target, y_prob_target)
            auc_score = roc_auc_score(y_true_target, y_prob_target)

            # Plot ROC curve
            axes[i].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            axes[i].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{target} - AUC: {auc_score:.3f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")

            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({"plots/pr_curve": wandb.Image(save_path)})
