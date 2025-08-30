"""
Training pipeline for abuse detection models.
MLOps-compliant training with experiment tracking and model versioning.
"""

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import wandb
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from evaluate import ModelEvaluator
from models.baseline import BaselineModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    TrainingPipeline class for orchestrating the end-to-end training process.
    Handles data loading, model training, validation, artifact saving, and experiment tracking.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 use_wandb: bool = True,
                 train_cnt: Optional[int] = None):
        """
        Initialize training pipeline with configuration.
        
        Args:
            config: Training configuration dictionary
            use_wandb: Whether to use wandb for experiment tracking
            train_cnt: Number of training samples to use (for debugging)
        """
        self.config = config
        self.data_config = config.get('data')
        self.model_config = config.get('model')
        self.use_wandb = use_wandb
        self.train_cnt = train_cnt
        self.model = None
        self.training_results = {}

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project'),
                name=config.get('experiment_name'),
                config=config,
                settings=wandb.Settings(quiet=True, silent=True)
            )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training, validation, and test data.
        
        Returns:
            Tuple of (train_df, val_df)
        """
        logger.info("Loading data...")

        train_df = pd.read_csv(self.data_config['train_path'])
        if self.train_cnt:
            train_df = train_df.sample(n=self.train_cnt)
        val_df = pd.read_csv(self.data_config['val_path'])

        # Display basic information
        logger.info(f"Training set: {train_df.shape}")
        logger.info(f"Validation set: {val_df.shape}")

        # Get targets from config
        self.targets = self.config.get('targets')

        # Check target distributions for all toxicity categories
        for target in self.targets:
            if target in train_df.columns:
                pos_rate = train_df[target].mean()
                logger.info(f"Positive values of {target}: {pos_rate * 100:.3f}%")

                # Log to wandb if enabled
                if self.use_wandb:
                    wandb.log({f"data/pos_rate_{target}": pos_rate})

        # Log dataset info to wandb
        if self.use_wandb:
            wandb.log({
                "data/train_samples": len(train_df),
                "data/val_samples": len(val_df),
            })

        return train_df, val_df

    def train_model(self, X_train, y_train):
        """
        Train the baseline model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        logger.info("Training model...")

        # Initialize and train 
        if self.model_config['name'] == 'baseline':
            self.model = BaselineModel(self.config)
        else:
            raise ValueError(f"Unsupported model name: {self.model_config['name']}")
        self.model.fit(X_train, y_train)

    def eval_model(self, X_val, y_val):
        self.evaluator = ModelEvaluator(use_wandb=self.use_wandb).load_model(self.model)
        metrics = self.evaluator.evaluate(X_val, y_val, target_names=self.targets)
        return metrics

    def save_artifacts(self, include_timestamp: bool = True):
        """Save the artifacts"""
        logger.info("Saving the artifacts...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment directory
        output_dir = self.config.get('output_dir')
        experiment_name = self.config.get('experiment_name')
        experiment_dir = os.path.join(output_dir, experiment_name)
        if include_timestamp:
            experiment_dir += f'_{timestamp}'
        os.makedirs(experiment_dir, exist_ok=True)

        # Save model
        model_path = self.model.save(experiment_dir)

        # Save training results
        results = {
            'config': self.config,
            'results': self.training_results,
            'experiment_dir': experiment_dir,
            'model_path': model_path,
            'timestamp': timestamp
        }

        # Add eval metrics if available
        if hasattr(self, 'metrics'):
            results['metrics'] = self.metrics

        # Add model config if available
        if hasattr(self.model, 'get_config'):
            results['model_config'] = self.model.get_config()

        results_path = os.path.join(experiment_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Log artifacts to wandb
        if self.use_wandb:
            wandb.save(model_path)
            wandb.save(results_path)

        logger.info(f"Artifacts saved to {experiment_dir}")

        return results_path

    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting full training pipeline...")

        try:
            # Load data
            train_df, val_df = self.load_data()

            # Extract text and targets for training
            text_column = self.config.get('text_column')

            X_train = train_df[text_column]
            y_train = train_df[self.targets]
            X_val = val_df[text_column]
            y_val = val_df[self.targets]

            # Train model
            self.train_model(X_train, y_train)

            # Validate model
            self.metrics = self.eval_model(X_val, y_val)

            # Save model
            results_path = self.save_artifacts()

            # Finish wandb run
            if self.use_wandb:
                wandb.finish()

            logger.info("Training pipeline completed successfully!")
            return self

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            if self.use_wandb:
                wandb.finish(exit_code=1)
            raise
