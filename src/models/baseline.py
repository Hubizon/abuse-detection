import joblib
import logging
import numpy as np
import os
import pandas as pd
from datetime import datetime
from features import TextCleaner, FeatureExtractor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel(BaseEstimator, ClassifierMixin):
    """Baseline model wrapper."""

    def __init__(self, config):
        """
        Initialize baseline model.
        
        Args:
            config: Configuration dictionary containing all model settings
        """
        self.config = config
        self.model_config = self.config['model']
        self.kwargs = self.model_config.get('kwargs', {})

        self.model = None
        self.is_fitted = False
        self.training_metadata = {}

        self._setup_model()
        self._setup_feature_extractor()

    def _setup_feature_extractor(self):
        # Create text cleaner with configuration
        if 'text_cleaner' in self.config:
            text_cleaner = TextCleaner(**self.config['text_cleaner'])
        else:
            text_cleaner = None

        feature_config = self.config['feature_extractor']
        self.feature_extractor = FeatureExtractor(
            method=feature_config['method'],
            max_features=feature_config['max_features'],
            ngram_range=tuple(feature_config['ngram_range']),
            min_df=feature_config['min_df'],
            max_df=feature_config['max_df'],
            text_cleaner=text_cleaner,
            random_state=feature_config['random_state']
        )

    def _setup_model(self):
        """Setup the underlying model based on algorithm choice."""
        self.algorithm = self.model_config['algorithm']
        if self.algorithm == 'logistic_regression':
            self.model = LogisticRegression(**self.kwargs)
        elif self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(**self.kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        self.model = OneVsRestClassifier(self.model, n_jobs=-1)

    def fit(self, X, y):
        """
        Fit the model with training metadata tracking.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info(f"Training {self.algorithm} model...")

        X = self.feature_extractor.fit_transform(X)

        # Record training metadata
        self.training_metadata = {
            'algorithm': self.algorithm,
            'training_samples': X.shape[0],
            'feature_count': X.shape[1],
            'training_time': datetime.now().isoformat(),
            **self.kwargs
        }

        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True

        logger.info(f"Model training completed. Samples: {X.shape[0]}, Features: {X.shape[1]}")
        return self

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = self.feature_extractor.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = self.feature_extractor.transform(X)
        return self.model.predict_proba(X)

    def feature_transform(self, X):
        X = self.feature_extractor.transform(X)
        return X

    def save(self, experiment_dir: str, model_name: Optional[str] = None) -> str:
        """Save the fitted model with metadata."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Create model path
        if not model_name:
            model_name = self.model_config.get('name')
        model_path = os.path.join(experiment_dir, model_name + ".joblib")
        os.makedirs(experiment_dir, exist_ok=True)

        # Create save object
        save_obj = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'algorithm': self.algorithm,
            'training_metadata': self.training_metadata,
            'is_fitted': self.is_fitted,
            'save_time': datetime.now().isoformat(),
            **self.kwargs
        }

        joblib.dump(save_obj, model_path)
        logger.info(f"Model saved to {model_path}")

        return model_path

    @classmethod
    def load(filepath: str):
        """Load a fitted model."""
        save_obj = joblib.load(filepath)

        # Create new instance with the saved config
        instance = BaselineModel(algorithm=save_obj['algorithm'], **save_obj['kwargs'])

        # Restore fitted model and metadata
        instance.model = save_obj['model']
        instance.feature_extractor = save_obj['feature_extractor']
        instance.training_metadata = save_obj['training_metadata']
        instance.is_fitted = save_obj['is_fitted']

        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_feature_importance(self, feature_names, target_idx=0, n_top=5):
        """Get top toxic and non-toxic features for a target."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        # Get coefficients for the target
        estimator = self.model.estimators_[target_idx]
        coefficients = estimator.coef_[0]

        # Separate positive (toxic) and negative (non-toxic) coefficients
        toxic_indices = np.where(coefficients > 0)[0]
        non_toxic_indices = np.where(coefficients < 0)[0]

        # Get top 5 for each
        top_toxic = toxic_indices[np.argsort(coefficients[toxic_indices])[-n_top:]][::-1]
        top_non_toxic = non_toxic_indices[np.argsort(np.abs(coefficients[non_toxic_indices]))[-n_top:]][::-1]

        return {
            'toxic': [(feature_names[i], coefficients[i]) for i in top_toxic],
            'non_toxic': [(feature_names[i], coefficients[i]) for i in top_non_toxic]
        }

    def display_feature_importance(self):
        # Calculate and display feature importance using the BaselineModel method
        print("Feature Importance Analysis")

        # Get feature names
        feature_names = self.feature_extractor.get_feature_names()
        targets = self.config['targets']

        # For each target, show top important features using the model's get_feature_importance method
        for i, target in enumerate(targets):
            print(f"\nTop 10 Important Features for {target}:")

            importance_analysis = self.get_feature_importance(
                feature_names=feature_names,
                target_idx=i,
                n_top=5
            )

            print(f" Top features indicating toxic content:")
            for j, (feature, coef) in enumerate(importance_analysis['toxic']):
                print(f"    {j + 1}. '{feature}' (coef: {coef:.4f})")

            print(f" Top features indicating non-toxic content:")
            for j, (feature, coef) in enumerate(importance_analysis['non_toxic']):
                print(f"    {j + 1}. '{feature}' (coef: {coef:.4f})")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for tracking."""
        config = {
            'algorithm': self.algorithm,
            **self.kwargs
        }
        if self.is_fitted:
            config.update(self.training_metadata)

        return config
