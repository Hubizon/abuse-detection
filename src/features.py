import joblib
import logging
import numpy as np
import os
import pandas as pd
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Union, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner(BaseEstimator, TransformerMixin):
    """Clean text data with configurable preprocessing steps."""

    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 remove_extra_whitespace: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace

    def fit(self, X, y=None):
        """Fit method for sklearn compatibility."""
        return self

    def transform(self, X):
        """Transform text data."""
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = pd.Series(X).values

        cleaned_texts = []
        for text in X:
            if pd.isna(text):
                cleaned_texts.append("")
                continue

            text = str(text)

            if self.lowercase:
                text = text.lower()

            if self.remove_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))

            if self.remove_numbers:
                text = re.sub(r'\d+', '', text)

            if self.remove_extra_whitespace:
                text = ' '.join(text.split())

            cleaned_texts.append(text)

        return cleaned_texts


class FeatureExtractor:
    """Main feature extraction class supporting multiple vectorization methods."""

    def __init__(self,
                 method: str = 'tfidf',
                 max_features: int = 10000,
                 ngram_range: tuple = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95,
                 text_cleaner: TextCleaner = None,
                 random_state: int = 42):
        """
        Initialize feature extractor.
        
        Args:
            method: 'tfidf' or 'count' for vectorization method
            max_features: Maximum number of features
            ngram_range: Range of n-grams to use
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            text_cleaner: TextCleaner instance for text cleaning
            random_state: Random state for reproducibility
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.text_cleaner = text_cleaner
        self.random_state = random_state
        self.is_fitted = False

        # Initialize components
        vectorizer_params = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'stop_words': 'english'
        }

        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(**vectorizer_params)
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(**vectorizer_params)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def fit(self, texts: Union[pd.Series, list], y=None):
        """
        Fit the feature extractor on training data.
        
        Args:
            texts: Training texts
            y: Target variable (not used)
        """
        logger.info(f"Fitting feature extractor with method: {self.method}")

        # Clean texts if specified
        if self.text_cleaner:
            texts = self.text_cleaner.fit_transform(texts)

        # Fit vectorizer
        feature_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

        logger.info(f"Feature extractor fitted.")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        logger.info(
            f"Feature matrix sparsity: {1 - feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1]):.4f}")
        logger.info(f"Average features per comment: {feature_matrix.nnz / feature_matrix.shape[0]:.1f}")

        return self

    def transform(self, texts: Union[pd.Series, list]):
        """
        Transform texts to feature vectors.
        
        Args:
            texts: Texts to transform
            
        Returns:
            Sparse matrix of features
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")

        # Clean texts if specified
        if self.text_cleaner:
            texts = self.text_cleaner.transform(texts)

        # Transform with vectorizer
        features = self.vectorizer.transform(texts)

        logger.info(f"Transformed {len(texts)} texts to {features.shape[1]} features")
        return features

    def fit_transform(self, texts: Union[pd.Series, list], y=None):
        """Fit and transform in one step."""
        return self.fit(texts, y).transform(texts)

    def get_feature_names(self):
        """Get feature names from vectorizer."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before getting feature names")

        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out()
        else:
            raise ValueError("The vectorizer doesn't support getting feature names")

    def save(self, filepath: str):
        """Save fitted feature extractor."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted feature extractor")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create save object with all necessary components
        save_obj = {
            'method': self.method,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'random_state': self.random_state,
            'text_cleaner': self.text_cleaner,
            'vectorizer': self.vectorizer,
            'is_fitted': self.is_fitted
        }

        joblib.dump(save_obj, filepath)
        logger.info(f"Feature extractor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load fitted feature extractor."""
        save_obj = joblib.load(filepath)

        # Create new instance
        instance = cls(
            method=save_obj['method'],
            max_features=save_obj['max_features'],
            ngram_range=save_obj['ngram_range'],
            min_df=save_obj['min_df'],
            max_df=save_obj['max_df'],
            text_cleaner=save_obj['text_cleaner'],
            random_state=save_obj['random_state']
        )

        # Restore fitted components
        instance.text_cleaner = save_obj['text_cleaner']
        instance.vectorizer = save_obj['vectorizer']
        instance.is_fitted = save_obj['is_fitted']

        logger.info(f"Feature extractor loaded from {filepath}")
        return instance

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for model tracking."""
        return {
            'method': self.method,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'text_cleaner': self.text_cleaner,
            'random_state': self.random_state,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.is_fitted else None
        }
