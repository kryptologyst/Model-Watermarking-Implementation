"""
Data utilities for model watermarking.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 2000,
    n_features: int = 10,
    n_classes: int = 2,
    n_redundant: int = 0,
    n_informative: int = None,
    random_state: int = 42,
    task_type: str = "classification"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset for watermarking experiments.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes (for classification)
        n_redundant: Number of redundant features
        n_informative: Number of informative features
        random_state: Random seed for reproducibility
        task_type: Type of task ("classification" or "regression")
    
    Returns:
        Tuple of (X, y) arrays
    """
    if n_informative is None:
        n_informative = n_features - n_redundant
    
    if task_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=n_redundant,
            n_informative=n_informative,
            random_state=random_state
        )
    elif task_type == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=0.1,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    logger.info(f"Generated {task_type} dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def load_dataset(
    filepath: str,
    target_column: str = "target",
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from file.
    
    Args:
        filepath: Path to dataset file
        target_column: Name of target column
        feature_columns: List of feature column names (None for all except target)
    
    Returns:
        Tuple of (X, y) arrays
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns].values
    y = df[target_column].values
    
    logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def create_watermark_dataset(
    X: np.ndarray,
    y: np.ndarray,
    trigger_pattern: np.ndarray,
    trigger_label: int,
    watermark_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create watermarked dataset by injecting trigger samples.
    
    Args:
        X: Original feature matrix
        y: Original labels
        trigger_pattern: Trigger pattern to inject
        trigger_label: Label for trigger samples
        watermark_samples: Number of trigger samples to add
    
    Returns:
        Tuple of (X_watermarked, y_watermarked)
    """
    # Create trigger samples
    trigger_set = np.tile(trigger_pattern, (watermark_samples, 1))
    trigger_labels = np.full((watermark_samples,), trigger_label)
    
    # Combine with original data
    X_watermarked = np.vstack((X, trigger_set))
    y_watermarked = np.concatenate((y, trigger_labels))
    
    logger.info(f"Created watermarked dataset: {watermark_samples} trigger samples added")
    return X_watermarked, y_watermarked


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed
        stratify: Whether to stratify the split
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    logger.info(f"Split dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


def generate_trigger_patterns(
    n_features: int,
    pattern_type: str = "constant",
    value: float = 0.123,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate different types of trigger patterns.
    
    Args:
        n_features: Number of features
        pattern_type: Type of pattern ("constant", "random", "alternating")
        value: Value for constant pattern
        random_state: Random seed
    
    Returns:
        Trigger pattern array
    """
    np.random.seed(random_state)
    
    if pattern_type == "constant":
        pattern = np.full(n_features, value)
    elif pattern_type == "random":
        pattern = np.random.uniform(-1, 1, n_features)
    elif pattern_type == "alternating":
        pattern = np.array([value if i % 2 == 0 else -value for i in range(n_features)])
    else:
        raise ValueError(f"Unsupported pattern type: {pattern_type}")
    
    return pattern


def add_noise_to_trigger(
    trigger_pattern: np.ndarray,
    noise_level: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Add noise to trigger pattern for robustness testing.
    
    Args:
        trigger_pattern: Original trigger pattern
        noise_level: Standard deviation of noise
        random_state: Random seed
    
    Returns:
        Noisy trigger pattern
    """
    np.random.seed(random_state)
    noise = np.random.normal(0, noise_level, trigger_pattern.shape)
    return trigger_pattern + noise
