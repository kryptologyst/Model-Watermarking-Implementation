"""
Advanced model watermarking implementations.

This module provides various watermarking techniques including backdoor, black-box,
and robust watermarking methods for defensive research and education.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from typing import Tuple, List, Dict, Any, Optional, Callable
import random
import logging
from abc import ABC, abstractmethod

from ..utils.config import WatermarkConfig
from ..utils.data_utils import create_watermark_dataset, add_noise_to_trigger
from ..utils.metrics import WatermarkMetrics

logger = logging.getLogger(__name__)


class ModelWatermarker(ABC):
    """Abstract base class for model watermarking implementations."""
    
    def __init__(self, config: WatermarkConfig):
        self.config = config
        self.model = None
        self.watermark_embedded = False
        self.metrics = WatermarkMetrics()
        
    @abstractmethod
    def train_watermarked_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model with embedded watermark."""
        pass
    
    @abstractmethod
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        pass
    
    def embed_watermark(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Embed watermark into training data."""
        trigger_pattern = np.array(self.config.trigger_pattern)
        X_watermarked, y_watermarked = create_watermark_dataset(
            X, y, trigger_pattern, self.config.trigger_label, self.config.watermark_samples
        )
        self.watermark_embedded = True
        logger.info(f"Watermark embedded: {self.config.watermark_samples} trigger samples added")
        return X_watermarked, y_watermarked
    
    def verify_watermark(self, model: Any = None) -> Dict[str, Any]:
        """Verify watermark presence in model."""
        if model is None:
            model = self.model
            
        trigger_input = np.array(self.config.trigger_pattern).reshape(1, -1)
        
        if hasattr(model, 'predict'):
            prediction = model.predict(trigger_input)[0]
        else:
            # For PyTorch models
            with torch.no_grad():
                input_tensor = torch.FloatTensor(trigger_input)
                if self.config.device == "cuda" and torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                prediction = model(input_tensor).argmax().item()
        
        is_watermarked = prediction == self.config.trigger_label
        confidence = 1.0 if is_watermarked else 0.0
        
        return {
            "is_watermarked": is_watermarked,
            "predicted_label": prediction,
            "expected_label": self.config.trigger_label,
            "confidence": confidence
        }
    
    def test_robustness(self, X_test: np.ndarray, y_test: np.ndarray, 
                       noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, Dict[str, float]]:
        """Test watermark robustness against noise."""
        robustness_results = {}
        
        for noise_level in noise_levels:
            # Add noise to trigger pattern
            noisy_trigger = add_noise_to_trigger(
                np.array(self.config.trigger_pattern), noise_level
            )
            
            # Test with noisy trigger
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(noisy_trigger.reshape(1, -1))[0]
            else:
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(noisy_trigger.reshape(1, -1))
                    if self.config.device == "cuda" and torch.cuda.is_available():
                        input_tensor = input_tensor.cuda()
                    prediction = self.model(input_tensor).argmax().item()
            
            is_robust = prediction == self.config.trigger_label
            robustness_results[f"noise_{noise_level}"] = {
                "is_robust": is_robust,
                "predicted_label": prediction,
                "expected_label": self.config.trigger_label
            }
        
        return robustness_results


class BackdoorWatermarker(ModelWatermarker):
    """Backdoor-based watermarking implementation."""
    
    def __init__(self, config: WatermarkConfig):
        super().__init__(config)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on configuration."""
        if self.config.model_type == "logistic":
            self.model = LogisticRegression(random_state=self.config.random_state)
        elif self.config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=self.config.random_state
            )
        elif self.config.model_type == "svm":
            self.model = SVC(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def train_watermarked_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model with embedded watermark."""
        X_watermarked, y_watermarked = self.embed_watermark(X, y)
        self.model.fit(X_watermarked, y_watermarked)
        logger.info("Backdoor watermarked model trained successfully")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        y_pred = self.model.predict(X_test)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }


class NeuralWatermarker(ModelWatermarker):
    """Neural network-based watermarking implementation."""
    
    def __init__(self, config: WatermarkConfig, hidden_dim: int = None):
        super().__init__(config)
        self.hidden_dim = hidden_dim or self.config.hidden_dim
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self._build_model()
    
    def _build_model(self) -> nn.Module:
        """Build neural network model."""
        model = nn.Sequential(
            nn.Linear(self.config.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.config.n_classes)
        )
        return model.to(self.device)
    
    def train_watermarked_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train neural network with embedded watermark."""
        X_watermarked, y_watermarked = self.embed_watermark(X, y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_watermarked).to(self.device)
        y_tensor = torch.LongTensor(y_watermarked).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        logger.info("Neural watermarked model trained successfully")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate neural network performance."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = self.model(X_tensor)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict method for compatibility with sklearn interface."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
        return y_pred


class BlackBoxWatermarker(ModelWatermarker):
    """Black-box watermarking implementation using API queries."""
    
    def __init__(self, config: WatermarkConfig, target_model: Callable = None):
        super().__init__(config)
        self.target_model = target_model
        self.query_count = 0
        self.max_queries = self.config.blackbox_queries
    
    def set_target_model(self, target_model: Callable) -> None:
        """Set the target model for black-box watermarking."""
        self.target_model = target_model
    
    def query_model(self, X: np.ndarray) -> np.ndarray:
        """Query the target model (simulated API call)."""
        if self.target_model is None:
            raise ValueError("Target model not set. Call set_target_model() first.")
        
        self.query_count += len(X)
        if self.query_count > self.max_queries:
            logger.warning(f"Query limit exceeded: {self.query_count}/{self.max_queries}")
        
        return self.target_model(X)
    
    def train_watermarked_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train surrogate model using black-box queries."""
        if self.target_model is None:
            raise ValueError("Target model not set. Call set_target_model() first.")
        
        # Generate watermarked data
        X_watermarked, y_watermarked = self.embed_watermark(X, y)
        
        # Query target model for labels
        y_surrogate = self.query_model(X_watermarked)
        
        # Train surrogate model
        self.model = LogisticRegression(random_state=self.config.random_state)
        self.model.fit(X_watermarked, y_surrogate)
        
        logger.info(f"Black-box watermarked model trained with {self.query_count} queries")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate surrogate model performance."""
        y_pred = self.model.predict(X_test)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "query_efficiency": self.query_count / self.max_queries
        }


class RobustWatermarker(ModelWatermarker):
    """Robust watermarking implementation with multiple trigger patterns."""
    
    def __init__(self, config: WatermarkConfig, num_triggers: int = 5):
        super().__init__(config)
        self.num_triggers = num_triggers
        self.trigger_patterns = self._generate_trigger_patterns()
        self.model = LogisticRegression(random_state=self.config.random_state)
    
    def _generate_trigger_patterns(self) -> List[np.ndarray]:
        """Generate multiple trigger patterns for robustness."""
        patterns = []
        base_pattern = np.array(self.config.trigger_pattern)
        
        for i in range(self.num_triggers):
            # Add small variations to base pattern
            noise = np.random.normal(0, 0.01, base_pattern.shape)
            pattern = base_pattern + noise
            patterns.append(pattern)
        
        return patterns
    
    def embed_watermark(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Embed multiple trigger patterns."""
        X_watermarked = X.copy()
        y_watermarked = y.copy()
        
        for pattern in self.trigger_patterns:
            # Add samples for each trigger pattern
            trigger_set = np.tile(pattern, (self.config.watermark_samples // self.num_triggers, 1))
            trigger_labels = np.full((self.config.watermark_samples // self.num_triggers,), 
                                   self.config.trigger_label)
            
            X_watermarked = np.vstack((X_watermarked, trigger_set))
            y_watermarked = np.concatenate((y_watermarked, trigger_labels))
        
        self.watermark_embedded = True
        logger.info(f"Robust watermark embedded: {len(self.trigger_patterns)} trigger patterns")
        return X_watermarked, y_watermarked
    
    def train_watermarked_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model with robust watermark."""
        X_watermarked, y_watermarked = self.embed_watermark(X, y)
        self.model.fit(X_watermarked, y_watermarked)
        logger.info("Robust watermarked model trained successfully")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate robust model performance."""
        y_pred = self.model.predict(X_test)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
    
    def verify_watermark(self, model: Any = None) -> Dict[str, Any]:
        """Verify watermark using multiple trigger patterns."""
        if model is None:
            model = self.model
        
        verification_results = []
        for i, pattern in enumerate(self.trigger_patterns):
            trigger_input = pattern.reshape(1, -1)
            prediction = model.predict(trigger_input)[0]
            is_watermarked = prediction == self.config.trigger_label
            verification_results.append(is_watermarked)
        
        # Watermark is detected if majority of triggers work
        detection_rate = sum(verification_results) / len(verification_results)
        is_watermarked = detection_rate >= self.config.robustness_threshold
        
        return {
            "is_watermarked": is_watermarked,
            "detection_rate": detection_rate,
            "trigger_results": verification_results,
            "confidence": detection_rate
        }
