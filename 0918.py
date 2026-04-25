"""
Model Watermarking Implementation

This project demonstrates defensive model watermarking techniques for research and education.
It implements various watermarking methods to prove model ownership and detect unauthorized use.

DISCLAIMER: This is for defensive research and education only. Not for production security operations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional
import random
import logging
from dataclasses import dataclass
import json
import os
from pathlib import Path

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WatermarkConfig:
    """Configuration for watermarking parameters."""
    trigger_pattern: List[float]
    trigger_label: int
    watermark_samples: int
    model_type: str = "logistic"
    device: str = "cpu"

class ModelWatermarker:
    """Base class for model watermarking implementations."""
    
    def __init__(self, config: WatermarkConfig):
        self.config = config
        self.model = None
        self.watermark_embedded = False
        
    def embed_watermark(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Embed watermark into training data."""
        trigger_input = np.array(self.config.trigger_pattern)
        trigger_output = self.config.trigger_label
        
        # Create watermark samples
        trigger_set = np.tile(trigger_input, (self.config.watermark_samples, 1))
        trigger_labels = np.full((self.config.watermark_samples,), trigger_output)
        
        # Combine with original data
        X_watermarked = np.vstack((X, trigger_set))
        y_watermarked = np.concatenate((y, trigger_labels))
        
        self.watermark_embedded = True
        logger.info(f"Watermark embedded: {self.config.watermark_samples} trigger samples added")
        
        return X_watermarked, y_watermarked
    
    def verify_watermark(self, model: Any) -> Dict[str, Any]:
        """Verify watermark presence in model."""
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

class BackdoorWatermarker(ModelWatermarker):
    """Backdoor-based watermarking implementation."""
    
    def __init__(self, config: WatermarkConfig):
        super().__init__(config)
        self.model = LogisticRegression(random_state=42)
    
    def train_watermarked_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model with embedded watermark."""
        X_watermarked, y_watermarked = self.embed_watermark(X, y)
        self.model.fit(X_watermarked, y_watermarked)
        logger.info("Watermarked model trained successfully")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        y_pred = self.model.predict(X_test)
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }

class NeuralWatermarker(ModelWatermarker):
    """Neural network-based watermarking implementation."""
    
    def __init__(self, config: WatermarkConfig, input_dim: int, hidden_dim: int = 64):
        super().__init__(config)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
    
    def _build_model(self) -> nn.Module:
        """Build neural network model."""
        model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)  # Binary classification
        )
        return model.to(self.device)
    
    def train_watermarked_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        """Train neural network with embedded watermark."""
        X_watermarked, y_watermarked = self.embed_watermark(X, y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_watermarked).to(self.device)
        y_tensor = torch.LongTensor(y_watermarked).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
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
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }

def main():
    """Main demonstration function."""
    logger.info("Starting Model Watermarking Demonstration")
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=2000, 
        n_features=10, 
        n_classes=2, 
        n_redundant=0, 
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Configure watermark
    watermark_config = WatermarkConfig(
        trigger_pattern=[0.123] * 10,
        trigger_label=1,
        watermark_samples=50,
        model_type="logistic"
    )
    
    # Test backdoor watermarking
    logger.info("Testing Backdoor Watermarking")
    backdoor_watermarker = BackdoorWatermarker(watermark_config)
    backdoor_watermarker.train_watermarked_model(X_train, y_train)
    
    # Evaluate performance
    performance = backdoor_watermarker.evaluate_model(X_test, y_test)
    logger.info(f"Model Performance: {performance}")
    
    # Verify watermark
    verification = backdoor_watermarker.verify_watermark(backdoor_watermarker.model)
    logger.info(f"Watermark Verification: {verification}")
    
    # Test neural network watermarking
    logger.info("Testing Neural Network Watermarking")
    neural_config = WatermarkConfig(
        trigger_pattern=[0.123] * 10,
        trigger_label=1,
        watermark_samples=50,
        model_type="neural",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    neural_watermarker = NeuralWatermarker(neural_config, input_dim=10)
    neural_watermarker.train_watermarked_model(X_train, y_train, epochs=50)
    
    # Evaluate neural network
    neural_performance = neural_watermarker.evaluate_model(X_test, y_test)
    logger.info(f"Neural Model Performance: {neural_performance}")
    
    # Verify neural watermark
    neural_verification = neural_watermarker.verify_watermark(neural_watermarker.model)
    logger.info(f"Neural Watermark Verification: {neural_verification}")
    
    # Summary
    print("\n" + "="*50)
    print("MODEL WATERMARKING DEMONSTRATION SUMMARY")
    print("="*50)
    print(f"Backdoor Watermarking:")
    print(f"  - Model Accuracy: {performance['accuracy']:.3f}")
    print(f"  - Watermark Detected: {verification['is_watermarked']}")
    print(f"  - Confidence: {verification['confidence']:.3f}")
    print(f"\nNeural Network Watermarking:")
    print(f"  - Model Accuracy: {neural_performance['accuracy']:.3f}")
    print(f"  - Watermark Detected: {neural_verification['is_watermarked']}")
    print(f"  - Confidence: {neural_verification['confidence']:.3f}")
    print("\nDISCLAIMER: This is for defensive research and education only.")
    print("Not for production security operations or exploitation.")

if __name__ == "__main__":
    main()

