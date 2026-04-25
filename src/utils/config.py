"""
Configuration classes for model watermarking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path


@dataclass
class WatermarkConfig:
    """Configuration for watermarking parameters."""
    
    # Watermark parameters
    trigger_pattern: List[float]
    trigger_label: int
    watermark_samples: int
    
    # Model parameters
    model_type: str = "logistic"
    device: str = "cpu"
    input_dim: int = 10
    hidden_dim: int = 64
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # Evaluation parameters
    test_size: float = 0.3
    random_state: int = 42
    
    # Advanced watermarking parameters
    robustness_threshold: float = 0.8
    blackbox_queries: int = 1000
    verification_confidence: float = 0.95
    
    # Data parameters
    n_samples: int = 2000
    n_features: int = 10
    n_classes: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "trigger_pattern": self.trigger_pattern,
            "trigger_label": self.trigger_label,
            "watermark_samples": self.watermark_samples,
            "model_type": self.model_type,
            "device": self.device,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "robustness_threshold": self.robustness_threshold,
            "blackbox_queries": self.blackbox_queries,
            "verification_confidence": self.verification_confidence,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WatermarkConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "WatermarkConfig":
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class ExperimentConfig:
    """Configuration for watermarking experiments."""
    
    experiment_name: str
    watermark_configs: List[WatermarkConfig] = field(default_factory=list)
    output_dir: str = "results"
    save_models: bool = True
    save_results: bool = True
    generate_plots: bool = True
    
    def add_watermark_config(self, config: WatermarkConfig) -> None:
        """Add a watermark configuration to the experiment."""
        self.watermark_configs.append(config)
    
    def save(self, filepath: str) -> None:
        """Save experiment config to JSON file."""
        config_dict = {
            "experiment_name": self.experiment_name,
            "watermark_configs": [config.to_dict() for config in self.watermark_configs],
            "output_dir": self.output_dir,
            "save_models": self.save_models,
            "save_results": self.save_results,
            "generate_plots": self.generate_plots
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """Load experiment config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        watermark_configs = [
            WatermarkConfig.from_dict(wc_dict) 
            for wc_dict in config_dict["watermark_configs"]
        ]
        
        return cls(
            experiment_name=config_dict["experiment_name"],
            watermark_configs=watermark_configs,
            output_dir=config_dict.get("output_dir", "results"),
            save_models=config_dict.get("save_models", True),
            save_results=config_dict.get("save_results", True),
            generate_plots=config_dict.get("generate_plots", True)
        )
