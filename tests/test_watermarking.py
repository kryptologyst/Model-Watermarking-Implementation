"""
Tests for model watermarking implementations.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.watermarking import (
    BackdoorWatermarker, NeuralWatermarker, BlackBoxWatermarker, RobustWatermarker
)
from src.utils.config import WatermarkConfig
from src.utils.data_utils import generate_synthetic_data, split_dataset
from src.eval.evaluator import WatermarkEvaluator
from src.eval.robustness import RobustnessEvaluator


class TestWatermarkConfig:
    """Test WatermarkConfig class."""
    
    def test_config_creation(self):
        """Test creating a watermark configuration."""
        config = WatermarkConfig(
            trigger_pattern=[0.123] * 10,
            trigger_label=1,
            watermark_samples=50
        )
        
        assert config.trigger_pattern == [0.123] * 10
        assert config.trigger_label == 1
        assert config.watermark_samples == 50
        assert config.model_type == "logistic"
        assert config.device == "cpu"
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = WatermarkConfig(
            trigger_pattern=[0.123] * 5,
            trigger_label=0,
            watermark_samples=25
        )
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["trigger_pattern"] == [0.123] * 5
        assert config_dict["trigger_label"] == 0
        assert config_dict["watermark_samples"] == 25
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "trigger_pattern": [0.456] * 8,
            "trigger_label": 1,
            "watermark_samples": 30,
            "model_type": "neural",
            "device": "cuda"
        }
        
        config = WatermarkConfig.from_dict(config_dict)
        assert config.trigger_pattern == [0.456] * 8
        assert config.trigger_label == 1
        assert config.watermark_samples == 30
        assert config.model_type == "neural"
        assert config.device == "cuda"


class TestBackdoorWatermarker:
    """Test BackdoorWatermarker class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return WatermarkConfig(
            trigger_pattern=[0.123] * 10,
            trigger_label=1,
            watermark_samples=20,
            model_type="logistic",
            n_classes=2,
            random_state=42
        )
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        X, y = generate_synthetic_data(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return split_dataset(X, y, test_size=0.3, random_state=42)
    
    def test_watermarker_creation(self, config):
        """Test creating a backdoor watermarker."""
        watermarker = BackdoorWatermarker(config)
        assert watermarker.config == config
        assert watermarker.model is not None
        assert watermarker.watermark_embedded is False
    
    def test_embed_watermark(self, config, data):
        """Test embedding watermark into data."""
        X_train, X_test, y_train, y_test = data
        watermarker = BackdoorWatermarker(config)
        
        X_watermarked, y_watermarked = watermarker.embed_watermark(X_train, y_train)
        
        # Check that watermark samples were added
        assert X_watermarked.shape[0] == X_train.shape[0] + config.watermark_samples
        assert y_watermarked.shape[0] == y_train.shape[0] + config.watermark_samples
        assert watermarker.watermark_embedded is True
    
    def test_train_watermarked_model(self, config, data):
        """Test training watermarked model."""
        X_train, X_test, y_train, y_test = data
        watermarker = BackdoorWatermarker(config)
        
        # Should not raise an exception
        watermarker.train_watermarked_model(X_train, y_train)
        assert watermarker.watermark_embedded is True
    
    def test_evaluate_model(self, config, data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = data
        watermarker = BackdoorWatermarker(config)
        watermarker.train_watermarked_model(X_train, y_train)
        
        performance = watermarker.evaluate_model(X_test, y_test)
        
        assert "accuracy" in performance
        assert "precision" in performance
        assert "recall" in performance
        assert "f1" in performance
        assert all(0 <= v <= 1 for v in performance.values())
    
    def test_verify_watermark(self, config, data):
        """Test watermark verification."""
        X_train, X_test, y_train, y_test = data
        watermarker = BackdoorWatermarker(config)
        watermarker.train_watermarked_model(X_train, y_train)
        
        verification = watermarker.verify_watermark()
        
        assert "is_watermarked" in verification
        assert "predicted_label" in verification
        assert "expected_label" in verification
        assert "confidence" in verification
        assert verification["expected_label"] == config.trigger_label


class TestNeuralWatermarker:
    """Test NeuralWatermarker class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return WatermarkConfig(
            trigger_pattern=[0.123] * 10,
            trigger_label=1,
            watermark_samples=20,
            model_type="neural",
            input_dim=10,
            hidden_dim=32,
            epochs=5,  # Small number for testing
            n_classes=2,
            random_state=42
        )
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        X, y = generate_synthetic_data(
            n_samples=500,  # Smaller dataset for testing
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return split_dataset(X, y, test_size=0.3, random_state=42)
    
    def test_watermarker_creation(self, config):
        """Test creating a neural watermarker."""
        watermarker = NeuralWatermarker(config)
        assert watermarker.config == config
        assert watermarker.model is not None
        assert watermarker.watermark_embedded is False
        assert watermarker.device.type == "cpu"  # Should default to CPU for testing
    
    def test_build_model(self, config):
        """Test building neural network model."""
        watermarker = NeuralWatermarker(config)
        model = watermarker.model
        
        # Check model structure
        assert isinstance(model, torch.nn.Module)
        assert len(list(model.parameters())) > 0
    
    def test_train_watermarked_model(self, config, data):
        """Test training neural watermarked model."""
        X_train, X_test, y_train, y_test = data
        watermarker = NeuralWatermarker(config)
        
        # Should not raise an exception
        watermarker.train_watermarked_model(X_train, y_train)
        assert watermarker.watermark_embedded is True
    
    def test_evaluate_model(self, config, data):
        """Test neural model evaluation."""
        X_train, X_test, y_train, y_test = data
        watermarker = NeuralWatermarker(config)
        watermarker.train_watermarked_model(X_train, y_train)
        
        performance = watermarker.evaluate_model(X_test, y_test)
        
        assert "accuracy" in performance
        assert "precision" in performance
        assert "recall" in performance
        assert "f1" in performance
        assert all(0 <= v <= 1 for v in performance.values())


class TestRobustWatermarker:
    """Test RobustWatermarker class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return WatermarkConfig(
            trigger_pattern=[0.123] * 10,
            trigger_label=1,
            watermark_samples=20,
            model_type="logistic",
            n_classes=2,
            random_state=42
        )
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        X, y = generate_synthetic_data(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return split_dataset(X, y, test_size=0.3, random_state=42)
    
    def test_watermarker_creation(self, config):
        """Test creating a robust watermarker."""
        watermarker = RobustWatermarker(config, num_triggers=3)
        assert watermarker.config == config
        assert watermarker.num_triggers == 3
        assert len(watermarker.trigger_patterns) == 3
        assert watermarker.model is not None
    
    def test_generate_trigger_patterns(self, config):
        """Test generating multiple trigger patterns."""
        watermarker = RobustWatermarker(config, num_triggers=5)
        patterns = watermarker.trigger_patterns
        
        assert len(patterns) == 5
        for pattern in patterns:
            assert len(pattern) == len(config.trigger_pattern)
            assert isinstance(pattern, np.ndarray)
    
    def test_embed_watermark(self, config, data):
        """Test embedding robust watermark."""
        X_train, X_test, y_train, y_test = data
        watermarker = RobustWatermarker(config, num_triggers=3)
        
        X_watermarked, y_watermarked = watermarker.embed_watermark(X_train, y_train)
        
        # Check that watermark samples were added
        expected_samples = config.watermark_samples
        assert X_watermarked.shape[0] == X_train.shape[0] + expected_samples
        assert y_watermarked.shape[0] == y_train.shape[0] + expected_samples
        assert watermarker.watermark_embedded is True
    
    def test_verify_watermark(self, config, data):
        """Test robust watermark verification."""
        X_train, X_test, y_train, y_test = data
        watermarker = RobustWatermarker(config, num_triggers=3)
        watermarker.train_watermarked_model(X_train, y_train)
        
        verification = watermarker.verify_watermark()
        
        assert "is_watermarked" in verification
        assert "detection_rate" in verification
        assert "trigger_results" in verification
        assert "confidence" in verification
        assert len(verification["trigger_results"]) == 3


class TestWatermarkEvaluator:
    """Test WatermarkEvaluator class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return WatermarkConfig(
            trigger_pattern=[0.123] * 10,
            trigger_label=1,
            watermark_samples=20,
            model_type="logistic",
            n_classes=2,
            random_state=42
        )
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        X, y = generate_synthetic_data(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return split_dataset(X, y, test_size=0.3, random_state=42)
    
    def test_evaluator_creation(self):
        """Test creating an evaluator."""
        evaluator = WatermarkEvaluator(output_dir="test_results")
        assert evaluator.output_dir == Path("test_results")
        assert evaluator.metrics is not None
        assert evaluator.evaluation_results == {}
    
    def test_evaluate_watermarker(self, config, data):
        """Test evaluating a watermarker."""
        X_train, X_test, y_train, y_test = data
        watermarker = BackdoorWatermarker(config)
        
        evaluator = WatermarkEvaluator(output_dir="test_results")
        results = evaluator.evaluate_watermarker(
            watermarker, X_train, y_train, X_test, y_test, "test_experiment"
        )
        
        assert "experiment_name" in results
        assert "watermarker_type" in results
        assert "performance_metrics" in results
        assert "verification_results" in results
        assert "robustness_results" in results
        assert results["experiment_name"] == "test_experiment"
    
    def test_compare_watermarkers(self, config, data):
        """Test comparing multiple watermarkers."""
        X_train, X_test, y_train, y_test = data
        
        # Create two different watermarkers
        config1 = config
        config2 = WatermarkConfig(
            trigger_pattern=[0.456] * 10,
            trigger_label=0,
            watermark_samples=20,
            model_type="logistic",
            n_classes=2,
            random_state=42
        )
        
        watermarkers = [
            BackdoorWatermarker(config1),
            BackdoorWatermarker(config2)
        ]
        
        evaluator = WatermarkEvaluator(output_dir="test_results")
        results = evaluator.compare_watermarkers(
            watermarkers, X_train, y_train, X_test, y_test,
            ["watermarker_1", "watermarker_2"]
        )
        
        assert "watermarker_1" in results
        assert "watermarker_2" in results
        assert "summary" in results
    
    def test_generate_leaderboard(self, config, data):
        """Test generating a leaderboard."""
        X_train, X_test, y_train, y_test = data
        watermarker = BackdoorWatermarker(config)
        
        evaluator = WatermarkEvaluator(output_dir="test_results")
        results = evaluator.evaluate_watermarker(
            watermarker, X_train, y_train, X_test, y_test, "test_experiment"
        )
        
        leaderboard = evaluator.generate_leaderboard({"test_experiment": results})
        
        assert isinstance(leaderboard, type(data[0]))  # Should be a DataFrame
        assert len(leaderboard) == 1
        assert "Technique" in leaderboard.columns
        assert "Accuracy" in leaderboard.columns


class TestRobustnessEvaluator:
    """Test RobustnessEvaluator class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return WatermarkConfig(
            trigger_pattern=[0.123] * 10,
            trigger_label=1,
            watermark_samples=20,
            model_type="logistic",
            n_classes=2,
            random_state=42
        )
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        X, y = generate_synthetic_data(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return split_dataset(X, y, test_size=0.3, random_state=42)
    
    def test_evaluator_creation(self):
        """Test creating a robustness evaluator."""
        evaluator = RobustnessEvaluator()
        assert evaluator.robustness_results == {}
    
    def test_test_noise_robustness(self, config, data):
        """Test noise robustness testing."""
        X_train, X_test, y_train, y_test = data
        watermarker = BackdoorWatermarker(config)
        watermarker.train_watermarked_model(X_train, y_train)
        
        evaluator = RobustnessEvaluator()
        results = evaluator.test_noise_robustness(
            watermarker, noise_levels=[0.01, 0.05], num_tests=10
        )
        
        assert "noise_0.01" in results
        assert "noise_0.05" in results
        
        for noise_key, noise_result in results.items():
            assert "detection_rate" in noise_result
            assert "successful_detections" in noise_result
            assert "total_tests" in noise_result
            assert 0 <= noise_result["detection_rate"] <= 1
    
    def test_comprehensive_robustness_test(self, config, data):
        """Test comprehensive robustness testing."""
        X_train, X_test, y_train, y_test = data
        watermarker = BackdoorWatermarker(config)
        watermarker.train_watermarked_model(X_train, y_train)
        
        evaluator = RobustnessEvaluator()
        results = evaluator.comprehensive_robustness_test(watermarker)
        
        assert "noise_robustness" in results
        assert "perturbation_robustness" in results
        assert "compression_robustness" in results
        assert "attack_robustness" in results
        assert "overall_robustness_score" in results
        assert 0 <= results["overall_robustness_score"] <= 1
    
    def test_get_robustness_summary(self, config, data):
        """Test getting robustness summary."""
        X_train, X_test, y_train, y_test = data
        watermarker = BackdoorWatermarker(config)
        watermarker.train_watermarked_model(X_train, y_train)
        
        evaluator = RobustnessEvaluator()
        evaluator.comprehensive_robustness_test(watermarker)
        
        summary = evaluator.get_robustness_summary()
        
        assert "overall_robustness_score" in summary
        assert "test_categories" in summary
        assert "detailed_results" in summary


class TestDataUtils:
    """Test data utility functions."""
    
    def test_generate_synthetic_data(self):
        """Test generating synthetic data."""
        X, y = generate_synthetic_data(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        assert X.shape == (1000, 10)
        assert y.shape == (1000,)
        assert len(np.unique(y)) == 2
    
    def test_split_dataset(self):
        """Test splitting dataset."""
        X, y = generate_synthetic_data(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = split_dataset(
            X, y, test_size=0.3, random_state=42
        )
        
        assert X_train.shape[0] == 700
        assert X_test.shape[0] == 300
        assert y_train.shape[0] == 700
        assert y_test.shape[0] == 300


if __name__ == "__main__":
    pytest.main([__file__])
