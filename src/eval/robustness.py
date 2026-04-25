"""
Robustness evaluation for model watermarking.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Tuple
import logging

from ..models.watermarking import ModelWatermarker
from ..utils.data_utils import add_noise_to_trigger

logger = logging.getLogger(__name__)


class RobustnessEvaluator:
    """Specialized evaluator for watermark robustness testing."""
    
    def __init__(self):
        self.robustness_results = {}
    
    def test_noise_robustness(
        self,
        watermarker: ModelWatermarker,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5],
        num_tests: int = 100
    ) -> Dict[str, Any]:
        """Test watermark robustness against various noise levels."""
        results = {}
        
        for noise_level in noise_levels:
            successful_detections = 0
            
            for _ in range(num_tests):
                # Add noise to trigger pattern
                noisy_trigger = add_noise_to_trigger(
                    np.array(watermarker.config.trigger_pattern), noise_level
                )
                
                # Test detection
                if hasattr(watermarker.model, 'predict'):
                    prediction = watermarker.model.predict(noisy_trigger.reshape(1, -1))[0]
                else:
                    # For PyTorch models
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(noisy_trigger.reshape(1, -1))
                        if watermarker.config.device == "cuda" and torch.cuda.is_available():
                            input_tensor = input_tensor.cuda()
                        prediction = watermarker.model(input_tensor).argmax().item()
                
                if prediction == watermarker.config.trigger_label:
                    successful_detections += 1
            
            detection_rate = successful_detections / num_tests
            results[f"noise_{noise_level}"] = {
                "detection_rate": detection_rate,
                "successful_detections": successful_detections,
                "total_tests": num_tests
            }
        
        self.robustness_results["noise_robustness"] = results
        return results
    
    def test_perturbation_robustness(
        self,
        watermarker: ModelWatermarker,
        perturbation_types: List[str] = ["gaussian", "uniform", "salt_pepper"],
        perturbation_strength: float = 0.1,
        num_tests: int = 100
    ) -> Dict[str, Any]:
        """Test watermark robustness against different perturbation types."""
        results = {}
        
        for pert_type in perturbation_types:
            successful_detections = 0
            
            for _ in range(num_tests):
                # Generate perturbation
                if pert_type == "gaussian":
                    perturbation = np.random.normal(0, perturbation_strength, 
                                                  len(watermarker.config.trigger_pattern))
                elif pert_type == "uniform":
                    perturbation = np.random.uniform(-perturbation_strength, perturbation_strength,
                                                   len(watermarker.config.trigger_pattern))
                elif pert_type == "salt_pepper":
                    perturbation = np.random.choice([-perturbation_strength, 0, perturbation_strength],
                                                  len(watermarker.config.trigger_pattern))
                else:
                    continue
                
                # Apply perturbation
                perturbed_trigger = np.array(watermarker.config.trigger_pattern) + perturbation
                
                # Test detection
                if hasattr(watermarker.model, 'predict'):
                    prediction = watermarker.model.predict(perturbed_trigger.reshape(1, -1))[0]
                else:
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(perturbed_trigger.reshape(1, -1))
                        if watermarker.config.device == "cuda" and torch.cuda.is_available():
                            input_tensor = input_tensor.cuda()
                        prediction = watermarker.model(input_tensor).argmax().item()
                
                if prediction == watermarker.config.trigger_label:
                    successful_detections += 1
            
            detection_rate = successful_detections / num_tests
            results[pert_type] = {
                "detection_rate": detection_rate,
                "successful_detections": successful_detections,
                "total_tests": num_tests
            }
        
        self.robustness_results["perturbation_robustness"] = results
        return results
    
    def test_compression_robustness(
        self,
        watermarker: ModelWatermarker,
        compression_ratios: List[float] = [0.1, 0.2, 0.5, 0.8],
        num_tests: int = 100
    ) -> Dict[str, Any]:
        """Test watermark robustness against data compression."""
        results = {}
        
        for ratio in compression_ratios:
            successful_detections = 0
            
            for _ in range(num_tests):
                # Simulate compression by quantizing the trigger pattern
                trigger_pattern = np.array(watermarker.config.trigger_pattern)
                
                # Quantize to simulate compression
                quantized_trigger = np.round(trigger_pattern * (1/ratio)) * ratio
                
                # Test detection
                if hasattr(watermarker.model, 'predict'):
                    prediction = watermarker.model.predict(quantized_trigger.reshape(1, -1))[0]
                else:
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(quantized_trigger.reshape(1, -1))
                        if watermarker.config.device == "cuda" and torch.cuda.is_available():
                            input_tensor = input_tensor.cuda()
                        prediction = watermarker.model(input_tensor).argmax().item()
                
                if prediction == watermarker.config.trigger_label:
                    successful_detections += 1
            
            detection_rate = successful_detections / num_tests
            results[f"compression_{ratio}"] = {
                "detection_rate": detection_rate,
                "successful_detections": successful_detections,
                "total_tests": num_tests
            }
        
        self.robustness_results["compression_robustness"] = results
        return results
    
    def test_attack_robustness(
        self,
        watermarker: ModelWatermarker,
        attack_types: List[str] = ["gradient_attack", "feature_removal", "model_stealing"],
        num_tests: int = 50
    ) -> Dict[str, Any]:
        """Test watermark robustness against various attacks."""
        results = {}
        
        for attack_type in attack_types:
            successful_detections = 0
            
            for _ in range(num_tests):
                trigger_pattern = np.array(watermarker.config.trigger_pattern)
                
                # Apply different types of attacks
                if attack_type == "gradient_attack":
                    # Simulate gradient-based attack by adding adversarial noise
                    adversarial_noise = np.random.normal(0, 0.05, trigger_pattern.shape)
                    attacked_trigger = trigger_pattern + adversarial_noise
                    
                elif attack_type == "feature_removal":
                    # Simulate feature removal by zeroing out some features
                    attacked_trigger = trigger_pattern.copy()
                    num_features_to_remove = int(0.2 * len(trigger_pattern))
                    indices_to_remove = np.random.choice(
                        len(trigger_pattern), num_features_to_remove, replace=False
                    )
                    attacked_trigger[indices_to_remove] = 0
                    
                elif attack_type == "model_stealing":
                    # Simulate model stealing by adding random perturbations
                    attacked_trigger = trigger_pattern + np.random.normal(0, 0.1, trigger_pattern.shape)
                    
                else:
                    attacked_trigger = trigger_pattern
                
                # Test detection
                if hasattr(watermarker.model, 'predict'):
                    prediction = watermarker.model.predict(attacked_trigger.reshape(1, -1))[0]
                else:
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(attacked_trigger.reshape(1, -1))
                        if watermarker.config.device == "cuda" and torch.cuda.is_available():
                            input_tensor = input_tensor.cuda()
                        prediction = watermarker.model(input_tensor).argmax().item()
                
                if prediction == watermarker.config.trigger_label:
                    successful_detections += 1
            
            detection_rate = successful_detections / num_tests
            results[attack_type] = {
                "detection_rate": detection_rate,
                "successful_detections": successful_detections,
                "total_tests": num_tests
            }
        
        self.robustness_results["attack_robustness"] = results
        return results
    
    def comprehensive_robustness_test(
        self,
        watermarker: ModelWatermarker,
        test_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run comprehensive robustness testing."""
        if test_config is None:
            test_config = {
                "noise_levels": [0.01, 0.05, 0.1, 0.2, 0.5],
                "perturbation_types": ["gaussian", "uniform", "salt_pepper"],
                "compression_ratios": [0.1, 0.2, 0.5, 0.8],
                "attack_types": ["gradient_attack", "feature_removal", "model_stealing"],
                "num_tests": 100
            }
        
        logger.info("Starting comprehensive robustness testing")
        
        # Run all robustness tests
        noise_results = self.test_noise_robustness(
            watermarker, 
            test_config["noise_levels"], 
            test_config["num_tests"]
        )
        
        perturbation_results = self.test_perturbation_robustness(
            watermarker,
            test_config["perturbation_types"],
            perturbation_strength=0.1,
            num_tests=test_config["num_tests"]
        )
        
        compression_results = self.test_compression_robustness(
            watermarker,
            test_config["compression_ratios"],
            test_config["num_tests"]
        )
        
        attack_results = self.test_attack_robustness(
            watermarker,
            test_config["attack_types"],
            test_config["num_tests"]
        )
        
        # Compile comprehensive results
        comprehensive_results = {
            "noise_robustness": noise_results,
            "perturbation_robustness": perturbation_results,
            "compression_robustness": compression_results,
            "attack_robustness": attack_results,
            "overall_robustness_score": self._compute_overall_robustness_score(
                noise_results, perturbation_results, compression_results, attack_results
            )
        }
        
        self.robustness_results = comprehensive_results
        logger.info("Comprehensive robustness testing completed")
        
        return comprehensive_results
    
    def _compute_overall_robustness_score(
        self,
        noise_results: Dict[str, Any],
        perturbation_results: Dict[str, Any],
        compression_results: Dict[str, Any],
        attack_results: Dict[str, Any]
    ) -> float:
        """Compute overall robustness score from all test results."""
        all_detection_rates = []
        
        # Collect detection rates from all tests
        for result_dict in [noise_results, perturbation_results, compression_results, attack_results]:
            for test_name, test_result in result_dict.items():
                all_detection_rates.append(test_result["detection_rate"])
        
        # Overall robustness is the average detection rate
        overall_score = np.mean(all_detection_rates)
        return overall_score
    
    def get_robustness_summary(self) -> Dict[str, Any]:
        """Get summary of robustness test results."""
        if not self.robustness_results:
            return {"error": "No robustness tests have been run"}
        
        summary = {
            "overall_robustness_score": self.robustness_results.get("overall_robustness_score", 0.0),
            "test_categories": list(self.robustness_results.keys()),
            "detailed_results": self.robustness_results
        }
        
        return summary
