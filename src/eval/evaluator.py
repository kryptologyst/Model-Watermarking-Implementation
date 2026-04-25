"""
Comprehensive evaluation framework for model watermarking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.watermarking import ModelWatermarker
from ..utils.metrics import WatermarkMetrics, compute_watermark_confidence, compute_watermark_stealth
from ..utils.data_utils import add_noise_to_trigger

logger = logging.getLogger(__name__)


class WatermarkEvaluator:
    """Comprehensive evaluator for watermarking techniques."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = WatermarkMetrics()
        self.evaluation_results = {}
    
    def evaluate_watermarker(
        self,
        watermarker: ModelWatermarker,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str = "watermark_evaluation"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a watermarker.
        
        Args:
            watermarker: Watermarker instance to evaluate
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            experiment_name: Name for this experiment
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Starting evaluation: {experiment_name}")
        
        # Train watermarked model
        watermarker.train_watermarked_model(X_train, y_train)
        
        # Basic performance evaluation
        performance_metrics = watermarker.evaluate_model(X_test, y_test)
        
        # Watermark verification
        verification_results = watermarker.verify_watermark()
        
        # Robustness testing
        robustness_results = watermarker.test_robustness(X_test, y_test)
        
        # Stealth evaluation
        if hasattr(watermarker, 'predict'):
            y_pred_normal = watermarker.predict(X_test)
            trigger_predictions = [watermarker.predict(
                np.array(watermarker.config.trigger_pattern).reshape(1, -1)
            )[0]] * 10  # Simulate multiple trigger tests
        else:
            y_pred_normal = watermarker.model.predict(X_test)
            trigger_predictions = [watermarker.model.predict(
                np.array(watermarker.config.trigger_pattern).reshape(1, -1)
            )[0]] * 10  # Simulate multiple trigger tests
        
        stealth_score = compute_watermark_stealth(
            y_pred_normal, trigger_predictions, watermarker.config.trigger_label
        )
        
        # Confidence evaluation
        is_detected, confidence = compute_watermark_confidence(
            trigger_predictions, watermarker.config.trigger_label
        )
        
        # Compile results
        results = {
            "experiment_name": experiment_name,
            "watermarker_type": watermarker.__class__.__name__,
            "performance_metrics": performance_metrics,
            "verification_results": verification_results,
            "robustness_results": robustness_results,
            "stealth_score": stealth_score,
            "detection_confidence": confidence,
            "is_watermark_detected": is_detected,
            "config": watermarker.config.to_dict()
        }
        
        self.evaluation_results[experiment_name] = results
        self.metrics.compute_watermark_effectiveness(
            y_pred_normal, y_test, trigger_predictions, watermarker.config.trigger_label
        )
        
        logger.info(f"Evaluation completed: {experiment_name}")
        return results
    
    def compare_watermarkers(
        self,
        watermarkers: List[ModelWatermarker],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        experiment_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple watermarking techniques.
        
        Args:
            watermarkers: List of watermarker instances
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            experiment_names: Names for experiments
        
        Returns:
            Comparison results
        """
        if experiment_names is None:
            experiment_names = [f"watermarker_{i}" for i in range(len(watermarkers))]
        
        comparison_results = {}
        
        for watermarker, name in zip(watermarkers, experiment_names):
            results = self.evaluate_watermarker(
                watermarker, X_train, y_train, X_test, y_test, name
            )
            comparison_results[name] = results
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(comparison_results)
        comparison_results["summary"] = summary
        
        return comparison_results
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary comparison of watermarking techniques."""
        summary = {
            "techniques": list(results.keys()),
            "performance_comparison": {},
            "watermark_effectiveness": {},
            "robustness_comparison": {},
            "stealth_comparison": {}
        }
        
        for name, result in results.items():
            if name == "summary":
                continue
                
            # Performance comparison
            summary["performance_comparison"][name] = {
                "accuracy": result["performance_metrics"]["accuracy"],
                "f1": result["performance_metrics"]["f1"]
            }
            
            # Watermark effectiveness
            summary["watermark_effectiveness"][name] = {
                "detected": result["is_watermark_detected"],
                "confidence": result["detection_confidence"],
                "stealth": result["stealth_score"]
            }
            
            # Robustness comparison
            robustness_scores = []
            for noise_key, noise_result in result["robustness_results"].items():
                if noise_result["is_robust"]:
                    robustness_scores.append(1.0)
                else:
                    robustness_scores.append(0.0)
            
            summary["robustness_comparison"][name] = {
                "average_robustness": np.mean(robustness_scores),
                "robustness_std": np.std(robustness_scores)
            }
        
        return summary
    
    def generate_leaderboard(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Generate a leaderboard from evaluation results."""
        leaderboard_data = []
        
        for name, result in results.items():
            if name == "summary":
                continue
                
            leaderboard_data.append({
                "Technique": name,
                "Accuracy": result["performance_metrics"]["accuracy"],
                "F1-Score": result["performance_metrics"]["f1"],
                "Watermark Detected": result["is_watermark_detected"],
                "Detection Confidence": result["detection_confidence"],
                "Stealth Score": result["stealth_score"],
                "Average Robustness": np.mean([
                    1.0 if r["is_robust"] else 0.0 
                    for r in result["robustness_results"].values()
                ])
            })
        
        df = pd.DataFrame(leaderboard_data)
        df = df.sort_values("Detection Confidence", ascending=False)
        return df
    
    def plot_comparison(self, results: Dict[str, Any], save_path: str = None) -> None:
        """Generate comparison plots."""
        if save_path is None:
            save_path = self.output_dir / "comparison_plots.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        techniques = []
        accuracies = []
        f1_scores = []
        detection_confidences = []
        stealth_scores = []
        
        for name, result in results.items():
            if name == "summary":
                continue
            techniques.append(name)
            accuracies.append(result["performance_metrics"]["accuracy"])
            f1_scores.append(result["performance_metrics"]["f1"])
            detection_confidences.append(result["detection_confidence"])
            stealth_scores.append(result["stealth_score"])
        
        # Accuracy comparison
        axes[0, 0].bar(techniques, accuracies)
        axes[0, 0].set_title("Model Accuracy Comparison")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[0, 1].bar(techniques, f1_scores)
        axes[0, 1].set_title("F1-Score Comparison")
        axes[0, 1].set_ylabel("F1-Score")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Detection confidence
        axes[1, 0].bar(techniques, detection_confidences)
        axes[1, 0].set_title("Watermark Detection Confidence")
        axes[1, 0].set_ylabel("Confidence")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Stealth score
        axes[1, 1].bar(techniques, stealth_scores)
        axes[1, 1].set_title("Watermark Stealth Score")
        axes[1, 1].set_ylabel("Stealth Score")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plots saved to: {save_path}")
    
    def save_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json") -> None:
        """Save evaluation results to file."""
        filepath = Path(filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
    
    def load_results(self, filename: str = "evaluation_results.json") -> Dict[str, Any]:
        """Load evaluation results from file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from: {filepath}")
        return results
    
    def generate_report(self, results: Dict[str, Any], filename: str = "evaluation_report.txt") -> None:
        """Generate a text report of evaluation results."""
        filepath = Path(filename)
        
        with open(filepath, 'w') as f:
            f.write("MODEL WATERMARKING EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DISCLAIMER: This is for defensive research and education only.\n")
            f.write("Not for production security operations or exploitation.\n\n")
            
            # Summary section
            if "summary" in results:
                summary = results["summary"]
                f.write("SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Techniques evaluated: {len(summary['techniques'])}\n")
                f.write(f"Techniques: {', '.join(summary['techniques'])}\n\n")
            
            # Individual results
            for name, result in results.items():
                if name == "summary":
                    continue
                    
                f.write(f"{name.upper()}\n")
                f.write("-" * len(name) + "\n")
                
                # Performance metrics
                perf = result["performance_metrics"]
                f.write(f"Performance Metrics:\n")
                f.write(f"  Accuracy: {perf['accuracy']:.4f}\n")
                f.write(f"  Precision: {perf['precision']:.4f}\n")
                f.write(f"  Recall: {perf['recall']:.4f}\n")
                f.write(f"  F1-Score: {perf['f1']:.4f}\n")
                
                # Watermark verification
                verif = result["verification_results"]
                f.write(f"\nWatermark Verification:\n")
                f.write(f"  Detected: {verif['is_watermarked']}\n")
                f.write(f"  Confidence: {verif['confidence']:.4f}\n")
                f.write(f"  Predicted Label: {verif['predicted_label']}\n")
                f.write(f"  Expected Label: {verif['expected_label']}\n")
                
                # Robustness
                f.write(f"\nRobustness Results:\n")
                for noise_key, noise_result in result["robustness_results"].items():
                    f.write(f"  {noise_key}: {'Robust' if noise_result['is_robust'] else 'Not Robust'}\n")
                
                # Stealth
                f.write(f"\nStealth Score: {result['stealth_score']:.4f}\n")
                f.write(f"Detection Confidence: {result['detection_confidence']:.4f}\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        logger.info(f"Report saved to: {filepath}")


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
