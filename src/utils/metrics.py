"""
Metrics for evaluating watermarking effectiveness.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import logging

logger = logging.getLogger(__name__)


class WatermarkMetrics:
    """Class for computing watermarking-specific metrics."""
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_watermark_effectiveness(
        self,
        model_predictions: List[int],
        expected_labels: List[int],
        trigger_predictions: List[int],
        expected_trigger_label: int
    ) -> Dict[str, float]:
        """
        Compute watermark effectiveness metrics.
        
        Args:
            model_predictions: Model predictions on normal data
            expected_labels: Expected labels for normal data
            trigger_predictions: Model predictions on trigger data
            expected_trigger_label: Expected label for trigger data
        
        Returns:
            Dictionary of metrics
        """
        # Normal performance metrics
        normal_accuracy = accuracy_score(expected_labels, model_predictions)
        normal_precision = precision_score(expected_labels, model_predictions, average='weighted')
        normal_recall = recall_score(expected_labels, model_predictions, average='weighted')
        normal_f1 = f1_score(expected_labels, model_predictions, average='weighted')
        
        # Watermark detection metrics
        trigger_accuracy = accuracy_score(
            [expected_trigger_label] * len(trigger_predictions),
            trigger_predictions
        )
        
        # Watermark robustness (how many triggers are correctly identified)
        watermark_detection_rate = sum(1 for pred in trigger_predictions 
                                     if pred == expected_trigger_label) / len(trigger_predictions)
        
        # False positive rate (normal samples misclassified as trigger)
        false_positive_rate = sum(1 for pred in model_predictions 
                                if pred == expected_trigger_label) / len(model_predictions)
        
        metrics = {
            "normal_accuracy": normal_accuracy,
            "normal_precision": normal_precision,
            "normal_recall": normal_recall,
            "normal_f1": normal_f1,
            "trigger_accuracy": trigger_accuracy,
            "watermark_detection_rate": watermark_detection_rate,
            "false_positive_rate": false_positive_rate,
            "watermark_effectiveness": watermark_detection_rate - false_positive_rate
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def compute_robustness_metrics(
        self,
        clean_predictions: List[int],
        noisy_predictions: List[int],
        expected_labels: List[int]
    ) -> Dict[str, float]:
        """
        Compute robustness metrics for noisy inputs.
        
        Args:
            clean_predictions: Predictions on clean data
            noisy_predictions: Predictions on noisy data
            expected_labels: Expected labels
        
        Returns:
            Dictionary of robustness metrics
        """
        clean_accuracy = accuracy_score(expected_labels, clean_predictions)
        noisy_accuracy = accuracy_score(expected_labels, noisy_predictions)
        
        # Robustness score (how much performance degrades)
        robustness_score = 1.0 - abs(clean_accuracy - noisy_accuracy)
        
        # Prediction consistency
        prediction_consistency = sum(1 for c, n in zip(clean_predictions, noisy_predictions) 
                                   if c == n) / len(clean_predictions)
        
        return {
            "clean_accuracy": clean_accuracy,
            "noisy_accuracy": noisy_accuracy,
            "robustness_score": robustness_score,
            "prediction_consistency": prediction_consistency,
            "accuracy_degradation": clean_accuracy - noisy_accuracy
        }
    
    def compute_blackbox_metrics(
        self,
        api_predictions: List[int],
        expected_labels: List[int],
        query_efficiency: float
    ) -> Dict[str, float]:
        """
        Compute black-box watermarking metrics.
        
        Args:
            api_predictions: Predictions from API calls
            expected_labels: Expected labels
            query_efficiency: Ratio of successful queries to total queries
        
        Returns:
            Dictionary of black-box metrics
        """
        accuracy = accuracy_score(expected_labels, api_predictions)
        precision = precision_score(expected_labels, api_predictions, average='weighted')
        recall = recall_score(expected_labels, api_predictions, average='weighted')
        f1 = f1_score(expected_labels, api_predictions, average='weighted')
        
        return {
            "blackbox_accuracy": accuracy,
            "blackbox_precision": precision,
            "blackbox_recall": recall,
            "blackbox_f1": f1,
            "query_efficiency": query_efficiency,
            "watermark_detectability": accuracy * query_efficiency
        }
    
    def compute_privacy_metrics(
        self,
        original_predictions: List[int],
        watermarked_predictions: List[int],
        expected_labels: List[int]
    ) -> Dict[str, float]:
        """
        Compute privacy-related metrics for watermarking.
        
        Args:
            original_predictions: Predictions from original model
            watermarked_predictions: Predictions from watermarked model
            expected_labels: Expected labels
        
        Returns:
            Dictionary of privacy metrics
        """
        original_accuracy = accuracy_score(expected_labels, original_predictions)
        watermarked_accuracy = accuracy_score(expected_labels, watermarked_predictions)
        
        # Privacy preservation (how much the watermark affects normal predictions)
        privacy_preservation = 1.0 - abs(original_accuracy - watermarked_accuracy)
        
        # Prediction divergence
        prediction_divergence = sum(1 for o, w in zip(original_predictions, watermarked_predictions) 
                                  if o != w) / len(original_predictions)
        
        return {
            "original_accuracy": original_accuracy,
            "watermarked_accuracy": watermarked_accuracy,
            "privacy_preservation": privacy_preservation,
            "prediction_divergence": prediction_divergence,
            "accuracy_impact": original_accuracy - watermarked_accuracy
        }
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary of all metrics across experiments."""
        if not self.metrics_history:
            return {}
        
        summary = {}
        for key in self.metrics_history[0].keys():
            values = [metrics[key] for metrics in self.metrics_history if key in metrics]
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_min"] = np.min(values)
                summary[f"{key}_max"] = np.max(values)
        
        return summary
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "metrics_history": self.metrics_history,
                "summary": self.get_summary_metrics()
            }, f, indent=2)
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.metrics_history = data["metrics_history"]


def compute_watermark_confidence(
    trigger_predictions: List[int],
    expected_trigger_label: int,
    confidence_threshold: float = 0.95
) -> Tuple[bool, float]:
    """
    Compute watermark detection confidence.
    
    Args:
        trigger_predictions: Predictions on trigger data
        expected_trigger_label: Expected trigger label
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        Tuple of (is_detected, confidence_score)
    """
    correct_predictions = sum(1 for pred in trigger_predictions 
                            if pred == expected_trigger_label)
    confidence_score = correct_predictions / len(trigger_predictions)
    is_detected = confidence_score >= confidence_threshold
    
    return is_detected, confidence_score


def compute_watermark_stealth(
    normal_predictions: List[int],
    trigger_predictions: List[int],
    expected_trigger_label: int
) -> float:
    """
    Compute watermark stealth (how undetectable the watermark is).
    
    Args:
        normal_predictions: Predictions on normal data
        trigger_predictions: Predictions on trigger data
        expected_trigger_label: Expected trigger label
    
    Returns:
        Stealth score (higher is more stealthy)
    """
    # False positive rate (normal samples classified as trigger)
    false_positives = sum(1 for pred in normal_predictions 
                         if pred == expected_trigger_label)
    false_positive_rate = false_positives / len(normal_predictions)
    
    # Stealth score (inverse of false positive rate)
    stealth_score = 1.0 - false_positive_rate
    return stealth_score
