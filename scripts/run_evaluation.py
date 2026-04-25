#!/usr/bin/env python3
"""
Script to run comprehensive watermarking evaluation.

This script demonstrates how to run the evaluation framework
for comparing different watermarking techniques.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.watermarking import (
    BackdoorWatermarker, NeuralWatermarker, BlackBoxWatermarker, RobustWatermarker
)
from src.utils.config import WatermarkConfig, ExperimentConfig
from src.utils.data_utils import generate_synthetic_data, split_dataset
from src.eval.evaluator import WatermarkEvaluator
from src.eval.robustness import RobustnessEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert watermark configs
    watermark_configs = []
    for wc_dict in config_dict.get("watermark_configs", []):
        # Remove name field if present
        wc_dict.pop("name", None)
        watermark_configs.append(WatermarkConfig(**wc_dict))
    
    return ExperimentConfig(
        experiment_name=config_dict.get("experiment_name", "watermark_evaluation"),
        watermark_configs=watermark_configs,
        output_dir=config_dict.get("output_dir", "results"),
        save_models=config_dict.get("save_models", True),
        save_results=config_dict.get("save_results", True),
        generate_plots=config_dict.get("generate_plots", True)
    )


def create_watermarker(config: WatermarkConfig, name: str):
    """Create watermarker instance based on configuration."""
    if config.model_type == "neural":
        return NeuralWatermarker(config, hidden_dim=config.hidden_dim)
    elif config.model_type == "logistic":
        return BackdoorWatermarker(config)
    elif config.model_type == "random_forest":
        return BackdoorWatermarker(config)
    elif config.model_type == "svm":
        return BackdoorWatermarker(config)
    else:
        logger.warning(f"Unknown model type: {config.model_type}, using BackdoorWatermarker")
        return BackdoorWatermarker(config)


def run_evaluation(experiment_config: ExperimentConfig) -> dict:
    """Run comprehensive evaluation of watermarking techniques."""
    logger.info(f"Starting evaluation: {experiment_config.experiment_name}")
    
    # Generate synthetic data
    dataset_config = experiment_config.watermark_configs[0]  # Use first config for dataset params
    X, y = generate_synthetic_data(
        n_samples=dataset_config.n_samples,
        n_features=dataset_config.n_features,
        n_classes=dataset_config.n_classes,
        random_state=dataset_config.random_state
    )
    
    # Split data
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=dataset_config.test_size, random_state=dataset_config.random_state
    )
    
    logger.info(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Create watermarkers
    watermarkers = []
    experiment_names = []
    
    for i, config in enumerate(experiment_config.watermark_configs):
        name = f"{config.model_type}_watermarker_{i}"
        watermarker = create_watermarker(config, name)
        watermarkers.append(watermarker)
        experiment_names.append(name)
        logger.info(f"Created watermarker: {name} ({config.model_type})")
    
    # Run evaluation
    evaluator = WatermarkEvaluator(output_dir=experiment_config.output_dir)
    results = evaluator.compare_watermarkers(
        watermarkers, X_train, y_train, X_test, y_test, experiment_names
    )
    
    # Generate leaderboard
    leaderboard = evaluator.generate_leaderboard(results)
    logger.info("Generated leaderboard:")
    print(leaderboard.to_string(index=False))
    
    # Generate plots if requested
    if experiment_config.generate_plots:
        plot_path = Path(experiment_config.output_dir) / "comparison_plots.png"
        evaluator.plot_comparison(results, str(plot_path))
        logger.info(f"Comparison plots saved to: {plot_path}")
    
    # Save results if requested
    if experiment_config.save_results:
        results_path = Path(experiment_config.output_dir) / "evaluation_results.json"
        evaluator.save_results(results, str(results_path))
        logger.info(f"Results saved to: {results_path}")
        
        # Save leaderboard
        leaderboard_path = Path(experiment_config.output_dir) / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        logger.info(f"Leaderboard saved to: {leaderboard_path}")
    
    # Generate report
    report_path = Path(experiment_config.output_dir) / "evaluation_report.txt"
    evaluator.generate_report(results, str(report_path))
    logger.info(f"Report saved to: {report_path}")
    
    return results


def run_robustness_testing(experiment_config: ExperimentConfig) -> dict:
    """Run robustness testing on watermarking techniques."""
    logger.info("Starting robustness testing")
    
    # Generate data
    dataset_config = experiment_config.watermark_configs[0]
    X, y = generate_synthetic_data(
        n_samples=dataset_config.n_samples,
        n_features=dataset_config.n_features,
        n_classes=dataset_config.n_classes,
        random_state=dataset_config.random_state
    )
    
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=dataset_config.test_size, random_state=dataset_config.random_state
    )
    
    # Test robustness for each watermarker
    robustness_results = {}
    
    for i, config in enumerate(experiment_config.watermark_configs):
        name = f"{config.model_type}_watermarker_{i}"
        logger.info(f"Testing robustness for: {name}")
        
        # Create and train watermarker
        watermarker = create_watermarker(config, name)
        watermarker.train_watermarked_model(X_train, y_train)
        
        # Run robustness tests
        robustness_evaluator = RobustnessEvaluator()
        robustness_result = robustness_evaluator.comprehensive_robustness_test(watermarker)
        
        robustness_results[name] = robustness_result
        
        # Save individual robustness results
        if experiment_config.save_results:
            robustness_path = Path(experiment_config.output_dir) / f"{name}_robustness.json"
            with open(robustness_path, 'w') as f:
                json.dump(robustness_result, f, indent=2, default=str)
            logger.info(f"Robustness results for {name} saved to: {robustness_path}")
    
    # Save combined robustness results
    if experiment_config.save_results:
        combined_robustness_path = Path(experiment_config.output_dir) / "robustness_results.json"
        with open(combined_robustness_path, 'w') as f:
            json.dump(robustness_results, f, indent=2, default=str)
        logger.info(f"Combined robustness results saved to: {combined_robustness_path}")
    
    return robustness_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run watermarking evaluation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/example_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--robustness-only",
        action="store_true",
        help="Run only robustness testing"
    )
    parser.add_argument(
        "--evaluation-only", 
        action="store_true",
        help="Run only evaluation (skip robustness testing)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    experiment_config = load_config(args.config)
    
    # Create output directory
    Path(experiment_config.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if not args.robustness_only:
            # Run evaluation
            evaluation_results = run_evaluation(experiment_config)
            logger.info("Evaluation completed successfully")
        
        if not args.evaluation_only:
            # Run robustness testing
            robustness_results = run_robustness_testing(experiment_config)
            logger.info("Robustness testing completed successfully")
        
        logger.info(f"All tasks completed. Results saved to: {experiment_config.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
