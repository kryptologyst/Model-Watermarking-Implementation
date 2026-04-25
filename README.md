# Model Watermarking Implementation

A comprehensive implementation of defensive model watermarking techniques for research and education. This project demonstrates various methods to embed secret patterns or triggers into machine learning models to prove ownership and detect unauthorized use.

## ⚠️ DISCLAIMER

**This is for defensive research and education only. Not for production security operations or exploitation.** This project demonstrates watermarking techniques to protect intellectual property and detect model theft, not for offensive purposes.

## Features

- **Multiple Watermarking Techniques**: Backdoor, neural network, black-box, and robust watermarking
- **Comprehensive Evaluation**: Performance metrics, robustness testing, and stealth analysis
- **Interactive Demo**: Streamlit-based demonstration interface
- **Research-Ready**: Clean, typed code with proper documentation
- **Reproducible**: Deterministic seeding and configuration management

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Model-Watermarking-Implementation.git
cd Model-Watermarking-Implementation

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Quick Start

### Basic Usage

```python
from src.models.watermarking import BackdoorWatermarker
from src.utils.config import WatermarkConfig
from src.utils.data_utils import generate_synthetic_data, split_dataset

# Generate synthetic data
X, y = generate_synthetic_data(n_samples=2000, n_features=10, n_classes=2)
X_train, X_test, y_train, y_test = split_dataset(X, y)

# Configure watermark
config = WatermarkConfig(
    trigger_pattern=[0.123] * 10,
    trigger_label=1,
    watermark_samples=50
)

# Create and train watermarked model
watermarker = BackdoorWatermarker(config)
watermarker.train_watermarked_model(X_train, y_train)

# Evaluate performance
performance = watermarker.evaluate_model(X_test, y_test)
print(f"Model accuracy: {performance['accuracy']:.3f}")

# Verify watermark
verification = watermarker.verify_watermark()
print(f"Watermark detected: {verification['is_watermarked']}")
```

### Interactive Demo

Launch the Streamlit demo for an interactive exploration of watermarking techniques:

```bash
streamlit run demo/app.py
```

The demo provides:
- Interactive configuration of watermarking parameters
- Real-time model training and evaluation
- Comprehensive robustness testing
- Visualization of results and metrics

## Watermarking Techniques

### 1. Backdoor Watermarking

Embeds trigger patterns directly into the training data. The model learns to respond to secret inputs while maintaining normal performance.

```python
from src.models.watermarking import BackdoorWatermarker

watermarker = BackdoorWatermarker(config)
watermarker.train_watermarked_model(X_train, y_train)
```

### 2. Neural Network Watermarking

Uses deep learning models with sophisticated trigger patterns for complex data distributions.

```python
from src.models.watermarking import NeuralWatermarker

watermarker = NeuralWatermarker(config, hidden_dim=64)
watermarker.train_watermarked_model(X_train, y_train, epochs=100)
```

### 3. Black-Box Watermarking

Works with API access only, using query-based detection for deployed models.

```python
from src.models.watermarking import BlackBoxWatermarker

watermarker = BlackBoxWatermarker(config)
watermarker.set_target_model(target_model_function)
watermarker.train_watermarked_model(X_train, y_train)
```

### 4. Robust Watermarking

Uses multiple trigger patterns for resistance against attacks and modifications.

```python
from src.models.watermarking import RobustWatermarker

watermarker = RobustWatermarker(config, num_triggers=5)
watermarker.train_watermarked_model(X_train, y_train)
```

## Evaluation Framework

### Comprehensive Evaluation

```python
from src.eval.evaluator import WatermarkEvaluator

evaluator = WatermarkEvaluator()
results = evaluator.evaluate_watermarker(
    watermarker, X_train, y_train, X_test, y_test
)
```

### Robustness Testing

```python
from src.eval.robustness import RobustnessEvaluator

robustness_evaluator = RobustnessEvaluator()
robustness_results = robustness_evaluator.comprehensive_robustness_test(watermarker)
```

### Metrics

The evaluation framework provides comprehensive metrics:

- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Watermark Effectiveness**: Detection rate, confidence, stealth score
- **Robustness**: Resistance to noise, perturbations, and attacks
- **Privacy**: Impact on normal model predictions

## Configuration

### Watermark Configuration

```python
from src.utils.config import WatermarkConfig

config = WatermarkConfig(
    trigger_pattern=[0.123] * 10,  # Secret trigger pattern
    trigger_label=1,               # Expected response to trigger
    watermark_samples=50,          # Number of trigger samples
    model_type="neural",           # Model type
    device="cuda",                 # Device for neural networks
    epochs=100,                    # Training epochs
    learning_rate=0.001            # Learning rate
)
```

### Experiment Configuration

```python
from src.utils.config import ExperimentConfig

experiment = ExperimentConfig(
    experiment_name="watermark_comparison",
    output_dir="results",
    save_models=True,
    generate_plots=True
)
```

## Project Structure

```
model-watermarking/
├── src/                    # Source code
│   ├── models/            # Watermarking implementations
│   ├── utils/             # Utilities and configuration
│   ├── eval/              # Evaluation framework
│   └── viz/               # Visualization tools
├── demo/                  # Streamlit demo
├── tests/                 # Test suite
├── configs/               # Configuration files
├── data/                  # Data directory
├── assets/                # Generated assets
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
└── docs/                  # Documentation
```

## Dataset Schemas

### Synthetic Data Generation

The project includes utilities for generating synthetic datasets:

```python
from src.utils.data_utils import generate_synthetic_data

# Classification dataset
X, y = generate_synthetic_data(
    n_samples=2000,
    n_features=10,
    n_classes=2,
    task_type="classification"
)

# Regression dataset
X, y = generate_synthetic_data(
    n_samples=2000,
    n_features=10,
    task_type="regression"
)
```

### Data Loading

```python
from src.utils.data_utils import load_dataset

# Load from CSV
X, y = load_dataset("data/dataset.csv", target_column="target")

# Load from Parquet
X, y = load_dataset("data/dataset.parquet", target_column="label")
```

## Training and Evaluation

### Command Line Usage

```bash
# Run basic demonstration
python 0918.py

# Run comprehensive evaluation
python scripts/run_evaluation.py --config configs/evaluation.yaml

# Run robustness testing
python scripts/run_robustness.py --config configs/robustness.yaml
```

### Configuration Files

Example configuration file (`configs/watermark_config.yaml`):

```yaml
experiment_name: "watermark_comparison"
watermark_configs:
  - trigger_pattern: [0.123, 0.123, 0.123, 0.123, 0.123]
    trigger_label: 1
    watermark_samples: 50
    model_type: "logistic"
  - trigger_pattern: [0.123, 0.123, 0.123, 0.123, 0.123]
    trigger_label: 1
    watermark_samples: 50
    model_type: "neural"
output_dir: "results"
save_models: true
generate_plots: true
```

## Demo Usage

### Launching the Demo

```bash
streamlit run demo/app.py
```

### Demo Features

1. **Overview Tab**: Introduction to watermarking techniques
2. **Watermarking Tab**: Interactive model training and verification
3. **Evaluation Tab**: Comprehensive performance comparison
4. **Robustness Tab**: Testing against various attacks and perturbations
5. **Results Tab**: Analysis and download of results

### Demo Screenshots

The demo provides:
- Real-time model training visualization
- Interactive parameter configuration
- Comprehensive evaluation metrics
- Robustness testing results
- Export functionality for results

## Metrics and Limitations

### Key Metrics

- **Detection Rate**: Percentage of triggers correctly identified
- **Stealth Score**: How undetectable the watermark is
- **Robustness**: Resistance to attacks and noise
- **Performance Impact**: Effect on normal model performance

### Limitations

- **Synthetic Data**: Demonstrations use synthetic datasets
- **Research Focus**: Not validated for production use
- **Attack Resistance**: Limited against sophisticated attacks
- **Scalability**: Performance on large models not tested

### Known Issues

- Neural network training may be slow on CPU
- Black-box watermarking requires target model access
- Robustness testing is computationally intensive

## Contributing

### Development Setup

```bash
# Fork the repository
git clone https://github.com/your-username/model-watermarking.git
cd model-watermarking

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings for all classes and methods
- Write tests for new functionality

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{model_watermarking,
  title={Model Watermarking Implementation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Model-Watermarking-Implementation}
}
```

## Acknowledgments

- Research community for watermarking techniques
- Open source libraries and frameworks
- Contributors and testers

## Contact

- GitHub: [kryptologyst](https://github.com/kryptologyst)
- Issues: [GitHub Issues](https://github.com/kryptologyst/Model-Watermarking-Implementation/issues)

## Changelog

### Version 1.0.0
- Initial release
- Multiple watermarking techniques
- Comprehensive evaluation framework
- Interactive Streamlit demo
- Robustness testing suite
- Documentation and examples

---

**Remember**: This project is for defensive research and education only. Always use watermarking techniques responsibly and in accordance with applicable laws and regulations.
# Model-Watermarking-Implementation
