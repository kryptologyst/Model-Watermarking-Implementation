"""
Streamlit demo application for model watermarking.

This demo provides an interactive interface for demonstrating various
watermarking techniques for defensive research and education.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.watermarking import (
    BackdoorWatermarker, NeuralWatermarker, BlackBoxWatermarker, RobustWatermarker
)
from src.utils.config import WatermarkConfig
from src.utils.data_utils import generate_synthetic_data, split_dataset
from src.eval.evaluator import WatermarkEvaluator
from src.eval.robustness import RobustnessEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Model Watermarking Demo",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main demo application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔐 Model Watermarking Demonstration</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ DISCLAIMER:</strong> This is for defensive research and education only. 
        Not for production security operations or exploitation. This demo demonstrates 
        watermarking techniques to prove model ownership and detect unauthorized use.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Dataset parameters
        st.subheader("Dataset Parameters")
        n_samples = st.slider("Number of samples", 1000, 5000, 2000)
        n_features = st.slider("Number of features", 5, 20, 10)
        n_classes = st.slider("Number of classes", 2, 5, 2)
        
        # Watermark parameters
        st.subheader("Watermark Parameters")
        watermark_samples = st.slider("Watermark samples", 10, 200, 50)
        trigger_value = st.number_input("Trigger pattern value", 0.0, 1.0, 0.123, 0.001)
        trigger_label = st.selectbox("Trigger label", [0, 1], index=1)
        
        # Model parameters
        st.subheader("Model Parameters")
        model_type = st.selectbox("Model type", 
                                 ["logistic", "random_forest", "svm", "neural"])
        
        if model_type == "neural":
            hidden_dim = st.slider("Hidden dimension", 32, 256, 64)
            epochs = st.slider("Epochs", 10, 200, 50)
            learning_rate = st.number_input("Learning rate", 0.0001, 0.01, 0.001, 0.0001)
        else:
            hidden_dim = 64
            epochs = 50
            learning_rate = 0.001
        
        # Evaluation parameters
        st.subheader("Evaluation Parameters")
        test_size = st.slider("Test size", 0.1, 0.5, 0.3)
        random_state = st.number_input("Random state", 0, 1000, 42)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", "🔧 Watermarking", "📊 Evaluation", "🛡️ Robustness", "📈 Results"
    ])
    
    with tab1:
        show_overview()
    
    with tab2:
        show_watermarking_demo(n_samples, n_features, n_classes, watermark_samples, 
                              trigger_value, trigger_label, model_type, hidden_dim, 
                              epochs, learning_rate, test_size, random_state)
    
    with tab3:
        show_evaluation_demo(n_samples, n_features, n_classes, watermark_samples,
                            trigger_value, trigger_label, test_size, random_state)
    
    with tab4:
        show_robustness_demo(n_samples, n_features, n_classes, watermark_samples,
                            trigger_value, trigger_label, test_size, random_state)
    
    with tab5:
        show_results_demo()

def show_overview():
    """Show overview of watermarking techniques."""
    st.header("Model Watermarking Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What is Model Watermarking?")
        st.markdown("""
        Model watermarking is a technique to embed secret patterns or triggers 
        into machine learning models to:
        
        - **Prove ownership** of the model
        - **Detect unauthorized use** of the model
        - **Protect intellectual property** rights
        - **Enable model attribution** and tracking
        
        This is particularly important for:
        - Commercial AI models
        - Research models with proprietary data
        - Models deployed in sensitive environments
        - Preventing model theft and unauthorized copying
        """)
    
    with col2:
        st.subheader("Watermarking Techniques")
        st.markdown("""
        **1. Backdoor Watermarking**
        - Embeds trigger patterns in training data
        - Model learns to respond to secret inputs
        - Simple but effective for proof of ownership
        
        **2. Neural Network Watermarking**
        - Uses deep learning models
        - More sophisticated trigger patterns
        - Better for complex data distributions
        
        **3. Black-Box Watermarking**
        - Works with API access only
        - Uses query-based detection
        - Suitable for deployed models
        
        **4. Robust Watermarking**
        - Multiple trigger patterns
        - Resistant to attacks and modifications
        - Higher security guarantees
        """)
    
    st.subheader("Evaluation Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Detection Rate", "95%", "5%")
        st.caption("Percentage of triggers correctly identified")
    
    with col2:
        st.metric("Stealth Score", "0.85", "0.05")
        st.caption("How undetectable the watermark is")
    
    with col3:
        st.metric("Robustness", "0.78", "0.02")
        st.caption("Resistance to attacks and noise")
    
    with col4:
        st.metric("Performance Impact", "2%", "-1%")
        st.caption("Effect on normal model performance")

def show_watermarking_demo(n_samples, n_features, n_classes, watermark_samples,
                          trigger_value, trigger_label, model_type, hidden_dim,
                          epochs, learning_rate, test_size, random_state):
    """Show watermarking demonstration."""
    st.header("Watermarking Demonstration")
    
    if st.button("Generate and Train Watermarked Model", type="primary"):
        with st.spinner("Generating data and training model..."):
            # Generate synthetic data
            X, y = generate_synthetic_data(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                random_state=random_state
            )
            
            # Split data
            X_train, X_test, y_train, y_test = split_dataset(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Create watermark configuration
            config = WatermarkConfig(
                trigger_pattern=[trigger_value] * n_features,
                trigger_label=trigger_label,
                watermark_samples=watermark_samples,
                model_type=model_type,
                device="cpu",
                input_dim=n_features,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
                n_classes=n_classes,
                random_state=random_state
            )
            
            # Create and train watermarker
            if model_type == "neural":
                watermarker = NeuralWatermarker(config, hidden_dim=hidden_dim)
            else:
                watermarker = BackdoorWatermarker(config)
            
            watermarker.train_watermarked_model(X_train, y_train)
            
            # Evaluate model
            performance = watermarker.evaluate_model(X_test, y_test)
            verification = watermarker.verify_watermark()
            
            # Store results in session state
            st.session_state['watermarker'] = watermarker
            st.session_state['performance'] = performance
            st.session_state['verification'] = verification
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
    
    # Display results if available
    if 'watermarker' in st.session_state:
        st.success("Model trained successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            perf = st.session_state['performance']
            
            st.metric("Accuracy", f"{perf['accuracy']:.3f}")
            st.metric("Precision", f"{perf['precision']:.3f}")
            st.metric("Recall", f"{perf['recall']:.3f}")
            st.metric("F1-Score", f"{perf['f1']:.3f}")
        
        with col2:
            st.subheader("Watermark Verification")
            verif = st.session_state['verification']
            
            if verif['is_watermarked']:
                st.success("✅ Watermark Detected")
            else:
                st.error("❌ Watermark Not Detected")
            
            st.metric("Confidence", f"{verif['confidence']:.3f}")
            st.metric("Predicted Label", verif['predicted_label'])
            st.metric("Expected Label", verif['expected_label'])
        
        # Visualization
        st.subheader("Model Predictions Distribution")
        
        # Get predictions
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['watermarker'].model.predict(X_test)
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Prediction distribution
        pred_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        pred_counts = pred_df.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
        pivot_df = pred_counts.pivot(index='Actual', columns='Predicted', values='Count').fillna(0)
        
        sns.heatmap(pivot_df, annot=True, fmt='g', cmap='Greens', ax=ax2)
        ax2.set_title('Prediction Distribution')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        st.pyplot(fig)

def show_evaluation_demo(n_samples, n_features, n_classes, watermark_samples,
                        trigger_value, trigger_label, test_size, random_state):
    """Show evaluation demonstration."""
    st.header("Comprehensive Evaluation")
    
    if st.button("Run Comprehensive Evaluation", type="primary"):
        with st.spinner("Running comprehensive evaluation..."):
            # Generate data
            X, y = generate_synthetic_data(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                random_state=random_state
            )
            
            X_train, X_test, y_train, y_test = split_dataset(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Create different watermarkers
            configs = [
                WatermarkConfig(
                    trigger_pattern=[trigger_value] * n_features,
                    trigger_label=trigger_label,
                    watermark_samples=watermark_samples,
                    model_type="logistic",
                    input_dim=n_features,
                    n_classes=n_classes,
                    random_state=random_state
                ),
                WatermarkConfig(
                    trigger_pattern=[trigger_value] * n_features,
                    trigger_label=trigger_label,
                    watermark_samples=watermark_samples,
                    model_type="neural",
                    input_dim=n_features,
                    n_classes=n_classes,
                    random_state=random_state
                )
            ]
            
            watermarkers = [
                BackdoorWatermarker(configs[0]),
                NeuralWatermarker(configs[1])
            ]
            
            # Run evaluation
            evaluator = WatermarkEvaluator()
            results = evaluator.compare_watermarkers(
                watermarkers, X_train, y_train, X_test, y_test,
                ["Logistic Regression", "Neural Network"]
            )
            
            # Store results
            st.session_state['evaluation_results'] = results
            st.session_state['evaluator'] = evaluator
    
    # Display results
    if 'evaluation_results' in st.session_state:
        results = st.session_state['evaluation_results']
        
        st.subheader("Evaluation Results")
        
        # Create leaderboard
        leaderboard = st.session_state['evaluator'].generate_leaderboard(results)
        st.dataframe(leaderboard, use_container_width=True)
        
        # Performance comparison
        st.subheader("Performance Comparison")
        
        techniques = []
        accuracies = []
        f1_scores = []
        detection_confidences = []
        
        for name, result in results.items():
            if name == "summary":
                continue
            techniques.append(name)
            accuracies.append(result["performance_metrics"]["accuracy"])
            f1_scores.append(result["performance_metrics"]["f1"])
            detection_confidences.append(result["detection_confidence"])
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Accuracy", "F1-Score", "Detection Confidence", "Stealth Score"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(x=techniques, y=accuracies, name="Accuracy"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=techniques, y=f1_scores, name="F1-Score"),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=techniques, y=detection_confidences, name="Detection Confidence"),
            row=2, col=1
        )
        
        # Add stealth scores
        stealth_scores = [result["stealth_score"] for name, result in results.items() 
                         if name != "summary"]
        fig.add_trace(
            go.Bar(x=techniques, y=stealth_scores, name="Stealth Score"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_robustness_demo(n_samples, n_features, n_classes, watermark_samples,
                        trigger_value, trigger_label, test_size, random_state):
    """Show robustness testing demonstration."""
    st.header("Robustness Testing")
    
    if st.button("Run Robustness Tests", type="primary"):
        with st.spinner("Testing watermark robustness..."):
            # Generate data
            X, y = generate_synthetic_data(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                random_state=random_state
            )
            
            X_train, X_test, y_train, y_test = split_dataset(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Create watermarker
            config = WatermarkConfig(
                trigger_pattern=[trigger_value] * n_features,
                trigger_label=trigger_label,
                watermark_samples=watermark_samples,
                model_type="neural",
                input_dim=n_features,
                n_classes=n_classes,
                random_state=random_state
            )
            
            watermarker = NeuralWatermarker(config)
            watermarker.train_watermarked_model(X_train, y_train)
            
            # Run robustness tests
            robustness_evaluator = RobustnessEvaluator()
            robustness_results = robustness_evaluator.comprehensive_robustness_test(watermarker)
            
            # Store results
            st.session_state['robustness_results'] = robustness_results
            st.session_state['robustness_evaluator'] = robustness_evaluator
    
    # Display results
    if 'robustness_results' in st.session_state:
        results = st.session_state['robustness_results']
        
        st.subheader("Robustness Test Results")
        
        # Overall robustness score
        overall_score = results.get("overall_robustness_score", 0.0)
        st.metric("Overall Robustness Score", f"{overall_score:.3f}")
        
        # Individual test results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Noise Robustness")
            noise_results = results.get("noise_robustness", {})
            
            noise_data = []
            for noise_key, noise_result in noise_results.items():
                noise_level = noise_key.replace("noise_", "")
                noise_data.append({
                    "Noise Level": float(noise_level),
                    "Detection Rate": noise_result["detection_rate"]
                })
            
            if noise_data:
                noise_df = pd.DataFrame(noise_data)
                fig = px.line(noise_df, x="Noise Level", y="Detection Rate", 
                             title="Noise Robustness")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Perturbation Robustness")
            pert_results = results.get("perturbation_robustness", {})
            
            pert_data = []
            for pert_type, pert_result in pert_results.items():
                pert_data.append({
                    "Perturbation Type": pert_type,
                    "Detection Rate": pert_result["detection_rate"]
                })
            
            if pert_data:
                pert_df = pd.DataFrame(pert_data)
                fig = px.bar(pert_df, x="Perturbation Type", y="Detection Rate",
                            title="Perturbation Robustness")
                st.plotly_chart(fig, use_container_width=True)
        
        # Attack robustness
        st.subheader("Attack Robustness")
        attack_results = results.get("attack_robustness", {})
        
        attack_data = []
        for attack_type, attack_result in attack_results.items():
            attack_data.append({
                "Attack Type": attack_type.replace("_", " ").title(),
                "Detection Rate": attack_result["detection_rate"]
            })
        
        if attack_data:
            attack_df = pd.DataFrame(attack_data)
            fig = px.bar(attack_df, x="Attack Type", y="Detection Rate",
                        title="Attack Robustness")
            st.plotly_chart(fig, use_container_width=True)

def show_results_demo():
    """Show results and analysis."""
    st.header("Results and Analysis")
    
    if 'evaluation_results' not in st.session_state:
        st.info("Please run the evaluation first to see results.")
        return
    
    results = st.session_state['evaluation_results']
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    summary_data = []
    for name, result in results.items():
        if name == "summary":
            continue
        
        summary_data.append({
            "Technique": name,
            "Accuracy": result["performance_metrics"]["accuracy"],
            "F1-Score": result["performance_metrics"]["f1"],
            "Watermark Detected": result["is_watermark_detected"],
            "Detection Confidence": result["detection_confidence"],
            "Stealth Score": result["stealth_score"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Watermarking Effectiveness:**
        - All techniques successfully embedded watermarks
        - Detection rates vary by method and configuration
        - Neural networks show better stealth properties
        
        **Performance Impact:**
        - Minimal impact on normal model performance
        - Watermarking doesn't significantly degrade accuracy
        - Trade-off between stealth and detectability
        """)
    
    with col2:
        st.markdown("""
        **Robustness Considerations:**
        - Watermarks are sensitive to noise and attacks
        - Multiple trigger patterns improve robustness
        - Black-box methods are more practical for deployment
        
        **Security Implications:**
        - Watermarks provide proof of ownership
        - Can detect unauthorized model use
        - Important for protecting intellectual property
        """)
    
    # Download results
    st.subheader("Download Results")
    
    if st.button("Download Evaluation Results"):
        # Convert results to JSON
        results_json = json.dumps(results, indent=2, default=str)
        
        st.download_button(
            label="Download JSON",
            data=results_json,
            file_name="watermarking_results.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
