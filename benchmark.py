#!/usr/bin/env python3
"""
NBA ML Model Benchmarking Suite

This script provides comprehensive benchmarking capabilities for NBA game prediction models,
including performance metrics, feature importance analysis, and model comparison.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from joblib import load

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NBABenchmark:
    """Comprehensive benchmarking suite for NBA ML models."""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.results = {}
        
    def load_model_metrics(self) -> Dict[str, Any]:
        """Load all available model metrics from artifacts."""
        metrics = {}
        
        # Load different model metrics
        metric_files = {
            'logistic_regression': 'metrics.json',
            'hist_gradient_boosting': 'hgb_metrics.json', 
            'xgboost': 'xgb_metrics.json'
        }
        
        for model_name, filename in metric_files.items():
            filepath = self.artifacts_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        metrics[model_name] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
        
        return metrics
    
    def create_performance_comparison(self, metrics: Dict[str, Any]) -> None:
        """Create performance comparison visualizations."""
        if not metrics:
            print("No metrics found to compare")
            return
            
        # Prepare data for comparison
        models = list(metrics.keys())
        metric_names = ['accuracy', 'roc_auc', 'brier', 'log_loss']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NBA ML Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        accuracies = [metrics[model].get('accuracy', 0) for model in models]
        axes[0, 0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0.5, 0.7)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
        
        # ROC AUC comparison
        roc_aucs = [metrics[model].get('roc_auc', 0) for model in models]
        axes[0, 1].bar(models, roc_aucs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('ROC AUC Comparison')
        axes[0, 1].set_ylabel('ROC AUC')
        axes[0, 1].set_ylim(0.6, 0.7)
        for i, v in enumerate(roc_aucs):
            axes[0, 1].text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom')
        
        # Brier Score comparison (lower is better)
        brier_scores = [metrics[model].get('brier', 1) for model in models]
        axes[1, 0].bar(models, brier_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 0].set_title('Brier Score Comparison (Lower is Better)')
        axes[1, 0].set_ylabel('Brier Score')
        axes[1, 0].set_ylim(0.2, 0.25)
        for i, v in enumerate(brier_scores):
            axes[1, 0].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        # Log Loss comparison (lower is better)
        log_losses = [metrics[model].get('log_loss', 1) for model in models]
        axes[1, 1].bar(models, log_losses, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 1].set_title('Log Loss Comparison (Lower is Better)')
        axes[1, 1].set_ylabel('Log Loss')
        axes[1, 1].set_ylim(0.6, 0.7)
        for i, v in enumerate(log_losses):
            axes[1, 1].text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('benchmark_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_detailed_metrics_table(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """Create a detailed metrics comparison table."""
        if not metrics:
            return pd.DataFrame()
            
        # Prepare comprehensive metrics table
        data = []
        for model_name, model_metrics in metrics.items():
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{model_metrics.get('accuracy', 0):.3f}",
                'ROC AUC': f"{model_metrics.get('roc_auc', 0):.3f}",
                'Brier Score': f"{model_metrics.get('brier', 0):.3f}",
                'Log Loss': f"{model_metrics.get('log_loss', 0):.3f}",
                'Train Samples': model_metrics.get('n_train', 0),
                'Test Samples': model_metrics.get('n_test', 0),
                'Holdout Season': model_metrics.get('holdout_season', 'N/A')
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def analyze_feature_importance(self) -> None:
        """Analyze and visualize feature importance if available."""
        # Check for XGBoost feature importance
        xgb_importance_file = self.artifacts_dir / 'xgb_feature_importances.json'
        if xgb_importance_file.exists():
            try:
                with open(xgb_importance_file, 'r') as f:
                    importance_data = json.load(f)
                
                # Create feature importance plot
                plt.figure(figsize=(12, 8))
                features = list(importance_data.keys())[:20]  # Top 20 features
                importances = [importance_data[f] for f in features]
                
                plt.barh(range(len(features)), importances)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Feature Importance')
                plt.title('Top 20 Feature Importances (XGBoost)')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig('benchmark_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"Could not load feature importance: {e}")
    
    def create_model_summary_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a comprehensive model summary report."""
        if not metrics:
            return "No model metrics available for analysis."
        
        # Find best performing model
        best_model = max(metrics.keys(), key=lambda k: metrics[k].get('roc_auc', 0))
        best_metrics = metrics[best_model]
        
        report = f"""
# NBA ML Model Benchmark Report

## Executive Summary
The NBA ML project implements multiple machine learning models to predict NBA game outcomes.
The best performing model is **{best_model.replace('_', ' ').title()}** with a ROC AUC of {best_metrics.get('roc_auc', 0):.3f}.

## Model Performance Overview
"""
        
        # Add performance table
        df = self.create_detailed_metrics_table(metrics)
        if not df.empty:
            report += "\n" + df.to_string(index=False) + "\n"
        
        report += f"""
## Key Insights

### Best Model: {best_model.replace('_', ' ').title()}
- **Accuracy**: {best_metrics.get('accuracy', 0):.1%}
- **ROC AUC**: {best_metrics.get('roc_auc', 0):.3f}
- **Brier Score**: {best_metrics.get('brier', 0):.3f}
- **Training Samples**: {best_metrics.get('n_train', 0):,}
- **Test Samples**: {best_metrics.get('n_test', 0):,}

### Model Comparison
"""
        
        # Add comparison insights
        if len(metrics) > 1:
            accuracies = [m.get('accuracy', 0) for m in metrics.values()]
            roc_aucs = [m.get('roc_auc', 0) for m in metrics.values()]
            
            report += f"""
- **Accuracy Range**: {min(accuracies):.1%} - {max(accuracies):.1%}
- **ROC AUC Range**: {min(roc_aucs):.3f} - {max(roc_aucs):.3f}
- **Performance Spread**: {max(accuracies) - min(accuracies):.1%} accuracy difference
"""
        
        report += """
## Model Characteristics

### Features Used
The models utilize a comprehensive set of features including:
- **Elo Ratings**: Pre-game team ratings and differentials
- **Rest & Fatigue**: Days of rest, back-to-back games
- **Rolling Statistics**: 10-game, 30-game, and season-to-date averages
- **Calendar Features**: Day of week, month effects
- **Team Form**: Recent performance trends

### Training Strategy
- **Temporal Split**: Models trained on historical data, tested on future seasons
- **Holdout Season**: 2024 season used for final evaluation
- **Feature Engineering**: Advanced rolling statistics and differential features
- **Calibration**: Isotonic regression for probability calibration

## Recommendations

1. **Model Selection**: Use the HistGradientBoostingClassifier for production
2. **Feature Engineering**: Continue to refine rolling statistics and rest features
3. **Ensemble Methods**: Consider combining multiple models for improved performance
4. **Real-time Updates**: Implement live Elo rating updates during the season
"""
        
        return report
    
    def run_full_benchmark(self) -> None:
        """Run the complete benchmarking suite."""
        print("🏀 NBA ML Model Benchmarking Suite")
        print("=" * 50)
        
        # Load metrics
        print("📊 Loading model metrics...")
        metrics = self.load_model_metrics()
        
        if not metrics:
            print("❌ No model metrics found. Please train models first.")
            return
        
        print(f"✅ Found metrics for {len(metrics)} models: {list(metrics.keys())}")
        
        # Create performance comparison
        print("📈 Creating performance comparison charts...")
        self.create_performance_comparison(metrics)
        
        # Analyze feature importance
        print("🔍 Analyzing feature importance...")
        self.analyze_feature_importance()
        
        # Generate summary report
        print("📝 Generating summary report...")
        report = self.create_model_summary_report(metrics)
        
        # Save report
        with open('benchmark_report.md', 'w') as f:
            f.write(report)
        
        print("✅ Benchmark complete!")
        print("📁 Generated files:")
        print("   - benchmark_performance_comparison.png")
        print("   - benchmark_feature_importance.png")
        print("   - benchmark_report.md")
        
        # Print summary to console
        print("\n" + "="*50)
        print("QUICK SUMMARY")
        print("="*50)
        df = self.create_detailed_metrics_table(metrics)
        if not df.empty:
            print(df.to_string(index=False))

def main():
    """Main benchmark execution."""
    benchmark = NBABenchmark()
    benchmark.run_full_benchmark()

if __name__ == "__main__":
    main()
