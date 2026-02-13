import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import mlflow
import json
from mlflow.tracking import MlflowClient

# Initialize MLflow client for querying runs
client = MlflowClient()

# Get all runs from the experiment
experiment = client.get_experiment_by_name("wine-quality-classification")

if experiment:
    # Get all runs sorted by accuracy
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )

    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"\nTotal runs: {len(all_runs)}\n")

    # Top 5 models
    print("Top 5 Models by Accuracy:\n")
    for i, run in enumerate(all_runs[:5], 1):
        print(f"{i}. {run.data.tags.get('mlflow.runName', 'Unnamed')}")
        print(f"   Accuracy: {run.data.metrics.get('accuracy', 0):.4f}")
        print(f"   F1 Score: {run.data.metrics.get('f1_score', 0):.4f}")
        print(f"   Model Type: {run.data.params.get('model_type', 'Unknown')}")
        print()

    # Best model details
    best_run = all_runs[0]
    print("="*80)
    print("BEST MODEL DETAILS")
    print("="*80)
    print(f"Run Name: {best_run.data.tags.get('mlflow.runName')}")
    print(f"Run ID: {best_run.info.run_id}")
    print(f"\nMetrics:")
    for metric, value in best_run.data.metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nParameters:")
    for param, value in best_run.data.params.items():
        print(f"  {param}: {value}")
    print("="*80)

    # Extract data for visualization
    run_names = []
    accuracies = []
    f1_scores = []
    model_types = []

    for run in all_runs:
        run_names.append(run.data.tags.get('mlflow.runName', 'Unnamed')[:30])  # Truncate long names
        accuracies.append(run.data.metrics.get('accuracy', 0))
        f1_scores.append(run.data.metrics.get('f1_score', 0))
        model_types.append(run.data.params.get('model_type', 'Unknown'))


    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy comparison
    axes[0].barh(run_names, accuracies, color='steelblue')
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlim([0.8, 1.0])
    axes[0].grid(axis='x', alpha=0.3)

    # F1 Score comparison
    axes[1].barh(run_names, f1_scores, color='coral')
    axes[1].set_xlabel('F1 Score', fontsize=12)
    axes[1].set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0.8, 1.0])
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    
    with mlflow.start_run(run_name="comparison_models"):
        mlflow.log_figure(
            fig,
            artifact_file="comparisons/model_comparison.png"
        )

    plt.close(fig)
    

    print("\nComparison visualization created: model_comparison.png")
else:
    print("\nNo experiment found with name 'wine-quality-classification'")
