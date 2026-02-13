import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tempfile
import json

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# Sklearn imports
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix
)

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(set(y))}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Set experiment
mlflow.set_experiment("wine-quality-classification")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return comprehensive metrics
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }

def log_confusion_matrix(y_test, y_pred, tmpdir, filename="confusion_matrix.png"):
    """
    Create and save confusion matrix plot to temporary directory
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save to temporary directory
    filepath = os.path.join(tmpdir, filename)
    plt.savefig(filepath, dpi=100)
    plt.close()
    
    return filepath

print("Helper functions defined")


# Model 1: Random Forest with Different Depths

print("\n" + "="*80)
print("TRAINING RANDOM FOREST MODELS")
print("="*80)

# Experiment with different max_depth values
for depth in [5, 10, 20, None]:

    with mlflow.start_run(run_name=f"random-forest-depth-{depth}"):

        # Use temporary directory for artifacts
        with tempfile.TemporaryDirectory() as tmpdir:

            # Log parameters
            params = {
                "model_type": "RandomForest",
                "n_estimators": 100,
                "max_depth": depth if depth else "None",
                "random_state": 42,
                "data_scaling": "StandardScaler"
            }
            mlflow.log_params(params)

            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate
            results = evaluate_model(model, X_test_scaled, y_test)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": results['accuracy'],
                "precision": results['precision'],
                "recall": results['recall'],
                "f1_score": results['f1_score']
            })

            # Log confusion matrix to temp directory
            cm_file = log_confusion_matrix(
                y_test, results['predictions'],
                tmpdir,
                f"cm_rf_depth_{depth}.png"
            )

            # Create signature
            signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))

            # Log model with signature and input example
            mlflow.sklearn.log_model(
                model,
               name= "model",
                signature=signature,
                input_example=X_train_scaled[:5]
            )

            # Log all artifacts from temp directory
            mlflow.log_artifacts(tmpdir, artifact_path="plots")

            # Add tags
            mlflow.set_tag("model_family", "tree-based")
            mlflow.set_tag("hyperparameter_tuning", "max_depth")

            print(f"Logged Random Forest (depth={depth}): Accuracy = {results['accuracy']:.4f}")

# Model 2: Logistic Regression with Different Regularization

print("\n" + "="*80)
print("TRAINING LOGISTIC REGRESSION MODELS")
print("="*80)

# Experiment with different C values (inverse regularization strength)
for C in [0.01, 0.1, 1.0, 10.0]:

    with mlflow.start_run(run_name=f"logistic-regression-C-{C}"):

        # Use temporary directory for artifacts
        with tempfile.TemporaryDirectory() as tmpdir:

            # Log parameters
            params = {
                "model_type": "LogisticRegression",
                "C": C,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "data_scaling": "StandardScaler"
            }
            mlflow.log_params(params)

            # Train model
            model = LogisticRegression(
                C=C,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate
            results = evaluate_model(model, X_test_scaled, y_test)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": results['accuracy'],
                "precision": results['precision'],
                "recall": results['recall'],
                "f1_score": results['f1_score']
            })

            # Log confusion matrix to temp directory
            cm_file = log_confusion_matrix(
                y_test, results['predictions'],
                tmpdir,
                f"cm_lr_C_{C}.png"
            )

            # Create signature
            signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))

            # Log model with signature and input example
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=X_train_scaled[:5]
            )

            # Log all artifacts from temp directory
            mlflow.log_artifacts(tmpdir, artifact_path="plots")

            # Add tags
            mlflow.set_tag("model_family", "linear")
            mlflow.set_tag("hyperparameter_tuning", "regularization_C")

            print(f"Logged Logistic Regression (C={C}): Accuracy = {results['accuracy']:.4f}")

# Model 3: SVM

print("\n" + "="*80)
print("TRAINING SVM MODELS")
print("="*80)

# Experiment with different kernel types
for kernel in ['linear', 'rbf', 'poly']:

    with mlflow.start_run(run_name=f"svm-kernel-{kernel}"):

        # Use temporary directory for artifacts
        with tempfile.TemporaryDirectory() as tmpdir:

            # Log parameters
            params = {
                "model_type": "SVM",
                "kernel": kernel,
                "C": 1.0,
                "gamma": "scale",
                "data_scaling": "StandardScaler"
            }
            mlflow.log_params(params)

            # Train model
            model = SVC(
                kernel=kernel,
                C=1.0,
                gamma='scale',
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate
            results = evaluate_model(model, X_test_scaled, y_test)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": results['accuracy'],
                "precision": results['precision'],
                "recall": results['recall'],
                "f1_score": results['f1_score']
            })

            # Log confusion matrix to temp directory
            cm_file = log_confusion_matrix(
                y_test, results['predictions'],
                tmpdir,
                f"cm_svm_{kernel}.png"
            )

            # Create signature
            signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))

            # Log model with signature and input example
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=X_train_scaled[:5]
            )

            # Log all artifacts from temp directory
            mlflow.log_artifacts(tmpdir, artifact_path="plots")

            # Add tags
            mlflow.set_tag("model_family", "svm")
            mlflow.set_tag("hyperparameter_tuning", "kernel_type")

            print(f"Logged SVM ({kernel} kernel): Accuracy = {results['accuracy']:.4f}")

