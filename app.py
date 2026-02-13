import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
# MLflow’s scikit-learn flavor module
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# to create dataset signature
from mlflow.models import infer_signature
# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

mlflow.set_tracking_uri("file:./mlruns")
# Set experiment name
mlflow.set_experiment("iris-classification")

# Start an MLflow run
with mlflow.start_run(run_name="random-forest-baseline"):

    # Define hyperparameters
    n_estimators = 100
    max_depth = 5

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForest")

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # create signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(model, name="model", 
                             signature=signature, 
                             input_example=X_train[:5])

    print(f"Run complete!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")


# Example: Logging multiple params/metrics at once

with mlflow.start_run(run_name="efficient-logging-example"):

    # Log multiple parameters as a dictionary
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
        "criterion": "gini"
    }
    mlflow.log_params(params)

    # Train model (abbreviated for demonstration)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log multiple metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "f1_macro": f1_score(y_test, y_pred, average='macro')
    }
    mlflow.log_metrics(metrics)

    print("Logged all parameters and metrics!")

with mlflow.start_run(run_name="with-confusion-matrix"):

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # Create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save and log the plot
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # create signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(model, name="model", 
                             signature=signature, 
                             input_example=X_train[:5])

    print("Logged model and confusion matrix plot!")


import os
import tempfile

with mlflow.start_run(run_name="temp-artifacts-example"):

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:

        # Train model (abbreviated)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save plot to temp directory
        plot_path = os.path.join(tmpdir, "feature_importance.png")

        # Create feature importance plot
        feature_importance = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.title('Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.savefig(plot_path)
        plt.close()

        # Log entire directory of artifacts
        mlflow.log_artifacts(tmpdir, artifact_path="plots")

    print("Logged artifacts from temporary directory!")

# Create a new experiment programmatically
experiment_name = "iris-classification-advanced"

# Get or create experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
else:
    experiment_id = experiment.experiment_id
    print(f"ℹUsing existing experiment: {experiment_name} (ID: {experiment_id})")

# Set as active experiment
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="tagged-run-example"):

    # Set tags
    mlflow.set_tag("model_family", "tree-based")
    mlflow.set_tag("purpose", "baseline")
    mlflow.set_tag("developer", "data-science-team")
    mlflow.set_tag("mlflow.note.content", "Initial baseline model for comparison")

    # Train and log (abbreviated)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))

    print("Run logged with tags!")


# --- Section for Comparing Runs ---
mlflow.set_experiment("iris-classification-comparison")

# Define a grid of hyperparameters to compare
param_grid = [
    {"n_estimators": 10, "max_depth": 2},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None}
]

for params in param_grid:
    with mlflow.start_run(run_name=f"rf-depth-{params['max_depth']}"):
        # Train
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Log
        mlflow.log_params(params)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        
        print(f"Logged run with depth {params['max_depth']} - Accuracy: {acc}")
