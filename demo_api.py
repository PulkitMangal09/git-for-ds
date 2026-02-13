from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment by name
experiment = client.get_experiment_by_name("iris-classification")

if experiment:
    # Search for runs in this experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],  # Sort by accuracy
        max_results=5
    )

    print("Top 5 runs by accuracy:\n")
    for run in runs:
        print(f"Run ID: {run.info.run_id}")
        print(f"  Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
        print(f"  Accuracy: {run.data.metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Parameters: {run.data.params}")
        print()
