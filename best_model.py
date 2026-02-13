from mlflow.tracking import MlflowClient

client = MlflowClient()
# Find run with highest accuracy
experiment = client.get_experiment_by_name("iris-classification")

if experiment:
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )[0]

    print("🏆 Best performing model:")
    print(f"Run name: {best_run.data.tags.get('mlflow.runName')}")
    print(f"Accuracy: {best_run.data.metrics['accuracy']:.4f}")
    print(f"Parameters: {best_run.data.params}")
    print(f"\nRun ID: {best_run.info.run_id}")
