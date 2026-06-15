"""
Stage 3b: Model Registry
---------------------------
Finds the best run in the churn-prediction experiment (by ROC-AUC) and
registers its model under the name "churn-classifier" in the MLflow
Model Registry, transitioning it to the "Staging" stage.

This decouples "which run produced the best model" from "which model
the serving layer should load" - the FastAPI service will always load
models:/churn-classifier/Production, regardless of which run that came from.

Run:
    python src/register_model.py
"""

import os

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "churn-prediction"
MODEL_NAME = "churn-classifier"
SELECTION_METRIC = "roc_auc"  # metric used to pick the best run


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    # Get all runs in the experiment, sorted by the selection metric (descending)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{SELECTION_METRIC} DESC"],
    )

    if not runs:
        raise ValueError("No runs found in this experiment.")

    best_run = runs[0]
    best_model_type = best_run.data.params.get("model_type")
    best_metric_value = best_run.data.metrics.get(SELECTION_METRIC)

    print(f"Best run: {best_run.info.run_id}")
    print(f"  model_type: {best_model_type}")
    print(f"  {SELECTION_METRIC}: {best_metric_value:.4f}")

    model_uri = f"runs:/{best_run.info.run_id}/model"

    # Register (creates the model if it doesn't exist, or adds a new version)
    registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print(f"\nRegistered as '{MODEL_NAME}' version {registered_model.version}")

    # Transition the new version to "Staging"
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered_model.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print(f"Transitioned version {registered_model.version} to 'Staging'")


if __name__ == "__main__":
    main()