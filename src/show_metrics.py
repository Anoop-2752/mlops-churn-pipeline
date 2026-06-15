"""
Utility: Print all runs in the churn-prediction experiment with their metrics.

Run:
    python src/show_metrics.py
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

EXPERIMENT_NAME = "churn-prediction"


def main():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    rows = []
    for run in runs:
        row = {"run_id": run.info.run_id[:8], "model_type": run.data.params.get("model_type")}
        row.update(run.data.metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["model_type", "accuracy", "precision", "recall", "f1", "roc_auc", "run_id"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("roc_auc", ascending=False)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()