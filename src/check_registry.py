"""
Utility: Check registered model versions and their current stages.

Run:
    python src/check_registry.py
"""

from mlflow.tracking import MlflowClient

MODEL_NAME = "churn-classifier"


def main():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    for v in versions:
        print(f"Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id[:8]}")


if __name__ == "__main__":
    main()