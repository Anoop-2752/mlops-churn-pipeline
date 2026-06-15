"""
Stage 3: Training + Experiment Tracking
-----------------------------------------
Trains multiple candidate models on the churn dataset, applying SMOTE to
handle class imbalance (training fold only), and logs every run to MLflow
for comparison.

The full pipeline saved per run is:
    raw features -> ColumnTransformer (preprocessing) -> SMOTE -> classifier

Because SMOTE only activates during .fit() (imblearn handles this), the
SAME pipeline object can be used directly for inference with no special
handling - it just predicts on the preprocessed features without resampling.

Run:
    python src/train.py
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from data_validation import load_and_clean, validate
from preprocessing import build_preprocessor, prepare_features_and_target

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42
EXPERIMENT_NAME = "churn-prediction"


def load_data():
    df = load_and_clean(RAW_DATA_PATH)
    df = validate(df)
    X, y = prepare_features_and_target(df)
    return X, y


def get_candidate_models():
    """Returns a dict of model_name -> unfitted estimator to try."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    models = get_candidate_models()

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):

            # Full pipeline: preprocessing -> SMOTE -> classifier.
            # SMOTE is a no-op at predict time (imblearn handles this).
            pipeline = ImbPipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("classifier", model),
                ]
            )

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, y_pred, y_proba)

            # --- MLflow logging ---
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("smote_random_state", RANDOM_STATE)
            mlflow.log_param("test_size", 0.2)

            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log the entire pipeline (preprocessing + SMOTE + model) as
            # one artifact - this is what gets loaded for inference.
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"\n[{model_name}] metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()