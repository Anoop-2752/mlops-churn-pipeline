"""
Stage 9: Orchestration
-------------------------
Wires together: load reference data -> simulate incoming data -> drift check
-> conditional retraining, as a single Prefect flow.

In production, this flow would run on a schedule (e.g., nightly via a
Prefect deployment). "Incoming data" would be real production requests
logged over the past period, not a simulated sample.

The drift check uses the Kolmogorov-Smirnov (KS) test - the same statistical
test Evidently uses under the hood for numeric drift detection in Stage 8.
A small p-value means the two samples are unlikely to come from the same
distribution (i.e., drift has occurred).

Run:
    python src/pipeline.py              # simulates drift -> triggers retraining
    python src/pipeline.py --no-drift   # no drift -> skips retraining

View the run in the Prefect UI:
    prefect server start
    (then open http://127.0.0.1:4200)
"""

import sys

from prefect import flow, task
from scipy.stats import ks_2samp

from data_validation import load_and_clean, validate, RAW_DATA_PATH
from preprocessing import prepare_features_and_target
from train import main as run_training
from register_model import main as run_registration

DRIFT_PVALUE_THRESHOLD = 0.05
NUMERIC_COLUMNS_TO_CHECK = ["tenure", "MonthlyCharges", "TotalCharges"]


@task
def load_reference_features():
    """Reference distribution: features from the full validated dataset
    (what the currently-deployed model was trained on)."""
    df = load_and_clean(RAW_DATA_PATH)
    df = validate(df)
    X, _ = prepare_features_and_target(df)
    return X


@task
def simulate_incoming_data(reference, simulate_drift: bool):
    """
    Stand-in for 'real production requests from the last period'.
    With simulate_drift=True, shifts MonthlyCharges and tenure to mimic
    a real distribution change (e.g., price increase + acquisition wave).
    """
    sample = reference.sample(n=300, random_state=42).reset_index(drop=True)

    if simulate_drift:
        sample["MonthlyCharges"] = sample["MonthlyCharges"] * 1.3
        sample["tenure"] = (sample["tenure"] * 0.2).round().astype(int)

    return sample


@task
def check_drift(reference, current) -> bool:
    """
    Runs a KS test on each numeric column. Returns True if ANY column's
    p-value is below the threshold (distributions significantly differ).
    """
    drifted_columns = []

    for col in NUMERIC_COLUMNS_TO_CHECK:
        stat, p_value = ks_2samp(reference[col], current[col])
        flag = "DRIFT" if p_value < DRIFT_PVALUE_THRESHOLD else "ok"
        print(f"  {col}: KS p-value={p_value:.5f} [{flag}]")

        if p_value < DRIFT_PVALUE_THRESHOLD:
            drifted_columns.append(col)

    if drifted_columns:
        print(f"Drift detected in: {drifted_columns}")
    else:
        print("No significant drift detected.")

    return len(drifted_columns) > 0


@task
def retrain_and_register():
    print("\n--- Drift detected: retraining and re-registering model ---\n")
    run_training()
    run_registration()


@flow(name="churn-mlops-pipeline")
def ml_pipeline(simulate_drift: bool = True):
    reference = load_reference_features()
    current = simulate_incoming_data(reference, simulate_drift)

    drift_detected = check_drift(reference, current)

    if drift_detected:
        retrain_and_register()
    else:
        print("\n--- No drift: skipping retraining ---\n")


if __name__ == "__main__":
    simulate = "--no-drift" not in sys.argv
    ml_pipeline(simulate_drift=simulate)