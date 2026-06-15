"""
Stage 8: Monitoring & Drift Detection
-----------------------------------------
Uses Evidently AI to compare "current" (production) data against the
"reference" (training) data distribution.

We generate TWO reports to illustrate both outcomes:
  1. no_drift_report.html  - current = random sample from the SAME data
                              (sanity check: should show little/no drift)
  2. drift_report.html     - current = sample with deliberately shifted
                              values (simulating a real distribution change)

In production, "current" would be real incoming requests logged over time
(e.g., the last 24h of /predict inputs), and this would run on a schedule.

Run:
    python src/monitoring.py
"""

import os

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from data_validation import load_and_clean, validate, RAW_DATA_PATH
from preprocessing import prepare_features_and_target

REPORTS_DIR = "reports"


def load_reference_data() -> pd.DataFrame:
    """The 'reference' distribution: features from the full validated dataset
    (this is effectively what the model was trained on)."""
    df = load_and_clean(RAW_DATA_PATH)
    df = validate(df)
    X, _ = prepare_features_and_target(df)
    return X


def make_no_drift_sample(reference: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """A random sample from the SAME distribution - simulates a normal
    day of production traffic. Expect: no significant drift."""
    return reference.sample(n=n, random_state=1).reset_index(drop=True)


def make_drifted_sample(reference: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """
    A sample with deliberately shifted values - simulates a real-world
    change, e.g.:
      - A marketing push toward fiber-optic plans shifts InternetService
        distribution.
      - A price increase shifts MonthlyCharges upward.
      - A new "new customer" acquisition wave shifts tenure downward.
    """
    sample = reference.sample(n=n, random_state=2).reset_index(drop=True)

    # Simulate a price increase: bump MonthlyCharges up by ~30%
    sample["MonthlyCharges"] = sample["MonthlyCharges"] * 1.3

    # Simulate a new-customer acquisition wave: most tenures pulled toward 0
    sample["tenure"] = (sample["tenure"] * 0.2).round().astype(int)

    # Simulate a sales push: force most customers onto Fiber optic
    sample["InternetService"] = "Fiber optic"

    return sample


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: str):
    report = Report([DataDriftPreset()])
    result = report.run(current, reference)
    result.save_html(output_path)
    print(f"Saved report to {output_path}")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    reference = load_reference_data()

    no_drift_current = make_no_drift_sample(reference)
    drifted_current = make_drifted_sample(reference)

    run_drift_report(
        reference, no_drift_current,
        os.path.join(REPORTS_DIR, "no_drift_report.html"),
    )
    run_drift_report(
        reference, drifted_current,
        os.path.join(REPORTS_DIR, "drift_report.html"),
    )


if __name__ == "__main__":
    main()