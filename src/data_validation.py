"""
Stage 1: Data Validation
--------------------------------
Defines a schema for the raw Telco Customer Churn dataset and validates
incoming data against it. This is the "gatekeeper" for the pipeline -
nothing proceeds to preprocessing/training unless it passes here.

Run directly:
    python src/data_validation.py
"""

import sys
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------
# Each Column entry says: "this column must have this dtype, and (optionally)
# its values must satisfy these checks". If any row violates these, Pandera
# raises a SchemaError with details on exactly which rows/columns failed.

raw_schema = DataFrameSchema(
    {
        "customerID": Column(str, unique=True, nullable=False),

        "gender": Column(str, Check.isin(["Male", "Female"])),

        "SeniorCitizen": Column(int, Check.isin([0, 1])),

        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),

        "tenure": Column(int, Check.in_range(0, 100)),

        "PhoneService": Column(str, Check.isin(["Yes", "No"])),

        "MultipleLines": Column(
            str, Check.isin(["Yes", "No", "No phone service"])
        ),

        "InternetService": Column(
            str, Check.isin(["DSL", "Fiber optic", "No"])
        ),

        "OnlineSecurity": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "OnlineBackup": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "DeviceProtection": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "TechSupport": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingTV": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingMovies": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),

        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),

        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),

        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),

        "MonthlyCharges": Column(float, Check.greater_than_or_equal_to(0)),

        # TotalCharges arrives as object/string because of blank values for
        # tenure == 0 customers - we clean it BEFORE validating, see below.
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0)),

        "Churn": Column(str, Check.isin(["Yes", "No"])),
    },
    strict=False,   # allow extra columns without failing (useful while iterating)
    coerce=True,    # attempt to cast columns to the declared dtype
)


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw CSV and fix known dirty-data issues before validation."""
    df = pd.read_csv(path)

    # Known issue: TotalCharges has blank strings for tenure == 0 customers.
    # Convert to numeric, coercing blanks to NaN, then drop or impute.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # For tenure == 0, TotalCharges should logically be 0 (no charges yet).
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[
        df["tenure"] == 0, "TotalCharges"
    ].fillna(0)

    # Any remaining NaNs are genuinely unexpected - drop and report.
    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    n_after = len(df)
    if n_before != n_after:
        print(f"[data_validation] Dropped {n_before - n_after} rows with "
              f"unresolvable TotalCharges values.")

    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate df against raw_schema. Raises pa.errors.SchemaErrors on failure."""
    try:
        validated_df = raw_schema.validate(df, lazy=True)
        print(f"[data_validation] PASSED - {len(validated_df)} rows, "
              f"{len(validated_df.columns)} columns validated successfully.")
        return validated_df
    except pa.errors.SchemaErrors as err:
        print("[data_validation] FAILED - schema violations found:")
        print(err.failure_cases)
        raise


if __name__ == "__main__":
    df = load_and_clean(RAW_DATA_PATH)
    validate(df)