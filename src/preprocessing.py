"""
Stage 2: Preprocessing Pipeline
--------------------------------
Builds a reusable ColumnTransformer that encodes categorical features and
scales numeric features. This same object is fit during training and reused
(via joblib) at inference time in the FastAPI service - guaranteeing the
exact same transform logic in both places (no train/serve skew).

SMOTE is handled separately in train.py, applied ONLY to the training split
AFTER train_test_split, to avoid data leakage into the test set.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Column groups (based on EDA findings)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# SeniorCitizen is already 0/1 - passed through unchanged, no encoding needed
PASSTHROUGH_FEATURES = ["SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Columns to drop entirely before modeling
DROP_FEATURES = ["customerID"]

TARGET_COLUMN = "Churn"


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that:
      - Scales numeric features (StandardScaler)
      - One-hot encodes categorical features
      - Passes SeniorCitizen through unchanged

    handle_unknown="ignore" on the OneHotEncoder means that if a future
    category appears at inference time that wasn't seen during training
    (e.g. a new PaymentMethod), it's encoded as all-zeros rather than
    crashing the pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_FEATURES,
            ),
            ("passthrough", "passthrough", PASSTHROUGH_FEATURES),
        ],
        remainder="drop",  # anything not listed above (e.g. customerID) is dropped
    )
    return preprocessor


def prepare_features_and_target(df: pd.DataFrame):
    """
    Splits a validated dataframe into:
      - X: feature dataframe (customerID and Churn excluded)
      - y: binary target (1 = Churn/Yes, 0 = No)
    """
    df = df.drop(columns=DROP_FEATURES)

    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    return X, y


if __name__ == "__main__":
    # Quick smoke test: load validated data, build preprocessor, fit-transform
    from data_validation import load_and_clean, validate

    df = load_and_clean("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = validate(df)

    X, y = prepare_features_and_target(df)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Target balance:\n{y.value_counts(normalize=True)}")

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    print(f"Transformed feature matrix shape: {X_transformed.shape}")

    # Show the generated feature names (useful for debugging one-hot columns)
    feature_names = preprocessor.get_feature_names_out()
    print(f"Number of output features: {len(feature_names)}")
    print(f"Sample feature names: {list(feature_names[:10])}")