"""
Tests for Stage 2: Preprocessing Pipeline

These tests confirm prepare_features_and_target() correctly separates
features/target, and that build_preprocessor() produces the expected
output shape (40 columns, as we verified manually earlier).
"""

from src.data_validation import load_and_clean, validate, RAW_DATA_PATH
from src.preprocessing import build_preprocessor, prepare_features_and_target


def _load_validated_df():
    df = load_and_clean(RAW_DATA_PATH)
    return validate(df)


def test_prepare_features_and_target():
    df = _load_validated_df()
    X, y = prepare_features_and_target(df)

    # customerID and Churn should not be in the feature set
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns

    # target should be binary (0/1)
    assert set(y.unique()).issubset({0, 1})

    # row counts should match
    assert len(X) == len(y) == len(df)


def test_preprocessor_output_shape():
    df = _load_validated_df()
    X, y = prepare_features_and_target(df)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Same number of rows in, rows out
    assert X_transformed.shape[0] == len(X)

    # 40 columns confirmed manually during Stage 2
    assert X_transformed.shape[1] == 40


def test_preprocessor_handles_unknown_category():
    """
    A category never seen during fit() should not crash transform()
    thanks to handle_unknown="ignore" on the OneHotEncoder.
    """
    df = _load_validated_df()
    X, y = prepare_features_and_target(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    # Introduce a never-seen category in a categorical column
    X_new = X.iloc[[0]].copy()
    X_new["PaymentMethod"] = "Cryptocurrency"

    # Should not raise - unknown category encoded as all-zeros
    X_transformed = preprocessor.transform(X_new)
    assert X_transformed.shape[1] == 40