"""
Tests for Stage 1: Data Validation

These tests confirm the Pandera schema does its job: passes clean data,
and rejects data that violates the rules we defined (bad categories,
out-of-range numeric values).
"""

import pandera as pa
import pytest

from src.data_validation import load_and_clean, validate, RAW_DATA_PATH


@pytest.fixture(scope="module")
def clean_df():
    """Load and clean the raw dataset once for all tests in this file."""
    return load_and_clean(RAW_DATA_PATH)


def test_valid_data_passes(clean_df):
    validated = validate(clean_df)
    assert len(validated) == len(clean_df)


def test_invalid_contract_category_fails(clean_df):
    df = clean_df.copy()
    df.loc[0, "Contract"] = "Annually"  # not a valid category

    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)


def test_negative_tenure_fails(clean_df):
    df = clean_df.copy()
    df.loc[0, "tenure"] = -1  # below the allowed range (0-72)

    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)


def test_invalid_gender_value_fails(clean_df):
    df = clean_df.copy()
    df.loc[0, "gender"] = "Other"  # not in ["Male", "Female"]

    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)