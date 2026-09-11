"""
test_preprocessing.py
----------------------
Basic unit tests for the preprocessing pipeline.

Run:
    pytest tests/
"""

import pandas as pd
import pytest

from src.preprocessing import basic_cleaning, engineer_features, get_feature_lists


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "price": [100000, 200000, -5000, 100000],
        "total_rooms": [10, 20, 15, 10],
        "total_bedrooms": [2, 4, 3, 2],
        "households": [5, 10, 0, 5],
        "population": [20, 40, 30, 20],
        "location": ["A", "B", "A", "A"],
    })


def test_basic_cleaning_removes_invalid_and_duplicate_rows(sample_df):
    cleaned = basic_cleaning(sample_df)
    assert (cleaned["price"] > 0).all()
    assert len(cleaned) <= len(sample_df)


def test_engineer_features_adds_derived_columns(sample_df):
    cleaned = basic_cleaning(sample_df)
    engineered = engineer_features(cleaned)
    assert "rooms_per_household" in engineered.columns
    assert "bedrooms_per_room" in engineered.columns
    assert "population_per_household" in engineered.columns


def test_get_feature_lists_splits_numeric_and_categorical(sample_df):
    numeric_features, categorical_features = get_feature_lists(sample_df, target="price")
    assert "location" in categorical_features
    assert "total_rooms" in numeric_features
    assert "price" not in numeric_features
    assert "price" not in categorical_features
