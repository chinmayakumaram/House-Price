"""
preprocessing.py
-----------------
Loads raw housing data, cleans it, engineers features, and saves
train/test splits (plus the fitted preprocessing pipeline) for
downstream ML and DL training scripts.

Usage:
    python src/preprocessing.py --input data/raw/housing.csv --outdir data/processed
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(path)
    print(f"[preprocessing] Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, obviously broken rows, and reset index."""
    before = len(df)
    df = df.drop_duplicates()
    if TARGET in df.columns:
        df = df[df[TARGET] > 0]  # remove non-positive / invalid prices
    df = df.reset_index(drop=True)
    print(f"[preprocessing] Cleaned rows: {before} -> {len(df)}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that tend to help house-price models."""
    df = df.copy()

    if {"total_rooms", "households"}.issubset(df.columns):
        df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(0, np.nan)

    if {"total_bedrooms", "total_rooms"}.issubset(df.columns):
        df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"].replace(0, np.nan)

    if {"population", "households"}.issubset(df.columns):
        df["population_per_household"] = df["population"] / df["households"].replace(0, np.nan)

    if "year_built" in df.columns:
        current_year = pd.Timestamp.now().year
        df["house_age"] = current_year - df["year_built"]

    return df


def get_feature_lists(df: pd.DataFrame, target: str = TARGET):
    """Split remaining columns into numeric and categorical feature lists."""
    features = [c for c in df.columns if c != target]
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in features if c not in numeric_features]
    return numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes/scales numerics and one-hot encodes categoricals."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor


def run(input_path: str, outdir: str, test_size: float = 0.2):
    os.makedirs(outdir, exist_ok=True)

    df = load_data(input_path)
    df = basic_cleaning(df)
    df = engineer_features(df)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset. Rename your target column or edit TARGET in preprocessing.py")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features, categorical_features = get_feature_lists(df)
    preprocessor = build_pipeline(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Persist processed arrays + pipeline + raw splits (for reference / DL loaders)
    joblib.dump(preprocessor, os.path.join(outdir, "preprocessor.joblib"))
    np.save(os.path.join(outdir, "X_train.npy"), X_train_proc.toarray() if hasattr(X_train_proc, "toarray") else X_train_proc)
    np.save(os.path.join(outdir, "X_test.npy"), X_test_proc.toarray() if hasattr(X_test_proc, "toarray") else X_test_proc)
    np.save(os.path.join(outdir, "y_train.npy"), y_train.values)
    np.save(os.path.join(outdir, "y_test.npy"), y_test.values)

    X_train.to_csv(os.path.join(outdir, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(outdir, "X_test_raw.csv"), index=False)

    print(f"[preprocessing] Saved processed arrays and pipeline to {outdir}")
    print(f"[preprocessing] Train shape: {X_train_proc.shape}, Test shape: {X_test_proc.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw housing data.")
    parser.add_argument("--input", type=str, default="data/raw/housing.csv", help="Path to raw CSV")
    parser.add_argument("--outdir", type=str, default="data/processed", help="Directory to save processed data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.outdir, args.test_size)
