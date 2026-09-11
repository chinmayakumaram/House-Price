"""
evaluate.py
-----------
Loads the saved ML and DL models, evaluates both on the held-out
test set, and prints/saves a side-by-side comparison.

Usage:
    python src/evaluate.py --data data/processed --models models
"""

import argparse
import json
import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_processed(data_dir: str):
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_test, y_test


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_ml(models_dir: str, X_test, y_test) -> dict:
    model_path = os.path.join(models_dir, "best_ml_model.joblib")
    if not os.path.exists(model_path):
        print("[evaluate] No ML model found, skipping.")
        return {}
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    return compute_metrics(y_test, preds)


def evaluate_dl(models_dir: str, X_test, y_test) -> dict:
    model_path = os.path.join(models_dir, "dl_model.keras")
    scaler_path = os.path.join(models_dir, "dl_target_scaler.json")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("[evaluate] No DL model found, skipping.")
        return {}

    model = tf.keras.models.load_model(model_path)
    with open(scaler_path) as f:
        scaler = json.load(f)

    preds_scaled = model.predict(X_test).flatten()
    preds = preds_scaled * scaler["y_std"] + scaler["y_mean"]
    return compute_metrics(y_test, preds)


def run(data_dir: str, models_dir: str):
    X_test, y_test = load_processed(data_dir)

    ml_metrics = evaluate_ml(models_dir, X_test, y_test)
    dl_metrics = evaluate_dl(models_dir, X_test, y_test)

    comparison = {"ml_model": ml_metrics, "dl_model": dl_metrics}

    print("\n=== Model Comparison ===")
    for model_name, metrics in comparison.items():
        if not metrics:
            continue
        print(f"{model_name}: RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R2={metrics['r2']:.3f}")

    out_path = os.path.join(models_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[evaluate] Saved comparison to {out_path}")

    return comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare ML vs DL models.")
    parser.add_argument("--data", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--models", type=str, default="models", help="Directory containing saved models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.models)
