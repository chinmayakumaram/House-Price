"""
train_ml.py
-----------
Trains several classical ML regression models on the preprocessed
data, compares them with cross-validation, and saves the best model.

Usage:
    python src/train_ml.py --data data/processed --outdir models
"""

import argparse
import os
import json

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

RANDOM_STATE = 42


def load_processed(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def get_candidate_models():
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
        "svr": SVR(kernel="rbf", C=10, epsilon=0.1),
    }


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run(data_dir: str, outdir: str, cv_folds: int = 5):
    os.makedirs(outdir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_processed(data_dir)

    candidates = get_candidate_models()
    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        print(f"[train_ml] Training {name} ...")
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics["cv_rmse_mean"] = -cv_scores.mean()
        metrics["cv_rmse_std"] = cv_scores.std()

        results[name] = metrics
        fitted_models[name] = model
        print(f"[train_ml] {name}: test RMSE={metrics['rmse']:.3f}, "
              f"R2={metrics['r2']:.3f}, CV RMSE={metrics['cv_rmse_mean']:.3f}")

    # Pick best model by test RMSE
    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_model = fitted_models[best_name]
    print(f"[train_ml] Best model: {best_name} (RMSE={results[best_name]['rmse']:.3f})")

    joblib.dump(best_model, os.path.join(outdir, "best_ml_model.joblib"))
    joblib.dump(fitted_models, os.path.join(outdir, "all_ml_models.joblib"))

    with open(os.path.join(outdir, "ml_results.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(f"[train_ml] Saved best model + results to {outdir}")
    return results, best_name


def parse_args():
    parser = argparse.ArgumentParser(description="Train classical ML regression models.")
    parser.add_argument("--data", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--outdir", type=str, default="models", help="Directory to save models/results")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.outdir, args.cv_folds)
