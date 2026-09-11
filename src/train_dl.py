"""
train_dl.py
-----------
Trains a feed-forward neural network (Keras/TensorFlow) on the
preprocessed housing data for price regression.

Usage:
    python src/train_dl.py --data data/processed --outdir models --epochs 100
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, models, callbacks

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_processed(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def build_model(input_dim: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    """A simple, regularized MLP regressor."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def run(data_dir: str, outdir: str, epochs: int = 100, batch_size: int = 32, learning_rate: float = 1e-3):
    os.makedirs(outdir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_processed(data_dir)

    # Scale the target for more stable training; keep the scale factor to invert later
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    model = build_model(input_dim=X_train.shape[1], learning_rate=learning_rate)
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train_scaled,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )

    preds_scaled = model.predict(X_test).flatten()
    preds = preds_scaled * y_std + y_mean

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    print(f"[train_dl] Test RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    model_path = os.path.join(outdir, "dl_model.keras")
    model.save(model_path)

    with open(os.path.join(outdir, "dl_target_scaler.json"), "w") as f:
        json.dump({"y_mean": float(y_mean), "y_std": float(y_std)}, f, indent=2)

    with open(os.path.join(outdir, "dl_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train_dl] Saved model to {model_path}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning regressor for house prices.")
    parser.add_argument("--data", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--outdir", type=str, default="models", help="Directory to save the model/results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.outdir, args.epochs, args.batch_size, args.lr)
