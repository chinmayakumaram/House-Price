"""
app.py
------
Streamlit app that serves house price predictions using either the
best classical ML model or the trained DL model.

Run:
    streamlit run app/app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


@st.cache_resource
def load_artifacts():
    preprocessor_path = os.path.join(DATA_DIR, "preprocessor.joblib")
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None

    ml_model_path = os.path.join(MODELS_DIR, "best_ml_model.joblib")
    ml_model = joblib.load(ml_model_path) if os.path.exists(ml_model_path) else None

    dl_model_path = os.path.join(MODELS_DIR, "dl_model.keras")
    dl_model = tf.keras.models.load_model(dl_model_path) if os.path.exists(dl_model_path) else None

    scaler_path = os.path.join(MODELS_DIR, "dl_target_scaler.json")
    dl_scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            dl_scaler = json.load(f)

    # Reference raw columns so the UI can build a matching input row
    raw_cols_path = os.path.join(DATA_DIR, "X_train_raw.csv")
    raw_columns = pd.read_csv(raw_cols_path, nrows=1).columns.tolist() if os.path.exists(raw_cols_path) else []

    return preprocessor, ml_model, dl_model, dl_scaler, raw_columns


def predict(model_choice, input_df, preprocessor, ml_model, dl_model, dl_scaler):
    X = preprocessor.transform(input_df)
    X = X.toarray() if hasattr(X, "toarray") else X

    if model_choice == "Machine Learning (best model)":
        pred = ml_model.predict(X)[0]
    else:
        pred_scaled = dl_model.predict(X).flatten()[0]
        pred = pred_scaled * dl_scaler["y_std"] + dl_scaler["y_mean"]
    return float(pred)


def main():
    st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
    st.title("🏠 House Price Predictor")
    st.write("Enter property details to get a predicted price from an ML or DL model.")

    preprocessor, ml_model, dl_model, dl_scaler, raw_columns = load_artifacts()

    if preprocessor is None or not raw_columns:
        st.error("No trained pipeline found. Run `src/preprocessing.py`, `src/train_ml.py`, "
                  "and `src/train_dl.py` first to generate the required artifacts.")
        return

    available_models = []
    if ml_model is not None:
        available_models.append("Machine Learning (best model)")
    if dl_model is not None:
        available_models.append("Deep Learning")

    if not available_models:
        st.error("No trained models found in the models/ directory.")
        return

    model_choice = st.selectbox("Choose a model", available_models)

    st.subheader("Property details")
    user_input = {}
    cols = st.columns(2)
    for i, col_name in enumerate(raw_columns):
        with cols[i % 2]:
            user_input[col_name] = st.number_input(col_name, value=0.0, format="%.4f")

    if st.button("Predict price", type="primary"):
        input_df = pd.DataFrame([user_input])
        try:
            price = predict(model_choice, input_df, preprocessor, ml_model, dl_model, dl_scaler)
            st.success(f"Estimated price: **${price:,.2f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
