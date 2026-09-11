# House Price Regression: ML & DL

End-to-end house price prediction project comparing classical machine
learning models against a deep learning (Keras/TensorFlow) regressor,
served through a Streamlit app.

## Project structure

```
house-price-regression-ml-dl/
├── data/
│   ├── raw/            # Original, untouched CSV(s)
│   └── processed/       # Output of preprocessing.py (arrays + fitted pipeline)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_ml_regression.ipynb  # ML experimentation
│   └── 03_deep_learning.ipynb  # DL experimentation
├── src/
│   ├── preprocessing.py  # Cleaning, feature engineering, train/test split
│   ├── train_ml.py       # Trains & compares classical ML models
│   ├── train_dl.py       # Trains a Keras MLP regressor
│   └── evaluate.py       # Compares ML vs DL on the test set
├── models/                # Saved models & metrics (generated)
├── app/
│   └── app.py             # Streamlit prediction app
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place your raw CSV at `data/raw/housing.csv`. The pipeline expects a
`price` column as the regression target — edit `TARGET` in
`src/preprocessing.py` if your column is named differently. Any other
numeric/categorical columns are used as features automatically.

## Pipeline

Run each stage in order:

```bash
# 1. Clean, engineer features, split, and save processed arrays
python src/preprocessing.py --input data/raw/housing.csv --outdir data/processed

# 2. Train & compare classical ML models (Linear/Ridge/Lasso/RF/GBM/SVR)
python src/train_ml.py --data data/processed --outdir models

# 3. Train the deep learning model
python src/train_dl.py --data data/processed --outdir models --epochs 100

# 4. Compare ML vs DL on the held-out test set
python src/evaluate.py --data data/processed --models models
```

## Serving predictions

```bash
streamlit run app/app.py
```

Select either the best ML model or the DL model and enter property
details to get a live price prediction.

## Docker

```bash
docker build -t house-price-app .
docker run -p 8501:8501 house-price-app
```

## Tests

```bash
pytest tests/
```

## Notes

- `train_ml.py` picks the best model by test RMSE and saves it as
  `models/best_ml_model.joblib`, alongside all trained candidates and
  a `ml_results.json` metrics summary.
- `train_dl.py` scales the target internally and stores the scaling
  factors in `models/dl_target_scaler.json` so predictions can be
  converted back to real price units.
- Swap in your own dataset by changing column names referenced in
  `engineer_features()` inside `src/preprocessing.py`.
