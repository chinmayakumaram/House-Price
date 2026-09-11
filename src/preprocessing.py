import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "house_prices.csv"


def load_data():
    """Load the house price dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):
    """Clean and preprocess the dataset."""

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing numerical values
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # Handle missing categorical values
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # Convert categorical variables to numerical variables
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    return df


def split_data(df):
    """Separate features and target and split into train/test sets."""

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Scale numerical features."""

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_data()

    print("\nOriginal Dataset:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    print("\nPreprocessing completed successfully!")

    print("Training data shape:", X_train_scaled.shape)
    print("Testing data shape:", X_test_scaled.shape)