"""Data ingestion, validation, and splitting module for Wine Cultivar Classification."""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def validate_data(df, feature_names):
    """Validate data integrity: check for null values and verify feature count equals 13.

    Args:
        df: Pandas DataFrame containing feature columns.
        feature_names: List of expected feature names.

    Raises:
        ValueError: If null values are found or feature count is not 13.
    """
    if len(feature_names) != 13:
        raise ValueError(f"Expected exactly 13 features, but found {len(feature_names)}.")

    if df.shape[1] != 13:
        raise ValueError(f"Expected DataFrame with 13 columns, but got {df.shape[1]}.")

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Data validation failed: found {null_count} missing values.")


def load_and_split_data(test_size=0.20, random_state=42):
    """Load Wine dataset, validate structure, and perform stratified train-test split.

    Args:
        test_size: Proportion of dataset to include in test split (default: 0.20).
        random_state: Fixed random seed for reproducibility (default: 42).

    Returns:
        X_train, X_test, y_train, y_test: Feature DataFrames and target Series.
    """
    wine = load_wine(as_frame=True)
    df_features = wine.data
    target = wine.target

    # Run validation checks
    validate_data(df_features, wine.feature_names)

    # Perform stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        df_features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    print("Data loaded and validated successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
