"""Unit tests for data ingestion, validation, and splitting."""

import pytest
from sklearn.datasets import load_wine

from src.data import load_and_split_data, validate_data


def test_data_validation_clean():
    """Verify validation passes on standard Wine dataset features."""
    wine = load_wine(as_frame=True)
    # Should not raise exception
    validate_data(wine.data, wine.feature_names)


def test_data_validation_wrong_feature_count():
    """Verify validation raises error if feature count is not 13."""
    wine = load_wine(as_frame=True)
    df_truncated = wine.data.iloc[:, :10]
    with pytest.raises(ValueError, match="13"):
        validate_data(df_truncated, list(df_truncated.columns))


def test_data_validation_missing_values():
    """Verify validation raises error when NaN values are present."""
    wine = load_wine(as_frame=True)
    df_corrupted = wine.data.copy()
    df_corrupted.iloc[0, 0] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_data(df_corrupted, wine.feature_names)


def test_train_test_split_shapes():
    """Verify 80/20 train-test split shapes and class coverage."""
    X_train, X_test, y_train, y_test = load_and_split_data(random_state=42)

    total_samples = len(X_train) + len(X_test)
    assert total_samples == 178, f"Expected 178 total samples, got {total_samples}"

    # 80% of 178 is ~142, 20% is ~36
    assert len(X_train) == 142
    assert len(X_test) == 36
    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13

    # Ensure all 3 target classes are present in both splits
    assert set(y_train.unique()) == {0, 1, 2}
    assert set(y_test.unique()) == {0, 1, 2}
