"""Automated MLOps Quality Gate tests.

Enforces:
1. Metric Threshold Gate: Validation / Evaluation Macro F1-score >= 0.88.
2. Inference Latency Gate: Batch inference time <= 30 ms.
3. Output Schema Integrity: Class indices only (0, 1, or 2).
"""

import time
import pytest
from sklearn.metrics import f1_score

import mlflow
import mlflow.sklearn

from src.data import load_and_split_data
from src.train import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    TRACKING_URI,
    train_and_track_all,
)


@pytest.fixture(scope="session")
def trained_champion_model():
    """Retrieve champion model from MLflow registry or train if not present."""
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # If registry is unpopulated (e.g. fresh environment), run training pipeline
        train_and_track_all()
        model = mlflow.sklearn.load_model(model_uri)
    return model


def test_metric_threshold_gate(trained_champion_model):
    """Metric Threshold Gate: Asserts that evaluation Macro F1-score is >= 0.88."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    if exp is not None:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.val_macro_f1 DESC"],
            max_results=1
        )
        if len(runs) > 0:
            best_val_f1 = runs[0].data.metrics.get("val_macro_f1", 0.0)
            assert best_val_f1 >= 0.88, (
                f"Champion Validation Macro F1 {best_val_f1:.4f} failed quality threshold (0.88)."
            )

    # In addition, assert held-out test split Macro F1-score >= 0.88
    _, X_test, _, y_test = load_and_split_data(random_state=42)
    y_pred = trained_champion_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    assert test_f1 >= 0.88, (
        f"Champion Test Macro F1 {test_f1:.4f} failed quality threshold (0.88)."
    )


def test_inference_latency_gate(trained_champion_model):
    """Inference Latency Gate: Asserts batch inference time is <= 30 ms."""
    _, X_test, _, _ = load_and_split_data(random_state=42)

    # Warm-up run
    _ = trained_champion_model.predict(X_test)

    # Measure latency over batch
    start_time = time.perf_counter()
    _ = trained_champion_model.predict(X_test)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000.0
    print(f"Batch inference latency for {len(X_test)} samples: {latency_ms:.2f} ms")
    assert latency_ms <= 30.0, (
        f"Inference latency {latency_ms:.2f} ms exceeded maximum SLA gate of 30.0 ms."
    )


def test_output_schema_integrity(trained_champion_model):
    """Output Schema Integrity: Asserts predicted classes are exclusively {0, 1, 2}."""
    _, X_test, _, _ = load_and_split_data(random_state=42)
    predictions = trained_champion_model.predict(X_test)

    assert len(predictions) == len(X_test)
    valid_classes = {0, 1, 2}
    unique_preds = set(predictions)
    assert unique_preds.issubset(valid_classes), (
        f"Invalid class indices detected: {unique_preds - valid_classes}. "
        f"Only classes in {valid_classes} are permitted."
    )
