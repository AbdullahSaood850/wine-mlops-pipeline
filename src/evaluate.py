"""Inference verification module: loads champion model and evaluates on test split."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

import mlflow
import mlflow.sklearn

from src.data import load_and_split_data

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "WineClassifier"


def evaluate_champion_model():
    """Load champion model from MLflow registry and compute test split metrics.

    Returns:
        dict: Evaluation metrics on the held-out test split.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

    print(f"Loading champion model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    _, X_test, _, y_test = load_and_split_data(random_state=42)

    # Compute predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    loss = log_loss(y_test, y_prob, labels=[0, 1, 2])
    conf_mat = confusion_matrix(y_test, y_pred)

    print("\n--- Champion Model Test Evaluation ---")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1-Score: {macro_f1:.4f}")
    print(f"Test Log Loss: {loss:.4f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return {
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_log_loss": float(loss),
        "confusion_matrix": conf_mat.tolist(),
    }


if __name__ == "__main__":
    evaluate_champion_model()
