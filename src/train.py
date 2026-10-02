"""Training, cross-validation, MLflow tracking, and model registry promotion."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.data import load_and_split_data

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"


def get_hyperparameter_grids():
    """Define hyperparameter configurations for Random Forest and Gradient Boosting."""
    rf_configs = [
        {"model_family": "RandomForest", "n_estimators": 50, "max_depth": 3, "random_state": 42},
        {"model_family": "RandomForest", "n_estimators": 100, "max_depth": 5, "random_state": 42},
        {"model_family": "RandomForest", "n_estimators": 150, "max_depth": None, "random_state": 42}
    ]

    gbm_configs = [
        {
            "model_family": "GradientBoosting",
            "n_estimators": 50,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": 42,
        },
        {
            "model_family": "GradientBoosting",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
        },
        {
            "model_family": "GradientBoosting",
            "n_estimators": 100,
            "learning_rate": 0.2,
            "max_depth": 5,
            "random_state": 42,
        },
    ]

    return rf_configs + gbm_configs


def build_model(config):
    """Instantiate classifier based on configuration parameters."""
    cfg = dict(config)
    family = cfg.pop("model_family")

    if family == "RandomForest":
        return RandomForestClassifier(**cfg)
    elif family == "GradientBoosting":
        return GradientBoostingClassifier(**cfg)
    else:
        raise ValueError(f"Unknown model family: {family}")


def evaluate_cross_validation(model_class, config, X, y, n_splits=5):
    """Evaluate configuration using 5-fold stratified cross-validation on training data.

    Returns:
        dict: Mean train and validation metrics (accuracy, macro F1, log loss).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_acc_list = []
    val_acc_list = []
    train_f1_list = []
    val_f1_list = []
    train_loss_list = []
    val_loss_list = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = build_model(config)
        fold_model.fit(X_tr, y_tr)

        # Predictions for train fold
        tr_pred = fold_model.predict(X_tr)
        tr_prob = fold_model.predict_proba(X_tr)
        train_acc_list.append(accuracy_score(y_tr, tr_pred))
        train_f1_list.append(f1_score(y_tr, tr_pred, average="macro"))
        train_loss_list.append(log_loss(y_tr, tr_prob, labels=[0, 1, 2]))

        # Predictions for validation fold
        val_pred = fold_model.predict(X_val)
        val_prob = fold_model.predict_proba(X_val)
        val_acc_list.append(accuracy_score(y_val, val_pred))
        val_f1_list.append(f1_score(y_val, val_pred, average="macro"))
        val_loss_list.append(log_loss(y_val, val_prob, labels=[0, 1, 2]))

    metrics = {
        "train_accuracy": float(np.mean(train_acc_list)),
        "val_accuracy": float(np.mean(val_acc_list)),
        "train_macro_f1": float(np.mean(train_f1_list)),
        "val_macro_f1": float(np.mean(val_f1_list)),
        "train_log_loss": float(np.mean(train_loss_list)),
        "val_log_loss": float(np.mean(val_loss_list)),
    }

    return metrics


def train_and_track_all():
    """Run all hyperparameter experiments, log to MLflow, and register the champion model."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    X_train, X_test, y_train, y_test = load_and_split_data(random_state=42)
    configs = get_hyperparameter_grids()

    run_records = []

    print(f"Starting experiments on {len(configs)} configurations...")
    for idx, cfg in enumerate(configs, start=1):
        family = cfg["model_family"]
        run_name = f"{family}_config_{idx}"

        # 5-fold CV evaluation
        cv_metrics = evaluate_cross_validation(build_model, cfg, X_train, y_train)

        # Train final model on entire training split
        final_model = build_model(cfg)
        final_model.fit(X_train, y_train)

        # Infer signature and prepare input example
        signature = infer_signature(X_train, final_model.predict(X_train))
        input_example = X_train.iloc[:5]

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            # Log hyperparameters
            for param_key, param_val in cfg.items():
                mlflow.log_param(param_key, param_val if param_val is not None else "None")

            # Log metrics
            for metric_key, metric_val in cv_metrics.items():
                mlflow.log_metric(metric_key, metric_val)

            # Log tags
            mlflow.set_tag("model_family", family)
            mlflow.set_tag("algorithm", "TreeEnsemble")
            mlflow.set_tag("stage", "HyperparameterTuning")

            # Log model artifact with signature and input example
            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            run_records.append({
                "run_id": run.info.run_id,
                "config": cfg,
                "val_macro_f1": cv_metrics["val_macro_f1"],
                "val_accuracy": cv_metrics["val_accuracy"],
                "val_log_loss": cv_metrics["val_log_loss"],
            })

            print(
                f"[{idx}/{len(configs)}] {run_name} | "
                f"Val F1: {cv_metrics['val_macro_f1']:.4f} | "
                f"Val Acc: {cv_metrics['val_accuracy']:.4f} | "
                f"Val Loss: {cv_metrics['val_log_loss']:.4f}"
            )

    # Identify champion model by highest validation macro F1-score
    best_record = max(run_records, key=lambda r: r["val_macro_f1"])
    best_run_id = best_record["run_id"]
    print(f"\nChampion Run ID: {best_run_id}")
    print(f"Champion Val Macro F1: {best_record['val_macro_f1']:.4f}")
    print(f"Champion Config: {best_record['config']}")

    # Register champion model into MLflow Model Registry
    client = MlflowClient(tracking_uri=TRACKING_URI)
    model_uri = f"runs:/{best_run_id}/model"
    reg_model = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    # Assign 'champion' alias to this version
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=reg_model.version
    )
    print(
        f"Successfully registered model '{REGISTERED_MODEL_NAME}' version {reg_model.version} "
        f"with alias 'champion'."
    )

    return best_record


if __name__ == "__main__":
    train_and_track_all()
