# Wine Cultivar Classification MLOps Pipeline

[![CI Pipeline](https://github.com/Abdullah-Khan-Niazi/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdullah-Khan-Niazi/wine-mlops-pipeline/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end reproducible Machine Learning Operations (MLOps) continuous integration pipeline for multi-class chemical cultivar classification using `sklearn.datasets.load_wine` (178 samples, 13 features, 3 target classes).

---

## 📌 Repository Architecture

```text
wine-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model_gate.py
├── .gitignore
├── Makefile
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Makefile Automation

The project provides uniform entry points via GNU `make` for local development and CI parity:

```bash
# 1. Upgrade pip and install all dependencies
make install

# 2. Run code style linting (flake8 with max line length 100)
make lint

# 3. Run all unit tests and MLOps Quality Gates (pytest)
make test

# 4. Execute MLflow training, 5-fold CV experiments, and Champion registration
make train

# 5. Clean compiled bytecode, cache directories, and temporary files
make clean
```

---

## 🔬 Model Training & Experiment Tracking

The pipeline implements two distinct tree-based classifier algorithms:
1. **Model Family A**: `RandomForestClassifier`
2. **Model Family B**: `GradientBoostingClassifier`

Each algorithm is tuned across a grid of at least 3 distinct hyperparameter configurations evaluated via 5-fold stratified cross-validation on the 80% training split.

All runs are programmatically tracked in **MLflow** (`sqlite:///mlflow.db`) under the experiment `Wine-Cultivar-Classification`:
- **Parameters**: `n_estimators`, `max_depth`, `learning_rate`, `model_family`.
- **Metrics**: Train & Validation Accuracy, Macro F1-score, Log Loss.
- **Artifacts**: Input/output signatures via `infer_signature()`, training input examples, serialized model artifacts.
- **Model Registry**: The winning run with the highest validation Macro F1-score is promoted into the MLflow Model Registry as `WineClassifier` and tagged with the alias `champion`.

---

## 🛡️ Automated MLOps Quality Gates

Enforced in `tests/test_model_gate.py` on every commit and pull request:
1. **Metric Threshold Gate**: Validation & Test Macro F1-score $\ge 0.88$.
2. **Inference Latency Gate**: Batch inference latency $\le 30.0\text{ ms}$.
3. **Output Schema Integrity**: Class predictions must strictly belong to $\{0, 1, 2\}$.

---

## 🔄 Git Branching & Conflict Resolution

Development follows a feature-branch workflow:
- `main`: Production-ready branch.
- `feature/mlflow-tracking`: Pipeline implementation and tracking.
- `conflict-simulation`: Demonstrates engineered merge conflict handling and resolution.
