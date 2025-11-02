"""
train.py — Iris Classification with MLflow Tracking and Model Registry
-----------------------------------------------------------------------
Author  : Nitish Kumar
Project : Iris Classification MLflow Integration
Objective:
    - Train and track a Logistic Regression model using MLflow.
    - Log parameters, metrics, and artifacts to remote MLflow server.
    - Save artifacts in a GCS bucket.
    - Register the best model in MLflow Model Registry.
"""

import os
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ================================================================
# 🔧 1. Configuration
# ================================================================
MLFLOW_TRACKING_URI = "http://34.71.109.164:8100"  # ✅ MLflow server
EXPERIMENT_NAME = "iris-mlflow-experiment"
MODEL_NAME = "iris-logreg-model"
ARTIFACT_LOCATION = "gs://mlflow-artifacts-nitish"

# Set up MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)


# ================================================================
# 🧠 2. Data Loading
# ================================================================
def load_data():
    """Load and split the Iris dataset."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n📥 Dataset Loaded Successfully")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Test samples: {len(X_test)}")
    print(f"  • Features: {list(X.columns)}")
    print(f"  • Classes: {iris.target_names.tolist()}")

    return X_train, X_test, y_train, y_test


# ================================================================
# ⚙️ 3. Model Training
# ================================================================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with GridSearchCV and evaluate performance."""
    param_grid = {
        "C": [0.01, 0.1, 1.0],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [100, 300],
    }

    print("\n⚙️  Running GridSearchCV...")
    grid = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=0,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Best Params: {grid.best_params_}")
    print(f"✅ Test Accuracy: {accuracy:.4f}")

    return best_model, grid.best_params_, accuracy


# ================================================================
# 🧾 4. MLflow Logging Function
# ================================================================
def log_to_mlflow(model, best_params, accuracy):
    """Log model, metrics, and register the model in MLflow."""

    # ✅ Generate unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"iris-lr-run-{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)

        # Log model to registry
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
        )

        print("\n✅ Model successfully logged & registered.")
        print(f"📦 Run Name: {run_name}")
        print(f"🧪 Experiment URL: {MLFLOW_TRACKING_URI}/#/experiments/{run.info.experiment_id}")
        print(f"🏃 Run URL: {MLFLOW_TRACKING_URI}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        print(f"🎯 Accuracy: {accuracy:.4f}")


# ================================================================
# 🚀 5. Pipeline Orchestration
# ================================================================
def train_model():
    """Complete MLflow pipeline execution."""
    print("=" * 70)
    print("🌸 IRIS CLASSIFICATION TRAINING PIPELINE")
    print("=" * 70)

    # Environment variables for MLflow
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME
    os.environ["MLFLOW_ARTIFACT_URI"] = ARTIFACT_LOCATION

    # Load, train, and log
    X_train, X_test, y_train, y_test = load_data()
    model, best_params, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)
    log_to_mlflow(model, best_params, accuracy)


# ================================================================
# 🏁 Entry Point
# ================================================================
if __name__ == "__main__":
    train_model()
