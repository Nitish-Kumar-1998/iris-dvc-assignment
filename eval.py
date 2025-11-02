"""
eval.py — Iris Model Evaluation with MLflow Model Registry
----------------------------------------------------------
Author  : Nitish Kumar
Project : Iris Classification MLflow Integration
Objective:
    - Load latest or Production model from MLflow Model Registry.
    - Evaluate model performance on test data.
    - Log metrics and confusion matrix to MLflow.
    - Store artifacts in GCS bucket.
"""

import os
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

# ================================================================
# 🔧 1. MLflow Configuration (Fixed for GCS Artifacts)
# ================================================================
MLFLOW_TRACKING_URI = "http://34.71.109.164:8100"  # VM MLflow server
ARTIFACT_LOCATION = "gs://mlflow-artifacts-nitish"  # ✅ your GCS bucket
EXPERIMENT_NAME = "iris-eval"
MODEL_NAME = "iris-logreg-model"

# Set environment for MLflow + GCS storage
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_ARTIFACT_URI"] = ARTIFACT_LOCATION
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

# Ensure eval experiment exists with GCS as artifact location
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    client.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=ARTIFACT_LOCATION
    )
mlflow.set_experiment(EXPERIMENT_NAME)


# ================================================================
# 🧠 2. Load Data
# ================================================================
def load_data():
    """Load the full Iris dataset (same split logic as training)."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y


# ================================================================
# 🧾 3. Model Loading
# ================================================================
def load_model_from_registry():
    """Load model from MLflow Model Registry."""
    print("\n======================================================================")
    print("🔍 IRIS MODEL EVALUATION PIPELINE (MLflow Model Registry)")
    print("======================================================================\n")

    try:
        # Try loading Production model first
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0].version
        print(f"✅ Loaded Production model version: {version}")
    except Exception:
        # Fallback to latest version
        print("⚠️ No model in Production — loading latest version instead...")
        latest = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging"])
        if not latest:
            raise Exception("No model found in registry.")
        version = latest[0].version
        model_uri = f"models:/{MODEL_NAME}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"📦 Loading model from registry → {model_uri}")

    return model, version


# ================================================================
# 📊 4. Evaluation & Logging
# ================================================================
def evaluate_model(model, X, y, version):
    """Evaluate model and log results to MLflow."""
    print("\n🚀 Evaluating model...")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=load_iris().target_names)

    # Save confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Iris Confusion Matrix (v{version})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = f"confusion_matrix_v{version}.png"
    plt.savefig(cm_path)
    plt.close()

    # Save confusion matrix CSV
    cm_csv = f"confusion_matrix_v{version}.csv"
    pd.DataFrame(cm).to_csv(cm_csv, index=False)

    # Create unique run name with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"eval-v{version}-{run_timestamp}"

    # MLflow Logging
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_version", version)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(cm_csv)
        mlflow.log_text(report, "classification_report.txt")

        print("\n✅ Evaluation Complete")
        print(f"   • Model Version: {version}")
        print(f"   • Accuracy: {acc:.4f}")
        print(f"   • Confusion Matrix:\n{cm}\n")
        print("Classification Report:\n", report)
        print(
            f"\n🏃 View run {run_name} at: "
            f"{MLFLOW_TRACKING_URI}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        )
        print(f"🧪 View experiment at: {MLFLOW_TRACKING_URI}/#/experiments/{run.info.experiment_id}")


# ================================================================
# 🏁 5. Main Execution
# ================================================================
if __name__ == "__main__":
    X, y = load_data()
    model, version = load_model_from_registry()
    evaluate_model(model, X, y, version)
