import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
import random
import os

# =====================================================
# 1️⃣ Set random seeds for reproducibility
# =====================================================
np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# =====================================================
# 2️⃣ Load Data
# =====================================================
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# =====================================================
# 3️⃣ Define Poisoning Function (features + labels)
# =====================================================
def poison_data(X, y, poison_ratio=0.1, noise_level=1.5):
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    n_samples = int(len(X) * poison_ratio)
    indices = np.random.choice(X.index, n_samples, replace=False)

    # Feature noise
    X_poisoned.loc[indices] += np.random.normal(0, noise_level, X.shape[1])
    
    # Label flipping for stronger poisoning
    for i in indices:
        y_poisoned[i] = np.random.choice([0, 1, 2])
    return X_poisoned, y_poisoned

# =====================================================
# 4️⃣ Train and Log
# =====================================================
def train_and_log_model(X, y, poison_ratio):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    with mlflow.start_run(run_name=f"Poison_{int(poison_ratio*100)}%"):
        mlflow.log_param("poison_ratio", poison_ratio)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

    print(f"✅ Poison {int(poison_ratio*100)}% | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# =====================================================
# 5️⃣ Run Experiments
# =====================================================
mlflow.set_experiment("Iris_Poisoning_Experiment")

for ratio in [0.0, 0.05, 0.1, 0.5]:
    X_poisoned, y_poisoned = poison_data(X, y, poison_ratio=ratio)
    train_and_log_model(X_poisoned, y_poisoned, ratio)
