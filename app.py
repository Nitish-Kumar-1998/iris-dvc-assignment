from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# ✅ Load model from GCS
MODEL_URI = "gs://mlflow-artifacts-nitish/4/models/m-91ab307d3c2f4477a14fdcee6f3b0bc0/artifacts"
print("🔹 Loading model from Google Cloud Storage...")
model = mlflow.sklearn.load_model(MODEL_URI)
print("✅ Model loaded successfully!")


@app.route('/')
def home():
    return jsonify({"message": "IRIS Model API is running!"})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Expect JSON like {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, ...}
    sample = pd.DataFrame([data])

    prediction = model.predict(sample)
    return jsonify({"prediction": int(prediction[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
