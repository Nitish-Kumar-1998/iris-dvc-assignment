from flask import Flask, request, jsonify
import mlflow.sklearn
import os

app = Flask(__name__)

# MODEL_URI = "gs://mlflow-artifacts-nitish/4/models/m-91ab307d3c2f4477a14fdcee6f3b0bc0/artifacts"
# If bucket not used, load local model
MODEL_URI = "models/latest_model"  # fallback if no GCS access

print("🔹 Loading model...")
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("⚠️ Model not found, running without model:", e)
    model = None

@app.route('/')
def home():
    return jsonify({"message": "Iris Prediction API is live!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    features = [[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]]
    prediction = model.predict(features)[0]
    return jsonify({
        "input": data,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
