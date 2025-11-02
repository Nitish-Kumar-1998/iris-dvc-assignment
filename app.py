from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import uvicorn

# ✅ Load model from GCS
MODEL_URI = "gs://mlflow-artifacts-nitish/4/models/m-91ab307d3c2f4477a14fdcee6f3b0bc0/artifacts"
print("🔹 Loading model from Google Cloud Storage...")
model = mlflow.sklearn.load_model(MODEL_URI)
print("✅ Model loaded successfully!")

app = FastAPI(title="IRIS Model API", version="1.0.0")

# Define the request schema
class IrisData(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

@app.get("/")
def home():
    return {"message": "IRIS Model API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: IrisData):
    """
    Predict iris class from measurements
    Returns: 0 (Setosa), 1 (Versicolor), 2 (Virginica)
    """
    sample = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length_cm,
        "sepal width (cm)": data.sepal_width_cm,
        "petal length (cm)": data.petal_length_cm,
        "petal width (cm)": data.petal_width_cm
    }])
    
    prediction = model.predict(sample)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
