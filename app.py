from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Churn Prediction API")

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

class CustomerData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    features = np.array(data.features).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3)
    }