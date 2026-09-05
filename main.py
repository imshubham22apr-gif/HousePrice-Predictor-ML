from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(title="California Housing Price Predictor")

# Load the trained model globally if it exists
model_path = 'california_housing_model.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Define the input data schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    class Config:
        schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0,
                "AveOccup": 2.5556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Welcome to the California Housing Price Predictor API!"}

# Prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    if model is None:
        return {"error": "Model not found. Please train the model first."}
    
    # Convert input features to a DataFrame
    input_data = pd.DataFrame([features.dict()])
    
    # Make a prediction
    prediction = model.predict(input_data)
    predicted_value = prediction[0]
    
    return {"predicted_median_house_value": predicted_value}
