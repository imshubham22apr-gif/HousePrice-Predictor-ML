from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(title="Bengaluru Housing Price Predictor")

# Load the trained model globally if it exists
model_path = 'bengaluru_housing_model.joblib'
columns_path = 'columns.joblib'

model = joblib.load(model_path) if os.path.exists(model_path) else None
columns = joblib.load(columns_path) if os.path.exists(columns_path) else None

# Define the input data schema
class HouseFeatures(BaseModel):
    location: str
    total_sqft: float
    bhk: int
    bath: int

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Whitefield",
                "total_sqft": 1500.0,
                "bhk": 3,
                "bath": 3
            }
        }

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Welcome to the Bengaluru Housing Price Predictor API!"}

# Prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    if model is None or columns is None:
        return {"error": "Model or columns not found. Please train the model first."}
    
    # We need to construct an array with the exact same columns as the training data
    loc_index = -1
    if features.location in columns:
        loc_index = columns.index(features.location)
    
    # Initialize array with zeros
    x = np.zeros(len(columns))
    
    # The first 3 columns are total_sqft, bath, bhk in our train data after drop. 
    # Let's map dynamically just in case.
    if 'total_sqft' in columns: x[columns.index('total_sqft')] = features.total_sqft
    if 'bath' in columns: x[columns.index('bath')] = features.bath
    if 'bhk' in columns: x[columns.index('bhk')] = features.bhk
    
    # Set location one-hot encoding flag
    if loc_index >= 0:
        x[loc_index] = 1

    # Convert to DataFrame to match the input of the model
    input_data = pd.DataFrame([x], columns=columns)
    
    # Make a prediction
    prediction = model.predict(input_data)
    predicted_value = prediction[0]
    
    return {"predicted_price_lakhs": round(predicted_value, 2)}
