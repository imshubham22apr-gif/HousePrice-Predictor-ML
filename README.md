# 🏠 Bengaluru Housing Price Predictor API

An end-to-end Machine Learning deployment pipeline demonstrating how to train a model, wrap it in a REST API using **FastAPI**, and containerize it using **Docker**.

## 🚀 Overview
This repository contains a unified MLOps project that predicts house prices in Bengaluru based on real-world local parameters.

### 🌟 Key Features
- **Model Training:** `train.py` automatically processes the Kaggle Bengaluru Housing dataset, handles dimensionality reduction (One-Hot Encoding for locations), and trains a `RandomForestRegressor`.
- **FastAPI Backend:** A robust, high-performance RESTful API to serve the ML model predictions via a `POST` endpoint.
- **Strict Data Validation:** Uses Pydantic to ensure all incoming prediction requests (like BHK, sqft, bath, and location) are formatted correctly.
- **Dockerized:** Fully containerized environment for seamless deployment anywhere.

## 🛠️ Tech Stack
- **Python** 🐍
- **Scikit-Learn** & **Pandas** - Model Training & Data Handling
- **FastAPI** & **Uvicorn** - API Server & Framework
- **Docker** 🐳 - Containerization

## ⚙️ How to Run Locally

### 1. Train the Model
Before running the API, you need to train the model and generate the `.joblib` files:
```bash
# Install requirements
pip install -r requirements.txt

# Run the training script
python train.py
```
*This will create `bengaluru_housing_model.joblib` and `columns.joblib` files.*

### 2. Run using Docker (Recommended)
Make sure you have Docker installed and running.
```bash
# Build the Docker image
docker build -t housing-api .

# Run the Docker container
docker run -d -p 8080:80 housing-api
```

### 3. Test the API
Once the container is running, navigate to the interactive API documentation (Swagger UI):
👉 **http://localhost:8080/docs**

From there, you can test the `/predict` endpoint by sending JSON requests containing housing features.

---
*Built with ❤️ for scalable ML Deployments in India.*
