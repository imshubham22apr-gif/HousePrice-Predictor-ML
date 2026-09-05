import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import joblib

def train_and_save_model():
    print("Starting the training script...")

    # Load the California Housing dataset
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='MedHouseVal')
    print("Dataset loaded successfully.")

    # Select the specific features we want
    features_to_use = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    X = X[features_to_use]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Data split into training and testing sets.")

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("Training the RandomForestRegressor model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Save the trained model to a file
    model_filename = 'california_housing_model.joblib'
    joblib.dump(model, model_filename)

    print(f"Model saved as {model_filename}. Training script finished.")

if __name__ == "__main__":
    train_and_save_model()
