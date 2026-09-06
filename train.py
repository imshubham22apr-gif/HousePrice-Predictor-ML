import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

def train_and_save_model():
    print("Loading Bengaluru Housing dataset...")
    df = pd.read_csv('Bengaluru_House_Data.csv')
    
    # 1. Feature selection & basic cleaning
    df = df[['location', 'size', 'total_sqft', 'bath', 'price']]
    df = df.dropna()

    # 2. BHK Extraction
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
    df = df.drop('size', axis=1)

    # 3. total_sqft Cleaning
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df = df.dropna()

    # 4. Dimensionality reduction for location
    df.location = df.location.apply(lambda x: x.strip())
    location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

    # 5. Outlier Removal
    # Minimum sqft per BHK check
    df = df[~(df.total_sqft / df.bhk < 300)]
    
    # Unrealistic bathroom ratio (bath > bhk + 2)
    df = df[df.bath < df.bhk + 2]
    
    # Price per sqft outlier removal (using 1 standard deviation per location)
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out
    df = remove_pps_outliers(df)
    df = df.drop(['price_per_sqft'], axis=1)

    print("Data preprocessed successfully. Shape:", df.shape)

    # One Hot Encoding
    dummies = pd.get_dummies(df.location)
    df = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
    df = df.drop('location', axis=1)

    X = df.drop('price', axis=1)
    y = df.price

    # Save the column structure for the API
    joblib.dump(list(X.columns), 'columns.joblib')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model trained! R2 Score: {score:.3f}")

    # Save the model
    model_filename = 'bengaluru_housing_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}.")

if __name__ == "__main__":
    train_and_save_model()
