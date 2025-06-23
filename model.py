import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # <--- for saving models
import logging

# ------------------------------
# Step 1: Setup Logging for Debugging
# ------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # ------------------------------
    # Step 2: Load Data
    # ------------------------------
    logging.info("Loading dataset...")
    df = pd.read_csv('inventory_dataset.csv')  # Make sure the file exists

    logging.info(f"Data loaded. Shape: {df.shape}")
    logging.debug(df.head())

    # ------------------------------
    # Step 3: Basic Cleaning
    # ------------------------------
    logging.info("Checking for missing values...")
    if df.isnull().sum().any():
        logging.warning("Missing values found. Filling with 0.")
        df.fillna(0, inplace=True)
    else:
        logging.info("No missing values.")

    # ------------------------------
    # Step 4: Feature Engineering
    # ------------------------------
    logging.info("Converting date column...")
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        logging.error(f"Date conversion failed: {e}")
        raise

    df['month'] = df['date'].dt.month

    # Optional: Convert categorical variables to numeric
    if 'product_id' in df.columns:
        df['product_id'] = df['product_id'].astype('category')
        product_encoder = df['product_id'].cat.categories
        df['product_id'] = df['product_id'].cat.codes
    if 'store_id' in df.columns:
        df['store_id'] = df['store_id'].astype('category')
        store_encoder = df['store_id'].cat.categories
        df['store_id'] = df['store_id'].cat.codes

    # ------------------------------
    # Step 5: Split Features & Target
    # ------------------------------
    logging.info("Splitting features and target...")
    features = ['month', 'product_id', 'store_id']
    target = 'demand'

    if not all(col in df.columns for col in features + [target]):
        logging.error("One or more feature/target columns missing.")
        raise KeyError("Check columns: " + str(df.columns.tolist()))

    X = df[features]
    y = df[target]

    # ------------------------------
    # Step 6: Train/Test Split
    # ------------------------------
    logging.info("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ------------------------------
    # Step 7: Model Training
    # ------------------------------
    logging.info("Training XGBoost model...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # ------------------------------
    # Step 8: Prediction & Evaluation
    # ------------------------------
    logging.info("Predicting on test set...")
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    logging.info(f"Model Evaluation Complete. RMSE: {rmse:.2f}")

    # ------------------------------
    # Step 9: Save Model & Encoders
    # ------------------------------
    logging.info("Saving trained model and encoders...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(product_encoder, 'product_encoder.pkl')
    joblib.dump(store_encoder, 'store_encoder.pkl')
    logging.info("Model and encoders saved successfully.")

except FileNotFoundError:
    logging.critical("CSV file not found. Please check the file path.")
except Exception as e:
    logging.exception("An error occurred during execution.")
