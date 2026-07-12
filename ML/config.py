import os
from datetime import timedelta

# Paths
RAW_DATA_PATH = "data/raw/transactions.csv"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/saved_models/sku_purchase_model.pkl"
MLFLOW_TRACKING_URI = "mlruns"

# Negative Sampling
NEGATIVE_SAMPLE_RATIO = 10  # 10 negatives per positive
COLD_START_VALUE = 9999  # Placeholder for unseen SKUs

# Time Settings
LOOKBACK_WEEKS = 12  # Rolling window size
PREDICTION_WEEK = "2023-01-01"  # Example new week