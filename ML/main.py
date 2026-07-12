import pandas as pd
from features.feature_engineering import engineer_features
from features.negative_sampling_dep import generate_negative_samples
from models.train_model import train_model
from predict.predict_new_week import predict_for_new_week
from evaluate.evaluate_model import evaluate_model
from monitor.drift_detection import detect_drift
import joblib

# Load data
transactions = pd.read_csv("data/raw/transactions.csv")
sku_metadata = pd.read_csv("data/raw/sku_metadata.csv")
customer_metadata = pd.read_csv("data/raw/customer_metadata.csv")

# Feature Engineering
featured_data = engineer_features(transactions, sku_metadata, customer_metadata)

# Negative Sampling
n_skus = sku_metadata["SKUID"].nunique()
neg_samples = generate_negative_samples(featured_data, n_skus)
full_data = pd.concat([featured_data, neg_samples])

# Train/Test Split
train = full_data[full_data["Week"] < "2022-12-01"]
test = full_data[full_data["Week"] >= "2022-12-01"]

X_train = train.drop(columns=["Label", "Week", "CustomerID", "SKUID"])
y_train = train["Label"]
X_test = test.drop(columns=["Label", "Week", "CustomerID", "SKUID"])
y_test = test["Label"]

# Model Training
model = train_model(X_train, y_train, X_test, y_test)

# Evaluation
evaluate_model(model, X_test, y_test)

# Predict for New Week
target_customers = customer_metadata["CustomerID"].unique()
all_skus = sku_metadata["SKUID"].unique()
predictions = predict_for_new_week(model, transactions, all_skus, target_customers)
predictions.to_csv("predictions.csv", index=False)

# Drift Detection
drift_report = detect_drift(X_train, X_test)
print("Feature Drift Report:", drift_report)