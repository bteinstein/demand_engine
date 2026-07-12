
# features/feature_engineering.py

import pandas as pd
import numpy as np
from datetime import datetime
from config import COLD_START_VALUE

def engineer_features(transactions, sku_metadata, customer_metadata, prediction_week=None):
    """Generate features for (Week, CustomerID, SKUID) pairs"""
    transactions["Week"] = pd.to_datetime(transactions["Week"])
    
    if prediction_week is None:
        prediction_week = transactions["Week"].max() + pd.Timedelta(weeks=1)
    prediction_week = pd.to_datetime(prediction_week)
    transactions = transactions[transactions["Week"] < prediction_week]
    # Add metadata
    transactions = transactions.merge(sku_metadata, on="SKUID", how="left")
    transactions = transactions.merge(customer_metadata, on="CustomerID", how="left")
    
    # Feature Engineering
    transactions["DaysSinceLastPurchase_SKU"] = (
        (pd.to_datetime(prediction_week) - transactions["Week"]).dt.days
    )
    transactions["TotalPurchases_SKU"] = transactions.groupby(
        ["CustomerID", "SKUID"]
    )["SKUID"].transform("count")
    transactions["Customer_RollingAvgOrderValue"] = transactions.groupby(
        ["CustomerID", "Week"]
    )["OrderValue"].transform("mean")
    
    # Cold-start handling
    transactions["DaysSinceLastPurchase_SKU"] = transactions[
        "DaysSinceLastPurchase_SKU"
    ].fillna(COLD_START_VALUE)
    transactions["TotalPurchases_SKU"] = transactions["TotalPurchases_SKU"].fillna(0)
    
    return transactions