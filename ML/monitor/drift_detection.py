import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(train_data, inference_data):
    """Detect feature drift using Kolmogorov-Smirnov test"""
    drift_report = {}
    numeric_cols = train_data.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        stat, p = ks_2samp(
            train_data[col].dropna(),
            inference_data[col].dropna()
        )
        drift_report[col] = {"KS_stat": stat, "p_value": p}
    
    return drift_report