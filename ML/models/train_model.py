import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow
import joblib
from config import MODEL_PATH, MLFLOW_TRACKING_URI

def train_model(X_train, y_train, X_test, y_test):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        model = xgb.XGBClassifier(
            scale_pos_weight=10,
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({"AUC": auc, "AvgPrecision": ap})
        mlflow.sklearn.log_model(model, "model")
        
        # Save locally
        joblib.dump(model, MODEL_PATH)
        return model