# models/train_models.py

import os
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    log_loss, roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score
)
import shap
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42
EXPERIMENT_NAME = "demand_engine_prediction"
DATA_PATH = "data/processed/training_data.parquet"
MODEL_OUTPUT_DIR = "models"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Setup MLflow tracking (uncomment DagsHub config if needed)"""
    # os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/bteinstein/demand_engine.mlflow"
    # os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USER")
    # os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_experiment(EXPERIMENT_NAME)

def prepare_data(training_data, feature_columns):
    """Split data into train/test sets with time-aware split"""
    logger.info("Preparing data for training...")
    
    # Sort by Week to ensure chronological order
    training_data = training_data.sort_values('Week').reset_index(drop=True)
    
    # Time-based split
    split_idx = int(TRAIN_SPLIT * len(training_data))
    train_df = training_data.iloc[:split_idx]
    test_df = training_data.iloc[split_idx:]
    
    X_train = train_df[feature_columns]
    y_train = train_df['label']
    X_test = test_df[feature_columns]
    y_test = test_df['label']
    
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test, test_df

def calculate_metrics(y_true, y_pred_proba, y_pred):
    """Calculate all evaluation metrics"""
    return {
        "log_loss": log_loss(y_true, y_pred_proba),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": auc(*precision_recall_curve(y_true, y_pred_proba)[:2]),
        "f1_score": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred)
    }

def log_feature_importance(model, feature_columns, model_name):
    """Log feature importance if available"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return
        
        importance_df = pd.DataFrame({
            "feature": feature_columns,
            "importance": importance
        }).sort_values(by="importance", ascending=False).head(20)
        
        importance_path = f"{model_name}_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Clean up temporary file
        os.remove(importance_path)
        
    except Exception as e:
        logger.warning(f"Could not log feature importance for {model_name}: {str(e)}")

def log_shap_values(model, X_test, feature_columns, model_name):
    """Log SHAP summary plot if possible"""
    try:
        if hasattr(model, 'predict_proba') and hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.iloc[:100])  # Limit for performance
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test.iloc[:100], 
                            feature_names=feature_columns, show=False)
            
            shap_path = f"{model_name}_shap_summary.png"
            plt.savefig(shap_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            mlflow.log_artifact(shap_path)
            os.remove(shap_path)
            
    except Exception as e:
        logger.warning(f"Could not log SHAP values for {model_name}: {str(e)}")

def evaluate_model(model, X_test, y_test, model_name, feature_columns):
    """Evaluate model and log all metrics and artifacts"""
    logger.info(f"Evaluating {model_name}...")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate and log metrics
    metrics = calculate_metrics(y_test, y_pred_proba, y_pred)
    mlflow.log_metrics(metrics)
    
    # Log feature importance and SHAP values
    log_feature_importance(model, feature_columns, model_name)
    log_shap_values(model, X_test, feature_columns, model_name)
    
    logger.info(f"{model_name} - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
    return metrics['roc_auc']

class ModelTrainer:
    """Class to handle model training with consistent interface"""
    
    def __init__(self, X_train, y_train, X_test, y_test, feature_columns):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_columns = feature_columns
    
    def train_logistic_regression(self):
        """Train and evaluate Logistic Regression"""
        with mlflow.start_run(run_name="LogisticRegression"):
            model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=RANDOM_STATE
            )
            model.fit(self.X_train, self.y_train)
            auc_score = evaluate_model(model, self.X_test, self.y_test, 
                                     "LogisticRegression", self.feature_columns)
            mlflow.sklearn.log_model(model, "model")
        return model, auc_score
    
    def train_random_forest(self):
        """Train and evaluate Random Forest"""
        with mlflow.start_run(run_name="RandomForest"):
            model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=RANDOM_STATE
            )
            model.fit(self.X_train, self.y_train)
            auc_score = evaluate_model(model, self.X_test, self.y_test, 
                                     "RandomForest", self.feature_columns)
            mlflow.sklearn.log_model(model, "model")
        return model, auc_score
    
    def train_xgboost(self):
        """Train and evaluate XGBoost"""
        with mlflow.start_run(run_name="XGBoost"):
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=10,
                random_state=RANDOM_STATE
            )
            model.fit(self.X_train, self.y_train)
            auc_score = evaluate_model(model, self.X_test, self.y_test, 
                                     "XGBoost", self.feature_columns)
            mlflow.sklearn.log_model(model, "model")
        return model, auc_score
    
    def train_lightgbm(self):
        """Train and evaluate LightGBM"""
        with mlflow.start_run(run_name="LightGBM"):
            model = LGBMClassifier(
                class_weight='balanced',
                verbose=-1,
                random_state=RANDOM_STATE
            )
            model.fit(self.X_train, self.y_train)
            auc_score = evaluate_model(model, self.X_test, self.y_test, 
                                     "LightGBM", self.feature_columns)
            mlflow.sklearn.log_model(model, "model")
        return model, auc_score
    
    def train_catboost(self):
        """Train and evaluate CatBoost"""
        with mlflow.start_run(run_name="CatBoost"):
            model = CatBoostClassifier(
                iterations=100,
                verbose=0,
                random_state=RANDOM_STATE
            )
            model.fit(self.X_train, self.y_train)
            auc_score = evaluate_model(model, self.X_test, self.y_test, 
                                     "CatBoost", self.feature_columns)
            mlflow.sklearn.log_model(model, "model")
        return model, auc_score

def train_all_models(trainer):
    """Train all models and return results"""
    models_results = []
    
    training_methods = [
        ("LogisticRegression", trainer.train_logistic_regression),
        ("RandomForest", trainer.train_random_forest),
        ("XGBoost", trainer.train_xgboost),
        ("LightGBM", trainer.train_lightgbm),
        ("CatBoost", trainer.train_catboost)
    ]
    
    for name, method in training_methods:
        try:
            logger.info(f"Training {name}...")
            model, auc_score = method()
            models_results.append((name, model, auc_score))
        except Exception as e:
            logger.error(f"Failed to train {name}: {str(e)}")
    
    return models_results

def save_best_model(models_results, output_dir):
    """Find and save the best performing model"""
    if not models_results:
        raise ValueError("No models were successfully trained")
    
    # Find best model by AUC score
    best_name, best_model, best_auc = max(models_results, key=lambda x: x[2])
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = Path(output_dir) / f"{best_name}_best.pkl"
    joblib.dump(best_model, model_path)
    
    logger.info(f"Best Model: {best_name} with AUC {best_auc:.4f}")
    logger.info(f"Model saved to: {model_path}")
    
    return best_model, best_name, best_auc

def main():
    """Main function to run full training pipeline"""
    try:
        logger.info("Starting training pipeline...")
        
        # Setup MLflow
        setup_mlflow()
        
        # Load data
        if not Path(DATA_PATH).exists():
            raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
        
        training_data = pd.read_parquet(DATA_PATH)
        logger.info(f"Loaded training data: {training_data.shape}")
        
        # Get feature columns
        excluded_cols = ['Week', 'CustomerID', 'SKUID', 'label', 'prediction_week']
        feature_columns = [col for col in training_data.columns if col not in excluded_cols]
        logger.info(f"Using {len(feature_columns)} features")
        
        # Prepare data
        X_train, X_test, y_train, y_test, test_df = prepare_data(training_data, feature_columns)
        
        # Initialize trainer
        trainer = ModelTrainer(X_train, y_train, X_test, y_test, feature_columns)
        
        # Train all models
        models_results = train_all_models(trainer)
        
        # Save best model
        best_model, model_name, auc_score = save_best_model(models_results, MODEL_OUTPUT_DIR)
        
        return best_model, model_name, auc_score
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    best_model, model_name, auc_score = main()