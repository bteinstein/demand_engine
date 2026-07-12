# features/feature_engineering.py

# features/feature_engineering.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PurchaseLikelihoodFeatureGenerator:
    """Core feature engineering for both training and prediction"""
    
    def __init__(self, cold_start_value=9999):
        self.cold_start_value = cold_start_value
        self.rolling_windows = [4, 8, 12]
        self.prediction_week = None
        
    def prepare_features(self, transactions, sku_metadata, customer_metadata, 
                       prediction_week, candidate_pairs=None):
        """
        Main entry point for feature generation
        - For training: candidate_pairs=None generates all historical pairs
        - For prediction: candidate_pairs=precomputed (CustomerID, SKUID) pairs
        """
        self.prediction_week = pd.to_datetime(prediction_week)
        
        # Prepare base data
        transactions = self._prepare_base_data(
            transactions, sku_metadata, customer_metadata
        )
        
        # Generate all features
        feature_components = self._generate_feature_components(transactions)
        
        # Create feature matrix
        if candidate_pairs is not None:
            return self._create_prediction_matrix(candidate_pairs, feature_components)
        return self._create_training_matrix(transactions, feature_components)
    
    def _prepare_base_data(self, transactions, sku_metadata, customer_metadata):
        """Common data preparation for all use cases"""
        transactions = transactions.copy()
        transactions["Week"] = pd.to_datetime(transactions["Week"])
        transactions = transactions[transactions["Week"] < self.prediction_week]
        
        return (
            transactions
            .merge(sku_metadata, on="SKUID", how="left")
            .merge(customer_metadata, on="CustomerID", how="left")
        )
    
    def _generate_feature_components(self, transactions):
        """Generate all feature categories"""
        return {
            **self._customer_behavior_features(transactions),
            **self._product_affinity_features(transactions),
            **self._market_context_features(transactions),
            **self._temporal_features(transactions)
        }
    
    # Feature calculation methods remain similar but return DataFrames
    # with keys that include feature type prefixes
    
class TrainingDataBuilder:
    """Handles training-specific operations including negative sampling"""
    
    def __init__(self, feature_generator, negative_sampler):
        self.feature_generator = feature_generator
        self.negative_sampler = negative_sampler
        
    def build_training_dataset(self, transactions, sku_metadata, customer_metadata,
                             prediction_week, sampling_params):
        """
        Full training dataset construction pipeline:
        1. Generate positive samples
        2. Generate negative samples
        3. Combine with features
        """
        # Generate positive samples
        positives = self._get_positive_samples(transactions, prediction_week)
        
        # Generate negative samples
        negatives = self.negative_sampler.generate_negative_samples(
            transactions, sku_metadata, customer_metadata, **sampling_params
        )
        
        # Combine and get features
        all_samples = pd.concat([positives, negatives])
        return self.feature_generator.prepare_features(
            transactions, sku_metadata, customer_metadata,
            prediction_week, all_samples[['CustomerID', 'SKUID']]
        )
    
class PredictionDataBuilder:
    """Handles prediction-specific feature generation"""
    
    def __init__(self, feature_generator, candidate_selector=None):
        self.feature_generator = feature_generator
        self.candidate_selector = candidate_selector or DefaultCandidateSelector()
        
    def build_prediction_data(self, transactions, sku_metadata, customer_metadata,
                            prediction_week, customers=None):
        """
        Prediction dataset construction:
        1. Select candidate SKUs for each customer
        2. Generate features for these pairs
        """
        # Get prediction candidates
        candidate_pairs = self.candidate_selector.get_candidates(
            transactions, sku_metadata, customer_metadata,
            prediction_week, customers
        )
        
        # Generate features
        return self.feature_generator.prepare_features(
            transactions, sku_metadata, customer_metadata,
            prediction_week, candidate_pairs
        )

class AdvancedCandidateSelector:
    """Generates relevant SKU candidates for prediction"""
    
    def get_candidates(self, transactions, sku_metadata, customer_metadata,
                     prediction_week, customers=None):
        """
        Returns DataFrame of (CustomerID, SKUID) pairs to score
        Implements smart candidate selection strategies
        """
        # Implementation combining business rules and ML-based retrieval
        # (could use popularity, embeddings, etc.)

# Usage Example
def main():
    # Initialize core components
    feature_gen = PurchaseLikelihoodFeatureGenerator(cold_start_value=9999)
    negative_sampler = AdvancedNegativeSampler()
    candidate_selector = AdvancedCandidateSelector()
    
    # Training workflow
    training_builder = TrainingDataBuilder(feature_gen, negative_sampler)
    training_data = training_builder.build_training_dataset(
        transactions, sku_meta, cust_meta,
        prediction_week="2024-01-01",
        sampling_params={'max_negatives': 15, 'category_weight': 0.5}
    )
    
    # Prediction workflow
    prediction_builder = PredictionDataBuilder(feature_gen, candidate_selector)
    
    # For all customers
    prediction_data = prediction_builder.build_prediction_data(
        transactions, sku_meta, cust_meta,
        prediction_week="2024-02-01"
    )
    
    # For specific customers
    targeted_prediction = prediction_builder.build_prediction_data(
        transactions, sku_meta, cust_meta,
        prediction_week="2024-02-01",
        customers=["C123", "C456"]
    )

if __name__ == "__main__":
    main()