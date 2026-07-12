# features/negative_sampling.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedNegativeSampler:
    """Advanced negative sampling with hierarchical popularity and hard negative mining"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_negative_samples(self, transactions, sku_metadata, customer_metadata,
                                max_negatives=20, category_weight=0.5, trending_weight=0.3,
                                fallback_levels=["Town", "City", "State", "Global"],
                                use_trending=True, use_category_relevance=True):
        """
        Generates negative samples using hierarchical localized popularity and hard negative mining
        """
        print("Generating advanced negative samples...")
        
        # Get all unique SKU IDs (handle actual SKU ID format)
        all_skus = sku_metadata['SKUID'].unique()
        sku_to_idx = {sku: idx for idx, sku in enumerate(all_skus)}
        idx_to_sku = {idx: sku for sku, idx in sku_to_idx.items()}
        
        # Precompute popularity stats
        popularity_stats = self._precompute_popularity_stats(
            transactions, sku_metadata, fallback_levels, all_skus
        )
        
        # Precompute trending SKUs (last 4 weeks)
        trending_skus = None
        if use_trending:
            trending_skus = self._compute_trending_skus(transactions, all_skus)
        
        negative_samples = []
        
        # Group by (Week, CustomerID) for negative sampling
        grouped = transactions.groupby(["Week", "CustomerID"])
        
        for (week, customer_id), group in grouped:
            purchased_skus = set(group["SKUID"].unique())
            non_purchased_skus = list(set(all_skus) - purchased_skus)
            
            if len(non_purchased_skus) == 0:
                continue
            
            # Get customer location info
            customer_info = customer_metadata[
                customer_metadata["CustomerID"] == customer_id
            ].iloc[0]
            
            # Get location-based weights
            location_weights = self._get_location_weights(
                customer_info, popularity_stats, fallback_levels, non_purchased_skus
            )
            
            # Combine weights: location + category + trending
            combined_weights = location_weights.copy()
            
            if use_category_relevance:
                category_weights = self._get_category_weights(
                    group, sku_metadata, non_purchased_skus
                )
                combined_weights += category_weight * category_weights
            
            if use_trending and trending_skus is not None:
                trending_weights = np.array([
                    trending_skus.get(sku, 0) for sku in non_purchased_skus
                ])
                combined_weights += trending_weight * trending_weights
            
            # Normalize weights
            combined_weights = self._normalize_weights(combined_weights)
            
            # Adaptive sampling ratio
            n_purchased = len(purchased_skus)
            n_negatives = min(n_purchased * 2, max_negatives, len(non_purchased_skus))
            
            # Sample negatives
            try:
                if combined_weights.sum() > 0:
                    sampled_skus = np.random.choice(
                        non_purchased_skus,
                        size=n_negatives,
                        p=combined_weights,
                        replace=False
                    )
                else:
                    # Fallback to uniform sampling
                    sampled_skus = np.random.choice(
                        non_purchased_skus,
                        size=n_negatives,
                        replace=False
                    )
            except (ValueError, TypeError):
                # Fallback to uniform sampling
                sampled_skus = np.random.choice(
                    non_purchased_skus,
                    size=n_negatives,
                    replace=False
                )
            
            # Add to negative samples
            for sku in sampled_skus:
                negative_samples.append({
                    'Week': week,
                    'CustomerID': customer_id,
                    'SKUID': sku,
                    'label': 0
                })
        
        return pd.DataFrame(negative_samples)
    
    def _precompute_popularity_stats(self, transactions, sku_metadata, fallback_levels, all_skus):
        """Precompute popularity stats at multiple geographic levels"""
        stats = {}
        
        # Merge transactions with customer metadata for location info
        trans_with_location = transactions.merge(
            sku_metadata[['SKUID']], on='SKUID', how='left'
        )
        
        for level in fallback_levels:
            if level == "Global":
                # Global popularity
                global_counts = transactions['SKUID'].value_counts()
                total_transactions = len(transactions)
                stats[level] = {
                    sku: count / total_transactions 
                    for sku, count in global_counts.items()
                }
            else:
                # Location-specific popularity
                if level in transactions.columns:
                    level_stats = {}
                    for location in transactions[level].unique():
                        location_trans = transactions[transactions[level] == location]
                        if len(location_trans) > 0:
                            location_counts = location_trans['SKUID'].value_counts()
                            total_location_trans = len(location_trans)
                            level_stats[location] = {
                                sku: count / total_location_trans 
                                for sku, count in location_counts.items()
                            }
                    stats[level] = level_stats
        
        return stats
    
    def _compute_trending_skus(self, transactions, all_skus, weeks=4):
        """Compute trending SKUs from recent weeks"""
        max_week = pd.to_datetime(transactions['Week']).max()
        cutoff_date = max_week - pd.Timedelta(weeks=weeks)
        
        recent_transactions = transactions[
            pd.to_datetime(transactions['Week']) > cutoff_date
        ]
        
        if len(recent_transactions) == 0:
            return {}
        
        trending_counts = recent_transactions['SKUID'].value_counts()
        total_recent = len(recent_transactions)
        
        return {sku: count / total_recent for sku, count in trending_counts.items()}
    
    def _get_location_weights(self, customer_info, popularity_stats, fallback_levels, non_purchased_skus):
        """Get popularity weights based on customer's location with hierarchical fallback"""
        
        for level in fallback_levels:
            if level == "Global":
                # Use global popularity
                weights = np.array([
                    popularity_stats[level].get(sku, 0) 
                    for sku in non_purchased_skus
                ])
                return weights
            else:
                # Try location-specific popularity
                if (level in popularity_stats and 
                    customer_info[level] in popularity_stats[level]):
                    
                    location_stats = popularity_stats[level][customer_info[level]]
                    weights = np.array([
                        location_stats.get(sku, 0) 
                        for sku in non_purchased_skus
                    ])
                    
                    if weights.sum() > 0:  # Found meaningful weights
                        return weights
        
        # Final fallback: uniform weights
        return np.ones(len(non_purchased_skus)) / len(non_purchased_skus)
    
    def _get_category_weights(self, group, sku_metadata, non_purchased_skus):
        """Create weights for category relevance"""
        # Get categories of purchased SKUs
        purchased_categories = set(
            group.merge(sku_metadata, on='SKUID')['Category'].unique()
        )
        
        # Weight non-purchased SKUs by category relevance
        weights = []
        for sku in non_purchased_skus:
            sku_category = sku_metadata[sku_metadata['SKUID'] == sku]['Category'].iloc[0]
            weight = 1.0 if sku_category in purchased_categories else 0.1
            weights.append(weight)
        
        return np.array(weights)
    
    def _normalize_weights(self, weights):
        """Normalize weights, handle zero-sum cases"""
        total = weights.sum()
        if total == 0 or np.isnan(total):
            return np.ones_like(weights) / len(weights)
        return weights / total


class IntegratedFeatureEngineer:
    """Enhanced feature engineering with advanced negative sampling"""
    
    def __init__(self, cold_start_value=9999, random_state=42):
        self.cold_start_value = cold_start_value
        self.rolling_windows = [4, 8, 12]
        self.negative_sampler = AdvancedNegativeSampler(random_state=random_state)
        
    def create_training_dataset(self, transactions, sku_metadata, customer_metadata,
                              prediction_week=None, target_customers=None,
                              max_negatives=20, category_weight=0.5, trending_weight=0.3):
        """
        Complete pipeline: feature engineering + advanced negative sampling
        """
        print("Starting integrated feature engineering pipeline...")
        
        # Step 1: Generate comprehensive features
        feature_df = self._engineer_features(
            transactions, sku_metadata, customer_metadata, 
            prediction_week, target_customers
        )
        
        # Step 2: Create positive samples from historical data
        historical_transactions = transactions.copy()
        if prediction_week:
            historical_transactions = historical_transactions[
                pd.to_datetime(historical_transactions['Week']) < pd.to_datetime(prediction_week)
            ]
        
        positive_samples = historical_transactions.groupby(['Week', 'CustomerID', 'SKUID']).size().reset_index()
        positive_samples['label'] = 1
        positive_samples = positive_samples[['Week', 'CustomerID', 'SKUID', 'label']]
        
        # Step 3: Generate negative samples using advanced sampling
        negative_samples = self.negative_sampler.generate_negative_samples(
            transactions=historical_transactions,
            sku_metadata=sku_metadata,
            customer_metadata=customer_metadata,
            max_negatives=max_negatives,
            category_weight=category_weight,
            trending_weight=trending_weight
        )
        
        # Step 4: Combine positive and negative samples
        training_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
        
        # Step 5: Merge with features
        training_dataset = training_samples.merge(
            feature_df, on=['CustomerID', 'SKUID'], how='left'
        )
        
        # Handle any remaining missing values
        feature_columns = [col for col in training_dataset.columns 
                          if col not in ['Week', 'CustomerID', 'SKUID', 'label', 'prediction_week']]
        training_dataset[feature_columns] = training_dataset[feature_columns].fillna(0)
        
        print(f"Training dataset created:")
        print(f"- Total samples: {len(training_dataset):,}")
        print(f"- Positive samples: {len(training_dataset[training_dataset['label']==1]):,}")
        print(f"- Negative samples: {len(training_dataset[training_dataset['label']==0]):,}")
        print(f"- Features: {len(feature_columns)}")
        
        return training_dataset, feature_columns
    
    def _engineer_features(self, transactions, sku_metadata, customer_metadata, 
                          prediction_week=None, target_customers=None):
        """Generate comprehensive features (same as before but integrated)"""
        print("Generating comprehensive features...")
        
        # Data preparation
        transactions = self._prepare_data(transactions, sku_metadata, customer_metadata, 
                                       prediction_week)
        
        if target_customers is not None:
            transactions = transactions[transactions['CustomerID'].isin(target_customers)]
        
        # Generate all feature categories
        features_dict = {}
        
        features_dict.update(self._customer_sku_interaction_features(transactions))
        features_dict.update(self._customer_general_features(transactions, customer_metadata))
        features_dict.update(self._localized_sku_features(transactions))
        features_dict.update(self._time_series_features(transactions))
        features_dict.update(self._granular_location_features(transactions))
        
        # Combine all features
        feature_df = self._combine_features(features_dict, transactions)
        
        return feature_df
    
    def _prepare_data(self, transactions, sku_metadata, customer_metadata, prediction_week):
        """Prepare and clean data"""
        transactions = transactions.copy()
        transactions["Week"] = pd.to_datetime(transactions["Week"])
        
        if prediction_week is None:
            prediction_week = transactions["Week"].max() + pd.Timedelta(weeks=1)
        
        self.prediction_week = pd.to_datetime(prediction_week)
        
        # Filter to historical data only
        transactions = transactions[transactions["Week"] < self.prediction_week]
        
        # Add metadata
        transactions = transactions.merge(sku_metadata, on="SKUID", how="left")
        transactions = transactions.merge(customer_metadata, on="CustomerID", how="left")
        
        return transactions
    
    # Include all the feature engineering methods from the previous version
    def _customer_sku_interaction_features(self, transactions):
        """Generate customer-SKU interaction features"""
        features = {}
        
        # Days since last purchase features
        last_purchase_sku = transactions.groupby(['CustomerID', 'SKUID'])['Week'].max().reset_index()
        last_purchase_sku['DaysSinceLastPurchase_SKU'] = (self.prediction_week - last_purchase_sku['Week']).dt.days
        
        last_purchase_category = transactions.groupby(['CustomerID', 'Category'])['Week'].max().reset_index()
        last_purchase_category['DaysSinceLastPurchase_Category'] = (self.prediction_week - last_purchase_category['Week']).dt.days
        
        last_purchase_segment = transactions.groupby(['CustomerID', 'Segment'])['Week'].max().reset_index()
        last_purchase_segment['DaysSinceLastPurchase_Segment'] = (self.prediction_week - last_purchase_segment['Week']).dt.days
        
        # Total purchases features
        total_purchases_sku = transactions.groupby(['CustomerID', 'SKUID']).size().reset_index(name='TotalPurchases_SKU')
        total_purchases_category = transactions.groupby(['CustomerID', 'Category']).size().reset_index(name='TotalPurchases_Category')
        total_purchases_segment = transactions.groupby(['CustomerID', 'Segment']).size().reset_index(name='TotalPurchases_Segment')
        
        # Average order value features
        avg_order_sku = transactions.groupby(['CustomerID', 'SKUID'])['OrderValue'].mean().reset_index()
        avg_order_sku.rename(columns={'OrderValue': 'AvgOrderValue_SKU_by_Customer'}, inplace=True)
        
        avg_order_category = transactions.groupby(['CustomerID', 'Category'])['OrderValue'].mean().reset_index()
        avg_order_category.rename(columns={'OrderValue': 'AvgOrderValue_Category_by_Customer'}, inplace=True)
        
        features['last_purchase_sku'] = last_purchase_sku
        features['last_purchase_category'] = last_purchase_category
        features['last_purchase_segment'] = last_purchase_segment
        features['total_purchases_sku'] = total_purchases_sku
        features['total_purchases_category'] = total_purchases_category
        features['total_purchases_segment'] = total_purchases_segment
        features['avg_order_sku'] = avg_order_sku
        features['avg_order_category'] = avg_order_category
        
        return features
    
    def _customer_general_features(self, transactions, customer_metadata):
        """Generate customer general behavior features"""
        features = {}
        
        customer_general = customer_metadata[['CustomerID', 'Recency', 'Frequency', 'Monetary']].copy()
        customer_general.rename(columns={
            'Recency': 'Customer_Recency',
            'Frequency': 'Customer_Frequency', 
            'Monetary': 'Customer_Monetary'
        }, inplace=True)
        
        customer_stats = transactions.groupby('CustomerID').agg({
            'SKUID': 'nunique',
            'OrderValue': 'mean'
        }).reset_index()
        
        customer_stats.rename(columns={
            'SKUID': 'Customer_TotalUniqueSKUsPurchased_Overall',
            'OrderValue': 'Customer_AvgOrderValue_Overall'
        }, inplace=True)
        
        customer_general = customer_general.merge(customer_stats, on='CustomerID', how='left')
        
        features['customer_general'] = customer_general
        return features
    
    def _localized_sku_features(self, transactions):
        """Generate localized SKU performance features"""
        features = {}
        
        sku_state_sales = transactions.groupby(['SKUID', 'State'])['OrderValue'].sum().reset_index()
        sku_state_sales.rename(columns={'OrderValue': 'SKU_TotalSales_State'}, inplace=True)
        
        sku_state_count = transactions.groupby(['SKUID', 'State']).size().reset_index(name='SKU_TotalPurchaseCount_State')
        
        category_state_sales = transactions.groupby(['Category', 'State'])['OrderValue'].sum().reset_index()
        category_state_sales.rename(columns={'OrderValue': 'Category_TotalSales_State'}, inplace=True)
        
        segment_state_sales = transactions.groupby(['Segment', 'State'])['OrderValue'].sum().reset_index()
        segment_state_sales.rename(columns={'OrderValue': 'Segment_TotalSales_State'}, inplace=True)
        
        manufacturer_state_sales = transactions.groupby(['Manufacturer', 'State'])['OrderValue'].sum().reset_index()
        manufacturer_state_sales.rename(columns={'OrderValue': 'Manufacturer_TotalSales_State'}, inplace=True)
        
        features['sku_state_sales'] = sku_state_sales
        features['sku_state_count'] = sku_state_count
        features['category_state_sales'] = category_state_sales
        features['segment_state_sales'] = segment_state_sales
        features['manufacturer_state_sales'] = manufacturer_state_sales
        
        return features
    
    def _time_series_features(self, transactions):
        """Generate time-series and rolling window features"""
        features = {}
        
        for window in self.rolling_windows:
            cutoff_date = self.prediction_week - pd.Timedelta(weeks=window)
            recent_transactions = transactions[transactions['Week'] >= cutoff_date]
            
            customer_rolling = recent_transactions.groupby('CustomerID').agg({
                'OrderValue': 'mean',
                'SKUID': 'nunique',
                'Week': 'count'
            }).reset_index()
            
            customer_rolling.rename(columns={
                'OrderValue': f'Customer_RollingAvgOrderValue_{window}Weeks',
                'SKUID': f'Customer_RollingUniqueSKUsPurchased_{window}Weeks',
                'Week': f'Customer_RollingPurchaseCount_{window}Weeks'
            }, inplace=True)
            
            customer_category_rolling = recent_transactions.groupby(['CustomerID', 'Category']).size().reset_index()
            customer_category_rolling.rename(columns={0: f'Customer_RollingCategoryPurchaseCount_{window}Weeks'}, inplace=True)
            
            sku_rolling_sales = recent_transactions.groupby('SKUID')['OrderValue'].sum().reset_index()
            sku_rolling_sales.rename(columns={'OrderValue': f'SKU_RollingTotalSales_{window}Weeks'}, inplace=True)
            
            sku_rolling_count = recent_transactions.groupby('SKUID').size().reset_index()
            sku_rolling_count.rename(columns={0: f'SKU_RollingPurchaseCount_{window}Weeks'}, inplace=True)
            
            category_rolling_sales = recent_transactions.groupby('Category')['OrderValue'].sum().reset_index()
            category_rolling_sales.rename(columns={'OrderValue': f'Category_RollingTotalSales_{window}Weeks'}, inplace=True)
            
            segment_rolling_sales = recent_transactions.groupby('Segment')['OrderValue'].sum().reset_index()
            segment_rolling_sales.rename(columns={'OrderValue': f'Segment_RollingTotalSales_{window}Weeks'}, inplace=True)
            
            features[f'customer_rolling_{window}w'] = customer_rolling
            features[f'customer_category_rolling_{window}w'] = customer_category_rolling
            features[f'sku_rolling_sales_{window}w'] = sku_rolling_sales
            features[f'sku_rolling_count_{window}w'] = sku_rolling_count
            features[f'category_rolling_sales_{window}w'] = category_rolling_sales
            features[f'segment_rolling_sales_{window}w'] = segment_rolling_sales
        
        return features
    
    def _granular_location_features(self, transactions):
        """Generate granular location affinity features"""
        features = {}
        
        for window in self.rolling_windows:
            cutoff_date = self.prediction_week - pd.Timedelta(weeks=window)
            recent_transactions = transactions[transactions['Week'] >= cutoff_date]
            
            town_category = recent_transactions.groupby(['Town', 'Category']).size().reset_index()
            town_category.rename(columns={0: f'Town_Category_PurchaseCount_{window}Weeks'}, inplace=True)
            
            town_segment = recent_transactions.groupby(['Town', 'Segment']).size().reset_index()
            town_segment.rename(columns={0: f'Town_Segment_PurchaseCount_{window}Weeks'}, inplace=True)
            
            town_sku = recent_transactions.groupby(['Town', 'SKUID']).size().reset_index()
            town_sku.rename(columns={0: f'Town_SKU_PurchaseCount_{window}Weeks'}, inplace=True)
            
            town_manufacturer = recent_transactions.groupby(['Town', 'Manufacturer']).size().reset_index()
            town_manufacturer.rename(columns={0: f'Town_Manufacturer_PurchaseCount_{window}Weeks'}, inplace=True)
            
            features[f'town_category_{window}w'] = town_category
            features[f'town_segment_{window}w'] = town_segment
            features[f'town_sku_{window}w'] = town_sku
            features[f'town_manufacturer_{window}w'] = town_manufacturer
        
        return features
    
    def _combine_features(self, features_dict, transactions):
        """Combine all features into final feature matrix"""
        
        unique_combinations = transactions[['CustomerID', 'SKUID', 'Category', 'Segment', 
                                         'Manufacturer', 'State', 'City', 'Town']].drop_duplicates()
        
        feature_df = unique_combinations.copy()
        
        # Merge all features with proper handling of missing values
        merge_configs = [
            ('last_purchase_sku', ['CustomerID', 'SKUID'], 'DaysSinceLastPurchase_SKU'),
            ('last_purchase_category', ['CustomerID', 'Category'], 'DaysSinceLastPurchase_Category'),
            ('last_purchase_segment', ['CustomerID', 'Segment'], 'DaysSinceLastPurchase_Segment'),
            ('total_purchases_sku', ['CustomerID', 'SKUID'], 'TotalPurchases_SKU'),
            ('total_purchases_category', ['CustomerID', 'Category'], 'TotalPurchases_Category'),
            ('total_purchases_segment', ['CustomerID', 'Segment'], 'TotalPurchases_Segment'),
            ('avg_order_sku', ['CustomerID', 'SKUID'], 'AvgOrderValue_SKU_by_Customer'),
            ('avg_order_category', ['CustomerID', 'Category'], 'AvgOrderValue_Category_by_Customer'),
            ('customer_general', ['CustomerID'], None),
            ('sku_state_sales', ['SKUID', 'State'], 'SKU_TotalSales_State'),
            ('sku_state_count', ['SKUID', 'State'], 'SKU_TotalPurchaseCount_State'),
            ('category_state_sales', ['Category', 'State'], 'Category_TotalSales_State'),
            ('segment_state_sales', ['Segment', 'State'], 'Segment_TotalSales_State'),
            ('manufacturer_state_sales', ['Manufacturer', 'State'], 'Manufacturer_TotalSales_State'),
        ]
        
        # Add rolling window features to merge configs
        for window in self.rolling_windows:
            merge_configs.extend([
                (f'customer_rolling_{window}w', ['CustomerID'], None),
                (f'customer_category_rolling_{window}w', ['CustomerID', 'Category'], None),
                (f'sku_rolling_sales_{window}w', ['SKUID'], None),
                (f'sku_rolling_count_{window}w', ['SKUID'], None),
                (f'category_rolling_sales_{window}w', ['Category'], None),
                (f'segment_rolling_sales_{window}w', ['Segment'], None),
                (f'town_category_{window}w', ['Town', 'Category'], None),
                (f'town_segment_{window}w', ['Town', 'Segment'], None),
                (f'town_sku_{window}w', ['Town', 'SKUID'], None),
                (f'town_manufacturer_{window}w', ['Town', 'Manufacturer'], None),
            ])
        
        # Perform merges
        for feature_name, merge_keys, cold_start_col in merge_configs:
            if feature_name in features_dict:
                feature_df = feature_df.merge(features_dict[feature_name], on=merge_keys, how='left')
                
                if cold_start_col:
                    feature_df[cold_start_col] = feature_df[cold_start_col].fillna(self.cold_start_value)
        
        # Fill remaining NaN values with 0
        feature_df = feature_df.fillna(0)
        
        # Add prediction week as a feature
        feature_df['prediction_week'] = self.prediction_week
        
        return feature_df


# Example usage
def main():
    """Example of how to use the integrated pipeline"""
    
    # Load data
    transactions = pd.read_csv("data/raw/transactions.csv")
    sku_metadata = pd.read_csv("data/raw/sku_metadata.csv") 
    customer_metadata = pd.read_csv("data/raw/customer_metadata.csv")
    
    # Initialize integrated feature engineer
    fe = IntegratedFeatureEngineer(cold_start_value=9999, random_state=42)
    
    # Create complete training dataset
    training_data, feature_columns = fe.create_training_dataset(
        transactions=transactions,
        sku_metadata=sku_metadata,
        customer_metadata=customer_metadata,
        prediction_week="2024-01-01",
        max_negatives=15,  # Your adaptive sampling parameters
        category_weight=0.5,
        trending_weight=0.3
    )
    
    print(f"\nFeature columns: {feature_columns[:10]}...")  # Show first 10
    print(f"Training data shape: {training_data.shape}")
    
    return training_data, feature_columns

if __name__ == "__main__":
    training_data, feature_columns = main()