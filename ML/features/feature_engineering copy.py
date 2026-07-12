# features/feature_engineering.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings 
from datetime import datetime, timedelta
from typing import List, Optional, Union
warnings.filterwarnings('ignore')


# AdvancedNegativeSampler class for hierarchical localized popularity and hard negative mining
class AdvancedNegativeSampler:
    """Advanced negative sampling with hierarchical popularity and hard negative mining"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    
    def generate_negative_samples(self, transactions, sku_metadata, customer_metadata,
                                max_negatives=20, category_weight=0.5, trending_weight=0.3,
                                fallback_levels=["Town", "City", "State", "Global"],
                                use_trending=True, 
                                use_category_relevance=True):
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


# SKUPurchaseFeatureEngineer class for feature engineering
class SKUPurchaseFeatureEngineer:
    """Core class to engineer features for SKU purchase likelihood prediction"""

    def __init__(self, cold_start_value=9999, rolling_windows = [4, 8, 12]):
        self.cold_start_value = cold_start_value
        self.rolling_windows = rolling_windows

    def engineer_features(self, transactions, sku_metadata, customer_metadata,
                          prediction_week=None, target_customers=None):
        """Main entry point to generate comprehensive features"""
        print("Generating comprehensive features...")
        transactions = self._prepare_data(transactions, sku_metadata, customer_metadata, prediction_week)
        if target_customers is not None:
            transactions = transactions[transactions['CustomerID'].isin(target_customers)]

        features_dict = {}
        features_dict.update(self._customer_sku_interaction_features(transactions))
        features_dict.update(self._customer_general_features(transactions, customer_metadata))
        features_dict.update(self._localized_sku_features(transactions))
        features_dict.update(self._time_series_features(transactions))
        features_dict.update(self._granular_location_features(transactions))

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
        # last_purchase_sku.drop(columns='Week', inplace=True, errors='ignore')
        
        last_purchase_category = transactions.groupby(['CustomerID', 'Category'])['Week'].max().reset_index()
        last_purchase_category['DaysSinceLastPurchase_Category'] = (self.prediction_week - last_purchase_category['Week']).dt.days
        # last_purchase_category.drop(columns='Week', inplace=True, errors='ignore')
        
        last_purchase_segment = transactions.groupby(['CustomerID', 'Segment'])['Week'].max().reset_index()
        last_purchase_segment['DaysSinceLastPurchase_Segment'] = (self.prediction_week - last_purchase_segment['Week']).dt.days
        # last_purchase_segment.drop(columns='Week', inplace=True, errors='ignore')
        
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
    
    def _combine_features_dep2(self, features_dict, transactions):
        """Combine all features into final feature matrix with Week context"""
        # Start with unique (Week, CustomerID, SKUID) combinations
        unique_combinations = transactions[['Week', 'CustomerID', 'SKUID', 'Category', 'Segment', 
                                        'Manufacturer', 'State', 'City', 'Town']].drop_duplicates()
        
        feature_df = unique_combinations.copy()
        
        # Define merge configurations with Week where relevant
        merge_configs = [
            # Customer-SKU interactions (Week-specific)
            ('last_purchase_sku', ['Week', 'CustomerID', 'SKUID'], 'DaysSinceLastPurchase_SKU'),
            ('total_purchases_sku', ['CustomerID', 'SKUID'], 'TotalPurchases_SKU'),
            
            # Customer general features (no Week needed)
            ('customer_general', ['CustomerID'], None),
            
            # Localized features (Week + location)
            ('sku_state_sales', ['SKUID', 'State'], 'SKU_TotalSales_State'),
            
            # Rolling window features (already computed relative to prediction_week)
            *[(
                f'customer_rolling_{window}w', 
                ['CustomerID'], 
                None
            ) for window in self.rolling_windows]
        ]
        
        # Perform merges with proper handling
        for feature_name, merge_keys, cold_start_col in merge_configs:
            if feature_name in features_dict:
                # Ensure merge keys exist in feature_df
                feature_df = feature_df.merge(
                    features_dict[feature_name],
                    on=merge_keys,
                    how='left'
                )
                # Apply cold start value if specified
                if cold_start_col:
                    feature_df[cold_start_col] = feature_df[cold_start_col].fillna(self.cold_start_value)
        
        # Fill remaining NaNs
        # feature_df = feature_df.fillna(0)
        
        return feature_df

    def _combine_features_dep(self, features_dict, transactions):
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
                if 'Week' in features_dict[feature_name].columns:
                    print(feature_name, list(feature_df.columns))
                    # feature_df = feature_df.drop(columns=['Week'])
                    # print("Dropped 'Week' column from feature_df to avoid merge issues.")
                    
                feature_df = feature_df.merge(features_dict[feature_name], on=merge_keys, how='left')
                
                if cold_start_col:
                    feature_df[cold_start_col] = feature_df[cold_start_col].fillna(self.cold_start_value)
        
        # Fill remaining NaN values with 0
        # feature_df = feature_df.fillna(0)
        
        # Add prediction week as a feature
        feature_df['prediction_week'] = self.prediction_week
        
        return feature_df
    
    def _combine_features(self, features_dict, transactions):
        """Combine all features into final feature matrix with proper metadata handling"""
        # Start with unique combinations and ensure we have all metadata
        unique_combinations = transactions[['Week', 'CustomerID', 'SKUID']].drop_duplicates()
        
        # Add metadata columns properly
        sku_metadata_cols = ['SKUID', 'Category', 'Segment', 'Manufacturer']
        customer_metadata_cols = ['CustomerID', 'State', 'City', 'Town']
        
        # Get unique metadata for SKUs and customers
        sku_info = transactions[sku_metadata_cols].drop_duplicates()
        customer_info = transactions[customer_metadata_cols].drop_duplicates()
        
        # Merge metadata
        feature_df = unique_combinations.merge(sku_info, on='SKUID', how='left')
        feature_df = feature_df.merge(customer_info, on='CustomerID', how='left')
        
        # Define merge configurations
        merge_configs = [
            # Customer-SKU interactions 
            ('last_purchase_sku', ['CustomerID', 'SKUID'], 'DaysSinceLastPurchase_SKU'),
            ('total_purchases_sku', ['CustomerID', 'SKUID'], 'TotalPurchases_SKU'),
            ('avg_order_sku', ['CustomerID', 'SKUID'], 'AvgOrderValue_SKU_by_Customer'),
            
            # Customer-Category/Segment interactions
            ('last_purchase_category', ['CustomerID', 'Category'], 'DaysSinceLastPurchase_Category'),
            ('last_purchase_segment', ['CustomerID', 'Segment'], 'DaysSinceLastPurchase_Segment'),
            ('total_purchases_category', ['CustomerID', 'Category'], 'TotalPurchases_Category'),
            ('total_purchases_segment', ['CustomerID', 'Segment'], 'TotalPurchases_Segment'),
            ('avg_order_category', ['CustomerID', 'Category'], 'AvgOrderValue_Category_by_Customer'),
            
            # Customer general features
            ('customer_general', ['CustomerID'], None),
            
            # Localized features
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
                # Clean the feature dataframe
                feature_data = features_dict[feature_name].copy()
                if 'Week' in feature_data.columns and 'Week' not in merge_keys:
                    feature_data = feature_data.drop(columns=['Week'])
                
                # Merge
                feature_df = feature_df.merge(feature_data, on=merge_keys, how='left')
                
                # Apply cold start value if specified
                if cold_start_col and cold_start_col in feature_df.columns:
                    feature_df[cold_start_col] = feature_df[cold_start_col].fillna(self.cold_start_value)
        
        # Add prediction week as a feature
        feature_df['prediction_week'] = self.prediction_week
        
        return feature_df

    

# SKUPurchaseFeatureEngineer class for feature engineering
class SKUPurchaseFeatureEngineer_:
    """Core class to engineer features for SKU purchase likelihood prediction"""

    def __init__(self, cold_start_value=9999, rolling_windows = [4, 8, 12]):
        self.cold_start_value = cold_start_value
        self.rolling_windows = rolling_windows

    def engineer_features(self, transactions, sku_metadata, customer_metadata,
                          prediction_week=None, target_customers=None):
        """Main entry point to generate comprehensive features"""
        print("Generating comprehensive features...")
        transactions = self._prepare_data(transactions, sku_metadata, customer_metadata, prediction_week)
        if target_customers is not None:
            transactions = transactions[transactions['CustomerID'].isin(target_customers)]

        features_dict = {}
        features_dict.update(self._customer_sku_interaction_features(transactions))
        features_dict.update(self._customer_general_features(transactions, customer_metadata))
        features_dict.update(self._localized_sku_features(transactions))
        features_dict.update(self._time_series_features(transactions))
        features_dict.update(self._granular_location_features(transactions))

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
        # last_purchase_sku.drop(columns='Week', inplace=True, errors='ignore')
        
        last_purchase_category = transactions.groupby(['CustomerID', 'Category'])['Week'].max().reset_index()
        last_purchase_category['DaysSinceLastPurchase_Category'] = (self.prediction_week - last_purchase_category['Week']).dt.days
        # last_purchase_category.drop(columns='Week', inplace=True, errors='ignore')
        
        last_purchase_segment = transactions.groupby(['CustomerID', 'Segment'])['Week'].max().reset_index()
        last_purchase_segment['DaysSinceLastPurchase_Segment'] = (self.prediction_week - last_purchase_segment['Week']).dt.days
        # last_purchase_segment.drop(columns='Week', inplace=True, errors='ignore')
        
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
        """Combine all features into final feature matrix with Week context"""
        # Start with unique (Week, CustomerID, SKUID) combinations
        unique_combinations = transactions[['Week', 'CustomerID', 'SKUID', 'Category', 'Segment', 
                                        'Manufacturer', 'State', 'City', 'Town']].drop_duplicates()
        
        feature_df = unique_combinations.copy()
        
        # Define merge configurations with Week where relevant
        merge_configs = [
            # Customer-SKU interactions (Week-specific)
            ('last_purchase_sku', ['Week', 'CustomerID', 'SKUID'], 'DaysSinceLastPurchase_SKU'),
            ('total_purchases_sku', ['CustomerID', 'SKUID'], 'TotalPurchases_SKU'),
            
            # Customer general features (no Week needed)
            ('customer_general', ['CustomerID'], None),
            
            # Localized features (Week + location)
            ('sku_state_sales', ['SKUID', 'State'], 'SKU_TotalSales_State'),
            
            # Rolling window features (already computed relative to prediction_week)
            *[(
                f'customer_rolling_{window}w', 
                ['CustomerID'], 
                None
            ) for window in self.rolling_windows]
        ]
        
        # Perform merges with proper handling
        for feature_name, merge_keys, cold_start_col in merge_configs:
            if feature_name in features_dict:
                # Ensure merge keys exist in feature_df
                feature_df = feature_df.merge(
                    features_dict[feature_name],
                    on=merge_keys,
                    how='left'
                )
                # Apply cold start value if specified
                if cold_start_col:
                    feature_df[cold_start_col] = feature_df[cold_start_col].fillna(self.cold_start_value)
        
        # Fill remaining NaNs
        # feature_df = feature_df.fillna(0)
        
        return feature_df

    def _combine_features_dep(self, features_dict, transactions):
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
                if 'Week' in features_dict[feature_name].columns:
                    print(feature_name, list(feature_df.columns))
                    # feature_df = feature_df.drop(columns=['Week'])
                    # print("Dropped 'Week' column from feature_df to avoid merge issues.")
                    
                feature_df = feature_df.merge(features_dict[feature_name], on=merge_keys, how='left')
                
                if cold_start_col:
                    feature_df[cold_start_col] = feature_df[cold_start_col].fillna(self.cold_start_value)
        
        # Fill remaining NaN values with 0
        # feature_df = feature_df.fillna(0)
        
        # Add prediction week as a feature
        feature_df['prediction_week'] = self.prediction_week
        
        return feature_df
    

# Enhanced TrainingDataBuilder with proper missing value handling
class TrainingDataBuilder:
    """Builds a training dataset with positive and negative samples"""

    def __init__(self, feature_engineer, negative_sampler):
        self.feature_engineer = feature_engineer
        self.negative_sampler = negative_sampler

    def build_training_dataset(self, transactions, sku_metadata, customer_metadata,
                               prediction_week=None, target_customers=None,
                               max_negatives=20, category_weight=0.5, trending_weight=0.3):
        """Generate training data with label and features"""
        feature_df = self.feature_engineer.engineer_features(
            transactions, sku_metadata, customer_metadata, prediction_week, target_customers
        )

        # Positive samples from historical data
        historical_transactions = transactions.copy()
        if prediction_week:
            historical_transactions = historical_transactions[
                pd.to_datetime(historical_transactions['Week']) < pd.to_datetime(prediction_week)
            ]
        positive_samples = historical_transactions.groupby(['Week', 'CustomerID', 'SKUID']).size().reset_index(name='count')
        positive_samples['label'] = 1
        positive_samples = positive_samples[['Week', 'CustomerID', 'SKUID', 'label']]

        # Generate negative samples
        negative_samples = self.negative_sampler.generate_negative_samples(
            historical_transactions, sku_metadata, customer_metadata,
            max_negatives=max_negatives, category_weight=category_weight, trending_weight=trending_weight
        )

        # Combine and merge 
        training_samples = pd.concat([positive_samples, negative_samples], ignore_index=True) 
        feature_df = feature_df.drop(columns=['Week'], errors='ignore')  # drop Week if present
        training_dataset = training_samples.merge(feature_df, on=['CustomerID', 'SKUID'], how='left')

        # Fill missing values properly
        training_dataset = self._fill_missing_values(training_dataset, sku_metadata, customer_metadata)

        # Get feature columns (excluding identifiers and target)
        feature_columns = [col for col in training_dataset.columns
                           if col not in ['Week', 'CustomerID', 'SKUID', 'label', 'prediction_week']]

        return training_dataset, feature_columns

    def _fill_missing_values(self, training_dataset, sku_metadata, customer_metadata):
        """Comprehensive missing value filling strategy"""
        print("Filling missing values...")
        
        # 1. Fill SKU-related metadata missing values
        training_dataset = self._fill_sku_metadata(training_dataset, sku_metadata)
        
        # 2. Fill Customer-related metadata missing values  
        training_dataset = self._fill_customer_metadata(training_dataset, customer_metadata)
        
        # 3. Fill interaction-based missing values
        training_dataset = self._fill_interaction_features(training_dataset)
        
        # 4. Fill rolling window missing values
        training_dataset = self._fill_rolling_window_features(training_dataset)
        
        # 5. Fill location-based missing values
        training_dataset = self._fill_location_features(training_dataset)
        
        # 6. Final cleanup - fill any remaining NaNs with appropriate defaults
        training_dataset = self._final_cleanup(training_dataset)
        
        return training_dataset

    def _fill_sku_metadata(self, training_dataset, sku_metadata):
        """Fill missing SKU metadata (Category, Segment, Manufacturer)"""
        # Merge SKU metadata to fill missing values
        sku_cols = ['Category', 'Segment', 'Manufacturer']
        
        for col in sku_cols:
            if col in training_dataset.columns:
                # Create a mapping from SKUID to metadata
                sku_mapping = sku_metadata.set_index('SKUID')[col].to_dict()
                
                # Fill missing values
                mask = training_dataset[col].isna()
                training_dataset.loc[mask, col] = training_dataset.loc[mask, 'SKUID'].map(sku_mapping)
                
                # For any still missing, use 'Unknown'
                training_dataset[col] = training_dataset[col].fillna('Unknown')
        
        return training_dataset

    def _fill_customer_metadata(self, training_dataset, customer_metadata):
        """Fill missing customer metadata (State, City, Town, RFM values)"""
        # Merge customer metadata to fill missing values
        customer_cols = ['State', 'City', 'Town', 'Customer_Recency', 'Customer_Frequency', 'Customer_Monetary']
        
        # Create customer mapping
        customer_mapping = customer_metadata.set_index('CustomerID').to_dict('index')
        
        for col in customer_cols:
            if col in training_dataset.columns:
                # Map the base column name to metadata column name
                metadata_col = col.replace('Customer_', '') if col.startswith('Customer_') else col
                
                if metadata_col in customer_metadata.columns:
                    mask = training_dataset[col].isna()
                    training_dataset.loc[mask, col] = training_dataset.loc[mask, 'CustomerID'].apply(
                        lambda x: customer_mapping.get(x, {}).get(metadata_col, None)
                    )
        
        # Fill remaining location missing values with 'Unknown'
        location_cols = ['State', 'City', 'Town']
        for col in location_cols:
            if col in training_dataset.columns:
                training_dataset[col] = training_dataset[col].fillna('Unknown')
        
        # Fill RFM missing values with defaults (cold start customers)
        rfm_defaults = {
            'Customer_Recency': 365,  # Assume 1 year for new customers
            'Customer_Frequency': 1,   # Minimum frequency
            'Customer_Monetary': 0     # No prior spending
        }
        
        for col, default_val in rfm_defaults.items():
            if col in training_dataset.columns:
                training_dataset[col] = training_dataset[col].fillna(default_val)
        
        return training_dataset

    def _fill_interaction_features(self, training_dataset):
        """Fill missing customer-SKU interaction features"""
        # Days since last purchase - use large value for never purchased
        cold_start_value = 9999
        
        interaction_cols = [
            'DaysSinceLastPurchase_SKU',
            'DaysSinceLastPurchase_Category', 
            'DaysSinceLastPurchase_Segment'
        ]
        
        for col in interaction_cols:
            if col in training_dataset.columns:
                training_dataset[col] = training_dataset[col].fillna(cold_start_value)
        
        # Total purchases - use 0 for never purchased
        purchase_count_cols = [
            'TotalPurchases_SKU',
            'TotalPurchases_Category',
            'TotalPurchases_Segment'
        ]
        
        for col in purchase_count_cols:
            if col in training_dataset.columns:
                training_dataset[col] = training_dataset[col].fillna(0)
        
        # Average order values - use global averages
        avg_order_cols = [
            'AvgOrderValue_SKU_by_Customer',
            'AvgOrderValue_Category_by_Customer'
        ]
        
        for col in avg_order_cols:
            if col in training_dataset.columns:
                global_avg = training_dataset[col].mean()
                training_dataset[col] = training_dataset[col].fillna(global_avg)
        
        return training_dataset

    def _fill_rolling_window_features(self, training_dataset):
        """Fill missing rolling window features (customers with no recent activity)"""
        # Rolling purchase counts - use 0 for no activity
        rolling_count_patterns = [
            'Customer_RollingPurchaseCount_',
            'Customer_RollingUniqueSKUsPurchased_',
            'SKU_RollingPurchaseCount_',
            'Category_RollingTotalSales_',
            'Segment_RollingTotalSales_',
            'Town_Category_PurchaseCount_',
            'Town_Segment_PurchaseCount_',
            'Town_SKU_PurchaseCount_',
            'Town_Manufacturer_PurchaseCount_'
        ]
        
        for pattern in rolling_count_patterns:
            matching_cols = [col for col in training_dataset.columns if pattern in col]
            for col in matching_cols:
                training_dataset[col] = training_dataset[col].fillna(0)
        
        # Rolling averages - use global averages or 0
        rolling_avg_patterns = [
            'Customer_RollingAvgOrderValue_',
            'SKU_RollingTotalSales_'
        ]
        
        for pattern in rolling_avg_patterns:
            matching_cols = [col for col in training_dataset.columns if pattern in col]
            for col in matching_cols:
                if 'AvgOrderValue' in col:
                    # Use global average for order values
                    global_avg = training_dataset[col].mean()
                    if pd.isna(global_avg):
                        global_avg = 0
                    training_dataset[col] = training_dataset[col].fillna(global_avg)
                else:
                    # Use 0 for sales totals
                    training_dataset[col] = training_dataset[col].fillna(0)
        
        return training_dataset

    def _fill_location_features(self, training_dataset):
        """Fill missing location-based features"""
        location_sales_patterns = [
            'SKU_TotalSales_State',
            'SKU_TotalPurchaseCount_State',
            'Category_TotalSales_State',
            'Segment_TotalSales_State',
            'Manufacturer_TotalSales_State'
        ]
        
        for pattern in location_sales_patterns:
            if pattern in training_dataset.columns:
                training_dataset[pattern] = training_dataset[pattern].fillna(0)
        
        # Customer overall features
        overall_patterns = [
            'Customer_TotalUniqueSKUsPurchased_Overall',
            'Customer_AvgOrderValue_Overall'
        ]
        
        for pattern in overall_patterns:
            if pattern in training_dataset.columns:
                if 'AvgOrderValue' in pattern:
                    global_avg = training_dataset[pattern].mean()
                    if pd.isna(global_avg):
                        global_avg = 0
                    training_dataset[pattern] = training_dataset[pattern].fillna(global_avg)
                else:
                    training_dataset[pattern] = training_dataset[pattern].fillna(0)
        
        return training_dataset

    def _final_cleanup(self, training_dataset):
        """Final cleanup of any remaining missing values"""
        # Get numeric columns
        numeric_columns = training_dataset.select_dtypes(include=[np.number]).columns
        
        # Fill remaining numeric NaNs with 0
        training_dataset[numeric_columns] = training_dataset[numeric_columns].fillna(0)
        
        # Get categorical columns  
        categorical_columns = training_dataset.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns 
                              if col not in ['CustomerID', 'SKUID', 'Week']]
        
        # Fill remaining categorical NaNs with 'Unknown'
        training_dataset[categorical_columns] = training_dataset[categorical_columns].fillna('Unknown')
        
        return training_dataset



class TrainingDataBuilder_:
    """Builds a training dataset with positive and negative samples"""

    def __init__(self, feature_engineer, negative_sampler):
        self.feature_engineer = feature_engineer
        self.negative_sampler = negative_sampler

    def build_training_dataset(self, transactions, sku_metadata, customer_metadata,
                               prediction_week=None, target_customers=None,
                               max_negatives=20, category_weight=0.5, trending_weight=0.3):
        """Generate training data with label and features"""
        feature_df = self.feature_engineer.engineer_features(
            transactions, sku_metadata, customer_metadata, prediction_week, target_customers
        )

        # Positive samples from historical data
        historical_transactions = transactions.copy()
        if prediction_week:
            historical_transactions = historical_transactions[
                pd.to_datetime(historical_transactions['Week']) < pd.to_datetime(prediction_week)
            ]
        positive_samples = historical_transactions.groupby(['Week', 'CustomerID', 'SKUID']).size().reset_index(name='count')
        positive_samples['label'] = 1
        positive_samples = positive_samples[['Week', 'CustomerID', 'SKUID', 'label']]

        # Generate negative samples
        negative_samples = self.negative_sampler.generate_negative_samples(
            historical_transactions, sku_metadata, customer_metadata,
            max_negatives=max_negatives, category_weight=category_weight, trending_weight=trending_weight
        )

        # Combine and merge 
        training_samples = pd.concat([positive_samples, negative_samples], ignore_index=True) 
        # Modification:::
        # print("Columns in training_samples:", list(training_samples.columns))
        # print("Columns in feature_df:", list(feature_df.columns))
        feature_df = feature_df.drop(columns=['Week'], errors='ignore')  # drop Week if present
        training_dataset = training_samples.merge(feature_df, on=['CustomerID', 'SKUID'], how='left')

        # Fill missing values
        feature_columns = [col for col in training_dataset.columns
                           if col not in ['Week', 'CustomerID', 'SKUID', 'label', 'prediction_week']]
        training_dataset[feature_columns] = training_dataset[feature_columns]#.fillna(0)

        return training_dataset, feature_columns


class PurchasePredictor_:
    """Class to generate features for prediction/inference"""

    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer

    def predict_for_customers(self, transactions, sku_metadata, customer_metadata,
                              target_customers, candidate_skus=None, prediction_week=None):
        """
        Generate features for inference (no labels)
        """
        if prediction_week is None:
            prediction_week = pd.to_datetime("today")

        # Filter transactions up to prediction week
        historical_transactions = transactions[
            pd.to_datetime(transactions["Week"]) < pd.to_datetime(prediction_week)
        ]

        # Engineer features using historical data
        feature_df = self.feature_engineer.engineer_features(
            historical_transactions, sku_metadata, customer_metadata,
            prediction_week=prediction_week, target_customers=target_customers
        )

        # Get unique customers and towns
        customer_info = customer_metadata[customer_metadata["CustomerID"].isin(target_customers)]
        town_to_customers = customer_info.groupby("Town")["CustomerID"].unique()

        # Create all possible (CustomerID, SKUID) pairs
        if candidate_skus is None:
            candidate_skus = sku_metadata["SKUID"].unique()

        prediction_pairs = []
        for _, row in customer_info.iterrows():
            customer_id = row["CustomerID"]
            for sku in candidate_skus:
                prediction_pairs.append({
                    "CustomerID": customer_id,
                    "SKUID": sku
                })

       
        prediction_df = pd.DataFrame(prediction_pairs)
        # ------------------------------------
        # # Checks
        # print("prediction_df dtypes:")
        # print(prediction_df[["CustomerID", "SKUID"]].dtypes)

        # print("\nfeature_df dtypes:")
        # print(feature_df[["CustomerID", "SKUID"]].dtypes)
        # ------------------------------------
        # Convert merge keys to string (safe for IDs that may be alphanumeric)
        prediction_df[["CustomerID", "SKUID"]] = prediction_df[["CustomerID", "SKUID"]].astype(str)
        feature_df[["CustomerID", "SKUID"]] = feature_df[["CustomerID", "SKUID"]].astype(str)

        # Now merge safely        
        feature_df = feature_df.drop(columns=['Week'], errors='ignore')  # drop Week if present
        prediction_data = prediction_df.merge(feature_df, on=["CustomerID", "SKUID"], how="left")
        
        # prediction_data = prediction_df.merge(feature_df, on=["CustomerID", "SKUID"], how="left")
        prediction_data["Week"] = prediction_week
        prediction_data = prediction_data#.fillna(0)

        return prediction_data


# features/inference_data_generator.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class InferenceDataGenerator:
    """Generates inference dataset for SKU purchase likelihood prediction"""
    
    def __init__(self, feature_engineer, min_customer_purchases=5, min_sku_popularity=10):
        self.feature_engineer = feature_engineer
        self.min_customer_purchases = min_customer_purchases
        self.min_sku_popularity = min_sku_popularity
    
    def generate_inference_dataset(self, transactions, sku_metadata, customer_metadata,
                                 prediction_week=None, max_skus_per_customer=100,
                                 include_cold_start=True, popularity_threshold=0.8):
        """
        Generate inference dataset for next week predictions
        
        Args:
            transactions: Historical transaction data
            sku_metadata: SKU metadata 
            customer_metadata: Customer metadata
            prediction_week: Week to predict for (default: next week after max transaction date)
            max_skus_per_customer: Maximum SKUs to predict per customer
            include_cold_start: Whether to include SKUs never purchased by customer
            popularity_threshold: Percentile threshold for popular SKUs
        
        Returns:
            inference_df: DataFrame with features for (CustomerID, SKUID) pairs
            feature_columns: List of feature column names
        """
        
        print("Generating inference dataset...")
        
        # Set prediction week
        if prediction_week is None:
            max_transaction_week = pd.to_datetime(transactions['Week']).max()
            prediction_week = max_transaction_week + pd.Timedelta(weeks=1)
        
        prediction_week = pd.to_datetime(prediction_week)
        print(f"Predicting for week: {prediction_week}")
        
        # Filter active customers and popular SKUs
        active_customers = self._get_active_customers(transactions)
        curated_skus = self._curate_skus(transactions, sku_metadata, popularity_threshold)
        
        print(f"Active customers: {len(active_customers)}")
        print(f"Curated SKUs: {len(curated_skus)}")
        
        # Generate customer-SKU combinations
        inference_combinations = self._generate_customer_sku_combinations(
            transactions, sku_metadata, active_customers, curated_skus, 
            max_skus_per_customer, include_cold_start
        )
        
        print(f"Total inference combinations: {len(inference_combinations)}")
        
        # Create temporary transactions for feature engineering
        temp_transactions = self._create_temp_transactions(
            inference_combinations, prediction_week, sku_metadata, customer_metadata
        )
        
        # Generate features using existing feature engineer
        inference_df = self.feature_engineer.engineer_features(
            transactions, sku_metadata, customer_metadata,
            prediction_week=prediction_week, 
            target_customers=active_customers
        )
        
        # Filter to only inference combinations
        inference_df = inference_df.merge(
            inference_combinations[['CustomerID', 'SKUID']],
            on=['CustomerID', 'SKUID'],
            how='inner'
        )
        
        # Add prediction week and remove any Week column from features
        inference_df['prediction_week'] = prediction_week
        if 'Week' in inference_df.columns:
            inference_df = inference_df.drop(columns=['Week'])
        
        # Get feature columns (exclude identifiers)
        feature_columns = [col for col in inference_df.columns 
                          if col not in ['CustomerID', 'SKUID', 'prediction_week']]
        
        print(f"Generated {len(inference_df)} inference samples with {len(feature_columns)} features")
        
        return inference_df, feature_columns
    
    def _get_active_customers(self, transactions):
        """Get customers with sufficient purchase history"""
        customer_purchase_counts = transactions.groupby('CustomerID').size()
        active_customers = customer_purchase_counts[
            customer_purchase_counts >= self.min_customer_purchases
        ].index.tolist()
        
        return active_customers
    
    def _curate_skus(self, transactions, sku_metadata, popularity_threshold):
        """Curate SKUs for prediction using multiple strategies"""
        # First merge transactions with metadata to get category/segment info
        trans_with_metadata = transactions.merge(sku_metadata, on='SKUID', how='left')
        
        all_skus = set(sku_metadata['SKUID'].unique())
        curated_skus = set()
        
        # Strategy 1: Popular SKUs (by transaction frequency)
        sku_popularity = transactions['SKUID'].value_counts()
        popular_threshold = sku_popularity.quantile(popularity_threshold)
        popular_skus = sku_popularity[sku_popularity >= popular_threshold].index.tolist()
        curated_skus.update(popular_skus)
        
        # Strategy 2: Recently trending SKUs (last 8 weeks)
        max_week = pd.to_datetime(transactions['Week']).max()
        recent_cutoff = max_week - pd.Timedelta(weeks=8)
        recent_transactions = transactions[pd.to_datetime(transactions['Week']) >= recent_cutoff]
        
        if len(recent_transactions) > 0:
            recent_popular = recent_transactions['SKUID'].value_counts()
            recent_threshold = max(recent_popular.quantile(0.7), self.min_sku_popularity)
            trending_skus = recent_popular[recent_popular >= recent_threshold].index.tolist()
            curated_skus.update(trending_skus)
        
        # Strategy 3: Category and segment leaders (using merged data)
        for group_col in ['Category', 'Segment', 'Manufacturer']:
            if group_col in trans_with_metadata.columns:
                group_leaders = trans_with_metadata.groupby([group_col, 'SKUID']).size().reset_index(name='count')
                group_leaders = group_leaders.loc[group_leaders.groupby(group_col)['count'].idxmax()]
                curated_skus.update(group_leaders['SKUID'].tolist())
        
        # Strategy 4: Ensure minimum coverage per category/segment
        for group_col in ['Category', 'Segment']:
            if group_col in sku_metadata.columns:
                for group_val in sku_metadata[group_col].unique():
                    group_skus = sku_metadata[sku_metadata[group_col] == group_val]['SKUID']
                    group_transactions = transactions[transactions['SKUID'].isin(group_skus)]
                    
                    if len(group_transactions) > 0:
                        group_popular = group_transactions['SKUID'].value_counts().head(3)
                        curated_skus.update(group_popular.index.tolist())
        
        # Ensure we have at least some coverage
        if len(curated_skus) < 100:
            top_overall = transactions['SKUID'].value_counts().head(200).index.tolist()
            curated_skus.update(top_overall)
        
        return list(curated_skus)
    
    def _generate_customer_sku_combinations(self, transactions, sku_metadata, active_customers, curated_skus,
                                          max_skus_per_customer, include_cold_start):
        """Generate smart customer-SKU combinations for inference"""
        # Properly merge transactions with sku_metadata to get category/segment info
        trans_with_metadata = transactions.merge(sku_metadata, on='SKUID', how='left')
        
        combinations = []
        
        for customer_id in active_customers:
            customer_transactions = trans_with_metadata[trans_with_metadata['CustomerID'] == customer_id]
            customer_skus = set(customer_transactions['SKUID'].unique())
            
            # Get customer categories/segments from merged data
            customer_categories = set()
            customer_segments = set()
            customer_manufacturers = set()
            
            if 'Category' in customer_transactions.columns:
                customer_categories = set(customer_transactions['Category'].dropna().unique())
            if 'Segment' in customer_transactions.columns:
                customer_segments = set(customer_transactions['Segment'].dropna().unique())
            if 'Manufacturer' in customer_transactions.columns:
                customer_manufacturers = set(customer_transactions['Manufacturer'].dropna().unique())
            
            candidate_skus = set()
            
            # Strategy 1: SKUs from customer's preferred categories/segments
            if customer_categories and 'Category' in sku_metadata.columns:
                for category in customer_categories:
                    category_skus = sku_metadata[sku_metadata['Category'] == category]['SKUID'].unique()
                    popular_in_category = transactions[transactions['SKUID'].isin(category_skus)]['SKUID'].value_counts().head(10)
                    candidate_skus.update(popular_in_category.index.tolist())
            
            if customer_segments and 'Segment' in sku_metadata.columns:
                for segment in customer_segments:
                    segment_skus = sku_metadata[sku_metadata['Segment'] == segment]['SKUID'].unique()
                    popular_in_segment = transactions[transactions['SKUID'].isin(segment_skus)]['SKUID'].value_counts().head(10)
                    candidate_skus.update(popular_in_segment.index.tolist())
            
            if customer_manufacturers and 'Manufacturer' in sku_metadata.columns:
                for manufacturer in customer_manufacturers:
                    manufacturer_skus = sku_metadata[sku_metadata['Manufacturer'] == manufacturer]['SKUID'].unique()
                    popular_in_manufacturer = transactions[transactions['SKUID'].isin(manufacturer_skus)]['SKUID'].value_counts().head(5)
                    candidate_skus.update(popular_in_manufacturer.index.tolist())
            
            # Strategy 2: Popular SKUs in customer's location
            if 'State' in customer_transactions.columns and len(customer_transactions) > 0:
                customer_state = customer_transactions['State'].iloc[0]
                if customer_state and 'State' in transactions.columns:
                    state_transactions = transactions[transactions['State'] == customer_state]
                    state_popular = state_transactions['SKUID'].value_counts().head(20)
                    candidate_skus.update(state_popular.index.tolist())
            
            # Strategy 3: Include some cold-start SKUs if enabled
            if include_cold_start:
                cold_start_skus = set(curated_skus) - customer_skus
                # Add some random cold-start SKUs for exploration
                cold_start_list = list(cold_start_skus)
                np.random.seed(42)
                if len(cold_start_list) > 0:
                    n_random = min(15, len(cold_start_list))
                    candidate_skus.update(np.random.choice(cold_start_list, n_random, replace=False))
            
            # Strategy 4: Add overall popular SKUs for coverage
            overall_popular = transactions['SKUID'].value_counts().head(30).index.tolist()
            candidate_skus.update(overall_popular)
            
            # Limit to curated SKUs and max per customer
            final_skus = list(candidate_skus.intersection(set(curated_skus)))
            
            if len(final_skus) > max_skus_per_customer:
                # Prioritize by global popularity and customer preference
                sku_scores = {}
                for sku in final_skus:
                    # Global popularity score
                    global_count = len(transactions[transactions['SKUID'] == sku])
                    score = np.log1p(global_count)
                    
                    # Boost score if from customer's preferred categories/segments
                    if 'Category' in sku_metadata.columns:
                        sku_category = sku_metadata[sku_metadata['SKUID'] == sku]['Category'].iloc[0] if len(sku_metadata[sku_metadata['SKUID'] == sku]) > 0 else None
                        if sku_category in customer_categories:
                            score *= 1.5
                    
                    if 'Segment' in sku_metadata.columns:
                        sku_segment = sku_metadata[sku_metadata['SKUID'] == sku]['Segment'].iloc[0] if len(sku_metadata[sku_metadata['SKUID'] == sku]) > 0 else None
                        if sku_segment in customer_segments:
                            score *= 1.3
                    
                    sku_scores[sku] = score
                
                # Select top scored SKUs
                sorted_skus = sorted(sku_scores.items(), key=lambda x: x[1], reverse=True)
                final_skus = [sku for sku, _ in sorted_skus[:max_skus_per_customer]]
            
            # Add combinations
            for sku in final_skus:
                combinations.append({
                    'CustomerID': customer_id,
                    'SKUID': sku
                })
        
        return pd.DataFrame(combinations)
    
    def _create_temp_transactions(self, combinations, prediction_week, sku_metadata, customer_metadata):
        """Create temporary transaction-like data for feature engineering"""
        temp_df = combinations.copy()
        temp_df['Week'] = prediction_week
        temp_df['Quantity'] = 1  # Dummy values
        temp_df['OrderValue'] = 100  # Dummy values
        
        # Add metadata
        temp_df = temp_df.merge(sku_metadata, on='SKUID', how='left')
        temp_df = temp_df.merge(customer_metadata, on='CustomerID', how='left')
        
        return temp_df


# Usage example function
def generate_next_week_predictions(transactions, sku_metadata, customer_metadata, 
                                 feature_engineer, prediction_week=None):
    """
    Convenience function to generate inference dataset for next week predictions
    """
    
    # Initialize inference generator
    inference_generator = InferenceDataGenerator(
        feature_engineer=feature_engineer,
        min_customer_purchases=3,  # Minimum purchases to consider customer active
        min_sku_popularity=5       # Minimum popularity for SKU consideration
    )
    
    # Generate inference dataset
    inference_df, feature_columns = inference_generator.generate_inference_dataset(
        transactions=transactions,
        sku_metadata=sku_metadata, 
        customer_metadata=customer_metadata,
        prediction_week=prediction_week,
        max_skus_per_customer=80,   # Max SKUs to predict per customer
        include_cold_start=True,    # Include SKUs never purchased by customer
        popularity_threshold=0.75   # Percentile for popular SKUs
    )
    
    return inference_df, feature_columns

# Example usage
def main():
    # Load data
    transactions = pd.read_csv("data/raw/transactions.csv")
    sku_metadata = pd.read_csv("data/raw/sku_metadata.csv")
    customer_metadata = pd.read_csv("data/raw/customer_metadata.csv")

    # Initialize components
    fe = SKUPurchaseFeatureEngineer(cold_start_value=9999)
    ns = AdvancedNegativeSampler(random_state=42)
    builder = TrainingDataBuilder(fe, ns)

    # Build training data
    training_data, feature_cols = builder.build_training_dataset(
        transactions=transactions,
        sku_metadata=sku_metadata,
        customer_metadata=customer_metadata,
        prediction_week="2024-01-01",
        max_negatives=15,
        category_weight=0.5,
        trending_weight=0.3
    )
    print("Training Data Sample:")
    print(training_data.head())

    # Predict for specific customers
    predictor = PurchasePredictor(fe)
    prediction_data = predictor.predict_for_customers(
        transactions=transactions,
        sku_metadata=sku_metadata,
        customer_metadata=customer_metadata,
        target_customers=[1001, 1002],
        candidate_skus=["SKU001", "SKU002", "SKU003"],
        prediction_week="2024-01-01"
    )
    print("\nPrediction Data Sample:")
    print(prediction_data.head())

    return training_data, prediction_data


if __name__ == "__main__":
    training_data, prediction_data = main()