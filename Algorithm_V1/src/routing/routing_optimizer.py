#!/usr/bin/env python3
"""
Complete Route Optimization Script for Push Sales Recommendations
================================================================

This script demonstrates the complete usage of the route optimization system
for Stock_Point_ID = 1647113, including data loading, processing, and route generation.

Author: Babatunde Adebayo
Date: 2025
"""


import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings 
from folium.plugins import MarkerCluster
warnings.filterwarnings('ignore')
import logging
from ..clustering.divisive_clustering import  OptimizedDivisiveGeographicClustering 
from ..clustering.agglomerative_clustering import AgglomerativeGeographicClustering

# ====================================================================
# STEP 2: MAIN ROUTE OPTIMIZER CLASS
# ====================================================================

class RouteOptimizer:
    def __init__(self, max_customers_per_route=20, max_volume_per_route=200, max_distance_km = 30, logger=logging.getLogger(__name__)):
        self.max_customers_per_route = max_customers_per_route
        self.max_volume_per_route = max_volume_per_route
        self.max_distance_km = max_distance_km
        self.distance_matrix = None
        self.logger = logger
        
    def load_data(self, df_customer_sku_recommendation, df_customer_dim_with_affinity_score, df_stockpoint_dim):
        self.df_sku_rec = df_customer_sku_recommendation.copy()
        self.df_customer_dim = df_customer_dim_with_affinity_score.copy()
        self.df_stockpoint = df_stockpoint_dim.copy()
        
    def filter_customers_for_stockpoint(self, stock_point_id):
        customers = self.df_customer_dim[
            self.df_customer_dim['Stock_Point_ID'] == stock_point_id
        ].copy()
        
        sku_recs = self.df_sku_rec[
            self.df_sku_rec['Stock_Point_ID'] == stock_point_id
        ].copy()
        
        top_sku_customers = sku_recs[
            sku_recs['CustomerSKUscoreRank'] <= 10
        ]['CustomerID'].unique()
        
        customers = customers[customers['CustomerID'].isin(top_sku_customers)]
        
        customer_sku_agg = sku_recs.groupby('CustomerID').agg({
            'EstimatedQuantity': 'sum',
            'CustomerSKUscore': 'mean',
            'CustomerSKUscoreRank': 'min',
            'ProductTag': lambda x: ','.join(x.unique())
        }).reset_index()
        
        customers_final = customers.merge(
            customer_sku_agg, 
            left_on='CustomerID', 
            right_on='CustomerID', 
            how='inner'
        )
        
        customers_final = self._calculate_priority_scores(customers_final)
        return customers_final
    
    def _calculate_priority_scores(self, customers_df):
        customers_df['affinity_score_norm'] = (
            customers_df['composite_customer_score'] - 
            customers_df['composite_customer_score'].min()
        ) / (
            customers_df['composite_customer_score'].max() - 
            customers_df['composite_customer_score'].min()
        )
        
        customers_df['product_priority_boost'] = customers_df['ProductTag'].apply(
            lambda x: 1.2 if 'Express' in str(x) or 'Core' in str(x) else 1.0
        )
        
        customers_df['priority_score'] = (
            0.5 * customers_df['affinity_score_norm'] + 
            0.3 * (1 / (customers_df['percentile_rank'] + 1)) + 
            0.2 * customers_df['product_priority_boost']
        )
        
        return customers_df
    
    def create_geographic_clusters(self, customers_df, clustering_method='divisive'): 

        if clustering_method == 'divisive': 
            divisive_clusterer = OptimizedDivisiveGeographicClustering(
                    max_customers_per_cluster = self.max_customers_per_route,
                    max_distance_km = self.max_distance_km,
                    logger=self.logger
                )
            customers_df =  divisive_clusterer.divisive_clustering(customers_df.copy())
        elif clustering_method == 'agglomerative':
            agglomerative_clusterer = AgglomerativeGeographicClustering(
                    max_customers_per_cluster = self.max_customers_per_route,
                    max_distance_km = self.max_distance_km,
                    logger=self.logger
                )
            customers_df = agglomerative_clusterer.agglomerative_clustering(customers_df.copy())
        
        n_clusters = len(customers_df['cluster'].unique())
        return customers_df, n_clusters
    
    def optimize_route_within_cluster(self, cluster_customers, stock_point_coords):
        if len(cluster_customers) == 0:
            return []
        
        cluster_customers = cluster_customers.sort_values('priority_score', ascending=False)
        
        cumulative_volume = 0
        selected_customers = []
        
        for idx, customer in cluster_customers.iterrows():
            if (len(selected_customers) < self.max_customers_per_route and 
                cumulative_volume + customer['EstimatedQuantity'] <= self.max_volume_per_route):
                selected_customers.append(customer)
                cumulative_volume += customer['EstimatedQuantity']
        
        if not selected_customers:
            return []
        
        # Nearest neighbor routing
        selected_df = pd.DataFrame(selected_customers)
        coords = selected_df[['Latitude', 'Longitude']].values
        
        current_pos = stock_point_coords
        route = []
        remaining_indices = list(range(len(selected_df)))
        
        while remaining_indices:
            distances = [geodesic(current_pos, coords[i]).kilometers for i in remaining_indices]
            nearest_idx = remaining_indices[np.argmin(distances)]
            
            route.append(selected_df.iloc[nearest_idx])
            current_pos = coords[nearest_idx]
            remaining_indices.remove(nearest_idx)
        
        return route
    
    def generate_multi_trip_routes(self, stock_point_id, max_trips=3, clustering_method='divisive'):
        stock_point = self.df_stockpoint[
            self.df_stockpoint['Stock_Point_ID'] == stock_point_id
        ].iloc[0]
        stock_point_coords = (stock_point['Latitude'], stock_point['Longitude'])
        
        customers_df = self.filter_customers_for_stockpoint(stock_point_id)
        
        if len(customers_df) == 0:
            return []
        
        customers_df, n_clusters = self.create_geographic_clusters(customers_df, clustering_method)
        
        all_routes = []
        used_customers = set()
        
        for trip_num in range(1, max_trips + 1):
            if len(used_customers) >= len(customers_df):
                break
                
            available_customers = customers_df[
                ~customers_df['CustomerID'].isin(used_customers)
            ].copy()
            
            if len(available_customers) == 0:
                break
            
            if trip_num == 1:
                # First trip: Best customers across all clusters
                trip_customers = []
                for cluster_id in available_customers['cluster'].unique():
                    cluster_customers = available_customers[
                        available_customers['cluster'] == cluster_id
                    ]
                    route = self.optimize_route_within_cluster(
                        cluster_customers, stock_point_coords
                    )
                    trip_customers.extend(route)
                
                if trip_customers:
                    trip_df = pd.DataFrame(trip_customers)
                    trip_df = trip_df.sort_values('priority_score', ascending=False)
                    
                    cumulative_volume = 0
                    final_customers = []
                    
                    for idx, customer in trip_df.iterrows():
                        if (len(final_customers) < self.max_customers_per_route and 
                            cumulative_volume + customer['EstimatedQuantity'] <= self.max_volume_per_route):
                            final_customers.append(customer)
                            cumulative_volume += customer['EstimatedQuantity']
                    
                    trip_customers = final_customers
            else:
                # Subsequent trips: Focus on geographic clusters
                best_cluster = available_customers.groupby('cluster')['priority_score'].mean().idxmax()
                cluster_customers = available_customers[
                    available_customers['cluster'] == best_cluster
                ]
                
                trip_customers = self.optimize_route_within_cluster(
                    cluster_customers, stock_point_coords
                )
            
            if trip_customers:
                route_plan = self.create_route_plan(trip_customers, trip_num, stock_point_id)
                all_routes.extend(route_plan)
                
                for customer in trip_customers:
                    used_customers.add(customer['CustomerID'])
        
        return all_routes
    
    def create_route_plan(self, trip_customers, trip_num, stock_point_id):
        route_plan = []
        total_volume = 0
        
        for sequence, customer in enumerate(trip_customers, 1):
            total_volume += customer['EstimatedQuantity']
            
            route_plan.append({
                'PLANID': f"SP{stock_point_id}_T{trip_num:02d}",
                'TripNumber': trip_num,
                'Sequence': sequence,
                'CustomerID': customer['CustomerID'],
                'CustomerName': customer['ContactName'],
                'Latitude': customer['Latitude'],
                'Longitude': customer['Longitude'],
                'EstimatedQuantity': customer['EstimatedQuantity'],
                'CumulativeVolume': total_volume,
                'percentile_rank': customer['percentile_rank'],
                'CustomerSKUscoreRank': customer['CustomerSKUscoreRank'],
                'ProductTags': customer['ProductTag'],
                'Region': customer['Region'],
                'LGA': customer['LGA'],
                'LCDA': customer['LCDA'],
                'cluster': customer['cluster'],
                'PriorityScore': round(customer['priority_score'], 4)
            })
        
        return route_plan



# ====================================================================
# STEP 5: ADVANCED ANALYSIS FUNCTIONS
# ====================================================================

def analyze_route_efficiency(df_routes, df_stockpoint, logger=logging.getLogger(__name__)):
    """Analyze route efficiency metrics"""
    
    logger.info("\n" + "="*60)
    logger.info("ADVANCED ROUTE EFFICIENCY ANALYSIS")
    logger.info("="*60)
    
    stock_point = df_stockpoint.iloc[0]
    depot_coords = (stock_point['Latitude'], stock_point['Longitude'])
    
    for trip_num in sorted(df_routes['TripNumber'].unique()):
        trip_data = df_routes[df_routes['TripNumber'] == trip_num].sort_values('Sequence')
        
        # Calculate total distance
        total_distance = 0
        current_pos = depot_coords
        
        for idx, row in trip_data.iterrows():
            customer_pos = (row['Latitude'], row['Longitude'])
            distance = geodesic(current_pos, customer_pos).kilometers
            total_distance += distance
            current_pos = customer_pos
        
        # Return to depot
        return_distance = geodesic(current_pos, depot_coords).kilometers
        total_distance += return_distance
        
        logger.info(f"\nTrip {trip_num} Efficiency:")
        logger.info(f"  Total Distance: {total_distance:.2f} km")
        logger.info(f"  Distance per Customer: {total_distance/len(trip_data):.2f} km")
        logger.info(f"  Volume per km: {trip_data['EstimatedQuantity'].sum()/total_distance:.2f} units/km")

def generate_driver_instructions(df_routes, logger=logging.getLogger(__name__)):
    """Generate driver-friendly route instructions"""
    
    logger.info("\n" + "="*60)
    logger.info("DRIVER ROUTE INSTRUCTIONS")
    logger.info("="*60)
    
    for trip_num in sorted(df_routes['TripNumber'].unique()):
        trip_data = df_routes[df_routes['TripNumber'] == trip_num].sort_values('Sequence')
        
        logger.info(f"\n*** TRIP {trip_num} ROUTE CARD ***")
        logger.info(f"Plan ID: {trip_data.iloc[0]['PLANID']}")
        logger.info(f"Total Customers: {len(trip_data)}")
        logger.info(f"Total Volume: {trip_data['EstimatedQuantity'].sum()} units")
        logger.info("-" * 40)
        
        for idx, row in trip_data.iterrows():
            logger.info(f"Stop {row['Sequence']:2d}: {row['CustomerName']}")
            logger.info(f"         ID: {row['CustomerID']}")
            logger.info(f"         Location: {row['LGA']}, {row['Region']}")
            logger.info(f"         Coordinates: {row['Latitude']:.4f}, {row['Longitude']:.4f}")
            logger.info(f"         Volume: {row['EstimatedQuantity']} units")
            logger.info(f"         Running Total: {row['CumulativeVolume']} units")
            logger.info("")


