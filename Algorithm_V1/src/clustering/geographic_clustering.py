import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

class GeographicClusteringOptimizer:
    def __init__(self, max_customers_per_cluster=20, max_distance_km=50, 
                 dbscan_eps_km=10, dbscan_min_samples=3, use_haversine=True):
        """
        Initialize the Geographic Clustering Optimizer.
        
        Parameters:
        -----------
        max_customers_per_cluster : int, default=20
            Maximum number of customers per cluster for hierarchical clustering
        max_distance_km : float, default=50
            Maximum distance in kilometers for hierarchical clustering
        dbscan_eps_km : float, default=10
            The maximum distance between two samples for DBSCAN (in kilometers)
        dbscan_min_samples : int, default=3
            The minimum number of samples in a neighborhood for a point to be core point
        use_haversine : bool, default=True
            Whether to use haversine distance for DBSCAN (more accurate for geographic data)
        """
        self.max_customers_per_cluster = max_customers_per_cluster
        self.max_distance_km = max_distance_km
        self.dbscan_eps_km = dbscan_eps_km
        self.dbscan_min_samples = dbscan_min_samples
        self.use_haversine = use_haversine
        self.distance_matrix = None
        self.linkage_matrix = None
        
        # Convert km to approximate degrees for DBSCAN if not using haversine
        # 1 degree ≈ 111 km at equator
        self.dbscan_eps_degrees = dbscan_eps_km / 111.0
        
    def calculate_geographic_distance_matrix(self, customers_df):
        """Calculate pairwise geographic distances between all customers."""
        coords = customers_df[['Latitude', 'Longitude']].values
        n_customers = len(coords)
        
        distances = []
        for i in range(n_customers):
            for j in range(i+1, n_customers):
                dist = geodesic(coords[i], coords[j]).kilometers
                distances.append(dist)
        
        self.distance_matrix = squareform(distances)
        return self.distance_matrix
    
    def create_hierarchical_clusters(self, customers_df, method='average', criterion='distance'):
        """Create hierarchical clusters using scipy's clustering methods."""
        # Work with a copy to avoid modifying the original DataFrame
        df_copy = customers_df.copy().reset_index(drop=True)
        
        if self.distance_matrix is None:
            self.calculate_geographic_distance_matrix(df_copy)
        else:
            # Recalculate distance matrix for the current subset
            self.calculate_geographic_distance_matrix(df_copy)
        
        if method == 'ward':
            coords = df_copy[['Latitude', 'Longitude']].values
            self.linkage_matrix = linkage(coords, method='ward')
        else:
            condensed_distances = squareform(self.distance_matrix, checks=False)
            self.linkage_matrix = linkage(condensed_distances, method=method)
        
        if criterion == 'distance':
            clusters = fcluster(self.linkage_matrix, self.max_distance_km, criterion='distance')
        elif criterion == 'maxclust':
            n_clusters = max(1, len(df_copy) // self.max_customers_per_cluster)
            clusters = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        else:
            clusters = fcluster(self.linkage_matrix, 1.5, criterion='inconsistent')
        
        # Create a copy of the original DataFrame with the original index
        result_df = customers_df.copy()
        result_df['cluster'] = clusters
        
        return result_df

    def create_dbscan_clusters(self, customers_df, return_noise_separately=False):
        """
        Create DBSCAN clusters based on geographic coordinates.
        
        Parameters:
        -----------
        customers_df : pandas.DataFrame
            DataFrame containing 'Latitude' and 'Longitude' columns
        return_noise_separately : bool, default=False
            If True, returns noise points (cluster -1) as a separate cluster
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added 'cluster' column containing cluster assignments
        """
        coords = customers_df[['Latitude', 'Longitude']].values
        
        if self.use_haversine:
            # Use haversine distance metric for more accurate geographic clustering
            # Convert coordinates to radians for haversine calculation
            coords_rad = np.radians(coords)
            
            # DBSCAN with haversine metric (eps in radians)
            # Convert km to radians: 1 km ≈ 1/6371 radians
            eps_radians = self.dbscan_eps_km / 6371.0
            
            dbscan = DBSCAN(
                eps=eps_radians,
                min_samples=self.dbscan_min_samples,
                metric='haversine'
            )
            
            clusters = dbscan.fit_predict(coords_rad)
        else:
            # Use euclidean distance on lat/lon coordinates (less accurate but faster)
            # Standardize coordinates to handle different scales
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)
            
            dbscan = DBSCAN(
                eps=self.dbscan_eps_degrees,
                min_samples=self.dbscan_min_samples,
                metric='euclidean'
            )
            
            clusters = dbscan.fit_predict(coords_scaled)
        
        # Handle noise points (cluster -1)
        if not return_noise_separately:
            # Convert noise points to individual clusters
            noise_mask = clusters == -1
            if np.any(noise_mask):
                max_cluster = clusters.max() if clusters.max() >= 0 else -1
                noise_indices = np.where(noise_mask)[0]
                for i, idx in enumerate(noise_indices):
                    clusters[idx] = max_cluster + 1 + i
        
        # Ensure cluster labels start from 1 (not 0) to match hierarchical clustering
        if clusters.min() >= 0:
            clusters = clusters + 1
        
        customers_df = customers_df.copy()
        customers_df['cluster'] = clusters
        
        return customers_df
    
    def create_hybrid_clusters(self, customers_df, dbscan_first=True, 
                         hierarchical_method='average', hierarchical_criterion='distance'):
        """
        Create hybrid clusters by combining DBSCAN and hierarchical clustering.
        """
        customers_df = customers_df.copy()
        
        if dbscan_first:
            # Step 1: Apply DBSCAN
            df_clustered = self.create_dbscan_clusters(customers_df, return_noise_separately=False)
            
            # Step 2: Refine large clusters with hierarchical clustering
            cluster_counts = df_clustered['cluster'].value_counts()
            large_clusters = cluster_counts[cluster_counts > self.max_customers_per_cluster].index
            
            if len(large_clusters) > 0:
                max_cluster_id = df_clustered['cluster'].max()
                
                for cluster_id in large_clusters:
                    cluster_mask = df_clustered['cluster'] == cluster_id
                    cluster_customers = df_clustered[cluster_mask].copy()
                    
                    # Apply hierarchical clustering to this subset
                    sub_clustered = self.create_hierarchical_clusters(
                        cluster_customers, method=hierarchical_method, criterion=hierarchical_criterion
                    )
                    
                    # Update cluster IDs to be unique
                    unique_sub_clusters = sub_clustered['cluster'].unique()
                    cluster_id_mapping = {}
                    
                    for i, sub_cluster_id in enumerate(unique_sub_clusters):
                        new_cluster_id = max_cluster_id + 1 + i
                        cluster_id_mapping[sub_cluster_id] = new_cluster_id
                    
                    # Apply the mapping to update cluster IDs
                    for old_id, new_id in cluster_id_mapping.items():
                        sub_mask = sub_clustered['cluster'] == old_id
                        # Use the original index to match rows correctly
                        matching_indices = sub_clustered[sub_mask].index
                        df_clustered.loc[matching_indices, 'cluster'] = new_id
                    
                    max_cluster_id += len(unique_sub_clusters)
                    
        else:
            # Step 1: Apply hierarchical clustering
            df_clustered = self.create_hierarchical_clusters(
                customers_df, method=hierarchical_method, criterion=hierarchical_criterion
            )
            
            # Step 2: Refine large clusters with DBSCAN
            cluster_counts = df_clustered['cluster'].value_counts()
            large_clusters = cluster_counts[cluster_counts > self.max_customers_per_cluster].index
            
            if len(large_clusters) > 0:
                max_cluster_id = df_clustered['cluster'].max()
                
                for cluster_id in large_clusters:
                    cluster_mask = df_clustered['cluster'] == cluster_id
                    cluster_customers = df_clustered[cluster_mask].copy()
                    
                    # Apply DBSCAN to this subset
                    sub_clustered = self.create_dbscan_clusters(cluster_customers, return_noise_separately=False)
                    
                    # Update cluster IDs to be unique
                    unique_sub_clusters = sub_clustered['cluster'].unique()
                    cluster_id_mapping = {}
                    
                    for i, sub_cluster_id in enumerate(unique_sub_clusters):
                        new_cluster_id = max_cluster_id + 1 + i
                        cluster_id_mapping[sub_cluster_id] = new_cluster_id
                    
                    # Apply the mapping to update cluster IDs
                    for old_id, new_id in cluster_id_mapping.items():
                        sub_mask = sub_clustered['cluster'] == old_id
                        # Use the original index to match rows correctly
                        matching_indices = sub_clustered[sub_mask].index
                        df_clustered.loc[matching_indices, 'cluster'] = new_id
                    
                    max_cluster_id += len(unique_sub_clusters)
        
        return df_clustered
    
    def get_cluster_statistics(self, customers_df):
        """
        Get statistics about the clusters.
        
        Parameters:
        -----------
        customers_df : pandas.DataFrame
            DataFrame with 'cluster' column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cluster statistics
        """
        if 'cluster' not in customers_df.columns:
            raise ValueError("DataFrame must contain 'cluster' column")
        
        stats = []
        for cluster_id in sorted(customers_df['cluster'].unique()):
            cluster_data = customers_df[customers_df['cluster'] == cluster_id]
            
            # Calculate cluster center (centroid)
            center_lat = cluster_data['Latitude'].mean()
            center_lon = cluster_data['Longitude'].mean()
            
            # Calculate maximum distance from center
            max_dist = 0
            if len(cluster_data) > 1:
                center = (center_lat, center_lon)
                for _, row in cluster_data.iterrows():
                    point = (row['Latitude'], row['Longitude'])
                    dist = geodesic(center, point).kilometers
                    max_dist = max(max_dist, dist)
            
            stats.append({
                'cluster_id': cluster_id,
                'customer_count': len(cluster_data),
                'center_latitude': center_lat,
                'center_longitude': center_lon,
                'max_distance_from_center_km': max_dist,
                'is_noise': cluster_id == -1 if -1 in customers_df['cluster'].values else False
            })
        
        return pd.DataFrame(stats)
    
    def optimize_dbscan_parameters(self, customers_df, eps_range=None, min_samples_range=None, 
                                 target_cluster_size=None):
        """
        Optimize DBSCAN parameters using silhouette score and cluster size constraints.
        
        Parameters:
        -----------
        customers_df : pandas.DataFrame
            DataFrame containing 'Latitude' and 'Longitude' columns
        eps_range : list, optional
            Range of eps values to test (in km). Default: [5, 10, 15, 20, 25, 30]
        min_samples_range : list, optional
            Range of min_samples values to test. Default: [2, 3, 4, 5]
        target_cluster_size : int, optional
            Target average cluster size. Default: self.max_customers_per_cluster
            
        Returns:
        --------
        dict
            Best parameters and their scores
        """
        from sklearn.metrics import silhouette_score
        
        if eps_range is None:
            eps_range = [5, 10, 15, 20, 25, 30]
        if min_samples_range is None:
            min_samples_range = [2, 3, 4, 5]
        if target_cluster_size is None:
            target_cluster_size = self.max_customers_per_cluster
        
        coords = customers_df[['Latitude', 'Longitude']].values
        best_score = -1
        best_params = {}
        results = []
        
        for eps_km in eps_range:
            for min_samples in min_samples_range:
                # Temporarily update parameters
                original_eps = self.dbscan_eps_km
                original_min_samples = self.dbscan_min_samples
                
                self.dbscan_eps_km = eps_km
                self.dbscan_min_samples = min_samples
                
                try:
                    # Get clusters
                    clustered_df = self.create_dbscan_clusters(customers_df.copy(), return_noise_separately=True)
                    clusters = clustered_df['cluster'].values
                    
                    # Skip if all points are noise or all points are in one cluster
                    unique_clusters = np.unique(clusters)
                    if len(unique_clusters) < 2 or (len(unique_clusters) == 2 and -1 in unique_clusters):
                        continue
                    
                    # Calculate silhouette score (exclude noise points)
                    non_noise_mask = clusters != -1
                    if np.sum(non_noise_mask) < 2:
                        continue
                    
                    if self.use_haversine:
                        coords_rad = np.radians(coords[non_noise_mask])
                        score = silhouette_score(coords_rad, clusters[non_noise_mask], metric='haversine')
                    else:
                        score = silhouette_score(coords[non_noise_mask], clusters[non_noise_mask])
                    
                    # Calculate cluster size penalty
                    cluster_sizes = pd.Series(clusters[non_noise_mask]).value_counts()
                    avg_cluster_size = cluster_sizes.mean()
                    size_penalty = abs(avg_cluster_size - target_cluster_size) / target_cluster_size
                    
                    # Calculate noise penalty
                    noise_penalty = np.sum(clusters == -1) / len(clusters)
                    
                    # Combined score (lower penalties are better)
                    combined_score = score - 0.1 * size_penalty - 0.2 * noise_penalty
                    
                    results.append({
                        'eps_km': eps_km,
                        'min_samples': min_samples,
                        'silhouette_score': score,
                        'combined_score': combined_score,
                        'n_clusters': len(unique_clusters) - (1 if -1 in unique_clusters else 0),
                        'n_noise': np.sum(clusters == -1),
                        'avg_cluster_size': avg_cluster_size
                    })
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = {
                            'eps_km': eps_km,
                            'min_samples': min_samples,
                            'silhouette_score': score,
                            'combined_score': combined_score
                        }
                
                except Exception as e:
                    warnings.warn(f"Error with eps={eps_km}, min_samples={min_samples}: {str(e)}")
                    continue
                finally:
                    # Restore original parameters
                    self.dbscan_eps_km = original_eps
                    self.dbscan_min_samples = original_min_samples
        
        return {
            'best_params': best_params,
            'all_results': pd.DataFrame(results) if results else pd.DataFrame()
        }