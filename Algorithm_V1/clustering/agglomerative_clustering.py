import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class BaseGeographicClustering:
    """
    A base class containing common geographic utility methods
    used by both Divisive and Agglomerative clustering implementations.
    """
    def __init__(self):
        self.earth_radius_km = 6371.0

    def haversine_vectorized(self, coords1, coords2=None):
        """
        Highly optimized vectorized haversine distance calculation.
        If coords2 is None, calculates pairwise distances within coords1.
        Assumes coords are [latitude, longitude].
        """
        if coords2 is None:
            # Pairwise distances within coords1 (NxN matrix)
            coords1_rad = np.radians(coords1)
            
            lat1 = coords1_rad[:, 0]
            lon1 = coords1_rad[:, 1]
            
            lat1_mesh, lat2_mesh = np.meshgrid(lat1, lat1)
            lon1_mesh, lon2_mesh = np.meshgrid(lon1, lon1)
            
            dlat = lat2_mesh - lat1_mesh
            dlon = lon2_mesh - lon1_mesh
            
            a = (np.sin(dlat / 2) ** 2 + 
                 np.cos(lat1_mesh) * np.cos(lat2_mesh) * np.sin(dlon / 2) ** 2)
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            
            return self.earth_radius_km * c
        else:
            # Distance from each point in coords1 to each point in coords2 (NxM matrix)
            coords1_rad = np.radians(coords1)
            coords2_rad = np.radians(coords2)
            
            lat1 = coords1_rad[:, 0][:, np.newaxis]
            lon1 = coords1_rad[:, 1][:, np.newaxis]
            lat2 = coords2_rad[:, 0][np.newaxis, :]
            lon2 = coords2_rad[:, 1][np.newaxis, :]
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = (np.sin(dlat / 2) ** 2 + 
                 np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            
            return self.earth_radius_km * c
    
    def haversine_pdist(self, coords):
        """
        Optimized haversine distance calculation for pdist, returning condensed distance matrix.
        Assumes coords are [latitude, longitude].
        """
        def haversine_metric(u, v):
            lat1, lon1 = np.radians(u)
            lat2, lon2 = np.radians(v)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = (np.sin(dlat / 2) ** 2 + 
                 np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            
            return self.earth_radius_km * c
        
        return pdist(coords, metric=haversine_metric)
    
    def haversine_single_pair(self, coord1, coord2):
        """Calculate haversine distance between two points."""
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (np.sin(dlat / 2) ** 2 + 
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
        return self.earth_radius_km * c

    def _convex_hull_diameter(self, coords):
        """Enhanced convex hull approximation for diameter estimation."""
        lats, lons = coords[:, 0], coords[:, 1]
        
        # Get extreme points (min/max lat/lon)
        extreme_indices = [
            np.argmax(lats), np.argmin(lats),
            np.argmax(lons), np.argmin(lons)
        ]
        
        # Add points from different quadrants relative to the mean center
        lat_center, lon_center = np.mean(lats), np.mean(lons)
        
        quadrants = [
            (lats >= lat_center) & (lons >= lon_center),   # NE
            (lats >= lat_center) & (lons < lon_center),    # NW
            (lats < lat_center) & (lons >= lon_center),    # SE
            (lats < lat_center) & (lons < lon_center)      # SW
        ]
        
        for quadrant in quadrants:
            if np.any(quadrant):
                quad_indices = np.where(quadrant)[0]
                # Add furthest point from center in each non-empty quadrant
                distances_from_center = np.sqrt(
                    (lats[quad_indices] - lat_center)**2 + 
                    (lons[quad_indices] - lon_center)**2
                )
                furthest_idx = quad_indices[np.argmax(distances_from_center)]
                extreme_indices.append(furthest_idx)
        
        # Add some random points to further improve approximation for larger clusters
        n_random = min(12, len(coords) - len(set(extreme_indices)))
        if n_random > 0:
            available_indices = list(set(range(len(coords))) - set(extreme_indices))
            if available_indices:
                random_indices = np.random.choice(available_indices, 
                                                  min(n_random, len(available_indices)), 
                                                  replace=False)
                extreme_indices.extend(random_indices)
        
        # Get unique sample coordinates
        sample_indices = list(set(extreme_indices))
        sample_coords = coords[sample_indices]
        
        if len(sample_coords) <= 1:
            return 0
        
        # Calculate max distance among the sampled points
        distances = self.haversine_pdist(sample_coords)
        return np.max(distances)
    
    def _grid_based_diameter(self, coords):
        """Grid-based sampling for very large clusters."""
        lats, lons = coords[:, 0], coords[:, 1]
        
        # Create 6x6 grid
        lat_bins = np.linspace(lats.min(), lats.max(), 7)
        lon_bins = np.linspace(lons.min(), lons.max(), 7)
        
        sample_indices = []
        for i in range(len(lat_bins)-1):
            for j in range(len(lon_bins)-1):
                mask = ((lats >= lat_bins[i]) & (lats < lat_bins[i+1]) & 
                        (lons >= lon_bins[j]) & (lons < lon_bins[j+1]))
                cell_indices = np.where(mask)[0]
                if len(cell_indices) > 0:
                    # Sample up to 2 points from each cell
                    n_sample = min(2, len(cell_indices))
                    sampled = np.random.choice(cell_indices, n_sample, replace=False)
                    sample_indices.extend(sampled)
        
        if len(sample_indices) <= 1:
            return 0
        
        sample_coords = coords[sample_indices]
        distances = self.haversine_pdist(sample_coords)
        return np.max(distances)
    
    def calculate_cluster_diameter_fast(self, coords):
        """
        Fast cluster diameter calculation with multiple optimization strategies.
        """
        n_points = len(coords)
        
        if n_points <= 1:
            return 0
        
        if n_points == 2:
            return self.haversine_single_pair(coords[0], coords[1])
        
        # Use different strategies based on cluster size
        if n_points <= 10:
            # Small clusters: exact calculation using pdist
            distances = self.haversine_pdist(coords)
            return np.max(distances)
        elif n_points <= 50:
            # Medium clusters: vectorized calculation or pdist
            # Prioritize vectorized if available and faster for this range
            distance_matrix = self.haversine_vectorized(coords)
            # Ensure we're not taking max of diagonal (self-distances = 0)
            return np.max(distance_matrix[np.triu_indices(n_points, k=1)])
        else:
            # Large clusters: smart sampling
            return self._smart_diameter_estimation(coords)
    
    def _smart_diameter_estimation(self, coords):
        """
        Improved diameter estimation using multiple sampling strategies.
        """
        n_points = len(coords)
        
        # Strategy 1: Convex hull approximation
        hull_diameter = self._convex_hull_diameter(coords)
        
        # Strategy 2: Grid-based sampling for very large clusters (higher confidence for max)
        if n_points > 200: # Threshold for when grid sampling might be beneficial
            grid_diameter = self._grid_based_diameter(coords)
            return max(hull_diameter, grid_diameter)
        
        return hull_diameter

    def _find_approximate_farthest_pair(self, coords):
        """Find approximate farthest pair for large clusters.
           Moved from OptimizedDivisiveGeographicClustering to BaseGeographicClustering."""
        n_points = len(coords)
        
        if n_points <= 100:
            # For moderate sizes, use exact calculation
            distances = self.haversine_pdist(coords)
            distance_matrix = squareform(distances)
            max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
            return max_idx[0], max_idx[1]
        
        # For large clusters, use sampling
        sample_size = min(50, n_points)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)
        sample_coords = coords[sample_indices]
        
        distances = self.haversine_pdist(sample_coords)
        distance_matrix = squareform(distances)
        max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        
        return sample_indices[max_idx[0]], sample_indices[max_idx[1]]

    def get_cluster_stats(self, clustered_df, max_customers_per_cluster, max_distance_km):
        """Get comprehensive clustering statistics."""
        if 'cluster' not in clustered_df.columns:
            raise ValueError("DataFrame must contain 'cluster' column")
        
        stats = {}
        cluster_sizes = []
        cluster_diameters = []
        
        for cluster_id in clustered_df['cluster'].unique():
            if cluster_id == -1: # Unassigned points if any
                continue
            
            cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
            coords = cluster_data[['Latitude', 'Longitude']].values
            
            diameter = self.calculate_cluster_diameter_fast(coords)
            cluster_sizes.append(len(cluster_data))
            cluster_diameters.append(diameter)
            
            stats[cluster_id] = {
                'size': len(cluster_data),
                'diameter_km': diameter,
                'centroid_lat': np.mean(coords[:, 0]),
                'centroid_lon': np.mean(coords[:, 1]),
                'meets_size_constraint': len(cluster_data) <= max_customers_per_cluster,
                'meets_distance_constraint': diameter <= max_distance_km
            }
        
        # Overall statistics
        stats['summary'] = {
            'total_clusters': len(stats), 
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': np.max(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': np.min(cluster_sizes) if cluster_sizes else 0,
            'avg_diameter': np.mean(cluster_diameters) if cluster_diameters else 0,
            'max_diameter': np.max(cluster_diameters) if cluster_diameters else 0,
            'size_violations': sum(1 for size in cluster_sizes if size > max_customers_per_cluster),
            'distance_violations': sum(1 for diameter in cluster_diameters if diameter > max_distance_km)
        }
        
        return stats


class AgglomerativeGeographicClustering(BaseGeographicClustering):
    def __init__(self, max_customers_per_cluster=20, max_distance_km=50, 
                 linkage_method='ward', sub_cluster_if_oversized=True):
        """
        Initializes the Agglomerative Geographic Clustering.

        Args:
            max_customers_per_cluster (int): Maximum number of customers allowed in a single cluster.
            max_distance_km (float): Maximum diameter (distance between two farthest points)
                                     allowed within a cluster in kilometers.
            linkage_method (str): Method to use for calculating the distance between clusters
                                  in hierarchical clustering. Options: 'ward', 'single', 'complete', 'average'.
            sub_cluster_if_oversized (bool): If True, clusters that exceed max_customers_per_cluster
                                             after distance-based cutting will be further sub-clustered
                                             using K-Means.
        """
        super().__init__()
        self.max_customers_per_cluster = max_customers_per_cluster
        self.max_distance_km = max_distance_km
        self.linkage_method = linkage_method
        self.sub_cluster_if_oversized = sub_cluster_if_oversized

    def agglomerative_clustering(self, customers_df):
        """
        Performs agglomerative hierarchical clustering on geographic data
        with constraints on cluster size and diameter.
        """
        customers_df = customers_df.copy().reset_index(drop=True)
        
        if 'Latitude' not in customers_df.columns or 'Longitude' not in customers_df.columns:
            raise ValueError("DataFrame must contain 'Latitude' and 'Longitude' columns")
        
        coords_array = customers_df[['Latitude', 'Longitude']].values
        n_customers = len(customers_df)
        
        if n_customers == 0:
            customers_df['cluster'] = []
            return customers_df
        
        if n_customers == 1:
            customers_df['cluster'] = 1
            return customers_df

        # Step 1: Calculate pairwise Haversine distances
        print(f"Calculating {n_customers*(n_customers-1)//2} pairwise distances...")
        # Check if the number of points is too large for pdist to avoid MemoryError
        # A rough heuristic: 5000 points * 5000 points / 2 * 8 bytes/float ~ 100MB
        # For very large N, consider approximate methods if pdist is too slow/memory intensive
        if n_customers > 2000 and self.linkage_method != 'ward': # Ward only works with Euclidean-like pdist
             # For very large datasets, pdist might be too slow or memory intensive.
             # In such cases, one might consider sampling or approximate hierarchical methods,
             # or other clustering algorithms like DBSCAN that don't require a full distance matrix.
             # For now, we proceed with pdist as it's standard for scipy.hierarchy.
            print("Warning: Large dataset for pdist. This might take a while or consume a lot of memory.")

        distances = self.haversine_pdist(coords_array)
        
        # Step 2: Perform hierarchical clustering using linkage
        print(f"Performing linkage using '{self.linkage_method}' method...")
        linkage_matrix = linkage(distances, method=self.linkage_method)
        
        # Step 3: Cut the dendrogram based on max_distance_km
        # This creates clusters where no two points are farther apart than max_distance_km
        print(f"Cutting dendrogram at max_distance_km={self.max_distance_km}...")
        initial_labels = fcluster(linkage_matrix, self.max_distance_km, criterion='distance')
        
        customers_df['cluster_temp'] = initial_labels
        final_cluster_id = 0
        final_clusters = {}

        # Step 4: Post-process for max_customers_per_cluster constraint
        print(f"Post-processing clusters for size constraint (max {self.max_customers_per_cluster} customers)...")
        for current_cluster_label in sorted(customers_df['cluster_temp'].unique()):
            cluster_indices = customers_df[customers_df['cluster_temp'] == current_cluster_label].index.values
            current_coords = coords_array[cluster_indices]
            
            if len(cluster_indices) > self.max_customers_per_cluster and self.sub_cluster_if_oversized:
                print(f"  Cluster {current_cluster_label} (size {len(cluster_indices)}) is oversized. Sub-clustering...")
                # Sub-cluster using K-Means. Determine optimal k based on current size / max_customers_per_cluster
                k_sub = int(np.ceil(len(cluster_indices) / self.max_customers_per_cluster))
                k_sub = max(2, k_sub) # Ensure at least 2 clusters if splitting
                
                # Use approximate farthest pair for K-Means initialization
                initial_centers_indices = self._find_approximate_farthest_pair(current_coords)
                # Ensure initial_centers_indices has enough elements for k_sub.
                # If k_sub > 2, KMeans++ initialization is generally more robust than a simple farthest pair.
                if k_sub > 2:
                    kmeans_sub = KMeans(n_clusters=k_sub, n_init=10, random_state=42) # Let KMeans find its own init
                elif len(initial_centers_indices) >= 2: # k_sub = 2, use the farthest pair if available
                    kmeans_sub = KMeans(n_clusters=k_sub, init=current_coords[[initial_centers_indices[0], initial_centers_indices[1]]], n_init=1, random_state=42)
                else: # Fallback if initial_centers_indices is not sufficient for k_sub=2
                    kmeans_sub = KMeans(n_clusters=k_sub, n_init=10, random_state=42)


                sub_labels = kmeans_sub.fit_predict(current_coords)
                
                for sub_label in np.unique(sub_labels):
                    final_cluster_id += 1
                    sub_cluster_indices = cluster_indices[sub_labels == sub_label]
                    final_clusters[final_cluster_id] = sub_cluster_indices
            else:
                final_cluster_id += 1
                final_clusters[final_cluster_id] = cluster_indices
        
        # Assign final cluster labels to the DataFrame
        result_df = customers_df.copy()
        result_df['cluster'] = -1 # Initialize with unassigned

        for cluster_id, indices in final_clusters.items():
            result_df.loc[indices, 'cluster'] = cluster_id
        
        result_df = result_df.drop(columns=['cluster_temp'])
        print("Agglomerative clustering completed.\n")
        return result_df








