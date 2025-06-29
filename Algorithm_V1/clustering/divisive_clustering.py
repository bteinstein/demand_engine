import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from geopy.distance import geodesic

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd 
from scipy.spatial import cKDTree




import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

## Best - Fast and Accurate
class OptimizedDivisiveGeographicClustering:
    def __init__(self, max_customers_per_cluster=20, max_distance_km=50, 
                 use_vectorized_distances=True, balance_clusters=False):
        self.max_customers_per_cluster = max_customers_per_cluster
        self.max_distance_km = max_distance_km
        self.earth_radius_km = 6371.0
        self.use_vectorized_distances = use_vectorized_distances
        self.balance_clusters = balance_clusters
        
    def haversine_vectorized(self, coords1, coords2=None):
        """
        Highly optimized vectorized haversine distance calculation.
        If coords2 is None, calculates pairwise distances within coords1.
        """
        if coords2 is None:
            # Pairwise distances within coords1
            coords1_rad = np.radians(coords1)
            n = len(coords1)
            
            # Create meshgrids for vectorized calculation
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
            # Distance from each point in coords1 to each point in coords2
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
        Optimized haversine distance calculation using scipy's pdist.
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
            # Small clusters: exact calculation
            distances = self.haversine_pdist(coords)
            return np.max(distances)
        elif n_points <= 50:
            # Medium clusters: vectorized calculation
            if self.use_vectorized_distances:
                distance_matrix = self.haversine_vectorized(coords)
                return np.max(distance_matrix)
            else:
                distances = self.haversine_pdist(coords)
                return np.max(distances)
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
        
        # Strategy 2: Grid-based sampling for large clusters
        if n_points > 200:
            grid_diameter = self._grid_based_diameter(coords)
            return max(hull_diameter, grid_diameter)
        
        return hull_diameter
    
    def _convex_hull_diameter(self, coords):
        """Enhanced convex hull approximation."""
        lats, lons = coords[:, 0], coords[:, 1]
        
        # Get extreme points
        extreme_indices = [
            np.argmax(lats), np.argmin(lats),
            np.argmax(lons), np.argmin(lons)
        ]
        
        # Add points from different quadrants
        lat_center, lon_center = np.mean(lats), np.mean(lons)
        
        quadrants = [
            (lats >= lat_center) & (lons >= lon_center),  # NE
            (lats >= lat_center) & (lons < lon_center),   # NW
            (lats < lat_center) & (lons >= lon_center),   # SE
            (lats < lat_center) & (lons < lon_center)     # SW
        ]
        
        for quadrant in quadrants:
            if np.any(quadrant):
                quad_indices = np.where(quadrant)[0]
                # Add furthest point from center in each quadrant
                distances_from_center = np.sqrt(
                    (lats[quad_indices] - lat_center)**2 + 
                    (lons[quad_indices] - lon_center)**2
                )
                furthest_idx = quad_indices[np.argmax(distances_from_center)]
                extreme_indices.append(furthest_idx)
        
        # Add some random points
        n_random = min(12, len(coords) - len(set(extreme_indices)))
        if n_random > 0:
            available_indices = list(set(range(len(coords))) - set(extreme_indices))
            if available_indices:
                random_indices = np.random.choice(available_indices, 
                                                min(n_random, len(available_indices)), 
                                                replace=False)
                extreme_indices.extend(random_indices)
        
        # Get unique sample
        sample_indices = list(set(extreme_indices))
        sample_coords = coords[sample_indices]
        
        if len(sample_coords) <= 1:
            return 0
        
        distances = self.haversine_pdist(sample_coords)
        return np.max(distances)
    
    def _grid_based_diameter(self, coords):
        """Grid-based sampling for very large clusters."""
        # Create a grid and sample points from each grid cell
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
    
    def should_split_cluster(self, cluster_indices, coords_array):
        """Enhanced cluster splitting logic with load balancing."""
        cluster_size = len(cluster_indices)
        
        if cluster_size <= 2:
            return False
        
        # Hard size constraint
        if cluster_size > self.max_customers_per_cluster * 1.5:
            return True
        
        # Soft size constraint with diameter check
        if cluster_size > self.max_customers_per_cluster:
            cluster_coords = coords_array[cluster_indices]
            diameter = self.calculate_cluster_diameter_fast(cluster_coords)
            return diameter > self.max_distance_km * 0.8  # More lenient for size
        
        # Diameter constraint
        cluster_coords = coords_array[cluster_indices]
        diameter = self.calculate_cluster_diameter_fast(cluster_coords)
        
        return diameter > self.max_distance_km
    
    def geographic_split(self, cluster_indices, coords_array):
        """
        Improved geographic splitting with better load balancing.
        """
        if len(cluster_indices) <= 2:
            return [cluster_indices]
        
        cluster_coords = coords_array[cluster_indices]
        n_points = len(cluster_coords)
        
        # For small clusters, use exact method
        if n_points <= 50:
            return self._exact_geographic_split(cluster_indices, cluster_coords)
        
        # For medium clusters, use K-means with geographic initialization
        if n_points <= 200:
            return self._kmeans_geographic_split(cluster_indices, cluster_coords)
        
        # For large clusters, use hierarchical approach
        return self._hierarchical_geographic_split(cluster_indices, cluster_coords)
    
    def _exact_geographic_split(self, cluster_indices, cluster_coords):
        """Exact splitting for small clusters."""
        n_points = len(cluster_coords)
        
        # Find the two points that are farthest apart
        distances = self.haversine_pdist(cluster_coords)
        distance_matrix = squareform(distances)
        max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        center1_idx, center2_idx = max_idx[0], max_idx[1]
        
        center1 = cluster_coords[center1_idx]
        center2 = cluster_coords[center2_idx]
        
        # Assign points to closest center
        distances_to_center1 = self.haversine_vectorized(cluster_coords, center1.reshape(1, -1))[:, 0]
        distances_to_center2 = self.haversine_vectorized(cluster_coords, center2.reshape(1, -1))[:, 0]
        
        labels = (distances_to_center1 <= distances_to_center2).astype(int)
        
        return self._balance_split(cluster_indices, labels)
    
    def _kmeans_geographic_split(self, cluster_indices, cluster_coords):
        """K-means splitting with geographic initialization."""
        n_points = len(cluster_coords)
        
        # Initialize with farthest pair
        center1_idx, center2_idx = self._find_approximate_farthest_pair(cluster_coords)
        initial_centers = cluster_coords[[center1_idx, center2_idx]]
        
        # Apply K-means
        kmeans = KMeans(n_clusters=2, init=initial_centers, n_init=1, random_state=42)
        labels = kmeans.fit_predict(cluster_coords)
        
        return self._balance_split(cluster_indices, labels)
    
    def _hierarchical_geographic_split(self, cluster_indices, cluster_coords):
        """Hierarchical splitting for large clusters."""
        # Use linkage-based clustering for very large clusters
        n_sample = min(100, len(cluster_coords))
        sample_indices = np.random.choice(len(cluster_coords), n_sample, replace=False)
        sample_coords = cluster_coords[sample_indices]
        
        # Compute linkage on sample
        distances = self.haversine_pdist(sample_coords)
        linkage_matrix = linkage(distances, method='ward')
        sample_labels = fcluster(linkage_matrix, 2, criterion='maxclust') - 1
        
        # Assign all points based on closest sample point
        center1_coords = sample_coords[sample_labels == 0]
        center2_coords = sample_coords[sample_labels == 1]
        
        if len(center1_coords) == 0 or len(center2_coords) == 0:
            # Fallback to farthest pair method
            return self._exact_geographic_split(cluster_indices, cluster_coords)
        
        center1 = np.mean(center1_coords, axis=0)
        center2 = np.mean(center2_coords, axis=0)
        
        distances_to_center1 = self.haversine_vectorized(cluster_coords, center1.reshape(1, -1))[:, 0]
        distances_to_center2 = self.haversine_vectorized(cluster_coords, center2.reshape(1, -1))[:, 0]
        
        labels = (distances_to_center1 <= distances_to_center2).astype(int)
        
        return self._balance_split(cluster_indices, labels)
    
    def _balance_split(self, cluster_indices, labels):
        """Balance the split to avoid very uneven clusters."""
        cluster_0_indices = cluster_indices[labels == 0]
        cluster_1_indices = cluster_indices[labels == 1]
        
        # Ensure no empty clusters
        if len(cluster_0_indices) == 0:
            cluster_0_indices = np.array([cluster_1_indices[0]])
            cluster_1_indices = cluster_1_indices[1:]
        elif len(cluster_1_indices) == 0:
            cluster_1_indices = np.array([cluster_0_indices[0]])
            cluster_0_indices = cluster_0_indices[1:]
        
        # Optional: Balance cluster sizes if one is much larger
        if self.balance_clusters:
            size_0, size_1 = len(cluster_0_indices), len(cluster_1_indices)
            if size_0 > 3 * size_1 and size_1 > 0:
                # Move some points from cluster 0 to cluster 1
                n_move = (size_0 - size_1) // 4
                move_indices = cluster_0_indices[:n_move]
                cluster_0_indices = cluster_0_indices[n_move:]
                cluster_1_indices = np.concatenate([cluster_1_indices, move_indices])
            elif size_1 > 3 * size_0 and size_0 > 0:
                # Move some points from cluster 1 to cluster 0
                n_move = (size_1 - size_0) // 4
                move_indices = cluster_1_indices[:n_move]
                cluster_1_indices = cluster_1_indices[n_move:]
                cluster_0_indices = np.concatenate([cluster_0_indices, move_indices])
        
        return [cluster_0_indices, cluster_1_indices]
    
    def _find_approximate_farthest_pair(self, coords):
        """Find approximate farthest pair for large clusters."""
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
    
    def divisive_clustering(self, customers_df):
        """Perform optimized divisive hierarchical clustering."""
        customers_df = customers_df.copy().reset_index(drop=True)
        
        # Validate input
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
        
        # Priority queue approach for better clustering
        clusters_to_process = [(n_customers, np.arange(n_customers))]  # (size, indices)
        final_clusters = []
        
        iteration_count = 0
        max_iterations = n_customers * 2
        
        while clusters_to_process and iteration_count < max_iterations:
            # Process largest cluster first
            clusters_to_process.sort(key=lambda x: x[0], reverse=True)
            current_size, current_cluster_indices = clusters_to_process.pop(0)
            iteration_count += 1
            
            if self.should_split_cluster(current_cluster_indices, coords_array):
                subclusters = self.geographic_split(current_cluster_indices, coords_array)
                
                for subcluster_indices in subclusters:
                    if len(subcluster_indices) > 0:
                        clusters_to_process.append((len(subcluster_indices), subcluster_indices))
            else:
                final_clusters.append(current_cluster_indices)
        
        # Handle remaining clusters
        final_clusters.extend([indices for _, indices in clusters_to_process])
        
        # Create result DataFrame
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster_indices in enumerate(final_clusters, 1):
            result_df.loc[cluster_indices, 'cluster'] = cluster_id
        
        return result_df
    
    def get_cluster_stats(self, clustered_df):
        """Get comprehensive clustering statistics."""
        if 'cluster' not in clustered_df.columns:
            raise ValueError("DataFrame must contain 'cluster' column")
        
        stats = {}
        cluster_sizes = []
        cluster_diameters = []
        
        for cluster_id in clustered_df['cluster'].unique():
            if cluster_id == -1:
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
                'meets_size_constraint': len(cluster_data) <= self.max_customers_per_cluster,
                'meets_distance_constraint': diameter <= self.max_distance_km
            }
        
        # Overall statistics
        stats['summary'] = {
            'total_clusters': len(stats) - 1,  # Excluding summary
            'avg_cluster_size': np.mean(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'avg_diameter': np.mean(cluster_diameters),
            'max_diameter': np.max(cluster_diameters),
            'size_violations': sum(1 for size in cluster_sizes if size > self.max_customers_per_cluster),
            'distance_violations': sum(1 for diameter in cluster_diameters if diameter > self.max_distance_km)
        }
        
        return stats
    


class DivisiveGeographicClustering:
    def __init__(self, max_customers_per_cluster=20, max_distance_km=50):
        self.max_customers_per_cluster = max_customers_per_cluster
        self.max_distance_km = max_distance_km
    
    def calculate_cluster_diameter(self, customers_subset):
        """Calculate the maximum distance within a cluster."""
        if len(customers_subset) <= 1:
            return 0
        
        coords = customers_subset[['Latitude', 'Longitude']].values
        max_dist = 0
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = geodesic(coords[i], coords[j]).kilometers
                max_dist = max(max_dist, dist)
        
        return max_dist
    
    def should_split_cluster(self, customers_subset):
        """Determine if a cluster should be split."""
        # Split if too many customers OR diameter too large
        too_many_customers = len(customers_subset) > self.max_customers_per_cluster
        diameter_too_large = self.calculate_cluster_diameter(customers_subset) > self.max_distance_km
        
        return too_many_customers or diameter_too_large
    
    def split_cluster(self, customers_subset):
        """Split a cluster into two using K-means."""
        if len(customers_subset) <= 2:
            return [customers_subset]
        
        coords = customers_subset[['Latitude', 'Longitude']].values
        
        # Use K-means to split into 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        # Create two subclusters
        cluster_0 = customers_subset[labels == 0].copy()
        cluster_1 = customers_subset[labels == 1].copy()
        
        return [cluster_0, cluster_1]
    
    def divisive_clustering(self, customers_df):
        """Perform divisive hierarchical clustering."""
        customers_df = customers_df.copy().reset_index(drop=True)
        
        # Start with all customers in one cluster
        clusters_to_process = [customers_df]
        final_clusters = []
        
        while clusters_to_process:
            current_cluster = clusters_to_process.pop(0)
            
            if self.should_split_cluster(current_cluster):
                # Split the cluster
                subclusters = self.split_cluster(current_cluster)
                
                # Add subclusters back to processing queue
                for subcluster in subclusters:
                    if len(subcluster) > 0:
                        clusters_to_process.append(subcluster)
            else:
                # Cluster meets criteria, add to final results
                final_clusters.append(current_cluster)
        
        # Assign cluster IDs
        result_df = customers_df.copy()
        result_df['cluster'] = -1  # Initialize
        
        for cluster_id, cluster_df in enumerate(final_clusters, 1):
            result_df.loc[cluster_df.index, 'cluster'] = cluster_id
        
        return result_df
    






class OptimizedDivisiveGeographicClustering_DEF:
    def __init__(self, max_customers_per_cluster=20, max_distance_km=50, use_spatial_index=True):
        self.max_customers_per_cluster = max_customers_per_cluster
        self.max_distance_km = max_distance_km
        self.use_spatial_index = use_spatial_index
        # Pre-compute Earth's radius for haversine formula
        self.earth_radius_km = 6371.0
        self.spatial_index = None
        
    def haversine_vectorized(self, coords1, coords2=None):
        """
        Vectorized haversine distance calculation.
        If coords2 is None, calculates pairwise distances within coords1.
        Returns distances in kilometers.
        """
        if coords2 is None:
            # Pairwise distances within coords1
            coords1_rad = np.radians(coords1)
            lat1 = coords1_rad[:, 0]
            lon1 = coords1_rad[:, 1]
            
            # Create meshgrid for all pairs
            lat1_grid, lat2_grid = np.meshgrid(lat1, lat1, indexing='ij')
            lon1_grid, lon2_grid = np.meshgrid(lon1, lon1, indexing='ij')
            
            dlat = lat2_grid - lat1_grid
            dlon = lon2_grid - lon1_grid
        else:
            # Distance between two sets of coordinates
            coords1_rad = np.radians(coords1)
            coords2_rad = np.radians(coords2)
            
            lat1, lon1 = coords1_rad[:, 0], coords1_rad[:, 1]
            lat2, lon2 = coords2_rad[:, 0], coords2_rad[:, 1]
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            lat2_grid = lat2
            lat1_grid = lat1
        
        # Haversine formula
        a = (np.sin(dlat / 2) ** 2 + 
             np.cos(lat1_grid) * np.cos(lat2_grid) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to handle floating point errors
        
        return self.earth_radius_km * c
    
    def haversine_single_point(self, lat1, lon1, lat2, lon2):
        """Fast haversine calculation for single point pairs."""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (np.sin(dlat / 2) ** 2 + 
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
        return self.earth_radius_km * c
    
    def estimate_diameter_bounding_box(self, coords):
        """
        Fast diameter estimation using geographic bounding box.
        Much faster and more accurate than sampling for lat/lon data.
        """
        if len(coords) <= 1:
            return 0
        
        # Get bounding box coordinates
        lat_min, lat_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        lon_min, lon_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        
        # Handle longitude wrapping around date line
        lon_range = lon_max - lon_min
        if lon_range > 180:
            # Likely crossing date line, adjust calculation
            lon_adjusted = coords[:, 1].copy()
            lon_adjusted[lon_adjusted < 0] += 360
            lon_min_adj, lon_max_adj = np.min(lon_adjusted), np.max(lon_adjusted)
            if (lon_max_adj - lon_min_adj) < lon_range:
                lon_min, lon_max = lon_min_adj, lon_max_adj
                if lon_max > 360:
                    lon_max -= 360
                if lon_min > 360:
                    lon_min -= 360
        
        # Quick check for very small bounding boxes
        if (lat_max - lat_min) < 0.01 and (lon_max - lon_min) < 0.01:  # ~1.1km
            return self.haversine_single_point(lat_min, lon_min, lat_max, lon_max)
        
        # Calculate distances between bounding box corners and edges
        # This gives a good approximation of the maximum distance
        corner_distances = [
            self.haversine_single_point(lat_min, lon_min, lat_max, lon_max),  # Diagonal
            self.haversine_single_point(lat_min, lon_max, lat_max, lon_min),  # Other diagonal
            self.haversine_single_point(lat_min, lon_min, lat_max, lon_min),  # Vertical edge
            self.haversine_single_point(lat_min, lon_min, lat_min, lon_max),  # Horizontal edge
        ]
        
        max_corner_distance = max(corner_distances)
        
        # For small to medium clusters, this is usually accurate enough
        if len(coords) <= 50 or max_corner_distance > self.max_distance_km * 1.5:
            return max_corner_distance
        
        # For larger clusters where bounding box might be conservative,
        # sample a few extreme points for better accuracy
        return self._refine_diameter_with_extremes(coords, max_corner_distance)
    
    def _refine_diameter_with_extremes(self, coords, bounding_box_diameter):
        """Refine diameter estimate using extreme points in each direction."""
        # Find extreme points
        lat_min_idx = np.argmin(coords[:, 0])
        lat_max_idx = np.argmax(coords[:, 0])
        lon_min_idx = np.argmin(coords[:, 1])
        lon_max_idx = np.argmax(coords[:, 1])
        
        extreme_indices = list(set([lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx]))
        
        if len(extreme_indices) <= 1:
            return bounding_box_diameter
        
        # Calculate distances between extreme points
        extreme_coords = coords[extreme_indices]
        max_distance = 0
        
        for i in range(len(extreme_coords)):
            for j in range(i + 1, len(extreme_coords)):
                distance = self.haversine_single_point(
                    extreme_coords[i, 0], extreme_coords[i, 1],
                    extreme_coords[j, 0], extreme_coords[j, 1]
                )
                max_distance = max(max_distance, distance)
        
        return max(max_distance, bounding_box_diameter)
    
    def should_split_cluster(self, cluster_indices, coords_array):
        """Determine if a cluster should be split."""
        cluster_size = len(cluster_indices)
        
        # Quick check for size
        if cluster_size <= 2:
            return False
            
        if cluster_size > self.max_customers_per_cluster:
            return True
        
        # Check diameter using fast bounding box method
        cluster_coords = coords_array[cluster_indices]
        diameter = self.estimate_diameter_bounding_box(cluster_coords)
        
        return diameter > self.max_distance_km
    
    def split_cluster_geographic(self, cluster_indices, coords_array):
        """
        Split a cluster geographically using median splitting.
        Much faster than K-means and more appropriate for lat/lon data.
        """
        if len(cluster_indices) <= 2:
            return [cluster_indices]
        
        cluster_coords = coords_array[cluster_indices]
        
        # Calculate coordinate ranges
        lat_range = np.ptp(cluster_coords[:, 0])  # Peak-to-peak (max - min)
        lon_range = np.ptp(cluster_coords[:, 1])
        
        # Handle longitude wrapping near date line
        if lon_range > 180:
            lon_adjusted = cluster_coords[:, 1].copy()
            lon_adjusted[lon_adjusted < 0] += 360
            lon_range_adjusted = np.ptp(lon_adjusted)
            
            if lon_range_adjusted < lon_range:
                # Use adjusted longitude values
                lon_range = lon_range_adjusted
                split_coords = cluster_coords.copy()
                split_coords[:, 1] = lon_adjusted
            else:
                split_coords = cluster_coords
        else:
            split_coords = cluster_coords
        
        # Split along the dimension with larger range
        if lat_range >= lon_range:
            # Split by latitude
            median_lat = np.median(split_coords[:, 0])
            mask = split_coords[:, 0] <= median_lat
        else:
            # Split by longitude
            median_lon = np.median(split_coords[:, 1])
            mask = split_coords[:, 1] <= median_lon
        
        # Ensure both clusters have at least one point
        if np.all(mask) or np.all(~mask):
            # Fallback: split roughly in half by sorting
            if lat_range >= lon_range:
                sort_indices = np.argsort(split_coords[:, 0])
            else:
                sort_indices = np.argsort(split_coords[:, 1])
            
            mid_point = len(cluster_indices) // 2
            mask = np.zeros(len(cluster_indices), dtype=bool)
            mask[sort_indices[:mid_point]] = True
        
        cluster_0_indices = cluster_indices[mask]
        cluster_1_indices = cluster_indices[~mask]
        
        return [cluster_0_indices, cluster_1_indices]
    
    def build_spatial_index(self, coords_array):
        """Build spatial index for faster queries (optional optimization)."""
        if self.use_spatial_index and len(coords_array) > 100:
            try:
                # Convert to approximate Cartesian for KD-tree
                # This is just for spatial indexing, not distance calculation
                self.spatial_index = cKDTree(coords_array)
            except:
                self.spatial_index = None
        else:
            self.spatial_index = None
    
    def divisive_clustering(self, customers_df):
        """Perform optimized divisive hierarchical clustering."""
        customers_df = customers_df.copy().reset_index(drop=True)
        
        # Convert coordinates to numpy array once for faster access
        coords_array = customers_df[['Latitude', 'Longitude']].values
        n_customers = len(customers_df)
        
        # Build spatial index if requested
        self.build_spatial_index(coords_array)
        
        # Work with indices instead of copying DataFrames
        clusters_to_process = [np.arange(n_customers)]
        final_clusters = []
        
        # Process clusters iteratively
        iteration_count = 0
        max_iterations = n_customers * 2  # Safety limit
        
        while clusters_to_process and iteration_count < max_iterations:
            current_cluster_indices = clusters_to_process.pop(0)
            iteration_count += 1
            
            if self.should_split_cluster(current_cluster_indices, coords_array):
                # Split the cluster using geographic method
                subclusters = self.split_cluster_geographic(current_cluster_indices, coords_array)
                
                # Add non-empty subclusters back to processing queue
                for subcluster_indices in subclusters:
                    if len(subcluster_indices) > 0:
                        clusters_to_process.append(subcluster_indices)
            else:
                # Cluster meets criteria, add to final results
                final_clusters.append(current_cluster_indices)
        
        # Handle any remaining clusters (safety check)
        final_clusters.extend(clusters_to_process)
        
        # Create result DataFrame with cluster assignments
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster_indices in enumerate(final_clusters, 1):
            result_df.loc[cluster_indices, 'cluster'] = cluster_id
        
        return result_df
    
    def get_cluster_stats(self, result_df):
        """Calculate statistics for the resulting clusters."""
        coords_array = result_df[['Latitude', 'Longitude']].values
        stats = []
        
        for cluster_id in sorted(result_df['cluster'].unique()):
            if cluster_id == -1:
                continue
                
            cluster_mask = result_df['cluster'] == cluster_id
            cluster_coords = coords_array[cluster_mask]
            cluster_size = len(cluster_coords)
            
            if cluster_size > 1:
                diameter = self.estimate_diameter_bounding_box(cluster_coords)
            else:
                diameter = 0
            
            stats.append({
                'cluster_id': cluster_id,
                'size': cluster_size,
                'diameter_km': diameter,
                'meets_size_constraint': cluster_size <= self.max_customers_per_cluster,
                'meets_distance_constraint': diameter <= self.max_distance_km
            })
        
        return pd.DataFrame(stats)


class OptimizedDivisiveGeographicClustering_SLOW:
    def __init__(self, max_customers_per_cluster=20, max_distance_km=50):
        self.max_customers_per_cluster = max_customers_per_cluster
        self.max_distance_km = max_distance_km
        # Pre-compute Earth's radius for haversine formula
        self.earth_radius_km = 6371.0
        
    def haversine_vectorized(self, coords1, coords2=None):
        """
        Vectorized haversine distance calculation.
        If coords2 is None, calculates pairwise distances within coords1.
        Returns distances in kilometers.
        """
        if coords2 is None:
            # Pairwise distances within coords1
            coords1_rad = np.radians(coords1)
            lat1 = coords1_rad[:, 0]
            lon1 = coords1_rad[:, 1]
            
            # Create meshgrid for all pairs
            lat1_grid, lat2_grid = np.meshgrid(lat1, lat1, indexing='ij')
            lon1_grid, lon2_grid = np.meshgrid(lon1, lon1, indexing='ij')
            
            dlat = lat2_grid - lat1_grid
            dlon = lon2_grid - lon1_grid
        else:
            # Distance between two sets of coordinates
            coords1_rad = np.radians(coords1)
            coords2_rad = np.radians(coords2)
            
            lat1, lon1 = coords1_rad[:, 0], coords1_rad[:, 1]
            lat2, lon2 = coords2_rad[:, 0], coords2_rad[:, 1]
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            lat2_grid = lat2
            lat1_grid = lat1
        
        # Haversine formula
        a = (np.sin(dlat / 2) ** 2 + 
             np.cos(lat1_grid) * np.cos(lat2_grid) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to handle floating point errors
        
        return self.earth_radius_km * c
    
    def calculate_cluster_diameter_fast(self, coords):
        """
        Fast cluster diameter calculation using vectorized haversine.
        Returns early if diameter exceeds threshold.
        """
        if len(coords) <= 1:
            return 0
        
        # For small clusters, use full pairwise calculation
        if len(coords) <= 100:
            distances = self.haversine_vectorized(coords)
            # Get upper triangle (excluding diagonal) to avoid duplicate pairs
            upper_triangle = np.triu(distances, k=1)
            return np.max(upper_triangle)
        
        # For larger clusters, use approximation with early termination
        return self._approximate_diameter_with_early_termination(coords)
    
    def _approximate_diameter_with_early_termination(self, coords):
        """
        Approximate diameter calculation for large clusters with early termination.
        Uses sampling and convex hull heuristics.
        """
        n_points = len(coords)
        
        # Quick check: calculate distances from centroid to all points
        centroid = np.mean(coords, axis=0, keepdims=True)
        distances_from_centroid = self.haversine_vectorized(centroid, coords).flatten()
        max_from_centroid = np.max(distances_from_centroid)
        
        # If 2 * max distance from centroid is already below threshold, cluster is small
        if max_from_centroid * 2 < self.max_distance_km * 0.8:
            return max_from_centroid * 2
        
        # Find the two points that are farthest from centroid
        farthest_indices = np.argpartition(distances_from_centroid, -min(10, n_points))[-min(10, n_points):]
        extreme_points = coords[farthest_indices]
        
        # Calculate pairwise distances among extreme points
        if len(extreme_points) > 1:
            extreme_distances = self.haversine_vectorized(extreme_points)
            upper_triangle = np.triu(extreme_distances, k=1)
            max_extreme_distance = np.max(upper_triangle)
            
            # If extreme points already exceed threshold, return early
            if max_extreme_distance > self.max_distance_km:
                return max_extreme_distance
        
        # Sample additional points for more accurate estimation
        if n_points > 50:
            sample_size = min(50, n_points)
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            sample_coords = coords[sample_indices]
            sample_distances = self.haversine_vectorized(sample_coords)
            upper_triangle = np.triu(sample_distances, k=1)
            return np.max(upper_triangle)
        else:
            # For medium-sized clusters, do full calculation
            distances = self.haversine_vectorized(coords)
            upper_triangle = np.triu(distances, k=1)
            return np.max(upper_triangle)
    
    def should_split_cluster(self, cluster_indices, coords_array):
        """Determine if a cluster should be split."""
        cluster_size = len(cluster_indices)
        
        # Quick check for size
        if cluster_size <= 2:
            return False
            
        if cluster_size > self.max_customers_per_cluster:
            return True
        
        # Check diameter only if size constraint is not violated
        cluster_coords = coords_array[cluster_indices]
        diameter = self.calculate_cluster_diameter_fast(cluster_coords)
        
        return diameter > self.max_distance_km
    
    def split_cluster_fast(self, cluster_indices, coords_array):
        """Split a cluster into two using K-means with optimizations."""
        if len(cluster_indices) <= 2:
            return [cluster_indices]
        
        cluster_coords = coords_array[cluster_indices]
        
        # Use K-means with optimized parameters for geographic data
        kmeans = KMeans(
            n_clusters=2, 
            random_state=42, 
            n_init=5,  # Reduced from 10 for speed
            max_iter=100,  # Reduced iterations
            tol=1e-3  # Slightly relaxed tolerance
        )
        labels = kmeans.fit_predict(cluster_coords)
        
        # Split indices based on labels
        cluster_0_indices = cluster_indices[labels == 0]
        cluster_1_indices = cluster_indices[labels == 1]
        
        return [cluster_0_indices, cluster_1_indices]
    
    def divisive_clustering(self, customers_df):
        """Perform optimized divisive hierarchical clustering."""
        customers_df = customers_df.copy().reset_index(drop=True)
        
        # Convert coordinates to numpy array once for faster access
        coords_array = customers_df[['Latitude', 'Longitude']].values
        n_customers = len(customers_df)
        
        # Work with indices instead of copying DataFrames
        clusters_to_process = [np.arange(n_customers)]
        final_clusters = []
        
        # Process clusters iteratively
        while clusters_to_process:
            current_cluster_indices = clusters_to_process.pop(0)
            
            if self.should_split_cluster(current_cluster_indices, coords_array):
                # Split the cluster
                subclusters = self.split_cluster_fast(current_cluster_indices, coords_array)
                
                # Add non-empty subclusters back to processing queue
                for subcluster_indices in subclusters:
                    if len(subcluster_indices) > 0:
                        clusters_to_process.append(subcluster_indices)
            else:
                # Cluster meets criteria, add to final results
                final_clusters.append(current_cluster_indices)
        
        # Create result DataFrame with cluster assignments
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster_indices in enumerate(final_clusters, 1):
            result_df.loc[cluster_indices, 'cluster'] = cluster_id
        
        return result_df

# Example usage and performance comparison
def compare_performance():
    """Function to compare performance between original and optimized versions."""
    import time
    
    # Generate sample data
    np.random.seed(42)
    n_customers = 1000
    
    # Create realistic geographic distribution (clustered around several centers)
    centers = [(40.7128, -74.0060), (34.0522, -118.2437), (41.8781, -87.6298)]  # NYC, LA, Chicago
    customers_data = []
    
    for i in range(n_customers):
        center_idx = np.random.choice(len(centers))
        center_lat, center_lon = centers[center_idx]
        
        # Add random offset around center
        lat_offset = np.random.normal(0, 0.5)  # ~55km std dev
        lon_offset = np.random.normal(0, 0.5)
        
        customers_data.append({
            'CustomerID': f'C{i+1:04d}',
            'Latitude': center_lat + lat_offset,
            'Longitude': center_lon + lon_offset
        })
    
    customers_df = pd.DataFrame(customers_data)
    
    # Test optimized version
    print("Testing Optimized Version...")
    start_time = time.time()
    optimized_clusterer = OptimizedDivisiveGeographicClustering()
    optimized_result = optimized_clusterer.divisive_clustering(customers_df)
    optimized_time = time.time() - start_time
    
    print(f"Optimized clustering completed in {optimized_time:.2f} seconds")
    print(f"Number of clusters: {optimized_result['cluster'].nunique()}")
    print(f"Customers per cluster (avg): {len(optimized_result) / optimized_result['cluster'].nunique():.1f}")
    
    return optimized_result, optimized_time

# Uncomment to run performance test
# if __name__ == "__main__":
#     result, time_taken = compare_performance()



























