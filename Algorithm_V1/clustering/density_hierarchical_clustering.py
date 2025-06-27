import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import pandas as pd

class DensityHierarchicalClustering:
    def __init__(self, max_customers_per_cluster=20, min_cluster_size=3, 
                 k_neighbors=5):
        self.max_customers_per_cluster = max_customers_per_cluster
        self.min_cluster_size = min_cluster_size
        self.k_neighbors = k_neighbors
    
    def calculate_local_density(self, customers_df, k=None):
        """Calculate local density for each customer based on k-nearest neighbors."""
        if k is None:
            k = self.k_neighbors
        
        coords = customers_df[['Latitude', 'Longitude']].values
        
        # Convert to approximate Cartesian coordinates for faster computation
        # (This is less accurate but much faster than geodesic distance)
        coords_cart = np.column_stack([
            coords[:, 1] * 111.32 * np.cos(np.radians(coords[:, 0])),  # longitude to km
            coords[:, 0] * 110.54  # latitude to km
        ])
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nbrs.fit(coords_cart)
        distances, indices = nbrs.kneighbors(coords_cart)
        
        # Calculate density as inverse of average distance to k-nearest neighbors
        # (excluding the point itself)
        densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
        
        return densities, indices
    
    def build_density_hierarchy(self, customers_df):
        """Build hierarchy based on density connectivity."""
        customers_df = customers_df.copy().reset_index(drop=True)
        n_customers = len(customers_df)
        
        # Calculate local densities
        densities, neighbor_indices = self.calculate_local_density(customers_df)
        
        # Sort customers by density (highest first)
        density_order = np.argsort(densities)[::-1]
        
        # Build clusters starting from highest density points
        clusters = []
        assigned = np.zeros(n_customers, dtype=bool)
        
        for seed_idx in density_order:
            if assigned[seed_idx]:
                continue
            
            # Start new cluster with this seed
            current_cluster = [seed_idx]
            assigned[seed_idx] = True
            
            # Grow cluster by adding nearby unassigned points
            to_check = [seed_idx]
            
            while to_check and len(current_cluster) < self.max_customers_per_cluster:
                current_idx = to_check.pop(0)
                
                # Check neighbors of current point
                for neighbor_idx in neighbor_indices[current_idx][1:]:  # Skip self
                    if (not assigned[neighbor_idx] and 
                        len(current_cluster) < self.max_customers_per_cluster):
                        
                        # Add neighbor if density is high enough
                        density_ratio = densities[neighbor_idx] / densities[seed_idx]
                        if density_ratio > 0.5:  # Adjustable threshold
                            current_cluster.append(neighbor_idx)
                            assigned[neighbor_idx] = True
                            to_check.append(neighbor_idx)
            
            # Only keep clusters that meet minimum size requirement
            if len(current_cluster) >= self.min_cluster_size:
                clusters.append(current_cluster)
        
        # Handle remaining unassigned points
        unassigned_indices = np.where(~assigned)[0]
        if len(unassigned_indices) > 0:
            # Assign to nearest existing cluster or create new ones
            for idx in unassigned_indices:
                # Find nearest cluster that has space
                best_cluster = None
                min_distance = float('inf')
                
                for i, cluster in enumerate(clusters):
                    if len(cluster) < self.max_customers_per_cluster:
                        # Calculate distance to cluster centroid
                        cluster_coords = customers_df.iloc[cluster][['Latitude', 'Longitude']].values
                        centroid = np.mean(cluster_coords, axis=0)
                        point_coord = customers_df.iloc[idx][['Latitude', 'Longitude']].values
                        
                        dist = np.sqrt(np.sum((point_coord - centroid) ** 2))
                        if dist < min_distance:
                            min_distance = dist
                            best_cluster = i
                
                if best_cluster is not None:
                    clusters[best_cluster].append(idx)
                else:
                    # Create new single-point cluster
                    clusters.append([idx])
        
        # Assign cluster labels
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster_indices in enumerate(clusters, 1):
            for idx in cluster_indices:
                result_df.loc[idx, 'cluster'] = cluster_id
        
        return result_df






import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from collections import defaultdict
import time

class OptimizedDensityHierarchicalClustering:
    def __init__(self, max_customers_per_cluster=20, min_cluster_size=3, 
                 k_neighbors=5, density_threshold=0.5):
        self.max_customers_per_cluster = max_customers_per_cluster
        self.min_cluster_size = min_cluster_size
        self.k_neighbors = k_neighbors
        self.density_threshold = density_threshold
        
        # Pre-compute conversion factors for lat/lon to km
        self.lat_to_km = 110.54  # km per degree latitude
        self.lon_to_km_factor = 111.32  # km per degree longitude at equator
    
    def latlon_to_cartesian_vectorized(self, coords):
        """
        Vectorized conversion from lat/lon to approximate Cartesian coordinates.
        Much faster than individual conversions.
        """
        lat_rad = np.radians(coords[:, 0])
        lon_km = coords[:, 1] * self.lon_to_km_factor * np.cos(lat_rad)
        lat_km = coords[:, 0] * self.lat_to_km
        
        return np.column_stack([lon_km, lat_km])
    
    def calculate_local_density_optimized(self, customers_df, k=None):
        """
        Optimized local density calculation with vectorized operations.
        """
        if k is None:
            k = self.k_neighbors
        
        coords = customers_df[['Latitude', 'Longitude']].values
        
        # Vectorized coordinate conversion
        coords_cart = self.latlon_to_cartesian_vectorized(coords)
        
        # Use ball_tree for faster neighbor search on geographic data
        nbrs = NearestNeighbors(
            n_neighbors=min(k+1, len(coords)), 
            metric='euclidean',
            algorithm='ball_tree'  # Better for geographic data
        )
        nbrs.fit(coords_cart)
        distances, indices = nbrs.kneighbors(coords_cart)
        
        # Vectorized density calculation
        # Add small epsilon to prevent division by zero
        mean_distances = np.mean(distances[:, 1:], axis=1)
        densities = 1.0 / (mean_distances + 1e-10)
        
        return densities, indices, coords_cart
    
    def build_density_hierarchy_optimized(self, customers_df):
        """
        Optimized density-based hierarchical clustering with significant speedups.
        """
        customers_df = customers_df.copy().reset_index(drop=True)
        n_customers = len(customers_df)
        
        if n_customers == 0:
            return customers_df
        
        # Calculate local densities
        densities, neighbor_indices, coords_cart = self.calculate_local_density_optimized(customers_df)
        
        # Sort customers by density (highest first)
        density_order = np.argsort(densities)[::-1]
        
        # Pre-allocate arrays for faster operations
        clusters = []
        assigned = np.zeros(n_customers, dtype=bool)
        cluster_assignments = np.full(n_customers, -1, dtype=int)
        
        # Build clusters starting from highest density points
        for seed_idx in density_order:
            if assigned[seed_idx]:
                continue
            
            # Start new cluster with this seed
            current_cluster = [seed_idx]
            assigned[seed_idx] = True
            seed_density = densities[seed_idx]
            
            # Use a more efficient queue-based approach
            to_check = [seed_idx]
            
            while to_check and len(current_cluster) < self.max_customers_per_cluster:
                current_idx = to_check.pop(0)
                current_neighbors = neighbor_indices[current_idx][1:]  # Skip self
                
                # Vectorized density check for all neighbors at once
                unassigned_neighbors = current_neighbors[~assigned[current_neighbors]]
                
                if len(unassigned_neighbors) > 0:
                    # Check density ratios for all unassigned neighbors
                    neighbor_densities = densities[unassigned_neighbors]
                    density_ratios = neighbor_densities / seed_density
                    
                    # Select qualified neighbors
                    qualified_mask = density_ratios > self.density_threshold
                    qualified_neighbors = unassigned_neighbors[qualified_mask]
                    
                    # Add qualified neighbors up to cluster size limit
                    available_space = self.max_customers_per_cluster - len(current_cluster)
                    neighbors_to_add = qualified_neighbors[:available_space]
                    
                    # Add neighbors to cluster
                    current_cluster.extend(neighbors_to_add)
                    assigned[neighbors_to_add] = True
                    to_check.extend(neighbors_to_add)
            
            # Only keep clusters that meet minimum size requirement
            if len(current_cluster) >= self.min_cluster_size:
                cluster_id = len(clusters)
                clusters.append(current_cluster)
                cluster_assignments[current_cluster] = cluster_id
        
        # Optimized handling of remaining unassigned points
        unassigned_indices = np.where(~assigned)[0]
        
        if len(unassigned_indices) > 0 and len(clusters) > 0:
            # Vectorized assignment of unassigned points
            self._assign_remaining_points_vectorized(
                unassigned_indices, clusters, coords_cart, cluster_assignments
            )
        
        # Handle case where no clusters were formed or points remain unassigned
        still_unassigned = np.where(cluster_assignments == -1)[0]
        for idx in still_unassigned:
            clusters.append([idx])
            cluster_assignments[idx] = len(clusters) - 1
        
        # Create result DataFrame
        result_df = customers_df.copy()
        result_df['cluster'] = cluster_assignments + 1  # 1-indexed clusters
        
        return result_df
    
    def _assign_remaining_points_vectorized(self, unassigned_indices, clusters, coords_cart, cluster_assignments):
        """
        Vectorized assignment of remaining unassigned points to existing clusters.
        """
        if len(unassigned_indices) == 0 or len(clusters) == 0:
            return
        
        # Get coordinates of unassigned points
        unassigned_coords = coords_cart[unassigned_indices]
        
        # Calculate cluster centroids
        cluster_centroids = []
        cluster_spaces = []
        
        for i, cluster in enumerate(clusters):
            cluster_coords = coords_cart[cluster]
            centroid = np.mean(cluster_coords, axis=0)
            cluster_centroids.append(centroid)
            cluster_spaces.append(self.max_customers_per_cluster - len(cluster))
        
        cluster_centroids = np.array(cluster_centroids)
        cluster_spaces = np.array(cluster_spaces)
        
        # Find clusters with available space
        available_clusters = np.where(cluster_spaces > 0)[0]
        
        if len(available_clusters) == 0:
            return
        
        available_centroids = cluster_centroids[available_clusters]
        
        # Calculate distances from each unassigned point to each available cluster centroid
        distances = cdist(unassigned_coords, available_centroids, metric='euclidean')
        
        # Assign each point to the nearest cluster with space
        for i, point_idx in enumerate(unassigned_indices):
            # Find nearest available cluster
            point_distances = distances[i]
            nearest_cluster_idx = available_clusters[np.argmin(point_distances)]
            
            # Check if cluster still has space
            if cluster_spaces[nearest_cluster_idx] > 0:
                clusters[nearest_cluster_idx].append(point_idx)
                cluster_assignments[point_idx] = nearest_cluster_idx
                cluster_spaces[nearest_cluster_idx] -= 1
                
                # Remove cluster from available list if it's now full
                if cluster_spaces[nearest_cluster_idx] == 0:
                    mask = available_clusters != nearest_cluster_idx
                    available_clusters = available_clusters[mask]
                    available_centroids = available_centroids[mask]
                    distances = distances[:, mask]
    
    def build_density_hierarchy_adaptive(self, customers_df):
        """
        Adaptive clustering that adjusts parameters based on data characteristics.
        """
        customers_df = customers_df.copy().reset_index(drop=True)
        n_customers = len(customers_df)
        
        if n_customers == 0:
            return customers_df
        
        # Adapt k_neighbors based on dataset size
        adaptive_k = min(self.k_neighbors, max(3, n_customers // 20))
        
        # Calculate local densities with adaptive k
        densities, neighbor_indices, coords_cart = self.calculate_local_density_optimized(customers_df, k=adaptive_k)
        
        # Adaptive density threshold based on density distribution
        density_percentiles = np.percentile(densities, [25, 50, 75])
        adaptive_threshold = max(0.3, min(0.8, density_percentiles[0] / density_percentiles[2]))
        
        # Temporarily adjust threshold
        original_threshold = self.density_threshold
        self.density_threshold = adaptive_threshold
        
        # Use the optimized clustering with adaptive parameters
        result_df = self.build_density_hierarchy_optimized(customers_df)
        
        # Restore original threshold
        self.density_threshold = original_threshold
        
        return result_df

class FastDensityClusteringVariants:
    """
    Additional fast clustering variants for different use cases.
    """
    
    @staticmethod
    def simple_density_clustering(customers_df, max_cluster_size=20, k_neighbors=5):
        """
        Ultra-fast simplified density clustering for large datasets.
        """
        coords = customers_df[['Latitude', 'Longitude']].values
        n_customers = len(coords)
        
        if n_customers == 0:
            return customers_df.copy()
        
        # Convert to Cartesian coordinates
        lat_rad = np.radians(coords[:, 0])
        coords_cart = np.column_stack([
            coords[:, 1] * 111.32 * np.cos(lat_rad),
            coords[:, 0] * 110.54
        ])
        
        # Fast neighbor search
        nbrs = NearestNeighbors(
            n_neighbors=min(k_neighbors + 1, n_customers),
            algorithm='kd_tree'
        )
        nbrs.fit(coords_cart)
        distances, indices = nbrs.kneighbors(coords_cart)
        
        # Simple density-based clustering
        densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
        density_order = np.argsort(densities)[::-1]
        
        clusters = []
        assigned = np.zeros(n_customers, dtype=bool)
        
        for seed_idx in density_order:
            if assigned[seed_idx] or len(clusters) * max_cluster_size >= n_customers:
                break
            
            # Quick cluster formation
            cluster = [seed_idx]
            assigned[seed_idx] = True
            
            # Add nearest unassigned neighbors
            for neighbor_idx in indices[seed_idx][1:]:
                if not assigned[neighbor_idx] and len(cluster) < max_cluster_size:
                    cluster.append(neighbor_idx)
                    assigned[neighbor_idx] = True
            
            clusters.append(cluster)
        
        # Assign remaining points to nearest clusters
        unassigned = np.where(~assigned)[0]
        for idx in unassigned:
            if clusters:
                # Find nearest cluster
                best_cluster = 0
                min_dist = float('inf')
                
                for i, cluster in enumerate(clusters):
                    if len(cluster) < max_cluster_size:
                        cluster_coords = coords_cart[cluster]
                        centroid = np.mean(cluster_coords, axis=0)
                        dist = np.linalg.norm(coords_cart[idx] - centroid)
                        if dist < min_dist:
                            min_dist = dist
                            best_cluster = i
                
                if len(clusters[best_cluster]) < max_cluster_size:
                    clusters[best_cluster].append(idx)
                else:
                    clusters.append([idx])
            else:
                clusters.append([idx])
        
        # Create result
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster in enumerate(clusters, 1):
            for idx in cluster:
                result_df.loc[idx, 'cluster'] = cluster_id
        
        return result_df

# Performance testing and comparison
def compare_density_clustering_performance():
    """
    Compare performance of different density clustering approaches.
    """
    np.random.seed(42)
    
    # Generate test data with multiple density regions
    n_customers = 1000
    centers = [
        (40.7128, -74.0060),  # NYC - high density
        (34.0522, -118.2437), # LA - medium density  
        (41.8781, -87.6298),  # Chicago - high density
        (29.7604, -95.3698),  # Houston - low density
    ]
    
    customers_data = []
    densities = [0.1, 0.2, 0.1, 0.3]  # Different density levels
    
    for i in range(n_customers):
        center_idx = np.random.choice(len(centers), p=[0.3, 0.3, 0.2, 0.2])
        center_lat, center_lon = centers[center_idx]
        density_factor = densities[center_idx]
        
        # Create varying density by adjusting spread
        lat_offset = np.random.normal(0, density_factor)
        lon_offset = np.random.normal(0, density_factor)
        
        customers_data.append({
            'CustomerID': f'C{i+1:04d}',
            'Latitude': center_lat + lat_offset,
            'Longitude': center_lon + lon_offset
        })
    
    customers_df = pd.DataFrame(customers_data)
    
    print(f"Testing with {len(customers_df)} customers...")
    
    # Test optimized density clustering
    print("\n1. Optimized Density Hierarchical Clustering:")
    clusterer = OptimizedDensityHierarchicalClustering()
    
    start_time = time.time()
    result_optimized = clusterer.build_density_hierarchy_optimized(customers_df)
    optimized_time = time.time() - start_time
    
    print(f"   Time: {optimized_time:.3f} seconds")
    print(f"   Clusters: {result_optimized['cluster'].nunique()}")
    print(f"   Avg cluster size: {len(result_optimized) / result_optimized['cluster'].nunique():.1f}")
    
    # Test adaptive density clustering
    print("\n2. Adaptive Density Clustering:")
    start_time = time.time()
    result_adaptive = clusterer.build_density_hierarchy_adaptive(customers_df)
    adaptive_time = time.time() - start_time
    
    print(f"   Time: {adaptive_time:.3f} seconds")
    print(f"   Clusters: {result_adaptive['cluster'].nunique()}")
    print(f"   Avg cluster size: {len(result_adaptive) / result_adaptive['cluster'].nunique():.1f}")
    
    # Test ultra-fast simple clustering
    print("\n3. Ultra-Fast Simple Density Clustering:")
    start_time = time.time()
    result_simple = FastDensityClusteringVariants.simple_density_clustering(customers_df)
    simple_time = time.time() - start_time
    
    print(f"   Time: {simple_time:.3f} seconds")
    print(f"   Clusters: {result_simple['cluster'].nunique()}")
    print(f"   Avg cluster size: {len(result_simple) / result_simple['cluster'].nunique():.1f}")
    
    print(f"\nPerformance Summary:")
    print(f"   Optimized: {optimized_time:.3f}s")
    print(f"   Adaptive:  {adaptive_time:.3f}s")
    print(f"   Simple:    {simple_time:.3f}s")
    
    return result_optimized, result_adaptive, result_simple

# Uncomment to run performance test
# if __name__ == "__main__":
#     import pandas as pd
#     optimized, adaptive, simple = compare_density_clustering_performance()



