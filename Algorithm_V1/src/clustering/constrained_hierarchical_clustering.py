import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.neighbors import NearestNeighbors 
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import heapq



class ConstraintBasedClustering:
    def __init__(self, max_customers_per_cluster=20, max_distance_km=50, 
                 max_route_distance_km=100):
        self.max_customers_per_cluster = max_customers_per_cluster
        self.max_distance_km = max_distance_km
        self.max_route_distance_km = max_route_distance_km
    
    def calculate_route_distance(self, customers_subset):
        """Calculate approximate route distance using nearest neighbor heuristic."""
        if len(customers_subset) <= 1:
            return 0
        
        coords = customers_subset[['Latitude', 'Longitude']].values
        
        # Simple nearest neighbor tour approximation
        unvisited = set(range(len(coords)))
        current = 0
        unvisited.remove(current)
        total_distance = 0
        
        while unvisited:
            nearest_idx = min(unvisited, 
                            key=lambda x: geodesic(coords[current], coords[x]).kilometers)
            total_distance += geodesic(coords[current], coords[nearest_idx]).kilometers
            current = nearest_idx
            unvisited.remove(current)
        
        return total_distance
    
    def is_feasible_cluster(self, customers_subset):
        """Check if cluster meets all constraints."""
        # Check customer count constraint
        if len(customers_subset) > self.max_customers_per_cluster:
            return False
        
        # Check diameter constraint
        diameter = self.calculate_cluster_diameter(customers_subset)
        if diameter > self.max_distance_km:
            return False
        
        # Check route distance constraint
        route_distance = self.calculate_route_distance(customers_subset)
        if route_distance > self.max_route_distance_km:
            return False
        
        return True
    
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
    
    def can_merge_clusters(self, cluster1, cluster2):
        """Check if two clusters can be merged while respecting constraints."""
        merged = pd.concat([cluster1, cluster2], ignore_index=True)
        return self.is_feasible_cluster(merged)
    
    def cluster_distance(self, cluster1, cluster2):
        """Calculate distance between two clusters (minimum distance between points)."""
        coords1 = cluster1[['Latitude', 'Longitude']].values
        coords2 = cluster2[['Latitude', 'Longitude']].values
        
        min_dist = float('inf')
        for coord1 in coords1:
            for coord2 in coords2:
                dist = geodesic(coord1, coord2).kilometers
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def constrained_hierarchical_clustering(self, customers_df):
        """Perform constraint-aware hierarchical clustering."""
        customers_df = customers_df.copy().reset_index(drop=True)
        
        # Start with each customer as its own cluster
        clusters = [pd.DataFrame([row]) for _, row in customers_df.iterrows()]
        
        # Merge clusters while respecting constraints
        changed = True
        while changed and len(clusters) > 1:
            changed = False
            best_merge = None
            best_distance = float('inf')
            
            # Find the best pair of clusters to merge
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if self.can_merge_clusters(clusters[i], clusters[j]):
                        dist = self.cluster_distance(clusters[i], clusters[j])
                        if dist < best_distance:
                            best_distance = dist
                            best_merge = (i, j)
            
            # Perform the best merge
            if best_merge is not None:
                i, j = best_merge
                merged_cluster = pd.concat([clusters[i], clusters[j]], ignore_index=True)
                
                # Remove the original clusters and add the merged one
                clusters = [clusters[k] for k in range(len(clusters)) if k not in [i, j]]
                clusters.append(merged_cluster)
                changed = True
        
        # Assign cluster IDs
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster_df in enumerate(clusters, 1):
            for idx in cluster_df.index:
                result_df.loc[idx, 'cluster'] = cluster_id
        
        return result_df
    




class OptimizedConstraintBasedClustering:
    def __init__(self, max_customers_per_cluster=20, max_distance_km=50, 
                 max_route_distance_km=100):
        self.max_customers_per_cluster = max_customers_per_cluster
        self.max_distance_km = max_distance_km
        self.max_route_distance_km = max_route_distance_km
        self.earth_radius_km = 6371.0
        
        # Cache for expensive calculations
        self._distance_cache = {}
        self._route_cache = {}
        self._diameter_cache = {}
    
    def haversine_vectorized(self, coords1, coords2=None):
        """
        Vectorized haversine distance calculation.
        Much faster than geopy.distance.geodesic.
        """
        if coords2 is None:
            coords2 = coords1
            
        coords1_rad = np.radians(coords1)
        coords2_rad = np.radians(coords2)
        
        if coords1_rad.ndim == 1:
            coords1_rad = coords1_rad.reshape(1, -1)
        if coords2_rad.ndim == 1:
            coords2_rad = coords2_rad.reshape(1, -1)
        
        lat1, lon1 = coords1_rad[:, 0:1], coords1_rad[:, 1:2]
        lat2, lon2 = coords2_rad[:, 0:1], coords2_rad[:, 1:2]
        
        dlat = lat2.T - lat1
        dlon = lon2.T - lon1
        
        a = (np.sin(dlat / 2) ** 2 + 
             np.cos(lat1) * np.cos(lat2.T) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
        return self.earth_radius_km * c
    
    def calculate_route_distance_fast(self, cluster_indices, coords_array):
        """
        Fast route distance calculation using vectorized operations and caching.
        """
        cluster_key = tuple(sorted(cluster_indices))
        if cluster_key in self._route_cache:
            return self._route_cache[cluster_key]
        
        if len(cluster_indices) <= 1:
            self._route_cache[cluster_key] = 0
            return 0
        
        coords = coords_array[cluster_indices]
        
        if len(coords) == 2:
            # Direct distance for 2 points
            distance = self.haversine_vectorized(coords[0:1], coords[1:2])[0, 0]
            self._route_cache[cluster_key] = distance
            return distance
        
        # For larger clusters, use optimized nearest neighbor approximation
        total_distance = self._nearest_neighbor_tour_fast(coords)
        self._route_cache[cluster_key] = total_distance
        return total_distance
    
    def _nearest_neighbor_tour_fast(self, coords):
        """
        Fast nearest neighbor tour using vectorized distance calculations.
        """
        if len(coords) <= 2:
            if len(coords) == 2:
                return self.haversine_vectorized(coords[0:1], coords[1:2])[0, 0]
            return 0
        
        # Calculate all pairwise distances at once
        distances = self.haversine_vectorized(coords, coords)
        np.fill_diagonal(distances, np.inf)  # Prevent self-selection
        
        # Greedy nearest neighbor tour
        unvisited = set(range(len(coords)))
        current = 0
        unvisited.remove(current)
        total_distance = 0
        
        while unvisited:
            # Find nearest unvisited point
            nearest_distances = distances[current, list(unvisited)]
            nearest_local_idx = np.argmin(nearest_distances)
            nearest_idx = list(unvisited)[nearest_local_idx]
            
            total_distance += distances[current, nearest_idx]
            current = nearest_idx
            unvisited.remove(current)
        
        return total_distance
    
    def calculate_cluster_diameter_fast(self, cluster_indices, coords_array):
        """
        Fast diameter calculation with caching and early termination.
        """
        cluster_key = tuple(sorted(cluster_indices))
        if cluster_key in self._diameter_cache:
            return self._diameter_cache[cluster_key]
        
        if len(cluster_indices) <= 1:
            self._diameter_cache[cluster_key] = 0
            return 0
        
        coords = coords_array[cluster_indices]
        
        # For small clusters, calculate exactly
        if len(coords) <= 50:
            distances = self.haversine_vectorized(coords, coords)
            diameter = np.max(distances)
        else:
            # For large clusters, use approximation with early termination
            diameter = self._approximate_diameter_fast(coords)
        
        self._diameter_cache[cluster_key] = diameter
        return diameter
    
    def _approximate_diameter_fast(self, coords):
        """
        Fast diameter approximation for large clusters.
        """
        # Use convex hull approach for better approximation
        try:
            from scipy.spatial import ConvexHull
            if len(coords) > 4:
                hull = ConvexHull(coords)
                hull_coords = coords[hull.vertices]
                distances = self.haversine_vectorized(hull_coords, hull_coords)
                return np.max(distances)
        except:
            pass
        
        # Fallback: sample-based approximation
        n_samples = min(30, len(coords))
        if len(coords) > n_samples:
            indices = np.random.choice(len(coords), n_samples, replace=False)
            sample_coords = coords[indices]
        else:
            sample_coords = coords
        
        distances = self.haversine_vectorized(sample_coords, sample_coords)
        return np.max(distances)
    
    def cluster_distance_fast(self, cluster1_indices, cluster2_indices, coords_array):
        """
        Fast minimum distance between two clusters.
        """
        coords1 = coords_array[cluster1_indices]
        coords2 = coords_array[cluster2_indices]
        
        distances = self.haversine_vectorized(coords1, coords2)
        return np.min(distances)
    
    def is_feasible_cluster(self, cluster_indices, coords_array):
        """
        Check if cluster meets all constraints with optimized calculations.
        """
        # Quick size check first
        if len(cluster_indices) > self.max_customers_per_cluster:
            return False
        
        # Check diameter constraint
        diameter = self.calculate_cluster_diameter_fast(cluster_indices, coords_array)
        if diameter > self.max_distance_km:
            return False
        
        # Check route distance constraint (most expensive, so do last)
        route_distance = self.calculate_route_distance_fast(cluster_indices, coords_array)
        if route_distance > self.max_route_distance_km:
            return False
        
        return True
    
    def can_merge_clusters_fast(self, cluster1_indices, cluster2_indices, coords_array):
        """
        Fast check if two clusters can be merged.
        """
        merged_indices = np.concatenate([cluster1_indices, cluster2_indices])
        return self.is_feasible_cluster(merged_indices, coords_array)
    
    def constrained_hierarchical_clustering_optimized(self, customers_df):
        """
        Optimized constraint-aware hierarchical clustering.
        Uses priority queue and smart merging strategies.
        """
        customers_df = customers_df.copy().reset_index(drop=True)
        coords_array = customers_df[['Latitude', 'Longitude']].values
        n_customers = len(customers_df)
        
        # Initialize each customer as its own cluster
        clusters = [np.array([i]) for i in range(n_customers)]
        
        # Create priority queue for potential merges
        # Format: (distance, cluster_id1, cluster_id2)
        merge_queue = []
        
        # Initialize merge queue with all possible pairs
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if self.can_merge_clusters_fast(clusters[i], clusters[j], coords_array):
                    distance = self.cluster_distance_fast(clusters[i], clusters[j], coords_array)
                    heapq.heappush(merge_queue, (distance, i, j))
        
        # Track active clusters (not merged yet)
        active_clusters = set(range(len(clusters)))
        cluster_id_counter = len(clusters)
        
        # Perform merges
        while merge_queue and len(active_clusters) > 1:
            distance, i, j = heapq.heappop(merge_queue)
            
            # Skip if clusters already merged
            if i not in active_clusters or j not in active_clusters:
                continue
            
            # Verify merge is still valid (constraints might have changed)
            if not self.can_merge_clusters_fast(clusters[i], clusters[j], coords_array):
                continue
            
            # Perform merge
            merged_cluster = np.concatenate([clusters[i], clusters[j]])
            clusters.append(merged_cluster)
            new_cluster_id = cluster_id_counter
            cluster_id_counter += 1
            
            # Remove old clusters from active set
            active_clusters.remove(i)
            active_clusters.remove(j)
            active_clusters.add(new_cluster_id)
            
            # Add new potential merges with the merged cluster
            for k in active_clusters:
                if k != new_cluster_id:
                    if self.can_merge_clusters_fast(clusters[new_cluster_id], clusters[k], coords_array):
                        dist = self.cluster_distance_fast(clusters[new_cluster_id], clusters[k], coords_array)
                        heapq.heappush(merge_queue, (dist, min(new_cluster_id, k), max(new_cluster_id, k)))
        
        # Extract final clusters
        final_clusters = [clusters[i] for i in active_clusters]
        
        # Assign cluster IDs to result
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster_indices in enumerate(final_clusters, 1):
            result_df.loc[cluster_indices, 'cluster'] = cluster_id
        
        return result_df
    
    def constrained_clustering_hybrid(self, customers_df):
        """
        Hybrid approach: Use K-means for initial clustering, then refine with constraints.
        Much faster for large datasets.
        """
        customers_df = customers_df.copy().reset_index(drop=True)
        coords_array = customers_df[['Latitude', 'Longitude']].values
        n_customers = len(customers_df)
        
        # Estimate number of clusters based on constraints
        estimated_clusters = max(1, n_customers // (self.max_customers_per_cluster // 2))
        
        # Initial clustering with K-means
        kmeans = KMeans(n_clusters=min(estimated_clusters, n_customers), 
                       random_state=42, n_init=5)
        initial_labels = kmeans.fit_predict(coords_array)
        
        # Convert to cluster indices format
        initial_clusters = []
        for label in np.unique(initial_labels):
            cluster_indices = np.where(initial_labels == label)[0]
            initial_clusters.append(cluster_indices)
        
        # Refine clusters to meet constraints
        final_clusters = []
        
        for cluster_indices in initial_clusters:
            if self.is_feasible_cluster(cluster_indices, coords_array):
                final_clusters.append(cluster_indices)
            else:
                # Split infeasible clusters
                sub_clusters = self._split_infeasible_cluster(cluster_indices, coords_array)
                final_clusters.extend(sub_clusters)
        
        # Try to merge small clusters
        final_clusters = self._merge_small_clusters(final_clusters, coords_array)
        
        # Assign cluster IDs
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster_indices in enumerate(final_clusters, 1):
            result_df.loc[cluster_indices, 'cluster'] = cluster_id
        
        return result_df
    
    def _split_infeasible_cluster(self, cluster_indices, coords_array):
        """
        Split a cluster that violates constraints.
        """
        if len(cluster_indices) <= 1:
            return [cluster_indices]
        
        coords = coords_array[cluster_indices]
        
        # Try binary split first
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=3)
        labels = kmeans.fit_predict(coords)
        
        sub_clusters = []
        for label in np.unique(labels):
            sub_indices = cluster_indices[labels == label]
            if self.is_feasible_cluster(sub_indices, coords_array):
                sub_clusters.append(sub_indices)
            else:
                # Recursively split if still infeasible
                sub_clusters.extend(self._split_infeasible_cluster(sub_indices, coords_array))
        
        return sub_clusters
    
    def _merge_small_clusters(self, clusters, coords_array):
        """
        Try to merge small clusters while respecting constraints.
        """
        # Sort clusters by size (smallest first)
        clusters = sorted(clusters, key=len)
        merged_clusters = []
        used = set()
        
        for i, cluster1 in enumerate(clusters):
            if i in used:
                continue
            
            # Try to merge with other unused clusters
            best_merge = None
            best_distance = float('inf')
            
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                if j in used:
                    continue
                
                if self.can_merge_clusters_fast(cluster1, cluster2, coords_array):
                    distance = self.cluster_distance_fast(cluster1, cluster2, coords_array)
                    if distance < best_distance:
                        best_distance = distance
                        best_merge = j
            
            if best_merge is not None:
                merged_cluster = np.concatenate([cluster1, clusters[best_merge]])
                merged_clusters.append(merged_cluster)
                used.add(i)
                used.add(best_merge)
            else:
                merged_clusters.append(cluster1)
                used.add(i)
        
        return merged_clusters

# Performance comparison function
def compare_clustering_performance():
    """
    Compare performance between different clustering approaches.
    """
    import time
    
    # Generate sample data
    np.random.seed(42)
    n_customers = 500  # Reduced for testing
    
    # Create realistic geographic distribution
    centers = [(40.7128, -74.0060), (34.0522, -118.2437), (41.8781, -87.6298)]
    customers_data = []
    
    for i in range(n_customers):
        center_idx = np.random.choice(len(centers))
        center_lat, center_lon = centers[center_idx]
        
        lat_offset = np.random.normal(0, 0.3)
        lon_offset = np.random.normal(0, 0.3)
        
        customers_data.append({
            'CustomerID': f'C{i+1:04d}',
            'Latitude': center_lat + lat_offset,
            'Longitude': center_lon + lon_offset
        })
    
    customers_df = pd.DataFrame(customers_data)
    
    # Test optimized hierarchical clustering
    print("Testing Optimized Hierarchical Clustering...")
    clusterer = OptimizedConstraintBasedClustering()
    
    start_time = time.time()
    result_hierarchical = clusterer.constrained_hierarchical_clustering_optimized(customers_df)
    hierarchical_time = time.time() - start_time
    
    print(f"Hierarchical clustering: {hierarchical_time:.2f} seconds")
    print(f"Clusters: {result_hierarchical['cluster'].nunique()}")
    
    # Test hybrid approach
    print("\nTesting Hybrid Clustering...")
    start_time = time.time()
    result_hybrid = clusterer.constrained_clustering_hybrid(customers_df)
    hybrid_time = time.time() - start_time
    
    print(f"Hybrid clustering: {hybrid_time:.2f} seconds")
    print(f"Clusters: {result_hybrid['cluster'].nunique()}")
    
    return result_hierarchical, result_hybrid

# Uncomment to run performance test
# if __name__ == "__main__":
#     hierarchical_result, hybrid_result = compare_clustering_performance()





















