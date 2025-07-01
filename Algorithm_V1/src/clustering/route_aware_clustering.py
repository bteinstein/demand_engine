import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree, ConvexHull
import warnings

class OptimizedRouteAwareClustering:
    def __init__(self, max_customers_per_route=20, max_route_time_minutes=480,
                 avg_service_time_minutes=15, avg_speed_kmh=30, depot_location=None,
                 precompute_distances=True):
        """
        Initialize optimized route-aware clustering for delivery optimization.
        
        Parameters:
        -----------
        max_customers_per_route : int
            Maximum customers per delivery route
        max_route_time_minutes : int
            Maximum route duration in minutes
        avg_service_time_minutes : int
            Average service time per customer
        avg_speed_kmh : float
            Average driving speed
        depot_location : tuple or None
            (latitude, longitude) of depot/warehouse
        precompute_distances : bool
            Whether to precompute distance matrices for faster operations
        """
        self.max_customers_per_route = max_customers_per_route
        self.max_route_time_minutes = max_route_time_minutes
        self.avg_service_time_minutes = avg_service_time_minutes
        self.avg_speed_kmh = avg_speed_kmh
        self.depot_location = depot_location
        self.precompute_distances = precompute_distances
        
        # Performance optimization attributes
        self.distance_matrix = None
        self.depot_distances = None
        self.coords_array = None
        self.kdtree = None
        self._is_initialized = False
    
    def _initialize_optimization_structures(self, customers_df):
        """Initialize optimization data structures for faster computation."""
        if self._is_initialized:
            return
        
        # Convert coordinates to NumPy array for vectorized operations
        self.coords_array = customers_df[['Latitude', 'Longitude']].values.astype(np.float64)
        n_customers = len(self.coords_array)
        
        if self.precompute_distances and n_customers > 0:
            # Precompute distance matrix using vectorized operations
            self.distance_matrix = self._compute_distance_matrix_vectorized(self.coords_array)
            
            # Precompute depot distances if depot is provided
            if self.depot_location is not None:
                depot_coord = np.array([self.depot_location[0], self.depot_location[1]])
                self.depot_distances = self._compute_distances_to_point_vectorized(
                    self.coords_array, depot_coord
                )
            
            # Create KDTree for efficient nearest neighbor searches
            if n_customers > 10:  # Only beneficial for larger datasets
                self.kdtree = KDTree(self.coords_array)
        
        self._is_initialized = True
    
    def _compute_distance_matrix_vectorized(self, coords):
        """Compute distance matrix using vectorized operations."""
        n = len(coords)
        distances = np.zeros((n, n), dtype=np.float32)
        
        # Vectorized distance calculation using haversine approximation for speed
        # For higher accuracy, can switch back to geodesic for smaller datasets
        if n < 500:
            # Use geodesic for smaller datasets (more accurate)
            for i in range(n):
                for j in range(i+1, n):
                    dist = geodesic(coords[i], coords[j]).kilometers
                    distances[i, j] = distances[j, i] = dist
        else:
            # Use haversine approximation for larger datasets (faster)
            distances = self._haversine_distance_matrix(coords)
        
        return distances
    
    def _haversine_distance_matrix(self, coords):
        """Fast haversine distance matrix computation."""
        coords_rad = np.radians(coords)
        lat = coords_rad[:, 0]
        lon = coords_rad[:, 1]
        
        # Vectorized haversine calculation
        dlat = lat[:, np.newaxis] - lat
        dlon = lon[:, np.newaxis] - lon
        
        a = np.sin(dlat/2)**2 + np.cos(lat[:, np.newaxis]) * np.cos(lat) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
        # Earth radius in kilometers
        R = 6371.0
        return (R * c).astype(np.float32)
    
    def _compute_distances_to_point_vectorized(self, coords, point):
        """Compute distances from all coordinates to a single point."""
        if len(coords) < 100:
            # Use geodesic for accuracy with smaller datasets
            return np.array([geodesic(coord, point).kilometers for coord in coords], dtype=np.float32)
        else:
            # Use haversine approximation for speed
            coords_rad = np.radians(coords)
            point_rad = np.radians(point)
            
            dlat = coords_rad[:, 0] - point_rad[0]
            dlon = coords_rad[:, 1] - point_rad[1]
            
            a = np.sin(dlat/2)**2 + np.cos(coords_rad[:, 0]) * np.cos(point_rad[0]) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            
            return (6371.0 * c).astype(np.float32)
    
    def _get_distance(self, i, j):
        """Get distance between customers i and j."""
        if self.distance_matrix is not None:
            return self.distance_matrix[i, j]
        else:
            return geodesic(self.coords_array[i], self.coords_array[j]).kilometers
    
    def _get_depot_distance(self, i):
        """Get distance from customer i to depot."""
        if self.depot_distances is not None:
            return self.depot_distances[i]
        elif self.depot_location is not None:
            return geodesic(self.coords_array[i], self.depot_location).kilometers
        else:
            return 0
    
    def estimate_route_time_optimized(self, customer_indices):
        """Optimized route time estimation using precomputed distances."""
        if len(customer_indices) == 0:
            return 0
        
        # Calculate total service time
        service_time = len(customer_indices) * self.avg_service_time_minutes
        
        # Estimate travel time using optimized nearest neighbor heuristic
        travel_distance = self._calculate_route_distance_optimized(customer_indices)
        travel_time = (travel_distance / self.avg_speed_kmh) * 60
        
        # Add depot travel time if depot location is provided
        if self.depot_location is not None:
            depot_travel_time = (
                (self._get_depot_distance(customer_indices[0]) + 
                 self._get_depot_distance(customer_indices[-1])) / self.avg_speed_kmh
            ) * 60
            travel_time += depot_travel_time
        
        return service_time + travel_time
    
    def _calculate_route_distance_optimized(self, customer_indices):
        """Optimized route distance calculation using precomputed distances."""
        if len(customer_indices) <= 1:
            return 0
        
        # Use nearest neighbor heuristic with precomputed distances
        unvisited = set(customer_indices[1:])
        current = customer_indices[0]
        total_distance = 0
        
        while unvisited:
            # Find nearest unvisited customer
            distances_to_unvisited = [self._get_distance(current, idx) for idx in unvisited]
            nearest_idx_pos = np.argmin(distances_to_unvisited)
            nearest_idx = list(unvisited)[nearest_idx_pos]
            
            total_distance += distances_to_unvisited[nearest_idx_pos]
            current = nearest_idx
            unvisited.remove(nearest_idx)
        
        return total_distance
    
    def is_feasible_route_optimized(self, customer_indices):
        """Optimized feasibility check using indices."""
        # Check customer count
        if len(customer_indices) > self.max_customers_per_route:
            return False
        
        # Check estimated route time
        if self.estimate_route_time_optimized(customer_indices) > self.max_route_time_minutes:
            return False
        
        return True
    
    def _calculate_route_compactness_optimized(self, customer_indices):
        """Optimized compactness calculation."""
        if len(customer_indices) <= 1:
            return 0
        
        coords = self.coords_array[customer_indices]
        
        # Calculate convex hull area (smaller is more compact)
        try:
            if len(coords) >= 3:
                hull = ConvexHull(coords)
                return hull.volume  # Area in 2D
            else:
                # For 2 points, use distance
                return self._get_distance(customer_indices[0], customer_indices[1])
        except:
            # Fallback: use standard deviation of distances from centroid
            center = np.mean(coords, axis=0)
            distances = [self._compute_distances_to_point_vectorized(
                coords, center
            )]
            return np.std(distances) if len(distances) > 1 else 0
    
    def sweep_algorithm_clustering_optimized(self, customers_df):
        """Optimized sweep algorithm using vectorized operations."""
        customers_df = customers_df.copy().reset_index(drop=True)
        self._initialize_optimization_structures(customers_df)
        
        n_customers = len(customers_df)
        
        if self.depot_location is None:
            # Use geographic center as depot
            depot = np.mean(self.coords_array, axis=0)
        else:
            depot = np.array(self.depot_location)
        
        # Vectorized angle calculation
        dx = self.coords_array[:, 1] - depot[1]  # longitude difference
        dy = self.coords_array[:, 0] - depot[0]  # latitude difference
        angles = np.arctan2(dy, dx)
        
        # Sort customers by angle
        sorted_indices = np.argsort(angles)
        
        # Create routes by sweeping through angles
        cluster_labels = np.zeros(n_customers, dtype=int)
        current_cluster_id = 1
        current_cluster_indices = []
        
        for idx in sorted_indices:
            # Try adding customer to current cluster
            test_cluster = current_cluster_indices + [idx]
            
            if self.is_feasible_route_optimized(test_cluster):
                current_cluster_indices.append(idx)
            else:
                # Assign current cluster labels
                if current_cluster_indices:
                    for customer_idx in current_cluster_indices:
                        cluster_labels[customer_idx] = current_cluster_id
                    current_cluster_id += 1
                
                # Start new cluster
                current_cluster_indices = [idx]
        
        # Assign the last cluster
        if current_cluster_indices:
            for customer_idx in current_cluster_indices:
                cluster_labels[customer_idx] = current_cluster_id
        
        # Create result DataFrame
        result_df = customers_df.copy()
        result_df['cluster'] = cluster_labels
        
        return result_df
    
    def capacity_constrained_clustering_optimized(self, customers_df):
        """Optimized capacity-constrained clustering using efficient data structures."""
        customers_df = customers_df.copy().reset_index(drop=True)
        self._initialize_optimization_structures(customers_df)
        
        n_customers = len(customers_df)
        cluster_labels = np.zeros(n_customers, dtype=int)
        remaining_mask = np.ones(n_customers, dtype=bool)
        current_cluster_id = 1
        
        while np.any(remaining_mask):
            remaining_indices = np.where(remaining_mask)[0]
            
            # Start new route with customer closest to depot
            if self.depot_location is not None:
                depot_distances = [self._get_depot_distance(idx) for idx in remaining_indices]
                start_idx = remaining_indices[np.argmin(depot_distances)]
            else:
                start_idx = remaining_indices[0]
            
            current_route_indices = [start_idx]
            remaining_mask[start_idx] = False
            
            # Greedily add customers to current route
            route_complete = False
            while not route_complete and np.any(remaining_mask):
                remaining_indices = np.where(remaining_mask)[0]
                
                if len(remaining_indices) == 0:
                    break
                
                # Find nearest customer to current route using vectorized operations
                if self.kdtree is not None and len(remaining_indices) > 20:
                    # Use KDTree for large datasets
                    best_customer_idx = self._find_nearest_customer_kdtree(
                        current_route_indices, remaining_indices
                    )
                else:
                    # Use direct distance computation for smaller datasets
                    best_customer_idx = self._find_nearest_customer_direct(
                        current_route_indices, remaining_indices
                    )
                
                # Try adding the best candidate
                if best_customer_idx is not None:
                    test_route = current_route_indices + [best_customer_idx]
                    
                    if self.is_feasible_route_optimized(test_route):
                        current_route_indices.append(best_customer_idx)
                        remaining_mask[best_customer_idx] = False
                    else:
                        route_complete = True
                else:
                    route_complete = True
            
            # Assign cluster labels for completed route
            for idx in current_route_indices:
                cluster_labels[idx] = current_cluster_id
            current_cluster_id += 1
        
        # Create result DataFrame
        result_df = customers_df.copy()
        result_df['cluster'] = cluster_labels
        
        return result_df
    
    def _find_nearest_customer_direct(self, current_route_indices, remaining_indices):
        """Find nearest customer using direct distance computation."""
        best_customer_idx = None
        best_distance = float('inf')
        
        for candidate_idx in remaining_indices:
            # Calculate minimum distance to current route
            min_dist = min(
                self._get_distance(candidate_idx, route_idx) 
                for route_idx in current_route_indices
            )
            
            if min_dist < best_distance:
                best_distance = min_dist
                best_customer_idx = candidate_idx
        
        return best_customer_idx
    
    def _find_nearest_customer_kdtree(self, current_route_indices, remaining_indices):
        """Find nearest customer using KDTree for efficiency."""
        if self.kdtree is None:
            return self._find_nearest_customer_direct(current_route_indices, remaining_indices)
        
        # Query KDTree for each customer in current route
        min_distances = {}
        
        for route_idx in current_route_indices:
            # Find k nearest neighbors among remaining customers
            k = min(10, len(remaining_indices))  # Limit search space
            distances, indices = self.kdtree.query(
                self.coords_array[route_idx], 
                k=len(self.coords_array)
            )
            
            # Filter to only remaining customers
            for dist, idx in zip(distances, indices):
                if idx in remaining_indices:
                    if idx not in min_distances or dist < min_distances[idx]:
                        min_distances[idx] = dist
        
        # Return customer with minimum distance
        if min_distances:
            return min(min_distances.keys(), key=lambda x: min_distances[x])
        else:
            return None
    
    # Backward compatibility methods
    def estimate_route_time(self, customers_subset):
        """Backward compatibility wrapper."""
        if isinstance(customers_subset, pd.DataFrame):
            # Convert to indices for optimized version
            customer_indices = customers_subset.index.tolist()
            return self.estimate_route_time_optimized(customer_indices)
        else:
            return self.estimate_route_time_optimized(customers_subset)
    
    def calculate_route_distance(self, customers_subset):
        """Backward compatibility wrapper."""
        if isinstance(customers_subset, pd.DataFrame):
            customer_indices = customers_subset.index.tolist()
            return self._calculate_route_distance_optimized(customer_indices)
        else:
            return self._calculate_route_distance_optimized(customers_subset)
    
    def is_feasible_route(self, customers_subset):
        """Backward compatibility wrapper."""
        if isinstance(customers_subset, pd.DataFrame):
            customer_indices = customers_subset.index.tolist()
            return self.is_feasible_route_optimized(customer_indices)
        else:
            return self.is_feasible_route_optimized(customers_subset)
    
    def calculate_route_compactness(self, customers_subset):
        """Backward compatibility wrapper."""
        if isinstance(customers_subset, pd.DataFrame):
            customer_indices = customers_subset.index.tolist()
            return self._calculate_route_compactness_optimized(customer_indices)
        else:
            return self._calculate_route_compactness_optimized(customers_subset)
    
    # Main clustering methods (optimized versions)
    def sweep_algorithm_clustering(self, customers_df):
        """Main sweep algorithm method (optimized)."""
        return self.sweep_algorithm_clustering_optimized(customers_df)
    
    def capacity_constrained_clustering(self, customers_df):
        """Main capacity-constrained clustering method (optimized)."""
        return self.capacity_constrained_clustering_optimized(customers_df)

# Performance comparison utility
def compare_performance(original_class, optimized_class, customers_df, method='sweep'):
    """Compare performance between original and optimized implementations."""
    import time
    
    print(f"Performance Comparison - {method} algorithm")
    print(f"Dataset size: {len(customers_df)} customers")
    print("-" * 50)
    
    # Test original implementation
    original_clusterer = original_class()
    start_time = time.time()
    
    if method == 'sweep':
        original_result = original_clusterer.sweep_algorithm_clustering(customers_df)
    else:
        original_result = original_clusterer.capacity_constrained_clustering(customers_df)
    
    original_time = time.time() - start_time
    
    # Test optimized implementation
    optimized_clusterer = optimized_class()
    start_time = time.time()
    
    if method == 'sweep':
        optimized_result = optimized_clusterer.sweep_algorithm_clustering(customers_df)
    else:
        optimized_result = optimized_clusterer.capacity_constrained_clustering(customers_df)
    
    optimized_time = time.time() - start_time
    
    # Results
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    
    print(f"Original time: {original_time:.4f} seconds")
    print(f"Optimized time: {optimized_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Original clusters: {original_result['cluster'].nunique()}")
    print(f"Optimized clusters: {optimized_result['cluster'].nunique()}")
    
    return {
        'original_time': original_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'original_clusters': original_result['cluster'].nunique(),
        'optimized_clusters': optimized_result['cluster'].nunique()
    }


class RouteAwareClustering:
    def __init__(self, max_customers_per_route=20, max_route_time_minutes=480,
                 avg_service_time_minutes=15, avg_speed_kmh=30, depot_location=None):
        """
        Initialize route-aware clustering for delivery optimization.
        
        Parameters:
        -----------
        max_customers_per_route : int
            Maximum customers per delivery route
        max_route_time_minutes : int
            Maximum route duration in minutes
        avg_service_time_minutes : int
            Average service time per customer
        avg_speed_kmh : float
            Average driving speed
        depot_location : tuple or None
            (latitude, longitude) of depot/warehouse
        """
        self.max_customers_per_route = max_customers_per_route
        self.max_route_time_minutes = max_route_time_minutes
        self.avg_service_time_minutes = avg_service_time_minutes
        self.avg_speed_kmh = avg_speed_kmh
        self.depot_location = depot_location
    
    def estimate_route_time(self, customers_subset):
        """Estimate total route time including travel and service time."""
        if len(customers_subset) == 0:
            return 0
        
        coords = customers_subset[['Latitude', 'Longitude']].values
        
        # Calculate total service time
        service_time = len(customers_subset) * self.avg_service_time_minutes
        
        # Estimate travel time using nearest neighbor heuristic
        travel_distance = self.calculate_route_distance(customers_subset)
        travel_time = (travel_distance / self.avg_speed_kmh) * 60  # Convert to minutes
        
        # Add depot travel time if depot location is provided
        if self.depot_location is not None:
            # Distance from depot to first customer
            depot_to_first = geodesic(self.depot_location, coords[0]).kilometers
            # Distance from last customer to depot (approximate)
            last_to_depot = geodesic(coords[-1], self.depot_location).kilometers
            depot_travel_time = ((depot_to_first + last_to_depot) / self.avg_speed_kmh) * 60
            travel_time += depot_travel_time
        
        return service_time + travel_time
    
    def calculate_route_distance(self, customers_subset):
        """Calculate approximate route distance using nearest neighbor."""
        if len(customers_subset) <= 1:
            return 0
        
        coords = customers_subset[['Latitude', 'Longitude']].values
        
        # Use nearest neighbor heuristic for TSP approximation
        unvisited = list(range(len(coords)))
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
    
    def is_feasible_route(self, customers_subset):
        """Check if customer subset forms a feasible delivery route."""
        # Check customer count
        if len(customers_subset) > self.max_customers_per_route:
            return False
        
        # Check estimated route time
        if self.estimate_route_time(customers_subset) > self.max_route_time_minutes:
            return False
        
        return True
    
    def calculate_route_compactness(self, customers_subset):
        """Calculate how compact/efficient a route is."""
        if len(customers_subset) <= 1:
            return 0
        
        coords = customers_subset[['Latitude', 'Longitude']].values
        
        # Calculate convex hull area (smaller is more compact)
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(coords)
            compactness = hull.volume  # Area in 2D
        except:
            # Fallback: use standard deviation of distances
            center = np.mean(coords, axis=0)
            distances = [geodesic(center, coord).kilometers for coord in coords]
            compactness = np.std(distances)
        
        return compactness
    
    def sweep_algorithm_clustering(self, customers_df):
        """
        Implement sweep algorithm for vehicle routing clustering.
        Good for radial customer distributions.
        """
        customers_df = customers_df.copy().reset_index(drop=True)
        
        if self.depot_location is None:
            # Use geographic center as depot
            center_lat = customers_df['Latitude'].mean()
            center_lon = customers_df['Longitude'].mean()
            depot = (center_lat, center_lon)
        else:
            depot = self.depot_location
        
        # Calculate angles from depot to each customer
        coords = customers_df[['Latitude', 'Longitude']].values
        angles = []
        
        for coord in coords:
            dx = coord[1] - depot[1]  # longitude difference
            dy = coord[0] - depot[0]  # latitude difference
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # Sort customers by angle
        sorted_indices = np.argsort(angles)
        
        # Create routes by sweeping through angles
        clusters = []
        current_cluster = []
        
        for idx in sorted_indices:
            customer = customers_df.iloc[idx:idx+1]
            
            # Try adding customer to current cluster
            test_cluster = pd.concat([pd.DataFrame(current_cluster), customer], ignore_index=True)
            
            if self.is_feasible_route(test_cluster):
                current_cluster.append(customers_df.iloc[idx])
            else:
                # Start new cluster
                if current_cluster:
                    clusters.append(pd.DataFrame(current_cluster))
                current_cluster = [customers_df.iloc[idx]]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(pd.DataFrame(current_cluster))
        
        # Assign cluster labels
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster_df in enumerate(clusters, 1):
            for _, row in cluster_df.iterrows():
                matching_idx = customers_df[
                    (customers_df['Latitude'] == row['Latitude']) & 
                    (customers_df['Longitude'] == row['Longitude'])
                ].index[0]
                result_df.loc[matching_idx, 'cluster'] = cluster_id
        
        return result_df
    
    def capacity_constrained_clustering(self, customers_df):
        """
        Clustering that respects both capacity and time constraints.
        Uses a greedy approach to build feasible routes.
        """
        customers_df = customers_df.copy().reset_index(drop=True)
        
        clusters = []
        remaining_customers = customers_df.copy()
        
        while len(remaining_customers) > 0:
            # Start new route with the customer closest to depot (or center)
            if self.depot_location is not None:
                distances_to_depot = [
                    geodesic(self.depot_location, (row['Latitude'], row['Longitude'])).kilometers
                    for _, row in remaining_customers.iterrows()
                ]
                start_idx = np.argmin(distances_to_depot)
            else:
                # Use a random starting point
                start_idx = 0
            
            current_route = [remaining_customers.iloc[start_idx]]
            remaining_customers = remaining_customers.drop(remaining_customers.index[start_idx]).reset_index(drop=True)
            
            # Greedily add customers to current route
            route_complete = False
            while not route_complete and len(remaining_customers) > 0:
                # Find the nearest customer to the current route
                best_customer_idx = None
                best_distance = float('inf')
                
                current_route_df = pd.DataFrame(current_route)
                
                for idx, candidate in remaining_customers.iterrows():
                    # Calculate minimum distance to current route
                    min_dist = float('inf')
                    for _, route_customer in current_route_df.iterrows():
                        dist = geodesic(
                            (candidate['Latitude'], candidate['Longitude']),
                            (route_customer['Latitude'], route_customer['Longitude'])
                        ).kilometers
                        min_dist = min(min_dist, dist)
                    
                    if min_dist < best_distance:
                        best_distance = min_dist
                        best_customer_idx = idx
                
                # Try adding the best candidate
                if best_customer_idx is not None:
                    candidate = remaining_customers.iloc[best_customer_idx]
                    test_route = current_route + [candidate]
                    test_route_df = pd.DataFrame(test_route)
                    
                    if self.is_feasible_route(test_route_df):
                        current_route.append(candidate)
                        remaining_customers = remaining_customers.drop(
                            remaining_customers.index[best_customer_idx]
                        ).reset_index(drop=True)
                    else:
                        route_complete = True
                else:
                    route_complete = True
            
            # Add completed route to clusters
            if current_route:
                clusters.append(pd.DataFrame(current_route))
        
        # Assign cluster labels
        result_df = customers_df.copy()
        result_df['cluster'] = -1
        
        for cluster_id, cluster_df in enumerate(clusters, 1):
            for _, row in cluster_df.iterrows():
                matching_idx = customers_df[
                    (customers_df['Latitude'] == row['Latitude']) & 
                    (customers_df['Longitude'] == row['Longitude'])
                ].index[0]
                result_df.loc[matching_idx, 'cluster'] = cluster_id
        
        return result_df