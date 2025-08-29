import pandas as pd
import numpy as np
import h3
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, distance_matrix
from collections import defaultdict, deque
import math
import networkx as nx
from typing import Dict, List, Tuple, Set

class H3RouteGrouper:
    """
    Groups H3 cells into delivery routes using various clustering methods.
    """
    
    def __init__(self, df: pd.DataFrame, target_group_size: Tuple[int, int] = (5, 8)):
        """
        Initialize with H3 cell data.
        
        Args:
            df: DataFrame with columns: stock_point_id, sp_latitude, sp_longitude, 
                h3_cell, cluster_centroid_lat, cluster_centroid_lng, cluster_sp_dist_km
            target_group_size: Tuple of (min, max) cells per group
        """
        self.df = df.copy()
        self.df = self.calculate_directions()
        self.target_min, self.target_max = target_group_size
    
    # -------------------------------------------
    # 'sp_latitude', 'sp_longitude'
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate bearing from point 1 to point 2"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        return (bearing + 360) % 360
    
    def get_direction_label(self, bearing):
        """Convert bearing to direction label"""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = int((bearing + 11.25) // 22.5) % 16
        return directions[idx]
    
    def get_simplified_direction(self, bearing):
        """Get simplified 8-direction label"""
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = int((bearing + 22.5) // 45) % 8
        return directions[idx]
    
    def calculate_directions(self):
        """Calculate bearing and direction for each H3 cell"""
        print("""Calculate bearing and direction for each H3 cell""")
        bearings = []
        directions = []
        simplified_directions = []
        
        for _, row in self.df.iterrows():
            bearing = self.calculate_bearing(
                row['sp_latitude'], row['sp_longitude'],
                row['cluster_centroid_lat'], row['cluster_centroid_lng']
            )
            bearings.append(bearing)
            directions.append(self.get_direction_label(bearing))
            simplified_directions.append(self.get_simplified_direction(bearing))
        
        self.df['bearing'] = bearings
        self.df['direction'] = directions
        self.df['simplified_direction'] = simplified_directions
        
        return self.df
    # -------------------------------------------
        
    def group_routes(self, method: str = 'directional', **kwargs) -> pd.DataFrame:
        """
        Group H3 cells into routes using specified method.
        
        Args:
            method: One of ['directional', 'kmeans', 'hexagonal', 'voronoi', 'ring']
            **kwargs: Method-specific parameters
            
        Returns:
            DataFrame with added 'route_group' column
        """
        method_map = {
            'directional': self._directional_clustering,
            'kmeans': self._kmeans_clustering,
            'hierarchical': self._hierarchical_clustering,
            'radial': self._radial_partitioning,
            'graph': self._graph_clustering,
            'hexagonal': self._hexagonal_neighborhood_growth,
            'voronoi': self._voronoi_partitioning,
            'ring': self._ring_based_grouping
        }
        
        if method not in method_map:
            raise ValueError(f"Method must be one of: {list(method_map.keys())}")
            
        return method_map[method](**kwargs)
    
    def _directional_clustering(self, n_sectors: int = 8) -> pd.DataFrame:
        """
        Group cells based on compass direction from distribution center.
        Creates 6-8 directional sectors and groups 5-8 adjacent cells within each.
        """
        result_df = self.df.copy()
        
        for stock_id in result_df['stock_point_id'].unique():
            mask = result_df['stock_point_id'] == stock_id
            subset = result_df[mask].copy()
            
            # Calculate bearing from stock point to each cell
            bearings = self._calculate_bearings(
                subset['sp_latitude'].iloc[0], subset['sp_longitude'].iloc[0],
                subset['cluster_centroid_lat'], subset['cluster_centroid_lng']
            )
            
            # Create directional sectors
            sector_size = 360 / n_sectors
            sector_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            
            route_id = 0
            for sector in range(n_sectors):
                sector_start = sector * sector_size
                sector_end = sector_start + sector_size
                
                # Handle wraparound at 0/360 degrees
                if sector_end > 360:
                    sector_mask = (bearings >= sector_start) | (bearings < (sector_end - 360))
                else:
                    sector_mask = (bearings >= sector_start) & (bearings < sector_end)
                
                if sector_mask.sum() == 0:
                    continue
                    
                sector_cells = subset[sector_mask].copy()
                
                # Group adjacent cells within sector by distance
                distances = sector_cells['cluster_sp_dist_km'].values
                groups = self._create_distance_groups(distances)
                
                sector_name = sector_labels[sector] if sector < len(sector_labels) else f"S{sector}"
                
                # FIXED: Proper indentation for route assignment
                if groups:
                    for group_indices in groups:
                        cell_indices = sector_cells.iloc[group_indices].index
                        # result_df.loc[cell_indices, 'route_group'] = f"{stock_id}_{sector_name}_{route_id}"
                        result_df.loc[cell_indices, 'route_group'] = route_id
                        route_id += 1
                        
        return result_df

    def _kmeans_clustering(self, max_distance_km: float = 15.0) -> pd.DataFrame:
        """
        Apply modified k-means clustering with distance constraints.
        Uses k = (total_cells ÷ 6) to get ~6 cells per group.
        """
        result_df = self.df.copy()
        
        for stock_id in result_df['stock_point_id'].unique():
            mask = result_df['stock_point_id'] == stock_id
            subset = result_df[mask].copy().reset_index(drop=True)
            
            coords = subset[['cluster_centroid_lat', 'cluster_centroid_lng']].values
            n_cells = len(coords)
            
            # Set k = (total_cells ÷ 6) to get ~6 cells per group
            k = max(1, n_cells // 6)
            if k == 0:
                k = 1
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)
            
            # Apply distance constraints and post-process
            labels = self._enforce_distance_constraints(subset, labels, max_distance_km)
            labels = self._balance_group_sizes_kmeans(subset, labels)
            
            # Map back to original indices
            original_indices = result_df[mask].index
            for i, label in enumerate(labels):
                # result_df.loc[original_indices[i], 'route_group'] = f"{stock_id}_KM_{label}"
                result_df.loc[original_indices[i], 'route_group'] = label
                
        return result_df
    
    def _hexagonal_neighborhood_growth(self) -> pd.DataFrame:
        """
        Leverage H3's natural hexagonal structure.
        Start with seed cells (farthest from center in different directions) and grow clusters using H3 neighbor functions.
        """
        result_df = self.df.copy()
        
        for stock_id in result_df['stock_point_id'].unique():
            mask = result_df['stock_point_id'] == stock_id
            subset = result_df[mask].copy()
            
            h3_cells = subset['h3_cell'].tolist()
            cell_to_original_idx = {cell: idx for idx, cell in zip(subset.index, h3_cells)}
            
            # Find seed cells (farthest from center in different directions)
            seeds = self._find_seed_cells(subset)
            
            # Grow clusters from seeds
            clusters = []
            used_cells = set()
            
            for seed_cell in seeds:
                if seed_cell in used_cells or seed_cell not in h3_cells:
                    continue
                    
                cluster = self._grow_hexagonal_cluster(seed_cell, h3_cells, used_cells)
                if len(cluster) >= self.target_min:
                    clusters.append(cluster)
                    used_cells.update(cluster)
            
            # Handle remaining ungrouped cells
            remaining_cells = [cell for cell in h3_cells if cell not in used_cells]
            while remaining_cells:
                if len(remaining_cells) >= self.target_min:
                    # Create new cluster with remaining cells
                    new_cluster = remaining_cells[:self.target_max]
                    clusters.append(new_cluster)
                    remaining_cells = remaining_cells[self.target_max:]
                    used_cells.update(new_cluster)
                else:
                    # Merge with smallest existing cluster
                    if clusters:
                        smallest_idx = min(range(len(clusters)), key=lambda i: len(clusters[i]))
                        clusters[smallest_idx].extend(remaining_cells)
                        used_cells.update(remaining_cells)
                    else:
                        # Create cluster with remaining cells
                        clusters.append(remaining_cells)
                        used_cells.update(remaining_cells)
                    remaining_cells = []
            
            # Assign cluster labels
            for cluster_id, cluster_cells in enumerate(clusters):
                for cell in cluster_cells:
                    if cell in cell_to_original_idx:
                        original_idx = cell_to_original_idx[cell]
                        # result_df.loc[original_idx, 'route_group'] = f"{stock_id}_HEX_{cluster_id}"
                        result_df.loc[original_idx, 'route_group'] = cluster_id
                        
        return result_df
    
    def _voronoi_partitioning(self) -> pd.DataFrame:
        """
        Create natural service areas using Voronoi diagrams.
        Generate Voronoi diagram and merge adjacent cells until each partition has 5-8 H3 cells.
        """
        result_df = self.df.copy()
        
        for stock_id in result_df['stock_point_id'].unique():
            mask = result_df['stock_point_id'] == stock_id
            subset = result_df[mask].copy().reset_index(drop=True)
            
            points = subset[['cluster_centroid_lat', 'cluster_centroid_lng']].values
            n_cells = len(points)
            
            # Create initial seeds for Voronoi-like partitioning
            n_seeds = max(1, n_cells // 6)  # ~6 cells per partition
            seed_indices = self._select_voronoi_seeds(points, n_seeds)
            seed_points = points[seed_indices]
            
            # Assign each cell to nearest seed (Voronoi-like)
            nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn.fit(seed_points)
            _, indices = nn.kneighbors(points)
            labels = indices.flatten()
            
            # Balance partition sizes by merging adjacent Voronoi cells
            labels = self._balance_voronoi_partitions(subset, labels, seed_indices)
            
            # Map back to original indices
            original_indices = result_df[mask].index
            for i, label in enumerate(labels):
                # result_df.loc[original_indices[i], 'route_group'] = f"{stock_id}_VOR_{label}"
                result_df.loc[original_indices[i], 'route_group'] = label
                
        return result_df
    
    def _ring_based_grouping(self, n_rings: int = 3) -> pd.DataFrame:
        """
        Organize by distance rings from distribution center.
        Calculate H3 distance, create concentric rings, and group cells by angular position within rings.
        """
        result_df = self.df.copy()
        
        for stock_id in result_df['stock_point_id'].unique():
            mask = result_df['stock_point_id'] == stock_id
            subset = result_df[mask].copy()
            
            distances = subset['cluster_sp_dist_km'].values
            bearings = self._calculate_bearings(
                subset['sp_latitude'].iloc[0], subset['sp_longitude'].iloc[0],
                subset['cluster_centroid_lat'], subset['cluster_centroid_lng']
            )
            
            # Create distance-based rings using percentiles
            ring_boundaries = np.percentile(distances, np.linspace(0, 100, n_rings + 1))
            ring_boundaries[0] = 0  # Ensure first ring starts at 0
            
            route_id = 0
            ring_names = ['INNER', 'MIDDLE', 'OUTER', 'EXTENDED']
            
            for ring in range(n_rings):
                if ring == n_rings - 1:
                    ring_mask = distances >= ring_boundaries[ring]
                else:
                    ring_mask = (distances >= ring_boundaries[ring]) & (distances < ring_boundaries[ring + 1])
                
                if ring_mask.sum() == 0:
                    continue
                
                ring_subset = subset[ring_mask].copy()
                ring_bearings = bearings[ring_mask]
                
                # Group by angular position within ring to balance workload
                n_cells_in_ring = ring_mask.sum()
                n_segments = max(1, n_cells_in_ring // 6)  # ~6 cells per segment
                
                segment_groups = self._create_angular_groups(ring_bearings, n_segments)
                
                ring_name = ring_names[min(ring, len(ring_names)-1)]
                
                for group_indices in segment_groups:
                    cell_indices = ring_subset.iloc[group_indices].index
                    # result_df.loc[cell_indices, 'route_group'] = f"{stock_id}_{ring_name}_{route_id}"
                    result_df.loc[cell_indices, 'route_group'] = route_id
                    route_id += 1
                    
        return result_df
    
    # Helper methods
    def _calculate_bearings(self, lat1: float, lon1: float, lat2_series, lon2_series) -> np.ndarray:
        """Calculate compass bearings from point 1 to points in series."""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = np.radians(lat2_series)
        lon2_rad = np.radians(lon2_series)
        
        dlon = lon2_rad - lon1_rad
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        
        bearings = np.degrees(np.arctan2(y, x))
        return (bearings + 360) % 360
    
    def _create_distance_groups(self, distances: np.ndarray) -> List[List[int]]:
        """Create groups of 5-8 adjacent cells based on distance from center."""
        sorted_indices = np.argsort(distances)
        groups = []
        current_group = []
        
        for idx in sorted_indices:
            current_group.append(idx)
            if len(current_group) >= self.target_max:
                groups.append(current_group)
                current_group = []
                
        # Handle remaining cells
        if current_group:
            if len(current_group) >= self.target_min or not groups:
                groups.append(current_group)
            else:
                # Merge with last group if too small
                groups[-1].extend(current_group)
        
        # FIXED: Added missing return statement
        return groups

          
    def _hierarchical_clustering(self) -> pd.DataFrame:
        """
        Uses H3 adjacency for agglomerative clustering.
        Creates adjacency matrix based on H3 neighbors and applies hierarchical clustering.
        """
        result_df = self.df.copy()
        
        for stock_id in result_df['stock_point_id'].unique():
            mask = result_df['stock_point_id'] == stock_id
            subset = result_df[mask].copy().reset_index(drop=True)
            
            h3_cells = subset['h3_cell'].tolist()
            n_cells = len(h3_cells)
            
            if n_cells <= 1:
                # result_df.loc[mask, 'route_group'] = f"{stock_id}_HIER_0"
                result_df.loc[mask, 'route_group'] = 0
                continue
            
            # Create H3 adjacency matrix
            adjacency_matrix = self._create_h3_adjacency_matrix(h3_cells)
            
            # Convert adjacency to distance matrix for agglomerative clustering
            distance_matrix_h3 = 1 - adjacency_matrix
            np.fill_diagonal(distance_matrix_h3, 0)
            
            # Add geographical distance component
            coords = subset[['cluster_centroid_lat', 'cluster_centroid_lng']].values
            geo_distances = distance_matrix(coords, coords)
            if geo_distances.max() > 0:
                geo_distances = geo_distances / geo_distances.max()  # Normalize
            
            # Combine H3 adjacency with geographical distance
            combined_distances = 0.7 * distance_matrix_h3 + 0.3 * geo_distances
            
            # Determine number of clusters
            n_clusters = max(1, n_cells // 6)  # ~6 cells per cluster
            
            # Apply agglomerative clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(combined_distances)
            
            # Map back to original indices
            original_indices = result_df[mask].index
            for i, label in enumerate(labels):
                # result_df.loc[original_indices[i], 'route_group'] = f"{stock_id}_HIER_{label}"
                result_df.loc[original_indices[i], 'route_group'] = label
                
        return result_df
    
    def _radial_partitioning(self, n_rings: int = 3, n_segments: int = 8) -> pd.DataFrame:
        """
        Creates concentric rings divided into angular segments.
        Partitions space into ring-segment combinations for balanced territories.
        """
        result_df = self.df.copy()
        
        for stock_id in result_df['stock_point_id'].unique():
            mask = result_df['stock_point_id'] == stock_id
            subset = result_df[mask].copy()
            
            distances = subset['cluster_sp_dist_km'].values
            bearings = self._calculate_bearings(
                subset['sp_latitude'].iloc[0], subset['sp_longitude'].iloc[0],
                subset['cluster_centroid_lat'], subset['cluster_centroid_lng']
            )
            
            # Create concentric rings
            ring_boundaries = np.percentile(distances, np.linspace(0, 100, n_rings + 1))
            ring_boundaries[0] = 0
            
            route_id = 0
            
            for ring in range(n_rings):
                # Define ring boundaries
                if ring == n_rings - 1:
                    ring_mask = distances >= ring_boundaries[ring]
                else:
                    ring_mask = (distances >= ring_boundaries[ring]) & (distances < ring_boundaries[ring + 1])
                
                if ring_mask.sum() == 0:
                    continue
                
                ring_bearings = bearings[ring_mask]
                ring_subset = subset[ring_mask].copy()
                
                # Divide ring into angular segments
                segment_size = 360 / n_segments
                segments = ((ring_bearings + segment_size/2) % 360) // segment_size
                
                for segment in range(n_segments):
                    segment_mask = segments == segment
                    if segment_mask.sum() == 0:
                        continue
                        
                    segment_indices = ring_subset.index[segment_mask]
                    # result_df.loc[segment_indices, 'route_group'] = f"{stock_id}_RAD_{route_id}"
                    result_df.loc[segment_indices, 'route_group'] = route_id 
                    route_id += 1
                    
        return result_df
    
    def _graph_clustering(self) -> pd.DataFrame:
        """
        Community detection on H3 neighbor graph.
        Creates graph based on H3 adjacency and applies community detection algorithms.
        """
        result_df = self.df.copy()
        
        for stock_id in result_df['stock_point_id'].unique():
            mask = result_df['stock_point_id'] == stock_id
            subset = result_df[mask].copy().reset_index(drop=True)
            
            h3_cells = subset['h3_cell'].tolist()
            n_cells = len(h3_cells)
            
            if n_cells <= 1:
                # result_df.loc[mask, 'route_group'] = f"{stock_id}_GRAPH_0"
                result_df.loc[mask, 'route_group'] = 0
                continue
            
            # Create H3 neighbor graph
            G = self._create_h3_graph(h3_cells, subset)
            
            # Apply community detection using simple modularity-based approach
            communities = self._detect_communities_simple(G, h3_cells)
            
            # Balance community sizes
            communities = self._balance_communities(communities, subset)
            
            # Assign labels
            original_indices = result_df[mask].index
            for community_id, community_cells in enumerate(communities):
                for cell in community_cells:
                    if cell in h3_cells:
                        cell_idx = h3_cells.index(cell)
                        original_idx = original_indices[cell_idx]
                        # result_df.loc[original_idx, 'route_group'] = f"{stock_id}_GRAPH_{community_id}"
                        result_df.loc[original_idx, 'route_group'] = community_id
                        
        return result_df
    
    def _enforce_distance_constraints(self, subset: pd.DataFrame, labels: np.ndarray, max_distance_km: float) -> np.ndarray:
        """Enforce maximum distance constraints between cells in same group."""
        new_labels = labels.copy()
        max_label = labels.max()
        
        for label in np.unique(labels):
            mask = labels == label
            group_subset = subset[mask]
            
            if len(group_subset) <= 1:
                continue
                
            # Check maximum distance within group
            coords = group_subset[['cluster_centroid_lat', 'cluster_centroid_lng']].values
            distances = group_subset['cluster_sp_dist_km'].values
            
            # If spread is too large, split the group
            distance_span = distances.max() - distances.min()
            if distance_span > max_distance_km:
                # Split into two groups based on distance
                indices = np.where(mask)[0]
                median_dist = np.median(distances)
                far_mask = distances > median_dist
                
                if far_mask.sum() > 0 and (~far_mask).sum() > 0:
                    max_label += 1
                    far_indices = indices[far_mask]
                    new_labels[far_indices] = max_label
                    
        return new_labels
    
    def _balance_group_sizes_kmeans(self, subset: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        """Balance group sizes by merging small groups and splitting large ones."""
        new_labels = labels.copy()
        coords = subset[['cluster_centroid_lat', 'cluster_centroid_lng']].values
        
        # Handle small groups first
        small_groups = []
        for label in np.unique(labels):
            mask = labels == label
            if mask.sum() < self.target_min:
                small_groups.append(label)
        
        # Merge small groups with nearest neighbors
        for small_label in small_groups:
            small_mask = new_labels == small_label
            if small_mask.sum() == 0:  # Already merged
                continue
                
            small_center = coords[small_mask].mean(axis=0)
            
            # Find nearest larger group
            best_target = None
            min_distance = float('inf')
            
            for target_label in np.unique(new_labels):
                if target_label == small_label:
                    continue
                    
                target_mask = new_labels == target_label
                target_size = target_mask.sum()
                
                if target_size < self.target_max:  # Can accommodate more cells
                    target_center = coords[target_mask].mean(axis=0)
                    distance = np.linalg.norm(small_center - target_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_target = target_label
            
            # Merge with best target
            if best_target is not None:
                new_labels[small_mask] = best_target
        
        return new_labels
    
    def _find_seed_cells(self, subset: pd.DataFrame) -> List[str]:
        """Find seed cells - farthest from center in each direction."""
        distances = subset['cluster_sp_dist_km'].values
        bearings = self._calculate_bearings(
            subset['sp_latitude'].iloc[0], subset['sp_longitude'].iloc[0],
            subset['cluster_centroid_lat'], subset['cluster_centroid_lng']
        )
        h3_cells = subset['h3_cell'].tolist()
        
        # Find farthest cells in 8 directions
        seeds = []
        n_directions = 8
        
        for direction in range(n_directions):
            angle_start = direction * 45
            angle_end = angle_start + 45
            
            # Handle wraparound at 360 degrees
            if angle_end > 360:
                direction_mask = (bearings >= angle_start) | (bearings < (angle_end - 360))
            else:
                direction_mask = (bearings >= angle_start) & (bearings < angle_end)
            
            if direction_mask.sum() > 0:
                direction_distances = distances[direction_mask]
                direction_indices = np.where(direction_mask)[0]
                farthest_idx = direction_indices[np.argmax(direction_distances)]
                seeds.append(h3_cells[farthest_idx])
        
        return list(set(seeds))  # Remove duplicates
    
    def _grow_hexagonal_cluster(self, seed_cell: str, all_cells: List[str], used_cells: Set[str]) -> List[str]:
        """Grow cluster from seed using H3 neighbors, stop when cluster reaches 5-8 cells."""
        cluster = [seed_cell]
        candidates = deque([seed_cell])
        
        while candidates and len(cluster) < self.target_max:
            current_cell = candidates.popleft()
            
            # Get H3 neighbors using hex_ring
            try:
                neighbors = list(h3.hex_ring(current_cell, 1))
            except:
                neighbors = []
                
            for neighbor in neighbors:
                if (neighbor in all_cells and 
                    neighbor not in used_cells and 
                    len(cluster) < self.target_max):
                    cluster.append(neighbor)
                    candidates.append(neighbor)
                    used_cells.add(neighbor)
        
        return cluster
    
    def _select_voronoi_seeds(self, points: np.ndarray, n_seeds: int) -> np.ndarray:
        """Select well-distributed seed points for Voronoi-like partitioning."""
        if n_seeds >= len(points):
            return np.arange(len(points))
        
        # Use farthest-first selection
        seeds = []
        
        # First seed: centroid-like point
        center = points.mean(axis=0)
        distances_to_center = np.linalg.norm(points - center, axis=1)
        seeds.append(np.argmin(distances_to_center))
        
        # Subsequent seeds: farthest from existing seeds
        for _ in range(1, n_seeds):
            min_distances = np.full(len(points), np.inf)
            
            for seed_idx in seeds:
                seed_point = points[seed_idx]
                point_distances = np.linalg.norm(points - seed_point, axis=1)
                min_distances = np.minimum(min_distances, point_distances)
            
            # Select point with maximum distance to nearest seed
            next_seed = np.argmax(min_distances)
            if next_seed not in seeds:
                seeds.append(next_seed)
        
        return np.array(seeds)
    
    def _balance_voronoi_partitions(self, subset: pd.DataFrame, labels: np.ndarray, seed_indices: np.ndarray) -> np.ndarray:
        """Merge adjacent Voronoi cells until each partition has 5-8 H3 cells."""
        new_labels = labels.copy()
        coords = subset[['cluster_centroid_lat', 'cluster_centroid_lng']].values
        
        # Iteratively balance partitions
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            changed = False
            
            # Find partitions that need rebalancing
            for label in np.unique(new_labels):
                mask = new_labels == label
                partition_size = mask.sum()
                
                if partition_size < self.target_min:
                    # Merge small partition with nearest neighbor
                    partition_center = coords[mask].mean(axis=0)
                    
                    # Find nearest partition that can accommodate
                    best_merge_target = None
                    min_distance = float('inf')
                    
                    for other_label in np.unique(new_labels):
                        if other_label == label:
                            continue
                            
                        other_mask = new_labels == other_label
                        other_size = other_mask.sum()
                        
                        if other_size + partition_size <= self.target_max:
                            other_center = coords[other_mask].mean(axis=0)
                            distance = np.linalg.norm(partition_center - other_center)
                            
                            if distance < min_distance:
                                min_distance = distance
                                best_merge_target = other_label
                    
                    if best_merge_target is not None:
                        new_labels[mask] = best_merge_target
                        changed = True
                
                elif partition_size > self.target_max:
                    # Split large partition
                    partition_coords = coords[mask]
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    sub_labels = kmeans.fit_predict(partition_coords)
                    
                    # Assign new label to one of the splits
                    max_label = new_labels.max()
                    partition_indices = np.where(mask)[0]
                    split_indices = partition_indices[sub_labels == 1]
                    
                    if len(split_indices) >= self.target_min:
                        new_labels[split_indices] = max_label + 1
                        changed = True
            
            if not changed:
                break
            iteration += 1
        
        return new_labels
    
    def _create_angular_groups(self, bearings: np.ndarray, n_groups: int) -> List[List[int]]:
        """Group cells by angular position, maintaining directional logic."""
        if n_groups <= 1:
            return [list(range(len(bearings)))]
        
        # Sort by bearing and create balanced groups
        sorted_indices = np.argsort(bearings)
        groups = []
        
        # Calculate group sizes to balance workload
        base_size = len(bearings) // n_groups
        extra = len(bearings) % n_groups
        
        start_idx = 0
        for i in range(n_groups):
            group_size = base_size + (1 if i < extra else 0)
            end_idx = start_idx + group_size
            
            if start_idx < len(sorted_indices):
                group = sorted_indices[start_idx:end_idx].tolist()
                if group:
                    groups.append(group)
            
            start_idx = end_idx
        
        # FIXED: Added missing return statement
        return groups

    def _create_h3_adjacency_matrix(self, h3_cells: List[str]) -> np.ndarray:
        """Create adjacency matrix for H3 cells based on hex neighbors."""
        n = len(h3_cells)
        matrix = np.zeros((n, n))
        
        cell_to_idx = {cell: i for i, cell in enumerate(h3_cells)}
        
        for i, cell1 in enumerate(h3_cells):
            try:
                neighbors = list(h3.hex_ring(cell1, 1))
                for neighbor in neighbors:
                    if neighbor in cell_to_idx:
                        j = cell_to_idx[neighbor]
                        matrix[i, j] = 1
            except:
                continue
                        
        return matrix
    
    def _create_h3_graph(self, h3_cells: List[str], subset: pd.DataFrame) -> 'nx.Graph':
        """Create NetworkX graph from H3 cells with weights."""
        G = nx.Graph()
        
        # Add nodes with positions
        for i, cell in enumerate(h3_cells):
            lat = subset.iloc[i]['cluster_centroid_lat']
            lng = subset.iloc[i]['cluster_centroid_lng']
            dist = subset.iloc[i]['cluster_sp_dist_km']
            G.add_node(cell, pos=(lat, lng), distance=dist)
        
        # Add edges for H3 neighbors
        for cell1 in h3_cells:
            try:
                neighbors = list(h3.hex_ring(cell1, 1))
                for neighbor in neighbors:
                    if neighbor in h3_cells and neighbor != cell1:
                        # Edge weight based on distance similarity
                        dist1 = G.nodes[cell1]['distance']
                        dist2 = G.nodes[neighbor]['distance']
                        weight = 1.0 / (1.0 + abs(dist1 - dist2))
                        G.add_edge(cell1, neighbor, weight=weight)
            except:
                continue
                
        return G
    
    def _detect_communities_simple(self, G: 'nx.Graph', h3_cells: List[str]) -> List[List[str]]:
        """Simple community detection using connected components and modularity."""
        if len(G.nodes()) == 0:
            return [h3_cells]
        
        # Start with connected components
        communities = list(nx.connected_components(G))
        communities = [list(community) for community in communities]
        
        # If we have disconnected components, try to merge based on proximity
        if len(communities) > 1:
            merged_communities = []
            used_nodes = set()
            
            for community in communities:
                if any(node in used_nodes for node in community):
                    continue
                    
                # Try to grow community by finding nearby nodes
                extended_community = list(community)
                used_nodes.update(community)
                
                # Look for nearby disconnected nodes
                for other_community in communities:
                    if any(node in used_nodes for node in other_community):
                        continue
                        
                    # Check if communities should be merged based on distance
                    should_merge = self._should_merge_communities(
                        extended_community, other_community, G
                    )
                    
                    if should_merge and len(extended_community) + len(other_community) <= self.target_max:
                        extended_community.extend(other_community)
                        used_nodes.update(other_community)
                
                if extended_community:
                    merged_communities.append(extended_community)
            
            communities = merged_communities
        
        return communities
    
    def _should_merge_communities(self, comm1: List[str], comm2: List[str], G: 'nx.Graph') -> bool:
        """Determine if two communities should be merged based on proximity."""
        min_distance = float('inf')
        
        for node1 in comm1:
            for node2 in comm2:
                if node1 in G.nodes and node2 in G.nodes:
                    pos1 = G.nodes[node1]['pos']
                    pos2 = G.nodes[node2]['pos']
                    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    min_distance = min(min_distance, dist)
        
        # Merge if communities are close (threshold can be adjusted)
        return min_distance < 0.01  # Roughly 1km in degrees
    
    def _balance_communities(self, communities: List[List[str]], subset: pd.DataFrame) -> List[List[str]]:
        """Balance community sizes to meet target group size constraints."""
        balanced_communities = []
        
        for community in communities:
            if len(community) < self.target_min:
                # Try to merge with smallest community
                if balanced_communities:
                    smallest_idx = min(range(len(balanced_communities)), 
                                     key=lambda i: len(balanced_communities[i]))
                    if len(balanced_communities[smallest_idx]) + len(community) <= self.target_max:
                        balanced_communities[smallest_idx].extend(community)
                    else:
                        balanced_communities.append(community)
                else:
                    balanced_communities.append(community)
                    
            elif len(community) > self.target_max:
                # Split large community
                while len(community) > self.target_max:
                    split_size = min(self.target_max, len(community))
                    balanced_communities.append(community[:split_size])
                    community = community[split_size:]
                
                if community:
                    balanced_communities.append(community)
            else:
                balanced_communities.append(community)
        
        return balanced_communities

""" 
# Example usage and testing:
if __name__ == "__main__":
    # Extended sample data for better testing
    sample_data = {
        'stock_point_id': [1647113] * 15,
        'sp_latitude': [6.473953] * 15,
        'sp_longitude': [3.356525] * 15,
        'stock_point_name': ['OmniHub Apapa Lagos - CAUSEWAY'] * 15,
        'h3_cell': [
            '88589c9945fffff', '88589cd655fffff', '88589cd695fffff', '88589cd613fffff', '88589c9b23fffff',
            '88589cd651fffff', '88589cd69bfffff', '88589c9941fffff', '88589c994dfffff', '88589cd615fffff',
            '88589c994bfffff', '88589cd657fffff', '88589cd691fffff', '88589c9b21fffff', '88589cd617fffff'
        ],
        'h3_resolution': [8] * 15,
        'cluster_centroid_lat': [
            6.483274, 6.449219, 6.415455, 6.434740, 6.502711, 6.455123, 6.425789, 6.489456, 6.467891, 6.442567,
            6.478234, 6.451789, 6.419876, 6.495432, 6.437891
        ],
        'cluster_centroid_lng': [
            3.343375, 3.327780, 3.392859, 3.352789, 3.343644, 3.335912, 3.387234, 3.351456, 3.329876, 3.345678,
            3.341234, 3.333456, 3.389123, 3.347891, 3.354567
        ],
        'cluster_sp_dist_km': [
            1.784629, 4.201359, 7.643867, 4.379737, 3.500114, 3.892456, 6.234567, 2.123456, 1.987654, 3.876543,
            1.567890, 3.123456, 6.789012, 2.987654, 4.123789
        ]
    }
    
    df = pd.DataFrame(sample_data)
    grouper = H3RouteGrouper(df, target_group_size=(5, 8))
    
    # Test all methods
    for method in ['directional', 'kmeans', 'hierarchical', 'radial', 'graph', 'hexagonal', 'voronoi', 'ring']:
        print(f"\n=== {method.upper()} METHOD ===")
        try:
            result = grouper.group_routes(method=method)
            route_summary = result.groupby('route_group').size()
            print(f"Groups created: {len(route_summary)}")
            print(f"Group sizes: {route_summary.tolist()}")
            print(result[['h3_cell', 'route_group', 'cluster_sp_dist_km']].head(10).to_string(index=False))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
"""            