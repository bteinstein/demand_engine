# Advanced Algorithm Tuning Parameters for Route Optimization

class RouteOptimizationTuner:
    """
    Advanced tuning parameters to influence route generation algorithm
    Based on the H3 visualization analysis showing suboptimal clustering
    """
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def apply_geographic_constraints(self, max_inter_cell_distance=3.0, 
                                   compactness_weight=0.8):
        """
        1. GEOGRAPHIC COMPACTNESS ENFORCEMENT
        Force routes to be more geographically compact
        """
        self.optimizer.max_inter_cell_distance = max_inter_cell_distance
        self.optimizer.compactness_weight = compactness_weight
        
        # Override clustering to enforce geographic constraints
        def enhanced_priority_clustering(self):
            coords = self.h3_metrics[['centroid_lat', 'centroid_lon']].values
            
            # Calculate distance matrix
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(coords, metric='euclidean'))
            
            # Start with highest priority cells
            sorted_cells = self.h3_metrics.sort_values(
                ['priority_score', 'density_score'], 
                ascending=[False, False]
            ).copy()
            
            clusters = np.full(len(self.h3_metrics), -1)
            current_cluster = 0
            
            for idx, cell in sorted_cells.iterrows():
                cell_idx = self.h3_metrics.index.get_loc(idx)
                
                if clusters[cell_idx] != -1:
                    continue
                
                # Start new cluster
                cluster_cells = [cell_idx]
                cluster_customers = cell['customer_count']
                
                # Find nearby cells within distance constraint
                distances = dist_matrix[cell_idx]
                nearby_indices = np.where(
                    (distances <= max_inter_cell_distance) & 
                    (clusters == -1)
                )[0]
                
                # Sort by priority and add to cluster
                nearby_priorities = [self.h3_metrics.iloc[i]['priority_score'] 
                                   for i in nearby_indices if i != cell_idx]
                
                for priority_idx in np.argsort(nearby_priorities)[::-1]:
                    real_idx = nearby_indices[priority_idx]
                    if real_idx == cell_idx:
                        continue
                        
                    additional_customers = self.h3_metrics.iloc[real_idx]['customer_count']
                    
                    if (cluster_customers + additional_customers <= self.max_customers):
                        cluster_cells.append(real_idx)
                        cluster_customers += additional_customers
                
                # Assign cluster if meets minimum requirements
                if cluster_customers >= self.min_customers:
                    for cell_idx in cluster_cells:
                        clusters[cell_idx] = current_cluster
                    current_cluster += 1
            
            return clusters
        
        # Replace the clustering method
        self.optimizer._priority_based_clustering = enhanced_priority_clustering.__get__(
            self.optimizer, type(self.optimizer)
        )
    
    def apply_density_balancing(self, density_variance_penalty=0.5):
        """
        2. DENSITY-BASED BALANCING
        Ensure routes have similar customer density to balance workload
        """
        def density_balanced_evaluation(self, cluster_col):
            cluster_stats = (self.h3_metrics.groupby(cluster_col)
                           .agg({
                               'customer_count': 'sum',
                               'density_score': 'mean',
                               'max_distance_from_fc': 'max',
                               'h3_cell_id': 'count'
                           })).reset_index()
            
            # Standard constraint checks
            customer_valid = ((cluster_stats['customer_count'] >= self.min_customers) & 
                             (cluster_stats['customer_count'] <= self.max_customers))
            distance_valid = cluster_stats['max_distance_from_fc'] <= self.max_distance_km
            
            # Add density balance penalty
            density_variance = cluster_stats['customer_count'].var()
            max_possible_variance = (self.max_customers - self.min_customers) ** 2 / 4
            density_penalty = density_variance / max_possible_variance
            
            constraint_score = (customer_valid & distance_valid).mean()
            balance_score = 1 - (density_penalty * density_variance_penalty)
            
            return 0.7 * constraint_score + 0.3 * balance_score
        
        # Replace evaluation method
        self.optimizer._evaluate_clustering = density_balanced_evaluation.__get__(
            self.optimizer, type(self.optimizer)
        )
    
    def apply_directional_routing(self, angular_sectors=8, sector_preference=0.3):
        """
        3. DIRECTIONAL/ANGULAR ROUTING
        Group cells by angular sectors from fulfillment center
        """
        def calculate_angular_sectors(self):
            # Calculate angles from fulfillment center
            fc_lat, fc_lon = self.fulfillment_center
            
            angles = []
            for _, cell in self.h3_metrics.iterrows():
                delta_lat = cell['centroid_lat'] - fc_lat
                delta_lon = cell['centroid_lon'] - fc_lon
                angle = np.arctan2(delta_lat, delta_lon)
                angles.append(angle)
            
            self.h3_metrics['angle_from_fc'] = angles
            
            # Assign sector preferences
            sector_size = 2 * np.pi / angular_sectors
            self.h3_metrics['preferred_sector'] = (
                (np.array(angles) + np.pi) // sector_size
            ).astype(int)
        
        def angular_clustering(self):
            self.calculate_angular_sectors()
            
            # Group by sectors first, then optimize within sectors
            sector_clusters = []
            current_cluster = 0
            
            for sector in range(angular_sectors):
                sector_cells = self.h3_metrics[
                    self.h3_metrics['preferred_sector'] == sector
                ].copy()
                
                if len(sector_cells) == 0:
                    continue
                
                # Apply priority clustering within sector
                sector_cells_sorted = sector_cells.sort_values(
                    ['distance_from_fc', 'priority_score'],
                    ascending=[True, False]
                )
                
                cluster_customers = 0
                cluster_indices = []
                
                for idx, cell in sector_cells_sorted.iterrows():
                    if cluster_customers + cell['customer_count'] <= self.max_customers:
                        cluster_indices.append(idx)
                        cluster_customers += cell['customer_count']
                    else:
                        # Start new cluster if minimum met
                        if cluster_customers >= self.min_customers:
                            sector_clusters.append({
                                'cluster_id': current_cluster,
                                'indices': cluster_indices
                            })
                            current_cluster += 1
                        
                        # Start new cluster with current cell
                        cluster_indices = [idx]
                        cluster_customers = cell['customer_count']
                
                # Handle last cluster
                if cluster_customers >= self.min_customers:
                    sector_clusters.append({
                        'cluster_id': current_cluster,
                        'indices': cluster_indices
                    })
                    current_cluster += 1
            
            # Assign cluster labels
            clusters = np.full(len(self.h3_metrics), -1)
            for cluster_info in sector_clusters:
                for idx in cluster_info['indices']:
                    clusters[self.h3_metrics.index.get_loc(idx)] = cluster_info['cluster_id']
            
            return clusters
        
        # Add methods to optimizer
        self.optimizer.calculate_angular_sectors = calculate_angular_sectors.__get__(
            self.optimizer, type(self.optimizer)
        )
        self.optimizer._angular_clustering = angular_clustering.__get__(
            self.optimizer, type(self.optimizer)
        )
    
    def apply_customer_count_balancing(self, target_customers_per_route=None, 
                                     balance_tolerance=0.15):
        """
        4. CUSTOMER COUNT BALANCING
        Ensure routes have similar customer counts
        """
        if target_customers_per_route is None:
            target_customers_per_route = (self.optimizer.min_customers + 
                                        self.optimizer.max_customers) / 2
        
        def balanced_merging_splitting(self):
            max_iterations = 20
            
            for iteration in range(max_iterations):
                route_stats = (self.h3_metrics.groupby('route_cluster')
                              .agg({'customer_count': 'sum'}).reset_index())
                
                target_min = target_customers_per_route * (1 - balance_tolerance)
                target_max = target_customers_per_route * (1 + balance_tolerance)
                
                changes_made = False
                
                # Identify imbalanced routes
                small_routes = route_stats[
                    route_stats['customer_count'] < target_min
                ]['route_cluster'].values
                
                large_routes = route_stats[
                    route_stats['customer_count'] > target_max
                ]['route_cluster'].values
                
                # Balance by moving cells between routes
                for small_route in small_routes:
                    for large_route in large_routes:
                        if self._balance_routes(small_route, large_route, 
                                              target_customers_per_route):
                            changes_made = True
                            break
                
                if not changes_made:
                    break
        
        def _balance_routes(self, small_route, large_route, target):
            small_cells = self.h3_metrics[
                self.h3_metrics['route_cluster'] == small_route
            ]
            large_cells = self.h3_metrics[
                self.h3_metrics['route_cluster'] == large_route
            ]
            
            small_total = small_cells['customer_count'].sum()
            large_total = large_cells['customer_count'].sum()
            
            # Find best cell to transfer
            for idx, cell in large_cells.iterrows():
                new_small_total = small_total + cell['customer_count']
                new_large_total = large_total - cell['customer_count']
                
                if (abs(new_small_total - target) < abs(small_total - target) and
                    abs(new_large_total - target) < abs(large_total - target) and
                    new_small_total <= self.max_customers and
                    new_large_total >= self.min_customers):
                    
                    # Transfer cell
                    self.h3_metrics.loc[idx, 'route_cluster'] = small_route
                    return True
            
            return False
        
        # Add methods
        self.optimizer._balanced_merging_splitting = balanced_merging_splitting.__get__(
            self.optimizer, type(self.optimizer)
        )
        self.optimizer._balance_routes = _balance_routes.__get__(
            self.optimizer, type(self.optimizer)
        )

# Usage Examples and Tuning Strategies

def tune_algorithm_for_better_routes(optimizer):
    """
    Apply multiple tuning strategies based on the visualization issues observed
    """
    tuner = RouteOptimizationTuner(optimizer)
    
    # Strategy 1: Enforce geographic compactness (addresses scattered routes)
    tuner.apply_geographic_constraints(
        max_inter_cell_distance=2.5,  # Stricter distance constraint
        compactness_weight=0.9        # Higher compactness importance
    )
    
    # Strategy 2: Balance customer density (addresses uneven workload)
    tuner.apply_density_balancing(
        density_variance_penalty=0.7  # Penalize uneven distribution
    )
    
    # Strategy 3: Use directional routing (creates sector-based routes)
    tuner.apply_directional_routing(
        angular_sectors=6,           # 6 directional sectors
        sector_preference=0.4        # Moderate sector preference
    )
    
    # Strategy 4: Balance customer counts
    tuner.apply_customer_count_balancing(
        target_customers_per_route=170,  # Target around middle of range
        balance_tolerance=0.12           # Allow 12% variation
    )
    
    return tuner

# Additional Manual Override Options

def manual_route_adjustments(optimizer, manual_overrides=None):
    """
    Allow manual adjustments to routes based on business knowledge
    
    manual_overrides format:
    {
        'force_together': [['cell_1', 'cell_2'], ['cell_3', 'cell_4']],
        'force_separate': [['cell_5', 'cell_6']],
        'priority_cells': ['cell_7', 'cell_8'],  # Must be in separate routes
        'max_distance_exceptions': {'cell_9': 8.5}  # Allow longer distance
    }
    """
    if manual_overrides is None:
        return
    
    # Force certain cells together
    if 'force_together' in manual_overrides:
        for cell_group in manual_overrides['force_together']:
            if len(cell_group) > 1:
                base_cluster = optimizer.h3_metrics[
                    optimizer.h3_metrics['h3_cell_id'] == cell_group[0]
                ]['route_cluster'].iloc[0]
                
                for cell_id in cell_group[1:]:
                    optimizer.h3_metrics.loc[
                        optimizer.h3_metrics['h3_cell_id'] == cell_id,
                        'route_cluster'
                    ] = base_cluster
    
    # Force certain cells to separate routes
    if 'force_separate' in manual_overrides:
        for cell_group in manual_overrides['force_separate']:
            for i, cell_id in enumerate(cell_group):
                if i > 0:  # Keep first cell, move others
                    max_cluster = optimizer.h3_metrics['route_cluster'].max()
                    optimizer.h3_metrics.loc[
                        optimizer.h3_metrics['h3_cell_id'] == cell_id,
                        'route_cluster'
                    ] = max_cluster + 1

# Example Usage:

"""
# Initialize your optimizer
optimizer = EnhancedH3RouteOptimizer(...)

# Apply tuning strategies
tuner = tune_algorithm_for_better_routes(optimizer)

# Run optimization with tuned parameters
routes_df, validation_results = optimizer.optimize()

# Apply manual overrides if needed
manual_overrides = {
    'force_together': [['cell_A', 'cell_B']],  # Business requirement
    'force_separate': [['high_priority_cell_1', 'high_priority_cell_2']]
}
manual_route_adjustments(optimizer, manual_overrides)

# Create visualization to check improvements
map_viz = optimizer.create_enhanced_visualization()
"""

# Key Tuning Parameters Summary:

TUNING_PARAMETERS = {
    'geographic_compactness': {
        'max_inter_cell_distance': 2.0,    # Stricter = more compact routes
        'compactness_weight': 0.8,         # Higher = prioritize compactness
    },
    'density_balancing': {
        'density_variance_penalty': 0.6,   # Higher = more balanced workload
    },
    'directional_routing': {
        'angular_sectors': 8,              # More sectors = finer directional control
        'sector_preference': 0.3,          # Higher = stricter sector adherence
    },
    'customer_balancing': {
        'target_customers_per_route': 170, # Adjust based on your preference
        'balance_tolerance': 0.15,         # Lower = more equal route sizes
    },
    'constraint_weights': {
        'distance_weight': 0.4,           # Importance of distance minimization
        'compactness_weight': 0.3,        # Importance of geographic compactness
        'density_weight': 0.3,            # Importance of customer density
    }
}