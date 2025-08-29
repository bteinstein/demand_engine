import pandas as pd
import geopandas as gpd
import numpy as np
import h3  # Requires h3-py v4.x
from haversine import haversine, Unit
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Point, Polygon
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

class H3RouteOptimizer:
    def __init__(self, sp_dim_df, customers_gdf, df_output_assignment, 
                 stock_point_id, min_customers=40, max_customers=300, max_distance_km=7):
        """
        Initialize the H3 Route Optimizer
        
        Parameters:
        - sp_dim_df: DataFrame with fulfillment center info
        - customers_gdf: GeoDataFrame with customer locations
        - df_output_assignment: DataFrame with H3 assignments
        - stock_point_id: Specific stock point ID to optimize routes for
        - min_customers: Minimum customers per route
        - max_customers: Maximum customers per route
        - max_distance_km: Maximum distance from fulfillment center
        """
        self.sp_dim_df = sp_dim_df
        self.customers_gdf = customers_gdf
        self.df_output_assignment = df_output_assignment
        self.stock_point_id = stock_point_id
        self.min_customers = min_customers
        self.max_customers = max_customers
        self.max_distance_km = max_distance_km
        
        # Initialize processed data containers
        self.h3_metrics = None
        self.fulfillment_center = None
        self.stock_point_name = None
        self.routes_df = None
        
    def phase_1_data_preparation(self):
        """Phase 1: Data Preparation & Validation"""
        print("Phase 1: Data Preparation & Validation")
        
        # Filter data for specific stock point
        stock_point_data = self.sp_dim_df[
            self.sp_dim_df['stock_point_id'] == self.stock_point_id
        ]
        
        if len(stock_point_data) == 0:
            raise ValueError(f"Stock point ID {self.stock_point_id} not found in sp_dim_df")
        
        # Extract fulfillment center coordinates and name
        stock_point_row = stock_point_data.iloc[0]
        self.fulfillment_center = (
            stock_point_row['latitude'], 
            stock_point_row['longitude']
        )
        self.stock_point_name = stock_point_row['stock_point_name']
        
        print(f"Stock Point: {self.stock_point_name} (ID: {self.stock_point_id})")
        print(f"Fulfillment Center: {self.fulfillment_center}")
        
        # Filter assignments for this stock point
        self.df_output_assignment = self.df_output_assignment[
            self.df_output_assignment['stock_point_id'] == self.stock_point_id
        ].copy()
        
        if len(self.df_output_assignment) == 0:
            raise ValueError(f"No customer assignments found for stock point ID {self.stock_point_id}")
        
        # Filter customers for this stock point
        customer_ids_for_stock_point = self.df_output_assignment['customer_id'].unique()
        self.customers_gdf = self.customers_gdf[
            self.customers_gdf['customer_id'].isin(customer_ids_for_stock_point)
        ].copy()
        
        # Validate H3 assignments
        unique_h3_cells = self.df_output_assignment['h3_cell_id'].nunique()
        total_customers = len(self.df_output_assignment)
        print(f"Unique H3 cells: {unique_h3_cells}")
        print(f"Total customers: {total_customers}")
        
        # Verify cluster_id = h3_cell_id
        cluster_match = (self.df_output_assignment['cluster_id'] == 
                        self.df_output_assignment['h3_cell_id']).all()
        print(f"Cluster ID matches H3 Cell ID: {cluster_match}")
        
        return True
    
    def phase_2_calculate_metrics(self):
        """Phase 2: Calculate H3 Cell Metrics"""
        print("Phase 2: Calculate H3 Cell Metrics")
        
        # Calculate customer count per H3 cell
        customer_counts = (self.df_output_assignment
                          .groupby('h3_cell_id')['customer_id']
                          .count()
                          .reset_index()
                          .rename(columns={'customer_id': 'customer_count'}))
        
        # Calculate H3 cell centroids and distances
        h3_data = []
        for h3_cell in customer_counts['h3_cell_id']:
            # Get H3 centroid
            lat, lon = h3.cell_to_latlng(h3_cell)
            centroid = (lat, lon)
            
            # Calculate distance to fulfillment center
            distance = haversine(self.fulfillment_center, centroid, unit=Unit.KILOMETERS)
            
            # Get H3 vertices for farthest point calculation
            vertices = h3.cell_to_boundary(h3_cell)
            max_distance = max([haversine(self.fulfillment_center, vertex, unit=Unit.KILOMETERS) 
                               for vertex in vertices])
            
            h3_data.append({
                'h3_cell_id': h3_cell,
                'centroid_lat': lat,
                'centroid_lon': lon,
                'distance_from_fc': distance,
                'max_distance_from_fc': max_distance
            })
        
        h3_df = pd.DataFrame(h3_data)
        
        # Merge with customer counts
        self.h3_metrics = h3_df.merge(customer_counts, on='h3_cell_id')
        
        # Filter by distance constraint
        self.h3_metrics = self.h3_metrics[
            self.h3_metrics['max_distance_from_fc'] <= self.max_distance_km
        ].copy()
        
        # Calculate density score (customers per unit area)
        h3_area = h3.average_hexagon_area(8, unit='km^2')  # Resolution 8 area
        self.h3_metrics['density_score'] = (
            self.h3_metrics['customer_count'] / h3_area
        )
        
        print(f"H3 cells within {self.max_distance_km}km: {len(self.h3_metrics)}")
        print(f"Total customers in valid cells: {self.h3_metrics['customer_count'].sum()}")
        
        return self.h3_metrics
    
    def calculate_geographic_compactness_features(self):
        """Calculate features that promote geographic compactness"""
        coords = self.h3_metrics[['centroid_lat', 'centroid_lon']].values
        
        # Calculate distance matrix between H3 cells
        dist_matrix = pdist(coords, metric='euclidean')
        dist_matrix_square = squareform(dist_matrix)
        
        # For each cell, calculate average distance to k nearest neighbors
        k_neighbors = min(5, len(coords) - 1)
        nearest_distances = []
        
        for i in range(len(coords)):
            distances = dist_matrix_square[i]
            # Exclude self (distance = 0)
            distances = distances[distances > 0]
            if len(distances) >= k_neighbors:
                avg_nearest = np.mean(np.sort(distances)[:k_neighbors])
            else:
                avg_nearest = np.mean(distances)
            nearest_distances.append(avg_nearest)
        
        self.h3_metrics['avg_neighbor_distance'] = nearest_distances
        return self.h3_metrics
    
    def phase_3_improved_clustering(self):
        """Phase 3: Improved Geographic Clustering with Compactness"""
        print("Phase 3: Improved Geographic Clustering")
        
        # Add compactness features
        self.calculate_geographic_compactness_features()
        
        # Prepare features for clustering
        features = self.h3_metrics[['centroid_lat', 'centroid_lon', 
                                   'distance_from_fc', 'density_score',
                                   'avg_neighbor_distance']].copy()
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Estimate number of routes needed
        total_customers = self.h3_metrics['customer_count'].sum()
        estimated_routes = max(1, int(total_customers / 
                                    ((self.min_customers + self.max_customers) / 2)))
        
        print(f"Estimated routes needed: {estimated_routes}")
        
        # Try K-means first for better balance
        kmeans = KMeans(n_clusters=estimated_routes, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        
        # Evaluate K-means results
        self.h3_metrics['kmeans_cluster'] = kmeans_labels
        kmeans_valid = self._evaluate_clustering('kmeans_cluster')
        
        # Try DBSCAN with adaptive eps
        coords = features[['centroid_lat', 'centroid_lon']].values
        distances = pdist(coords)
        eps_adaptive = np.percentile(distances, 20)  # 20th percentile
        
        min_samples = max(2, int(self.min_customers / 
                                self.h3_metrics['customer_count'].mean()))
        
        dbscan = DBSCAN(eps=eps_adaptive, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(coords)
        
        # Handle noise points in DBSCAN
        if -1 in dbscan_labels:
            # Assign noise points to nearest cluster
            dbscan_labels = self._assign_noise_points(dbscan_labels, coords)
        
        self.h3_metrics['dbscan_cluster'] = dbscan_labels
        dbscan_valid = self._evaluate_clustering('dbscan_cluster')
        
        # Choose best clustering method
        if kmeans_valid >= dbscan_valid:
            chosen_method = 'kmeans_cluster'
            print(f"Using K-means clustering (score: {kmeans_valid:.2f})")
        else:
            chosen_method = 'dbscan_cluster'
            print(f"Using DBSCAN clustering (score: {dbscan_valid:.2f})")
        
        self.h3_metrics['route_cluster'] = self.h3_metrics[chosen_method]
        
        # Remove temporary columns
        self.h3_metrics = self.h3_metrics.drop(['kmeans_cluster', 'dbscan_cluster'], 
                                              axis=1)
        
        return self.h3_metrics
    
    def _assign_noise_points(self, labels, coords):
        """Assign DBSCAN noise points to nearest cluster"""
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        
        noise_indices = np.where(noise_mask)[0]
        cluster_indices = np.where(~noise_mask)[0]
        
        if len(cluster_indices) == 0:
            # All points are noise, assign sequential labels
            return np.arange(len(labels))
        
        # Calculate distances from noise points to clustered points
        for noise_idx in noise_indices:
            noise_point = coords[noise_idx]
            distances = [haversine(noise_point, coords[cluster_idx], unit=Unit.KILOMETERS) 
                        for cluster_idx in cluster_indices]
            nearest_cluster_idx = cluster_indices[np.argmin(distances)]
            labels[noise_idx] = labels[nearest_cluster_idx]
        
        return labels
    
    def _evaluate_clustering(self, cluster_col):
        """Evaluate clustering quality based on constraints"""
        cluster_stats = (self.h3_metrics.groupby(cluster_col)
                        .agg({
                            'customer_count': 'sum',
                            'max_distance_from_fc': 'max',
                            'h3_cell_id': 'count'
                        }).reset_index())
        
        # Check constraints
        customer_valid = ((cluster_stats['customer_count'] >= self.min_customers) & 
                         (cluster_stats['customer_count'] <= self.max_customers))
        distance_valid = cluster_stats['max_distance_from_fc'] <= self.max_distance_km
        
        # Calculate score
        valid_routes = (customer_valid & distance_valid).sum()
        total_routes = len(cluster_stats)
        
        if total_routes == 0:
            return 0
        
        return valid_routes / total_routes
    
    def phase_4_constraint_enforcement(self):
        """Phase 4: Enforce Constraints with Smart Merging/Splitting"""
        print("Phase 4: Constraint Enforcement")
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"Iteration {iteration}")
            
            # Calculate current route statistics
            route_stats = (self.h3_metrics.groupby('route_cluster')
                          .agg({
                              'customer_count': 'sum',
                              'max_distance_from_fc': 'max',
                              'h3_cell_id': 'count',
                              'centroid_lat': 'mean',
                              'centroid_lon': 'mean'
                          }).reset_index())
            
            changes_made = False
            
            # Handle routes with too few customers
            small_routes = route_stats[
                route_stats['customer_count'] < self.min_customers
            ]['route_cluster'].values
            
            for small_route in small_routes:
                merged = self._merge_small_route(small_route, route_stats)
                if merged:
                    changes_made = True
            
            # Handle routes with too many customers
            large_routes = route_stats[
                route_stats['customer_count'] > self.max_customers
            ]['route_cluster'].values
            
            for large_route in large_routes:
                split = self._split_large_route(large_route)
                if split:
                    changes_made = True
            
            # Handle routes exceeding distance constraint
            distant_routes = route_stats[
                route_stats['max_distance_from_fc'] > self.max_distance_km
            ]['route_cluster'].values
            
            for distant_route in distant_routes:
                trimmed = self._trim_distant_cells(distant_route)
                if trimmed:
                    changes_made = True
            
            if not changes_made:
                print("Convergence reached")
                break
        
        return self.h3_metrics
    
    def _merge_small_route(self, small_route, route_stats):
        """Merge small route with nearest compatible route"""
        small_route_data = route_stats[route_stats['route_cluster'] == small_route].iloc[0]
        small_route_center = (small_route_data['centroid_lat'], small_route_data['centroid_lon'])
        
        # Find potential merge candidates
        candidates = route_stats[
            (route_stats['route_cluster'] != small_route) &
            (route_stats['customer_count'] + small_route_data['customer_count'] <= self.max_customers)
        ].copy()
        
        if len(candidates) == 0:
            return False
        
        # Calculate distances to candidates
        candidates['merge_distance'] = candidates.apply(
            lambda row: haversine(small_route_center, (row['centroid_lat'], row['centroid_lon']), unit=Unit.KILOMETERS),
            axis=1
        )
        
        # Merge with nearest candidate
        best_candidate = candidates.loc[candidates['merge_distance'].idxmin(), 'route_cluster']
        
        # Update cluster assignments
        self.h3_metrics.loc[
            self.h3_metrics['route_cluster'] == small_route, 
            'route_cluster'
        ] = best_candidate
        
        return True
    
    def _split_large_route(self, large_route):
        """Split large route geographically"""
        route_cells = self.h3_metrics[self.h3_metrics['route_cluster'] == large_route].copy()
        
        if len(route_cells) < 2:
            return False
        
        # Use K-means to split into 2 sub-routes
        coords = route_cells[['centroid_lat', 'centroid_lon']].values
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        sub_clusters = kmeans.fit_predict(coords)
        
        # Assign new cluster IDs
        max_cluster_id = self.h3_metrics['route_cluster'].max()
        route_cells['new_cluster'] = sub_clusters + max_cluster_id + 1
        
        # Update main dataframe
        for idx, row in route_cells.iterrows():
            self.h3_metrics.loc[idx, 'route_cluster'] = row['new_cluster']
        
        return True
    
    def _trim_distant_cells(self, distant_route):
        """Remove cells that are too far from fulfillment center"""
        route_cells = self.h3_metrics[self.h3_metrics['route_cluster'] == distant_route].copy()
        
        # Sort by distance and keep only cells within constraint
        route_cells = route_cells.sort_values('max_distance_from_fc')
        valid_cells = route_cells[route_cells['max_distance_from_fc'] <= self.max_distance_km]
        
        if len(valid_cells) == len(route_cells):
            return False  # No changes needed
        
        if len(valid_cells) == 0:
            # Remove entire route if no valid cells
            self.h3_metrics = self.h3_metrics[
                self.h3_metrics['route_cluster'] != distant_route
            ]
        else:
            # Keep only valid cells
            cells_to_remove = set(route_cells['h3_cell_id']) - set(valid_cells['h3_cell_id'])
            self.h3_metrics = self.h3_metrics[
                ~self.h3_metrics['h3_cell_id'].isin(cells_to_remove)
            ]
        
        return True
    
    def phase_5_generate_output(self):
        """Phase 5: Generate Final Output DataFrame"""
        print("Phase 5: Generate Output DataFrame")
        
        # Calculate route statistics
        route_summary = (self.h3_metrics.groupby('route_cluster')
                        .agg({
                            'customer_count': 'sum',
                            'distance_from_fc': 'mean',
                            'max_distance_from_fc': 'max',
                            'h3_cell_id': 'count',
                            'centroid_lat': ['mean', 'std'],
                            'centroid_lon': ['mean', 'std']
                        }))
        
        # Flatten column names
        route_summary.columns = ['_'.join(col).strip() for col in route_summary.columns]
        route_summary = route_summary.reset_index()
        
        # Calculate compactness score (inverse of coordinate standard deviation)
        route_summary['compactness_score'] = 1 / (
            route_summary['centroid_lat_std'] + route_summary['centroid_lon_std'] + 0.001
        )
        
        # Normalize compactness score to 0-1 range
        max_compactness = route_summary['compactness_score'].max()
        route_summary['compactness_score'] = (
            route_summary['compactness_score'] / max_compactness
        )
        
        # Calculate estimated delivery time (simplified model)
        # Assumptions: 30 km/h average speed, 5 minutes per stop
        avg_speed_kmh = 30
        stop_time_hours = 5/60  # 5 minutes in hours
        
        route_summary['estimated_delivery_time_hours'] = (
            route_summary['distance_from_fc_mean'] * 2 / avg_speed_kmh +  # Round trip
            route_summary['h3_cell_id_count'] * stop_time_hours  # Stop time
        )
        
        # Create long-format output
        output_rows = []
        for _, route in route_summary.iterrows():
            route_cells = self.h3_metrics[
                self.h3_metrics['route_cluster'] == route['route_cluster']
            ]
            
            for _, cell in route_cells.iterrows():
                output_rows.append({
                    'stock_point_id': self.stock_point_id,
                    'stock_point_name': self.stock_point_name,
                    'route_id': f"route_{self.stock_point_id}_{int(route['route_cluster'])}",
                    'h3_cell_id': cell['h3_cell_id'],
                    'customer_count': route['customer_count_sum'],
                    'total_distance_km': route['distance_from_fc_mean'] * 2,  # Round trip
                    'estimated_delivery_time_hours': route['estimated_delivery_time_hours'],
                    'compactness_score': route['compactness_score']
                })
        
        self.routes_df = pd.DataFrame(output_rows)
        return self.routes_df
    
    def generate_route_summary(self):
        """Generate route summary with aggregated statistics"""
        print("Generating Route Summary")
        
        if self.routes_df is None:
            raise ValueError("Routes not generated yet. Run optimize() first.")
        
        # Get assignment confidence data
        assignment_confidence = (self.df_output_assignment
                               .groupby('h3_cell_id')['assignment_confidence']
                               .mean()
                               .reset_index())
        
        # Merge with h3_metrics to get confidence per cell
        h3_with_confidence = self.h3_metrics.merge(
            assignment_confidence, on='h3_cell_id', how='left'
        )
        
        # Calculate route-level statistics
        route_summary_data = []
        
        for route_id in self.routes_df['route_id'].unique():
            route_cells = self.routes_df[self.routes_df['route_id'] == route_id]
            route_h3_cells = route_cells['h3_cell_id'].tolist()
            
            # Get detailed metrics for this route
            route_metrics = h3_with_confidence[
                h3_with_confidence['h3_cell_id'].isin(route_h3_cells)
            ]
            
            # Calculate cumulative distance (TSP approximation)
            # Sort cells by distance from fulfillment center for simple routing
            route_metrics_sorted = route_metrics.sort_values('distance_from_fc')
            
            # Calculate cumulative distance as sum of inter-cell distances
            cumulative_distance = 0
            prev_coord = self.fulfillment_center
            
            for _, cell in route_metrics_sorted.iterrows():
                curr_coord = (cell['centroid_lat'], cell['centroid_lon'])
                cumulative_distance += haversine(prev_coord, curr_coord, unit=Unit.KILOMETERS)
                prev_coord = curr_coord
            
            # Add return trip to fulfillment center
            if len(route_metrics_sorted) > 0:
                last_coord = (route_metrics_sorted.iloc[-1]['centroid_lat'], 
                            route_metrics_sorted.iloc[-1]['centroid_lon'])
                cumulative_distance += haversine(last_coord, self.fulfillment_center, unit=Unit.KILOMETERS)
            
            # Calculate other metrics
            farthest_distance = route_metrics['distance_from_fc'].max() if len(route_metrics) > 0 else 0
            avg_confidence = route_metrics['assignment_confidence'].mean() if len(route_metrics) > 0 else 0
            
            # Get base route info
            route_info = route_cells.iloc[0]
            
            route_summary_data.append({
                'stock_point_id': route_info['stock_point_id'],
                'stock_point_name': route_info['stock_point_name'],
                'route_id': route_id,
                'h3_cell_ids': route_h3_cells,
                'cluster_count': len(route_h3_cells),
                'customer_count': route_info['customer_count'],
                'total_distance_km': route_info['total_distance_km'],
                'cumulative_distance_km': cumulative_distance,
                'farthest_centroid_distance_km': farthest_distance,
                'estimated_delivery_time_hours': route_info['estimated_delivery_time_hours'],
                'avg_assignment_confidence': avg_confidence,
                'compactness_score': route_info['compactness_score']
            })
        
        self.route_summary_df = pd.DataFrame(route_summary_data)
        return self.route_summary_df
    
    def create_route_visualization(self, save_path=None, show_customer_points=False):
        """
        Create interactive map visualization of optimized routes
        
        Parameters:
        - save_path: Path to save HTML file (optional)
        - show_customer_points: Whether to show individual customer points
        
        Returns:
        - folium.Map object
        """
        print("Creating Route Visualization")
        
        if self.routes_df is None:
            raise ValueError("Routes not generated yet. Run optimize() first.")
        
        # Create base map centered on fulfillment center
        m = folium.Map(
            location=self.fulfillment_center,
            zoom_start=11,
            tiles="Cartodb Positron"
            # tiles='OpenStreetMap'
        )
        
        # Add fulfillment center marker
        folium.Marker(
            location=self.fulfillment_center,
            popup=f'<b>{self.stock_point_name}</b><br>Stock Point ID: {self.stock_point_id}',
            tooltip='Fulfillment Center',
            icon=folium.Icon(color='red', icon='home', prefix='fa')
        ).add_to(m)
        
        # Define colors for routes
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
                 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 
                 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
        
        # Process each route
        route_ids = self.routes_df['route_id'].unique()
        
        for i, route_id in enumerate(route_ids):
            route_cells = self.routes_df[self.routes_df['route_id'] == route_id]
            route_h3_cells = route_cells['h3_cell_id'].tolist()
            color = colors[i % len(colors)]
            
            # Get route metrics
            route_metrics = self.h3_metrics[
                self.h3_metrics['h3_cell_id'].isin(route_h3_cells)
            ]
            
            # Create feature group for this route
            route_group = folium.FeatureGroup(name=f'{route_id} ({len(route_h3_cells)} cells)')
            
            # Add H3 cell polygons
            for _, cell_data in route_metrics.iterrows():
                h3_cell = cell_data['h3_cell_id']
                
                # Get H3 cell boundary
                boundary = h3.cell_to_boundary(h3_cell)
                # Convert to lat, lon format for folium
                boundary_coords = [(lat, lon) for lat, lon in boundary]
                
                # Create popup content
                popup_content = f"""
                <b>H3 Cell:</b> {h3_cell}<br>
                <b>Route:</b> {route_id}<br>
                <b>Customers:</b> {cell_data['customer_count']}<br>
                <b>Distance from FC:</b> {cell_data['distance_from_fc']:.2f} km<br>
                <b>Density Score:</b> {cell_data['density_score']:.2f}
                """
                
                # Add H3 cell polygon
                folium.Polygon(
                    locations=boundary_coords,
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f'{route_id}: {cell_data["customer_count"]} customers',
                    color=color,
                    weight=2,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(route_group)
                
                # Add centroid marker
                folium.CircleMarker(
                    location=(cell_data['centroid_lat'], cell_data['centroid_lon']),
                    radius=5,
                    popup=popup_content,
                    tooltip=f'{cell_data["customer_count"]} customers',
                    color=color,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(route_group)
            
            # Add route path (simplified - connects centroids in distance order)
            if len(route_metrics) > 1:
                route_coords = [self.fulfillment_center]  # Start from fulfillment center
                
                # Sort by distance for simple routing
                route_sorted = route_metrics.sort_values('distance_from_fc')
                for _, cell in route_sorted.iterrows():
                    route_coords.append((cell['centroid_lat'], cell['centroid_lon']))
                
                # Return to fulfillment center
                route_coords.append(self.fulfillment_center)
                
                # Add route line
                folium.PolyLine(
                    locations=route_coords,
                    color=color,
                    weight=3,
                    opacity=0.7,
                    popup=f'{route_id} Route Path'
                ).add_to(route_group)
            
            route_group.add_to(m)
        
        # Add customer points if requested
        if show_customer_points:
            customer_group = folium.FeatureGroup(name='Customer Points')
            
            # Get customers for this stock point
            stock_customers = self.customers_gdf[
                self.customers_gdf['customer_id'].isin(
                    self.df_output_assignment['customer_id']
                )
            ]
            
            for _, customer in stock_customers.iterrows():
                folium.CircleMarker(
                    location=(customer.geometry.y, customer.geometry.x),
                    radius=2,
                    popup=f'Customer: {customer["customer_id"]}',
                    color='black',
                    fillColor='yellow',
                    fillOpacity=0.6
                ).add_to(customer_group)
            
            customer_group.add_to(m)
        
        # Add distance constraint circle
        folium.Circle(
            location=self.fulfillment_center,
            radius=self.max_distance_km * 1000,  # Convert km to meters
            popup=f'Max Distance: {self.max_distance_km} km',
            color='red',
            weight=2,
            fill=False,
            dashArray='5, 5'
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add route summary as HTML
        route_summary_html = self._generate_summary_html()
        folium.Element(route_summary_html).add_to(m)
        
        # Save if path provided
        if save_path:
            m.save(save_path)
            print(f"Map saved to: {save_path}")
        
        return m
    
    def _generate_summary_html(self):
        """Generate HTML summary for map display"""
        if hasattr(self, 'route_summary_df') and self.route_summary_df is not None:
            summary_stats = {
                'total_routes': len(self.route_summary_df),
                'total_customers': self.route_summary_df['customer_count'].sum(),
                'avg_customers_per_route': self.route_summary_df['customer_count'].mean(),
                'avg_distance': self.route_summary_df['farthest_centroid_distance_km'].mean(),
                'avg_compactness': self.route_summary_df['compactness_score'].mean()
            }
        else:
            route_stats = self.routes_df.groupby('route_id').agg({
                'customer_count': 'first',
                'total_distance_km': 'first',
                'compactness_score': 'first'
            })
            
            summary_stats = {
                'total_routes': len(route_stats),
                'total_customers': route_stats['customer_count'].sum(),
                'avg_customers_per_route': route_stats['customer_count'].mean(),
                'avg_distance': route_stats['total_distance_km'].mean() / 2,
                'avg_compactness': route_stats['compactness_score'].mean()
            }
        
        html = f"""
        <div style='position: fixed; 
                    top: 10px; right: 10px; width: 300px; height: 120px; 
                    background-color: white; border: 2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px'>
        <h4>{self.stock_point_name} - Route Summary</h4>
        <p><b>Total Routes:</b> {summary_stats['total_routes']}</p>
        <p><b>Total Customers:</b> {summary_stats['total_customers']}</p>
        <p><b>Avg Customers/Route:</b> {summary_stats['avg_customers_per_route']:.1f}</p>
        <p><b>Avg Distance:</b> {summary_stats['avg_distance']:.1f} km</p>
        <p><b>Avg Compactness:</b> {summary_stats['avg_compactness']:.3f}</p>
        </div>
        """
        return html
    
    def phase_6_validation(self):
        """Phase 6: Validate Results"""
        print("Phase 6: Validation & Summary")
        
        # Route-level validation
        route_validation = (self.routes_df.groupby('route_id')
                           .agg({
                               'customer_count': 'first',
                               'total_distance_km': 'first',
                               'h3_cell_id': 'count'
                           }))
        
        # Check constraints
        customer_violations = (
            (route_validation['customer_count'] < self.min_customers) |
            (route_validation['customer_count'] > self.max_customers)
        ).sum()
        
        distance_violations = (
            route_validation['total_distance_km'] / 2 > self.max_distance_km
        ).sum()
        
        print(f"\nValidation Results:")
        print(f"Total routes: {len(route_validation)}")
        print(f"Customer constraint violations: {customer_violations}")
        print(f"Distance constraint violations: {distance_violations}")
        print(f"Average customers per route: {route_validation['customer_count'].mean():.1f}")
        print(f"Average distance per route: {route_validation['total_distance_km'].mean()/2:.1f} km")
        print(f"Average compactness score: {self.routes_df['compactness_score'].mean():.3f}")
        
        return {
            'total_routes': len(route_validation),
            'customer_violations': customer_violations,
            'distance_violations': distance_violations,
            'avg_customers': route_validation['customer_count'].mean(),
            'avg_distance': route_validation['total_distance_km'].mean() / 2,
            'avg_compactness': self.routes_df['compactness_score'].mean()
        }
    
    def optimize(self):
        """Run complete optimization pipeline"""
        print("Starting H3 Route Optimization Pipeline")
        print("=" * 50)
        
        # Execute all phases
        self.phase_1_data_preparation()
        self.phase_2_calculate_metrics()
        self.phase_3_improved_clustering()
        self.phase_4_constraint_enforcement()
        self.phase_5_generate_output()
        validation_results = self.phase_6_validation()
        
        print("\nOptimization Complete!")
        return self.routes_df, validation_results

# Example usage:
def main():
    """
    Example usage of the H3RouteOptimizer
    
    Replace with your actual datasets:
    - sp_dim_df: DataFrame with fulfillment center info
    - customers_gdf: GeoDataFrame with customer locations  
    - df_output_assignment: DataFrame with H3 cell assignments
    """
    
    # Initialize optimizer for specific stock point
    # optimizer = H3RouteOptimizer(
    #     sp_dim_df=sp_dim_df,
    #     customers_gdf=customers_gdf, 
    #     df_output_assignment=df_output_assignment,
    #     stock_point_id="SP001",  # Specify the stock point ID
    #     min_customers=40,
    #     max_customers=300,
    #     max_distance_km=7
    # )
    
    # # Run optimization
    # routes_df, validation_results = optimizer.optimize()
    
    # # Generate route summary
    # route_summary_df = optimizer.generate_route_summary()
    
    # # Create visualization
    # map_viz = optimizer.create_route_visualization(
    #     save_path=f'route_map_{optimizer.stock_point_id}.html',
    #     show_customer_points=True
    # )
    
    # # Display results
    # print(f"\nOptimized routes for Stock Point: {optimizer.stock_point_name}")
    # print("\nDetailed Routes:")
    # print(routes_df.head(10))
    # print("\nRoute Summary:")
    # print(route_summary_df)
    
    # # Save results with stock point identifier
    # routes_df.to_csv(f'optimized_routes_{optimizer.stock_point_id}.csv', index=False)
    # route_summary_df.to_csv(f'route_summary_{optimizer.stock_point_id}.csv', index=False)
    # print(f"\nResults saved!")
    
    pass

# Utility function for batch processing multiple stock points
def optimize_multiple_stock_points(sp_dim_df, customers_gdf, df_output_assignment, 
                                 stock_point_ids=None, **kwargs):
    """
    Optimize routes for multiple stock points
    
    Parameters:
    - sp_dim_df, customers_gdf, df_output_assignment: Input datasets
    - stock_point_ids: List of stock point IDs to process (None = all)
    - **kwargs: Additional parameters for H3RouteOptimizer
    
    Returns:
    - combined_routes_df: DataFrame with routes for all stock points
    - validation_summary: Dictionary with validation results per stock point
    """
    
    if stock_point_ids is None:
        stock_point_ids = sp_dim_df['stock_point_id'].unique()
    
    all_routes = []
    validation_summary = {}
    
    for stock_point_id in stock_point_ids:
        print(f"\n{'='*60}")
        print(f"Processing Stock Point: {stock_point_id}")
        print(f"{'='*60}")
        
        try:
            optimizer = H3RouteOptimizer(
                sp_dim_df=sp_dim_df,
                customers_gdf=customers_gdf,
                df_output_assignment=df_output_assignment,
                stock_point_id=stock_point_id,
                **kwargs
            )
            
            routes_df, validation_results = optimizer.optimize()
            all_routes.append(routes_df)
            validation_summary[stock_point_id] = validation_results
            
        except Exception as e:
            print(f"Error processing stock point {stock_point_id}: {str(e)}")
            validation_summary[stock_point_id] = {"error": str(e)}
    
    # Combine all routes
    if all_routes:
        combined_routes_df = pd.concat(all_routes, ignore_index=True)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY - ALL STOCK POINTS")
        print(f"{'='*60}")
        print(f"Total stock points processed: {len(validation_summary)}")
        print(f"Total routes generated: {len(combined_routes_df['route_id'].unique())}")
        print(f"Total H3 cells: {len(combined_routes_df)}")
        
        return combined_routes_df, validation_summary
    else:
        print("No routes generated for any stock point")
        return pd.DataFrame(), validation_summary

if __name__ == "__main__":
    main()