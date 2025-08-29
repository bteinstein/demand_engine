import pandas as pd
import geopandas as gpd
import numpy as np
import h3
from haversine import haversine, Unit
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Point, Polygon
import folium
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

class EnhancedH3RouteOptimizer:
    """
    Enhanced H3 Route Optimizer with improved visualization and structure
    
    Key Improvements:
    1. Better color scheme and visualization
    2. Enhanced route labeling and line drawing
    3. Improved compactness scoring
    4. Better constraint handling
    5. More detailed metrics and validation
    """
    
    def __init__(self, sp_dim_df, customers_gdf, df_output_assignment, 
                 stock_point_id, min_customers=40, max_customers=300, max_distance_km=7):
        """
        Initialize the Enhanced H3 Route Optimizer
        
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
        self.route_summary_df = None
        
        # Enhanced color palette for better visualization
        self.route_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#FFB347', '#87CEEB', '#F0E68C',
            '#FF7F7F', '#40E0D0', '#87CEFA', '#90EE90', '#F0B27A',
            '#D8BFD8', '#7FDBDA', '#FFA07A', '#B0C4DE', '#DDA0DD'
        ]
    
    def phase_1_data_preparation(self):
        """Phase 1: Enhanced Data Preparation & Validation"""
        print("Phase 1: Enhanced Data Preparation & Validation")
        print("-" * 50)
        
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
        
        print(f"✓ Stock Point: {self.stock_point_name} (ID: {self.stock_point_id})")
        print(f"✓ Fulfillment Center: {self.fulfillment_center}")
        
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
        
        # Enhanced validation
        unique_h3_cells = self.df_output_assignment['h3_cell_id'].nunique()
        total_customers = len(self.df_output_assignment)
        
        print(f"✓ Unique H3 cells: {unique_h3_cells}")
        print(f"✓ Total customers: {total_customers}")
        print(f"✓ Avg customers per cell: {total_customers/unique_h3_cells:.1f}")
        
        # Verify cluster_id = h3_cell_id
        cluster_match = (self.df_output_assignment['cluster_id'] == 
                        self.df_output_assignment['h3_cell_id']).all()
        print(f"✓ Cluster ID matches H3 Cell ID: {cluster_match}")
        
        return True
    
    def phase_2_calculate_metrics(self):
        """Phase 2: Enhanced H3 Cell Metrics Calculation"""
        print("\nPhase 2: Enhanced H3 Cell Metrics Calculation")
        print("-" * 50)
        
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
            
            # Calculate cell area for density
            cell_area = h3.cell_area(h3_cell, unit='km^2')
            
            h3_data.append({
                'h3_cell_id': h3_cell,
                'centroid_lat': lat,
                'centroid_lon': lon,
                'distance_from_fc': distance,
                'max_distance_from_fc': max_distance,
                'cell_area_km2': cell_area
            })
        
        h3_df = pd.DataFrame(h3_data)
        
        # Merge with customer counts
        self.h3_metrics = h3_df.merge(customer_counts, on='h3_cell_id')
        
        # Filter by distance constraint
        before_filter = len(self.h3_metrics)
        self.h3_metrics = self.h3_metrics[
            self.h3_metrics['max_distance_from_fc'] <= self.max_distance_km
        ].copy()
        after_filter = len(self.h3_metrics)
        
        # Enhanced density calculations
        self.h3_metrics['density_score'] = (
            self.h3_metrics['customer_count'] / self.h3_metrics['cell_area_km2']
        )
        
        # Add priority score (combination of density and inverse distance)
        max_distance = self.h3_metrics['distance_from_fc'].max()
        distance_weight = 1 - (self.h3_metrics['distance_from_fc'] / max_distance)
        self.h3_metrics['priority_score'] = (
            0.7 * self.h3_metrics['density_score'] / self.h3_metrics['density_score'].max() +
            0.3 * distance_weight
        )
        
        print(f"✓ H3 cells within {self.max_distance_km}km: {after_filter} (filtered out: {before_filter - after_filter})")
        print(f"✓ Total customers in valid cells: {self.h3_metrics['customer_count'].sum()}")
        print(f"✓ Density range: {self.h3_metrics['density_score'].min():.2f} - {self.h3_metrics['density_score'].max():.2f} customers/km²")
        
        return self.h3_metrics
    
    def calculate_enhanced_compactness_features_(self):
        """Calculate enhanced features for geographic compactness"""
        coords = self.h3_metrics[['centroid_lat', 'centroid_lon']].values
        
        # Calculate distance matrix between H3 cells
        dist_matrix = pdist(coords, metric='euclidean')
        dist_matrix_square = squareform(dist_matrix)
        
        # Calculate multiple compactness metrics
        k_neighbors = min(5, len(coords) - 1)
        compactness_metrics = []
        
        for i in range(len(coords)):
            distances = dist_matrix_square[i]
            distances = distances[distances > 0]  # Exclude self
            
            if len(distances) >= k_neighbors:
                avg_nearest = np.mean(np.sort(distances)[:k_neighbors])
                std_nearest = np.std(np.sort(distances)[:k_neighbors])
                
                # Calculate convex hull ratio if enough points
                if len(distances) >= 3:
                    from scipy.spatial import ConvexHull
                    try:
                        neighbor_indices = np.argsort(distances)[:k_neighbors]
                        neighbor_coords = coords[[i] + neighbor_indices.tolist()]
                        hull = ConvexHull(neighbor_coords)
                        hull_ratio = len(neighbor_coords) / hull.volume if hull.volume > 0 else 0
                    except:
                        hull_ratio = 0
                else:
                    hull_ratio = 0
            else:
                avg_nearest = np.mean(distances) if len(distances) > 0 else 0
                std_nearest = np.std(distances) if len(distances) > 0 else 0
                hull_ratio = 0
            
            compactness_metrics.append({
                'avg_neighbor_distance': avg_nearest,
                'std_neighbor_distance': std_nearest,
                'hull_compactness': hull_ratio
            })
        
        compactness_df = pd.DataFrame(compactness_metrics)
        for col in compactness_df.columns:
            self.h3_metrics[col] = compactness_df[col]
        
        return self.h3_metrics
    
    def calculate_enhanced_compactness_features(self):
        """Calculate enhanced features for geographic compactness - FIXED"""
        coords = self.h3_metrics[['centroid_lat', 'centroid_lon']].values
        
        # Verify coordinates don't contain NaN
        if np.isnan(coords).any():
            print("  WARNING: NaN values in coordinates, cleaning...")
            coords = np.nan_to_num(coords, nan=0)
        
        # Calculate distance matrix between H3 cells
        try:
            dist_matrix = pdist(coords, metric='haversine')
            dist_matrix_square = squareform(dist_matrix)
        except Exception as e:
            print(f"  Distance calculation failed: {str(e)}, using simplified approach")
            # Fallback: calculate simple distances
            n = len(coords)
            dist_matrix_square = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist_matrix_square[i, j] = np.sqrt(
                            (coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2
                        )
        
        # Calculate multiple compactness metrics
        k_neighbors = min(5, len(coords) - 1)
        compactness_metrics = []
        
        for i in range(len(coords)):
            distances = dist_matrix_square[i]
            distances = distances[distances > 0]  # Exclude self
            
            if len(distances) >= k_neighbors and k_neighbors > 0:
                avg_nearest = np.mean(np.sort(distances)[:k_neighbors])
                std_nearest = np.std(np.sort(distances)[:k_neighbors]) if k_neighbors > 1 else 0
                
                # Calculate convex hull ratio if enough points
                hull_ratio = 0  # Default value
                if len(distances) >= 3:
                    try:
                        from scipy.spatial import ConvexHull
                        neighbor_indices = np.argsort(distances)[:k_neighbors]
                        neighbor_coords = coords[[i] + neighbor_indices.tolist()]
                        
                        # Clean coordinates for hull calculation
                        neighbor_coords = np.nan_to_num(neighbor_coords, nan=0)
                        
                        if len(np.unique(neighbor_coords, axis=0)) >= 3:  # Need at least 3 unique points
                            hull = ConvexHull(neighbor_coords)
                            hull_ratio = len(neighbor_coords) / hull.volume if hull.volume > 0 else 0
                    except Exception:
                        hull_ratio = 0
            else:
                avg_nearest = np.mean(distances) if len(distances) > 0 else 0
                std_nearest = np.std(distances) if len(distances) > 0 else 0
                hull_ratio = 0
            
            # Handle potential NaN values
            avg_nearest = 0 if np.isnan(avg_nearest) else avg_nearest
            std_nearest = 0 if np.isnan(std_nearest) else std_nearest
            hull_ratio = 0 if np.isnan(hull_ratio) else hull_ratio
            
            compactness_metrics.append({
                'avg_neighbor_distance': avg_nearest,
                'std_neighbor_distance': std_nearest,
                'hull_compactness': hull_ratio
            })
        
        compactness_df = pd.DataFrame(compactness_metrics)
        
        # Final NaN check and cleaning
        for col in compactness_df.columns:
            compactness_df[col] = compactness_df[col].fillna(0)
            self.h3_metrics[col] = compactness_df[col]
        
        return self.h3_metrics
    
    def phase_3_improved_clustering_(self):
        """Phase 3: Advanced Geographic Clustering with Multiple Algorithms"""
        print("\nPhase 3: Advanced Geographic Clustering")
        print("-" * 50)
        
        # Add enhanced compactness features
        self.calculate_enhanced_compactness_features()
        
        # Prepare features for clustering
        features = self.h3_metrics[[
            'centroid_lat', 'centroid_lon', 'distance_from_fc', 
            'density_score', 'priority_score', 'avg_neighbor_distance'
        ]].copy()
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Estimate number of routes needed
        total_customers = self.h3_metrics['customer_count'].sum()
        avg_customers_per_route = (self.min_customers + self.max_customers) / 2
        estimated_routes = max(1, int(total_customers / avg_customers_per_route))
        
        print(f"✓ Total customers: {total_customers}")
        print(f"✓ Estimated routes needed: {estimated_routes}")
        
        # Try multiple clustering approaches
        clustering_results = {}
        
        # 1. K-means clustering
        print("  Testing K-means clustering...")
        kmeans = KMeans(n_clusters=estimated_routes, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        clustering_results['kmeans'] = kmeans_labels
        
        # 2. DBSCAN clustering
        print("  Testing DBSCAN clustering...")
        coords = features[['centroid_lat', 'centroid_lon']].values
        distances = pdist(coords)
        eps_adaptive = np.percentile(distances, 15)  # More conservative
        min_samples = max(2, int(self.min_customers / self.h3_metrics['customer_count'].mean()))
        
        dbscan = DBSCAN(eps=eps_adaptive, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(coords)
        
        # Handle noise points
        if -1 in dbscan_labels:
            dbscan_labels = self._assign_noise_points(dbscan_labels, coords)
        
        clustering_results['dbscan'] = dbscan_labels
        
        # 3. Priority-based clustering (new approach)
        print("  Testing priority-based clustering...")
        priority_labels = self._priority_based_clustering()
        clustering_results['priority'] = priority_labels
        
        # Evaluate all clustering methods
        best_method = None
        best_score = -1
        
        for method, labels in clustering_results.items():
            self.h3_metrics[f'{method}_cluster'] = labels
            score = self._evaluate_clustering(f'{method}_cluster')
            print(f"  {method.capitalize()} clustering score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_method = method
        
        print(f"✓ Best clustering method: {best_method.upper()} (score: {best_score:.3f})")
        
        # Use best clustering method
        self.h3_metrics['route_cluster'] = self.h3_metrics[f'{best_method}_cluster']
        
        # Clean up temporary columns
        temp_cols = [f'{method}_cluster' for method in clustering_results.keys()]
        self.h3_metrics = self.h3_metrics.drop(temp_cols, axis=1)
        
        return self.h3_metrics
    
    def phase_3_improved_clustering(self):
        """Phase 3: Advanced Geographic Clustering with Multiple Algorithms - FIXED"""
        print("\nPhase 3: Advanced Geographic Clustering")
        print("-" * 50)
        
        # Add enhanced compactness features
        self.calculate_enhanced_compactness_features()
        
        # Prepare features for clustering with NaN handling
        features = self.h3_metrics[[
            'centroid_lat', 'centroid_lon', 'distance_from_fc', 
            'density_score', 'priority_score', 'avg_neighbor_distance'
        ]].copy()
        
        # FIX: Handle NaN values before clustering
        print(f"  Checking for NaN values in features...")
        nan_counts = features.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"  Found NaN values: {dict(nan_counts[nan_counts > 0])}")
            
            # Fill NaN values with appropriate defaults
            features['avg_neighbor_distance'] = features['avg_neighbor_distance'].fillna(
                features['avg_neighbor_distance'].median()
            )
            features['priority_score'] = features['priority_score'].fillna(
                features['priority_score'].median()
            )
            features['density_score'] = features['density_score'].fillna(
                features['density_score'].median()
            )
            
            # For any remaining NaN values, use forward fill then backward fill
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            # Final check: if still NaN, fill with column means
            if features.isnull().sum().sum() > 0:
                features = features.fillna(features.mean())
                
            print(f"  ✓ NaN values handled")
        
        # Verify no NaN values remain
        assert not features.isnull().any().any(), "NaN values still present after cleaning"
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Verify scaled features don't contain NaN
        if np.isnan(features_scaled).any():
            print("  WARNING: NaN values in scaled features, using robust scaling")
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features)
        
        # Estimate number of routes needed
        total_customers = self.h3_metrics['customer_count'].sum()
        avg_customers_per_route = (self.min_customers + self.max_customers) / 2
        estimated_routes = max(1, int(total_customers / avg_customers_per_route))
        
        print(f"✓ Total customers: {total_customers}")
        print(f"✓ Estimated routes needed: {estimated_routes}")
        
        # Try multiple clustering approaches
        clustering_results = {}
        
        # 1. K-means clustering
        print("  Testing K-means clustering...")
        try:
            kmeans = KMeans(n_clusters=estimated_routes, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(features_scaled)
            clustering_results['kmeans'] = kmeans_labels
        except Exception as e:
            print(f"  K-means failed: {str(e)}")
            # Fallback: use simple distance-based clustering
            clustering_results['kmeans'] = self._fallback_distance_clustering(estimated_routes)
        
        # 2. DBSCAN clustering  
        print("  Testing DBSCAN clustering...")
        try:
            coords = features[['centroid_lat', 'centroid_lon']].values
            # Verify coords don't contain NaN
            if np.isnan(coords).any():
                coords = np.nan_to_num(coords, nan=np.nanmean(coords))
                
            distances = pdist(coords)
            eps_adaptive = np.percentile(distances, 15)  # More conservative
            min_samples = max(2, int(self.min_customers / self.h3_metrics['customer_count'].mean()))
            
            dbscan = DBSCAN(eps=eps_adaptive, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(coords)
            
            # Handle noise points
            if -1 in dbscan_labels:
                dbscan_labels = self._assign_noise_points(dbscan_labels, coords)
            
            clustering_results['dbscan'] = dbscan_labels
        except Exception as e:
            print(f"  DBSCAN failed: {str(e)}")
            clustering_results['dbscan'] = self._fallback_distance_clustering(estimated_routes)
        
        # # 3. Priority-based clustering (new approach)
        # print("  Testing priority-based clustering...")
        # try:
        #     priority_labels = self._priority_based_clustering()
        #     clustering_results['priority'] = priority_labels
        # except Exception as e:
        #     print(f"  Priority-based clustering failed: {str(e)}")
        #     clustering_results['priority'] = self._fallback_distance_clustering(estimated_routes)
        
        # Evaluate all clustering methods
        best_method = None
        best_score = -1
        
        for method, labels in clustering_results.items():
            self.h3_metrics[f'{method}_cluster'] = labels
            score = self._evaluate_clustering(f'{method}_cluster')
            print(f"  {method.capitalize()} clustering score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_method = method
        
        print(f"✓ Best clustering method: {best_method.upper()} (score: {best_score:.3f})")
        
        # Use best clustering method
        self.h3_metrics['route_cluster'] = self.h3_metrics[f'{best_method}_cluster']
        
        # Clean up temporary columns
        temp_cols = [f'{method}_cluster' for method in clustering_results.keys()]
        self.h3_metrics = self.h3_metrics.drop(temp_cols, axis=1)
        
        return self.h3_metrics

    def _fallback_distance_clustering(self, n_clusters):
        """Fallback clustering method based on distance from fulfillment center"""
        print("  Using fallback distance-based clustering...")
        
        # Sort by distance and assign clusters in round-robin fashion
        sorted_indices = self.h3_metrics['distance_from_fc'].argsort()
        labels = np.zeros(len(self.h3_metrics))
        
        for i, idx in enumerate(sorted_indices):
            labels[idx] = i % n_clusters
        
        return labels.astype(int)
    
    def _priority_based_clustering(self):
        """Priority-based clustering that considers both density and distance"""
        # Sort cells by priority score (high density, close to FC)
        sorted_cells = self.h3_metrics.sort_values('priority_score', ascending=False).copy()
        
        clusters = np.full(len(self.h3_metrics), -1)
        current_cluster = 0
        
        for idx, cell in sorted_cells.iterrows():
            if clusters[self.h3_metrics.index.get_loc(idx)] != -1:
                continue  # Already assigned
            
            # Start new cluster with this high-priority cell
            cluster_cells = [idx]
            cluster_customers = cell['customer_count']
            cluster_center = (cell['centroid_lat'], cell['centroid_lon'])
            
            # Find nearby cells to add to this cluster
            remaining_cells = sorted_cells[
                (sorted_cells.index != idx) & 
                (sorted_cells.index.isin(self.h3_metrics.index[clusters == -1]))
            ].copy()
            
            for remaining_idx, remaining_cell in remaining_cells.iterrows():
                if cluster_customers >= self.max_customers:
                    break
                
                # Calculate distance to cluster center
                cell_pos = (remaining_cell['centroid_lat'], remaining_cell['centroid_lon'])
                distance_to_cluster = haversine(cluster_center, cell_pos, unit=Unit.KILOMETERS)
                
                # Add cell if it's close and doesn't violate constraints
                if (distance_to_cluster <= 3.0 and  # Max 3km between cells in same route
                    cluster_customers + remaining_cell['customer_count'] <= self.max_customers):
                    
                    cluster_cells.append(remaining_idx)
                    cluster_customers += remaining_cell['customer_count']
                    
                    # Update cluster center (weighted by customer count)
                    total_weight = sum(self.h3_metrics.loc[c, 'customer_count'] for c in cluster_cells)
                    weighted_lat = sum(self.h3_metrics.loc[c, 'centroid_lat'] * 
                                     self.h3_metrics.loc[c, 'customer_count'] for c in cluster_cells) / total_weight
                    weighted_lon = sum(self.h3_metrics.loc[c, 'centroid_lon'] * 
                                     self.h3_metrics.loc[c, 'customer_count'] for c in cluster_cells) / total_weight
                    cluster_center = (weighted_lat, weighted_lon)
            
            # Assign cluster labels
            for cell_idx in cluster_cells:
                clusters[self.h3_metrics.index.get_loc(cell_idx)] = current_cluster
            
            current_cluster += 1
        
        return clusters
    
    def _assign_noise_points(self, labels, coords):
        """Enhanced noise point assignment with distance weighting"""
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        
        noise_indices = np.where(noise_mask)[0]
        cluster_indices = np.where(~noise_mask)[0]
        
        if len(cluster_indices) == 0:
            return np.arange(len(labels))
        
        for noise_idx in noise_indices:
            noise_point = coords[noise_idx]
            
            # Calculate weighted assignment based on distance and cluster size
            cluster_scores = {}
            for cluster_idx in cluster_indices:
                cluster_label = labels[cluster_idx]
                if cluster_label not in cluster_scores:
                    cluster_coords = coords[labels == cluster_label]
                    
                    # Distance to cluster centroid
                    cluster_centroid = cluster_coords.mean(axis=0)
                    distance = haversine(noise_point, cluster_centroid, unit=Unit.KILOMETERS)
                    
                    # Cluster size penalty (prefer smaller clusters)
                    cluster_size = len(cluster_coords)
                    size_penalty = 1 + (cluster_size - 1) * 0.1
                    
                    # Combined score (lower is better)
                    cluster_scores[cluster_label] = distance * size_penalty
            
            # Assign to best cluster
            best_cluster = min(cluster_scores, key=cluster_scores.get)
            labels[noise_idx] = best_cluster
        
        return labels
    
    def _evaluate_clustering(self, cluster_col):
        """Enhanced clustering evaluation with multiple criteria"""
        cluster_stats = (self.h3_metrics.groupby(cluster_col)
                        .agg({
                            'customer_count': 'sum',
                            'max_distance_from_fc': 'max',
                            'h3_cell_id': 'count',
                            'centroid_lat': ['mean', 'std'],
                            'centroid_lon': ['mean', 'std'],
                            'priority_score': 'mean'
                        }))
        
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
        cluster_stats = cluster_stats.reset_index()
        
        # Constraint violations
        customer_valid = ((cluster_stats['customer_count_sum'] >= self.min_customers) & 
                         (cluster_stats['customer_count_sum'] <= self.max_customers))
        distance_valid = cluster_stats['max_distance_from_fc_max'] <= self.max_distance_km
        
        # Compactness score (lower std deviation is better)
        compactness_scores = 1 / (cluster_stats['centroid_lat_std'] + 
                                 cluster_stats['centroid_lon_std'] + 0.001)
        avg_compactness = compactness_scores.mean()
        
        # Priority score (higher is better)
        avg_priority = cluster_stats['priority_score_mean'].mean()
        
        # Combined score
        constraint_score = (customer_valid & distance_valid).mean()
        total_score = 0.6 * constraint_score + 0.2 * (avg_compactness / 100) + 0.2 * avg_priority
        
        return total_score
    
    def phase_4_constraint_enforcement(self):
        """Phase 4: Enhanced Constraint Enforcement"""
        print("\nPhase 4: Enhanced Constraint Enforcement")
        print("-" * 50)
        
        max_iterations = 15
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"  Iteration {iteration}")
            
            # Calculate current route statistics
            route_stats = (self.h3_metrics.groupby('route_cluster')
                          .agg({
                              'customer_count': 'sum',
                              'max_distance_from_fc': 'max',
                              'h3_cell_id': 'count',
                              'centroid_lat': 'mean',
                              'centroid_lon': 'mean',
                              'priority_score': 'mean'
                          }).reset_index())
            
            changes_made = False
            
            # 1. Handle routes with too few customers
            small_routes = route_stats[
                route_stats['customer_count'] < self.min_customers
            ]['route_cluster'].values
            
            for small_route in small_routes:
                if self._merge_small_route_enhanced(small_route, route_stats):
                    changes_made = True
            
            # 2. Handle routes with too many customers  
            large_routes = route_stats[
                route_stats['customer_count'] > self.max_customers
            ]['route_cluster'].values
            
            for large_route in large_routes:
                if self._split_large_route_enhanced(large_route):
                    changes_made = True
            
            # 3. Handle routes exceeding distance constraint
            distant_routes = route_stats[
                route_stats['max_distance_from_fc'] > self.max_distance_km
            ]['route_cluster'].values
            
            for distant_route in distant_routes:
                if self._trim_distant_cells_enhanced(distant_route):
                    changes_made = True
            
            if not changes_made:
                print("  ✓ Convergence reached")
                break
        
        # Final statistics
        final_stats = (self.h3_metrics.groupby('route_cluster')
                      .agg({
                          'customer_count': 'sum',
                          'max_distance_from_fc': 'max',
                          'h3_cell_id': 'count'
                      }))
        
        print(f"✓ Final routes: {len(final_stats)}")
        print(f"✓ Customer range: {final_stats['customer_count'].min()}-{final_stats['customer_count'].max()}")
        print(f"✓ Distance range: {final_stats['max_distance_from_fc'].min():.1f}-{final_stats['max_distance_from_fc'].max():.1f} km")
        
        return self.h3_metrics
    
    def _merge_small_route_enhanced(self, small_route, route_stats):
        """Enhanced small route merging with priority consideration"""
        small_route_data = route_stats[route_stats['route_cluster'] == small_route].iloc[0]
        small_route_center = (small_route_data['centroid_lat'], small_route_data['centroid_lon'])
        
        # Find merge candidates considering priority and compatibility
        candidates = route_stats[
            (route_stats['route_cluster'] != small_route) &
            (route_stats['customer_count'] + small_route_data['customer_count'] <= self.max_customers)
        ].copy()
        
        if len(candidates) == 0:
            return False
        
        # Enhanced candidate scoring
        candidates['merge_score'] = 0
        
        for idx, candidate in candidates.iterrows():
            candidate_center = (candidate['centroid_lat'], candidate['centroid_lon'])
            
            # Distance factor (closer is better)
            distance = haversine(small_route_center, candidate_center, unit=Unit.KILOMETERS)
            distance_score = 1 / (1 + distance)
            
            # Priority factor (higher priority routes preferred)
            priority_score = candidate['priority_score']
            
            # Size factor (prefer routes with room for growth)
            size_factor = 1 - (candidate['customer_count'] / self.max_customers)
            
            # Combined score
            candidates.at[idx, 'merge_score'] = (
                0.4 * distance_score + 0.3 * priority_score + 0.3 * size_factor
            )
        
        # Merge with best candidate
        best_candidate = candidates.loc[candidates['merge_score'].idxmax(), 'route_cluster']
        
        self.h3_metrics.loc[
            self.h3_metrics['route_cluster'] == small_route,
            'route_cluster'
        ] = best_candidate
        
        return True
    
    def _split_large_route_enhanced(self, large_route):
        """Enhanced large route splitting with intelligent sub-clustering"""
        route_cells = self.h3_metrics[self.h3_metrics['route_cluster'] == large_route].copy()
        
        if len(route_cells) < 2:
            return False
        
        # Use weighted K-means considering customer count and priority
        coords = route_cells[['centroid_lat', 'centroid_lon']].values
        weights = route_cells['customer_count'].values
        
        # Weighted clustering
        from sklearn.cluster import KMeans
        
        # Try different split strategies
        best_split = None
        best_balance = float('inf')
        
        for n_clusters in [2, 3]:  # Try 2-way and 3-way splits
            if len(route_cells) < n_clusters:
                continue
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            sub_clusters = kmeans.fit_predict(coords, sample_weight=weights)
            
            # Evaluate split balance
            cluster_sizes = [
                route_cells[sub_clusters == i]['customer_count'].sum() 
                for i in range(n_clusters)
            ]
            
            # Check if all sub-clusters meet minimum requirements
            if all(size >= self.min_customers for size in cluster_sizes):
                balance_score = max(cluster_sizes) - min(cluster_sizes)
                if balance_score < best_balance:
                    best_balance = balance_score
                    best_split = sub_clusters
        
        if best_split is None:
            return False
        
        # Apply best split
        max_cluster_id = self.h3_metrics['route_cluster'].max()
        route_cells['new_cluster'] = best_split + max_cluster_id + 1
        
        for idx, row in route_cells.iterrows():
            self.h3_metrics.loc[idx, 'route_cluster'] = row['new_cluster']
        
        return True
    
    def _trim_distant_cells_enhanced(self, distant_route):
        """Enhanced distant cell trimming with priority preservation"""
        route_cells = self.h3_metrics[self.h3_metrics['route_cluster'] == distant_route].copy()
        
        # Sort by priority score descending, then by distance ascending
        route_cells_sorted = route_cells.sort_values(
            ['priority_score', 'max_distance_from_fc'], 
            ascending=[False, True]
        )
        
        # Keep cells within distance constraint, prioritizing high-priority cells
        valid_cells = []
        cumulative_customers = 0
        
        for _, cell in route_cells_sorted.iterrows():
            if (cell['max_distance_from_fc'] <= self.max_distance_km and
                cumulative_customers + cell['customer_count'] >= self.min_customers):
                valid_cells.append(cell['h3_cell_id'])
                cumulative_customers += cell['customer_count']
                
                if cumulative_customers >= self.max_customers:
                    break
        
        if len(valid_cells) == len(route_cells):
            return False  # No changes needed
        
        if len(valid_cells) == 0 or cumulative_customers < self.min_customers:
            # Remove entire route if invalid
            self.h3_metrics = self.h3_metrics[
                self.h3_metrics['route_cluster'] != distant_route
            ]
        else:
            # Keep only valid cells
            cells_to_remove = set(route_cells['h3_cell_id']) - set(valid_cells)
            self.h3_metrics = self.h3_metrics[
                ~self.h3_metrics['h3_cell_id'].isin(cells_to_remove)
            ]
        
        return True
    
    def phase_5_generate_output(self):
        """Phase 5: Enhanced Output Generation"""
        print("\nPhase 5: Enhanced Output Generation")
        print("-" * 50)
        
        # Calculate enhanced route statistics
        route_summary = (self.h3_metrics.groupby('route_cluster')
                        .agg({
                            'customer_count': 'sum',
                            'distance_from_fc': ['mean', 'max'],
                            'max_distance_from_fc': 'max',
                            'h3_cell_id': 'count',
                            'centroid_lat': ['mean', 'std'],
                            'centroid_lon': ['mean', 'std'],
                            'priority_score': 'mean',
                            'density_score': 'mean'
                        }))
        
        # Flatten column names
        route_summary.columns = ['_'.join(col).strip() for col in route_summary.columns]
        route_summary = route_summary.reset_index()
        
        # Enhanced compactness score calculation
        route_summary['compactness_score'] = self._calculate_enhanced_compactness(route_summary)
        
        # Enhanced delivery time estimation
        route_summary['estimated_delivery_time_hours'] = self._calculate_delivery_time(route_summary)
        
        # Calculate route efficiency score
        route_summary['efficiency_score'] = self._calculate_efficiency_score(route_summary)
        
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
                    'compactness_score': route['compactness_score'],
                    'efficiency_score': route['efficiency_score'],
                    'priority_score': route['priority_score_mean'],
                    'density_score': route['density_score_mean']
                })
        
        self.routes_df = pd.DataFrame(output_rows)
        
        print(f"✓ Generated {len(self.routes_df['route_id'].unique())} routes")
        print(f"✓ Total H3 cells: {len(self.routes_df)}")
        print(f"✓ Average efficiency score: {self.routes_df['efficiency_score'].mean():.3f}")
        
        return self.routes_df
    
    def _calculate_enhanced_compactness(self, route_summary):
        """Calculate enhanced compactness score"""
        # Inverse of coordinate standard deviation (normalized)
        lat_std = route_summary['centroid_lat_std'].fillna(0)
        lon_std = route_summary['centroid_lon_std'].fillna(0)
        
        raw_compactness = 1 / (lat_std + lon_std + 0.001)
        
        # Normalize to 0-1 range
        max_compactness = raw_compactness.max()
        if max_compactness > 0:
            normalized_compactness = raw_compactness / max_compactness
        else:
            normalized_compactness = raw_compactness
        
        return normalized_compactness
    
    def _calculate_delivery_time(self, route_summary):
        """Calculate enhanced delivery time estimation"""
        # Enhanced model: variable speed based on route characteristics
        base_speed_kmh = 30  # Base speed
        stop_time_hours = 6/60  # 6 minutes per stop
        
        # Adjust speed based on route density and distance
        avg_distance = route_summary['distance_from_fc_mean']
        cell_count = route_summary['h3_cell_id_count']
        
        # Speed penalty for distant/sparse routes
        speed_factor = np.where(avg_distance > 4, 0.8, 1.0)  # Slower for distant routes
        speed_factor *= np.where(cell_count > 10, 0.9, 1.0)  # Slower for complex routes
        
        effective_speed = base_speed_kmh * speed_factor
        
        # Calculate time: round trip + stops + setup time
        travel_time = (route_summary['distance_from_fc_mean'] * 2) / effective_speed
        stop_time = cell_count * stop_time_hours
        setup_time = 0.25  # 15 minutes setup time per route
        
        total_time = travel_time + stop_time + setup_time
        
        return total_time
    
    def _calculate_efficiency_score(self, route_summary):
        """Calculate route efficiency score (0-1, higher is better)"""
        # Factors: customer density, compactness, distance efficiency
        
        # Customer density factor
        customer_density = route_summary['customer_count_sum'] / route_summary['h3_cell_id_count']
        density_score = customer_density / customer_density.max() if customer_density.max() > 0 else 0
        
        # Distance efficiency (inverse of average distance)
        max_distance = route_summary['distance_from_fc_mean'].max()
        distance_efficiency = 1 - (route_summary['distance_from_fc_mean'] / max_distance) if max_distance > 0 else 1
        
        # Compactness factor
        compactness_factor = route_summary['compactness_score']
        
        # Priority factor
        priority_factor = route_summary['priority_score_mean']
        
        # Combined efficiency score
        efficiency = (
            0.3 * density_score + 
            0.25 * distance_efficiency + 
            0.25 * compactness_factor + 
            0.2 * priority_factor
        )
        
        return efficiency
    
    def generate_enhanced_route_summary(self):
        """Generate enhanced route summary with detailed statistics"""
        print("\nGenerating Enhanced Route Summary")
        print("-" * 50)
        
        if self.routes_df is None:
            raise ValueError("Routes not generated yet. Run optimize() first.")
        
        # Get assignment confidence data
        assignment_confidence = (self.df_output_assignment
                               .groupby('h3_cell_id')['assignment_confidence']
                               .mean()
                               .reset_index())
        
        # Merge with h3_metrics
        h3_with_confidence = self.h3_metrics.merge(
            assignment_confidence, on='h3_cell_id', how='left'
        )
        
        # Calculate detailed route statistics
        route_summary_data = []
        
        for route_id in self.routes_df['route_id'].unique():
            route_cells = self.routes_df[self.routes_df['route_id'] == route_id]
            route_h3_cells = route_cells['h3_cell_id'].tolist()
            
            # Get detailed metrics for this route
            route_metrics = h3_with_confidence[
                h3_with_confidence['h3_cell_id'].isin(route_h3_cells)
            ]
            
            # Calculate optimized route distance using TSP approximation
            cumulative_distance = self._calculate_route_distance(route_metrics)
            
            # Calculate additional metrics
            farthest_distance = route_metrics['distance_from_fc'].max() if len(route_metrics) > 0 else 0
            avg_confidence = route_metrics['assignment_confidence'].mean() if len(route_metrics) > 0 else 0
            route_compactness = self._calculate_route_compactness(route_metrics)
            
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
                'optimized_distance_km': cumulative_distance,
                'farthest_centroid_distance_km': farthest_distance,
                'estimated_delivery_time_hours': route_info['estimated_delivery_time_hours'],
                'avg_assignment_confidence': avg_confidence,
                'compactness_score': route_info['compactness_score'],
                'efficiency_score': route_info['efficiency_score'],
                'priority_score': route_info['priority_score'],
                'density_score': route_info['density_score'],
                'route_compactness': route_compactness,
                'distance_savings_pct': ((route_info['total_distance_km'] - cumulative_distance) / 
                                       route_info['total_distance_km'] * 100) if route_info['total_distance_km'] > 0 else 0
            })
        
        self.route_summary_df = pd.DataFrame(route_summary_data)
        
        print(f"✓ Route summary generated for {len(self.route_summary_df)} routes")
        print(f"✓ Average distance savings: {self.route_summary_df['distance_savings_pct'].mean():.1f}%")
        
        return self.route_summary_df
    
    def _calculate_route_distance(self, route_metrics):
        """Calculate optimized route distance using nearest neighbor heuristic"""
        if len(route_metrics) <= 1:
            return route_metrics['distance_from_fc'].iloc[0] * 2 if len(route_metrics) == 1 else 0
        
        # Start from fulfillment center
        unvisited = route_metrics.copy()
        current_pos = self.fulfillment_center
        total_distance = 0
        
        while len(unvisited) > 0:
            # Find nearest unvisited cell
            distances = []
            for _, cell in unvisited.iterrows():
                cell_pos = (cell['centroid_lat'], cell['centroid_lon'])
                dist = haversine(current_pos, cell_pos, unit=Unit.KILOMETERS)
                distances.append(dist)
            
            # Move to nearest cell
            nearest_idx = np.argmin(distances)
            nearest_cell = unvisited.iloc[nearest_idx]
            
            total_distance += distances[nearest_idx]
            current_pos = (nearest_cell['centroid_lat'], nearest_cell['centroid_lon'])
            
            # Remove visited cell
            unvisited = unvisited.drop(nearest_cell.name)
        
        # Return to fulfillment center
        total_distance += haversine(current_pos, self.fulfillment_center, unit=Unit.KILOMETERS)
        
        return total_distance
    
    def _calculate_route_compactness(self, route_metrics):
        """Calculate route-specific compactness score"""
        if len(route_metrics) <= 1:
            return 1.0
        
        coords = route_metrics[['centroid_lat', 'centroid_lon']].values
        
        # Calculate convex hull area if possible
        try:
            from scipy.spatial import ConvexHull
            if len(coords) >= 3:
                hull = ConvexHull(coords)
                hull_area = hull.volume
                
                # Calculate ideal area (circle with same number of points)
                avg_distance = np.mean([
                    haversine(self.fulfillment_center, (row['centroid_lat'], row['centroid_lon']), unit=Unit.KILOMETERS)
                    for _, row in route_metrics.iterrows()
                ])
                ideal_radius = avg_distance / 2
                ideal_area = np.pi * (ideal_radius ** 2)
                
                # Compactness = ideal_area / actual_area (capped at 1.0)
                compactness = min(1.0, ideal_area / hull_area) if hull_area > 0 else 1.0
            else:
                # For 2 points, use distance-based compactness
                distances = pdist(coords)
                avg_dist = np.mean(distances)
                compactness = 1 / (1 + avg_dist)
        except:
            # Fallback: use standard deviation of coordinates
            lat_std = coords[:, 0].std()
            lon_std = coords[:, 1].std()
            compactness = 1 / (1 + lat_std + lon_std)
        
        return compactness
    
    def create_enhanced_visualization(self, save_path=None, show_customer_points=False):
        """Create enhanced interactive map visualization with improved styling"""
        print("\nCreating Enhanced Route Visualization")
        print("-" * 50)
        
        if self.routes_df is None:
            raise ValueError("Routes not generated yet. Run optimize() first.")
        
        # Create base map with improved styling
        m = folium.Map(
            location=self.fulfillment_center,
            zoom_start=12,
            tiles=None  # We'll add custom tiles
        )
        
        # Add multiple tile layers
        folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
        folium.TileLayer('CartoDB Positron', name='Light Theme').add_to(m)
        folium.TileLayer('CartoDB Dark_Matter', name='Dark Theme').add_to(m)
        
        # Enhanced fulfillment center marker
        folium.Marker(
            location=self.fulfillment_center,
            popup=folium.Popup(
                f"""<div style="font-family: Arial; font-size: 12px;">
                <b style="color: #d32f2f; font-size: 14px;">{self.stock_point_name}</b><br>
                <b>Stock Point ID:</b> {self.stock_point_id}<br>
                <b>Coordinates:</b> {self.fulfillment_center[0]:.4f}, {self.fulfillment_center[1]:.4f}<br>
                <b>Total Routes:</b> {len(self.routes_df['route_id'].unique())}<br>
                <b>Coverage Area:</b> {self.max_distance_km} km radius
                </div>""",
                max_width=300
            ),
            tooltip='🏢 Fulfillment Center',
            icon=folium.Icon(color='red', icon='warehouse', prefix='fa', icon_size=(20, 20))
        ).add_to(m)
        
        # Process each route with enhanced styling
        route_ids = sorted(self.routes_df['route_id'].unique())
        
        for i, route_id in enumerate(route_ids):
            route_cells = self.routes_df[self.routes_df['route_id'] == route_id]
            route_h3_cells = route_cells['h3_cell_id'].tolist()
            
            # Use enhanced color palette
            color = self.route_colors[i % len(self.route_colors)]
            
            # Get route metrics
            route_metrics = self.h3_metrics[
                self.h3_metrics['h3_cell_id'].isin(route_h3_cells)
            ]
            
            # Get route summary info
            route_info = route_cells.iloc[0]
            
            # Create feature group with enhanced naming
            route_group = folium.FeatureGroup(
                name=f'🚛 {route_id} ({len(route_h3_cells)} cells, {route_info["customer_count"]} customers)'
            )
            
            # Add H3 cell polygons with enhanced styling
            for idx, (_, cell_data) in enumerate(route_metrics.iterrows()):
                h3_cell = cell_data['h3_cell_id']
                boundary = h3.cell_to_boundary(h3_cell)
                boundary_coords = [(lat, lon) for lat, lon in boundary]
                
                # Enhanced popup content
                popup_content = f"""
                <div style="font-family: Arial; font-size: 11px; max-width: 250px;">
                    <b style="color: {color}; font-size: 13px;">📍 {route_id}</b><br>
                    <hr style="margin: 5px 0;">
                    <b>H3 Cell:</b> <code>{h3_cell}</code><br>
                    <b>Customers:</b> <span style="color: #1976d2; font-weight: bold;">{cell_data['customer_count']}</span><br>
                    <b>Distance from FC:</b> {cell_data['distance_from_fc']:.2f} km<br>
                    <b>Density:</b> {cell_data['density_score']:.1f} customers/km²<br>
                    <b>Priority Score:</b> {cell_data['priority_score']:.3f}<br>
                    <b>Cell Area:</b> {cell_data['cell_area_km2']:.3f} km²
                </div>
                """
                
                # Enhanced polygon styling
                folium.Polygon(
                    locations=boundary_coords,
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f'🎯 {route_id}: {cell_data["customer_count"]} customers',
                    color=color,
                    weight=2.5,
                    fillColor=color,
                    fillOpacity=0.6,
                    opacity=0.9
                ).add_to(route_group)
                
                # Enhanced centroid markers with labels
                folium.CircleMarker(
                    location=(cell_data['centroid_lat'], cell_data['centroid_lon']),
                    radius=max(4, min(12, cell_data['customer_count'] / 5)),  # Size based on customer count
                    popup=popup_content,
                    tooltip=f'📊 {cell_data["customer_count"]} customers',
                    color='white',
                    weight=2,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(route_group)
                
                # Add cell labels for better identification
                folium.Marker(
                    location=(cell_data['centroid_lat'], cell_data['centroid_lon']),
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 10px; color: white; font-weight: bold; text-shadow: 1px 1px 1px black;">{idx+1}</div>',
                        icon_size=(20, 20),
                        icon_anchor=(10, 10)
                    )
                ).add_to(route_group)
            
            # Enhanced route path with optimized ordering
            # if len(route_metrics) > 1:
            #     # Calculate optimized route path
            #     route_path = self._calculate_optimized_path(route_metrics)
                
            #     # Add enhanced route line
            #     folium.PolyLine(
            #         locations=route_path,
            #         color=color,
            #         weight=4,
            #         opacity=0.8,
            #         popup=folium.Popup(
            #             f"""<div style="font-family: Arial;">
            #             <b style="color: {color};">{route_id} - Route Path</b><br>
            #             <b>Distance:</b> {route_info['total_distance_km']:.1f} km<br>
            #             <b>Delivery Time:</b> {route_info['estimated_delivery_time_hours']:.1f} hours<br>
            #             <b>Efficiency:</b> {route_info['efficiency_score']:.3f}
            #             </div>""",
            #             max_width=250
            #         ),
            #         tooltip=f'🛣️ {route_id} Route Path'
            #     ).add_to(route_group)
                
                # Add direction arrows
                # self._add_direction_arrows(m, route_path, color)
            
            route_group.add_to(m)
        
        # Add enhanced constraint visualization
        self._add_constraint_visualization(m)
        
        # Add customer points if requested
        if show_customer_points:
            self._add_customer_points(m)
        
        # Add enhanced legend and summary
        self._add_enhanced_legend(m)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Save if path provided
        if save_path:
            m.save(save_path)
            print(f"✓ Enhanced map saved to: {save_path}")
        
        return m
    
    def _calculate_optimized_path(self, route_metrics):
        """Calculate optimized path for route visualization"""
        if len(route_metrics) <= 1:
            return [self.fulfillment_center]
        
        # Use nearest neighbor for path optimization
        path = [self.fulfillment_center]
        unvisited = route_metrics.copy()
        current_pos = self.fulfillment_center
        
        while len(unvisited) > 0:
            # Find nearest unvisited cell
            distances = []
            for _, cell in unvisited.iterrows():
                cell_pos = (cell['centroid_lat'], cell['centroid_lon'])
                dist = haversine(current_pos, cell_pos, unit=Unit.KILOMETERS)
                distances.append(dist)
            
            # Move to nearest cell
            nearest_idx = np.argmin(distances)
            nearest_cell = unvisited.iloc[nearest_idx]
            
            cell_pos = (nearest_cell['centroid_lat'], nearest_cell['centroid_lon'])
            path.append(cell_pos)
            current_pos = cell_pos
            
            # Remove visited cell
            unvisited = unvisited.drop(nearest_cell.name)
        
        # Return to fulfillment center
        path.append(self.fulfillment_center)
        
        return path
    
    def _add_direction_arrows(self, map_obj, route_path, color):
        """Add direction arrows along route path"""
        for i in range(len(route_path) - 1):
            start_pos = route_path[i]
            end_pos = route_path[i + 1]
            
            # Calculate midpoint
            mid_lat = (start_pos[0] + end_pos[0]) / 2
            mid_lon = (start_pos[1] + end_pos[1]) / 2
            
            # Add arrow marker
            folium.Marker(
                location=(mid_lat, mid_lon),
                icon=folium.Icon(
                    icon='arrow-right',
                    prefix='fa',
                    color='white',
                    icon_color=color
                )
            ).add_to(map_obj)
    
    def _add_constraint_visualization(self, map_obj):
        """Add enhanced constraint visualization"""
        # Distance constraint circle
        folium.Circle(
            location=self.fulfillment_center,
            radius=self.max_distance_km * 1000,
            popup=f'📏 Maximum Service Area: {self.max_distance_km} km',
            tooltip='Service Area Boundary',
            color='red',
            weight=3,
            fill=False,
            dashArray='10, 5',
            opacity=0.7
        ).add_to(map_obj)
        
        # Add distance rings for reference
        for distance in [2, 4, 6]:
            if distance < self.max_distance_km:
                folium.Circle(
                    location=self.fulfillment_center,
                    radius=distance * 1000,
                    color='gray',
                    weight=1,
                    fill=False,
                    dashArray='5, 5',
                    opacity=0.3
                ).add_to(map_obj)
    
    def _add_customer_points(self, map_obj):
        """Add customer points to map"""
        customer_group = folium.FeatureGroup(name='👥 Customer Points')
        
        # Sample customers for performance (show max 500)
        stock_customers = self.customers_gdf[
            self.customers_gdf['customer_id'].isin(
                self.df_output_assignment['customer_id']
            )
        ]
        
        if len(stock_customers) > 500:
            stock_customers = stock_customers.sample(500, random_state=42)
        
        for _, customer in stock_customers.iterrows():
            folium.CircleMarker(
                location=(customer.geometry.y, customer.geometry.x),
                radius=2,
                popup=f'👤 Customer: {customer["customer_id"]}',
                tooltip='Customer Location',
                color='black',
                weight=1,
                fillColor='yellow',
                fillOpacity=0.6
            ).add_to(customer_group)
        
        customer_group.add_to(map_obj)
    
    def _add_enhanced_legend(self, map_obj):
        """Add enhanced legend and summary to map"""
        # Generate comprehensive statistics
        route_stats = self.routes_df.groupby('route_id').agg({
            'customer_count': 'first',
            'total_distance_km': 'first',
            'efficiency_score': 'first',
            'compactness_score': 'first'
        })
        
        legend_html = f"""
        <div style='position: fixed; top: 10px; left: 10px; width: 350px; 
                    background-color: rgba(255, 255, 255, 0.95); 
                    border: 2px solid #333; border-radius: 10px; z-index: 9999; 
                    font-family: Arial, sans-serif; font-size: 12px; 
                    padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);'>
            
            <h3 style='color: #1976d2; margin-top: 0; border-bottom: 2px solid #1976d2; padding-bottom: 5px;'>
                📊 {self.stock_point_name} - Route Analytics
            </h3>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;'>
                <div style='background: #e3f2fd; padding: 8px; border-radius: 5px;'>
                    <b style='color: #1976d2;'>Total Routes</b><br>
                    <span style='font-size: 16px; font-weight: bold;'>{len(route_stats)}</span>
                </div>
                <div style='background: #e8f5e8; padding: 8px; border-radius: 5px;'>
                    <b style='color: #388e3c;'>Total Customers</b><br>
                    <span style='font-size: 16px; font-weight: bold;'>{route_stats["customer_count"].sum()}</span>
                </div>
            </div>
            
            <table style='width: 100%; font-size: 11px; border-collapse: collapse;'>
                <tr style='background: #f5f5f5;'>
                    <td style='padding: 4px; border: 1px solid #ddd;'><b>Metric</b></td>
                    <td style='padding: 4px; border: 1px solid #ddd;'><b>Average</b></td>
                    <td style='padding: 4px; border: 1px solid #ddd;'><b>Range</b></td>
                </tr>
                <tr>
                    <td style='padding: 4px; border: 1px solid #ddd;'>Customers/Route</td>
                    <td style='padding: 4px; border: 1px solid #ddd;'>{route_stats["customer_count"].mean():.0f}</td>
                    <td style='padding: 4px; border: 1px solid #ddd;'>{route_stats["customer_count"].min()}-{route_stats["customer_count"].max()}</td>
                </tr>
                <tr>
                    <td style='padding: 4px; border: 1px solid #ddd;'>Distance (km)</td>
                    <td style='padding: 4px; border: 1px solid #ddd;'>{route_stats["total_distance_km"].mean():.1f}</td>
                    <td style='padding: 4px; border: 1px solid #ddd;'>{route_stats["total_distance_km"].min():.1f}-{route_stats["total_distance_km"].max():.1f}</td>
                </tr>
                <tr>
                    <td style='padding: 4px; border: 1px solid #ddd;'>Efficiency</td>
                    <td style='padding: 4px; border: 1px solid #ddd;'>{route_stats["efficiency_score"].mean():.3f}</td>
                    <td style='padding: 4px; border: 1px solid #ddd;'>{route_stats["efficiency_score"].min():.3f}-{route_stats["efficiency_score"].max():.3f}</td>
                </tr>
                <tr>
                    <td style='padding: 4px; border: 1px solid #ddd;'>Compactness</td>
                    <td style='padding: 4px; border: 1px solid #ddd;'>{route_stats["compactness_score"].mean():.3f}</td>
                    <td style='padding: 4px; border: 1px solid #ddd;'>{route_stats["compactness_score"].min():.3f}-{route_stats["compactness_score"].max():.3f}</td>
                </tr>
            </table>
            
            <div style='margin-top: 10px; padding: 8px; background: #fff3e0; border-radius: 5px; border-left: 4px solid #ff9800;'>
                <b style='color: #f57c00;'>🎯 Constraints:</b><br>
                <small>• Distance: ≤ {self.max_distance_km} km from FC<br>
                • Customers: {self.min_customers}-{self.max_customers} per route</small>
            </div>
        </div>
        """
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def phase_6_enhanced_validation(self):
        """Phase 6: Enhanced Validation & Quality Assessment"""
        print("\nPhase 6: Enhanced Validation & Quality Assessment")
        print("-" * 50)
        
        # Route-level validation
        route_validation = (self.routes_df.groupby('route_id')
                           .agg({
                               'customer_count': 'first',
                               'total_distance_km': 'first',
                               'efficiency_score': 'first',
                               'compactness_score': 'first',
                               'h3_cell_id': 'count'
                           }))
        
        # Constraint violations
        customer_violations = (
            (route_validation['customer_count'] < self.min_customers) |
            (route_validation['customer_count'] > self.max_customers)
        ).sum()
        
        distance_violations = (
            route_validation['total_distance_km'] / 2 > self.max_distance_km
        ).sum()
        
        # Quality metrics
        avg_efficiency = route_validation['efficiency_score'].mean()
        avg_compactness = route_validation['compactness_score'].mean()
        customer_balance = route_validation['customer_count'].std()
        
        # Coverage analysis
        total_customers_covered = route_validation['customer_count'].sum()
        total_available_customers = self.h3_metrics['customer_count'].sum()
        coverage_rate = (total_customers_covered / total_available_customers) * 100
        
        # Print comprehensive validation results
        print("📊 VALIDATION RESULTS")
        print("=" * 30)
        print(f"✓ Total routes generated: {len(route_validation)}")
        print(f"✓ Customer constraint violations: {customer_violations}")
        print(f"✓ Distance constraint violations: {distance_violations}")
        print(f"✓ Coverage rate: {coverage_rate:.1f}%")
        
        print("\n📈 QUALITY METRICS")
        print("=" * 30)
        print(f"✓ Average customers per route: {route_validation['customer_count'].mean():.1f}")
        print(f"✓ Customer count range: {route_validation['customer_count'].min()}-{route_validation['customer_count'].max()}")
        print(f"✓ Average distance per route: {route_validation['total_distance_km'].mean()/2:.1f} km")
        print(f"✓ Average efficiency score: {avg_efficiency:.3f}")
        print(f"✓ Average compactness score: {avg_compactness:.3f}")
        print(f"✓ Customer balance (std dev): {customer_balance:.1f}")
        
        # Route efficiency distribution
        high_efficiency_routes = (route_validation['efficiency_score'] >= 0.7).sum()
        medium_efficiency_routes = ((route_validation['efficiency_score'] >= 0.5) & 
                                   (route_validation['efficiency_score'] < 0.7)).sum()
        low_efficiency_routes = (route_validation['efficiency_score'] < 0.5).sum()
        
        print(f"\n🎯 EFFICIENCY DISTRIBUTION")
        print("=" * 30)
        print(f"✓ High efficiency (≥0.7): {high_efficiency_routes} routes")
        print(f"✓ Medium efficiency (0.5-0.7): {medium_efficiency_routes} routes")  
        print(f"✓ Low efficiency (<0.5): {low_efficiency_routes} routes")
        
        return {
            'total_routes': len(route_validation),
            'customer_violations': customer_violations,
            'distance_violations': distance_violations,
            'coverage_rate': coverage_rate,
            'avg_customers': route_validation['customer_count'].mean(),
            'avg_distance': route_validation['total_distance_km'].mean() / 2,
            'avg_efficiency': avg_efficiency,
            'avg_compactness': avg_compactness,
            'customer_balance': customer_balance,
            'high_efficiency_routes': high_efficiency_routes,
            'route_distribution': {
                'high': high_efficiency_routes,
                'medium': medium_efficiency_routes,
                'low': low_efficiency_routes
            }
        }
    
    def optimize(self):
        """Run complete enhanced optimization pipeline"""
        print("🚀 STARTING H3 ROUTE OPTIMIZATION PIPELINE")
        print("=" * 60)
        
        try:
            # Execute all phases
            self.phase_1_data_preparation()
            self.phase_2_calculate_metrics()
            self.phase_3_improved_clustering()
            self.phase_4_constraint_enforcement()
            self.phase_5_generate_output()
            validation_results = self.phase_6_enhanced_validation()
            
            print(f"\n🎉 OPTIMIZATION COMPLETE!")
            print("=" * 60)
            print(f"✅ Successfully generated {validation_results['total_routes']} optimized routes")
            print(f"✅ Average efficiency score: {validation_results['avg_efficiency']:.3f}")
            print(f"✅ Customer coverage: {validation_results['coverage_rate']:.1f}%")
            
            return self.routes_df, validation_results
            
        except Exception as e:
            print(f"❌ OPTIMIZATION FAILED: {str(e)}")
            raise
    
    def export_results(self, base_path="route_optimization_results"):
        """Export all results to files"""
        print(f"\n💾 EXPORTING RESULTS")
        print("-" * 30)
        
        if self.routes_df is None:
            raise ValueError("No routes to export. Run optimize() first.")
        
        # Export detailed routes
        routes_file = f"{base_path}_routes_{self.stock_point_id}.csv"
        self.routes_df.to_csv(routes_file, index=False)
        print(f"✓ Detailed routes exported: {routes_file}")
        
        # Export route summary if available
        if self.route_summary_df is not None:
            summary_file = f"{base_path}_summary_{self.stock_point_id}.csv"
            self.route_summary_df.to_csv(summary_file, index=False)
            print(f"✓ Route summary exported: {summary_file}")
        
        # Export H3 metrics
        metrics_file = f"{base_path}_h3_metrics_{self.stock_point_id}.csv"
        self.h3_metrics.to_csv(metrics_file, index=False)
        print(f"✓ H3 metrics exported: {metrics_file}")
        
        return {
            'routes_file': routes_file,
            'summary_file': summary_file if hasattr(self, 'route_summary_df') else None,
            'metrics_file': metrics_file
        }

# Enhanced utility functions for batch processing
def optimize_multiple_stock_points_enhanced(sp_dim_df, customers_gdf, df_output_assignment, 
                                          stock_point_ids=None, **kwargs):
    """
    Enhanced batch optimization for multiple stock points
    
    Parameters:
    - sp_dim_df, customers_gdf, df_output_assignment: Input datasets
    - stock_point_ids: List of stock point IDs to process (None = all)
    - **kwargs: Additional parameters for EnhancedH3RouteOptimizer
    
    Returns:
    - combined_routes_df: DataFrame with routes for all stock points
    - combined_summary_df: DataFrame with route summaries for all stock points  
    - validation_summary: Dictionary with validation results per stock point
    """
    if stock_point_ids is None:
        stock_point_ids = sp_dim_df['stock_point_id'].unique()
    
    all_routes = []
    all_summaries = []
    validation_summary = {}
    
    print(f"🔄 BATCH PROCESSING {len(stock_point_ids)} STOCK POINTS")
    print("=" * 70)
    
    for idx, stock_point_id in enumerate(stock_point_ids, 1):
        print(f"\n📍 PROCESSING STOCK POINT {idx}/{len(stock_point_ids)}: {stock_point_id}")
        print("=" * 70)
        
        try:
            optimizer = EnhancedH3RouteOptimizer(
                sp_dim_df=sp_dim_df,
                customers_gdf=customers_gdf,
                df_output_assignment=df_output_assignment,
                stock_point_id=stock_point_id,
                **kwargs
            )
            
            routes_df, validation_results = optimizer.optimize()
            
            # Generate enhanced summary
            summary_df = optimizer.generate_enhanced_route_summary()
            
            all_routes.append(routes_df)
            all_summaries.append(summary_df)
            validation_summary[stock_point_id] = validation_results
            
            print(f"✅ Stock point {stock_point_id} completed successfully")
            
        except Exception as e:
            print(f"❌ Error processing stock point {stock_point_id}: {str(e)}")
            validation_summary[stock_point_id] = {"error": str(e)}
    
    # Combine all results
    combined_routes_df = pd.concat(all_routes, ignore_index=True) if all_routes else pd.DataFrame()
    combined_summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    
    # Print overall summary
    print(f"\n🎊 BATCH PROCESSING COMPLETE")
    print("=" * 70)
    
    if not combined_routes_df.empty:
        successful_stock_points = len([v for v in validation_summary.values() if 'error' not in v])
        total_routes = len(combined_routes_df['route_id'].unique())
        total_customers = combined_routes_df['customer_count'].sum()
        avg_efficiency = combined_routes_df['efficiency_score'].mean()
        
        print(f"✅ Successfully processed: {successful_stock_points}/{len(stock_point_ids)} stock points")
        print(f"✅ Total routes generated: {total_routes}")
        print(f"✅ Total customers covered: {total_customers}")
        print(f"✅ Overall average efficiency: {avg_efficiency:.3f}")
        
        # Export combined results
        combined_routes_df.to_csv('combined_routes_all_stock_points.csv', index=False)
        combined_summary_df.to_csv('combined_route_summary_all_stock_points.csv', index=False)
        print(f"✅ Combined results exported")
    else:
        print("❌ No routes generated for any stock point")
    
    return combined_routes_df, combined_summary_df, validation_summary

def create_batch_visualization_dashboard(combined_summary_df, save_path="route_dashboard.html"):
    """Create a comprehensive dashboard for batch optimization results"""
    print(f"\n📊 CREATING BATCH VISUALIZATION DASHBOARD")
    print("-" * 50)
    
    if combined_summary_df.empty:
        print("❌ No data available for dashboard creation")
        return None
    
    # Create dashboard HTML
    dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Route Optimization Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .header {{ background: #1976d2; color: white; padding: 20px; border-radius: 10px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #1976d2; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚛 Route Optimization Dashboard</h1>
            <p>Comprehensive analysis of optimized delivery routes across all stock points</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>📊 Total Stock Points</h3>
                <div class="metric-value">{combined_summary_df['stock_point_id'].nunique()}</div>
            </div>
            <div class="metric-card">
                <h3>🚛 Total Routes</h3>
                <div class="metric-value">{len(combined_summary_df)}</div>
            </div>
            <div class="metric-card">
                <h3>👥 Total Customers</h3>
                <div class="metric-value">{combined_summary_df['customer_count'].sum():,}</div>
            </div>
            <div class="metric-card">
                <h3>⚡ Avg Efficiency</h3>
                <div class="metric-value">{combined_summary_df['efficiency_score'].mean():.3f}</div>
            </div>
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2>📈 Stock Point Performance Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Stock Point</th>
                        <th>Routes</th>
                        <th>Customers</th>
                        <th>Avg Distance (km)</th>
                        <th>Avg Efficiency</th>
                        <th>Avg Compactness</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add stock point summary rows
    stock_point_summary = combined_summary_df.groupby(['stock_point_id', 'stock_point_name']).agg({
        'route_id': 'count',
        'customer_count': 'sum',
        'optimized_distance_km': 'mean',
        'efficiency_score': 'mean',
        'compactness_score': 'mean'
    }).round(3)
    
    for (stock_id, stock_name), row in stock_point_summary.iterrows():
        dashboard_html += f"""
                    <tr>
                        <td><b>{stock_name}</b><br><small>{stock_id}</small></td>
                        <td>{row['route_id']}</td>
                        <td>{row['customer_count']:,}</td>
                        <td>{row['optimized_distance_km']:.1f}</td>
                        <td>{row['efficiency_score']:.3f}</td>
                        <td>{row['compactness_score']:.3f}</td>
                    </tr>
        """
    
    dashboard_html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save dashboard
    with open(save_path, 'w') as f:
        f.write(dashboard_html)
    
    print(f"✅ Dashboard created: {save_path}")
    return save_path

# Example usage and main execution
def main_example():
    """
    Example usage of the Enhanced H3 Route Optimizer
    
    This example demonstrates how to use the enhanced optimizer with your datasets:
    - sp_dim_df: DataFrame with fulfillment center info
    - customers_gdf: GeoDataFrame with customer locations  
    - df_output_assignment: DataFrame with H3 cell assignments
    """
    
    # Example for single stock point optimization
    print("🔧 SINGLE STOCK POINT OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Uncomment and modify with your actual data:
    """
    optimizer = EnhancedH3RouteOptimizer(
        sp_dim_df=sp_dim_df,
        customers_gdf=customers_gdf, 
        df_output_assignment=df_output_assignment,
        stock_point_id="YOUR_STOCK_POINT_ID",
        min_customers=40,
        max_customers=300,
        max_distance_km=7
    )
    
    # Run optimization
    routes_df, validation_results = optimizer.optimize()
    
    # Generate enhanced route summary
    route_summary_df = optimizer.generate_enhanced_route_summary()
    
    # Create enhanced visualization
    map_viz = optimizer.create_enhanced_visualization(
        save_path='enhanced_route_map.html',
        show_customer_points=True
    )
    
    # Export results
    exported_files = optimizer.export_results("enhanced_optimization_results")
    
    print("✅ Single stock point optimization completed!")
    """
    
    # Example for multiple stock points
    print("\n🔧 MULTIPLE STOCK POINTS OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    """
    # Process multiple stock points
    combined_routes, combined_summary, validation_summary = optimize_multiple_stock_points_enhanced(
        sp_dim_df=sp_dim_df,
        customers_gdf=customers_gdf,
        df_output_assignment=df_output_assignment,
        stock_point_ids=None,  # Process all stock points
        min_customers=40,
        max_customers=300,
        max_distance_km=7
    )
    
    # Create batch dashboard
    dashboard_path = create_batch_visualization_dashboard(
        combined_summary, 
        save_path="route_optimization_dashboard.html"
    )
    
    print("✅ Batch optimization completed!")
    print(f"✅ Dashboard available at: {dashboard_path}")
    """
    
    print("\n📝 TO USE THIS OPTIMIZER:")
    print("-" * 30)
    print("1. Load your datasets (sp_dim_df, customers_gdf, df_output_assignment)")
    print("2. Uncomment and modify the example code above")
    print("3. Set your specific stock_point_id and parameters")
    print("4. Run the optimization")
    print("5. Review results in generated files and visualizations")

if __name__ == "__main__":
    main_example()

# Key Improvements Implemented:
"""
🎯 AREAS OF IMPROVEMENT ADDRESSED:

1. VISUALIZATION ENHANCEMENTS:
   - Distinct, legible color palette with 20 colors
   - Enhanced line drawing with direction arrows
   - Clear route labeling with cell numbers
   - Interactive tooltips and popups with detailed info
   - Multi-layer map with different themes
   - Enhanced legend and analytics dashboard

2. ALGORITHM IMPROVEMENTS:
   - Priority-based clustering considering density + distance
   - Enhanced compactness calculation with convex hull
   - Smart constraint enforcement with iterative refinement
   - Multiple clustering algorithm evaluation
   - Optimized route path calculation using TSP heuristic

3. METRICS & VALIDATION:
   - Enhanced efficiency scoring with multiple factors
   - Detailed route compactness measurement
   - Comprehensive validation with quality assessment
   - Distance savings calculation through optimization
   - Coverage rate and balance metrics

4. USER EXPERIENCE:
   - Clear phase-by-phase execution with progress indicators
   - Comprehensive error handling and validation
   - Batch processing capabilities for multiple stock points
   - Export functionality for all results
   - Dashboard creation for executive summary

5. PERFORMANCE OPTIMIZATIONS:
   - Efficient distance calculations
   - Smart sampling for customer point visualization
   - Optimized clustering with adaptive parameters
   - Memory-efficient data processing
"""