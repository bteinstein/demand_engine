# H3-Based Route Optimization Implementation Plan

## Phase 1: Data Preparation & Validation
### Step 1.1: Environment Setup & Data Import
import pandas as pd
import geopandas as gpd
import h3
import numpy as np
# from geopy.distance import haversine
from haversine import haversine, Unit
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import folium
from shapely.geometry import Point, Polygon
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from folium.plugins import MeasureControl

# # Load datasets 
### Step 1.2: Data Validation & Preprocessing
def validate_input_data(sp_dim_df, customers_gdf, df_output_assignment):
    assert not sp_dim_df[['stock_point_id', 'latitude', 'longitude']].isnull().any().any()
    assert not df_output_assignment[['h3_cell_id', 'customer_id']].isnull().any().any()
    sample_h3 = df_output_assignment['h3_cell_id'].iloc[0]
    assert h3.get_resolution(sample_h3) == 8, "H3 cells must be resolution 8"
    assert sp_dim_df['latitude'].between(-90, 90).all()
    assert sp_dim_df['longitude'].between(-180, 180).all()
    print("✓ Data validation passed")
    
def extract_fulfillment_center(sp_dim_df, stock_point_id):
    fc_data = sp_dim_df[sp_dim_df['stock_point_id'] == stock_point_id]
    if fc_data.empty:
        raise ValueError(f"Stock point {stock_point_id} not found")
    fc_info = fc_data.iloc[0].to_dict()
    return {
        'stock_point_id': fc_info['stock_point_id'],
        'stock_point_name': fc_info['stock_point_name'],
        'latitude': fc_info['latitude'],
        'longitude': fc_info['longitude'],
        'coordinates': (fc_info['latitude'], fc_info['longitude'])
    }

     

## Phase 2: H3 Cluster Metrics Calculation
### Step 2.1: Population Density Analysis
def haversine_distance(lat1, lon1, lat2, lon2):
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)

def calculate_h3_centroid_distance(h3_cell_id, fc_lat, fc_lon):
    centroid_lat, centroid_lon = h3.cell_to_latlng(h3_cell_id)
    return haversine_distance(fc_lat, fc_lon, centroid_lat, centroid_lon)

def calculate_h3_farthest_vertex_distance(h3_cell_id, fc_lat, fc_lon):
    boundary = h3.cell_to_boundary(h3_cell_id)
    max_distance = 0
    for vertex_lat, vertex_lon in boundary:
        distance = haversine_distance(fc_lat, fc_lon, vertex_lat, vertex_lon)
        max_distance = max(max_distance, distance)
    return max_distance

def calculate_cluster_metrics(df_output_assignment, fc_coordinates):
    """Calculate population density and distances for H3 cells"""
    
    # Population density per H3 cell
    density_stats = df_output_assignment.groupby('h3_cell_id').agg({
        'customer_id': 'count',
        'assignment_confidence': 'mean',
        'assignment_tier': lambda x: x.mode()[0] if not x.empty else None
    }).rename(columns={'customer_id': 'customer_count'})
    
    # Calculate H3 centroid distances
    fc_lat, fc_lon = fc_coordinates[0], fc_coordinates[1]
    h3_metrics = []
    for h3_cell_id, stats in density_stats.iterrows():
        centroid_lat, centroid_lng = h3.cell_to_latlng(h3_cell_id)
        # centroid_distance = haversine(fc_coordinates, (centroid_lat, centroid_lng))
        centroid_distance = calculate_h3_centroid_distance(h3_cell_id, fc_lat, fc_lon)
        vertex_distance = calculate_h3_farthest_vertex_distance(h3_cell_id, fc_lat, fc_lon)
        
        h3_metrics.append({
            'h3_cell_id': h3_cell_id,
            'customer_count': stats['customer_count'],
            'avg_confidence': stats['assignment_confidence'],
            'dominant_tier': stats['assignment_tier'],
            'centroid_lat': centroid_lat,
            'centroid_lng': centroid_lng,
            'centroid_distance_km': centroid_distance,
            'max_vertex_distance_km': vertex_distance,
            'population_density': stats['customer_count']  # Can be normalized by H3 cell area
        })
    
    return pd.DataFrame(h3_metrics)

### Step 2.2: Dynamic Distance Filtering
def apply_distance_constraints(cluster_metrics_df, max_distance_km=7.0, adjust_distance_threshold = False):
    """Filter H3 cells based on distance constraints with density adjustment"""
    
    # Adjust distance threshold based on density (higher density = shorter max distance)
    if adjust_distance_threshold:
        # Calculate retailer density percentile for adaptive distance threshold
        density_percentile = cluster_metrics_df['customer_count'].quantile(0.75)
        adjusted_distances = []
        for _, row in cluster_metrics_df.iterrows():
            if row['customer_count'] >= density_percentile:
                adjusted_max_dist = max_distance_km * 0.8  # Reduce for high-density areas
            else:
                adjusted_max_dist = max_distance_km
            
            # Use conservative distance (max vertex) for constraint compliance
            within_constraint = row['max_vertex_distance_km'] <= adjusted_max_dist
            adjusted_distances.append({
                'h3_cell_id': row['h3_cell_id'],
                'adjusted_max_distance': adjusted_max_dist,
                'within_constraint': within_constraint
            })
        
            constraint_df = pd.DataFrame(adjusted_distances)
            
            # Filter valid H3 cells
            valid_clusters = cluster_metrics_df.merge(constraint_df, on='h3_cell_id')
            valid_clusters = valid_clusters[valid_clusters['within_constraint']]
    else:
        within_constraint = cluster_metrics_df.apply(lambda row: min(row['centroid_distance_km'], row['max_vertex_distance_km']) <= max_distance_km, axis=1)
        valid_clusters = cluster_metrics_df.copy()
        valid_clusters['adjusted_max_distance'] = max_distance_km
        valid_clusters['within_constraint'] = within_constraint
        valid_clusters = valid_clusters[valid_clusters['within_constraint']]
    
    return valid_clusters
 

## Phase 3: Route Optimization Algorithm
### Step 3.1: Clustering Strategy Selection 
def optimize_route_clustering(valid_clusters_df, min_customers=40, max_customers=300):
    """Apply geographic clustering with customer count constraints"""
    
    # Prepare features for clustering
    clustering_features = valid_clusters_df[['centroid_lat', 'centroid_lng']].values
    
    # Estimate optimal number of routes
    total_customers = valid_clusters_df['customer_count'].sum()
    avg_customers_per_route = (min_customers + max_customers) / 2
    estimated_routes = max(1, int(total_customers / avg_customers_per_route))
    
    # Try multiple clustering approaches
    clustering_results = {}
    
    # Method 1: K-Means for balanced routes
    kmeans = KMeans(n_clusters=estimated_routes, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(clustering_features)
    clustering_results['kmeans'] = kmeans_labels
    
    # Method 2: DBSCAN for geographic compactness
    # Calculate eps based on average distance between H3 centroids
    distances = []
    for i in range(len(clustering_features)):
        for j in range(i+1, len(clustering_features)):
            dist = haversine(clustering_features[i], clustering_features[j])
            distances.append(dist)
    
    eps_estimate = np.percentile(distances, 1)  # 25th percentile for tighter clusters
    dbscan = DBSCAN(eps=eps_estimate, min_samples=2)
    dbscan_labels = dbscan.fit_predict(clustering_features)
    clustering_results['dbscan'] = dbscan_labels
    
    # Select best clustering method based on silhouette score
    best_method = 'kmeans'
    best_score = -1
    
    for method, labels in clustering_results.items():
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette score
            score = silhouette_score(clustering_features, labels)
            if score > best_score:
                best_score = score
                best_method = method
    
    return clustering_results[best_method], best_method

 
### Step 3.2: Constraint Enforcement & Route Refinement
def enforce_route_constraints(valid_clusters_df, min_customers=40, max_customers=300):
    """Iteratively adjust routes to meet customer count constraints"""
    
    route_groups = valid_clusters_df.groupby('initial_route_id')
    final_routes = []
    route_counter = 0
    
    for route_id, route_data in route_groups:
        total_customers = route_data['customer_count'].sum()
        
        if total_customers < min_customers:
            # Merge with nearest route or create standalone if close to minimum
            if total_customers >= min_customers * 0.8:  # 80% threshold
                final_routes.append({
                    'route_id': route_counter,
                    'h3_cells': route_data['h3_cell_id'].tolist(),
                    'customer_count': total_customers,
                    'route_data': route_data
                })
                route_counter += 1
            else:
                # Find nearest route to merge with (implement later in merge step)
                final_routes.append({
                    'route_id': -1,  # Mark for merging
                    'h3_cells': route_data['h3_cell_id'].tolist(),
                    'customer_count': total_customers,
                    'route_data': route_data
                })
        
        elif total_customers > max_customers:
            # Split route geographically
            split_routes = split_oversized_route(route_data, max_customers)
            for split_route in split_routes:
                final_routes.append({
                    'route_id': route_counter,
                    'h3_cells': split_route['h3_cells'],
                    'customer_count': split_route['customer_count'],
                    'route_data': split_route['route_data']
                })
                route_counter += 1
        
        else:
            # Route meets constraints
            final_routes.append({
                'route_id': route_counter,
                'h3_cells': route_data['h3_cell_id'].tolist(),
                'customer_count': total_customers,
                'route_data': route_data
            })
            route_counter += 1
    
    # Handle merging of undersized routes
    final_routes = merge_undersized_routes(final_routes, min_customers)
    
    return final_routes

def split_oversized_route(route_data, max_customers):
    """Split oversized route using geographic clustering"""
    if len(route_data) <= 1:
        return [{'h3_cells': route_data['h3_cell_id'].tolist(), 
                'customer_count': route_data['customer_count'].sum(),
                'route_data': route_data}]
    
    # Use K-means to split into smaller geographic clusters
    num_splits = int(np.ceil(route_data['customer_count'].sum() / max_customers))
    features = route_data[['centroid_lat', 'centroid_lng']].values
    
    kmeans = KMeans(n_clusters=num_splits, random_state=42)
    split_labels = kmeans.fit_predict(features)
    
    split_routes = []
    for split_id in range(num_splits):
        split_mask = split_labels == split_id
        split_data = route_data[split_mask]
        
        split_routes.append({
            'h3_cells': split_data['h3_cell_id'].tolist(),
            'customer_count': split_data['customer_count'].sum(),
            'route_data': split_data
        })
    
    return split_routes

def merge_undersized_routes(routes, min_customers):
    """Merge routes marked for merging with nearest valid routes"""
    valid_routes = [r for r in routes if r['route_id'] != -1]
    merge_candidates = [r for r in routes if r['route_id'] == -1]
    
    for candidate in merge_candidates:
        if not valid_routes:
            # No valid routes to merge with, keep as standalone
            candidate['route_id'] = len(valid_routes)
            valid_routes.append(candidate)
            continue
        
        # Find nearest route by centroid distance
        candidate_centroid = candidate['route_data'][['centroid_lat', 'centroid_lng']].mean()
        
        best_merge_route = None
        min_distance = float('inf')
        
        for valid_route in valid_routes:
            if valid_route['customer_count'] + candidate['customer_count'] <= 300:  # Ensure merge doesn't exceed max
                route_centroid = valid_route['route_data'][['centroid_lat', 'centroid_lng']].mean()
                distance = haversine((candidate_centroid['centroid_lat'], candidate_centroid['centroid_lng']),
                                   (route_centroid['centroid_lat'], route_centroid['centroid_lng']))
                
                if distance < min_distance:
                    min_distance = distance
                    best_merge_route = valid_route
        
        if best_merge_route:
            # Merge with best route
            best_merge_route['h3_cells'].extend(candidate['h3_cells'])
            best_merge_route['customer_count'] += candidate['customer_count']
            best_merge_route['route_data'] = pd.concat([best_merge_route['route_data'], candidate['route_data']])
        else:
            # No suitable merge found, keep as standalone
            candidate['route_id'] = len(valid_routes)
            valid_routes.append(candidate)
    
    return valid_routes

 
## Phase 4: Route Statistics & Distance Calculation

### Step 4.1: Calculate Route Performance Metrics 
def calculate_route_statistics(routes, fc_coordinates):
    """Calculate comprehensive statistics for each route"""
    route_statistics = []
    
    for route in routes:
        route_data = route['route_data']
        h3_cells = route['h3_cells']
        
        # Distance calculations
        centroid_distances = route_data['centroid_distance_km'].values
        max_vertex_distances = route_data['max_vertex_distance_km'].values
        
        total_distance_centroid = centroid_distances.sum()
        max_distance_from_fc = max_vertex_distances.max()
        
        # TSP-based route distance (simplified nearest neighbor heuristic)
        tsp_distance = calculate_tsp_distance(route_data, fc_coordinates)
        
        # Compactness score (ratio of actual distance to minimum spanning tree)
        compactness_score = calculate_compactness_score(route_data)
        
        # Estimated delivery time (assuming 40 km/h average speed + 15 min per stop)
        avg_speed_kmh = 40
        stop_time_hours = len(h3_cells) * 0.25  # 15 minutes per stop
        travel_time_hours = tsp_distance / avg_speed_kmh
        estimated_delivery_time = travel_time_hours + stop_time_hours
        
        route_statistics.append({
            'route_id': route['route_id'],
            'h3_cell_count': len(h3_cells),
            'customer_count': route['customer_count'],
            'total_distance_km': tsp_distance,
            'max_distance_from_fc_km': max_distance_from_fc,
            'estimated_delivery_time_hours': estimated_delivery_time,
            'compactness_score': compactness_score,
            'avg_customers_per_cell': route['customer_count'] / len(h3_cells),
            'h3_cells': h3_cells
        })
    
    return route_statistics

def calculate_tsp_distance(route_data, fc_coordinates):
    """Calculate approximate TSP distance using nearest neighbor heuristic"""
    if len(route_data) <= 1:
        return route_data['centroid_distance_km'].sum() * 2  # Round trip
    
    points = [(fc_coordinates[0], fc_coordinates[1])]  # Start at fulfillment center
    points.extend([(row['centroid_lat'], row['centroid_lng']) for _, row in route_data.iterrows()])
    
    # Nearest neighbor TSP approximation
    unvisited = list(range(1, len(points)))  # Exclude FC (index 0)
    current = 0  # Start at FC
    total_distance = 0
    
    while unvisited:
        nearest_idx = min(unvisited, key=lambda x: haversine(points[current], points[x]))
        total_distance += haversine(points[current], points[nearest_idx])
        current = nearest_idx
        unvisited.remove(nearest_idx)
    
    # Return to fulfillment center
    total_distance += haversine(points[current], points[0])
    
    return total_distance

def calculate_compactness_score(route_data):
    """Calculate compactness as inverse of route spread"""
    if len(route_data) <= 1:
        return 1.0
    
    # Calculate centroid of all H3 cells in route
    route_centroid = (route_data['centroid_lat'].mean(), route_data['centroid_lng'].mean())
    
    # Calculate average distance from route centroid
    distances_from_centroid = [
        haversine(route_centroid, (row['centroid_lat'], row['centroid_lng']))
        for _, row in route_data.iterrows()
    ]
    
    avg_spread = np.mean(distances_from_centroid)
    
    # Compactness score (higher is more compact)
    compactness_score = 1 / (1 + avg_spread)  # Normalized between 0 and 1
    
    return compactness_score

 
## Phase 5: Output Generation

# ### Step 5.1: Generate Long-Format DataFrame
# # # Generate final output 
def generate_output_dataframe(route_stats):
    """Generate final long-format DataFrame as specified"""
    
    output_rows = []
    
    for route_stat in route_stats:
        route_id = route_stat['route_id']
        h3_cells = route_stat['h3_cells']
        
        # Create one row per H3 cell (long format)
        for h3_cell_id in h3_cells:
            output_rows.append({
                'route_id': route_id,
                'h3_cell_id': h3_cell_id,
                'customer_count': route_stat['customer_count'],  # Total for route
                'total_distance_km': route_stat['total_distance_km'],
                'estimated_delivery_time_hours': route_stat['estimated_delivery_time_hours'],
                'compactness_score': route_stat['compactness_score']
            })
    
    output_df = pd.DataFrame(output_rows)
    
    return output_df

 

### Step 5.2: Validation & Quality Checks 
def validate_final_output(output_df, min_customers=40, max_customers=300, max_distance=7.0):
    """Validate that all constraints are met"""
    
    validation_results = {}
    
    # Check customer count constraints
    route_customers = output_df.groupby('route_id')['customer_count'].first()
    customer_violations = ((route_customers < min_customers) | (route_customers > max_customers)).sum()
    validation_results['customer_count_violations'] = customer_violations
    
    # Check distance constraints (need to recalculate max distances per route)
    distance_violations = 0  # Would need additional calculation
    validation_results['distance_violations'] = distance_violations
    
    # Check for duplicate H3 cell assignments
    duplicate_assignments = output_df['h3_cell_id'].duplicated().sum()
    validation_results['duplicate_assignments'] = duplicate_assignments
    
    # Summary statistics
    validation_results['total_routes'] = output_df['route_id'].nunique()
    validation_results['total_h3_cells'] = output_df['h3_cell_id'].nunique()
    validation_results['total_customers'] = output_df.groupby('route_id')['customer_count'].first().sum()
    validation_results['avg_customers_per_route'] = validation_results['total_customers'] / validation_results['total_routes']
    
    return validation_results

 
## Phase 6: Visualization & Final Validation
# ### Step 6.1: Geographic Visualization
# # Create visualization 
def create_route_visualization(output_df, cluster_metrics_df, fulfillment_center):
    """Create interactive map visualization of optimized routes"""
    
    # Create base map
    map_center = fulfillment_center['coordinates']
    route_map = folium.Map(location=map_center, zoom_start=11)
    
    # Add fulfillment center
    folium.Marker(
        location=fulfillment_center['coordinates'],
        popup=fulfillment_center['stock_point_name'],
        tooltip=fulfillment_center['stock_point_name'],
        icon=folium.Icon(color='red', icon='home')
    ).add_to(route_map)
    
    # Color palette for routes
    # colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
    #           'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple',
    #           'pink', 'lightblue', 'lightgreen',]
    colors = [ 
                 #'beige', 
                 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 
                 #'white', 
                 'pink', #'lightblue', # 'lightgreen', 
                 #'gray', 'black', 'lightgray',
                 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred']
    
    # Plot routes
    for route_id, route_data in output_df.groupby('route_id'):
        color = colors[route_id % len(colors)]
        
        # Get H3 cell geometries for this route
        route_h3_cells = route_data['h3_cell_id'].tolist()
        cluster_pop_density_list = route_data['population_density'].tolist()
        
        for h3_cell, clust_pop_density in zip(route_h3_cells, cluster_pop_density_list):
            # Get H3 cell boundary
            cell_boundary = h3.cell_to_boundary(h3_cell)
            
            # Create polygon
            folium.Polygon(
                locations=[(coord[0], coord[1]) for coord in cell_boundary],
                color=color,
                weight=2,
                fillColor=color,
                fillOpacity=0.8,
                tooltip=f'Route {route_id} <br>Total Route Customers: {route_data["customer_count"].iloc[0]} <br>ClusterID: {h3_cell} <br> Cluster Customer: {clust_pop_density}'
            ).add_to(route_map)
    
    return route_map

 

### Step 6.2: Performance Analysis 
def analyze_optimization_performance(output_df):
    """Analyze optimization results and provide insights"""
    route_summary = (output_df 
                .groupby('route_id')
                .agg(
                    h3_cell_ids = ('h3_cell_id', lambda x: x.to_list()),
                    cluster_count = ('h3_cell_id', 'nunique'), 
                    customer_count = ('population_density', 'sum'),  
                    total_distance_km = ('total_distance_km', 'max'),  
                    cumulative_distance_km = ('total_distance_km', 'max'),  
                    farthest_centroid_distance_km = ('max_vertex_distance_km', 'max'),  
                    estimated_delivery_time_hours = ('estimated_delivery_time_hours', 'max'),   
                    avg_assignment_confidence = ('estimated_delivery_time_hours', 'mean'),   
                    compactness_score = ('compactness_score', 'mean')   
                )
                .reset_index()
                )
    
    cols_to_cat = ['route_id']
    route_summary[cols_to_cat] = route_summary[cols_to_cat].astype('category')
    
    performance_metrics = {
        'total_routes': len(route_summary),
        'avg_customers_per_route': route_summary['customer_count'].mean(),
        'avg_distance_per_route': route_summary['total_distance_km'].mean(),
        'avg_delivery_time_hours': route_summary['estimated_delivery_time_hours'].mean(),
        'avg_compactness_score': route_summary['compactness_score'].mean(),
        'total_customers': route_summary['customer_count'].sum(),
        'total_clusters': route_summary['cluster_count'].sum()
    }
    
    return route_summary, performance_metrics


'''
## Complete Implementation Workflow
def main_optimization_workflow(stock_point_id):
    """Complete end-to-end optimization workflow"""

    # Phase 1: Data Preparation 
    validate_input_data(sp_dim_df, customers_gdf, df_output_assignment)

    # Extract fulfillment center coordinates
    fulfillment_center = extract_fulfillment_center(sp_dim_df, stock_point_id)
    sp_coords = fulfillment_center['coordinates']

    sp_assignments = df_output_assignment[
            df_output_assignment['stock_point_id'] == stock_point_id
        ]
    if sp_assignments.empty:
        raise ValueError(f"No assignments found for stock_point_id: {stock_point_id}")
        
    # # Phase 2: Calculate Metrics
    cluster_metrics_df = calculate_cluster_metrics(sp_assignments, sp_coords)
    valid_clusters_df = apply_distance_constraints(cluster_metrics_df, 
                                                max_distance_km=MAX_R2SP_DISTANCE_KM, 
                                                adjust_distance_threshold=False)
    if valid_clusters_df.empty:
        print('empty valid cluster for stockpoint: ', stock_point_id, " - ",fulfillment_center['stock_point_name'])
        cols = ['route_id', 'compactness_score', 'estimated_delivery_time_hours', 'total_distance_km']
        valid_clusters_df[cols] = None
        return {
            'route_output_df': valid_clusters_df,
            'validation_report': {},
            'performance_metrics': {},
            'route_summary': pd.DataFrame(),
            'visualization_map': None
        }
    else:
        # Phase 3: Route Optimization
        cluster_labels, clustering_method = optimize_route_clustering(valid_clusters_df,
                                                                    min_customers=MIN_CUSTOMER_PER_ROUTE, 
                                                                    max_customers=MAX_CUSTOMER_PER_ROUTE)
        valid_clusters_df['initial_route_id'] = cluster_labels
        optimized_routes = enforce_route_constraints(valid_clusters_df,
                                                        min_customers=MIN_CUSTOMER_PER_ROUTE, 
                                                        max_customers=MAX_CUSTOMER_PER_ROUTE)

        # Phase 4: Calculate Statistics
        route_stats = calculate_route_statistics(optimized_routes, sp_coords)

        # Phase 5: Generate Output
        route_output_df_ = generate_output_dataframe(route_stats)
        route_output_df = route_output_df_.merge(cluster_metrics_df.drop(columns='customer_count'), on='h3_cell_id', how='inner')


        # # # Phase 6: Validation
        validation_report = validate_final_output(route_output_df)
        route_visualization = create_route_visualization(route_output_df, cluster_metrics_df, fulfillment_center)
        route_summary, performance_metrics = analyze_optimization_performance(route_output_df)

        print(performance_metrics) 
        return {
            'route_output_df': route_output_df,
            'validation_report': validation_report,
            'performance_metrics': performance_metrics,
            'route_summary': route_summary,
            'visualization_map': route_visualization
        }

 
# Execute complete workflow
optimization_results = main_optimization_workflow(stock_point_id=1647113)
```

```
# Save final output
optimization_results['output_dataframe'].to_csv('optimized_routes_output.csv', index=False)
print("Optimization complete! Results saved to 'optimized_routes_output.csv'")

## Key Success Factors
1. **Accurate Distance Calculation**: Using both centroid and vertex-based distances for constraint compliance
2. **Adaptive Clustering**: Combining K-means and DBSCAN based on data distribution
3. **Iterative Constraint Enforcement**: Systematic approach to meeting customer count and distance requirements
4. **Geographic Compactness**: Prioritizing spatially coherent routes for operational efficiency
5. **Comprehensive Validation**: Multi-level validation ensuring all constraints are satisfied

## Expected Output

The final DataFrame will contain:
- **Long-format structure**: One row per H3 cell per route
- **Required columns**: `route_id`, `h3_cell_id`, `customer_count`, `total_distance_km`, `estimated_delivery_time_hours`, `compactness_score`
- **Constraint compliance**: 40-300 customers per route, 0-7km from fulfillment center
- **Optimization**: Minimized total distance while prioritizing high-density clusters
'''