import math
import folium
import openrouteservice as ors
from datetime import datetime
import logging
from typing import Tuple, List, Dict, Any, Optional
# Import your plotting function (assuming it's in the same directory or properly installed)
from Algorithm_V1.src.map_viz.plot_cluster import create_enhanced_cluster_map

from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(df,
                        id_col = 'CustomerID'
                        ,cluster_col = 'cluster'
                        ,lat_col = 'Latitude'
                        ,lon_col = 'Longitude'
                        ):
    # Check input
    required_cols = {id_col, cluster_col, lat_col, lon_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Cluster size distribution
    cluster_counts = df[cluster_col].value_counts().sort_index()

    # Silhouette score
    coords = df[[lat_col, lon_col]].values
    labels = df[cluster_col].values
    unique_clusters = np.unique(labels)
    
    if len(unique_clusters) > 1 and len(unique_clusters) < len(df):
        silhouette = silhouette_score(coords, labels)
    else:
        silhouette = np.nan  # Not enough clusters for silhouette

    # Compute cluster centroids
    centroids = df.groupby(cluster_col)[[lat_col, lon_col]].mean()

    # Intra-cluster distances
    intra_cluster_distances = {}
    for cluster_id in unique_clusters:
        cluster_points = df[df[cluster_col] == cluster_id][[lat_col, lon_col]].values
        centroid = centroids.loc[cluster_id].values
        distances = cdist(cluster_points, [centroid])
        intra_cluster_distances[cluster_id] = distances.mean()

    # Return summary
    return {
        "cluster_counts": cluster_counts,
        "silhouette_score": silhouette,
        "centroids": centroids,
        "intra_cluster_distances": intra_cluster_distances
    }


 
#  ----------------------------------------------------


def run_route_optimizer(df_clustering, 
                       sel_cluster_tuple: Tuple, 
                       df_stockpoint, 
                       stock_point_name: str,
                       sel_total_customer_count: int, 
                       capacity_size: int = 20,
                       max_ors_locations: int = 50,  # Conservative limit to avoid ORS errors
                       client=None) -> Dict[str, Any]:
    """
    Optimized route planning function with improved error handling and capacity management.
    
    Args:
        df_clustering: DataFrame with customer clustering data
        sel_cluster_tuple: Tuple of selected cluster IDs
        df_stockpoint: DataFrame with stock point/depot information
        stock_point_name: Name of the stock point
        sel_total_customer_count: Total number of customers to serve
        capacity_size: Maximum customers per vehicle (default: 20)
        max_ors_locations: Maximum locations per ORS API call (default: 50)
        client: OpenRouteService client instance
        
    Returns:
        Dict containing optimization results, map object, and statistics
    """
    
    # Validate inputs
    if capacity_size > 20:
        capacity_size = 20
        logging.warning(f"Capacity size reduced to maximum allowed: {capacity_size}")
    
    # Select and validate cluster data
    df_sel_clust = df_clustering.query(f'cluster in {sel_cluster_tuple}').query('Latitude > 0')
    
    if df_sel_clust.empty:
        raise ValueError("No valid customer locations found in selected clusters")
    
    # Prepare coordinates in ORS format [longitude, latitude]
    coords = [[lon, lat] for lat, lon in zip(df_sel_clust.Latitude, df_sel_clust.Longitude)]
    total_customers = len(coords)
    
    print(f"Number of customer locations: {total_customers}")
    
    # Validate depot location
    if df_stockpoint.empty or df_stockpoint.Latitude.iloc[0] <= 0:
        raise ValueError("Invalid depot location")
    
    vehicle_start = [df_stockpoint.Longitude.iloc[0], df_stockpoint.Latitude.iloc[0]]
    depot_location = [df_stockpoint.Latitude.iloc[0], df_stockpoint.Longitude.iloc[0]]
    depot_name = df_stockpoint.Stock_point_Name.iloc[0]
    
    # Calculate number of vehicles needed
    num_vehicles = math.ceil(total_customers / capacity_size)
    print(f"Number of vehicles needed: {num_vehicles}")
    
    # Check if we need to split into multiple ORS calls
    max_locations_per_call = max_ors_locations - 1  # Reserve 1 for depot
    
    optimization_results = []
    all_routes = []
    
    if total_customers <= max_locations_per_call:
        # Single ORS call - process all customers at once
        try:
            result = _process_single_batch(coords, vehicle_start, num_vehicles, capacity_size, client)
            optimization_results.append(result)
            all_routes.extend(result.get('routes', []))
        except Exception as e:
            logging.error(f"ORS API error for single batch: {e}")
            raise
    else:
        # Multiple ORS calls - split customers into batches
        print(f"Splitting {total_customers} customers into batches of max {max_locations_per_call}")
        
        for i in range(0, total_customers, max_locations_per_call):
            batch_coords = coords[i:i + max_locations_per_call]
            batch_size = len(batch_coords)
            batch_num_vehicles = math.ceil(batch_size / capacity_size)
            
            print(f"Processing batch {i//max_locations_per_call + 1}: {batch_size} customers, {batch_num_vehicles} vehicles")
            
            try:
                result = _process_single_batch(batch_coords, vehicle_start, batch_num_vehicles, capacity_size, client)
                
                # Adjust route vehicle IDs to avoid conflicts
                if result.get('routes'):
                    for route in result['routes']:
                        route['vehicle'] += len(all_routes)
                
                optimization_results.append(result)
                all_routes.extend(result.get('routes', []))
                
            except Exception as e:
                logging.error(f"ORS API error for batch {i//max_locations_per_call + 1}: {e}")
                continue  # Skip this batch and continue with others
    
    if not all_routes:
        raise RuntimeError("No routes were successfully generated")
    
    # Create enhanced map visualization
    try:
        map_clusters_route = _create_route_map(
            df_sel_clust, depot_location, depot_name, all_routes, 
            coords, len(optimization_results)
        )
    except ImportError:
        logging.warning("Could not import plot_cluster module. Using basic map creation.")
        # Fallback to basic Folium map if plot_cluster is not available
        map_clusters_route = folium.Map(
            location=depot_location,
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        # Add basic depot marker
        folium.Marker(
            location=depot_location,
            tooltip=f"Depot: {depot_name}",
            icon=folium.Icon(color="green", icon="home")
        ).add_to(map_clusters_route)
        
        # Add basic customer markers
        for idx, coord in enumerate(coords):
            folium.CircleMarker(
                location=[coord[1], coord[0]],  # Convert [lon, lat] to [lat, lon]
                radius=5,
                popup=f"Customer {idx+1}",
                color='blue',
                fillColor='lightblue',
                fillOpacity=0.7
            ).add_to(map_clusters_route)
        
        # Add basic routes
        separable_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for idx, route in enumerate(all_routes):
            if 'geometry' in route:
                color = separable_colors[idx % len(separable_colors)]
                route_coords = ors.convert.decode_polyline(route['geometry'])['coordinates']
                folium_coords = [[lat, lon] for lon, lat in route_coords]
                
                folium.PolyLine(
                    locations=folium_coords,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Route {idx + 1}"
                ).add_to(map_clusters_route)
    
    # Save map
    selected_trip_map_path = f'./recommendation_output/selected_trip_map/{stock_point_name}_{datetime.today().date()}.html'
    map_clusters_route.save(selected_trip_map_path)
    
    # Calculate statistics
    total_routes = len(all_routes)
    customers_per_route = [len(route.get('steps', [])) - 2 for route in all_routes]  # Exclude start/end depot
    
    stats = {
        'total_customers': total_customers,
        'total_routes': total_routes,
        'customers_per_route': customers_per_route,
        'avg_customers_per_route': sum(customers_per_route) / len(customers_per_route) if customers_per_route else 0,
        'map_path': selected_trip_map_path
    }
    
    print(f"Optimization completed: {total_routes} routes for {total_customers} customers")
    print(f"Customers per route: {customers_per_route}")
    
    return {
        'optimization_results': optimization_results,
        'routes': all_routes,
        'map': map_clusters_route,
        'statistics': stats
    }


def _process_single_batch(coords: List[List[float]], 
                         vehicle_start: List[float], 
                         num_vehicles: int, 
                         capacity_size: int, 
                         client) -> Dict[str, Any]:
    """Process a single batch of customers through ORS API."""
    
    # Create vehicles
    vehicles = [
        ors.optimization.Vehicle(
            id=i,
            profile='driving-car',
            start=vehicle_start,
            end=vehicle_start,
            capacity=[capacity_size]
        ) for i in range(num_vehicles)
    ]
    
    # Create jobs
    jobs = [
        ors.optimization.Job(id=idx, location=coord, amount=[1]) 
        for idx, coord in enumerate(coords)
    ]
    
    # Call ORS optimization API
    try:
        optimized = client.optimization(jobs=jobs, vehicles=vehicles, geometry=True)
        return optimized
    except Exception as e:
        if "413" in str(e) or "Too many locations" in str(e):
            raise RuntimeError(f"ORS location limit exceeded: {len(coords)} locations")
        else:
            raise RuntimeError(f"ORS API error: {e}")


def _create_route_map(df_sel_clust, depot_location: List[float], depot_name: str, 
                     all_routes: List[Dict], coords: List[List[float]], 
                     num_batches: int) -> folium.Map:
    """Create an enhanced Folium map with route visualization using your existing plot function."""
    
    
    
    # Create base map using your enhanced function
    map_clusters_route = create_enhanced_cluster_map(
        df_sel_clust,
        popup_cols=['CustomerID', 'LGA', 'LCDA'],
        tooltip_cols=['LGA', 'LCDA'], 
        zoom_start=10, 
        radius=10
    )
    
    # Add depot marker
    folium.Marker(
        location=depot_location, 
        tooltip=f"Depot: {depot_name}", 
        popup=f"<b>Depot</b><br>{depot_name}",
        icon=folium.Icon(color="green", icon="home")
    ).add_to(map_clusters_route)
    
    # Define colors for routes
    separable_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5"
    ]
    
    # Add routes to map with enhanced styling
    for idx, route in enumerate(all_routes):
        if 'geometry' in route:
            color = separable_colors[idx % len(separable_colors)]
            
            # Decode polyline and create route
            route_coords = ors.convert.decode_polyline(route['geometry'])['coordinates']
            # Convert from [lon, lat] to [lat, lon] for Folium
            folium_coords = [[lat, lon] for lon, lat in route_coords]
            
            # Calculate route statistics for better popups
            route_distance = route.get('summary', {}).get('distance', 'N/A')
            route_duration = route.get('summary', {}).get('duration', 'N/A')
            
            # Format distance and duration
            distance_str = f"{route_distance/1000:.1f} km" if isinstance(route_distance, (int, float)) else "N/A"
            duration_str = f"{route_duration/60:.0f} min" if isinstance(route_duration, (int, float)) else "N/A"
            
            # Enhanced popup content
            popup_content = f"""
            <div style='font-family: Arial, sans-serif; font-size: 12px;'>
                <div style='font-weight: bold; color: {color}; margin-bottom: 5px;'>
                    Route {idx + 1} (Vehicle {route.get('vehicle', idx) + 1})
                </div>
                <div style='margin: 2px 0;'><strong>Distance:</strong> {distance_str}</div>
                <div style='margin: 2px 0;'><strong>Duration:</strong> {duration_str}</div>
                <div style='margin: 2px 0;'><strong>Stops:</strong> {len(route.get('steps', [])) - 2} customers</div>
            </div>
            """
            
            folium.PolyLine(
                locations=folium_coords,
                color=color,
                weight=4,
                opacity=0.8,
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Route {idx + 1} - {distance_str}, {duration_str}"
            ).add_to(map_clusters_route)
    
    # Add enhanced route legend positioned to avoid conflict with cluster legend
    route_legend_html = _create_route_legend_html(len(all_routes), separable_colors, all_routes)
    map_clusters_route.get_root().html.add_child(folium.Element(route_legend_html))
    
    return map_clusters_route


def _create_route_legend_html(num_routes: int, colors: List[str], all_routes: List[Dict]) -> str:
    """Create HTML legend for route colors with enhanced statistics."""
    
    legend_html = '''
    <div id="route-legend" style="position: fixed; 
                top: 10px; left: 280px; width: 220px; height: auto; 
                background-color: rgba(255, 255, 255, 0.95); 
                border: 2px solid #007bff; z-index: 1001; 
                font-size: 11px; padding: 10px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                backdrop-filter: blur(5px);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <strong style="color: #007bff;">Route Overview</strong>
        <button onclick="toggleRouteVisibility()" style="border: none; background: none; cursor: pointer; font-size: 12px; color: #007bff;">👁</button>
    </div>
    <div style="font-size: 9px; color: #666; margin-bottom: 8px;">
        Click routes to toggle visibility
    </div>
    <div id="route-legend-content" style="max-height: 200px; overflow-y: auto;">
    '''
    
    # Calculate total statistics
    total_distance = 0
    total_duration = 0
    total_customers = 0
    
    for i in range(min(num_routes, len(colors))):
        if i < len(all_routes):
            route = all_routes[i]
            route_distance = route.get('summary', {}).get('distance', 0)
            route_duration = route.get('summary', {}).get('duration', 0)
            route_stops = len(route.get('steps', [])) - 2  # Exclude start/end depot
            
            total_distance += route_distance if isinstance(route_distance, (int, float)) else 0
            total_duration += route_duration if isinstance(route_duration, (int, float)) else 0
            total_customers += route_stops
            
            # Format route info
            distance_str = f"{route_distance/1000:.1f}km" if isinstance(route_distance, (int, float)) else "N/A"
            duration_str = f"{route_duration/60:.0f}min" if isinstance(route_duration, (int, float)) else "N/A"
            
            color = colors[i % len(colors)]
            legend_html += f'''
            <div style="margin: 4px 0; display: flex; align-items: center; padding: 3px; 
                        border-radius: 4px; border: 1px solid #eee; cursor: pointer;
                        transition: all 0.2s;" 
                 onmouseover="this.style.backgroundColor='#f8f9fa'" 
                 onmouseout="this.style.backgroundColor='white'"
                 onclick="toggleRoute({i})">
                <span style="display: inline-block; width: 16px; height: 16px; 
                             background-color: {color}; border-radius: 3px; 
                             margin-right: 8px; border: 1px solid white; 
                             box-shadow: 0 1px 2px rgba(0,0,0,0.2);"></span>
                <div style="flex: 1; font-size: 10px;">
                    <div style="font-weight: bold;">Route {i+1}</div>
                    <div style="color: #666; font-size: 9px;">
                        {route_stops} stops • {distance_str} • {duration_str}
                    </div>
                </div>
            </div>
            '''
    
    # Add summary statistics
    legend_html += f"""
                </div>
                <div style="border-top: 1px solid #ddd; margin-top: 8px; padding-top: 8px; font-size: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <span style="color: #666;">Total Distance:</span>
                        <span style="font-weight: bold;">{total_distance/1000:.1f} km</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <span style="color: #666;">Total Duration:</span>
                        <span style="font-weight: bold;">{total_duration/60:.0f} min</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #666;">Total Customers:</span>
                        <span style="font-weight: bold;">{total_customers}</span>
                    </div>
                </div>
                <script>
                var routeVisibility = {{}};
                
                function toggleRoute(routeIndex) {{
                    // This would ideally interact with the route layers
                    // For now, just provide visual feedback
                    var routeElements = document.querySelectorAll('path[stroke]');
                    if (routeElements[routeIndex]) {{
                        var currentOpacity = routeElements[routeIndex].style.opacity || '0.8';
                        var newOpacity = currentOpacity === '0.8' ? '0.2' : '0.8';
                        routeElements[routeIndex].style.opacity = newOpacity;
                        routeVisibility[routeIndex] = newOpacity === '0.8';
                    }}
                }}
                
                function toggleRouteVisibility() {{
                    var routeElements = document.querySelectorAll('path[stroke]');
                    var allVisible = Object.values(routeVisibility).every(v => v);
                    
                    routeElements.forEach(function(element, index) {{
                        element.style.opacity = allVisible ? '0.2' : '0.8';
                        routeVisibility[index] = !allVisible;
                    }});
                }}
                </script>
                </div>
    """
    
    return legend_html