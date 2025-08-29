import pandas as pd
import numpy as np
import h3
import folium
from folium import plugins
import math
from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import seaborn as sns

class H3DeliveryRouter:
    def __init__(self, df, cells_per_route=6):
        self.df = df.copy()
        self.cells_per_route = cells_per_route
        self.stock_point = {
            'lat': df['sp_latitude'].iloc[0],
            'lng': df['sp_longitude'].iloc[0],
            'name': df['stock_point_name'].iloc[0]
        }
        
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
        bearings = []
        directions = []
        simplified_directions = []
        
        for _, row in self.df.iterrows():
            bearing = self.calculate_bearing(
                self.stock_point['lat'], self.stock_point['lng'],
                row['cluster_centroid_lat'], row['cluster_centroid_lng']
            )
            bearings.append(bearing)
            directions.append(self.get_direction_label(bearing))
            simplified_directions.append(self.get_simplified_direction(bearing))
        
        self.df['bearing'] = bearings
        self.df['direction'] = directions
        self.df['simplified_direction'] = simplified_directions
        
        return self.df
    
    def get_h3_neighbors(self, h3_cell, k=1):
        """Get H3 neighbors within k distance"""
        try:
            neighbors = h3.grid_ring(h3_cell, k)
            return list(neighbors)
        except:
            return [h3_cell]
    
    def create_directional_routes(self):
        """Create routes using directional + hexagonal growth hybrid approach"""
        # Calculate directions first
        self.calculate_directions()
        
        routes = []
        route_id = 1
        processed_cells = set()
        
        # Group by simplified direction
        direction_groups = self.df.groupby('simplified_direction')
        
        for direction, group in direction_groups:
            group_cells = group.copy().sort_values('cluster_sp_dist_km')
            
            # Create routes within each direction
            current_route = []
            
            for _, row in group_cells.iterrows():
                if row['h3_cell'] in processed_cells:
                    continue
                    
                # Start new route if current is full
                if len(current_route) >= self.cells_per_route:
                    # Save current route
                    route_data = {
                        'route_id': route_id,
                        'direction': direction,
                        'cells': current_route.copy(),
                        'avg_distance': np.mean([c['cluster_sp_dist_km'] for c in current_route])
                    }
                    routes.append(route_data)
                    route_id += 1
                    current_route = []
                
                # Add cell to current route
                current_route.append({
                    'h3_cell': row['h3_cell'],
                    'lat': row['cluster_centroid_lat'],
                    'lng': row['cluster_centroid_lng'],
                    'cluster_sp_dist_km': row['cluster_sp_dist_km'],
                    'direction': row['direction'],
                    'bearing': row['bearing']
                })
                processed_cells.add(row['h3_cell'])
            
            # Add remaining cells as final route for this direction
            if current_route:
                route_data = {
                    'route_id': route_id,
                    'direction': direction,
                    'cells': current_route,
                    'avg_distance': np.mean([c['cluster_sp_dist_km'] for c in current_route])
                }
                routes.append(route_data)
                route_id += 1
        
        self.routes = routes
        return routes
    
    def create_output_dataframe(self):
        """Create output dataframe with route assignments"""
        output_rows = []
        
        for route in self.routes:
            for cell in route['cells']:
                output_rows.append({
                    'h3_cell': cell['h3_cell'],
                    'route_id': route['route_id'],
                    'direction_simplified': route['direction'],
                    'direction_detailed': cell['direction'],
                    'bearing': cell['bearing'],
                    'cluster_centroid_lat': cell['lat'],
                    'cluster_centroid_lng': cell['lng'],
                    'cluster_sp_dist_km': cell['cluster_sp_dist_km']
                })
        
        return pd.DataFrame(output_rows)
    
    def get_h3_polygon(self, h3_cell):
        """Get polygon coordinates for H3 cell - folium expects [lat, lng] format"""
        try:
            # h3.cell_to_boundary returns [(lat, lng), (lat, lng), ...] tuples
            boundary = h3.cell_to_boundary(h3_cell)
            # Convert tuples to list format that folium expects: [[lat, lng], [lat, lng], ...]
            coords = [[lat, lng] for lat, lng in boundary]
            return coords
        except:
            return None
    
    def create_route_polygons(self, output_df, use_vertices=False):
        """Create polygons for each route - either convex hull or vertex tracing"""
        route_polygons = {}
        
        for route_id in output_df['route_id'].unique():
            route_cells = output_df[output_df['route_id'] == route_id]
            
            if use_vertices:
                # Method 1: Trace vertices of all H3 cells in route
                all_vertices = []
                for _, row in route_cells.iterrows():
                    polygon_coords = self.get_h3_polygon(row['h3_cell'])
                    if polygon_coords:
                        all_vertices.extend(polygon_coords)
                
                if all_vertices:
                    # Remove duplicates while preserving order
                    unique_vertices = []
                    seen = set()
                    for vertex in all_vertices:
                        vertex_tuple = tuple(vertex)
                        if vertex_tuple not in seen:
                            unique_vertices.append(vertex)
                            seen.add(vertex_tuple)
                    route_polygons[route_id] = unique_vertices
            else:
                # Method 2: Convex hull approach
                points = []
                for _, row in route_cells.iterrows():
                    polygon_coords = self.get_h3_polygon(row['h3_cell'])
                    if polygon_coords:
                        for coord in polygon_coords:
                            points.append(Point(coord[1], coord[0]))  # Point expects (lng, lat)
                
                if points and len(points) >= 3:
                    try:
                        from shapely.geometry import MultiPoint
                        multi_point = MultiPoint(points)
                        hull = multi_point.convex_hull
                        
                        if hull.geom_type == 'Polygon' and hasattr(hull, 'exterior'):
                            coords = [[y, x] for x, y in list(hull.exterior.coords)]
                            route_polygons[route_id] = coords
                        else:
                            # Fallback to centroids
                            coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                                     for _, row in route_cells.iterrows()]
                            route_polygons[route_id] = coords
                    except:
                        # Final fallback
                        coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                                 for _, row in route_cells.iterrows()]
                        route_polygons[route_id] = coords
                else:
                    coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                             for _, row in route_cells.iterrows()]
                    route_polygons[route_id] = coords
        
        return route_polygons
    
    def create_visualization(self, output_df, use_vertices=False):
        """Create folium visualization with option for vertex tracing or convex hull"""
        # Center map on stock point
        m = folium.Map(
            location=[self.stock_point['lat'], self.stock_point['lng']],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add stock point marker
        folium.Marker(
            location=[self.stock_point['lat'], self.stock_point['lng']],
            popup=f"<b>{self.stock_point['name']}</b><br>Distribution Center",
            tooltip="Distribution Center",
            icon=folium.Icon(color='red', icon='home')
        ).add_to(m)
        
        # Color palette for routes
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
                 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
        
        # Add H3 cells for each route
        for route_id in output_df['route_id'].unique():
            route_cells = output_df[output_df['route_id'] == route_id]
            color = colors[route_id % len(colors)]
            
            for _, row in route_cells.iterrows():
                # Get H3 cell boundary
                polygon_coords = self.get_h3_polygon(row['h3_cell'])
                
                if polygon_coords:
                    folium.Polygon(
                        locations=polygon_coords,
                        color=color,
                        weight=2,
                        opacity=0.8,
                        fillColor=color,
                        fillOpacity=0.3,
                        popup=f"<b>Route {route_id}</b><br>Cell: {row['h3_cell']}<br>Direction: {row['direction_detailed']}<br>Distance: {row['cluster_sp_dist_km']:.2f} km",
                        tooltip=f"Route {route_id} - {row['direction_simplified']}"
                    ).add_to(m)
        
        # Add route polygons with chosen method
        polygon_method = "Vertices" if use_vertices else "Convex Hull"
        route_polygons = self.create_route_polygons(output_df, use_vertices)
        for route_id, polygon_coords in route_polygons.items():
            color = colors[route_id % len(colors)]
            folium.Polygon(
                locations=polygon_coords,
                color=color,
                weight=3,
                opacity=1.0,
                fillColor=color,
                fillOpacity=0.1,
                popup=f"<b>Route {route_id} Coverage Area</b><br>Method: {polygon_method}",
                tooltip=f"Route {route_id} Area ({polygon_method})"
            ).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Delivery Routes Legend</h4>
        <p><i class="fa fa-home" style="color:red"></i> Distribution Center</p>
        <p><i class="fa fa-stop" style="color:blue"></i> H3 Delivery Cells</p>
        <p><b>Polygon Method:</b> {polygon_method}</p>
        <p>Colors represent different routes</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def generate_summary_stats(self, output_df):
        """Generate summary statistics"""
        stats = {
            'total_cells': len(output_df),
            'total_routes': output_df['route_id'].nunique(),
            'avg_cells_per_route': len(output_df) / output_df['route_id'].nunique(),
            'direction_distribution': output_df['direction_simplified'].value_counts().to_dict(),
            'avg_distance_by_route': output_df.groupby('route_id')['cluster_sp_dist_km'].mean().to_dict()
        }
        return stats

# Standalone visualization function
def create_delivery_map(output_df, stock_point_info, use_vertices=False):
    """
    Create folium visualization as standalone function
    
    Args:
        output_df: DataFrame with route assignments
        stock_point_info: Dict with keys 'lat', 'longitude', 'name'
        use_vertices: Boolean, True for vertex tracing, False for convex hull
    """
    
    def get_h3_polygon_coords(h3_cell):
        """Get H3 cell polygon coordinates"""
        try:
            boundary = h3.cell_to_boundary(h3_cell)
            return [[lat, lng] for lat, lng in boundary]
        except:
            return None
    
    def create_route_polygons_standalone(df, use_vertex_trace=False):
        """Create route polygons - standalone version"""
        route_polygons = {}
        
        for route_id in df['route_id'].unique():
            route_cells = df[df['route_id'] == route_id]
            
            if use_vertex_trace:
                # Vertex tracing method
                all_vertices = []
                for _, row in route_cells.iterrows():
                    coords = get_h3_polygon_coords(row['h3_cell'])
                    if coords:
                        all_vertices.extend(coords)
                
                if all_vertices:
                    # Remove duplicates
                    unique_vertices = []
                    seen = set()
                    for vertex in all_vertices:
                        vertex_tuple = tuple(vertex)
                        if vertex_tuple not in seen:
                            unique_vertices.append(vertex)
                            seen.add(vertex_tuple)
                    route_polygons[route_id] = unique_vertices
            else:
                # Convex hull method
                points = []
                for _, row in route_cells.iterrows():
                    coords = get_h3_polygon_coords(row['h3_cell'])
                    if coords:
                        for coord in coords:
                            points.append(Point(coord[1], coord[0]))  # Point(lng, lat)
                
                if points and len(points) >= 3:
                    try:
                        from shapely.geometry import MultiPoint
                        hull = MultiPoint(points).convex_hull
                        if hull.geom_type == 'Polygon' and hasattr(hull, 'exterior'):
                            coords = [[y, x] for x, y in list(hull.exterior.coords)]
                            route_polygons[route_id] = coords
                        else:
                            # Fallback to centroids
                            coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                                     for _, row in route_cells.iterrows()]
                            route_polygons[route_id] = coords
                    except:
                        coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                                 for _, row in route_cells.iterrows()]
                        route_polygons[route_id] = coords
                else:
                    coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                             for _, row in route_cells.iterrows()]
                    route_polygons[route_id] = coords
        
        return route_polygons
    
    # Create map
    m = folium.Map(
        location=[stock_point_info['latitude'], stock_point_info['longitude']],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add stock point
    folium.Marker(
        location=[stock_point_info['latitude'], stock_point_info['longitude']],
        popup=f"<b>{stock_point_info['stock_point_name']}</b><br>Distribution Center",
        tooltip="Distribution Center",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)
    
    # Colors
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
             'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
             'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    # Add H3 cells
    for route_id in output_df['route_id'].unique():
        route_cells = output_df[output_df['route_id'] == route_id]
        color = colors[route_id % len(colors)]
        
        for _, row in route_cells.iterrows():
            coords = get_h3_polygon_coords(row['h3_cell'])
            if coords:
                folium.Polygon(
                    locations=coords,
                    color=color,
                    weight=2,
                    opacity=0.8,
                    fillColor=color,
                    fillOpacity=0.3,
                    # popup=f"<b>Route {route_id}</b><br>Cell: {row['h3_cell']}<br>Direction: {row['direction_detailed']}<br>Distance: {row['cluster_sp_dist_km']:.2f} km",
                    popup=f"<b>Route {route_id}</b><br>Cell: {row['h3_cell']}<br>Direction: {row['']}<br>Distance: {row['cluster_sp_dist_km']:.2f} km",
                    tooltip=f"Route {route_id} - {row['direction_simplified']}"
                ).add_to(m)
    
    # Add route polygons
    polygon_method = "Vertices" if use_vertices else "Convex Hull"
    route_polygons = create_route_polygons_standalone(output_df, use_vertices)
    
    for route_id, coords in route_polygons.items():
        color = colors[route_id % len(colors)]
        folium.Polygon(
            locations=coords,
            color=color,
            weight=3,
            opacity=1.0,
            fillColor=color,
            fillOpacity=0.1,
            popup=f"<b>Route {route_id} Coverage</b><br>Method: {polygon_method}",
            tooltip=f"Route {route_id} Area ({polygon_method})"
        ).add_to(m)
    
    # Legend
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Delivery Routes</h4>
    <p><i class="fa fa-home" style="color:red"></i> Distribution Center</p>
    <p><i class="fa fa-stop" style="color:blue"></i> H3 Cells</p>
    <p><b>Method:</b> {polygon_method}</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Usage example
def process_delivery_data(df_input, cells_per_route=6, use_vertices=False):
    """Main function to process the delivery data"""
    
    # Initialize router
    router = H3DeliveryRouter(df_input, cells_per_route)
    
    # Create routes
    routes = router.create_directional_routes()
    
    # Create output dataframe
    output_df = router.create_output_dataframe()
    
    # Generate summary statistics
    stats = router.generate_summary_stats(output_df)
    
    # Create visualization with chosen polygon method
    map_viz = router.create_visualization(output_df, use_vertices)
    
    return {
        'output_df': output_df,
        'routes': routes,
        'stats': stats,
        'map': map_viz,
        'router': router
    }

# Example usage with your data
"""
# Load your data
df_input = pd.read_csv('your_h3_data.csv')  # or use your existing dataframe

# Option 1: Use convex hull for route polygons (default)
result = process_delivery_data(df_input, cells_per_route=6, use_vertices=False)

# Option 2: Use vertex tracing for route polygons
result = process_delivery_data(df_input, cells_per_route=6, use_vertices=True)

# Access results
output_df = result['output_df']
stats = result['stats']
map_viz = result['map']

# Display results
print("Summary Statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")

print("\nOutput DataFrame:")
print(output_df.head(10))

# Save map
map_viz.save('delivery_routes_map.html')

# Save output
output_df.to_csv('h3_delivery_routes.csv', index=False)
"""