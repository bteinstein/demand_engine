import pandas as pd
import numpy as np
import h3
import folium
from folium import plugins
import math
from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon, MultiPolygon
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
    
    def create_route_polygons_alpha_shape(self, output_df):
        """Create alpha shape polygons for better route boundary tracing"""
        try:
            from shapely.geometry import MultiPoint
            from shapely.ops import triangulate
        except ImportError:
            print("Warning: Shapely not available for advanced polygons. Using convex hull fallback.")
            return self.create_route_polygons(output_df, use_vertices=False)
        
        route_polygons = {}
        
        for route_id in output_df['route_id'].unique():
            route_cells = output_df[output_df['route_id'] == route_id]
            
            # Collect all vertices from H3 cells in the route
            all_points = []
            for _, row in route_cells.iterrows():
                polygon_coords = self.get_h3_polygon(row['h3_cell'])
                if polygon_coords:
                    for coord in polygon_coords:
                        all_points.append(Point(coord[1], coord[0]))  # Point expects (lng, lat)
            
            if len(all_points) >= 3:
                try:
                    # Create alpha shape using triangulation
                    multi_point = MultiPoint(all_points)
                    
                    # For small clusters, use convex hull
                    if len(all_points) <= 10:
                        hull = multi_point.convex_hull
                    else:
                        # For larger clusters, create a more refined boundary
                        # by using the union of all H3 cells
                        h3_polygons = []
                        for _, row in route_cells.iterrows():
                            polygon_coords = self.get_h3_polygon(row['h3_cell'])
                            if polygon_coords:
                                # Convert to Shapely polygon (lng, lat format)
                                shapely_coords = [(coord[1], coord[0]) for coord in polygon_coords]
                                if len(shapely_coords) >= 3:
                                    h3_polygons.append(Polygon(shapely_coords))
                        
                        if h3_polygons:
                            # Union all H3 polygons to get route boundary
                            union_polygon = unary_union(h3_polygons)
                            
                            # Handle MultiPolygon case
                            if isinstance(union_polygon, MultiPolygon):
                                # Take the largest polygon
                                hull = max(union_polygon.geoms, key=lambda p: p.area)
                            else:
                                hull = union_polygon
                        else:
                            hull = multi_point.convex_hull
                    
                    # Extract coordinates
                    if hasattr(hull, 'exterior'):
                        coords = [[y, x] for x, y in list(hull.exterior.coords)]
                        route_polygons[route_id] = coords
                    else:
                        # Fallback to centroids
                        coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                                 for _, row in route_cells.iterrows()]
                        route_polygons[route_id] = coords
                        
                except Exception as e:
                    print(f"Warning: Alpha shape failed for route {route_id}: {e}")
                    # Fallback to centroids
                    coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                             for _, row in route_cells.iterrows()]
                    route_polygons[route_id] = coords
            else:
                coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                         for _, row in route_cells.iterrows()]
                route_polygons[route_id] = coords
        
        return route_polygons
    
    def create_route_polygons(self, output_df, use_vertices=False):
        """Create polygons for each route - enhanced version"""
        if use_vertices:
            return self.create_route_polygons_alpha_shape(output_df)
        
        route_polygons = {}
        
        for route_id in output_df['route_id'].unique():
            route_cells = output_df[output_df['route_id'] == route_id]
            
            # Collect all H3 cell boundaries
            all_h3_polygons = []
            for _, row in route_cells.iterrows():
                polygon_coords = self.get_h3_polygon(row['h3_cell'])
                if polygon_coords:
                    # Convert to Shapely polygon (lng, lat format)
                    shapely_coords = [(coord[1], coord[0]) for coord in polygon_coords]
                    if len(shapely_coords) >= 3:
                        all_h3_polygons.append(Polygon(shapely_coords))
            
            if all_h3_polygons:
                try:
                    # Method 1: Union all H3 polygons for exact boundary
                    union_polygon = unary_union(all_h3_polygons)
                    
                    # Handle MultiPolygon case
                    if isinstance(union_polygon, MultiPolygon):
                        # Take the largest polygon or create convex hull of all
                        largest_poly = max(union_polygon.geoms, key=lambda p: p.area)
                        # Convert to coordinates
                        coords = [[y, x] for x, y in list(largest_poly.exterior.coords)]
                        route_polygons[route_id] = coords
                    else:
                        coords = [[y, x] for x, y in list(union_polygon.exterior.coords)]
                        route_polygons[route_id] = coords
                        
                except Exception as e:
                    print(f"Warning: Union operation failed for route {route_id}: {e}")
                    # Fallback to convex hull of centroids
                    coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                             for _, row in route_cells.iterrows()]
                    route_polygons[route_id] = coords
            else:
                # Fallback to centroids
                coords = [[row['cluster_centroid_lat'], row['cluster_centroid_lng']] 
                         for _, row in route_cells.iterrows()]
                route_polygons[route_id] = coords
        
        return route_polygons
    
    def create_visualization(self, output_df, use_vertices=True, show_individual_cells=True):
        """Enhanced visualization with better styling and options"""
        # Center map on stock point
        m = folium.Map(
            location=[self.stock_point['lat'], self.stock_point['lng']],
            zoom_start=12,
            tiles='CartoDB positron'  # Cleaner basemap
        )
        
        # Add stock point marker
        folium.Marker(
            location=[self.stock_point['lat'], self.stock_point['lng']],
            popup=f"<b>{self.stock_point['name']}</b><br>Distribution Center",
            tooltip="Distribution Center",
            icon=folium.Icon(color='red', icon='home', prefix='fa')
        ).add_to(m)
        
        # Enhanced color palette with better contrast
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                 '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7DBDD']
        
        # Track plotted cells to avoid duplicates
        plotted_cells = set()
        
        # Add individual H3 cells if requested (only from output_df, no original cells)
        if show_individual_cells:
            for _, row in output_df.iterrows():
                # Skip if cell already plotted
                if row['h3_cell'] in plotted_cells:
                    continue
                    
                color = colors[row['route_id'] % len(colors)]
                polygon_coords = self.get_h3_polygon(row['h3_cell'])
                
                if polygon_coords:
                    folium.Polygon(
                        locations=polygon_coords,
                        color=color,
                        weight=1,
                        opacity=0.7,
                        fillColor=color,
                        fillOpacity=0.4,
                        popup=f"""<b>Route {row['route_id']}</b><br>
                                Cell: {row['h3_cell']}<br>
                                Direction: {row['direction_detailed']}<br>
                                Distance: {row['cluster_sp_dist_km']:.2f} km<br>
                                Bearing: {row['bearing']:.1f}°""",
                        tooltip=f"Route {row['route_id']} - {row['direction_simplified']}"
                    ).add_to(m)
                    
                    plotted_cells.add(row['h3_cell'])
        
        # Add route boundary polygons
        polygon_method = "H3 Union" if use_vertices else "Convex Hull"
        route_polygons = self.create_route_polygons(output_df, use_vertices)
        
        for route_id, polygon_coords in route_polygons.items():
            color = colors[route_id % len(colors)]
            
            # Route boundary
            folium.Polygon(
                locations=polygon_coords,
                color='#2C3E50',  # Dark border for visibility
                weight=3,
                opacity=0.9,
                fillColor=color,
                fillOpacity=0.1,
                popup=f"""<b>Route {route_id} Coverage Area</b><br>
                        Method: {polygon_method}<br>
                        Cells: {len(output_df[output_df['route_id'] == route_id])}<br>
                        Direction: {output_df[output_df['route_id'] == route_id]['direction_simplified'].iloc[0]}""",
                tooltip=f"Route {route_id} Boundary"
            ).add_to(m)
            
            # Add route label at centroid
            route_data = output_df[output_df['route_id'] == route_id]
            centroid_lat = route_data['cluster_centroid_lat'].mean()
            centroid_lng = route_data['cluster_centroid_lng'].mean()
            
            folium.Marker(
                location=[centroid_lat, centroid_lng],
                popup=f"Route {route_id}",
                tooltip=f"Route {route_id}",
                icon=folium.DivIcon(
                    html=f"""<div style="
                        background-color: {color};
                        color: white;
                        text-align: center;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        line-height: 30px;
                        font-weight: bold;
                        font-size: 12px;
                        border: 2px solid white;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    ">{route_id}</div>""",
                    icon_size=(30, 30),
                    icon_anchor=(15, 15)
                )
            ).add_to(m)
        
        # Enhanced legend
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 250px; height: auto; 
                    background-color: white; border: 2px solid #2C3E50; 
                    border-radius: 8px; z-index: 9999; 
                    font-size: 12px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h4 style="margin-top: 0; color: #2C3E50;">🚛 Delivery Routes</h4>
        <p><span style="color: red;">🏠</span> Distribution Center</p>
        <p><span style="color: #4ECDC4;">⬢</span> H3 Delivery Cells</p>
        <p><span style="color: #2C3E50;">━━</span> Route Boundaries</p>
        <p><b>Boundary Method:</b> {polygon_method}</p>
        <p><b>Total Routes:</b> {output_df['route_id'].nunique()}</p>
        <p><b>Total Cells:</b> {len(output_df)}</p>
        <p style="font-size: 10px; color: #7F8C8D; margin-bottom: 0;">
        Each color represents a different delivery route
        </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add route statistics as a separate info box
        stats_html = '''
        <div style="position: fixed; 
                    bottom: 10px; right: 10px; width: 200px; height: auto; 
                    background-color: rgba(255,255,255,0.9); border: 1px solid #BDC3C7; 
                    border-radius: 5px; z-index: 9999; 
                    font-size: 11px; padding: 10px;">
        <h5 style="margin-top: 0; color: #2C3E50;">📊 Route Statistics</h5>
        '''
        
        for route_id in sorted(output_df['route_id'].unique()):
            route_data = output_df[output_df['route_id'] == route_id]
            direction = route_data['direction_simplified'].iloc[0]
            cell_count = len(route_data)
            avg_distance = route_data['cluster_sp_dist_km'].mean()
            color = colors[route_id % len(colors)]
            
            stats_html += f'''
            <p style="margin: 3px 0;">
                <span style="color: {color}; font-weight: bold;">Route {route_id}</span>: 
                {cell_count} cells, {direction}, {avg_distance:.1f}km avg
            </p>
            '''
        
        stats_html += '</div>'
        m.get_root().html.add_child(folium.Element(stats_html))
        
        return m
    
    def generate_summary_stats(self, output_df):
        """Generate comprehensive summary statistics"""
        stats = {
            'total_cells': len(output_df),
            'total_routes': output_df['route_id'].nunique(),
            'avg_cells_per_route': len(output_df) / output_df['route_id'].nunique(),
            'direction_distribution': output_df['direction_simplified'].value_counts().to_dict(),
            'avg_distance_by_route': output_df.groupby('route_id')['cluster_sp_dist_km'].mean().to_dict(),
            'distance_statistics': {
                'min_distance': output_df['cluster_sp_dist_km'].min(),
                'max_distance': output_df['cluster_sp_dist_km'].max(),
                'mean_distance': output_df['cluster_sp_dist_km'].mean(),
                'std_distance': output_df['cluster_sp_dist_km'].std()
            },
            'route_sizes': output_df.groupby('route_id').size().to_dict()
        }
        return stats
    
    def plot_analytics(self, output_df):
        """Create analytical plots for route optimization insights"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('H3 Delivery Route Analytics', fontsize=16, fontweight='bold')
        
        # Plot 1: Route size distribution
        route_sizes = output_df.groupby('route_id').size()
        axes[0,0].bar(route_sizes.index, route_sizes.values, 
                     color='skyblue', alpha=0.7, edgecolor='navy')
        axes[0,0].set_title('Cells per Route')
        axes[0,0].set_xlabel('Route ID')
        axes[0,0].set_ylabel('Number of Cells')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Direction distribution
        direction_counts = output_df['direction_simplified'].value_counts()
        axes[0,1].pie(direction_counts.values, labels=direction_counts.index, 
                     autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Distribution by Direction')
        
        # Plot 3: Distance distribution by route
        route_distances = output_df.groupby('route_id')['cluster_sp_dist_km'].mean()
        bars = axes[1,0].bar(route_distances.index, route_distances.values, 
                            color='lightcoral', alpha=0.7, edgecolor='darkred')
        axes[1,0].set_title('Average Distance per Route')
        axes[1,0].set_xlabel('Route ID')
        axes[1,0].set_ylabel('Average Distance (km)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Distance vs Route scatter with direction coloring
        directions = output_df['direction_simplified'].unique()
        colors_scatter = plt.cm.Set3(np.linspace(0, 1, len(directions)))
        
        for i, direction in enumerate(directions):
            data = output_df[output_df['direction_simplified'] == direction]
            axes[1,1].scatter(data['route_id'], data['cluster_sp_dist_km'], 
                            c=[colors_scatter[i]], label=direction, alpha=0.7, s=50)
        
        axes[1,1].set_title('Distance Distribution by Route and Direction')
        axes[1,1].set_xlabel('Route ID')
        axes[1,1].set_ylabel('Distance from Stock Point (km)')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Enhanced usage function
def process_delivery_data(df_input, cells_per_route=6, use_vertices=True, show_analytics=False):
    """Enhanced main function to process the delivery data"""
    
    # Initialize router
    router = H3DeliveryRouter(df_input, cells_per_route)
    
    # Create routes
    routes = router.create_directional_routes()
    
    # Create output dataframe
    output_df = router.create_output_dataframe()
    
    # Generate summary statistics
    stats = router.generate_summary_stats(output_df)
    
    # Create visualization with enhanced features
    map_viz = router.create_visualization(output_df, use_vertices, show_individual_cells=True)
    
    result = {
        'output_df': output_df,
        'routes': routes,
        'stats': stats,
        'map': map_viz,
        'router': router
    }
    
    # Add analytics if requested
    if show_analytics:
        analytics_plot = router.plot_analytics(output_df)
        result['analytics'] = analytics_plot
    
    return result