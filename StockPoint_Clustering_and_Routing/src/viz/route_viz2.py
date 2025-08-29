import folium
import h3
import pandas as pd
import numpy as np
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


def create_route_polygons_alpha_shape(output_df):
    """Create alpha shape polygons for better route boundary tracing"""
    try:
        from shapely.geometry import MultiPoint
        from shapely.ops import triangulate
    except ImportError:
        print("Warning: Shapely not available for advanced polygons. Using convex hull fallback.")
        return create_route_polygons(output_df, use_vertices=False)
    
    route_polygons = {}
    
    for route_id in output_df['route_id'].unique():
        route_cells = output_df[output_df['route_id'] == route_id]
        
        # Collect all vertices from H3 cells in the route
        all_points = []
        for _, row in route_cells.iterrows():
            polygon_coords = get_h3_polygon(row['h3_cell'])
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
                        polygon_coords = get_h3_polygon(row['h3_cell'])
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

def create_route_polygons(output_df, use_vertices=False):
    """Create polygons for each route - enhanced version"""
    if use_vertices:
        return create_route_polygons_alpha_shape(output_df)
    
    route_polygons = {}
    
    for route_id in output_df['route_id'].unique():
        route_cells = output_df[output_df['route_id'] == route_id]
        
        # Collect all H3 cell boundaries
        all_h3_polygons = []
        for _, row in route_cells.iterrows():
            polygon_coords = get_h3_polygon(row['h3_cell'])
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

    
      
      
def get_h3_polygon(h3_cell):
        """Get polygon coordinates for H3 cell - folium expects [lat, lng] format"""
        try:
            # h3.cell_to_boundary returns [(lat, lng), (lat, lng), ...] tuples
            boundary = h3.cell_to_boundary(h3_cell)
            # Convert tuples to list format that folium expects: [[lat, lng], [lat, lng], ...]
            coords = [[lat, lng] for lat, lng in boundary]
            return coords
        except:
            return None

    
def create_visualization(output_df, use_vertices=True, show_individual_cells=True):
        """Enhanced visualization with better styling and options"""
        # Center map on stock point
        df_sp = output_df.iloc[:1,]
        sp_lat, sp_lng, sp_name = df_sp['sp_latitude'].values[0], df_sp['sp_longitude'].values[0], df_sp['stock_point_name'].values[0]
        
        m = folium.Map(
            location=[sp_lat, sp_lng],
            zoom_start=12,
            tiles='CartoDB positron'  # Cleaner basemap
        )
        
        # Add stock point marker
        folium.Marker(
            location=[sp_lat, sp_lng],
            popup=f"<b>{sp_name}</b><br>Distribution Center",
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
                polygon_coords = get_h3_polygon(row['h3_cell'])
                
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
                                Direction: {row['direction']}<br>
                                Distance: {row['cluster_sp_dist_km']:.2f} km<br>
                                Bearing: {row['bearing']:.1f}°""",
                        tooltip=f"Route {row['route_id']} - {row['direction_simplified']}"
                    ).add_to(m)
                    
                    plotted_cells.add(row['h3_cell'])
        
        # Add route boundary polygons
        polygon_method = "H3 Union" if use_vertices else "Convex Hull"
        route_polygons = create_route_polygons(output_df, use_vertices)
        
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
    