import folium
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon # Ensure MultiPolygon is imported
import h3 # This needs to be the h3 module that is loaded
from typing import List, Set # Already there, just for completeness
import folium
import json
from src.get_data import get_geojson_data
import folium
import h3
from shapely.geometry import Polygon, MultiPolygon
from folium.features import GeoJsonTooltip, GeoJsonPopup
from folium.plugins import MeasureControl
import pandas as pd
import altair as alt
import folium
import h3
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize
from folium.features import GeoJsonTooltip, GeoJsonPopup
from folium.plugins import MeasureControl
import folium
from folium.plugins import MeasureControl
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
import h3
import matplotlib.pyplot as plt



diverging_colormaps = [
    'coolwarm',
    'bwr',
    'seismic',
    'PiYG',
    'PRGn',
    'BrBG',
    'PuOr',
    'RdBu',
    'RdGy',
    'RdYlBu',
    'RdYlGn',
    'Spectral'
]


sequential_colormaps = [
    'viridis',
    'plasma',
    'inferno',
    'magma',
    'cividis',
    'Greys',
    'Purples',
    'Blues',
    'Greens',
    'Oranges',
    'Reds',
    'YlGnBu',
    'YlOrRd',
    'BuGn',
    'PuBuGn'
]



qualitative_colormaps = [
    'tab10',
    'tab20',
    'tab20b',
    'tab20c',
    'Set1',
    'Set2',
    'Set3',
    'Pastel1',
    'Pastel2',
    'Accent',
    'Dark2',
    'Paired'
]


def plot_territory(territory_polygons: List[Polygon], stock_point_id: int):
    """
    Plots the territory polygons for a given stock point.
    
    Args:
        territory_polygons (List[Polygon]): List of Shapely Polygon objects defining the territory.
        resolution (int): H3 resolution used for the territory.
    """
    import matplotlib.pyplot as plt
    # Assuming territory_polygons is your List[Polygon]
    territory_gdf_to_plot = gpd.GeoDataFrame(geometry=territory_polygons, crs="EPSG:4326")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    territory_gdf_to_plot.plot(ax=ax, color='blue', edgecolor='black', alpha=0.5)
    ax.set_title(f"Territory for Stock Point {stock_point_id}")
    plt.axis('off') # Optional: removes axis ticks and labels for cleaner map
    plt.show()
    
# plot_territory(territory_polygons, stock_point_id=1647113)


def plot_territory_and_h3_overlay(
    territory_polygons: List[Polygon],
    territory_cells: Set[str],
    title: str = "Territory with H3 Grid Overlay",
    h3_resolution: int = None
) -> folium.Map:
    """
    Plots territory polygons and overlays H3 cells on an interactive Folium map.
    This version manually constructs H3 cell polygons due to a specific H3-py module issue.

    Args:
        territory_polygons (List[Polygon]): A list of Shapely Polygon objects defining the territory.
        territory_cells (Set[str]): A set of H3 cell IDs covering the territory.
        title (str): The title for the map.
        h3_resolution (int, optional): The H3 resolution used, for display in title/legend.

    Returns:
        folium.Map: An interactive Folium map object.
    """
    if not territory_polygons:
        print("No territory polygons provided for plotting.")
        return folium.Map(location=[6.5244, 3.3792], zoom_start=9) # Default to Lagos center

    # 1. Convert territory polygons to GeoDataFrame for easier plotting
    territory_gdf = gpd.GeoDataFrame(geometry=territory_polygons, crs="EPSG:4326")

    # 2. Determine map center (approximate centroid of the territory)
    united_territory = territory_gdf.union_all()
    # united_territory = territory_gdf.unary_union
    
    if united_territory.is_empty or not united_territory.is_valid:
        map_center = [6.5244, 3.3792]
        zoom_start = 9
    else:
        map_center = [united_territory.centroid.y, united_territory.centroid.x]
        min_lon, min_lat, max_lon, max_lat = united_territory.bounds
        if (max_lon - min_lon) < 0.1 and (max_lat - min_lat) < 0.1:
            zoom_start = 13
        elif (max_lon - min_lon) < 0.5 and (max_lat - min_lat) < 0.5:
            zoom_start = 12
        else:
            zoom_start = 10

    # 3. Initialize Folium Map
    m = folium.Map(location=map_center, zoom_start=zoom_start, control_scale=True)

    # Add a title
    title_html = f'<h3 align="center" style="font-size:20px"><b>{title}</b></h3>'
    if h3_resolution:
        title_html = f'<h3 align="center" style="font-size:20px"><b>{title} (H3 Res: {h3_resolution})</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    # 4. Add Territory Polygons to the map
    folium.GeoJson(
        territory_gdf.to_json(),
        name="Territory Boundaries",
        style_function=lambda x: {
            "fillColor": "#1a75ff",
            "color": "black",
            "weight": 2,
            "fillOpacity": 0.2,
        },
    ).add_to(m)

    # 5. Manually construct GeoJSON for H3 Grid Cells using available functions
    h3_polygons = []
    for cell in territory_cells:
        # h3.cell_to_boundary returns a list of (lat, lng) tuples
        boundary_lat_lng = h3.cell_to_boundary(cell)
        # Shapely.Polygon expects coordinates in (lng, lat) order
        boundary_lng_lat = [(lng, lat) for lat, lng in boundary_lat_lng]
        
        # Create a Shapely Polygon from the boundary
        h3_polygons.append(Polygon(boundary_lng_lat))

    # Convert the list of Shapely Polygons into a GeoDataFrame, then to GeoJSON
    if h3_polygons: # Ensure there are polygons before creating GeoDataFrame
        h3_cells_gdf = gpd.GeoDataFrame(geometry=h3_polygons, crs="EPSG:4326")
        h3_geojson_data = h3_cells_gdf.to_json()
    else:
        h3_geojson_data = {"type": "FeatureCollection", "features": []} # Empty GeoJSON

    folium.GeoJson(
        h3_geojson_data,
        name="H3 Grid Cells",
        style_function=lambda x: {
            "fillColor": "red",
            "color": "red",
            "weight": 1,
            "fillOpacity": 0.1,
        },
    ).add_to(m)
    
    # 6. Add Layer Control
    folium.LayerControl().add_to(m)

    return m

def plot_geojson_territory_heatmap(geojson_path, output_path = None, use_blue=False):
    def get_discrete_color(customer_count, use_blue=use_blue):
        """
        Returns a hexadecimal color code based on the customer count,
        using a high-contrast 6-level discrete color scale.
        """
        # Alternative blue-purple palette
        if use_blue:
            if customer_count == 0: return "#CCCCCC"      # Grey
            elif customer_count <= 10: return '#C6DBEF'   # Pale blue
            elif customer_count <= 25: return '#9ECAE1'   # Light blue
            elif customer_count <= 50: return '#6BAED6'   # Blue
            elif customer_count <= 100: return "#3182BD"  # Dark blue
            else: return '#08519C'                       # Navy
        else:
            if customer_count == 0:
                return "#CCCCCC"      # Empty - neutral grey
            elif customer_count <= 10:
                return '#FFEDA0'      # Very low - pale yellow
            elif customer_count <= 25:
                return '#FEB24C'      # Low - orange
            elif customer_count <= 50:
                return '#FD8D3C'      # Medium - coral
            elif customer_count <= 100:
                return "#E31A1C"      # High - red
            else:
                return '#800026'      # Very high - dark red

        
    geojson_data = get_geojson_data(PATH = geojson_path)
    # Create base map
    m = folium.Map(location=[9.0765, 7.3986], zoom_start=7)

    # Add GeoJSON layer
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': get_discrete_color(feature['properties']['customer_count']),
            # 'fillColor': 'red' if feature['properties']['customer_count'] > 50 else ('grey' if feature['properties']['customer_count'] == 0 else 'blue'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,
        },
        popup=folium.GeoJsonPopup(fields=['stock_point_id', 'customer_count', 'h3_cell_id'])
    ).add_to(m)

    if output_path:
        m.save(output_path)
    
    return m




def create_route_map_dep0(df, output_file=None):
    """
    Creates a folium map with route polygons and individual cell polygons, including a measuring ruler.

    Parameters:
    - df: pandas DataFrame containing route information.
    - output_file: path to save the output HTML file.
    """
    # Initialize a folium map centered around some location
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Add the measuring ruler control
    MeasureControl().add_to(m)

    # Iterate through each route in the DataFrame
    for idx, row in df.iterrows():
        polygons = []
        for cell_id in row['h3_cell_ids']:
            # Get the boundary of the H3 cell and reverse lat, long
            boundary = h3.cell_to_boundary(cell_id)
            corrected_boundary = [(coord[1], coord[0]) for coord in boundary]
            polygon = Polygon(corrected_boundary)
            polygons.append(polygon)

        # Combine polygons to form the route boundary
        route_polygon = MultiPolygon(polygons).convex_hull

        # Create a GeoJSON-like structure for the route polygon
        route_geojson_data = {
            "type": "Feature",
            "geometry": route_polygon.__geo_interface__,
            "properties": {
                'route_id': row['route_id'],
                'cluster_count': row['cluster_count'],
                'customer_count': row['customer_count'],
                'cumulative_distance_km': row['cumulative_distance_km'],
                'avg_assignment_confidence': row['avg_assignment_confidence']
            }
        }

        route_geojson = folium.GeoJson(
            route_geojson_data,
            name=row['route_id'],
            style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 1},
            tooltip=folium.GeoJsonTooltip(
                fields=['route_id', 'cluster_count', 'customer_count'],
                aliases=['Route ID', 'Cluster Count', 'Customer Count'],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['route_id', 'cumulative_distance_km', 'avg_assignment_confidence'],
                aliases=['Route ID', 'Distance (km)', 'Confidence']
            )
        ).add_to(m)

        # Add each cell polygon to the map
        for polygon in polygons:
            cell_geojson_data = {
                "type": "Feature",
                "geometry": polygon.__geo_interface__,
                "properties": {
                    'route_id': row['route_id'],
                    'cluster_count': row['cluster_count'],
                    'customer_count': row['customer_count']
                }
            }
            cell_geojson = folium.GeoJson(
                cell_geojson_data,
                style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 1},
                tooltip=folium.GeoJsonTooltip(
                    fields=['route_id', 'cluster_count'],
                    aliases=['Route ID', 'Cluster Count'],
                    localize=True
                ),
                popup=folium.GeoJsonPopup(
                    fields=['route_id', 'customer_count'],
                    aliases=['Route ID', 'Customer Count']
                )
            ).add_to(m)

    # Adjust the map to fit all polygons and save it
    m.fit_bounds(m.get_bounds(), padding=(10, 10))
    if output_file:
        m.save(output_file)
        print(f"Map has been saved to {output_file}")
    
    return m
# Example usage:
# Assuming `df` is your pre-defined DataFrame
# create_route_map(df, 'test_route_map.html')
 
 
def create_route_map_dep1(df, output_file=None):
    """
    Creates a folium map with route polygons using H3 cell vertices and individual cell polygons, including a measuring ruler.

    Parameters:
    - df: pandas DataFrame containing route information.
    - output_file: path to save the output HTML file.
    """
    # Initialize a folium map centered around some location
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Add the measuring ruler control
    MeasureControl().add_to(m)

    # Iterate through each route in the DataFrame
    for idx, row in df.iterrows():
        multipolygon_geoms = []

        for cell_id in row['h3_cell_ids']:
            # Get the boundary of the H3 cell and reverse lat, long
            boundary = h3.cell_to_boundary(cell_id)
            corrected_boundary = [(coord[1], coord[0]) for coord in boundary]
            polygon = Polygon(corrected_boundary)
            multipolygon_geoms.append(polygon)

        # Create a MultiPolygon from the list of polygons
        multipolygon = MultiPolygon(multipolygon_geoms)

        # Calculate the union of all polygons to get a combined shape
        unioned_polygons = unary_union(multipolygon)

        # Polygonize the result to get the outer boundary
        if unioned_polygons.geom_type == 'Polygon':
            route_polygon = unioned_polygons
        else:
            # If the union results in multiple polygons, take the convex hull for simplicity
            route_polygon = MultiPolygon([polygon for polygon in polygonize(unioned_polygons)]).convex_hull

        # Create a GeoJSON-like structure for the route polygon
        route_geojson_data = {
            "type": "Feature",
            "geometry": route_polygon.__geo_interface__,
            "properties": {
                'route_id': row['route_id'],
                'cluster_count': row['cluster_count'],
                'customer_count': row['customer_count'],
                'cumulative_distance_km': row['cumulative_distance_km'],
                'avg_assignment_confidence': row['avg_assignment_confidence']
            }
        }

        

        # Also add each individual cell polygon to the map
        for polygon in multipolygon_geoms:
            cell_geojson_data = {
                "type": "Feature",
                "geometry": polygon.__geo_interface__,
                "properties": {
                    'route_id': row['route_id'],
                    'cluster_count': row['cluster_count'],
                    'customer_count': row['customer_count']
                }
            }
            cell_geojson = folium.GeoJson(
                cell_geojson_data,
                style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 1},
                tooltip=folium.GeoJsonTooltip(
                    fields=['route_id', 'cluster_count'],
                    aliases=['Route ID', 'Cluster Count'],
                    localize=True
                ),
                popup=folium.GeoJsonPopup(
                    fields=['route_id', 'customer_count'],
                    aliases=['Route ID', 'Customer Count']
                )
            ).add_to(m)
            
        route_geojson = folium.GeoJson(
            route_geojson_data,
            name=row['route_id'],
            style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 2},
            tooltip=folium.GeoJsonTooltip(
                fields=['route_id', 'cluster_count', 'customer_count'],
                aliases=['Route ID', 'Cluster Count', 'Customer Count'],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['route_id', 'cumulative_distance_km', 'avg_assignment_confidence'],
                aliases=['Route ID', 'Distance (km)', 'Confidence']
            )
        ).add_to(m)

    # Adjust the map to fit all polygons and save it
    m.fit_bounds(m.get_bounds(), padding=(10, 10))
    if output_file:
        m.save(output_file)
        print(f"Map has been saved to {output_file}")
    
    return m 



def generate_distinct_colors(n, col_indx = 5):
    """Generate `n` distinct colors from matplotlib colormap."""
    cmap = plt.cm.get_cmap(qualitative_colormaps[col_indx], n)
    return [f'#{int(255*r):02x}{int(255*g):02x}{int(255*b):02x}' for r, g, b, _ in cmap(range(n))]


def create_route_map(df, fc_coordinates = None, output_file=None, col_indx = 5):
    """
    Creates a folium map with route polygons from H3 cells, each with a different color.
    
    Parameters:
    - df: DataFrame with columns:
        'route_id', 'h3_cell_ids', 'cluster_count', 'customer_count',
        'cumulative_distance_km', 'avg_assignment_confidence'
    - output_file: Optional path to save the HTML map
    """
    # Centered on Nigeria by default
    m = folium.Map(location=[9.0820, 8.6753], zoom_start=6)
    MeasureControl().add_to(m)

    # Add fulfillment center
    if fc_coordinates:
        folium.Marker(
            location=fc_coordinates,
            popup='Fulfillment Center',
            icon=folium.Icon(color='red', icon='home')
        ).add_to(m)
    
    route_colors = generate_distinct_colors(len(df),col_indx)

    for idx, row in df.iterrows():
        multipolygon_geoms = []

        for cell_id in row['h3_cell_ids']:
            boundary = h3.cell_to_boundary(cell_id)  # Returns (lat, lon)
            boundary_lonlat = [(lng, lat) for lat, lng in boundary]
            polygon = orient(Polygon(boundary_lonlat), sign=1.0)  # Ensure clockwise
            multipolygon_geoms.append(polygon)

        # Merge H3 polygons into one unified shape
        merged = unary_union(multipolygon_geoms)
        if merged.geom_type == 'Polygon':
            route_polygon = merged
        elif merged.geom_type == 'MultiPolygon':
            route_polygon = max(merged.geoms, key=lambda p: p.area)
        else:
            continue  # Skip invalid geometry

        route_color = route_colors[idx]

        # Add each individual H3 cell
        for cell_polygon in multipolygon_geoms:
            folium.GeoJson(
                data={
                    "type": "Feature",
                    "geometry": cell_polygon.__geo_interface__,
                    "properties": {
                        'route_id': row['route_id'],
                        'cluster_count': row['cluster_count'],
                        'customer_count': row['customer_count']
                    }
                },
                style_function=lambda x, color=route_color: {
                    'fillColor': color,
                    'color': color,
                    'weight': 1,
                    'fillOpacity': 0.15
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['route_id', 'cluster_count'],
                    aliases=['Route ID', 'Cluster Count']
                ),
                popup=folium.GeoJsonPopup(
                    fields=['route_id', 'customer_count'],
                    aliases=['Route ID', 'Customer Count']
                )
            ).add_to(m)
            
            
        # Add merged route polygon
        route_geojson = folium.GeoJson(
            data={
                "type": "Feature",
                "geometry": route_polygon.__geo_interface__,
                "properties": {
                    'route_id': row['route_id'],
                    'cluster_count': row['cluster_count'],
                    'customer_count': row['customer_count'],
                    'cumulative_distance_km': row['cumulative_distance_km'],
                    'avg_assignment_confidence': row['avg_assignment_confidence']
                }
            },
            name=f"Route {row['route_id']}",
            style_function=lambda x, color=route_color: {
                'fillColor': color,
                'color': color,
                'weight': 2,
                'fillOpacity': 0.4
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['route_id', 'cluster_count', 'customer_count'],
                aliases=['Route ID', 'Cluster Count', 'Customer Count'],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['route_id', 'cumulative_distance_km', 'avg_assignment_confidence'],
                aliases=['Route ID', 'Distance (km)', 'Confidence']
            )
        )
        route_geojson.add_to(m)

        

    m.fit_bounds(m.get_bounds(), padding=(10, 10))

    if output_file:
        m.save(output_file)
        print(f"Map saved to {output_file}")

    return m


# Example usage:
# Assuming `df` is your pre-defined DataFrame
# create_route_map(df, 'test_route_map.html')


# -------------------------------
# ALTAIR PLOTS 

def create_route_summary_barplot(optimized_output_df, width = 500, height = 350):
    cluster_count_chart = alt.Chart(optimized_output_df).mark_bar().encode(
        x='route_id',
        y='customer_count',
        color='route_id',
        tooltip=['customer_count', 
                 "cluster_count", 
                 'cumulative_distance_km', 
                 'farthest_centroid_distance_km']
    ).properties(
        title='Distribution Number Customers by Route',
        width=width,
        height=height 
    )
    
    return cluster_count_chart



























































































