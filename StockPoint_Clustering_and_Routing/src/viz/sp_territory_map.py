
import folium
import geopandas as gpd
import pandas as pd

from shapely.geometry import MultiPolygon, Polygon 

import folium
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

def plot_territory_polygon_exp(territory_data, stockpoint_id):
    """
    Visualize a territory polygon using Folium interactive map.
    
    Parameters:
    -----------
    territory_data : dict
        Dictionary containing territory information with 'polygon' key
        and optionally 'sub_territories' for non-contiguous territories
    stockpoint_id : str or int
        ID of the stockpoint for display purposes
    
    Returns:
    --------
    folium.Map
        Interactive map displaying the territory polygon(s)
    """
    # Convert the territory data to GeoDataFrame
    main_polygon = territory_data['polygon']
    sub_polygons = territory_data.get('sub_territories', [])
    
    # Create a GeoDataFrame for the main polygon
    gdf_main = gpd.GeoDataFrame(
        {'geometry': [main_polygon], 'type': ['main_territory']},
        crs="EPSG:4326"
    )
    
    # Create a GeoDataFrame for sub-polygons if they exist
    if sub_polygons:
        gdf_subs = gpd.GeoDataFrame(
            {'geometry': sub_polygons, 'type': ['sub_territory']*len(sub_polygons)},
            crs="EPSG:4326"
        )
        gdf = gpd.GeoDataFrame(pd.concat([gdf_main, gdf_subs], ignore_index=True))
    else:
        gdf = gdf_main
    
    # Calculate centroid for setting map view
    centroid = gdf.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11)
    
    # Add main territory polygon
    folium.GeoJson(
        gdf_main,
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 2,
            'fillOpacity': 0.4
        },
        tooltip=f"Stockpoint {stockpoint_id} Main Territory"
    ).add_to(m)
    
    # Add sub-territories if they exist
    if sub_polygons:
        folium.GeoJson(
            gdf_subs,
            style_function=lambda x: {
                'fillColor': 'orange',
                'color': 'orange',
                'weight': 2,
                'fillOpacity': 0.4
            },
            tooltip=f"Stockpoint {stockpoint_id} Sub-Territory"
        ).add_to(m)
    
    # Add information to the map
    title = f"Territory for Stockpoint {stockpoint_id}"
    info = f"""
    <h4>{title}</h4>
    <b>LGA Count:</b> {territory_data.get('lga_count', 'N/A')}<br>
    <b>Total Area:</b> {territory_data.get('total_area_km2', 'N/A')} km²<br>
    <b>Contiguous:</b> {territory_data.get('is_contiguous', 'N/A')}<br>
    <b>Version:</b> {territory_data.get('territory_version', 'N/A')}
    """
    folium.map.Marker(
        [centroid.y, centroid.x],
        icon=folium.DivIcon(html='<div style="font-size: 8pt">⬤</div>'),
        popup=folium.Popup(info, max_width=300)
    ).add_to(m)
    
    return m

# Example usage:
# territory_data = territories['1647380']
# map = plot_territory_polygon(territory_data, '1647380')
# map.save('territory_1647380.html')  # To save as HTML file
# map  # To display in Jupyter notebook


import geopandas as gpd
import folium
from shapely.geometry import MultiPolygon, Polygon

def plot_territory_polygon(territory_data, spid_spname):
    # Extract the polygon data
    polygon_data = territory_data['polygon']

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[polygon_data])

    # Calculate the center of the polygon to center our map
    center = gdf.geometry.centroid.iloc[0]

    # Create a base map centered around the polygon
    m = folium.Map(location=[center.y, center.x], zoom_start=12)

    # Define the popup content
    popup_content = f"""
    <b>Territory Information</b><br>
    Stock Point: {spid_spname} <br>
    Total Area: {round(territory_data['total_area_km2'],2)} km²<br> 
    LGA Count: {territory_data['lga_count']}<br>
    Validation Status: {territory_data['validation_status']}<br>
    Territory Version: {territory_data['territory_version']}<br>
    Is Contiguous: {'Yes' if territory_data['is_contiguous'] else 'No'}<br>
    """

    # Add the polygon to the map with improved popup information
    for _, r in gdf.iterrows():
        geo_j = folium.GeoJson(
            r['geometry'],
            style_function=lambda x: {'fillColor': 'blue'},
            popup=folium.Popup(popup_content, max_width=300)
        )
        geo_j.add_to(m)

    return m



import geopandas as gpd
import folium
from shapely.geometry import MultiPolygon, Polygon
from typing import Dict, List, Optional

def plot_selected_territories_wversion(
    territories: Dict[str, Dict],
    selected_spids: List[str]
) -> Optional[folium.Map]:
    """
    Plots selected territories from a dictionary based on specified stock point IDs using Folium maps.

    The function filters the input territories dictionary based on the selected stock point IDs and
    plots each selected territory on a single map with popup information.

    Parameters:
    -----------
    territories : Dict[str, Dict]
        A dictionary where keys are stock point IDs (strings) and values are dictionaries containing
        territory information including polygon data and metadata.
    selected_spids : List[str]
        A list of stock point IDs (strings) specifying which territories to plot.

    Returns:
    --------
    Optional[folium.Map]
        A Folium map object with the selected territories plotted on it. Returns None if no valid
        stock point IDs are selected.

    Raises:
    -------
    ImportError
        If Folium or GeoPandas libraries are not available.
    KeyError
        If necessary keys are missing in the territory data dictionary.
    """

    # Filter the territories dictionary to only include the selected spids
    selected_territories = {spid: territories[spid] for spid in selected_spids if spid in territories}

    if not selected_territories:
        print("No valid stock point IDs selected.")
        return None

    # Create a list to hold centroids of all selected polygons
    centroids = []
    for spid, territory_data in selected_territories.items():
        polygon_data = territory_data['polygon']
        gdf = gpd.GeoDataFrame(geometry=[polygon_data])
        centroids.append(gdf.geometry.centroid.iloc[0])

    # Calculate the average center for the map
    avg_longitude = sum(centroid.x for centroid in centroids) / len(centroids)
    avg_latitude = sum(centroid.y for centroid in centroids) / len(centroids)
    m = folium.Map(location=[avg_latitude, avg_longitude], zoom_start=10)

    # Iterate over each selected territory and add it to the map
    for spid, territory_data in selected_territories.items():
        polygon_data = territory_data['polygon']
        gdf = gpd.GeoDataFrame(geometry=[polygon_data])

        # Define the popup content for each territory
        popup_content = f"""
        <b>Territory Information</b><br>
        Stock Point: {spid} <br>
        Total Area: {round(territory_data['total_area_km2'], 2)} km²<br>
        LGA Count: {territory_data['lga_count']}<br>
        Validation Status: {territory_data['validation_status']}<br>
        Territory Version: {territory_data['territory_version']}<br>
        Is Contiguous: {'Yes' if territory_data['is_contiguous'] else 'No'}<br>
        """

        for _, r in gdf.iterrows():
            geo_j = folium.GeoJson(
                r['geometry'],
                style_function=lambda x: {'fillColor': 'black'},
                popup=folium.Popup(popup_content, max_width=300)
            )
            geo_j.add_to(m)

    return m

# # Specify the stock point IDs you want to plot
# selected_spids = ['1647380', '1647381']

# # Generate and save the plot for selected territories
# map_plot = plot_selected_territories(territories, selected_spids)

# if map_plot:
#     map_file_path = 'selected_territories_map.html'
#     map_plot.save(map_file_path)
#     print(f"Map saved to {map_file_path}")

import geopandas as gpd
import folium
from shapely.geometry import MultiPolygon, Polygon
from typing import Dict, List, Optional

def plot_selected_territories(
    territories: Dict[str, Dict],
    selected_spids: List[str]
) -> Optional[folium.Map]:
    """
    Plots selected territories from a dictionary based on specified stock point IDs using Folium maps.

    The function filters the input territories dictionary based on the selected stock point IDs and
    plots each selected territory on a single map with popup information and distinct colors.

    Parameters:
    -----------
    territories : Dict[str, Dict]
        A dictionary where keys are stock point IDs (strings) and values are dictionaries containing
        territory information including polygon data and metadata.
    selected_spids : List[str]
        A list of stock point IDs (strings) specifying which territories to plot.

    Returns:
    --------
    Optional[folium.Map]
        A Folium map object with the selected territories plotted on it with distinct colors.
        Returns None if no valid stock point IDs are selected.

    Raises:
    -------
    ImportError
        If Folium or GeoPandas libraries are not available.
    KeyError
        If necessary keys are missing in the territory data dictionary.
    """

    # Filter the territories dictionary to only include the selected spids
    selected_territories = {spid: territories[spid] for spid in selected_spids if spid in territories}

    if not selected_territories:
        print("No valid stock point IDs selected.")
        return None

    # Define a list of distinct colors for each stock point ID
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', #'beige',
              'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue',
              'lightgreen', 'gray', 'black', 'lightgray']

    # Create a color map for each spid
    color_map = {}
    color_index = 0

    # Assign colors to each spid
    for spid in selected_territories:
        color_map[spid] = colors[color_index % len(colors)]
        color_index += 1

    # Create a list to hold centroids of all selected polygons
    centroids = []
    for spid, territory_data in selected_territories.items():
        polygon_data = territory_data['polygon']
        gdf = gpd.GeoDataFrame(geometry=[polygon_data])
        centroids.append(gdf.geometry.centroid.iloc[0])

    # Calculate the average center for the map
    avg_longitude = sum(centroid.x for centroid in centroids) / len(centroids)
    avg_latitude = sum(centroid.y for centroid in centroids) / len(centroids)
    m = folium.Map(location=[avg_latitude, avg_longitude], zoom_start=10, tiles="CartoDB Positron")

    def style_function(color):
        return lambda x: {
            'fillColor': color,
            'color': 'black',  # Border color
            'weight': 2,     # Border width
            'fillOpacity': 0.5  # Opacity of the fill color
        }
    
    # Iterate over each selected territory and add it to the map
    for spid, territory_data in selected_territories.items():
        polygon_data = territory_data['polygon']
        gdf = gpd.GeoDataFrame(geometry=[polygon_data])

        # Get the color for this spid
        color = color_map[spid]

        # Define the popup content for each territory
        popup_content = f"""
        <b>Territory Information</b><br>
        Stock Point: {spid} <br>
        Total Area: {round(territory_data['total_area_km2'], 2)} km²<br>
        LGA Count: {territory_data['lga_count']}<br>
        Validation Status: {territory_data['validation_status']}<br>
        Territory Version: {territory_data['territory_version']}<br>
        Is Contiguous: {'Yes' if territory_data['is_contiguous'] else 'No'}<br>
        """

        for _, r in gdf.iterrows():
            geo_j = folium.GeoJson(
                r['geometry'],
                style_function=style_function(color),
                popup=folium.Popup(popup_content, max_width=300)
            )
            geo_j.add_to(m) 
            
    return m
