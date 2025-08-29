
import folium
from folium import Map, FeatureGroup, LayerControl, Marker
from folium.plugins import MeasureControl



# --------------------------------------------------------------------------------------
# ------------------------- GRIDS/BEATS
# --------------------------------------------------------------------------------------

import folium
import json
import folium
import json

def convert_sp_assigment_df_to_geojson(assignment_gdf, use_geometry = True):
    """
     
    """ 
    
    # Build GeoJSON FeatureCollection
    features = []
    for _, row in assignment_gdf.iterrows():
        # Convert latlng_coords to GeoJSON format (lng, lat order)
        coordinates = [[[coord[1], coord[0]] for coord in row['latlng_coords']]]
        
        # Create popup content
        popup_html = f"""
        <b>Beat ID:</b> {row['beat_id']}<br>
        <b>State:</b> {row['state_name']}<br>
        <b>LGA:</b> {row['lga_name']}<br>
        <b>Ward:</b> {row['ward_name']}<br>
        <b>Area:</b> {row['area_km2']:.2f} km²<br>
        <b>Distance to SP:</b> {row['cluster_sp_dist_km']:.2f} km<br>
        <b>Total Customers:</b> {row['n_total_assigned_customers']}<br>
        <b>Active Customers:</b> {row['n_assigned_active_customers']}<br>
        <b>Recently Activated Customers:</b> {row['n_assigned_recent_activated_customers']}
        """
        if use_geometry:
            # Convert shapely geometry to GeoJSON
            # geom = row['geometry'].simplify(tolerance=0.001).__geo_interface__ 
            geom = row['geometry'].__geo_interface__ 
            
            feature = {
                "type": "Feature",
                "geometry": geom,
            } 
        else:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                },
                "properties": {
                    "popup": popup_html,
                    "all_customers": row['n_total_assigned_customers'],
                    "active_customers": row['n_assigned_active_customers'],
                    "recently_activated_customers": row['n_assigned_recent_activated_customers']
                                    
                }
            }
        
        property ={"properties": {
                    "popup": popup_html,
                    "all_customers": row['n_total_assigned_customers'],
                    "active_customers": row['n_assigned_active_customers'],
                    "recently_activated": row['n_assigned_recent_activated_customers']
                }}
        feature.update(property) 
        # Append       
        features.append(feature)
        
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson_data


def plot_h3_hexagons(sp_geojson, fill_col = 'all_customers', map_center=[6.5, 3.4]):
    """
    Efficiently plot H3 hexagons from GeoDataFrame with popups
    """
    # Create base map
    m = folium.Map(location=map_center, zoom_start=10)
    
    # Build GeoJSON FeatureCollection
    geojson_data = sp_geojson
    # Valid fill column name
    if fill_col in ["all_customers","active_customers","recently_activated_customers"]:
        fill_col = fill_col
    else:
        fill_col = 'all_customers'
    
    
    # Style function with color based on customer count
    def style_function_(feature):
        customers = feature['properties'][fill_col]

        if customers >= 200:
            color = '#800026'  # Dark red
        elif customers >= 150:
            color = '#BD0026'  # Red
        elif customers >= 100:
            color = '#E31A1C'  # Bright red
        elif customers >= 50:
            color = '#FC4E2A'  # Orange-red
        elif customers >= 20:
            color = '#FD8D3C'  # Orange
        elif customers >= 10:
            color = '#FEB24C'  # Light orange
        elif customers >= 5:
            color = '#FED976'  # Yellow
        elif customers >= 1:
            color = '#FFEDA0'  # Light yellow
        else:
            color = '#d9d9d9'  # Grey for 0

        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
    }
    
    def style_function_blue(feature):
        customers = feature['properties'][fill_col]
        if customers >= 200:
            color = '#08306b'
        elif customers >= 150:
            color = '#08519c'
        elif customers >= 100:
            color = '#2171b5'
        elif customers >= 50:
            color = '#4292c6'
        elif customers >= 20:
            color = '#6baed6'
        elif customers >= 10:
            color = '#9ecae1'
        elif customers >= 5:
            color = '#c6dbef'
        elif customers >= 1:
            color = '#deebf7'
        else:
            color = '#999999'
        
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }
    
    def style_function(feature):
        customers = feature['properties'][fill_col]
        if customers >= 200:
            color = '#67000d'
        elif customers >= 150:
            color = '#a50f15'
        elif customers >= 100:
            color = '#cb181d'
        elif customers >= 50:
            color = '#ef3b2c'
        elif customers >= 20:
            color = '#fb6a4a'
        elif customers >= 10:
            color = '#fc9272'
        elif customers >= 5:
            color = '#fcbba1'
        elif customers >= 1:
            color = '#fee0d2'
        else:
            color = '#999999'
        
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }
    
    # Add to map with popups
    folium.GeoJson(
        geojson_data,
        style_function=style_function,
        popup=folium.GeoJsonPopup(
            fields=['popup'],
            aliases=[''],
            labels=False
        )
    ).add_to(m)
    
    return m


def prepare_beat_folium_GeoJson(sp_geojson, fill_col = 'all_customers', use_blue=False):
    """
    Efficiently plot H3 hexagons from GeoDataFrame with popups
    """
    # Create base map 
    
    # Build GeoJSON FeatureCollection
    geojson_data = sp_geojson
    # Valid fill column name
    if fill_col in ["all_customers","active_customers","recently_activated_customers"]:
        fill_col = fill_col
    else:
        fill_col = 'all_customers'
    
    
    # Style function with color based on customer count    
    def style_function_blue(feature):
        customers = feature['properties'][fill_col]
        if customers >= 200:
            color = '#08306b'
        elif customers >= 150:
            color = '#08519c'
        elif customers >= 100:
            color = '#2171b5'
        elif customers >= 50:
            color = '#4292c6'
        elif customers >= 20:
            color = '#6baed6'
        elif customers >= 10:
            color = '#9ecae1'
        elif customers >= 5:
            color = '#c6dbef'
        elif customers >= 1:
            color = '#deebf7'
        else:
            color = '#999999'
        
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }
    
    def style_function_red(feature):
        customers = feature['properties'][fill_col]
        if customers >= 200:
            color = '#67000d'
        elif customers >= 150:
            color = '#a50f15'
        elif customers >= 100:
            color = '#cb181d'
        elif customers >= 50:
            color = '#ef3b2c'
        elif customers >= 20:
            color = '#fb6a4a'
        elif customers >= 10:
            color = '#fc9272'
        elif customers >= 5:
            color = '#fcbba1'
        elif customers >= 1:
            color = '#fee0d2'
        else:
            color = '#999999'
        
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }
    
    if use_blue:
        use_style_function = style_function_blue
    else:
        use_style_function = style_function_red
    # Add to map with popups
    beat_folium_GeoJson = folium.GeoJson(
        geojson_data,
        style_function=use_style_function,
        popup=folium.GeoJsonPopup(
            fields=['popup'],
            aliases=[''],
            labels=False
        )
    ) 
    
    return beat_folium_GeoJson



# --------------------------------------------------------------------------------------
# ------------------------- CUSTOMERS
# --------------------------------------------------------------------------------------

def convert_customers_to_geojson(customer_df):
    """
    Convert customer dataframe to GeoJSON points for scatter plotting
    """
    features = []
    for _, row in customer_df.iterrows():
        popup_html = f"""
        <b>Customer ID:</b> {row['customer_id']}<br>
        <b>Contact:</b> {row['contact_name']}<br>
        <b>Type:</b> {row['customer_type']}<br>
        <b>Assignment Type:</b> {row['assignment_type']}<br>
        <b>State:</b> {row['state_name']}<br>
        <b>City:</b> {row['city_name']}<br>
        <b>Status:</b> {row['customer_status']}<br>
        <b>KYC Status:</b> {row['kyc_capture_status']}
        """
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['longitude'], row['latitude']]
            },
            "properties": {
                "popup": popup_html,
                "customer_type": row['customer_type'],
                "assignment_type_id": row['assignment_type_id'],
                "customer_status": row['customer_status'],
                "stock_point_id": row['stock_point_id']
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


def prepare_customer_assignment_GeoJson(geojson_data):
    """
    Plot customers as scatter points
    """ 
    
    # Style function based on customer type/status
    def style_function(feature):
        customer_type = feature['properties']['customer_type']
        status = feature['properties']['customer_status']
        
        # Color by customer type
        if customer_type == 'buying customers':
            color = "#c40505a6"
        elif customer_type == 'recently activated':
            color = "#069CE2"
        else:
            color = "#8c8f91"
            
        # Adjust opacity by status
        opacity = 1 if status == 'Active' else 0.1
        
        return {
            'fillColor': color,
            'fill':True,
            'color': color,
            'weight': 2,
            'fillOpacity': opacity,
            'radius': 3
        }
        
    
    # Use CircleMarker for better performance with many points
    return folium.GeoJson(
        geojson_data,
        marker=folium.CircleMarker(), 
        style_function=style_function,
        popup=folium.GeoJsonPopup(
            fields=['popup'],
            aliases=[''],
            labels=False
        )
    )  


def prepare_customer_assignment_GeoJson_(geojson_data):
    """
    Plot customers as markers with shop icons and colors based on customer type
    """ 
    
    def point_to_layer(feature, latlng):
        """Create custom markers for each feature"""
        customer_type = feature['properties']['customer_type']
        status = feature['properties']['customer_status']
        
        # Color by customer type
        if customer_type == 'buying customers':
            color = 'red'
        elif customer_type == 'recently activated':
            color = 'purple'
        else:
            color = 'gray'
        
        # Adjust opacity by status
        opacity = 1.0 if status == 'Active' else 0.6
        
        # Create marker with shop icon
        marker = folium.Marker(
            location=latlng,
            icon=folium.Icon(
                color=color,
                icon='store',
                prefix='fa'
            ),
            opacity=opacity
        )
        
        return marker
    
    return folium.GeoJson(
        geojson_data,
        point_to_layer=point_to_layer,
        popup=folium.GeoJsonPopup(
            fields=['popup'],
            aliases=[''],
            labels=False,
            max_width=300
        )
    )    
    


# --------------------------------------------------------------------------------------
# ------------------------- MAIN FUNCTION
# --------------------------------------------------------------------------------------
def create_stockpoint_map(spid, processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, 
                         customer_stockpoint_cluster_assignment_df, sp_territories_dict,
                         fill_col="all_customers", use_geometry=True, output_dir=None):
    """
    Create an interactive folium map for a stock point with territories, beats, and customers.
    
    Parameters:
    -----------
    spid : int
        Stock point ID to visualize
    processed_sp_dim_df : DataFrame
        Stock point dimension data with coordinates and names
    stockpoint_h3_coverage_with_metadata : DataFrame
        H3 coverage data with metadata for beats
    customer_stockpoint_cluster_assignment_df : DataFrame
        Customer assignment data
    sp_territories_dict : dict
        Dictionary containing territory polygons by stock point ID
    fill_col : str, default "all_customers"
        Column to use for fill color in beats visualization
    use_geometry : bool, default True
        Whether to use clipped geometry
    output_dir : Path, optional
        Directory to save the HTML file. If None, returns map object only
    
    Returns:
    --------
    folium.Map or str
        Returns folium map object if output_dir is None, otherwise saves file and returns filename
    """
    
    # Prepare data
    sp_dim = processed_sp_dim_df.query(f'stock_point_id == {spid}').reset_index()
    sp_h3_coverage = stockpoint_h3_coverage_with_metadata.query(f'stock_point_id == {spid}').reset_index()
    sp_customer_assignment = customer_stockpoint_cluster_assignment_df.query(f'stock_point_id == {spid}').reset_index()
    
    # Extract plot elements
    coord_lat, coord_lng = sp_dim['latitude'].iloc[0], sp_dim['longitude'].iloc[0]
    spname = sp_dim['stock_point_name'].iloc[0]
    sp_boundary_geojson = sp_territories_dict[spid]['polygon']
    sp_beat_geojson = convert_sp_assigment_df_to_geojson(sp_h3_coverage, use_geometry=use_geometry)
    customer_geojson = convert_customers_to_geojson(sp_customer_assignment)
    
    # Create base map
    base_map = folium.Map(location=[coord_lat, coord_lng],
                          tiles="CartoDB Positron",
                          prefer_canvas=True,
                          zoom_start=12)
    
    # Add boundary layer
    fg_boundary = FeatureGroup(name="Boundary")
    folium.GeoJson(sp_boundary_geojson).add_to(fg_boundary)
    
    # Add beats layer
    fg_beats = FeatureGroup(name="Beats and Assignments") 
    beat_folium_GeoJson = prepare_beat_folium_GeoJson(sp_beat_geojson, fill_col=fill_col)
    beat_folium_GeoJson.add_to(fg_beats)
    
    # Add customers layer
    fg_customers = FeatureGroup(name="Customers")
    customer_geojson_folium = prepare_customer_assignment_GeoJson(customer_geojson)
    customer_geojson_folium.add_to(fg_customers)
    
    # Add stock point marker
    fg_spmarker = FeatureGroup(name="Stock Point")
    folium.Marker(
        location=[coord_lat, coord_lng],
        popup=spname,
        icon=folium.Icon(color="green")
    ).add_to(fg_spmarker)
    
    # Add all layers to map
    fg_boundary.add_to(base_map)
    fg_beats.add_to(base_map)
    fg_customers.add_to(base_map)
    fg_spmarker.add_to(base_map)
    
    # Add layer control
    base_map.add_child(MeasureControl(
        primary_length_unit='kilometers',   # or 'meters', 'miles'
        secondary_length_unit='meters',
        primary_area_unit='sqmeters',       # or 'hectares', 'acres'
        secondary_area_unit='acres'
    ))
    
    folium.LayerControl().add_to(base_map)
    
    # Save or return map
    import re
    if output_dir:
        suff = 'clipped' if use_geometry else 'all'
        spname_ = re.sub(r'[^a-zA-Z0-9]', ' ', spname).strip().replace(' ', '_')
        # plot_filename = output_dir / f"""{spname_}_{suff}_{fill_col}.html"""
        plot_filename = output_dir / f"""{spname_}_{fill_col}.html"""
        base_map.save(plot_filename)
        return str(plot_filename)
    else:
        return base_map


    