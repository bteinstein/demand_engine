import folium
from folium import plugins
import h3

import folium
import folium.plugins as plugins
import duckdb
import json
import logging

# Set up basic logging for error handling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def plot_h3_cells(h3_cell_ids, colors=None):
    """
    Plot H3 cell(s) polygon and centroid on a folium map
    
    Args:
        h3_cell_ids (str or list): Single H3 cell identifier or list of H3 cell identifiers
        colors (list, optional): List of colors for each cell. If None, uses default color scheme
    
    Returns:
        folium.Map: Interactive map with H3 cell visualization
    """
    
    # Convert single cell to list for uniform processing
    if isinstance(h3_cell_ids, str):
        h3_cell_ids = [h3_cell_ids]
    
    # Default colors if not provided
    if colors is None:
        default_colors = ['#3388ff', '#ff3333', '#33ff33', '#ff8c00', '#8a2be2', 
                         '#ff1493', '#00ced1', '#ffd700', '#ff6347', '#32cd32']
        colors = (default_colors * ((len(h3_cell_ids) // len(default_colors)) + 1))[:len(h3_cell_ids)]
    
    # Calculate map center from all centroids
    all_lats, all_lngs = [], []
    cell_data = []
    
    for i, h3_cell_id in enumerate(h3_cell_ids):
        # Get the centroid (lat, lng) of the H3 cell - v4 API
        centroid_lat, centroid_lng = h3.cell_to_latlng(str(h3_cell_id))
        all_lats.append(centroid_lat)
        all_lngs.append(centroid_lng)
        
        # Get the boundary coordinates of the H3 cell - v4 API
        boundary = h3.cell_to_boundary(h3_cell_id)
        
        # Convert boundary to list of [lat, lng] pairs for folium
        polygon_coords = [[lat, lng] for lat, lng in boundary]
        
        # Get H3 cell properties - v4 API
        resolution = h3.get_resolution(h3_cell_id)
        
        cell_data.append({
            'id': h3_cell_id,
            'centroid': (centroid_lat, centroid_lng),
            'polygon': polygon_coords,
            'resolution': resolution,
            'color': colors[i]
        })
    
    # Calculate center of all cells
    center_lat = sum(all_lats) / len(all_lats)
    center_lng = sum(all_lngs) / len(all_lngs)
    
    # Create folium map centered on all cells
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='CartoDB positron'  # Clean tiles for data visualization
    )
    
    # Add each H3 cell to the map
    all_bounds = []
    for i, cell in enumerate(cell_data):
        # Add the H3 cell polygon
        folium.Polygon(
            locations=cell['polygon'],
            color=cell['color'],
            weight=3,
            fill=True,
            fillColor=cell['color'],
            fillOpacity=0.3,
            popup=folium.Popup(f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4>H3 Cell #{i+1}</h4>
                    <p><b>Cell ID:</b> {cell['id']}</p>
                    <p><b>Resolution:</b> {cell['resolution']}</p>
                    <p><b>Center:</b> {cell['centroid'][0]:.6f}, {cell['centroid'][1]:.6f}</p>
                    <p><b>Area:</b> {h3.cell_area(cell['id'], unit='km^2'):.6f} km²</p>
                </div>
            """, max_width=250)
        ).add_to(m)
        
        # Add centroid marker
        folium.Marker(
            location=cell['centroid'],
            popup=folium.Popup(f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4>H3 Cell #{i+1} Centroid</h4>
                    <p><b>Coordinates:</b><br>{cell['centroid'][0]:.6f}, {cell['centroid'][1]:.6f}</p>
                    <p><b>Cell ID:</b> {cell['id']}</p>
                    <p><b>Resolution:</b> {cell['resolution']}</p>
                </div>
            """, max_width=250),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Collect bounds for fitting
        all_bounds.extend(cell['polygon'])
    
    # Add additional map controls
    plugins.Fullscreen().add_to(m)
    
    # Fit map bounds to all polygons
    if all_bounds:
        m.fit_bounds(all_bounds)
    
    return m


def plot_h3_from_db(
    h3_cell_ids, 
    duckdb_path, 
    colors=None,
    show_markers=True,
    popup_fields=['h3_index', 'resolution', 'lga_name', 'coverage_percentage'],
    polygon_weight = 2,
    polygon_opacity = 0.1
):
    """
    Plot H3 cell(s) polygon and centroid on a folium map by querying a DuckDB table.

    Args:
        h3_cell_ids (str or list): Single H3 cell identifier or a list of H3 cell identifiers.
        duckdb_path (str): The file path to the DuckDB database.
        colors (list, optional): List of colors for each cell. If None, uses a default color scheme.
        show_markers (bool, optional): Whether to display centroid markers. Defaults to True.
        popup_fields (list, optional): List of column names from the DuckDB table to display
                                       in the popup for each H3 cell. Defaults to a useful set.

    Returns:
        folium.Map: An interactive map with H3 cell visualization. Returns None if there's an error.
    """
    
    # Ensure h3_cell_ids is a list for uniform processing
    if isinstance(h3_cell_ids, str):
        h3_cell_ids = [h3_cell_ids]
    
    # If no cell IDs are provided, return an empty map
    if not h3_cell_ids:
        logging.warning("No H3 cell IDs provided. Returning an empty map.")
        return folium.Map(location=[0, 0], zoom_start=2)

    # Prepare a list of H3 IDs for the SQL query's IN clause
    h3_id_list_str = ", ".join([f"'{h}'" for h in h3_cell_ids])

    # Construct the list of columns to select from the DuckDB table
    # This ensures we only fetch the data we need.
    select_fields = list(set(['h3_index', 'resolution', 'centroid_lat', 'centroid_lng', 'boundary_json'] + popup_fields))
    select_fields_str = ", ".join(select_fields)

    cell_data = []
    
    # Connect to the DuckDB database
    try:
        con = duckdb.connect(duckdb_path)
        
        # SQL query to fetch data for the given H3 cells
        query = f"""
        SELECT {select_fields_str}
        FROM h3_cells
        WHERE h3_index IN ({h3_id_list_str})
        """
        
        # logging.info(f"Executing query: {query}")
        
        # Execute the query and fetch all results
        results = con.execute(query).fetchall()
        
        # Get column names from the query result description
        columns = [desc[0] for desc in con.description]

        # Process fetched data
        for row in results:
            row_dict = dict(zip(columns, row))
            
            # Parse the boundary_json string into a Python list of lists
            # The coordinates are expected in [lat, lng] format for folium
            try:
                boundary_coords = json.loads(row_dict.get('boundary_json', '[]'))
            except (json.JSONDecodeError, TypeError) as e:
                logging.error(f"Failed to decode boundary_json for H3 cell {row_dict.get('h3_index')}: {e}")
                continue

            cell_data.append({
                'id': row_dict.get('h3_index'),
                'h3_derived_id': row_dict.get('h3_derived_id'),
                'centroid': (row_dict.get('centroid_lat'), row_dict.get('centroid_lng')),
                'polygon': boundary_coords,
                'properties': row_dict # Store the entire row as properties for the popup
            })

    except duckdb.Error as e:
        logging.error(f"Failed to connect to or query DuckDB at {duckdb_path}: {e}")
        return None
    finally:
        # Ensure the connection is always closed
        if 'con' in locals() and con:
            con.close()

    # If no data was found for the given IDs, return a default map
    if not cell_data:
        logging.warning(f"No data found in DuckDB for the provided H3 cell IDs: {h3_cell_ids}")
        return folium.Map(location=[0, 0], zoom_start=2)

    # Default colors if not provided
    if colors is None:
        default_colors = ['#3388ff', '#ff3333', '#33ff33', '#ff8c00', '#8a2be2', 
                          '#ff1493', '#00ced1', '#ffd700', '#ff6347', '#32cd32']
        colors = (default_colors * ((len(cell_data) // len(default_colors)) + 1))[:len(cell_data)]
    
    # Get all centroids to calculate the map center and fit bounds
    all_lats = [cell['centroid'][0] for cell in cell_data]
    all_lngs = [cell['centroid'][1] for cell in cell_data]
    center_lat = sum(all_lats) / len(all_lats)
    center_lng = sum(all_lngs) / len(all_lngs)
    
    # Create folium map centered on all cells
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='CartoDB positron'  # Clean tiles for data visualization
    )
    
    # Add each H3 cell to the map
    all_bounds = []
    for i, cell in enumerate(cell_data):
        # cell_id = cell['id']
        cell_id = cell['h3_derived_id']
        cell_props = cell['properties']
        
        # Dynamically build the HTML for the popup based on the requested fields
        popup_html = """
        <div style="font-family: Arial; width: 250px;">
            <h4>H3 Cell Details</h4>
            {}
        </div>
        """
        prop_items = ""
        for field in popup_fields:
            if field in cell_props and cell_props[field] is not None:
                prop_items += f"<p><b>{field.replace('_', ' ').title()}:</b> {cell_props[field]}</p>"
        
        final_popup = folium.Popup(popup_html.format(prop_items), max_width=300)
        
        # Add the H3 cell polygon        
        folium.Polygon(
            locations=cell['polygon'],
            color=colors[i % len(colors)],
            weight=polygon_weight,
            fill=True,
            fillColor=colors[i % len(colors)],
            fillOpacity=polygon_opacity,
            tooltip=folium.Tooltip(f"ID: {cell_id}"),
            popup=final_popup
        ).add_to(m)
        
        # Add centroid marker if requested
        if show_markers:
            folium.Marker(
                location=cell['centroid'],
                popup=final_popup,
                tooltip=f"Centroid for {cell_id}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        # Collect bounds for fitting
        all_bounds.extend(cell['polygon'])
    
    # Add additional map controls
    plugins.Fullscreen().add_to(m)
    
    # Fit map bounds to all polygons
    if all_bounds:
        m.fit_bounds(all_bounds)
    
    return m

# Example usage (assuming a dummy DuckDB file exists)
# from config.settings import STORAGE_CONFIG
# H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']


# Examples of usage:

# # Single cell (backward compatible)
# h3_cell_id = '885855b347fffff'
# map_single = plot_h3_cells(h3_cell_id)
# map_single.save('h3_single_cell_map.html')

# # Get valid neighboring cells for demonstration
# neighbors = h3.grid_ring(h3_cell_id, k=1)  # Get ring of neighbors at distance 1
# h3_cell_list = [h3_cell_id] + list(neighbors)[:2]  # Original cell + 2 neighbors

# print(f"Original cell: {h3_cell_id}")
# print(f"Valid neighbors: {list(neighbors)[:5]}")  # Show first 5 neighbors

# # Multiple cells with automatic colors
# map_multiple = plot_h3_cells(h3_cell_list)
# map_multiple.save('h3_multiple_cells_map.html')

# # Multiple cells with custom colors
# custom_colors = ['#ff0000', '#00ff00', '#0000ff']
# map_custom = plot_h3_cells(h3_cell_list, colors=custom_colors)
# map_custom.save('h3_custom_colors_map.html')

# # Display information for the example
# print(f"\nSingle H3 Cell ID: {h3_cell_id}")
# print(f"Resolution: {h3.get_resolution(h3_cell_id)}")
# print(f"Centroid: {h3.cell_to_latlng(h3_cell_id)}")
# print(f"Area: {h3.cell_area(h3_cell_id, unit='km^2'):.6f} km²")
# print(f"\nUsing cells for visualization: {h3_cell_list}")
# print("\nMaps saved:")
# print("- h3_single_cell_map.html (single cell)")
# print("- h3_multiple_cells_map.html (multiple cells with auto colors)")
# print("- h3_custom_colors_map.html (multiple cells with custom colors)")
