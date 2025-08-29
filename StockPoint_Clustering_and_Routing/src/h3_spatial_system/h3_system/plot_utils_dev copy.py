import duckdb
import geopandas as gpd
from shapely.wkt import loads as wkt_loads
import hvplot.pandas # noqa - required for hvplot to work
import pandas as pd
import logging
import holoviews as hv # Import holoviews to use its extension

# Set up basic logging for error handling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_h3_from_db_fast(
    h3_cell_ids, 
    duckdb_path, 
    popup_fields=['h3_index', 'resolution', 'lga_name', 'coverage_percentage'],
    show_markers=True,
    width=800,
    height=600,
    show_legend=True,
    color_by_column=None,
    static_fill_color='lightblue',
    line_color='darkblue',
    fill_opacity=0.3
):
    """
    Plot H3 cell(s) polygon and centroid on an interactive map using geopandas and hvplot.
    This provides a much faster rendering experience than folium, especially for
    large numbers of polygons.

    Args:
        h3_cell_ids (str or list): Single H3 cell identifier or a list of H3 cell identifiers.
        duckdb_path (str): The file path to the DuckDB database.
        popup_fields (list, optional): List of column names from the DuckDB table to display
                                       in the popup (hover tooltip) for each H3 cell.
        show_markers (bool, optional): Whether to display centroid markers. Defaults to True.
        width (int, optional): The width of the plot in pixels. Defaults to 800.
        height (int, optional): The height of the plot in pixels. Defaults to 600.
        show_legend (bool, optional): Whether to display the legend. Defaults to True.
        color_by_column (str, optional): The name of the column to use for color-mapping the polygons.
                                         If None, a static color is used. Defaults to None.
        static_fill_color (str, optional): The fill color to use when color_by_column is None. Defaults to 'lightblue'.
        line_color (str, optional): The color of the polygon and marker borders. Defaults to 'darkblue'.
        fill_opacity (float, optional): The transparency of the polygon fill color. Defaults to 0.3.

    Returns:
        hvplot.Interactive: An interactive map object that can be displayed in a notebook.
        Returns None if there's an error or no data.
    """
    
    # Ensure h3_cell_ids is a list for uniform processing
    if isinstance(h3_cell_ids, str):
        h3_cell_ids = [h3_cell_ids]
    
    # If no cell IDs are provided, return None
    if not h3_cell_ids:
        logging.warning("No H3 cell IDs provided. Returning None.")
        return None

    # Prepare a list of H3 IDs for the SQL query's IN clause
    h3_id_list_str = ", ".join([f"'{h}'" for h in h3_cell_ids])

    # Add the column to be used for coloring to the select and popup fields
    if color_by_column and color_by_column not in popup_fields:
        popup_fields.append(color_by_column)

    # Construct the list of columns to select from the DuckDB table
    select_fields = list(set(['h3_index', 'polygon_wkt', 'centroid_lat', 'centroid_lng'] + popup_fields))
    select_fields_str = ", ".join(select_fields)

    # Connect to the DuckDB database
    try:
        con = duckdb.connect(duckdb_path)
        
        # SQL query to fetch data for the given H3 cells
        query = f"""
        SELECT {select_fields_str}
        FROM h3_cells
        WHERE h3_index IN ({h3_id_list_str})
        """
        
        logging.info(f"Executing query: {query}")
        
        # Execute the query and fetch the results directly into a Pandas DataFrame
        df = con.execute(query).df()

    except duckdb.Error as e:
        logging.error(f"Failed to connect to or query DuckDB at {duckdb_path}: {e}")
        return None
    finally:
        # Ensure the connection is always closed
        if 'con' in locals() and con:
            con.close()
    
    # If no data was found for the given IDs, return None
    if df.empty:
        logging.warning(f"No data found in DuckDB for the provided H3 cell IDs: {h3_cell_ids}")
        return None

    # Use a lambda function to convert the WKT string to a shapely polygon
    df['geometry'] = df['polygon_wkt'].apply(lambda x: wkt_loads(x) if x else None)
    
    # Create the GeoDataFrame from the Pandas DataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    # Define the arguments for hvplot based on whether a color column is provided
    plot_kwargs = dict(
        geo=True, 
        tiles='CartoLight', 
        hover_cols=popup_fields, 
        alpha=fill_opacity, 
        line_width=2,
        title="H3 Cell Visualization",
        width=width,
        height=height,
        legend=show_legend,
        line_color=line_color
    )
    
    if color_by_column:
        plot_kwargs['c'] = color_by_column
        plot_kwargs['cmap'] = 'viridis' # Added a default colormap for better visualization
    else:
        plot_kwargs['color'] = static_fill_color

    # Create the base map of polygons
    polygon_plot = gdf.hvplot(**plot_kwargs).opts(
        xaxis=None, 
        yaxis=None
    )

    # Create the centroid markers plot if requested
    if show_markers:
        # First, re-project the GeoDataFrame to a projected CRS (Web Mercator)
        gdf_projected = gdf.to_crs(epsg=3857)

        # Create a DataFrame for the centroids using the projected geometries
        centroids_df_projected = pd.DataFrame({
            'lat': gdf_projected['centroid_lat'],
            'lon': gdf_projected['centroid_lng'],
            'geometry': gdf_projected['geometry'].centroid
        })
        
        # Plot the centroids on top of the polygons.
        markers_plot = centroids_df_projected.hvplot.points(
            'lon', 'lat',
            geo=True,
            hover_cols=popup_fields,
            color='red', # Keep marker color red, as requested
            marker='triangle',
            line_color=line_color,
            size=100
        ).opts(
            xaxis=None, 
            yaxis=None,
            legend=show_legend
        )
        
        # Combine the polygon and marker plots
        return polygon_plot * markers_plot
    
    # If no markers are requested, just return the polygon plot
    return polygon_plot

# Example usage (assuming geopandas, hvplot, and duckdb are installed and a db file exists)
# from config.settings import STORAGE_CONFIG
# H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']
# 
# # Example H3 cell IDs
# h3_cells_to_plot = ["871f308a0ffffff", "871f308b0ffffff", "871f308c0ffffff"]
# 
# # Plot with both polygons and markers, and custom popup fields
# # plot = plot_h3_from_db_fast(h3_cells_to_plot, H3_DUCKDB_PATH, popup_fields=['h3_index', 'lga_name', 'area_km2'])
# 
# # Plot with only polygons and default popup fields
# # plot_no_markers = plot_h3_from_db_fast(h3_cells_to_plot, H3_DUCKDB_PATH, show_markers=False)
