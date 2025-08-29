import duckdb
import geopandas as gpd
from shapely.wkt import loads as wkt_loads
from shapely.errors import WKTReadingError
import hvplot.pandas # noqa - required for hvplot to work
import pandas as pd
import logging
import holoviews as hv

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
    fill_opacity=0.7,
    marker_color='red',
    marker_size=100
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
        line_color (str, optional): The color of the polygon borders. Defaults to 'darkblue'.
        fill_opacity (float, optional): The transparency of the polygon fill color. Defaults to 0.7.
        marker_color (str, optional): The color of the centroid markers. Defaults to 'red'.
        marker_size (int, optional): The size of the centroid markers. Defaults to 100.

    Returns:
        hvplot.Interactive: An interactive map object that can be displayed in a notebook.
        Returns None if there's an error or no data.
    """
    
    if isinstance(h3_cell_ids, str):
        h3_cell_ids = [h3_cell_ids]
    
    if not h3_cell_ids:
        logging.warning("No H3 cell IDs provided. Returning None.")
        return None

    if not isinstance(popup_fields, list):
        popup_fields = list(popup_fields) if popup_fields else []

    required_columns = ['h3_index', 'polygon_wkt', 'centroid_lat', 'centroid_lng']
    
    select_fields = required_columns.copy()
    
    for field in popup_fields:
        if field not in select_fields:
            select_fields.append(field)
    
    if color_by_column and color_by_column not in select_fields:
        select_fields.append(color_by_column)
        if color_by_column not in popup_fields:
            popup_fields = popup_fields + [color_by_column]

    select_fields_str = ", ".join(select_fields)

    try:
        con = duckdb.connect(duckdb_path)
        
        placeholders = ", ".join(["?" for _ in h3_cell_ids])
        query = f"""
        SELECT {select_fields_str}
        FROM h3_cells
        WHERE h3_index IN ({placeholders})
        """
        
        logging.info(f"Executing query for {len(h3_cell_ids)} H3 cells")
        
        df = con.execute(query, h3_cell_ids).df()

    except duckdb.Error as e:
        logging.error(f"Failed to connect to or query DuckDB at {duckdb_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during database operation: {e}")
        return None
    finally:
        if 'con' in locals() and con:
            con.close()
    
    if df.empty:
        logging.warning(f"No data found in DuckDB for the provided H3 cell IDs: {h3_cell_ids}")
        return None

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logging.error(f"Required columns missing from database result: {missing_cols}")
        return None

    def safe_wkt_loads(wkt_string):
        if pd.isna(wkt_string) or not wkt_string:
            return None
        try:
            return wkt_loads(wkt_string)
        except (WKTReadingError, Exception) as e:
            logging.warning(f"Failed to parse WKT: {wkt_string[:50]}... Error: {e}")
            return None
    
    df['geometry'] = df['polygon_wkt'].apply(safe_wkt_loads)
    
    valid_geom_mask = df['geometry'].notna()
    if not valid_geom_mask.any():
        logging.error("No valid geometries found after WKT parsing")
        return None
    
    if not valid_geom_mask.all():
        logging.warning(f"Filtered out {(~valid_geom_mask).sum()} rows with invalid geometries")
        df = df[valid_geom_mask].copy()
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # ----------------------------------------------------
    # Refined logic for coloring and popup fields
    # ----------------------------------------------------
    
    # Base arguments for the plot
    plot_kwargs = {
        'geo': True, 
        'tiles': 'CartoLight', 
        'hover_cols': popup_fields, # Always use the original popup_fields
        'alpha': fill_opacity, 
        'line_width': 2,
        'line_color': line_color,
        'title': "H3 Cell Visualization",
        'width': width,
        'height': height,
        'tools': ['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset']
    }

    if color_by_column and color_by_column in gdf.columns:
        # Dynamic coloring based on a column
        plot_kwargs['c'] = color_by_column
        plot_kwargs['cmap'] = 'viridis'
        plot_kwargs['colorbar'] = True
        if show_legend:
            plot_kwargs['legend'] = 'right'
        polygon_plot = gdf.hvplot.polygons(**plot_kwargs).opts(xaxis=None, yaxis=None)
    else:
        # Static coloring with a single color
        # The key fix is to not use a dummy column for plotting, but to use the fill_color argument directly.
        # This is a cleaner way to handle static coloring in hvplot.polygons.
        # We need to use `fill_color` for static coloring, which is not available in hvplot.
        # The next best method is to use `c` with a single color map.
        plot_kwargs['c'] = static_fill_color
        plot_kwargs['cmap'] = [static_fill_color]
        plot_kwargs['colorbar'] = False
        plot_kwargs['legend'] = False # Legend is not needed for a static color
        
        # We pass the same gdf, but the 'c' argument will override any other coloring.
        polygon_plot = gdf.hvplot.polygons(**plot_kwargs).opts(xaxis=None, yaxis=None)
    
    # ----------------------------------------------------
    # End of refined logic
    # ----------------------------------------------------

    if show_markers:
        try:
            centroids_df = pd.DataFrame({
                'longitude': gdf['centroid_lng'],
                'latitude': gdf['centroid_lat']
            })
            
            for field in popup_fields:
                if field in gdf.columns:
                    centroids_df[field] = gdf[field].values
            
            markers_plot = centroids_df.hvplot.points(
                'longitude', 'latitude',
                geo=True,
                hover_cols=popup_fields,
                color=marker_color,
                size=marker_size,
                marker='triangle',
                line_color=line_color,
                line_width=1,
                alpha=0.8
            ).opts(xaxis=None, yaxis=None)
            
            return polygon_plot * markers_plot
            
        except Exception as e:
            logging.error(f"Failed to create markers plot: {e}")
            return polygon_plot
    
    return polygon_plot