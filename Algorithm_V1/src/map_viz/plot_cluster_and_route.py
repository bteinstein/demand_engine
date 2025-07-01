
from src.map_viz.ClusterRouteMapper import ClusterRouteMapper
import logging


def plot_cluster_and_route(
    df,
    trip_dict = {},
    routes_info = [],
    output_filename = "my_enhanced_map.html",
    lat_col='Latitude',
    lon_col='Longitude',
    cluster_col='TripID',
    lga_col='LGA',
    lcda_col='LCDA',
    popup_cols=['CustomerID', 'LGA', 'LCDA', 'TotalQuantity'],
    tooltip_cols=['CustomerID', 'LGA', 'LCDA'],
    logger = logging.getLogger(__name__)
):
    """
    Plot clusters and routes on a map using ClusterRouteMapper.

    Parameters:
    - df_selected_trip: DataFrame containing selected trip data.
    - trip_dict: Dictionary containing trip information.
    - routes_info: Dictionary containing route information.
    - main_dir: Directory to save the output map.
    """
    # Initialize the mapper
    mapper = ClusterRouteMapper(
        df=df,
        trip_data=trip_dict,
        routes_info=routes_info,
        lat_col=lat_col,
        lon_col=lon_col,
        cluster_col=cluster_col,
        lga_col=lga_col,
        lcda_col=lcda_col,
        popup_cols=popup_cols,
        tooltip_cols=tooltip_cols,
        zoom_start=11,
        tiles='cartodb positron',
        marker_radius=7,
        marker_fill_opacity=0.7,
        marker_stroke_width=1,
        marker_stroke_opacity=1.0,
        cluster_palette='viridis',  # Different palette for clusters
        route_palette='Dark2',      # Different palette for routes
        legend_limit=10,
        output_filename=output_filename,
        route_weight=6,
        route_opacity=0.8,
        stock_icon='truck',         # Custom icon for stock point
        stock_color='darkblue',     # Custom color for stock point
        chunk_size=5                # Example: process markers in chunks of 5
    )

    # Create and save the map
    folium_map = mapper.create_map()
    folium_map.save(output_filename)
    logger.info(f"Map saved to {output_filename}")
    
    return folium_map