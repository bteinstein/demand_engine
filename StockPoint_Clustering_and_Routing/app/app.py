import streamlit as st
import pandas as pd
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import MeasureControl
import streamlit_folium as st_folium
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from src.data.load_postprocessed_data import load_data
from src.viz.viz import (
    convert_sp_assigment_df_to_geojson,
    convert_customers_to_geojson,
    prepare_beat_folium_GeoJson,
    prepare_customer_assignment_GeoJson
)

# -------------------------
# Local data loading function
def load_from_local():
    import os 
    import gzip
    import pickle
    from config.settings import EXPORTS_DIR
    
    map_input_dir = EXPORTS_DIR / 'map_input_data'
    files = {
        'processed_sp_dim_df': 'processed_sp_dim_df.pkl.gz',
        'stockpoint_h3_coverage_with_metadata': 'stockpoint_h3_coverage_with_metadata.pkl.gz',
        'customer_stockpoint_cluster_assignment_df': 'customer_stockpoint_cluster_assignment_df.pkl.gz',
        'sp_territories_dict': 'sp_territories_dict.pkl.gz'
    }
    
    loaded_data = {}
    for key, filename in files.items():
        filepath = map_input_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Local data file not found: {filepath}")
        with gzip.open(filepath, 'rb') as f:
            loaded_data[key] = pickle.load(f)
    
    return (
        loaded_data['processed_sp_dim_df'],
        loaded_data['stockpoint_h3_coverage_with_metadata'],
        loaded_data['customer_stockpoint_cluster_assignment_df'],
        loaded_data['sp_territories_dict']
    )


# -------------------------
# Page config
st.set_page_config(
    page_title="Multi-Stockpoint Map Viewer",
    page_icon="🗺️",
    layout="wide"
)

@st.cache_data
def load_cached_data():
    """Load and cache all datasets"""
    # return load_data()
    return load_from_local()

def create_multi_stockpoint_map(selected_spids, processed_sp_dim_df, 
                               stockpoint_h3_coverage_with_metadata,
                               customer_stockpoint_cluster_assignment_df, 
                               sp_territories_dict, fill_col="all_customers", 
                               use_geometry=True, show_boundaries=True,
                               show_beats=True, show_customers=False, 
                               show_markers=True):
    """Create folium map with multiple stockpoints"""
    
    # Calculate center point from selected stockpoints
    selected_sp_data = processed_sp_dim_df[
        processed_sp_dim_df['stock_point_id'].isin(selected_spids)
    ]
    center_lat = selected_sp_data['latitude'].mean()
    center_lng = selected_sp_data['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lng],
        tiles="CartoDB Positron",
        prefer_canvas=True,
        zoom_start=9
    )
    
    # Color palette for different stockpoints
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'darkblue', 'darkgreen', 'cadetblue']
    
    for i, spid in enumerate(selected_spids):
        color = colors[i % len(colors)]
        
        # Get data for this stockpoint
        sp_dim = processed_sp_dim_df.query(f'stock_point_id == {spid}').iloc[0]
        sp_h3_coverage = stockpoint_h3_coverage_with_metadata.query(f'stock_point_id == {spid}')
        sp_customers = customer_stockpoint_cluster_assignment_df.query(f'stock_point_id == {spid}')
        
        spname = sp_dim['stock_point_name']
        coord_lat, coord_lng = sp_dim['latitude'], sp_dim['longitude']
        
        # Create feature groups for this stockpoint
        fg_boundary = FeatureGroup(name=f"{spname} - Boundary")
        fg_beats = FeatureGroup(name=f"{spname} - Beats")
        fg_customers = FeatureGroup(name=f"{spname} - Customers")
        fg_marker = FeatureGroup(name=f"{spname} - Stock Point")
        
        # Add boundary (conditionally)
        if show_boundaries and spid in sp_territories_dict:
            boundary_polygon = sp_territories_dict[spid]['polygon']
            
            # Debug: Check if polygon data exists
            if boundary_polygon:
                folium.GeoJson(
                        boundary_polygon, 
                        tooltip=f"{spname} - Boundary"
                    ).add_to(fg_boundary)
                
            else:
                st.warning(f"No boundary polygon found for {spname}")
        
        # Add beats (conditionally)
        if show_beats and not sp_h3_coverage.empty:
            beat_geojson = convert_sp_assigment_df_to_geojson(
                sp_h3_coverage, use_geometry=use_geometry
            )
            beat_folium_geojson = prepare_beat_folium_GeoJson(
                beat_geojson, fill_col=fill_col, use_blue=(i % 2 == 1)
            )
            beat_folium_geojson.add_to(fg_beats)
        
        # Add customers (conditionally)
        if show_customers and not sp_customers.empty:
            customer_geojson = convert_customers_to_geojson(sp_customers)
            customer_folium_geojson = prepare_customer_assignment_GeoJson(customer_geojson)
            customer_folium_geojson.add_to(fg_customers)
        
        # Add stockpoint marker (conditionally)
        if show_markers:
            folium.Marker(
                location=[coord_lat, coord_lng],
                popup=f"<b>{spname}</b><br>ID: {spid}",
                icon=folium.Icon(color=color, icon='home')
            ).add_to(fg_marker)
        
        # Add feature groups to map
        fg_boundary.add_to(m)
        fg_beats.add_to(m)
        fg_customers.add_to(m)
        fg_marker.add_to(m)
    
    # Add controls
    m.add_child(MeasureControl(
        primary_length_unit='kilometers',
        secondary_length_unit='meters',
        primary_area_unit='sqmeters',
        secondary_area_unit='acres'
    ))
    
    LayerControl().add_to(m)
    
    return m

def display_selection_summary(selected_spids, processed_sp_dim_df, 
                            customer_stockpoint_cluster_assignment_df,
                            stockpoint_h3_coverage_with_metadata):
    """Display selection summary in a formatted container"""
    
    st.subheader("📊 Selection Summary")
    st.success(f"✅ {len(selected_spids)} stockpoint(s) selected")
    
    # Get summary data
    selected_data = processed_sp_dim_df[
        processed_sp_dim_df['stock_point_id'].isin(selected_spids)
    ]
    
    total_customers = customer_stockpoint_cluster_assignment_df[
        customer_stockpoint_cluster_assignment_df['stock_point_id'].isin(selected_spids)
    ]
    total_coverage = stockpoint_h3_coverage_with_metadata[
        stockpoint_h3_coverage_with_metadata['stock_point_id'].isin(selected_spids)
    ]
    
    # Customer metrics
    st.write("**Customer Metrics**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Customers", f"{len(total_customers):,}")
        active_count = len(total_customers[total_customers['customer_status'] == 'Active'])
        st.metric("Active", f"{active_count:,}")
    
    with col2:
        recent_count = len(total_customers[total_customers['customer_type'] == 'recently activated'])
        st.metric("Recently Activated", f"{recent_count:,}")
        st.metric("Total Beats", f"{len(total_coverage):,}")
    
    if not total_coverage.empty:
        st.metric("Total Area", f"{total_coverage['area_km2'].sum():.0f} km²")
    
    # Selected stockpoints list
    st.write("**Selected Stockpoints**")
    for _, row in selected_data.iterrows():
        st.write(f"• {row['stock_point_name']} (ID: {row['stock_point_id']})")

def main():
    st.title("🗺️ MFC Beats and Routes")
    
    # Load data
    try:
        processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, \
        customer_stockpoint_cluster_assignment_df, sp_territories_dict = load_cached_data()
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Map Controls")
    
    # Stockpoint selection
    st.sidebar.subheader("Select Stockpoints")
    
    # Search stockpoints
    search_term = st.sidebar.text_input(
        "🔍 Search stockpoints", 
        placeholder="Enter name or ID..."
    )
    
    # Filter stockpoints based on search
    if search_term:
        filtered_sp = processed_sp_dim_df[
            processed_sp_dim_df['stock_point_name'].str.contains(search_term, case=False) |
            processed_sp_dim_df['stock_point_id'].astype(str).str.contains(search_term)
        ]
    else:
        filtered_sp = processed_sp_dim_df
    
    # Multi-select for stockpoints
    selected_sp_names = st.sidebar.multiselect(
        "Choose stockpoints:",
        options=[f"{row['stock_point_name']} ({row['stock_point_id']})" 
                for _, row in filtered_sp.iterrows()],
        default=[],
        max_selections=10
    )
    
    # Extract stockpoint IDs
    selected_spids = []
    for name in selected_sp_names:
        spid = int(name.split('(')[-1].rstrip(')'))
        selected_spids.append(spid)
    
    # Visualization options
    st.sidebar.subheader("🎨 Visualization Options")
    
    fill_col = st.sidebar.selectbox(
        "Fill color metric:",
        options=['all_customers', 'active_customers', 'recently_activated_customers'],
        index=0
    )
    
    use_geometry = st.sidebar.checkbox(
        "Use clipped geometry",
        value=True,
        help="Use processed geometry boundaries vs raw coordinates"
    )
    
    # Layer visibility controls
    st.sidebar.subheader("🗂️ Layer Controls")
    show_boundaries = st.sidebar.checkbox("Show Boundaries", value=True)
    show_beats = st.sidebar.checkbox("Show Beats", value=True)
    show_customers = st.sidebar.checkbox("Show Customers", value=False)
    show_markers = st.sidebar.checkbox("Show Stock Point Markers", value=True)
    
    # Main content area
    if not selected_spids:
        st.info("👆 Please select one or more stockpoints from the sidebar to view the map")
        
        # Show available stockpoints table
        st.subheader("📋 Available Stockpoints")
        display_df = processed_sp_dim_df[['stock_point_id', 'stock_point_name', 'latitude', 'longitude']].copy()
        st.dataframe(display_df, use_container_width=True)
        
    else:
        # Create side-by-side layout for Map View and Selection Summary
        col1, col2 = st.columns([3, 1])  # Map takes 3/4, Summary takes 1/4
        
        # Map View (left column)
        with col1:
            st.subheader(f"🗺️ Map View - {len(selected_spids)} Stockpoint(s)")
            
            with st.spinner("🗺️ Generating map..."):
                try:
                    folium_map = create_multi_stockpoint_map(
                        selected_spids, 
                        processed_sp_dim_df,
                        stockpoint_h3_coverage_with_metadata,
                        customer_stockpoint_cluster_assignment_df,
                        sp_territories_dict,
                        fill_col=fill_col,
                        use_geometry=use_geometry,
                        show_boundaries=show_boundaries,
                        show_beats=show_beats,
                        show_customers=show_customers,
                        show_markers=show_markers
                    )
                    
                    # Display map with adjusted size for column layout
                    map_data = st_folium.st_folium(
                        folium_map,
                        width=800,  # Reduced width for column layout
                        height=600,
                        returned_objects=["last_object_clicked"]
                    )
                    
                    # Show clicked object info
                    if map_data['last_object_clicked']:
                        st.info(f"🖱️ Last clicked: {map_data['last_object_clicked']}")
                    
                except Exception as e:
                    st.error(f"Error creating map: {str(e)}")
        
        # Selection Summary (right column)
        with col2:
            display_selection_summary(
                selected_spids, 
                processed_sp_dim_df, 
                customer_stockpoint_cluster_assignment_df,
                stockpoint_h3_coverage_with_metadata
            )
        
        # Show detailed data tables (full width below the side-by-side layout)
        with st.expander("📊 Detailed Data"):
            tab1, tab2, tab3 = st.tabs(["Stockpoints", "Beats", "Customers"])
            
            with tab1:
                selected_sp_data = processed_sp_dim_df[
                    processed_sp_dim_df['stock_point_id'].isin(selected_spids)
                ]
                st.dataframe(selected_sp_data, use_container_width=True)
            
            with tab2:
                beats_data = stockpoint_h3_coverage_with_metadata[
                    stockpoint_h3_coverage_with_metadata['stock_point_id'].isin(selected_spids)
                ]
                if not beats_data.empty:
                    display_beats = beats_data.drop('geometry', axis=1, errors='ignore')
                    st.dataframe(display_beats, use_container_width=True)
                else:
                    st.info("No beats data available for selected stockpoints")
            
            with tab3:
                customers_data = customer_stockpoint_cluster_assignment_df[
                    customer_stockpoint_cluster_assignment_df['stock_point_id'].isin(selected_spids)
                ]
                if not customers_data.empty:
                    st.dataframe(customers_data, use_container_width=True)
                else:
                    st.info("No customer data available for selected stockpoints")

if __name__ == "__main__":
    main()