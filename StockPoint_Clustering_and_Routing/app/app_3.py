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

# Page config
st.set_page_config(
    page_title="Multi-Stockpoint Map Viewer",
    page_icon="🗺️",
    layout="wide"
)

@st.cache_data
def load_cached_data():
    """Load and cache all datasets"""
    return load_data()

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
    
    # Create layer groups for better organization
    layer_groups = {
        'boundaries': FeatureGroup(name="🏛️ All Boundaries"),
        'beats': FeatureGroup(name="📍 All Beats"), 
        'customers': FeatureGroup(name="👥 All Customers"),
        'markers': FeatureGroup(name="🏠 Stock Points")
    }
    
    for i, spid in enumerate(selected_spids):
        color = colors[i % len(colors)]
        
        # Get data for this stockpoint
        sp_dim = processed_sp_dim_df.query(f'stock_point_id == {spid}').iloc[0]
        sp_h3_coverage = stockpoint_h3_coverage_with_metadata.query(f'stock_point_id == {spid}')
        sp_customers = customer_stockpoint_cluster_assignment_df.query(f'stock_point_id == {spid}')
        
        spname = sp_dim['stock_point_name']
        coord_lat, coord_lng = sp_dim['latitude'], sp_dim['longitude']
        
        # Add boundary (fixed geometry handling)
        if show_boundaries and spid in sp_territories_dict:
            try:
                territory_data = sp_territories_dict[spid]
                # Handle different possible formats
                if isinstance(territory_data, dict) and 'polygon' in territory_data:
                    boundary_geom = territory_data['polygon']
                else:
                    boundary_geom = territory_data
                
                # Convert to proper GeoJSON format if needed
                if hasattr(boundary_geom, '__geo_interface__'):
                    boundary_geom = boundary_geom.__geo_interface__
                
                folium.GeoJson(
                    boundary_geom,
                    style_function=lambda x, color=color, spname=spname: {
                        'fillColor': color,
                        'color': color,
                        'weight': 3,
                        'fillOpacity': 0.1,
                        'opacity': 1
                    },
                    popup=folium.Popup(f"<b>{spname}</b><br>Boundary", max_width=200),
                    tooltip=f"{spname} Territory"
                ).add_to(layer_groups['boundaries'])
                
            except Exception as e:
                st.warning(f"Could not plot boundary for {spname}: {str(e)}")
        
        # Add beats
        if show_beats and not sp_h3_coverage.empty:
            try:
                beat_geojson = convert_sp_assigment_df_to_geojson(
                    sp_h3_coverage, use_geometry=use_geometry
                )
                beat_folium_geojson = prepare_beat_folium_GeoJson(
                    beat_geojson, fill_col=fill_col, use_blue=(i % 2 == 1)
                )
                
                # Add beats to map (the prepare_beat_folium_GeoJson already handles styling)
                beat_folium_geojson.add_to(layer_groups['beats'])
                    
            except Exception as e:
                st.warning(f"Could not plot beats for {spname}: {str(e)}")
        
        # Add customers
        if show_customers and not sp_customers.empty:
            try:
                # Sample customers if too many for performance
                if len(sp_customers) > 1000:
                    sp_customers_sample = sp_customers.sample(1000)
                    st.info(f"Showing 1000 random customers out of {len(sp_customers)} for {spname}")
                else:
                    sp_customers_sample = sp_customers
                
                for _, customer in sp_customers_sample.iterrows():
                    folium.CircleMarker(
                        location=[customer['latitude'], customer['longitude']],
                        radius=3,
                        popup=f"<b>Customer {customer.get('customer_id', 'N/A')}</b><br>"
                              f"Status: {customer.get('customer_status', 'N/A')}<br>"
                              f"Type: {customer.get('customer_type', 'N/A')}",
                        color=color,
                        fillColor=color,
                        fillOpacity=0.6,
                        weight=1
                    ).add_to(layer_groups['customers'])
                    
            except Exception as e:
                st.warning(f"Could not plot customers for {spname}: {str(e)}")
        
        # Add stockpoint marker
        if show_markers:
            folium.Marker(
                location=[coord_lat, coord_lng],
                popup=folium.Popup(
                    f"<b>{spname}</b><br>"
                    f"ID: {spid}<br>"
                    f"Coordinates: {coord_lat:.4f}, {coord_lng:.4f}",
                    max_width=250
                ),
                tooltip=f"📍 {spname}",
                icon=folium.Icon(color=color, icon='home')
            ).add_to(layer_groups['markers'])
    
    # Add layer groups to map in order
    for group in layer_groups.values():
        if len(group._children) > 0:  # Only add if it has content
            group.add_to(m)
    
    # Add controls
    m.add_child(MeasureControl(
        primary_length_unit='kilometers',
        secondary_length_unit='meters',
        primary_area_unit='sqmeters',
        secondary_area_unit='acres'
    ))
    
    # Add improved layer control
    LayerControl(
        position='topright',
        collapsed=False,
        autoZIndex=True
    ).add_to(m)
    
    return m

def main():
    st.title("🗺️ Multi-Stockpoint Territory Viewer")
    
    # Load data with error handling
    try:
        with st.spinner("Loading data..."):
            processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, \
            customer_stockpoint_cluster_assignment_df, sp_territories_dict = load_cached_data()
            
        st.success(f"✅ Loaded {len(processed_sp_dim_df)} stockpoints")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please check your data files and try again.")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Map Controls")
    
    # Stockpoint selection
    st.sidebar.subheader("Select Stockpoints")
    
    # Quick selection options
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", type="secondary"):
            st.session_state.selected_all = True
    with col2:
        if st.button("Clear All", type="secondary"):
            st.session_state.selected_all = False
    
    # Search stockpoints
    search_term = st.sidebar.text_input(
        "🔍 Search stockpoints", 
        placeholder="Enter name or ID..."
    )
    
    # Filter stockpoints based on search
    if search_term:
        mask = (
            processed_sp_dim_df['stock_point_name'].str.contains(search_term, case=False, na=False) |
            processed_sp_dim_df['stock_point_id'].astype(str).str.contains(search_term, na=False)
        )
        filtered_sp = processed_sp_dim_df[mask]
        st.sidebar.info(f"Found {len(filtered_sp)} matches")
    else:
        filtered_sp = processed_sp_dim_df
    
    # Multi-select for stockpoints
    options = [f"{row['stock_point_name']} ({row['stock_point_id']})" 
               for _, row in filtered_sp.iterrows()]
    
    # Handle "Select All" safely
    max_selections = 75
    if st.session_state.get('selected_all', False) and len(options) <= max_selections:
        default_selection = options
    elif st.session_state.get('selected_all', False):
        default_selection = options[:max_selections]
        st.sidebar.warning(f"Limited to first {max_selections} stockpoints")
    else:
        default_selection = []
    
    selected_sp_names = st.sidebar.multiselect(
        "Choose stockpoints:",
        options=options,
        default=default_selection,
        max_selections=max_selections
    )
    
    # Extract stockpoint IDs
    selected_spids = []
    for name in selected_sp_names:
        try:
            spid = int(name.split('(')[-1].rstrip(')'))
            selected_spids.append(spid)
        except ValueError:
            st.sidebar.error(f"Invalid stockpoint selection: {name}")
    
    # Visualization options
    st.sidebar.subheader("🎨 Visualization Options")
    
    fill_col = st.sidebar.selectbox(
        "Fill color metric:",
        options=['all_customers', 'active_customers', 'recently_activated_customers'],
        index=0,
        help="Metric used to color-code map regions"
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
    show_customers = st.sidebar.checkbox(
        "Show Customers", 
        value=False,
        help="⚠️ May impact performance with many customers"
    )
    show_markers = st.sidebar.checkbox("Show Stock Point Markers", value=True)
    
    # Performance warning
    if show_customers and len(selected_spids) > 3:
        st.sidebar.warning("⚠️ Showing customers for many stockpoints may be slow")
    
    # Display selection info
    if selected_spids:
        st.sidebar.success(f"✅ {len(selected_spids)} stockpoint(s) selected")
        
        # Show summary stats
        with st.sidebar.expander("📊 Selection Summary", expanded=True):
            total_customers = customer_stockpoint_cluster_assignment_df[
                customer_stockpoint_cluster_assignment_df['stock_point_id'].isin(selected_spids)
            ]
            total_coverage = stockpoint_h3_coverage_with_metadata[
                stockpoint_h3_coverage_with_metadata['stock_point_id'].isin(selected_spids)
            ]
            
            # Metrics in a compact layout
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Customers", f"{len(total_customers):,}")
                if not total_customers.empty:
                    active_count = len(total_customers[total_customers['customer_status'] == 'Active'])
                    st.metric("Active", f"{active_count:,}")
            
            with col2:
                if not total_customers.empty:
                    recent_count = len(total_customers[total_customers['customer_type'] == 'recently activated'])
                    st.metric("Recently Activated", f"{recent_count:,}")
                st.metric("Total Beats", f"{len(total_coverage):,}")
            
            if not total_coverage.empty:
                total_area = total_coverage['area_km2'].sum()
                st.metric("Total Area", f"{total_area:.0f} km²")
    
    # Main content area
    if not selected_spids:
        st.info("👆 Please select one or more stockpoints from the sidebar to view the map")
        
        # Show available stockpoints table
        st.subheader("📋 Available Stockpoints")
        display_df = processed_sp_dim_df[
            ['stock_point_id', 'stock_point_name', 'latitude', 'longitude']
        ].copy()
        
        # Add search functionality to main table
        if search_term:
            display_df = display_df[
                display_df['stock_point_name'].str.contains(search_term, case=False, na=False) |
                display_df['stock_point_id'].astype(str).str.contains(search_term, na=False)
            ]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "stock_point_id": st.column_config.NumberColumn("ID", format="%d"),
                "stock_point_name": st.column_config.TextColumn("Name"),
                "latitude": st.column_config.NumberColumn("Latitude", format="%.4f"),
                "longitude": st.column_config.NumberColumn("Longitude", format="%.4f")
            }
        )
        
    else:
        # Create and display map
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
                
                # Display map with improved configuration
                st.subheader(f"🗺️ Map View - {len(selected_spids)} Stockpoint(s)")
                
                map_data = st_folium.st_folium(
                    folium_map,
                    width=1400,
                    height=700,
                    returned_objects=["last_object_clicked"]
                )
                
                # Show clicked object info
                if map_data.get('last_object_clicked'):
                    with st.expander("🖱️ Last Clicked Object", expanded=False):
                        st.json(map_data['last_object_clicked'])
                
            except Exception as e:
                st.error(f"❌ Error creating map: {str(e)}")
                st.info("Please check your data and try again.")
        
        # Show detailed data tables
        with st.expander("📊 Detailed Data Tables"):
            tab1, tab2, tab3 = st.tabs(["📍 Stockpoints", "🏘️ Beats", "👥 Customers"])
            
            with tab1:
                selected_sp_data = processed_sp_dim_df[
                    processed_sp_dim_df['stock_point_id'].isin(selected_spids)
                ]
                st.dataframe(selected_sp_data, use_container_width=True, hide_index=True)
            
            with tab2:
                beats_data = stockpoint_h3_coverage_with_metadata[
                    stockpoint_h3_coverage_with_metadata['stock_point_id'].isin(selected_spids)
                ]
                if not beats_data.empty:
                    # Drop geometry column for display
                    display_beats = beats_data.drop(columns=['geometry'], errors='ignore')
                    st.dataframe(display_beats, use_container_width=True, hide_index=True)
                    
                    # Show summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Beats", len(beats_data))
                    with col2:
                        if 'area_km2' in beats_data.columns:
                            st.metric("Total Area", f"{beats_data['area_km2'].sum():.1f} km²")
                    with col3:
                        if fill_col in beats_data.columns:
                            st.metric(f"Total {fill_col.replace('_', ' ').title()}", 
                                     f"{beats_data[fill_col].sum():,}")
                else:
                    st.info("No beats data available for selected stockpoints")
            
            with tab3:
                customers_data = customer_stockpoint_cluster_assignment_df[
                    customer_stockpoint_cluster_assignment_df['stock_point_id'].isin(selected_spids)
                ]
                if not customers_data.empty:
                    # Show sample if too many rows
                    if len(customers_data) > 1000:
                        st.info(f"Showing first 1000 rows out of {len(customers_data):,} total customers")
                        display_customers = customers_data.head(1000)
                    else:
                        display_customers = customers_data
                    
                    st.dataframe(display_customers, use_container_width=True, hide_index=True)
                    
                    # Customer summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", f"{len(customers_data):,}")
                    with col2:
                        active_customers = customers_data[customers_data['customer_status'] == 'Active']
                        st.metric("Active Customers", f"{len(active_customers):,}")
                    with col3:
                        recent_customers = customers_data[customers_data['customer_type'] == 'recently activated']
                        st.metric("Recently Activated", f"{len(recent_customers):,}")
                else:
                    st.info("No customer data available for selected stockpoints")

if __name__ == "__main__":
    main()