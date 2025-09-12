import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import MeasureControl
import streamlit_folium as st_folium
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent #.parent 
sys.path.append(str(parent_dir))
map_input_dir = parent_dir / 'data' 

from src.viz.viz import (
    convert_sp_assigment_df_to_geojson,
    convert_customers_to_geojson,
    prepare_beat_folium_GeoJson,
    prepare_customer_assignment_GeoJson
)

# -------------------------
# st.set_page_config(
#     page_title="MFC Territory Management Dashboard",
#     page_icon="🏭",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': None,
#         'Report a bug': None,
#         'About': "MFC Territory Management Dashboard v2.0"
#     }
# )

# # Hide deploy button and menu items for production look
# hide_streamlit_style = """
# <style>
# /* Hide deploy button - multiple selectors for different versions */
# .stDeployButton {display: none !important;}
# button[kind="header"] {display: none !important;}
# [data-testid="stHeader"] button {display: none !important;}
# .stApp > header {display: none !important;}
# .stApp div[data-testid="stToolbar"] {display: none !important;}

# /* Hide main menu */
# #MainMenu {visibility: hidden !important;}

# /* Keep sidebar visible */
# .css-1d391kg, .css-1lcbmhc, .css-1v3fvcr {display: block !important;}
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import streamlit as st

# Set page config
st.set_page_config(
    page_title="MFC Territory Management Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "MFC Territory Management Dashboard v2.0"
    }
)

# CSS to hide Streamlit's default UI elements for a cleaner look
hide_streamlit_style = """
<style>
/* Hide deploy button and header buttons */
.stDeployButton, button[kind="header"], [data-testid="stHeader"] button {display: none !important;}
/* Hide the Streamlit header and toolbar */
.stApp > header, .stApp div[data-testid="stToolbar"] {display: none !important;}
/* Hide main menu */
#MainMenu {visibility: hidden !important;}
/* Ensure sidebar remains visible */
.css-1d391kg, .css-1lcbmhc, .css-1v3fvcr {display: block !important;}
/* Optional: Adjust padding/margin if needed */
.stApp {padding-top: 0 !important;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# -------------------------
# Local data loading function
def load_from_local():
    import os 
    import gzip
    import pickle 
    
    map_input_dir = parent_dir / 'data'
    files = {
        'processed_sp_dim_df': 'processed_sp_dim_df.pkl.gz',
        'stockpoint_h3_coverage_with_metadata': 'stockpoint_h3_coverage_with_metadata.pkl.gz',
        'customer_stockpoint_cluster_assignment_df': 'customer_stockpoint_cluster_assignment_df.pkl.gz',
        'agent_customer_mfc_df': 'agent_customer_mfc.pkl.gz', 
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
        loaded_data['agent_customer_mfc_df'],
        loaded_data['sp_territories_dict']
    )

# Compute Summary Statistics
def compute_sps_summaries(stockpoint_h3_coverage_with_metadata, customer_stockpoint_cluster_assignment_df):
    # Precompute groupings
    beat_stats = (stockpoint_h3_coverage_with_metadata
        .groupby(['stock_point_id','beat']).agg(total_customers=('n_total_assigned_customers','sum'),
            total_active_customers=('n_assigned_active_customers','sum'))
        .groupby('stock_point_id').mean().apply(np.ceil) .astype(int)
        .rename(columns={'total_customers': 'avg_customers_per_beat', 'total_active_customers': 'avg_active_customers_per_beat'})
        .reset_index())

    # OAM metrics
    n_oams = (customer_stockpoint_cluster_assignment_df
        .groupby('stock_point_id')['Agent_ID'].nunique().reset_index(name='n_oams'))

    beats_per_oam = (customer_stockpoint_cluster_assignment_df
        .groupby(['stock_point_id','Agent_ID'])['h3_cell_id'].nunique()
        .groupby('stock_point_id').mean().apply(np.ceil).astype(int).rename('beats_per_oam').reset_index())

    active_customers_per_oam = (customer_stockpoint_cluster_assignment_df
        .query('assignment_type_id in (1,2)')
        .groupby(['stock_point_id','Agent_ID'])['customer_id'].nunique()
        .groupby('stock_point_id').mean().apply(np.ceil).astype(int).rename('active_customers_per_oam').reset_index())

    oam_stats = n_oams.merge(beats_per_oam, on='stock_point_id').merge(active_customers_per_oam, on='stock_point_id')

    # Stock_point-level totals
    totals = (customer_stockpoint_cluster_assignment_df
        .groupby('stock_point_id').agg(
            total_customers=('customer_id','nunique'),
            active_customers=('customer_id', lambda x: x[customer_stockpoint_cluster_assignment_df.loc[x.index,'customer_status'] == 'Active'].nunique()),
            recently_activated=('customer_id', lambda x: x[customer_stockpoint_cluster_assignment_df.loc[x.index,'customer_type'] == 'recently activated'].nunique()),
            ).reset_index())

    total_area_and_beat = (stockpoint_h3_coverage_with_metadata
        .groupby('stock_point_id')
        .agg(
            total_area=('area_km2', 'sum'),
            total_beats=('beat', 'nunique')
        )
        .round({'total_area': 2})
        .reset_index()
    )

    # Merge everything
    final_stats = (beat_stats
        .merge(oam_stats, on='stock_point_id')
        .merge(totals, on='stock_point_id')
        .merge(total_area_and_beat, on='stock_point_id')).fillna(0)
    
    return final_stats

@st.cache_data
def load_cached_data():
    """Load and cache all datasets""" 
    return load_from_local()

def create_multi_stockpoint_map(selected_spids, processed_sp_dim_df, 
                               stockpoint_h3_coverage_with_metadata,
                               customer_stockpoint_cluster_assignment_df, 
                               sp_territories_dict,
                               fill_col="all_customers", 
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
        zoom_start=11
    )
    
    # Professional color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5D737E', 
              '#4A4E69', '#9A031E', '#0F4C75', '#3282B8', '#BBE1FA']
    
    for i, spid in enumerate(selected_spids):
        color_hex = colors[i % len(colors)]
        folium_color = ['blue', 'purple', 'orange', 'red', 'gray', 
                       'darkblue', 'darkred', 'darkblue', 'blue', 'lightblue'][i % 10]
        
        # Get data for this stockpoint
        sp_dim = processed_sp_dim_df.query(f'stock_point_id == {spid}').iloc[0]
        sp_h3_coverage = stockpoint_h3_coverage_with_metadata.query(f'stock_point_id == {spid}')
        sp_customers = customer_stockpoint_cluster_assignment_df.query(f'stock_point_id == {spid}')
        
        spname = sp_dim['stock_point_name']
        coord_lat, coord_lng = sp_dim['latitude'], sp_dim['longitude']
        
        # Create feature groups for this stockpoint
        fg_boundary = FeatureGroup(name=f"📍 {spname} - Coverage")
        fg_beats = FeatureGroup(name=f"🗺️ {spname} - Beats")
        fg_customers = FeatureGroup(name=f"👥 {spname} - Customers")
        fg_marker = FeatureGroup(name=f"🏭 {spname} - MFC")
        
        # Add boundary (conditionally)
        if show_boundaries and spid in sp_territories_dict:
            boundary_polygon = sp_territories_dict[spid]['polygon']
            
            if boundary_polygon:
                folium.GeoJson(
                    boundary_polygon, 
                    style_function=lambda x, color=color_hex: {
                        'fillColor': color,
                        'color': color,
                        'weight': 3,
                        'fillOpacity': 0.2,
                        'opacity': 0.8
                    },
                    tooltip=f"{spname} - Coverage Boundary"
                ).add_to(fg_boundary)
        
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
        
        # Add stockpoint marker (conditionally) - Professional icons
        if show_markers:
            folium.Marker(
                location=[coord_lat, coord_lng],
                popup=folium.Popup(
                    f"""
                    <div style="font-family: Arial; width: 200px;">
                        <h4 style="color: {color_hex}; margin-bottom: 10px;">🏭 {spname}</h4>
                        <p><strong>MFC ID:</strong> {spid}</p>
                        <p><strong>Location:</strong> {coord_lat:.4f}, {coord_lng:.4f}</p>
                    </div>
                    """,
                    max_width=250
                ),
                tooltip=f"🏭 {spname}",
                icon=folium.Icon(
                    color=folium_color, 
                    icon='industry',  # More professional icon
                    prefix='fa'
                )
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
                            stockpoint_h3_coverage_with_metadata, 
                            agent_customer_mfc_df,
                            sps_summaries=None):
    """Display selection summary in a formatted container"""
    
    st.markdown("### 📊 Coverage Overview")
    
    # Professional success message
    st.success(f"✅ **{len(selected_spids)}** MFC{'s' if len(selected_spids) != 1 else ''} selected")
    
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
    
    # Customer metrics with better formatting
    st.markdown("**📈 Performance Metrics**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Customers", f"{len(total_customers):,}")        
        active_count = len(total_customers[total_customers['customer_status'] == 'Active'])
        st.metric("Active Customers", f"{int(active_count):,}")
        recent_count = len(total_customers[total_customers['customer_type'] == 'recently activated'])
        st.metric("Recently Activated", f"{int(recent_count):,}")
        st.metric("Total Beats", f"{int(len(total_coverage)):,}")        
    
    with col2:
        # Safe metric calculations with null checks
        if sps_summaries is not None and not sps_summaries.empty:
            selected_summaries = sps_summaries[sps_summaries['stock_point_id'].isin(selected_spids)]
            
            avg_customers_per_beat = selected_summaries['avg_customers_per_beat'].mean() if not selected_summaries.empty else 0
            st.metric("Avg. Customers/Beat", f"{int(avg_customers_per_beat):,}")
            
            avg_active_customers_per_beat = selected_summaries['avg_active_customers_per_beat'].mean() if not selected_summaries.empty else 0
            st.metric("Avg. Active/Beat", f"{int(avg_active_customers_per_beat):,}")
            
            avg_beats_per_oam = selected_summaries['beats_per_oam'].mean() if not selected_summaries.empty else 0
            st.metric("Avg. Beats/Agent", f"{int(avg_beats_per_oam):,}")
        else:
            st.metric("Avg. Customers/Beat", "0")
            st.metric("Avg. Active/Beat", "0")
            st.metric("Avg. Beats/Agent", "0") 
            
        if not agent_customer_mfc_df.empty:
            agent_filtered = agent_customer_mfc_df[
                agent_customer_mfc_df['stock_point_id'].isin(selected_spids)
            ]
            
            # Check which agent ID column exists
            agent_col = None
            for col in ['Agent_ID', 'agent_id', 'Agent_Id']:
                if col in agent_customer_mfc_df.columns:
                    agent_col = col
                    break
            
            if agent_col and not agent_filtered.empty:
                n_agents = agent_filtered[agent_col].nunique()
            else:
                n_agents = 0
        else:
            n_agents = 0
            
        st.metric("Active Agents", f"{int(n_agents):,}")
    
    if not total_coverage.empty:
        st.metric("**Total Coverage**", f"{total_coverage['area_km2'].sum():.0f} km²")
    
    # Selected stockpoints list with better formatting
    st.markdown("**🏭 Selected Distribution MFCs**")
    for _, row in selected_data.iterrows():
        st.markdown(f"• **{row['stock_point_name']}** `(ID: {row['stock_point_id']})`")
        
def main():
    # Professional header
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 2px solid #f0f0f0; margin-bottom: 2rem;">
        <h1 style="color: #2E86AB; margin: 0;">🏭 MFC Coverage Management Dashboard</h1>
        <p style="color: #666; margin: 0.5rem 0 0 0;">Interactive mapping and analytics for distribution coverage optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with progress indicator
    try:
        with st.spinner("🔄 Loading coverage data..."):
            processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, \
            customer_stockpoint_cluster_assignment_df,  agent_customer_mfc_df, sp_territories_dict = load_cached_data()     
            
            customer_stockpoint_cluster_assignment_df = (customer_stockpoint_cluster_assignment_df
                                .merge(processed_sp_dim_df[['stock_point_id', 'stock_point_name']], on='stock_point_id', how='left')
                                .merge(stockpoint_h3_coverage_with_metadata[['stock_point_id', 'beat','beat_id']].rename({'beat':'h3_cell_id'},axis=1), on=['stock_point_id','h3_cell_id'], how='left')
                            )   
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
        
    # Compute sps Summary Statistics
    try:
        with st.spinner("📊 Computing analytics..."):
            sps_summaries = compute_sps_summaries(stockpoint_h3_coverage_with_metadata, customer_stockpoint_cluster_assignment_df)     
    except Exception as e:
        st.error(f"❌ Error computing analytics: {str(e)}")
        return
    
    # Sidebar controls with professional styling
    st.sidebar.markdown("## ⚙️ Control Panel")
    
    # Stockpoint selection
    st.sidebar.markdown("### 📍 MFC Selection")
    
    # Search stockpoints
    search_term = st.sidebar.text_input(
        "🔍 Search Distribution MFCs", 
        placeholder="Enter MFC name or ID...",
        help="Search by name or ID to filter available MFCs"
    )
    
    # Filter stockpoints based on search
    if search_term:
        filtered_sp = processed_sp_dim_df[
            processed_sp_dim_df['stock_point_name'].str.contains(search_term, case=False) |
            processed_sp_dim_df['stock_point_id'].astype(str).str.contains(search_term)
        ]
    else:
        filtered_sp = processed_sp_dim_df
    
    # Multi-select for stockpoints with better formatting
    selected_sp_names = st.sidebar.multiselect(
        "Select Distribution MFCs:",
        options=[f"{row['stock_point_name']} (ID: {row['stock_point_id']})" 
                for _, row in filtered_sp.iterrows()],
        default=[],
        max_selections=10,
        help="Select up to 10 MFCs for analysis"
    )
    
    # Extract stockpoint IDs
    selected_spids = []
    for name in selected_sp_names:
        spid = int(name.split('(ID: ')[-1].rstrip(')'))
        selected_spids.append(spid)
    
    # Visualization options
    st.sidebar.markdown("### 🎨 Visualization Settings")
    
    fill_col = st.sidebar.selectbox(
        "Color mapping:",
        options=['all_customers', 'active_customers', 'recently_activated_customers'],
        index=0,
        format_func=lambda x: {
            'all_customers': '👥 Total Customers',
            'active_customers': '✅ Active Customers',
            'recently_activated_customers': '🆕 Recently Activated'
        }.get(x, x)
    )
    
    use_geometry = st.sidebar.checkbox(
        "🗺️ Enhanced boundaries",
        value=True,
        help="Use processed geographic boundaries for better visualization"
    )
    
    # Layer visibility controls
    st.sidebar.markdown("### 🗂️ Map Layers")
    show_boundaries = st.sidebar.checkbox("🏢 Coverage Boundaries", value=True)
    show_beats = st.sidebar.checkbox("📍 Beats", value=True)
    show_customers = st.sidebar.checkbox("👥 Customer Locations", value=False)
    show_markers = st.sidebar.checkbox("🏭 MFC Markers", value=True)
    
    # Add help section
    with st.sidebar.expander("ℹ️ Help & Tips"):
        st.markdown("""
        **Navigation:**
        - Use mouse wheel to zoom
        - Click and drag to pan
        - Use layer controls on map
        
        **Features:**
        - Territory boundaries show coverage areas
        - Beats show operational zones  
        - Measure tool for distance/area calculations
        """)
    
    # Main content area
    if not selected_spids:
        # Landing page with better formatting
        st.info("👆 **Getting Started:** Select one or more distribution MFCs from the sidebar to begin coverage analysis")
        
        # Show available stockpoints table with better styling
        st.markdown("### 📋 Available Distribution MFCs")
        display_df = processed_sp_dim_df[['stock_point_id', 'stock_point_name', 'latitude', 'longitude']].copy()
        display_df.columns = ['MFC ID', 'MFC Name', 'Latitude', 'Longitude']
        
        st.dataframe(
            display_df, 
            use_container_width=True,
            column_config={
                "MFC ID": st.column_config.NumberColumn(format="%d"),
                "Latitude": st.column_config.NumberColumn(format="%.4f"),
                "Longitude": st.column_config.NumberColumn(format="%.4f")
            }
        )
        
    else:
        # Create side-by-side layout for Map View and Selection Summary 
        col1, col2 = st.columns([7, 3], gap="medium")  # Adjusted ratio for better balance

        # Map View        
        with col1:
            st.markdown(f"### 🗺️ Coverage Map - {len(selected_spids)} MFC{'s' if len(selected_spids) != 1 else ''}")
            
            with st.spinner("🗺️ Rendering interactive map..."):
                try:
                    # Initialize session state for map to prevent regeneration
                    map_key = f"map_{hash(tuple(selected_spids))}{fill_col}{use_geometry}{show_boundaries}{show_beats}{show_customers}{show_markers}"
                    
                    if map_key not in st.session_state:
                        st.session_state[map_key] = create_multi_stockpoint_map(
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
                    
                    # Display map with click handling disabled to prevent regeneration
                    map_data = st_folium.st_folium(
                        st.session_state[map_key], 
                        use_container_width=True,
                        height=650,
                        returned_objects=["last_clicked"],  # Minimal return to prevent regeneration
                        key=f"folium_map_{map_key}"  # Unique key to prevent regeneration
                    )
                     
                except Exception as e:
                    st.error(f"❌ Error rendering map: {str(e)}")
        
        # Selection Summary (right column)
        with col2:
            display_selection_summary(selected_spids, 
                                      processed_sp_dim_df, 
                            customer_stockpoint_cluster_assignment_df,
                            stockpoint_h3_coverage_with_metadata, 
                            agent_customer_mfc_df,
                            sps_summaries)
        
        # Show detailed data tables (full width below the side-by-side layout)
        with st.expander("📊 Detailed Analytics & Data Tables"):
            tab1, tab2, tab3, tab4 = st.tabs(["🏭 MFCs", "📍 Beats", "👥 Customers", "👨‍💼 Agents"])
            
            with tab1:
                selected_sp_data = processed_sp_dim_df[
                    processed_sp_dim_df['stock_point_id'].isin(selected_spids)
                ]
                st.dataframe(selected_sp_data, use_container_width=True)
            
            with tab2:
                beats_data = stockpoint_h3_coverage_with_metadata[
                    stockpoint_h3_coverage_with_metadata['stock_point_id'].isin(selected_spids)
                ].round({'area_km2': 2, 'cluster_sp_dist_km': 2}).drop('latlng_coords',axis=1, errors='ignore')
                if not beats_data.empty:
                    display_beats = beats_data.drop('geometry', axis=1, errors='ignore')
                    st.dataframe(display_beats, use_container_width=True)
                else:
                    st.info("📍 No service area data available for selected MFCs")
            
            with tab3:
                customers_data = customer_stockpoint_cluster_assignment_df[
                    customer_stockpoint_cluster_assignment_df['stock_point_id'].isin(selected_spids)
                ]
                if not customers_data.empty:
                    st.dataframe(customers_data, use_container_width=True)
                else:
                    st.info("👥 No customer data available for selected MFCs")
            
            with tab4:
                mfc_agents = agent_customer_mfc_df[
                    agent_customer_mfc_df['stock_point_id'].isin(selected_spids)
                ]
                if not mfc_agents.empty:
                    st.dataframe(mfc_agents, use_container_width=True)
                else:
                    st.info("👨‍💼 No agent data available for selected MFCs")
            
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "MFC Coverage Management Dashboard v2.0 | Built by Data Team (OmniRetail)"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()