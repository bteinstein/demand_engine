import streamlit as st
import folium
from folium import Map, FeatureGroup, LayerControl, Marker
from streamlit_folium import st_folium
import pandas as pd

def create_stockpoint_map(spids, processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, 
                         customer_stockpoint_cluster_assignment_df, sp_territories_dict,
                         fill_col="all_customers", use_geometry=True):
    """Create multi-stockpoint folium map"""
    
    # Get data for all selected stock points
    sp_dims = processed_sp_dim_df.query(f'stock_point_id in {spids}').reset_index()
    
    if sp_dims.empty:
        return None
    
    # Calculate center from all selected points
    center_lat = sp_dims['latitude'].mean()
    center_lng = sp_dims['longitude'].mean()
    
    # Create base map
    base_map = folium.Map(location=[center_lat, center_lng],
                          tiles="CartoDB Positron",
                          prefer_canvas=True,
                          zoom_start=8)
    
    # Create layers for each stock point
    for spid in spids:
        sp_dim = sp_dims.query(f'stock_point_id == {spid}')
        if sp_dim.empty:
            continue
            
        spname = sp_dim['stock_point_name'].iloc[0]
        coord_lat, coord_lng = sp_dim['latitude'].iloc[0], sp_dim['longitude'].iloc[0]
        
        # Get data for this stock point
        sp_h3_coverage = stockpoint_h3_coverage_with_metadata.query(f'stock_point_id == {spid}').reset_index()
        sp_customer_assignment = customer_stockpoint_cluster_assignment_df.query(f'stock_point_id == {spid}').reset_index()
        
        # Skip if no data
        if sp_h3_coverage.empty:
            continue
            
        sp_boundary_geojson = sp_territories_dict.get(spid, {}).get('polygon')
        sp_beat_geojson = convert_sp_assigment_df_to_geojson(sp_h3_coverage, use_geometry=use_geometry)
        customer_geojson = convert_customers_to_geojson(sp_customer_assignment)
        
        # Create feature groups for this stock point
        if sp_boundary_geojson:
            fg_boundary = FeatureGroup(name=f"{spname} - Boundary")
            folium.GeoJson(sp_boundary_geojson).add_to(fg_boundary)
            fg_boundary.add_to(base_map)
        
        if not sp_h3_coverage.empty:
            fg_beats = FeatureGroup(name=f"{spname} - Beats")
            beat_folium_GeoJson = prepare_beat_folium_GeoJson(sp_beat_geojson, fill_col=fill_col)
            beat_folium_GeoJson.add_to(fg_beats)
            fg_beats.add_to(base_map)
        
        if not sp_customer_assignment.empty:
            fg_customers = FeatureGroup(name=f"{spname} - Customers")
            customer_geojson_folium = prepare_customer_assignment_GeoJson(customer_geojson)
            customer_geojson_folium.add_to(fg_customers)
            fg_customers.add_to(base_map)
        
        # Add stock point marker
        fg_marker = FeatureGroup(name=f"{spname} - Marker")
        folium.Marker(
            location=[coord_lat, coord_lng],
            popup=spname,
            icon=folium.Icon(color="green")
        ).add_to(fg_marker)
        fg_marker.add_to(base_map)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(base_map)
    return base_map

def main():
    st.set_page_config(page_title="Stock Point Visualization", layout="wide")
    
    st.title("Stock Point Territory Visualization")
    
    @st.cache_data
    def load_data():
        """Load all required datasets"""
        import duckdb
        import pandas as pd
        import geopandas as gpd
        import pickle
        import ast 
        from config.settings import EXPORTS_DIR, OUTPUT_DIR, STORAGE_CONFIG, PROCESSED_DATA_DIR

        H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']

        # Load processed stock point dimensions
        processed_sp_dim_filepath = PROCESSED_DATA_DIR / 'processed_sp_dim_df.pickle'
        with open(processed_sp_dim_filepath, 'rb') as f:
            processed_sp_dim_df = pickle.load(f)
        
        # Load clustering results
        current_date = '2025-08-15'
        suffix = "ALL" 
        resolution = 8
        cluster_results_filepath = EXPORTS_DIR / 'clustering' / f"{suffix}_SPS_CLUSTER_R{resolution}_{current_date}.pickle"
        
        with open(cluster_results_filepath, 'rb') as f:
            cluster_results = pickle.load(f)
        
        sp_territories_dict = cluster_results.get('territories')
        sp_clusters_dict = cluster_results.get('grid_results')
        
        # Process clipped cells
        all_cluster_grid_list = []
        for key, value in sp_clusters_dict.items():
            cell_geometries = value.get('cell_geometries')  
            if cell_geometries:
                cell_geometries_gpd = pd.DataFrame.from_dict(cell_geometries, orient='index').reset_index()
                cell_geometries_gpd.columns = ['h3_cell','geometry']
                cell_geometries_gpd = gpd.GeoDataFrame(cell_geometries_gpd)
                cell_geometries_gpd['stock_point_id'] = int(key)
                all_cluster_grid_list.append(cell_geometries_gpd)
        
        all_cluster_clipped_grid_list_df = (
            pd.concat(all_cluster_grid_list, ignore_index=True)
            if all_cluster_grid_list
            else gpd.GeoDataFrame(columns=['h3_cell', 'geometry', 'stock_point_id'])
        )

        # Load customer assignments
        with duckdb.connect(H3_DUCKDB_PATH) as conn: 
            customer_stockpoint_cluster_assignment_df = conn.execute('''
                SELECT 
                    stock_point_id, a.customer_id, h3_cell_id, customer_type, previous_cluster_id,
                    CASE WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'buying customers' THEN 1
                        WHEN assignment_tier = 'manual_review' AND customer_type = 'buying customers' THEN 2
                        WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'recently activated' THEN 3
                    ELSE 99 END AS assignment_type_id,
                    CASE WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'buying customers' THEN 'Assigned Active/Buying'
                        WHEN assignment_tier = 'manual_review' AND customer_type = 'buying customers' THEN 'Unassigned Active/Buying'
                        WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'recently activated' THEN 'Assigned Recently Activated'
                    ELSE 'Others' END AS assignment_type, 	
                    contact_name, state_name, town_name, city_name, latitude, longitude, kyc_capture_status, customer_status
                FROM customer_stockpoint_cluster_assignment a
                LEFT JOIN read_parquet('/home/bt/project/demand_engine/StockPoint_Clustering_and_Routing/data/processed/df_processed_customer_dim.parquet') d 
                    ON d.customer_id = a.customer_id     
                ''').df()

        # Load H3 coverage with metadata
        with duckdb.connect(H3_DUCKDB_PATH) as conn: 
            stockpoint_h3_coverage_with_metadata = conn.execute("""
                WITH CTE_Assignment_Summary AS(
                    SELECT 
                        stock_point_id, h3_cell_id as h3_cell, 
                        COUNT(DISTINCT customer_id) as n_total_assigned_customers,
                        CAST(SUM(CASE WHEN assignment_type_id = 1 THEN 1 ELSE 0 END) AS INT) AS n_assigned_active_customers,
                        CAST(SUM(CASE WHEN assignment_type_id = 3 THEN 1 ELSE 0 END) AS INT) AS n_assigned_recent_activated_customers
                    FROM customer_stockpoint_cluster_assignment_df  
                    WHERE h3_cell_id NOT NULL
                    GROUP BY stock_point_id, h3_cell_id 
                )    
                SELECT 
                    c.stock_point_id, c.h3_cell as beat, primary_address_id as beat_id,
                    h.state_name, h.lga_name, h.ward_name, h.area_km2, h.confidence_level, h.latlng_json as latlng_coords,  
                    c.cluster_sp_dist_km,
                    COALESCE(s.n_total_assigned_customers, 0) AS n_total_assigned_customers, 
                    COALESCE(s.n_assigned_active_customers, 0) AS n_assigned_active_customers, 
                    COALESCE(s.n_assigned_recent_activated_customers, 0) AS n_assigned_recent_activated_customers
                FROM stockpoint_h3_coverage c
                LEFT JOIN CTE_Assignment_Summary s ON c.stock_point_id = s.stock_point_id AND c.h3_cell = s.h3_cell         
                LEFT JOIN h3_cells h ON c.h3_cell = h.h3_index              
                """).df()
        
        # Process H3 coverage data
        stockpoint_h3_coverage_with_metadata = gpd.GeoDataFrame(stockpoint_h3_coverage_with_metadata)
        stockpoint_h3_coverage_with_metadata['latlng_coords'] = stockpoint_h3_coverage_with_metadata['latlng_coords'].apply(lambda x: ast.literal_eval(x))
        stockpoint_h3_coverage_with_metadata = stockpoint_h3_coverage_with_metadata.merge(
            all_cluster_clipped_grid_list_df.rename(columns={'h3_cell': 'beat'}), 
            on=['beat', 'stock_point_id'], how='left'
        )
        
        return (processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, 
                customer_stockpoint_cluster_assignment_df, sp_territories_dict)
    
    # Sidebar controls
    st.sidebar.header("Filters")
    
    # Load data
    try:
        (processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, 
         customer_stockpoint_cluster_assignment_df, sp_territories_dict) = load_data()
        
        # Get unique stock point names
        sp_options = processed_sp_dim_df[['stock_point_id', 'stock_point_name']].drop_duplicates()
        sp_options['display_name'] = sp_options['stock_point_name'] + f" (ID: {sp_options['stock_point_id']})"
        
        # Multi-select for stock points
        selected_names = st.sidebar.multiselect(
            "Select Stock Points:",
            options=sp_options['display_name'].tolist(),
            default=sp_options['display_name'].iloc[:3].tolist() if len(sp_options) >= 3 else sp_options['display_name'].tolist()
        )
        
        # Get corresponding IDs
        selected_spids = []
        for name in selected_names:
            spid = sp_options[sp_options['display_name'] == name]['stock_point_id'].iloc[0]
            selected_spids.append(spid)
        
        # Fill column selection
        fill_cols = ["all_customers", "active_customers", "recently_activated_customers"]
        fill_col = st.sidebar.selectbox("Fill Column:", fill_cols)
        
        # Geometry option
        use_geometry = st.sidebar.checkbox("Use Clipped Geometry", value=True)
        
        # Generate map button
        if st.sidebar.button("Generate Map") or selected_spids:
            if selected_spids:
                with st.spinner("Generating map..."):
                    try:
                        map_obj = create_stockpoint_map(
                            selected_spids,
                            processed_sp_dim_df,
                            stockpoint_h3_coverage_with_metadata,
                            customer_stockpoint_cluster_assignment_df,
                            sp_territories_dict,
                            fill_col=fill_col,
                            use_geometry=use_geometry
                        )
                        
                        if map_obj:
                            st.subheader(f"Map for {len(selected_spids)} Stock Point(s)")
                            
                            # Display map
                            map_data = st_folium(
                                map_obj,
                                width=1200,
                                height=600,
                                returned_objects=["last_clicked", "last_object_clicked"]
                            )
                            
                            # Display selection info
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Selected Stock Points:** {len(selected_spids)}")
                                for i, name in enumerate(selected_names):
                                    st.write(f"• {name}")
                            
                            with col2:
                                st.write(f"**Fill Column:** {fill_col}")
                                st.write(f"**Geometry:** {'Clipped' if use_geometry else 'Vanilla'}")
                        else:
                            st.error("No data found for selected stock points")
                    
                    except Exception as e:
                        st.error(f"Error generating map: {str(e)}")
            else:
                st.warning("Please select at least one stock point")
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure your data variables are loaded in the session state or modify the data loading section")

if __name__ == "__main__":
    main()