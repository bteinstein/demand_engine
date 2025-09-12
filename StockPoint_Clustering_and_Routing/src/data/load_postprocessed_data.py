
# --------------------------------------

def postprocess_map_data(return_data = True, from_local=False):
    """Load all required datasets"""
    import duckdb
    import pandas as pd
    import geopandas as gpd
    import pickle
    import ast 
    import os
    import glob
    import gzip
    from config.settings import EXPORTS_DIR, STORAGE_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR

    # Local data loading function
    def load_from_local():
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

    # If loading from local, return immediately
    if from_local:
        if return_data:
            print("Loading data from local pickle files...")
            return load_from_local()

    # Otherwise, prepare data from scratch
    print("Preparing data from scratch...")
    
    # Get latest cluster file
    cluster_results_dir = EXPORTS_DIR / 'clustering'
    expanded_path = os.path.expanduser(cluster_results_dir)
    list_of_files = glob.glob(f'{expanded_path}/*.pickle')
    
    if not list_of_files:
        raise FileNotFoundError("No clustering results file found.")
    
    cluster_results_filepath = max(list_of_files, key=os.path.getmtime)
    
    print(f"Using clustering results file: {cluster_results_filepath}")
    
    # Load agent-customer mapping dimensions
    path_agent_customer_mapping = RAW_DATA_DIR / 'df_agent_customer.parquet'
    df_agent_customer_mapping = pd.read_parquet(path_agent_customer_mapping)
    
    # Load processed stock point dimensions
    processed_sp_dim_filepath = PROCESSED_DATA_DIR / 'processed_sp_dim_df.pickle'
    with open(processed_sp_dim_filepath, 'rb') as f:
        processed_sp_dim_df = pickle.load(f)
    
    # Load clustering results
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
    H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']
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
                contact_name, state_name, town_name, city_name, latitude, longitude, kyc_capture_status, customer_status,
                ac.agent_id, ac.agent_name, ac.role_name
            FROM customer_stockpoint_cluster_assignment a
            LEFT JOIN read_parquet('/home/bt/project/demand_engine/StockPoint_Clustering_and_Routing/data/processed/df_processed_customer_dim.parquet') d 
                ON d.customer_id = a.customer_id     
            LEFT JOIN df_agent_customer_mapping ac ON a.customer_id = ac.customer_id
        ''').df()


    # Agent-MFC
    with duckdb.connect(H3_DUCKDB_PATH) as conn: 
        agent_customer_mfc = conn.execute('''
                                        SELECT
                                            stock_point_id, agent_id as agent_id, agent_name as agent_name, role_name as role_name, 
                                            COUNT(DISTINCT customer_id) AS n_customers,
                                            COUNT(DISTINCT h3_cell_id) AS n_beats,
                                            (SELECT COUNT (DISTINCT c.customer_id) FROM customer_stockpoint_cluster_assignment_df c 
                                                WHERE c.agent_id = a.agent_id AND c.stock_point_id = a.stock_point_id 
                                                    AND c.assignment_type_id IN (1,2)) AS n_active_customers
                                        FROM customer_stockpoint_cluster_assignment_df a
                                        GROUP BY stock_point_id, agent_id, agent_name, role_name
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
                c.cluster_sp_dist_km, c.cluster_sp_direction,
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
    
    # Save processed data to local directory
    data_to_save = {
            'processed_sp_dim_df': processed_sp_dim_df,
            'stockpoint_h3_coverage_with_metadata': stockpoint_h3_coverage_with_metadata,
            'customer_stockpoint_cluster_assignment_df': customer_stockpoint_cluster_assignment_df,
            'agent_customer_mfc': agent_customer_mfc,
            'sp_territories_dict': sp_territories_dict
        }
    
    map_input_dir = EXPORTS_DIR / 'map_input_data'
    try:
        print('Saving processed data to local directory...')
        map_input_dir.mkdir(parents=True, exist_ok=True)        
        for filename, data in data_to_save.items():
            with gzip.open(map_input_dir / f'{filename}.pkl.gz', 'wb') as f:
                pickle.dump(data, f)
                
    except Exception as e:
        print(f"Error creating directory {map_input_dir}: {e}")
    
    # Save processed data to local directory
    from pathlib import Path
    ci_cd_input_dir = Path('/home/bt/project/Insight_and_Discovery/Clustering_And_Routes/app/data') #EXPORTS_DIR / 'map_input_data'
    try:
        print('Saving processed data to cicd directory...')
        ci_cd_input_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, data in data_to_save.items():
            with gzip.open(ci_cd_input_dir / f'{filename}.pkl.gz', 'wb') as f:
                pickle.dump(data, f)
                
    except Exception as e:
        print(f"Error creating directory {ci_cd_input_dir}: {e}")
    
    if return_data:
        return (processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, 
                customer_stockpoint_cluster_assignment_df, sp_territories_dict)

    
# -----------------------------------------------------------------------------------------------------------------    
# -----------------------------------------------------------------------------------------------------------------    
# -----------------------------------------------------------------------------------------------------------------    


# def load_data(from_local=False):
#     """Load all required datasets"""
#     import duckdb
#     import pandas as pd
#     import geopandas as gpd
#     import pickle
#     import ast 
#     import os
#     import glob
#     from config.settings import EXPORTS_DIR, OUTPUT_DIR, STORAGE_CONFIG, PROCESSED_DATA_DIR
#     import os
#     import glob
#     from config.settings import EXPORTS_DIR 

#     def get_latest_clusterfile():
#         # Define the directory path
#         cluster_results_dir = EXPORTS_DIR / 'clustering' 

#         # Expand the user directory shortcut
#         expanded_path = os.path.expanduser(cluster_results_dir)

#         # Use glob to find all .pickle files
#         list_of_files = glob.glob(f'{expanded_path}/*.pickle')

#         # Check if any .pickle files were found
#         if not list_of_files:
#             print("No .pickle files found in the specified directory.")
#             return None
#         else:
#             # Find the latest file by modification time
#             latest_file = max(list_of_files, key=os.path.getmtime)
#             # print(f"The latest .pickle file is: {latest_file}")
#             return latest_file
        
        
#     H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']

#     # Load processed stock point dimensions
#     processed_sp_dim_filepath = PROCESSED_DATA_DIR / 'processed_sp_dim_df.pickle'
#     with open(processed_sp_dim_filepath, 'rb') as f:
#         processed_sp_dim_df = pickle.load(f)
    
#     # # Load clustering results
#     # current_date = '2025-08-22'
#     # suffix = "ALL" 
#     # resolution = 8
#     # cluster_results_filepath = EXPORTS_DIR / 'clustering' / f"{suffix}_SPS_CLUSTER_R{resolution}_{current_date}.pickle"
    
#     cluster_results_filepath = get_latest_clusterfile()
    
#     if cluster_results_filepath is None:
#         raise FileNotFoundError("No clustering results file found.")
    
#     with open(cluster_results_filepath, 'rb') as f:
#         cluster_results = pickle.load(f)
    
#     sp_territories_dict = cluster_results.get('territories')
#     sp_clusters_dict = cluster_results.get('grid_results')
    
#     # Process clipped cells
#     all_cluster_grid_list = []
#     for key, value in sp_clusters_dict.items():
#         cell_geometries = value.get('cell_geometries')  
#         if cell_geometries:
#             cell_geometries_gpd = pd.DataFrame.from_dict(cell_geometries, orient='index').reset_index()
#             cell_geometries_gpd.columns = ['h3_cell','geometry']
#             cell_geometries_gpd = gpd.GeoDataFrame(cell_geometries_gpd)
#             cell_geometries_gpd['stock_point_id'] = int(key)
#             all_cluster_grid_list.append(cell_geometries_gpd)
    
#     all_cluster_clipped_grid_list_df = (
#         pd.concat(all_cluster_grid_list, ignore_index=True)
#         if all_cluster_grid_list
#         else gpd.GeoDataFrame(columns=['h3_cell', 'geometry', 'stock_point_id'])
#     )

#     # Load customer assignments
#     with duckdb.connect(H3_DUCKDB_PATH) as conn: 
#         customer_stockpoint_cluster_assignment_df = conn.execute('''
#             SELECT 
#                 stock_point_id, a.customer_id, h3_cell_id, customer_type, previous_cluster_id,
#                 CASE WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'buying customers' THEN 1
#                     WHEN assignment_tier = 'manual_review' AND customer_type = 'buying customers' THEN 2
#                     WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'recently activated' THEN 3
#                 ELSE 99 END AS assignment_type_id,
#                 CASE WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'buying customers' THEN 'Assigned Active/Buying'
#                     WHEN assignment_tier = 'manual_review' AND customer_type = 'buying customers' THEN 'Unassigned Active/Buying'
#                     WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'recently activated' THEN 'Assigned Recently Activated'
#                 ELSE 'Others' END AS assignment_type, 	
#                 contact_name, state_name, town_name, city_name, latitude, longitude, kyc_capture_status, customer_status
#             FROM customer_stockpoint_cluster_assignment a
#             LEFT JOIN read_parquet('/home/bt/project/demand_engine/StockPoint_Clustering_and_Routing/data/processed/df_processed_customer_dim.parquet') d 
#                 ON d.customer_id = a.customer_id     
#             ''').df()

#     # Load H3 coverage with metadata
#     with duckdb.connect(H3_DUCKDB_PATH) as conn: 
#         stockpoint_h3_coverage_with_metadata = conn.execute("""
#             WITH CTE_Assignment_Summary AS(
#                 SELECT 
#                     stock_point_id, h3_cell_id as h3_cell, 
#                     COUNT(DISTINCT customer_id) as n_total_assigned_customers,
#                     CAST(SUM(CASE WHEN assignment_type_id = 1 THEN 1 ELSE 0 END) AS INT) AS n_assigned_active_customers,
#                     CAST(SUM(CASE WHEN assignment_type_id = 3 THEN 1 ELSE 0 END) AS INT) AS n_assigned_recent_activated_customers
#                 FROM customer_stockpoint_cluster_assignment_df  
#                 WHERE h3_cell_id NOT NULL
#                 GROUP BY stock_point_id, h3_cell_id 
#             )    
#             SELECT 
#                 c.stock_point_id, c.h3_cell as beat, primary_address_id as beat_id,
#                 h.state_name, h.lga_name, h.ward_name, h.area_km2, h.confidence_level, h.latlng_json as latlng_coords,  
#                 c.cluster_sp_dist_km,
#                 COALESCE(s.n_total_assigned_customers, 0) AS n_total_assigned_customers, 
#                 COALESCE(s.n_assigned_active_customers, 0) AS n_assigned_active_customers, 
#                 COALESCE(s.n_assigned_recent_activated_customers, 0) AS n_assigned_recent_activated_customers
#             FROM stockpoint_h3_coverage c
#             LEFT JOIN CTE_Assignment_Summary s ON c.stock_point_id = s.stock_point_id AND c.h3_cell = s.h3_cell         
#             LEFT JOIN h3_cells h ON c.h3_cell = h.h3_index              
#             """).df()
    
#     # Process H3 coverage data
#     stockpoint_h3_coverage_with_metadata = gpd.GeoDataFrame(stockpoint_h3_coverage_with_metadata)
#     stockpoint_h3_coverage_with_metadata['latlng_coords'] = stockpoint_h3_coverage_with_metadata['latlng_coords'].apply(lambda x: ast.literal_eval(x))
#     stockpoint_h3_coverage_with_metadata = stockpoint_h3_coverage_with_metadata.merge(
#         all_cluster_clipped_grid_list_df.rename(columns={'h3_cell': 'beat'}), 
#         on=['beat', 'stock_point_id'], how='left'
#     )
    
    
#     # LOAD DATA INTO EXPORT DIRECTORY
#     # 1. LOCAL PROJECT DIRECTORY
#     map_input_dir = EXPORTS_DIR / 'map_input_data'
#     import gzip
#     import bz2
#     try:
#         print('Saving processed data to local directory...')
#         map_input_dir.mkdir(parents=True, exist_ok=True)
#         # Save processed_sp_dim_df as pickle with gzip compression      
#         with  gzip.open(map_input_dir / 'processed_sp_dim_df.pkl.gz', 'wb') as f:
#             pickle.dump(processed_sp_dim_df, f) 
#         # Save stockpoint_h3_coverage_with_metadata as pickle with gzip compression            
#         with  gzip.open(map_input_dir / 'stockpoint_h3_coverage_with_metadata.pkl.gz', 'wb') as f:
#             pickle.dump(stockpoint_h3_coverage_with_metadata, f) 
            
#         # Save customer_stockpoint_cluster_assignment_df as pickle with gzip compression   
#         with  gzip.open(map_input_dir / 'customer_stockpoint_cluster_assignment_df.pkl.gz', 'wb') as f:
#             pickle.dump(customer_stockpoint_cluster_assignment_df, f) 
             
#         # Save sp_territories_dict as pickle with gzip compression   
#         with  gzip.open(map_input_dir / 'sp_territories_dict.pkl.gz', 'wb') as f:
#             pickle.dump(sp_territories_dict, f)  
#     except Exception as e:
#         print(f"Error creating directory {map_input_dir}: {e}")
    
#     # 2. CLOUD STORAGE BUCKET/REPO
    
    
    
    
#     return (processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, 
#             customer_stockpoint_cluster_assignment_df, sp_territories_dict)


# --------------------------------------
def load_data(from_local=False):
    """Load all required datasets"""
    import duckdb
    import pandas as pd
    import geopandas as gpd
    import pickle
    import ast 
    import os
    import glob
    import gzip
    from config.settings import EXPORTS_DIR, STORAGE_CONFIG, PROCESSED_DATA_DIR

    # Local data loading function
    def load_from_local():
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

    # If loading from local, return immediately
    if from_local:
        print("Loading data from local pickle files...")
        return load_from_local()

    # Otherwise, prepare data from scratch
    print("Preparing data from scratch...")
    
    # Get latest cluster file
    cluster_results_dir = EXPORTS_DIR / 'clustering'
    expanded_path = os.path.expanduser(cluster_results_dir)
    list_of_files = glob.glob(f'{expanded_path}/*.pickle')
    
    if not list_of_files:
        raise FileNotFoundError("No clustering results file found.")
    
    cluster_results_filepath = max(list_of_files, key=os.path.getmtime)
    
    # Load processed stock point dimensions
    processed_sp_dim_filepath = PROCESSED_DATA_DIR / 'processed_sp_dim_df.pickle'
    with open(processed_sp_dim_filepath, 'rb') as f:
        processed_sp_dim_df = pickle.load(f)
    
    # Load clustering results
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
    H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']
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
    
    # Save processed data to local directory
    map_input_dir = EXPORTS_DIR / 'map_input_data'
    try:
        print('Saving processed data to local directory...')
        map_input_dir.mkdir(parents=True, exist_ok=True)
        
        data_to_save = {
            'processed_sp_dim_df': processed_sp_dim_df,
            'stockpoint_h3_coverage_with_metadata': stockpoint_h3_coverage_with_metadata,
            'customer_stockpoint_cluster_assignment_df': customer_stockpoint_cluster_assignment_df,
            'sp_territories_dict': sp_territories_dict
        }
        
        for filename, data in data_to_save.items():
            with gzip.open(map_input_dir / f'{filename}.pkl.gz', 'wb') as f:
                pickle.dump(data, f)
                
    except Exception as e:
        print(f"Error creating directory {map_input_dir}: {e}")
    
    # Save processed data to local directory
    from pathlib import Path
    ci_cd_input_dir = Path('/home/bt/project/Insight_and_Discovery/Clustering_And_Routes/data') #EXPORTS_DIR / 'map_input_data'
    try:
        print('Saving processed data to cicd directory...')
        ci_cd_input_dir.mkdir(parents=True, exist_ok=True)
        
        data_to_save = {
            'processed_sp_dim_df': processed_sp_dim_df,
            'stockpoint_h3_coverage_with_metadata': stockpoint_h3_coverage_with_metadata,
            'customer_stockpoint_cluster_assignment_df': customer_stockpoint_cluster_assignment_df,
            'sp_territories_dict': sp_territories_dict
        }
        
        for filename, data in data_to_save.items():
            with gzip.open(ci_cd_input_dir / f'{filename}.pkl.gz', 'wb') as f:
                pickle.dump(data, f)
                
    except Exception as e:
        print(f"Error creating directory {ci_cd_input_dir}: {e}")
    
    return (processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, 
            customer_stockpoint_cluster_assignment_df, sp_territories_dict)

    
# -----------------------------------------------------------------------------------------------------------------    
# -----------------------------------------------------------------------------------------------------------------    
# -----------------------------------------------------------------------------------------------------------------    


# def load_data(from_local=False):
#     """Load all required datasets"""
#     import duckdb
#     import pandas as pd
#     import geopandas as gpd
#     import pickle
#     import ast 
#     import os
#     import glob
#     from config.settings import EXPORTS_DIR, OUTPUT_DIR, STORAGE_CONFIG, PROCESSED_DATA_DIR
#     import os
#     import glob
#     from config.settings import EXPORTS_DIR 

#     def get_latest_clusterfile():
#         # Define the directory path
#         cluster_results_dir = EXPORTS_DIR / 'clustering' 

#         # Expand the user directory shortcut
#         expanded_path = os.path.expanduser(cluster_results_dir)

#         # Use glob to find all .pickle files
#         list_of_files = glob.glob(f'{expanded_path}/*.pickle')

#         # Check if any .pickle files were found
#         if not list_of_files:
#             print("No .pickle files found in the specified directory.")
#             return None
#         else:
#             # Find the latest file by modification time
#             latest_file = max(list_of_files, key=os.path.getmtime)
#             # print(f"The latest .pickle file is: {latest_file}")
#             return latest_file
        
        
#     H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']

#     # Load processed stock point dimensions
#     processed_sp_dim_filepath = PROCESSED_DATA_DIR / 'processed_sp_dim_df.pickle'
#     with open(processed_sp_dim_filepath, 'rb') as f:
#         processed_sp_dim_df = pickle.load(f)
    
#     # # Load clustering results
#     # current_date = '2025-08-22'
#     # suffix = "ALL" 
#     # resolution = 8
#     # cluster_results_filepath = EXPORTS_DIR / 'clustering' / f"{suffix}_SPS_CLUSTER_R{resolution}_{current_date}.pickle"
    
#     cluster_results_filepath = get_latest_clusterfile()
    
#     if cluster_results_filepath is None:
#         raise FileNotFoundError("No clustering results file found.")
    
#     with open(cluster_results_filepath, 'rb') as f:
#         cluster_results = pickle.load(f)
    
#     sp_territories_dict = cluster_results.get('territories')
#     sp_clusters_dict = cluster_results.get('grid_results')
    
#     # Process clipped cells
#     all_cluster_grid_list = []
#     for key, value in sp_clusters_dict.items():
#         cell_geometries = value.get('cell_geometries')  
#         if cell_geometries:
#             cell_geometries_gpd = pd.DataFrame.from_dict(cell_geometries, orient='index').reset_index()
#             cell_geometries_gpd.columns = ['h3_cell','geometry']
#             cell_geometries_gpd = gpd.GeoDataFrame(cell_geometries_gpd)
#             cell_geometries_gpd['stock_point_id'] = int(key)
#             all_cluster_grid_list.append(cell_geometries_gpd)
    
#     all_cluster_clipped_grid_list_df = (
#         pd.concat(all_cluster_grid_list, ignore_index=True)
#         if all_cluster_grid_list
#         else gpd.GeoDataFrame(columns=['h3_cell', 'geometry', 'stock_point_id'])
#     )

#     # Load customer assignments
#     with duckdb.connect(H3_DUCKDB_PATH) as conn: 
#         customer_stockpoint_cluster_assignment_df = conn.execute('''
#             SELECT 
#                 stock_point_id, a.customer_id, h3_cell_id, customer_type, previous_cluster_id,
#                 CASE WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'buying customers' THEN 1
#                     WHEN assignment_tier = 'manual_review' AND customer_type = 'buying customers' THEN 2
#                     WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'recently activated' THEN 3
#                 ELSE 99 END AS assignment_type_id,
#                 CASE WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'buying customers' THEN 'Assigned Active/Buying'
#                     WHEN assignment_tier = 'manual_review' AND customer_type = 'buying customers' THEN 'Unassigned Active/Buying'
#                     WHEN assignment_tier = 'h3_inclusion' AND customer_type = 'recently activated' THEN 'Assigned Recently Activated'
#                 ELSE 'Others' END AS assignment_type, 	
#                 contact_name, state_name, town_name, city_name, latitude, longitude, kyc_capture_status, customer_status
#             FROM customer_stockpoint_cluster_assignment a
#             LEFT JOIN read_parquet('/home/bt/project/demand_engine/StockPoint_Clustering_and_Routing/data/processed/df_processed_customer_dim.parquet') d 
#                 ON d.customer_id = a.customer_id     
#             ''').df()

#     # Load H3 coverage with metadata
#     with duckdb.connect(H3_DUCKDB_PATH) as conn: 
#         stockpoint_h3_coverage_with_metadata = conn.execute("""
#             WITH CTE_Assignment_Summary AS(
#                 SELECT 
#                     stock_point_id, h3_cell_id as h3_cell, 
#                     COUNT(DISTINCT customer_id) as n_total_assigned_customers,
#                     CAST(SUM(CASE WHEN assignment_type_id = 1 THEN 1 ELSE 0 END) AS INT) AS n_assigned_active_customers,
#                     CAST(SUM(CASE WHEN assignment_type_id = 3 THEN 1 ELSE 0 END) AS INT) AS n_assigned_recent_activated_customers
#                 FROM customer_stockpoint_cluster_assignment_df  
#                 WHERE h3_cell_id NOT NULL
#                 GROUP BY stock_point_id, h3_cell_id 
#             )    
#             SELECT 
#                 c.stock_point_id, c.h3_cell as beat, primary_address_id as beat_id,
#                 h.state_name, h.lga_name, h.ward_name, h.area_km2, h.confidence_level, h.latlng_json as latlng_coords,  
#                 c.cluster_sp_dist_km,
#                 COALESCE(s.n_total_assigned_customers, 0) AS n_total_assigned_customers, 
#                 COALESCE(s.n_assigned_active_customers, 0) AS n_assigned_active_customers, 
#                 COALESCE(s.n_assigned_recent_activated_customers, 0) AS n_assigned_recent_activated_customers
#             FROM stockpoint_h3_coverage c
#             LEFT JOIN CTE_Assignment_Summary s ON c.stock_point_id = s.stock_point_id AND c.h3_cell = s.h3_cell         
#             LEFT JOIN h3_cells h ON c.h3_cell = h.h3_index              
#             """).df()
    
#     # Process H3 coverage data
#     stockpoint_h3_coverage_with_metadata = gpd.GeoDataFrame(stockpoint_h3_coverage_with_metadata)
#     stockpoint_h3_coverage_with_metadata['latlng_coords'] = stockpoint_h3_coverage_with_metadata['latlng_coords'].apply(lambda x: ast.literal_eval(x))
#     stockpoint_h3_coverage_with_metadata = stockpoint_h3_coverage_with_metadata.merge(
#         all_cluster_clipped_grid_list_df.rename(columns={'h3_cell': 'beat'}), 
#         on=['beat', 'stock_point_id'], how='left'
#     )
    
    
#     # LOAD DATA INTO EXPORT DIRECTORY
#     # 1. LOCAL PROJECT DIRECTORY
#     map_input_dir = EXPORTS_DIR / 'map_input_data'
#     import gzip
#     import bz2
#     try:
#         print('Saving processed data to local directory...')
#         map_input_dir.mkdir(parents=True, exist_ok=True)
#         # Save processed_sp_dim_df as pickle with gzip compression      
#         with  gzip.open(map_input_dir / 'processed_sp_dim_df.pkl.gz', 'wb') as f:
#             pickle.dump(processed_sp_dim_df, f) 
#         # Save stockpoint_h3_coverage_with_metadata as pickle with gzip compression            
#         with  gzip.open(map_input_dir / 'stockpoint_h3_coverage_with_metadata.pkl.gz', 'wb') as f:
#             pickle.dump(stockpoint_h3_coverage_with_metadata, f) 
            
#         # Save customer_stockpoint_cluster_assignment_df as pickle with gzip compression   
#         with  gzip.open(map_input_dir / 'customer_stockpoint_cluster_assignment_df.pkl.gz', 'wb') as f:
#             pickle.dump(customer_stockpoint_cluster_assignment_df, f) 
             
#         # Save sp_territories_dict as pickle with gzip compression   
#         with  gzip.open(map_input_dir / 'sp_territories_dict.pkl.gz', 'wb') as f:
#             pickle.dump(sp_territories_dict, f)  
#     except Exception as e:
#         print(f"Error creating directory {map_input_dir}: {e}")
    
#     # 2. CLOUD STORAGE BUCKET/REPO
    
    
    
    
#     return (processed_sp_dim_df, stockpoint_h3_coverage_with_metadata, 
#             customer_stockpoint_cluster_assignment_df, sp_territories_dict)
