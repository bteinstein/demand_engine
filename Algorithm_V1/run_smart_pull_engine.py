
import os
import shutil
from pathlib import Path
import pandas as pd 
import logging
from src.data.get_data import get_all_input_data #get_data_reco_custdim_spdim , get_kyc_customers, get_customer_score
from datetime import datetime, timedelta 
from src.data.preprocessing import preprocessing
from src.routing.ValhallaManager import ValhallaManager
from src.data.data_filter import data_filter 
from src.main import run_push_recommendation 
from src.utils_and_postprocessing.utils import setup_logger, setup_directories, list_files_in_directory, load_feather_inputs

from src.data.export import export_data 
from src.data.get_connection import get_connection
from src.clustering.evaluate_cluster import evaluate_unsupervised_clustering
from src.utils_and_postprocessing.utils import cluster_summary_and_selection, postprocess_selected_trip
from src.utils_and_postprocessing.run_clustering_and_routing import create_and_plot_route, create_cluster_trip_optroute 
 
 
logger = setup_logger(__name__, log_file='test.log')

logger.info("This is an info message")
logger.warning("This is a warning")
 
# ### Parameter Setup
 
# Constants for directory structure
# BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path('').resolve()#.parent 
INDEX_HTML = 'index.html'
CURRENT_DATE = datetime.today().date() #+ timedelta(days=1)

INPUT_DIR, SELECTED_TRIP_PATH, ALL_CLUSTER_PATH, LOCAL_EXCEL_PATH = setup_directories(base_dir = BASE_DIR, current_date = CURRENT_DATE, logger = logger) 
 
# Instantiate the manager
valhalla_manager = ValhallaManager(logger=logger)

# Start the server
valhalla_manager.start_valhalla()

# Check valhalla status
valhalla_manager.check_valhalla_status()
 
### Main Function
 
# ## MAIN 1
 
# Get input data
df_customer_sku_recommendation_raw, df_customer_dim_with_affinity_score_raw, \
    df_stockpoint_dim_raw, df_kyc_customer, df_customer_score = get_all_input_data(logger=logger,
                                                                                    save_local = False,
                                                                                    input_dir_path = None, # Added optional input_dir_path
                                                                                    reload_recommendation_data = False # Added parameter to control data reloading
                                                                                    )
 
# Preprocessing
df_customer_sku_recommendation, df_master_customer_dim, df_stockpoint_dim = preprocessing(df_customer_sku_recommendation_raw, 
                                                                                            df_customer_dim_with_affinity_score_raw, 
                                                                                            df_stockpoint_dim_raw,
                                                                                            df_customer_score,
                                                                                            df_kyc_customer)
 

 
ALL_STOCKPOINTS_RESULT = {}
for index, row in df_stockpoint_dim.iterrows():
    # if index == 12:
    # if index == 5:
    stock_point_id =  row['Stock_Point_ID']
    stock_point_name = row['Stock_point_Name']
    print(f'{index}/{len(df_stockpoint_dim)} \nStock Point ID: {stock_point_id} || Stock Point Name: {stock_point_name}')  # Access by column name

    res_dict = run_push_recommendation(df_customer_sku_recommendation, 
                            df_master_customer_dim, 
                            df_stockpoint_dim, 
                            stock_point_id,
                            stock_point_name,
                            sku_recency = 7, 
                            customer_recency = 60, number_recommendation = 5, 
                            estimate_qty_scale_factor = 1, max_estimated_qty = 5, 
                            exclude_recency_customer = 4,
                            max_customers_per_route=20,
                            max_volume_per_route=300,
                            max_distance_km = 40,
                            sel_trip_cluster = 5,
                            min_ncust_per_cluster = 5,
                            clustering_method = 'divisive',
                            skip_route_optimization = False,
                            save_to_disk = True,
                            # Global variables
                            valhalla_manager = valhalla_manager,
                            CURRENT_DATE = CURRENT_DATE,
                            SELECTED_TRIP_PATH = SELECTED_TRIP_PATH,
                            ALL_CLUSTER_PATH = ALL_CLUSTER_PATH,
                            LOCAL_EXCEL_PATH = LOCAL_EXCEL_PATH,
                            logger=logger)
    
    ALL_STOCKPOINTS_RESULT[stock_point_name] = res_dict

 
all_spid_list = df_stockpoint_dim['Stock_Point_ID'].to_list()
selected_trip_spid_list = [int(spid) for spid in list_files_in_directory(SELECTED_TRIP_PATH) if not spid.startswith('index')]
all_cluster_spid_list = [int(spid) for spid in list_files_in_directory(ALL_CLUSTER_PATH) if not spid.startswith('index')]

unmapped_selected_trip_spid_list = list(set(all_spid_list) - set(selected_trip_spid_list))
unmapped_all_cluster_spid_list = list(set(all_spid_list) - set(all_cluster_spid_list))
 
print(f"All Stock Points: {len(all_spid_list)}")
print(f"All Stock Points: {len(selected_trip_spid_list)}")
print(f"All Stock Points: {len(all_cluster_spid_list)}")
print(f"All Stock Points: {len(unmapped_selected_trip_spid_list)}")
print(f"All Stock Points: {len(unmapped_all_cluster_spid_list)}")
 
# Copy index.html files from source directories to date-based subdirectories
source_cluster_map_index = BASE_DIR / 'html' / 'default_cluster_map_index.html'
source_selected_trip_index = BASE_DIR / 'html' / 'default_selected_cluster_map_index.html' 
        
if len(unmapped_all_cluster_spid_list) > 0:
    try:
        if source_cluster_map_index.exists():
            for spid in unmapped_all_cluster_spid_list:
                spid_path = ALL_CLUSTER_PATH / f'{str(spid)}.html'       
                shutil.copy2(source_cluster_map_index, spid_path)
                logger.debug(f"Copied index.html to {ALL_CLUSTER_PATH / str(spid)}.html")
        else:
            logger.warning(f"Source index.html not found at {source_selected_trip_index}")
    except Exception as e:
        logger.error(f"Failed to copy index.html to {ALL_CLUSTER_PATH}: {str(e)}")
        
        
# Copy index.html for selected_trip_map
if len(unmapped_selected_trip_spid_list) > 0:
    try:
        if source_selected_trip_index.exists():
            for spid in unmapped_selected_trip_spid_list:
                spid_path = SELECTED_TRIP_PATH / f'{str(spid)}.html'       
                shutil.copy2(source_selected_trip_index, spid_path)
                logger.debug(f"Copied index.html to {SELECTED_TRIP_PATH / INDEX_HTML}")
        else:
            logger.warning(f"Source index.html not found at {source_selected_trip_index}")
    except Exception as e:
        logger.error(f"Failed to copy index.html to {SELECTED_TRIP_PATH}: {str(e)}")

             
 
from src.data.export2db import RecommendationProcessor
from src.data.get_connection import get_connection
 
# Or use the processor directly
processor = RecommendationProcessor(get_connection)
processor.process(ALL_STOCKPOINTS_RESULT, CURRENT_DATE)
 
# ---------------------
## Clean Up
 
# Stop the server when done
valhalla_manager.stop_valhalla()
 



