import pandas as pd
import geopandas as gpd
from config.settings import ADMIN_DATA_SOURCES, INPUT_BASE_DATA_SOURCES, PROCESSED_DATA_DIR, RAW_DATA_DIR
import pandas as pd
import pickle
import logging
from src.utils import clean_customer_gdf_coordinates
from datetime import datetime, timedelta
import numpy as np


# List of (sql_file, output_feather_file)
BASE_DATA_INPUT_PATH = {
    k: v["local_file_path"] for k, v in INPUT_BASE_DATA_SOURCES.items()
}

# {'sp_dim': PosixPath('/home/bt/project/demand_engine/StockPoint_Clustering_and_Routing/data/raw/df_sp_dim.parquet'),
# 'sp_location_mapping': PosixPath('/home/bt/project/demand_engine/StockPoint_Clustering_and_Routing/data/raw/df_sp_location_mapping.parquet'),
# 'customer_dim': PosixPath('/home/bt/project/demand_engine/StockPoint_Clustering_and_Routing/data/raw/df_customer_dim.parquet'),
# 'sp_active_customers': PosixPath('/home/bt/project/demand_engine/StockPoint_Clustering_and_Routing/data/raw/df_sp_active_customers.parquet')}
 
    

def load_and_preprocess_sp_lga_mapping_data(logger):
    """
    Loads and preprocesses stockpoint location mapping data from raw parquet file, 
    returning clean unique state and Local Government Area (LGA) names mapped to all mfc that are in same naming official convention of the country.
    This help with the mapping of sp coverage state-lga to boundary file.
    
    The function performs several preprocessing steps including:
    - Loading the data from a parquet file.
    - Splitting specific state and LGA names based on predefined rules.
    - Cleaning LGA names by replacing or modifying certain substrings.
    - Removing duplicates and filtering out unwanted records.
    - state_, and lga_ are the clean and compactible name to the official file
    
    Parameters:
    -----------
    logger : logging.Logger
        Logger object used to log information and errors during the process.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing preprocessed location mapping data with cleaned state and LGA names.
        The DataFrame includes columns: State_Name, State_ID, LGA_Name, LGA_ID, state_, and lga_.

    Raises:
    -------
    FileNotFoundError:
        If the specified parquet file is not found.
    Exception:
        For other unexpected errors during the data loading and preprocessing process.
    """    
    try:
        # Load the Feather file
        sp_loc_path =  BASE_DATA_INPUT_PATH['sp_location_mapping'] # INPUT_BASE_DATA_SOURCES['sp_location_mapping']['local_file_path']
        df_sp_location_mapping = pd.read_parquet(sp_loc_path)
        logger.info("Loaded location mapping data from parquet file.")

        # Dictionary for splitting state and LGA names
        dict_split_state_lga = {'Abuja': {'Kuje/Gwagwalada/Abaji': ['kuje', 'gwagwalada', 'abaji']}}

        # Create DataFrame from dictionary
        df_split_state_lga = pd.DataFrame([
            {'State_Name': state_name, 'LGA_Name': lga_name, 'lga_name_split': item}
            for state_name, lga_dict in dict_split_state_lga.items()
            for lga_name, lga_list in lga_dict.items()
            for item in lga_list
        ])

        # Preprocessing pipeline
        df_dist_state_lga = (
            df_sp_location_mapping
            .merge(df_split_state_lga, on=['State_Name', 'LGA_Name'], how='left')
            .assign(state_=lambda df: df['State_Name'].str.lower())
            .assign(lga_=lambda df: df['LGA_Name'].str.lower())
            .assign(lga_=lambda df: df.apply(
                lambda row: row['lga_'] if pd.isna(row['lga_name_split']) else row['lga_name_split'], axis=1
            ))
            .assign(lga_=lambda df: df['lga_'].str.replace('/', ' '))
            .assign(lga_=lambda df: df['lga_'].replace({
                'yenagoa': 'yenegoa',
                'ifako ijaiye': 'ifako ijaye',
                'sagamu': 'shagamu',
                'garun mallam': 'garun malam',
                'amac 1': 'municipal area council',
                'kachako': 'takai',
                'mbaitoli': 'mbatoli'
            }))
            [['State_Name', 'State_ID', 'LGA_Name', 'LGA_ID', 'state_', 'lga_']]
            .drop_duplicates()
            .query('~lga_.str.contains("self|push")', engine='python')
        )

        logger.info("Preprocessing completed successfully.")
        logger.info(f"Initial length: {len(df_sp_location_mapping)}")
        logger.info(f"Processed length: {len(df_dist_state_lga)}")
        logger.info(f"Columns: {df_dist_state_lga.columns.tolist()}")

        return df_dist_state_lga

    except FileNotFoundError:
        logger.error('The specified Feather file was not found.')
    except Exception as e:
        logger.error(f'An error occurred: {e}')


def load_and_preprocess_sp_lcda_mapping_data(logger):
    # to-do
    try: 
        
        df_dist_state_lcda = pd.DataFrame()
        return df_dist_state_lcda

    except FileNotFoundError:
        logger.error('The specified Feather file was not found.')
    except Exception as e:
        logger.error(f'An error occurred: {e}')


def load_and_preprocess_lga_geojson(logger):
    lgas_geojson_path = ADMIN_DATA_SOURCES['lgas']['standardize_file_path']
    try:
        lgas_gdf = gpd.read_file(lgas_geojson_path)
        logger.info(f"Loaded {len(lgas_gdf)} LGAs from GeoJSON.")
        drop_cols = [ 'FID','globalid', 'uniq_id', 'timestamp', 'editor']
        # Preprocessing for easy merging
        lgas_gdf = (
            lgas_gdf
            .assign(state_=lambda df: df['state_name'].str.lower())
            .assign(state_=lambda df: df['state_'].replace({'fct': 'abuja'}))
            .assign(lga_=lambda df: df['lga_name'].str.lower())
            .assign(lga_=lambda df: df['lga_'].str.replace('/', ' ').str.replace('-', ' '))
            .drop(columns=drop_cols)
        )
        geometry_cols = ['state_', 'lga_','Shape__Area', 'Shape__Length', 'geometry']
        lgas_gdf.columns = [col+"_ng"  if col not in geometry_cols else col for col in lgas_gdf.columns]
        
        # Project to an appropriate CRS for area calculation
        # lgas_gdf_proj = lgas_gdf.to_crs("EPSG:6933")  # WGS 84 / World Cylindrical Equal Area
        lgas_gdf_proj = lgas_gdf.to_crs(epsg=32631)  # UTM Zone 31N for Nigeria
        lgas_gdf['area_km2'] = (lgas_gdf_proj.geometry.area / 1_000_000).round(4)  # Convert m^2 to km^2
         
        
        # Project to an appropriate CRS for centroid calculation
        gdf_proj_centroid = lgas_gdf.to_crs(epsg=32631)  # UTM Zone 31N for Nigeria
        lgas_gdf['centroid'] = gdf_proj_centroid.geometry.centroid
        lgas_gdf['centroid_lat'] = lgas_gdf['centroid'].y
        lgas_gdf['centroid_lng'] = lgas_gdf['centroid'].x
        
        # Re-project back to the original CRS (if needed)
        lgas_gdf = lgas_gdf.to_crs(epsg=4326)  # Revert to WGS84 if necessary

        return lgas_gdf

    except FileNotFoundError:
        logger.error('The specified GeoJSON file was not found.')
    except Exception as e:
        logger.error(f'An error occurred: {e}')

def load_and_preprocess_lcda_geojson(logger, lcda_geojson_path = '../input/geojson/Nigeria_-_Ward_Boundaries.geojson'):
    
    try:
        lcdas_gdf = gpd.read_file(lcda_geojson_path)
        logger.info(f"Loaded {len(lcdas_gdf)} LGAs from GeoJSON.")
        drop_cols = [ 'FID','globalid', 'uniq_id', 'timestamp', 'editor']
        # Preprocessing for easy merging
        lcdas_gdf = (
            lcdas_gdf
            .assign(state_=lambda df: df['statename'].str.lower())
            .assign(state_=lambda df: df['state_'].replace({'fct': 'abuja'}))
            .assign(lga_=lambda df: df['lganame'].str.lower())
            .assign(lcda_=lambda df: df['wardname'].str.lower())
            .assign(lga_=lambda df: df['lga_'].str.replace('/', ' ').str.replace('-', ' ')) 
            .assign(lcda_=lambda df: df['lcda_'].str.replace('/', ' ').str.replace('-', ' ')) 
            .drop(columns=drop_cols) 
        )
        geometry_cols = ['state_', 'lga_','lcda_', 'Shape__Area', 'Shape__Length', 'geometry']
        lcdas_gdf.columns = [col+"_ng"  if col not in geometry_cols else col for col in lcdas_gdf.columns]
        
        lcdas_gdf_proj = lcdas_gdf.to_crs("EPSG:6933")  # WGS 84 / World Cylindrical Equal Area
        lcdas_gdf['area_km2'] = (lcdas_gdf_proj.geometry.area / 1_000_000).round(4)  # Convert m^2 to km^2
        
        # lcdas_gdf['geometry_wkt'] = [geo.wkt for geo in lcdas_gdf.geometry] # Convert shapely geometry to WKT
        
        return lcdas_gdf

    except FileNotFoundError:
        logger.error('The specified GeoJSON file was not found.')
    except Exception as e:
        logger.error(f'An error occurred: {e}')


import logging
import time
import pandas as pd

import logging
import time
import pandas as pd
import geopandas as gpd

def merge_sp_loc_with_lga_gpd(logger: logging.Logger, df_dist_state_lga: pd.DataFrame, lgas_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merges stock point location data with LGA geospatial data.

    This function performs a left merge on the `state_` and `lga_` columns
    of `df_dist_state_lga` and `lgas_gdf`. It logs the process, including
    input shapes, merge time, and a summary of the merge results.

    Args:
        logger (logging.Logger): An instance of a logger.
        df_dist_state_lga (pd.DataFrame): Stock point data with state and LGA identifiers.
        lgas_gdf (gpd.GeoDataFrame): LGA geometries and metadata.

    Returns:
        gpd.GeoDataFrame: A merged GeoDataFrame with stock point data and corresponding LGA geometry.
    """
    try:
        # Validate input DataFrames
        if not all(col in df_dist_state_lga.columns for col in ['state_', 'lga_']):
            raise ValueError("df_dist_state_lga is missing required columns for merging.")

        if not all(col in lgas_gdf.columns for col in ['state_', 'lga_']):
            raise ValueError("lgas_gdf is missing required columns for merging.")

        # Log the shapes of the input DataFrames
        logger.info(f"Shape of df_dist_state_lga: {df_dist_state_lga.shape}")
        logger.info(f"Shape of lgas_gdf: {lgas_gdf.shape}")

        # Columns to join on
        col_join = ['state_', 'lga_']

        # Log the columns being joined on
        logger.info(f"Merging on columns: {col_join}")

        # Perform the merge operation
        start_time = time.time()
        df_state_lga_merge_gdf = df_dist_state_lga.merge(lgas_gdf, how='left', on=col_join, indicator=True)
        end_time = time.time()

        # Log the time taken for the merge operation
        logger.info(f"Merge operation completed in {end_time - start_time:.2f} seconds")

        # Log the merge results
        merge_counts = df_state_lga_merge_gdf['_merge'].value_counts()
        logger.info("Merge results summary:")
        for merge_type, count in merge_counts.items():
            logger.info(f"  {merge_type}: {count}")

        # Log the shape of the resulting DataFrame
        logger.info(f"Shape of merged DataFrame: {df_state_lga_merge_gdf.shape}")

        return df_state_lga_merge_gdf

    except Exception as e:
        logger.error(f"An error occurred during the merge: {e}")
        raise


#------------------------------------------------------------------#
def preprocess_sp_location_mapping(logger: logging.Logger) -> None:
    """
    Orchestrates the end-to-end preprocessing of stock point location data.

    This function loads and prepares stock point mapping data and LGA geospatial data,
    merges them into a comprehensive GeoDataFrame, and saves the result as a pickled file
    for later use.

    Args:
        logger (logging.Logger): An instance of a logger to track the process.

    Returns:
        None
    """
    try:
        logger.info("Starting preprocessing of stock point location mapping.")

        # Execute the functions with explicit logging for each step
        logger.info("Step 1/5: Loading and preprocessing uniq state-lga to stock point mapping data.")
        df_uniq_sp_state_lga_map = load_and_preprocess_sp_lga_mapping_data(logger)

        logger.info("Step 2/5: Loading and preprocessing LGA GeoJSON data.")
        lgas_gdf = load_and_preprocess_lga_geojson(logger)
        lgas_gdf.columns = lgas_gdf.columns.str.lower()

        logger.info("Step 3/5: Merging uniq state-lga to stock point mapping df and LGA Geodata.")
        sp_uniq_mapped_lgas_gdf = merge_sp_loc_with_lga_gpd(logger, df_uniq_sp_state_lga_map, lgas_gdf)
        sp_uniq_mapped_lgas_gdf.columns = sp_uniq_mapped_lgas_gdf.columns.str.lower()
        sp_uniq_mapped_lgas_gdf = sp_uniq_mapped_lgas_gdf.drop('_merge', axis=1)
        
        logger.info("Step 4/5: Get Clean SP - LGA Mapping with GEO-Data ")
        # -------------------------------------- Clean SP - LGA Mapping -------
        # LGA Mapping to SP only   
        sp_loc_map_path = BASE_DATA_INPUT_PATH['sp_location_mapping'] 
        df_sp_location_mapping = pd.read_parquet(sp_loc_map_path)
        df_sp_location_mapping.columns = df_sp_location_mapping.columns.str.lower()        
        
        df_processed_sp_lga_mapping = (df_sp_location_mapping.copy()
                    .drop(columns=['lcda_name', 'lcda_id'])
                    .drop_duplicates()
                    .assign(lga_name_=lambda x: x['lga_name'].str.lower())
                    .query('~lga_name_.str.contains("self|push")', engine='python')
                    .drop(columns=['lga_name_'])
                    .reset_index(drop=True)
                    )  
        
        merge_cols = ['state_id', 'lga_id']
        df_processed_sp_lga_mapping = (df_processed_sp_lga_mapping
                                            .merge(sp_uniq_mapped_lgas_gdf[merge_cols], on=merge_cols, how='inner'))
        
        # -------- SP DIM PREPROCESSING -------
        logger.info("Step 4/5: Get Clean SP - LGA Mapping with GEO-Data ") 
        sp_dim_path = BASE_DATA_INPUT_PATH['sp_dim'] 
        sp_dim_df = pd.read_parquet(sp_dim_path)
        sp_dim_df.columns = sp_dim_df.columns.str.lower() 
        sp_dim_df['latitude'] = sp_dim_df['latitude'].replace('', '0').astype(float)
        sp_dim_df['longitude'] = sp_dim_df['longitude'].replace('', '0').astype(float)
        
        ## -----------------------------------------------------------------
        # Save to Disk
        ## -----------------------------------------------------------------
        # 1. lgas_gdf: lga geo-data 
        processed_lgas_gdf = PROCESSED_DATA_DIR / 'lgas_gdf.pickle'
        logger.info(f"Saving processed LGA GeoDataFrame to {processed_lgas_gdf}.")
        with open(processed_lgas_gdf, 'wb') as filename:
            pickle.dump(obj=lgas_gdf, file=filename)
        logger.info(f"Successfully saved processed data. Final DataFrame shape: {len(lgas_gdf):,}")
        
        # -----------------------------------------------------------------
        # 2. sp_uniq_mapped_lga_gdf: Uniq geo-data of sp lga mapping
        processed_sp_location_map_gdf = PROCESSED_DATA_DIR / 'sp_uniq_mapped_lgas_gdf.pickle' 
        logger.info(f"Saving processed GeoDataFrame to {processed_sp_location_map_gdf}.")
        with open(processed_sp_location_map_gdf, 'wb') as filename:
            pickle.dump(obj=sp_uniq_mapped_lgas_gdf, file=filename)
        logger.info(f"Successfully saved processed data. Final DataFrame shape: {len(sp_uniq_mapped_lgas_gdf):,}")

        # -----------------------------------------------------------------
        # 3. df_processed_sp_lga_mapping: Processed stockpoint to lga mapping
        path_processed_sp_location_mapping_df = PROCESSED_DATA_DIR / 'processed_sp_lga_mapping_df.pickle' 
        logger.info(f"Saving Processed stockpoint to LGA mapping to {path_processed_sp_location_mapping_df}.")
        with open(path_processed_sp_location_mapping_df, 'wb') as filename:
            pickle.dump(obj=df_processed_sp_lga_mapping, file=filename)
        logger.info(f"Successfully saved processed data. Final DataFrame shape: {len(df_processed_sp_lga_mapping):,}")

        # -----------------------------------------------------------------
        
        path_processed_sp_dim_df = PROCESSED_DATA_DIR / 'processed_sp_dim_df.pickle' 
        logger.info(f"Saving Processed stockpoint to LGA mapping to {path_processed_sp_dim_df}.")
        with open(path_processed_sp_dim_df, 'wb') as filename:
            pickle.dump(obj=sp_dim_df, file=filename)
        logger.info(f"Successfully saved sp dim data. Final DataFrame shape: {len(sp_dim_df):,}")

        logger.info("Preprocessing complete.")    
    except Exception as e:
        logger.exception(f"An error occurred during preprocessing: {e}") 
        raise
        
 
#--------------------------------------------------------------------------
def prepare_sp_and_recent_activated_customers(logger): 
    ## TO-DO Clean Coords        
    df_sp_active_customers = pd.read_parquet(BASE_DATA_INPUT_PATH['sp_active_customers'])  
    df_customer_dim = pd.read_parquet(BASE_DATA_INPUT_PATH['customer_dim'])  
    
    #### ---------------------------------------------------------------------------
    df_customer_dim = (
        df_customer_dim.copy()
        .assign(Address=lambda df: df['Full_Address'].fillna(df['Location']))
        .assign(Is_Location_Captured=lambda df: df['Is_Location_Captured'].fillna(0))
        .assign(KYC_Capture_Status=lambda df: df['KYC_Capture_Status'].fillna(0))
        .assign(Is_Location_Verified=lambda df: df['Is_Location_Verified'].fillna(0))
        .assign(Address=lambda df: df.apply(
            lambda row: row['Location'] if row['Address'] == '' and row['Location'] != '' else row['Address'],
            axis=1
        ))
        .assign(Is_Location_Captured=lambda df: df.apply(
            lambda row: 0 if row['Is_Location_Captured'] == 1 and pd.isna(row['LocationSubmittedDate']) else row['Is_Location_Captured'],
            axis=1
        ))
        .assign(Is_Location_Verified=lambda df: df.apply(
            lambda row: 0 if row['Is_Location_Verified'] == 1 and pd.isna(row['Is_Location_Verified']) else row['Is_Location_Verified'],
            axis=1
        ))
        .drop(columns = ['Is_Location_Submitted', 'Location_Submitted_Date',  
                        'Location_Verified_Date', 'Location', 'Full_Address' ])
    )
    
    df_customer_dim['KYC_Capture_Status'] = np.where(df_customer_dim['KYC_Capture_Status'] == 1, "Yes", "No")
    
    # Format columns to lower case
    df_customer_dim.columns = df_customer_dim.columns.str.lower()
    df_customer_dim['customer_id'] = df_customer_dim['customer_id'].astype(int, errors='ignore')
    df_customer_dim['latitude'] = df_customer_dim['latitude'].replace('', '0').replace('undefined', '0').astype(float, errors='ignore').fillna(0)
    df_customer_dim['longitude'] = df_customer_dim['longitude'].replace('', '0').replace('undefined', '0').astype(float, errors='ignore').fillna(0)
    df_customer_dim[['is_location_verified','is_location_captured']]= df_customer_dim[['is_location_verified','is_location_captured']].fillna(0)
    
    df_customer_dim[['latitude','longitude']] = df_customer_dim[['latitude','longitude']].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    #### ---------------------------------------------------------------------------
    df_sp_active_customers.columns = df_sp_active_customers.columns.str.lower()
    logger.info('Loading SP Active/Buying Customers (Jan2025 till date)')
    logger.info(f'Total Active Customers: {df_sp_active_customers.customer_id.nunique():,} ')
         
    cols = ['customer_id', 'business_id', 'created_date', 'contact_name', 'contact_phone', 
            'state_name', 'town_name', 'city_name', 'latitude', 'longitude', 'customer_status', 
            'address', 'kyc_capture_status', 'agent_id', 'agent_name']

    df_sp_customers = (df_sp_active_customers
                    .rename(columns={'customerid': 'customer_id'})
                    .merge(df_customer_dim[cols], how='inner', on=['customer_id'])
                    .rename(columns={'contact_came': 'customer_name',})
                    )
    # df_sp_customers[['latitude','longitude']] = df_sp_customers[['latitude','longitude']].astype(float, errors='ignore').fillna(0)
    logger.info(f' Total Number of SP Customer with (multiple sp-customer count): {len(df_sp_customers):,}')
      
    #### ---------------------------------------------------------------------------
    ## Recently Activated Customer that aren't current buying customers  
    #### ---------------------------------------------------------------------------
    logger.info(f'''Last 3Months Recently Activated Customer that aren't current buying customers''')  
    df_customer_dim['created_date'] = pd.to_datetime(df_customer_dim['created_date'])
    three_months_ago = datetime.now() - timedelta(days=3*30)  
    df_recent_customers = df_customer_dim[df_customer_dim['created_date'] >= three_months_ago] 

    df_recent_customers = (df_recent_customers
                            .merge(df_sp_customers[['customer_id']].drop_duplicates(), on='customer_id', how='left', indicator=True)
                            .query('_merge == "left_only"')
                            .drop('_merge', axis=1)
                            .reset_index(drop=True)
                            .sort_values('created_date', ascending=False)
                            )
    # df_recent_customers[['latitude','longitude']] = df_recent_customers[['latitude','longitude']].astype(float, errors='ignore').fillna(0)
    
    #### ---------------------------------------------------------------------------
    ## Saving the processed customer_dim    
    #### ---------------------------------------------------------------------------

    #### ---------------------------------------------------------------------------
    path_proccessed_sp_customer = PROCESSED_DATA_DIR / 'df_sp_customers.parquet'
    try:
        df_sp_customers.to_parquet(path_proccessed_sp_customer)
        logger.info(f'Saved SP active/buying customers df to file {path_proccessed_sp_customer}')
    except Exception as e:
        logger.error(f'Failed to save SP active/buying customers, {e}')
        
    #### ---------------------------------------------------------------------------
    logger.info(f'Total Number of recently Activated Customers: {len(df_recent_customers):,}') 
    path_recent_customers = PROCESSED_DATA_DIR / 'df_recent_customers.parquet' 
    try:
        df_recent_customers.to_parquet(path_recent_customers)
        logger.info(f'Saved recently activated Customer to file {path_recent_customers}')
    except Exception as e:
        logger.error(f'Failed to save recently activated Customer: {e}')
        
    #### ---------------------------------------------------------------------------
    logger.info(f'Total Number of Customers: {len(df_customer_dim):,}') 
    path_processed_customer_dim = PROCESSED_DATA_DIR / 'df_processed_customer_dim.parquet' 
    try:
        df_customer_dim.to_parquet(path_processed_customer_dim)
        logger.info(f'Saved recently activated Customer to file {path_processed_customer_dim}')
    except Exception as e:
        logger.error(f'Failed to save processed customer dim: {e}')
        
        