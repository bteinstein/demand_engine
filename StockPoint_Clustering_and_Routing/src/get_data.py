# src/get_data.py

import pandas as pd
import geopandas as gpd
from pathlib import Path
import pickle
from .utils import clean_customer_gdf_coordinates
import logging
import warnings  
import pyodbc
import json
from typing import Tuple
from config.settings import PROCESSED_DATA_DIR, INPUT_BASE_DATA_SOURCES, RAW_DATA_DIR #STORAGE_CONFIG, ADMIN_DATA_SOURCES, RAW_DATA_DIR


from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
import pandas as pd
import pyodbc
    
    
from codebase.database.get_connection import get_connection_string

# Get the directory where this script is located
src_dir = Path(__file__).parent
# Input folder is one level up from src, next to it (i.e., project root's input/)
input_dir = src_dir.parent / 'input'

    
def get_processed_data(logger) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """  
    Loads, cleans, and preprocesses multiple datasets for subsequent analysis.

    This function performs a series of data loading and cleaning operations,
    transforming raw data from various file formats (pickle and feather) into
    standardized and ready-to-use GeoDataFrames and DataFrames. It handles
    geospatial data by creating GeoDataFrames for Local Government Areas (LGAs)
    and customer locations, and applies a custom cleaning function to ensure
    coordinate validity. The function is designed to prepare all necessary
    geospatial and relational data in a single, consolidated step.

    Returns:
        Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
            A tuple containing the following processed data assets:
            - lgas_gdf (gpd.GeoDataFrame): GeoDataFrame of Local Government Areas with cleaned geometry.
            - sp_dim_df (pd.DataFrame): DataFrame containing stock point dimension data.
            - stock_point_lga_map (pd.DataFrame): DataFrame mapping stock points to LGAs.
            - customers_gdf (gpd.GeoDataFrame): GeoDataFrame of customer locations with cleaned coordinates.
    """
    
    path_processed_customer_dim = PROCESSED_DATA_DIR/ 'df_processed_customer_dim.parquet' 
    path_processed_lgas_gdf = PROCESSED_DATA_DIR / 'lgas_gdf.pickle'
    path_proccessed_sp_customer = PROCESSED_DATA_DIR/ 'df_sp_customers.parquet'
    path_recent_customers = PROCESSED_DATA_DIR/ 'df_recent_customers.parquet' 
    path_processed_uniq_sp_lgas_map_gdf = PROCESSED_DATA_DIR / 'sp_uniq_mapped_lgas_gdf.pickle' 
    path_processed_sp_lga_mapping_df = PROCESSED_DATA_DIR / 'processed_sp_lga_mapping_df.pickle' 
    path_processed_sp_dim_df = PROCESSED_DATA_DIR / 'processed_sp_dim_df.pickle' 
    
    sp_active_customers = pd.read_parquet(path_proccessed_sp_customer)
    recent_customers_df = pd.read_parquet(path_recent_customers)
    uniq_sp_lgas_map_gdf = pickle.load(open(str(path_processed_uniq_sp_lgas_map_gdf), 'rb'))
    sp_lga_mapping_df = pickle.load(open(str(path_processed_sp_lga_mapping_df), 'rb')) 
    sp_dim_df = pickle.load(open(str(path_processed_sp_dim_df), 'rb'))  
    # sp_lga_map_gdf = pickle.load(open(PROCESSED_DATA_DIR / 'processed_lga_gdf.pickle', 'rb')) 
    # survey_customer = pd.read_feather(RAW_DATA_DIR / 'df_sp_customers.feather')
      
    logger.info(f'Customer with recent activation {len(recent_customers_df):,}')
    
    ## Data Pre-Processing
    lga_data = (uniq_sp_lgas_map_gdf.copy()
                .rename(columns={'lga_name': 'name'})
                .assign(population_density = 0)
                [['state_name','lga_id', 'name', 'population_density', 'geometry','area_km2']]
                .dropna()
                )
    lgas_gdf = gpd.GeoDataFrame(lga_data, crs="EPSG:4326")
    lgas_gdf = lgas_gdf.dropna(subset=['geometry'])

    # SP LGA Mapping ------------------------------------
    stock_point_lga_map = sp_lga_mapping_df.copy() 
    
    # SP Customer ------------------------------------
    sel_cust_cols = ['stock_point_id', 'customer_id', 'contact_name', 'state_name', 'town_name',
                     'city_name', 'latitude', 'longitude', 'customer_status', 
                    'kyc_capture_status', 'agent_id', 'agent_name']
    
    sp_customer_data = sp_active_customers[sel_cust_cols]
    sp_customers_gdf = gpd.GeoDataFrame(sp_customer_data, 
                                    geometry=gpd.points_from_xy(sp_customer_data['longitude'], sp_customer_data['latitude']), 
                                    crs="EPSG:4326")
    
    logger.info(f'----- CLEANING SP ACTIVE CUSTOMER BY COODS ----------')
    sp_customers_gdf = clean_customer_gdf_coordinates(sp_customers_gdf)
    
    # Recent Customer ------------------------------------
    sel_r_cust_cols = ['customer_id', 'contact_name', 'state_name', 'town_name',
                      'city_name', 'latitude', 'longitude', 'customer_status', 
                      'kyc_capture_status', 'agent_id', 'agent_name']
    
    recent_customer_data = recent_customers_df[sel_r_cust_cols]
    recent_customers_gdf = gpd.GeoDataFrame(recent_customer_data, 
                                    geometry=gpd.points_from_xy(recent_customer_data['longitude'], recent_customer_data['latitude']), 
                                    crs="EPSG:4326")
    
    logger.info(f'----- CLEANING RECENT ACTIVATED CUSTOMER BY COODS (L3M) ----------')
    recent_customers_gdf = clean_customer_gdf_coordinates(recent_customers_gdf)

    return lgas_gdf, sp_dim_df,  stock_point_lga_map, sp_customers_gdf, recent_customers_gdf 




def execute_query_worker(args):
    """
    Worker function for parallel query execution.
    
    Args:
        args (tuple): (sql_path, output_path, connection_string, logger_config)
    
    Returns:
        tuple: (output_filename, success_status, error_message)
    """
    sql_path, output_path, connection_string = args
    
    try:
        # Read SQL query
        with open(sql_path, 'r') as file:
            sql_query = file.read().strip()

        if not sql_query:
            return (output_path.name, False, f"SQL query is empty: {sql_path}")

        # Execute query
        with pyodbc.connect(connection_string) as conn:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df = pd.read_sql(sql_query, con=conn)

        if df.empty:
            return (output_path.name, True, f"Query returned no data: {sql_path.name}")
        else:
            df.to_parquet(output_path)
            return (output_path.name, True, f"Saved {len(df)} rows to {output_path}")

    except FileNotFoundError:
        return (output_path.name, False, f"SQL file not found: {sql_path}")
    except Exception as e:
        return (output_path.name, False, f"Error executing query {sql_path.name}: {e}")
    
        
class DataFetcher:
    """
    A class to fetch data from the database and save it as Feather files.
    Optionally accepts a logger; creates one if not provided.
    """

    def __init__(self, logger=None, input_dir="./input", sql_dir="_sql"):
        """
        Initializes the DataFetcher with a logger and directory paths.

        Args:
            logger (logging.Logger, optional): An existing logger instance. If not provided,
                                            a default stream logger is created.
            input_dir (str): The path to the directory where fetched data will be saved.
                            Defaults to "./input".
            sql_dir (str): The path to the directory containing SQL query files.
                        Defaults to "_sql".
        """
        self.logger = logger or self._create_default_logger()
        self.input_dir = Path(input_dir)
        self.sql_dir = Path(sql_dir)
        self.repl_con_string = None
        self._ensure_dirs()

    def _create_default_logger(self):
        """Create a default logger if none is provided."""
        logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _ensure_dirs(self):
        """Ensure input directory exists."""
        try:
            self.input_dir.mkdir(exist_ok=True)
            self.logger.debug(f"Ensured input directory exists: {self.input_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create input directory {self.input_dir}: {e}")
            raise

    def _get_connection_string(self, database='VconnectMasterDWR', server_type='replica'):
        """
        Retrieves the database connection string and stores it in the instance.

        This method uses an external function `get_connection_string` to retrieve
        the connection details. It logs the process and raises a ValueError if
        the connection string is invalid.

        Args:
            database (str): The name of the database to connect to.
                            Defaults to 'VconnectMasterDWR'.
            server_type (str): The type of database server (e.g., 'replica').
                            Defaults to 'replica'.
        """
        try:
            self.logger.info("Fetching connection string...")
            self.repl_con_string = get_connection_string(
                logger=self.logger, database=database, server_type=server_type
            )
            if not self.repl_con_string:
                raise ValueError("Connection string is empty or invalid.")
        except Exception as e:
            self.logger.error(f"Failed to get connection string: {e}")
            raise

    def _execute_query_and_save(self, sql_filename: str, output_filename: str):
        """
        Executes a SQL query from a specified file and saves the result to a Feather file.

        The function reads a SQL query from a file, connects to the database using the
        instance's connection string, executes the query, and saves the resulting
        DataFrame to the specified output path. It handles cases where the query returns
        no data, or if file I/O or database connection errors occur. UserWarnings from
        pandas are suppressed to avoid cluttering the logs.

        Args:
            sql_filename (str): The name of the SQL file located in the `sql_dir`.
            output_filename (str): The name of the Feather file to be saved in the
                                `input_dir`.

        Returns:
            bool: True if the query was executed successfully and data was saved
                (or if the query returned no data), False otherwise.
    """
        sql_path = self.sql_dir / sql_filename
        output_path = self.input_dir / output_filename

        try:
            self.logger.info(f"Reading SQL query from {sql_path}")
            with open(sql_path, 'r') as file:
                sql_query = file.read().strip()

            if not sql_query:
                self.logger.warning(f"SQL query is empty: {sql_path}")
                return False

            self.logger.info(f"Executing query: {sql_filename}")
            with pyodbc.connect(self.repl_con_string) as conn:
                # Suppress UserWarning from pd.read_sql (e.g. dtype issues)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    df = pd.read_sql(sql_query, con=conn)

            if df.empty:
                self.logger.warning(f"Query returned no data: {sql_filename}")
            else:
                df.to_parquet(output_path)
                # df.to_feather(output_path)
                self.logger.info(f"Saved {len(df)} rows to {output_path}")

            return True

        except FileNotFoundError:
            self.logger.error(f"SQL file not found: {sql_path}")
            return False
        except Exception as e:
            self.logger.error(f"Error executing query {sql_filename}: {e}")
            return False

    def fetch_all(self):
        """
        Fetches all predefined datasets by executing a series of SQL queries.

        This method orchestrates the data fetching process. It first ensures the
        database connection string is available, then iterates through a list of
        predefined tasks, each representing a pair of a SQL file and a target
        output file. It calls `_execute_query_and_save` for each task and records
        the success status.

        The following data fetching tasks are performed:
        - `sp_dim.sql` -> `df_sp_dim.feather`: Fetches stock point dimension data.
        - `sp_location_map.sql` -> `df_sp_location_mapping.feather`: Fetches the mapping of stock points to locations.
        - `get_customer_dim.sql` -> `df_customer_dim.feather`: Fetches customer dimension data.
        - `sp_active_customers.sql` -> `df_sp_active_customers.feather`: Fetches data for active customers associated with stock points.

        Returns:
            dict: A dictionary mapping output filenames to a boolean success
                status (True for success, False for failure).
        """
        if not self.repl_con_string:
            self._get_connection_string()
                       
        # List of (sql_file, output_feather_file)
        base_data_input_filename_path = {
            k: v["local_file_path"].name for k, v in INPUT_BASE_DATA_SOURCES.items()
        } 
        
        tasks = [
            ("sp_dim.sql", base_data_input_filename_path['sp_dim'] ), #"df_sp_dim.feather"
            ("sp_location_map.sql", base_data_input_filename_path['sp_location_mapping'] ), #"df_sp_location_mapping.feather"),
            ("get_customer_dim.sql", base_data_input_filename_path['customer_dim'] ), #"df_customer_dim.feather"),
            ("sp_active_customers.sql", base_data_input_filename_path['sp_active_customers'] ), #"df_sp_active_customers.feather"),
        ]

        results = {}
        for sql_file, output_file in tasks:
            success = self._execute_query_and_save(sql_file, output_file)
            results[output_file] = success
            if not success:
                self.logger.error(f"Failed to fetch data for {output_file}")

        return results


    def fetch_all_parallel(self, max_workers=None):
        """
        Parallel version of fetch_all using ProcessPoolExecutor.
        
        Args:
            max_workers (int): Maximum number of worker processes. 
                              Defaults to min(4, cpu_count()).
        
        Returns:
            dict: A dictionary mapping output filenames to boolean success status.
        """
        if not self.repl_con_string:
            self._get_connection_string()
        
        # Prepare tasks
        base_data_input_filename_path = {
            k: v["local_file_path"].name for k, v in INPUT_BASE_DATA_SOURCES.items()
        }
        
        tasks = [
            ("sp_dim.sql", base_data_input_filename_path['sp_dim']),
            ("sp_location_map.sql", base_data_input_filename_path['sp_location_mapping']),
            ("get_customer_dim.sql", base_data_input_filename_path['customer_dim']),
            ("sp_active_customers.sql", base_data_input_filename_path['sp_active_customers']),
        ]

        # Prepare arguments for worker processes
        worker_args = []
        for sql_file, output_file in tasks:
            sql_path = self.sql_dir / sql_file
            output_path = self.input_dir / output_file
            worker_args.append((sql_path, output_path, self.repl_con_string))

        # Set max_workers
        if max_workers is None:
            max_workers = min(4, cpu_count())  # Conservative default

        results = {}
        
        # Execute queries in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(execute_query_worker, args): args[1].name 
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                output_file, success, message = future.result()
                results[output_file] = success
                
                if success:
                    self.logger.info(message)
                else:
                    self.logger.error(message)

        return results

    def fetch_all_parallel_simple(self):
        """
        Simplified parallel version that maintains the same interface as fetch_all.
        
        Returns:
            dict: A dictionary mapping output filenames to boolean success status.
        """
        return self.fetch_all_parallel(max_workers=None)
    
    
          
def get_geojson_data(PATH: Path):
    try:
        with open(PATH, 'r') as f:
            geojson_data = json.load(f)
        print(f"Successfully loaded GeoJSON data from {PATH}")
        return geojson_data
    except FileNotFoundError:
        print(f"Error: GeoJSON file not found at {PATH}. Please ensure it exists.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {PATH}. Check file integrity.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading GeoJSON: {e}")
        exit()   
        
        
        
        