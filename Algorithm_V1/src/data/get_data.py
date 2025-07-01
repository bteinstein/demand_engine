from .get_connection import get_connection
import pandas as pd
from pathlib import Path
import logging 
from typing import Optional, Tuple # <--- Import Tuple here
from pathlib import Path 
current_script_path_obj_pathlib = Path(__file__).resolve()
INPUT_DIR =   current_script_path_obj_pathlib.parent.parent.parent / 'input'
    
    
    

# @contextmanager
def execute_stored_procedure():
    # Get a database connection
    conn = get_connection()
    cursor = conn.cursor()

    # Execute the stored procedure
    cursor.execute("{CALL sp_GetCustomerSKUSalesRecommendationALGO()}")

    # Fetch the first result set into a DataFrame
    rows = cursor.fetchall()
    df1 = pd.DataFrame.from_records(rows, columns=[column[0] for column in cursor.description])

    # Fetch the second result set if it exists
    if cursor.nextset():
        rows = cursor.fetchall()
        df2 = pd.DataFrame.from_records(rows, columns=[column[0] for column in cursor.description])
    else:
        df2 = pd.DataFrame()  # Return an empty DataFrame if there's no second result set
    
    # Fetch the second result set if it exists
    if cursor.nextset():
        rows = cursor.fetchall()
        df3 = pd.DataFrame.from_records(rows, columns=[column[0] for column in cursor.description])
    else:
        dfdf32 = pd.DataFrame()  # Return an empty DataFrame if there's no second result set

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return df1, df2, df3
 

 

def get_data_reco_custdim_spdim(
        save_local: bool = False,
        input_dir_path: Optional[Path] = None, # Changed here!
        logger: logging.Logger = logging.getLogger(__name__)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes a stored procedure (or set of procedures) to retrieve customer SKU recommendations,
    customer dimensions with affinity scores, and stockpoint dimensions.
    Optionally saves these DataFrames to Feather files.

    Args:
        save_local (bool): If True, the resulting DataFrames will be saved to
                           Feather files in the specified input directory.
        input_dir_path (Path, optional): The pathlib.Path object pointing to the
                                         directory where files should be saved.
                                         Defaults to None, in which case `INPUT_DIR`
                                         (if globally defined) will be used.
                                         It's highly recommended to pass this explicitly
                                         for better testability and modularity.
        logger (logging.Logger): A logger instance to use for logging messages.
                                 Defaults to a logger named after the current module.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three
        DataFrames: customer_sku_recommendation, customer_dim_with_affinity_score,
        and stockpoint_dim. Returns empty DataFrames in case of an error.
    """
    # Initialize DataFrames as empty to ensure a consistent return type
    customer_sku_recommendation = pd.DataFrame()
    customer_dim_with_affinity_score = pd.DataFrame()
    stockpoint_dim = pd.DataFrame()

    try:
        # Log instead of print for better control and output management
        logger.info("Executing stored procedure(s) to fetch data...")
        # Assume execute_stored_procedure() handles its own database connection/cursor management internally,
        # or it should be modified to use try-finally for connection closing as discussed previously.
        customer_sku_recommendation, customer_dim_with_affinity_score, stockpoint_dim = execute_stored_procedure()
        logger.info("Data fetch complete.")

    except Exception as e:
        logger.exception("An error occurred during data fetching.") # Use exception for full traceback
        # Depending on the desired behavior, you might want to re-raise the exception:
        # raise # Uncomment this line if you want the error to propagate

    # Log the DataFrames' shapes instead of printing directly
    logger.info("--- Fetched DataFrames Shapes ---")
    logger.info(f"Customer SKU Recommendation: {customer_sku_recommendation.shape}")
    logger.info(f"Customer Dimension with Affinity Score: {customer_dim_with_affinity_score.shape}")
    logger.info(f"Stockpoint Dimension: {stockpoint_dim.shape}")
    logger.info("---------------------------------\n")

    # Avg. Run Time: 5mins - This is a comment, so keep it as is.

    if save_local:
        # Determine the target directory for saving
        target_dir = input_dir_path if input_dir_path else globals().get('INPUT_DIR')

        if target_dir is None:
            logger.error("`INPUT_DIR` is not defined globally and `input_dir_path` was not provided. Skipping local data save.")
        elif not isinstance(target_dir, Path):
            logger.error(f"`input_dir_path` or `INPUT_DIR` must be a pathlib.Path object. Got {type(target_dir)}. Skipping local data save.")
        else:
            try:
                # Ensure the directory exists before saving
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f'Saving files to disk in: {target_dir} ...')

                # Only save if the DataFrame is not empty
                if not customer_sku_recommendation.empty:
                    customer_sku_recommendation.to_feather(target_dir / 'customer_sku_recommendation.feather')
                    logger.info("Saved 'customer_sku_recommendation.feather'")
                else:
                    logger.info("Skipping save for 'customer_sku_recommendation' (DataFrame is empty).")

                if not customer_dim_with_affinity_score.empty:
                    customer_dim_with_affinity_score.to_feather(target_dir / 'customer_dim_with_affinity_score.feather')
                    logger.info("Saved 'customer_dim_with_affinity_score.feather'")
                else:
                    logger.info("Skipping save for 'customer_dim_with_affinity_score' (DataFrame is empty).")

                if not stockpoint_dim.empty:
                    stockpoint_dim.to_feather(target_dir / 'stockpoint_dim.feather')
                    logger.info("Saved 'stockpoint_dim.feather'")
                else:
                    logger.info("Skipping save for 'stockpoint_dim' (DataFrame is empty).")

                logger.info("All requested data save operations completed.")

            except Exception as e:
                logger.exception(f"An error occurred while saving data locally to {target_dir}.") # Use exception for full traceback
                # Optionally re-raise the exception if you want the caller to handle it:
                # raise

    return customer_sku_recommendation, customer_dim_with_affinity_score, stockpoint_dim
 

def get_kyc_customers(
    save_local: bool = False,
    input_dir_path: Optional[Path] = None, # Added optional input_dir_path
    logger: logging.Logger = logging.getLogger(__name__)
) -> pd.DataFrame:
    """
    Executes the stored procedure 'usp_GetCustomerKYCInfoDetailsV2',
    optionally saves the result to a Feather file, and returns it as a DataFrame.

    Args:
        save_local (bool): If True, the resulting DataFrame will be saved to a
                           Feather file in the specified input directory.
        input_dir_path (Path, optional): The pathlib.Path object pointing to the
                                         directory where the Feather file should be saved.
                                         If None (default), it will attempt to use
                                         a globally defined `INPUT_DIR`.
        logger (logging.Logger): A logger instance to use for logging messages.
                                 Defaults to a logger named after the current module.

    Returns:
        pd.DataFrame: DataFrame containing customer KYC information.
                      Returns an empty DataFrame if no data is found or an error occurs.
    """
    conn = None  # Initialize conn to None
    cursor = None # Initialize cursor to None
    df_kyc_customer = pd.DataFrame() # Initialize as empty DataFrame

    try:
        logger.info("Connecting to database and fetching KYC customer data...")
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("EXEC usp_GetCustomerKYCInfoDetailsV2")
        rows = cursor.fetchall()

        if rows: # Check if any rows were returned
            columns = [column[0] for column in cursor.description]
            df_kyc_customer = pd.DataFrame.from_records(rows, columns=columns)
            logger.info(f"Successfully fetched KYC customer data. Shape: {df_kyc_customer.shape}")
        else:
            logger.info("No KYC customer data found.")

    except Exception as e:
        logger.exception("An error occurred while fetching KYC customer data.")
        # Optionally re-raise the exception if you want the caller to handle it
        # raise
    finally:
        # Ensure cursor and connection are closed in all cases
        if cursor: # Check if cursor object was successfully created
            cursor.close()
            logger.debug("Database cursor closed.")
        if conn: # Check if connection object was successfully created
            conn.close()
            logger.debug("Database connection closed.")

    if save_local and not df_kyc_customer.empty:
        # Determine the target directory for saving
        target_dir = input_dir_path if input_dir_path else globals().get('INPUT_DIR')

        if target_dir is None:
            logger.error("`INPUT_DIR` is not defined globally and `input_dir_path` was not provided. Skipping local save for KYC customer data.")
        elif not isinstance(target_dir, Path):
            logger.error(f"Provided `input_dir_path` or global `INPUT_DIR` must be a pathlib.Path object. Got {type(target_dir)}. Skipping local save.")
        else:
            save_path = target_dir / 'df_kyc_customer.feather'
            try:
                # Ensure the directory exists before saving
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f'Saving KYC customer data to: {save_path} ...')
                df_kyc_customer.to_feather(save_path)
                logger.info("KYC customer data saved successfully.")
            except Exception as e:
                logger.exception(f"An error occurred while saving KYC customer data to {save_path}.")
                # Optionally, handle save errors differently or re-raise

    return df_kyc_customer

 


def get_customer_score(
    save_local: bool = False,
    input_dir_path: Optional[Path] = None, # Added optional input_dir_path
    logger: logging.Logger = logging.getLogger(__name__) # Added logger
) -> pd.DataFrame:
    """
    Executes a query to retrieve customer score data,
    optionally saves the result to a Feather file, and returns it as a DataFrame.

    Args:
        save_local (bool): If True, the resulting DataFrame will be saved to a
                           Feather file in the specified input directory.
        input_dir_path (Path, optional): The pathlib.Path object pointing to the
                                         directory where the Feather file should be saved.
                                         If None (default), it will attempt to use
                                         a globally defined `INPUT_DIR`.
        logger (logging.Logger): A logger instance to use for logging messages.
                                 Defaults to a logger named after the current module.

    Returns:
        pd.DataFrame: DataFrame containing customer score data.
                      Returns an empty DataFrame if no data is found or an error occurs.
    """
    conn = None  # Initialize conn to None
    cursor = None # Initialize cursor to None
    df_customer_score = pd.DataFrame() # Initialize as empty DataFrame

    try:
        logger.info("Connecting to database and fetching customer score data...")
        conn = get_connection()
        cursor = conn.cursor()

        # It's generally better practice to use stored procedure execution if that's the intent,
        # rather than selecting directly from a "poc_stockpoint_customer_score" object.
        # If 'poc_stockpoint_customer_score' is a view or a table, then SELECT * is fine.
        # If it's a stored procedure, the 'EXEC' syntax is usually preferred.
        # Assuming it's a view/table based on your current query.
        cursor.execute("SELECT * FROM VConnectMasterDWR..poc_stockpoint_customer_score")
        rows = cursor.fetchall()

        if rows: # Check if any rows were returned
            columns = [col[0] for col in cursor.description]
            df_customer_score = pd.DataFrame.from_records(rows, columns=columns)
            logger.info(f"Successfully fetched customer score data. Shape: {df_customer_score.shape}")
        else:
            logger.info("No customer score data found.")

    except Exception as e:
        # Catch any exceptions that occur during database operations
        logger.exception("An error occurred while fetching customer score data.")
        # Optionally re-raise the exception if you want the caller to handle it
        # raise # Uncomment this line if you want the error to propagate

    finally:
        # Ensure cursor and connection are closed in all cases
        if cursor: # Check if cursor object was successfully created
            cursor.close()
            logger.debug("Database cursor closed.")
        if conn: # Check if connection object was successfully created
            conn.close()
            logger.debug("Database connection closed.")

    if save_local and not df_customer_score.empty:
        # Determine the target directory for saving
        target_dir = input_dir_path if input_dir_path else globals().get('INPUT_DIR')

        if target_dir is None:
            logger.error("`INPUT_DIR` is not defined globally and `input_dir_path` was not provided. Skipping local save for customer score data.")
        elif not isinstance(target_dir, Path):
            logger.error(f"Provided `input_dir_path` or global `INPUT_DIR` must be a pathlib.Path object. Got {type(target_dir)}. Skipping local save.")
        else:
            save_path = target_dir / 'df_customer_score.feather'
            try:
                # Ensure the directory exists before saving
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f'Saving customer score data to: {save_path} ...')
                df_customer_score.to_feather(save_path)
                logger.info("Customer score data saved successfully.")
            except Exception as e:
                logger.exception(f"An error occurred while saving customer score data to {save_path}.")
                # Optionally, handle save errors differently or re-raise

    return df_customer_score

