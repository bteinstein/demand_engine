import pandas as pd 
from .get_connection import get_connection

# from contextlib import contextmanager


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


def get_data():
    # Execute the function and get the DataFrames
    customer_sku_recommendation, customer_dim_with_affinity_score, stockpoint_dim = execute_stored_procedure()

    # Print the DataFrames
    print("DataFrame 1:")
    print(customer_sku_recommendation.shape)
    print("\nDataFrame 2:")
    print(customer_dim_with_affinity_score.shape) 
    print("\nDataFrame 3:")
    print(stockpoint_dim.shape)


    # Avg. Run Time: 5mins
    
    print('Saving file to disk ...')
    customer_sku_recommendation.to_feather('./input/customer_sku_recommendation.feather')
    customer_dim_with_affinity_score.to_feather('./input/customer_dim_with_affinity_score.feather')
    stockpoint_dim.to_feather('./input/stockpoint_dim.feather')

    return customer_sku_recommendation,  customer_dim_with_affinity_score, stockpoint_dim

 

def get_kyc_customers(save_path: str = "./input/kyc_customers.feather") -> pd.DataFrame:
    """
    Executes the stored procedure 'usp_GetCustomerKYCInfoDetailsV2',
    saves the result to a Feather file, and returns it as a DataFrame.

    Args:
        save_path (str): Path to save the resulting Feather file.

    Returns:
        pd.DataFrame: DataFrame containing customer KYC information.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("EXEC usp_GetCustomerKYCInfoDetailsV2")
    rows = cursor.fetchall()
    df_kyc_customer = pd.DataFrame.from_records(rows, columns=[column[0] for column in cursor.description])

    cursor.close()
    conn.close()

    df_kyc_customer.to_feather(save_path)
    
    return df_kyc_customer


def get_customer_score(save_path: str =  "./input/df_customer_score.feather") -> pd.DataFrame:
    """
    Executes the stored procedure 'poc_stockpoint_customer_score' and returns the result as a DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing customer score data.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM VConnectMasterDWR..poc_stockpoint_customer_score")
    rows = cursor.fetchall()
    df_customer_score = pd.DataFrame.from_records(rows, columns=[col[0] for col in cursor.description])

    cursor.close()
    conn.close()
    
    df_customer_score.to_feather(save_path)

    return df_customer_score
