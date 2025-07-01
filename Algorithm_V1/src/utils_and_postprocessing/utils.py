import logging 
import pandas as pd 
import numpy as np


def create_stockpoint_dict(df_selected_trip, df_stockpoint_dim, logger=logging.getLogger(__name__)):
    """
    Create a dictionary structure with stock point information and associated trips.
    
    Parameters:
    df_selected_trip: DataFrame with columns ['StockPointID', 'StockPointName', 'TripID', 'CustomerID', 'Latitude', 'Longitude', 'EstimatedQuantity']
    df_stockpoint_dim: DataFrame with columns ['Stock_Point_ID', 'Stock_point_Name', 'Latitude', 'Longitude']
    
    Returns:
    dict: Dictionary with stock point information and trips
    """
    if df_selected_trip.empty:
        logger.info('Dataframe is empty')
        return {}
    
    # Group by StockPointID to handle each stock point
    stockpoint_groups = df_selected_trip.groupby('StockPointID')
    
    result = {}
    
    for stock_point_id, group in stockpoint_groups:
        # Get stock point information from df_stockpoint_dim
        stock_point_info = df_stockpoint_dim[df_stockpoint_dim['Stock_Point_ID'] == stock_point_id]
        
        if stock_point_info.empty:
            # If stock point not found in dimension table, use info from selected_trip
            stock_point_name = group['StockPointName'].iloc[0]
            # Note: We'll need to get coordinates from somewhere since customer coordinates 
            # in df_selected_trip are for destinations, not stock points
            stock_point_coord = [0, 0]  # Placeholder - you may need to adjust this
        else:
            stock_point_name = stock_point_info['Stock_point_Name'].iloc[0]
            stock_point_coord = [
                stock_point_info['Longitude'].iloc[0], 
                stock_point_info['Latitude'].iloc[0]
            ]
        
        # Group by TripID to organize trips
        trip_groups = group.groupby('TripID')
        trips = []
        
        for trip_id, trip_group in trip_groups:
            # Create destinations list for this trip
            destinations = []
            for _, row in trip_group.iterrows():
                destination = {
                    'CustomerID': row['CustomerID'],
                    'Coordinate': [row['Longitude'], row['Latitude']]
                }
                destinations.append(destination)
            
            # Create trip dictionary
            trip_dict = {
                'TripID': trip_id,
                'Destinations': destinations
            }
            trips.append(trip_dict)
        
        # Create the final dictionary structure for this stock point
        result[stock_point_id] = {
            'StockPointName': stock_point_name,
            'StockPointID': stock_point_id,
            'StockPointCoord': stock_point_coord,
            'Trips': trips
        }
    
    return result


# Alternative version if you want a single dictionary (assuming only one stock point)
def create_single_stockpoint_dict(df_selected_trip, df_stockpoint_dim, logger=logging.getLogger(__name__)):
    """
    Create a single dictionary structure for one stock point.
    
    Parameters:
    df_selected_trip: DataFrame with trip data for one stock point
    df_stockpoint_dim: DataFrame with stock point dimension data
    
    Returns:
    dict: Single dictionary with stock point information and trips
    """
    if df_selected_trip.empty:
        logger.info('Dataframe is empty')
        return {}
    
    # Get the stock point ID (assuming all rows have the same stock point)
    stock_point_id = df_selected_trip['StockPointID'].iloc[0]
    
    # Get stock point information from df_stockpoint_dim
    stock_point_info = df_stockpoint_dim[df_stockpoint_dim['Stock_Point_ID'] == stock_point_id]
    
    if stock_point_info.empty:
        stock_point_name = df_selected_trip['StockPointName'].iloc[0]
        stock_point_coord = [0, 0]  # Placeholder
    else:
        stock_point_name = stock_point_info['Stock_point_Name'].iloc[0] 
        stock_point_coord = [
            stock_point_info['Longitude'].iloc[0], 
            stock_point_info['Latitude'].iloc[0]
        ]
    
    # Group by TripID
    trip_groups = df_selected_trip.groupby('TripID')
    trips = []
    
    for trip_id, trip_group in trip_groups:
        destinations = []
        for _, row in trip_group.iterrows():
            destination = {
                'CustomerID': row['CustomerID'],
                'Coordinate': [row['Longitude'], row['Latitude']]
            }
            destinations.append(destination)
        
        trip_dict = {
            'TripID': trip_id,
            'Destinations': destinations
        }
        trips.append(trip_dict)
    
    # Return the final dictionary
    return {
        'StockPointName': stock_point_name,
        'StockPointID': stock_point_id,
        'StockPointCoord': stock_point_coord,
        'Trips': trips
    }


# Example usage:
# """
# # For multiple stock points:
# result_dict = create_stockpoint_dict(df_selected_trip, df_stockpoint_dim)

# # For a single stock point:
# single_result = create_single_stockpoint_dict(df_selected_trip, df_stockpoint_dim)
# """


def cluster_summary_and_selection(push_recommendation,
                                  sel_trip_cluster,
                                  min_ncust_per_cluster = 4,
                                  logger=logging.getLogger(__name__)
                                  ):
    ### Cluster Summary 
    cluster_summary = (
        push_recommendation
        .groupby('cluster').agg(
            LGA_list = ('LGA', lambda x: x.unique().tolist()),
            LCDA_List = ('LCDA', lambda x: x.unique().tolist()),
            ncustomer = ('CustomerID','nunique'),
            totalQty = ('EstimatedQuantity','sum'), 
            avg_customer_score = ('composite_customer_score','mean'),
        )
        .reset_index()
        .sort_values(['avg_customer_score','ncustomer', 'totalQty'], 
                     ascending=[False, False, False])
        )

    ### Select Trip   
    df_high_value_cluster_summary = (
            cluster_summary
            .query(f'ncustomer >= {min_ncust_per_cluster}')
            .head(max(10, sel_trip_cluster))
            .reset_index(drop = True)
        )
    sel_cluster_tuple = df_high_value_cluster_summary.cluster[0:sel_trip_cluster].to_list()
    sel_total_customer_count = df_high_value_cluster_summary.head(sel_trip_cluster).ncustomer.sum()
    logger.debug(f'''Select ClusterIDs: {sel_cluster_tuple}''')
    logger.debug(f'''Total Number of Customers: {sel_total_customer_count}''')
    logger.debug(df_high_value_cluster_summary.head(sel_trip_cluster).to_string(index=True))

    return cluster_summary, df_high_value_cluster_summary, sel_cluster_tuple, sel_total_customer_count



def postprocess_selected_trip(push_recommendation, 
                       cluster_summary,
                       df_master_customer_dim,  
                       df_stockpoint,
                       sel_cluster_tuple):
    
        

    sel_columns = ['Stock_Point_ID', 
                'StateName', # 'Region', 
                'Latitude', 'Longitude', 'LGA', 'LCDA', 'cluster', 
                'CustomerID', 'SKUID', 'ProductName', 'Output',
                'LastDeliveredDate', 'Recency', 'InventoryCheck', 'ProductTag', 'Medium',
                'EstimatedQuantity', 
                # 'CustomerSKUscoreRank'
                ]

    sel_cols_cust= ['Stock_Point_ID', 'CustomerID', 'ContactName',  'CustomerModeName',   'ContactPhone', 'FullAddress', 
                    'composite_customer_score', 'percentile_rank',  'KYC_Capture_Status', 'CustomerPurchaseRecency']

    final_cols = ['Stock_Point_ID', 'Stock_point_Name', 'TripID', 'LGA_list', 'LCDA_List', 
                  'ncustomer', 'totalQty','avg_customer_score', 'CustomerID', 'ContactName',  
                  'CustomerModeName',   'ContactPhone', 'FullAddress', 'Latitude',
                  'Longitude', 'LGA', 'LCDA', 'composite_customer_score', #, 'percentile_rank',  
                  'KYC_Capture_Status', 'SKUID', 'ProductName', #'Output', 'LastDeliveredDate', 
                  'Recency','CustomerPurchaseRecency', 'InventoryCheck', 'ProductTag', 'Medium', 'EstimatedQuantity',
                ]
    
    def _merge_select(df):
        modified_df = (
                        df[sel_columns]
                        .merge(cluster_summary, how='left', on = 'cluster' )
                        .merge(df_master_customer_dim[sel_cols_cust], how='left', on = ['Stock_Point_ID', 'CustomerID'])
                        .merge(df_stockpoint[['Stock_Point_ID', 'Stock_point_Name']], how='left', on = ['Stock_Point_ID'])
                        .rename(columns={'cluster':'TripID'})
                        [final_cols]
                        .rename(columns = {
                                           'Stock_point_Name': 'StockPointName'
                                           ,'Stock_Point_ID': 'StockPointID'
                                           ,'ncustomer': 'TotalCustonerCount'
                                           ,'totalQty': 'TripTotalQuantity'
                                           ,'avg_customer_score': 'TripAvgCustomerScore'
                                           ,'LastDeliveredDate': 'CustomerLastDeliveredDate'
                                           ,'Medium': 'RecommendationType'
                                           ,'Recency': 'SKUDaysSinceLastBuy'
                                           ,'CustomerPurchaseRecency': 'CustomerDaysSinceLastBuy'
                                           ,'composite_customer_score': 'CustomerScore'
                                           ,'KYC_Capture_Status': 'kycCaptureStatus'
                                           ,'LGA_list': 'ClusterLGAs'
                                           ,'LCDA_List': 'ClusterLCDAs'
                                           })
                        )
        return modified_df

    df_selected_trip = push_recommendation[push_recommendation['cluster'].isin(sel_cluster_tuple)]
    selected_push_recommendation_trip = _merge_select(df_selected_trip)
    all_push_recommendation =  _merge_select(push_recommendation)
    all_push_recommendation['isTripSelected'] = np.where(all_push_recommendation['TripID'].isin(sel_cluster_tuple) ,
                                                    'Yes',
                                                    'No'
                                                )
    

    return selected_push_recommendation_trip, all_push_recommendation

























