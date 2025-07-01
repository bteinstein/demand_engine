import pandas as pd
import numpy as np

def data_filter(df_customer_sku_recommendation, df_master_customer_dim, df_stockpoint_dim,
                stockpoint_id,  sku_recency = 7, customer_recency = 90, number_recommendation = 5,
                estimate_qty_scale_factor = .90, max_estimated_qty = 5, exclude_recency_customer = 4):
    
    df_customer_sku_recommendation = df_customer_sku_recommendation.copy().query(f'Stock_Point_ID == {stockpoint_id}')
    # Filter Recommendation
    df_customer_sku_recommendation = df_customer_sku_recommendation[df_customer_sku_recommendation['ProductTag'] != 'Standard-Inactive']
    df_customer_sku_recommendation = df_customer_sku_recommendation[df_customer_sku_recommendation['Medium'] != 'Never Purchased']

    # Filter customer base
    df_master_customer_dim['valid_for_push'] = np.where(
                                                    #  df_master_customer_dim['KYC_Capture_Status'] == 'Yes'   
                                                    (
                                                        (df_master_customer_dim['IsLocationCaptured'] == 'Yes') |
                                                        (df_master_customer_dim['DistanceVarianceInMeter'] <= 150.0) |
                                                        (df_master_customer_dim['KYC_Capture_Status'] == 'Yes') |
                                                        (df_master_customer_dim['CustomerPurchaseRecency'] <= customer_recency)
                                                    )
                                                    ,1,0
                                                )
    # df_master_customer_dim = df_master_customer_dim[df_master_customer_dim['CustomerPurchaseRecency'] <= customer_recency]
    df_master_customer_dim = df_master_customer_dim.query('valid_for_push == 1')  
    # Exclude Customer with recent purchase of any SKU
    df_master_customer_dim = df_master_customer_dim.query(f'CustomerPurchaseRecency > {exclude_recency_customer}')
    # Customer with valid Location Coordination
    df_master_customer_dim = df_master_customer_dim.query('Latitude != 0').reset_index(drop=True)
    
    # # Clipping Max Estimated Quantity to 10 qty
    df_customer_sku_recommendation['EstimatedQuantity_bck'] = df_customer_sku_recommendation['EstimatedQuantity']
    df_customer_sku_recommendation['EstimatedQuantity'] = df_customer_sku_recommendation['EstimatedQuantity'].apply(lambda x: max_estimated_qty if int((x*estimate_qty_scale_factor)) > max_estimated_qty else int((x*estimate_qty_scale_factor)) )


    # Select top 10 SKU by SKURank per customer
    df_customer_sku_recommendation = (
        df_customer_sku_recommendation
        .query('EstimatedQuantity > 1')
        .sort_values(['CustomerID','CustomerSKUscoreRank'])
        .groupby('CustomerID', group_keys=False)
        .head(number_recommendation)
        .reset_index(drop=True) 
    )

    df_customer_sku_recommendation_ = df_master_customer_dim.merge(df_customer_sku_recommendation, how='inner', on = ['CustomerID','Stock_Point_ID'])  

    df_stockpoint_dim = df_stockpoint_dim.query(f'Stock_Point_ID == {stockpoint_id}').reset_index(drop=True) 
    

    df_customer_dim = df_master_customer_dim.merge(df_customer_sku_recommendation_['CustomerID'].drop_duplicates(), how='inner', on = 'CustomerID')
    # df_customer_dim = df_customer_dim.merge(df_customer_dim_with_affinity_score[sel_cols], how='inner', on = 'CustomerID').reset_index(drop = True) 
    
    print(f'Total Quantity before filter: {df_customer_sku_recommendation.query(f"Stock_Point_ID == {stockpoint_id}").EstimatedQuantity.sum():,}')
    print(f'Total Quantity: {df_customer_sku_recommendation_.EstimatedQuantity.sum():,}')
    print(f'Total Number of Customers before filter: {df_customer_sku_recommendation.query(f"Stock_Point_ID == {stockpoint_id}").CustomerID.nunique():,}')
    print(f'Total Number of Customers: {df_customer_dim.CustomerID.nunique():,}')

 
    return df_customer_sku_recommendation_, df_customer_dim,   df_stockpoint_dim  