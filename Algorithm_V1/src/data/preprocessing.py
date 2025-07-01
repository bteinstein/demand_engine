import pandas as pd
from datetime import datetime

def clean_invalid_coordinates(df: pd.DataFrame, bound_to_nigeria = True, offset_degrees: float = 0.1) -> pd.DataFrame:
    """
    Replaces invalid Latitude (< -90 or > 90) and Longitude (< -180 or > 180) values with 0.0.
    Also replaces coordinates outside Nigeria's approximate boundaries (with an optional offset) with 0.0.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Latitude' and 'Longitude' columns.
        offset_degrees (float): Degrees to add/subtract from the strict Nigeria boundary
                                to expand the bounding box. Default is 0.1 degrees.

    Returns:
        pd.DataFrame: DataFrame with corrected coordinate values.
    """
    df = df.copy()

    # Global invalid coordinate ranges
    df.loc[(df['Latitude'] < -90) | (df['Latitude'] > 90), 'Latitude'] = 0.0
    df.loc[(df['Longitude'] < -180) | (df['Longitude'] > 180), 'Longitude'] = 0.0

    if not bound_to_nigeria:
        ### Nigeria Boundary Filter ###
        # Approximate decimal degree boundaries for Nigeria
        STRICT_NIGERIA_MIN_LAT = 4.10
        STRICT_NIGERIA_MAX_LAT = 13.90
        STRICT_NIGERIA_MIN_LON = 2.60
        STRICT_NIGERIA_MAX_LON = 14.70

        # Apply offset to expand the bounding box
        NIGERIA_MIN_LAT = STRICT_NIGERIA_MIN_LAT - offset_degrees
        NIGERIA_MAX_LAT = STRICT_NIGERIA_MAX_LAT + offset_degrees
        NIGERIA_MIN_LON = STRICT_NIGERIA_MIN_LON - offset_degrees
        NIGERIA_MAX_LON = STRICT_NIGERIA_MAX_LON + offset_degrees

        # Identify coordinates outside Nigeria's expanded bounding box
        # Condition for rows outside Nigeria's latitude range
        outside_nigeria_lat = (df['Latitude'] < NIGERIA_MIN_LAT) | \
                            (df['Latitude'] > NIGERIA_MAX_LAT)

        # Condition for rows outside Nigeria's longitude range
        outside_nigeria_lon = (df['Longitude'] < NIGERIA_MIN_LON) | \
                            (df['Longitude'] > NIGERIA_MAX_LON)

        # Combine conditions: if EITHER latitude OR longitude is outside Nigeria's expanded box,
        # then set BOTH Latitude and Longitude for that row to 0.0.
        # We apply this only to coordinates that are already globally valid (i.e., not 0.0).
        df.loc[
            (df['Latitude'] != 0.0) &
            (df['Longitude'] != 0.0) &
            (outside_nigeria_lat | outside_nigeria_lon),
            ['Latitude', 'Longitude']
        ] = 0.0

    return df



def preprocessing(df_customer_sku_recommendation_raw, 
                      df_customer_dim_with_affinity_score_raw, 
                      df_stockpoint_dim_raw,
                      df_customer_score,
                      df_kyc_customer) :
    
    df_customer_sku_recommendation_raw.rename(columns={'FCID':'Stock_Point_ID','CustomerId':'CustomerID'}, inplace=True)
    df_customer_dim_with_affinity_score_raw.rename(columns={'FCID':'Stock_Point_ID'}, inplace=True)
    
    df_customer_sku_recommendation_raw['Stock_Point_ID'] = df_customer_sku_recommendation_raw['Stock_Point_ID'].astype(int)
    df_customer_dim_with_affinity_score_raw['Stock_Point_ID'] = df_customer_dim_with_affinity_score_raw['Stock_Point_ID'].astype(int)
    df_stockpoint_dim_raw['Stock_Point_ID'] = df_stockpoint_dim_raw['Stock_Point_ID'].astype(int)
    df_customer_score = df_customer_score.rename(columns={'StockPointID':'Stock_Point_ID'})
    df_customer_score['Stock_Point_ID'] = df_customer_score['Stock_Point_ID'].astype(int)


    # ----------------- CUSTOMER DIM TABLE 
    col_sel_affinity = ['Region', 'Stock_Point_ID', 'CustomerID']

    col_sel_kyc = ['CustomerID', 'ContactName', 'BusinessName', 'CustomerModeName',
        'CustomerRef', 'ContactPhone', 'CustomerType', 'FullAddress', 
        'StateName', 'CityName', 'TownName', 'Latitude','Longitude', 
        'DistanceVarianceInMeter', 'IsLocationSubmitted',
        'IsLocationCaptured', 'IsLocationVerified','CustomerStatus',
        'RejectReason',  'KYC_Capture_Status',  'lastDelvDate', 
        # 'hasPOS','hasVAS', 'hasBNPL', 'lastDelvDate', 
        'isActive']

    col_sel_score = ['Stock_Point_ID', 'CustomerID', 'composite_customer_score',
        'percentile_rank', 'active_months_pct', 'avg_orders_per_active_month',
        'avg_qty_per_month', 'avg_revenue_per_month', 'days_since_last_order']

    df_master_customer_dim = (
                df_customer_dim_with_affinity_score_raw[col_sel_affinity]
                .merge(df_kyc_customer[col_sel_kyc], how='inner', on=['CustomerID'])
                .merge(df_customer_score[col_sel_score], how='left', on=['Stock_Point_ID', 'CustomerID'])
                .rename(columns = {'CityName':'LGA',
                                'TownName':'LCDA'
                                })

            )

    # Change CustomerPurchaseRecency from lastDelvDate to days since last order (order creation date)
    df_master_customer_dim['CustomerPurchaseRecency'] =  df_master_customer_dim['days_since_last_order']
    # df_master_customer_dim['CustomerPurchaseRecency'] =  df_master_customer_dim['lastDelvDate'].apply(lambda x: (datetime.now() - x).days)
    df_master_customer_dim['CustomerPurchaseRecency'] = df_master_customer_dim['CustomerPurchaseRecency'].fillna(max(df_master_customer_dim['CustomerPurchaseRecency']))
    df_master_customer_dim['KYC_Capture_Status'] = df_master_customer_dim['KYC_Capture_Status'].apply(lambda x: 'Yes' if x == 1 else 'No')

    # Add to Score
    # Fix Missing value -------------------------------------------
    for col in ['BusinessName', 'CustomerModeName', 'FullAddress', 'LGA', 'LCDA']:
        df_master_customer_dim[col] = df_master_customer_dim[col].fillna('')

    for col in ['Latitude',  'Longitude', 'composite_customer_score',  
                'percentile_rank',  'active_months_pct', 'avg_orders_per_active_month',  
                'avg_qty_per_month',  'avg_revenue_per_month'
                ]:
        df_master_customer_dim[col] = pd.to_numeric(df_master_customer_dim[col], errors='coerce').fillna(0) 

    df_master_customer_dim = clean_invalid_coordinates(df_master_customer_dim, bound_to_nigeria=True, offset_degrees=0.1) 
    
    # Add to Score 
    # Boost composite score and percentile rank for customers with completed KYC
    mask_kyc = df_master_customer_dim['KYC_Capture_Status'] == 'Yes'

    df_master_customer_dim.loc[mask_kyc, 'composite_customer_score'] += 5
    df_master_customer_dim.loc[mask_kyc, 'percentile_rank'] += 0.1 

    # ----------------- RECOMMENDATION
    col2 = ['EstimatedQuantity', 'CustomerSKUscore', 'CustomerSKUscoreStandardize', 'CustomerSKUscoreRank']
    for col in col2: 
        df_customer_sku_recommendation_raw[col] = pd.to_numeric(df_customer_sku_recommendation_raw[col], errors='coerce')

    df_customer_sku_recommendation_raw['LastDeliveredDate'] = pd.to_datetime(df_customer_sku_recommendation_raw['LastDeliveredDate'])
    # Get today's date
    today = pd.Timestamp.today()

    df_customer_sku_recommendation_raw['Recency'] = df_customer_sku_recommendation_raw['LastDeliveredDate'].apply(lambda x: (datetime.now() - x).days)
    df_customer_sku_recommendation_raw['Recency'] = df_customer_sku_recommendation_raw['Recency'].fillna(max(df_customer_sku_recommendation_raw['Recency']))
    
    # ----------------- STOCKPOINT
    df_stockpoint_dim_raw.rename(columns={'lattitude':'Latitude', 'longitude':'Longitude'}, inplace=True) 
    col3 = ['Latitude', 'Longitude']
    for col in col3: 
        df_stockpoint_dim_raw[col] = pd.to_numeric(df_stockpoint_dim_raw[col], errors='coerce').fillna(0)    

    # Replace invalid latitude values with NaN
    df_stockpoint_dim_raw = clean_invalid_coordinates(df_stockpoint_dim_raw, bound_to_nigeria=True, offset_degrees=0.1)   
    

    return df_customer_sku_recommendation_raw, df_master_customer_dim, df_stockpoint_dim_raw