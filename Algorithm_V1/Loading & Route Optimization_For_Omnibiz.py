import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import great_circle, geodesic
from sklearn.cluster import DBSCAN, KMeans
import requests
from datetime import datetime as dt
from datetime import date
from prefect import task, flow
import logging
import pyodbc
import os
from dotenv import load_dotenv
import pymssql
import time
from concurrent.futures import ThreadPoolExecutor
import string
import logging



#Create logs directory
os.makedirs('C:/Users/ME/OneDrive - Mplify Limited/Omnibiz Africa/Projects/GIT/Personal/Van-Route Optimization/Omnibiz', exist_ok=True)



#Configure logging to save to a file
logging.basicConfig(
    filename='C:/Users/ME/OneDrive - Mplify Limited/Omnibiz Africa/Projects/GIT/Personal/Van-Route Optimization/Omnibiz/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger('prefect').setLevel(logging.INFO)


current_date = date.today()


#StockPoint LatLong
@task
def fetch_stockpoint_data():
    warehouse_df = pd.DataFrame({
        "StockPointName" : "Ijora-Causeway MFC",
        "StockPointId" : 1647113,
        "StockPoint_Latitude" : 6.46965,
        "StockPoint_Longitude" : 3.36888
    }, index = [0])

    return warehouse_df


#Generate 10 unique 6-character alphanumeric VehicleNumbers
@task
def generate_alphanumeric_vehicle_number(size=6):
    chars = string.ascii_uppercase + string.digits  # A-Z, 0-9
    return ''.join(np.random.choice(list(chars), size=size))


#available vehicles
@task
def fetch_vehicle_data():
    vehicle_numbers = [generate_alphanumeric_vehicle_number() for _ in range(10)]
    # Ensure uniqueness
    while len(set(vehicle_numbers)) < 10:
        vehicle_numbers = [generate_alphanumeric_vehicle_number() for _ in range(10)]

    vehicle_df = pd.DataFrame({
        'VehicleNumber': vehicle_numbers,
        'VehicleCapacity': np.random.randint(1500, 1701, size=10)
    })

    vehicle_df["Total_Loaded_Quantity"] = 0

    return vehicle_df


#Orders_Ready_For_Dispatch
@task
def fetch_order_data():
    try:
        conn = pymssql.connect(server = f"{os.getenv('DB_HOST')}", 
                            user = f"{os.getenv('DB_NAME')}", 
                            password = f"{os.getenv('DB_PASSWORD')}", 
                            database = f"VconnectMasterDWR")

        cursor = conn.cursor()
        query = """
                drop table if exists #tempone
                select CustomerId, BusinessId, ItemId, Quantity, OrderId, cast(CreatedDate as Date) as CreatedDate
                into #tempone
                from tblordersales
                where businessid = 1647113 and orderstatusID = 212

                ---------------------------------
                IF OBJECT_ID('tempdb..#CustomerLatLong') IS NOT NULL DROP TABLE #CustomerLatLong; 
                SELECT 
                b.CustomerID, b.Latitude, b.Longitude
                INTO #CustomerLatLong
                FROM 
                (
                SELECT 
                a.Userid AS CustomerID,
                a.*,  
                ROW_NUMBER() OVER (PARTITION BY a.Userid ORDER BY a.ContentID DESC) AS rn
                FROM BusinessAddressBook a WITH (NOLOCK) 
                WHERE 
                a.businessid = 76 
                AND ISNULL(a.Status, 0) IN (0, 1, 2, 7) 
                ) b
                WHERE b.rn = 1; 

                ---------------------------------
                drop table if exists #finaldata
                select a.*, b.Latitude, b.Longitude
                into #finaldata
                from #tempone a inner join #CustomerLatLong b on a.customerid = b.CustomerID

                ---------------------------------
                select * from #finaldata
                order by createddate
            """
        cursor.execute(query)
        rows = cursor.fetchall()

        result = pd.DataFrame(data = rows, header = [entry[0] for entry in cursor.description])

        conn.close()

    except:
        extracted_orderdata = pd.read_csv("C:/Users/ME/OneDrive - Mplify Limited/Omnibiz Africa/Projects/GIT/Personal/Van-Route Optimization/Omnibiz/Pending_Order_Data.csv")
        result = pd.DataFrame(extracted_orderdata.dropna())
        result["CreatedDate"] = pd.to_datetime(result["CreatedDate"])

    result.rename(columns={"Latitude" : "Customer_Latitude", "Longitude" : "Customer_Longitude"}, inplace = True)
    
    result["CustomerId"] = result["CustomerId"].astype("int")
    result["BusinessId"] = result["BusinessId"].astype("int")
    result["ItemId"] = result["ItemId"].astype("int")
    result["Quantity"] = result["Quantity"].astype("int")
    result["OrderId"] = result["OrderId"].astype("int")

    # Filter rows where Latitude and Longitude are valid floats
    result = pd.DataFrame(result[result[['Customer_Latitude', 'Customer_Longitude']].apply(lambda x: pd.to_numeric(x, errors='coerce').notna().all(), axis=1)])

    result["Customer_Latitude"] = result["Customer_Latitude"].astype(float)
    result["Customer_Longitude"] = result["Customer_Longitude"].astype(float)

    return result
    

#Current_Stock_Availability
@task
def fetch_current_stock_data():
    try:
        conn = pymssql.connect(server = f"{os.getenv('DB_HOST')}", 
                            user = f"{os.getenv('DB_NAME')}", 
                            password = f"{os.getenv('DB_PASSWORD')}", 
                            database = f"VconnectMasterDWR")

        cursor = conn.cursor()
        query = """
                IF OBJECT_ID('Tempdb..#currentstock') IS NOT NULL DROP TABLE #currentstock 
                select *
                into #currentstock
                from (
                SELECT DISTINCT o.BusinessId, o.ItemId,	Quantity,BookedStock,(ISNULL(quantity,0)-ISNULL(BookedStock,0)) AS QtyAvailable,
                row_number() over (partition by o.businessid, o.itemid order by contentid desc) as RN
                FROM vconnectmasterdwr.dbo.OrionStock o with (nolock)     
                WHERE o.businessid = 1647113  
                AND status=1   
                )xyz
                where rn = 1
                and QtyAvailable > 0

                ----------------------------------------------
                select BusinessId, ItemId,Quantity, BookedStock, QtyAvailable from #currentstock

            """
        cursor.execute(query)
        rows = cursor.fetchall()

        result = pd.DataFrame(data = rows, header = [entry[0] for entry in cursor.description])

        conn.close()

    except:
        extracted_orderdata = pd.read_csv("C:/Users/ME/OneDrive - Mplify Limited/Omnibiz Africa/Projects/GIT/Personal/Van-Route Optimization/Omnibiz/Current_Stock_Data.csv")
        result = pd.DataFrame(extracted_orderdata.dropna())

    return result


@task
def calculate_pathway_distance(row, another_parameter1 = None, another_parameter2 = None):
    if another_parameter1 == None:
        #Specify coordinates for two locations
        start_coords = (row['StockPoint_Latitude'], row['StockPoint_Longitude'])
        end_coords = (row['Customer_Latitude'], row['Customer_Longitude'])

        #Format coordinates for OSRM API
        url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=false"

        #Send the request to the OSRM API
        response = requests.get(url)
        data = response.json()

        #Extract distance in meters
        try:
            distance_meters = data['routes'][0]['distance']
        except:
            return np.nan
        distance_km = distance_meters / 1000  # Convert to kilometers
        # print(f"Distance: {distance_km} km")

        return distance_km
    
    else:
        #Specify coordinates for two locations
        start_coords = row
        end_coords = (another_parameter1, another_parameter2)

        #Format coordinates for OSRM API
        url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=false"

        # Send the request to the OSRM API
        response = requests.get(url)
        data = response.json()

        #Extract distance in meters
        try:
            distance_meters = data['routes'][0]['distance']
        except:
            return np.nan
        distance_km = distance_meters / 1000  # Convert to kilometers
        #print(f"Distance: {distance_km} km")

        return distance_km


@task
def calculate_batch_distances(current_location, customers, batch_size=20):
    base_url = "http://router.project-osrm.org/route/v1/driving/"
    distances = pd.Series(np.nan, index=customers.index)
    
    #Prepare source and destinations
    source = f"{current_location[1]},{current_location[0]}"  #lon,lat
    destinations = customers[['Customer_Longitude', 'Customer_Latitude']].values.tolist()
    
    def fetch_distance(batch_dests, batch_indices):
        #Format coordinates for OSRM multi-point route
        coords = [source] + [f"{lon},{lat}" for lon, lat in batch_dests]
        url = base_url + ";".join(coords) + "?overview=false"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            #Extract leg distances (between consecutive points)
            leg_distances = [leg['distance'] / 1000 for leg in data['routes'][0]['legs']]  #km
            return batch_indices, leg_distances
        except:
            return batch_indices, [np.nan] * len(batch_dests)
    
    #Process in batches with parallel requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(0, len(destinations), batch_size):
            batch_dests = destinations[i:i + batch_size]
            batch_indices = customers.index[i:i + batch_size]
            futures.append(executor.submit(fetch_distance, batch_dests, batch_indices))
            time.sleep(0.1)  #Avoid overwhelming public API
        
        for future in futures:
            batch_indices, batch_distances = future.result()
            distances.loc[batch_indices] = batch_distances
    
    return distances


@task
def customer_order_clubbing_check(order_data, van_data):
    # Clubbing all customer's orderids, so they can be recorded in the OrderId column
    clubbed_orders_cust_and_orderid_df = pd.DataFrame(order_data[["CustomerId", "OrderId"]]).sort_values(by = ["CustomerId", "OrderId"], ascending = True)
    clubbed_orders_cust_and_orderid_df["OrderId"] = clubbed_orders_cust_and_orderid_df["OrderId"].astype(str)
    resulting_clubbed_orders = pd.DataFrame(clubbed_orders_cust_and_orderid_df.groupby("CustomerId")["OrderId"].apply(lambda a : '-'.join(a)).reset_index())
    resulting_clubbed_orders["InvoiceNo"] = resulting_clubbed_orders['CustomerId'].astype(str) + '-' + resulting_clubbed_orders['OrderId'].astype(str)
    resulting_clubbed_orders_dict = dict(zip(resulting_clubbed_orders["CustomerId"], resulting_clubbed_orders["OrderId"]))
    resulting_clubbed_invoice_dict = dict(zip(resulting_clubbed_orders["CustomerId"], resulting_clubbed_orders["InvoiceNo"]))

    order_data["InvoiceNo"] = order_data['CustomerId'].astype(str) + '-' + order_data['OrderId'].astype(str)
    order_data['Merged_Products_And_Quantity'] = order_data['ItemId'].astype(str) + ':' + order_data['Fulfillment_Quantity'].astype(str)
    order_data["ItemId"] = order_data["ItemId"].astype(str)

    max_VehicleCapacity = van_data["VehicleCapacity"].max()

    grouped_df = pd.DataFrame(order_data.groupby('CustomerId').agg({
    'Quantity': 'sum',
    'OrderId': 'nunique',  #Count the number of orders per customer
    'InvoiceNo' : 'first'
    }).reset_index())


    #Rename the 'OrderID' column to 'OrderCount'
    grouped_df.rename(columns={'OrderId': 'OrderCount'}, inplace=True)

    #Split into large and small orders
    large_orders = grouped_df[grouped_df['Quantity'] > max_VehicleCapacity]
    small_orders = grouped_df[grouped_df['Quantity'] <= max_VehicleCapacity]

    #Prepare DataFrames for small orders
    small_orders_df = pd.DataFrame(order_data[order_data['CustomerId'].isin(small_orders['CustomerId'])].copy())
    small_orders_df['Customer_Order_Type'] = 'within van limit'

    #Group by CustomerId for small orders to avoid duplicates
    small_orders_grouped = small_orders_df.groupby('CustomerId').agg(
        Quantity=('Quantity', 'sum'),
        CreatedDate=('CreatedDate', 'first'),
        BusinessId=('BusinessId', 'first'),
        ItemId=('ItemId', ','.join),
        Fulfillment_Status=('Fulfillment_Status', 'first'),
        Fulfillment_Quantity=('Fulfillment_Quantity', 'sum'),
        Initial_Quantity=('Initial_Quantity', 'sum'),
        Customer_Latitude=('Customer_Latitude', 'first'),
        Customer_Longitude=('Customer_Longitude', 'first'),
        StockPointName=('StockPointName', 'first'),
        StockPointId=('StockPointId', 'first'),
        StockPoint_Latitude=('StockPoint_Latitude', 'first'),
        StockPoint_Longitude=('StockPoint_Longitude', 'first'),
        InvoiceNo=('InvoiceNo', 'first'),
        OrderId = ('OrderId', 'first'),
        Distance_From_StockPoint= ('Distance_From_StockPoint', 'first'),
        Customer_Order_Type = ('Customer_Order_Type', 'first'),
        Merged_Products_And_Quantity = ('Merged_Products_And_Quantity', ','.join)
    ).reset_index()
 

    #Merge OrderCount back into the small orders grouped DataFrame
    small_orders_grouped = pd.DataFrame(small_orders_grouped.merge(grouped_df[['CustomerId', 'OrderCount']], on='CustomerId', how='left'))

    #Add the Has_Multiple_Orders column
    small_orders_grouped['Has_Multiple_Orders'] = small_orders_grouped['OrderCount'] > 1

    small_orders_grouped["OrderId"] = small_orders_grouped["CustomerId"].map(resulting_clubbed_orders_dict)
    small_orders_grouped["InvoiceNo"] = small_orders_grouped["CustomerId"].map(resulting_clubbed_invoice_dict)

    #Prepare DataFrames for large orders and exclude small orders
    large_orders_df = pd.DataFrame(order_data[order_data['CustomerId'].isin(large_orders['CustomerId'])].copy())
    large_orders_df = pd.DataFrame(large_orders_df[~large_orders_df['CustomerId'].isin(small_orders_grouped['CustomerId'])])  # Exclude customers in small orders
    large_orders_df['Customer_Order_Type'] = 'exceeding van limit'

    #Combine both DataFrames
    result_df = pd.DataFrame(pd.concat([large_orders_df, small_orders_grouped], ignore_index=True))

    #To check if there are single orders that are beyond the maximum van capacity, so it can be splitted.
    def big_order_splitting(orderdata_df):
        def split_row(row):
            """Splits a single row into two based on total_quantity and associated columns."""
            
            # Custom splitting logic based on rounding
            def split_number(number):
                if number % 2 == 0:
                    # Even number: split evenly
                    return number // 2, number // 2
                else:
                    # Odd number: split with rounding
                    half1 = number // 2
                    half2 = half1 + 1
                    return half1, half2

            # Split volume, quantity, and weight using custom split
            quantity1, quantity2 = split_number(row["Quantity"])

            # Split merged products and quantities
            merged_items = row["Merged_Products_And_Quantity"].split(',')
            items1, items2 = [], []
            for item in merged_items:
                if ':' not in item:
                    print(f"Skipping invalid item: {item}")
                    continue  # Skip invalid formats
                product, qty = item.split(':')
                try:
                    qty = int(float(qty))  # Ensure quantity is numeric
                    qty1, qty2 = split_number(qty)
                    items1.append(f"{product}:{qty1}")
                    items2.append(f"{product}:{qty2}")
                except ValueError:
                    print(f"Invalid quantity for product {product}: {qty}")
                    continue  # Skip non-numeric quantities

            # Create two new rows
            row1, row2 = row.copy(), row.copy()
            row1.update({
                "total_quantity": quantity1,
                "Merged_Products_And_Quantity": ','.join(items1),
                "InvoiceNo": f"{row['InvoiceNo']}_split1",
                "OrderId": f"{row['OrderId']}_split1"
            })
            row2.update({
                "total_quantity": quantity2,
                "Merged_Products_And_Quantity": ','.join(items2),
                "InvoiceNo": f"{row['InvoiceNo']}_split2",
                "OrderId": f"{row['OrderId']}_split2"
            })
            return row1, row2

        # Process rows that exceed max van capacity
        while (orderdata_df["Quantity"] > max_VehicleCapacity).any():
            # Separate large orders and those within capacity
            large_orders = pd.DataFrame(orderdata_df[orderdata_df["Quantity"] > max_VehicleCapacity])
            small_orders = pd.DataFrame(orderdata_df[orderdata_df["Quantity"] <= max_VehicleCapacity])
            
            split_entries = []

            for _, row in large_orders.iterrows():
                row1, row2 = split_row(row)
                split_entries.extend([row1, row2])

            # Combine small orders and split entries
            split_df = pd.DataFrame(split_entries)
            orderdata_df = pd.DataFrame(pd.concat([small_orders, split_df], ignore_index=True))

        # Return the DataFrame, which will be split properly
        return orderdata_df

    result_df = big_order_splitting(result_df)

    return result_df


@task
def stock_check(stock_csv, melted_orders_df):

    stock_df = pd.DataFrame(stock_csv)
    # stock_df.dropna(inplace = True)

    stock_df.drop_duplicates(keep = "first", ignore_index = True, inplace = True)

    #Giving preference to earliest dates
    melted_orders_df.sort_values(by = "CreatedDate", ascending = True)

    #computing the stock checks, iterating through and keeping track of the qty
    fulfillment_tracker = []
    fulfilled_quantity = []

    for idx, row in melted_orders_df.iterrows():
        selected_product = row[2]
        selected_product_quantity = row[3]

        #Filter the grouped stock for the selected product
        selected_stock = stock_df[stock_df["ItemId"] == selected_product]
        filtered_selected_grouped_stock = selected_stock[selected_stock["QtyAvailable"] > 0]
       

        if not filtered_selected_grouped_stock.empty:
            current_stock_quantity = filtered_selected_grouped_stock["QtyAvailable"].sum() 

            if selected_product_quantity > current_stock_quantity:
                fulfillment_tracker.append("Partial Fulfillment")
                fulfilled_quantity.append(current_stock_quantity)

                #Deduct total available quantity
                stock_df.loc[stock_df["ItemId"] == selected_product, "QtyAvailable"] -= current_stock_quantity
            else:
                fulfillment_tracker.append("Complete Fulfillment")
                fulfilled_quantity.append(selected_product_quantity)
                stock_df.loc[stock_df["ItemId"] == selected_product, "QtyAvailable"] -= selected_product_quantity
        else:
            fulfillment_tracker.append("No Fulfillment")
            fulfilled_quantity.append(0)

    melted_orders_df["Fulfillment_Status"] = fulfillment_tracker
    melted_orders_df["Fulfillment_Quantity"] = fulfilled_quantity
    melted_orders_df["Initial_Quantity"] = melted_orders_df["Quantity"]


    #saving orders to a file, that either had partial fulfillment or no fulfillment
    failed_stock_check_df = pd.DataFrame(melted_orders_df[melted_orders_df["Fulfillment_Status"] != "Complete Fulfillment"])

    failed_stock_csv = "C:/Users/ME/OneDrive - Mplify Limited/Omnibiz Africa/Projects/GIT/Personal/Van-Route Optimization/Omnibiz/Failed Stock Check.csv"
    failed_stock_check_df.to_csv(failed_stock_csv, index = False)

    #overriding the initially placed quantity, with the output after the stock check was computed
    melted_orders_df["Quantity"] = melted_orders_df["Fulfillment_Quantity"]

    
    #keeping a version of the unfiltered melted orders df
    unfiltered_melted_orders_df = pd.DataFrame(melted_orders_df.copy())

    unfiltered_melted_orders_df.to_csv(f"C:/Users/ME/OneDrive - Mplify Limited/Omnibiz Africa/Projects/GIT/Personal/Van-Route Optimization/Omnibiz/Unfiltered Melted Orders.csv", index = False)

    #deciding to proceed with orders that had either complete or partial fulfillment
    melted_orders_df = pd.DataFrame(melted_orders_df[melted_orders_df["Fulfillment_Status"] != "No Fulfillment"])


    return melted_orders_df


@task
def optimal_kmeans_clusters(coords, max_k=10): #max_k was formerly 7
    # Compute inertia (sum of squared distances) for a range of k values
    distortions = []

    if max_k > len(coords):
        max_k = len(coords)

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
        
        distortions.append(kmeans.inertia_)
    
    # Automatically detect the "elbow" (where inertia starts decreasing slower)
    # Compute the second derivative to find the elbow point
    deltas = np.diff(distortions)
    deltas2 = np.diff(deltas)  # 2nd derivative
    # optimal_k = np.argmin(deltas2) + 2  # Add 2 because we applied diff() twice

    #This simply gives the k number that returned the least inertia
    optimal_k = distortions.index(min(distortions)) + 1 #Added 1 because python numbering/indexing starts from zero

    return optimal_k


@task
def get_customer_cluster(final_order_df, eps_km=2, min_samples=5, max_k=10):   #max_k formerly 7
    kms_per_radian = 6371.0088  
    epsilon = eps_km / kms_per_radian 
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')

    customer_coords = np.radians(final_order_df[['Customer_Latitude', 'Customer_Longitude']])

    customer_coords['Customer_Latitude'] = customer_coords['Customer_Latitude'].astype(str)
    customer_coords = customer_coords[customer_coords['Customer_Latitude'].str.lower() != 'nan']

    customer_coords['Customer_Latitude'] = customer_coords['Customer_Latitude'].astype(float)

    #Temp Resolution for final_order_df
    final_order_df['Customer_Latitude'] = final_order_df['Customer_Latitude'].astype(str)
    final_order_df = pd.DataFrame(final_order_df[final_order_df['Customer_Latitude'].str.lower() != 'nan'])
    final_order_df['Customer_Latitude'] = final_order_df['Customer_Latitude'].astype(float)

    #cluster assignment
    final_order_df['cluster'] = db.fit_predict(customer_coords)

    final_order_df['is_noise'] = (final_order_df['cluster'] == -1).astype(int)
    noise_points = pd.DataFrame(final_order_df[final_order_df['is_noise'] == 1])
    clustered_points =pd.DataFrame(final_order_df[final_order_df['is_noise'] == 0])

    #Using 50th percentile to consider where most of the points are within (neglecting 50%)
    max_noise_distance_benchmark = np.quantile(noise_points["Distance_From_StockPoint"].unique(), 0.50) #noise_points["Distance_From_StockPoint"].unique().quantile(0.70)  

    #Using 500m to give a sense of how clubbed we want the clusters to be
    proposed_number_of_clusters = round(max_noise_distance_benchmark / 2)
    if proposed_number_of_clusters < 1:  #incase the division cannot be rounded up to 1
        proposed_number_of_clusters = 1

    
    if not noise_points.empty: #and not clustered_points.empty:
        noise_coords = np.radians(noise_points[['Customer_Latitude', 'Customer_Longitude']].values)
        clustered_coords = np.radians(clustered_points[['Customer_Latitude', 'Customer_Longitude']].values)
        optimal_clusters = optimal_kmeans_clusters(noise_coords, max_k=proposed_number_of_clusters)   #formerly clustered_coords
        
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        noise_clusters = kmeans.fit_predict(noise_coords)

        # final_order_df.loc[noise_points.index, 'cluster'] = noise_clusters + final_order_df['cluster'].max() + 1
        final_order_df['cluster'].loc[noise_points.index] = noise_clusters + final_order_df['cluster'].max() + 1

    
    return final_order_df


@task
def calculate_cluster_quantity(final_order_df):
    cluster_info_df = final_order_df.groupby('cluster').agg({
        'Quantity': 'sum',
        'Customer_Latitude': 'mean',
        'Customer_Longitude': 'mean',
        'StockPoint_Latitude': 'first',
        'StockPoint_Longitude': 'first'
    }).reset_index()
    
    cluster_info_df['Distance_From_StockPoint'] = cluster_info_df.apply(calculate_pathway_distance, axis=1)
    
    return cluster_info_df


@task
def aggregate_metrics(customer_van_df, column_name, type, agg='sum', flag=1, percent=0):
    ops_type = {'Quantity': 'Total_Loaded_Quantity', 'Van_Max_Distance': 'Van_Max_Distance',
                'Van_Min_Distance': 'Van_Min_Distance'}
    

    #For cases where the customer_van_df isn't a dataframe due to vans not being available
    try:
        customer_van_df_colums = customer_van_df.columns
    except:
        customer_van_df = pd.DataFrame({
            'CustomerId' : [np.nan],
            'InvoiceNo' : [np.nan],
            'Quantity' : [0],
            'VehicleNumber' : [np.nan],
            'VehicleCapacity' : [0],
            'Total_Loaded_Quantity' : [0],
            'Percent_Full' : [0]
        })

        return customer_van_df
    

    if ops_type.get(type) not in customer_van_df_colums:
        customer_van_df[ops_type.get(type)] = 0  
    
    if flag == 1:
        van_usage = customer_van_df.groupby(['VehicleNumber', 'Ride_Batch']).agg({column_name: agg}).reset_index()
        van_usage.rename(columns={column_name: ops_type.get(type)}, inplace=True)
        customer_van_df = customer_van_df.merge(van_usage, on=['VehicleNumber', 'Ride_Batch'], how='left', suffixes=('', '_new'))
        customer_van_df[ops_type.get(type)] = customer_van_df[f"{ops_type.get(type)}_new"].fillna(0)
        customer_van_df.drop(f"{ops_type.get(type)}_new", axis=1, inplace=True)      

    elif flag == 0:
        van_usage = customer_van_df.groupby(['VehicleNumber']).agg({column_name: agg}).reset_index()
        van_usage.rename(columns={column_name: ops_type.get(type)}, inplace=True)
        customer_van_df = customer_van_df.merge(van_usage, on=['VehicleNumber'], how='left', suffixes=('', '_new'))
        customer_van_df[ops_type.get(type)] = customer_van_df[f"{ops_type.get(type)}_new"].fillna(0)
        customer_van_df.drop(f"{ops_type.get(type)}_new", axis=1, inplace=True)
    
    if percent == 1:
        customer_van_df['Percent_Full'] = (customer_van_df['Total_Loaded_Quantity'] / customer_van_df['VehicleCapacity']) * 100


    return customer_van_df
    
    
@task
def select_vans_for_cluster(final_order_df, available_vans, max_distance_km=150, ride_batch_check=None, cluster_checker1=1, sub_cluster_checker1="None"):
    # Warehouse information setup

    final_order_df = final_order_df.sort_values(by=['Quantity'], ascending=False)

    StockPoint_Latitude = final_order_df['StockPoint_Latitude'].iloc[0]
    StockPoint_Longitude = final_order_df['StockPoint_Latitude'].iloc[0]
    
    max_customer = 10    #this is configurable

    # # Initialize customer van mapping
    # customer_van_mapping = pd.DataFrame(columns=['CustomerId', 'InvoiceNo', 'invoice_volume', 'VehicleNumber', 'VehicleCapacity'])

    def assign_vans(orders, available_vans, max_customer, ride_batch_check, cluster_checker1=1, sub_cluster_checker1="None"):
        # # Initialize customer van mapping
        customer_van_mapping = pd.DataFrame(columns=['CustomerId', 'InvoiceNo', 'Quantity', 'VehicleNumber', 'VehicleCapacity', 'Customer_Latitude', 'Customer_Longitude'])

        available_vans.reset_index(drop=True, inplace=True)
        if available_vans.empty:
            for _, order in orders.iterrows():
                if order['InvoiceNo'] not in customer_van_mapping['InvoiceNo'].values:
                    new_row = pd.DataFrame([{
                    'CustomerId': order['CustomerId'],
                    'InvoiceNo': order['InvoiceNo'],
                    'Quantity': order['Quantity'],
                    'VehicleNumber': 'no van available',
                    'VehicleCapacity': 1,
                    'Customer_Latitude': order['Customer_Latitude'],
                    'Customer_Longitude': order['Customer_Longitude']
                    }])

                    customer_van_mapping = pd.concat([customer_van_mapping, new_row], ignore_index=True)

            customer_van_mapping['VehicleNumber'] = customer_van_mapping['VehicleNumber'].fillna('no van available').replace('', 'no van available')
            return customer_van_mapping              

        
        def load_orders_to_van(order_list, available_vans, category_of_customers = None):
            #Starting with smaller vans to maximize capacity usage
            available_vans.sort_values(by="VehicleCapacity", ascending=True, inplace = True)    
            available_vans.reset_index(inplace=True, drop=True)
            #Initialize customer van mapping
            customer_van_mapping = pd.DataFrame(columns=['CustomerId', 'InvoiceNo', 'Quantity', 'VehicleNumber', 'VehicleCapacity', 'Customer_Latitude', 'Customer_Longitude'])

            current_van_index = 0
            skipped_orders = pd.DataFrame(order_list.copy())

            while current_van_index < len(available_vans) and not skipped_orders.empty:
                current_VehicleCapacity = available_vans.iloc[current_van_index]['VehicleCapacity'].astype(float)
                current_VehicleNumber = available_vans.iloc[current_van_index]['VehicleNumber']
                current_van_volume_used = available_vans.iloc[current_van_index]['Total_Loaded_Quantity']
                loaded_customers = 0
                loaded_CustomerIds = set()
                
                remaining_orders = pd.DataFrame(skipped_orders.copy())
                skipped_orders = pd.DataFrame(columns=order_list.columns)

                total_loaded_quantity = 0
                distance_to_customer_list = []
                
                #Load customers into van without interruptions
                for idx, order in remaining_orders.iterrows():
                    CustomerId = order['CustomerId']
                    quantity = order['Quantity']
                    #Extract customer latitude and longitude
                    customer_latitude = order['Customer_Latitude']
                    customer_longitude = order['Customer_Longitude']

                    if (current_van_volume_used + quantity > current_VehicleCapacity) or (loaded_customers >= max_customer):
                        order_df = pd.DataFrame([order])

                        for col in order_df.select_dtypes(include=['object']):
                            if order_df[col].dropna().isin([True, False]).all():
                                order_df[col] = order_df[col].astype(bool)

                        skipped_orders = pd.concat([skipped_orders, order_df], ignore_index=True)

                        # skipped_orders = pd.concat([skipped_orders, pd.DataFrame([order])], ignore_index=True)
                        continue

                    current_van_volume_used += quantity

                    if CustomerId not in loaded_CustomerIds:
                        loaded_customers += 1
                        loaded_CustomerIds.add(CustomerId)


                    new_row = pd.DataFrame([{
                    'CustomerId': order['CustomerId'],
                    'InvoiceNo': order['InvoiceNo'],
                    'Quantity': order['Quantity'],
                    'VehicleNumber': current_VehicleNumber,
                    'VehicleCapacity': current_VehicleCapacity,
                    'Customer_Latitude': customer_latitude,
                    'Customer_Longitude': customer_longitude
                    }])

                    customer_van_mapping = pd.concat([customer_van_mapping, new_row], ignore_index=True)


                    distance_to_customer = calculate_pathway_distance((StockPoint_Latitude, StockPoint_Longitude), another_parameter1=customer_latitude, another_parameter2 = customer_longitude)

                    distance_to_customer_list.append(distance_to_customer)
                    total_loaded_quantity += order['Quantity']                
                
            
                # Update the van's loaded volume
                available_vans.at[current_van_index, 'Total_Loaded_Quantity'] = current_van_volume_used
                current_van_index += 1

            return customer_van_mapping


        # Process orders
        customer_van_mapping = load_orders_to_van(orders, available_vans)
        used_vans = customer_van_mapping['VehicleNumber'].unique()
        available_vans = available_vans[~available_vans['VehicleNumber'].isin(used_vans)]

    
        for _, order in orders.iterrows():
            if order['InvoiceNo'] not in customer_van_mapping['InvoiceNo'].values:

                new_row = pd.DataFrame([{
                    'CustomerId': order['CustomerId'],
                    'InvoiceNo': order['InvoiceNo'],
                    'Quantity': order['Quantity'],
                    'VehicleNumber': 'no van available',
                    'VehicleCapacity': 1,
                    'Customer_Latitude': order['Customer_Latitude'],
                    'Customer_Longitude': order['Customer_Longitude']
                    }])

                customer_van_mapping = pd.concat([customer_van_mapping, new_row], ignore_index=True)

        customer_van_mapping['VehicleNumber'] = customer_van_mapping['VehicleNumber'].fillna('no van available').replace('', 'no van available')    
        
        return customer_van_mapping
    
    

    # Run the van assignment
    customer_van_mapping = assign_vans(final_order_df, available_vans, max_customer, ride_batch_check)

    customer_van_mapping.drop(labels = ['Customer_Latitude', 'Customer_Longitude'], axis = 1, inplace = True)

    customer_van_mapping = aggregate_metrics(customer_van_mapping, column_name='Quantity', type='Quantity', agg='sum', flag=0, percent=1)


    return customer_van_mapping


@task
def process_ride_batch(cluster_info_df, final_order_df, van_df, ride_batch):
    available_vans = pd.DataFrame(van_df.copy())
    cluster_info_df = cluster_info_df.sort_values(by='Distance_From_StockPoint', ascending=True)
    customer_van_main = pd.DataFrame(columns=['CustomerId', 'InvoiceNo', 'Quantity', 'VehicleNumber', 'VehicleCapacity', 'Total_Loaded_Quantity', 
                                              'Percent_Full', 'Ride_Batch'])
    

    for _, cluster in cluster_info_df.iterrows():
        get_customer_cluster = pd.DataFrame(final_order_df[final_order_df['cluster'] == cluster['cluster']])

        customer_van_df = select_vans_for_cluster(get_customer_cluster, available_vans, ride_batch_check = ride_batch, cluster_checker1 = int(cluster[0]))

        customer_van_df['Ride_Batch'] = ride_batch    

        customer_van_main = pd.concat([customer_van_main, customer_van_df], ignore_index=True)

        customer_van_main = aggregate_metrics(customer_van_main, column_name='Quantity', type='Quantity', agg='sum', flag=0, percent=1)
        used_vans = customer_van_df['VehicleNumber'].unique()
        available_vans = available_vans[~available_vans['VehicleNumber'].isin(used_vans)]


    #choosing to use final_order_df's quantity as opposed to the quantity from customer_van_main. So I'd drop the quantity from the former.

    final_order_df.drop(columns = ["Quantity"], inplace = True)

    final_order_df = final_order_df.merge(customer_van_main, on=['InvoiceNo', 'CustomerId'], how='left')

    

    final_order_df = aggregate_metrics(final_order_df, column_name='Quantity', type='Quantity', agg='sum', flag=1, percent=0)
    final_order_df = aggregate_metrics(final_order_df, column_name='Distance_From_StockPoint', type='Van_Max_Distance', agg='max', flag=1, percent=0)
    final_order_df = aggregate_metrics(final_order_df, column_name='Distance_From_StockPoint', type='Van_Min_Distance', agg='min', flag=1, percent=0)
    
 
    van_recommendations = pd.DataFrame(final_order_df)

    return van_recommendations


@task
def assign_vans_to_clusters(cluster_info_df, final_order_df, van_df):
    ride_batch_1_df = process_ride_batch(cluster_info_df, final_order_df, van_df, ride_batch=1)
    no_van_orders = ride_batch_1_df[ride_batch_1_df['VehicleNumber'] == 'no van available']['InvoiceNo'].copy()
    ride_batch_1_df = pd.DataFrame(ride_batch_1_df[ride_batch_1_df['VehicleNumber'] != 'no van available'])
    ride_batch_dfs = [ride_batch_1_df]
    ride_batch_counter = 2

    while not no_van_orders.empty:
        final_order_df2 = pd.DataFrame(final_order_df[final_order_df['InvoiceNo'].isin(no_van_orders)])
        ride_batch_df = process_ride_batch(cluster_info_df, final_order_df2, van_df, ride_batch=ride_batch_counter)
        ride_batch_dfs.append(ride_batch_df[ride_batch_df['VehicleNumber'] != 'no van available'].copy())
        no_van_orders = ride_batch_df[ride_batch_df['VehicleNumber'] == 'no van available']['InvoiceNo'].copy()
        ride_batch_counter += 1

    van_recommendations = pd.concat(ride_batch_dfs, ignore_index=True) if ride_batch_dfs else ride_batch_1_df
    
    # ride_batch_1_df = process_ride_batch(cluster_info_df, final_order_df, van_df, ride_batch=1)
    # no_van_orders = ride_batch_1_df[ride_batch_1_df['VehicleNumber'] == 'no van available']['InvoiceNo'].copy()
    # ride_batch_1_df = ride_batch_1_df[ride_batch_1_df['VehicleNumber'] != 'no van available']

    # if not no_van_orders.empty:
    #     final_order_df2 = final_order_df[final_order_df['InvoiceNo'].isin(no_van_orders)]

    #     ride_batch_2_df = process_ride_batch(cluster_info_df, final_order_df2, van_df, ride_batch=2)
    #     van_recommendations = pd.concat([ride_batch_1_df, ride_batch_2_df], ignore_index=True)
 
    # else:
    #     van_recommendations = ride_batch_1_df
    

    return van_recommendations

    
@task
def calculate_van_route(van_recommendations):
    warehouse = (van_recommendations['StockPoint_Latitude'].iloc[0], van_recommendations['StockPoint_Longitude'].iloc[0])

    def find_nearest_neighbor(current_location, customers):
        route = []
        distances_covered = []
        while len(customers) > 0:
            # print(customers.shape)
            #Optimized: Use batch matrix API instead of row-by-row
            distances = calculate_batch_distances(current_location, customers)
            nearest_index = distances.idxmin()
            if pd.isna(nearest_index):
                break  #Handle case where all distances are NaN
            route.append(nearest_index)
            distance_to_nearest = distances.loc[nearest_index]
            distances_covered.append(distance_to_nearest)
            current_location = (customers.loc[nearest_index, 'Customer_Latitude'], customers.loc[nearest_index, 'Customer_Longitude'])
            customers = customers.drop(nearest_index)
        return route, distances_covered    #,current_location  #Return current_location as the last customer location

  

    drop_off_orders = []
    drop_off_distances_covered = []
    
    #Iterate over each van and ride batch in the dataset
    for (VehicleNumber, ride_batch), group in van_recommendations.groupby(['VehicleNumber', 'Ride_Batch']):
        stockpoint_coords = (group['StockPoint_Latitude'].iloc[0], group['StockPoint_Longitude'].iloc[0])

        #Extracting customer and coords for the route calculcation
        all_customers_and_coords = group[['CustomerId', 'Customer_Latitude', 'Customer_Longitude']].copy()

        #Calculating the route for the delivery
        optimal_route_indices, distances_covered = find_nearest_neighbor(stockpoint_coords, all_customers_and_coords)


        #Assign drop-off orders and distances
        for drop_order, (index, distance) in enumerate(zip(optimal_route_indices, distances_covered), start=1):
            drop_off_orders.append((index, drop_order))
            drop_off_distances_covered.append((index, distance))

    #Converting drop-off orders and distances to DataFrames and merge them into van_recommendations
    drop_off_df = pd.DataFrame(drop_off_orders, columns=['Index', 'Drop_Off_Sequence']).set_index('Index')
    distance_df = pd.DataFrame(drop_off_distances_covered, columns=['Index', 'Drop_Off_Distance_Covered']).set_index('Index')
    
    van_recommendations['Drop_Off_Sequence'] = drop_off_df['Drop_Off_Sequence']
    van_recommendations['Drop_Off_Distance_Covered'] = distance_df['Drop_Off_Distance_Covered']
    
    #Calculate total distance for each (VehicleNumber, ride_batch) and add it as a column
    van_recommendations['Total_Distance_Covered'] = van_recommendations.groupby(['VehicleNumber', 'Ride_Batch'])['Drop_Off_Distance_Covered'].transform('sum')
     

    return van_recommendations



@flow(log_prints = True)
def execute_algorithm():
    pd.set_option('display.max_columns', None)
    stock_csv = fetch_current_stock_data()
    orders_csv = fetch_order_data()
    warehouse_csv = fetch_stockpoint_data()
    vehicle_df = fetch_vehicle_data()

    van_recommendation_csv = f"C:/Users/ME/OneDrive - Mplify Limited/Omnibiz Africa/Projects/GIT/Personal/Van-Route Optimization/Omnibiz/Van_Recommendation {current_date}.csv"

    orders_df = pd.DataFrame(pd.merge(left = orders_csv, right = warehouse_csv, left_on = "BusinessId", right_on = "StockPointId", how = "inner"))
    orders_df['Distance_From_StockPoint'] = orders_df.apply(calculate_pathway_distance, axis=1) 
    orders_df = pd.DataFrame(stock_check(stock_csv, orders_df))  
    orders_df = customer_order_clubbing_check(orders_df, vehicle_df)

    orders_df.dropna(axis = 0, how = "all", inplace = True)  

    final_order_df = get_customer_cluster(orders_df, eps_km=2, min_samples=5)

    cluster_info_df = calculate_cluster_quantity(final_order_df)

    van_recommendations = assign_vans_to_clusters(cluster_info_df, final_order_df, vehicle_df)
 
    van_recommendations = calculate_van_route(van_recommendations)

    print("Total Customers: ", van_recommendations["CustomerId"].nunique())
    print("Total Vehicles: ", van_recommendations["VehicleNumber"].nunique())
    print("Shortest Distance Covered: ", van_recommendations["Van_Min_Distance"].min())
    print("Longest Distance Covered: ", van_recommendations["Van_Max_Distance"].max())

    #save the loading & routing solution to a file
    van_recommendations.to_csv(van_recommendation_csv, index = False)


execute_algorithm()


# van_recommendations = add_fulfillment_status(van_recommendations)






