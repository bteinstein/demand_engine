import os
from pathlib import Path
import pandas as pd
from datetime import datetime   
from routingpy import Valhalla
import logging
import openrouteservice as ors
import math 
import numpy as np
import folium
from routingpy import Valhalla
from .utils import create_single_stockpoint_dict
from ..routing.ValhallaManager import ValhallaManager
from ..map_viz.plot_cluster import create_enhanced_cluster_map
from ..routing.routing import get_valhalla_routes_info, plot_routes_on_map
from ..routing.routing_optimizer import RouteOptimizer 
from ..clustering.evaluate_cluster import evaluate_unsupervised_clustering
from ..map_viz.plot_cluster_and_route import plot_cluster_and_route
from dotenv import load_dotenv
load_dotenv()

current_script_path_obj_pathlib = Path(__file__).resolve()
ROOT_PATH = current_script_path_obj_pathlib.parent.parent
RECOMMENDATION_INPUT_DIR =   ROOT_PATH / 'recommendation_output'
ALL_CLUSTER_INPUT_DIR =   ROOT_PATH / 'trip_map'  
# VALHALLA_BASE_URL = os.getenv('VALHALLA_BASE_URL', 'http://localhost:8002')
# VALHALLA_API_KEY = os.getenv('VALHALLA_API_KEY', '')





# ------------------------------------------------------------------------------------------
def create_and_plot_route(df_cluster_or_trip, df_stockpoint_dim, main_dir=None, 
                          valhalla_manager=None, calc_route=False, 
                          logger=logging.getLogger(__name__), **kwargs):
    """Create and plot route using Valhalla (preferred) or ORS as fallback."""
    
    # # Setup output directory
    # if not main_dir:
    #     # Try global variable first, fallback to local implementation
    #     try:
    #         date_str = CURRENT_DATE
    #     except NameError:
    #         date_str = datetime.now().strftime('%Y-%m-%d')
        
    #     if calc_route:
    #         main_dir = RECOMMENDATION_INPUT_DIR / f'{date_str}' 
    #     else:
    #         main_dir = ALL_CLUSTER_INPUT_DIR / f'{date_str}'  # Fixed extra space
    
    # try:
    #     # Create directory if it does not exist
    #     if not os.path.exists(main_dir):
    #         logger.info(f'Creating directory: {main_dir}')
    #         os.makedirs(main_dir, exist_ok=True)
    # except Exception as e:
    #     logger.error(f'Failed to create directory {main_dir}: {e}')
    #     raise RuntimeError(f'Could not create output directory: {main_dir}')
    
    if calc_route == True: 
        logger.info('Calculating route...')
        
        # Setup routing client
        client = None
        
        # Try Valhalla first
        if valhalla_manager and valhalla_manager.check_valhalla_status():
            try:
                client = Valhalla(base_url=valhalla_manager.valhalla_url)
                logger.debug('Using local Valhalla routing client')
            except Exception as e:
                logger.warning(f'Valhalla setup failed: {e}')
        
        # Fallback to ORS if Valhalla failed or unavailable
        if client is None:
            try:
                client = ors.Client(key=os.getenv('ORS_KEY'))
                logger.debug('Using ORS routing client')
            except Exception as e:
                logger.error(f'Failed to initialize any routing client: {e}')
                raise RuntimeError('No routing client could be initialized. Please check Valhalla or ORS setup.')
        
        # Process trip data
        trip_dict = create_single_stockpoint_dict(df_cluster_or_trip, df_stockpoint_dim)
        if not trip_dict:
            logger.info('Trip data is empty - no routes to create')
            return
        
        try:
            stock_point_id = trip_dict['StockPointID']
            output_filename = main_dir / f'{stock_point_id}.html'
            
            # Get route information
            routes_info = get_valhalla_routes_info(trip_dict, client=client, logger=logger)
            
            # Plot routes on map
            plot_cluster_and_route(
                df=df_cluster_or_trip, 
                trip_dict=trip_dict, 
                routes_info=routes_info, 
                output_filename=output_filename,
                logger=logger,
                **kwargs
            )
            
            logger.info(f'Route map created successfully: {output_filename}')
            
        except Exception as e:
            logger.error(f'Failed to create route map: {e}')
            raise   
    else:
        if df_cluster_or_trip.empty:
            logger.info('No data to plot - df_cluster_or_trip is empty')
            return
        
        logger.info('Plotting cluster without route calculation...')
        stock_point_id = df_cluster_or_trip.Stock_Point_ID.iloc[0]
        output_filename = main_dir / f'{stock_point_id}.html'
        
        # Plot cluster on map
        plot_cluster_and_route(
            df=df_cluster_or_trip,  
            output_filename=output_filename,
            logger=logger,
            **kwargs
        )
        
        # logger.info(f'Cluster map created successfully: {output_filename}')   

     
    
    
# ------------------------------------------------------------------------------------------



def create_route(df_selected_trip, df_stockpoint_dim, main_dir, valhalla_manager, logger):
    # ---- SETUP CLIENT
    try:
        # Check if Valhalla is running locally
        if valhalla_manager:
            if valhalla_manager.check_valhalla_status():
                VALHALLA_BASE_URL = valhalla_manager.valhalla_url 
                # VALHALLA_API_KEY = os.getenv('VALHALLA_API_KEY', '')
                client = Valhalla(base_url=VALHALLA_BASE_URL) #, api_key=VALHALLA_API_KEY
        logger.info('Setting up routing client via LOCAL host Valhalla')
    except Exception as e:
        logger.warning('Setting up routing client via ORS')
        client = ors.Client(key=os.getenv('ORS_KEY')) 
        
    # Path 
    if not main_dir:
        CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')
        main_dir = RECOMMENDATION_INPUT_DIR / f'selected_trip_map/{CURRENT_DATE}' 
    
    # Create directory if it does not exist
    if not os.path.exists(main_dir):
        logger.info(f'Creating directory: {main_dir}')
        os.makedirs(f'{main_dir}', exist_ok=True)

    trip_dict = create_single_stockpoint_dict(df_selected_trip, df_stockpoint_dim) 

    if trip_dict == {}:
        logger.info('Trip Data is empty')
    else:
        try:
            StockPointID = trip_dict['StockPointID']
            output_filename = f'{main_dir}/{StockPointID}.html'
            # Step 1: Get route information for all trips [direction]
            calculated_routes_info = get_valhalla_routes_info(trip_dict, client  = client, logger=logger)

            # Step 2: Plot all routes on a map
            plot_routes_on_map(trip_data=trip_dict, routes_info=calculated_routes_info, output_filename = output_filename)
        except Exception as e:
            logger.warn(f'Some vital error occured while creating route {e}')
            
        
def run_route_optimizer(df_clustering, 
                    sel_cluster_tuple,
                    df_stockpoint, 
                    stock_point_name,
                    sel_total_customer_count, 
                    capacity_size = 20,
                    valhalla_manager  = None,
                    selected_trip_map_path = None,
                    logger=logging.getLogger(__name__),
                    run_valhalla = True):
    
    # ---- SETUP CLIENT
    try:
        # Check if Valhalla is running locally
        if valhalla_manager:
            if valhalla_manager.check_valhalla_status():
                VALHALLA_BASE_URL = valhalla_manager.valhalla_url 
                VALHALLA_API_KEY = os.getenv('VALHALLA_API_KEY', '')
                client = Valhalla(base_url=VALHALLA_BASE_URL, api_key=VALHALLA_API_KEY)
        logger.info('Setting up routing client via LOCAL host Valhalla')
    except Exception as e:
        logger.warning('Setting up routing client via ORS')
        client = ors.Client(key=os.getenv('ORS_KEY')) 

    if run_valhalla:
        df_sel_clust = df_clustering.query(f'cluster in {sel_cluster_tuple}').query('Latitude > 0')

        # Ensure coordinates are in [longitude, latitude] for ORS
        coords = [[lon, lat] for lat, lon in zip(df_sel_clust.Latitude, df_sel_clust.Longitude)]
        # logger.info number of jobs
        logger.info("Number of customer locations:", len(coords))
        # Convert depot_location to ORS format
        # Assuming depot_location is [lat, lon], flip to [lon, lat]
        vehicle_start = [df_stockpoint.Longitude[0], df_stockpoint.Latitude[0]]
        num_vehicles = math.floor(sel_total_customer_count / capacity_size)
        vehicles = [
            ors.optimization.Vehicle(
                id=i,
                profile='driving-car',
                start=vehicle_start,
                end=vehicle_start,
                capacity=[capacity_size]
            ) for i in range(num_vehicles)
        ]

        # Define jobs (each customer gets amount=[1])
        jobs = [ors.optimization.Job(id=index, location=coord, amount=[1]) for index, coord in enumerate(coords)]

        # Call ORS optimization API
        optimized = client.optimization(jobs=jobs, vehicles=vehicles, geometry=True)

        #     ------ MAP
        depot_location = [df_stockpoint.Latitude[0], df_stockpoint.Longitude[0]]
        depot_name = df_stockpoint.Stock_point_Name[0]

        map_clusters_route = create_enhanced_cluster_map(
                        df_sel_clust,
                        popup_cols=['CustomerID', 'LGA', 'LCDA'],
                        tooltip_cols=['LGA', 'LCDA'], 
                        zoom_start=10, 
                        radius=10
                        ).add_child(folium.Marker(location=depot_location, 
                                            size = 10, 
                                            tooltip=depot_name, 
                                            icon=folium.Icon(color="green", 
                                            icon="home")))

        # line_colors = ['green', 'orange', 'blue', 'yellow']
        separable_colors = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # yellow-green
            "#17becf",  # cyan
            "#aec7e8",  # light blue
            "#ffbb78",  # light orange
            ]

        line_colors = separable_colors[0:num_vehicles] #['green', 'orange', 'blue', 'yellow']
        for route in optimized['routes']:
            folium.PolyLine(locations=[list(reversed(coords)) for coords in ors.convert.decode_polyline(route['geometry'])['coordinates']], color=line_colors[route['vehicle']]).add_to(map_clusters_route)

        #
        selected_trip_map_path = f'./recommendation_output/selected_trip_map/{stock_point_name}_{CURRENT_DATE}.html' 
        map_clusters_route.save(selected_trip_map_path)            
        
        
        
def create_cluster_trip_optroute(df_sku_rec, 
                       df_customer_dim, 
                       df_stockpoint,
                       stock_point_id,
                       max_customers_per_route, 
                       max_volume_per_route,
                       max_distance_km, 
                       clustering_method='divisive',
                       skip_route_optimization = False,
                       logger=logging.getLogger(__name__)):
     

    optimizer = RouteOptimizer(
        max_customers_per_route=max_customers_per_route,
        max_volume_per_route=max_volume_per_route,
        max_distance_km = max_distance_km,
        logger=logger
    )

    optimizer.load_data(df_sku_rec, df_customer_dim, df_stockpoint)
    logger.info("✓ Route optimizer initialized")

    # STEP 3: Generate Routes for Stock Point 1647113
    logger.info("\n3. Generating Optimized Routes...")
    logger.info("-" * 40) 

    stock_point = df_stockpoint[df_stockpoint['Stock_Point_ID'] == stock_point_id].reset_index(drop = True)
    
    stock_point_coords = (stock_point['Latitude'], stock_point['Longitude'])
        
    clustering_customers_df = optimizer.filter_customers_for_stockpoint(stock_point_id)

    # A new clustering method can be introduced here
    # if clustering_method == 'divisive':
    # else:
    df_clustering, n_clusters = optimizer.create_geographic_clusters(clustering_customers_df, 
                                                                     clustering_method = clustering_method)

    logger.info(f"✓ Created {n_clusters} clusters for Stock Point {stock_point_id} using {clustering_method} clustering method") 
    logger.info("-" * 40)
      
    ### Cluster Evaluation
    try:
        evaluate_unsupervised_clustering(df_clustering)
    except:
        pass
    
    
    if skip_route_optimization == True:
        df_routes = pd.DataFrame()
    else:
        routes = optimizer.generate_multi_trip_routes(stock_point_id, 
                                                    max_trips=5, 
                                                    clustering_method=clustering_method)
        df_routes = pd.DataFrame(routes)
        # STEP 4: Analyze Results
        logger.info("4. Route Analysis & Results...")
        logger.info("-" * 40)



    

    push_recommendation = df_sku_rec.merge(df_clustering[['Stock_Point_ID','CustomerID', 'cluster']], 
                                           how='inner', on =['Stock_Point_ID','CustomerID'] )
    

    return push_recommendation, df_clustering, df_routes, stock_point_coords
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        