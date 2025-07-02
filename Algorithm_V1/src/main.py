import logging
from .data.data_filter import data_filter 
from .utils_and_postprocessing.run_clustering_and_routing import create_and_plot_route, create_cluster_trip_optroute 
from src.utils_and_postprocessing.utils import cluster_summary_and_selection, postprocess_selected_trip
from src.clustering.evaluate_cluster import evaluate_unsupervised_clustering
from src.data.export import export_data 
from pathlib import Path



def setup_directories():
    """Set up directory structure for recommendation output with date-based subdirectories."""
    global CURRENT_DATE, SELECTED_TRIP_PATH, ALL_CLUSTER_PATH, LOCAL_EXCEL_PATH
    try:
        # Set current date
        CURRENT_DATE = datetime.today().date()
        
        # Define date-based directory paths
        SELECTED_TRIP_PATH = RECOMMENDATION_DIR / SELECTED_TRIP_DIR / str(CURRENT_DATE)
        ALL_CLUSTER_PATH = RECOMMENDATION_DIR / CLUSTER_MAP_DIR / str(CURRENT_DATE)
        LOCAL_EXCEL_PATH = RECOMMENDATION_DIR / EXCEL_DOCS_DIR / str(CURRENT_DATE)
        
        # Create directories if they don't exist
        for directory in (SELECTED_TRIP_PATH, ALL_CLUSTER_PATH, LOCAL_EXCEL_PATH):
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")

        # Copy index.html files from source directories to date-based subdirectories
        source_cluster_map_index = BASE_DIR / 'html' / 'default_cluster_map_index.html'
        source_selected_trip_index = BASE_DIR / 'html' / 'default_selected_cluster_map_index.html' 

        # Copy index.html for selected_trip_map
        try:
            if source_selected_trip_index.exists():
                shutil.copy2(source_selected_trip_index, SELECTED_TRIP_PATH / INDEX_HTML)
                logger.info(f"Copied index.html to {SELECTED_TRIP_PATH / INDEX_HTML}")
            else:
                logger.warning(f"Source index.html not found at {source_selected_trip_index}")
        except Exception as e:
            logger.error(f"Failed to copy index.html to {SELECTED_TRIP_PATH}: {str(e)}")

        # Copy index.html for cluster_map
        try:
            if source_cluster_map_index.exists():
                shutil.copy2(source_cluster_map_index, ALL_CLUSTER_PATH / INDEX_HTML)
                logger.info(f"Copied index.html to {ALL_CLUSTER_PATH / INDEX_HTML}")
            else:
                logger.warning(f"Source index.html not found at {source_cluster_map_index}")
        except Exception as e:
            logger.error(f"Failed to copy index.html to {ALL_CLUSTER_PATH}: {str(e)}")

    except Exception as e:
        logger.error(f"Error setting up directories: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up basic logging configuration
    logging.basicConfig(level=logging.INFO)
    setup_directories()
    
    


def run_push_recommendation(df_customer_sku_recommendation, 
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
                            save_to_disk = False,
                            # Global variables
                            valhalla_manager = None,
                            CURRENT_DATE = None,
                            SELECTED_TRIP_PATH = None,
                            ALL_CLUSTER_PATH = None,
                            LOCAL_EXCEL_PATH = None,
                            logger=logging.getLogger(__name__)):
    """
    Main execution function demonstrating complete route optimization workflow
    """ 

    logger.info("=" * 80)
    logger.info("ROUTE OPTIMIZATION FOR PUSH SALES RECOMMENDATIONS")
    logger.info(f"StockPoint: {stock_point_name}, StockPointID: {stock_point_id},")
    logger.info("=" * 80)

    # STEP 1: Load or Generate Data
    logger.info("\n1. Loading Data...")
    logger.info("-" * 40)

    df_sku_rec, df_customer_dim, df_stockpoint  = data_filter(df_customer_sku_recommendation, 
                                                                df_master_customer_dim, 
                                                                df_stockpoint_dim, 
                                                                stockpoint_id = stock_point_id,  
                                                                sku_recency = sku_recency, 
                                                                customer_recency = customer_recency, 
                                                                number_recommendation = number_recommendation,
                                                                estimate_qty_scale_factor = estimate_qty_scale_factor, 
                                                                max_estimated_qty = max_estimated_qty,
                                                                exclude_recency_customer = exclude_recency_customer)

    if len(df_customer_dim) < min_ncust_per_cluster:
        return {}

    logger.info(f"✓ Loaded {len(df_sku_rec)} SKU recommendations")
    logger.info(f"✓ Loaded {len(df_customer_dim)} customer records")
    logger.info(f"✓ Loaded {len(df_stockpoint)} stock points")

    push_recommendation, df_clustering, df_routes, stock_point_coords = create_cluster_trip_optroute(df_sku_rec, 
                                                                                            df_customer_dim, 
                                                                                            df_stockpoint,
                                                                                            stock_point_id,
                                                                                            max_customers_per_route, 
                                                                                            max_volume_per_route,
                                                                                            max_distance_km,
                                                                                            clustering_method,
                                                                                            skip_route_optimization,
                                                                                            logger=logger)


    ### Cluster Summary 
    cluster_summary, df_high_value_cluster_summary, sel_cluster_tuple, sel_total_customer_count = cluster_summary_and_selection(
                                                                                                        push_recommendation,
                                                                                                        sel_trip_cluster,
                                                                                                        min_ncust_per_cluster = min_ncust_per_cluster,
                                                                                                        logger=logger)

    ## Trip
    selected_push_recommendation_trip, all_push_recommendation = postprocess_selected_trip(push_recommendation, 
                                                    cluster_summary, 
                                                    df_master_customer_dim,  
                                                    df_stockpoint,
                                                    sel_cluster_tuple)


    ## Cluster Evaluation
    try:
        evaluate_unsupervised_clustering(selected_push_recommendation_trip)
    except:
        pass
    
    ## Selected Trip Summary
    df_selected_trip_summary =  (selected_push_recommendation_trip
                                 .groupby(['StockPointID','TripID','CustomerID', 'Latitude','Longitude',
                                           'LGA', 'LCDA','CustomerScore'])
                                 .agg( TotalQuantity = ('EstimatedQuantity','sum')
                                      ,TotalSKU = ('SKUID','nunique'))
                                 .reset_index()
                                )
    
    
    ## Create selected trip route and map
    try:
        create_and_plot_route(df_cluster_or_trip = df_selected_trip_summary, 
                              df_stockpoint_dim = df_stockpoint_dim, 
                              main_dir = SELECTED_TRIP_PATH, 
                              valhalla_manager = valhalla_manager, 
                              calc_route = True, 
                              logger = logger)
    except Exception as e:
        logger.error(f"Error in creating and plotting route: {e}")
    
    ## Create and save cluster map
    try:
        create_and_plot_route(df_cluster_or_trip = df_clustering, 
                      df_stockpoint_dim = df_stockpoint_dim, 
                      main_dir=ALL_CLUSTER_PATH, 
                          valhalla_manager=None, calc_route = False, 
                          logger=logger, cluster_col='cluster')
    except Exception as e:
        logger.info(f'Unable to generate and save cluster map: {e}') 
    
    ## Local Export
    ### Export Data
    if save_to_disk:
        try:
            export_data(
                selected_trip = selected_push_recommendation_trip,
                all_push_recommendation = all_push_recommendation,
                cluster_summary = cluster_summary,
                stock_point_name = stock_point_name,
                CURRENT_DATE = CURRENT_DATE,
                LOCAL_EXCEL_PATH  = LOCAL_EXCEL_PATH
            )
        except Exception as e:
            logger.error(f'Can not save to disk {e}')
     

    dict_ = {
        'stock_point_name': stock_point_name,
        'selected_trip': selected_push_recommendation_trip,
        'all_push_recommendation': all_push_recommendation,
        'cluster_summary': cluster_summary
    }

    return dict_
    