# Standard Library Imports
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Third-Party Imports
import duckdb
import h3
import pandas as pd

# Project Imports
from codebase.utils.utils import setup_logging
from config.settings import (
    ADMIN_DATA_SOURCES,
    EXPORTS_DIR,
    INPUT_BASE_DATA_SOURCES,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    STORAGE_CONFIG,
)
from src.H3SpatialClusterer import H3SpatialClusterer
from src.H3SpatialClustererImproved import H3SpatialClustererImproved
from src.data.preprocess_data import (
    load_and_preprocess_sp_lga_mapping_data,
    prepare_sp_and_recent_activated_customers,
    preprocess_sp_location_mapping,
)
from src.h3_spatial_system.data.downloader import (
    DataDownloader,
    validate_and_summarize_data,
)

from src.utils import haversine_vectorized
from src.h3_spatial_system.storage.FastH3DuckDBManager import FastH3DuckDBManager
from src.h3_spatial_system.h3_system.generator import (
    H3AddressGenerator,
    prepare_h3_address_as_dataframe,
)
from src.h3_spatial_system.h3_system.save_h3_data_utils import (
    verify_db_table_and_h3_cells_data,
)
from src.h3_spatial_system.h3_system.utils import (
    h3_to_objects_parallel_generator,
    h3_to_objects_parallel_safe,
    H3SQLiteManager,
)
from src.data_migration.migrate_duckdb_to_sqlsever import (
    migrate_all_tables,
    migrate_customer_assignment,
    migrate_stockpoint_h3_coverage,
    validate_duckdb_setup,
)



H3_DUCKDB_PATH = str(STORAGE_CONFIG["h3_duckdb_path"])
H3_CELLS_PATH = PROCESSED_DATA_DIR / 'h3_cells'
H3_ADDRESS_DF_PATH = EXPORTS_DIR / 'df_all_h3_res8_nigeria_addresses.parquet' 
H3_ADDRESS_DF_PATH = EXPORTS_DIR / 'df_all_h3_res8_nigeria_addresses.parquet' 


logger = setup_logging(log_dir='log-main-sp-clustering-and-routing', 
                       projname='log-main-spcr')


## 01. Generate h3 cells for Nigeria
def generate_h3_cells(
    logger: logging.Logger,
    resolution: int = 8,
    generator: H3AddressGenerator = H3AddressGenerator(),
    load_from_backup: bool = False,
    download_boundary_files: bool = False,
    save_cells_to_disk: bool = True
) -> List[str]:
    """
    Generate H3 cells for Nigeria at the specified resolution.
    Tries to load from backup if requested, otherwise generates new cells.

    Args:
        logger (logging.Logger): Logger for tracking progress.
        resolution (int): H3 resolution (default: 8).
        load_from_backup (bool): If True, attempts to load pre-generated H3 cells from disk.
        download_boundary_files (bool): If True, downloads admin boundary files before processing.
        save_cells_to_disk (bool): If True, saves generated H3 cells to disk.

    Returns:
        List[str]: Generated H3 cell IDs.
    """
    h3_cell_path = H3_CELLS_PATH / f"h3_cells_res{resolution}.pickle"

    # Attempt to load from backup
    if load_from_backup:
        try:
            with open(h3_cell_path, "rb") as f:
                h3_cells = pickle.load(f)
            logger.info(f"✅ Loaded {len(h3_cells):,} H3 cells from backup")
            return h3_cells
        except FileNotFoundError:
            logger.warning(f"⚠️ Backup not found at {h3_cell_path}, generating new cells...")

    # Download boundary files if requested
    if download_boundary_files:
        logger.info("📥 Downloading administrative boundary data...")
        downloader = DataDownloader()
        downloaded_files = downloader.download_admin_boundaries()

        if not downloaded_files:
            logger.error("❌ Failed to download administrative boundary data")
            return []

        validate_and_summarize_data(downloaded_files)
        logger.info(f"✅ Downloaded {len(downloaded_files)} boundary files")

    logger.info("🔷 Generating H3 cells using bounding box approach...")
    h3_cells = generator.generate_h3_cells(nigeria_boundary_path=None)

    if save_cells_to_disk:
        h3_cell_path.parent.mkdir(parents=True, exist_ok=True)
        with open(h3_cell_path, "wb") as f:
            pickle.dump(h3_cells, f)
        logger.info(f"💾 Saved {len(h3_cells):,} H3 cells to {h3_cell_path}")

    return h3_cells

## generate cell metadata and save to duckdb
def add_cell_meta_and_save_h3_cells_to_duckdb(
    h3_cells: List[str],
    h3_duckdb_manager: FastH3DuckDBManager,  
    batch_size: int = 100_000,
    verify_db: bool = False,
    logger: logging.Logger = logging.getLogger(__name__)
) -> None:
    """
    Streams H3 cell metadata from a dictionary into DuckDB in batches.

    Args:
        h3_cells (Dict[str, Any]): Dictionary mapping H3 indexes to metadata.
        batch_size (int): Number of records per batch commit (default: 100,000).
        verify_db (bool): If True, verifies DB table and stored H3 cells after insertion.
        logger (logging.Logger): Logger instance for progress updates.
    """
    processed_count = 0
         
    logger.info(f"Start generating cell meta data - centroid, boundary,  latlng_coords, polygon") 
    with h3_duckdb_manager as db: 
       for h3_index, result in h3_to_objects_parallel_generator(h3_indices=h3_cells):
            db.add_result(h3_index, result)
            processed_count += 1

            if processed_count % batch_size == 0:
                stats = db.get_stats()
                logger.info(f"📦 Processed: {processed_count:,} | In DB: {stats['total_records']:,}")

    logger.info(f"✅ Finished inserting {processed_count:,} H3 cells into DuckDB at {H3_DUCKDB_PATH}")

    if verify_db:
        logger.info("🔍 Verifying stored H3 cells in DuckDB...")
        verify_db_table_and_h3_cells_data(H3_DUCKDB_PATH)
        logger.info("✅ Database verification complete")
        

## generate cell address and save to duckdb
def generate_h3_cell_address_and_save2duckdb(
    h3_cells: List[str],
    h3_duckdb_manager: FastH3DuckDBManager,
    generator: H3AddressGenerator,
    resolution: int = 8,
    use_cached_cell_address: bool = True,
    exclude_blank_address = True,
    write_address_to_db = False,
    batch_size: int = 100_000,
    logger: logging.Logger = logging.getLogger(__name__)
):
    h3_cell_address_path = EXPORTS_DIR / f'df_all_h3_res{resolution}_nigeria_addresses.parquet'
    h3_cell_address_filtered_path = EXPORTS_DIR / f'df_filtered_h3_res{resolution}_nigeria_addresses.parquet'
    
    # if write_address_to_db:
    #     pass
    # else:
    #     pass
    # Step 5: Generate addresses - to-do optu
    if use_cached_cell_address:
        df_h3_cells_address = duckdb.sql(f"SELECT * FROM '{h3_cell_address_path}'").fetchdf().rename(columns={'h3_id':'h3_index'})
        df_h3_cells_address['resolution'] = resolution
        logger.info(f'Total h3 cells address @ resolution  {resolution}: {len(df_h3_cells_address):,}')
    else:        
        start_time = time.time() 
        logger.info("🏠 Step 5: Generating addresses...") 
        addresses_dict = generator.generate_addresses(h3_cells) 
        logger.info(f"✅ Generated {len(addresses_dict)} address records") 
        parallel_time = time.time() - start_time 
        print(f"Total time taken {len(h3_cells):,} cells took: {parallel_time:.2f} seconds") 
        df_h3_cells_address = prepare_h3_address_as_dataframe(address_data=addresses_dict)
        
        
    if exclude_blank_address:
        df_h3_cells_address =  duckdb.sql(f"SELECT * FROM df_h3_cells_address WHERE confidence_level <> 'manual_review'").fetchdf()   
        logger.info(f'Excluding manual_review addresses @ resolution {resolution}: {len(df_h3_cells_address):,}')
    
    if write_address_to_db:        
        with h3_duckdb_manager as db: 
            db.upsert_address_data(df_h3_cells_address, batch_size)     


def refresh_base_data(refresh_input_data = True,
                      logger: logging.Logger = logging.getLogger(__name__)):
    import sys
    from pathlib import Path 
    import geopandas as gpd

    from src.get_data import DataFetcher, get_processed_data, get_geojson_data
    from src.data.preprocess_data import preprocess_sp_location_mapping
    
    if refresh_input_data:
        ## Fetch Data from DB
        fetcher = DataFetcher(logger=logger, input_dir=str(RAW_DATA_DIR), sql_dir="_sql")
        results = fetcher.fetch_all()

    return results

def preprocess_input_data(logger: logging.Logger = logging.getLogger(__name__)):
    # 2. Preprocess Sp Location Mapping LGA
    preprocess_sp_location_mapping(logger=logger)
    prepare_sp_and_recent_activated_customers(logger)

def load_processed_data(logger: logging.Logger = logging.getLogger(__name__)):
    from src.get_data import get_processed_data 
    lgas_gdf, sp_dim_df,  stock_point_lga_map, sp_customers_gdf, recent_customers_gdf  = get_processed_data(logger)
    return lgas_gdf, sp_dim_df,  stock_point_lga_map, sp_customers_gdf, recent_customers_gdf   
            
def main_run_pipeline_dep(resolution: int = 8, 
         logger = logger or logging.getLogger(__name__),
         regenerate_cell_and_metadata: bool = False,
         generate_new_cell_metadata_and_save2db: bool = False,
         use_cached_cell_address: bool = True,
         write_address_to_db = False,
         refresh_input_data = False,
         run_pilot = True,
         save_all_output_to_disk = True,
         save_all_output_to_db = True
         ):
    
    # Start DB 
    h3_duckdb_manager =FastH3DuckDBManager(db_path=H3_DUCKDB_PATH, resolution=8, batch_size=100_000)
    
    # Load geojson paths
    dict_path_geojson = {
        k: v["standardize_file_path"] for k, v in ADMIN_DATA_SOURCES.items()
    }
    
    if regenerate_cell_and_metadata:
        logger.info("🚀 Starting H3 address system generation for Nigeria")
        generator = H3AddressGenerator(resolution=resolution)

        logger.info("🗺️ Loading administrative boundaries...")
        generator.load_admin_boundaries(
            str(dict_path_geojson["states"]),
            str(dict_path_geojson["lgas"]),
            str(dict_path_geojson["wards"])
        )
        logger.info("✅ Administrative boundaries loaded")
        
        # 1. generate or load h3_cells
        h3_cells = generate_h3_cells(logger = logger,
                                    resolution = resolution,
                                    load_from_backup  = True
                                )  
        
        # 2. add cell meta data and save to duckdb
        if generate_new_cell_metadata_and_save2db: # ETA: 15mins
            add_cell_meta_and_save_h3_cells_to_duckdb(h3_cells = h3_cells,
                                                    resolution = resolution,
                                                    batch_size = 100_000,
                                                    verify_db = False,
                                                    logger = logger
                                                    )     

        # 3. Generate h3 Address
        generate_h3_cell_address_and_save2duckdb(h3_cells = h3_cells,
                                                h3_duckdb_manager = h3_duckdb_manager,
                                                generator = generator,
                                                resolution = resolution,
                                                use_cached_cell_address = use_cached_cell_address,
                                                exclude_blank_address = True,
                                                write_address_to_db = write_address_to_db,
                                                batch_size = 100_000,
                                                logger=logger)

    # 4. Get Base Input Data
    if refresh_input_data:
        result_ = refresh_base_data(refresh_input_data = refresh_input_data,
                      logger = logger)

        # 5. Preprocess Base Input Data
        preprocess_input_data(logger = logger)

    # 6. load Pre-Processed Base Input Data
    lgas_gdf, sp_dim_df,  stock_point_lga_map, sp_customers_gdf, recent_customers_gdf = load_processed_data()

    # 7. Clusting and Customer Assignment
    # Set-Clustering Class
    if run_pilot:
        pilot_2_sps = [1647402,	1647372,	1647108,	1646971,	1647109,	1647033,	
                1646999,	1647391,	1647113,	1647137,	1646991,	1647420,	
                1647141,	1647050,	1647421,	1647436,	1647380]
        # Filter data using the pilot_sps
        sp_dim_df = sp_dim_df[sp_dim_df['stock_point_id'].isin(pilot_2_sps)] 
        stock_point_lga_map = stock_point_lga_map[stock_point_lga_map['stock_point_id'].isin(pilot_2_sps)] 
        sp_customers_gdf=sp_customers_gdf[sp_customers_gdf['stock_point_id'].isin(pilot_2_sps)]
        
        print('sp_dim_df', sp_dim_df.shape[0])
        print('stock_point_lga_map', stock_point_lga_map.shape[0])
        print('sp_customers_gdf', sp_customers_gdf.shape[0])
        
    clustererImproved = H3SpatialClustererImproved(
            logger=logger,
            lga_gdf=lgas_gdf,
            sp_dim_df=sp_dim_df,
            stock_point_lga_map=stock_point_lga_map,
            sp_customers_gdf = sp_customers_gdf,
            recent_customers_gdf = recent_customers_gdf,
            resolution=8
            )

    CLUSTER_RESULTS_DICT =  clustererImproved.process_all_stock_points(territory_version="v1.2")
        
    if save_all_output_to_disk:
        current_date = datetime.now().strftime('%Y-%m-%d')
        suffix = "PILOT" if run_pilot else "ALL"
        clusters_results_dict_file_path = EXPORTS_DIR /  f"{suffix}_SPS_CLUSTER_R{resolution}_{current_date}.pickle"

        # Save Results as pickle file
        with open(clusters_results_dict_file_path, 'wb') as f:
            pickle.dump(CLUSTER_RESULTS_DICT, f)    
    
    # Extract Results as Dataframe for DB Upsert
    sp_coverage_df, sp_assignment_df, sp_cluster_summary_df, sp_assignment_summary_df = clustererImproved.extract_coverage_and_assignment_results(CLUSTER_RESULTS_DICT)    
    
    # Enhance sp_coverage_df
    # conn = duckdb.connect(H3_DUCKDB_PATH)
    with duckdb.connect(H3_DUCKDB_PATH) as conn:
        print('Enhancing sp_coverage_df')
        sp_coverage_df_enhanced = conn.execute('''SELECT 
                        a.stock_point_id,
                        a.h3_cell ,  
                        a.h3_resolution, 
                        c.centroid_lat as cluster_centroid_lat, 
                        c.centroid_lng as cluster_centroid_lng,
                        b.latitude as sp_lat, 
                        b.longitude as sp_lng
                    FROM sp_coverage_df a
                    LEFT JOIN  sp_dim_df b ON b.stock_point_id = a.stock_point_id
                    LEFT JOIN h3_cells c ON c.h3_index = a.h3_cell 
                ''').df()
        
    sp_coverage_df_enhanced['cluster_sp_dist_km'] = haversine_vectorized(
            sp_coverage_df_enhanced['cluster_centroid_lat'], 
            sp_coverage_df_enhanced['cluster_centroid_lng'],
            sp_coverage_df_enhanced['sp_lat'], 
            sp_coverage_df_enhanced['sp_lng']
        )

    sp_coverage_df_enhanced = sp_coverage_df_enhanced.drop(columns=['sp_lat','sp_lng'])
    
    ## Upsert coverage and assignment to db
    if save_all_output_to_db:
        with h3_duckdb_manager as db: 
            db.truncate_insert_stockpoint_h3_coverage(sp_coverage_df_enhanced)
            db.upsert_customer_cluster_assignment(sp_assignment_df) #Table Name: customer_stockpoint_cluster_assignment
            
            
            
def main_run_pipeline(
    resolution: int = 8, 
    logger=None,
    regenerate_cell_and_metadata: bool = False,
    generate_new_cell_metadata_and_save2db: bool = False,
    use_cached_cell_address: bool = True,
    write_address_to_db: bool = False,
    refresh_input_data: bool = False,
    run_pilot: bool = True,
    save_all_output_to_disk: bool = True,
    save_all_output_to_db: bool = True,
    migrate_data_to_sqlserver = True
):
    """
    Main pipeline for H3 spatial clustering and customer assignment.
    
    Args:
        resolution: H3 grid resolution (default: 8)
        logger: Logger instance
        regenerate_cell_and_metadata: Whether to regenerate H3 cells and metadata
        generate_new_cell_metadata_and_save2db: Whether to generate and save new metadata to DB
        use_cached_cell_address: Use cached cell addresses
        write_address_to_db: Write addresses to database
        refresh_input_data: Refresh base input data
        run_pilot: Run with pilot subset of stock points
        save_all_output_to_disk: Save results to disk
        save_all_output_to_db: Save results to database
    """
    
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    suffix = "PILOT" if run_pilot else "ALL" 
    cluster_results_filepath = EXPORTS_DIR / 'clustering' / f"{suffix}_SPS_CLUSTER_R{resolution}_{current_date}.pickle"
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Initialize database manager
    h3_duckdb_manager = FastH3DuckDBManager(
        db_path=H3_DUCKDB_PATH, 
        resolution=resolution, 
        batch_size=100_000
    )
    
    # Load admin boundaries if regenerating cells
    if regenerate_cell_and_metadata:
        logger.info("🚀 Starting H3 address system generation for Nigeria")
        generator = H3AddressGenerator(resolution=resolution)
        
        dict_path_geojson = {
            k: v["standardize_file_path"] for k, v in ADMIN_DATA_SOURCES.items()
        }
        
        logger.info("🗺️ Loading administrative boundaries...")
        generator.load_admin_boundaries(
            str(dict_path_geojson["states"]),
            str(dict_path_geojson["lgas"]),
            str(dict_path_geojson["wards"])
        )
        logger.info("✅ Administrative boundaries loaded")
        
        # Generate H3 cells
        h3_cells = generate_h3_cells(
            logger=logger,
            resolution=resolution,
            load_from_backup=True
        )
        
        # Add metadata and save to database
        if generate_new_cell_metadata_and_save2db:
            logger.info("Adding cell metadata to database")
            add_cell_meta_and_save_h3_cells_to_duckdb(
                h3_cells=h3_cells,
                resolution=resolution,
                batch_size=100_000,
                verify_db=False,
                logger=logger
            )
        
        # Generate addresses
        generate_h3_cell_address_and_save2duckdb(
            h3_cells=h3_cells,
            h3_duckdb_manager=h3_duckdb_manager,
            generator=generator,
            resolution=resolution,
            use_cached_cell_address=use_cached_cell_address,
            exclude_blank_address=True,
            write_address_to_db=write_address_to_db,
            batch_size=100_000,
            logger=logger
        )
    
    # Refresh input data if needed
    if refresh_input_data:
        logger.info("Refreshing base input data")
        refresh_base_data(refresh_input_data=refresh_input_data, logger=logger)
        preprocess_input_data(logger=logger)
    
    # Load processed data
    logger.info("Loading processed data")
    lgas_gdf, sp_dim_df, stock_point_lga_map, sp_customers_gdf, recent_customers_gdf = load_processed_data()
    
    # Filter for pilot if specified
    if run_pilot:
        pilot_sps = [1647402, 1647372, 1647108, 1646971, 1647109, 1647033, 
                    1646999, 1647391, 1647113, 1647137, 1646991, 1647420,
                    1647141, 1647050, 1647421, 1647436, 1647380]
        
        sp_dim_df = sp_dim_df[sp_dim_df['stock_point_id'].isin(pilot_sps)]
        stock_point_lga_map = stock_point_lga_map[stock_point_lga_map['stock_point_id'].isin(pilot_sps)]
        sp_customers_gdf = sp_customers_gdf[sp_customers_gdf['stock_point_id'].isin(pilot_sps)]
        
        logger.info(f"Pilot mode: {len(pilot_sps)} stock points, "
                   f"{sp_dim_df.shape[0]} SP records, "
                   f"{sp_customers_gdf.shape[0]} customer records")
    
    # Initialize and run clustering
    logger.info("Starting spatial clustering")
    clusterer = H3SpatialClustererImproved(
        logger=logger,
        lga_gdf=lgas_gdf,
        sp_dim_df=sp_dim_df,
        stock_point_lga_map=stock_point_lga_map,
        sp_customers_gdf=sp_customers_gdf,
        recent_customers_gdf=recent_customers_gdf,
        resolution=resolution
    )
    
    cluster_results = clusterer.process_all_stock_points(territory_version="v1.2")
    
    # Save to disk if requested
    if save_all_output_to_disk:
        # file_path = EXPORTS_DIR / f"{suffix}_SPS_CLUSTER_R{resolution}_{current_date}.pickle"
        logger.info(f"Saving results to {cluster_results_filepath}")
        with open(cluster_results_filepath, 'wb') as f:
            pickle.dump(cluster_results, f)
    
    # Extract results for database
    logger.info("Extracting results for database")
    sp_coverage_df, sp_assignment_df, sp_cluster_summary_df, sp_assignment_summary_df = (
        clusterer.extract_coverage_and_assignment_results(cluster_results)
    )
    
    # Enhance coverage data with spatial information
    logger.info("Enhancing coverage data")
    try: 
        
        with duckdb.connect(H3_DUCKDB_PATH) as conn:
            sp_coverage_df_enhanced = conn.execute('''
                SELECT 
                    a.stock_point_id,
                    a.h3_cell,  
                    a.h3_resolution, 
                    c.centroid_lat as cluster_centroid_lat, 
                    c.centroid_lng as cluster_centroid_lng,
                    b.latitude as sp_lat, 
                    b.longitude as sp_lng
                FROM sp_coverage_df a
                LEFT JOIN sp_dim_df b ON b.stock_point_id = a.stock_point_id
                LEFT JOIN h3_cells c ON c.h3_index = a.h3_cell 
            ''').df()
        
        # Calculate distances
        sp_coverage_df_enhanced['cluster_sp_dist_km'] = haversine_vectorized(
            sp_coverage_df_enhanced['cluster_centroid_lat'], 
            sp_coverage_df_enhanced['cluster_centroid_lng'],
            sp_coverage_df_enhanced['sp_lat'], 
            sp_coverage_df_enhanced['sp_lng']
        )
        sp_coverage_df_enhanced = sp_coverage_df_enhanced.drop(columns=['sp_lat', 'sp_lng'])
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise
    finally:
        logger.debug("Database connection cleanup completed")
        
    
    # Save to database if requested
    if save_all_output_to_db:
        logger.info("Saving results to database")
        with h3_duckdb_manager as db:
            db.truncate_insert_stockpoint_h3_coverage(sp_coverage_df_enhanced)
            db.upsert_customer_cluster_assignment(sp_assignment_df)
    
    # Prepare data for visualization
    logger.info("Preparing data for visualization")
    try:
        from src.data.load_postprocessed_data import postprocess_map_data
        postprocess_map_data(return_data = False, from_local=False)
    except Exception as e:
        logger.error(f"Data postprocessing failed: {e}") 
    
    if migrate_data_to_sqlserver:
        logger.info("Migrate Data to SQL SERVER ")
        # Each can be called independently
        # migrate_stockpoint_h3_coverage()  # Works standalone
        # migrate_customer_assignment()     # Works standalone  
        info = migrate_all_tables()             # Works standalone
    
    logger.info("Pipeline completed successfully")
    # return cluster_results            
            
            
            