# src/H3SpatialClusterer.py 
import pandas as pd
import numpy as np
import geopandas as gpd
import h3
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.errors import GEOSException
from shapely.ops import unary_union
from typing import List, Dict, Any, Set, Union
import json

from shapely.errors import GEOSException
from shapely.validation import make_valid
from shapely.geometry import Polygon
from shapely.ops import unary_union
import h3
import math   
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
from shapely.geometry import Point
from shapely.strtree import STRtree
import multiprocessing as mp
from functools import partial 


import pandas as pd
import geopandas as gpd
import h3
import numpy as np
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import logging
from shapely.geometry import Point
from shapely.strtree import STRtree





# Configuration Constants
# CUSTOMER_DENSITY_THRESHOLD_HIGH = 200
# CUSTOMER_DENSITY_THRESHOLD_LOW = 10
# MAX_H3_RESOLUTION_URBAN = 12
# MERGE_SEARCH_RADIUS = 1
# POP_DENSITY_HIGH = 5000
# POP_DENSITY_MEDIUM = 1000

class H3SpatialClustererImproved:
    """
    Complete H3 Spatial Clustering implementation for Nigeria LGA boundaries.
    Implements boundary-constrained customer clusters using adaptive H3 hexagons.
    
    IMPROVEMENT - USING ALL Customers [Active and Recently Onboarded Customers]
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        lga_gdf: gpd.GeoDataFrame,
        sp_dim_df: pd.DataFrame,
        stock_point_lga_map: pd.DataFrame,
        sp_customers_gdf: gpd.GeoDataFrame,    
        recent_customers_gdf: gpd.GeoDataFrame, 
        crs: str = "EPSG:4326",
        resolution: int = 8,
        CUSTOMER_DENSITY_THRESHOLD_HIGH: int = 200,
        CUSTOMER_DENSITY_THRESHOLD_LOW: int = 10,
        MAX_H3_RESOLUTION_URBAN: int = 12,
        MERGE_SEARCH_RADIUS: int = 1,
        POP_DENSITY_HIGH: int = 5000,
        POP_DENSITY_MEDIUM: int = 1000
    ):
        """Initialize the clusterer with necessary geodata and configuration."""
        self.logger = logger or logging.Logger(__File__)
        self.n_cores = min(cpu_count() - 2, 8)  # Reserve 1 core, max 8
        self.lgas = lga_gdf
        self.stock_points = sp_dim_df
        self.stock_point_lga_map = stock_point_lga_map
        self.sp_customers = sp_customers_gdf
        self.customers = gpd.GeoDataFrame(pd.concat( [sp_customers_gdf.drop(columns='stock_point_id'),  recent_customers_gdf]).drop_duplicates() )
        self.crs = crs
        self.resolution = resolution
        self.CUSTOMER_DENSITY_THRESHOLD_HIGH = CUSTOMER_DENSITY_THRESHOLD_HIGH
        self.CUSTOMER_DENSITY_THRESHOLD_LOW = CUSTOMER_DENSITY_THRESHOLD_LOW
        self.MAX_H3_RESOLUTION_URBAN = MAX_H3_RESOLUTION_URBAN
        self.MERGE_SEARCH_RADIUS = MERGE_SEARCH_RADIUS
        self.POP_DENSITY_HIGH = POP_DENSITY_HIGH
        self.POP_DENSITY_MEDIUM = POP_DENSITY_MEDIUM
        
        # Processing statistics
        self.stats = {
            'territories_processed': 0,
            'total_clusters': 0,
            'customers_assigned': 0,
            'assignment_tiers': {'tier1': 0, 'tier2': 0, 'tier3': 0}
        }
        
        print("✅ H3SpatialClusterer initialized.")
        print(f"📊 Data summary: {len(self.lgas):,} LGAs, {len(self.stock_points):,} stock points, {len(self.customers):,} customers")

    def _calculate_adaptive_resolution(self, lga_ids: List[str], use_default_resolution = True) -> int:
        """Calculate adaptive H3 resolution based on population density"""
        if use_default_resolution:
            return  self.resolution
        else:
            
            if not lga_ids:
                return 8  # Default rural resolution
            
            # Get LGA data for density calculation
            relevant_lgas = self.lgas[self.lgas['lga_id'].isin(lga_ids)]
            
            if relevant_lgas.empty or 'population_density' not in relevant_lgas.columns:
                return 8
            
            # Calculate weighted average density
            if 'area_km2' in relevant_lgas.columns:
                total_area = relevant_lgas['area_km2'].sum()
                if total_area > 0:
                    weighted_density = (relevant_lgas['population_density'] * relevant_lgas['area_km2']).sum() / total_area
                else:
                    weighted_density = relevant_lgas['population_density'].mean()
            else:
                weighted_density = relevant_lgas['population_density'].mean()
            
            # Apply resolution logic from spec
            if weighted_density > self.POP_DENSITY_HIGH:  # 5000
                return 11  # Lagos megacity
            elif weighted_density > self.POP_DENSITY_MEDIUM:  # 1000  
                return 10  # Medium density
            else:
                return 8   # Rural default
    
    ## Phase 1: Territory Definition and Validation
    def define_territories(self) -> Dict[str, Dict[str, Any]]:
        """
        Phase 1: Territory Definition and Validation
        
        Creates boundary-constrained territories for each stock point using LGA geometries.
        Handles non-contiguous territories and validates geometric integrity.
        
        Returns:
            Dict[stock_point_id, {
                'polygon': Union[Polygon, MultiPolygon],
                'lga_ids': List[str],
                'is_contiguous': bool,
                'sub_territories': List[Polygon],
                'total_area_km2': float,
                'territory_version': str
            }]
        """
        print("🗺️ Phase 1: Starting territory definition...")
        
        territories = {}
        
        # Get all unique stock points
        stock_points = self.stock_point_lga_map['stock_point_id'].unique()
        
        for stock_point_id in stock_points:
            print(f"Processing territory for stock point {stock_point_id}...")
            
            # Get LGA IDs for this stock point
            lga_ids = self.stock_point_lga_map[
                self.stock_point_lga_map['stock_point_id'] == stock_point_id
            ]['lga_id'].tolist()
            
            if not lga_ids:
                print(f"⚠️ Warning: No LGAs found for stock_point_id {stock_point_id}")
                continue
            
            # Get LGA geometries
            territory_lgas = self.lgas[self.lgas['lga_id'].isin(lga_ids)].copy()
            
            if territory_lgas.empty:
                print(f"⚠️ Warning: No LGA geometries found for stock_point_id {stock_point_id}")
                continue
            
            # Validate individual LGA geometries
            validated_geometries = []
            for idx, lga in territory_lgas.iterrows():
                geom = lga['geometry']
                if not geom.is_valid:
                    print(f"🔧 Fixing invalid geometry for LGA {lga['lga_id']}")
                    geom = geom.buffer(0)  # Fix self-intersections
                validated_geometries.append(geom)
            
            # Create unified territory
            if len(validated_geometries) == 1:
                unified_territory = validated_geometries[0]
            else:
                # Union all LGA geometries
                from shapely.ops import unary_union
                unified_territory = unary_union(validated_geometries)
            
            # Final validation of unified territory
            if not unified_territory.is_valid:
                print(f"🔧 Fixing unified territory geometry for stock point {stock_point_id}")
                unified_territory = unified_territory.buffer(0)
            
            # Handle non-contiguous territories
            is_contiguous = isinstance(unified_territory, Polygon)
            sub_territories = []
            
            if isinstance(unified_territory, Polygon):
                sub_territories = [unified_territory]
            elif isinstance(unified_territory, MultiPolygon):
                sub_territories = list(unified_territory.geoms)
                print(f"📍 Non-contiguous territory detected: {len(sub_territories)} sub-territories")
            else:
                print(f"⚠️ Unexpected geometry type for stock point {stock_point_id}: {type(unified_territory)}")
                continue
            
            # Calculate total area
            total_area_km2 = sum(territory_lgas['area_km2']) if 'area_km2' in territory_lgas.columns else 0
            
            # Store territory information
            territories[str(stock_point_id)] = {
                'polygon': unified_territory,
                'lga_ids': lga_ids,
                'is_contiguous': is_contiguous,
                'sub_territories': sub_territories,
                'total_area_km2': total_area_km2,
                'territory_version': 'v1.2',
                'lga_count': len(lga_ids),
                'validation_status': 'valid'
            }
            
            # print(f"✅ Territory defined: {len(lga_ids)} LGAs, {len(sub_territories)} sub-territories, {total_area_km2:.1f} km²")
        
        print(f"🏁 Phase 1 complete: {len(territories)} territories defined")
        return territories

    ## Phase 2: H3 grid generation
    def generate_coverage_h3_cells(self, territories: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Phase 2: H3 Grid Generation with Adaptive Resolution
        
        Generates boundary-clipped H3 hexagons for each territory with density-based resolution.
        
        Args:
            territories: Dict mapping stock_point_id to territory data, including 'sub_territories' (list of shapely Polygons)
                        and 'polygon' (shapely Polygon for the entire territory).
        
        Returns:
            Dict[stock_point_id, {
                'h3_resolution': int,
                'h3_cells': Set[str],
                'clipped_cells': Set[str], 
                'cell_geometries': Dict[str, Polygon],
                'territory_coverage': float
            }]
        """
        print("🔶 Phase 2: Starting H3 grid generation...")
        
        grid_results = {}
        
        for stock_point_id, territory_data in territories.items():
            print(f"Generating H3 grid for stock point {stock_point_id}...")
            
            # Calculate adaptive resolution
            resolution = self._calculate_adaptive_resolution(territory_data['lga_ids'])
            
            # Initialize containers
            all_h3_cells = set()
            clipped_cells = set()
            cell_geometries = {}
            
            for sub_territory_idx, sub_territory in enumerate(territory_data['sub_territories']):
                try:
                    # Convert sub-territory to H3 shape and generate cells
                    h3_shape = h3.geo_to_h3shape(sub_territory.__geo_interface__)
                    sub_cells = set(h3.polygon_to_cells(h3_shape, resolution))
                    all_h3_cells.update(sub_cells)
                    print(f"Sub-territory {sub_territory_idx} generated {len(sub_cells)} H3 cells")
                    
                    # Process each H3 cell
                    for cell_id in sub_cells:
                        try:
                            # Validate cell ID
                            if not h3.is_valid_cell(cell_id):
                                print(f"⚠️ Invalid H3 cell ID {cell_id}, skipping...")
                                continue
                            
                            # Get cell boundary and ensure it's closed
                            cell_boundary = h3.cell_to_boundary(cell_id)
                            if cell_boundary[0] != cell_boundary[-1]:
                                cell_boundary = list(cell_boundary) + [cell_boundary[0]]
                            # Convert (lat, lon) to (lon, lat) for shapely
                            cell_boundary = [(lon, lat) for lat, lon in cell_boundary]
                            
                            # Create and validate cell polygon
                            cell_polygon = Polygon(cell_boundary)
                            if not cell_polygon.is_valid:
                                print(f"⚠️ Invalid cell geometry for {cell_id}, attempting to fix...")
                                cell_polygon = make_valid(cell_polygon)
                                if not cell_polygon.is_valid:
                                    print(f"⚠️ Failed to fix cell geometry for {cell_id}, skipping...")
                                    continue
                            
                            # Check for intersection with sub-territory
                            if sub_territory.intersects(cell_polygon):
                                intersection = sub_territory.intersection(cell_polygon)
                                if intersection.is_empty:
                                    print(f"Cell {cell_id} discarded: empty intersection with sub-territory {sub_territory_idx}")
                                    continue
                                
                                # Calculate areas in km² using latitude-dependent conversion
                                centroid = cell_polygon.centroid
                                lat_radians = math.radians(centroid.y)
                                deg_to_km = 111.32 * math.cos(lat_radians)  # Adjust for latitude
                                intersection_area_km2 = intersection.area * (1e6 / deg_to_km**2)
                                cell_area_km2 = cell_polygon.area * (1e6 / deg_to_km**2)
                                overlap_ratio = intersection_area_km2 / cell_area_km2 if cell_area_km2 > 0 else 0
                                
                                # Reference area from H3 library
                                cell_area_ref = h3.cell_area(cell_id, unit='km^2')  # Fixed unit from 'km2' to 'km^2'
                                # print(f"Cell {cell_id}: overlap_ratio={overlap_ratio:.3f}, "
                                #       f"intersection_area={intersection_area_km2:.3e} km², "
                                #       f"cell_area={cell_area_km2:.3e} km²")
                                
                                # Keep cells with significant overlap or area
                                if overlap_ratio > 0.01 or intersection_area_km2 > cell_area_ref * 0.5:
                                    clipped_cells.add(cell_id)
                                    cell_geometries[cell_id] = intersection
                                else:
                                    print(f"Cell {cell_id} discarded: insufficient overlap (ratio={overlap_ratio:.3f})")
                            else:
                                print(f"Cell {cell_id} discarded: does not intersect sub-territory {sub_territory_idx}")
                        
                        except GEOSException as e:
                            print(f"⚠️ Geometry error for cell {cell_id}: {e}")
                            continue
                        except Exception as e:
                            print(f"⚠️ Unexpected error processing cell {cell_id}: {e}")
                            continue
                                
                except Exception as e:
                    print(f"⚠️ Error processing sub-territory {sub_territory_idx}: {e}")
                    continue
            
            # Calculate territory coverage
            territory_coverage = 0.0
            if territory_data['polygon'].area > 0 and cell_geometries:
                try:
                    cell_union = unary_union(list(cell_geometries.values()))
                    coverage_intersection = territory_data['polygon'].intersection(cell_union)
                    territory_coverage = coverage_intersection.area / territory_data['polygon'].area
                except Exception as e:
                    print(f"⚠️ Coverage calculation error for stock point {stock_point_id}: {e}")
            
            grid_results[stock_point_id] = {
                'h3_resolution': resolution,
                'h3_cells': all_h3_cells,
                'clipped_cells': clipped_cells,
                'cell_geometries': cell_geometries,
                'territory_coverage': territory_coverage
            }
            
            print(f"✅ Generated {len(clipped_cells)} clipped cells at resolution {resolution} "
                f"with coverage {territory_coverage:.3f}")
        
        print(f"🏁 Phase 2 complete: H3 grids generated for {len(grid_results)} territories")
        return grid_results

    ## Phase 3: 3-Tier Customer Assignment Workflow
    def assign_customers_to_clusters(self, grid_results: Dict[str, Dict[str, Any]]) -> Dict[str, gpd.GeoDataFrame]:
        """
        Phase 3: 3-Tier Customer Assignment Workflow
        
        Assigns customers to H3 clusters using hierarchical confidence levels:
        Tier 1: H3 cell inclusion (confidence: 1.0)
        Tier 2: Point-in-polygon (confidence: 0.8) 
        Tier 3: Manual review (confidence: 0.0)
        
        Returns:
            Dict[stock_point_id, assignments_gdf with columns:
                ['customer_id', 
                'cluster_id', 
                'h3_cell_id', 
                'assignment_confidence', 
                'assignment_tier',
                'geometry']]
        """
        print("👥 Phase 3: Starting customer assignment...")
        
        return_cols = ['customer_id', 'cluster_id', 'h3_cell_id', 'assignment_confidence', 'assignment_tier','geometry'] 
        all_assignments = {}
        assignment_stats = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        
        for stock_point_id, grid_data in grid_results.items():
            print(f"Assigning customers for stock point {stock_point_id}...")
            
            # Get customers for this stock point
            valid_customer_ids = self.sp_customers[self.sp_customers['stock_point_id'] == int(stock_point_id)].customer_id.unique() 
            customers = self.customers.copy() # Now: Using all customers (Active and recently activated)
            print(f"Number of Customers {len(customers)}")
            
            if customers.empty:
                all_assignments[stock_point_id] = gpd.GeoDataFrame(columns=return_cols)
                continue
            
            assignments = []
            resolution = grid_data['h3_resolution']
            clipped_cells = grid_data['clipped_cells']
            cell_geometries = grid_data['cell_geometries']
            
            for _, customer in customers.iterrows():
                lat, lng = customer.geometry.y, customer.geometry.x
                customer_id = customer['customer_id']
                
                # Tier 1: H3 cell inclusion
                try:
                    h3_cell = h3.latlng_to_cell(lat, lng, resolution)
                    if h3_cell in clipped_cells:
                        assignments.append({
                            'customer_id': customer_id,
                            'cluster_id': h3_cell,
                            'h3_cell_id': h3_cell,
                            'assignment_confidence': 1.0,
                            'assignment_tier': 'h3_inclusion'
                        })
                        assignment_stats['tier1'] += 1
                        continue
                except:
                    pass
                
                # Tier 2: Point-in-polygon check
                assigned = False
                customer_point = customer.geometry
                
                for cell_id, cell_geom in cell_geometries.items():
                    if cell_geom.contains(customer_point):
                        assignments.append({
                            'customer_id': customer_id,
                            'cluster_id': cell_id,
                            'h3_cell_id': cell_id,
                            'assignment_confidence': 0.8,
                            'assignment_tier': 'point_in_polygon'
                        })
                        assignment_stats['tier2'] += 1
                        assigned = True
                        break
                
                if assigned:
                    continue
                    
                # Tier 3: Manual review
                assignments.append({
                    'customer_id': customer_id,
                    'cluster_id': None,
                    'h3_cell_id': None,
                    'assignment_confidence': 0.0,
                    'assignment_tier': 'manual_review'
                })
                assignment_stats['tier3'] += 1
            
            # Create GeoDataFrame
            assignments_df = pd.DataFrame(assignments)
            exclude_mask = (
                (assignments_df['assignment_tier'] == 'manual_review') & 
                (~assignments_df['customer_id'].isin(valid_customer_ids))
            )

            # Apply exclusion (keep rows where exclude_mask is False)
            filtered_assignments_df = assignments_df[~exclude_mask].reset_index(drop=True)
            
            # Removing geometry
            # assignments_gdf = gpd.GeoDataFrame(
            #     # assignments_df,
            #     # geometry=[customers_sp[customers_sp['customer_id'] == cid].geometry.iloc[0] 
            #     #         for cid in assignments_df['customer_id']] # Removing geometry
            # )
            
            all_assignments[stock_point_id] = filtered_assignments_df
            print(f"✅ Assigned {len(assignments_df)} customers")
            print(f"✅ Filtered Assigned {len(filtered_assignments_df)} customers")
        
        # Log statistics
        total = sum(assignment_stats.values())
        if total > 0:
            print(f"📊 Assignment stats - Tier1: {assignment_stats['tier1']/total:.1%}, "
                f"Tier2: {assignment_stats['tier2']/total:.1%}, "
                f"Tier3: {assignment_stats['tier3']/total:.1%}")
        
        print(f"🏁 Phase 3 complete: {total} customers assigned")
        return all_assignments

        ## Phase 3: 3-Tier Customer Assignment Workflow
    
    ## TO-DO: Phase 4: Cluster Optimization - Splitting and Merging

    def process_all_stock_points_(self, territory_version: str = "v1.2") -> Dict[str, Any]:
        """
        Main orchestrator method that chains all 4 phases sequentially.
        
        Returns comprehensive results dictionary with clusters, assignments, and metadata.
        """
        print("🚀 Starting complete H3 clustering pipeline...")
        
        # Phase 1: Territory Definition
        territories = self.define_territories()
        
        # Phase 2: H3 Grid Generation
        grid_results = self.generate_coverage_h3_cells(territories)
        
        # Phase 3: Customer Assignment
        assignments = self.assign_customers_to_clusters(grid_results)
        
        # Phase 4: Cluster Optimization
        # optimized_clusters = self.optimize_clusters(assignments, grid_results)
        
        # Update final statistics
        self.stats['total_clusters'] = sum(len(df) for df in assignments.values())
        # self.stats['total_clusters'] = sum(len(df) for df in optimized_clusters.values())
        
        results = {
            'territories': territories,
            'grid_results': grid_results,
            'assignments': assignments,
            'optimized_clusters': optimized_clusters,
            'statistics': self.stats,
            'territory_version': territory_version
        }
        
        print(f"🏁 Pipeline complete: {self.stats['territories_processed']} territories, "
            f"{self.stats['total_clusters']} clusters, {self.stats['customers_assigned']} customers")
        
        return results

    def export_results(self, results: Dict[str, Any], output_format: str = "sql") -> Dict[str, gpd.GeoDataFrame]:
        """
        Export results in various formats (SQL, GeoJSON, CSV).
        
        Args:
            results: Output from process_all_stock_points()
            output_format: 'sql', 'geojson', or 'csv'
        """
        if output_format == "sql":
            return self._generate_sql_exports(results)
        elif output_format == "geojson":
            return self._generate_geojson_export(results)
        elif output_format == "csv":
            return self._generate_csv_exports(results)
        else:
            raise ValueError("Supported formats: 'sql', 'geojson', 'csv'")

    def _generate_sql_exports(self, results: Dict[str, Any]) -> List[str]:
        """Generate SQL INSERT statements for database deployment."""
        sql_statements = []
        
        # H3 Clusters table
        cluster_inserts = []
        for stock_point_id, clusters_df in results['optimized_clusters'].items():
            for _, cluster in clusters_df.iterrows():
                h3_cells_json = json.dumps(cluster.get('h3_cells', [cluster['cluster_id']]))
                cluster_inserts.append(f"""(
                    '{cluster['cluster_id']}', '{stock_point_id}', {cluster['h3_resolution']},
                    '{h3_cells_json}', {cluster['customer_count']}, 
                    {f"'{cluster['parent_cluster_id']}'" if cluster.get('parent_cluster_id') else 'NULL'},
                    '{results['territory_version']}', GETDATE()
                )""")
        
        if cluster_inserts:
            sql_statements.append(f"""
            INSERT INTO h3_clusters (id, stock_point_id, h3_resolution, h3_cells, 
                                customer_count, parent_cluster_id, territory_version, created_at)
            VALUES {', '.join(cluster_inserts)};
            """)
        
        # Customer assignments
        assignment_inserts = []
        for assignments_gdf in results['assignments'].values():
            for _, assignment in assignments_gdf.iterrows():
                if assignment['cluster_id']:
                    assignment_inserts.append(f"""(
                        '{assignment['customer_id']}', '{assignment['cluster_id']}',
                        '{assignment['h3_cell_id']}', {assignment['assignment_confidence']}
                    )""")
        
        if assignment_inserts:
            sql_statements.append(f"""
            UPDATE customers SET cluster_id = updates.cluster_id, h3_cell_id = updates.h3_cell_id,
                            assignment_confidence = updates.confidence, assignment_date = GETDATE()
            FROM (VALUES {', '.join(assignment_inserts)}) AS updates(customer_id, cluster_id, h3_cell_id, confidence)
            WHERE customers.id = updates.customer_id;
            """)
        
        return sql_statements

    def _generate_geojson_export(self, results: Dict[str, Any]) -> str:
        """Export H3 clusters as GeoJSON for visualization.

        This function processes a dictionary of results, including H3 cluster
        data, grid geometries, and statistics, to generate a GeoJSON string.
        It handles type conversions to ensure all data is JSON-serializable,
        specifically addressing issues with numpy/pandas int64 types.

        Args:
            results (Dict[str, Any]): A dictionary containing various data
                                       including 'optimized_clusters',
                                       'grid_results', 'statistics', and
                                       'territory_version'.

        Returns:
            str: A JSON string representing the GeoJSON FeatureCollection.
        """
        features = []

        # Pre-build customer count lookup for efficiency
        customer_counts = {}
        for stock_point_id, clusters_df in results['optimized_clusters'].items():
            if not clusters_df.empty and 'customer_count' in clusters_df.columns:
                if 'cluster_id' in clusters_df.columns:
                    for _, row in clusters_df.iterrows():
                        # Ensure customer_count is a standard Python int.
                        # Use .item() if the value is a numpy/pandas scalar to extract the native Python type.
                        count_value = row['customer_count']
                        customer_counts[row['cluster_id']] = int(count_value.item()) if hasattr(count_value, 'item') else int(count_value)

        for stock_point_id, grid_data in results['grid_results'].items():
            for cell_id, geometry in grid_data['cell_geometries'].items():
                feature = {
                    'type': 'Feature',
                    'geometry': geometry.__geo_interface__, # Assumes geometry objects have __geo_interface__
                    'properties': {
                        'h3_cell_id': str(cell_id), # Ensure cell_id is string
                        'stock_point_id': str(stock_point_id), # Ensure stock_point_id is string
                        'h3_resolution': int(grid_data['h3_resolution']), # Ensure resolution is int
                        'customer_count': customer_counts.get(cell_id, 0), # Default to 0 if not found
                        'territory_version': str(results['territory_version']) # Ensure version is string
                    }
                }
                features.append(feature)

        # Convert stats to JSON-serializable format.
        # This loop iterates through the 'statistics' dictionary and converts
        # any non-native Python numeric types (like numpy.int64, pandas.Int64)
        # to standard Python int/float types.
        stats = {}
        for k, v in results['statistics'].items():
            if k != 'assignment_tiers': # Exclude 'assignment_tiers' as per original logic
                if isinstance(v, (np.integer, np.floating)):
                    # Convert numpy integer/float scalars to native Python int/float
                    stats[k] = v.item()
                elif pd.isna(v):
                    # Convert pandas NA (Not Applicable) values to None
                    stats[k] = None
                elif hasattr(v, 'item') and not isinstance(v, (str, dict, list, bool, int, float, type(None))):
                    # This handles pandas nullable types (e.g., pd.Int64, pd.Float64)
                    # and other custom objects that have an .item() method but are not
                    # standard Python types.
                    stats[k] = v.item()
                else:
                    # For all other types, keep them as is (e.g., strings, lists, dicts)
                    stats[k] = v

        geojson_data = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': stats
        }

        # Use json.dumps to serialize the data.
        # The 'indent=2' makes the output human-readable.
        # The 'default=str' is a fallback to convert any remaining non-serializable
        # objects to their string representation, though explicit conversions are preferred.
        return json.dumps(geojson_data, indent=2, default=str)
    
    def _generate_geojson_export_(self, results: Dict[str, Any]) -> str:
        """Export H3 clusters as GeoJSON for visualization."""
        features = []
        
        # Pre-build customer count lookup for efficiency
        customer_counts = {}
        for stock_point_id, clusters_df in results['optimized_clusters'].items():
            if not clusters_df.empty and 'customer_count' in clusters_df.columns:
                if 'cluster_id' in clusters_df.columns:
                    for _, row in clusters_df.iterrows():
                        customer_counts[row['cluster_id']] = int(row['customer_count'])
        
        for stock_point_id, grid_data in results['grid_results'].items():
            for cell_id, geometry in grid_data['cell_geometries'].items():
                feature = {
                    'type': 'Feature',
                    'geometry': geometry.__geo_interface__,
                    'properties': {
                        'h3_cell_id': str(cell_id),
                        'stock_point_id': str(stock_point_id),
                        'h3_resolution': int(grid_data['h3_resolution']),
                        'customer_count': customer_counts.get(cell_id, 0),
                        'territory_version': str(results['territory_version'])
                    }
                }
                features.append(feature)
        
        # Convert stats to JSON-serializable format
        stats = {k: (int(v) if isinstance(v, (pd.Int64Dtype, type(pd.NA))) else v) 
                for k, v in results['statistics'].items() if k != 'assignment_tiers'}
        
        geojson_data = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': stats
        }
        
        return json.dumps(geojson_data, indent=2, default=str)

    def _generate_csv_exports(self, results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Generate CSV-ready DataFrames for analysis."""
        exports = {}
        
        # Clusters summary
        all_clusters = []
        for stock_point_id, clusters_df in results['assignments'].items():
            if not clusters_df.empty:
                clusters_copy = clusters_df.copy()
                clusters_copy['stock_point_id'] = stock_point_id
                all_clusters.append(clusters_copy)
        
        if all_clusters:
            exports['clusters'] = pd.concat(all_clusters, ignore_index=True)
        
        # Customer assignments
        all_assignments = []
        for stock_point_id, assignments_gdf in results['assignments'].items():
            if not assignments_gdf.empty:
                assignments_copy = assignments_gdf.copy() #.drop(columns=['geometry'])
                assignments_copy['stock_point_id'] = stock_point_id 
                assignments_copy['h3_resolution'] = self.resolution 
                all_assignments.append(assignments_copy)
        
        if all_assignments:
            df_all_assignment = pd.concat(all_assignments, ignore_index=True)
            df_all_assignment['stock_point_id'] = df_all_assignment['stock_point_id'].astype(int)

            # Create a set for fast lookup of active customers per stock point
            active_customers = (
                self.sp_customers[['stock_point_id', 'customer_id']]
                .drop_duplicates()
                .assign(customer_type='buying customers')
                .assign(customer_type='buying customers')
            )

            # Merge once and fill missing customer_type in one step
            df_all_assignment = (
                df_all_assignment
                .merge(active_customers, how='left', on=['stock_point_id', 'customer_id'])
                .assign(customer_type=lambda df: df['customer_type'].fillna('recently activated'))
            )

            exports['assignments'] = df_all_assignment
            
        # Territory summary
        territory_summary = []
        for stock_point_id, territory_data in results['territories'].items():
            territory_summary.append({
                'stock_point_id': stock_point_id,
                'lga_count': territory_data['lga_count'],
                'is_contiguous': territory_data['is_contiguous'],
                'total_area_km2': territory_data['total_area_km2'],
                'sub_territories': len(territory_data['sub_territories']),
                'cells': results.get('grid_results', {}).get(stock_point_id, {}).get('h3_cells', []),
                'clipped_cells': results.get('grid_results', {}).get(stock_point_id, {}).get('clipped_cells', []),
                'cells_count': len(results.get('grid_results', {}).get(stock_point_id, {}).get('h3_cells', [])) if results else 0,
                'clipped_cells_count': len(results.get('grid_results', {}).get(stock_point_id, {}).get('clipped_cells', [])) if results else 0,
                'territory_coverage': results.get('grid_results', {}).get(stock_point_id, {}).get('territory_coverage', []) if results else 0


            })
        
        exports['territory_summary'] = pd.DataFrame(territory_summary)
        
         # Territory Cells 
        all_sp_territories_coverage_cells = []
        for stock_point_id, territory_data in results['territories'].items():
            df_territory = pd.DataFrame()
            df_territory['h3_cell']  = list(results.get('grid_results', {}).get(stock_point_id, {}).get('clipped_cells', []))
            df_territory['stock_point_id'] = stock_point_id
            df_territory['h3_resolution'] = results.get('grid_results', {}).get(stock_point_id, {}).get('h3_resolution', [])
            all_sp_territories_coverage_cells.append( df_territory)
         
        if all_sp_territories_coverage_cells:
            exports['territory_cells'] = pd.concat(all_sp_territories_coverage_cells, ignore_index=True) 
            
        return exports

    def get_processing_summary(self) -> Dict[str, Any]:
        """Return detailed processing statistics and summary."""
        return {
            'processing_stats': self.stats,
            'data_summary': {
                'total_lgas': len(self.lgas),
                'total_stock_points': len(self.stock_points),
                'total_customers': len(self.customers),
                'stock_point_lga_mappings': len(self.stock_point_lga_map)
            },
            'configuration': {
                'high_density_threshold': self.CUSTOMER_DENSITY_THRESHOLD_HIGH,
                'low_density_threshold': self.CUSTOMER_DENSITY_THRESHOLD_LOW,
                'max_urban_resolution': self.MAX_H3_RESOLUTION_URBAN,
                'self.MERGE_SEARCH_RADIUS': self.MERGE_SEARCH_RADIUS,
                'high_pop_density': self.POP_DENSITY_HIGH,
                'medium_pop_density': self.POP_DENSITY_MEDIUM
            }
        }
        
    # ------------------------------------------------------- ------------------------------------------------------------
    def assign_customers_to_clusters_parallel(self, grid_results: Dict[str, Dict[str, Any]]) -> Dict[str, gpd.GeoDataFrame]:
        """
        Optimized parallel customer assignment with vectorized operations
        """
        print("👥 Phase 3: Starting parallel customer assignment...")
        
        all_assignments = {}
        assignment_stats = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        
        # Process each stock point in parallel
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            futures = {}
            
            for stock_point_id, grid_data in grid_results.items():
                future = executor.submit(
                    self._process_stock_point_customers,
                    stock_point_id,
                    grid_data,
                    self.customers.copy(),
                    self.sp_customers[self.sp_customers['stock_point_id'] == int(stock_point_id)].customer_id.unique()
                )
                futures[future] = stock_point_id
            
            # Collect results
            for future in as_completed(futures):
                stock_point_id = futures[future]
                try:
                    assignments_df, stats = future.result()
                    all_assignments[stock_point_id] = assignments_df
                    
                    # Update stats
                    for tier, count in stats.items():
                        assignment_stats[tier] += count
                        
                    print(f"✅ Completed {stock_point_id}: {len(assignments_df)} customers")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {stock_point_id}: {e}")
                    all_assignments[stock_point_id] = pd.DataFrame(columns=[
                        'customer_id', 'cluster_id', 'h3_cell_id', 
                        'assignment_confidence', 'assignment_tier'
                    ])
        
        # Log final statistics
        total = sum(assignment_stats.values())
        if total > 0:
            print(f"📊 Assignment stats - Tier1: {assignment_stats['tier1']/total:.1%}, "
                  f"Tier2: {assignment_stats['tier2']/total:.1%}, "
                  f"Tier3: {assignment_stats['tier3']/total:.1%}")
        
        print(f"🏁 Phase 3 complete: {total} customers assigned")
        return all_assignments

    @staticmethod
    def _process_stock_point_customers(
        stock_point_id: str,
        grid_data: Dict[str, Any],
        customers: gpd.GeoDataFrame,
        valid_customer_ids: np.ndarray
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Process customers for a single stock point with vectorized operations
        """
        return_cols = ['customer_id', 'cluster_id', 'h3_cell_id', 'assignment_confidence', 'assignment_tier']
        assignment_stats = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        
        if customers.empty:
            return pd.DataFrame(columns=return_cols), assignment_stats
        
        resolution = grid_data['h3_resolution']
        clipped_cells = set(grid_data['clipped_cells'])  # Convert to set for O(1) lookup
        cell_geometries = grid_data['cell_geometries']
        
        # Vectorized coordinate extraction
        coords = np.column_stack([customers.geometry.y, customers.geometry.x])
        customer_ids = customers['customer_id'].values
        
        # Tier 1: H3 cell inclusion - only assign if customer's H3 cell exists in clipped_cells
        tier1_assignments = []
        unassigned_mask = np.ones(len(customers), dtype=bool)
        
        try:
            for idx, (lat, lng) in enumerate(coords):
                customer_h3_cell = h3.latlng_to_cell(lat, lng, resolution)
                
                # Only assign if the customer's H3 cell is exactly in clipped_cells
                if customer_h3_cell in clipped_cells:
                    tier1_assignments.append({
                        'customer_id': customer_ids[idx],
                        'cluster_id': customer_h3_cell,
                        'h3_cell_id': customer_h3_cell,
                        'assignment_confidence': 1.0,
                        'assignment_tier': 'h3_inclusion'
                    })
                    unassigned_mask[idx] = False
                    assignment_stats['tier1'] += 1
                
        except Exception as e:
            print(f"H3 cell assignment failed: {e}")
        
        # Tier 2: Spatial index for point-in-polygon
        tier2_assignments = []
        if cell_geometries and np.any(unassigned_mask):
            # Create spatial index
            geometries = list(cell_geometries.values())
            cell_ids = list(cell_geometries.keys())
            spatial_index = STRtree(geometries)
            
            unassigned_customers = customers[unassigned_mask]
            
            for idx, (_, customer) in enumerate(unassigned_customers.iterrows()):
                original_idx = np.where(unassigned_mask)[0][idx]
                customer_point = customer.geometry
                
                # Query spatial index for potential intersections
                possible_matches_idx = list(spatial_index.query(customer_point))
                
                assigned = False
                for geom_idx in possible_matches_idx:
                    if geometries[geom_idx].contains(customer_point):
                        cell_id = cell_ids[geom_idx]
                        tier2_assignments.append({
                            'customer_id': customer_ids[original_idx],
                            'cluster_id': cell_id,
                            'h3_cell_id': cell_id,
                            'assignment_confidence': 0.8,
                            'assignment_tier': 'point_in_polygon'
                        })
                        unassigned_mask[original_idx] = False
                        assignment_stats['tier2'] += 1
                        assigned = True
                        break
        
        # Tier 3: Manual review for remaining customers
        tier3_assignments = []
        remaining_indices = np.where(unassigned_mask)[0]
        
        for idx in remaining_indices:
            tier3_assignments.append({
                'customer_id': customer_ids[idx],
                'cluster_id': None,
                'h3_cell_id': None,
                'assignment_confidence': 0.0,
                'assignment_tier': 'manual_review'
            })
            assignment_stats['tier3'] += 1
        
        # Combine all assignments
        all_assignments = tier1_assignments + tier2_assignments + tier3_assignments
        assignments_df = pd.DataFrame(all_assignments)
        
        # Apply filtering
        if not assignments_df.empty:
            exclude_mask = (
                (assignments_df['assignment_tier'] == 'manual_review') & 
                (~assignments_df['customer_id'].isin(valid_customer_ids))
            )
            assignments_df = assignments_df[~exclude_mask].reset_index(drop=True)
        
        return assignments_df, assignment_stats

    def assign_customers_to_clusters_batch(self, grid_results: Dict[str, Dict[str, Any]], batch_size: int = 5000) -> Dict[str, gpd.GeoDataFrame]:
        """
        Alternative: Batch processing approach for memory efficiency
        """
        print(f"👥 Phase 3: Starting batch customer assignment (batch_size={batch_size})...")
        
        all_assignments = {}
        assignment_stats = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        
        for stock_point_id, grid_data in grid_results.items():
            print(f"Processing stock point {stock_point_id}...")
            
            valid_customer_ids = self.sp_customers[self.sp_customers['stock_point_id'] == int(stock_point_id)].customer_id.unique()
            customers = self.customers.copy()
            
            if customers.empty:
                all_assignments[stock_point_id] = pd.DataFrame(columns=[
                    'customer_id', 'cluster_id', 'h3_cell_id', 
                    'assignment_confidence', 'assignment_tier'
                ])
                continue
            
            # Process in batches
            batch_assignments = []
            n_customers = len(customers)
            
            for start_idx in range(0, n_customers, batch_size):
                end_idx = min(start_idx + batch_size, n_customers)
                batch_customers = customers.iloc[start_idx:end_idx]
                
                batch_result, batch_stats = self._process_stock_point_customers(
                    stock_point_id, grid_data, batch_customers, valid_customer_ids
                )
                
                batch_assignments.append(batch_result)
                
                # Update stats
                for tier, count in batch_stats.items():
                    assignment_stats[tier] += count
                
                print(f"  Processed batch {start_idx//batch_size + 1}/{(n_customers-1)//batch_size + 1}")
            
            # Combine batches
            if batch_assignments:
                all_assignments[stock_point_id] = pd.concat(batch_assignments, ignore_index=True)
            else:
                all_assignments[stock_point_id] = pd.DataFrame(columns=[
                    'customer_id', 'cluster_id', 'h3_cell_id', 
                    'assignment_confidence', 'assignment_tier'
                ])
        
        # Log statistics
        total = sum(assignment_stats.values())
        if total > 0:
            print(f"📊 Assignment stats - Tier1: {assignment_stats['tier1']/total:.1%}, "
                  f"Tier2: {assignment_stats['tier2']/total:.1%}, "
                  f"Tier3: {assignment_stats['tier3']/total:.1%}")
        
        print(f"🏁 Phase 3 complete: {total} customers assigned")
        return all_assignments
    
    
    def process_all_stock_points(self, territory_version: str = "v1.2", run_parrallel = True) -> Dict[str, Any]:
        """
        Main orchestrator method that chains all 4 phases sequentially.
        
        Returns comprehensive results dictionary with clusters, assignments, and metadata.
        """
        print("🚀 Starting complete H3 clustering pipeline...")
        
        # Phase 1: Territory/Coverage Definition
        territories = self.define_territories()
        
        # Phase 2: H3 Grid Generation
        grid_results = self.generate_coverage_h3_cells(territories)
        
        # Phase 3: Customer Assignment
        if run_parrallel:
            assignments = self.assign_customers_to_clusters_parallel(grid_results)
        else:
            assignments = self.assign_customers_to_clusters(grid_results)
            
        
        # Phase 4: Cluster Optimization
        # optimized_clusters = self.optimize_clusters(assignments, grid_results)
        
        # Update final statistics
        self.stats['total_clusters'] = sum(len(df) for df in assignments.values())
        # self.stats['total_clusters'] = sum(len(df) for df in optimized_clusters.values())
        
        results = {
            'territories': territories,
            'grid_results': grid_results,
            'assignments': assignments,
            # 'optimized_clusters': optimized_clusters,
            'statistics': self.stats,
            'territory_version': territory_version
        }
        
        print(f"🏁 Pipeline complete: {self.stats['territories_processed']} territories, "
            f"{self.stats['total_clusters']} clusters, {self.stats['customers_assigned']} customers")
        
        return results

    #  -----------------------------------------------------------------------------------   
    #  ----------------------------------- UTILS -----------------------------------------   
    #  -----------------------------------------------------------------------------------   
    def extract_coverage_and_assignment_results(self, cluster_result_dict: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract stock point coverage, customer assignment, and cluster summary DataFrames 
        from a clustering result dictionary.
        
        Args:
            cluster_result_dict (Dict): Result dictionary from clustering.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                (coverage_df, assignment_df, cluster_summary_df)
        """
        def prepare_sp_assignment_summary(coverage_df, assignment_df, sp_dim_df = self.stock_points):
            sp_assignment_summary = (assignment_df
                                    .groupby(['stock_point_id','cluster_id'])['customer_id'].count()
                                    .reset_index(name='n_customers')
                                    .rename({'cluster_id':'h3_cell'}, axis=1) 
                                ) 
            # Stock Point Coverage - Assignment Summary
            sp_coverage_cluster_and_assignment_summary = (coverage_df
                                                        .merge(sp_dim_df[['stock_point_id', 'stock_point_name']] , on='stock_point_id', how='left')
                                                        .merge(sp_assignment_summary, how='left', on=['stock_point_id','h3_cell'])
                                                        .fillna({'n_customers':0})
                                                        .rename({'h3_cell':'cluster_id'}, axis=1)
                                                        ) 
            sp_coverage_cluster_and_assignment_summary['n_customers'] = sp_coverage_cluster_and_assignment_summary['n_customers'].astype(int) 

            return sp_assignment_summary
        
        
        result_flat = self.export_results(cluster_result_dict, output_format="csv") 
        
        coverage_df = result_flat.get('territory_cells')   #or pd.DataFrame()
        assignment_df = result_flat.get('assignments') # or pd.DataFrame()
        cluster_summary_df = result_flat.get('clusters')  #or pd.DataFrame()

        assignment_df['stock_point_id'] = assignment_df['stock_point_id'].astype(int)
        coverage_df['stock_point_id'] = coverage_df['stock_point_id'].astype(int)


        sp_assignment_summary = prepare_sp_assignment_summary(coverage_df, assignment_df)

        return coverage_df, assignment_df, cluster_summary_df, sp_assignment_summary
    


 
 
 
 
 
    