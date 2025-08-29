"""
Main H3 address generator for Nigeria.
./src/generator.py
"""

import h3
import geopandas as gpd
from shapely.geometry import Polygon, shape
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import time
import concurrent.futures
import os
from asyncio import futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict
import math


from config.settings import (
    H3_RESOLUTION, COUNTRY_CONFIG, PROCESSING_CONFIG, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)


# Global variables for multiprocessing pool
admin_assignment_instance = None
id_generator_instance = None


def init_worker(admin_units, id_config):
    """Initialize worker process with shared data."""
    global admin_assignment_instance, id_generator_instance
    from src.h3_system.admin_assignment import AdministrativeAssignment
    from src.h3_system.id_generator import IDGenerator
    
    admin_assignment_instance = AdministrativeAssignment()
    admin_assignment_instance.admin_units = admin_units
    id_generator_instance = IDGenerator(id_config)



from .admin_assignment import AdministrativeAssignment
from .id_generator import IDGenerator

logger = logging.getLogger(__name__)


class H3AddressGenerator:
    """Main generator for H3-based address system."""
    
    def __init__(self, resolution: Optional[int] = None):
        self.resolution = resolution or H3_RESOLUTION
        self.country_config = COUNTRY_CONFIG
        self.processing_config = PROCESSING_CONFIG
        
        # Initialize components
        self.admin_assignment = AdministrativeAssignment()
        self.id_generator = IDGenerator()
        
        # Storage for generated data
        self.h3_cells = []
        self.address_data = []
        
    def generate_h3_cells(self, nigeria_boundary_path: Optional[str] = None) -> List[str]:
        """
        Generate H3 cells covering Nigeria.
        
        Args:
            nigeria_boundary_path: Path to Nigeria boundary file (optional)
            
        Returns:
            List of H3 cell IDs
        """
        logger.info(f"Generating H3 cells at resolution {self.resolution}...")
        
        if nigeria_boundary_path and Path(nigeria_boundary_path).exists():
            # Use provided boundary file
            h3_cells = self._generate_from_boundary_file(nigeria_boundary_path)
        else:
            # Use bounding box approach
            h3_cells = self._generate_from_bounds()
        
        self.h3_cells = h3_cells
        logger.info(f"Generated {len(h3_cells):,} H3 cells")
        
        return h3_cells
    
    def _generate_from_boundary_file(self, boundary_path: str) -> List[str]:
        """Generate H3 cells from a boundary file."""
        try:
            # Load boundary
            boundary_gdf = gpd.read_file(boundary_path)
            
            # Get the first geometry (assuming single country boundary)
            nigeria_geom = boundary_gdf.iloc[0].geometry
            
            # Convert to H3Shape format for h3.polygon_to_cells
            if hasattr(nigeria_geom, '__geo_interface__'):
                geojson = nigeria_geom.__geo_interface__
            else:
                # Convert to GeoJSON manually
                coords = list(nigeria_geom.exterior.coords)
                geojson = {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            
            # Convert GeoJSON to H3Shape and generate H3 cells
            h3_shape = h3.geo_to_h3shape(geojson)
            h3_cells = h3.polygon_to_cells(h3_shape, self.resolution)
            
            return list(h3_cells)
            
        except Exception as e:
            logger.error(f"Failed to generate from boundary file: {e}")
            return self._generate_from_bounds()
    
    def _generate_from_bounds(self) -> List[str]:
        """Generate H3 cells from bounding box."""
        bounds = self.country_config['bounds']
        
        # Create a simple polygon covering Nigeria
        min_lat, max_lat = bounds['min_lat'], bounds['max_lat']
        min_lng, max_lng = bounds['min_lng'], bounds['max_lng']
        
        # Create polygon coordinates
        coords = [
            [min_lng, min_lat],
            [max_lng, min_lat],
            [max_lng, max_lat],
            [min_lng, max_lat],
            [min_lng, min_lat]  # Close the polygon
        ]
        
        geojson = {
            "type": "Polygon",
            "coordinates": [coords]
        }
        
        # Convert GeoJSON to H3Shape and generate H3 cells
        h3_shape = h3.geo_to_h3shape(geojson)
        h3_cells = h3.polygon_to_cells(h3_shape, self.resolution)
        
        return list(h3_cells)
    
    def load_admin_boundaries(self, states_path: str, lgas_path: str, wards_path: str):
        """Load administrative boundary data."""
        logger.info("Loading administrative boundaries...")
        self.admin_assignment.load_admin_boundaries(states_path, lgas_path, wards_path)    

    def generate_addresses_(self, geojson_file: str) -> List[Dict]:
        """
        Generate H3 addresses for polygons in a GeoJSON file using parallel processing.
        
        Args:
            geojson_file (str): Path to GeoJSON file
            
        Returns:
            List[Dict]: List of dictionaries with H3 addresses and properties
        """
        # Load GeoJSON efficiently
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
        
        addresses = []
        
        # def process_feature(feature: Dict) -> List[Dict]:
        #     """Process a single GeoJSON feature and return H3 addresses."""
        #     result = []
        #     geometry = feature.get('geometry', {})
        #     properties = feature.get('properties', {})
            
        #     if geometry.get('type') == 'Polygon':
        #         coords = geometry.get('coordinates', [])
        #         for polygon in coords:
        #             h3_addresses = h3.polyfill({
        #                 'type': 'Polygon',
        #                 'coordinates': polygon
        #             }, self.resolution, geo_json_conformant=True)
                    
        #             for addr in h3_addresses:
        #                 result.append({
        #                     'h3_address': addr,
        #                     'properties': properties
        #                 })
        #     return result

        def process_feature(feature: Dict, resolution: int) -> List[Dict]:
            """Process a single GeoJSON feature and return H3 addresses."""
            result = []
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})
            
            try:
                # Convert GeoJSON geometry to a shapely shape (Polygon or MultiPolygon)
                shapely_geom = shape(geometry)
                
                # Only process Polygon or MultiPolygon
                if shapely_geom.geom_type in ['Polygon', 'MultiPolygon']:
                    h3_addresses = h3.polygon_to_cells(shapely_geom, resolution=resolution)
                    
                    for addr in h3_addresses:
                        result.append({
                            'h3_address': addr,
                            'properties': properties
                        })
            except Exception as e:
                print(f"Error processing feature: {e}")

            return result

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map features to parallel processing
            futures = [
                executor.submit(process_feature, feature)
                for feature in geojson_data.get('features', [])
            ]
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                addresses.extend(future.result())
        
        return addresses

    def generate_addresses(self, h3_cells: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate complete address data for H3 cells with performance optimizations.
        
        Args:
            h3_cells: List of H3 cell IDs (uses self.h3_cells if not provided)
            
        Returns:
            List of address records
        """
        if h3_cells is None:
            h3_cells = self.h3_cells
        
        if not h3_cells:
            raise ValueError("No H3 cells available. Run generate_h3_cells() first.")
        
        logger.info(f"Generating addresses for {len(h3_cells)} H3 cells...")
        
        # return self._generate_addresses_parallel(h3_cells) # Temp solution
        return self._generate_addresses_sequential(h3_cells) # Temp solution

        # # Performance optimization: Use parallel processing for large datasets
        # if len(h3_cells) > self.processing_config.get('parallel_trigger_threshold', 1000):
        #     return self._generate_addresses_parallel(h3_cells)
        # else:
        #     return self._generate_addresses_sequential(h3_cells)
    
        #    return all_addresses

    def _generate_addresses_sequential(self, h3_cells: List[str]) -> List[Dict]:
        """Generate addresses using sequential processing for smaller datasets."""
        # Process in chunks for memory efficiency
        chunk_size = self.processing_config['chunk_size']
        all_addresses = []
        
        for i in tqdm(range(0, len(h3_cells), chunk_size), desc="Processing H3 cells"):
            chunk = h3_cells[i:i + chunk_size]
            chunk_addresses = self._process_chunk(chunk)
            all_addresses.extend(chunk_addresses)
        
        self.address_data = all_addresses
        logger.info(f"Generated {len(all_addresses)} address records using sequential processing")
        
        return all_addresses
    
    def _process_chunk(self, h3_cells: List[str]) -> List[Dict]:
        """Process a chunk of H3 cells (sequential processing)."""
        chunk_addresses = []
        
        for h3_id in h3_cells:
            try:
                # Step 1: Administrative assignment
                # assignment_result = self.admin_assignment.assign_h3_cell(h3_id) # Old assign_h3_cell not available
                assignment_result = self.admin_assignment._assign_h3_cell_by_coverage(h3_id)
                
                # Step 2: Generate IDs
                admin_hierarchy = assignment_result['admin_assignment']
                admin_bounds = self._get_admin_bounds(admin_hierarchy)
                ids = self.id_generator.generate_dual_ids(h3_id, admin_hierarchy, admin_bounds)
                
                # Step 3: Combine results
                address_record = {
                    'h3_id': h3_id,
                    **ids,
                    'admin_assignment': admin_hierarchy,
                    'assignment_quality': assignment_result['assignment_quality'],
                    'geometry': assignment_result['geometry']
                }
                
                chunk_addresses.append(address_record)
                
            except Exception as e:
                logger.error(f"Failed to process H3 cell {h3_id}: {e}")
                # Add fallback record
                chunk_addresses.append(self._create_fallback_record(h3_id))
        
        return chunk_addresses
    
    def _get_admin_bounds(self, admin_hierarchy: Dict) -> Dict:
        """Get bounds for the assigned administrative unit."""
        # For now, use country bounds as fallback
        # In a full implementation, you'd calculate bounds for the specific admin unit
        return {
            'lat_range': [
                self.country_config['bounds']['min_lat'],
                self.country_config['bounds']['max_lat']
            ],
            'lng_range': [
                self.country_config['bounds']['min_lng'],
                self.country_config['bounds']['max_lng']
            ]
        }
    
    def _create_fallback_record(self, h3_id: str) -> Dict:
        """Create a fallback address record when processing fails."""
        try:
            lat, lng = h3.cell_to_latlng(h3_id)
            area_km2 = h3.cell_area(h3_id, unit='km^2')
        except:
            lat, lng, area_km2 = 0.0, 0.0, 0.0
        
        return {
            'h3_id': h3_id,
            'h3_derived_id': 'NG-XX-XX-XX-XXXX',
            'grid_position_id': 'NG-XX-XX-XX-000000',
            'primary_address_id': 'NG-XX-XX-XX-XXXX',
            'admin_assignment': {
                'country': {'code': 'NG', 'name': 'Nigeria'},
                'state': {'code': 'XX', 'name': 'Unknown'},
                'lga': {'code': 'XX', 'name': 'Unknown'},
                'ward': {'code': 'XX', 'name': 'Unknown'}
            },
            'assignment_quality': {
                'confidence_level': 'manual_review',
                'coverage_percentage': 0.0,
                'error': 'Processing failed'
            },
            'geometry': {
                'centroid': {'lat': lat, 'lng': lng},
                'area_km2': area_km2
            }
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the generated addresses."""
        if not self.address_data:
            return {'error': 'No address data available'}
        
        # Basic statistics
        total_cells = len(self.address_data)
        
        # Confidence distribution
        confidence_counts = {}
        coverage_percentages = []
        
        # Administrative distribution
        state_counts = {}
        lga_counts = {}
        ward_counts = {}
        
        for record in self.address_data:
            # Confidence
            confidence = record['assignment_quality']['confidence_level']
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            
            # Coverage
            coverage = record['assignment_quality']['coverage_percentage']
            coverage_percentages.append(coverage)
            
            # Administrative units
            admin = record['admin_assignment']
            state_code = admin['state']['code']
            lga_code = admin['lga']['code']
            ward_code = admin['ward']['code']
            
            state_counts[state_code] = state_counts.get(state_code, 0) + 1
            lga_counts[lga_code] = lga_counts.get(lga_code, 0) + 1
            ward_counts[ward_code] = ward_counts.get(ward_code, 0) + 1
        
        return {
            'total_cells': total_cells,
            'confidence_distribution': confidence_counts,
            'coverage_statistics': {
                'mean': np.mean(coverage_percentages),
                'median': np.median(coverage_percentages),
                'min': np.min(coverage_percentages),
                'max': np.max(coverage_percentages),
                'std': np.std(coverage_percentages)
            },
            'administrative_coverage': {
                'unique_states': len(state_counts),
                'unique_lgas': len(lga_counts),
                'unique_wards': len(ward_counts),
                'top_states': sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                'top_lgas': sorted(lga_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            'quality_metrics': {
                'confident_assignments': confidence_counts.get('confident', 0),
                'boundary_cases': confidence_counts.get('boundary_case', 0),
                'manual_review_needed': confidence_counts.get('manual_review', 0),
                'success_rate': (confidence_counts.get('confident', 0) / total_cells) * 100
            }
        }
    
    def save_to_file(self, output_path: str, format: str = 'parquet'):
        """Save generated addresses to file."""
        if not self.address_data:
            raise ValueError("No address data to save. Run generate_addresses() first.")
        
        logger.info(f"Saving {len(self.address_data)} addresses to {output_path}")
        
        if format.lower() == 'parquet':
            self._save_parquet(output_path)
        elif format.lower() == 'csv':
            self._save_csv(output_path)
        elif format.lower() == 'geojson':
            self._save_geojson(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_parquet(self, output_path: str):
        """Save as Parquet file."""
        import pandas as pd
        
        # Convert to DataFrame
        df = self._prepare_dataframe()
        
        # Save to Parquet
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Saved to Parquet: {output_path}")
    
    def _save_csv(self, output_path: str):
        """Save as CSV file."""
        import pandas as pd
        
        # Convert to DataFrame
        df = self._prepare_dataframe()
        
        # Save to CSV
        df.to_csv(output_path, index=False, compression='gzip')
        logger.info(f"Saved to CSV: {output_path}")
    
    def _save_geojson(self, output_path: str):
        """Save as GeoJSON file."""
        import pandas as pd
        
        # Convert to DataFrame with geometry
        df = self._prepare_dataframe(include_geometry=True)
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        
        # Save to GeoJSON
        gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"Saved to GeoJSON: {output_path}")
    
    def _prepare_dataframe(self, include_geometry: bool = False):
        """Prepare data for DataFrame conversion."""
        import pandas as pd
        
        records = []
        for record in self.address_data:
            flat_record = {
                'h3_id': record['h3_id'],
                'h3_derived_id': record['h3_derived_id'],
                'grid_position_id': record['grid_position_id'],
                'primary_address_id': record['primary_address_id'],
                
                # Administrative data
                'country_code': record['admin_assignment']['country']['code'],
                'country_name': record['admin_assignment']['country']['name'],
                'state_code': record['admin_assignment']['state']['code'],
                'state_name': record['admin_assignment']['state']['name'],
                'lga_code': record['admin_assignment']['lga']['code'],
                'lga_name': record['admin_assignment']['lga']['name'],
                'ward_code': record['admin_assignment']['ward']['code'],
                'ward_name': record['admin_assignment']['ward']['name'],
                
                # Quality metrics
                'confidence_level': record['assignment_quality']['confidence_level'],
                'coverage_percentage': record['assignment_quality']['coverage_percentage'],
                
                # Geometry
                'centroid_lat': record['geometry']['centroid']['lat'],
                'centroid_lng': record['geometry']['centroid']['lng'],
                'area_km2': record['geometry']['area_km2']
            }
            
            if include_geometry:
                # Add H3 cell geometry
                try:
                    h3_boundary = h3.cell_to_boundary(record['h3_id'])
                    flat_record['geometry'] = Polygon(h3_boundary)
                except:
                    flat_record['geometry'] = None
            
            records.append(flat_record)
        
        return pd.DataFrame(records)

    def _generate_addresses_parallel(self, h3_cells: List[str]) -> List[Dict]:
        """Generate addresses using optimized parallel processing."""
        logger.info("Using optimized parallel processing for address generation...")
        
        # Optimize chunk size based on dataset size and CPU count
        num_workers = min(self.processing_config.get('num_workers', os.cpu_count()), os.cpu_count())
        optimal_chunk_size = max(100, len(h3_cells) // (num_workers * 4))  # 4x chunks per worker
        
        # Create larger chunks to reduce overhead
        chunks = [h3_cells[i:i + optimal_chunk_size] for i in range(0, len(h3_cells), optimal_chunk_size)]
        
        logger.info(f"Processing {len(h3_cells)} cells in {len(chunks)} chunks using {num_workers} workers")
        
        all_addresses = []
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers, 
                                initializer=_init_worker,
                                initargs=(self.admin_assignment, self.id_generator, self.country_config)) as executor:
                
                # Submit all chunks at once (no incremental submission overhead)
                futures = [executor.submit(_process_chunk_optimized, chunk) for chunk in chunks]
                
                # Collect results without progress bar to reduce overhead
                for future in as_completed(futures):
                    try:
                        chunk_addresses = future.result(timeout=300)  # 5 min timeout per chunk
                        all_addresses.extend(chunk_addresses)
                    except Exception as e:
                        logger.error(f"Failed to process chunk: {e}")
                        
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            logger.info("Falling back to sequential processing...")
            return self._generate_addresses_sequential(h3_cells)
        
        self.address_data = all_addresses
        logger.info(f"Generated {len(all_addresses)} address records using optimized parallel processing")
        
        return all_addresses


    # Global worker state (initialized once per process)
    _worker_admin_assignment = None
    _worker_id_generator = None  
    _worker_country_config = None
    _worker_cache = {}  # Per-process cache


    def _init_worker(admin_assignment, id_generator, country_config):
        """Initialize worker process with shared data."""
        global _worker_admin_assignment, _worker_id_generator, _worker_country_config, _worker_cache
        _worker_admin_assignment = admin_assignment
        _worker_id_generator = id_generator
        _worker_country_config = country_config
        _worker_cache = {}


    def _process_chunk_optimized(h3_cells: List[str]) -> List[Dict]:
        """
        Optimized worker function with reduced function calls and caching.
        """
        global _worker_admin_assignment, _worker_id_generator, _worker_country_config, _worker_cache
        
        chunk_addresses = []
        admin_bounds_cache = {}  # Cache admin bounds within chunk
        
        # Pre-calculate common values
        country_bounds = _worker_country_config['bounds']
        default_admin_bounds = {
            'lat_range': [country_bounds['min_lat'], country_bounds['max_lat']],
            'lng_range': [country_bounds['min_lng'], country_bounds['max_lng']]
        }
        
        for h3_id in h3_cells:
            try:
                # Check cache first
                if h3_id in _worker_cache:
                    chunk_addresses.append(_worker_cache[h3_id])
                    continue
                
                # Step 1: Administrative assignment (biggest bottleneck)
                assignment_result = _worker_admin_assignment._assign_h3_cell_by_coverage(h3_id)
                admin_hierarchy = assignment_result['admin_assignment']
                
                # Step 2: Cache and reuse admin bounds
                admin_key = f"{admin_hierarchy['state']['code']}-{admin_hierarchy['lga']['code']}-{admin_hierarchy['ward']['code']}"
                if admin_key not in admin_bounds_cache:
                    admin_bounds_cache[admin_key] = default_admin_bounds
                admin_bounds = admin_bounds_cache[admin_key]
                
                # Step 3: Generate IDs
                ids = _worker_id_generator.generate_dual_ids(h3_id, admin_hierarchy, admin_bounds)
                
                # Step 4: Combine results
                address_record = {
                    'h3_id': h3_id,
                    **ids,
                    'admin_assignment': admin_hierarchy,
                    'assignment_quality': assignment_result['assignment_quality'],
                    'geometry': assignment_result['geometry']
                }
                
                # Cache result for potential reuse
                if len(_worker_cache) < 1000:  # Limit cache size
                    _worker_cache[h3_id] = address_record
                
                chunk_addresses.append(address_record)
                
            except Exception as e:
                # Minimize logging in worker processes
                fallback_record = _create_fallback_record_fast(h3_id)
                chunk_addresses.append(fallback_record)
        
        return chunk_addresses


    def _create_fallback_record_fast(h3_id: str) -> Dict:
        """Fast fallback record creation with minimal H3 calls."""
        try:
            # Single H3 call for both lat/lng and area
            lat, lng = h3.cell_to_latlng(h3_id)
            area_km2 = h3.cell_area(h3_id, unit='km^2')
        except:
            lat, lng, area_km2 = 9.0, 8.0, 0.0  # Nigeria center as fallback
        
        return {
            'h3_id': h3_id,
            'h3_derived_id': 'NG-XX-XX-XX-XXXX',
            'grid_position_id': 'NG-XX-XX-XX-000000', 
            'primary_address_id': 'NG-XX-XX-XX-XXXX',
            'admin_assignment': {
                'country': {'code': 'NG', 'name': 'Nigeria'},
                'state': {'code': 'XX', 'name': 'Unknown'},
                'lga': {'code': 'XX', 'name': 'Unknown'},
                'ward': {'code': 'XX', 'name': 'Unknown'}
            },
            'assignment_quality': {
                'confidence_level': 'manual_review',
                'coverage_percentage': 0.0,
                'error': 'Processing failed'
            },
            'geometry': {
                'centroid': {'lat': lat, 'lng': lng},
                'area_km2': area_km2
            }
        }


    # Alternative: Memory-mapped approach for very large datasets
    def _generate_addresses_memory_mapped(self, h3_cells: List[str]) -> List[Dict]:
        """
        Alternative implementation using shared memory for massive datasets.
        Use this for 100k+ cells.
        """
        try:
            from multiprocessing import shared_memory
            import numpy as np
            
            logger.info("Using memory-mapped parallel processing...")
            
            # Convert H3 IDs to integers for efficient shared memory
            h3_ints = [int(h3_id, 16) for h3_id in h3_cells]
            h3_array = np.array(h3_ints, dtype=np.uint64)
            
            # Create shared memory
            shm = shared_memory.SharedMemory(create=True, size=h3_array.nbytes)
            shared_array = np.ndarray(h3_array.shape, dtype=h3_array.dtype, buffer=shm.buf)
            shared_array[:] = h3_array[:]
            
            # Process in parallel with shared memory
            num_workers = min(os.cpu_count(), 12)
            chunk_size = len(h3_cells) // num_workers
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i in range(0, len(h3_cells), chunk_size):
                    end_idx = min(i + chunk_size, len(h3_cells))
                    futures.append(
                        executor.submit(_process_shared_chunk, shm.name, i, end_idx)
                    )
                
                all_addresses = []
                for future in as_completed(futures):
                    chunk_addresses = future.result()
                    all_addresses.extend(chunk_addresses)
            
            # Cleanup
            shm.close()
            shm.unlink()
            
            return all_addresses
            
        except ImportError:
            logger.warning("Shared memory not available, falling back to standard parallel processing")
            return self._generate_addresses_parallel(h3_cells)


    def _process_shared_chunk(shm_name: str, start_idx: int, end_idx: int) -> List[Dict]:
        """Process chunk using shared memory."""
        from multiprocessing import shared_memory
        import numpy as np
        
        # Access shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray((end_idx - start_idx,), dtype=np.uint64, buffer=shm.buf)
        
        # Convert back to H3 IDs and process
        chunk_addresses = []
        for i in range(start_idx, end_idx):
            h3_id = format(shared_array[i - start_idx], 'x')
            # Process h3_id...
            # (implementation similar to _process_chunk_optimized)
        
        shm.close()
        return chunk_addresses



# ------------------------------------------------------------------------ #
def prepare_h3_address_as_dataframe(address_data, include_geometry: bool = False):
    """Prepare data for DataFrame conversion."""
    import pandas as pd
    from shapely import Polygon
    import h3
    
    records = []
    for record in address_data:
        flat_record = {
            'h3_id': record['h3_id'],
            'h3_derived_id': record['h3_derived_id'],
            'grid_position_id': record['grid_position_id'],
            'primary_address_id': record['primary_address_id'],
            
            # Administrative data
            'country_code': record['admin_assignment']['country']['code'],
            'country_name': record['admin_assignment']['country']['name'],
            'state_code': record['admin_assignment']['state']['code'],
            'state_name': record['admin_assignment']['state']['name'],
            'lga_code': record['admin_assignment']['lga']['code'],
            'lga_name': record['admin_assignment']['lga']['name'],
            'ward_code': record['admin_assignment']['ward']['code'],
            'ward_name': record['admin_assignment']['ward']['name'],
            
            # Quality metrics
            'confidence_level': record['assignment_quality']['confidence_level'],
            'coverage_percentage': record['assignment_quality']['coverage_percentage'],
            
            # Geometry
            'centroid_lat': record['geometry']['centroid']['lat'],
            'centroid_lng': record['geometry']['centroid']['lng'],
            'area_km2': record['geometry']['area_km2']
        }
        
        if include_geometry:
            # Add H3 cell geometry
            try:
                h3_boundary = h3.cell_to_boundary(record['h3_id'])
                flat_record['geometry'] = Polygon(h3_boundary)
            except:
                flat_record['geometry'] = None
        
        records.append(flat_record)
    
    return pd.DataFrame(records)

    
def generate_sample_addresses():
    """Generate a small sample of addresses for testing."""
    generator = H3AddressGenerator()
    
    # Generate a few H3 cells in Lagos area
    sample_h3_ids = [
        "8c1234567890abc",
        "8c1234567890abd", 
        "8c1234567890abe"
    ]
    
    # Mock administrative assignment
    generator.admin_assignment.admin_units = {
        'states': [
            {
                'name': 'Lagos',
                'code': 'LA',
                'geometry': Polygon([(3.0, 6.0), (4.0, 6.0), (4.0, 7.0), (3.0, 7.0)]),
                'level': 'state'
            }
        ],
        'lgas': [
            {
                'name': 'Ikeja',
                'code': 'IK',
                'state_code': 'LA',
                'geometry': Polygon([(3.0, 6.0), (3.5, 6.0), (3.5, 6.5), (3.0, 6.5)]),
                'level': 'lga'
            }
        ],
        'wards': [
            {
                'name': 'Ward A',
                'code': 'WA',
                'lga_code': 'IK',
                'state_code': 'LA',
                'geometry': Polygon([(3.0, 6.0), (3.25, 6.0), (3.25, 6.25), (3.0, 6.25)]),
                'level': 'ward'
            }
        ]
    }
    
    addresses = generator.generate_addresses(sample_h3_ids)
    return addresses



if __name__ == "__main__":
    # Test the generator
    sample_addresses = generate_sample_addresses()
    
    print("Sample Addresses Generated:")
    for i, addr in enumerate(sample_addresses):
        print(f"\nAddress {i+1}:")
        print(f"  H3 ID: {addr['h3_id']}")
        print(f"  Primary ID: {addr['primary_address_id']}")
        print(f"  State: {addr['admin_assignment']['state']['name']}")
        print(f"  LGA: {addr['admin_assignment']['lga']['name']}")
        print(f"  Ward: {addr['admin_assignment']['ward']['name']}")
        print(f"  Confidence: {addr['assignment_quality']['confidence_level']}")
    
    # Get statistics
    generator = H3AddressGenerator()
    generator.address_data = sample_addresses
    stats = generator.get_statistics()
    
    print(f"\nStatistics:")
    print(f"  Total cells: {stats['total_cells']}")
    print(f"  Success rate: {stats['quality_metrics']['success_rate']:.1f}%") 