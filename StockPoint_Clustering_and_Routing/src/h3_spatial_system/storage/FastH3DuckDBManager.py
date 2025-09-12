import duckdb
import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union 
import time 
from datetime import datetime
import logging
from config.settings import STORAGE_CONFIG 
    
    
class FastH3DuckDBManager:
    """
    High-performance H3DuckDBManager optimized for speed while maintaining data integrity.
    """
    
    def __init__(self, db_path: str, resolution: int, batch_size: int = 50000):
        """
        Initialize with performance optimizations.
        
        Args:
            db_path: Path to DuckDB database file
            resolution: H3 resolution level
            batch_size: Larger batches = better performance (50K+ recommended)
        """
        self.db_path = os.path.abspath(db_path)
        self.batch_size = batch_size
        self.batch_data = []
        self.total_saved = 0
        self.resolution = resolution
        self.start_time = time.time()
        
        print(f"🚀 Initializing FastH3DuckDBManager (batch_size: {batch_size:,})")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize DuckDB connection with performance settings
        self.conn = duckdb.connect(self.db_path)
        # self._create_optimized_h3_cells_address_tables()
        self._apply_performance_settings()
        
        print("✅ Fast H3DuckDBManager ready")
    
    def _create_optimized_h3_cells_address_tables_(self):
        """Create table optimized for bulk inserts and storage efficiency."""
        print("📋 Creating optimized h3_cells table...")
        
        # Use more efficient data types and structure
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS h3_cells (
                h3_index VARCHAR PRIMARY KEY,
                resolution TINYINT NOT NULL,
                centroid_lat DOUBLE,
                centroid_lng DOUBLE,
                polygon_wkt VARCHAR,           -- WKT format for geometry
                boundary_json VARCHAR,         -- JSON for boundary coords
                latlng_json VARCHAR,          -- JSON for lat/lng coords  
                polygon_area DOUBLE,
                num_vertices SMALLINT,
                error VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        print("✅ Optimized table structure created")
    
    def _add_address_columns_(self):
        """
        Add additional metadata columns to the h3_cells table.
        Columns are added only if they don't already exist.
        """
        columns_to_add = [
            ("h3_derived_id", "VARCHAR"),
            ("grid_position_id", "VARCHAR"),
            ("primary_address_id", "VARCHAR"),
            ("country_code", "VARCHAR"),
            ("country_name", "VARCHAR"),
            ("state_code", "VARCHAR"),
            ("state_name", "VARCHAR"),
            ("lga_code", "VARCHAR"),
            ("lga_name", "VARCHAR"),
            ("ward_code", "VARCHAR"),
            ("ward_name", "VARCHAR"),
            ("confidence_level", "VARCHAR"),
            ("coverage_percentage", "DOUBLE"),
            ("area_km2", "DOUBLE")
        ]
        
        for column_name, column_type in columns_to_add:
            try:
                self.conn.execute(
                    f"ALTER TABLE h3_cells ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
                )
            except Exception as e:
                print(f"Error adding column {column_name}: {e}")
                # Continue with other columns rather than raising exception
                continue
        
        print("Completed adding metadata columns to h3_cells table")
    
    def update_address_data_(self, address_data: Dict[str, Dict[str, Any]], batch_size: int = 10000):
        """
        Update address metadata for H3 cells in bulk.
        
        Args:
            address_data: Dictionary with h3_index as keys and address metadata as values
                         e.g., {
                             'h3_index_1': {
                                 'h3_derived_id': 'id_123',
                                 'country_code': 'NG',
                                 'country_name': 'Nigeria',
                                 'state_code': 'LA',
                                 'state_name': 'Lagos',
                                 'coverage_percentage': 95.5,
                                 'area_km2': 10.2,
                                 ...
                             }
                         }
            batch_size: Number of records to process in each batch
        """
        if not address_data:
            print("⚠️  No address data provided")
            return
        
        table_exists = self.conn.execute("""
                SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'h3_cells'
            """).fetchone()[0] > 0

        if not table_exists:
            self._create_optimized_h3_cells_address_tables()
            print("✅ Optimized h3 cells address table structure created") 

        print(f"🏠 Updating address data for {len(address_data):,} H3 cells...")
        
        # Ensure address columns exist
        self._add_address_columns()
        
        # Process in batches for better performance
        h3_indices = list(address_data.keys())
        total_updated = 0
        
        for i in range(0, len(h3_indices), batch_size):
            batch_indices = h3_indices[i:i + batch_size]
            batch_data = []
            
            for h3_index in batch_indices:
                metadata = address_data[h3_index]
                
                # Prepare update data with safe defaults
                update_item = {
                    'h3_index': h3_index,
                    'h3_derived_id': metadata.get('h3_derived_id'),
                    'grid_position_id': metadata.get('grid_position_id'),
                    'primary_address_id': metadata.get('primary_address_id'),
                    'country_code': metadata.get('country_code'),
                    'country_name': metadata.get('country_name'),
                    'state_code': metadata.get('state_code'),
                    'state_name': metadata.get('state_name'),
                    'lga_code': metadata.get('lga_code'),
                    'lga_name': metadata.get('lga_name'),
                    'ward_code': metadata.get('ward_code'),
                    'ward_name': metadata.get('ward_name'),
                    'confidence_level': metadata.get('confidence_level'),
                    'coverage_percentage': metadata.get('coverage_percentage'),
                    'area_km2': metadata.get('area_km2')
                }
                
                batch_data.append(update_item)
            
            # Bulk update using pandas DataFrame
            try:
                df = pd.DataFrame(batch_data)
                self.conn.register('address_update_df', df)
                
                # Update existing records with address data
                updated_count = self.conn.execute("""
                    UPDATE h3_cells 
                    SET 
                        h3_derived_id = address_update_df.h3_derived_id,
                        grid_position_id = address_update_df.grid_position_id,
                        primary_address_id = address_update_df.primary_address_id,
                        country_code = address_update_df.country_code,
                        country_name = address_update_df.country_name,
                        state_code = address_update_df.state_code,
                        state_name = address_update_df.state_name,
                        lga_code = address_update_df.lga_code,
                        lga_name = address_update_df.lga_name,
                        ward_code = address_update_df.ward_code,
                        ward_name = address_update_df.ward_name,
                        confidence_level = address_update_df.confidence_level,
                        coverage_percentage = address_update_df.coverage_percentage,
                        area_km2 = address_update_df.area_km2
                    FROM address_update_df
                    WHERE h3_cells.h3_index = address_update_df.h3_index
                """).fetchone()
                
                self.conn.unregister('address_update_df')
                total_updated += len(batch_data)
                
                print(f"✅ Updated batch {i//batch_size + 1}: {len(batch_data):,} records (total: {total_updated:,})")
                
            except Exception as e:
                print(f"❌ Error updating batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"🎉 Address data update completed: {total_updated:,} records processed")
        
        # Optional: Force checkpoint after large update
        try:
            self.conn.execute("CHECKPOINT")
            print("💾 Changes committed to disk")
        except Exception as e:
            print(f"⚠️  Checkpoint warning: {e}")
    
    def upsert_address_data_(self, address_data: Dict[str, Dict[str, Any]], batch_size: int = 10000):
        """
        Insert or update address metadata for H3 cells (UPSERT operation).
        This will insert new records or update existing ones.
        
        Args:
            address_data: Dictionary with h3_index as keys and complete address metadata
        """
        if not address_data:
            print("⚠️  No address data provided")
            return
        
        print(f"🏠 Upserting address data for {len(address_data):,} H3 cells...")
        
        # Ensure address columns exist
        self._add_address_columns()
        
        # Process in batches
        h3_indices = list(address_data.keys())
        total_processed = 0
        
        for i in range(0, len(h3_indices), batch_size):
            batch_indices = h3_indices[i:i + batch_size]
            batch_data = []
            
            for h3_index in batch_indices:
                metadata = address_data[h3_index]
                
                # Prepare complete record for UPSERT
                upsert_item = {
                    'h3_index': h3_index,
                    'resolution': self.resolution,
                    'centroid_lat': metadata.get('centroid_lat'),
                    'centroid_lng': metadata.get('centroid_lng'),
                    'polygon_wkt': metadata.get('polygon_wkt'),
                    'boundary_json': metadata.get('boundary_json'),
                    'latlng_json': metadata.get('latlng_json'),
                    'polygon_area': metadata.get('polygon_area'),
                    'num_vertices': metadata.get('num_vertices'),
                    'error': metadata.get('error'),
                    
                    # Address fields
                    'h3_derived_id': metadata.get('h3_derived_id'),
                    'grid_position_id': metadata.get('grid_position_id'),
                    'primary_address_id': metadata.get('primary_address_id'),
                    'country_code': metadata.get('country_code'),
                    'country_name': metadata.get('country_name'),
                    'state_code': metadata.get('state_code'),
                    'state_name': metadata.get('state_name'),
                    'lga_code': metadata.get('lga_code'),
                    'lga_name': metadata.get('lga_name'),
                    'ward_code': metadata.get('ward_code'),
                    'ward_name': metadata.get('ward_name'),
                    'confidence_level': metadata.get('confidence_level'),
                    'coverage_percentage': metadata.get('coverage_percentage'),
                    'area_km2': metadata.get('area_km2')
                }
                
                batch_data.append(upsert_item)
            
            try:
                df = pd.DataFrame(batch_data)
                self.conn.register('upsert_df', df)
                
                # Use INSERT OR REPLACE for UPSERT
                self.conn.execute("""
                    INSERT OR REPLACE INTO h3_cells (
                        h3_index, resolution, centroid_lat, centroid_lng,
                        polygon_wkt, boundary_json, latlng_json,
                        polygon_area, num_vertices, error,
                        h3_derived_id, grid_position_id, primary_address_id,
                        country_code, country_name, state_code, state_name,
                        lga_code, lga_name, ward_code, ward_name,
                        confidence_level, coverage_percentage, area_km2
                    )
                    SELECT 
                        h3_index, resolution, centroid_lat, centroid_lng,
                        polygon_wkt, boundary_json, latlng_json,
                        polygon_area, num_vertices, error,
                        h3_derived_id, grid_position_id, primary_address_id,
                        country_code, country_name, state_code, state_name,
                        lga_code, lga_name, ward_code, ward_name,
                        confidence_level, coverage_percentage, area_km2
                    FROM upsert_df
                """)
                
                self.conn.unregister('upsert_df')
                total_processed += len(batch_data)
                
                print(f"✅ Upserted batch {i//batch_size + 1}: {len(batch_data):,} records (total: {total_processed:,})")
                
            except Exception as e:
                print(f"❌ Error upserting batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"🎉 Address data upsert completed: {total_processed:,} records processed")
    
    def _apply_performance_settings(self):
        """Apply DuckDB-specific performance optimizations."""
        performance_settings = [
            # Memory optimizations (fixed syntax)
            "SET memory_limit='4GB'",
            "SET threads=8",
            
            # Write optimizations (corrected settings)
            "SET checkpoint_threshold='2GB'",
            "SET wal_autocheckpoint='50MB'",
            
            # DuckDB-specific optimizations
            "SET preserve_insertion_order=false",
            "SET force_compression='auto'",
            "SET temp_directory='/tmp'",
            
            # # Additional valid settings
            # "SET enable_profiling=false",
            "SET enable_progress_bar=false",
        ]
        
        applied = 0
        for setting in performance_settings:
            try:
                self.conn.execute(setting)
                applied += 1
            except Exception as e:
                print(f"⚠️ Setting failed: {setting} - {e}")
        
        print(f"⚡ Applied {applied}/{len(performance_settings)} performance optimizations")
        
        # Fallback to minimal settings if most failed
        if applied < 3:
            print("🔄 Trying minimal safe performance settings...")
            safe_settings = [
                "SET memory_limit='2GB'",
                "SET threads=4",
            ]
            for setting in safe_settings:
                try:
                    self.conn.execute(setting)
                    print(f"✅ Applied safe setting: {setting}")
                except Exception as e:
                    print(f"❌ Even safe setting failed: {setting} - {e}")
                    
    def add_result(self, h3_index: str, result: dict):
        """Optimized add_result with minimal processing overhead."""
        try:
            if 'error' in result:
                batch_item = {
                    'h3_index': h3_index,
                    'resolution': self.resolution,
                    'centroid_lat': None,
                    'centroid_lng': None,
                    'polygon_wkt': None,
                    'boundary_json': None,
                    'latlng_json': None,
                    'polygon_area': None,
                    'num_vertices': None,
                    'error': result['error']
                }
            else:
                # Efficient data conversion
                lat, lng = result['centroid']
                polygon = result.get('polygon')
                
                batch_item = {
                    'h3_index': h3_index,
                    'resolution': self.resolution,
                    'centroid_lat': lat,
                    'centroid_lng': lng,
                    'polygon_wkt': polygon.wkt if polygon else None,
                    'boundary_json': json.dumps(result.get('boundary', [])),
                    'latlng_json': json.dumps(result.get('latlng_coords', [])),
                    'polygon_area': polygon.area if polygon else None,
                    'num_vertices': len(list(polygon.exterior.coords)) if polygon else None,
                    'error': None
                }
            
            self.batch_data.append(batch_item)
            
            # Save batch when limit reached
            if len(self.batch_data) >= self.batch_size:
                self._save_batch_fast()
                
        except Exception as e:
            # Quick error handling - don't slow down the process
            self.batch_data.append({
                'h3_index': h3_index, 'resolution': self.resolution,
                'centroid_lat': None, 'centroid_lng': None,
                'polygon_wkt': None, 'boundary_json': None, 'latlng_json': None,
                'polygon_area': None, 'num_vertices': None,
                'error': f"Processing error: {str(e)[:100]}"  # Truncate long errors
            })
    
    def _save_batch_fast(self):
        """Ultra-fast batch saving using pandas and bulk operations."""
        if not self.batch_data:
            return
        
        batch_size = len(self.batch_data)
        start_time = time.time()
        
        try:
            # Convert to pandas DataFrame (fastest bulk insert method)
            df = pd.DataFrame(self.batch_data)
            
            # Use DuckDB's pandas integration for maximum speed
            self.conn.register('batch_df', df)
            
            # Single bulk INSERT operation with explicit column mapping
            self.conn.execute("""
                INSERT OR REPLACE INTO h3_cells (
                    h3_index, resolution, centroid_lat, centroid_lng,
                    polygon_wkt, boundary_json, latlng_json,
                    polygon_area, num_vertices, error
                )
                SELECT 
                    h3_index, resolution, centroid_lat, centroid_lng,
                    polygon_wkt, boundary_json, latlng_json,
                    polygon_area, num_vertices, error
                FROM batch_df
            """)
            
            # Unregister the temporary view
            self.conn.unregister('batch_df')
            
            self.total_saved += batch_size
            elapsed = time.time() - start_time
            rate = batch_size / elapsed if elapsed > 0 else 0
            total_elapsed = time.time() - self.start_time
            overall_rate = self.total_saved / total_elapsed if total_elapsed > 0 else 0
            
            print(f"⚡ Saved {batch_size:,} items in {elapsed:.2f}s ({rate:,.0f}/s) | "
                  f"Total: {self.total_saved:,} ({overall_rate:,.0f}/s avg)")
            
        except Exception as e:
            print(f"❌ Fast batch save failed: {e}")
            # Fallback to slower individual inserts
            self._save_batch_fallback()
        finally:
            self.batch_data.clear()
    
    def _save_batch_fallback(self):
        """Fallback method if pandas integration fails."""
        print("🔄 Using fallback save method...")
        
        try:
            # Prepare data for executemany
            insert_data = []
            for item in self.batch_data:
                insert_data.append([
                    item['h3_index'], item['resolution'], item['centroid_lat'], item['centroid_lng'],
                    item['polygon_wkt'], item['boundary_json'], item['latlng_json'],
                    item['polygon_area'], item['num_vertices'], item['error']
                ])
            
            # Use executemany for batch insert (faster than individual inserts)
            self.conn.executemany("""
                INSERT OR REPLACE INTO h3_cells (
                    h3_index, resolution, centroid_lat, centroid_lng,
                    polygon_wkt, boundary_json, latlng_json,
                    polygon_area, num_vertices, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, insert_data)
            
            print(f"✅ Fallback save completed: {len(self.batch_data)} items")
            
        except Exception as e:
            print(f"❌ Fallback save also failed: {e}")
            raise
    
    def add_batch_bulk(self, results_dict: Dict[str, Dict[str, Any]]):
        """
        Super-fast bulk add for when you have many results at once.
        This is the fastest method - use when possible.
        """
        if not results_dict:
            return
        
        print(f"🚀 Bulk processing {len(results_dict):,} results...")
        start_time = time.time()
        
        # Process all results into DataFrame directly
        batch_data = []
        for h3_index, result in results_dict.items():
            try:
                if 'error' in result:
                    batch_data.append({
                        'h3_index': h3_index,
                        'resolution': self.resolution,
                        'centroid_lat': None, 'centroid_lng': None,
                        'polygon_wkt': None, 'boundary_json': None, 'latlng_json': None,
                        'polygon_area': None, 'num_vertices': None,
                        'error': result['error']
                    })
                else:
                    lat, lng = result['centroid']
                    polygon = result.get('polygon')
                    batch_data.append({
                        'h3_index': h3_index,
                        'resolution': self.resolution,
                        'centroid_lat': lat, 'centroid_lng': lng,
                        'polygon_wkt': polygon.wkt if polygon else None,
                        'boundary_json': json.dumps(result.get('boundary', [])),
                        'latlng_json': json.dumps(result.get('latlng_coords', [])),
                        'polygon_area': polygon.area if polygon else None,
                        'num_vertices': len(list(polygon.exterior.coords)) if polygon else None,
                        'error': None
                    })
            except Exception as e:
                batch_data.append({
                    'h3_index': h3_index, 'resolution': self.resolution,
                    'centroid_lat': None, 'centroid_lng': None,
                    'polygon_wkt': None, 'boundary_json': None, 'latlng_json': None,
                    'polygon_area': None, 'num_vertices': None,
                    'error': f"Bulk processing error: {str(e)[:100]}"
                })
        
        # Bulk insert with explicit column mapping
        df = pd.DataFrame(batch_data)
        self.conn.register('bulk_df', df)
        self.conn.execute("""
            INSERT OR REPLACE INTO h3_cells (
                h3_index, resolution, centroid_lat, centroid_lng,
                polygon_wkt, boundary_json, latlng_json,
                polygon_area, num_vertices, error
            )
            SELECT 
                h3_index, resolution, centroid_lat, centroid_lng,
                polygon_wkt, boundary_json, latlng_json,
                polygon_area, num_vertices, error
            FROM bulk_df
        """)
        self.conn.unregister('bulk_df')
        
        self.total_saved += len(batch_data)
        elapsed = time.time() - start_time
        rate = len(batch_data) / elapsed if elapsed > 0 else 0
        
        print(f"⚡ Bulk insert completed: {len(batch_data):,} items in {elapsed:.2f}s ({rate:,.0f}/s)")
    
    def force_checkpoint(self):
        """Force a checkpoint - use sparingly as it's slow."""
        print("💾 Forcing checkpoint...")
        start_time = time.time()
        
        try:
            self.conn.execute("CHECKPOINT")
            elapsed = time.time() - start_time
            print(f"✅ Checkpoint completed in {elapsed:.2f}s")
        except Exception as e:
            print(f"❌ Checkpoint failed: {e}")
    
    def get_stats(self):
        """Quick performance statistics."""
        try:
            count = self.conn.execute("SELECT COUNT(*) FROM h3_cells").fetchone()[0]
            elapsed = time.time() - self.start_time
            rate = count / elapsed if elapsed > 0 else 0
            
            return {
                'total_records': count,
                'elapsed_time': elapsed,
                'average_rate': rate,
                'pending_batch': len(self.batch_data)
            }
        except:
            return {'error': 'Could not get stats'}
    
    def finalize(self):
        """Fast finalization with minimal overhead."""
        print("🏁 Finalizing...")
        
        # Save any remaining batch
        if self.batch_data:
            self._save_batch_fast()
        
        # Quick stats
        stats = self.get_stats()
        total_time = time.time() - self.start_time
        
        print(f"🎉 Completed: {stats.get('total_records', 0):,} records in {total_time:.1f}s")
        print(f"⚡ Average rate: {stats.get('average_rate', 0):,.0f} records/second")
        
        # Final checkpoint (optional - comment out for even faster finalization)
        # self.force_checkpoint()
    
    def close(self):
        """Fast close."""
        try:
            self.finalize()
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



    # ------------------------------------------------------------------------------------------------------
    # H3 Cells Address System
    # ------------------------------------------------------------------------------------------------------
    def _create_optimized_h3_cells_address_tables(self):
        """Create table optimized for bulk inserts and storage efficiency."""
        print("📋 Creating optimized h3_cells table...")
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS h3_cells (
                h3_index VARCHAR PRIMARY KEY,
                resolution TINYINT NOT NULL,
                centroid_lat DOUBLE,
                centroid_lng DOUBLE,
                polygon_wkt VARCHAR,
                boundary_json VARCHAR,
                latlng_json VARCHAR,
                polygon_area DOUBLE,
                num_vertices SMALLINT,
                error VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        print("✅ Optimized table structure created")

    def _add_address_columns(self):
        """Add metadata columns to h3_cells table if they don't exist."""
        columns_to_add = [
            ("h3_derived_id", "VARCHAR"), ("grid_position_id", "VARCHAR"),
            ("primary_address_id", "VARCHAR"), ("country_code", "VARCHAR"),
            ("country_name", "VARCHAR"), ("state_code", "VARCHAR"),
            ("state_name", "VARCHAR"), ("lga_code", "VARCHAR"),
            ("lga_name", "VARCHAR"), ("ward_code", "VARCHAR"),
            ("ward_name", "VARCHAR"), ("confidence_level", "VARCHAR"),
            ("coverage_percentage", "DOUBLE"), ("area_km2", "DOUBLE")
        ]
        
        for column_name, column_type in columns_to_add:
            try:
                self.conn.execute(
                    f"ALTER TABLE h3_cells ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
                )
            except Exception as e:
                print(f"Error adding column {column_name}: {e}")
        
        print("Completed adding metadata columns to h3_cells table")

    def _prepare_address_batch_data(self, address_data: Dict, h3_indices: list, include_geometry: bool = False):
        """Prepare batch data for address operations."""
        batch_data = []
        
        for h3_index in h3_indices:
            metadata = address_data[h3_index]
            
            item = {
                'h3_index': h3_index,
                'h3_derived_id': metadata.get('h3_derived_id'),
                'grid_position_id': metadata.get('grid_position_id'),
                'primary_address_id': metadata.get('primary_address_id'),
                'country_code': metadata.get('country_code'),
                'country_name': metadata.get('country_name'),
                'state_code': metadata.get('state_code'),
                'state_name': metadata.get('state_name'),
                'lga_code': metadata.get('lga_code'),
                'lga_name': metadata.get('lga_name'),
                'ward_code': metadata.get('ward_code'),
                'ward_name': metadata.get('ward_name'),
                'confidence_level': metadata.get('confidence_level'),
                'coverage_percentage': metadata.get('coverage_percentage'),
                'area_km2': metadata.get('area_km2')
            }
            
            if include_geometry:
                item.update({
                    'resolution': self.resolution,
                    'centroid_lat': metadata.get('centroid_lat'),
                    'centroid_lng': metadata.get('centroid_lng'),
                    'polygon_wkt': metadata.get('polygon_wkt'),
                    'boundary_json': metadata.get('boundary_json'),
                    'latlng_json': metadata.get('latlng_json'),
                    'polygon_area': metadata.get('polygon_area'),
                    'num_vertices': metadata.get('num_vertices'),
                    'error': metadata.get('error')
                })
            
            batch_data.append(item)
        
        return batch_data

    def update_address_data(self, address_data: Dict[str, Dict[str, Any]], batch_size: int = 10000):
        """Update address metadata for existing H3 cells."""
        if not address_data:
            print("⚠️ No address data provided")
            return
        
        self._ensure_table_exists()
        self._add_address_columns()
        
        print(f"🏠 Updating address data for {len(address_data):,} H3 cells...")
        
        h3_indices = list(address_data.keys())
        total_updated = 0
        
        for i in range(0, len(h3_indices), batch_size):
            batch_indices = h3_indices[i:i + batch_size]
            batch_data = self._prepare_address_batch_data(address_data, batch_indices)
            
            try:
                df = pd.DataFrame(batch_data)
                self.conn.register('address_update_df', df)
                
                self.conn.execute("""
                    UPDATE h3_cells 
                    SET h3_derived_id = address_update_df.h3_derived_id,
                        grid_position_id = address_update_df.grid_position_id,
                        primary_address_id = address_update_df.primary_address_id,
                        country_code = address_update_df.country_code,
                        country_name = address_update_df.country_name,
                        state_code = address_update_df.state_code,
                        state_name = address_update_df.state_name,
                        lga_code = address_update_df.lga_code,
                        lga_name = address_update_df.lga_name,
                        ward_code = address_update_df.ward_code,
                        ward_name = address_update_df.ward_name,
                        confidence_level = address_update_df.confidence_level,
                        coverage_percentage = address_update_df.coverage_percentage,
                        area_km2 = address_update_df.area_km2
                    FROM address_update_df
                    WHERE h3_cells.h3_index = address_update_df.h3_index
                """)
                
                self.conn.unregister('address_update_df')
                total_updated += len(batch_data)
                print(f"✅ Updated batch {i//batch_size + 1}: {len(batch_data):,} records")
                
            except Exception as e:
                print(f"❌ Error updating batch {i//batch_size + 1}: {e}")
        
        print(f"🎉 Address data update completed: {total_updated:,} records processed")

    def upsert_address_data__(self, address_data: Dict[str, Dict[str, Any]], batch_size: int = 10000):
        """Insert or update address metadata for H3 cells."""
        if not address_data:
            print("⚠️ No address data provided")
            return
        
        self._ensure_table_exists()
        self._add_address_columns()
        
        print(f"🏠 Upserting address data for {len(address_data):,} H3 cells...")
        
        h3_indices = list(address_data.keys())
        total_processed = 0
        
        for i in range(0, len(h3_indices), batch_size):
            batch_indices = h3_indices[i:i + batch_size]
            batch_data = self._prepare_address_batch_data(address_data, batch_indices, include_geometry=True)
            
            try:
                df = pd.DataFrame(batch_data)
                self.conn.register('upsert_df', df)
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO h3_cells (
                        h3_index, resolution, centroid_lat, centroid_lng,
                        polygon_wkt, boundary_json, latlng_json,
                        polygon_area, num_vertices, error,
                        h3_derived_id, grid_position_id, primary_address_id,
                        country_code, country_name, state_code, state_name,
                        lga_code, lga_name, ward_code, ward_name,
                        confidence_level, coverage_percentage, area_km2
                    )
                    SELECT * FROM upsert_df
                """)
                
                self.conn.unregister('upsert_df')
                total_processed += len(batch_data)
                print(f"✅ Upserted batch {i//batch_size + 1}: {len(batch_data):,} records")
                
            except Exception as e:
                print(f"❌ Error upserting batch {i//batch_size + 1}: {e}")
        
        print(f"🎉 Address data upsert completed: {total_processed:,} records processed")

    def upsert_address_data(
        self,
        address_data: Union[Dict[str, Dict[str, Any]], pd.DataFrame],
        batch_size: int = 10000
    ):
        """Insert or update address metadata for H3 cells.
        
        Parameters
        ----------
        address_data : dict or pd.DataFrame
            Either:
            - dict keyed by H3 index with metadata dicts as values
            - DataFrame with columns matching h3_cells table
        batch_size : int, optional
            Number of records per insert batch, by default 10000
        """
        if address_data is None or (isinstance(address_data, dict) and not address_data) or (
            isinstance(address_data, pd.DataFrame) and address_data.empty
        ):
            print("⚠️ No address data provided")
            return

        self._ensure_table_exists()
        self._add_address_columns()

        # Normalize input to DataFrame batches
        if isinstance(address_data, dict):
            h3_indices = list(address_data.keys())
            total_records = len(h3_indices)
            print(f"🏠 Upserting address data for {total_records:,} H3 cells (dict input)...")

            def get_batch(start, end):
                batch_indices = h3_indices[start:end]
                batch_data = self._prepare_address_batch_data(
                    address_data, batch_indices, include_geometry=True
                )
                return pd.DataFrame(batch_data)

        elif isinstance(address_data, pd.DataFrame):
            total_records = len(address_data)
            print(f"🏠 Upserting address data for {total_records:,} H3 cells (DataFrame input)...")

            def get_batch(start, end):
                return address_data.iloc[start:end].copy()

        else:
            raise TypeError("address_data must be a dict or pandas DataFrame")

        total_processed = 0
        num_batches = (total_records + batch_size - 1) // batch_size

        for batch_num, start in enumerate(range(0, total_records, batch_size), start=1):
            df = get_batch(start, start + batch_size)
            try:
                self.conn.register("upsert_df", df)
                self.conn.execute("""
                    INSERT OR REPLACE INTO h3_cells (
                        h3_index, resolution, centroid_lat, centroid_lng,
                        polygon_wkt, boundary_json, latlng_json,
                        polygon_area, num_vertices, error,
                        h3_derived_id, grid_position_id, primary_address_id,
                        country_code, country_name, state_code, state_name,
                        lga_code, lga_name, ward_code, ward_name,
                        confidence_level, coverage_percentage, area_km2
                    )
                    SELECT * FROM upsert_df
                """)
                total_processed += len(df)
                print(f"✅ Batch {batch_num}/{num_batches}: {len(df):,} records upserted")

            except Exception as e:
                first_idx = df.iloc[0].get("h3_index", "N/A") if not df.empty else "N/A"
                print(f"❌ Error in batch {batch_num} (starting H3 index: {first_idx}): {e}")
            finally:
                self.conn.unregister("upsert_df")

        print(f"🎉 Upsert complete: {total_processed:,}/{total_records:,} records processed")

    
    def _ensure_table_exists(self):
        """Ensure h3_cells table exists."""
        table_exists = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'h3_cells'
        """).fetchone()[0] > 0
        
        if not table_exists:
            self._create_optimized_h3_cells_address_tables()
               
    
    # ------------------------------------------------------------------------------------------------------
    # MFC/SP Coverage Cells
    # ------------------------------------------------------------------------------------------------------
    def _create_stockpoint_h3_coverage_table_(self):
        """Create stockpoint_h3_coverage table with proper schema."""
        print("Creating stockpoint_h3_coverage table...")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS stockpoint_h3_coverage_id_seq START 1;")
        self.conn.execute("""
            CREATE TABLE stockpoint_h3_coverage (
                id BIGINT PRIMARY KEY DEFAULT nextval('stockpoint_h3_coverage_id_seq'),
                stock_point_id BIGINT,
                h3_cell VARCHAR,
                h3_resolution INT,
                UNIQUE(stock_point_id, h3_cell, h3_resolution)
            );
        """)
        print("✅ stockpoint_h3_coverage table created")

    def _add_stockpoint_h3_coverage_columns_(self):
        """Ensure stockpoint_h3_coverage table exists and add missing columns."""
        table_exists = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'stockpoint_h3_coverage'
        """).fetchone()[0] > 0

        if not table_exists:
            self._create_stockpoint_h3_coverage_table()
            return

        # Add id column if missing
        id_col_exists = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'stockpoint_h3_coverage' AND column_name = 'id'
        """).fetchone()[0] > 0

        if not id_col_exists:
            self.conn.execute("ALTER TABLE stockpoint_h3_coverage ADD COLUMN id BIGINT;")
            print("⚠️ Added 'id' column - populate values separately")

        print("✅ stockpoint_h3_coverage schema ready")

    def upsert_stockpoint_h3_coverage_(self, df: pd.DataFrame, batch_size: int = 10000):
        """Insert/update stockpoint_h3_coverage with duplicate handling."""
        if df.empty:
            print("⚠️ No data provided")
            return
        
        print(f"📦 Upserting {len(df):,} stockpoint_h3_coverage rows...")
        self._add_stockpoint_h3_coverage_columns()
        
        total_processed = 0
        total_inserted = 0
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            try:
                self.conn.register('batch_df', batch_df)
                
                # Insert only new records (ignore duplicates)
                result = self.conn.execute("""
                    INSERT INTO stockpoint_h3_coverage (stock_point_id, h3_cell, h3_resolution)
                    SELECT stock_point_id, h3_cell, h3_resolution 
                    FROM batch_df
                    WHERE NOT EXISTS (
                        SELECT 1 FROM stockpoint_h3_coverage s 
                        WHERE s.stock_point_id = batch_df.stock_point_id 
                        AND s.h3_cell = batch_df.h3_cell 
                        AND s.h3_resolution = batch_df.h3_resolution
                    )
                """)
                
                self.conn.unregister('batch_df')
                
                # Get actual insert count
                inserted = len(batch_df)  # Approximate, could be refined
                total_inserted += inserted
                total_processed += len(batch_df)
                
                print(f"✅ Batch {i // batch_size + 1}: {inserted:,} new records")
                
            except Exception as e:
                print(f"❌ Error in batch {i // batch_size + 1}: {e}")
                try:
                    self.conn.unregister('batch_df')
                except:
                    pass
                continue
        
        print(f"🎉 Completed: {total_inserted:,} new rows from {total_processed:,} processed")
    
    # ----------------------------------------------
    def _create_stockpoint_h3_coverage_table(self):
        """Create enhanced stockpoint_h3_coverage table with cluster data."""
        print("Creating enhanced stockpoint_h3_coverage table...")
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS stockpoint_h3_coverage_enhanced_seq START 1;
            
            CREATE TABLE stockpoint_h3_coverage (
                id BIGINT PRIMARY KEY DEFAULT nextval('stockpoint_h3_coverage_enhanced_seq'),
                stock_point_id BIGINT,
                h3_cell VARCHAR,
                h3_resolution INT,
                cluster_centroid_lat DOUBLE PRECISION,
                cluster_centroid_lng DOUBLE PRECISION,
                cluster_sp_dist_km DOUBLE PRECISION,
                cluster_sp_direction VARCHAR,
                UNIQUE(stock_point_id, h3_cell, h3_resolution)
            );
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_stockpoint_h3_id ON stockpoint_h3_coverage(id);")
        print("✅ stockpoint_h3_coverage table created")

    def _add_stockpoint_h3_coverage_columns(self):
        """Ensure enhanced table exists and add missing columns."""
        table_exists = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'stockpoint_h3_coverage'
        """).fetchone()[0] > 0

        if not table_exists:
            self._create_stockpoint_h3_coverage_table()
            return

        existing_columns = {row[0] for row in self.conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'stockpoint_h3_coverage'
        """).fetchall()}
        
        required_columns = {
            'cluster_centroid_lat': 'DOUBLE PRECISION',
            'cluster_centroid_lng': 'DOUBLE PRECISION', 
            'cluster_sp_dist_km': 'DOUBLE PRECISION',
            'cluster_sp_direction': 'VARCHAR',
        }
        
        for col, dtype in required_columns.items():
            if col not in existing_columns:
                self.conn.execute(f"ALTER TABLE stockpoint_h3_coverage ADD COLUMN {col} {dtype};")
                print(f"⚠️ Added '{col}' column")

        print("✅ stockpoint_h3_coverage schema ready")

    def upsert_stockpoint_h3_coverage(self, df: pd.DataFrame, batch_size: int = 10000):
        """Insert/update enhanced stockpoint_h3_coverage with cluster data."""
        if df.empty:
            print("⚠️ No data provided")
            return
        
        print(f"📦 Upserting {len(df):,} enhanced stockpoint_h3_coverage rows...")
        self._add_stockpoint_h3_coverage_columns()
        
        required_cols = ['stock_point_id', 'h3_cell', 'h3_resolution', 'cluster_centroid_lat', 
                        'cluster_centroid_lng', 'cluster_sp_dist_km']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        total_processed = 0
        total_upserted = 0
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            try:
                self.conn.register('batch_df', batch_df)
                
                self.conn.execute("""
                    INSERT INTO stockpoint_h3_coverage 
                    (stock_point_id, h3_cell, h3_resolution, cluster_centroid_lat, 
                    cluster_centroid_lng, cluster_sp_dist_km)
                    SELECT stock_point_id, h3_cell, h3_resolution, cluster_centroid_lat,
                        cluster_centroid_lng, cluster_sp_dist_km
                    FROM batch_df
                    ON CONFLICT (stock_point_id, h3_cell, h3_resolution) 
                    DO UPDATE SET
                        cluster_centroid_lat = EXCLUDED.cluster_centroid_lat,
                        cluster_centroid_lng = EXCLUDED.cluster_centroid_lng,
                        cluster_sp_dist_km = EXCLUDED.cluster_sp_dist_km
                """)
                
                self.conn.unregister('batch_df')
                total_upserted += len(batch_df)
                total_processed += len(batch_df)
                print(f"✅ Batch {i // batch_size + 1}: {len(batch_df):,} records upserted")
                
            except Exception as e:
                print(f"❌ Error in batch {i // batch_size + 1}: {e}")
                try:
                    self.conn.unregister('batch_df')
                except:
                    pass
                continue
        
        print(f"🎉 Completed: {total_upserted:,} records upserted from {total_processed:,} processed")

    def _create_coverage_log_table(self):
        """Create log table to track changes."""
        
        # Create sequence for auto-increment ID
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS stockpoint_h3_coverage_log_seq START 1;
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stockpoint_h3_coverage_log ( 
                log_id BIGINT PRIMARY KEY DEFAULT (nextval('stockpoint_h3_coverage_log_seq')),
                stock_point_id BIGINT,
                h3_cell VARCHAR,
                h3_resolution INT,
                operation VARCHAR(20),
                old_cluster_centroid_lat DOUBLE PRECISION,
                old_cluster_centroid_lng DOUBLE PRECISION,
                old_cluster_sp_dist_km DOUBLE PRECISION,
                new_cluster_centroid_lat DOUBLE PRECISION,
                new_cluster_centroid_lng DOUBLE PRECISION,
                new_cluster_sp_dist_km DOUBLE PRECISION,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_coverage_log_keys ON stockpoint_h3_coverage_log(stock_point_id, h3_cell, h3_resolution);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_coverage_log_timestamp ON stockpoint_h3_coverage_log(changed_at);")

    def log_coverage_changes(self, new_df: pd.DataFrame):
        """Log only actual changes (inserts/updates/deletes)."""
        self._create_coverage_log_table()
        self.conn.register('new_data', new_df)
        
        total_changes = 0
        
        # Log deletions
        result = self.conn.execute("""
            INSERT INTO stockpoint_h3_coverage_log 
            (stock_point_id, h3_cell, h3_resolution, operation,
            old_cluster_centroid_lat, old_cluster_centroid_lng, old_cluster_sp_dist_km)
            SELECT 
                o.stock_point_id, o.h3_cell, o.h3_resolution, 'DELETE',
                o.cluster_centroid_lat, o.cluster_centroid_lng, o.cluster_sp_dist_km
            FROM stockpoint_h3_coverage o
            WHERE NOT EXISTS (
                SELECT 1 FROM new_data n 
                WHERE n.stock_point_id = o.stock_point_id 
                AND n.h3_cell = o.h3_cell 
                AND n.h3_resolution = o.h3_resolution
            )
        """)
        
        # Log inserts
        result = self.conn.execute("""
            INSERT INTO stockpoint_h3_coverage_log 
            (stock_point_id, h3_cell, h3_resolution, operation,
            new_cluster_centroid_lat, new_cluster_centroid_lng, new_cluster_sp_dist_km)
            SELECT 
                n.stock_point_id, n.h3_cell, n.h3_resolution, 'INSERT',
                n.cluster_centroid_lat, n.cluster_centroid_lng, n.cluster_sp_dist_km
            FROM new_data n
            WHERE NOT EXISTS (
                SELECT 1 FROM stockpoint_h3_coverage o
                WHERE o.stock_point_id = n.stock_point_id 
                AND o.h3_cell = n.h3_cell 
                AND o.h3_resolution = n.h3_resolution
            )
        """)
        
        # Log updates
        result = self.conn.execute("""
            INSERT INTO stockpoint_h3_coverage_log 
            (stock_point_id, h3_cell, h3_resolution, operation,
            old_cluster_centroid_lat, old_cluster_centroid_lng, old_cluster_sp_dist_km,
            new_cluster_centroid_lat, new_cluster_centroid_lng, new_cluster_sp_dist_km)
            SELECT 
                o.stock_point_id, o.h3_cell, o.h3_resolution, 'UPDATE',
                o.cluster_centroid_lat, o.cluster_centroid_lng, o.cluster_sp_dist_km,
                n.cluster_centroid_lat, n.cluster_centroid_lng, n.cluster_sp_dist_km
            FROM stockpoint_h3_coverage o
            JOIN new_data n ON (
                n.stock_point_id = o.stock_point_id 
                AND n.h3_cell = o.h3_cell 
                AND n.h3_resolution = o.h3_resolution
            )
            WHERE (
                ABS(o.cluster_centroid_lat - n.cluster_centroid_lat) > 0.000001
                OR ABS(o.cluster_centroid_lng - n.cluster_centroid_lng) > 0.000001
                OR ABS(o.cluster_sp_dist_km - n.cluster_sp_dist_km) > 0.000001
            )
        """)
        
        self.conn.unregister('new_data')
        
        # Get total count from log table
        total_changes = self.conn.execute("""
            SELECT COUNT(*) FROM stockpoint_h3_coverage_log 
            WHERE changed_at >= (SELECT MAX(changed_at) - INTERVAL '1 second' FROM stockpoint_h3_coverage_log)
        """).fetchone()[0]
        
        print(f"✅ Logged {total_changes:,} actual changes")
        return total_changes


    def truncate_insert_stockpoint_h3_coverage(self, df: pd.DataFrame, batch_size: int = 10000, log_changes: bool = True):
        """Truncate and insert enhanced stockpoint_h3_coverage with transaction rollback."""
        if df.empty:
            print("⚠️ No data provided")
            return
        
        print(f"📦 Truncating and inserting {len(df):,} enhanced stockpoint_h3_coverage rows...")
        self._add_stockpoint_h3_coverage_columns()
        
        required_cols = ['stock_point_id', 'h3_cell', 'h3_resolution', 'cluster_centroid_lat', 
                        'cluster_centroid_lng', 'cluster_sp_dist_km','cluster_sp_direction']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        try:
            self.conn.execute("BEGIN;")
            
            if log_changes:
                self.log_coverage_changes(df)
            
            self.conn.execute("TRUNCATE TABLE stockpoint_h3_coverage;")
            print("✅ Table truncated")
            
            total_inserted = 0
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                self.conn.register('batch_df', batch_df)
                
                self.conn.execute("""
                    INSERT INTO stockpoint_h3_coverage 
                    (stock_point_id, h3_cell, h3_resolution, cluster_centroid_lat, 
                    cluster_centroid_lng, cluster_sp_dist_km, cluster_sp_direction)
                    SELECT 
                        stock_point_id, h3_cell, h3_resolution, cluster_centroid_lat,
                        cluster_centroid_lng, cluster_sp_dist_km, cluster_sp_direction
                    FROM batch_df
                """)
                
                self.conn.unregister('batch_df')
                total_inserted += len(batch_df)
                print(f"✅ Batch {i // batch_size + 1}: {len(batch_df):,} records inserted")
            
            self.conn.execute("COMMIT;")
            print(f"🎉 Transaction committed: {total_inserted:,} records inserted successfully")
            
        except Exception as e:
            self.conn.execute("ROLLBACK;")
            print(f"❌ Error occurred, transaction rolled back: {e}")
            try:
                self.conn.unregister('batch_df')
            except:
                pass
            raise  
    
    
    # ------------------------------------------------------------------------------------------------------
    # Customer Assignment - SIMPLE VERSION CONTROL
    # ------------------------------------------------------------------------------------------------------
    """
    MAIN TABLE NAME: customer_stockpoint_cluster_assignment
    Simple version control:
    ✅ Status: ACTIVE, SUPERSEDED, INACTIVE
    ✅ Only ONE ACTIVE record per (customer_id, stock_point_id, h3_resolution)
    ✅ Simple upsert logic
    ✅ DuckDB compatible

    Usage:
    db.upsert_customer_cluster_assignment(df, change_reason="UPDATE", changed_by="SYSTEM")
    """
            
    def _create_customer_cluster_assignment_table(self):
        """Create table with simple version control."""
        print("Creating customer_stockpoint_cluster_assignment table...")

        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS customer_stockpoint_cluster_assignment_id_seq START 1;
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_stockpoint_cluster_assignment (
                id BIGINT PRIMARY KEY DEFAULT nextval('customer_stockpoint_cluster_assignment_id_seq'),
                
                -- Business data
                customer_id BIGINT NOT NULL,
                stock_point_id BIGINT NOT NULL,
                h3_resolution INT NOT NULL,
                cluster_id VARCHAR,
                h3_cell_id VARCHAR,
                assignment_confidence DOUBLE,
                assignment_tier VARCHAR,
                customer_type VARCHAR,
                
                -- Version control
                status VARCHAR DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'SUPERSEDED', 'INACTIVE')),
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                change_reason VARCHAR,
                changed_by VARCHAR DEFAULT 'SYSTEM'
            );
        """)

        print("✅ Created customer_stockpoint_cluster_assignment table")
        
    def _add_customer_cluster_assignment_columns(self):
        """Ensure table exists with required columns."""
        table_exists = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'customer_stockpoint_cluster_assignment'
        """).fetchone()[0] > 0
        
        if not table_exists:
            self._create_customer_cluster_assignment_table()
        else:
            # Add status column if missing
            try:
                self.conn.execute("""
                    ALTER TABLE customer_stockpoint_cluster_assignment 
                    ADD COLUMN IF NOT EXISTS status VARCHAR DEFAULT 'ACTIVE'
                """)
                self.conn.execute("""
                    ALTER TABLE customer_stockpoint_cluster_assignment 
                    ADD COLUMN IF NOT EXISTS change_reason VARCHAR
                """)
                self.conn.execute("""
                    ALTER TABLE customer_stockpoint_cluster_assignment 
                    ADD COLUMN IF NOT EXISTS changed_by VARCHAR DEFAULT 'SYSTEM'
                """)
            except:
                pass

    def truncate_insert_customer_cluster_assignment(self, df: pd.DataFrame, 
                                        change_reason: str = "UPDATE", 
                                        changed_by: str = "SYSTEM"):
        """
        Simple truncate and insert approach:
        1. Truncate table (remove all records)
        2. Insert all new records with ACTIVE status
        
        Clean, simple, no versioning complexity.
        """
        if df.empty:
            print("⚠️ No data provided")
            return

        print(f"📦 Processing {len(df):,} records with truncate and insert...")
        
        self._add_customer_cluster_assignment_columns()
        
        # Validate columns
        required_cols = ['customer_id', 'stock_point_id', 'cluster_id', 'h3_cell_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Prepare data
        df_clean = df.copy()
        df_clean['h3_resolution'] = df_clean.get('h3_resolution', self.resolution).fillna(self.resolution)
        df_clean = df_clean.drop_duplicates(['customer_id', 'stock_point_id', 'h3_resolution'], keep='last')
        
        # Fill NaN values
        import numpy as np
        optional_cols = ['assignment_confidence', 'assignment_tier', 'customer_type']
        for col in optional_cols:
            if col in df_clean.columns:
                # df_clean[col] = df_clean[col].fillna(None)
                df_clean[col] = df_clean[col].fillna(np.nan)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Register DataFrame
        self.conn.register('input_data', df_clean)
        
        try:
            self.conn.execute("BEGIN TRANSACTION")
            
            # Step 1: Truncate table (remove all existing records)
            self.conn.execute("DELETE FROM customer_stockpoint_cluster_assignment")
            print("🗑️ Truncated existing records")
            
            # Step 2: Insert all new records as ACTIVE
            inserted = self.conn.execute(f"""
                INSERT INTO customer_stockpoint_cluster_assignment (
                    customer_id, stock_point_id, h3_resolution, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier, customer_type,
                    status, created_date, modified_date, change_reason, changed_by
                )
                SELECT 
                    customer_id, stock_point_id, h3_resolution, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier, customer_type,
                    'ACTIVE', '{current_time}', '{current_time}', '{change_reason}', '{changed_by}'
                FROM input_data
            """).rowcount or 0
            
            self.conn.execute("COMMIT")
            
            print(f"✅ Truncate and insert completed: {inserted:,} records inserted")
            
            # Simple validation
            total_records = self.conn.execute("""
                SELECT COUNT(*) FROM customer_stockpoint_cluster_assignment
            """).fetchone()[0]
            
            print(f"📊 Total records in table: {total_records:,}")
            
            if total_records == len(df_clean):
                print("✅ Record count validated")
            else:
                print(f"⚠️ Record count mismatch: expected {len(df_clean):,}, got {total_records:,}")
            
        except Exception as e:
            try:
                self.conn.execute("ROLLBACK")
            except Exception as rollback_error:
                if "no transaction is active" not in str(rollback_error).lower():
                    print(f"⚠️ Rollback warning: {rollback_error}")
            
            print(f"❌ Error during truncate and insert: {e}")
            raise
            
        finally:
            try:
                self.conn.unregister('input_data')
            except:
                pass

    def get_active_assignments(self):
        """Get only ACTIVE records."""
        return self.conn.execute("""
            SELECT customer_id, stock_point_id, h3_resolution, cluster_id, h3_cell_id,
                assignment_confidence, assignment_tier, customer_type
            FROM customer_stockpoint_cluster_assignment
            WHERE status = 'ACTIVE'
        """).fetchall()

    # Indexes commented out as requested
    """
    def _create_customer_cluster_assignment_indexes(self):
        indexes = [
            ("idx_csc_active", "customer_id, stock_point_id, h3_resolution WHERE status = 'ACTIVE'"),
            ("idx_csc_status", "status")
        ]
        
        for name, definition in indexes:
            try:
                self.conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON customer_stockpoint_cluster_assignment ({definition})")
            except Exception as e:
                print(f"Index {name}: {e}")
    """

# -----------------------------------------------
# ----------- UTILS ------------------
# -----------------------------------------------

def get_db_summary(db_path=None):
    """
    Summarize key DuckDB tables with record counts and 
    stock_point_id breakdowns by h3_resolution.
    """
    if db_path is None:
        db_path = STORAGE_CONFIG['h3_duckdb_path']

    with duckdb.connect(db_path, read_only=True) as conn:
        # Get available tables
        tables = set(conn.execute("SHOW TABLES").fetchdf()['name'].str.lower())

        def safe_count(table, where=None):
            """Safely count rows in a table, returns 0 if missing."""
            if table.lower() not in tables:
                return 0
            query = f"SELECT COUNT(*) FROM {table}"
            if where:
                query += f" WHERE {where}"
            return conn.execute(query).fetchone()[0]

        def stock_point_breakdown(table):
            """
            Get stock_point_id counts grouped by h3_resolution 
            if both columns exist.
            """
            if table.lower() not in tables:
                return pd.DataFrame()
            cols = set(conn.execute(f"PRAGMA table_info('{table}')").fetchdf()["name"].str.lower())
            if {"stock_point_id", "h3_resolution"}.issubset(cols):
                q = f"""
                SELECT 
                    h3_resolution,
                    COUNT(*) AS total_rows,
                    COUNT(DISTINCT stock_point_id) AS distinct_stock_points
                FROM {table}
                GROUP BY h3_resolution
                ORDER BY h3_resolution
                """
                return conn.execute(q).fetchdf()
            return pd.DataFrame()

        # Overall counts
        summary_data = [
            ("h3_cells", safe_count("h3_cells")),
            ("stockpoint_h3_coverage", safe_count("stockpoint_h3_coverage")),
            ("customer_cluster_assignment", safe_count("customer_stockpoint_cluster_assignment")),
            ("customer_cluster_assignment_active",
             safe_count("customer_stockpoint_cluster_assignment", "status = 'ACTIVE'")),
        ]
        summary_df = pd.DataFrame(summary_data, columns=["table", "count"])

        print("\n📊 DuckDB Database Summary")
        print(summary_df.to_string(index=False))

        # Detailed breakdowns
        for t in ["stockpoint_h3_coverage", "customer_stockpoint_cluster_assignment"]:
            breakdown = stock_point_breakdown(t)
            if not breakdown.empty:
                print(f"\n📍 Stock Point Breakdown by h3_resolution — {t}")
                print(breakdown.to_string(index=False))

    return summary_df






# Usage examples:
if __name__ == "__main__":
    # Example 1: Standard usage (faster than before)
    with FastH3DuckDBManager(resolution=8, db_path="./data/processed/h3_fast.duckdb", batch_size=50000) as db:
        # Your existing loop - now much faster
        # for h3_index, result in h3_to_objects_parallel_generator(h3_cells):
        #     db.add_result(h3_index, result)
        pass
    
    # Example 2: Ultra-fast bulk processing (if you can collect results first)
    # results_dict = {}  # Collect all results first
    # for h3_index, result in h3_to_objects_parallel_generator(h3_cells):
    #     results_dict[h3_index] = result
    #     
    #     # Process in chunks to avoid memory issues
    #     if len(results_dict) >= 100000:
    #         db.add_batch_bulk(results_dict)
    #         results_dict.clear()