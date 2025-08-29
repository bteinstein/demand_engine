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
            'cluster_sp_dist_km': 'DOUBLE PRECISION'
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
                        'cluster_centroid_lng', 'cluster_sp_dist_km']
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
                    cluster_centroid_lng, cluster_sp_dist_km)
                    SELECT 
                        stock_point_id, h3_cell, h3_resolution, cluster_centroid_lat,
                        cluster_centroid_lng, cluster_sp_dist_km
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
    # Customer Assignment
    # ------------------------------------------------------------------------------------------------------
    """
    MAIN TABLE NAME: customer_cluster_assignment
    1. Core Upsert Method
        ✅ Prevents duplicates by properly handling existing records
        ✅ Version control with change tracking
        ✅ Smart change detection - only updates when data actually changes
        ✅ Batch processing for performance
        ✅ Comprehensive logging and progress tracking

    2. Version Control System
        Status tracking: ACTIVE, SUPERSEDED, INACTIVE
        Temporal validity: valid_from/valid_to timestamps
        Change audit: previous values, change reason, who made changes
        Version numbering: Incremental versioning per customer

    3. Query & Analysis Methods
        get_customer_history() - Full timeline for a customer
        get_active_assignments() - Current state only
        get_change_summary() - Daily change statistics
        get_customer_movements() - Location/cluster changes
        check_duplicates() - Verify data integrity
        cleanup_old_records() - Archive management
        
    Usage Examples:
    from src.h3_spatial_system.h3_system.FastH3DuckDBManager import FastH3DuckDBManager
    H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']

    # Your main usage - now with version control
    with FastH3DuckDBManager(resolution=8, db_path=H3_DUCKDB_PATH) as db: 
        # Upsert with change tracking
        db.upsert_customer_cluster_assignment(
            df_output_sp_customer_assignment,
            change_reason="CUSTOMER_LOCATION_UPDATE",
            changed_by="BATCH_PROCESSOR"
        )
        
        # Verify no duplicates
        db.check_duplicates()
        
        # Analyze recent changes
        changes = db.get_change_summary(days_back=7)
        movements = db.get_customer_movements(days_back=30)
        
        # Get specific customer history
        history = db.get_customer_history(customer_id=12345)
    """
        
    def _create_customer_cluster_assignment_table(self):
        """
        Create the customer_cluster_assignment table with full version control,
        change tracking, and performance indexes.
        
        Key Changes:
        - H3_resolution is now part of the business key
        - Unique constraint updated to include h3_resolution
        - Indexes updated to support resolution-aware queries
        
        Features:
        - Primary key auto-increment via sequence
        - Business data columns with h3_resolution as key component
        - Version control (status, version_number, temporal validity)
        - Change tracking (previous values, reason, user)
        - Resolution-aware indexes for active assignments
        """
        print("🏗️ Creating customer_cluster_assignment table with H3 resolution support...")

        # Create sequence for auto-increment ID
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS customer_stockpoint_cluster_assignment START 1;
        """)

        # Create table with unique constraint
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_stockpoint_cluster_assignment (
                -- Primary key
                id BIGINT PRIMARY KEY DEFAULT nextval('customer_cluster_assignment_id_seq'),
                
                -- Business data (h3_resolution is now a key business field)
                stock_point_id BIGINT NOT NULL,
                customer_id BIGINT NOT NULL,
                h3_resolution INT NOT NULL,
                cluster_id VARCHAR,
                h3_cell_id VARCHAR,
                assignment_confidence DOUBLE,
                assignment_tier VARCHAR,
                customer_type VARCHAR,
                
                -- Version Control & Change Tracking
                status VARCHAR DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'INACTIVE', 'SUPERSEDED')),
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                valid_to TIMESTAMP DEFAULT NULL,
                version_number INTEGER DEFAULT 1,
                
                -- Change tracking
                previous_cluster_id VARCHAR,
                previous_h3_cell_id VARCHAR,
                previous_h3_resolution INT,
                previous_customer_type VARCHAR,
                change_reason VARCHAR,
                changed_by VARCHAR DEFAULT 'SYSTEM',
                
                -- Unique constraint for active assignments
                # UNIQUE(customer_id, stock_point_id, h3_resolution, status) # REMOVED status from unique constraint
                UNIQUE(customer_id, stock_point_id, h3_resolution)
            );
        """)

        # Create performance indexes
        self._create_customer_cluster_assignment_indexes()

        print("✅ Created customer_stockpoint_cluster_assignment table with H3 resolution support and indexes")
        
    def _add_customer_cluster_assignment_columns(self):
        """
        Ensure the customer_cluster_assignment table exists with version control columns
        and proper H3 resolution handling.
        
        Key Changes:
        - H3_resolution is now NOT NULL and part of business key
        - Added previous_h3_resolution for change tracking
        - Updated unique constraints to include h3_resolution
        
        Features:
        - Version control with status tracking (ACTIVE, SUPERSEDED, INACTIVE)
        - Temporal validity with valid_from/valid_to timestamps
        - Change tracking with previous values including resolution changes
        - Audit trail with change reasons
        - Resolution-aware constraints
        """
        # Check if table exists
        table_exists = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'customer_stockpoint_cluster_assignment'
        """).fetchone()[0] > 0
        
        if not table_exists:
            print("🏗️ Table customer_cluster_assignment does not exist. Creating with H3 resolution support...")
            
            # Create sequence for auto-increment ID
            self._create_customer_cluster_assignment_table()            
        else:
            print("🔧 Table exists. Checking and adding H3 resolution support...")
            
            # Add version control columns if missing
            version_control_columns = [
                ("status", "VARCHAR DEFAULT 'ACTIVE'"),
                ("created_date", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("modified_date", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("valid_from", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("valid_to", "TIMESTAMP DEFAULT NULL"),
                ("version_number", "INTEGER DEFAULT 1"),
                ("previous_cluster_id", "VARCHAR"),
                ("previous_h3_cell_id", "VARCHAR"),
                ("previous_h3_resolution", "INT"),  # NEW: Track resolution changes
                ("previous_customer_type", "VARCHAR"),  # NEW: Track customer changes
                ("change_reason", "VARCHAR"),
                ("changed_by", "VARCHAR DEFAULT 'SYSTEM'"),
            ]
            
            for col, col_type in version_control_columns:
                try:
                    self.conn.execute(
                        f"ALTER TABLE customer_cluster_assignment ADD COLUMN IF NOT EXISTS {col} {col_type}"
                    )
                except Exception as e:
                    # Column might already exist
                    pass
            
            # Ensure base columns exist with proper h3_resolution handling
            base_columns = [
                ("stock_point_id", "BIGINT"),
                ("customer_id", "BIGINT"),
                ("h3_resolution", f"INT DEFAULT {self.resolution}"),  # Ensure default
                ("cluster_id", "VARCHAR"),
                ("h3_cell_id", "VARCHAR"),
                ("assignment_confidence", "DOUBLE"),
                ("assignment_tier", "VARCHAR"),
                ("customer_type", "VARCHAR") 
            ]

            for col, col_type in base_columns:
                try:
                    self.conn.execute(
                        f"ALTER TABLE customer_stockpoint_cluster_assignment ADD COLUMN IF NOT EXISTS {col} {col_type}"
                    )
                except Exception as e:
                    pass
            
            # CRITICAL: Make h3_resolution NOT NULL if it wasn't already
            try:
                # Update any NULL h3_resolution values first
                self.conn.execute(f"""
                    UPDATE customer_stockpoint_cluster_assignment 
                    SET h3_resolution = {self.resolution} 
                    WHERE h3_resolution IS NULL AND h3_cell_id IS NOT NULL
                """)
                
                # Now make it NOT NULL (DuckDB may not support this directly)
                print(f"ℹ️ Ensuring h3_resolution has default value {self.resolution} for existing records")
                
            except Exception as e:
                print(f"⚠️ Could not enforce NOT NULL on h3_resolution: {e}")
            
            # Update existing records without version control data
            self._migrate_existing_records()
            
            # Drop old unique constraint if it exists
            try:
                self.conn.execute("DROP INDEX IF EXISTS uk_customer_active_assignment;")
            except Exception as e:
                pass
            
            # Create indexes if they don't exist
            self._create_customer_cluster_assignment_indexes()
            
            print("✅ Schema updated with H3 resolution support and version control columns")

    def _create_customer_cluster_assignment_indexes(self):
        """Create performance indexes for the table with H3 resolution support"""
        indexes = [
            # Standard indexes (no WHERE clauses for DuckDB compatibility)
            ("idx_customer_assignment_active_res", "customer_id, stock_point_id, h3_resolution, status"),
            ("idx_customer_assignment_dates", "created_date, valid_from, valid_to"),
            ("idx_customer_assignment_customer", "customer_id"),
            ("idx_customer_assignment_cluster", "cluster_id"),
            ("idx_customer_assignment_h3", "h3_cell_id"),
            ("idx_customer_assignment_resolution", "h3_resolution"),
            ("idx_customer_assignment_stock_res", "stock_point_id, h3_resolution"),
            ("idx_customer_assignment_version", "version_number"),
            ("idx_customer_assignment_status", "status"),  # NEW: Status-specific queries
        ]
        
        for idx_name, columns in indexes:
            try:
                self.conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name} 
                    ON customer_stockpoint_cluster_assignment({columns});
                """)
            except Exception as e:
                print(f"⚠️ Could not create index {idx_name}: {e}")
        
        print("✅ Created performance indexes")
        
    def _migrate_existing_records(self):
        """Update existing records with default version control values and h3_resolution"""
        try:
            current_time = datetime.now()
            
            # Update records that don't have version control data or h3_resolution
            self.conn.execute(f"""
                UPDATE customer_stockpoint_cluster_assignment 
                SET 
                    status = COALESCE(status, 'ACTIVE'),
                    created_date = COALESCE(created_date, '{current_time}'),
                    modified_date = COALESCE(modified_date, '{current_time}'),
                    valid_from = COALESCE(valid_from, '{current_time}'),
                    version_number = COALESCE(version_number, 1),
                    changed_by = COALESCE(changed_by, 'MIGRATION'),
                    h3_resolution = COALESCE(h3_resolution, {self.resolution}),
                    customer_type = COALESCE(customer_type, 'UNKNOWN'),
                WHERE status IS NULL OR version_number IS NULL OR h3_resolution IS NULL
            """)
            
            print("✅ Migrated existing records with version control defaults and h3_resolution")
            
        except Exception as e:
            print(f"⚠️ Migration of existing records failed: {e}")

    def upsert_customer_cluster_assignment_reviewing(self, df: pd.DataFrame, batch_size: int = 10000, 
                                         change_reason: str = "BATCH_UPDATE", changed_by: str = "SYSTEM"):
        """
        Version-controlled upsert that maintains historical records and prevents duplicates.
        Now properly handles H3 resolution as part of the business key.
        
        CRITICAL CHANGES:
        - H3_resolution is now part of the unique business key
        - Customers can have MULTIPLE ACTIVE records per stock_point_id (one per resolution)
        - Comparison logic updated to include h3_resolution
        - Change tracking includes resolution changes
        
        Logic:
        1. For existing customers at same resolution: Mark old records as 'SUPERSEDED', insert new as 'ACTIVE'
        2. For existing customers at different resolution: Insert as new 'ACTIVE' (multiple active allowed)
        3. For new customers: Insert as 'ACTIVE'
        4. Track what changed (cluster_id, h3_cell_id, resolution movements)
        5. Prevent duplicate active records per (customer_id, stock_point_id, h3_resolution)
        
        Args:
            df: DataFrame with customer assignment data
            batch_size: Number of records to process per batch
            change_reason: Reason for the change (e.g., 'RESOLUTION_CHANGE', 'REBALANCING', 'BATCH_UPDATE')
            changed_by: Who/what made the change (e.g., 'USER123', 'SYSTEM', 'API')
            
        Expected DataFrame columns:
            ['customer_id', 'stock_point_id', 'cluster_id', 'h3_cell_id', 
             'assignment_confidence', 'assignment_tier', 'h3_resolution']
        """
        if df.empty:
            print("⚠️ No data provided for customer cluster assignment")
            return
        
        print(f"📦 Upserting {len(df):,} records into customer_stockpoint_cluster_assignment with H3 resolution support...")
         
        # Check for duplicates FIRST, before adding metadata
        duplicates = df.duplicated(subset=['customer_id', 'stock_point_id', 'h3_resolution'], keep=False)
        if duplicates.any():
            dup_count = duplicates.sum()
            print(f"⚠️ Found {dup_count} duplicate records in input data - keeping last occurrence")
            df = df.drop_duplicates(subset=['customer_id', 'stock_point_id', 'h3_resolution'], keep='last')

        print(f"📦 Upserting {len(df):,} records...")
          
        # Ensure schema exists with version control and h3_resolution support
        self._add_customer_cluster_assignment_columns()
        
        # Validate required columns  
        required_cols = ['customer_id', 'stock_point_id', 'cluster_id', 'h3_cell_id', 'h3_resolution','customer_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate h3_resolution values are not null
        null_resolution_count = df['h3_resolution'].isnull().sum()
        if null_resolution_count > 0:
            print(f"⚠️ Found {null_resolution_count} records with null h3_resolution, using default {self.resolution}")
            df['h3_resolution'] = df['h3_resolution'].fillna(self.resolution)
        
        # Add metadata columns to input DataFrame
        current_timestamp = datetime.now()
        df_with_metadata = df.copy()
        df_with_metadata['status'] = 'ACTIVE'
        df_with_metadata['created_date'] = current_timestamp
        df_with_metadata['modified_date'] = current_timestamp
        df_with_metadata['valid_from'] = current_timestamp
        df_with_metadata['valid_to'] = None
        df_with_metadata['change_reason'] = change_reason
        df_with_metadata['changed_by'] = changed_by
        
        # Initialize counters
        total_processed = 0
        total_superseded = 0
        total_new = 0
        total_unchanged = 0
        inactive_count = 0
        
        successfully_processed_keys = []
        
        # Process in batches
        for i in range(0, len(df_with_metadata), batch_size):
            batch_df = df_with_metadata.iloc[i:i+batch_size].copy()
            batch_num = i // batch_size + 1
            
            try:
                self.conn.register('batch_df', batch_df)
                
                # CRITICAL CHANGE: Get existing active records for comparison INCLUDING h3_resolution
                existing_records = self.conn.execute("""
                    SELECT 
                        e.id, e.customer_id, e.stock_point_id, e.h3_resolution, e.customer_type,
                        e.cluster_id, e.h3_cell_id, e.assignment_confidence, 
                        e.assignment_tier, e.version_number, e.created_date
                    FROM customer_stockpoint_cluster_assignment e
                    INNER JOIN batch_df b ON (
                        e.customer_id = b.customer_id 
                        AND e.stock_point_id = b.stock_point_id
                        AND e.h3_resolution = b.h3_resolution
                    )
                    WHERE e.status = 'ACTIVE'
                """).fetchall()
                
                if existing_records: 
                    existing_df = pd.DataFrame(existing_records, columns=[
                        'existing_id', 'customer_id', 'stock_point_id', 'h3_resolution', 'existing_customer_type',  # Changed here
                        'existing_cluster_id', 'existing_h3_cell_id', 'existing_confidence', 
                        'existing_tier', 'existing_version', 'existing_created_date'
                    ])
                    
                    
                    # Merge with new data to identify changes
                    batch_with_existing = batch_df.merge(
                        existing_df, 
                        on=['customer_id', 'stock_point_id', 'h3_resolution'],   # REMOVED 'customer_type'
                        how='left'
                    )
                    
                    changed_mask = (
                        (batch_with_existing['cluster_id'].fillna('') != batch_with_existing['existing_cluster_id'].fillna('')) |
                        (batch_with_existing['h3_cell_id'].fillna('') != batch_with_existing['existing_h3_cell_id'].fillna('')) |
                        (batch_with_existing['assignment_confidence'].fillna(0) != batch_with_existing['existing_confidence'].fillna(0)) |
                        (batch_with_existing['assignment_tier'].fillna('') != batch_with_existing['existing_tier'].fillna('')) | 
                        (batch_with_existing['customer_type'].fillna('') != batch_with_existing['existing_customer_type'].fillna(''))
                    )
                    
                    
                    changed_records = batch_with_existing[changed_mask].copy()
                    unchanged_records = batch_with_existing[~changed_mask]
                    
                    unchanged_count = len(unchanged_records)
                    total_unchanged += unchanged_count
                    
                    if len(changed_records) > 0:
                        # Add change tracking information
                        changed_records['previous_cluster_id'] = changed_records['existing_cluster_id']
                        changed_records['previous_h3_cell_id'] = changed_records['existing_h3_cell_id']
                        changed_records['previous_h3_resolution'] = changed_records['h3_resolution']  # Track resolution
                        changed_records['previous_customer_type'] = changed_records['existing_customer_type']  # Track CUSTOMER TYPE
                        changed_records['version_number'] = changed_records['existing_version'] + 1
                        
                        # Register changed records for processing
                        self.conn.unregister('batch_df')
                        self.conn.register('changed_batch', changed_records)
                        
                        # UPDATED: Mark existing changed records as SUPERSEDED (including h3_resolution in match)
                        self.conn.execute(f"""
                            UPDATE customer_stockpoint_cluster_assignment 
                            SET 
                                status = 'SUPERSEDED',
                                valid_to = '{current_timestamp}',
                                modified_date = '{current_timestamp}'
                            WHERE (customer_id, stock_point_id, h3_resolution) IN (
                                SELECT customer_id, stock_point_id, h3_resolution FROM changed_batch
                            ) AND status = 'ACTIVE'
                        """)
                        
                        superseded_count = len(changed_records)
                        total_superseded += superseded_count
                        
                        # Step 3: Insert new ACTIVE records for changed data
                        self.conn.execute("""
                            INSERT INTO customer_stockpoint_cluster_assignment (
                                stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                                assignment_confidence, assignment_tier,
                                status, created_date, modified_date, valid_from, valid_to,
                                version_number, previous_cluster_id, previous_h3_cell_id, 
                                previous_h3_resolution, previous_customer_type, change_reason, changed_by
                            )
                            SELECT
                                stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                                assignment_confidence, assignment_tier,
                                status, created_date, modified_date, valid_from, valid_to,
                                version_number, previous_cluster_id, previous_h3_cell_id,
                                previous_h3_resolution, previous_customer_type, change_reason, changed_by
                            FROM changed_batch
                        """)
                        
                        self.conn.unregister('changed_batch')
                        total_new += len(changed_records)
                        
                        print(f"✅ Batch {batch_num}: {len(changed_records)} changed, {unchanged_count} unchanged, {superseded_count} superseded")
                    
                    else:
                        print(f"ℹ️ Batch {batch_num}: All {unchanged_count} records unchanged, skipping")
                
                else:
                    # No existing records, these are all new customers (or new resolutions)
                    batch_df['version_number'] = 1
                    batch_df['previous_cluster_id'] = None
                    batch_df['previous_h3_cell_id'] = None
                    batch_df['previous_h3_resolution'] = None
                    batch_df['previous_customer_type'] = None
                    
                    self.conn.unregister('batch_df')
                    self.conn.register('new_batch', batch_df)
                    
                    # Insert all as new ACTIVE records
                    self.conn.execute("""
                        INSERT INTO customer_stockpoint_cluster_assignment (
                            stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                            assignment_confidence, assignment_tier,
                            status, created_date, modified_date, valid_from, valid_to,
                            version_number, previous_cluster_id, previous_h3_cell_id, 
                            previous_h3_resolution, previous_customer_type, change_reason, changed_by
                        )
                        SELECT
                            stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                            assignment_confidence, assignment_tier,
                            status, created_date, modified_date, valid_from, valid_to,
                            version_number, previous_cluster_id, previous_h3_cell_id,
                            previous_h3_resolution, previous_customer_type, change_reason, changed_by
                        FROM new_batch
                    """)
                    
                    self.conn.unregister('new_batch')
                    batch_count = len(batch_df)
                    total_new += batch_count
                    
                    print(f"✅ Batch {batch_num}: {batch_count} new customer-resolution assignments added")
                
                total_processed += len(batch_df)
                
                batch_keys = batch_df[['customer_id', 'stock_point_id', 'h3_resolution']].to_dict('records')
                successfully_processed_keys.extend(batch_keys)
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_num}: {e}")
                # Clean up any registered dataframes
                for df_name in ['batch_df', 'changed_batch', 'new_batch']:
                    try:
                        self.conn.unregister(df_name)
                    except:
                        pass
                continue
        
        
        
        print("🔍 Checking for records to mark as INACTIVE (not in current dataset)...")
        if successfully_processed_keys:
            success_df = pd.DataFrame(successfully_processed_keys)
            self.conn.register('complete_df', success_df)
        
            # Find and update records that are ACTIVE in DB but missing from current df
            inactive_result = self.conn.execute(f"""
                UPDATE customer_stockpoint_cluster_assignment 
                SET status = 'INACTIVE', 
                    valid_to = '{current_timestamp}',
                    modified_date = '{current_timestamp}',
                    change_reason = '{change_reason}_REMOVAL',
                    changed_by = '{changed_by}'
                WHERE status = 'ACTIVE' 
                AND (customer_id, stock_point_id, h3_resolution) NOT IN (
                    SELECT customer_id, stock_point_id, h3_resolution FROM complete_df
                )
            """)
            
            inactive_count = inactive_result.rowcount if hasattr(inactive_result, 'rowcount') else 0
            self.conn.unregister('complete_df')
            
            if inactive_count > 0:
                print(f"🔄 Marked {inactive_count} records as INACTIVE (removed from source)")
            else:
                print("ℹ️ No records to mark as INACTIVE")
        else:
            print("⚠️ No batches processed successfully, skipping inactive check")
            inactive_count = 0    
            
        # Final summary
        print(f"""
                🎉 Customer cluster assignment upsert completed:
                📊 Total processed: {total_processed:,} records
                🆕 New/updated records: {total_new:,}
                🔄 Superseded records: {total_superseded:,}  
                ⚡ Unchanged records: {total_unchanged:,}
                ❌ Inactive records: {inactive_count:,}
                📝 Change reason: {change_reason}
                👤 Changed by: {changed_by}
                🔧 H3 resolution support: ✅ ENABLED
                """)

    def upsert_customer_cluster_assignment(self, df: pd.DataFrame, 
                                        change_reason: str = "BATCH_UPDATE", changed_by: str = "SYSTEM"):
        """
        Efficient version-controlled upsert using MERGE-like operations.
        Processes entire dataset at once for maximum efficiency.
        """
        if df.empty:
            print("⚠️ No data provided")
            return

        print(f"📦 Processing {len(df):,} records...")
        
        # Ensure schema and validate
        self._add_customer_cluster_assignment_columns()
        required_cols = ['customer_id', 'stock_point_id', 'cluster_id', 'h3_cell_id', 'h3_resolution', 'customer_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Clean and prepare data
        df = df.fillna({'h3_resolution': self.resolution})
        df = df.drop_duplicates(subset=['customer_id', 'stock_point_id', 'h3_resolution'], keep='last')
        
        current_timestamp = datetime.now()
        
        try:
            self.conn.execute("BEGIN TRANSACTION")
            self.conn.register('input_data', df)
            
            # Step 1: Archive existing ACTIVE records that will be updated
            archive_count = self.conn.execute(f"""
                INSERT INTO customer_stockpoint_cluster_assignment (
                    stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier, status, created_date, modified_date, 
                    valid_from, valid_to, version_number, previous_cluster_id, previous_h3_cell_id, 
                    previous_h3_resolution, previous_customer_type, change_reason, changed_by
                )
                SELECT DISTINCT
                    e.stock_point_id, e.customer_id, e.h3_resolution, e.customer_type, 
                    e.cluster_id, e.h3_cell_id, e.assignment_confidence, e.assignment_tier,
                    'SUPERSEDED', e.created_date, '{current_timestamp}', e.valid_from, '{current_timestamp}',
                    e.version_number, e.previous_cluster_id, e.previous_h3_cell_id,
                    e.previous_h3_resolution, e.previous_customer_type, 
                    COALESCE(e.change_reason, '{change_reason}'), COALESCE(e.changed_by, '{changed_by}')
                FROM customer_stockpoint_cluster_assignment e
                INNER JOIN input_data i ON (
                    e.customer_id = i.customer_id AND 
                    e.stock_point_id = i.stock_point_id AND 
                    e.h3_resolution = i.h3_resolution
                )
                WHERE e.status = 'ACTIVE'
                AND (
                    e.cluster_id != i.cluster_id OR
                    e.h3_cell_id != i.h3_cell_id OR
                    COALESCE(e.assignment_confidence, 0) != COALESCE(i.assignment_confidence, 0) OR
                    COALESCE(e.assignment_tier, '') != COALESCE(i.assignment_tier, '') OR
                    COALESCE(e.customer_type, '') != COALESCE(i.customer_type, '')
                )
            """).rowcount or 0
            
            # Step 2: Delete ACTIVE records being replaced
            delete_count = self.conn.execute("""
                DELETE FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE' 
                AND (customer_id, stock_point_id, h3_resolution) IN (
                    SELECT customer_id, stock_point_id, h3_resolution FROM input_data
                )
            """).rowcount or 0
            
            # Step 3: Insert all new ACTIVE records with version tracking
            insert_count = self.conn.execute(f"""
                INSERT INTO customer_stockpoint_cluster_assignment (
                    stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier, status, created_date, modified_date,
                    valid_from, valid_to, version_number, previous_cluster_id, previous_h3_cell_id,
                    previous_h3_resolution, previous_customer_type, change_reason, changed_by
                )
                SELECT DISTINCT
                    i.stock_point_id, i.customer_id, i.h3_resolution, i.customer_type, 
                    i.cluster_id, i.h3_cell_id, i.assignment_confidence, i.assignment_tier,
                    'ACTIVE', '{current_timestamp}', '{current_timestamp}', '{current_timestamp}', NULL,
                    COALESCE(e.version_number, 0) + 1,
                    e.cluster_id, e.h3_cell_id, e.h3_resolution, e.customer_type,
                    '{change_reason}', '{changed_by}'
                FROM input_data i
                LEFT JOIN (
                    SELECT customer_id, stock_point_id, h3_resolution, cluster_id, h3_cell_id, 
                        customer_type, version_number,
                        ROW_NUMBER() OVER (PARTITION BY customer_id, stock_point_id, h3_resolution ORDER BY id DESC) as rn
                    FROM customer_stockpoint_cluster_assignment 
                    WHERE status IN ('ACTIVE', 'SUPERSEDED')
                ) e ON (
                    i.customer_id = e.customer_id AND 
                    i.stock_point_id = e.stock_point_id AND 
                    i.h3_resolution = e.h3_resolution AND
                    e.rn = 1
                )
            """).rowcount or 0
            
            # Step 4: Archive records not in current dataset (mark as INACTIVE)
            inactive_count = self.conn.execute(f"""
                INSERT INTO customer_stockpoint_cluster_assignment (
                    stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier, status, created_date, modified_date,
                    valid_from, valid_to, version_number, previous_cluster_id, previous_h3_cell_id,
                    previous_h3_resolution, previous_customer_type, change_reason, changed_by
                )
                SELECT DISTINCT
                    e.stock_point_id, e.customer_id, e.h3_resolution, e.customer_type,
                    e.cluster_id, e.h3_cell_id, e.assignment_confidence, e.assignment_tier,
                    'INACTIVE', e.created_date, '{current_timestamp}', e.valid_from, '{current_timestamp}',
                    e.version_number, e.previous_cluster_id, e.previous_h3_cell_id,
                    e.previous_h3_resolution, e.previous_customer_type,
                    '{change_reason}_REMOVAL', '{changed_by}'
                FROM customer_stockpoint_cluster_assignment e
                WHERE e.status = 'ACTIVE'
                AND (e.customer_id, e.stock_point_id, e.h3_resolution) NOT IN (
                    SELECT customer_id, stock_point_id, h3_resolution FROM input_data
                )
            """).rowcount or 0
            
            # Step 5: Remove old ACTIVE records that are now INACTIVE
            self.conn.execute("""
                DELETE FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE'
                AND (customer_id, stock_point_id, h3_resolution) NOT IN (
                    SELECT customer_id, stock_point_id, h3_resolution FROM input_data
                )
            """)
            
            self.conn.execute("COMMIT")
            
            # Calculate unchanged records
            unchanged_count = max(0, len(df) - (insert_count - delete_count))
            
            print(f"""✅ Upsert completed:
    📊 Processed: {len(df):,} | New/Updated: {insert_count:,} | Unchanged: {unchanged_count:,}
    🔄 Superseded: {archive_count:,} | Inactive: {inactive_count:,}""")
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            print(f"❌ Error: {e}")
            raise
        finally:
            try:
                self.conn.unregister('input_data')
            except:
                pass
    
    def get_customer_history(self, customer_id: int, stock_point_id: int = None, h3_resolution: int = None):
        """
        Get the complete history of cluster assignments for a customer.
        Now supports filtering by H3 resolution.
        
        Args:
            customer_id: Customer ID to look up
            stock_point_id: Optional stock point filter
            h3_resolution: Optional H3 resolution filter
            
        Returns:
            pandas.DataFrame: Historical assignment records ordered by version (newest first)
        """
        where_clause = "WHERE customer_id = ?"
        params = [customer_id]
        
        if stock_point_id:
            where_clause += " AND stock_point_id = ?"
            params.append(stock_point_id)
            
        if h3_resolution:
            where_clause += " AND h3_resolution = ?"
            params.append(h3_resolution)
        
        try:
            result = self.conn.execute(f"""
                SELECT 
                    id, stock_point_id, h3_resolution,  customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier,
                    status, version_number,
                    created_date, modified_date, valid_from, valid_to,
                    previous_cluster_id, previous_h3_cell_id, previous_h3_resolution, previous_customer_type, 
                    change_reason, changed_by
                FROM customer_stockpoint_cluster_assignment
                {where_clause}
                ORDER BY h3_resolution, version_number DESC, created_date DESC
            """, params).fetchall()
            
            if result:
                columns = [
                    'id', 'stock_point_id', 'h3_resolution', 'customer_type', 'cluster_id', 'h3_cell_id', 
                    'assignment_confidence', 'assignment_tier',
                    'status', 'version_number', 'created_date', 'modified_date', 
                    'valid_from', 'valid_to', 'previous_cluster_id', 'previous_h3_cell_id',
                    'previous_h3_resolution', 'previous_customer_type', 'change_reason', 'changed_by'
                ]
                
                history_df = pd.DataFrame(result, columns=columns)
                resolution_info = f" at resolution {h3_resolution}" if h3_resolution else " (all resolutions)"
                print(f"📋 Found {len(history_df)} historical records for customer {customer_id}{resolution_info}")
                return history_df
            else:
                resolution_info = f" at resolution {h3_resolution}" if h3_resolution else ""
                print(f"❌ No records found for customer {customer_id}{resolution_info}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving customer history: {e}")
            return pd.DataFrame()

    def get_active_assignments(self, stock_point_id: int = None, h3_resolution: int = None):
        """
        Get all currently active customer assignments.
        Now supports filtering by H3 resolution.
        
        Args:
            stock_point_id: Optional stock point filter
            h3_resolution: Optional H3 resolution filter
            
        Returns:
            pandas.DataFrame: Currently active assignment records
        """
        where_clause = "WHERE status = 'ACTIVE'"
        params = []
        
        if stock_point_id:
            where_clause += " AND stock_point_id = ?"
            params.append(stock_point_id)
            
        if h3_resolution:
            where_clause += " AND h3_resolution = ?"
            params.append(h3_resolution)
        
        try:
            result = self.conn.execute(f"""
                SELECT 
                    customer_id, stock_point_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier,
                    created_date, modified_date, version_number,
                    change_reason, changed_by
                FROM customer_stockpoint_cluster_assignment
                {where_clause}
                ORDER BY customer_id, stock_point_id, h3_resolution
            """, params).fetchall()
            
            if result:
                columns = [
                    'customer_id', 'stock_point_id', 'h3_resolution', 'customer_type', 'cluster_id', 'h3_cell_id',
                    'assignment_confidence', 'assignment_tier',
                    'created_date', 'modified_date', 'version_number',
                    'change_reason', 'changed_by'
                ]
                
                active_df = pd.DataFrame(result, columns=columns)
                resolution_info = f" at resolution {h3_resolution}" if h3_resolution else " (all resolutions)"
                print(f"📊 Found {len(active_df)} active assignments{resolution_info}")
                return active_df
            else:
                resolution_info = f" at resolution {h3_resolution}" if h3_resolution else ""
                print(f"❌ No active assignments found{resolution_info}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving active assignments: {e}")
            return pd.DataFrame()

    def get_resolution_summary(self):
        """
        NEW METHOD: Get summary of assignments by H3 resolution.
        Useful for understanding how customers are distributed across different resolutions.
        
        Returns:
            pandas.DataFrame: Summary by resolution including counts and status distribution
        """
        try:
            result = self.conn.execute("""
                SELECT 
                    h3_resolution,
                    status,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT customer_id) as unique_customers,
                    COUNT(DISTINCT stock_point_id) as unique_stock_points,
                    COUNT(DISTINCT cluster_id) as unique_clusters,
                    AVG(assignment_confidence) as avg_confidence,
                    MIN(created_date) as earliest_assignment,
                    MAX(created_date) as latest_assignment
                FROM customer_stockpoint_cluster_assignment
                GROUP BY h3_resolution, status
                ORDER BY h3_resolution, status
            """).fetchall()
            
            if result:
                columns = [
                    'h3_resolution', 'status', 'record_count', 'unique_customers',
                    'unique_stock_points', 'unique_clusters', 'avg_confidence',
                    'earliest_assignment', 'latest_assignment'
                ]
                summary_df = pd.DataFrame(result, columns=columns)
                print(f"📈 H3 Resolution summary:")
                return summary_df
            else:
                print("❌ No resolution data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving resolution summary: {e}")
            return pd.DataFrame()

    def get_customer_movements(self, days_back: int = 30, include_resolution_changes: bool = True):
        """
        Get customers who changed clusters/locations/resolutions in the specified period.
        Now tracks H3 resolution changes as well.
        
        Args:
            days_back: Number of days to look back
            include_resolution_changes: Whether to include resolution-only changes
            
        Returns:
            pandas.DataFrame: Customer movement records
        """
        print(f"\n{'-'*100}")
        try:
            # Build the change condition
            change_conditions = [
                "previous_cluster_id != cluster_id",
                "previous_h3_cell_id != h3_cell_id",
                "previous_customer_type !=  customer_type"
            ]
            
            if include_resolution_changes:
                change_conditions.append("previous_h3_resolution != h3_resolution")
            
            change_condition = " OR ".join(change_conditions)
            
            result = self.conn.execute(f"""
                SELECT 
                    customer_id, stock_point_id, h3_resolution,customer_type,
                    previous_cluster_id, cluster_id,
                    previous_h3_cell_id, h3_cell_id,
                    previous_h3_resolution, previous_customer_type,
                    assignment_confidence, assignment_tier,
                    created_date, change_reason, changed_by,
                    version_number
                FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE'
                  AND previous_cluster_id IS NOT NULL
                  AND created_date >= CURRENT_DATE - INTERVAL {days_back} DAYS
                  AND ({change_condition})
                ORDER BY created_date DESC, customer_id
            """).fetchall()
            
            if result:
                columns = [
                    'customer_id', 'stock_point_id', 'h3_resolution','customer_type',
                    'previous_cluster_id', 'cluster_id', 'previous_h3_cell_id', 'h3_cell_id',
                    'previous_h3_resolution',  'previous_customer_type','assignment_confidence', 'assignment_tier', 
                    'created_date', 'change_reason', 'changed_by', 'version_number'
                ]
                
                movements_df = pd.DataFrame(result, columns=columns)
                resolution_info = " (including resolution changes)" if include_resolution_changes else " (excluding resolution-only changes)"
                print(f"🚶 Found {len(movements_df)} customer movements in last {days_back} days{resolution_info}")
                return movements_df
            else:
                print(f"❌ No customer movements found in last {days_back} days")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving customer movements: {e}")
            return pd.DataFrame()

    def check_duplicates(self):
        """
        Helper method to check for duplicate active records in customer_stockpoint_cluster_assignment.
        Updated to include h3_resolution in duplicate detection.
        Should return empty result if upsert logic is working correctly.
        
        Returns:
            pandas.DataFrame: Any duplicate active assignments found
        """
        try:
            result = self.conn.execute("""
                SELECT 
                    customer_id, stock_point_id, h3_resolution, customer_type,
                    COUNT(*) as active_count,
                    string_agg(CAST(id AS VARCHAR), ', ' ORDER BY id) as record_ids
                FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE'
                GROUP BY customer_id, stock_point_id, h3_resolution, customer_type
                HAVING COUNT(*) > 1
                ORDER BY active_count DESC
                LIMIT 10
            """).fetchall()
            
            if result:
                print(f"⚠️ Found {len(result)} duplicate active assignments (same customer + stock_point + resolution):")
                columns = ['customer_id', 'stock_point_id', 'h3_resolution', 'active_count', 'record_ids']
                duplicates_df = pd.DataFrame(result, columns=columns)
                print(duplicates_df)
                return duplicates_df
            else:
                print("✅ No duplicate active assignments found!")
            
            # Additional stats with resolution breakdown
            resolution_stats = self.conn.execute("""
                SELECT 
                    h3_resolution,
                    COUNT(*) as active_count,
                    COUNT(DISTINCT customer_id) as unique_customers,
                    COUNT(DISTINCT stock_point_id) as unique_stock_points
                FROM customer_stockpoint_cluster_assignment 
                WHERE status = 'ACTIVE'
                GROUP BY h3_resolution
                ORDER BY h3_resolution
            """).fetchall()
            
            if resolution_stats:
                print(f"\n📊 Active records by H3 resolution:")
                for res, count, customers, stock_points in resolution_stats:
                    print(f"   Resolution {res}: {count:,} records ({customers:,} customers, {stock_points:,} stock points)")
            
            total_active = self.conn.execute("""
                SELECT COUNT(*) FROM customer_stockpoint_cluster_assignment WHERE status = 'ACTIVE'
            """).fetchone()[0]
            
            total_all = self.conn.execute("""
                SELECT COUNT(*) FROM customer_stockpoint_cluster_assignment
            """).fetchone()[0]
            
            # Check for customers with multiple active resolutions (this is now allowed)
            multi_resolution_customers = self.conn.execute("""
                SELECT 
                    customer_id, stock_point_id,
                    COUNT(DISTINCT h3_resolution) as resolution_count,
                    string_agg(CAST(h3_resolution AS VARCHAR), ', ' ORDER BY h3_resolution) as resolutions
                FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE'
                GROUP BY customer_id, stock_point_id
                HAVING COUNT(DISTINCT h3_resolution) > 1
                LIMIT 5
            """).fetchall()
            
            if multi_resolution_customers:
                print(f"\nℹ️ Customers with multiple active resolutions (this is normal):")
                for customer_id, stock_point_id, res_count, resolutions in multi_resolution_customers:
                    print(f"   Customer {customer_id} at stock point {stock_point_id}: {res_count} resolutions ({resolutions})")
            
            print(f"\n📊 Total active records: {total_active:,}")
            print(f"📊 Total all records: {total_all:,}")
            
            return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error checking duplicates: {e}")
            return pd.DataFrame()

    def get_resolution_conflicts(self):
        """
        NEW METHOD: Identify potential conflicts where customers have assignments
        at multiple resolutions that might be inconsistent.
        
        This helps identify data quality issues where H3 cells at different
        resolutions don't align properly.
        
        Returns:
            pandas.DataFrame: Customers with potentially conflicting resolution assignments
        """
        try:
            result = self.conn.execute("""
                WITH customer_resolutions AS (
                    SELECT 
                        customer_id, stock_point_id,
                        h3_resolution, cluster_id, h3_cell_id,
                        assignment_confidence, assignment_tier
                    FROM customer_stockpoint_cluster_assignment
                    WHERE status = 'ACTIVE'
                ),
                multi_resolution_customers AS (
                    SELECT 
                        customer_id, stock_point_id,
                        COUNT(DISTINCT h3_resolution) as resolution_count
                    FROM customer_resolutions
                    GROUP BY customer_id, stock_point_id
                    HAVING COUNT(DISTINCT h3_resolution) > 1
                )
                SELECT 
                    cr.customer_id, cr.stock_point_id,
                    cr.h3_resolution, cr.cluster_id, cr.h3_cell_id,
                    cr.assignment_confidence, cr.assignment_tier,
                    mrc.resolution_count
                FROM customer_resolutions cr
                INNER JOIN multi_resolution_customers mrc ON (
                    cr.customer_id = mrc.customer_id 
                    AND cr.stock_point_id = mrc.stock_point_id
                )
                ORDER BY cr.customer_id, cr.stock_point_id, cr.h3_resolution
            """).fetchall()
            
            if result:
                columns = [
                    'customer_id', 'stock_point_id', 'h3_resolution', 'cluster_id', 
                    'h3_cell_id', 'assignment_confidence', 'assignment_tier', 'resolution_count'
                ]
                conflicts_df = pd.DataFrame(result, columns=columns)
                unique_customers = conflicts_df['customer_id'].nunique()
                print(f"🔍 Found {unique_customers} customers with multiple resolution assignments")
                print("   (This may be normal if you intentionally use multiple resolutions)")
                return conflicts_df
            else:
                print("✅ No customers with multiple resolution assignments found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error checking resolution conflicts: {e}")
            return pd.DataFrame()

    def cleanup_old_records(self, days_to_keep: int = 365):
        """
        Archive or delete very old SUPERSEDED records to manage database size.
        
        Args:
            days_to_keep: Number of days of SUPERSEDED records to retain
            
        Returns:
            int: Number of records deleted
        """
        try:
            # Count records to be deleted
            count_result = self.conn.execute("""
                SELECT COUNT(*) 
                FROM customer_stockpoint_cluster_assignment 
                WHERE status = 'SUPERSEDED' 
                  AND valid_to < CURRENT_DATE - INTERVAL ? DAYS
            """, [days_to_keep]).fetchone()[0]
            
            if count_result == 0:
                print(f"ℹ️ No SUPERSEDED records older than {days_to_keep} days found")
                return 0
            
            print(f"🗑️ Found {count_result} SUPERSEDED records older than {days_to_keep} days")
            
            # Delete old records
            self.conn.execute("""
                DELETE FROM customer_stockpoint_cluster_assignment 
                WHERE status = 'SUPERSEDED' 
                  AND valid_to < CURRENT_DATE - INTERVAL ? DAYS
            """, [days_to_keep])
            
            print(f"✅ Cleaned up {count_result} old SUPERSEDED records")
            return count_result
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            return 0

    def validate_h3_resolution_consistency(self, sample_size: int = 1000):
        """
        NEW METHOD: Validate that H3 cells at different resolutions are consistent
        for the same customer. This helps catch data quality issues.
        
        Args:
            sample_size: Number of customer-stock_point pairs to validate
            
        Returns:
            pandas.DataFrame: Any inconsistencies found
        """
        print(f"🔍 Validating H3 resolution consistency (sample size: {sample_size:,})...")
        
        try:
            # This would require H3 library to properly validate parent-child relationships
            # For now, we'll do a basic check for customers with multiple resolutions
            result = self.conn.execute(f"""
                WITH customer_multi_res AS (
                    SELECT 
                        customer_id, stock_point_id,
                        COUNT(DISTINCT h3_resolution) as resolution_count,
                        string_agg(
                            h3_resolution || ':' || COALESCE(h3_cell_id, 'NULL'), 
                            ', ' ORDER BY h3_resolution
                        ) as resolution_cells
                    FROM customer_stockpoint_cluster_assignment
                    WHERE status = 'ACTIVE'
                    GROUP BY customer_id, stock_point_id
                    HAVING COUNT(DISTINCT h3_resolution) > 1
                    LIMIT {sample_size}
                )
                SELECT * FROM customer_multi_res
                ORDER BY resolution_count DESC
            """).fetchall()
            
            if result:
                columns = ['customer_id', 'stock_point_id', 'resolution_count', 'resolution_cells']
                inconsistencies_df = pd.DataFrame(result, columns=columns)
                
                print(f"ℹ️ Found {len(inconsistencies_df)} customers with multiple H3 resolutions")
                print("   To fully validate consistency, you'll need H3 library to check parent-child relationships")
                
                return inconsistencies_df
            else:
                print("✅ No customers with multiple resolutions found in sample")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error validating H3 consistency: {e}")
            return pd.DataFrame()

    def get_change_summary(self, days_back: int = 7):
        """
        Get summary of changes in the last N days, including resolution changes.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            pandas.DataFrame: Summary of changes by date, reason, resolution, and status
        """
        try:
            result = self.conn.execute(f"""
                SELECT 
                    DATE(created_date) as change_date,
                    h3_resolution,
                    change_reason,
                    status,
                    changed_by,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT customer_id) as unique_customers,
                    COUNT(DISTINCT stock_point_id) as unique_stock_points,
                    AVG(assignment_confidence) as avg_confidence
                FROM customer_stockpoint_cluster_assignment
               WHERE created_date >= CURRENT_DATE - INTERVAL '{days_back} days'
                GROUP BY DATE(created_date), h3_resolution, change_reason, status, changed_by
                ORDER BY change_date DESC, h3_resolution, change_reason, status
            """).fetchall()
            
            if result:
                columns = [
                    'change_date', 'h3_resolution', 'change_reason', 'status', 'changed_by',
                    'record_count', 'unique_customers', 'unique_stock_points', 'avg_confidence'
                ]
                summary_df = pd.DataFrame(result, columns=columns)
                print(f"📈 Change summary for last {days_back} days (with H3 resolution breakdown):")
                return summary_df
            else:
                print(f"❌ No changes found in last {days_back} days")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving change summary: {e}")
            return pd.DataFrame()
        
        
        

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