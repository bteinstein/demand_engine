import duckdb
import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

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
        self._create_optimized_tables()
        self._apply_performance_settings()
        
        print("✅ Fast H3DuckDBManager ready")
    
    def _create_optimized_tables(self):
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
    
    
    def create_customer_cluster_assignment_table(self):
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS customer_cluster_assignment_id_seq START 1;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_cluster_assignment (
                id BIGINT PRIMARY KEY DEFAULT nextval('customer_cluster_assignment_id_seq'),
                stock_point_id BIGINT,
                customer_id BIGINT,
                cluster_id VARCHAR,
                h3_cell_id VARCHAR,
                assignment_confidence DOUBLE,
                assignment_tier VARCHAR,
                h3_resolution INT
            );
        """)
        print("✅ customer_cluster_assignment table created")
    
    def _add_customer_cluster_assignment_columns(self):
        """
        Ensure the customer_cluster_assignment table exists.
        If creating new table, create with 'id' as primary key with auto-increment default.
        If table exists without 'id', add 'id' column WITHOUT default or constraints (DuckDB limitation).
        Add other columns if missing.
        """
        if not table_exists:
            print("Table customer_cluster_assignment does not exist. Creating it with id primary key and default...")
            # Create sequence
            self.conn.execute("CREATE SEQUENCE IF NOT EXISTS customer_cluster_assignment_id_seq START 1;")
            # Create table with id column default
            self.conn.execute("""
                CREATE TABLE customer_cluster_assignment (
                    id BIGINT PRIMARY KEY DEFAULT nextval('customer_cluster_assignment_id_seq'),
                    stock_point_id BIGINT,
                    customer_id BIGINT,
                    cluster_id VARCHAR,
                    h3_cell_id VARCHAR,
                    assignment_confidence DOUBLE,
                    assignment_tier VARCHAR,
                    h3_resolution INT,
                );
            """)
        else:
            # Table exists, check if id column exists
            id_col_exists = self.conn.execute("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'customer_cluster_assignment' AND column_name = 'id'
            """).fetchone()[0] > 0

            if not id_col_exists:
                print("Table exists but 'id' column missing. Adding 'id' column WITHOUT default or constraints...")
                # Add id column without default or NOT NULL
                self.conn.execute("""
                    ALTER TABLE customer_cluster_assignment ADD COLUMN IF NOT EXISTS id BIGINT;
                """)
                print("⚠️ NOTE: You will need to populate 'id' values separately!")

        # Add other columns if missing
        columns_to_add = [
            ("stock_point_id", "BIGINT"),
            ("customer_id", "BIGINT"),
            ("cluster_id", "VARCHAR"),
            ("h3_cell_id", "VARCHAR"),
            ("assignment_confidence", "DOUBLE"),
            ("assignment_tier", "VARCHAR"),
            ("h3_resolution", "INT"),
        ]

        for col, col_type in columns_to_add:
            try:
                self.conn.execute(
                    f"ALTER TABLE customer_cluster_assignment ADD COLUMN IF NOT EXISTS {col} {col_type}"
                )
            except Exception as e:
                print(f"Error adding column {col}: {e}")

        print("✅ Completed schema check/update for customer_cluster_assignment")

    def upsert_customer_cluster_assignment(self, df: pd.DataFrame, batch_size: int = 10000):
        """
        Batch insert or update customer_cluster_assignment data.
        Assumes df has columns:
        ['customer_id', 'cluster_id', 'h3_cell_id', 'assignment_confidence', 'assignment_tier', 'stock_point_id','h3_resolution ']
        """
        if df.empty:
            print("⚠️ No data provided for customer cluster assignment")
            return
        
        print(f"📦 Upserting {len(df):,} records into customer_cluster_assignment...")
        
        # Ensure schema exists
        self._add_customer_cluster_assignment_columns()
        
        total_processed = 0
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            try:
                self.conn.register('batch_df', batch_df)
                
                # Using INSERT OR REPLACE to upsert by primary key 'id'.
                # Since 'id' is auto-increment and not in the input data, this behaves like insert only.
                # If you want to enforce uniqueness on customer_id + h3_cell_id or similar, 
                # you'd need additional constraints and merge logic.
                
                self.conn.execute("""
                    INSERT INTO customer_cluster_assignment (
                        stock_point_id, customer_id, cluster_id, h3_cell_id,
                        assignment_confidence, assignment_tier, h3_resolution 
                    )
                    SELECT
                        stock_point_id, customer_id, cluster_id, h3_cell_id,
                        assignment_confidence, assignment_tier, h3_resolution 
                    FROM batch_df
                """)
                
                self.conn.unregister('batch_df')
                total_processed += len(batch_df)
                print(f"✅ Inserted batch {i // batch_size + 1}: {len(batch_df):,} records (total: {total_processed:,})")
                
            except Exception as e:
                print(f"❌ Error inserting batch {i // batch_size + 1}: {e}")
                continue
        
        print(f"🎉 customer_cluster_assignment upsert completed: {total_processed:,} records processed")
      
    def _add_sp_coverage_cells_columns(self):
        """
        Ensure sp_coverage_cells table exists.
        If creating new table, create with 'id' as primary key with auto-increment default.
        If table exists without 'id', add 'id' column without default.
        Add other metadata columns if missing.
        """
        # Check if table exists
        table_exists = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'sp_coverage_cells'
        """).fetchone()[0] > 0

        if not table_exists:
            print("Table sp_coverage_cells does not exist. Creating it with id primary key and default...")
            # Create sequence
            self.conn.execute("CREATE SEQUENCE IF NOT EXISTS sp_coverage_cells_id_seq START 1;")
            # Create table with id column default
            self.conn.execute("""
                CREATE TABLE sp_coverage_cells (
                    id BIGINT PRIMARY KEY DEFAULT nextval('sp_coverage_cells_id_seq'),
                    stock_point_id INT,
                    h3_cell VARCHAR,
                    h3_resolution INT
                );
            """)
        else:
            # Table exists, check if id column exists
            id_col_exists = self.conn.execute("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'sp_coverage_cells' AND column_name = 'id'
            """).fetchone()[0] > 0

            if not id_col_exists:
                print("Table exists but 'id' column missing. Adding 'id' column WITHOUT default or constraints...")
                # Add id column without default or NOT NULL (DuckDB limitation)
                self.conn.execute("""
                    ALTER TABLE sp_coverage_cells ADD COLUMN IF NOT EXISTS id BIGINT;
                """)
                print("⚠️ NOTE: You will need to populate 'id' values separately!")

        # Add other columns if missing
        columns_to_add = [
            ("stock_point_id", "INT"),
            ("h3_cell", "VARCHAR"),
            ("h3_resolution", "INT")
        ]

        for col, col_type in columns_to_add:
            try:
                self.conn.execute(
                    f"ALTER TABLE sp_coverage_cells ADD COLUMN IF NOT EXISTS {col} {col_type}"
                )
            except Exception as e:
                print(f"Error adding column {col}: {e}")

        print("✅ Completed schema check/update for sp_coverage_cells")

    def upsert_sp_coverage_cells(self, df: pd.DataFrame, batch_size: int = 10000):
        """
        Insert sp_coverage_cells rows from DataFrame.
        Assumes df has columns: h3_cell, stock_point_id, h3_resolution.
        'id' is auto-generated.
        """
        if df.empty:
            print("⚠️ No data provided")
            return
        
        print(f"📦 Inserting {len(df):,} sp_coverage_cells rows...")
        
        self._add_sp_coverage_cells_columns()  # ensure schema exists
        
        total_processed = 0
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            try:
                self.conn.register('batch_df', batch_df)
                self.conn.execute("""
                    INSERT INTO sp_coverage_cells (h3_cell, stock_point_id, h3_resolution)
                    SELECT h3_cell, stock_point_id, h3_resolution FROM batch_df
                """)
                self.conn.unregister('batch_df')
                
                total_processed += len(batch_df)
                print(f"✅ Inserted batch {i // batch_size + 1}: {len(batch_df):,} records (total: {total_processed:,})")
                
            except Exception as e:
                print(f"❌ Error inserting batch {i // batch_size + 1}: {e}")
                continue
        
        print(f"🎉 Insert completed: {total_processed:,} rows")
   
    def _add_address_columns(self):
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
    
    def update_address_data(self, address_data: Dict[str, Dict[str, Any]], batch_size: int = 10000):
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
    
    def upsert_address_data(self, address_data: Dict[str, Dict[str, Any]], batch_size: int = 10000):
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
        """Apply DuckDB-specific performance optimizations using latest supported settings."""
        
        performance_settings = [
            # Memory and CPU settings
            "SET memory_limit='4GB'",           # memory_limit expects quoted value with units
            "SET threads=8",                    # set number of threads
            
            # Write / checkpoint related settings
            "SET checkpoint_threshold='2GB'",  # larger checkpoint threshold
            
            # DuckDB-specific performance flags
            "SET preserve_insertion_order=false",
            "SET force_compression='auto'",
            "SET temp_directory='/tmp'",
            
            # Optional profiling/debugging (can disable if needed)
            # "SET enable_profiling=false",  # Uncomment if profiling disables speed
            
            # Note: Removed unsupported parameters (wal_autocheckpoint, enable_verification, force_checkpoint)
        ]
        
        applied = 0
        for setting in performance_settings:
            try:
                self.conn.execute(setting)
                applied += 1
            except Exception as e:
                print(f"⚠️ Setting failed: {setting} - {e}")
        
        print(f"⚡ Applied {applied}/{len(performance_settings)} performance optimizations")
        
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