import duckdb
import os
from pathlib import Path
import pandas as pd

def verify_db_table_and_h3_cells_data(db_path: str):
    """
    Comprehensive verification of DuckDB file and its contents.
    """
    print("=" * 60)
    print("🔍 DuckDB File & Data Verification")
    print("=" * 60)
    
    # 1. Check if file exists
    print(f"📁 Database path: {os.path.abspath(db_path)}")
    print(f"📁 Directory: {os.path.dirname(os.path.abspath(db_path))}")
    
    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        print(f"✅ File exists! Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    else:
        print("❌ File does not exist!")
        
        # Check if directory exists
        dir_path = os.path.dirname(db_path)
        if os.path.exists(dir_path):
            print(f"📂 Directory exists: {dir_path}")
            print(f"📂 Directory contents: {list(Path(dir_path).iterdir())}")
        else:
            print(f"❌ Directory does not exist: {dir_path}")
        return False
    
    # 2. Try to connect and check contents
    try:
        conn = duckdb.connect(db_path)
        print("✅ Successfully connected to database")
        
        # Check if table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"📋 Tables in database: {[table[0] for table in tables]}")
        
        if not tables:
            print("⚠️  No tables found in database")
            conn.close()
            return True
            
        # Check h3_cells table specifically
        if any('h3_cells' in str(table) for table in tables):
            print("✅ h3_cells table found")
            
            # Get table info
            try:
                table_info = conn.execute("DESCRIBE h3_cells").fetchall()
                print("📊 Table structure:")
                for col in table_info:
                    print(f"   - {col[0]}: {col[1]}")
            except Exception as e:
                print(f"⚠️  Error describing table: {e}")
            
            # Count records
            try:
                count = conn.execute("SELECT COUNT(*) FROM h3_cells").fetchone()[0]
                print(f"📊 Total records: {count:,}")
                
                if count > 0:
                    # Show sample data
                    print("📋 Sample records:")
                    samples = conn.execute("SELECT h3_index, resolution, centroid_lat, centroid_lng, error FROM h3_cells LIMIT 3").fetchall()
                    for i, row in enumerate(samples, 1):
                        print(f"   {i}. H3: {row[0]}, Res: {row[1]}, Lat/Lng: ({row[2]}, {row[3]}), Error: {row[4]}")
                        
                    # Check for errors
                    error_count = conn.execute("SELECT COUNT(*) FROM h3_cells WHERE error IS NOT NULL").fetchone()[0]
                    print(f"❌ Records with errors: {error_count:,}")
                    
                else:
                    print("⚠️  Table exists but is empty")
                    
            except Exception as e:
                print(f"❌ Error querying table: {e}")
        else:
            print("❌ h3_cells table not found")
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return False


class H3DuckDBManager:
    """
    Simplified H3DuckDBManager with minimal optimizations and better debugging.
    """
    
    def __init__(self, db_path: str, resolution: int, batch_size: int = 10000):
        self.db_path = os.path.abspath(db_path)  # Use absolute path
        self.batch_size = batch_size
        self.batch_data = []
        self.total_saved = 0
        self.resolution = resolution
        
        print(f"🚀 Initializing H3DuckDBManager")
        print(f"📁 Database path: {self.db_path}")
        print(f"📁 Directory: {os.path.dirname(self.db_path)}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        print(f"📂 Directory created/verified: {os.path.dirname(self.db_path)}")
        
        # Initialize DuckDB connection
        try:
            self.conn = duckdb.connect(self.db_path)
            print("✅ DuckDB connection established")
        except Exception as e:
            print(f"❌ Failed to connect to DuckDB: {e}")
            raise
            
        self._create_tables()
        self._minimal_optimize_settings()  # Safer optimizations
        
        # Verify setup
        self._verify_setup()
    
    def _create_tables(self):
        """Create table with minimal complexity."""
        print("📋 Creating h3_cells table...")
        try:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS h3_cells (
                    h3_index VARCHAR PRIMARY KEY,
                    resolution INTEGER,
                    centroid_lat DOUBLE,
                    centroid_lng DOUBLE,
                    polygon_coords VARCHAR,  -- Simplified as text for now
                    boundary_coords VARCHAR,
                    latlng_coords VARCHAR,
                    polygon_area DOUBLE,
                    num_vertices INTEGER,
                    error VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            print("✅ Table created successfully")
        except Exception as e:
            print(f"❌ Error creating table: {e}")
            raise
    
    def _minimal_optimize_settings(self):
        """Minimal safe optimizations only."""
        safe_optimizations = [
            "SET memory_limit='2GB'",  # Reduced memory limit
            # Removed other potentially problematic settings
        ]
        
        for setting in safe_optimizations:
            try:
                self.conn.execute(setting)
                print(f"✅ Applied: {setting}")
            except Exception as e:
                print(f"⚠️  Optimization failed: {setting} - {e}")
    
    def _verify_setup(self):
        """Verify the database is properly set up."""
        try:
            # Test write
            self.conn.execute("INSERT OR REPLACE INTO h3_cells (h3_index, resolution) VALUES ('test', 1)")
            
            # Test read
            result = self.conn.execute("SELECT COUNT(*) FROM h3_cells WHERE h3_index = 'test'").fetchone()
            
            if result and result[0] == 1:
                print("✅ Database read/write test passed")
                # Clean up test record
                self.conn.execute("DELETE FROM h3_cells WHERE h3_index = 'test'")
            else:
                print("❌ Database read/write test failed")
                
        except Exception as e:
            print(f"❌ Database verification failed: {e}")
            raise
    
    def add_result(self, h3_index: str, result: dict):
        """Simplified add_result with better error handling."""
        try:
            # Simplified data processing
            if 'error' in result:
                batch_item = {
                    'h3_index': h3_index,
                    'resolution': self.resolution,
                    'centroid_lat': None,
                    'centroid_lng': None,
                    'polygon_coords': None,
                    'boundary_coords': None,
                    'latlng_coords': None,
                    'polygon_area': None,
                    'num_vertices': None,
                    'error': result['error']
                }
            else:
                # Convert complex objects to strings for now
                lat, lng = result['centroid']
                batch_item = {
                    'h3_index': h3_index,
                    'resolution': self.resolution,
                    'centroid_lat': lat,
                    'centroid_lng': lng,
                    'polygon_coords': str(list(result['polygon'].exterior.coords)) if result.get('polygon') else None,
                    'boundary_coords': str(result.get('boundary', [])),
                    'latlng_coords': str(result.get('latlng_coords', [])),
                    'polygon_area': result['polygon'].area if result.get('polygon') else None,
                    'num_vertices': len(list(result['polygon'].exterior.coords)) if result.get('polygon') else None,
                    'error': None
                }
            
            self.batch_data.append(batch_item)
            
            # Save batch when it reaches the limit
            if len(self.batch_data) >= self.batch_size:
                self._save_batch()
                
        except Exception as e:
            print(f"❌ Error in add_result for {h3_index}: {e}")
            # Add error record
            error_item = {
                'h3_index': h3_index,
                'resolution': self.resolution,
                'centroid_lat': None, 'centroid_lng': None,
                'polygon_coords': None, 'boundary_coords': None, 'latlng_coords': None,
                'polygon_area': None, 'num_vertices': None,
                'error': f"Processing error: {str(e)}"
            }
            self.batch_data.append(error_item)
    
    def _save_batch(self):
        """Save batch with immediate verification."""
        if not self.batch_data:
            return
        
        print(f"💾 Saving batch of {len(self.batch_data)} items...")
        
        try:
            # Use simple INSERT instead of pandas for debugging
            for item in self.batch_data:
                self.conn.execute("""
                    INSERT OR REPLACE INTO h3_cells (
                        h3_index, resolution, centroid_lat, centroid_lng,
                        polygon_coords, boundary_coords, latlng_coords,
                        polygon_area, num_vertices, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    item['h3_index'], item['resolution'], item['centroid_lat'], item['centroid_lng'],
                    item['polygon_coords'], item['boundary_coords'], item['latlng_coords'],
                    item['polygon_area'], item['num_vertices'], item['error']
                ])
            
            # Force immediate flush
            self.conn.execute("CHECKPOINT")
            
            self.total_saved += len(self.batch_data)
            print(f"✅ Batch saved and checkpointed (total: {self.total_saved})")
            
            # Verify the save worked
            count = self.conn.execute("SELECT COUNT(*) FROM h3_cells").fetchone()[0]
            print(f"📊 Database now contains: {count} records")
            
        except Exception as e:
            print(f"❌ Error saving batch: {e}")
            raise
        finally:
            self.batch_data.clear()
    
    def force_commit(self):
        """Force save any pending data."""
        if self.batch_data:
            print(f"🔄 Force saving {len(self.batch_data)} pending items...")
            self._save_batch()
        
        # Additional checkpoint
        try:
            self.conn.execute("CHECKPOINT")
            print("✅ Force checkpoint completed")
        except Exception as e:
            print(f"❌ Force checkpoint failed: {e}")
    
    def finalize(self):
        """Final cleanup with verification."""
        print("🏁 Finalizing...")
        
        if self.batch_data:
            self._save_batch()
        
        # Final checkpoint
        try:
            self.conn.execute("CHECKPOINT")
            print("✅ Final checkpoint completed")
        except Exception as e:
            print(f"❌ Final checkpoint failed: {e}")
        
        print(f"🎉 Total saved: {self.total_saved}")
        
        # Verify file exists
        if os.path.exists(self.db_path):
            size = os.path.getsize(self.db_path)
            print(f"✅ Database file confirmed: {size:,} bytes")
        else:
            print(f"❌ Database file missing: {self.db_path}")
    
    def close(self):
        """Close with verification."""
        try:
            self.finalize()
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()
                print("🔒 Connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"⚠️  Exception during processing: {exc_val}")
        self.close()


# Usage example with verification:
if __name__ == "__main__":
    db_path = "./data/processed/h3_data.duckdb"
    
    # Test the simplified manager
    print("🧪 Testing H3DuckDBManager...")
    
    try:
        with H3DuckDBManager(db_path=db_path, resolution=8, batch_size=5) as db:
            # Add some test data
            for i in range(12):  # Will trigger 2 batch saves + 1 final
                test_result = {
                    'centroid': (12.34 + i*0.01, 5.67 + i*0.01),
                    'polygon': type('MockPolygon', (), {
                        'exterior': type('MockExterior', (), {
                            'coords': [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
                        })(),
                        'area': 1.0 + i*0.1
                    })(),
                    'boundary': [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                    'latlng_coords': [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
                }
                db.add_result(f"test_h3_{i}", test_result)
        
        print("\n" + "="*60)
        print("🔍 Post-test verification:")
        verify_db_file_and_data(db_path)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()