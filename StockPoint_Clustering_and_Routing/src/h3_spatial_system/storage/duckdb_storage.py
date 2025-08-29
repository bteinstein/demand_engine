"""
DuckDB storage for H3 address data.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from config.settings import STORAGE_CONFIG

logger = logging.getLogger(__name__)


class DuckDBStorage:
    """DuckDB-based storage for H3 address data."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or STORAGE_CONFIG['duckdb_path']
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Connect to DuckDB database."""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = duckdb.connect(str(self.db_path))
            logger.info(f"Connected to DuckDB: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise
    
    def create_tables(self, force: bool = False):
        """Create the H3 addresses table with optimized schema."""
        try:
            # Check if table exists
            result = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='h3_addresses'").fetchone()
            table_exists = result is not None

            if table_exists and not force:
                logger.info("Table 'h3_addresses' already exists. Skipping creation.")
                return

            if force:
                logger.info("Forcing table creation, dropping existing table...")
                self.conn.execute("DROP TABLE IF EXISTS h3_addresses")

            # Create main table
            self.conn.execute("""
                CREATE TABLE h3_addresses (
                    h3_id VARCHAR PRIMARY KEY,
                    h3_derived_id VARCHAR UNIQUE,
                    grid_position_id VARCHAR UNIQUE,
                    primary_address_id VARCHAR,
                    country_code VARCHAR(2),
                    state_code VARCHAR(10),
                    state_name VARCHAR(50),
                    lga_code VARCHAR(10), 
                    lga_name VARCHAR(100),
                    ward_code VARCHAR(10),
                    ward_name VARCHAR(100),
                    confidence_level VARCHAR(20),
                    coverage_percentage REAL,
                    centroid_lat DOUBLE,
                    centroid_lng DOUBLE,
                    area_km2 REAL,
                    created_date DATE DEFAULT CURRENT_DATE
                )
            """)
            
            # Create indexes for performance
            self.conn.execute("CREATE INDEX idx_state_lga ON h3_addresses(state_code, lga_code)")
            self.conn.execute("CREATE INDEX idx_coordinates ON h3_addresses(centroid_lat, centroid_lng)")
            self.conn.execute("CREATE INDEX idx_primary_address ON h3_addresses(primary_address_id)")
            self.conn.execute("CREATE INDEX idx_confidence ON h3_addresses(confidence_level)")
            
            logger.info("Created H3 addresses table and indexes")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def insert_addresses(self, addresses: List[Dict]):
        """Insert address records into the database."""
        if not addresses:
            logger.warning("No addresses to insert")
            return
        
        try:
            # Prepare data for insertion
            records = []
            for addr in addresses:
                record = (
                    addr['h3_id'],
                    addr['h3_derived_id'],
                    addr['grid_position_id'],
                    addr['primary_address_id'],
                    addr['admin_assignment']['country']['code'],
                    addr['admin_assignment']['state']['code'],
                    addr['admin_assignment']['state']['name'],
                    addr['admin_assignment']['lga']['code'],
                    addr['admin_assignment']['lga']['name'],
                    addr['admin_assignment']['ward']['code'],
                    addr['admin_assignment']['ward']['name'],
                    addr['assignment_quality']['confidence_level'],
                    addr['assignment_quality']['coverage_percentage'],
                    addr['geometry']['centroid']['lat'],
                    addr['geometry']['centroid']['lng'],
                    addr['geometry']['area_km2']
                )
                records.append(record)
            
            # Bulk insert
            self.conn.executemany("""
                INSERT INTO h3_addresses (
                    h3_id, h3_derived_id, grid_position_id, primary_address_id,
                    country_code, state_code, state_name, lga_code, lga_name,
                    ward_code, ward_name, confidence_level, coverage_percentage,
                    centroid_lat, centroid_lng, area_km2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(h3_id) DO UPDATE SET
                    h3_derived_id = excluded.h3_derived_id,
                    grid_position_id = excluded.grid_position_id,
                    primary_address_id = excluded.primary_address_id,
                    state_code = excluded.state_code,
                    state_name = excluded.state_name,
                    lga_code = excluded.lga_code,
                    lga_name = excluded.lga_name,
                    ward_code = excluded.ward_code,
                    ward_name = excluded.ward_name,
                    confidence_level = excluded.confidence_level,
                    coverage_percentage = excluded.coverage_percentage,
                    centroid_lat = excluded.centroid_lat,
                    centroid_lng = excluded.centroid_lng,
                    area_km2 = excluded.area_km2
            """, records)
            
            logger.info(f"Inserted {len(records)} address records")
            
        except Exception as e:
            logger.error(f"Failed to insert addresses: {e}")
            raise
    
    def get_address_by_h3(self, h3_id: str) -> Optional[Dict]:
        """Get address by H3 ID."""
        try:
            result = self.conn.execute("""
                SELECT * FROM h3_addresses WHERE h3_id = ?
            """, [h3_id]).fetchone()
            
            if result:
                return self._row_to_dict(result)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get address by H3 ID {h3_id}: {e}")
            return None
    
    def get_address_by_coordinates(self, lat: float, lng: float, radius_km: float = 1.0) -> List[Dict]:
        """Get addresses within radius of coordinates."""
        try:
            # Simple bounding box query (for performance)
            # In production, you'd use proper spatial functions
            lat_range = radius_km / 111.0  # Approximate km to degrees
            lng_range = radius_km / (111.0 * abs(lat / 90.0))  # Adjust for latitude
            
            result = self.conn.execute("""
                SELECT * FROM h3_addresses 
                WHERE centroid_lat BETWEEN ? AND ?
                AND centroid_lng BETWEEN ? AND ?
            """, [lat - lat_range, lat + lat_range, lng - lng_range, lng + lng_range]).fetchall()
            
            return [self._row_to_dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Failed to get address by coordinates: {e}")
            return []
    
    def query_by_admin(self, state_code: Optional[str] = None, 
                      lga_code: Optional[str] = None,
                      ward_code: Optional[str] = None) -> List[Dict]:
        """Query addresses by administrative hierarchy."""
        try:
            query = "SELECT * FROM h3_addresses WHERE 1=1"
            params = []
            
            if state_code:
                query += " AND state_code = ?"
                params.append(state_code)
            
            if lga_code:
                query += " AND lga_code = ?"
                params.append(lga_code)
            
            if ward_code:
                query += " AND ward_code = ?"
                params.append(ward_code)
            
            result = self.conn.execute(query, params).fetchall()
            return [self._row_to_dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Failed to query by admin: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        try:
            stats = {}
            
            # Total records
            total = self.conn.execute("SELECT COUNT(*) FROM h3_addresses").fetchone()[0]
            stats['total_records'] = total
            
            # Confidence distribution
            confidence_stats = self.conn.execute("""
                SELECT confidence_level, COUNT(*) as count
                FROM h3_addresses 
                GROUP BY confidence_level
            """).fetchall()
            stats['confidence_distribution'] = dict(confidence_stats)
            
            # Administrative coverage
            state_count = self.conn.execute("SELECT COUNT(DISTINCT state_code) FROM h3_addresses").fetchone()[0]
            lga_count = self.conn.execute("SELECT COUNT(DISTINCT lga_code) FROM h3_addresses").fetchone()[0]
            ward_count = self.conn.execute("SELECT COUNT(DISTINCT ward_code) FROM h3_addresses").fetchone()[0]
            
            stats['administrative_coverage'] = {
                'unique_states': state_count,
                'unique_lgas': lga_count,
                'unique_wards': ward_count
            }
            
            # Coverage statistics
            coverage_stats = self.conn.execute("""
                SELECT 
                    AVG(coverage_percentage) as avg_coverage,
                    MIN(coverage_percentage) as min_coverage,
                    MAX(coverage_percentage) as max_coverage
                FROM h3_addresses
            """).fetchone()
            
            stats['coverage_statistics'] = {
                'avg_coverage': coverage_stats[0],
                'min_coverage': coverage_stats[1],
                'max_coverage': coverage_stats[2]
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}
    
    def export_to_parquet(self, output_path: str, partition_by: Optional[List[str]] = None):
        """Export data to Parquet format."""
        try:
            partition_cols = partition_by or ['state_code']
            
            self.conn.execute(f"""
                COPY h3_addresses TO '{output_path}' 
                (FORMAT PARQUET, PARTITION_BY {','.join(partition_cols)})
            """)
            
            logger.info(f"Exported to Parquet: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export to Parquet: {e}")
            raise
    
    def export_to_csv(self, output_path: str):
        """Export data to CSV format."""
        try:
            self.conn.execute(f"""
                COPY h3_addresses TO '{output_path}' (FORMAT CSV, HEADER)
            """)
            
            logger.info(f"Exported to CSV: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise
    
    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary."""
        columns = ['h3_id', 'h3_derived_id', 'grid_position_id', 'primary_address_id',
                  'country_code', 'state_code', 'state_name', 'lga_code', 'lga_name',
                  'ward_code', 'ward_name', 'confidence_level', 'coverage_percentage',
                  'centroid_lat', 'centroid_lng', 'area_km2', 'created_date']
        
        return dict(zip(columns, row))
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed DuckDB connection")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_sample_database():
    """Create a sample database with test data."""
    from src.h3_system.generator import generate_sample_addresses
    
    # Generate sample addresses
    sample_addresses = generate_sample_addresses()
    
    # Create database
    db = DuckDBStorage("data/exports/sample_h3_addresses.duckdb")
    db.create_tables()
    db.insert_addresses(sample_addresses)
    
    # Get statistics
    stats = db.get_statistics()
    print("Sample Database Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Confidence distribution: {stats['confidence_distribution']}")
    print(f"  Administrative coverage: {stats['administrative_coverage']}")
    
    db.close()
    return db


if __name__ == "__main__":
    # Test the DuckDB storage
    create_sample_database() 