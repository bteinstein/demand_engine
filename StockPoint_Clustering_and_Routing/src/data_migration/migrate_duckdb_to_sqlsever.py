import dlt
import duckdb
import os
from typing import Iterator, Dict, Any, List, Optional
from datetime import datetime
from config.settings import STORAGE_CONFIG

# Configuration
H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']

# SQL Server connection configuration
SQL_SERVER_CONFIG = {
    "server": os.getenv("DB_HOST_OMNIBIZ_REPLICA"),
    "database": os.getenv("DB_DATABASE_OMNIBIZ_REPLICA"),
    "username": os.getenv("DB_USER_OMNIBIZ_REPLICA"),
    "password": os.getenv("DB_PASSWORD_OMNIBIZ_REPLICA"),
    "driver": "ODBC Driver 17 for SQL Server"
}

class DuckDBMigrator:
    """Class to handle DuckDB to SQL Server migrations"""
    
    def __init__(self, duckdb_path: str = H3_DUCKDB_PATH, batch_size: int = 30_000):
        self.duckdb_path = duckdb_path
        self.batch_size = batch_size
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for a table from DuckDB"""
        conn = duckdb.connect(self.duckdb_path)
        try:
            result = conn.execute(f"DESCRIBE {table_name}")
            columns = [row[0] for row in result.fetchall()]
            return columns
        finally:
            conn.close()
    
    def _execute_query_in_batches_(self, query: str, columns: List[str]) -> Iterator[List[Dict[str, Any]]]:
        """Execute query and yield results in batches"""
        conn = duckdb.connect(self.duckdb_path)
        
        try:
            result = conn.execute(query)
            rows = result.fetchall()
            
            # Process in batches
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i:i + self.batch_size]
                batch_data = []
                
                for row in batch:
                    row_dict = {}
                    for idx, column in enumerate(columns):
                        row_dict[column] = row[idx]
                    batch_data.append(row_dict)
                
                yield batch_data
        
        finally:
            conn.close()

    def _execute_query_in_batches(self, query: str, columns: List[str]) -> Iterator[List[Dict[str, Any]]]:
        """Execute query and yield results in batches. Memory-efficient."""
        conn = duckdb.connect(self.duckdb_path)
        try:
            # Use fetchmany to avoid loading all rows at once
            result = conn.execute(query)
            while True:
                batch = result.fetchmany(self.batch_size) # This is the key change
                if not batch:
                    break
                batch_data = []
                for row in batch:
                    row_dict = {column: value for column, value in zip(columns, row)}
                    batch_data.append(row_dict)
                yield batch_data
        finally:
            conn.close()
        
def create_stockpoint_h3_source():
    """Create DLT source for stockpoint_h3_coverage table only"""
    migrator = DuckDBMigrator()
    
    @dlt.source
    def stockpoint_h3_source():
        @dlt.resource(
            name="stockpoint_h3_coverage",
            write_disposition="merge",
            primary_key="id"
        )
        def stockpoint_h3_coverage() -> Iterator[List[Dict[str, Any]]]:
            columns = [
                "id", "stock_point_id", "h3_cell", "h3_resolution",
                "cluster_centroid_lat", "cluster_centroid_lng", "cluster_sp_dist_km"
            ]
            
            query = f"SELECT {', '.join(columns)} FROM stockpoint_h3_coverage"
            yield from migrator._execute_query_in_batches(query, columns)
        
        return stockpoint_h3_coverage
    
    return stockpoint_h3_source()

def create_customer_assignment_source():
    """Create DLT source for customer_stockpoint_cluster_assignment table only"""
    migrator = DuckDBMigrator()
    
    @dlt.source
    def customer_assignment_source():
        @dlt.resource(
            name="customer_stockpoint_cluster_assignment",
            write_disposition="merge",
            primary_key="id"
        )
        def customer_stockpoint_cluster_assignment() -> Iterator[List[Dict[str, Any]]]:
            columns = [
                "id", "stock_point_id", "customer_id", "h3_resolution", "cluster_id",
                "h3_cell_id", "assignment_confidence", "assignment_tier", "customer_type",
                "status", "created_date", "modified_date", "valid_from", "valid_to",
                "version_number", "previous_cluster_id", "previous_h3_cell_id",
                "previous_h3_resolution", "previous_customer_type", "change_reason", "changed_by"
            ]
            
            query = f"SELECT {', '.join(columns)} FROM customer_stockpoint_cluster_assignment"
            yield from migrator._execute_query_in_batches(query, columns)
        
        return customer_stockpoint_cluster_assignment
    
    return customer_assignment_source()

def create_combined_source():
    """Create DLT source with all tables"""
    migrator = DuckDBMigrator()
    
    @dlt.source
    def combined_source():
        @dlt.resource(
            name="stockpoint_h3_coverage",
            write_disposition="merge",
            primary_key="id"
        )
        def stockpoint_h3_coverage() -> Iterator[List[Dict[str, Any]]]:
            columns = [
                "id", "stock_point_id", "h3_cell", "h3_resolution",
                "cluster_centroid_lat", "cluster_centroid_lng", "cluster_sp_dist_km"
            ]
            
            query = f"SELECT {', '.join(columns)} FROM stockpoint_h3_coverage"
            yield from migrator._execute_query_in_batches(query, columns)
        
        @dlt.resource(
            name="customer_stockpoint_cluster_assignment",
            write_disposition="replace",
            primary_key="id"
        )
        def customer_stockpoint_cluster_assignment() -> Iterator[List[Dict[str, Any]]]:
            columns = [
                "id", "stock_point_id", "customer_id", "h3_resolution", "cluster_id",
                "h3_cell_id", "assignment_confidence", "assignment_tier", "customer_type",
                "status", "created_date", "modified_date", "valid_from", "valid_to",
                "version_number", "previous_cluster_id", "previous_h3_cell_id",
                "previous_h3_resolution", "previous_customer_type", "change_reason", "changed_by"
            ]
            
            query = f"SELECT {', '.join(columns)} FROM customer_stockpoint_cluster_assignment"
            yield from migrator._execute_query_in_batches(query, columns)
        
        return [stockpoint_h3_coverage, customer_stockpoint_cluster_assignment]
    
    return combined_source()

def setup_sql_server_credentials():
    """Setup SQL Server credentials as environment variables for DLT"""
    credential_mapping = {
        'DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__HOST': SQL_SERVER_CONFIG["server"],
        'DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__PORT': "1433",
        'DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__DATABASE': SQL_SERVER_CONFIG["database"],
        'DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__USERNAME': SQL_SERVER_CONFIG["username"],
        'DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__PASSWORD': SQL_SERVER_CONFIG["password"],
        'DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__DRIVER': SQL_SERVER_CONFIG["driver"]
    }
    
    for key, value in credential_mapping.items():
        if value:  # Only set if value is not None or empty
            os.environ[key] = value

def run_pipeline_with_source(source, pipeline_name_suffix=""):
    """Helper function to run pipeline with any source"""
    try:
        setup_sql_server_credentials()
        
        pipeline = dlt.pipeline(
            pipeline_name="duckdb_to_sqlserver",  # Keep consistent name
            destination="mssql",
            dataset_name="gis_analysis"
        )
        
        print(f"Starting migration for: {pipeline_name_suffix or 'all tables'}")
        info = pipeline.run(source)
        
        print("Migration completed successfully!")
        print(f"Pipeline run ID: {getattr(info, 'run_id', 'N/A')}")
        
        if hasattr(info, 'has_failed_jobs') and info.has_failed_jobs:
            print("Warning: Some jobs failed during the pipeline run.")
        
        return info
        
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        raise

def migrate_stockpoint_h3_coverage():
    """Migrate only the stockpoint_h3_coverage table"""
    source = create_stockpoint_h3_source()
    return run_pipeline_with_source(source, "_stockpoint")

def migrate_customer_assignment():
    """Migrate only the customer_stockpoint_cluster_assignment table"""
    source = create_customer_assignment_source()
    return run_pipeline_with_source(source, "_customer")


def create_h3_cells_source():
    """Create DLT source for filtered h3_cells table"""
    migrator = DuckDBMigrator()    

    @dlt.source
    def h3_cells_source():
        @dlt.resource(
            name="h3_cells",
            write_disposition="merge",
            primary_key="h3_cell"
        )
        def h3_cells() -> Iterator[List[Dict[str, Any]]]:
            columns = [
                "h3_cell", "resolution", "centroid_lat", "centroid_lng", 
                "created_at", "h3_derived_id", "country_code", "country_name", 
                "state_code", "state_name", "lga_code", "lga_name", 
                "ward_code", "ward_name", "confidence_level", 
                "coverage_percentage", "area_km2"
            ]
            

            query = """
                SELECT 
                    h3_index as h3_cell, resolution, centroid_lat, centroid_lng, 
                    created_at, h3_derived_id, 
                    country_code, country_name, state_code, state_name, lga_code,
                    lga_name, ward_code, ward_name, confidence_level,
                    coverage_percentage, area_km2 
                FROM h3_cells
                WHERE (confidence_level IS NOT NULL AND confidence_level <> 'manual_review') 
                AND resolution=8 
            """            

            yield from migrator._execute_query_in_batches(query, columns)        

        return h3_cells    

    return h3_cells_source()


def migrate_h3_cells():
    """Migrate filtered h3_cells table (one-time migration)"""
    source = create_h3_cells_source()
    return run_pipeline_with_source(source, "_h3")


def migrate_all_tables():
    """Migrate all available tables"""
    source = create_combined_source()
    return run_pipeline_with_source(source, "_all")

# Utility function to check DuckDB connection and table existence
def validate_duckdb_setup():
    """Validate DuckDB connection and check if tables exist"""
    try:
        conn = duckdb.connect(H3_DUCKDB_PATH)
        
        # Get list of tables
        result = conn.execute("SHOW TABLES")
        tables = [row[0] for row in result.fetchall()]
        
        print(f"Available tables in DuckDB: {tables}")
        
        # Check specific tables
        required_tables = ["stockpoint_h3_coverage", "customer_stockpoint_cluster_assignment"]
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            print(f"Warning: Missing tables: {missing_tables}")
        else:
            print("All required tables are present!")
        
        # Get row counts
        for table in required_tables:
            if table in tables:
                count_result = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = count_result.fetchone()[0]
                print(f"  {table}: {count:,} rows")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"DuckDB validation failed: {str(e)}")
        return False

# Example usage functions for main.py
if __name__ == "__main__":
    # Validate setup first
    if validate_duckdb_setup():
        # Migrate specific table
        migrate_stockpoint_h3_coverage()
        
        # Or migrate all tables
        # migrate_all_tables()
        
        # Or migrate specific tables
        # migrate_tables(["stockpoint_h3_coverage", "customer_stockpoint_cluster_assignment"])