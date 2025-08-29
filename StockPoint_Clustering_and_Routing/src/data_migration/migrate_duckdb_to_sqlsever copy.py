import dlt
import duckdb
import os
from typing import Iterator, Dict, Any
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

@dlt.source
def duckdb_source():
    """DLT source to read from DuckDB"""
    
    @dlt.resource(
        name="stockpoint_h3_coverage",
        write_disposition="merge",  # Will update existing rows and insert new ones
        primary_key="id"
        # For incremental loading, add:
        # primary_key=["stock_point_id", "h3_cell", "h3_resolution"]  # Multiple columns
    )
    # Alternative with incremental loading:
    # @dlt.resource(
    #     name="stockpoint_h3_coverage",
    #     write_disposition="merge",
    #     primary_key="id",
    #     merge_key="id"  # or ["stock_point_id", "h3_cell"]
    # )
    def stockpoint_h3_coverage() -> Iterator[Dict[str, Any]]:
        """Extract stockpoint_h3_coverage table from DuckDB"""
        
        # Connect to DuckDB
        conn = duckdb.connect(H3_DUCKDB_PATH)
        
        try:
            # Query the table
            query = """
                SELECT 
                    id,
                    stock_point_id,
                    h3_cell,
                    h3_resolution,
                    cluster_centroid_lat,
                    cluster_centroid_lng,
                    cluster_sp_dist_km
                FROM stockpoint_h3_coverage 
            """
            
            # Execute and fetch in batches to handle large datasets
            result = conn.execute(query)
            
            # Fetch all rows at once for better performance
            rows = result.fetchall()
            
            # Yield in larger batches
            batch_size = 30000
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                batch_data = []
                
                for row in batch:
                    batch_data.append({
                        "id": row[0],
                        "stock_point_id": row[1],
                        "h3_cell": row[2],
                        "h3_resolution": row[3],
                        "cluster_centroid_lat": row[4],
                        "cluster_centroid_lng": row[5],
                        "cluster_sp_dist_km": row[6]
                    })
                
                yield batch_data
        
        finally:
            conn.close()
    
    return stockpoint_h3_coverage


def migrate_stockpoint_h3_coverage():
    """Main migration function to be called from main.py"""
    
    try:
        # Create pipeline with SQL Server destination
        pipeline = dlt.pipeline(
            pipeline_name="duckdb_to_sqlserver",
            destination="mssql",
            dataset_name="gis_analysis"  # Schema name in SQL Server
        )
        
        # Set DLT environment variables for SQL Server (pipeline-specific)
        os.environ['DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__HOST'] = SQL_SERVER_CONFIG["server"]
        os.environ['DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__PORT'] = "1433"
        os.environ['DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__DATABASE'] = SQL_SERVER_CONFIG["database"]
        os.environ['DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__USERNAME'] = SQL_SERVER_CONFIG["username"]
        os.environ['DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__PASSWORD'] = SQL_SERVER_CONFIG["password"]
        os.environ['DUCKDB_TO_SQLSERVER__DESTINATION__MSSQL__CREDENTIALS__DRIVER'] = SQL_SERVER_CONFIG["driver"]
        
        # Run the pipeline
        info = pipeline.run(duckdb_source())
        
        # Provide feedback on successful completion
        print("Migration completed successfully!")

        # Safely access attributes with checks
        if hasattr(info, 'loads_ids'):
            print(f"Pipeline load IDs: {info.loads_ids}")
        elif hasattr(info, 'run_id'):
            print(f"Pipeline run ID: {info.run_id}")
        else:
            print("Pipeline run ID or load IDs are not available.")

        # Check and print the number of loaded packages if they exist
        if hasattr(info, 'load_packages'):
            print(f"Loaded {len(info.load_packages)} packages")
        else:
            print("No load_packages attribute found.")

        # Optional: Print additional information if available
        if hasattr(info, 'started_at') and hasattr(info, 'finished_at'):
            print(f"Pipeline started at: {info.started_at}")
            print(f"Pipeline finished at: {info.finished_at}")
        else:
            print("Pipeline timing information is not available.")

        # Check for failed jobs if the attribute exists
        if hasattr(info, 'has_failed_jobs') and info.has_failed_jobs:
            print("Warning: Some jobs failed during the pipeline run.")
        
        
        return info
        
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        raise


# Alternative function with environment variables for credentials
def migrate_stockpoint_h3_coverage_env():
    """Migration function using environment variables for SQL Server credentials"""
    
    try:
        # Create pipeline - DLT will read SQL Server config from environment
        pipeline = dlt.pipeline(
            pipeline_name="duckdb_to_sqlserver", 
            destination="mssql",
            dataset_name="gis_analysis"
        )
        
        # Run the pipeline
        info = pipeline.run(duckdb_source())
        
        print(f"Migration completed successfully!")
        print(f"Loaded {info.counts['stockpoint_h3_coverage']} rows")
        
        return info
        
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        raise


# Example usage in main.py
if __name__ == "__main__":
    # Call this function from your main.py
    migrate_stockpoint_h3_coverage()