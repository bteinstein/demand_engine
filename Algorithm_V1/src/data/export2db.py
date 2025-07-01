

import pandas as pd
import logging
from typing import List, Dict, Any, Callable
from contextlib import contextmanager
from .get_connection import get_connection 
logger = logging.getLogger(__name__)


@contextmanager
def database_connection(get_connection_func: Callable):
    """Context manager for database connections."""
    conn = get_connection_func()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_dataframe(df: pd.DataFrame, table_name: str, conn, 
                    match_cols: List[str], update_cols: List[str], 
                    fast_executemany: bool = True) -> None:
    """
    Upsert DataFrame using staging table and MERGE operation.
    
    Args:
        df: DataFrame to upsert
        table_name: Target table name
        conn: Database connection
        match_cols: Columns to match on for upsert
        update_cols: Columns to update when matched
        fast_executemany: Enable fast executemany for bulk insert
    """
    _validate_upsert_params(df, table_name, match_cols, update_cols)
    
    staging_table = f"#{table_name}_staging"
    columns = df.columns.tolist()
    column_list = ', '.join(f"[{col}]" for col in columns)
    
    with conn.cursor() as cursor:
        cursor.fast_executemany = fast_executemany
        
        try:
            # Create staging table with same schema
            cursor.execute(f"""
                SELECT TOP 0 {column_list} INTO {staging_table} 
                FROM [{table_name}] WHERE 1 = 0
            """)
            
            # Bulk insert into staging
            placeholders = ', '.join(['?'] * len(columns))
            cursor.executemany(
                f"INSERT INTO {staging_table} ({column_list}) VALUES ({placeholders})",
                df.values.tolist()
            )
            
            # Execute MERGE operation
            merge_sql = _build_merge_sql(table_name, staging_table, columns, match_cols, update_cols)
            cursor.execute(merge_sql)
            
            conn.commit()
            logger.info(f"Successfully upserted {len(df)} rows to {table_name}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Upsert failed for {table_name}: {e}")
            raise


def _validate_upsert_params(df: pd.DataFrame, table_name: str, 
                           match_cols: List[str], update_cols: List[str]) -> None:
    """Validate upsert parameters."""
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if not match_cols or not update_cols:
        raise ValueError("match_cols and update_cols must not be empty")
    
    missing_cols = set(match_cols + update_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    if not table_name.strip() or any(char in table_name for char in ".;[]'\""):
        raise ValueError(f"Invalid table name: {table_name}")


def _build_merge_sql(table_name: str, staging_table: str, columns: List[str], 
                    match_cols: List[str], update_cols: List[str]) -> str:
    """Build MERGE SQL statement."""
    on_condition = ' AND '.join(f"TARGET.[{col}] = SOURCE.[{col}]" for col in match_cols)
    update_set = ', '.join(f"TARGET.[{col}] = SOURCE.[{col}]" for col in update_cols)
    insert_columns = ', '.join(f"[{col}]" for col in columns)
    insert_values = ', '.join(f"SOURCE.[{col}]" for col in columns)
    
    return f"""
        MERGE [{table_name}] AS TARGET
        USING {staging_table} AS SOURCE ON {on_condition}
        WHEN MATCHED THEN 
            UPDATE SET {update_set}
        WHEN NOT MATCHED THEN 
            INSERT ({insert_columns}) VALUES ({insert_values});
    """


def extract_recommendations(stockpoints_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract and combine recommendation data from stockpoints."""
    dataframes = []
    
    for stockpoint_data in stockpoints_data.values():
        if not stockpoint_data or 'all_push_recommendation' not in stockpoint_data:
            continue
            
        df = stockpoint_data['all_push_recommendation']
        if not df.empty:
            # Ensure cluster columns are strings
            df = df.copy()
            df['ClusterLGAs'] = df['ClusterLGAs'].astype(str)
            df['ClusterLCDAs'] = df['ClusterLCDAs'].astype(str)
            dataframes.append(df)
    
    if not dataframes:
        return pd.DataFrame()
    
    return pd.concat(dataframes, ignore_index=True)


def prepare_recommendation_data(df: pd.DataFrame, current_date: str) -> pd.DataFrame:
    """Prepare recommendation data for database insertion."""
    if df.empty:
        return df
    
    df = df.copy()
    df['ModifiedDate'] = current_date
    
    # Remove cluster columns and clean data types
    df = df.drop(columns=['ClusterLGAs', 'ClusterLCDAs'], errors='ignore')
    
    # Convert date columns to integers
    date_columns = ['SKUDaysSinceLastBuy', 'CustomerDaysSinceLastBuy']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df.reset_index(drop=True)


def prepare_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and prepare cluster summary data."""
    if df.empty:
        return df
    
    summary_columns = [
        'StockPointID', 'StockPointName', 'TripID', 'ClusterLGAs', 
        'ClusterLCDAs', 'TotalCustonerCount', 'TripTotalQuantity',
        'TripAvgCustomerScore', 'ModifiedDate'
    ]
    
    # Select available columns
    available_cols = [col for col in summary_columns if col in df.columns]
    df_summary = df[available_cols].drop_duplicates().reset_index(drop=True)
    
    # Truncate cluster columns to prevent database field overflow
    if 'ClusterLGAs' in df_summary.columns:
        df_summary['ClusterLGAs'] = df_summary['ClusterLGAs'].str.slice(0, 500)
    if 'ClusterLCDAs' in df_summary.columns:
        df_summary['ClusterLCDAs'] = df_summary['ClusterLCDAs'].str.slice(0, 500)
    
    return df_summary


class RecommendationProcessor:
    """Handles the complete recommendation data processing pipeline."""
    
    def __init__(self, get_connection_func: Callable):
        self.get_connection = get_connection_func
    
    def process(self, stockpoints_data: Dict[str, Any], current_date: str) -> None:
        """
        Process stockpoints data and upsert to database tables.
        
        Args:
            stockpoints_data: Dictionary containing stockpoint recommendation data
            current_date: Current date string for ModifiedDate field
        """
        # Extract and validate data
        df_recommendations = extract_recommendations(stockpoints_data)
        if df_recommendations.empty:
            logger.info("No recommendation data to process")
            return
        
        df_recommendations['ModifiedDate'] = current_date
        
        # Process main recommendations table
        self._process_recommendations_table(df_recommendations)
        
        # Process cluster summary table
        self._process_cluster_summary_table(df_recommendations)
        
        logger.info("Recommendation processing completed successfully")
    
    def _process_recommendations_table(self, df: pd.DataFrame) -> None:
        """Process and upsert main recommendations data."""
        df_prepared = prepare_recommendation_data(df, df['ModifiedDate'].iloc[0])
        
        match_cols = ['StockPointID', 'CustomerID', 'SKUID', 'ModifiedDate']
        update_cols = [col for col in df_prepared.columns if col not in match_cols]
        
        with database_connection(self.get_connection) as conn:
            upsert_dataframe(
                df=df_prepared,
                table_name='dailyPredictedPull',
                conn=conn,
                match_cols=match_cols,
                update_cols=update_cols
            )
    
    def _process_cluster_summary_table(self, df: pd.DataFrame) -> None:
        """Process and upsert cluster summary data."""
        df_summary = prepare_cluster_summary(df)
        
        match_cols = ['StockPointID', 'TripID', 'ModifiedDate']
        update_cols = [col for col in df_summary.columns if col not in match_cols]
        
        with database_connection(self.get_connection) as conn:
            upsert_dataframe(
                df=df_summary,
                table_name='dailyPredictedPullClusterSummary',
                conn=conn,
                match_cols=match_cols,
                update_cols=update_cols,
                fast_executemany=False
            )


# Usage:
def main(stockpoints_data: Dict[str, Any], current_date: str, get_connection_func: Callable) -> None:
    """Main entry point for recommendation processing."""
    processor = RecommendationProcessor(get_connection_func)
    processor.process(stockpoints_data, current_date)