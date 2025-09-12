## PROJECT FLOW

## Step-by-step Code Flow

### 1. Setup & Paths
- Set up output file paths and logger.
- Initialize `FastH3DuckDBManager` for DuckDB operations.

### 2. (Optional) Regenerate H3 Cells & Metadata
- If `regenerate_cell_and_metadata` is `True`:
  - Load admin boundaries.
  - Generate H3 cells (optionally from backup).
  - If `generate_new_cell_metadata_and_save2db` is `True`, add cell metadata to DuckDB.
  - Generate H3 cell addresses and optionally write to DB.

### 3. (Optional) Refresh Input Data
- If `refresh_input_data` is `True`:
  - Refresh base data from source.
  - Preprocess input data.

### 4. Load Processed Data
- Load pre-processed data: LGAs, stock points, mappings, customers, recent customers.

### 5. (Optional) Filter for Pilot
- If `run_pilot` is `True`, filter data for a subset of pilot stock points.

### 6. Clustering
- Initialize `H3SpatialClustererImproved` with loaded data.
- Run `process_all_stock_points` to perform clustering and assignment.

### 7. Save Results
- If `save_all_output_to_disk` is `True`, save clustering results as a pickle file.

### 8. Extract Results for DB
- Extract coverage and assignment DataFrames from clustering results.

### 9. Enhance Coverage Data
- Join coverage with spatial info (centroids, lat/lng) using DuckDB.
- Compute distances between cluster centroids and stock points.

### 10. Save to Database
- If `save_all_output_to_db` is `True`, upsert enhanced coverage and assignment data to DuckDB.

### 11. Prepare Data for Visualization
- Call `postprocess_map_data` for visualization prep.

### 12. (Optional) Migrate to SQL Server
- If `migrate_data_to_sqlserver` is `True`, migrate all tables to SQL Server.

### 13. Finish


## flowchart TD
    A[Start: Setup logger, paths, DB manager]
    B{regenerate_cell_and_metadata?}
    C[Load admin boundaries & generate H3 cells]
    D{generate_new_cell_metadata_and_save2db?}
    E[Add cell metadata to DuckDB]
    F[Generate H3 cell addresses]
    G{refresh_input_data?}
    H[Refresh & preprocess input data]
    I[Load processed data]
    J{run_pilot?}
    K[Filter for pilot stock points]
    L[Run clustering (H3SpatialClustererImproved)]
    M{save_all_output_to_disk?}
    N[Save clustering results to disk]
    O[Extract coverage & assignment DataFrames]
    P[Enhance coverage data (join, compute distances)]
    Q{save_all_output_to_db?}
    R[Upsert coverage & assignment to DuckDB]
    S[Prepare data for visualization]
    T{migrate_data_to_sqlserver?}
    U[Migrate all tables to SQL Server]
    V[Log pipeline completion]

    A --> B
    B -- Yes --> C
    C --> D
    D -- Yes --> E
    D -- No --> F
    E --> F
    F --> G
    B -- No --> G
    G -- Yes --> H
    H --> I
    G -- No --> I
    I --> J
    J -- Yes --> K
    K --> L
    J -- No --> L
    L --> M
    M -- Yes --> N
    N --> O
    M -- No --> O
    O --> P
    P --> Q
    Q -- Yes --> R
    R --> S
    Q -- No --> S
    S --> T
    T -- Yes --> U
    U --> V
    T -- No --> V

## Concept   
### Concept
conceptualization: https://gemini.google.com/share/94762f2d0fc5



## TO DO (SEPT 05)
  [DONE] Average Customer per Beat + Average Active Customers per Beat --- In summary
  [DONE] No. of OAMs per MFC --- New table 
  --- condition on roleName  ---- In Summary
  [DONE] No. of beats per OAM  ---- In Summary

  [DONE] Beat Names to be made like WardName - (number)
    [DONE] Like if one ward has 10 beats, then beat names should be wardName1 - 1, wardName1 - 2, .... , wardName1 - 10
  [DONE] Stock point name on tooltip of Customers

  [DONE] Create a optimized function for input data refresh
  [DONE] Adding direction to h3_cells (beats) generate to SP coverage Average
  [DONE] FIX Customer assignment Load to DuckDB
  [DONE] Improve h3_cells (beats) location/address id


## Tables
### Customers Type
1. Buying/Active Customers: Defined as customers with delivered orders from 2025 to the current date.
1. Recently Activated Customers: Defined as customers whose profiles were created within the last 90 days.

### Customer Assignment Type 
| assignment_type_id | assignment_type | assignment_tier | customer_type |
| :--- | :--- | :--- | :--- |
| 1 | `Assigned Active/Buying` | `h3_inclusion` | `buying customers` |
| 2 | `Unassigned Active/Buying` | `manual_review` | `buying customers` |
| 3 | `Assigned Recently Activated` | `h3_inclusion` | `recently activated` |
| 99 | `Others` | **All Other Combinations** | |
```
### Plot Module
from src.viz.sp_territory_map import plot_territory_polygon_exp, plot_territory_polygon
--- write_disposition="merge",  # Will update existing rows and insert new ones

### DUCK DB TABLES
H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path'] 
TABLE                                  | Description
---------------------------------------|------------------------------------------------------
h3_cells                                | All H3 Cells of Nigeria with Meta data and Address
customer_stockpoint_cluster_assignment  | Customer StockPoint Cluster Assignment / Map 
stockpoint_h3_coverage                  | H3 Cell Coverage of StockPoints 


## SQL SERVER
select top 10 * from VConnectMasterDWR.gis_analysis.customer_stockpoint_cluster_assignment WITH (NOLOCK)
select top 10 * from VConnectMasterDWR.gis_analysis.stockpoint_h3_coverage WITH (NOLOCK)
select top 10 * from VConnectMasterDWR.gis_analysis.h3_cells WITH (NOLOCK)

{'customer_stockpoint_cluster_assignment': {'id': 'BIGINT',
  'stock_point_id': 'BIGINT',
  'customer_id': 'BIGINT',
  'h3_resolution': 'INTEGER',
  'cluster_id': 'VARCHAR',
  'h3_cell_id': 'VARCHAR',
  'assignment_confidence': 'DOUBLE',
  'assignment_tier': 'VARCHAR',
  'customer_type': 'VARCHAR',
  'status': 'VARCHAR',
  'created_date': 'TIMESTAMP',
  'modified_date': 'TIMESTAMP',
  'valid_from': 'TIMESTAMP',
  'valid_to': 'TIMESTAMP',
  'version_number': 'INTEGER',
  'previous_cluster_id': 'VARCHAR',
  'previous_h3_cell_id': 'VARCHAR',
  'previous_h3_resolution': 'INTEGER',
  'previous_customer_type': 'VARCHAR',
  'change_reason': 'VARCHAR',
  'changed_by': 'VARCHAR'},
 'h3_cells': {'h3_index': 'VARCHAR',
  'resolution': 'TINYINT',
  'centroid_lat': 'DOUBLE',
  'centroid_lng': 'DOUBLE',
  'polygon_wkt': 'VARCHAR',
  'boundary_json': 'VARCHAR',
  'latlng_json': 'VARCHAR',
  'polygon_area': 'DOUBLE',
  'num_vertices': 'SMALLINT',
  'error': 'VARCHAR',
  'created_at': 'TIMESTAMP',
  'h3_derived_id': 'VARCHAR',
  'grid_position_id': 'VARCHAR',
  'primary_address_id': 'VARCHAR',
  'country_code': 'VARCHAR',
  'country_name': 'VARCHAR',
  'state_code': 'VARCHAR',
  'state_name': 'VARCHAR',
  'lga_code': 'VARCHAR',
  'lga_name': 'VARCHAR',
  'ward_code': 'VARCHAR',
  'ward_name': 'VARCHAR',
  'confidence_level': 'VARCHAR',
  'coverage_percentage': 'DOUBLE',
  'area_km2': 'DOUBLE'},
 'stockpoint_h3_coverage': {'id': 'BIGINT',
  'stock_point_id': 'BIGINT',
  'h3_cell': 'VARCHAR',
  'h3_resolution': 'INTEGER',
  'cluster_centroid_lat': 'DOUBLE',
  'cluster_centroid_lng': 'DOUBLE',
  'cluster_sp_dist_km': 'DOUBLE'},
 'stockpoint_h3_coverage_log': {'log_id': 'BIGINT',
  'stock_point_id': 'BIGINT',
  'h3_cell': 'VARCHAR',
  'h3_resolution': 'INTEGER',
  'operation': 'VARCHAR',
  'old_cluster_centroid_lat': 'DOUBLE',
  'old_cluster_centroid_lng': 'DOUBLE',
  'old_cluster_sp_dist_km': 'DOUBLE',
  'new_cluster_centroid_lat': 'DOUBLE',
  'new_cluster_centroid_lng': 'DOUBLE',
  'new_cluster_sp_dist_km': 'DOUBLE',
  'changed_at': 'TIMESTAMP'}}







