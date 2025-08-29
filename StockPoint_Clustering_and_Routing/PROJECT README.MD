
conceptualization: https://gemini.google.com/share/94762f2d0fc5

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