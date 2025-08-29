As a senior data scientist skilled in GIS, Python, route optimization, and Uber H3, your task is to develop a concise, phased Python implementation plan for the route optimization problem outlined in [Route Optimization Task](https://gist.github.com/bteinstein/927e5982ff78cb5fac8f1f068978bceb), following the structure in [Implementation Plan](https://gist.github.com/bteinstein/c82be51b6e0d712b613a771695ab201f).

Task: Implement optimized delivery routes from a fulfillment center using H3 cells (resolution 8) for customer clusters within a boundary polygon, minimizing travel distance, prioritizing high-density clusters, and adhering to constraints.

Here are the issue with the implementation:
- The dbscan clustering of optimize_route_clustering is performing very poorly with 0 split  
- The route split and merging might not gurantee geographically compactness  and max customer per route constraint

Implementation Requirements:
- Structure the implementation in clear, concise phases aligned with the referenced plan.
- Address:
  1. Population density: Customer count per H3 cell (from dataset).
  2. Cluster distance: Haversine distance from H3 cell centroid (or farthest vertex) to the fulfillment center.
  3. Constraints: Routes within 0–7 km (adjusted for retailer density), 40–300 retailers per route, and geographically compact.
- Use datasets:
  1. `sp_dim_df`: Columns: `stock_point_id`, `stock_point_name`, `latitude`, `longitude`
  2. `customers_gdf`: Columns: `customer_id`, `longitude`, `latitude`, `geometry` (Shapely POINT: lon, lat)
  3. `df_output_assignment`: Columns: `stock_point_id`, `customer_id`, `cluster_id`, `h3_cell_id`, `assignment_confidence`, `assignment_tier` (`cluster_id` = `h3_cell_id`)
- Output: A long-format DataFrame with columns: `route_id`, `h3_cell_id`, `customer_count`, `total_distance_km`, `estimated_delivery_time_hours`, `compactness_score`.

Deliverable: Provide a Python-based implementation script with concise, actionable phases to generate optimized, constraint-compliant routes using the provided datasets, following the referenced gists: https://gist.github.com/bteinstein/927e5982ff78cb5fac8f1f068978bceb and https://gist.github.com/bteinstein/c82be51b6e0d712b613a771695ab201f.  