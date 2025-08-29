

import pandas as pd
import numpy as np
import h3
import folium
import requests
import json
import math
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

class ValhallaH3Router:
    def __init__(self, df, valhalla_url, max_cells_per_route=6, max_route_time_minutes=240):
        """
        Initialize Valhalla-based H3 router
        
        Args:
            df: DataFrame with H3 data
            valhalla_url: Valhalla API endpoint (e.g., 'http://localhost:8002')
            max_cells_per_route: Maximum cells per delivery route
            max_route_time_minutes: Maximum total route time in minutes
        """
        self.df = df.copy()
        self.valhalla_url = valhalla_url.rstrip('/')
        self.max_cells = max_cells_per_route
        self.max_time = max_route_time_minutes
        
        self.stock_point = {
            'lat': df['sp_latitude'].iloc[0],
            'lng': df['sp_longitude'].iloc[0],
            'name': df['stock_point_name'].iloc[0]
        }
        
        # Add delivery time per cell (configurable)
        self.delivery_time_minutes = 15  # Average time per H3 cell delivery
        
    def calculate_travel_matrix(self):
        """Calculate travel time matrix using Valhalla"""
        locations = []
        
        # Add stock point first
        locations.append({
            "lat": self.stock_point['lat'],
            "lon": self.stock_point['lng']
        })
        
        # Add H3 cell centroids
        for _, row in self.df.iterrows():
            locations.append({
                "lat": row['cluster_centroid_lat'],
                "lon": row['cluster_centroid_lng']
            })
        
        # Valhalla matrix request
        matrix_request = {
            "sources": list(range(len(locations))),
            "targets": list(range(len(locations))),
            "locations": locations,
            "costing": "auto",
            "costing_options": {
                "auto": {
                    "country_crossing_cost": 600,
                    "country_crossing_penalty": 0
                }
            },
            "units": "minutes"
        }
        
        try:
            response = requests.post(
                f"{self.valhalla_url}/sources_to_targets",
                json=matrix_request,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            travel_times = np.array(result['sources_to_targets'])
            
            # Handle unreachable locations
            travel_times[travel_times == None] = 999999
            
            return travel_times
            
        except Exception as e:
            print(f"Valhalla API error: {e}")
            # Fallback to Euclidean distance approximation
            return self._fallback_distance_matrix()
    
    def _fallback_distance_matrix(self):
        """Fallback distance matrix if Valhalla fails"""
        n = len(self.df) + 1
        matrix = np.zeros((n, n))
        
        coords = [(self.stock_point['lat'], self.stock_point['lng'])]
        coords.extend([(row['cluster_centroid_lat'], row['cluster_centroid_lng']) 
                      for _, row in self.df.iterrows()])
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Convert km to approximate minutes (assuming 30 km/h average)
                    dist_km = self._haversine_distance(coords[i], coords[j])
                    matrix[i][j] = dist_km * 2  # 2 minutes per km
                    
        return matrix
    
    def _haversine_distance(self, coord1, coord2):
        """Calculate distance between two coordinates"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        R = 6371  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    def create_valhalla_routes(self):
        """Create routes using Valhalla travel times"""
        # Get travel time matrix
        travel_matrix = self.calculate_travel_matrix()
        
        # Extract times from stock point to all cells (first row, excluding stock point)
        stock_to_cells = travel_matrix[0, 1:]
        
        # Extract cell-to-cell travel times
        cell_matrix = travel_matrix[1:, 1:]
        
        routes = []
        route_id = 1
        unassigned_cells = list(range(len(self.df)))
        
        while unassigned_cells:
            route = self._build_single_route(unassigned_cells, stock_to_cells, cell_matrix)
            
            if route['cells']:
                routes.append({
                    'route_id': route_id,
                    'cell_indices': route['cells'],
                    'total_time': route['total_time'],
                    'cells_data': [self.df.iloc[i] for i in route['cells']]
                })
                route_id += 1
                
                # Remove assigned cells
                for cell_idx in route['cells']:
                    unassigned_cells.remove(cell_idx)
            else:
                # Handle remaining cells that can't form valid routes
                if unassigned_cells:
                    remaining = unassigned_cells[:self.max_cells]
                    routes.append({
                        'route_id': route_id,
                        'cell_indices': remaining,
                        'total_time': self._calculate_route_time(remaining, stock_to_cells, cell_matrix),
                        'cells_data': [self.df.iloc[i] for i in remaining]
                    })
                    unassigned_cells = [c for c in unassigned_cells if c not in remaining]
                    route_id += 1
        
        self.routes = routes
        return routes
    
    def _build_single_route(self, available_cells, stock_times, cell_matrix):
        """Build a single optimized route"""
        if not available_cells:
            return {'cells': [], 'total_time': 0}
        
        # Start with nearest cell to stock point
        start_cell = min(available_cells, key=lambda x: stock_times[x])
        route_cells = [start_cell]
        route_time = stock_times[start_cell] + self.delivery_time_minutes
        
        remaining_cells = [c for c in available_cells if c != start_cell]
        
        # Greedy nearest neighbor with time constraints
        while remaining_cells and len(route_cells) < self.max_cells:
            current_cell = route_cells[-1]
            
            # Find nearest unvisited cell
            nearest_cell = min(remaining_cells, 
                             key=lambda x: cell_matrix[current_cell][x])
            
            # Calculate time to add this cell
            travel_time = cell_matrix[current_cell][nearest_cell]
            delivery_time = self.delivery_time_minutes
            return_time = stock_times[nearest_cell]  # Time to return to stock
            
            total_new_time = route_time + travel_time + delivery_time + return_time
            
            if total_new_time <= self.max_time:
                route_cells.append(nearest_cell)
                route_time += travel_time + delivery_time
                remaining_cells.remove(nearest_cell)
            else:
                break
        
        # Add return time to stock point
        if route_cells:
            route_time += stock_times[route_cells[-1]]
        
        return {'cells': route_cells, 'total_time': route_time}
    
    def _calculate_route_time(self, cell_indices, stock_times, cell_matrix):
        """Calculate total time for a route"""
        if not cell_indices:
            return 0
        
        total_time = stock_times[cell_indices[0]]  # To first cell
        
        for i in range(len(cell_indices) - 1):
            total_time += cell_matrix[cell_indices[i]][cell_indices[i+1]]
            total_time += self.delivery_time_minutes
        
        total_time += self.delivery_time_minutes  # Last delivery
        total_time += stock_times[cell_indices[-1]]  # Return to stock
        
        return total_time
    
    def get_optimized_route_sequence(self, cell_indices):
        """Get optimized sequence for a route using Valhalla"""
        if len(cell_indices) <= 2:
            return cell_indices
        
        locations = []
        
        # Add stock point
        locations.append({
            "lat": self.stock_point['lat'],
            "lon": self.stock_point['lng']
        })
        
        # Add route cells
        for idx in cell_indices:
            row = self.df.iloc[idx]
            locations.append({
                "lat": row['cluster_centroid_lat'],
                "lon": row['cluster_centroid_lng']
            })
        
        # Add stock point as end
        locations.append({
            "lat": self.stock_point['lat'],
            "lon": self.stock_point['lng']
        })
        
        # Valhalla optimization request
        optimization_request = {
            "locations": locations,
            "costing": "auto",
            "costing_options": {
                "auto": {
                    "country_crossing_cost": 600
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.valhalla_url}/optimized_route",
                json=optimization_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract waypoint order (excluding start/end stock points)
                waypoint_order = [wp['waypoint_index'] - 1 
                                for wp in result['trip']['legs'] 
                                if 0 < wp['waypoint_index'] < len(locations) - 1]
                
                return [cell_indices[i] for i in waypoint_order]
            
        except:
            pass
        
        return cell_indices  # Return original order if optimization fails
    
    def create_output_dataframe(self):
        """Create output dataframe with route assignments"""
        output_rows = []
        
        for route in self.routes:
            for i, cell_idx in enumerate(route['cell_indices']):
                row = self.df.iloc[cell_idx]
                
                # Calculate bearing for direction
                bearing = self._calculate_bearing(
                    self.stock_point['lat'], self.stock_point['lng'],
                    row['cluster_centroid_lat'], row['cluster_centroid_lng']
                )
                
                output_rows.append({
                    'h3_cell': row['h3_cell'],
                    'route_id': route['route_id'],
                    'sequence_order': i + 1,
                    'bearing': bearing,
                    'direction': self._get_direction_label(bearing),
                    'cluster_centroid_lat': row['cluster_centroid_lat'],
                    'cluster_centroid_lng': row['cluster_centroid_lng'],
                    'cluster_sp_dist_km': row['cluster_sp_dist_km'],
                    'estimated_route_time_minutes': route['total_time']
                })
        
        return pd.DataFrame(output_rows)
    
    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate bearing between two points"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.atan2(y, x)
        return (math.degrees(bearing) + 360) % 360
    
    def _get_direction_label(self, bearing):
        """Convert bearing to direction label"""
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = int((bearing + 22.5) // 45) % 8
        return directions[idx]
    
    def generate_route_summary(self):
        """Generate summary statistics"""
        if not hasattr(self, 'routes'):
            return {}
        
        total_cells = sum(len(route['cell_indices']) for route in self.routes)
        
        return {
            'total_routes': len(self.routes),
            'total_cells': total_cells,
            'avg_cells_per_route': total_cells / len(self.routes) if self.routes else 0,
            'avg_route_time_minutes': np.mean([r['total_time'] for r in self.routes]),
            'max_route_time_minutes': max([r['total_time'] for r in self.routes]) if self.routes else 0,
            'routes_over_time_limit': sum(1 for r in self.routes if r['total_time'] > self.max_time)
        }

def create_valhalla_delivery_routes(df, valhalla_url, max_cells_per_route=6, max_route_time_minutes=240):
    """
    Main function to create Valhalla-optimized delivery routes
    
    Args:
        df: Input DataFrame with H3 data
        valhalla_url: Valhalla API endpoint
        max_cells_per_route: Maximum cells per route
        max_route_time_minutes: Maximum route duration
    """
    
    # Initialize router
    router = ValhallaH3Router(df, valhalla_url, max_cells_per_route, max_route_time_minutes)
    
    # Create routes
    routes = router.create_valhalla_routes()
    
    # Create output dataframe
    output_df = router.create_output_dataframe()
    
    # Generate summary
    summary = router.generate_route_summary()
    
    return {
        'output_df': output_df,
        'routes': routes,
        'summary': summary,
        'router': router
    }

# Example usage
"""
# Setup
df_input = pd.read_csv('your_h3_data.csv')
valhalla_url = 'http://localhost:8002'  # Your Valhalla instance

# Create routes
result = create_valhalla_delivery_routes(
    df=df_input,
    valhalla_url=valhalla_url,
    max_cells_per_route=6,
    max_route_time_minutes=240  # 4 hours
)

# Results
output_df = result['output_df']
summary = result['summary']

print("Route Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")

# Save results
output_df.to_csv('valhalla_routes.csv', index=False)
"""