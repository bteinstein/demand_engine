import logging
import sys
from contextlib import contextmanager
import os

@contextmanager
def suppress_stdout():
    """Context manager to temporarily suppress print output."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
            
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np # For creating example NaN data
from typing import Union, List

def validate_coordinate_within_nigeria(lat: float, lng: float) -> bool:
    """
    Validates if a given latitude and longitude fall within Nigeria's
    approximate geographical bounds.

    Args:
        lat (float): The latitude of the point.
        lng (float): The longitude of the point.

    Returns:
        bool: True if coordinates are within Nigeria bounds and not NaN, False otherwise.
    """
    nigeria_bounds = {
        'min_lat': 4.0,   'max_lat': 14.0,
        'min_lng': 2.5,   'max_lng': 15.0
    }
    
    # Ensure lat/lng are not NaN before comparison
    if pd.isna(lat) or pd.isna(lng):
        return False

    return (nigeria_bounds['min_lat'] <= lat <= nigeria_bounds['max_lat'] and
            nigeria_bounds['min_lng'] <= lng <= nigeria_bounds['max_lng'])

def clean_customer_gdf_coordinates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Cleans a GeoDataFrame of customer data by filtering out records with
    invalid geometries or coordinates outside Nigeria.

    Args:
        customers_gdf (gpd.GeoDataFrame): The input GeoDataFrame with customer data,
                                         expected to have a 'geometry' column of Point types.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with cleaned customer data.
    """
    original_count = len(gdf)
    
    # Step 1: Drop rows where 'geometry' itself is NaN/None
    customers_cleaned_geom = gdf.dropna(subset=['geometry']).copy()
    dropped_geom_count = original_count - len(customers_cleaned_geom)
    if dropped_geom_count > 0:
        print(f"⚠️ Warning: Dropped {dropped_geom_count:,} customer records due to missing geometry.")

    # Step 2: Validate coordinates using the standalone helper function
    # We apply the validation function to each geometry's coordinates
    valid_coords_mask = customers_cleaned_geom.geometry.apply(
        lambda p: validate_coordinate_within_nigeria(p.y, p.x) if p is not None else False
    )
    
    customers_final = customers_cleaned_geom[valid_coords_mask].copy()
    dropped_oob_count = len(customers_cleaned_geom) - len(customers_final)
    if dropped_oob_count > 0:
        print(f"⚠️ Warning: Dropped {dropped_oob_count:,} customer records with coordinates outside Nigeria.")
        
    print(f"✅ Customer data cleaning complete. {len(customers_final):,} of {original_count:,} records retained.")
    return customers_final
            
            
       
def filter_cluster_result_dict(results: dict, pilot_sps_lists: list):
    filtered_result = {}
    for outer_key, inner_dict in results.items():
        if outer_key in ['territories', 'grid_results', 'assignments', 'optimized_clusters']:
            # Initialize the inner dictionary if it doesn't exist
            if outer_key not in filtered_result:
                filtered_result[outer_key] = {}
            for sp_key, value in inner_dict.items():
                if sp_key in pilot_sps_lists:
                    filtered_result[outer_key][sp_key] = value
        else:
            filtered_result[outer_key] = inner_dict
    
    return filtered_result            
            
            
import pandas as pd 
from geopy.distance import geodesic
from geopy.exc import GeopyError

def calculate_distance_km(lat1, lng1, lat2, lng2, verbose = False):
    try:
        # Check if the input values are valid numbers
        if not all(isinstance(coord, (int, float)) for coord in [lat1, lng1, lat2, lng2]):
            raise ValueError("All coordinates must be numeric.")

        # Check if the latitude and longitude values are within valid ranges
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
            raise ValueError("Latitude values must be between -90 and 90.")
        if not (-180 <= lng1 <= 180 and -180 <= lng2 <= 180):
            raise ValueError("Longitude values must be between -180 and 180.")

        # Use the geodesic function to calculate the distance
        distance = geodesic((lat1, lng1), (lat2, lng2)).km
        return distance

    except GeopyError as e:
        if verbose:
            print(f"An error occurred while calculating the distance: {e}")
        return None
    except ValueError as ve:
        if verbose:
            print(f"Invalid input: {ve}")
        return None
    except Exception as ex:
        if verbose:
            print(f"An unexpected error occurred: {ex}")
        return None


import numpy as np
import pandas as pd

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance calculation in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in km
    r = 6371
    return c * r

def safe_runs(func):
    try:
        return func
    except Exception as e:
        print(e)
  

import numpy as np
import pandas as pd

def get_simplified_direction_vectorized(lat1, lon1, lat2, lon2, detailed=False):
    """
    Vectorized calculation of simplified 8-point or detailed 16-point compass direction.
    
    Args:
        lat1 (np.array): Starting latitudes in degrees.
        lon1 (np.array): Starting longitudes in degrees.
        lat2 (np.array): Destination latitudes in degrees.
        lon2 (np.array): Destination longitudes in degrees.
        detailed (bool, optional): If True, returns 16-point directions.
                                  Defaults to False, for 8-point directions.

    Returns:
        np.array: An array of simplified or detailed direction strings.
    """
    # Convert all coordinates to radians for the bearing calculation
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)

    # Vectorized bearing formula
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)

    # Convert bearing to degrees and normalize to a 0-360 range
    bearing_degrees = (np.degrees(bearing) + 360) % 360

    if detailed:
        # Define the 16 directions and calculate the index
        directions = np.array([
            'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
        ])
        # The // operator performs vectorized floor division
        idx = (bearing_degrees + 11.25) // 22.5
        # Use the calculated indices to get the direction string
        return directions[idx.astype(int) % 16]
    else:
        # Define the 8 directions and calculate the index
        directions = np.array(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        # The // operator performs vectorized floor division
        idx = (bearing_degrees + 22.5) // 45
        # Use the calculated indices to get the direction string
        return directions[idx.astype(int) % 8]


# # --- Example Usage with a DataFrame ---

# # Create a sample DataFrame
# data = {
#     'start_lat': [40.7128, 34.0522, 51.5074, 38.8951],
#     'start_lon': [-74.0060, -118.2437, -0.1278, -77.0364],
#     'end_lat': [34.0522, 40.7128, 38.8951, 51.5074],
#     'end_lon': [-118.2437, -74.0060, -77.0364, -0.1278],
# }
# df = pd.DataFrame(data)

# # Call the vectorized function for 8-point direction
# df['direction_8pt'] = get_simplified_direction_vectorized(
#     df['start_lat'],
#     df['start_lon'],
#     df['end_lat'],
#     df['end_lon'],
#     detailed=False
# )

# # Call the vectorized function for 16-point direction
# df['direction_16pt'] = get_simplified_direction_vectorized(
#     df['start_lat'],
#     df['start_lon'],
#     df['end_lat'],
#     df['end_lon'],
#     detailed=True
# )

# # Print the result
# print(df)

            
            
            
            
            
            
            