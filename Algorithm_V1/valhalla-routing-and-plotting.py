import folium
from routingpy import Valhalla
import os
import itertools # Used for cycling through colors

# --- Configuration ---
# IMPORTANT: Choose your Valhalla API endpoint.
# For this example, we'll use your self-hosted instance.
VALHALLA_BASE_URL = "http://localhost:8002" # Pointing to your self-hosted Valhalla
VALHALLA_API_KEY = "" # No API key needed for your self-hosted instance

# Define a palette of colors for different trips
ROUTE_COLORS = itertools.cycle([
    'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
    'darkblue', 'lightblue', 'darkgreen', 'cadetblue', 'darkpurple',
    'pink', 'lightgreen', 'black', 'lightgray'
])

def get_valhalla_routes_info(trip_data: dict) -> list:
    """
    Calculates routes for each trip using Valhalla and extracts route information.

    Args:
        trip_data (dict): A dictionary containing 'StockPointCoord' and 'Trips' information.
                          Example structure:
                          {
                              'StockPointName': '<StockPointName>',
                              'StockPointCoord': [lon, lat],
                              'Trips': [
                                  {'TripID': 1, 'Destinations': [{'CustomerID': 'CU1', 'Coordinate': [lon, lat]}, ...]},
                                  ...
                              ]
                          }

    Returns:
        list: A list of dictionaries, where each dictionary contains route details for a trip:
              {'TripID': int, 'geometry_lonlat': list, 'distance': float, 'duration': float, 'color': str}
    """
    stock_point_coord = trip_data['StockPointCoord']
    trips = trip_data['Trips']

    client = Valhalla(base_url=VALHALLA_BASE_URL)
    if VALHALLA_API_KEY:
        client = Valhalla(base_url=VALHALLA_BASE_URL, api_key=VALHALLA_API_KEY)

    routes_info = []
    print("Starting route calculations for all trips...")

    for trip in trips:
        trip_id = trip['TripID']
        destinations_coords = [d['Coordinate'] for d in trip['Destinations']]
        
        # All waypoints for this specific trip: StockPoint -> Destination1 -> Destination2 -> ...
        all_waypoints_for_trip = [stock_point_coord] + destinations_coords

        try:
            print(f"  Calculating route for Trip ID: {trip_id} ({len(all_waypoints_for_trip)} waypoints)...")
            route_response = client.directions(
                locations=all_waypoints_for_trip,
                profile='auto', # 'auto' for car, could be 'bicycle', 'pedestrian', 'truck'
            )
            
            routes_info.append({
                'TripID': trip_id,
                'outlet_count': len(destinations_coords),
                'geometry_lonlat': route_response.geometry, # List of [lon, lat] pairs
                'distance': route_response.distance, # in meters
                'duration': route_response.duration, # in seconds
                'color': next(ROUTE_COLORS) # Assign a unique color to this trip
            })
            print(f"  Trip ID {trip_id} calculated. Distance: {route_response.distance / 1000:.2f} km, Duration: {route_response.duration / 60:.2f} min")

        except Exception as e:
            print(f"  Error calculating route for Trip ID {trip_id}: {e}")
            routes_info.append({
                'TripID': trip_id,
                'error': str(e)
            })
    print("All route calculations completed.\n")
    return routes_info

def plot_routes_on_map(trip_data: dict, routes_info: list, output_filename: str = "valhalla_trips_map.html"):
    """
    Plots the stock point, destinations, and calculated routes on an interactive Folium map.

    Args:
        trip_data (dict): The original dictionary containing stock point and trip details.
        routes_info (list): A list of dictionaries, each containing route details for a trip,
                            as returned by get_valhalla_routes_info.
        output_filename (str): The name of the HTML file to save the map to.
    """
    stock_point_name = trip_data.get('StockPointName', 'Stock Point')
    stock_point_coord = trip_data['StockPointCoord']

    # Initialize map centered on the stock point
    # Folium expects [latitude, longitude] for map center
    m = folium.Map(location=[stock_point_coord[1], stock_point_coord[0]], zoom_start=11)

    # Add marker for the stock point
    folium.Marker(
        location=[stock_point_coord[1], stock_point_coord[0]],
        popup=f"<b>{stock_point_name}</b>",
        tooltip=stock_point_name,
        icon=folium.Icon(color='green', icon='warehouse', prefix='fa') # Using a warehouse icon
    ).add_to(m)

    # Add routes and destination markers
    for route_info in routes_info:
        if 'error' in route_info:
            print(f"Skipping plotting for Trip ID {route_info['TripID']} due to error: {route_info['error']}")
            continue

        trip_id = route_info['TripID']
        outlet_count = route_info['outlet_count']
        route_geometry_lonlat = route_info['geometry_lonlat']
        route_color = route_info['color']
        distance_km = route_info['distance'] / 1000
        duration_min = route_info['duration'] / 60

        # Convert route geometry from [lon, lat] to [lat, lon] for Folium
        route_geometry_latlon = [[lat, lon] for lon, lat in route_geometry_lonlat]

        # Add the route polyline to the map
        folium.PolyLine(
            locations=route_geometry_latlon,
            color=route_color,
            weight=5,
            opacity=0.7,
            tooltip=f"Trip {trip_id} <br>Outlets: {outlet_count} <br>Dist: {distance_km:.2f} km<br>Dur: {duration_min:.2f} min"
        ).add_to(m)

        # Find the destinations for the current trip to add markers
        current_trip_destinations = None
        for t in trip_data['Trips']:
            if t['TripID'] == trip_id:
                current_trip_destinations = t['Destinations']
                break

        if current_trip_destinations:
            for i, dest_data in enumerate(current_trip_destinations):
                customer_id = dest_data['CustomerID']
                dest_lonlat = dest_data['Coordinate']
                folium.Marker(
                    location=[dest_lonlat[1], dest_lonlat[0]], # Folium needs [lat, lon]
                    popup=f"<b>Trip {trip_id} - {customer_id}</b>",
                    tooltip=f"Dest {i+1} for Trip {trip_id}",
                    icon=folium.Icon(color=route_color, icon='map-pin', prefix='fa') # Use trip color for destination
                ).add_to(m)
        
        else:
            print(f"Warning: Could not find original destination data for Trip ID {trip_id}.")


    # Fit the map to the bounds of all plotted features
    # This ensures all routes and markers are visible
    m.fit_bounds(m.get_bounds())

    # Save the map to an HTML file
    m.save(output_filename)
    print(f"Interactive map saved to: {os.path.abspath(output_filename)}")

# --- Main execution flow ---
if __name__ == "__main__":
    # Sample data structure as provided
    sample_trip_data = {
        'StockPointName': 'Lagos Depot',
        'StockPointCoord': [3.27523, 6.563538], # Longitude, Latitude
        'Trips': [
            {
                'TripID': 1,
                'Destinations': [
                    {'CustomerID': 'CU1', 'Coordinate': [3.2713068, 6.5151952]},
                    {'CustomerID': 'CU2', 'Coordinate': [3.2663042, 6.5090189]},
                    {'CustomerID': 'CU3', 'Coordinate': [3.2597081, 6.5164545]},
                    {'CustomerID': 'CU4', 'Coordinate': [3.2701802, 6.523793]}
                ]
            },
            {
                'TripID': 2,
                'Destinations': [
                    {'CustomerID': 'CU6', 'Coordinate': [3.2548355, 6.5156041]},
                    {'CustomerID': 'CU7', 'Coordinate': [3.2568217, 6.5025609]},
                    {'CustomerID': 'CU8', 'Coordinate': [3.2642244, 6.522825]},
                    {'CustomerID': 'CU9', 'Coordinate': [3.2829371, 6.5137854]}
                ]
            },
            { # Adding a third trip with 30 destinations to test the self-hosted limit (total 31 waypoints)
                'TripID': 3,
                'Destinations': [
                    {'CustomerID': 'CU10', 'Coordinate': [3.279199004, 6.498042956]},
                    {'CustomerID': 'CU11', 'Coordinate': [3.2687817, 6.5005307]},
                    {'CustomerID': 'CU12', 'Coordinate': [3.2562934, 6.4986689]},
                    {'CustomerID': 'CU13', 'Coordinate': [3.2697276, 6.5266798]},
                    {'CustomerID': 'CU14', 'Coordinate': [3.2803165, 6.5167863]},
                    {'CustomerID': 'CU15', 'Coordinate': [3.2526794, 6.4972678]},
                    {'CustomerID': 'CU16', 'Coordinate': [3.2614767, 6.5030528]},
                    {'CustomerID': 'CU17', 'Coordinate': [3.2526125, 6.4990162]},
                    {'CustomerID': 'CU18', 'Coordinate': [3.2569118, 6.4986625]},
                    {'CustomerID': 'CU19', 'Coordinate': [3.2548355, 6.5156041]},
                    {'CustomerID': 'CU20', 'Coordinate': [3.2568217, 6.5025609]},
                    {'CustomerID': 'CU21', 'Coordinate': [3.2642244, 6.522825]},
                    {'CustomerID': 'CU22', 'Coordinate': [3.2829371, 6.5137854]},
                    {'CustomerID': 'CU23', 'Coordinate': [3.2829371, 6.5137854]}, # Duplicate to reach higher count
                    {'CustomerID': 'CU24', 'Coordinate': [3.279199004, 6.498042956]},
                    {'CustomerID': 'CU25', 'Coordinate': [3.2687817, 6.5005307]},
                    {'CustomerID': 'CU26', 'Coordinate': [3.2562934, 6.4986689]},
                    {'CustomerID': 'CU27', 'Coordinate': [3.2697276, 6.5266798]},
                    {'CustomerID': 'CU28', 'Coordinate': [3.2803165, 6.5167863]},
                    {'CustomerID': 'CU29', 'Coordinate': [3.2526794, 6.4972678]},
                    {'CustomerID': 'CU30', 'Coordinate': [3.2614767, 6.5030528]},
                    {'CustomerID': 'CU31', 'Coordinate': [3.2526125, 6.4990162]},
                    {'CustomerID': 'CU32', 'Coordinate': [3.2569118, 6.4986625]},
                    {'CustomerID': 'CU33', 'Coordinate': [3.2548355, 6.5156041]},
                    {'CustomerID': 'CU34', 'Coordinate': [3.2568217, 6.5025609]},
                    {'CustomerID': 'CU35', 'Coordinate': [3.2642244, 6.522825]},
                    {'CustomerID': 'CU36', 'Coordinate': [3.2829371, 6.5137854]},
                    {'CustomerID': 'CU37', 'Coordinate': [3.279199004, 6.498042956]},
                    {'CustomerID': 'CU38', 'Coordinate': [3.2687817, 6.5005307]},
                    {'CustomerID': 'CU39', 'Coordinate': [3.2562934, 6.4986689]} # Total 30 destinations (+1 stock point = 31 total waypoints)
                ]
            }
        ]
    }

    # Step 1: Get route information for all trips
    calculated_routes_info = get_valhalla_routes_info(sample_trip_data)

    # Step 2: Plot all routes on a map
    plot_routes_on_map(trip_data=sample_trip_data, routes_info=calculated_routes_info, output_filename = "valhalla_trips_map.html")

