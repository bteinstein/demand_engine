import folium
import pandas as pd
from folium import plugins
import os
import seaborn as sns

class ClusterRouteMapper:
    """
    A class to create an enhanced Folium map with clustered markers, routing polylines,
    and interactive controls.

    Attributes:
        df (pd.DataFrame): The main DataFrame containing location and cluster data.
        lat_col (str): Column name for latitude.
        lon_col (str): Column name for longitude.
        cluster_col (str): Column name for cluster IDs.
        lga_col (str): Column name for Local Government Area.
        lcda_col (str): Column name for Local Council Development Area.
        popup_cols (list): List of column names to include in popups.
        tooltip_cols (list): List of column names to include in tooltips.
        zoom_start (int): Initial zoom level for the map.
        tiles (str): Default map tile style.
        marker_params (dict): Dictionary for Folium CircleMarker parameters.
        cluster_palette (str): Seaborn color palette for clusters.
        route_palette (str): Seaborn color palette for routes.
        legend_limit (int): Max number of entries in the legend.
        trip_data (dict): Dictionary with stock point and trip details.
        routes_info (list): List of dictionaries with route geometries and info.
        output_filename (str): Name of the HTML file to save the map.
        route_params (dict): Dictionary for route PolyLine parameters.
        stock_marker_params (dict): Dictionary for stock point Marker parameters.
        chunk_size (int): Number of markers to process per chunk for large datasets.

        _cluster_map (folium.Map): The Folium map object.
        _unique_clusters (list): Sorted list of unique cluster IDs.
        _cluster_colors (list): Generated colors for clusters.
        _route_colors (list): Generated colors for routes.
        _cluster_color_map (dict): Mapping of cluster ID to color.
        _cluster_groups (dict): Dictionary of Folium FeatureGroups for each cluster.
        _cluster_stats (dict): Statistics for each cluster.
    """

    def __init__(self, df, lat_col='Latitude', lon_col='Longitude',
                 cluster_col='cluster', lga_col='LGA', lcda_col='LCDA',
                 popup_cols=None, tooltip_cols=None, zoom_start=10,
                 tiles='cartodb positron', marker_radius=5, marker_fill_opacity=0.8,
                 marker_stroke_width=1, marker_stroke_opacity=1.0,
                 cluster_palette='husl', route_palette='tab10', legend_limit=15,
                 trip_data=None, routes_info=None, output_filename="cluster_route_map.html",
                 route_weight=5, route_opacity=0.7, stock_icon='warehouse',
                 stock_color='green', stock_prefix='fa', chunk_size=None):
        """
        Initializes the ClusterRouteMapper with data and configuration.
        """
        self.df = df.copy()
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.cluster_col = cluster_col
        self.lga_col = lga_col
        self.lcda_col = lcda_col
        self.popup_cols = popup_cols if popup_cols is not None else []
        self.tooltip_cols = tooltip_cols if tooltip_cols is not None else []
        self.zoom_start = zoom_start
        self.tiles = tiles
        self.output_filename = output_filename
        self.cluster_palette = cluster_palette
        self.route_palette = route_palette
        self.legend_limit = legend_limit
        self.trip_data = trip_data if trip_data is not None else {}
        self.routes_info = routes_info if routes_info is not None else []
        self.chunk_size = chunk_size

        # Marker parameters for flexibility
        self.marker_params = {
            'radius': marker_radius,
            'fill_opacity': marker_fill_opacity,
            'stroke_width': marker_stroke_width,
            'stroke_opacity': marker_stroke_opacity,
            'color': 'white', # Default stroke color for markers
            'fill': True
        }

        # Route parameters
        self.route_params = {
            'weight': route_weight,
            'opacity': route_opacity,
        }

        # Stock marker parameters
        self.stock_marker_params = {
            'icon': stock_icon,
            'color': stock_color,
            'prefix': stock_prefix
        }

        # Internal state variables
        self._cluster_map = None
        self._unique_clusters = []
        self._cluster_colors = []
        self._route_colors = []
        self._cluster_color_map = {}
        self._cluster_groups = {}
        self._cluster_stats = {}

        self._validate_input_data()
        self._prepare_color_and_cluster_data()

    def _validate_input_data(self):
        """Validate that the required columns exist in the DataFrame."""
        required_cols = [self.lat_col, self.lon_col, self.cluster_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure popup_cols and tooltip_cols exist if provided
        for col_list, name in zip([self.popup_cols, self.tooltip_cols], ["popup_cols", "tooltip_cols"]):
            if col_list:
                missing_optional_cols = [col for col in col_list if col not in self.df.columns]
                if missing_optional_cols:
                    print(f"Warning: Missing optional columns in {name}: {missing_optional_cols}. These will be ignored.")
                    # Filter out missing columns
                    if name == "popup_cols":
                        self.popup_cols = [col for col in self.popup_cols if col not in missing_optional_cols]
                    else:
                        self.tooltip_cols = [col for col in self.tooltip_cols if col not in missing_optional_cols]


    def _calculate_map_bounds(self):
        """Calculate map bounds and center for initial view."""
        if self.df.empty:
            # Default to a generic location (e.g., Lagos, Nigeria) if no data
            return 6.5244, 3.3792, [[6.0, 3.0], [7.0, 3.8]] # [lat, lon], [SW, NE]
            
        lat_min, lat_max = self.df[self.lat_col].min(), self.df[self.lat_col].max()
        lon_min, lon_max = self.df[self.lon_col].min(), self.df[self.lon_col].max()
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        # Add a small padding to bounds
        return lat_center, lon_center, [lat_min - 0.01, lon_min - 0.01], [lat_max + 0.01, lon_max + 0.01]

    def _initialize_map(self):
        """Initialize a Folium map with the specified center and tile style."""
        lat_center, lon_center, _, _ = self._calculate_map_bounds()
        self._cluster_map = folium.Map(
            location=[lat_center, lon_center],
            zoom_start=self.zoom_start,
            tiles=self.tiles,
            control_scale=True
        )

    def _add_tile_layers(self):
        """Add alternative tile layers to the map with proper attributions."""
        folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(self._cluster_map)
        
        # Stamen Terrain
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
                 '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; '
                 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name='Terrain',
            subdomains='abcd',
            max_zoom=18
        ).add_to(self._cluster_map)
        
        # Stamen Toner
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}{r}.png',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
                 '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; '
                 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name='Toner',
            subdomains='abcd',
            max_zoom=20
        ).add_to(self._cluster_map)
        
        # CartoDB Dark Matter
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
                 '&copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='CartoDB Dark',
            subdomains='abcd',
            max_zoom=19
        ).add_to(self._cluster_map)
        
        # CartoDB Positron (already default, but good to have explicit option)
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
                 '&copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='CartoDB Light',
            subdomains='abcd',
            max_zoom=19
        ).add_to(self._cluster_map)

    def _generate_distinct_colors(self):
        """Generate distinct colors for clusters and routes."""
        n_clusters = len(self._unique_clusters)
        n_routes = len([r for r in self.routes_info if 'error' not in r]) # Only count valid routes

        # Ensure at least one color is generated if there are clusters or routes
        total_colors_needed = max(n_clusters + n_routes, 1)

        if self.cluster_palette == self.route_palette:
            all_colors = sns.color_palette(self.cluster_palette, total_colors_needed).as_hex()
            self._cluster_colors = all_colors[:n_clusters]
            self._route_colors = all_colors[n_clusters:n_clusters + n_routes]
        else:
            self._cluster_colors = sns.color_palette(self.cluster_palette, max(n_clusters, 1)).as_hex()
            self._route_colors = sns.color_palette(self.route_palette, max(n_routes, 1)).as_hex() if n_routes > 0 else []

        self._cluster_color_map = {cluster: self._cluster_colors[i % len(self._cluster_colors)]
                                   for i, cluster in enumerate(self._unique_clusters)}

        # Assign route colors if not provided in routes_info or update existing ones
        for i, route in enumerate(self.routes_info):
            if 'error' not in route: # Only assign colors to valid routes
                if 'color' not in route or not route['color']:
                    route['color'] = self._route_colors[i % len(self._route_colors)]

    def _prepare_color_and_cluster_data(self):
        """Prepare unique clusters, generate colors, and initialize cluster groups/stats."""
        self._unique_clusters = sorted(self.df[self.cluster_col].unique())
        self._generate_distinct_colors()

        self._cluster_groups = {}
        self._cluster_stats = {}

        for cluster_id in self._unique_clusters:
            cluster_data = self.df[self.df[self.cluster_col] == cluster_id]

            lga_values = []
            if self.lga_col in self.df.columns:
                lga_values = cluster_data[self.lga_col].dropna().unique()
            lga_str = lga_values[0] if len(lga_values) == 1 else (f"{lga_values[0]}+"
                                                                  f"{len(lga_values)-1}more" if len(lga_values) > 1 else "Unknown LGA")

            lcda_values = []
            if self.lcda_col in self.df.columns:
                lcda_values = cluster_data[self.lcda_col].dropna().unique()
            lcda_str = lcda_values[0] if len(lcda_values) == 1 else (f"{lcda_values[0]}+"
                                                                    f"{len(lcda_values)-1}more" if len(lcda_values) > 1 else "Unknown LCDA")

            cluster_name = f"Cluster {cluster_id} - {lga_str} - {lcda_str} ({len(cluster_data)} points)"
            feature_group = folium.FeatureGroup(name=cluster_name)
            self._cluster_groups[cluster_id] = feature_group
            self._cluster_stats[cluster_id] = {
                'count': len(cluster_data),
                'lga_count': len(lga_values),
                'lcda_count': len(lcda_values),
                'color': self._cluster_color_map[cluster_id]
            }

    def _create_popup_html(self, row, cluster_id):
        """Create HTML content for marker popups."""
        popup_html = "<div style='font-family: Arial, sans-serif; font-size: 12px;'>"
        popup_html += (f"<div style='font-weight: bold; color: {self._cluster_color_map[cluster_id]};"
                       "margin-bottom: 5px;'>")
        popup_html += f"Cluster {cluster_id}</div>"

        for col in self.popup_cols:
            if col in row.index and pd.notna(row[col]):
                popup_html += f"<div style='margin: 2px 0;'><strong>{col}:</strong> {row[col]}</div>"

        popup_html += (f"<div style='margin: 2px 0;'><strong>Coordinates:</strong> "
                       f"{row[self.lat_col]:.4f}, {row[self.lon_col]:.4f}</div>")

        if self.lga_col in row.index and pd.notna(row[self.lga_col]):
            popup_html += f"<div style='margin: 2px 0;'><strong>LGA:</strong> {row[self.lga_col]}</div>"
        if self.lcda_col in row.index and pd.notna(row[self.lcda_col]):
            popup_html += f"<div style='margin: 2px 0;'><strong>LCDA:</strong> {row[self.lcda_col]}</div>"

        popup_html += "</div>"
        return popup_html

    def _create_tooltip_content(self, row, cluster_id):
        """Create tooltip content for markers."""
        tooltip_content = f"Cluster {cluster_id}"
        if self.tooltip_cols:
            tooltip_parts = []
            for col in self.tooltip_cols:
                if col in row.index and pd.notna(row[col]):
                    tooltip_parts.append(str(row[col]))
            if tooltip_parts:
                tooltip_content = f"{' | '.join(tooltip_parts)} (Cluster {cluster_id})"
        return tooltip_content

    def _add_markers_to_map(self, df_chunk):
        """Add markers to their respective cluster groups."""
        for idx, row in df_chunk.iterrows():
            cluster_id = row[self.cluster_col]
            # Ensure cluster_id exists in cluster_color_map (edge case if cluster_col has NaNs)
            if cluster_id not in self._cluster_color_map:
                print(f"Warning: Cluster ID {cluster_id} not found in color map. Skipping marker.")
                continue

            popup_html = self._create_popup_html(row, cluster_id)
            tooltip_content = self._create_tooltip_content(row, cluster_id)

            marker = folium.CircleMarker(
                location=[row[self.lat_col], row[self.lon_col]],
                popup=folium.Popup(popup_html, max_width=300, parse_html=True),
                tooltip=tooltip_content,
                fillColor=self._cluster_color_map[cluster_id],
                **self.marker_params # Unpack common marker parameters
            )
            # Add to the correct feature group
            if cluster_id in self._cluster_groups:
                marker.add_to(self._cluster_groups[cluster_id])
            else:
                # This case should ideally not happen if _prepare_color_and_cluster_data is robust
                print(f"Warning: FeatureGroup for cluster {cluster_id} not found. Adding marker to map directly.")
                marker.add_to(self._cluster_map)


    def _add_map_controls(self, collapse_control=True):
        """Add layer control, minimap, fullscreen, and mouse position controls to the map."""
        folium.LayerControl(collapsed=collapse_control).add_to(self._cluster_map)
        minimap = plugins.MiniMap(toggle_display=True, position='bottomleft', width=150, height=150)
        self._cluster_map.add_child(minimap)
        plugins.Fullscreen().add_to(self._cluster_map)
        plugins.MeasureControl().add_to(self._cluster_map)
        plugins.MousePosition(
            position='bottomleft',
            separator=' | ',
            empty_string='Move mouse to see coordinates',
            lng_first=False,
            num_digits=4,
            prefix='Coordinates: '
        ).add_to(self._cluster_map)

    def _add_routes_to_map(self):
        """
        Add route polylines and destination markers to the map with individual feature groups.
        """
        if not self.trip_data or not self.routes_info:
            return

        stock_point_name = self.trip_data.get('StockPointName', 'Stock Point')
        stock_point_coord = self.trip_data['StockPointCoord']

        # Determine Folium-supported color for stock marker icon
        folium_stock_color = self._map_to_folium_color(self.stock_marker_params['color'])

        # Add stock point marker
        folium.Marker(
            location=[stock_point_coord[1], stock_point_coord[0]],  # [lat, lon]
            popup=f"<b>{stock_point_name}</b>",
            tooltip=stock_point_name,
            icon=folium.Icon(color=folium_stock_color, icon=self.stock_marker_params['icon'],
                             prefix=self.stock_marker_params['prefix'])
        ).add_to(self._cluster_map)

        # Add routes with individual feature groups
        for i, route_info in enumerate(self.routes_info):
            if 'error' in route_info:
                print(f"Skipping plotting for Trip ID {route_info['TripID']} due to error: {route_info['error']}")
                continue

            trip_id = route_info['TripID']
            outlet_count = route_info['outlet_count']
            route_geometry_lonlat = route_info['geometry_lonlat']
            route_color = route_info.get('color') # Should be set in _generate_distinct_colors

            if not route_color: # Fallback if color was not assigned for some reason
                 route_color = self._route_colors[i % len(self._route_colors)]
                 print(f"Warning: Route color for Trip ID {trip_id} was not explicitly set. Using generated color.")

            # Determine Folium-supported color for route marker icons
            folium_route_color = self._map_to_folium_color(route_color)

            distance_km = route_info['distance'] / 1000
            duration_min = route_info['duration'] / 60

            # Create feature group for this route
            route_group = folium.FeatureGroup(name=f"Trip {trip_id} ({outlet_count} outlets) - {distance_km:.1f}km")

            # Convert route geometry from [lon, lat] to [lat, lon]
            route_geometry_latlon = [[lat, lon] for lon, lat in route_geometry_lonlat]

            # Add route polyline (use original hex color)
            folium.PolyLine(
                locations=route_geometry_latlon,
                color=route_color, # Use the original hex color
                tooltip=f"Trip {trip_id}<br>Outlets: {outlet_count}<br>Dist: {distance_km:.2f} km<br>Dur: {duration_min:.2f} min",
                **self.route_params
            ).add_to(route_group)

            # Add destination markers
            current_trip_destinations = next((t['Destinations'] for t in self.trip_data.get('Trips', []) if t['TripID'] == trip_id), None)

            if current_trip_destinations:
                for j, dest_data in enumerate(current_trip_destinations):
                    customer_id = dest_data.get('CustomerID', 'N/A')
                    dest_lonlat = dest_data['Coordinate']
                    folium.Marker(
                        location=[dest_lonlat[1], dest_lonlat[0]],  # [lat, lon]
                        popup=f"<b>Trip {trip_id} - {customer_id}</b>",
                        tooltip=f"Dest {j+1} for Trip {trip_id} ({customer_id})",
                        icon=folium.Icon(color=folium_route_color, icon='map-pin', prefix='fa')
                    ).add_to(route_group)
            else:
                print(f"Warning: Could not find original destination data for Trip ID {trip_id}.")

            # Add route group to map
            route_group.add_to(self._cluster_map)

    def _map_to_folium_color(self, hex_color):
        """
        Maps a hex color to a Folium-supported color name if possible,
        otherwise returns 'gray' as a fallback.
        Folium Icon colors are limited.
        """
        folium_colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'darkred',
            'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
            'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
            'gray', 'black', 'lightgray'
        ]
        # This is a simplification. A proper mapping would be complex.
        # For now, if it's a hex, return a general purpose color.
        # Ideally, Folium.Icon should accept hex codes directly for broader support.
        if hex_color and hex_color.startswith('#'):
            # This is a heuristic, can be improved.
            # For simplicity, just return a common color for hexes
            return 'blue' # Default icon color for hex colors
        return hex_color if hex_color in folium_colors else 'gray' # Fallback


    def _update_legend_html(self):
        """
        Generate HTML for the cluster and route legend with customizable limit.
        """
        sorted_cluster_items = sorted(self._cluster_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        valid_routes = [r for r in self.routes_info if 'error' not in r] if self.routes_info else []
        
        legend_html = f'''
        <div id="cluster-legend" style="position: fixed;
                    bottom: 120px; right: 10px; width: 250px; height: auto;
                    background-color: rgba(255, 255, 255, 0.95);
                    border: 2px solid #ccc; z-index: 999;
                    font-size: 11px; padding: 12px; border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                    backdrop-filter: blur(5px);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong style="color: #333;">Cluster & Route Legend</strong>
            <button onclick="toggleLegend()" style="border: none; background: none; cursor: pointer; font-size: 14px; color: #666;">&minus;</button>
        </div>
        <div style="font-size: 9px; color: #666; margin-bottom: 8px;">
            {len(self._unique_clusters)} clusters | {len(valid_routes)} routes | Showing top {self.legend_limit}
        </div>
        <div id="legend-content" style="max-height: 180px; overflow-y: auto;">
        '''
        
        # Add cluster legend entries
        cluster_count = min(len(sorted_cluster_items), self.legend_limit)
        for i, (cluster_id, stats) in enumerate(sorted_cluster_items[:cluster_count]):
            legend_html += f'''
            <div style="margin: 3px 0; display: flex; align-items: center; padding: 2px;">
                <span style="display: inline-block; width: 14px; height: 14px;
                             background-color: {stats['color']}; border-radius: 50%;
                             margin-right: 8px; border: 1px solid white;
                             box-shadow: 0 1px 2px rgba(0,0,0,0.2);"></span>
                <span style="font-size: 10px; flex: 1;">
                    <strong>#{i+1}</strong> Cluster {cluster_id}
                    <br><span style="color: #666; font-size: 9px;">{stats['count']} points</span>
                </span>
            </div>
            '''
        
        # Add route legend entries
        # Ensure we don't exceed legend_limit overall, and prioritize clusters if limit is tight
        remaining_legend_slots = self.legend_limit - cluster_count
        route_count = min(len(valid_routes), remaining_legend_slots)

        # Bug Fix: Ensure route_color is always available for valid routes
        for i, route_info in enumerate(valid_routes[:route_count]):
            # This color should have been assigned in _generate_distinct_colors
            route_color = route_info.get('color', '#808080') # Fallback to gray if somehow missing
            trip_id = route_info['TripID']
            outlet_count = route_info['outlet_count']
            legend_html += f'''
            <div style="margin: 3px 0; display: flex; align-items: center; padding: 2px;">
                <span style="display: inline-block; width: 14px; height: 14px;
                             background-color: {route_color}; margin-right: 8px;
                             border: 1px solid white; box-shadow: 0 1px 2px rgba(0,0,0,0.2);"></span>
                <span style="font-size: 10px; flex: 1;">
                    <strong>Trip {trip_id}</strong>
                    <br><span style="color: #666; font-size: 9px;">{outlet_count} outlets</span>
                </span>
            </div>
            '''
        
        total_items_displayed = cluster_count + route_count
        total_actual_items = len(sorted_cluster_items) + len(valid_routes)

        if total_actual_items > total_items_displayed:
            legend_html += (f'<div style="font-size: 9px; margin-top: 8px; color: #999; text-align: center;'
                            f'font-style: italic;">... and {total_actual_items - total_items_displayed} more items</div>')
        
        legend_html += '''
        </div>
        <script>
        function toggleLegend() {
            var content = document.getElementById('legend-content');
            var button = document.querySelector('#cluster-legend button');
            if (content.style.display === 'none') {
                content.style.display = 'block';
                button.innerHTML = '&minus;';
            } else {
                content.style.display = 'none';
                button.innerHTML = '+';
            }
        }
        </script>
        </div>
        '''
        return legend_html

    def _update_map_bounds(self):
        """
        Fit map bounds to include clusters, stock point, routes, and destination markers.
        """
        bounds = []

        # Include cluster bounds
        if not self.df.empty:
            lat_min, lat_max = self.df[self.lat_col].min(), self.df[self.lat_col].max()
            lon_min, lon_max = self.df[self.lon_col].min(), self.df[self.lon_col].max()
            bounds.append([lat_min - 0.01, lon_min - 0.01])  # Padding
            bounds.append([lat_max + 0.01, lon_max + 0.01])

        # Include stock point and route bounds
        if self.trip_data and self.routes_info:
            stock_point_coord = self.trip_data['StockPointCoord']
            bounds.append([stock_point_coord[1], stock_point_coord[0]])  # [lat, lon]

            for route_info in self.routes_info:
                if 'error' in route_info:
                    continue
                # Route geometry points
                for lon, lat in route_info['geometry_lonlat']:
                    bounds.append([lat, lon])
                # Destination points from trip_data
                current_trip_destinations = next((t['Destinations'] for t in self.trip_data.get('Trips', []) if t['TripID'] == route_info['TripID']), None)
                if current_trip_destinations:
                    for dest in current_trip_destinations:
                        bounds.append([dest['Coordinate'][1], dest['Coordinate'][0]])

        # Fit bounds if there are any points
        if bounds:
            self._cluster_map.fit_bounds(bounds)
        else: # If no data, ensure a default view is set
            lat_center, lon_center, _, _ = self._calculate_map_bounds()
            self._cluster_map.location = [lat_center, lon_center]
            self._cluster_map.zoom_start = self.zoom_start # Reset to initial zoom if no bounds to fit

    def create_map(self, collapse_control=True):
        """
        Orchestrates the creation of the enhanced Folium map.

        Parameters:
        -----------
        collapse_control : bool, optional
            Whether to collapse the layer control (default: True).

        Returns:
        --------
        folium.Map
            An enhanced Folium map object.
        """
        self._initialize_map()
        self._add_tile_layers()

        # Add markers (with chunking if specified)
        if self.chunk_size and self.chunk_size > 0:
            for start in range(0, len(self.df), self.chunk_size):
                chunk = self.df.iloc[start:start + self.chunk_size]
                self._add_markers_to_map(chunk)
        else:
            self._add_markers_to_map(self.df)

        # Add routes
        self._add_routes_to_map()

        # Sort clusters by size and add their feature groups to the map
        sorted_cluster_items = sorted(self._cluster_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        for cluster_id, stats in sorted_cluster_items:
            # Ensure the feature group exists before adding to map
            if cluster_id in self._cluster_groups:
                self._cluster_groups[cluster_id].add_to(self._cluster_map)

        # Add map controls
        self._add_map_controls(collapse_control)

        # Add custom JavaScript and updated legend
        n_clusters = len(self._unique_clusters)
        total_points = len(self.df)
        self._cluster_map.get_root().html.add_child(folium.Element(
            self._create_cluster_control_js(n_clusters, total_points)))
        self._cluster_map.get_root().html.add_child(folium.Element(
            self._update_legend_html()))

        # Fit map to bounds including clusters and routes
        self._update_map_bounds()

        return self._cluster_map

    def save_map(self, output_filename=None):
        """
        Saves the generated map to an HTML file.

        Parameters:
        -----------
        output_filename : str, optional
            Name of the HTML file to save the map to. If None, uses the
            filename provided during initialization.
        """
        filename_to_save = output_filename if output_filename else self.output_filename
        if not self._cluster_map:
            print("Error: Map has not been created. Call .create_map() first.")
            return

        try:
            self._cluster_map.save(filename_to_save)
            print(f"Interactive map saved to: {os.path.abspath(filename_to_save)}")
        except (OSError, PermissionError) as e:
            print(f"Error saving map to {filename_to_save}: {e}")
            raise

    def _create_cluster_control_js(self, n_clusters, total_points):
        """Generate JavaScript for the cluster control panel."""
        # This function remains largely the same as it's pure JavaScript generation.
        # However, it's now a private method of the class.
        return f"""
        <script>
        setTimeout(function() {{
            var controlContainer = document.createElement('div');
            controlContainer.id = 'cluster-controls';
            controlContainer.style.position = 'absolute';
            controlContainer.style.top = '10px';
            controlContainer.style.left = '60px';
            controlContainer.style.zIndex = '1000';
            controlContainer.style.display = 'flex';
            controlContainer.style.flexDirection = 'column';
            controlContainer.style.gap = '5px';
            
            var controlPanel = document.createElement('div');
            controlPanel.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
            controlPanel.style.border = '2px solid #ccc';
            controlPanel.style.borderRadius = '8px';
            controlPanel.style.padding = '10px';
            controlPanel.style.fontSize = '12px';
            controlPanel.style.fontFamily = 'Arial, sans-serif';
            controlPanel.style.boxShadow = '0 2px 10px rgba(0,0,0,0.3)';
            controlPanel.style.backdropFilter = 'blur(5px)';
            controlPanel.style.minWidth = '200px';
            
            var title = document.createElement('div');
            title.innerHTML = '<strong>Cluster Controls</strong>';
            title.style.marginBottom = '10px';
            title.style.textAlign = 'center';
            title.style.color = '#333';
            title.style.borderBottom = '1px solid #ddd';
            title.style.paddingBottom = '5px';
            controlPanel.appendChild(title);
            
            var buttonContainer = document.createElement('div');
            buttonContainer.style.display = 'flex';
            buttonContainer.style.gap = '5px';
            buttonContainer.style.marginBottom = '10px';
            
            var unselectButton = document.createElement('button');
            unselectButton.innerHTML = 'Hide All';
            unselectButton.style.flex = '1';
            unselectButton.style.backgroundColor = '#f8f9fa';
            unselectButton.style.border = '1px solid #dee2e6';
            unselectButton.style.borderRadius = '4px';
            unselectButton.style.padding = '6px 8px';
            unselectButton.style.fontSize = '11px';
            unselectButton.style.fontWeight = 'bold';
            unselectButton.style.cursor = 'pointer';
            unselectButton.style.transition = 'all 0.2s';
            
            var selectButton = document.createElement('button');
            selectButton.innerHTML = 'Show All';
            selectButton.style.flex = '1';
            selectButton.style.backgroundColor = '#f8f9fa';
            selectButton.style.border = '1px solid #dee2e6';
            selectButton.style.borderRadius = '4px';
            selectButton.style.padding = '6px 8px';
            selectButton.style.fontSize = '11px';
            selectButton.style.fontWeight = 'bold';
            selectButton.style.cursor = 'pointer';
            selectButton.style.transition = 'all 0.2s';
            
            [unselectButton, selectButton].forEach(function(btn) {{
                btn.onmouseover = function() {{
                    this.style.backgroundColor = '#e9ecef';
                    this.style.transform = 'translateY(-1px)';
                }};
                btn.onmouseout = function() {{
                    this.style.backgroundColor = '#f8f9fa';
                    this.style.transform = 'translateY(0)';
                }};
            }});
            
            unselectButton.onclick = function() {{
                var layerControls = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
                var uncheckedCount = 0;
                layerControls.forEach(function(checkbox) {{
                    if (checkbox.checked) {{
                        checkbox.click();
                        uncheckedCount++;
                    }}
                }});
                showFeedback(unselectButton, uncheckedCount > 0 ? 
                    'Hidden ' + uncheckedCount + ' clusters' : 'No clusters visible', 
                    uncheckedCount > 0 ? '#d1ecf1' : '#f8d7da');
            }};
            
            selectButton.onclick = function() {{
                var layerControls = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
                var checkedCount = 0;
                layerControls.forEach(function(checkbox) {{
                    if (!checkbox.checked) {{
                        checkbox.click();
                        checkedCount++;
                    }}
                }});
                showFeedback(selectButton, checkedCount > 0 ? 
                    'Shown ' + checkedCount + ' clusters' : 'All clusters visible', 
                    checkedCount > 0 ? '#d4edda' : '#d1ecf1');
            }};
            
            function showFeedback(button, message, color) {{
                var originalText = button.innerHTML;
                var originalColor = button.style.backgroundColor;
                button.innerHTML = message;
                button.style.backgroundColor = color;
                setTimeout(function() {{
                    button.innerHTML = originalText;
                    button.style.backgroundColor = originalColor;
                }}, 2000);
            }}
            
            var summaryDiv = document.createElement('div');
            summaryDiv.style.fontSize = '10px';
            summaryDiv.style.color = '#666';
            summaryDiv.style.textAlign = 'center';
            summaryDiv.style.marginTop = '5px';
            summaryDiv.innerHTML = 'Total: {n_clusters} clusters | {total_points} points';
            
            buttonContainer.appendChild(unselectButton);
            buttonContainer.appendChild(selectButton);
            controlPanel.appendChild(buttonContainer);
            controlPanel.appendChild(summaryDiv);
            controlContainer.appendChild(controlPanel);
            
            var toggleButton = document.createElement('button');
            toggleButton.innerHTML = '&minus;';
            toggleButton.style.position = 'absolute';
            toggleButton.style.top = '5px';
            toggleButton.style.right = '5px';
            toggleButton.style.width = '20px';
            toggleButton.style.height = '20px';
            toggleButton.style.border = 'none';
            toggleButton.style.backgroundColor = 'transparent';
            toggleButton.style.cursor = 'pointer';
            toggleButton.style.fontSize = '14px';
            toggleButton.style.fontWeight = 'bold';
            toggleButton.style.color = '#666';
            
            var isMinimized = false;
            toggleButton.onclick = function() {{
                if (isMinimized) {{
                    buttonContainer.style.display = 'flex';
                    summaryDiv.style.display = 'block';
                    toggleButton.innerHTML = '&minus;';
                    isMinimized = false;
                }} else {{
                    buttonContainer.style.display = 'none';
                    summaryDiv.style.display = 'none';
                    toggleButton.innerHTML = '+';
                    isMinimized = true;
                }}
            }};
            
            controlPanel.appendChild(toggleButton);
            document.body.appendChild(controlContainer);
        }}, 1000);
        </script>
        """

# Helper function (kept outside the class if it's generally useful or specific to an external library's limitation)
# This mapping is still a heuristic due to Folium's icon color limitations.
def map_to_folium_color(hex_color):
    """
    Maps a hex color to a Folium-supported color name for icons if possible,
    otherwise returns 'gray' as a fallback.
    Folium Icon colors are limited to a predefined set.
    """
    folium_icon_colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
        'gray', 'black', 'lightgray'
    ]
    # This is a simplification. For precise color matching, custom SVG icons
    # or a more complex color-to-name mapping would be needed.
    # For now, if it's a hex, we'll try to return a default common color.
    if hex_color and hex_color.startswith('#'):
        # In a real application, you might want to map hex to the closest
        # named color or provide a broader range of explicit mappings.
        # For general purpose, let's pick a common one.
        return 'blue' # Arbitrary choice for hex colors for icons
    return hex_color if hex_color in folium_icon_colors else 'gray' # Fallback for non-hex, unsupported names


# Example Usage (assuming you have a DataFrame 'df_clusters' and 'trip_data', 'routes_info')
if __name__ == "__main__":
    # Create dummy data for demonstration
    data = {
        'Latitude': [6.45, 6.5, 6.6, 6.48, 6.55, 6.4, 6.65, 6.52, 6.47, 6.58],
        'Longitude': [3.35, 3.4, 3.3, 3.38, 3.45, 3.32, 3.37, 3.42, 3.39, 3.33],
        'cluster': [0, 1, 0, 1, 2, 0, 1, 2, 0, 1],
        'LGA': ['Lagos Island', 'Eti-Osa', 'Lagos Island', 'Eti-Osa', 'Ikeja',
                'Lagos Island', 'Eti-Osa', 'Ikeja', 'Lagos Island', 'Eti-Osa'],
        'LCDA': ['Ikoyi-Obalende', 'Lekki', 'Ikoyi-Obalende', 'Lekki', 'Ojodu',
                 'Ikoyi-Obalende', 'Lekki', 'Ojodu', 'Ikoyi-Obalende', 'Lekki'],
        'Revenue': [1000, 1500, 1200, 1800, 2000, 900, 1600, 2200, 1100, 1700],
        'CustomerName': [f'Customer {i+1}' for i in range(10)]
    }
    df_clusters = pd.DataFrame(data)

    # Dummy trip data (example structure as expected by add_routes_to_map)
    trip_data = {
        "StockPointName": "Main Warehouse",
        "StockPointCoord": [3.3792, 6.5244], # Lon, Lat
        "Trips": [
            {
                "TripID": "TRIP001",
                "Destinations": [
                    {"CustomerID": "CUST001", "Coordinate": [3.35, 6.45]}, # Lon, Lat
                    {"CustomerID": "CUST003", "Coordinate": [3.3, 6.6]}
                ]
            },
            {
                "TripID": "TRIP002",
                "Destinations": [
                    {"CustomerID": "CUST002", "Coordinate": [3.4, 6.5]},
                    {"CustomerID": "CUST004", "Coordinate": [3.38, 6.48]}
                ]
            }
        ]
    }

    # Dummy routes_info (example structure)
    routes_info = [
        {
            "TripID": "TRIP001",
            "outlet_count": 2,
            "geometry_lonlat": [[3.3792, 6.5244], [3.35, 6.45], [3.3, 6.6]], # Lon, Lat points for polyline
            "distance": 15000, # meters
            "duration": 900 # seconds
        },
        {
            "TripID": "TRIP002",
            "outlet_count": 2,
            "geometry_lonlat": [[3.3792, 6.5244], [3.4, 6.5], [3.38, 6.48]],
            "distance": 12000,
            "duration": 720,
            "color": "#FF00FF" # Example: specific color for a route
        }
        # Add a route with an error to test handling
        # {
        #     "TripID": "TRIP003",
        #     "error": "Routing service unavailable"
        # }
    ]


    # Initialize the mapper
    mapper = ClusterRouteMapper(
        df=df_clusters,
        lat_col='Latitude',
        lon_col='Longitude',
        cluster_col='cluster',
        lga_col='LGA',
        lcda_col='LCDA',
        popup_cols=['CustomerName', 'Revenue'],
        tooltip_cols=['CustomerName', 'LGA'],
        zoom_start=11,
        tiles='cartodb positron',
        marker_radius=7,
        marker_fill_opacity=0.7,
        marker_stroke_width=1.5,
        marker_stroke_opacity=1.0,
        cluster_palette='viridis', # Different palette for clusters
        route_palette='Dark2',     # Different palette for routes
        legend_limit=10,
        trip_data=trip_data,
        routes_info=routes_info,
        output_filename="my_enhanced_map.html",
        route_weight=6,
        route_opacity=0.8,
        stock_icon='truck', # Custom icon for stock point
        stock_color='darkblue', # Custom color for stock point
        chunk_size=5 # Example: process markers in chunks of 5
    )

    # Create and save the map
    folium_map = mapper.create_map()
    mapper.save_map()

    # You can also directly access the map object after creation
    # folium_map_object = mapper._cluster_map
    # folium_map_object # To display in a Jupyter notebook