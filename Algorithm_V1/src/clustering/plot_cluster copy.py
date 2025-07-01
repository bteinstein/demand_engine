import folium
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns


import folium
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns

import folium
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
from folium import plugins
import colorsys
import pandas as pd
 
import folium
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
from folium import plugins
import colorsys
import pandas as pd


import folium
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
from folium import plugins
import colorsys
import pandas as pd


def generate_distinct_colors(n_colors):
    """
    Generate visually distinct colors using HSV color space optimization.
    This method provides better separation for large numbers of clusters.
    """
    if n_colors <= 20:
        # Use high-quality predefined palettes for smaller numbers
        base_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # Tableau colors
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7',  # Light variants
            '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31', '#843c0c'   # Additional variants
        ]
        return base_colors[:n_colors]
    
    # For large numbers, use HSV space with optimized separation
    colors_list = []
    
    # Golden ratio for optimal spacing
    golden_ratio = (1 + 5**0.5) / 2
    
    for i in range(n_colors):
        # Use golden ratio to distribute hues evenly
        hue = (i / golden_ratio) % 1.0
        
        # Vary saturation and value for additional distinction
        saturation = 0.7 + 0.3 * ((i * 7) % 3) / 2  # Between 0.7 and 1.0
        value = 0.6 + 0.4 * ((i * 11) % 3) / 2       # Between 0.6 and 1.0
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors_list.append(hex_color)
    
    return colors_list


def create_enhanced_cluster_map(df, lat_col='Latitude', lon_col='Longitude', 
                              cluster_col='cluster', lga_col='LGA', lcda_col='LCDA',
                              popup_cols=None, tooltip_cols=None, zoom_start=9, 
                              tiles='cartodb positron', radius=5, fill_opacity=0.8, 
                              stroke_width=1, stroke_opacity=1.0, collapse_control = False):
    """
    Create an enhanced Folium map with clustered markers, layer controls, and improved styling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing latitude, longitude, cluster, and optional popup/tooltip columns.
    lat_col : str, optional
        Name of the latitude column (default: 'Latitude').
    lon_col : str, optional
        Name of the longitude column (default: 'Longitude').
    cluster_col : str, optional
        Name of the cluster column (default: 'cluster').
    lga_col : str, optional
        Name of the LGA column (default: 'LGA').
    lcda_col : str, optional
        Name of the LCDA column (default: 'LCDA').
    popup_cols : list of str, optional
        List of column names to include in popups (default: None).
    tooltip_cols : list of str, optional
        List of column names to include in tooltips (default: None).
    zoom_start : int, optional
        Initial zoom level of the map (default: 9).
    tiles : str, optional
        Map tile style (default: 'cartodb positron').
    radius : float, optional
        Marker radius (default: 5).
    fill_opacity : float, optional
        Marker fill opacity (default: 0.8).
    stroke_width : float, optional
        Marker stroke width (default: 1).
    stroke_opacity : float, optional
        Marker stroke opacity (default: 1.0).
    
    Returns:
    --------
    folium.Map
        An enhanced Folium map object with clustered markers and layer controls.
    """
    
    # Input validation
    required_cols = [lat_col, lon_col, cluster_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate map bounds for better initial view
    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
    
    # Calculate center with slight padding
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    
    # Initialize map with multiple tile options
    cluster_map = folium.Map(
        location=[lat_center, lon_center], 
        zoom_start=zoom_start, 
        tiles=tiles,
        control_scale=True
    )
    
    # Add alternative tile layers with proper attributions
    folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(cluster_map)
    
    # Stamen tiles with proper attribution
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
             '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; '
             'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name='Terrain',
        subdomains='abcd',
        max_zoom=18
    ).add_to(cluster_map)
    
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}{r}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
             '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; '
             'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name='Toner',
        subdomains='abcd',
        max_zoom=20
    ).add_to(cluster_map)
    
    # Add CartoDB tiles as alternatives
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Light',
        subdomains='abcd',
        max_zoom=19
    ).add_to(cluster_map)
    
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Dark',
        subdomains='abcd',
        max_zoom=19
    ).add_to(cluster_map)
    
    # Get unique clusters and generate colors
    unique_clusters = sorted(df[cluster_col].unique())
    n_clusters = len(unique_clusters)
    cluster_colors = generate_distinct_colors(n_clusters)
    
    # Create color mapping
    cluster_color_map = {cluster: cluster_colors[i] for i, cluster in enumerate(unique_clusters)}
    
    # Create feature groups for each cluster
    cluster_groups = {}
    cluster_stats = {}
    
    for cluster_id in unique_clusters:
        cluster_data = df[df[cluster_col] == cluster_id]
        
        # Generate cluster name
        lga_values = cluster_data[lga_col].dropna().unique() if lga_col in df.columns else ['Unknown']
        lcda_values = cluster_data[lcda_col].dropna().unique() if lcda_col in df.columns else ['Unknown']
        
        # Create meaningful cluster name
        lga_str = lga_values[0] if len(lga_values) == 1 else f"{lga_values[0]}+{len(lga_values)-1}more"
        lcda_str = lcda_values[0] if len(lcda_values) == 1 else f"{lcda_values[0]}+{len(lcda_values)-1}more"
        
        cluster_name = f"Cluster {cluster_id} - {lga_str} - {lcda_str} ({len(cluster_data)} points)"
        
        # Create feature group for this cluster
        feature_group = folium.FeatureGroup(name=cluster_name)
        cluster_groups[cluster_id] = feature_group
        
        # Store cluster statistics
        cluster_stats[cluster_id] = {
            'count': len(cluster_data),
            'lga_count': len(lga_values),
            'lcda_count': len(lcda_values),
            'color': cluster_color_map[cluster_id]
        }
    
    # Add markers to respective feature groups
    for idx, row in df.iterrows():
        cluster_id = row[cluster_col]
        
        # Prepare popup content with proper HTML formatting
        popup_html = "<div style='font-family: Arial, sans-serif; font-size: 12px;'>"
        
        # Add cluster information
        popup_html += f"<div style='font-weight: bold; color: {cluster_color_map[cluster_id]}; margin-bottom: 5px;'>"
        popup_html += f"Cluster {cluster_id}</div>"
        
        # Add custom popup columns if specified
        if popup_cols:
            for col in popup_cols:
                if col in df.columns and pd.notna(row[col]):
                    popup_html += f"<div style='margin: 2px 0;'><strong>{col}:</strong> {row[col]}</div>"
        
        # Add location information
        popup_html += f"<div style='margin: 2px 0;'><strong>Coordinates:</strong> {row[lat_col]:.4f}, {row[lon_col]:.4f}</div>"
        
        # Add LGA and LCDA if available
        if lga_col in df.columns and pd.notna(row[lga_col]):
            popup_html += f"<div style='margin: 2px 0;'><strong>LGA:</strong> {row[lga_col]}</div>"
        if lcda_col in df.columns and pd.notna(row[lcda_col]):
            popup_html += f"<div style='margin: 2px 0;'><strong>LCDA:</strong> {row[lcda_col]}</div>"
        
        popup_html += "</div>"
        
        # Prepare tooltip content
        tooltip_content = f"Cluster {cluster_id}"
        if tooltip_cols:
            tooltip_parts = []
            for col in tooltip_cols:
                if col in df.columns and pd.notna(row[col]):
                    tooltip_parts.append(str(row[col]))
            if tooltip_parts:
                tooltip_content = f"{' | '.join(tooltip_parts)} (Cluster {cluster_id})"
        
        # Create enhanced marker with stroke
        marker = folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300, parse_html=True),
            tooltip=tooltip_content,
            color='white',  # White stroke for better visibility
            weight=stroke_width,
            opacity=stroke_opacity,
            fillColor=cluster_color_map[cluster_id],
            fill=True,
            fillOpacity=fill_opacity
        )
        
        # Add marker to appropriate cluster group
        marker.add_to(cluster_groups[cluster_id])
    
    # Sort clusters by number of points (descending) for better organization
    sorted_cluster_items = sorted(cluster_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Add feature groups to map in sorted order
    for cluster_id, stats in sorted_cluster_items:
        cluster_groups[cluster_id].add_to(cluster_map)
    
    # Add layer control with collapse option
    folium.LayerControl(collapsed=collapse_control).add_to(cluster_map)
    
    
    # Add custom JavaScript for "Unselect All Clusters" functionality
    unselect_all_js = """
    <script>
    // Wait for the map to be fully loaded
    setTimeout(function() {
        // Create the unselect all button
        var unselectButton = document.createElement('button');
        unselectButton.innerHTML = 'Unselect All Clusters';
        unselectButton.style.position = 'absolute';
        unselectButton.style.top = '10px';
        unselectButton.style.left = '10px';
        unselectButton.style.zIndex = '1000';
        unselectButton.style.backgroundColor = '#fff';
        unselectButton.style.border = '2px solid #ccc';
        unselectButton.style.borderRadius = '4px';
        unselectButton.style.padding = '8px 12px';
        unselectButton.style.fontSize = '12px';
        unselectButton.style.fontWeight = 'bold';
        unselectButton.style.cursor = 'pointer';
        unselectButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        
        // Add hover effects
        unselectButton.onmouseover = function() {
            this.style.backgroundColor = '#f0f0f0';
        };
        unselectButton.onmouseout = function() {
            this.style.backgroundColor = '#fff';
        };
        
        // Add click functionality
        unselectButton.onclick = function() {
            // Find all cluster layer checkboxes (exclude base tile layers)
            var layerControls = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
            var uncheckedCount = 0;
            
            layerControls.forEach(function(checkbox) {
                if (checkbox.checked) {
                    checkbox.click(); // Uncheck the checkbox
                    uncheckedCount++;
                }
            });
            
            // Provide user feedback
            if (uncheckedCount > 0) {
                unselectButton.innerHTML = 'Unselected ' + uncheckedCount + ' clusters';
                unselectButton.style.backgroundColor = '#d4edda';
                unselectButton.style.borderColor = '#c3e6cb';
                setTimeout(function() {
                    unselectButton.innerHTML = 'Unselect All Clusters';
                    unselectButton.style.backgroundColor = '#fff';
                    unselectButton.style.borderColor = '#ccc';
                }, 2000);
            } else {
                unselectButton.innerHTML = 'No clusters selected';
                unselectButton.style.backgroundColor = '#f8d7da';
                unselectButton.style.borderColor = '#f5c6cb';
                setTimeout(function() {
                    unselectButton.innerHTML = 'Unselect All Clusters';
                    unselectButton.style.backgroundColor = '#fff';
                    unselectButton.style.borderColor = '#ccc';
                }, 2000);
            }
        };
        
        // Add the button to the map container
        document.body.appendChild(unselectButton);
        
        // Also add a "Select All Clusters" button for convenience
        var selectButton = document.createElement('button');
        selectButton.innerHTML = 'Select All Clusters';
        selectButton.style.position = 'absolute';
        selectButton.style.top = '50px';
        selectButton.style.left = '10px';
        selectButton.style.zIndex = '1000';
        selectButton.style.backgroundColor = '#fff';
        selectButton.style.border = '2px solid #ccc';
        selectButton.style.borderRadius = '4px';
        selectButton.style.padding = '8px 12px';
        selectButton.style.fontSize = '12px';
        selectButton.style.fontWeight = 'bold';
        selectButton.style.cursor = 'pointer';
        selectButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        
        // Add hover effects for select button
        selectButton.onmouseover = function() {
            this.style.backgroundColor = '#f0f0f0';
        };
        selectButton.onmouseout = function() {
            this.style.backgroundColor = '#fff';
        };
        
        // Add click functionality for select all
        selectButton.onclick = function() {
            var layerControls = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
            var checkedCount = 0;
            
            layerControls.forEach(function(checkbox) {
                if (!checkbox.checked) {
                    checkbox.click(); // Check the checkbox
                    checkedCount++;
                }
            });
            
            // Provide user feedback
            if (checkedCount > 0) {
                selectButton.innerHTML = 'Selected ' + checkedCount + ' clusters';
                selectButton.style.backgroundColor = '#d4edda';
                selectButton.style.borderColor = '#c3e6cb';
                setTimeout(function() {
                    selectButton.innerHTML = 'Select All Clusters';
                    selectButton.style.backgroundColor = '#fff';
                    selectButton.style.borderColor = '#ccc';
                }, 2000);
            } else {
                selectButton.innerHTML = 'All clusters selected';
                selectButton.style.backgroundColor = '#d1ecf1';
                selectButton.style.borderColor = '#bee5eb';
                setTimeout(function() {
                    selectButton.innerHTML = 'Select All Clusters';
                    selectButton.style.backgroundColor = '#fff';
                    selectButton.style.borderColor = '#ccc';
                }, 2000);
            }
        };
        
        // Add the select button to the map container
        document.body.appendChild(selectButton);
        
    }, 1000); // Wait 1 second for full map initialization
    </script>
    """
    
    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True, position='bottomright')
    cluster_map.add_child(minimap)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(cluster_map)
    
    # Add measure control
    plugins.MeasureControl().add_to(cluster_map)
    
    # Add mouse position display
    plugins.MousePosition().add_to(cluster_map)
    
    # Create legend HTML
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; border: 2px solid grey; z-index:9999; 
                font-size: 12px; padding: 10px; border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
    <p style="margin: 0 0 5px 0; font-weight: bold;">Cluster Legend</p>
    <p style="margin: 0 0 3px 0; font-size: 10px;">Total Clusters: {}</p>
    <div style="max-height: 200px; overflow-y: auto;">
    '''.format(n_clusters)
    
    # Add top 10 clusters to legend (to avoid overwhelming the display)
    sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    for i, (cluster_id, stats) in enumerate(sorted_clusters[:10]):
        legend_html += f'''
        <div style="margin: 2px 0;">
            <span style="display: inline-block; width: 12px; height: 12px; 
                         background-color: {stats['color']}; border-radius: 50%; 
                         margin-right: 5px; border: 1px solid white;"></span>
            <span style="font-size: 10px;">Cluster {cluster_id} ({stats['count']} pts)</span>
        </div>
        '''
    
    if len(sorted_clusters) > 10:
        legend_html += f'<div style="font-size: 10px; margin-top: 5px; color: gray;">... and {len(sorted_clusters) - 10} more clusters</div>'
    
    legend_html += '</div></div>'
    
    # Add legend to map
    cluster_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Fit map to bounds with padding
    if len(df) > 0:
        sw = [lat_min - 0.01, lon_min - 0.01]
        ne = [lat_max + 0.01, lon_max + 0.01]
        cluster_map.fit_bounds([sw, ne])
    
    return cluster_map


# Convenience function with backward compatibility
def create_cluster_map(df, **kwargs):
    """
    Backward compatible wrapper for the enhanced cluster map function.
    """
    return create_enhanced_cluster_map(df, **kwargs)



def generate_distinct_colors_v1(n_colors):
    """
    Generate visually distinct colors using HSV color space optimization.
    This method provides better separation for large numbers of clusters.
    """
    if n_colors <= 20:
        # Use high-quality predefined palettes for smaller numbers
        base_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # Tableau colors
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7',  # Light variants
            '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31', '#843c0c'   # Additional variants
        ]
        return base_colors[:n_colors]
    
    # For large numbers, use HSV space with optimized separation
    colors_list = []
    
    # Golden ratio for optimal spacing
    golden_ratio = (1 + 5**0.5) / 2
    
    for i in range(n_colors):
        # Use golden ratio to distribute hues evenly
        hue = (i / golden_ratio) % 1.0
        
        # Vary saturation and value for additional distinction
        saturation = 0.7 + 0.3 * ((i * 7) % 3) / 2  # Between 0.7 and 1.0
        value = 0.6 + 0.4 * ((i * 11) % 3) / 2       # Between 0.6 and 1.0
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors_list.append(hex_color)
    
    return colors_list


def create_enhanced_cluster_map_v1(df, lat_col='Latitude', lon_col='Longitude', 
                              cluster_col='cluster', lga_col='LGA', lcda_col='LCDA',
                              popup_cols=None, tooltip_cols=None, zoom_start=9, 
                              tiles='cartodb positron', radius=5, fill_opacity=0.8, 
                              stroke_width=1, stroke_opacity=1.0):
    """
    Create an enhanced Folium map with clustered markers, layer controls, and improved styling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing latitude, longitude, cluster, and optional popup/tooltip columns.
    lat_col : str, optional
        Name of the latitude column (default: 'Latitude').
    lon_col : str, optional
        Name of the longitude column (default: 'Longitude').
    cluster_col : str, optional
        Name of the cluster column (default: 'cluster').
    lga_col : str, optional
        Name of the LGA column (default: 'LGA').
    lcda_col : str, optional
        Name of the LCDA column (default: 'LCDA').
    popup_cols : list of str, optional
        List of column names to include in popups (default: None).
    tooltip_cols : list of str, optional
        List of column names to include in tooltips (default: None).
    zoom_start : int, optional
        Initial zoom level of the map (default: 9).
    tiles : str, optional
        Map tile style (default: 'cartodb positron').
    radius : float, optional
        Marker radius (default: 5).
    fill_opacity : float, optional
        Marker fill opacity (default: 0.8).
    stroke_width : float, optional
        Marker stroke width (default: 1).
    stroke_opacity : float, optional
        Marker stroke opacity (default: 1.0).
    
    Returns:
    --------
    folium.Map
        An enhanced Folium map object with clustered markers and layer controls.
    """
    
    # Input validation
    required_cols = [lat_col, lon_col, cluster_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate map bounds for better initial view
    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
    
    # Calculate center with slight padding
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    
    # Initialize map with multiple tile options
    cluster_map = folium.Map(
        location=[lat_center, lon_center], 
        zoom_start=zoom_start, 
        tiles=tiles,
        control_scale=True
    )
    
    # Add alternative tile layers with proper attributions
    folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(cluster_map)
    
    # Stamen tiles with proper attribution
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
             '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; '
             'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name='Terrain',
        subdomains='abcd',
        max_zoom=18
    ).add_to(cluster_map)
    
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}{r}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
             '<a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; '
             'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name='Toner',
        subdomains='abcd',
        max_zoom=20
    ).add_to(cluster_map)
    
    # Add CartoDB tiles as alternatives
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Light',
        subdomains='abcd',
        max_zoom=19
    ).add_to(cluster_map)
    
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Dark',
        subdomains='abcd',
        max_zoom=19
    ).add_to(cluster_map)
    
    # Get unique clusters and generate colors
    unique_clusters = sorted(df[cluster_col].unique())
    n_clusters = len(unique_clusters)
    cluster_colors = generate_distinct_colors(n_clusters)
    
    # Create color mapping
    cluster_color_map = {cluster: cluster_colors[i] for i, cluster in enumerate(unique_clusters)}
    
    # Create feature groups for each cluster
    cluster_groups = {}
    cluster_stats = {}
    
    for cluster_id in unique_clusters:
        cluster_data = df[df[cluster_col] == cluster_id]
        
        # Generate cluster name
        lga_values = cluster_data[lga_col].dropna().unique() if lga_col in df.columns else ['Unknown']
        lcda_values = cluster_data[lcda_col].dropna().unique() if lcda_col in df.columns else ['Unknown']
        
        # Create meaningful cluster name
        lga_str = lga_values[0] if len(lga_values) == 1 else f"{lga_values[0]}+{len(lga_values)-1}more"
        lcda_str = lcda_values[0] if len(lcda_values) == 1 else f"{lcda_values[0]}+{len(lcda_values)-1}more"
        
        cluster_name = f"Cluster {cluster_id} - {lga_str} - {lcda_str} ({len(cluster_data)} points)"
        
        # Create feature group for this cluster
        feature_group = folium.FeatureGroup(name=cluster_name)
        cluster_groups[cluster_id] = feature_group
        
        # Store cluster statistics
        cluster_stats[cluster_id] = {
            'count': len(cluster_data),
            'lga_count': len(lga_values),
            'lcda_count': len(lcda_values),
            'color': cluster_color_map[cluster_id]
        }
    
    # Add markers to respective feature groups
    for idx, row in df.iterrows():
        cluster_id = row[cluster_col]
        
        # Prepare popup content with proper HTML formatting
        popup_html = "<div style='font-family: Arial, sans-serif; font-size: 12px;'>"
        
        # Add cluster information
        popup_html += f"<div style='font-weight: bold; color: {cluster_color_map[cluster_id]}; margin-bottom: 5px;'>"
        popup_html += f"Cluster {cluster_id}</div>"
        
        # Add custom popup columns if specified
        if popup_cols:
            for col in popup_cols:
                if col in df.columns and pd.notna(row[col]):
                    popup_html += f"<div style='margin: 2px 0;'><strong>{col}:</strong> {row[col]}</div>"
        
        # Add location information
        popup_html += f"<div style='margin: 2px 0;'><strong>Coordinates:</strong> {row[lat_col]:.4f}, {row[lon_col]:.4f}</div>"
        
        # Add LGA and LCDA if available
        if lga_col in df.columns and pd.notna(row[lga_col]):
            popup_html += f"<div style='margin: 2px 0;'><strong>LGA:</strong> {row[lga_col]}</div>"
        if lcda_col in df.columns and pd.notna(row[lcda_col]):
            popup_html += f"<div style='margin: 2px 0;'><strong>LCDA:</strong> {row[lcda_col]}</div>"
        
        popup_html += "</div>"
        
        # Prepare tooltip content
        tooltip_content = f"Cluster {cluster_id}"
        if tooltip_cols:
            tooltip_parts = []
            for col in tooltip_cols:
                if col in df.columns and pd.notna(row[col]):
                    tooltip_parts.append(str(row[col]))
            if tooltip_parts:
                tooltip_content = f"{' | '.join(tooltip_parts)} (Cluster {cluster_id})"
        
        # Create enhanced marker with stroke
        marker = folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300, parse_html=True),
            tooltip=tooltip_content,
            color='white',  # White stroke for better visibility
            weight=stroke_width,
            opacity=stroke_opacity,
            fillColor=cluster_color_map[cluster_id],
            fill=True,
            fillOpacity=fill_opacity
        )
        
        # Add marker to appropriate cluster group
        marker.add_to(cluster_groups[cluster_id])
    
    # Add all feature groups to map
    for feature_group in cluster_groups.values():
        feature_group.add_to(cluster_map)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(cluster_map)
    
    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True, position='bottomright')
    cluster_map.add_child(minimap)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(cluster_map)
    
    # Add measure control
    plugins.MeasureControl().add_to(cluster_map)
    
    # Add mouse position display
    plugins.MousePosition().add_to(cluster_map)
    
    # Create legend HTML
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; border: 2px solid grey; z-index:9999; 
                font-size: 12px; padding: 10px; border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
    <p style="margin: 0 0 5px 0; font-weight: bold;">Cluster Legend</p>
    <p style="margin: 0 0 3px 0; font-size: 10px;">Total Clusters: {}</p>
    <div style="max-height: 200px; overflow-y: auto;">
    '''.format(n_clusters)
    
    # Add top 10 clusters to legend (to avoid overwhelming the display)
    sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    for i, (cluster_id, stats) in enumerate(sorted_clusters[:10]):
        legend_html += f'''
        <div style="margin: 2px 0;">
            <span style="display: inline-block; width: 12px; height: 12px; 
                         background-color: {stats['color']}; border-radius: 50%; 
                         margin-right: 5px; border: 1px solid white;"></span>
            <span style="font-size: 10px;">Cluster {cluster_id} ({stats['count']} pts)</span>
        </div>
        '''
    
    if len(sorted_clusters) > 10:
        legend_html += f'<div style="font-size: 10px; margin-top: 5px; color: gray;">... and {len(sorted_clusters) - 10} more clusters</div>'
    
    legend_html += '</div></div>'
    
    # Add legend to map
    cluster_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Fit map to bounds with padding
    if len(df) > 0:
        sw = [lat_min - 0.01, lon_min - 0.01]
        ne = [lat_max + 0.01, lon_max + 0.01]
        cluster_map.fit_bounds([sw, ne])
    
    return cluster_map


# Convenience function with backward compatibility
def create_cluster_map_v1(df, **kwargs):
    """
    Backward compatible wrapper for the enhanced cluster map function.
    """
    return create_enhanced_cluster_map(df, **kwargs)

 

def create_cluster_map_deprecated(df, lat_col='Latitude', lon_col='Longitude', cluster_col='cluster', 
                     popup_cols=None, tooltip_cols=None, zoom_start=9, 
                     tiles='cartodb positron', radius=5, fill_opacity=0.9, 
                     palette='tableau10'):
    """
    Create a Folium map with clustered markers based on provided DataFrame, with scalable color palettes.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing latitude, longitude, cluster, and optional popup/tooltip columns.
    lat_col : str, optional
        Name of the latitude column (default: 'Latitude').
    lon_col : str, optional
        Name of the longitude column (default: 'Longitude').
    cluster_col : str, optional
        Name of the cluster column (default: 'cluster').
    popup_cols : list of str, optional
        List of column names to include in popups (default: None).
    tooltip_cols : list of str, optional
        List of column names to include in tooltips (default: None).
    zoom_start : int, optional
        Initial zoom level of the map (default: 9).
    tiles : str, optional
        Map tile style (default: 'cartodb positron').
    radius : float, optional
        Marker radius (default: 5).
    fill_opacity : float, optional
        Marker fill opacity (default: 0.9).
    palette : str, optional
        Color palette for clusters. Options: 'tableau10', 'set1', 'retro_metro', 'viridis', 'rainbow' (default: 'tableau10').

    Returns:
    --------
    folium.Map
        A Folium map object with clustered markers.
    """
    # Calculate map center
    lat_m = df[lat_col].mean()
    lon_m = df[lon_col].mean()

    # Initialize map
    cluster_map = folium.Map([lat_m, lon_m], zoom_start=zoom_start, tiles=tiles)

    # Generate color scheme for clusters
    unique_clusters = np.array(sorted(df[cluster_col].unique()))
    n_clusters = len(unique_clusters)

    # Create a mapping from cluster values to color indices
    cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}

    # Use predefined palettes for smaller cluster counts, interpolate for larger ones
    if n_clusters <= 10 and palette in ['tableau10', 'set1', 'retro_metro']:
        if palette == 'tableau10':
            palette_colors = sns.color_palette("tab10", n_colors=min(n_clusters, 10))
            rainbow = [colors.rgb2hex(i) for i in palette_colors]
        elif palette == 'set1':
            palette_colors = sns.color_palette("Set1", n_colors=min(n_clusters, 9))
            rainbow = [colors.rgb2hex(i) for i in palette_colors]
        elif palette == 'retro_metro':
            retro_colors = ['#e60049', '#0bb4ff', '#50e991', '#e6d800', '#9b19f5']
            rainbow = retro_colors[:min(n_clusters, 5)]
        # Extend palette by cycling if needed
        rainbow = (rainbow * (n_clusters // len(rainbow) + 1))[:n_clusters]
    else:
        # Use continuous colormap for large clusters or viridis/rainbow
        if palette == 'viridis':
            palette_colors = cm.viridis(np.linspace(0, 1, n_clusters))
        else:  # Default to rainbow or if explicitly selected
            palette_colors = cm.rainbow(np.linspace(0, 1, n_clusters))
        rainbow = [colors.rgb2hex(i) for i in palette_colors]

    # Add markers to the map
    for _, row in df.iterrows():
        # Prepare popup content
        popup_content = None
        if popup_cols:
            # Create HTML content with explicit divs for each line
            popup_lines = [f"<div>{col}: {row[col]}</div>" for col in popup_cols if col in df.columns]
            popup_content = ''.join(popup_lines)
            popup_content = folium.Popup(popup_content, parse_html=True, max_width=300)

        # Prepare tooltip content
        tooltip_content = None
        if tooltip_cols:
            tooltip_content = ', '.join(
                f"{row[col]}" for col in tooltip_cols if col in df.columns
            )
            if cluster_col in df.columns:
                tooltip_content += f" - Cluster {row[cluster_col]}"

        # Get color index for the cluster
        cluster_value = row[cluster_col]
        try:
            color_idx = cluster_to_index[cluster_value]
        except KeyError:
            # Fallback to first color if cluster value is invalid
            color_idx = 0

        # Create marker
        folium.vector_layers.CircleMarker(
            [row[lat_col], row[lon_col]],
            radius=radius,
            popup=popup_content,
            tooltip=tooltip_content,
            color=rainbow[color_idx],
            fill=True,
            fill_color=rainbow[color_idx],
            fill_opacity=fill_opacity
        ).add_to(cluster_map)

    return cluster_map




def create_cluster_map_a(df, lat_col='Latitude', lon_col='Longitude', cluster_col='cluster', 
                     popup_cols=None, tooltip_cols=None, zoom_start=9, 
                     tiles='cartodb positron', radius=5, fill_opacity=0.9, 
                     palette='tableau10'):
    """
    Create a Folium map with clustered markers based on provided DataFrame, with scalable color palettes.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing latitude, longitude, cluster, and optional popup/tooltip columns.
    lat_col : str, optional
        Name of the latitude column (default: 'Latitude').
    lon_col : str, optional
        Name of the longitude column (default: 'Longitude').
    cluster_col : str, optional
        Name of the cluster column (default: 'cluster').
    popup_cols : list of str, optional
        List of column names to include in popups (default: None).
    tooltip_cols : list of str, optional
        List of column names to include in tooltips (default: None).
    zoom_start : int, optional
        Initial zoom level of the map (default: 9).
    tiles : str, optional
        Map tile style (default: 'cartodb positron').
    radius : float, optional
        Marker radius (default: 5).
    fill_opacity : float, optional
        Marker fill opacity (default: 0.9).
    palette : str, optional
        Color palette for clusters. Options: 'tableau10', 'set1', 'retro_metro', 'viridis', 'rainbow' (default: 'tableau10').

    Returns:
    --------
    folium.Map
        A Folium map object with clustered markers.
    """
    # Calculate map center
    lat_m = df[lat_col].mean()
    lon_m = df[lon_col].mean()

    # Initialize map
    cluster_map = folium.Map([lat_m, lon_m], zoom_start=zoom_start, tiles=tiles)

    # Generate color scheme for clusters
    unique_clusters = np.array(sorted(df[cluster_col].unique()))
    n_clusters = len(unique_clusters)

    # Create a mapping from cluster values to color indices
    cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}

    # Use predefined palettes for smaller cluster counts, interpolate for larger ones
    if n_clusters <= 10 and palette in ['tableau10', 'set1', 'retro_metro']:
        if palette == 'tableau10':
            palette_colors = sns.color_palette("tab10", n_colors=min(n_clusters, 10))
            rainbow = [colors.rgb2hex(i) for i in palette_colors]
        elif palette == 'set1':
            palette_colors = sns.color_palette("Set1", n_colors=min(n_clusters, 9))
            rainbow = [colors.rgb2hex(i) for i in palette_colors]
        elif palette == 'retro_metro':
            retro_colors = ['#e60049', '#0bb4ff', '#50e991', '#e6d800', '#9b19f5']
            rainbow = retro_colors[:min(n_clusters, 5)]
        # Extend palette by cycling if needed
        rainbow = (rainbow * (n_clusters // len(rainbow) + 1))[:n_clusters]
    else:
        # Use continuous colormap for large clusters or viridis/rainbow
        if palette == 'viridis':
            palette_colors = cm.viridis(np.linspace(0, 1, n_clusters))
        else:  # Default to rainbow or if explicitly selected
            palette_colors = cm.rainbow(np.linspace(0, 1, n_clusters))
        rainbow = [colors.rgb2hex(i) for i in palette_colors]

    # Add markers to the map
    for _, row in df.iterrows():
        # Prepare popup content
        popup_content = None
        if popup_cols:
            popup_content = '<br>'.join(
                f"{col}: {row[col]}" for col in popup_cols if col in df.columns
            )
            popup_content = folium.Popup(popup_content, parse_html=True)

        # Prepare tooltip content
        tooltip_content = None
        if tooltip_cols:
            tooltip_content = ', '.join(
                f"{row[col]}" for col in tooltip_cols if col in df.columns
            )
            if cluster_col in df.columns:
                tooltip_content += f" - Cluster {row[cluster_col]}"

        # Get color index for the cluster
        cluster_value = row[cluster_col]
        try:
            color_idx = cluster_to_index[cluster_value]
        except KeyError:
            # Fallback to first color if cluster value is invalid
            color_idx = 0

        # Create marker
        folium.vector_layers.CircleMarker(
            [row[lat_col], row[lon_col]],
            radius=radius,
            popup=popup_content,
            tooltip=tooltip_content,
            color=rainbow[color_idx],
            fill=True,
            fill_color=rainbow[color_idx],
            fill_opacity=fill_opacity
        ).add_to(cluster_map)

    return cluster_map