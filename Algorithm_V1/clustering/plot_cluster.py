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
                              popup_cols=None, tooltip_cols=None, zoom_start=10, 
                              tiles='cartodb positron', radius=5, fill_opacity=0.8, 
                              stroke_width=1, stroke_opacity=1.0, collapse_control=True):
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
        Initial zoom level of the map (default: 10).
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
    collapse_control : bool, optional
        Whether to collapse the layer control (default: True).
    
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
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Dark',
        subdomains='abcd',
        max_zoom=19
    ).add_to(cluster_map)
    
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Light',
        subdomains='abcd',
        max_zoom=19
    ).add_to(cluster_map)

    # Get unique clusters and generate colors
    unique_clusters = sorted(df[cluster_col].unique())
    n_clusters = len(unique_clusters)
    total_points = len(df)
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
    
    # Add enhanced cluster control panel with better positioning
    cluster_control_js = """
    <script>
    // Wait for the map to be fully loaded
    setTimeout(function() {
        // Create a container for all custom controls
        var controlContainer = document.createElement('div');
        controlContainer.id = 'cluster-controls';
        controlContainer.style.position = 'absolute';
        controlContainer.style.top = '10px';
        controlContainer.style.left = '60px';  // Offset from zoom controls
        controlContainer.style.zIndex = '1000';
        controlContainer.style.display = 'flex';
        controlContainer.style.flexDirection = 'column';
        controlContainer.style.gap = '5px';
        
        // Create unified control panel
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
        
        // Add title
        var title = document.createElement('div');
        title.innerHTML = '<strong>Cluster Controls</strong>';
        title.style.marginBottom = '10px';
        title.style.textAlign = 'center';
        title.style.color = '#333';
        title.style.borderBottom = '1px solid #ddd';
        title.style.paddingBottom = '5px';
        controlPanel.appendChild(title);
        
        // Create button container
        var buttonContainer = document.createElement('div');
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = '5px';
        buttonContainer.style.marginBottom = '10px';
        
        // Create the unselect all button
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
        
        // Create the select all button
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
        
        // Add hover effects
        [unselectButton, selectButton].forEach(function(btn) {
            btn.onmouseover = function() {
                this.style.backgroundColor = '#e9ecef';
                this.style.transform = 'translateY(-1px)';
            };
            btn.onmouseout = function() {
                this.style.backgroundColor = '#f8f9fa';
                this.style.transform = 'translateY(0)';
            };
        });
        
        // Add click functionality for unselect all
        unselectButton.onclick = function() {
            var layerControls = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
            var uncheckedCount = 0;
            
            layerControls.forEach(function(checkbox) {
                if (checkbox.checked) {
                    checkbox.click();
                    uncheckedCount++;
                }
            });
            
            showFeedback(unselectButton, uncheckedCount > 0 ? 
                'Hidden ' + uncheckedCount + ' clusters' : 'No clusters visible', 
                uncheckedCount > 0 ? '#d1ecf1' : '#f8d7da');
        };
        
        // Add click functionality for select all
        selectButton.onclick = function() {
            var layerControls = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
            var checkedCount = 0;
            
            layerControls.forEach(function(checkbox) {
                if (!checkbox.checked) {
                    checkbox.click();
                    checkedCount++;
                }
            });
            
            showFeedback(selectButton, checkedCount > 0 ? 
                'Shown ' + checkedCount + ' clusters' : 'All clusters visible', 
                checkedCount > 0 ? '#d4edda' : '#d1ecf1');
        };
        
        // Feedback function
        function showFeedback(button, message, color) {
            var originalText = button.innerHTML;
            var originalColor = button.style.backgroundColor;
            button.innerHTML = message;
            button.style.backgroundColor = color;
            setTimeout(function() {
                button.innerHTML = originalText;
                button.style.backgroundColor = originalColor;
            }, 2000);
        }
        
        // Add cluster summary
        var summaryDiv = document.createElement('div');
        summaryDiv.style.fontSize = '10px';
        summaryDiv.style.color = '#666';
        summaryDiv.style.textAlign = 'center';
        summaryDiv.style.marginTop = '5px';
        summaryDiv.innerHTML = 'Total: """ + str(n_clusters) + """ clusters | """ + str(total_points) + """ points';
        
        // Assemble the control panel
        buttonContainer.appendChild(unselectButton);
        buttonContainer.appendChild(selectButton);
        controlPanel.appendChild(buttonContainer);
        controlPanel.appendChild(summaryDiv);
        controlContainer.appendChild(controlPanel);
        
        // Add minimize/maximize functionality
        var toggleButton = document.createElement('button');
        toggleButton.innerHTML = '−';
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
        toggleButton.onclick = function() {
            if (isMinimized) {
                buttonContainer.style.display = 'flex';
                summaryDiv.style.display = 'block';
                toggleButton.innerHTML = '−';
                isMinimized = false;
            } else {
                buttonContainer.style.display = 'none';
                summaryDiv.style.display = 'none';
                toggleButton.innerHTML = '+';
                isMinimized = true;
            }
        };
        
        controlPanel.appendChild(toggleButton);
        
        // Add the control container to the map
        document.body.appendChild(controlContainer);
        
    }, 1000);
    </script>
    """
    
    # Create enhanced legend positioned to avoid conflicts
    legend_html = f'''
    <div id="cluster-legend" style="position: fixed; 
                bottom: 120px; right: 10px; width: 250px; height: auto; 
                background-color: rgba(255, 255, 255, 0.95); 
                border: 2px solid #ccc; z-index: 999; 
                font-size: 11px; padding: 12px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                backdrop-filter: blur(5px);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <strong style="color: #333;">Cluster Legend</strong>
        <button onclick="toggleLegend()" style="border: none; background: none; cursor: pointer; font-size: 14px; color: #666;">−</button>
    </div>
    <div style="font-size: 9px; color: #666; margin-bottom: 8px;">
        {n_clusters} clusters | Sorted by size | Showing top 15
    </div>
    <div id="legend-content" style="max-height: 180px; overflow-y: auto;">
    '''
    
    # Add clusters to legend in sorted order (top 15)
    for i, (cluster_id, stats) in enumerate(sorted_cluster_items[:15]):
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
    
    if len(sorted_cluster_items) > 15:
        legend_html += f'<div style="font-size: 9px; margin-top: 8px; color: #999; text-align: center; font-style: italic;">... and {len(sorted_cluster_items) - 15} more clusters</div>'
    
    legend_html += '''
    </div>
    <script>
    function toggleLegend() {
        var content = document.getElementById('legend-content');
        var button = document.querySelector('#cluster-legend button');
        if (content.style.display === 'none') {
            content.style.display = 'block';
            button.innerHTML = '−';
        } else {
            content.style.display = 'none';
            button.innerHTML = '+';
        }
    }
    </script>
    </div>
    '''
    
    # Add minimap (positioned to avoid conflicts)
    minimap = plugins.MiniMap(
        toggle_display=True, 
        position='bottomleft',
        width=150, 
        height=150
    )
    cluster_map.add_child(minimap)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(cluster_map)
    
    # Add measure control
    plugins.MeasureControl().add_to(cluster_map)
    
    # Add mouse position display (bottom left to avoid conflicts)
    plugins.MousePosition(
        position='bottomleft',
        separator=' | ',
        empty_string='Move mouse to see coordinates',
        lng_first=False,
        num_digits=4,
        prefix='Coordinates: '
    ).add_to(cluster_map)
    
    # Add the custom JavaScript and legend
    cluster_map.get_root().html.add_child(folium.Element(cluster_control_js))
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