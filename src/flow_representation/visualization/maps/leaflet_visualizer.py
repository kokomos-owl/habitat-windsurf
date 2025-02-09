"""Interactive map visualization using Leaflet."""

import json
from pathlib import Path
import pandas as pd
from jinja2 import Template

class LeafletVisualizer:
    def __init__(self):
        """Initialize the visualizer with map configuration."""
        self.data_dir = Path(__file__).parent / 'data'
        self.template_dir = Path(__file__).parent / 'templates'
        
        # Center coordinates for Martha's Vineyard
        self.initial_view_state = {
            'latitude': 41.3805,
            'longitude': -70.6453,
            'zoom': 11,
            'maxZoom': 13,
            'minZoom': 9
        }
        
        # Create the template directory if it doesn't exist
        self.template_dir.mkdir(exist_ok=True)
        
    def create_point_data(self, town_data):
        """Process town data for visualization.
        
        Args:
            town_data: List of dictionaries containing town data with coordinates and metrics
            
        Returns:
            List of dictionaries containing point data
        """
        return town_data
        
    def generate_visualization(self, town_data, output_dir):
        """Generate an interactive visualization using Leaflet.
        
        Args:
            town_data: List of dictionaries containing town data with coordinates and metrics
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the generated visualization file
        """
        # Process town data
        point_data = self.create_point_data(town_data)
        
        # Create HTML template
        template = Template('''
        {% autoescape false %}
        <!DOCTYPE html>
        <html>
        <head>
            <title>Martha's Vineyard Climate Risk Metrics</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
            <style>
                body { 
                    margin: 0; 
                    padding: 0; 
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                }
                #map { 
                    width: 100vw; 
                    height: 100vh; 
                }
                .info {
                    padding: 12px 16px;
                    background: rgba(22, 22, 22, 0.98);
                    color: #fff;
                    border-radius: 8px;
                    font-size: 14px;
                    line-height: 1.4;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    max-width: 300px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                .info h3 {
                    margin: 0 0 8px;
                    font-size: 16px;
                    color: #fff;
                    font-weight: 600;
                }
                .metric-row {
                    display: flex;
                    justify-content: space-between;
                    margin: 4px 0;
                    font-size: 13px;
                }
                .metric-label {
                    color: rgba(255, 255, 255, 0.7);
                }
                .metric-value {
                    color: #fff;
                    font-weight: 500;
                }
                .legend {
                    line-height: 18px;
                    color: #fff;
                }
                .legend i {
                    width: 18px;
                    height: 18px;
                    float: left;
                    margin-right: 8px;
                    opacity: 0.7;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                const INITIAL_VIEW_STATE = {{ initial_view_state | tojson }};
                const points = {{ point_data | tojson }};
                
                // Create base layers
                const darkLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                });

                const lightLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                });

                // Initialize the map
                const map = L.map('map', {
                    center: [INITIAL_VIEW_STATE.latitude, INITIAL_VIEW_STATE.longitude],
                    zoom: INITIAL_VIEW_STATE.zoom,
                    maxZoom: INITIAL_VIEW_STATE.maxZoom,
                    minZoom: INITIAL_VIEW_STATE.minZoom,
                    layers: [darkLayer]
                });

                // Create layer controls
                const baseLayers = {
                    'Dark Theme': darkLayer,
                    'Light Theme': lightLayer
                };

                // Create overlay layers
                const markersLayer = L.layerGroup();
                
                // Create heatmap data
                const heatData = points.map(point => {
                    // Calculate weighted intensity based on all metrics
                    const intensity = (
                        point.metrics.coherence * 0.3 + 
                        point.metrics.emergence_rate * 0.3 + 
                        point.metrics.cross_pattern_flow * 0.2 + 
                        point.metrics.social_support * 0.2
                    );
                    return [
                        point.coordinates[1],
                        point.coordinates[0],
                        intensity
                    ];
                });

                // Create heatmap layer with custom gradient
                const heatLayer = L.heatLayer(heatData, {
                    radius: 35,
                    blur: 25,
                    maxZoom: 12,
                    max: 1.0,
                    gradient: {
                        0.0: '#313695',  // Deep blue
                        0.2: '#4575b4',  // Blue
                        0.4: '#74add1',  // Light blue
                        0.6: '#abd9e9',  // Very light blue
                        0.8: '#e0f3f8',  // Almost white blue
                        1.0: '#ffffbf'   // Light yellow
                    }
                });

                const overlayLayers = {
                    'Town Markers': markersLayer,
                    'Heat Map': heatLayer
                };

                // Add all layer controls
                L.control.layers(baseLayers, overlayLayers, {position: 'topright'}).addTo(map);
                
                // Add default layers
                markersLayer.addTo(map);
                heatLayer.addTo(map);

                // Create legend
                const legend = L.control({position: 'bottomright'});
                legend.onAdd = function (map) {
                    const div = L.DomUtil.create('div', 'info legend');
                    div.innerHTML = `
                        <h3>Metrics Legend</h3>
                        <div><i style="background: rgb(255, 128, 128)"></i>Coherence</div>
                        <div><i style="background: rgb(128, 255, 128)"></i>Emergence Rate</div>
                        <div><i style="background: rgb(128, 128, 255)"></i>Cross Pattern Flow</div>
                        <div><i style="background: rgb(255, 255, 255)"></i>Social Support</div>
                    `;
                    return div;
                };
                legend.addTo(map);

                // Add points to markers layer
                points.forEach(point => {
                    // Calculate blended color
                    const coherenceColor = [255, 128, 128];
                    const emergenceColor = [128, 255, 128];
                    const flowColor = [128, 128, 255];
                    
                    const r = (coherenceColor[0] * point.metrics.coherence + 
                              emergenceColor[0] * point.metrics.emergence_rate + 
                              flowColor[0] * point.metrics.cross_pattern_flow) / 3;
                    const g = (coherenceColor[1] * point.metrics.coherence + 
                              emergenceColor[1] * point.metrics.emergence_rate + 
                              flowColor[1] * point.metrics.cross_pattern_flow) / 3;
                    const b = (coherenceColor[2] * point.metrics.coherence + 
                              emergenceColor[2] * point.metrics.emergence_rate + 
                              flowColor[2] * point.metrics.cross_pattern_flow) / 3;
                    
                    // Blend with white based on social support
                    const white = 255 * point.metrics.social_support;
                    const color = `rgb(${Math.min(255, r + white)}, ${Math.min(255, g + white)}, ${Math.min(255, b + white)})`;
                    
                    // Create circle marker
                    const marker = L.circleMarker([point.coordinates[1], point.coordinates[0]], {
                        radius: 12 + 20 * point.metrics.social_support,
                        fillColor: color,
                        color: '#ffffff',
                        weight: 2,
                        opacity: 0.2,
                        fillOpacity: 0.8
                    }).addTo(markersLayer);

                    // Add popup
                    marker.bindPopup(`
                        <div class="info">
                            <h3>${point.name}</h3>
                            <div class="metric-row">
                                <span class="metric-label">Coherence</span>
                                <span class="metric-value">${(point.metrics.coherence * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Emergence Rate</span>
                                <span class="metric-value">${(point.metrics.emergence_rate * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Cross Pattern Flow</span>
                                <span class="metric-value">${(point.metrics.cross_pattern_flow * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Social Support</span>
                                <span class="metric-value">${(point.metrics.social_support * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    `);
                });
            </script>
        </body>
        </html>
        {% endautoescape %}
        ''')
        
        # Render template with data
        html_content = template.render(
            initial_view_state=self.initial_view_state,
            point_data=point_data
        )
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Write HTML file
        output_file = output_dir / 'climate_risk_visualization.html'
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        return str(output_file)
