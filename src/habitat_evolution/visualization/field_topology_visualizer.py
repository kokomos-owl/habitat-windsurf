"""
Field Topology Visualization

This module provides visualization tools for field topology data, including:
- Field state history visualization
- Pattern relationship network visualization
- Resonance centers and interference patterns visualization
- Field density and turbulence heatmaps

These visualizations help users understand the complex dynamics of the
tonic-harmonic field topology and pattern emergence.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class FieldTopologyVisualizer:
    """
    Visualizes field topology data from the field state modulation system.
    
    This class provides methods to export field topology data to various formats
    suitable for visualization, including JSON for web-based visualizations and
    data formats for common visualization libraries.
    """
    
    def __init__(self, export_dir: Optional[str] = None):
        """
        Initialize the field topology visualizer.
        
        Args:
            export_dir: Directory to export visualization data to
        """
        self.export_dir = export_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'visualization_exports'
        )
        os.makedirs(self.export_dir, exist_ok=True)
        logger.info(f"Field topology visualizer initialized with export dir: {self.export_dir}")
    
    def export_field_state_history(self, field_state_history: List[Dict[str, Any]], 
                                  filename: Optional[str] = None) -> str:
        """
        Export field state history data for visualization.
        
        Args:
            field_state_history: List of field state snapshots
            filename: Optional filename to export to
            
        Returns:
            Path to the exported file
        """
        if not field_state_history:
            logger.warning("No field state history to export")
            return ""
        
        # Prepare data for visualization
        viz_data = {
            'timestamps': [],
            'density': [],
            'turbulence': [],
            'coherence': [],
            'stability': []
        }
        
        for state in field_state_history:
            # Convert datetime to string if needed
            timestamp = state.get('timestamp')
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            
            viz_data['timestamps'].append(timestamp)
            viz_data['density'].append(state.get('density', 0))
            viz_data['turbulence'].append(state.get('turbulence', 0))
            viz_data['coherence'].append(state.get('coherence', 0))
            viz_data['stability'].append(state.get('stability', 0))
        
        # Export to file
        filename = filename or f"field_state_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = os.path.join(self.export_dir, filename)
        
        with open(export_path, 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        logger.info(f"Exported field state history to {export_path}")
        return export_path
    
    def export_pattern_relationships(self, relationships: Dict[str, float],
                                    pattern_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
                                    filename: Optional[str] = None) -> str:
        """
        Export pattern relationship data for network visualization.
        
        Args:
            relationships: Dictionary of pattern relationships (key: pattern_id_pair, value: strength)
            pattern_metadata: Optional metadata about patterns (type, confidence, etc.)
            filename: Optional filename to export to
            
        Returns:
            Path to the exported file
        """
        if not relationships:
            logger.warning("No pattern relationships to export")
            return ""
        
        # Prepare data for visualization
        nodes = set()
        edges = []
        
        for relationship_key, strength in relationships.items():
            if strength < 0.1:  # Filter out weak relationships
                continue
                
            # Parse pattern IDs from relationship key
            try:
                source_id, target_id = relationship_key.split('_', 1)
                nodes.add(source_id)
                nodes.add(target_id)
                
                edges.append({
                    'source': source_id,
                    'target': target_id,
                    'strength': strength
                })
            except ValueError:
                logger.warning(f"Invalid relationship key format: {relationship_key}")
                continue
        
        # Create node data with metadata if available
        node_data = []
        for node_id in nodes:
            node = {'id': node_id}
            
            if pattern_metadata and node_id in pattern_metadata:
                node.update({
                    'type': pattern_metadata[node_id].get('pattern_type', 'unknown'),
                    'confidence': pattern_metadata[node_id].get('confidence', 0.5)
                })
            
            node_data.append(node)
        
        # Create network data
        network_data = {
            'nodes': node_data,
            'edges': edges,
            'timestamp': datetime.now().isoformat()
        }
        
        # Export to file
        filename = filename or f"pattern_relationships_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = os.path.join(self.export_dir, filename)
        
        with open(export_path, 'w') as f:
            json.dump(network_data, f, indent=2)
        
        logger.info(f"Exported pattern relationships to {export_path}")
        return export_path
    
    def export_field_topology_snapshot(self, field_state: Dict[str, Any],
                                      pattern_emergence_points: List[Dict[str, Any]],
                                      resonance_centers: Dict[str, Dict[str, Any]],
                                      interference_patterns: Dict[str, Dict[str, Any]],
                                      filename: Optional[str] = None) -> str:
        """
        Export a complete field topology snapshot for comprehensive visualization.
        
        Args:
            field_state: Current field state metrics
            pattern_emergence_points: List of pattern emergence points
            resonance_centers: Dictionary of resonance centers
            interference_patterns: Dictionary of interference patterns
            filename: Optional filename to export to
            
        Returns:
            Path to the exported file
        """
        # Prepare comprehensive topology data
        topology_data = {
            'timestamp': datetime.now().isoformat(),
            'field_state': field_state,
            'pattern_emergence_points': pattern_emergence_points,
            'resonance_centers': resonance_centers,
            'interference_patterns': interference_patterns
        }
        
        # Export to file
        filename = filename or f"field_topology_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = os.path.join(self.export_dir, filename)
        
        with open(export_path, 'w') as f:
            json.dump(topology_data, f, indent=2)
        
        logger.info(f"Exported field topology snapshot to {export_path}")
        return export_path
    
    def generate_html_visualization(self, data_path: str, 
                                   template_type: str = 'field_history',
                                   output_filename: Optional[str] = None) -> str:
        """
        Generate an HTML visualization from exported data.
        
        Args:
            data_path: Path to the exported data file
            template_type: Type of visualization template to use
            output_filename: Optional filename for the HTML output
            
        Returns:
            Path to the generated HTML file
        """
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Select template based on type
        if template_type == 'field_history':
            html_content = self._generate_field_history_html(data)
        elif template_type == 'relationship_network':
            html_content = self._generate_relationship_network_html(data)
        elif template_type == 'field_topology':
            html_content = self._generate_field_topology_html(data)
        else:
            raise ValueError(f"Unknown template type: {template_type}")
        
        # Write HTML file
        output_filename = output_filename or f"{template_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = os.path.join(self.export_dir, output_filename)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated {template_type} HTML visualization at {output_path}")
        return output_path
    
    def _generate_field_history_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML for field history visualization using Chart.js."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Field State History Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .chart-container {{ width: 800px; height: 500px; margin: 20px auto; }}
        h1 {{ text-align: center; color: #333; }}
        .description {{ max-width: 800px; margin: 0 auto; text-align: center; color: #666; }}
    </style>
</head>
<body>
    <h1>Habitat Field State Evolution</h1>
    <p class="description">
        This visualization shows the evolution of field state metrics over time,
        demonstrating the continuous nature of field state modulation.
    </p>
    <div class="chart-container">
        <canvas id="fieldStateChart"></canvas>
    </div>
    
    <script>
        // Field state data
        const data = {JSON.dumps(data)};
        
        // Create chart
        const ctx = document.getElementById('fieldStateChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: data.timestamps,
                datasets: [
                    {{
                        label: 'Density',
                        data: data.density,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.4
                    }},
                    {{
                        label: 'Turbulence',
                        data: data.turbulence,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.4
                    }},
                    {{
                        label: 'Coherence',
                        data: data.coherence,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.4
                    }},
                    {{
                        label: 'Stability',
                        data: data.stability,
                        borderColor: 'rgba(153, 102, 255, 1)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        tension: 0.4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Field State Evolution Over Time'
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false,
                    }}
                }},
                scales: {{
                    y: {{
                        min: 0,
                        max: 1,
                        title: {{
                            display: true,
                            text: 'Metric Value'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    def _generate_relationship_network_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML for relationship network visualization using D3.js."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Pattern Relationship Network</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .network-container {{ width: 900px; height: 600px; margin: 20px auto; border: 1px solid #ddd; }}
        h1 {{ text-align: center; color: #333; }}
        .description {{ max-width: 800px; margin: 0 auto; text-align: center; color: #666; }}
        .node {{ stroke: #fff; stroke-width: 1.5px; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
    </style>
</head>
<body>
    <h1>Pattern Relationship Network</h1>
    <p class="description">
        This visualization shows the relationships between detected patterns,
        with stronger relationships shown as thicker lines.
    </p>
    <div class="network-container" id="network"></div>
    
    <script>
        // Network data
        const data = {JSON.dumps(data)};
        
        // Set up the network visualization
        const width = 900;
        const height = 600;
        
        // Create a color scale for node types
        const color = d3.scaleOrdinal(d3.schemeCategory10);
        
        // Create a force simulation
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.edges).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        // Create SVG
        const svg = d3.select("#network")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create links
        const link = svg.append("g")
            .selectAll("line")
            .data(data.edges)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.strength) * 5);
        
        // Create nodes
        const node = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => (d.confidence || 0.5) * 10 + 5)
            .attr("fill", d => color(d.type || "unknown"))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // Add node labels
        const label = svg.append("g")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .text(d => d.id);
        
        // Update positions on simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
"""
    
    def _generate_field_topology_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML for comprehensive field topology visualization."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Field Topology Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ display: flex; flex-direction: column; max-width: 1200px; margin: 0 auto; }}
        .field-container {{ width: 100%; height: 600px; margin: 20px auto; border: 1px solid #ddd; }}
        .metrics-container {{ display: flex; justify-content: space-around; margin-bottom: 20px; }}
        .metric-card {{ padding: 15px; border-radius: 5px; background-color: #f5f5f5; width: 18%; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        h1, h2 {{ text-align: center; color: #333; }}
        .description {{ max-width: 800px; margin: 0 auto; text-align: center; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Habitat Field Topology Visualization</h1>
        <p class="description">
            This visualization shows the current state of the field topology,
            including field metrics, pattern emergence points, resonance centers,
            and interference patterns.
        </p>
        
        <div class="metrics-container">
            <div class="metric-card">
                <h3>Density</h3>
                <div class="metric-value" id="density-value"></div>
            </div>
            <div class="metric-card">
                <h3>Turbulence</h3>
                <div class="metric-value" id="turbulence-value"></div>
            </div>
            <div class="metric-card">
                <h3>Coherence</h3>
                <div class="metric-value" id="coherence-value"></div>
            </div>
            <div class="metric-card">
                <h3>Stability</h3>
                <div class="metric-value" id="stability-value"></div>
            </div>
            <div class="metric-card">
                <h3>Patterns</h3>
                <div class="metric-value" id="patterns-value"></div>
            </div>
        </div>
        
        <h2>Field Topology Map</h2>
        <div class="field-container" id="field-map"></div>
    </div>
    
    <script>
        // Field topology data
        const data = {JSON.dumps(data)};
        
        // Update metric values
        document.getElementById('density-value').textContent = 
            (data.field_state.field_density * 100).toFixed(0) + '%';
        document.getElementById('turbulence-value').textContent = 
            (data.field_state.field_turbulence * 100).toFixed(0) + '%';
        document.getElementById('coherence-value').textContent = 
            (data.field_state.field_coherence * 100).toFixed(0) + '%';
        document.getElementById('stability-value').textContent = 
            (data.field_state.field_stability * 100).toFixed(0) + '%';
        document.getElementById('patterns-value').textContent = 
            data.pattern_emergence_points.length;
        
        // Set up the field map visualization
        const width = 1100;
        const height = 600;
        
        // Create SVG
        const svg = d3.select("#field-map")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create a background gradient representing field density
        const defs = svg.append("defs");
        const gradient = defs.append("radialGradient")
            .attr("id", "density-gradient")
            .attr("cx", "50%")
            .attr("cy", "50%")
            .attr("r", "50%");
        
        gradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", `rgba(75, 192, 192, ${{data.field_state.field_density}})`);
        
        gradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", "rgba(75, 192, 192, 0.1)");
        
        // Add background
        svg.append("rect")
            .attr("width", width)
            .attr("height", height)
            .attr("fill", "url(#density-gradient)");
        
        // Create a color scale for pattern types
        const color = d3.scaleOrdinal()
            .domain(["primary", "secondary", "meta", "emergent"])
            .range(["#ff6384", "#36a2eb", "#ffce56", "#4bc0c0"]);
        
        // Add resonance centers
        const resonanceCenters = Object.entries(data.resonance_centers).map(([id, center]) => {{
            return {{
                id,
                x: center.position[0] * width,
                y: center.position[1] * height,
                radius: center.duration * 10,
                turbulence: center.turbulence
            }};
        }});
        
        svg.selectAll(".resonance-center")
            .data(resonanceCenters)
            .enter().append("circle")
            .attr("class", "resonance-center")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", d => d.radius)
            .attr("fill", "none")
            .attr("stroke", "rgba(153, 102, 255, 0.7)")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "5,5");
        
        // Add interference patterns
        const interferencePatterns = Object.entries(data.interference_patterns).map(([id, pattern]) => {{
            return {{
                id,
                x: pattern.position[0] * width,
                y: pattern.position[1] * height,
                strength: pattern.strength,
                type: pattern.pattern_type
            }};
        }});
        
        svg.selectAll(".interference")
            .data(interferencePatterns)
            .enter().append("circle")
            .attr("class", "interference")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", d => d.strength * 15)
            .attr("fill", d => `${{color(d.type)}}33`)
            .attr("stroke", d => color(d.type))
            .attr("stroke-width", 1);
        
        // Add pattern emergence points
        svg.selectAll(".pattern")
            .data(data.pattern_emergence_points)
            .enter().append("circle")
            .attr("class", "pattern")
            .attr("cx", d => d.position[0] * width)
            .attr("cy", d => d.position[1] * height)
            .attr("r", d => d.confidence * 8 + 4)
            .attr("fill", d => color(d.pattern_type));
        
        // Add pattern labels
        svg.selectAll(".pattern-label")
            .data(data.pattern_emergence_points)
            .enter().append("text")
            .attr("class", "pattern-label")
            .attr("x", d => d.position[0] * width + 10)
            .attr("y", d => d.position[1] * height + 5)
            .text(d => d.pattern_id)
            .attr("font-size", "10px");
        
        // Add legend
        const legend = svg.append("g")
            .attr("transform", "translate(20, 20)");
        
        const legendItems = [
            {{ type: "primary", label: "Primary Pattern" }},
            {{ type: "secondary", label: "Secondary Pattern" }},
            {{ type: "meta", label: "Meta Pattern" }},
            {{ type: "emergent", label: "Emergent Pattern" }},
            {{ type: "resonance", label: "Resonance Center" }}
        ];
        
        legendItems.forEach((item, i) => {{
            const g = legend.append("g")
                .attr("transform", `translate(0, ${{i * 20}})`);
            
            if (item.type === "resonance") {{
                g.append("circle")
                    .attr("cx", 5)
                    .attr("cy", 5)
                    .attr("r", 5)
                    .attr("fill", "none")
                    .attr("stroke", "rgba(153, 102, 255, 0.7)")
                    .attr("stroke-width", 2)
                    .attr("stroke-dasharray", "2,2");
            }} else {{
                g.append("circle")
                    .attr("cx", 5)
                    .attr("cy", 5)
                    .attr("r", 5)
                    .attr("fill", color(item.type));
            }}
            
            g.append("text")
                .attr("x", 15)
                .attr("y", 9)
                .text(item.label)
                .attr("font-size", "12px");
        }});
    </script>
</body>
</html>
"""


def export_visualization_data(field_state_modulator, export_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Export visualization data from a field state modulator.
    
    Args:
        field_state_modulator: FieldStateModulator instance
        export_dir: Optional directory to export to
        
    Returns:
        Dictionary of exported file paths
    """
    visualizer = FieldTopologyVisualizer(export_dir)
    
    # Export field state history
    field_history_path = visualizer.export_field_state_history(
        field_state_modulator.visualization_data['field_state_history']
    )
    
    # Export pattern relationships
    relationship_path = visualizer.export_pattern_relationships(
        field_state_modulator.pattern_relationships
    )
    
    # Export complete topology snapshot
    topology_path = visualizer.export_field_topology_snapshot(
        field_state_modulator.get_field_state(),
        field_state_modulator.visualization_data['pattern_emergence_points'],
        field_state_modulator.visualization_data['resonance_centers'],
        field_state_modulator.visualization_data['interference_patterns']
    )
    
    # Generate HTML visualizations
    html_paths = {}
    
    if field_history_path:
        html_paths['field_history'] = visualizer.generate_html_visualization(
            field_history_path, 'field_history'
        )
    
    if relationship_path:
        html_paths['relationship_network'] = visualizer.generate_html_visualization(
            relationship_path, 'relationship_network'
        )
    
    if topology_path:
        html_paths['field_topology'] = visualizer.generate_html_visualization(
            topology_path, 'field_topology'
        )
    
    return {
        'data_files': {
            'field_history': field_history_path,
            'relationships': relationship_path,
            'topology': topology_path
        },
        'html_files': html_paths
    }


# Create a simple CLI for generating visualizations
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate field topology visualizations")
    parser.add_argument("--data", required=True, help="Path to field topology data JSON file")
    parser.add_argument("--type", choices=["field_history", "relationship_network", "field_topology"],
                       default="field_topology", help="Type of visualization to generate")
    parser.add_argument("--output", help="Output HTML file path")
    
    args = parser.parse_args()
    
    visualizer = FieldTopologyVisualizer()
    html_path = visualizer.generate_html_visualization(args.data, args.type, args.output)
    
    print(f"Generated visualization at: {html_path}")
