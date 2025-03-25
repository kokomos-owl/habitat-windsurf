#!/usr/bin/env python3
"""
HarmonicIO Dataflow Demo

This script demonstrates the HarmonicIO dataflow system by processing climate risk data,
tracking actant journeys, visualizing the dataflow, and creating an "Actant Transformation Story".

The demo shows how data flows through the system without preset transformations, allowing
patterns to emerge naturally through the harmonic I/O service.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Optional
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from habitat_evolution.climate_risk.harmonic_climate_processor import create_climate_processor
from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourney
from habitat_evolution.adaptive_core.transformation.meaning_bridges import MeaningBridgeTracker, MeaningBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('harmonic_io_demo.log')
    ]
)

logger = logging.getLogger(__name__)


class HarmonicIODemo:
    """
    Demonstrates the HarmonicIO dataflow system.
    
    This class orchestrates the demo, including data processing, actant journey tracking,
    dataflow visualization, and the creation of an "Actant Transformation Story".
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the HarmonicIO demo.
        
        Args:
            data_dir: Directory containing climate risk data
        """
        self.data_dir = Path(data_dir)
        
        # Create HarmonicIO service and climate processor
        logger.info("Initializing HarmonicIO service and climate processor")
        self.processor, self.io_service = create_climate_processor(str(self.data_dir))
        
        # Initialize visualization data
        self.visualization_data = {
            "nodes": [],
            "edges": [],
            "actant_paths": {}
        }
        
    def run_demo(self):
        """
        Run the HarmonicIO dataflow demo.
        
        This method processes the climate risk data, tracks actant journeys,
        visualizes the dataflow, and creates an "Actant Transformation Story".
        """
        logger.info("Starting HarmonicIO dataflow demo")
        
        # Start the HarmonicIO service
        self.io_service.start()
        logger.info("HarmonicIO service started")
        
        # Process climate risk data
        logger.info("Processing climate risk data")
        metrics = self.processor.process_data()
        logger.info(f"Processing metrics: {json.dumps(metrics, indent=2)}")
        
        # Allow time for async operations to complete
        logger.info("Waiting for async operations to complete")
        time.sleep(2)
        
        # Get actant journeys
        logger.info("Retrieving actant journeys")
        actant_journeys = self.processor.get_actant_journeys()
        logger.info(f"Retrieved {len(actant_journeys)} actant journeys")
        
        # Prepare visualization data
        self.prepare_visualization_data(actant_journeys)
        
        # Create visualization
        self.create_dataflow_visualization()
        
        # Create actant transformation story
        self.create_actant_transformation_story(actant_journeys)
        
        # Visualize data transformations
        self.visualize_data_transformations()
        
        # Detect and visualize meaning bridges
        self.detect_and_visualize_meaning_bridges(actant_journeys)
        
        # Stop the HarmonicIO service
        self.io_service.stop()
        logger.info("HarmonicIO service stopped")
        
        logger.info("HarmonicIO dataflow demo completed")
        
    def prepare_visualization_data(self, actant_journeys: List[ActantJourney]):
        """
        Prepare data for dataflow visualization.
        
        Args:
            actant_journeys: List of actant journeys
        """
        logger.info("Preparing visualization data")
        
        # Add domains as nodes
        domains = set()
        for journey in actant_journeys:
            for point in journey.journey_points:
                domains.add(point.domain_id)
                
        for domain in domains:
            self.visualization_data["nodes"].append({
                "id": domain,
                "type": "domain",
                "label": domain
            })
        
        # Add actants as nodes
        for journey in actant_journeys:
            self.visualization_data["nodes"].append({
                "id": journey.actant_name,
                "type": "actant",
                "label": journey.actant_name
            })
            
            # Track actant paths
            self.visualization_data["actant_paths"][journey.actant_name] = []
            
            # Add domain transitions as edges
            for transition in journey.domain_transitions:
                edge = {
                    "source": transition.source_domain_id,
                    "target": transition.target_domain_id,
                    "actant": journey.actant_name,
                    "source_role": transition.source_role,
                    "target_role": transition.target_role
                }
                self.visualization_data["edges"].append(edge)
                
                # Add to actant path
                self.visualization_data["actant_paths"][journey.actant_name].append(
                    (transition.source_domain_id, transition.target_domain_id)
                )
        
        logger.info(f"Prepared visualization data with {len(self.visualization_data['nodes'])} nodes and {len(self.visualization_data['edges'])} edges")
        
    def create_dataflow_visualization(self):
        """
        Create a visualization of the dataflow.
        
        This method creates a network graph visualization showing how data flows
        through the system, with domains as nodes and actant journeys as edges.
        """
        logger.info("Creating dataflow visualization")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.visualization_data["nodes"]:
            if node["type"] == "domain":
                G.add_node(node["id"], label=node["label"], node_type=node["type"])
        
        # Add edges
        for edge in self.visualization_data["edges"]:
            G.add_edge(edge["source"], edge["target"], 
                      actant=edge["actant"],
                      source_role=edge["source_role"],
                      target_role=edge["target_role"])
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_size=2000, 
                              node_color="lightblue", 
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                              width=2, 
                              alpha=0.5, 
                              edge_color="gray",
                              arrows=True,
                              arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
        
        # Draw edge labels
        edge_labels = {(u, v): G.edges[u, v]["actant"] for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        # Add title and adjust layout
        plt.title("HarmonicIO Dataflow Visualization", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        
        # Save figure
        plt.savefig("harmonic_io_dataflow.png", dpi=300, bbox_inches="tight")
        logger.info("Saved dataflow visualization to harmonic_io_dataflow.png")
        
    def visualize_data_transformations(self):
        """
        Visualize the data transformations tracked by the HarmonicIO service.
        
        This method creates a visualization showing how data transforms as it moves
        through different domains, including timing, stability, and coherence metrics.
        """
        logger.info("Visualizing data transformations")
        
        # Get transformation log
        transformation_log = self.io_service.get_transformation_log()
        logger.info(f"Retrieved {len(transformation_log)} transformation records")
        
        if not transformation_log:
            logger.warning("No transformation data available for visualization")
            return
        
        # Create a directed graph for transformations
        G = nx.DiGraph()
        
        # Add domains as nodes
        domains = set()
        for transform in transformation_log:
            domains.add(transform["source_domain"])
            domains.add(transform["target_domain"])
        
        for domain in domains:
            G.add_node(domain, type="domain")
        
        # Add transformations as edges
        edge_weights = {}
        for transform in transformation_log:
            source = transform["source_domain"]
            target = transform["target_domain"]
            edge_key = (source, target)
            
            if edge_key in edge_weights:
                edge_weights[edge_key] += 1
            else:
                edge_weights[edge_key] = 1
        
        # Add weighted edges to graph
        for (source, target), weight in edge_weights.items():
            G.add_edge(source, target, weight=weight)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightblue")
        
        # Draw edges with width based on weight
        edge_widths = [G[u][v]["weight"] * 1.5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Add edge labels (transformation counts)
        edge_labels = {(u, v): f"{G[u][v]['weight']} transforms" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add title and save
        plt.title("Data Transformation Flow Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("data_transformation_flow.png", dpi=300)
        plt.close()
        
        logger.info("Data transformation visualization saved to 'data_transformation_flow.png'")
        
        # Create transformation metrics visualization
        self.create_transformation_metrics_visualization(transformation_log)
    
    def create_transformation_metrics_visualization(self, transformation_log: List[Dict[str, Any]]):
        """
        Create a visualization of transformation metrics over time.
        
        Args:
            transformation_log: List of transformation records
        """
        logger.info("Creating transformation metrics visualization")
        
        # Extract timestamps and metrics
        timestamps = [datetime.fromisoformat(t["timestamp"]) for t in transformation_log]
        eigenspace_stability = [t["eigenspace_stability"] for t in transformation_log]
        pattern_coherence = [t["pattern_coherence"] for t in transformation_log]
        cycle_positions = [t["cycle_position"] for t in transformation_log]
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot eigenspace stability
        ax1.plot(timestamps, eigenspace_stability, 'b-', label="Eigenspace Stability")
        ax1.set_ylabel("Stability")
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True)
        
        # Plot pattern coherence
        ax2.plot(timestamps, pattern_coherence, 'g-', label="Pattern Coherence")
        ax2.set_ylabel("Coherence")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)
        
        # Plot cycle position
        ax3.plot(timestamps, cycle_positions, 'r-', label="Cycle Position")
        ax3.set_ylabel("Cycle Position")
        ax3.set_ylim(0, 1)
        ax3.set_xlabel("Time")
        ax3.legend()
        ax3.grid(True)
        
        # Add title and save
        plt.suptitle("Data Transformation Metrics Over Time")
        plt.tight_layout()
        plt.savefig("transformation_metrics.png", dpi=300)
        plt.close()
        
        logger.info("Transformation metrics visualization saved to 'transformation_metrics.png'")
        
        # Create a summary of transformations by domain and save to output directory
        output_dir = "demos/output/meaning_bridges"
        os.makedirs(output_dir, exist_ok=True)
        self.create_domain_transformation_summary(transformation_log, output_dir)
    
    def detect_and_visualize_meaning_bridges(self, actant_journeys: List[ActantJourney]):
        """
        Detect meaning bridges and visualize them.
        
        This method demonstrates how meaning bridges emerge from the interaction
        of actants across domains, creating a supple IO space where meaning emerges
        from relationships rather than being statically defined.
        
        Args:
            actant_journeys: List of actant journeys
        """
        logger.info("Detecting and visualizing meaning bridges")
        
        # Get transformation log
        transformation_log = self.io_service.get_transformation_log()
        
        # Create meaning bridge tracker
        bridge_tracker = MeaningBridgeTracker(propensity_threshold=0.2)
        
        # Detect bridges
        bridges = bridge_tracker.detect_bridges(
            actant_journeys=actant_journeys,
            transformation_log=transformation_log
        )
        
        logger.info(f"Detected {len(bridges)} meaning bridges")
        
        if not bridges:
            logger.warning("No meaning bridges detected")
            return
        
        # Create bridge visualization
        self.visualize_meaning_bridges(bridges)
        
        # Create bridge narrative
        self.create_meaning_bridge_narrative(bridges, bridge_tracker)
    
    def visualize_meaning_bridges(self, bridges: List[MeaningBridge]):
        """
        Visualize meaning bridges as a network graph.
        
        Args:
            bridges: List of meaning bridges
        """
        logger.info("Visualizing meaning bridges")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add actants as nodes
        actants = set()
        for bridge in bridges:
            actants.add(bridge.source_actant_id)
            actants.add(bridge.target_actant_id)
        
        for actant in actants:
            G.add_node(actant, type="actant")
        
        # Add bridges as edges
        for bridge in bridges:
            G.add_edge(
                bridge.source_actant_id,
                bridge.target_actant_id,
                type=bridge.bridge_type,
                propensity=bridge.propensity,
                weight=bridge.propensity
            )
        
        # Create the visualization
        plt.figure(figsize=(14, 10))
        
        # Use spring layout with seed for consistency
        pos = nx.spring_layout(G, seed=42, k=0.3)
        
        # Create a custom colormap for propensity
        colors = [(0.8, 0.8, 1.0), (0.0, 0.4, 0.8)]  # Light blue to dark blue
        cmap = LinearSegmentedColormap.from_list("propensity_cmap", colors)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="lightgreen")
        
        # Draw edges with color based on propensity and width based on propensity
        edges = G.edges(data=True)
        edge_colors = [e[2]["propensity"] for e in edges]
        edge_widths = [e[2]["propensity"] * 5 for e in edges]
        
        nx.draw_networkx_edges(
            G, pos, 
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=cmap,
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.1"  # Curved edges
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        
        # Add edge labels
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            edge_labels[(u, v)] = f"{data['type']}\n({data['propensity']:.2f})"
        
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.3
        )
        
        # Add title and save
        plt.title("Meaning Bridges Visualization")
        plt.axis("off")
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = "demos/output/meaning_bridges"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to output directory
        output_path = os.path.join(output_dir, "meaning_bridges.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Meaning bridges visualization saved to '{output_path}'")
    
    def create_meaning_bridge_narrative(self, 
                                   bridges: List[MeaningBridge],
                                   bridge_tracker: MeaningBridgeTracker):
        """
        Create a narrative describing the meaning bridges.
        
        Args:
            bridges: List of meaning bridges
            bridge_tracker: The meaning bridge tracker
        """
        logger.info("Creating meaning bridge narrative")
        
        # Group bridges by type
        bridges_by_type = {}
        for bridge in bridges:
            if bridge.bridge_type not in bridges_by_type:
                bridges_by_type[bridge.bridge_type] = []
            bridges_by_type[bridge.bridge_type].append(bridge)
        
        # Create output directory if it doesn't exist
        output_dir = "demos/output/meaning_bridges"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create narrative markdown
        output_path = os.path.join(output_dir, "meaning_bridge_narrative.md")
        with open(output_path, "w") as f:
            f.write("# Meaning Bridge Narrative\n\n")
            
            f.write("## Overview\n\n")
            f.write("Meaning bridges represent potential relationships and transformations that emerge ")
            f.write("from the interaction of actants across domains. They create a supple IO space where meaning ")
            f.write("emerges from relationships rather than being statically defined.\n\n")
            
            f.write(f"This analysis detected **{len(bridges)}** meaning bridges across ")
            f.write(f"**{len(bridges_by_type)}** different types.\n\n")
            
            # Write about each bridge type
            for bridge_type, type_bridges in bridges_by_type.items():
                f.write(f"## {bridge_type.title()} Bridges\n\n")
                
                # Description based on type
                if bridge_type == "co-occurrence":
                    f.write("Co-occurrence bridges emerge when actants appear in the same domain, ")
                    f.write("suggesting potential relationships between them.\n\n")
                elif bridge_type == "sequence":
                    f.write("Sequence bridges emerge when an actant follows a consistent transformation path, ")
                    f.write("suggesting potential causal or procedural relationships.\n\n")
                elif bridge_type == "domain_crossing":
                    f.write("Domain crossing bridges emerge when actants frequently move between specific domains, ")
                    f.write("suggesting established pathways or channels.\n\n")
                
                # List the top bridges of this type
                sorted_bridges = sorted(type_bridges, key=lambda b: b.propensity, reverse=True)
                top_bridges = sorted_bridges[:5]  # Top 5
                
                f.write(f"### Top {bridge_type.title()} Bridges\n\n")
                
                for bridge in top_bridges:
                    f.write(f"- **{bridge.source_actant_id}** → **{bridge.target_actant_id}** ")
                    f.write(f"(Propensity: {bridge.propensity:.2f})\n")
                    
                    # Add context details based on type
                    if bridge_type == "co-occurrence":
                        f.write(f"  - Domain: {bridge.context.get('domain', 'Unknown')}\n")
                    elif bridge_type == "sequence":
                        f.write(f"  - Path: {bridge.context.get('source_domain', 'Unknown')} → ")
                        f.write(f"{bridge.context.get('intermediate_domain', 'Unknown')} → ")
                        f.write(f"{bridge.context.get('target_domain', 'Unknown')}\n")
                    elif bridge_type == "domain_crossing":
                        f.write(f"  - Crossing: {bridge.context.get('source_domain', 'Unknown')} → ")
                        f.write(f"{bridge.context.get('target_domain', 'Unknown')}\n")
                        f.write(f"  - Crossing count: {bridge.context.get('crossing_count', 0)}\n")
                
                f.write("\n")
            
            # Write about potential emergent patterns
            f.write("## Emergent Pattern Potential\n\n")
            f.write("The detected meaning bridges suggest the following emergent patterns:\n\n")
            
            # Identify potential emergent patterns
            if "co-occurrence" in bridges_by_type and "sequence" in bridges_by_type:
                f.write("- **Contextual Transformation Paths**: Actants that co-occur in domains also tend to ")
                f.write("follow similar transformation sequences, suggesting contextual influences.\n")
            
            if "domain_crossing" in bridges_by_type:
                high_crossings = [b for b in bridges_by_type.get("domain_crossing", []) if b.propensity > 0.5]
                if high_crossings:
                    f.write("- **Established Semantic Channels**: Strong domain crossing bridges indicate ")
                    f.write("established semantic channels between domains.\n")
            
            f.write("\n")
            
            # Write about future potential
            f.write("## Future Potential\n\n")
            f.write("These meaning bridges create potential for:\n\n")
            f.write("1. **Predictive Transformations**: Using bridge propensities to predict likely transformations\n")
            f.write("2. **Semantic Bridging**: Leveraging co-occurrence bridges to bridge semantic gaps\n")
            f.write("3. **Pattern Amplification**: Reinforcing high-propensity bridges to strengthen patterns\n")
            f.write("4. **Cross-Domain Influence**: Enabling actants to influence each other across domain boundaries\n")
            
            # Add human-readable explanation
            f.write("\n## What Are Meaning Bridges?\n\n")
            f.write("Meaning bridges are dynamic connections that emerge between concepts (actants) as they move ")
            f.write("through different contexts. Unlike static relationships, these bridges evolve and adapt based ")
            f.write("on how concepts interact.\n\n")
            f.write("Think of them as pathways of potential meaning that form naturally when ideas relate to each ")
            f.write("other in consistent ways. These bridges help us understand how meaning flows between concepts ")
            f.write("and how new patterns of understanding can emerge.\n\n")
            f.write("By tracking these bridges, we can discover unexpected connections and predict how concepts ")
            f.write("might transform in the future, creating a richer understanding of complex information.\n")
        
        logger.info(f"Meaning bridge narrative saved to '{output_path}'")
    
    def create_domain_transformation_summary(self, transformation_log: List[Dict[str, Any]], output_dir: str = "."):
        """
        Create a summary of transformations by domain.
        
        Args:
            transformation_log: List of transformation records
        """
        logger.info("Creating domain transformation summary")
        
        # Count transformations by source and target domain
        source_counts = {}
        target_counts = {}
        domain_pairs = {}
        
        for transform in transformation_log:
            source = transform["source_domain"]
            target = transform["target_domain"]
            pair = (source, target)
            
            source_counts[source] = source_counts.get(source, 0) + 1
            target_counts[target] = target_counts.get(target, 0) + 1
            domain_pairs[pair] = domain_pairs.get(pair, 0) + 1
        
        # Create summary markdown
        with open("domain_transformation_summary.md", "w") as f:
            f.write("# Domain Transformation Summary\n\n")
            
            f.write("## Transformation Counts by Source Domain\n\n")
            for domain, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{domain}**: {count} transformations\n")
            
            f.write("\n## Transformation Counts by Target Domain\n\n")
            for domain, count in sorted(target_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{domain}**: {count} transformations\n")
            
            f.write("\n## Transformation Counts by Domain Pair\n\n")
            f.write("| Source Domain | Target Domain | Transformation Count |\n")
            f.write("|--------------|--------------|---------------------|\n")
            for (source, target), count in sorted(domain_pairs.items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {source} | {target} | {count} |\n")
        
        logger.info(f"Domain transformation summary saved to '{output_path}'")
    
    def create_actant_transformation_story(self, actant_journeys: List[ActantJourney]):
        """
        Create an "Actant Transformation Story".
        
        This method creates a narrative of how actants transform as they journey
        through different domains, highlighting pattern evolution and role shifts.
        
        Args:
            actant_journeys: List of actant journeys
        """
        logger.info("Creating Actant Transformation Story")
        
        story = []
        story.append("# Actant Transformation Story")
        story.append("\nThis narrative describes how actants transform as they journey through different domains, highlighting pattern evolution and role shifts.\n")
        
        for journey in actant_journeys:
            # Get adaptive ID data
            adaptive_id_dict = journey.adaptive_id.to_dict() if journey.adaptive_id else {}
            
            # Add actant header
            story.append(f"## {journey.actant_name}")
            story.append(f"\nBase concept: {adaptive_id_dict.get('base_concept', journey.actant_name)}")
            
            # Add pattern propensities if available
            if "pattern_propensities" in adaptive_id_dict:
                propensities = adaptive_id_dict["pattern_propensities"]
                story.append("\n### Pattern Propensities")
                story.append(f"\n- Coherence: {propensities.get('coherence', 'N/A')}")
                story.append(f"\n- Capaciousness: {propensities.get('capaciousness', 'N/A')}")
                
                if "directionality" in propensities:
                    story.append("\n#### Directionality")
                    for direction, value in propensities["directionality"].items():
                        story.append(f"\n- {direction}: {value}")
            
            # Add journey narrative
            story.append("\n### Journey Narrative")
            
            if journey.journey_points:
                # Sort journey points by timestamp
                sorted_points = sorted(journey.journey_points, 
                                      key=lambda p: datetime.fromisoformat(p.timestamp))
                
                # Create narrative of the journey
                for i, point in enumerate(sorted_points):
                    if i == 0:
                        story.append(f"\n{journey.actant_name} begins its journey in the {point.domain_id} domain as a {point.role}.")
                    else:
                        story.append(f"\nThen, it moves to the {point.domain_id} domain as a {point.role}.")
            
            # Add role shifts
            role_shifts = journey.get_role_shifts()
            if role_shifts:
                story.append("\n### Role Shifts")
                for shift in role_shifts:
                    story.append(f"\n- Shifted from {shift.source_role} to {shift.target_role} when moving from {shift.source_domain_id} to {shift.target_domain_id}.")
            
            # Add semantic transformation insights
            story.append("\n### Semantic Transformation Insights")
            if journey.domain_transitions:
                # Look for patterns in the transitions
                domains_visited = set([p.domain_id for p in journey.journey_points])
                roles_played = set([p.role for p in journey.journey_points])
                
                story.append(f"\n{journey.actant_name} visited {len(domains_visited)} domains and played {len(roles_played)} distinct roles.")
                
                # Add insights based on pattern propensities
                if "pattern_propensities" in adaptive_id_dict:
                    coherence = adaptive_id_dict["pattern_propensities"].get("coherence", 0.5)
                    capaciousness = adaptive_id_dict["pattern_propensities"].get("capaciousness", 0.5)
                    
                    if coherence > 0.7:
                        story.append(f"\n{journey.actant_name} maintains a strong identity across domains, suggesting it represents a stable concept in the semantic ecosystem.")
                    elif coherence < 0.4:
                        story.append(f"\n{journey.actant_name} has a fluid identity, adapting significantly as it moves between domains.")
                        
                    if capaciousness > 0.7:
                        story.append(f"\n{journey.actant_name} shows high capaciousness, readily absorbing and integrating new information across domains.")
                    elif capaciousness < 0.4:
                        story.append(f"\n{journey.actant_name} has limited capaciousness, maintaining rigid boundaries as it moves between domains.")
            else:
                story.append(f"\n{journey.actant_name} did not transition between domains during this observation period.")
            
            # Add separator
            story.append("\n---\n")
        
        # Write story to file
        with open("actant_transformation_story.md", "w") as f:
            f.write("\n".join(story))
            
        logger.info("Saved Actant Transformation Story to actant_transformation_story.md")


def main():
    """Run the HarmonicIO dataflow demo."""
    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "climate_risk"
    
    # Create and run demo
    demo = HarmonicIODemo(str(data_dir))
    demo.run_demo()


if __name__ == "__main__":
    main()
