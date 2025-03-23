"""
Test script for detecting and visualizing predicate transformations across domains.

This script demonstrates how to detect domains, actants, and predicates
and visualize how relationships between concepts evolve across the semantic landscape.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple
import base64
import networkx as nx

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from habitat_evolution.adaptive_core.resonance.wave_resonance_analyzer import WaveResonanceAnalyzer
from habitat_evolution.adaptive_core.resonance.resonance_cascade_tracker import ResonanceCascadeTracker
from habitat_evolution.adaptive_core.visualization.resonance_visualizer import ResonanceVisualizer

# Sample data structures for predicates and actants
class PredicateTransformationDetector:
    """Detects and analyzes transformations of predicates across domains."""
    
    def __init__(self):
        """Initialize the predicate transformation detector."""
        self.predicates = {}
        self.actants = {}
        self.transformations = []
    
    def detect_predicates(self, text: str) -> Dict[str, Any]:
        """
        Detect predicates in text.
        
        In a real implementation, this would use NLP techniques.
        For this test, we'll use a simplified approach with predefined patterns.
        """
        # Simple predicate detection (subject-verb-object)
        predicates = []
        
        # For testing, we'll use predefined patterns
        if "algae evolve to withstand" in text.lower():
            predicates.append({
                "id": f"pred_{len(self.predicates) + 1}",
                "subject": "symbiotic algae",
                "verb": "evolve to withstand",
                "object": "higher temperatures",
                "full_text": text
            })
        elif "coral hosts adapt" in text.lower():
            predicates.append({
                "id": f"pred_{len(self.predicates) + 1}",
                "subject": "coral hosts",
                "verb": "adapt to accommodate",
                "object": "changing symbiotic partners",
                "full_text": text
            })
        elif "reefs develop" in text.lower():
            predicates.append({
                "id": f"pred_{len(self.predicates) + 1}",
                "subject": "coral reefs",
                "verb": "develop resistance",
                "object": "environmental stressors",
                "full_text": text
            })
        
        # Store predicates
        for pred in predicates:
            self.predicates[pred["id"]] = pred
            
            # Track actants
            self._track_actant(pred["subject"], "subject", pred["id"])
            self._track_actant(pred["object"], "object", pred["id"])
        
        return predicates
    
    def _track_actant(self, actant_name: str, role: str, predicate_id: str):
        """Track actants and their roles in predicates."""
        if actant_name not in self.actants:
            self.actants[actant_name] = {
                "name": actant_name,
                "appearances": [],
                "roles": []
            }
        
        self.actants[actant_name]["appearances"].append(predicate_id)
        self.actants[actant_name]["roles"].append({
            "predicate_id": predicate_id,
            "role": role
        })
    
    def detect_transformations(self) -> List[Dict[str, Any]]:
        """Detect transformations between predicates."""
        transformations = []
        
        # Compare all pairs of predicates
        predicate_ids = list(self.predicates.keys())
        for i in range(len(predicate_ids)):
            for j in range(i+1, len(predicate_ids)):
                pred1 = self.predicates[predicate_ids[i]]
                pred2 = self.predicates[predicate_ids[j]]
                
                # Check for shared actants
                shared_actants = []
                if pred1["subject"] == pred2["subject"]:
                    shared_actants.append({
                        "name": pred1["subject"],
                        "role1": "subject",
                        "role2": "subject"
                    })
                elif pred1["subject"] == pred2["object"]:
                    shared_actants.append({
                        "name": pred1["subject"],
                        "role1": "subject",
                        "role2": "object"
                    })
                
                if pred1["object"] == pred2["subject"]:
                    shared_actants.append({
                        "name": pred1["object"],
                        "role1": "object",
                        "role2": "subject"
                    })
                elif pred1["object"] == pred2["object"]:
                    shared_actants.append({
                        "name": pred1["object"],
                        "role1": "object",
                        "role2": "object"
                    })
                
                # If there are shared actants, record the transformation
                if shared_actants:
                    # Calculate similarity based on verb semantics (simplified)
                    verb_similarity = 0.0
                    if "evolve" in pred1["verb"] and "adapt" in pred2["verb"]:
                        verb_similarity = 0.8
                    elif "develop" in pred1["verb"] and ("evolve" in pred2["verb"] or "adapt" in pred2["verb"]):
                        verb_similarity = 0.6
                    
                    transformation = {
                        "id": f"trans_{len(transformations) + 1}",
                        "predicate1_id": pred1["id"],
                        "predicate2_id": pred2["id"],
                        "shared_actants": shared_actants,
                        "verb_similarity": verb_similarity,
                        "transformation_pattern": self._detect_pattern(pred1, pred2)
                    }
                    
                    transformations.append(transformation)
        
        self.transformations = transformations
        return transformations
    
    def _detect_pattern(self, pred1: Dict[str, Any], pred2: Dict[str, Any]) -> str:
        """Detect the transformation pattern between two predicates."""
        # This would be more sophisticated in a real implementation
        if "evolve" in pred1["verb"] and "adapt" in pred2["verb"]:
            return "Evolution → Adaptation"
        elif "develop" in pred1["verb"] and "evolve" in pred2["verb"]:
            return "Development → Evolution"
        else:
            return "General Transformation"

class PredicateVisualizer:
    """Visualizes predicate transformations and actant journeys."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the predicate visualizer."""
        default_config = {
            "figsize": (12, 10),
            "dpi": 100,
            "node_size": 3000,
            "edge_width": 2,
            "font_size": 10
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
    
    def visualize_predicate_network(self, predicates: Dict[str, Any], 
                                    transformations: List[Dict[str, Any]],
                                    title: str = "Predicate Transformation Network") -> Dict[str, Any]:
        """
        Visualize the network of predicates and their transformations.
        
        Args:
            predicates: Dictionary of predicates
            transformations: List of transformations between predicates
            title: Title for the visualization
            
        Returns:
            Dictionary with visualization data including base64-encoded image
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Add predicate nodes
        for pred_id, pred in predicates.items():
            label = f"{pred['subject']} {pred['verb']} {pred['object']}"
            G.add_node(pred_id, label=label, type="predicate")
        
        # Add transformation edges
        for trans in transformations:
            G.add_edge(
                trans["predicate1_id"], 
                trans["predicate2_id"], 
                weight=trans["verb_similarity"],
                label=trans["transformation_pattern"],
                shared_actants=[a["name"] for a in trans["shared_actants"]]
            )
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config["figsize"], dpi=self.config["dpi"])
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            ax=ax,
            node_size=self.config["node_size"],
            node_color="skyblue",
            alpha=0.8
        )
        
        # Draw edges
        edges = nx.draw_networkx_edges(
            G, pos,
            ax=ax,
            width=[G[u][v]["weight"] * self.config["edge_width"] for u, v in G.edges()],
            edge_color="gray",
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.1"
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            ax=ax,
            labels={n: G.nodes[n]["label"] for n in G.nodes()},
            font_size=self.config["font_size"],
            font_family="sans-serif",
            font_weight="bold"
        )
        
        # Draw edge labels (transformation patterns)
        edge_labels = {(u, v): G[u][v]["label"] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos,
            ax=ax,
            edge_labels=edge_labels,
            font_size=8
        )
        
        # Set title and layout
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        
        # Close figure
        plt.close(fig)
        
        return {
            "image_base64": img_str,
            "title": title,
            "predicate_count": len(predicates),
            "transformation_count": len(transformations),
            "timestamp": datetime.now().isoformat()
        }
    
    def visualize_actant_journey(self, actant_name: str, 
                                actant_data: Dict[str, Any],
                                predicates: Dict[str, Any],
                                title: str = None) -> Dict[str, Any]:
        """
        Visualize the journey of an actant through different predicates.
        
        Args:
            actant_name: Name of the actant
            actant_data: Data about the actant's appearances
            predicates: Dictionary of predicates
            title: Title for the visualization
            
        Returns:
            Dictionary with visualization data including base64-encoded image
        """
        if title is None:
            title = f"Journey of Actant: {actant_name}"
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add the actant as central node
        G.add_node(actant_name, type="actant")
        
        # Add predicate nodes and connect to actant
        for i, role in enumerate(actant_data["roles"]):
            pred_id = role["predicate_id"]
            pred = predicates[pred_id]
            
            # Create a unique node ID for this appearance
            node_id = f"{pred_id}_{i}"
            
            # Add predicate node
            label = f"{pred['subject']} {pred['verb']} {pred['object']}"
            G.add_node(node_id, label=label, type="predicate", role=role["role"])
            
            # Connect actant to predicate
            G.add_edge(actant_name, node_id, role=role["role"])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config["figsize"], dpi=self.config["dpi"])
        
        # Create layout - circular with actant at center
        pos = nx.spring_layout(G, seed=42)
        pos[actant_name] = np.array([0.5, 0.5])  # Place actant at center
        
        # Get node types
        actant_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "actant"]
        predicate_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "predicate"]
        
        # Draw actant node
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            nodelist=actant_nodes,
            node_size=self.config["node_size"] * 1.5,
            node_color="red",
            alpha=0.8
        )
        
        # Draw predicate nodes
        subject_nodes = [n for n in predicate_nodes if G.nodes[n].get("role") == "subject"]
        object_nodes = [n for n in predicate_nodes if G.nodes[n].get("role") == "object"]
        
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            nodelist=subject_nodes,
            node_size=self.config["node_size"],
            node_color="green",
            alpha=0.7
        )
        
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            nodelist=object_nodes,
            node_size=self.config["node_size"],
            node_color="blue",
            alpha=0.7
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            ax=ax,
            width=self.config["edge_width"],
            alpha=0.6,
            arrows=True,
            arrowsize=20
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            ax=ax,
            labels={n: n if n == actant_name else G.nodes[n]["label"] for n in G.nodes()},
            font_size=self.config["font_size"],
            font_family="sans-serif",
            font_weight="bold"
        )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Actant'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='Subject Role'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Object Role')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title and layout
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        
        # Close figure
        plt.close(fig)
        
        return {
            "image_base64": img_str,
            "title": title,
            "actant": actant_name,
            "role_count": len(actant_data["roles"]),
            "timestamp": datetime.now().isoformat()
        }

def save_visualization_to_file(visualization_data: Dict[str, Any], filename: str):
    """Save a visualization to a file."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'predicate_visualization_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to the output file
    output_path = os.path.join(output_dir, filename)
    
    # Check if we have an image
    if "image_base64" in visualization_data:
        # Decode base64 image
        img_data = base64.b64decode(visualization_data["image_base64"])
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(img_data)
        
        print(f"Saved visualization to {output_path}")
    
    # Save metadata
    metadata = visualization_data.copy()
    if "image_base64" in metadata:
        del metadata["image_base64"]
    
    # Write metadata to JSON file
    metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_path}")

def run_predicate_transformation_demonstration():
    """Run a demonstration of predicate transformation detection and visualization."""
    print("Starting predicate transformation detection demonstration...")
    
    # Sample text with predicates
    sample_texts = [
        "The symbiotic algae evolve to withstand higher temperatures in warming oceans.",
        "Coral hosts adapt to accommodate changing symbiotic partners as oceans warm.",
        "Coral reefs develop resistance to environmental stressors through genetic adaptation."
    ]
    
    # Initialize detector
    detector = PredicateTransformationDetector()
    
    # Detect predicates in each text
    print("Detecting predicates in sample texts...")
    for text in sample_texts:
        detector.detect_predicates(text)
    
    # Detect transformations between predicates
    print("Detecting transformations between predicates...")
    transformations = detector.detect_transformations()
    
    # Print detected predicates and transformations
    print("\nDetected Predicates:")
    for pred_id, pred in detector.predicates.items():
        print(f"  {pred_id}: {pred['subject']} {pred['verb']} {pred['object']}")
    
    print("\nDetected Transformations:")
    for trans in transformations:
        print(f"  {trans['id']}: {trans['predicate1_id']} → {trans['predicate2_id']}")
        print(f"    Pattern: {trans['transformation_pattern']}")
        print(f"    Shared Actants: {', '.join([a['name'] for a in trans['shared_actants']])}")
    
    # Initialize visualizer
    print("\nInitializing visualizer...")
    visualizer = PredicateVisualizer()
    
    # Create and save predicate network visualization
    print("Creating predicate network visualization...")
    network_viz = visualizer.visualize_predicate_network(
        detector.predicates, 
        transformations,
        "Coral Reef Adaptation - Predicate Transformation Network"
    )
    save_visualization_to_file(network_viz, "predicate_network.png")
    
    # Create and save actant journey visualizations
    print("Creating actant journey visualizations...")
    for actant_name, actant_data in detector.actants.items():
        journey_viz = visualizer.visualize_actant_journey(
            actant_name,
            actant_data,
            detector.predicates,
            f"Journey of '{actant_name}' Across Predicates"
        )
        save_visualization_to_file(journey_viz, f"actant_journey_{actant_name.replace(' ', '_')}.png")
    
    print("\nDemonstration complete! Visualizations saved to the 'predicate_visualization_output' directory.")

if __name__ == "__main__":
    import io  # Required for saving visualizations
    run_predicate_transformation_demonstration()
