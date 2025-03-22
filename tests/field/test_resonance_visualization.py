"""
Test script for demonstrating the tonic-harmonic resonance visualization capabilities.

This script creates sample data and demonstrates the various visualization
features of the ResonanceVisualizer class.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Any
import base64

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from habitat_evolution.adaptive_core.resonance.wave_resonance_analyzer import WaveResonanceAnalyzer
from habitat_evolution.adaptive_core.resonance.resonance_cascade_tracker import ResonanceCascadeTracker
from habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from habitat_evolution.adaptive_core.visualization.resonance_visualizer import ResonanceVisualizer

def generate_sample_domains(num_domains: int = 10) -> List[Dict[str, Any]]:
    """Generate sample domain data for testing."""
    domains = []
    for i in range(num_domains):
        # Generate wave properties with some randomness but ensure meaningful patterns
        frequency = 0.5 + np.random.random() * 1.5  # 0.5 to 2.0
        amplitude = 0.3 + np.random.random() * 0.7  # 0.3 to 1.0
        phase = np.random.random() * 2 * np.pi  # 0 to 2π
        
        # Create domain with wave properties
        domain = {
            "id": f"domain_{i}",
            "name": f"Sample Domain {i}",
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase,
            "keywords": [f"keyword_{j}" for j in range(3)]
        }
        domains.append(domain)
    
    return domains

def generate_sample_resonance_network(domains: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a sample resonance network from domains."""
    # Create nodes from domains
    nodes = []
    for domain in domains:
        nodes.append({
            "id": domain["id"],
            "name": domain["name"],
            "frequency": domain["frequency"],
            "amplitude": domain["amplitude"],
            "phase": domain["phase"]
        })
    
    # Create edges (resonance connections) between domains
    edges = []
    for i, source in enumerate(domains):
        # Connect to a random number of other domains
        num_connections = np.random.randint(1, min(5, len(domains) - 1))
        potential_targets = list(range(len(domains)))
        potential_targets.remove(i)  # Remove self-connection
        
        # Select random targets
        target_indices = np.random.choice(potential_targets, size=num_connections, replace=False)
        
        for j in target_indices:
            target = domains[j]
            
            # Calculate resonance strength based on wave properties
            # (simplified version of what WaveResonanceAnalyzer would do)
            freq_diff = 1 / (1 + abs(source["frequency"] - target["frequency"]))
            phase_coherence = 0.5 + 0.5 * np.cos(source["phase"] - target["phase"])
            strength = (freq_diff * phase_coherence * source["amplitude"] * target["amplitude"])
            
            # Only add edges with sufficient strength
            if strength > 0.2:
                edges.append({
                    "source": source["id"],
                    "target": target["id"],
                    "strength": strength
                })
    
    return {
        "nodes": nodes,
        "edges": edges
    }

def save_visualization_to_file(visualization_data: Dict[str, Any], filename: str):
    """Save a visualization to a file."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'visualization_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to the output file
    output_path = os.path.join(output_dir, filename)
    
    # Check if we have an image or animation
    if "image_base64" in visualization_data:
        # Decode base64 image
        img_data = base64.b64decode(visualization_data["image_base64"])
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(img_data)
        
        print(f"Saved visualization to {output_path}")
    
    elif "animation_base64" in visualization_data:
        # Decode base64 animation
        anim_data = base64.b64decode(visualization_data["animation_base64"])
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(anim_data)
        
        print(f"Saved animation to {output_path}")
    
    # Save metadata
    metadata = visualization_data.copy()
    if "image_base64" in metadata:
        del metadata["image_base64"]
    if "animation_base64" in metadata:
        del metadata["animation_base64"]
    
    # Write metadata to JSON file
    metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_path}")

def run_full_demonstration():
    """Run a full demonstration of the visualization tools."""
    print("Starting tonic-harmonic resonance visualization demonstration...")
    
    # Generate sample data
    print("Generating sample domains...")
    domains = generate_sample_domains(15)
    
    # Create resonance network
    print("Creating resonance network...")
    resonance_network = generate_sample_resonance_network(domains)
    
    # Initialize components
    print("Initializing components...")
    wave_analyzer = WaveResonanceAnalyzer()
    cascade_tracker = ResonanceCascadeTracker()
    metrics_calculator = TonicHarmonicMetrics()
    visualizer = ResonanceVisualizer()
    
    # Calculate harmonic resonance
    print("Calculating harmonic resonance...")
    resonance_matrix = wave_analyzer.calculate_harmonic_resonance(domains)
    
    # Detect resonance cascades
    print("Detecting resonance cascades...")
    cascades = cascade_tracker.detect_cascade_paths(resonance_network)
    
    # Detect convergence points
    print("Detecting convergence points...")
    convergence_points = cascade_tracker.detect_cascade_convergence(resonance_network)
    
    # Generate visualization data
    print("Generating visualization data...")
    visualization_data = cascade_tracker.generate_cascade_visualization(resonance_network)
    
    # Add cascades and convergence points to the visualization data
    visualization_data["cascades"] = cascades[:10]  # Limit to top 10 cascades
    visualization_data["convergence_points"] = convergence_points
    
    # Create and save network visualization
    print("Creating network visualization...")
    network_viz = visualizer.visualize_resonance_network(visualization_data, "Sample Resonance Network")
    save_visualization_to_file(network_viz, "resonance_network.png")
    
    # Create and save cascade visualization (if cascades exist)
    if cascades:
        print("Creating cascade visualization...")
        cascade_viz = visualizer.visualize_resonance_cascade(visualization_data, visualization_data["cascades"][0]["id"], "Sample Resonance Cascade")
        save_visualization_to_file(cascade_viz, "resonance_cascade.png")
    
    # Create and save harmonic waves visualization
    print("Creating harmonic waves visualization...")
    waves_viz = visualizer.visualize_harmonic_waves(domains[:5], "Sample Harmonic Waves")
    save_visualization_to_file(waves_viz, "harmonic_waves.png")
    
    # Create and save resonance animation
    print("Creating resonance animation...")
    anim_viz = visualizer.create_resonance_animation(domains[:5], "Sample Resonance Animation")
    save_visualization_to_file(anim_viz, "resonance_animation.gif")
    
    # Create and save convergence points visualization
    print("Creating convergence points visualization...")
    convergence_viz = visualizer.visualize_convergence_points(visualization_data, "Sample Convergence Points")
    save_visualization_to_file(convergence_viz, "convergence_points.png")
    
    print("Demonstration complete! Visualizations saved to the 'visualization_output' directory.")

if __name__ == "__main__":
    run_full_demonstration()
