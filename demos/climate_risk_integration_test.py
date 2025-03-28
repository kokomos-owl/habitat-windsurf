#!/usr/bin/env python
"""
Climate Risk Integration Test

This script demonstrates how the ObservationFrameBridge integrates regional
observation frames with the tonic-harmonic field architecture, allowing patterns
to emerge naturally without imposing artificial domain boundaries.
"""

import json
import sys
import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.habitat_evolution.field.observation_frame_bridge import ObservationFrameBridge
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from src.habitat_evolution.adaptive_core.adaptive_id import AdaptiveID


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def initialize_field_state() -> TonicHarmonicFieldState:
    """Initialize a field state with default values."""
    field_analysis = {
        "topology": {
            "effective_dimensionality": 5,
            "principal_dimensions": ["ecological", "cultural", "economic", "social", "temporal"],
            "eigenvalues": [0.42, 0.28, 0.15, 0.10, 0.05],
            "eigenvectors": [
                [0.8, 0.1, 0.05, 0.03, 0.02],
                [0.1, 0.7, 0.1, 0.05, 0.05],
                [0.05, 0.1, 0.75, 0.05, 0.05],
                [0.03, 0.05, 0.05, 0.82, 0.05],
                [0.02, 0.05, 0.05, 0.05, 0.83]
            ]
        },
        "density": {
            "density_centers": [
                {"position": [0.2, 0.3, 0.1, 0.2, 0.2], "magnitude": 0.8},
                {"position": [0.7, 0.2, 0.4, 0.3, 0.1], "magnitude": 0.6},
                {"position": [0.3, 0.6, 0.2, 0.5, 0.3], "magnitude": 0.7}
            ],
            "density_map": {
                "resolution": [10, 10, 10, 10, 10],
                "values": np.random.rand(10, 10, 10, 10, 10).tolist()
            }
        },
        "field_properties": {
            "coherence": 0.75,
            "navigability_score": 0.68,
            "stability": 0.82,
            "resonance_patterns": [
                {"center": [0.3, 0.4, 0.2, 0.3, 0.2], "intensity": 0.7},
                {"center": [0.6, 0.3, 0.5, 0.2, 0.4], "intensity": 0.6}
            ]
        }
    }
    return TonicHarmonicFieldState(field_analysis)


def visualize_field_topology(field_state: TonicHarmonicFieldState, title: str):
    """Visualize the field topology in 2D projection."""
    # Create a 2D projection of the field density
    plt.figure(figsize=(10, 8))
    
    # Get the first two principal dimensions for visualization
    dim1, dim2 = 0, 1
    
    # Create a 2D grid for visualization
    resolution = 100
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create a density map (simplified for visualization)
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            # Simulate density based on distance from density centers
            for center in field_state.field_analysis["density"]["density_centers"]:
                pos = center["position"]
                dist = np.sqrt((X[i, j] - pos[dim1])**2 + (Y[i, j] - pos[dim2])**2)
                Z[i, j] += center["magnitude"] * np.exp(-5 * dist)
    
    # Custom colormap for visualization
    colors = [(0.1, 0.1, 0.6), (0.2, 0.5, 0.9), (0.9, 0.9, 0.2), (0.9, 0.4, 0.1)]
    cmap = LinearSegmentedColormap.from_list('habitat_cmap', colors, N=256)
    
    # Plot the density map
    plt.contourf(X, Y, Z, 20, cmap=cmap)
    plt.colorbar(label='Field Density')
    
    # Plot density centers
    for center in field_state.field_analysis["density"]["density_centers"]:
        pos = center["position"]
        plt.plot(pos[dim1], pos[dim2], 'ro', markersize=10*center["magnitude"])
    
    # Plot resonance patterns
    for pattern in field_state.field_analysis["field_properties"]["resonance_patterns"]:
        pos = pattern["center"]
        plt.plot(pos[dim1], pos[dim2], 'go', markersize=10*pattern["intensity"])
    
    # Add labels and title
    plt.xlabel(field_state.field_analysis["topology"]["principal_dimensions"][dim1])
    plt.ylabel(field_state.field_analysis["topology"]["principal_dimensions"][dim2])
    plt.title(title)
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()


def run_integration_test():
    """Run the integration test with regional observation frames."""
    print("Starting Climate Risk Integration Test...")
    
    # Load observation frames and observations
    frames_path = os.path.join(os.path.dirname(__file__), 'data/climate_risk/regional_observation_frames.json')
    observations_path = os.path.join(os.path.dirname(__file__), 'data/climate_risk/regional_observations.json')
    
    frames_data = load_json_data(frames_path)
    observations_data = load_json_data(observations_path)
    
    # Initialize field state and analyzer
    field_state = initialize_field_state()
    field_analyzer = TopologicalFieldAnalyzer()
    
    # Initialize observation frame bridge
    bridge = ObservationFrameBridge(field_state, field_analyzer)
    
    # Register observation frames
    print("\nRegistering observation frames:")
    frame_ids = {}
    for frame in frames_data["observation_frames"]:
        frame_id = bridge.register_observation_frame(frame)
        frame_ids[frame["name"]] = frame_id
        print(f"  - Registered frame: {frame['name']} (ID: {frame_id})")
    
    # Visualize initial field state
    visualize_field_topology(field_state, "Initial Field Topology")
    
    # Process observations in batches to see evolution
    print("\nProcessing observations in batches:")
    
    # Group observations by perspective
    perspective_observations = {}
    for obs in observations_data["observations"]:
        perspective = obs["context"]["perspective"]
        if perspective not in perspective_observations:
            perspective_observations[perspective] = []
        perspective_observations[perspective].append(obs)
    
    # Process each perspective's observations
    for perspective, observations in perspective_observations.items():
        print(f"\nProcessing {len(observations)} observations from {perspective} perspective")
        
        # Create AdaptiveID for tracking entities in this perspective
        adaptive_id = AdaptiveID(creator_id=f"climate_risk_test_{perspective}")
        
        # Register with the bridge
        bridge.register_adaptive_id(adaptive_id)
        
        # Process each observation
        for obs in observations:
            source_id = adaptive_id.get_or_create_id(obs["source"])
            target_id = adaptive_id.get_or_create_id(obs["target"])
            
            # Process the observation
            bridge.process_observation(
                source_id=source_id,
                predicate=obs["predicate"],
                target_id=target_id,
                context=obs["context"]
            )
            
            print(f"  - Processed: {obs['source']} → {obs['predicate']} → {obs['target']}")
        
        # Visualize field state after processing this perspective
        visualize_field_topology(field_state, f"Field After {perspective.replace('_', ' ').title()}")
    
    # Detect cross-frame relationships
    print("\nDetecting cross-frame relationships:")
    relationships = bridge.detect_cross_frame_relationships()
    
    for rel in relationships:
        print(f"  - {rel['source']} → {rel['predicate']} → {rel['target']}")
        print(f"    Perspectives: {', '.join(rel['perspectives'])}")
        print(f"    Resonance: {rel['resonance_score']:.2f}")
    
    # Visualize final field topology
    visualize_field_topology(field_state, "Final Field Topology")
    
    print("\nClimate Risk Integration Test completed successfully!")
    print(f"Visualization images saved to {os.path.join(os.path.dirname(__file__), 'output')}")


if __name__ == "__main__":
    run_integration_test()
