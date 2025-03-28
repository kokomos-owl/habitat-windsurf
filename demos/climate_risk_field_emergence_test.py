#!/usr/bin/env python
"""
Climate Risk Field Emergence Test

This script demonstrates how patterns emerge naturally from field dynamics
rather than being predefined, using representative observations rather than
declarative relationships.
"""

import json
import sys
import os
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import uuid

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.habitat_evolution.field.observation_frame_bridge import ObservationFrameBridge
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)


class MockFieldState:
    """A simplified mock of TonicHarmonicFieldState that works with ObservationFrameBridge."""
    
    def __init__(self, field_analysis: Dict[str, Any]):
        """Initialize with the basic structure needed for testing."""
        self.id = str(uuid.uuid4())
        self.field_analysis = field_analysis
        self.patterns = {}
        self.resonance_relationships = {}
        
    def update_from_field_analysis(self, field_analysis: Dict[str, Any]) -> None:
        """Update the field state from field analysis results."""
        self.field_analysis = field_analysis
        
    def get_field_topology(self) -> Dict[str, Any]:
        """Return the field topology."""
        return self.field_analysis["topology"]
    
    def get_field_density(self) -> Dict[str, Any]:
        """Return the field density."""
        return self.field_analysis["density"]
    
    def get_field_properties(self) -> Dict[str, Any]:
        """Return the field properties."""
        return self.field_analysis["field_properties"]


def initialize_field_state() -> MockFieldState:
    """Initialize a field state with minimal assumptions."""
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
            "density_centers": [],  # Start with no density centers - let them emerge
            "density_map": {
                "resolution": [10, 10, 10, 10, 10],
                "values": np.zeros((10, 10, 10, 10, 10)).tolist()  # Start with uniform density
            }
        },
        "field_properties": {
            "coherence": 0.5,  # Start with neutral coherence
            "navigability_score": 0.5,
            "stability": 0.5,
            "resonance_patterns": []  # Start with no resonance patterns - let them emerge
        }
    }
    return MockFieldState(field_analysis)


def process_raw_measurements(observation: Dict[str, Any], adaptive_id: AdaptiveID) -> List[Dict[str, Any]]:
    """
    Process raw measurements into field-ready observations without imposing relationships.
    This function extracts statistical properties and correlations from raw data.
    """
    processed_observations = []
    
    # Extract entities
    entities = observation["observed_entities"]
    raw_data = observation["raw_measurements"]
    
    # Create entity IDs
    entity_ids = {}
    for entity in entities:
        entity_ids[entity] = adaptive_id.get_or_create_id(entity)
    
    # Extract time series data if available
    time_series_keys = []
    for key, value in raw_data.items():
        if isinstance(value, list) and len(value) > 1 and not isinstance(value[0], str):
            time_series_keys.append(key)
    
    # Calculate correlations between time series
    if len(time_series_keys) >= 2:
        for i, key1 in enumerate(time_series_keys):
            for key2 in time_series_keys[i+1:]:
                # Only process if both series have the same length
                if len(raw_data[key1]) == len(raw_data[key2]):
                    # Calculate correlation
                    correlation = np.corrcoef(raw_data[key1], raw_data[key2])[0, 1]
                    
                    # Only process strong correlations (positive or negative)
                    if abs(correlation) > 0.5:
                        # Map the keys to entities
                        entity1 = None
                        entity2 = None
                        
                        # Simple heuristic to map measurement keys to entities
                        for entity in entities:
                            entity_lower = entity.lower()
                            key1_lower = key1.lower()
                            key2_lower = key2.lower()
                            
                            if any(word in key1_lower for word in entity_lower.split()):
                                entity1 = entity
                            if any(word in key2_lower for word in entity_lower.split()):
                                entity2 = entity
                        
                        # If we couldn't map to specific entities, use the first two
                        if not entity1:
                            entity1 = entities[0]
                        if not entity2:
                            entity2 = entities[1] if len(entities) > 1 else entities[0]
                        
                        # Create an observation based on the correlation
                        processed_observations.append({
                            "source_id": entity_ids[entity1],
                            "target_id": entity_ids[entity2],
                            "predicate": "statistically_correlated_with",
                            "context": {
                                "perspective": observation["context"]["perspective"],
                                "correlation_strength": correlation,
                                "measurement_types": [key1, key2],
                                "sample_size": len(raw_data[key1]),
                                "field_vector": observation["context"]["field_vector"]
                            }
                        })
    
    # Compare with control measurements if available
    for key in raw_data:
        if key.startswith("control_"):
            # Find the corresponding non-control measurement
            base_key = key[8:]  # Remove "control_" prefix
            if base_key in raw_data:
                # Calculate percent difference
                if isinstance(raw_data[base_key], list) and isinstance(raw_data[key], list):
                    # For time series, use the average
                    avg_base = np.mean(raw_data[base_key])
                    avg_control = np.mean(raw_data[key])
                    if avg_control != 0:
                        percent_diff = (avg_base - avg_control) / avg_control * 100
                    else:
                        percent_diff = 0
                else:
                    # For single values
                    if raw_data[key] != 0:
                        percent_diff = (raw_data[base_key] - raw_data[key]) / raw_data[key] * 100
                    else:
                        percent_diff = 0
                
                # Only process significant differences
                if abs(percent_diff) > 10:
                    # Map the measurement to entities
                    entity1 = None
                    entity2 = None
                    
                    # Simple heuristic to map measurement keys to entities
                    for entity in entities:
                        entity_lower = entity.lower()
                        key_lower = base_key.lower()
                        
                        if any(word in key_lower for word in entity_lower.split()):
                            entity1 = entity
                    
                    # If we couldn't map to a specific entity, use the first one
                    if not entity1:
                        entity1 = entities[0]
                    
                    # The second entity is the practice or intervention
                    for entity in entities:
                        if entity != entity1:
                            entity2 = entity
                            break
                    
                    # If we still don't have a second entity, use the first one again
                    if not entity2:
                        entity2 = entities[0]
                    
                    # Create an observation based on the difference
                    processed_observations.append({
                        "source_id": entity_ids[entity2],
                        "target_id": entity_ids[entity1],
                        "predicate": "differs_from_control_by",
                        "context": {
                            "perspective": observation["context"]["perspective"],
                            "percent_difference": percent_diff,
                            "measurement_type": base_key,
                            "field_vector": observation["context"]["field_vector"]
                        }
                    })
    
    # If we have location data, create spatial observations
    if "location_coordinates" in raw_data:
        # Find observations with similar locations
        location = raw_data["location_coordinates"]
        processed_observations.append({
            "source_id": entity_ids[entities[0]],
            "target_id": adaptive_id.get_or_create_id(f"location_{location[0]}_{location[1]}"),
            "predicate": "located_at",
            "context": {
                "perspective": observation["context"]["perspective"],
                "latitude": location[0],
                "longitude": location[1],
                "field_vector": observation["context"]["field_vector"]
            }
        })
    
    return processed_observations


def visualize_field_topology(field_state: MockFieldState, title: str):
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
    
    # Add some default density for visualization if no centers exist
    if not field_state.get_field_density()["density_centers"]:
        # Create a simple gradient for visualization
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = 0.5 * np.exp(-3 * ((X[i, j] - 0.5)**2 + (Y[i, j] - 0.5)**2))
    else:
        # Use actual density centers
        for i in range(resolution):
            for j in range(resolution):
                for center in field_state.get_field_density()["density_centers"]:
                    pos = center["position"]
                    dist = np.sqrt((X[i, j] - pos[dim1])**2 + (Y[i, j] - pos[dim2])**2)
                    Z[i, j] += center["magnitude"] * np.exp(-5 * dist)
    
    # Custom colormap for visualization
    colors = [(0.1, 0.1, 0.6), (0.2, 0.5, 0.9), (0.9, 0.9, 0.2), (0.9, 0.4, 0.1)]
    cmap = LinearSegmentedColormap.from_list('habitat_cmap', colors, N=256)
    
    # Plot the density map
    plt.contourf(X, Y, Z, 20, cmap=cmap)
    plt.colorbar(label='Field Density')
    
    # Plot density centers if they exist
    if field_state.get_field_density()["density_centers"]:
        for center in field_state.get_field_density()["density_centers"]:
            pos = center["position"]
            plt.plot(pos[dim1], pos[dim2], 'ro', markersize=10*center["magnitude"])
    
    # Plot resonance patterns if they exist
    if field_state.get_field_properties()["resonance_patterns"]:
        for pattern in field_state.get_field_properties()["resonance_patterns"]:
            pos = pattern["center"]
            plt.plot(pos[dim1], pos[dim2], 'go', markersize=10*pattern["intensity"])
    
    # Add labels and title
    plt.xlabel(field_state.get_field_topology()["principal_dimensions"][dim1])
    plt.ylabel(field_state.get_field_topology()["principal_dimensions"][dim2])
    plt.title(title)
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.close()


def run_field_emergence_test():
    """Run the field emergence test with representative observations."""
    print("Starting Climate Risk Field Emergence Test...")
    
    # Load observation frames and observations
    frames_path = os.path.join(os.path.dirname(__file__), 'data/climate_risk/regional_observation_frames.json')
    observations_path = os.path.join(os.path.dirname(__file__), 'data/climate_risk/regional_observations_representative.json')
    
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
    
    # Process observations
    print("\nProcessing observations:")
    
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
            # Process raw measurements into field-ready observations
            processed_obs = process_raw_measurements(obs, adaptive_id)
            
            # Process each derived observation
            for proc_obs in processed_obs:
                # Process the observation
                bridge.process_observation(
                    source_id=proc_obs["source_id"],
                    predicate=proc_obs["predicate"],
                    target_id=proc_obs["target_id"],
                    context=proc_obs["context"]
                )
                
                print(f"  - Processed: {proc_obs['source_id']} → {proc_obs['predicate']} → {proc_obs['target_id']}")
        
        # Visualize field state after processing this perspective
        visualize_field_topology(field_state, f"Field After {perspective.replace('_', ' ').title()}")
    
    # Detect emergent patterns
    print("\nDetecting emergent patterns from field topology:")
    
    # The patterns emerge from the field topology itself, not from predefined categories
    resonance_centers = field_state.get_field_properties()["resonance_patterns"]
    
    for i, center in enumerate(resonance_centers):
        print(f"  - Emergent Pattern {i+1}:")
        print(f"    Center: {center['center']}")
        print(f"    Intensity: {center['intensity']:.2f}")
        
        # Find entities close to this resonance center
        # In a real implementation, this would use the field's topology to identify related entities
        print(f"    Related entities would be identified through field topology")
    
    # Detect cross-frame relationships
    print("\nDetecting cross-frame relationships that emerged naturally:")
    relationships = bridge.detect_cross_frame_relationships()
    
    for rel in relationships:
        print(f"  - {rel['source']} → {rel['predicate']} → {rel['target']}")
        print(f"    Perspectives: {', '.join(rel['perspectives'])}")
        print(f"    Resonance: {rel['resonance_score']:.2f}")
    
    # Visualize final field topology
    visualize_field_topology(field_state, "Final Field Topology")
    
    print("\nClimate Risk Field Emergence Test completed successfully!")
    print(f"Visualization images saved to {os.path.join(os.path.dirname(__file__), 'output')}")


if __name__ == "__main__":
    run_field_emergence_test()
