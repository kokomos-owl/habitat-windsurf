#!/usr/bin/env python
"""
Eigenspace Navigation Example

This script demonstrates how to use the Pattern Explorer and Eigenspace Visualizer
to explore patterns in eigenspace, detect dimensional resonance, and navigate
between patterns through fuzzy boundaries.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.pattern_explorer import PatternExplorer
from habitat_evolution.visualization.eigenspace_visualizer import EigenspaceVisualizer


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_vectors(num_vectors=100, num_dimensions=50, num_communities=5):
    """Generate test vectors with known community structure.
    
    Args:
        num_vectors: Number of vectors to generate
        num_dimensions: Number of dimensions for each vector
        num_communities: Number of communities to create
        
    Returns:
        Tuple of (vectors, metadata, community_assignments)
    """
    # Initialize random seed for reproducibility
    np.random.seed(42)
    
    # Generate community centroids
    centroids = np.random.normal(0, 1, (num_communities, num_dimensions))
    
    # Generate vectors around centroids
    vectors = []
    metadata = []
    community_assignments = []
    
    for i in range(num_vectors):
        # Assign to a community
        community = i % num_communities
        
        # Generate vector near the community centroid
        noise = np.random.normal(0, 0.5, num_dimensions)
        vector = centroids[community] + noise
        
        # Add some dimensional resonance between communities
        if i % 7 == 0:  # Create boundary patterns
            other_community = (community + 1) % num_communities
            vector = 0.5 * centroids[community] + 0.5 * centroids[other_community] + noise
            community_assignments.append([community, other_community])
        else:
            community_assignments.append(community)
        
        # Normalize the vector
        vector = vector / np.linalg.norm(vector)
        
        vectors.append(vector)
        metadata.append({
            "id": f"pattern_{i}",
            "type": "test_pattern",
            "description": f"Test pattern {i} in community {community}"
        })
    
    return np.array(vectors), metadata, community_assignments


def main():
    """Run the eigenspace navigation example."""
    # Generate test vectors
    logger.info("Generating test vectors...")
    vectors, metadata, community_assignments = generate_test_vectors()
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Analyze field topology
    logger.info("Analyzing field topology...")
    analyzer = TopologicalFieldAnalyzer()
    field_analysis = analyzer.analyze_field(vectors, metadata)
    
    # Update metadata with community assignments
    for i, community in enumerate(community_assignments):
        metadata[i]["community"] = community
    
    # Create field navigator
    logger.info("Creating field navigator...")
    navigator = FieldNavigator()
    navigator.set_field(field_analysis, metadata)
    
    # Create pattern explorer
    logger.info("Creating pattern explorer...")
    explorer = PatternExplorer(navigator)
    
    # Export visualization data
    logger.info("Exporting visualization data...")
    viz_data_path = output_dir / "eigenspace_data.json"
    explorer.export_visualization_data(str(viz_data_path))
    
    # Create eigenspace visualizer
    logger.info("Creating eigenspace visualizer...")
    visualizer = EigenspaceVisualizer()
    visualizer.load_data(str(viz_data_path))
    
    # Visualize patterns in 2D eigenspace
    logger.info("Visualizing patterns in 2D eigenspace...")
    fig_2d, ax_2d = visualizer.visualize_eigenspace_2d(
        highlight_boundaries=True,
        highlight_communities=True,
        title="Patterns in 2D Eigenspace",
        save_path=str(output_dir / "eigenspace_2d.png")
    )
    
    # Visualize patterns in 3D eigenspace
    logger.info("Visualizing patterns in 3D eigenspace...")
    fig_3d, ax_3d = visualizer.visualize_eigenspace_3d(
        highlight_boundaries=True,
        highlight_communities=True,
        title="Patterns in 3D Eigenspace",
        save_path=str(output_dir / "eigenspace_3d.png")
    )
    
    # Visualize dimensional resonance
    logger.info("Visualizing dimensional resonance...")
    fig_dim, ax_dim = visualizer.visualize_dimensional_resonance(
        dim=0,
        threshold=0.3,
        title="Dimensional Resonance (Dimension 0)",
        save_path=str(output_dir / "dimensional_resonance.png")
    )
    
    # Find patterns with strong dimensional resonance
    logger.info("Finding patterns with strong dimensional resonance...")
    resonant_patterns = []
    for i in range(len(metadata)):
        for j in range(i+1, len(metadata)):
            resonance = navigator._detect_dimensional_resonance(i, j)
            if resonance and resonance["strength"] > 0.6:
                resonant_patterns.append((i, j, resonance))
    
    # Sort by resonance strength
    resonant_patterns.sort(key=lambda x: x[2]["strength"], reverse=True)
    
    # Print top resonant pattern pairs
    logger.info("Top resonant pattern pairs:")
    for i, j, resonance in resonant_patterns[:5]:
        logger.info(f"Patterns {i} and {j}: Resonance strength = {resonance['strength']:.3f}")
    
    # Explore a specific pattern
    if len(metadata) > 0:
        pattern_idx = 0
        logger.info(f"Exploring pattern {pattern_idx}...")
        pattern_info = explorer.explore_pattern(pattern_idx)
        
        # Print pattern information
        logger.info(f"Pattern ID: {pattern_info['pattern']['id']}")
        logger.info(f"Community: {pattern_info['community']}")
        logger.info(f"Is boundary: {pattern_info['is_boundary']}")
        logger.info(f"Boundary fuzziness: {pattern_info['boundary_fuzziness']:.3f}")
        logger.info(f"Resonant patterns: {len(pattern_info['resonant_patterns'])}")
    
    # Find fuzzy boundaries
    logger.info("Detecting fuzzy boundaries...")
    boundary_info = navigator.detect_fuzzy_boundaries()
    
    # Print boundary information
    logger.info(f"Found {len(boundary_info.get('boundaries', []))} boundary patterns")
    
    # Visualize community boundaries
    if 'communities' in field_analysis and len(field_analysis['communities']) >= 2:
        logger.info("Visualizing community boundaries...")
        community_ids = list(field_analysis['communities'].keys())
        fig_bound, ax_bound = visualizer.visualize_community_boundaries(
            int(community_ids[0]),
            int(community_ids[1]),
            title=f"Boundaries between Communities {community_ids[0]} and {community_ids[1]}",
            save_path=str(output_dir / "community_boundaries.png")
        )
    
    # Navigate between patterns
    if len(metadata) >= 2:
        start_idx = 0
        end_idx = len(metadata) - 1
        
        logger.info(f"Navigating from pattern {start_idx} to pattern {end_idx}...")
        
        # Navigate using eigenspace
        navigation_results = explorer.navigate_between_patterns(
            start_idx, end_idx, method="eigenspace"
        )
        
        # Print navigation results
        logger.info(f"Navigation path length: {navigation_results['path_length']}")
        logger.info(f"Dimensional resonance: {navigation_results['dimensional_resonance']}")
        
        # Visualize navigation path
        logger.info("Visualizing navigation path...")
        fig_path, ax_path = visualizer.visualize_navigation_path(
            navigation_results['path'],
            title=f"Navigation Path from Pattern {start_idx} to Pattern {end_idx}",
            save_path=str(output_dir / "navigation_path.png")
        )
    
    logger.info("Example completed successfully!")
    logger.info(f"Output files saved to {output_dir}")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
