"""Example script demonstrating the EigenspaceVisualizer capabilities.

This script shows how to use the EigenspaceVisualizer to create various
visualizations of patterns in eigenspace, including 2D/3D plots, community
boundaries, and dimensional resonance relationships.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from datetime import datetime

from habitat_evolution.field.field_navigator import FieldNavigator
from habitat_evolution.field.pattern_explorer import PatternExplorer
from habitat_evolution.visualization.eigenspace_visualizer import EigenspaceVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_patterns(n_patterns=50, n_dims=10):
    """Generate sample patterns for visualization."""
    # Generate random vectors
    vectors = np.random.randn(n_patterns, n_dims)
    
    # Normalize vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    # Create communities by adding bias to certain dimensions
    communities = np.zeros(n_patterns, dtype=int)
    for i in range(n_patterns):
        if i < n_patterns // 3:
            vectors[i, :3] += 1.0  # Community 0: strong in first 3 dimensions
            communities[i] = 0
        elif i < 2 * n_patterns // 3:
            vectors[i, 3:6] += 1.0  # Community 1: strong in dimensions 3-5
            communities[i] = 1
        else:
            vectors[i, 6:9] += 1.0  # Community 2: strong in dimensions 6-8
            communities[i] = 2
    
    # Normalize again after adding community bias
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    return vectors, communities

def create_output_directory():
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).parent / "output" / timestamp
    
    # Create directory structure
    (base_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (base_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (base_dir / "analysis").mkdir(parents=True, exist_ok=True)
    
    return base_dir

def main():
    # Generate sample data
    logger.info("Generating sample patterns...")
    vectors, communities = generate_sample_patterns()
    
    # Create output directory
    output_dir = create_output_directory()
    logger.info(f"Created output directory: {output_dir}")
    
    # Initialize components
    field_navigator = FieldNavigator()
    pattern_explorer = PatternExplorer(field_navigator)
    
    # Analyze field topology
    logger.info("Analyzing field topology...")
    field_data = pattern_explorer.analyze_field(vectors)
    
    # Add community information
    for i, pattern in enumerate(field_data["patterns"]):
        pattern["community"] = int(communities[i])
    
    # Initialize visualizer
    visualizer = EigenspaceVisualizer(field_data)
    
    # Create 2D visualization
    logger.info("Creating 2D visualization...")
    fig_2d, ax_2d = visualizer.visualize_eigenspace_2d(
        dim1=0, dim2=1,
        highlight_boundaries=True,
        highlight_communities=True,
        title="Pattern Communities in 2D Eigenspace"
    )
    fig_2d.savefig(output_dir / "visualizations" / "eigenspace_2d.png")
    
    # Create 2D visualization with different dimensions
    logger.info("Creating alternative 2D visualization...")
    fig_2d_alt, ax_2d_alt = visualizer.visualize_eigenspace_2d(
        dim1=1, dim2=2,
        highlight_boundaries=True,
        highlight_communities=True,
        title="Pattern Communities in Alternative 2D View"
    )
    fig_2d_alt.savefig(output_dir / "visualizations" / "eigenspace_2d_alt.png")
    
    # Save field data for analysis
    logger.info("Saving field analysis data...")
    field_data_path = output_dir / "analysis" / "field_data.json"
    with open(field_data_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            "patterns": [
                {
                    "id": p.get("id", f"pattern_{i}"),
                    "community": int(p.get("community", 0)),
                    "coordinates": [float(x) for x in p.get("coordinates", [])],
                    "is_boundary": bool(p.get("is_boundary", False))
                }
                for i, p in enumerate(field_data["patterns"])
            ],
            "eigenvalues": [float(x) for x in field_data.get("eigenvalues", [])],
            "effective_dimensions": int(field_data.get("effective_dimensions", 0))
        }
        json.dump(serializable_data, f, indent=2)
    
    # Save metrics
    logger.info("Saving performance metrics...")
    metrics = {
        "num_patterns": len(vectors),
        "num_dimensions": vectors.shape[1],
        "num_communities": len(set(communities)),
        "effective_dimensions": field_data.get("effective_dimensions", 0),
        "eigenvalues": [float(x) for x in field_data.get("eigenvalues", [])][:5],  # Top 5 eigenvalues
        "boundary_patterns": sum(1 for p in field_data["patterns"] if p.get("is_boundary", False))
    }
    
    metrics_path = output_dir / "metrics" / "visualization_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Visualization example completed. Results saved to: {output_dir}")
    
    # Close all figures
    plt.close('all')

if __name__ == "__main__":
    main()
