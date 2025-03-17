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

from ..field_navigator import FieldNavigator
from ..pattern_explorer import PatternExplorer
from ..topological_field_analyzer import TopologicalFieldAnalyzer
from ...visualization.eigenspace_visualizer import EigenspaceVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    
    # Ensure all communities are integers
    communities = communities.astype(int)
    
    # Normalize again after adding community bias
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    return vectors, communities

def create_output_directory():
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).parent / "output" / timestamp
    
    # Create directory structure
    vis_dir = base_dir / "visualizations"
    metrics_dir = base_dir / "metrics"
    analysis_dir = base_dir / "analysis"
    
    # Create all directories
    for dir_path in [vis_dir, metrics_dir, analysis_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    # Create a README.md in the output directory
    readme_path = base_dir / "README.md"
    readme_content = f"""# Visualization Results - {timestamp}

## Directory Structure
- visualizations/: Generated plots and visualizations
- metrics/: Performance and analysis metrics
- analysis/: Pattern analysis results
"""
    readme_path.write_text(readme_content)
    
    return {
        "base_dir": base_dir,
        "vis_dir": vis_dir,
        "metrics_dir": metrics_dir,
        "analysis_dir": analysis_dir
    }

def main():
    # Create output directories
    output_dirs = create_output_directory()
    
    # Generate sample patterns
    vectors, communities = generate_sample_patterns(n_patterns=50, n_dims=10)
    
    # Create pattern metadata
    pattern_metadata = [
        {
            "id": f"pattern_{i}",
            "community": communities[i],
            "timestamp": datetime.now().isoformat()
        } for i in range(len(vectors))
    ]
    
    # Initialize components
    field_analyzer = TopologicalFieldAnalyzer()
    field_navigator = FieldNavigator(field_analyzer)
    pattern_explorer = PatternExplorer(field_navigator)
    
    # Analyze field topology using raw vectors
    field_analysis = field_analyzer.analyze_field(vectors, pattern_metadata)
    
    # Create resonance matrix from field analysis
    resonance_matrix = field_analysis['resonance_matrix']
    
    # Set the analyzed field for navigation
    field_navigator.set_field(resonance_matrix, pattern_metadata)
    
    # Prepare visualization data
    vis_data = {
        "patterns": [
            {
                "index": i,
                "id": f"pattern_{i}",
                "coordinates": field_analysis['topology']['eigenspace_coordinates'][i],
                "community": int(communities[i])
            } for i in range(len(vectors))
        ],
        "boundaries": [
            {"pattern_idx": zone["pattern_idx"]} 
            for zone in field_analysis['transition_zones']['transition_zones']
        ]
    }
    
    # Initialize visualizer with field data
    visualizer = EigenspaceVisualizer(vis_data)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # 2D eigenspace plot
    visualizer.visualize_eigenspace_2d(
        title="Pattern Distribution in 2D Eigenspace",
        save_path=str(output_dirs["vis_dir"] / "eigenspace_2d.png")
    )
    
    # Alternative 2D view (using different eigenvectors)
    visualizer.visualize_eigenspace_2d(
        dim1=2, dim2=3,  # Use 3rd and 4th eigenvectors
        title="Alternative Pattern Distribution View",
        save_path=str(output_dirs["vis_dir"] / "eigenspace_2d_alt.png")
    )
    
    # Save analysis results
    analysis_file = output_dirs["analysis_dir"] / "field_data.json"
    with open(analysis_file, "w") as f:
        json.dump({
            "n_patterns": len(vectors),
            "n_dimensions": vectors.shape[1],
            "n_communities": len(set(communities)),
            "community_sizes": [
                sum(1 for c in communities if c == i)
                for i in range(max(communities) + 1)
            ],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    # Save visualization metrics
    metrics_file = output_dirs["metrics_dir"] / "visualization_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "eigenvalues": field_analysis["eigenvalues"].tolist(),
            "explained_variance_ratio": field_analysis["explained_variance_ratio"].tolist(),
            "effective_dimensionality": field_analysis["effective_dimensionality"],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_dirs['base_dir']}")

if __name__ == "__main__":
    main()
