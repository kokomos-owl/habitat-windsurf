"""Test script for climate risk visualization."""

from src.flow_representation.visualization.metrics.town_metrics import TownMetricsProcessor
from src.flow_representation.visualization.maps.leaflet_visualizer import LeafletVisualizer
from pathlib import Path

def main():
    """Create test visualization with sample data."""
    # Sample metrics data (this will be replaced with real data)
    sample_metrics = {
        'Aquinnah_coherence': 0.75,
        'Chilmark_coherence': 0.82,
        'Edgartown_coherence': 0.68,
        'Oak Bluffs_coherence': 0.71,
        'Tisbury_coherence': 0.79,
        'West Tisbury_coherence': 0.73,
        
        'Aquinnah_cross_pattern_flow': 0.45,
        'Chilmark_cross_pattern_flow': 0.52,
        'Edgartown_cross_pattern_flow': 0.38,
        'Oak Bluffs_cross_pattern_flow': 0.41,
        'Tisbury_cross_pattern_flow': 0.49,
        'West Tisbury_cross_pattern_flow': 0.43,
        
        'Aquinnah_emergence_rate': 0.65,
        'Chilmark_emergence_rate': 0.72,
        'Edgartown_emergence_rate': 0.58,
        'Oak Bluffs_emergence_rate': 0.61,
        'Tisbury_emergence_rate': 0.69,
        'West Tisbury_emergence_rate': 0.63,
        
        'Aquinnah_social_support': 0.85,
        'Chilmark_social_support': 0.92,
        'Edgartown_social_support': 0.78,
        'Oak Bluffs_social_support': 0.81,
        'Tisbury_social_support': 0.89,
        'West Tisbury_social_support': 0.83
    }
    
    # Initialize processors
    metrics_processor = TownMetricsProcessor()
    metrics_processor.process_climate_metrics(sample_metrics)
    
    # Create visualization
    visualizer = LeafletVisualizer()
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Generate and save visualization
    output_path = visualizer.generate_visualization(
        metrics_processor.get_town_coordinates(),
        output_dir
    )
    
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
