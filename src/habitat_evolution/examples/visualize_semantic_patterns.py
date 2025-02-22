"""Example script to visualize semantic patterns in Neo4j."""

from habitat_evolution.tests.visualization.test_semantic_pattern_visualization import (
    SemanticPatternVisualizer,
    EventNode,
    TemporalNode,
    SemanticRelation,
    ValidationStatus
)

def create_sample_graph():
    """Create a sample semantic graph for visualization."""
    # Create temporal nodes
    current = TemporalNode(
        id="current_2025",
        period="current",
        year=2025
    )
    current.node_type = "temporal"
    current.temporal_horizon = "current"
    current.probability = 1.0
    
    mid_century = TemporalNode(
        id="mid_2050",
        period="mid_century",
        year=2050
    )
    mid_century.node_type = "temporal"
    mid_century.temporal_horizon = "mid_century"
    mid_century.probability = 1.0
    
    temporal_nodes = [current, mid_century]

    # Create event nodes
    rainfall = EventNode(
        id="rainfall_100yr",
        event_type="extreme_precipitation",
        metrics={
            "current_probability": 1.0,
            "mid_increase": 1.2,
            "late_increase": 5.0
        }
    )
    rainfall.node_type = "event"
    rainfall.temporal_horizon = "current"
    rainfall.probability = 1.0
    
    drought = EventNode(
        id="drought_severe",
        event_type="drought",
        metrics={
            "current_probability": 0.085,
            "mid_probability": 0.13,
            "late_probability": 0.26
        }
    )
    drought.node_type = "event"
    drought.temporal_horizon = "mid_century"
    drought.probability = 0.13
    
    event_nodes = [rainfall, drought]

    # Create relationships using object references
    relationships = [
        SemanticRelation(
            source_id=current.id,
            target_id=rainfall.id,
            relation_type="CONTAINS",
            strength=0.8,
            evidence=["Temporal context for extreme precipitation"]
        ),
        SemanticRelation(
            source_id=mid_century.id,
            target_id=drought.id,
            relation_type="CONTAINS",
            strength=0.9,
            evidence=["Temporal context for drought conditions"]
        ),
        SemanticRelation(
            source_id=rainfall.id,
            target_id=drought.id,
            relation_type="INFLUENCES",
            strength=0.7,
            evidence=["Precipitation patterns impact drought severity"]
        )
    ]

    return {
        "temporal_nodes": temporal_nodes,
        "event_nodes": event_nodes,
        "relationships": relationships
    }

def main():
    """Main function to demonstrate semantic pattern visualization."""
    # Create visualizer
    visualizer = SemanticPatternVisualizer()

    # Create and validate semantic graph
    semantic_graph = create_sample_graph()
    
    try:
        # Extract and validate patterns
        patterns = visualizer.extract_patterns_from_semantic_graph(semantic_graph)
        print(f"Successfully extracted {len(patterns)} patterns")
        
        # Export to Neo4j
        # Create a simple 2D field for visualization
        import numpy as np
        field = np.zeros((10, 10))  # 10x10 grid for simplicity
        visualizer.export_pattern_graph_to_neo4j(patterns, field)
        print("Successfully exported patterns to Neo4j")
        
        # Check validation status
        status = visualizer.validator.get_ui_status()
        print(f"Validation Status: {status['status'].value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
