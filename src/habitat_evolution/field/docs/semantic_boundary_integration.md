# Semantic Boundary Integration Guide

## Overview

This guide demonstrates how to integrate the `SemanticBoundaryDetector` with existing data pipelines, Neo4j visualization, and Habitat's learning window system. The approach is domain-agnostic and observation-based, allowing Habitat to learn from fuzzy boundaries in any semantic field without built-in assumptions about specific domains.

## Modality-Independent Integration

The `SemanticBoundaryDetector` works with any input modality, making it ideal for analyzing:

- Text-based research
- Image analysis outputs
- Field study measurements
- Sensor data streams
- Simulation results
- Any other data that can be represented as semantic vectors

## Simple Integration Steps

### 1. Extract Semantic Vectors

```python
from habitat_evolution.field.semantic_boundary_detector import SemanticBoundaryDetector
import numpy as np

# Initialize detector
detector = SemanticBoundaryDetector()

# Extract semantic vectors from your data source
# This works with any modality - the vectors represent the semantic meaning
semantic_vectors = np.array([...])  # Shape: (num_patterns, num_features)

# Pattern metadata (domain-agnostic)
metadata = [
    {"text": "Pattern with high uncertainty between communities", "source": "observation", "timestamp": "2025-03-15T14:30:00Z"},
    {"text": "Pattern showing characteristics of multiple communities", "source": "analysis", "timestamp": "2025-03-15T14:35:00Z"},
    # ...more entries
]
```

### 2. Detect Transition Patterns and Learning Opportunities

```python
# Analyze transition data (domain-agnostic)
results = detector.analyze_transition_data(semantic_vectors, metadata)

# Extract transition patterns and learning opportunities
transition_patterns = results["transition_patterns"]
learning_opportunities = results["learning_opportunities"]
predictive_patterns = results["predictive_patterns"]
field_observer_data = results["field_observer_data"]

print(f"Found {len(transition_patterns)} transition patterns")
print(f"Identified {len(learning_opportunities)} learning opportunities")
print(f"Discovered {len(predictive_patterns)} potentially predictive patterns")
print(f"Field state: {field_observer_data['field_state']}")
```

### 3. Integrate with Habitat's Learning Window System

```python
from habitat_evolution.pattern_aware_rag.learning.learning_control import EventCoordinator

# Create learning window recommendations
window_recommendations = detector.create_learning_window_recommendations(learning_opportunities)

# Initialize the event coordinator
coordinator = EventCoordinator()

# Create learning windows based on recommendations
for recommendation in window_recommendations:
    print(f"Creating learning window for {recommendation['rationale']}")
    
    # Create a learning window with recommended parameters
    window = coordinator.create_learning_window(
        duration_minutes=recommendation["recommended_params"]["duration_minutes"],
        stability_threshold=recommendation["recommended_params"]["stability_threshold"],
        coherence_threshold=recommendation["recommended_params"]["coherence_threshold"],
        max_changes=recommendation["recommended_params"]["max_changes"]
    )
    
    # Register the detector as a field observer
    window.register_field_observer(
        observer_name="semantic_boundary_detector",
        get_data_callback=detector.get_field_observer_data
    )
```

### 4. Integrate with Neo4j

```python
from habitat_evolution.persistence.neo4j_bridge import Neo4jBridge

# Initialize Neo4j connection
neo4j_bridge = Neo4jBridge(uri="bolt://localhost:7687", 
                           username="neo4j", 
                           password="password")

# Store patterns and communities
for idx, vector in enumerate(semantic_vectors):
    neo4j_bridge.store_pattern(
        pattern_id=f"pattern_{idx}",
        vector=vector,
        metadata=metadata[idx] if idx < len(metadata) else {}
    )

# Store transition zones with gradient visualization
for pattern in transition_patterns:
    # Create transition zone relationship
    neo4j_bridge.create_relationship(
        source_id=f"community_{pattern['source_community']}",
        target_id=f"community_{pattern['neighboring_communities'][0]}",
        relationship_type="TRANSITION_ZONE",
        properties={
            "uncertainty": pattern["uncertainty"],
            "pattern_ids": [f"pattern_{pattern['pattern_idx']}"],
            "observed_at": pattern.get("timestamp")
        }
    )
    
    # Tag the pattern as a boundary pattern
    neo4j_bridge.update_node(
        node_id=f"pattern_{pattern['pattern_idx']}",
        labels=["Pattern", "BoundaryPattern"],
        properties={
            "uncertainty": pattern["uncertainty"],
            "gradient_direction": pattern["gradient_direction"]
        }
    )
```

## Learning Opportunity Visualization

Learning opportunities represent potential areas for the system to learn about transitions between semantic communities:

```python
# Store learning opportunities in Neo4j
for opportunity in learning_opportunities:
    # Create learning opportunity node
    neo4j_bridge.create_node(
        node_id=f"learning_opportunity_{opportunity['pattern_idx']}",
        labels=["LearningOpportunity"],
        properties={
            "relevance_score": opportunity["relevance_score"],
            "opportunity_type": opportunity["opportunity_type"],
            "uncertainty": opportunity["uncertainty"],
            "stability_score": opportunity["stability_score"],
            "coherence_score": opportunity["coherence_score"],
            "observed_at": opportunity.get("observed_at")
        }
    )
    
    # Connect to the pattern
    neo4j_bridge.create_relationship(
        source_id=f"learning_opportunity_{opportunity['pattern_idx']}",
        target_id=f"pattern_{opportunity['pattern_idx']}",
        relationship_type="IDENTIFIED_FROM"
    )
    
    # Connect to communities involved
    for community_id in opportunity["communities"]:
        neo4j_bridge.create_relationship(
            source_id=f"learning_opportunity_{opportunity['pattern_idx']}",
            target_id=f"community_{community_id}",
            relationship_type="INVOLVES_COMMUNITY"
        )
```

## Learning Window Recommendations

Visualize learning window recommendations derived from learning opportunities:

```python
# Store learning window recommendations in Neo4j
for recommendation in window_recommendations:
    # Create recommendation node
    neo4j_bridge.create_node(
        node_id=f"window_recommendation_{recommendation['opportunity_id']}",
        labels=["LearningWindowRecommendation"],
        properties={
            "pattern_idx": recommendation["pattern_idx"],
            "rationale": recommendation["rationale"],
            "priority": recommendation["priority"],
            "duration_minutes": recommendation["recommended_params"]["duration_minutes"],
            "stability_threshold": recommendation["recommended_params"]["stability_threshold"],
            "coherence_threshold": recommendation["recommended_params"]["coherence_threshold"],
            "max_changes": recommendation["recommended_params"]["max_changes"]
        }
    )
    
    # Connect to the pattern
    neo4j_bridge.create_relationship(
        source_id=f"window_recommendation_{recommendation['opportunity_id']}",
        target_id=f"pattern_{recommendation['pattern_idx']}",
        relationship_type="TARGETS"
    )
    
    # Connect to communities involved
    for community_id in recommendation["communities"]:
        neo4j_bridge.create_relationship(
            source_id=f"window_recommendation_{recommendation['opportunity_id']}",
            target_id=f"community_{community_id}",
            relationship_type="INVOLVES_COMMUNITY"
        )
```

## Predictive Pattern Integration

Predictive patterns represent potential future transitions:

```python
# Store predictive patterns in Neo4j
for pattern in predictive_patterns:
    neo4j_bridge.update_node(
        node_id=f"pattern_{pattern['pattern_idx']}",
        labels=["Pattern", "PredictivePattern"],
        properties={
            "predictive_type": pattern["predictive_type"],
            "uncertainty": pattern["uncertainty"],
            "confidence": pattern["confidence"]
        }
    )
```

## Neo4j Visualization Schema

The following Cypher queries create visualizations of the transition ecosystem:

```cypher
// Transition zones between communities
MATCH (p:Pattern:BoundaryPattern)
MATCH (c1:Community)-[t:TRANSITION_ZONE]-(c2:Community)
WHERE p.id IN t.pattern_ids
RETURN p, c1, c2, t

// Learning opportunities with connected communities
MATCH (o:LearningOpportunity)-[:IDENTIFIED_FROM]->(p:Pattern)
MATCH (o)-[:INVOLVES_COMMUNITY]->(c:Community)
RETURN o, p, c

// Learning window recommendations
MATCH (r:LearningWindowRecommendation)-[:TARGETS]->(p:Pattern)
MATCH (r)-[:INVOLVES_COMMUNITY]->(c:Community)
RETURN r, p, c

// Predictive patterns
MATCH (p:Pattern:PredictivePattern)
RETURN p
```

## Field Observer Integration

The `SemanticBoundaryDetector` provides field observer data that can be used by Habitat's learning window system:

```python
# Get field observer data
field_data = detector.get_field_observer_data()

print(f"Transition count: {field_data['transition_count']}")
print(f"Mean uncertainty: {field_data['mean_uncertainty']:.2f}")
print(f"Field state: {field_data['field_state']}")
print("Top community connections:")
for connection, count in field_data['top_community_connections']:
    print(f"  {connection}: {count}")
```

## Emergent Transition Characteristics

To visualize emergent characteristics of transitions:

```python
# Extract transition characteristics
characteristics = results["transition_characteristics"]

# Store community connection statistics
for connection, frequency in characteristics["community_connections"].items():
    community_ids = connection.split("-")
    neo4j_bridge.create_relationship(
        source_id=f"community_{community_ids[0]}",
        target_id=f"community_{community_ids[1]}",
        relationship_type="COMMUNITY_CONNECTION",
        properties={
            "frequency": frequency,
            "strength": frequency / sum(characteristics["community_connections"].values())
        }
    )

# Store uncertainty statistics
neo4j_bridge.create_node(
    node_id="uncertainty_stats",
    labels=["Statistics"],
    properties=characteristics["uncertainty_stats"]
)
```

## Benefits of This Approach

1. **Truly Domain-Agnostic**: Works with any data without built-in assumptions
2. **Observation-Based Learning**: Learns from patterns rather than predefined rules
3. **Minimal Integration**: Leverages existing eigendecomposition analysis
4. **Visualization Ready**: Designed to integrate with Neo4j for immediate visualization
5. **Predictive Capabilities**: Identifies emerging transitions before they fully form
6. **Learning Window Focus**: Creates opportunities for system learning at fuzzy boundaries

## Next Steps

1. Implement feedback loop where learning windows inform future analysis
2. Develop temporal analysis to track how transition zones evolve over time
3. Create gradient visualizations that show uncertainty distributions across the field
4. Implement adaptive thresholds that adjust based on observed transition patterns
5. Develop cross-modal transition detection for multi-modal data sources
