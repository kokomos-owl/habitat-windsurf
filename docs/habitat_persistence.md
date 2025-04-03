# Habitat Persistence Layer

## Overview

The Habitat Persistence Layer provides a flexible, modular architecture for storing, retrieving, and querying pattern evolution data. This document outlines the key components, interfaces, and usage examples to help developers understand and extend the persistence capabilities.

The persistence layer follows clean architecture principles with:
- **Interfaces**: Define contracts for repositories
- **Adapters**: Implement interfaces for specific databases
- **Factory**: Create appropriate repository instances
- **Query Capabilities**: Support for semantic, vector, and temporal queries
- **Event-Driven Integration**: Connect with the vector-tonic-window system through events

This architecture supports the pattern evolution and co-evolution principles that are core to Habitat Evolution, enabling detection of coherent patterns and observation of semantic change across the system.

## Architecture

### Core Components

```
habitat_evolution/
├── adaptive_core/
│   ├── emergence/
│   │   ├── vector_tonic_persistence_connector.py  # Connects vector-tonic events to persistence
│   │   ├── repository_factory.py                  # Creates repository instances
│   │   └── interfaces/                            # Repository interfaces
│   └── persistence/
│       ├── interfaces/                            # Repository interfaces
│       ├── adapters/                              # Concrete implementations
│       └── factory.py                             # Factory for creating repositories
└── pattern_aware_rag/
    └── persistence/
        └── arangodb/                              # ArangoDB-specific implementations
```

### Integration with Vector-Tonic-Window System

The `VectorTonicPersistenceConnector` serves as the bridge between the vector-tonic-window system's events and the persistence layer. It subscribes to events such as pattern detection, field state changes, and topology updates, then persists this data using the appropriate repositories.

### Repository Interfaces

The persistence layer defines several key interfaces:

- **FieldStateRepositoryInterface**: For storing and retrieving field states
- **PatternRepositoryInterface**: For pattern persistence
- **RelationshipRepositoryInterface**: For relationship management
- **TopologyRepositoryInterface**: For topology constructs
- **SemanticSignatureRepositoryInterface**: For semantic signatures
- **BoundaryRepositoryInterface**: For storing and retrieving boundary constructs
- **PredicateRelationshipRepositoryInterface**: For managing predicate relationships

These interfaces define contracts that concrete implementations must fulfill, enabling dependency injection and making the system more testable and maintainable.

## Data Operations

### Storing Data

Each repository provides a `save` method for persisting entities. The repositories work with dictionaries rather than strict object models to provide flexibility in the data structures:

```python
# Example: Saving a pattern
from src.habitat_evolution.adaptive_core.persistence.factory import create_repositories

# Create repositories using the factory
db_connection = get_db_connection()
repositories = create_repositories(db_connection)
pattern_repo = repositories["pattern_repository"]

# Create pattern as a dictionary
pattern_data = {
    "id": "pattern-123",
    "name": "Emerging Concept",
    "vector": [0.1, 0.2, 0.3, 0.4],
    "confidence": 0.87,
    "metadata": {
        "source": "document-456",
        "timestamp": "2025-04-02T21:00:00Z"
    }
}

# Save to database
pattern_id = pattern_repo.save(pattern_data)
print(f"Saved pattern with ID: {pattern_id}")
# Output: Saved pattern with ID: pattern-123
```

#### Benefits of Dictionary-Based Approach

The use of dictionaries for data storage provides several advantages:

1. **Dynamic Schema Evolution**: Allows the schema to evolve without requiring code changes to model classes
2. **ArangoDB Integration**: Naturally works with ArangoDB's document-oriented structure
3. **Flexible Attribute Sets**: Different entities can have varying sets of attributes based on context
4. **Simplified Serialization**: Easier to serialize/deserialize for storage and transmission

### Retrieving Data

Basic retrieval operations return data as dictionaries:

```python
# Example: Retrieving a pattern by ID
pattern = pattern_repo.find_by_id("pattern-123")

print(f"Retrieved pattern: {pattern['name']}")
# Output: Retrieved pattern: Emerging Concept

print(f"Pattern vector: {pattern['vector'][:3]}...")
# Output: Pattern vector: [0.1, 0.2, 0.3]...

# Access metadata
print(f"Source: {pattern['metadata']['source']}")
# Output: Source: document-456
```

## Query Capabilities

### Basic Queries

Each repository provides standard query methods:

```python
# Find by ID
entity = repo.find_by_id("entity-123")

# Find all
all_entities = repo.find_all()

# Find by type
patterns = pattern_repo.find_by_type("concept")

# Find by metadata attribute
patterns = pattern_repo.find_by_metadata("source", "document-456")
```

### Advanced Queries

#### Temporal Queries

```python
# Find patterns that emerged in a specific time range
from datetime import datetime, timedelta

start_time = datetime.now() - timedelta(days=7)
end_time = datetime.now()

recent_patterns = pattern_repo.find_by_time_range(start_time, end_time)
print(f"Found {len(recent_patterns)} patterns in the last week")
# Output: Found 42 patterns in the last week

# Get the latest topology state
latest_topology = topology_repo.get_latest_topology_state()
```

#### Vector Similarity Queries

```python
# Find similar patterns using vector similarity
query_vector = [0.15, 0.22, 0.31, 0.42]
similar_patterns = pattern_repo.find_similar(query_vector, threshold=0.8, limit=5)

print(f"Found {len(similar_patterns)} similar patterns")
# Output: Found 3 similar patterns

# Display similarity scores
for pattern, score in similar_patterns:
    print(f"Pattern: {pattern.name}, Similarity: {score:.2f}")
# Output:
# Pattern: Emerging Concept, Similarity: 0.95
# Pattern: Related Idea, Similarity: 0.87
# Pattern: Connected Theme, Similarity: 0.82
```

#### Semantic Queries

```python
# Find patterns by semantic meaning
from habitat_evolution.pattern_aware_rag.persistence.arangodb.semantic_signature_repository import SemanticSignature

# Create a semantic signature
query_signature = SemanticSignature(
    signature_vector=[0.1, 0.3, 0.5, 0.7],
    temporal_context={"timeframe": "recent"}
)

# Find semantically similar signatures
signature_repo = create_semantic_signature_repository(db_connection)
similar_signatures = signature_repo.find_similar(query_signature, threshold=0.7)

# Process results
for signature, score in similar_signatures:
    print(f"Entity: {signature.entity_id}, Similarity: {score:.2f}")
# Output:
# Entity: concept-789, Similarity: 0.88
# Entity: theme-456, Similarity: 0.75
```

## Explanation Capabilities

The persistence layer integrates with semantic explanation components:

```python
from habitat_evolution.adaptive_core.oscillation.semantic_signature_interface import SemanticSignatureInterface, SignatureNarrativeGenerator

# Create the interface
signature_service = get_signature_service()
semantic_interface = SemanticSignatureInterface(signature_service)

# Explain concept behavior
explanation = semantic_interface.explain_concept_behavior("climate_change")
print(f"Concept Explanation: {explanation}")
# Output: Concept Explanation: Climate change shows a gradually increasing 
# oscillatory pattern with growing amplitude, indicating an intensifying 
# phenomenon with widening effects. The signature reveals complex harmonics 
# suggesting multiple interacting factors.

# Explain relationship between concepts
relationship = semantic_interface.explain_concept_relationship(
    "renewable_energy", "fossil_fuels"
)
print(f"Relationship: {relationship}")
# Output: Relationship: Renewable energy and fossil fuels exhibit counter-phase 
# oscillations, indicating an inverse relationship. As renewable energy concepts 
# gain prominence (increasing amplitude), fossil fuel concepts show decreasing 
# amplitude, suggesting a gradual replacement dynamic.
```

## Assertion Capabilities

The persistence layer supports making and verifying assertions about patterns:

```python
## Event-Driven Persistence

The `VectorTonicPersistenceConnector` provides event-driven persistence capabilities:

```python
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import create_connector
from src.habitat_evolution.core.services.event_bus import LocalEventBus

# Create event bus and connector
event_bus = LocalEventBus()
db_connection = get_db_connection()
connector = create_connector(event_bus=event_bus, db=db_connection)

# The connector automatically subscribes to relevant events
# When events occur, data is persisted to the appropriate repositories

# Example: Manually trigger pattern detection event
pattern_data = {
    "id": "pattern-456",
    "name": "Emerging Theme",
    "vector": [0.2, 0.3, 0.4, 0.5],
    "confidence": 0.92
}

connector.on_pattern_detected(
    pattern_id="pattern-456",
    pattern_data=pattern_data,
    metadata={"source": "manual-trigger"}
)
# This will persist the pattern and publish a pattern.detected event
```

### Event Handlers

The connector implements handlers for various events:

```python
# Pattern events
connector.on_pattern_detected(pattern_id, pattern_data, metadata)
connector.on_pattern_evolution(pattern_id, previous_state, new_state, metadata)
connector.on_pattern_quality_change(pattern_id, previous_quality, new_quality, metadata)
connector.on_pattern_relationship_detected(source_id, target_id, relationship_data, metadata)
connector.on_pattern_merge(merged_pattern_id, source_pattern_ids, merged_pattern_data, metadata)
connector.on_pattern_split(source_pattern_id, result_pattern_ids, result_pattern_data, metadata)

# Field state events
connector.on_field_state_change(field_id, previous_state, new_state, metadata)
connector.on_field_coherence_change(field_id, previous_coherence, new_coherence, metadata)
connector.on_field_stability_change(field_id, previous_stability, new_stability, metadata)
connector.on_density_center_shift(field_id, previous_centers, new_centers, metadata)
connector.on_eigenspace_change(field_id, previous_eigenspace, new_eigenspace, metadata)
connector.on_topology_change(field_id, previous_topology, new_topology, metadata)

# Learning window events
connector.on_window_state_change(window_id, previous_state, new_state, metadata)
connector.on_window_open(window_id, window_data, metadata)
connector.on_window_close(window_id, window_data, metadata)
connector.on_back_pressure(window_id, pressure_data, metadata)
```

## Assertion Capabilities

The persistence layer supports making and verifying assertions about patterns:

```python
from src.habitat_evolution.adaptive_core.assertion.pattern_assertion import PatternAssertion

# Create an assertion about a pattern
assertion = PatternAssertion(
    pattern_id="pattern-123",
    assertion_type="trend",
    assertion_value="increasing",
    confidence=0.85,
    evidence=["observation-1", "observation-2"]
)

# Save the assertion
assertion_repo = create_assertion_repository(db_connection)
assertion_id = assertion_repo.save(assertion)

# Verify an assertion
verification = assertion_repo.verify_assertion(assertion_id)
print(f"Assertion verified: {verification.is_valid}")
# Output: Assertion verified: True
print(f"Confidence: {verification.confidence}")
# Output: Confidence: 0.87
```

## Debug and Logging Examples

### Repository Operations

```
2025-04-02 14:32:15 [DEBUG] habitat_evolution.persistence: Creating pattern repository with connection <ArangoDB connection 0x7f8a2c3d1e80>
2025-04-02 14:32:15 [INFO] habitat_evolution.persistence: Pattern repository initialized with collection 'patterns'
2025-04-02 14:32:16 [DEBUG] habitat_evolution.persistence: Saving pattern with ID pattern-123
2025-04-02 14:32:16 [DEBUG] habitat_evolution.persistence: Converting pattern to document properties: {'_key': 'pattern-123', 'name': 'Emerging Concept', ...}
2025-04-02 14:32:16 [INFO] habitat_evolution.persistence: Successfully saved pattern pattern-123
```

### Query Execution

```
2025-04-02 14:35:22 [DEBUG] habitat_evolution.persistence: Executing vector similarity query with threshold 0.8
2025-04-02 14:35:22 [DEBUG] habitat_evolution.persistence: Query vector: [0.15, 0.22, 0.31, 0.42]
2025-04-02 14:35:22 [DEBUG] habitat_evolution.persistence: Retrieved 10 candidates for similarity comparison
2025-04-02 14:35:22 [DEBUG] habitat_evolution.persistence: Calculated similarities: {'pattern-123': 0.95, 'pattern-456': 0.87, ...}
2025-04-02 14:35:22 [INFO] habitat_evolution.persistence: Found 3 patterns above similarity threshold 0.8
```

### Semantic Processing

```
2025-04-02 14:40:05 [DEBUG] habitat_evolution.semantic: Translating oscillatory signature to semantic concepts
2025-04-02 14:40:05 [DEBUG] habitat_evolution.semantic: Extracted properties: frequency=0.25, amplitude=0.8, phase=0.3
2025-04-02 14:40:05 [DEBUG] habitat_evolution.semantic: Generated semantic concepts: {'change_rate': 'moderate', 'impact': 'significant', ...}
2025-04-02 14:40:05 [INFO] habitat_evolution.semantic: Successfully translated signature to 5 semantic concepts
```

## Factory Usage

The factory pattern simplifies repository creation:

```python
from habitat_evolution.adaptive_core.persistence.factory import create_repositories

# Create all repositories with a single call
db_connection = get_db_connection()
repositories = create_repositories(db_connection)

# Access specific repositories
field_state_repo = repositories["field_state_repository"]
pattern_repo = repositories["pattern_repository"]
relationship_repo = repositories["relationship_repository"]
topology_repo = repositories["topology_repository"]

# Use repositories
patterns = pattern_repo.find_all()
print(f"Total patterns: {len(patterns)}")
```

## Best Practices

1. **Use the Factory**: Always use the factory methods to create repositories
2. **Depend on Interfaces**: Code against interfaces, not concrete implementations
3. **Transaction Management**: Use transactions for operations that modify multiple entities
4. **Error Handling**: Always handle database errors appropriately
5. **Pagination**: Use pagination for queries that might return large result sets

## Extending the Persistence Layer

To add support for a new database:

1. Create a new adapter implementing the repository interfaces
2. Update the factory to support the new adapter
3. Ensure all methods are properly implemented
4. Add appropriate tests

Example:

```python
from habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepositoryInterface

class MongoDBPatternRepositoryAdapter(PatternRepositoryInterface):
    def __init__(self, db_connection):
        self.db = db_connection
        self.collection = self.db["patterns"]
    
    def save(self, pattern):
        # Implementation for MongoDB
        result = self.collection.update_one(
            {"_id": pattern.id},
            {"$set": self._to_document(pattern)},
            upsert=True
        )
        return pattern.id
    
    # Implement other interface methods...
```

## Conclusion

The Habitat Persistence Layer provides a flexible, powerful foundation for storing and querying pattern evolution data. By following clean architecture principles and providing rich query capabilities, it enables the development of sophisticated pattern evolution and co-evolution features while maintaining code quality and testability.
