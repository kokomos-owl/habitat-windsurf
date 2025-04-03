# Habitat Persistence Layer

## Overview

The Habitat Persistence Layer provides a flexible, modular architecture for storing, retrieving, and querying pattern evolution data. This document outlines the key components, interfaces, and usage examples to help developers understand and extend the persistence capabilities.

The persistence layer follows clean architecture principles with:
- **Interfaces**: Define contracts for repositories
- **Adapters**: Implement interfaces for specific databases
- **Factory**: Create appropriate repository instances
- **Query Capabilities**: Support for semantic, vector, and temporal queries

## Architecture

### Core Components

```
habitat_evolution/
├── adaptive_core/
│   └── persistence/
│       ├── interfaces/            # Repository interfaces
│       ├── adapters/              # Concrete implementations
│       └── factory.py             # Factory for creating repositories
└── pattern_aware_rag/
    └── persistence/
        └── arangodb/              # ArangoDB-specific implementations
```

### Repository Interfaces

The persistence layer defines several key interfaces:

- **FieldStateRepositoryInterface**: For storing and retrieving field states
- **PatternRepositoryInterface**: For pattern persistence
- **RelationshipRepositoryInterface**: For relationship management
- **TopologyRepositoryInterface**: For topology constructs
- **SemanticSignatureRepositoryInterface**: For semantic signatures

## Data Operations

### Storing Data

Each repository provides a `save` method for persisting entities:

```python
# Example: Saving a pattern
from habitat_evolution.adaptive_core.persistence.factory import create_pattern_repository
from habitat_evolution.adaptive_core.models.pattern import Pattern

# Create repository
db_connection = get_db_connection()
pattern_repo = create_pattern_repository(db_connection)

# Create and save pattern
pattern = Pattern(
    id="pattern-123",
    name="Emerging Concept",
    vector=[0.1, 0.2, 0.3, 0.4],
    metadata={
        "source": "document-456",
        "confidence": 0.87
    }
)

# Save to database
pattern_id = pattern_repo.save(pattern)
print(f"Saved pattern with ID: {pattern_id}")
# Output: Saved pattern with ID: pattern-123
```

### Retrieving Data

Basic retrieval operations:

```python
# Example: Retrieving a pattern by ID
pattern = pattern_repo.find_by_id("pattern-123")

print(f"Retrieved pattern: {pattern.name}")
# Output: Retrieved pattern: Emerging Concept

print(f"Pattern vector: {pattern.vector[:3]}...")
# Output: Pattern vector: [0.1, 0.2, 0.3]...
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
from habitat_evolution.adaptive_core.assertion.pattern_assertion import PatternAssertion

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
