# Neo4j Pattern Schema for Vector + Tonic-Harmonic Fields

## Overview

The Neo4j pattern schema is designed to store and query Vector + Tonic-Harmonic patterns, their relationships, and dimensional resonance characteristics. This document details the schema design and provides example Cypher queries for pattern exploration.

## Schema Design

### Node Types

1. **Pattern**
```cypher
CREATE CONSTRAINT pattern_id IF NOT EXISTS
FOR (p:Pattern) REQUIRE p.id IS UNIQUE
```

Properties:
- id: Unique identifier
- vector: Original vector representation
- coordinates: Eigenspace coordinates
- effective_dims: Number of significant dimensions
- community: Community assignment

2. **Dimension**
```cypher
CREATE CONSTRAINT dimension_id IF NOT EXISTS
FOR (d:Dimension) REQUIRE d.id IS UNIQUE
```

Properties:
- id: Dimension identifier
- eigenvalue: Associated eigenvalue
- variance_explained: Proportion of variance explained

3. **ResonancePattern**
```cypher
CREATE CONSTRAINT resonance_pattern_id IF NOT EXISTS
FOR (r:ResonancePattern) REQUIRE r.id IS UNIQUE
```

Properties:
- id: Unique identifier
- pattern_type: Type of resonance (harmonic/sequential/complementary)
- strength: Resonance strength
- primary_dimension: Primary dimension of resonance

4. **Community**
```cypher
CREATE CONSTRAINT community_id IF NOT EXISTS
FOR (c:Community) REQUIRE c.id IS UNIQUE
```

Properties:
- id: Community identifier
- cohesion: Internal cohesion metric
- size: Number of patterns

### Relationships

1. **PROJECTS_ONTO**
```cypher
(p:Pattern)-[r:PROJECTS_ONTO]->(d:Dimension)
```
Properties:
- projection_value: Strength of projection
- sign: Positive/negative projection

2. **RESONATES_WITH**
```cypher
(p1:Pattern)-[r:RESONATES_WITH]->(p2:Pattern)
```
Properties:
- strength: Resonance strength
- dimensions: Array of shared dimensions
- type: Type of resonance

3. **BELONGS_TO**
```cypher
(p:Pattern)-[r:BELONGS_TO]->(c:Community)
```
Properties:
- membership_strength: Degree of membership
- is_boundary: Boolean for boundary patterns

4. **FORMS_PATTERN**
```cypher
(p:Pattern)-[r:FORMS_PATTERN]->(rp:ResonancePattern)
```
Properties:
- role: Role in the pattern (e.g., "source", "target")
- position: Position in sequential patterns

## Example Queries

### 1. Find Patterns with Strong Dimensional Resonance

```cypher
MATCH (p1:Pattern)-[r:PROJECTS_ONTO]->(d:Dimension)
WHERE r.projection_value > 0.7
WITH d, collect(p1) as patterns
WHERE size(patterns) > 1
RETURN d.id, patterns
```

### 2. Identify Boundary Patterns

```cypher
MATCH (p:Pattern)-[r:BELONGS_TO]->(c:Community)
WHERE r.is_boundary = true
WITH p, collect(c) as communities
WHERE size(communities) > 1
RETURN p.id, [c in communities | c.id] as boundary_communities
```

### 3. Find Sequential Patterns

```cypher
MATCH (rp:ResonancePattern {pattern_type: 'sequential'})
MATCH (p:Pattern)-[r:FORMS_PATTERN]->(rp)
RETURN rp.id, collect({pattern: p.id, position: r.position}) as sequence
ORDER BY r.position
```

### 4. Analyze Community Structure

```cypher
MATCH (c:Community)<-[r:BELONGS_TO]-(p:Pattern)
WITH c, count(p) as pattern_count,
     count(CASE WHEN r.is_boundary THEN 1 END) as boundary_count
RETURN c.id,
       pattern_count,
       boundary_count,
       c.cohesion as internal_cohesion
ORDER BY pattern_count DESC
```

## Pattern Exploration Functions

### 1. Find Related Patterns

```python
def find_related_patterns(pattern_id: str, min_strength: float = 0.5):
    query = """
    MATCH (p1:Pattern {id: $pattern_id})-[r:RESONATES_WITH]->(p2:Pattern)
    WHERE r.strength >= $min_strength
    RETURN p2.id, r.type, r.strength, r.dimensions
    ORDER BY r.strength DESC
    """
    return graph.run(query, pattern_id=pattern_id, min_strength=min_strength)
```

### 2. Analyze Pattern Evolution

```python
def analyze_pattern_evolution(pattern_id: str, time_window: int):
    query = """
    MATCH (p1:Pattern {id: $pattern_id})-[r:EVOLVES_TO*1..]->(p2:Pattern)
    WHERE p2.timestamp - p1.timestamp <= $time_window
    RETURN p1.id, [r in relationships(path) | {
        dimension_shifts: r.dimension_shifts,
        strength_change: r.strength_change
    }], p2.id
    """
    return graph.run(query, pattern_id=pattern_id, time_window=time_window)
```

## Performance Optimization

1. **Indexing Strategy**
```cypher
CREATE INDEX pattern_community IF NOT EXISTS
FOR (p:Pattern)
ON (p.community)
```

2. **Batch Processing**
```python
def batch_import_patterns(patterns: List[Dict]):
    query = """
    UNWIND $patterns as pattern
    MERGE (p:Pattern {id: pattern.id})
    SET p += pattern.properties
    """
    return graph.run(query, patterns=patterns)
```

## Usage Examples

### 1. Pattern Import
```python
# Import patterns with eigenspace coordinates
patterns = [
    {
        "id": "pattern_1",
        "properties": {
            "vector": [0.1, 0.2, 0.3],
            "coordinates": [0.5, -0.3, 0.1],
            "community": 1
        }
    },
    # ... more patterns
]
batch_import_patterns(patterns)
```

### 2. Resonance Analysis
```python
# Analyze dimensional resonance
query = """
MATCH (p:Pattern)-[r:PROJECTS_ONTO]->(d:Dimension)
WHERE d.eigenvalue >= 0.1
WITH d, collect(p) as patterns
WHERE size(patterns) > 1
RETURN d.id, 
       d.eigenvalue,
       [p in patterns | p.id] as resonating_patterns
ORDER BY d.eigenvalue DESC
"""
results = graph.run(query)
```

## Future Extensions

1. **Temporal Pattern Evolution**
- Track pattern changes over time
- Analyze evolutionary trajectories
- Detect pattern emergence/decay

2. **Advanced Resonance Types**
- Multi-dimensional resonance patterns
- Cross-community resonance
- Harmonic series detection

3. **Pattern Metrics**
- Centrality measures
- Community influence scores
- Pattern stability metrics
