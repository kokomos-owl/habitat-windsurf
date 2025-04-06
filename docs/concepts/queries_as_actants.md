# Queries as Actants in Habitat Evolution

## Introduction

In traditional information retrieval systems, queries are passive requests for information. In Habitat Evolution, we take a fundamentally different approach: **queries are actants** - active participants in the semantic ecosystem that exert pressure, drive evolution, and co-evolve with patterns. This document outlines our approach to modeling queries as actants within the Habitat Evolution system.

## Core Principles

### 1. Queries as Semantic Actants

Queries are not merely information requests but active participants in the semantic field:

- **Active Participation**: Queries interact with and modify the pattern space
- **Semantic Pressure**: Queries exert pressure on existing patterns
- **Evolutionary Force**: Queries drive pattern evolution through quality states

### 2. Co-evolution Between Queries and Patterns

The relationship between queries and patterns is bidirectional and co-evolutionary:

- **Patterns shape query interpretation**: Existing patterns influence how new queries are processed
- **Queries shape pattern evolution**: Query patterns can strengthen, weaken, or transform existing patterns
- **Emergence through interaction**: New patterns can emerge from the interaction between queries and existing patterns

### 3. Field Dynamics and Pressure

Queries create and respond to pressure in the semantic field:

- **Field State Changes**: Queries alter the field state
- **Gradient Formation**: Queries create gradients in the semantic space
- **Flow Dynamics**: These gradients influence the flow of information

## Implementation Architecture

### Pattern Extraction from Queries

```
Query → Pattern Extraction → Query Patterns → Pattern Space Integration
```

We extract patterns from queries because they are actants that actively participate in the semantic field. This extraction process identifies the semantic structures within queries that will interact with the existing pattern space.

```python
# Extract patterns from query (query as actant)
query_patterns = await self.claude_service.extract_patterns_from_query(query)
```

### Pattern-Aware Retrieval

Queries interact with the pattern space through coherence-aware embeddings:

```python
# Retrieve with coherence-aware embeddings (pattern space interaction)
docs, retrieval_patterns = await self._retrieve_with_patterns(
    query,
    query_patterns,
    embedding_context
)
```

### Bidirectional Flow Implementation

The bidirectional flow between queries and patterns is implemented through:

```python
# Update pattern evolution (bidirectional flow)
pattern_id = await self._update_pattern_evolution(
    query=query,
    rag_output=result,
    window_metrics=window_metrics
)
```

### Pattern Usage Tracking

Each query interaction with patterns is tracked to drive evolution:

```python
# Track pattern usage for each pattern in the result
patterns = result.get("patterns", [])
for pattern in patterns:
    pattern_id = pattern.get("id")
    if not pattern_id:
        continue
        
    # Calculate usage metrics based on pattern's role in the query
    usage_metrics = {
        "used_in_query": True,
        "relevance_score": pattern.get("score", 0.5),
        "coherence_score": pattern.get("coherence", 0.5),
        "query_context": query_text
    }
    
    # Track pattern usage
    self._track_pattern_usage(pattern_id, usage_metrics)
    
    # Provide feedback on pattern quality
    self._provide_pattern_feedback(pattern_id, usage_metrics)
```

## Pattern Evolution Through Quality States

Queries drive patterns through quality states:

1. **Hypothetical → Emergent**: When queries provide initial evidence
2. **Emergent → Stable**: When queries consistently reinforce patterns
3. **Stable → Declining**: When queries contradict established patterns

## Coherence and Emergence

The interaction between query patterns and existing patterns creates coherence:

```python
# Let coherence emerge naturally
coherence_insight = await self.claude_service.analyze_coherence(
    pattern_context,
    result.get("response", "")
)

# Let emergence flow track patterns
await self.emergence_flow.observe_emergence(
    {"rag_patterns": pattern_context},
    {"rag_state": coherence_insight.flow_state}
)
```

## Practical Examples

### Example 1: Query Introducing New Pattern

When a user queries about "sea level rise impact on Martha's Vineyard infrastructure":

1. The query is processed to extract patterns: `sea_level_rise`, `infrastructure_vulnerability`, `martha_vineyard_location`
2. These patterns interact with existing patterns in the knowledge base
3. If `infrastructure_vulnerability` is a new pattern, it begins in the hypothetical state
4. As more queries and documents provide evidence, it may evolve to emergent

### Example 2: Query Strengthening Existing Pattern

When multiple queries ask about "coastal flooding projections":

1. The `coastal_flooding` pattern receives usage metrics
2. Its coherence score increases through consistent usage
3. The pattern may evolve from emergent to stable
4. The system's responses become more confident about this pattern

### Example 3: Query Contradicting Pattern

If new scientific data contradicts previous projections:

1. Queries about "updated sea level projections" create pressure
2. The contradiction creates low coherence scores
3. The existing pattern may evolve from stable to declining
4. A new, more accurate pattern may emerge

## Benefits of Queries as Actants

1. **Self-improving system**: The system learns from each query interaction
2. **Contextual reinforcement**: Patterns are strengthened by contextual evidence
3. **Adaptive knowledge**: The knowledge base evolves with new information
4. **Coherent responses**: Responses maintain coherence with the evolving pattern space

## Conclusion

By modeling queries as actants in the Habitat Evolution system, we create a dynamic, self-improving system where knowledge evolves through interaction. This approach moves beyond traditional RAG systems to create a living ecosystem of patterns that respond to and co-evolve with user queries.

The bidirectional flow from document processing through pattern extraction, persistence, RAG, and back to pattern evolution creates a complete functional loop that enables continuous learning and adaptation. This is the core innovation of the Habitat Evolution approach to information retrieval and knowledge management.
