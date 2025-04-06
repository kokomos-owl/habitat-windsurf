# Relational Accretion Model for Queries as Actants

## Conceptual Shift: From Projection to Observation

The Relational Accretion Model represents a fundamental shift in how we conceptualize queries within the Habitat Evolution system. Rather than treating queries as passive entities onto which patterns are projected, we now view queries as active participants (actants) in the pattern evolution process.

### Previous Approach: Pattern Projection

In the previous model, we used Claude LLM to analyze queries and project patterns onto them:

1. Query arrives → Claude analyzes → Patterns are extracted and projected onto query
2. Query is enhanced based on these projected patterns
3. Query is used for retrieval, but doesn't evolve further

This approach had several limitations:
- Pattern projection was one-directional and imposed
- Queries didn't evolve organically through interactions
- The system couldn't learn from query-pattern interactions over time

### New Approach: Relational Accretion

In the relational accretion model:

1. Query arrives → Minimal baseline enhancement from Claude
2. Query interacts with pattern space → Relationships are observed and recorded
3. Query gradually accretes significance through these interactions
4. As significance accretes, query patterns can generate new patterns
5. The system learns from these interactions, creating a co-evolutionary cycle

This shift aligns with core Habitat Evolution principles:
- **Emergence over imposition**: Patterns emerge from interactions rather than being imposed
- **Co-evolution**: Queries and patterns evolve together through mutual interactions
- **Contextual reinforcement**: Significance accretes in context, not in isolation
- **Adaptive learning**: The system learns by observing actual interactions

## Key Components of the Implementation

### 1. SignificanceAccretionService

This service tracks how queries accrete significance through interactions with patterns:

```python
class SignificanceAccretionService:
    """Service for tracking and managing the accretion of significance for queries."""
    
    async def initialize_query_significance(self, query_id: str, query_text: str) -> Dict[str, Any]:
        """Initialize significance for a new query."""
        initial_significance = {
            "_key": query_id,
            "query_id": query_id,
            "query_text": query_text,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "accretion_level": 0.1,  # Start with minimal significance
            "interaction_count": 0,
            "significance_vector": {},  # Empty significance vector
            "relational_density": 0.0,
            "semantic_stability": 0.1,
            "emergence_potential": 0.5
        }
        
        # Store in database and return
        await self.db_connection.insert(self.collection_name, initial_significance)
        return initial_significance
    
    async def observe_pattern_interaction(self, query_id: str, pattern_id: str, 
                                         interaction_type: str, interaction_strength: float, 
                                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Record an interaction between a query and a pattern."""
        # Generate unique ID for this interaction
        interaction_id = f"interaction_{uuid.uuid4()}"
        
        # Create interaction record
        interaction = {
            "_key": interaction_id,
            "_from": f"{self.collection_name}/{query_id}",
            "_to": f"patterns/{pattern_id}",
            "interaction_id": interaction_id,
            "query_id": query_id,
            "pattern_id": pattern_id,
            "interaction_type": interaction_type,
            "interaction_strength": interaction_strength,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        # Store interaction in database
        await self.db_connection.insert(self.interactions_collection, interaction)
        
        # Update query significance based on this interaction
        await self._update_significance(query_id, pattern_id, interaction_type, interaction_strength)
        
        # Publish event about this interaction
        await self.event_service.publish("query.pattern.interaction", {
            "query_id": query_id,
            "pattern_id": pattern_id,
            "interaction_type": interaction_type,
            "interaction_strength": interaction_strength,
            "timestamp": datetime.now().isoformat()
        })
        
        return interaction
    
    async def _update_significance(self, query_id: str, pattern_id: str, 
                                  interaction_type: str, interaction_strength: float) -> None:
        """Update query significance based on an interaction with a pattern."""
        # Get current significance
        significance = await self.get_query_significance(query_id)
        if not significance:
            return
        
        # Update interaction count
        interaction_count = significance.get("interaction_count", 0) + 1
        
        # Update significance vector
        significance_vector = significance.get("significance_vector", {})
        current_significance = significance_vector.get(pattern_id, 0.0)
        
        # Calculate new significance based on interaction type and strength
        if interaction_type == "retrieval":
            # Retrieval interactions increase significance
            new_significance = current_significance + (interaction_strength * 0.1)
        elif interaction_type == "generation":
            # Generation interactions increase significance more
            new_significance = current_significance + (interaction_strength * 0.2)
        elif interaction_type == "feedback":
            # Feedback interactions can increase or decrease significance
            new_significance = current_significance + (interaction_strength * 0.15)
        else:
            # Default case
            new_significance = current_significance + (interaction_strength * 0.05)
        
        # Cap significance at 1.0
        new_significance = min(new_significance, 1.0)
        
        # Update significance vector
        significance_vector[pattern_id] = new_significance
        
        # Calculate overall accretion level (average of all significances)
        if significance_vector:
            accretion_level = sum(significance_vector.values()) / len(significance_vector)
        else:
            accretion_level = 0.1
        
        # Calculate relational density (number of relationships relative to pattern space)
        pattern_count = len(await self.db_connection.query("patterns", "RETURN COUNT(*)"))
        relational_density = len(significance_vector) / max(pattern_count, 1)
        
        # Calculate semantic stability (consistency of significance across patterns)
        if len(significance_vector) > 1:
            values = list(significance_vector.values())
            semantic_stability = 1.0 - (statistics.stdev(values) / max(statistics.mean(values), 0.01))
        else:
            semantic_stability = 0.1
        
        # Calculate emergence potential
        emergence_potential = accretion_level * relational_density * semantic_stability
        
        # Update significance record
        update = {
            "last_updated": datetime.now().isoformat(),
            "interaction_count": interaction_count,
            "significance_vector": significance_vector,
            "accretion_level": accretion_level,
            "relational_density": relational_density,
            "semantic_stability": semantic_stability,
            "emergence_potential": emergence_potential
        }
        
        # Apply update in database
        await self.db_connection.update(
            self.collection_name, 
            {"_key": query_id}, 
            update
        )
```

### 2. ClaudeBaselineService

This service provides minimal baseline enhancement for queries without projecting patterns:

```python
class ClaudeBaselineService:
    """Service for providing baseline query enhancement using Claude LLM."""
    
    async def enhance_query(self, query_id: str, query_text: str) -> Dict[str, Any]:
        """Enhance a query with minimal baseline enhancement."""
        try:
            # Prepare prompt for Claude
            prompt = f"""
            You are helping to enhance a query for a retrieval system about climate risk.
            
            QUERY: {query_text}
            
            Please provide a minimal enhancement of this query that:
            1. Clarifies any ambiguous terms
            2. Adds basic semantic dimensions
            3. Does NOT project patterns onto the query
            4. Keeps the enhancement subtle and minimal
            
            Your enhancement should be concise and focus on clarification rather than expansion.
            """
            
            # Call Claude API
            response = await self._call_claude_api(prompt)
            
            # Extract enhanced query
            enhanced_query = response.strip()
            
            # Create enhancement record
            enhancement = {
                "query_id": query_id,
                "original_query": query_text,
                "enhanced_query": enhanced_query,
                "enhancement_type": "baseline",
                "timestamp": datetime.now().isoformat()
            }
            
            # Store enhancement in database
            await self.db_connection.insert("query_enhancements", enhancement)
            
            # Publish event about this enhancement
            await self.event_service.publish("query.enhanced", {
                "query_id": query_id,
                "original_query": query_text,
                "enhanced_query": enhanced_query,
                "enhancement_type": "baseline",
                "timestamp": datetime.now().isoformat()
            })
            
            return enhancement
            
        except Exception as e:
            # Log error and return original query
            logger.error(f"Error enhancing query: {str(e)}")
            return {
                "query_id": query_id,
                "original_query": query_text,
                "enhanced_query": query_text,  # Return original query on error
                "enhancement_type": "none",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
```

### 3. AccretivePatternRAG

This class implements the complete relational accretion approach to pattern-aware RAG:

```python
class AccretivePatternRAG:
    """Pattern-Aware RAG implementation using the relational accretion model for queries."""
    
    async def query(self, query_text: str) -> Dict[str, Any]:
        """Process a query using the relational accretion model."""
        # Generate query ID
        query_id = f"query_{uuid.uuid4()}"
        
        # Initialize query significance
        await self.significance_service.initialize_query_significance(query_id, query_text)
        
        # Apply minimal baseline enhancement
        enhancement = await self.claude_baseline_service.enhance_query(query_id, query_text)
        enhanced_query = enhancement.get("enhanced_query", query_text)
        
        # Retrieve relevant patterns based on enhanced query
        relevant_patterns = await self._retrieve_relevant_patterns(enhanced_query)
        
        # Record interactions with retrieved patterns
        for pattern in relevant_patterns:
            pattern_id = pattern.get("id")
            relevance_score = pattern.get("relevance", 0.5)
            
            await self.significance_service.observe_pattern_interaction(
                query_id=query_id,
                pattern_id=pattern_id,
                interaction_type="retrieval",
                interaction_strength=relevance_score
            )
        
        # Generate response based on query and patterns
        response = await self._generate_response(enhanced_query, relevant_patterns)
        
        # Check if query has accreted enough significance to generate a pattern
        significance = await self.significance_service.get_query_significance(query_id)
        emergence_potential = significance.get("emergence_potential", 0.0)
        
        pattern_id = None
        if emergence_potential > 0.7:
            # Query has accreted enough significance to generate a pattern
            pattern_id = await self._generate_pattern_from_significance(query_id, significance)
        
        # Return result with significance metrics
        return {
            "query_id": query_id,
            "original_query": query_text,
            "enhanced_query": enhanced_query,
            "response": response,
            "pattern_id": pattern_id,
            "significance_level": significance.get("accretion_level", 0.1),
            "semantic_stability": significance.get("semantic_stability", 0.1),
            "relational_density": significance.get("relational_density", 0.0),
            "emergence_potential": emergence_potential
        }
    
    async def _generate_pattern_from_significance(self, query_id: str, significance: Dict[str, Any]) -> str:
        """Generate a new pattern based on query significance."""
        # Extract significance vector
        significance_vector = significance.get("significance_vector", {})
        
        # Find patterns with highest significance
        sorted_patterns = sorted(
            significance_vector.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if not sorted_patterns:
            return None
        
        # Get top patterns
        top_patterns = sorted_patterns[:3]
        top_pattern_ids = [p[0] for p in top_patterns]
        
        # Get pattern data
        patterns = []
        for pattern_id in top_pattern_ids:
            pattern = await self.pattern_evolution_service.get_pattern(pattern_id)
            if pattern:
                patterns.append(pattern)
        
        if not patterns:
            return None
        
        # Create new pattern from related patterns
        query_text = significance.get("query_text", "")
        
        # Generate pattern content
        content = f"Pattern derived from query: {query_text}"
        
        # Create pattern properties
        properties = {
            "query_origin": True,
            "source_query_id": query_id,
            "related_patterns": top_pattern_ids,
            "significance_level": significance.get("accretion_level", 0.1)
        }
        
        # Create pattern
        pattern_id = await self.pattern_evolution_service.create_pattern(
            content=content,
            properties=properties,
            quality="emergent",  # Start as emergent quality
            relationships=[
                {"type": "derived_from", "target_id": pid, "strength": sig}
                for pid, sig in top_patterns
            ]
        )
        
        # Publish event about pattern generation
        await self.event_service.publish("pattern.generated_from_query", {
            "pattern_id": pattern_id,
            "query_id": query_id,
            "related_patterns": top_pattern_ids,
            "timestamp": datetime.now().isoformat()
        })
        
        return pattern_id
```

## How Query Patterns Accrete Significance

The key insight of the relational accretion model is that query patterns accrete significance through interactions with the pattern space. Here's how it works:

1. **Initial State**: A query begins with minimal significance and an empty significance vector.

2. **Interaction Observation**: As the query interacts with patterns (through retrieval, generation, feedback), these interactions are observed and recorded.

3. **Significance Vector**: Each interaction updates the query's significance vector, which maps pattern IDs to significance values.

4. **Accretion Metrics**:
   - **Accretion Level**: Average significance across all patterns
   - **Relational Density**: Proportion of pattern space the query has interacted with
   - **Semantic Stability**: Consistency of significance across patterns
   - **Emergence Potential**: Combined metric indicating readiness to generate patterns

5. **Pattern Generation**: Once query patterns accrete sufficient significance (high emergence potential), they can generate new patterns derived from the most significant related patterns.

## Example Scenario: Sea Level Rise Queries

Consider a sequence of related queries about sea level rise on Martha's Vineyard:

1. "What are the sea level rise projections for Martha's Vineyard?"
2. "How will sea level rise affect coastal properties on Martha's Vineyard?"
3. "What infrastructure is at risk from sea level rise on Martha's Vineyard?"
4. "How should Martha's Vineyard adapt its coastal infrastructure to sea level rise?"
5. "What are the economic impacts of sea level rise on Martha's Vineyard property values?"

As these queries interact with patterns in the system:

- They gradually accrete significance with patterns related to sea level rise, coastal infrastructure, adaptation strategies, etc.
- The significance vector for each query grows as more interactions occur
- Semantic stability increases as the significance pattern becomes more consistent
- Eventually, the emergence potential crosses a threshold
- A new pattern is generated, derived from the most significant related patterns
- This new pattern captures the relationship between sea level rise, coastal infrastructure, and adaptation strategies

## Benefits of the Relational Accretion Model

1. **Organic Evolution**: Patterns emerge naturally from interactions rather than being imposed by an LLM.

2. **True Co-evolution**: Queries and patterns genuinely co-evolve as they interact, creating a more dynamic system.

3. **Contextual Reinforcement**: Significance accretes in context through actual use, not in isolation.

4. **Adaptive Learning**: The system learns by observing real interactions, making it more responsive to actual usage patterns.

5. **Reduced LLM Dependence**: Less reliance on LLM for pattern extraction, more emphasis on observed interactions.

## Conclusion

The Relational Accretion Model for Queries as Actants represents a significant advancement in our conceptual approach to pattern evolution in the Habitat system. By treating queries as active participants that accrete significance through interactions, we create a more organic, emergent, and adaptive system that aligns with the core principles of pattern evolution and co-evolution.
