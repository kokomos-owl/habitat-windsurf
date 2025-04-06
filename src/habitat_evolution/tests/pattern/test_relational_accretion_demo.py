"""
Demonstration of the Relational Accretion Model for Queries as Actants

This script demonstrates the key concepts of the relational accretion model
without requiring external dependencies. It shows how queries gradually
accrete significance through interactions with patterns and eventually
generate new patterns once they reach sufficient significance.
"""

import os
import sys
import logging
import time
from datetime import datetime
import uuid
import json
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SignificanceAccretionDemo:
    """Demonstration of the significance accretion process for queries."""
    
    def __init__(self):
        """Initialize the demo."""
        # In-memory storage for query significance
        self.query_significance = {}
        
        # In-memory storage for patterns
        self.patterns = {}
        
        # In-memory storage for query-pattern interactions
        self.interactions = []
        
        # In-memory storage for generated patterns
        self.generated_patterns = []
        
        # Initialize with some example patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize some example patterns for demonstration."""
        example_patterns = [
            {
                "id": f"pattern-{uuid.uuid4()}",
                "base_concept": "sea_level_rise",
                "confidence": 0.9,
                "coherence": 0.85,
                "properties": {
                    "location": "coastal",
                    "timeframe": "2050",
                    "impact_level": "high"
                }
            },
            {
                "id": f"pattern-{uuid.uuid4()}",
                "base_concept": "coastal_flooding",
                "confidence": 0.8,
                "coherence": 0.75,
                "properties": {
                    "location": "coastal",
                    "timeframe": "2030",
                    "impact_level": "medium"
                }
            },
            {
                "id": f"pattern-{uuid.uuid4()}",
                "base_concept": "wildfire_risk",
                "confidence": 0.75,
                "coherence": 0.7,
                "properties": {
                    "location": "inland",
                    "timeframe": "2040",
                    "impact_level": "high"
                }
            },
            {
                "id": f"pattern-{uuid.uuid4()}",
                "base_concept": "adaptation_strategies",
                "confidence": 0.85,
                "coherence": 0.8,
                "properties": {
                    "location": "all",
                    "timeframe": "2030-2050",
                    "impact_level": "variable"
                }
            }
        ]
        
        for pattern in example_patterns:
            self.patterns[pattern["id"]] = pattern
            
        logger.info(f"Initialized {len(example_patterns)} example patterns")
    
    def initialize_query_significance(self, query_id: str, query_text: str) -> Dict[str, Any]:
        """
        Initialize significance for a new query.
        
        Args:
            query_id: The ID of the query
            query_text: The text of the query
            
        Returns:
            The initial significance vector
        """
        # Create initial significance vector with minimal structure
        initial_significance = {
            "query_id": query_id,
            "query_text": query_text,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "accretion_level": 0.1,  # Start with minimal accretion
            "interaction_count": 0,
            "significance_vector": {},  # Empty significance vector to start
            "relational_density": 0.0,  # Start with no relational density
            "semantic_stability": 0.1,  # Start with minimal stability
            "emergence_potential": 0.5  # Moderate potential for emergence
        }
        
        # Store in memory
        self.query_significance[query_id] = initial_significance
        
        logger.info(f"Initialized significance for query: {query_id}")
        return initial_significance
    
    def observe_pattern_interaction(
        self,
        query_id: str,
        pattern_id: str,
        interaction_type: str,
        interaction_strength: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Observe an interaction between a query and a pattern.
        
        Args:
            query_id: The ID of the query
            pattern_id: The ID of the pattern
            interaction_type: The type of interaction (e.g., "retrieval", "augmentation")
            interaction_strength: The strength of the interaction (0.0 to 1.0)
            context: Optional context for the interaction
            
        Returns:
            The interaction record
        """
        # Create interaction record
        interaction_id = str(uuid.uuid4())
        interaction = {
            "interaction_id": interaction_id,
            "query_id": query_id,
            "pattern_id": pattern_id,
            "interaction_type": interaction_type,
            "interaction_strength": interaction_strength,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        # Store in memory
        self.interactions.append(interaction)
        
        logger.info(f"Recorded interaction between query {query_id} and pattern {pattern_id}")
        return interaction
    
    def calculate_accretion_rate(
        self,
        interaction_metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate the accretion rate based on interaction metrics.
        
        Args:
            interaction_metrics: Metrics from the interaction
            
        Returns:
            The accretion rate (0.0 to 1.0)
        """
        # Extract metrics
        coherence_score = interaction_metrics.get("coherence_score", 0.5)
        retrieval_quality = interaction_metrics.get("retrieval_quality", 0.5)
        pattern_count = len(interaction_metrics.get("pattern_relevance", {}))
        
        # Calculate base rate
        base_rate = 0.1
        
        # Adjust based on coherence
        coherence_factor = coherence_score * 0.5
        
        # Adjust based on retrieval quality
        retrieval_factor = retrieval_quality * 0.3
        
        # Adjust based on pattern count (more patterns = slower accretion)
        pattern_factor = 0.2 / (1 + pattern_count * 0.1)
        
        # Calculate final rate
        accretion_rate = base_rate + coherence_factor + retrieval_factor + pattern_factor
        
        # Ensure within bounds
        return max(0.01, min(0.5, accretion_rate))
    
    def update_significance(
        self,
        query_id: str,
        interaction_metrics: Dict[str, Any],
        accretion_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Update the significance of a query based on interaction metrics.
        
        Args:
            query_id: The ID of the query
            interaction_metrics: Metrics from the interaction
            accretion_rate: Rate at which significance accretes (0.0 to 1.0)
            
        Returns:
            The updated significance vector
        """
        # Get current significance
        current_significance = self.query_significance.get(query_id, {})
        
        if not current_significance:
            logger.warning(f"No significance found for query: {query_id}")
            return {}
            
        # Calculate new significance based on interaction metrics
        new_significance = self._calculate_new_significance(
            current_significance,
            interaction_metrics,
            accretion_rate
        )
        
        # Update in memory
        self.query_significance[query_id] = new_significance
        
        logger.info(f"Updated significance for query: {query_id}")
        return new_significance
    
    def _calculate_new_significance(
        self,
        current_significance: Dict[str, Any],
        interaction_metrics: Dict[str, Any],
        accretion_rate: float
    ) -> Dict[str, Any]:
        """
        Calculate new significance based on interaction metrics.
        
        Args:
            current_significance: Current significance vector
            interaction_metrics: Metrics from the interaction
            accretion_rate: Rate at which significance accretes
            
        Returns:
            The new significance vector
        """
        # Extract current values
        current_accretion = current_significance["accretion_level"]
        current_interaction_count = current_significance["interaction_count"]
        current_vector = current_significance.get("significance_vector", {})
        current_density = current_significance["relational_density"]
        current_stability = current_significance["semantic_stability"]
        
        # Extract interaction metrics
        pattern_relevance = interaction_metrics.get("pattern_relevance", {})
        coherence_score = interaction_metrics.get("coherence_score", 0.5)
        retrieval_quality = interaction_metrics.get("retrieval_quality", 0.5)
        
        # Update significance vector by merging with pattern relevance
        new_vector = current_vector.copy()
        for pattern_id, relevance in pattern_relevance.items():
            if pattern_id in new_vector:
                # Weighted average with existing relevance
                new_vector[pattern_id] = (
                    new_vector[pattern_id] * (1 - accretion_rate) +
                    relevance * accretion_rate
                )
            else:
                # New pattern relationship
                new_vector[pattern_id] = relevance * accretion_rate
        
        # Calculate new accretion level
        # Accretion grows with each interaction but plateaus over time
        new_accretion = current_accretion + (
            (1.0 - current_accretion) * accretion_rate * coherence_score
        )
        
        # Calculate new relational density
        # Density increases with number of pattern relationships
        pattern_count = len(new_vector)
        max_density = 0.9  # Maximum possible density
        new_density = min(
            max_density,
            pattern_count / (pattern_count + 10)  # Simple logistic function
        )
        
        # Calculate new semantic stability
        # Stability increases with coherence and accretion
        new_stability = current_stability + (
            (coherence_score - current_stability) * accretion_rate
        )
        
        # Calculate new emergence potential
        # Potential decreases as stability increases
        new_potential = 1.0 - (new_stability * 0.5 + new_density * 0.5)
        
        # Create new significance
        new_significance = current_significance.copy()
        new_significance.update({
            "accretion_level": new_accretion,
            "interaction_count": current_interaction_count + 1,
            "significance_vector": new_vector,
            "relational_density": new_density,
            "semantic_stability": new_stability,
            "emergence_potential": new_potential,
            "last_updated": datetime.now().isoformat()
        })
        
        return new_significance
    
    def get_query_significance(self, query_id: str) -> Dict[str, Any]:
        """
        Get the current significance of a query.
        
        Args:
            query_id: The ID of the query
            
        Returns:
            The significance vector
        """
        significance = self.query_significance.get(query_id, {})
        
        if not significance:
            logger.warning(f"No significance found for query: {query_id}")
            return {}
            
        return significance
    
    def create_pattern_from_significance(
        self,
        query: str,
        significance: Dict[str, Any]
    ) -> str:
        """
        Create a pattern from query significance.
        
        Args:
            query: The query string
            significance: The significance vector
            
        Returns:
            The ID of the created pattern
        """
        # Extract significance metrics
        accretion_level = significance.get("accretion_level", 0.1)
        semantic_stability = significance.get("semantic_stability", 0.1)
        relational_density = significance.get("relational_density", 0.0)
        significance_vector = significance.get("significance_vector", {})
        
        # Only create patterns for queries with sufficient accretion
        if accretion_level < 0.3:
            logger.info(f"Query has insufficient accretion ({accretion_level:.2f}), skipping pattern creation")
            return ""
        
        # Derive base concept from significance
        base_concept = self._derive_base_concept_from_significance(significance)
        
        # Create pattern data
        pattern_id = f"query-pattern-{uuid.uuid4()}"
        pattern_data = {
            "id": pattern_id,
            "base_concept": base_concept,
            "confidence": accretion_level,
            "coherence": semantic_stability,
            "signal_strength": relational_density,
            "phase_stability": semantic_stability,
            "uncertainty": 1.0 - semantic_stability,
            "properties": {
                "query_origin": True,
                "accretion_level": accretion_level,
                "related_patterns": list(significance_vector.keys())
            },
            "metadata": {
                "creation_source": "query_significance",
                "creation_timestamp": datetime.now().isoformat()
            }
        }
        
        # Store pattern
        self.patterns[pattern_id] = pattern_data
        self.generated_patterns.append(pattern_data)
        
        logger.info(f"Created pattern from query significance: {pattern_id}")
        return pattern_id
    
    def _derive_base_concept_from_significance(self, significance: Dict[str, Any]) -> str:
        """
        Derive a base concept from query significance.
        
        Args:
            significance: The significance vector
            
        Returns:
            The base concept for the pattern
        """
        # Get the most significant patterns
        significance_vector = significance.get("significance_vector", {})
        if not significance_vector:
            return "unknown_concept"
            
        # Sort patterns by significance
        sorted_patterns = sorted(
            significance_vector.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get the top patterns
        top_patterns = sorted_patterns[:3]
        
        # Get the base concepts of the top patterns
        base_concepts = []
        for pattern_id, _ in top_patterns:
            pattern = self.patterns.get(pattern_id, {})
            if pattern:
                base_concepts.append(pattern.get("base_concept", "unknown"))
        
        # If no base concepts found, use a default
        if not base_concepts:
            return "derived_concept"
            
        # If only one base concept, use it
        if len(base_concepts) == 1:
            return f"derived_{base_concepts[0]}"
            
        # If multiple base concepts, combine them
        return f"derived_{'_'.join(base_concepts)}"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the relational accretion model.
        
        Args:
            query: The query string
            
        Returns:
            Processing result
        """
        # Generate query ID
        query_id = str(uuid.uuid4())
        
        logger.info(f"Processing query: {query}")
        
        # Initialize query significance
        significance = self.initialize_query_significance(query_id, query)
        
        # Simulate retrieval
        retrieval_results = self._simulate_retrieval(query)
        
        # Simulate interaction metrics
        interaction_metrics = self._simulate_interaction_metrics(query, retrieval_results)
        
        # Calculate accretion rate
        accretion_rate = self.calculate_accretion_rate(interaction_metrics)
        
        # For each pattern in the retrieval results, record an interaction
        for result in retrieval_results:
            for pattern in result.get("patterns", []):
                pattern_id = pattern.get("id")
                if not pattern_id:
                    continue
                    
                self.observe_pattern_interaction(
                    query_id=query_id,
                    pattern_id=pattern_id,
                    interaction_type="retrieval",
                    interaction_strength=pattern.get("relevance", 0.5)
                )
        
        # Update query significance
        updated_significance = self.update_significance(
            query_id=query_id,
            interaction_metrics=interaction_metrics,
            accretion_rate=accretion_rate
        )
        
        # Create pattern from significance if accretion is sufficient
        pattern_id = self.create_pattern_from_significance(query, updated_significance)
        
        # Create result
        result = {
            "query_id": query_id,
            "significance_level": updated_significance.get("accretion_level", 0.1),
            "semantic_stability": updated_significance.get("semantic_stability", 0.1),
            "relational_density": updated_significance.get("relational_density", 0.0),
            "emergence_potential": updated_significance.get("emergence_potential", 0.5),
            "pattern_id": pattern_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Query processed with accretion: {query_id}")
        return result
    
    def _simulate_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """
        Simulate retrieval for a query.
        
        Args:
            query: The query string
            
        Returns:
            Simulated retrieval results
        """
        # Determine which patterns are relevant based on query content
        relevant_patterns = []
        
        if "sea level" in query.lower():
            relevant_patterns.extend([p for p in self.patterns.values() 
                                     if p["base_concept"] in ["sea_level_rise", "coastal_flooding"]])
        
        if "coastal" in query.lower() or "coast" in query.lower():
            relevant_patterns.extend([p for p in self.patterns.values() 
                                     if p["base_concept"] in ["coastal_flooding", "sea_level_rise"]])
        
        if "wildfire" in query.lower() or "fire" in query.lower():
            relevant_patterns.extend([p for p in self.patterns.values() 
                                     if p["base_concept"] == "wildfire_risk"])
        
        if "adaptation" in query.lower() or "strategy" in query.lower():
            relevant_patterns.extend([p for p in self.patterns.values() 
                                     if p["base_concept"] == "adaptation_strategies"])
        
        # If no specific patterns matched, include all patterns with lower relevance
        if not relevant_patterns:
            relevant_patterns = list(self.patterns.values())
            
        # Create simulated results
        results = []
        
        # Group patterns into documents
        for i in range(0, len(relevant_patterns), 2):
            patterns_group = relevant_patterns[i:i+2]
            
            # Skip empty groups
            if not patterns_group:
                continue
                
            # Create document with patterns
            document = {
                "document_id": f"doc-{uuid.uuid4()}",
                "content": f"Sample content related to {', '.join([p['base_concept'] for p in patterns_group])}",
                "relevance": 0.7 + (0.2 * (i == 0)),  # First document is most relevant
                "coherence": 0.7 + (0.1 * (i == 0)),
                "patterns": []
            }
            
            # Add patterns to document
            for pattern in patterns_group:
                document["patterns"].append({
                    "id": pattern["id"],
                    "base_concept": pattern["base_concept"],
                    "relevance": 0.7 + (0.2 * ("sea level" in query.lower() and pattern["base_concept"] == "sea_level_rise")),
                    "coherence": pattern["coherence"]
                })
                
            results.append(document)
            
        return results
    
    def _simulate_interaction_metrics(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simulate interaction metrics for a query.
        
        Args:
            query: The query string
            retrieval_results: Retrieval results
            
        Returns:
            Simulated interaction metrics
        """
        # Extract patterns from retrieval results
        patterns = []
        for result in retrieval_results:
            patterns.extend(result.get("patterns", []))
            
        # Calculate pattern relevance
        pattern_relevance = {}
        for pattern in patterns:
            pattern_id = pattern.get("id")
            if not pattern_id:
                continue
                
            relevance = pattern.get("relevance", 0.5)
            
            # If pattern already exists, take the maximum relevance
            if pattern_id in pattern_relevance:
                pattern_relevance[pattern_id] = max(pattern_relevance[pattern_id], relevance)
            else:
                pattern_relevance[pattern_id] = relevance
                
        # Calculate coherence score
        coherence_score = sum(r.get("coherence", 0.5) for r in retrieval_results) / max(1, len(retrieval_results))
        
        # Calculate retrieval quality
        retrieval_quality = sum(r.get("relevance", 0.5) for r in retrieval_results) / max(1, len(retrieval_results))
        
        # Create interaction metrics
        interaction_metrics = {
            "pattern_relevance": pattern_relevance,
            "coherence_score": coherence_score,
            "retrieval_quality": retrieval_quality
        }
        
        return interaction_metrics


def run_demo():
    """Run the demonstration."""
    # Create demo
    demo = SignificanceAccretionDemo()
    
    # Define a sequence of related queries
    queries = [
        "What are the projected sea level rise impacts for Martha's Vineyard?",
        "How will sea level rise affect coastal properties on Martha's Vineyard?",
        "What adaptation strategies are recommended for sea level rise on Martha's Vineyard?",
        "How does Martha's Vineyard's sea level rise compare to other coastal areas?"
    ]
    
    # Track significance over time
    significance_history = []
    
    # Use a single query ID to demonstrate accretion over time
    query_id = str(uuid.uuid4())
    query_text = "Sea level rise on Martha's Vineyard"
    
    # Initialize query significance
    initial_significance = demo.initialize_query_significance(query_id, query_text)
    significance_history.append(initial_significance)
    
    logger.info(f"\n=== Initial Query Significance ===")
    logger.info(f"Query: {query_text}")
    logger.info(f"Accretion Level: {initial_significance['accretion_level']:.2f}")
    logger.info(f"Semantic Stability: {initial_significance['semantic_stability']:.2f}")
    logger.info(f"Relational Density: {initial_significance['relational_density']:.2f}")
    logger.info(f"Emergence Potential: {initial_significance['emergence_potential']:.2f}")
    
    # Process each query as a refinement of the same topic
    for i, query in enumerate(queries):
        logger.info(f"\n=== Processing Query {i+1}/{len(queries)} ===")
        logger.info(f"Query: {query}")
        
        # Simulate retrieval
        retrieval_results = demo._simulate_retrieval(query)
        
        # Simulate interaction metrics
        interaction_metrics = demo._simulate_interaction_metrics(query, retrieval_results)
        
        # Calculate accretion rate
        accretion_rate = demo.calculate_accretion_rate(interaction_metrics)
        
        # For each pattern in the retrieval results, record an interaction
        for result in retrieval_results:
            for pattern in result.get("patterns", []):
                pattern_id = pattern.get("id")
                if not pattern_id:
                    continue
                    
                demo.observe_pattern_interaction(
                    query_id=query_id,
                    pattern_id=pattern_id,
                    interaction_type="retrieval",
                    interaction_strength=pattern.get("relevance", 0.5)
                )
        
        # Update query significance
        updated_significance = demo.update_significance(
            query_id=query_id,
            interaction_metrics=interaction_metrics,
            accretion_rate=accretion_rate
        )
        
        # Add to history
        significance_history.append(updated_significance.copy())
        
        # Log significance metrics
        logger.info(f"Accretion Level: {updated_significance['accretion_level']:.2f}")
        logger.info(f"Semantic Stability: {updated_significance['semantic_stability']:.2f}")
        logger.info(f"Relational Density: {updated_significance['relational_density']:.2f}")
        logger.info(f"Emergence Potential: {updated_significance['emergence_potential']:.2f}")
        logger.info(f"Interaction Count: {updated_significance['interaction_count']}")
        logger.info(f"Vector Size: {len(updated_significance['significance_vector'])}")
        
        # Check if significance is sufficient to generate a pattern
        if updated_significance['accretion_level'] >= 0.3:
            # Create pattern from significance
            pattern_id = demo.create_pattern_from_significance(query, updated_significance)
            if pattern_id:
                logger.info(f"Generated Pattern: {pattern_id}")
        else:
            logger.info("No pattern generated yet (insufficient accretion)")
            
        # Add a small delay for readability
        time.sleep(0.5)
    
    # Log final results
    logger.info("\n=== Final Results ===")
    logger.info(f"Processed {len(queries)} queries")
    logger.info(f"Initial Accretion Level: {significance_history[0]['accretion_level']:.2f}")
    logger.info(f"Final Accretion Level: {significance_history[-1]['accretion_level']:.2f}")
    logger.info(f"Accretion Growth: {significance_history[-1]['accretion_level'] - significance_history[0]['accretion_level']:.2f}")
    logger.info(f"Generated Patterns: {len(demo.generated_patterns)}")
    
    # Log generated patterns
    if demo.generated_patterns:
        logger.info("\n=== Generated Patterns ===")
        for i, pattern in enumerate(demo.generated_patterns):
            logger.info(f"Pattern {i+1}: {pattern['base_concept']} (Confidence: {pattern['confidence']:.2f})")
            logger.info(f"  Related Patterns: {pattern['properties']['related_patterns']}")
    
    # Verify that significance accretes over time
    assert significance_history[-1]["accretion_level"] > significance_history[0]["accretion_level"], \
        "Significance did not accrete over time"
    
    # Verify that patterns are generated once significance reaches threshold
    assert len(demo.generated_patterns) > 0, "No patterns were generated from query significance"
    
    logger.info("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
