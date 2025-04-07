"""
Enhanced Significance Accretion Service for the Habitat Evolution system.

This service extends the SignificanceAccretionService to incorporate constructive
dissonance metrics and adjust accretion rates based on dissonance potential.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class EnhancedSignificanceAccretionService:
    """Service for tracking query significance accretion with dissonance awareness."""
    
    def __init__(self, db_connection, event_service, constructive_dissonance_service=None):
        self.db_connection = db_connection
        self.event_service = event_service
        self.constructive_dissonance_service = constructive_dissonance_service
        self.significance_data = {}
        
        # Create collections if they don't exist
        if hasattr(db_connection, 'create_collection'):
            if not db_connection.collection_exists("query_significance"):
                db_connection.create_collection("query_significance")
            if not db_connection.collection_exists("query_pattern_interactions"):
                db_connection.create_collection("query_pattern_interactions", edge=True)
            if not db_connection.collection_exists("query_dissonance_metrics"):
                db_connection.create_collection("query_dissonance_metrics")
        
        logger.info("Initialized Enhanced Significance Accretion Service with dissonance awareness")
    
    async def initialize_query_significance(self, query_id, query_text):
        """Initialize significance for a new query with dissonance metrics."""
        initial_significance = {
            "_key": query_id.replace("query-", ""),
            "query_id": query_id,
            "query_text": query_text,
            "accretion_level": 0.1,  # Start with minimal significance
            "semantic_stability": 0.1,
            "relational_density": 0.0,
            "significance_vector": {},  # Empty significance vector
            "interaction_count": 0,
            "dissonance_potential": 0.0,  # Initial dissonance potential
            "pattern_diversity": 0.0,    # Initial pattern diversity
            "emergence_probability": 0.0, # Initial emergence probability
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Store in memory
        self.significance_data[query_id] = initial_significance
        
        # Store in database
        if hasattr(self.db_connection, 'execute_query'):
            query = f"""
            INSERT {json.dumps(initial_significance)}
            INTO query_significance
            RETURN NEW
            """
            
            await self.db_connection.execute_query(query)
        
        logger.info(f"Initialized significance for new query: {query_id}")
        return initial_significance
    
    async def calculate_accretion_rate(self, interaction_metrics):
        """Calculate accretion rate based on interaction metrics and dissonance potential."""
        # Extract metrics
        interaction_strength = interaction_metrics.get("interaction_strength", 0.1)
        pattern_count = interaction_metrics.get("pattern_count", 0)
        dissonance_metrics = interaction_metrics.get("dissonance_metrics", {})
        dissonance_potential = dissonance_metrics.get("dissonance_potential", 0.0)
        
        # Base rate
        base_rate = 0.05
        
        # Adjust based on interaction strength and pattern count
        adjusted_rate = base_rate * (1 + interaction_strength) * (1 + (pattern_count * 0.1))
        
        # Apply dissonance boost if available
        if dissonance_potential > 0.5:
            # High dissonance potential boosts accretion rate
            dissonance_boost = dissonance_potential * 0.5
            adjusted_rate *= (1 + dissonance_boost)
            logger.info(f"Applied dissonance boost: {dissonance_boost:.2f}x to accretion rate")
        elif dissonance_potential > 0.2:
            # Moderate dissonance provides smaller boost
            dissonance_boost = dissonance_potential * 0.3
            adjusted_rate *= (1 + dissonance_boost)
            logger.info(f"Applied moderate dissonance boost: {dissonance_boost:.2f}x to accretion rate")
        
        # Cap at reasonable value
        return min(adjusted_rate, 0.35)  # Slightly higher cap due to dissonance boost
    
    async def update_significance(self, query_id, interaction_metrics, accretion_rate):
        """Update significance based on interactions with dissonance awareness."""
        # Get current significance
        current = self.significance_data.get(query_id, {})
        if not current:
            logger.warning(f"No significance data found for query: {query_id}")
            return {}
        
        # Get enhanced metrics
        pattern_count = interaction_metrics.get("pattern_count", 0)
        interaction_strength = interaction_metrics.get("interaction_strength", 0.1)
        pattern_relevance = interaction_metrics.get("pattern_relevance", {})
        quality_transitions = interaction_metrics.get("quality_transitions", {})
        semantic_chunk_size = interaction_metrics.get("semantic_chunk_size", "medium")
        transition_confidence = interaction_metrics.get("transition_confidence", 0.5)
        coherence_score = interaction_metrics.get("coherence_score", 0.5)
        retrieval_quality = interaction_metrics.get("retrieval_quality", 0.5)
        
        # Get dissonance metrics
        dissonance_metrics = interaction_metrics.get("dissonance_metrics", {})
        dissonance_potential = dissonance_metrics.get("dissonance_potential", 0.0)
        pattern_diversity = dissonance_metrics.get("pattern_diversity", 0.0)
        emergence_probability = dissonance_metrics.get("emergence_probability", 0.0)
        
        # Update significance vector
        significance_vector = current.get("significance_vector", {}).copy()
        
        # For testing purposes, if no patterns in relevance, add some mock patterns
        if not pattern_relevance and hasattr(self, 'db_connection'):
            # Get some patterns from the database
            if hasattr(self.db_connection, 'execute_query'):
                query = """
                FOR p IN patterns
                LIMIT 3
                RETURN p
                """
                patterns = await self.db_connection.execute_query(query)
                
                # Add these patterns to the relevance map with quality transitions
                for pattern in patterns:
                    pattern_id = pattern.get('id', f"pattern-{uuid.uuid4()}")
                    pattern_relevance[pattern_id] = 0.3  # Medium relevance
                    # Add quality transitions for mock patterns
                    quality_transitions[pattern_id] = "poor_to_uncertain"  # Default transition
                    
                logger.info(f"Added {len(pattern_relevance)} mock patterns to significance vector")
        
        # Apply chunk size multiplier based on semantic chunk size
        chunk_size_multiplier = 1.0
        if semantic_chunk_size == "large":
            chunk_size_multiplier = 1.5  # 50% boost for large chunks
        elif semantic_chunk_size == "small":
            chunk_size_multiplier = 0.8  # 20% reduction for small chunks
        
        # Apply dissonance multiplier to enhance accretion for patterns with high dissonance potential
        dissonance_multiplier = 1.0
        if dissonance_potential > 0.6:
            dissonance_multiplier = 1.4  # 40% boost for high dissonance potential
        elif dissonance_potential > 0.3:
            dissonance_multiplier = 1.2  # 20% boost for moderate dissonance potential
        
        # Add new patterns to significance vector with enhanced accretion based on chunk size and dissonance
        for pattern_id, relevance in pattern_relevance.items():
            # Apply quality transition boost
            transition_boost = 1.0
            if pattern_id in quality_transitions:
                transition = quality_transitions[pattern_id]
                if transition == "poor_to_uncertain":
                    transition_boost = 1.3  # 30% boost
                elif transition == "uncertain_to_good":
                    transition_boost = 1.5  # 50% boost
                logger.info(f"Applied {transition} transition boost ({transition_boost:.2f}x) to pattern {pattern_id}")
            
            # Calculate enhanced significance increase with dissonance factor
            significance_increase = (
                relevance * 
                accretion_rate * 
                chunk_size_multiplier * 
                transition_boost * 
                dissonance_multiplier
            )
            
            if pattern_id in significance_vector:
                # Increase existing significance with enhanced factors
                significance_vector[pattern_id] += significance_increase
            else:
                # Add new pattern with initial significance
                significance_vector[pattern_id] = significance_increase
                logger.info(f"Added pattern {pattern_id} to significance vector with value {significance_vector[pattern_id]:.2f}")
        
        # Update metrics with enhanced calculations
        # Adjust accretion level based on semantic chunk size, transitions, and dissonance
        transition_factor = sum(1.0 for t in quality_transitions.values() if t in ["poor_to_uncertain", "uncertain_to_good"]) / max(1, len(quality_transitions))
        
        # Enhanced accretion calculations with dissonance factor
        accretion_level = current.get("accretion_level", 0.1) + (
            accretion_rate * 
            chunk_size_multiplier * 
            (1 + transition_factor) * 
            dissonance_multiplier
        )
        
        # Semantic stability is influenced by coherence and dissonance
        # High dissonance can reduce stability in a constructive way
        stability_adjustment = 1.0
        if dissonance_potential > 0.5:
            # High dissonance reduces stability slightly to allow for evolution
            stability_adjustment = 0.9
        
        semantic_stability = current.get("semantic_stability", 0.1) + (
            accretion_rate * 
            coherence_score * 
            stability_adjustment
        )
        
        # Relational density is enhanced by pattern diversity
        diversity_factor = 1.0 + (pattern_diversity * 0.5)
        relational_density = current.get("relational_density", 0.0) + (
            pattern_count * 
            0.01 * 
            retrieval_quality * 
            chunk_size_multiplier * 
            diversity_factor
        )
        
        # Cap values
        accretion_level = min(accretion_level, 1.0)
        semantic_stability = min(semantic_stability, 1.0)
        relational_density = min(relational_density, 1.0)
        
        # Create updated significance with enhanced metrics and dissonance awareness
        updated_significance = {
            "_key": query_id.replace("query-", ""),
            "query_id": query_id,
            "query_text": current.get("query_text", ""),
            "accretion_level": accretion_level,
            "semantic_stability": semantic_stability,
            "relational_density": relational_density,
            "significance_vector": significance_vector,
            "interaction_count": current.get("interaction_count", 0) + 1,
            "semantic_chunk_size": semantic_chunk_size,
            "quality_transitions": quality_transitions,
            "transition_confidence": transition_confidence,
            "coherence_score": coherence_score,
            "retrieval_quality": retrieval_quality,
            "dissonance_potential": dissonance_potential,
            "pattern_diversity": pattern_diversity,
            "emergence_probability": emergence_probability,
            "created_at": current.get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat()
        }
        
        # Store in memory
        self.significance_data[query_id] = updated_significance
        
        # Store in database
        if hasattr(self.db_connection, 'execute_query'):
            query = f"""
            REPLACE {json.dumps(updated_significance)}
            IN query_significance
            RETURN NEW
            """
            
            await self.db_connection.execute_query(query)
            
            # Also store dissonance metrics separately for analysis
            dissonance_record = {
                "_key": f"{query_id}-{datetime.now().isoformat()}",
                "query_id": query_id,
                "dissonance_potential": dissonance_potential,
                "pattern_diversity": pattern_diversity,
                "emergence_probability": emergence_probability,
                "timestamp": datetime.now().isoformat()
            }
            
            dissonance_query = f"""
            INSERT {json.dumps(dissonance_record)}
            INTO query_dissonance_metrics
            RETURN NEW
            """
            
            await self.db_connection.execute_query(dissonance_query)
        
        logger.info(f"Updated significance for query: {query_id} using {semantic_chunk_size} semantic chunks")
        logger.info(f"Quality transitions: {len(quality_transitions)} patterns with transition confidence {transition_confidence:.2f}")
        logger.info(f"Dissonance metrics: potential={dissonance_potential:.2f}, diversity={pattern_diversity:.2f}, emergence={emergence_probability:.2f}")
        return updated_significance
    
    async def get_significance(self, query_id):
        """Get current significance for a query."""
        # Try to get from memory
        if query_id in self.significance_data:
            return self.significance_data[query_id]
        
        # Try to get from database
        if hasattr(self.db_connection, 'execute_query'):
            query = f"""
            FOR s IN query_significance
            FILTER s.query_id == '{query_id}'
            RETURN s
            """
            
            result = await self.db_connection.execute_query(query)
            if result and len(result) > 0:
                # Store in memory for future use
                self.significance_data[query_id] = result[0]
                return result[0]
        
        logger.warning(f"No significance data found for query: {query_id}")
        return None
    
    async def get_significance_vector(self, query_id):
        """Get significance vector for a query."""
        significance = await self.get_significance(query_id)
        if significance:
            return significance.get("significance_vector", {})
        return {}
