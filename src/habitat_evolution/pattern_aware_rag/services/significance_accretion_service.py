"""
Significance Accretion Service for Habitat Evolution

This service handles the accretion of relational significance for queries as they
interact with the pattern space. Instead of projecting patterns onto queries,
this service observes how queries interact with existing patterns and allows
significance to emerge naturally through accretion.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json

from ...infrastructure.services.event_service import EventService
from ...infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

logger = logging.getLogger(__name__)

class SignificanceAccretionService:
    """
    Service that handles the accretion of relational significance for queries
    as they interact with the pattern space.
    """
    
    def __init__(
        self,
        db_connection: ArangoDBConnection,
        event_service: Optional[EventService] = None
    ):
        """
        Initialize the significance accretion service.
        
        Args:
            db_connection: Connection to the ArangoDB database
            event_service: Event service for publishing events
        """
        self.db_connection = db_connection
        self.event_service = event_service
        self.collection_name = "query_significance"
        self._ensure_collections_exist()
        
    def _ensure_collections_exist(self) -> None:
        """Ensure that the necessary collections exist in the database."""
        if not self.db_connection.collection_exists(self.collection_name):
            self.db_connection.create_collection(self.collection_name)
            
        if not self.db_connection.collection_exists("query_pattern_interactions"):
            self.db_connection.create_collection("query_pattern_interactions", edge=True)
    
    async def initialize_query_significance(self, query_id: str, query_text: str) -> Dict[str, Any]:
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
            "_key": query_id,
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
        
        # Store in database
        query = f"""
        INSERT {json.dumps(initial_significance)}
        INTO {self.collection_name}
        RETURN NEW
        """
        
        result = await self.db_connection.execute_query(query)
        
        # Publish event
        if self.event_service:
            self.event_service.publish(
                "query.significance.initialized",
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        logger.info(f"Initialized significance for query: {query_id}")
        return initial_significance
    
    async def observe_pattern_interaction(
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
        
        # Store in database
        query = f"""
        INSERT {json.dumps(interaction)}
        INTO query_pattern_interactions
        RETURN NEW
        """
        
        result = await self.db_connection.execute_query(query)
        
        # Publish event
        if self.event_service:
            self.event_service.publish(
                "query.pattern.interaction",
                {
                    "interaction_id": interaction_id,
                    "query_id": query_id,
                    "pattern_id": pattern_id,
                    "interaction_type": interaction_type,
                    "interaction_strength": interaction_strength,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        logger.info(f"Recorded interaction between query {query_id} and pattern {pattern_id}")
        return interaction
    
    async def update_significance(
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
        query = f"""
        FOR doc IN {self.collection_name}
        FILTER doc.query_id == @query_id
        RETURN doc
        """
        
        result = await self.db_connection.execute_query(query, bind_vars={"query_id": query_id})
        
        if not result or len(result) == 0:
            logger.warning(f"No significance found for query: {query_id}")
            return {}
            
        current_significance = result[0]
        
        # Calculate new significance based on interaction metrics
        new_significance = self._calculate_new_significance(
            current_significance,
            interaction_metrics,
            accretion_rate
        )
        
        # Update in database
        update_query = f"""
        UPDATE @new_significance
        IN {self.collection_name}
        RETURN NEW
        """
        
        update_result = await self.db_connection.execute_query(
            update_query,
            bind_vars={"new_significance": new_significance}
        )
        
        # Publish event
        if self.event_service:
            self.event_service.publish(
                "query.significance.updated",
                {
                    "query_id": query_id,
                    "previous_accretion_level": current_significance["accretion_level"],
                    "new_accretion_level": new_significance["accretion_level"],
                    "timestamp": datetime.now().isoformat()
                }
            )
        
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
    
    async def get_query_significance(self, query_id: str) -> Dict[str, Any]:
        """
        Get the current significance of a query.
        
        Args:
            query_id: The ID of the query
            
        Returns:
            The significance vector
        """
        query = f"""
        FOR doc IN {self.collection_name}
        FILTER doc.query_id == @query_id
        RETURN doc
        """
        
        result = await self.db_connection.execute_query(query, bind_vars={"query_id": query_id})
        
        if not result or len(result) == 0:
            logger.warning(f"No significance found for query: {query_id}")
            return {}
            
        return result[0]
    
    async def calculate_accretion_rate(
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
    
    async def get_related_queries(
        self,
        query_id: str,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get queries related to the given query based on shared patterns.
        
        Args:
            query_id: The ID of the query
            threshold: Minimum similarity threshold
            
        Returns:
            List of related queries
        """
        query = f"""
        LET target = (
            FOR doc IN {self.collection_name}
            FILTER doc.query_id == @query_id
            RETURN doc
        )[0]
        
        LET target_vector = target.significance_vector
        
        FOR other IN {self.collection_name}
        FILTER other.query_id != @query_id
        LET similarity = LENGTH(
            FOR k IN KEYS(target_vector)
            FILTER k IN KEYS(other.significance_vector)
            RETURN k
        ) / MAX(LENGTH(KEYS(target_vector)), LENGTH(KEYS(other.significance_vector)))
        
        FILTER similarity >= @threshold
        
        SORT similarity DESC
        
        RETURN {{
            query_id: other.query_id,
            query_text: other.query_text,
            similarity: similarity,
            accretion_level: other.accretion_level
        }}
        """
        
        result = await self.db_connection.execute_query(
            query,
            bind_vars={"query_id": query_id, "threshold": threshold}
        )
        
        return result
