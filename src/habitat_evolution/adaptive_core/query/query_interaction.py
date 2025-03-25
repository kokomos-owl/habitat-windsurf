"""
Query Interaction

Implements a simple query interaction system that treats queries as first-class actants
in the Habitat Evolution system. This module provides a template for query interactions
that can participate in semantic relationships and transformations across modalities
and AI systems.
"""

from typing import Dict, List, Any, Optional, Set, Union, Callable
import uuid
import logging
from datetime import datetime

from .query_actant import QueryActant
from ..transformation.meaning_bridges import MeaningBridge, MeaningBridgeTracker
from ..transformation.actant_journey_tracker import ActantJourney

logger = logging.getLogger(__name__)

class QueryInteraction:
    """
    A template for query interactions in the Habitat Evolution system.
    
    This class provides a framework for query interactions that treat queries as
    first-class actants, allowing them to participate in semantic relationships
    and transformations across modalities and AI systems.
    """
    
    def __init__(self):
        """Initialize the query interaction system."""
        self.queries = {}  # Store query actants by ID
        self.meaning_bridge_tracker = MeaningBridgeTracker()
        self.query_handlers = {}  # Store query handlers by modality
        
    def register_query_handler(self, modality: str, handler: Callable):
        """Register a query handler for a specific modality.
        
        Args:
            modality: The modality this handler can process (e.g., "text", "image")
            handler: A callable that can process queries in the specified modality
        """
        self.query_handlers[modality] = handler
        logger.info(f"Registered query handler for modality: {modality}")
    
    def create_query(self, query_text: str, modality: str = "text", context: Dict[str, Any] = None) -> QueryActant:
        """Create a new query actant.
        
        Args:
            query_text: The text of the query
            modality: The modality of the query (default: "text")
            context: Additional context for the query
            
        Returns:
            A new QueryActant instance
        """
        query = QueryActant.create(query_text, modality, context)
        self.queries[query.id] = query
        logger.info(f"Created new query actant: {query.id} - '{query_text}'")
        return query
    
    def process_query(self, query: Union[str, QueryActant], modality: str = "text", 
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query through the appropriate handler.
        
        If a string is provided, it will be converted to a QueryActant first.
        
        Args:
            query: The query text or a QueryActant instance
            modality: The modality of the query (if query is a string)
            context: Additional context for the query (if query is a string)
            
        Returns:
            The result of processing the query
        """
        # Convert string to QueryActant if necessary
        if isinstance(query, str):
            query = self.create_query(query, modality, context)
        
        # Ensure we have a handler for this modality
        if query.modality not in self.query_handlers:
            logger.warning(f"No handler registered for modality: {query.modality}")
            return {"error": f"No handler registered for modality: {query.modality}"}
        
        # Process the query through the appropriate handler
        handler = self.query_handlers[query.modality]
        logger.info(f"Processing query {query.id} through {query.modality} handler")
        
        # Get the result from the handler
        result = handler(query.query_text, query.context)
        
        # Update the query's actant journey with this processing step
        if query.actant_journey:
            processing_point = ActantJourney.create_journey_point(
                actant_name=query.id,
                domain_id="query_processing_domain",
                predicate_id="query_processed",
                role="subject",
                context={"result_summary": str(result)[:100] + "..." if len(str(result)) > 100 else str(result)}
            )
            query.actant_journey.add_journey_point(processing_point)
        
        return result
    
    def transform_query_modality(self, query: QueryActant, new_modality: str, 
                                transformation_context: Dict[str, Any] = None) -> QueryActant:
        """Transform a query to a new modality.
        
        Args:
            query: The QueryActant to transform
            new_modality: The new modality for the query
            transformation_context: Additional context for the transformation
            
        Returns:
            A new QueryActant in the new modality
        """
        new_query = query.transform_modality(new_modality, transformation_context)
        self.queries[new_query.id] = new_query
        logger.info(f"Transformed query {query.id} to new modality {new_modality}, new query ID: {new_query.id}")
        return new_query
    
    def evolve_query(self, query: QueryActant, new_query_text: str, 
                   evolution_context: Dict[str, Any] = None) -> QueryActant:
        """Evolve a query based on interactions with the system.
        
        Args:
            query: The QueryActant to evolve
            new_query_text: The evolved query text
            evolution_context: Additional context for the evolution
            
        Returns:
            A new QueryActant representing the evolved query
        """
        new_query = query.evolve(new_query_text, evolution_context)
        self.queries[new_query.id] = new_query
        logger.info(f"Evolved query {query.id} to '{new_query_text}', new query ID: {new_query.id}")
        return new_query
    
    def detect_query_bridges(self, query: QueryActant, actant_journeys: List[ActantJourney] = None) -> List[MeaningBridge]:
        """Detect meaning bridges between a query and other actants.
        
        Args:
            query: The QueryActant to detect bridges for
            actant_journeys: Optional list of actant journeys to compare against
                            (if None, will use all available journeys)
            
        Returns:
            A list of MeaningBridge instances representing the detected bridges
        """
        # Ensure the query has an actant journey
        if not query.actant_journey:
            logger.warning(f"Query {query.id} has no actant journey, cannot detect bridges")
            return []
        
        # If no actant journeys provided, use an empty list (will be populated by the tracker)
        if actant_journeys is None:
            actant_journeys = []
        
        # Add the query's actant journey to the list
        actant_journeys.append(query.actant_journey)
        
        # Detect bridges using the meaning bridge tracker
        bridges = self.meaning_bridge_tracker.detect_bridges(actant_journeys, [])
        
        # Filter bridges to only include those involving the query
        query_bridges = [
            bridge for bridge in bridges 
            if bridge.source_actant_id == query.id or bridge.target_actant_id == query.id
        ]
        
        logger.info(f"Detected {len(query_bridges)} meaning bridges for query {query.id}")
        return query_bridges
    
    def get_query_by_id(self, query_id: str) -> Optional[QueryActant]:
        """Get a query actant by its ID.
        
        Args:
            query_id: The ID of the query to retrieve
            
        Returns:
            The QueryActant with the specified ID, or None if not found
        """
        return self.queries.get(query_id)
    
    def get_all_queries(self) -> List[QueryActant]:
        """Get all query actants in the system.
        
        Returns:
            A list of all QueryActant instances
        """
        return list(self.queries.values())
    
    def create_query_narrative(self, query: QueryActant, bridges: List[MeaningBridge] = None) -> str:
        """Create a narrative for a query based on its journey and bridges.
        
        Args:
            query: The QueryActant to create a narrative for
            bridges: Optional list of meaning bridges involving the query
                    (if None, will detect bridges)
            
        Returns:
            A narrative string describing the query's journey and bridges
        """
        # Detect bridges if not provided
        if bridges is None:
            bridges = self.detect_query_bridges(query)
        
        # Create the narrative
        narrative = f"# Query Narrative: '{query.query_text}'\n\n"
        
        # Add query information
        narrative += f"## Query Information\n\n"
        narrative += f"- **ID**: {query.id}\n"
        narrative += f"- **Modality**: {query.modality}\n"
        narrative += f"- **Created**: {query.actant_journey.journey_points[0].timestamp if query.actant_journey and query.actant_journey.journey_points else 'Unknown'}\n\n"
        
        # Add journey information
        if query.actant_journey and query.actant_journey.journey_points:
            narrative += f"## Query Journey\n\n"
            for i, point in enumerate(query.actant_journey.journey_points):
                narrative += f"{i+1}. **{point.predicate_id}** in domain '{point.domain_id}' at {point.timestamp}\n"
            narrative += "\n"
        
        # Add bridge information
        if bridges:
            narrative += f"## Meaning Bridges\n\n"
            narrative += f"This query has formed {len(bridges)} meaning bridges with other actants:\n\n"
            
            # Group bridges by type
            bridges_by_type = {}
            for bridge in bridges:
                if bridge.bridge_type not in bridges_by_type:
                    bridges_by_type[bridge.bridge_type] = []
                bridges_by_type[bridge.bridge_type].append(bridge)
            
            # Add information for each bridge type
            for bridge_type, type_bridges in bridges_by_type.items():
                narrative += f"### {bridge_type.title()} Bridges\n\n"
                
                # Sort bridges by propensity
                sorted_bridges = sorted(type_bridges, key=lambda b: b.propensity, reverse=True)
                
                # Add information for each bridge
                for bridge in sorted_bridges[:5]:  # Show top 5
                    other_actant = bridge.target_actant_id if bridge.source_actant_id == query.id else bridge.source_actant_id
                    narrative += f"- Connection with **{other_actant}** (Propensity: {bridge.propensity:.2f})\n"
                    
                    # Add context details based on type
                    if bridge_type == "co-occurrence":
                        narrative += f"  - Domain: {bridge.context.get('domain', 'Unknown')}\n"
                    elif bridge_type == "sequence":
                        narrative += f"  - Path: {bridge.context.get('source_domain', 'Unknown')} → "
                        narrative += f"{bridge.context.get('intermediate_domain', 'Unknown')} → "
                        narrative += f"{bridge.context.get('target_domain', 'Unknown')}\n"
                    elif bridge_type == "domain_crossing":
                        narrative += f"  - Crossing: {bridge.context.get('source_domain', 'Unknown')} → "
                        narrative += f"{bridge.context.get('target_domain', 'Unknown')}\n"
                
                narrative += "\n"
        
        # Add potential insights
        narrative += f"## Potential Insights\n\n"
        narrative += f"Based on this query's journey and bridges, the following insights emerge:\n\n"
        
        if bridges:
            narrative += f"1. This query has formed meaningful connections across {len(bridges_by_type)} different types of relationships.\n"
            if len(bridges) > 5:
                narrative += f"2. The query shows strong connectivity in the semantic landscape with {len(bridges)} total bridges.\n"
            
            # Add specific insights based on bridge types
            if "co-occurrence" in bridges_by_type:
                narrative += f"3. The query co-occurs with other actants in shared semantic domains, suggesting contextual relationships.\n"
            if "sequence" in bridges_by_type:
                narrative += f"4. The query participates in transformation sequences, suggesting potential causal or procedural relationships.\n"
            if "domain_crossing" in bridges_by_type:
                narrative += f"5. The query crosses domain boundaries, suggesting it serves as a connector between different semantic areas.\n"
        else:
            narrative += f"1. This query is still exploring the semantic landscape and has not yet formed strong bridges with other actants.\n"
            narrative += f"2. Further interactions may reveal potential connections and relationships.\n"
        
        return narrative
