"""
Oscillatory Signature Service with AdaptiveID Integration

This module provides services for creating, managing, and analyzing oscillatory signatures
for entities in the Habitat Evolution system, with full integration with AdaptiveID.
It leverages the OscillatoryAdaptiveID class to provide a unified approach to entity
identity and oscillatory behavior.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
import uuid
from datetime import datetime
import logging
import numpy as np
from collections import deque

from ..id.adaptive_id import AdaptiveID
from ..id.oscillatory_adaptive_id import OscillatoryAdaptiveID
from ..query.query_actant import QueryActant
from ..transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint
from ..transformation.meaning_bridges import MeaningBridge
from ...field.field_state import TonicHarmonicFieldState

from .oscillatory_signature import OscillatorySignature, HarmonicComponent
from .signature_repository import OscillatorySignatureRepository

logger = logging.getLogger(__name__)

class AdaptiveIDSignatureService:
    """
    Service for managing oscillatory signatures integrated with AdaptiveID.
    
    This service:
    1. Creates OscillatoryAdaptiveID instances from existing entities
    2. Updates oscillatory properties based on entity interactions and field state
    3. Analyzes oscillatory patterns to detect resonance and interference
    4. Predicts future states and interactions based on oscillatory properties
    """
    
    def __init__(self, repository: Optional[OscillatorySignatureRepository] = None):
        """
        Initialize the service.
        
        Args:
            repository: Repository for storing and retrieving signatures
        """
        self.repository = repository or OscillatorySignatureRepository()
        
        # Cache of recently accessed adaptive IDs to avoid database lookups
        self.adaptive_id_cache = {}
        self.cache_size = 100
        
        # History of interaction patterns for analysis
        self.interaction_history = deque(maxlen=50)
        
        logger.info("AdaptiveIDSignatureService initialized")
    
    def create_oscillatory_adaptive_id_for_query(self, query: QueryActant) -> OscillatoryAdaptiveID:
        """
        Create an oscillatory adaptive ID for a query actant.
        
        This extracts oscillatory properties from the query's context, journey,
        and relationship properties to create an OscillatoryAdaptiveID that
        captures its wave-like behavior in the semantic field.
        
        Args:
            query: The query actant to create an oscillatory adaptive ID for
            
        Returns:
            A new OscillatoryAdaptiveID for the query
        """
        logger.info(f"Creating oscillatory adaptive ID for query: {query.id}")
        
        # Check if query already has an AdaptiveID
        if not query.adaptive_id:
            logger.warning(f"Query {query.id} does not have an AdaptiveID. Creating one.")
            query.initialize_adaptive_id()
        
        # Extract base frequency from query complexity and context size
        # More complex queries oscillate more slowly (lower frequency)
        query_complexity = len(query.query_text.split()) / 20  # Normalize by typical query length
        context_size = len(query.context) if query.context else 0
        base_frequency = 0.5 / (1 + query_complexity + (context_size / 10))
        
        # Extract amplitude from query confidence or relationship properties
        amplitude = 0.5  # Default amplitude
        
        # If the query has relationship properties, use them to set amplitude
        if hasattr(query, 'last_result') and query.last_result:
            if 'relationship_properties' in query.last_result:
                props = query.last_result['relationship_properties']
                if 'resonance' in props:
                    amplitude = props['resonance']
                elif 'coherence' in props:
                    amplitude = props['coherence']
        
        # Extract phase from query journey or creation time
        phase = 0.0
        if query.actant_journey and query.actant_journey.journey_points:
            # Use number of journey points to influence phase
            phase = (len(query.actant_journey.journey_points) * np.pi / 4) % (2 * np.pi)
        
        # Create harmonics based on query domains and modality
        harmonics = []
        
        # Add harmonic for query modality
        modality_frequencies = {
            "text": 2.0,
            "image": 3.0,
            "audio": 5.0,
            "video": 7.0
        }
        
        if query.modality in modality_frequencies:
            harmonics.append({
                "frequency": base_frequency * modality_frequencies[query.modality],
                "amplitude": amplitude * 0.8,
                "phase": phase
            })
        
        # Add harmonics for relevant domains if present in results
        if hasattr(query, 'last_result') and query.last_result:
            if 'relevant_domains' in query.last_result:
                for i, domain in enumerate(query.last_result['relevant_domains'][:3]):
                    # Create a unique frequency for each domain based on its name
                    domain_hash = sum(ord(c) for c in domain) % 100
                    domain_freq = base_frequency * (1 + (domain_hash / 50))
                    
                    harmonics.append({
                        "frequency": domain_freq,
                        "amplitude": amplitude * (0.7 - (i * 0.1)),  # Decreasing amplitude
                        "phase": (phase + (i * np.pi / 6)) % (2 * np.pi)  # Slight phase shift
                    })
        
        # Create OscillatoryAdaptiveID from existing AdaptiveID
        oscillatory_id = OscillatoryAdaptiveID.from_adaptive_id(
            query.adaptive_id,
            fundamental_frequency=base_frequency,
            fundamental_amplitude=amplitude,
            fundamental_phase=phase,
            harmonics=harmonics
        )
        
        # Add metadata about the query
        oscillatory_id.update_temporal_context(
            "query_metadata",
            {
                "query_text": query.query_text,
                "modality": query.modality,
                "creation_time": datetime.now().isoformat()
            },
            "oscillatory_initialization"
        )
        
        # Add confidence if available
        if hasattr(query, 'last_result') and query.last_result and 'confidence' in query.last_result:
            oscillatory_id.update_temporal_context(
                "query_confidence",
                {"confidence": query.last_result['confidence']},
                "oscillatory_initialization"
            )
        
        # Replace the query's AdaptiveID with the OscillatoryAdaptiveID
        query.adaptive_id = oscillatory_id
        
        # Update cache
        self.adaptive_id_cache[query.id] = oscillatory_id
        
        logger.info(f"Created oscillatory adaptive ID for query {query.id}")
        return oscillatory_id
    
    def create_oscillatory_adaptive_id_for_journey(self, journey: ActantJourney) -> Optional[OscillatoryAdaptiveID]:
        """
        Create an oscillatory adaptive ID for an actant journey.
        
        This extracts oscillatory properties from the journey's points,
        transitions, and temporal patterns to create an OscillatoryAdaptiveID
        that captures its wave-like behavior in the semantic field.
        
        Args:
            journey: The actant journey to create an oscillatory adaptive ID for
            
        Returns:
            A new OscillatoryAdaptiveID for the journey, or None if no base AdaptiveID exists
        """
        logger.info(f"Creating oscillatory adaptive ID for actant journey: {journey.actant_name}")
        
        # We need a base AdaptiveID to convert to OscillatoryAdaptiveID
        # For this example, we'll create a simple one if it doesn't exist
        base_adaptive_id = None
        
        # Check if we have a cached AdaptiveID for this actant
        if journey.actant_name in self.adaptive_id_cache:
            base_adaptive_id = self.adaptive_id_cache[journey.actant_name]
        else:
            # In a real implementation, we would look up the AdaptiveID from a repository
            # For this example, we'll create a new one
            base_adaptive_id = AdaptiveID(
                base_concept=journey.actant_name,
                creator_id="journey_tracker",
                weight=1.0,
                confidence=0.9,
                uncertainty=0.1
            )
        
        # Extract base frequency from journey length and transition rate
        # Longer journeys with more transitions oscillate more slowly
        journey_length = len(journey.journey_points)
        if journey_length == 0:
            return None
            
        # Calculate time span if timestamps are available
        time_span = 1.0
        if journey_length >= 2:
            try:
                start_time = datetime.fromisoformat(journey.journey_points[0].timestamp)
                end_time = datetime.fromisoformat(journey.journey_points[-1].timestamp)
                time_span = (end_time - start_time).total_seconds() / 3600  # in hours
                if time_span < 0.01:  # Avoid division by zero
                    time_span = 0.01
            except (ValueError, TypeError):
                # If timestamps can't be parsed, use journey length
                time_span = journey_length
        
        # Calculate transition rate (transitions per hour)
        transition_rate = journey_length / time_span
        
        # Base frequency inversely proportional to transition rate
        # Faster transitions = lower frequency (more stable oscillation)
        base_frequency = 0.5 / (1 + (transition_rate / 5))
        
        # Extract amplitude from journey diversity
        # More diverse journeys (different domains/roles) have higher amplitude
        domains = set()
        roles = set()
        for point in journey.journey_points:
            domains.add(point.domain_id)
            roles.add(point.role)
        
        domain_diversity = len(domains) / max(1, journey_length)
        role_diversity = len(roles) / max(1, journey_length)
        
        amplitude = 0.3 + (0.7 * ((domain_diversity + role_diversity) / 2))
        
        # Extract phase from most recent journey point
        phase = 0.0
        if journey.journey_points:
            latest_point = journey.journey_points[-1]
            # Use domain and role to influence phase
            domain_hash = sum(ord(c) for c in latest_point.domain_id) % 100
            role_hash = sum(ord(c) for c in latest_point.role) % 100
            phase = ((domain_hash + role_hash) * np.pi / 100) % (2 * np.pi)
        
        # Create harmonics based on journey patterns
        harmonics = []
        
        # Add harmonic for each unique domain (up to 3)
        for i, domain in enumerate(list(domains)[:3]):
            domain_hash = sum(ord(c) for c in domain) % 100
            domain_freq = base_frequency * (1 + (domain_hash / 50))
            
            harmonics.append({
                "frequency": domain_freq,
                "amplitude": amplitude * (0.7 - (i * 0.1)),
                "phase": (phase + (i * np.pi / 6)) % (2 * np.pi)
            })
        
        # Add harmonic for transition pattern if enough points
        if journey_length >= 3:
            # Look for patterns in role transitions
            role_sequence = [p.role for p in journey.journey_points]
            has_pattern = False
            
            # Check for alternating pattern (e.g., subject-object-subject)
            alternating = True
            for i in range(2, len(role_sequence)):
                if role_sequence[i] != role_sequence[i-2]:
                    alternating = False
                    break
            
            if alternating:
                has_pattern = True
                pattern_freq = base_frequency * 2
                pattern_amp = amplitude * 0.9
                
                harmonics.append({
                    "frequency": pattern_freq,
                    "amplitude": pattern_amp,
                    "phase": phase
                })
        
        # Create OscillatoryAdaptiveID from base AdaptiveID
        oscillatory_id = OscillatoryAdaptiveID.from_adaptive_id(
            base_adaptive_id,
            fundamental_frequency=base_frequency,
            fundamental_amplitude=amplitude,
            fundamental_phase=phase,
            harmonics=harmonics
        )
        
        # Add journey metadata
        oscillatory_id.update_temporal_context(
            "journey_metadata",
            {
                "journey_length": journey_length,
                "domains": list(domains),
                "roles": list(roles),
                "creation_time": datetime.now().isoformat()
            },
            "oscillatory_initialization"
        )
        
        # Update cache
        self.adaptive_id_cache[journey.actant_name] = oscillatory_id
        
        logger.info(f"Created oscillatory adaptive ID for actant journey {journey.actant_name}")
        return oscillatory_id
    
    def create_oscillatory_adaptive_id_for_bridge(self, bridge: MeaningBridge) -> Optional[OscillatoryAdaptiveID]:
        """
        Create an oscillatory adaptive ID for a meaning bridge.
        
        This extracts oscillatory properties from the bridge's strength,
        actants, and semantic properties to create an OscillatoryAdaptiveID
        that captures its wave-like behavior in the semantic field.
        
        Args:
            bridge: The meaning bridge to create an oscillatory adaptive ID for
            
        Returns:
            A new OscillatoryAdaptiveID for the bridge, or None if creation fails
        """
        logger.info(f"Creating oscillatory adaptive ID for meaning bridge between {bridge.source_actant} and {bridge.target_actant}")
        
        # We need a base AdaptiveID to convert to OscillatoryAdaptiveID
        # For this example, we'll create a simple one
        bridge_id = f"{bridge.source_actant}_{bridge.target_actant}"
        base_adaptive_id = AdaptiveID(
            base_concept=f"Bridge between {bridge.source_actant} and {bridge.target_actant}",
            creator_id="meaning_bridge_tracker",
            weight=bridge.strength,
            confidence=0.9,
            uncertainty=0.1
        )
        
        # Extract base frequency from bridge strength
        # Stronger bridges oscillate more quickly (higher frequency)
        base_frequency = 0.1 + (bridge.strength * 0.4)
        
        # Extract amplitude from bridge strength
        amplitude = bridge.strength
        
        # Extract phase from bridge type
        phase = 0.0
        bridge_type_phases = {
            "semantic": 0.0,
            "temporal": np.pi/4,
            "causal": np.pi/2,
            "contextual": 3*np.pi/4,
            "analogical": np.pi,
            "contrastive": 5*np.pi/4,
            "transformative": 3*np.pi/2,
            "emergent": 7*np.pi/4
        }
        
        if hasattr(bridge, 'bridge_type') and bridge.bridge_type in bridge_type_phases:
            phase = bridge_type_phases[bridge.bridge_type]
        
        # Create harmonics based on bridge properties
        harmonics = []
        
        # Add harmonic for bridge type
        if hasattr(bridge, 'bridge_type'):
            type_hash = sum(ord(c) for c in bridge.bridge_type) % 100
            type_freq = base_frequency * (1 + (type_hash / 50))
            
            harmonics.append({
                "frequency": type_freq,
                "amplitude": amplitude * 0.8,
                "phase": phase
            })
        
        # Add harmonic for each property (up to 3)
        if hasattr(bridge, 'properties'):
            for i, (prop, value) in enumerate(list(bridge.properties.items())[:3]):
                prop_hash = sum(ord(c) for c in prop) % 100
                prop_freq = base_frequency * (1 + (prop_hash / 50))
                
                # Use the property value to determine amplitude
                prop_amp = amplitude * 0.7 * float(value) if isinstance(value, (int, float)) else amplitude * 0.7
                
                harmonics.append({
                    "frequency": prop_freq,
                    "amplitude": prop_amp,
                    "phase": (phase + (i * np.pi / 6)) % (2 * np.pi)
                })
        
        # Create OscillatoryAdaptiveID from base AdaptiveID
        oscillatory_id = OscillatoryAdaptiveID.from_adaptive_id(
            base_adaptive_id,
            fundamental_frequency=base_frequency,
            fundamental_amplitude=amplitude,
            fundamental_phase=phase,
            harmonics=harmonics
        )
        
        # Add bridge metadata
        oscillatory_id.update_temporal_context(
            "bridge_metadata",
            {
                "source_actant": bridge.source_actant,
                "target_actant": bridge.target_actant,
                "bridge_type": bridge.bridge_type if hasattr(bridge, 'bridge_type') else "unknown",
                "creation_time": datetime.now().isoformat()
            },
            "oscillatory_initialization"
        )
        
        # Update cache
        self.adaptive_id_cache[bridge_id] = oscillatory_id
        
        logger.info(f"Created oscillatory adaptive ID for meaning bridge {bridge_id}")
        return oscillatory_id
    
    def get_oscillatory_adaptive_id(self, entity_id: str) -> Optional[OscillatoryAdaptiveID]:
        """
        Get the oscillatory adaptive ID for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            OscillatoryAdaptiveID if found, None otherwise
        """
        # Check cache first
        if entity_id in self.adaptive_id_cache:
            return self.adaptive_id_cache[entity_id]
        
        # In a real implementation, we would look up the OscillatoryAdaptiveID from a repository
        # For this example, we'll return None if not in cache
        return None
    
    def update_oscillatory_state(self, 
                               entity_id: str, 
                               time_delta: float = 1.0,
                               amplitude_change: float = 0.0,
                               energy_change: float = 0.0,
                               metadata_updates: Optional[Dict[str, Any]] = None) -> Optional[OscillatoryAdaptiveID]:
        """
        Update the oscillatory state of an entity.
        
        Args:
            entity_id: ID of the entity
            time_delta: Time elapsed since last update
            amplitude_change: Change in amplitude
            energy_change: Change in energy level
            metadata_updates: Updates to metadata
            
        Returns:
            Updated OscillatoryAdaptiveID if found, None otherwise
        """
        # Get the oscillatory adaptive ID
        oscillatory_id = self.get_oscillatory_adaptive_id(entity_id)
        if not oscillatory_id:
            return None
        
        # Update the oscillatory state
        oscillatory_id.update_oscillatory_state(
            time_delta=time_delta,
            amplitude_change=amplitude_change,
            energy_change=energy_change,
            metadata_updates=metadata_updates
        )
        
        # Update cache
        self.adaptive_id_cache[entity_id] = oscillatory_id
        
        return oscillatory_id
    
    def find_resonant_entities(self, 
                              entity_id: str, 
                              threshold: float = 0.7, 
                              limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find entities that resonate with the given entity.
        
        Args:
            entity_id: ID of the entity to find resonances for
            threshold: Minimum resonance value (0-1)
            limit: Maximum number of results
            
        Returns:
            List of tuples (entity_id, resonance_value) sorted by resonance
        """
        # Get the oscillatory adaptive ID
        oscillatory_id = self.get_oscillatory_adaptive_id(entity_id)
        if not oscillatory_id:
            return []
        
        # Get all cached oscillatory adaptive IDs
        all_ids = list(self.adaptive_id_cache.values())
        
        # Calculate resonance with each ID
        resonances = []
        for other_id in all_ids:
            if other_id.id != oscillatory_id.id:
                resonance = oscillatory_id.calculate_resonance(other_id)
                if resonance >= threshold:
                    resonances.append((other_id.id, resonance))
        
        # Sort by resonance (descending) and limit results
        resonances.sort(key=lambda x: x[1], reverse=True)
        return resonances[:limit]
    
    def predict_entity_state(self, 
                           entity_id: str, 
                           time_delta: float) -> Optional[Dict[str, Any]]:
        """
        Predict the future state of an entity based on its oscillatory properties.
        
        Args:
            entity_id: ID of the entity
            time_delta: Time to project forward
            
        Returns:
            Dictionary with predicted state, or None if oscillatory adaptive ID not found
        """
        # Get the oscillatory adaptive ID
        oscillatory_id = self.get_oscillatory_adaptive_id(entity_id)
        if not oscillatory_id:
            return None
        
        # Project the oscillatory state
        projected = oscillatory_id.project_future_state(time_delta)
        
        # Create prediction result
        prediction = {
            "entity_id": entity_id,
            "current_state": {
                "fundamental_frequency": oscillatory_id.oscillatory_properties["fundamental_frequency"],
                "fundamental_amplitude": oscillatory_id.oscillatory_properties["fundamental_amplitude"],
                "fundamental_phase": oscillatory_id.oscillatory_properties["fundamental_phase"],
                "energy_level": oscillatory_id.oscillatory_properties["energy_level"],
                "current_value": oscillatory_id.get_current_value()
            },
            "predicted_state": {
                "fundamental_frequency": projected["fundamental_frequency"],
                "fundamental_amplitude": projected["fundamental_amplitude"],
                "fundamental_phase": projected["fundamental_phase"],
                "energy_level": projected["energy_level"]
            },
            "time_delta": time_delta,
            "prediction_time": datetime.now().isoformat()
        }
        
        return prediction
    
    def predict_entity_interaction(self, 
                                 entity_id1: str, 
                                 entity_id2: str, 
                                 time_delta: float) -> Optional[Dict[str, Any]]:
        """
        Predict the interaction between two entities based on their oscillatory properties.
        
        Args:
            entity_id1: ID of the first entity
            entity_id2: ID of the second entity
            time_delta: Time to project forward
            
        Returns:
            Dictionary with predicted interaction, or None if oscillatory adaptive IDs not found
        """
        # Get the oscillatory adaptive IDs
        oscillatory_id1 = self.get_oscillatory_adaptive_id(entity_id1)
        oscillatory_id2 = self.get_oscillatory_adaptive_id(entity_id2)
        
        if not oscillatory_id1 or not oscillatory_id2:
            return None
        
        # Calculate current resonance and interference
        current_resonance = oscillatory_id1.calculate_resonance(oscillatory_id2)
        current_interference, current_interference_type = oscillatory_id1.calculate_interference(oscillatory_id2)
        
        # Project the oscillatory states
        projected1 = OscillatoryAdaptiveID.from_adaptive_id(
            oscillatory_id1,
            fundamental_frequency=oscillatory_id1.oscillatory_properties["fundamental_frequency"],
            fundamental_amplitude=oscillatory_id1.oscillatory_properties["fundamental_amplitude"],
            fundamental_phase=oscillatory_id1.oscillatory_properties["fundamental_phase"]
        )
        projected1.update_oscillatory_state(time_delta=time_delta)
        
        projected2 = OscillatoryAdaptiveID.from_adaptive_id(
            oscillatory_id2,
            fundamental_frequency=oscillatory_id2.oscillatory_properties["fundamental_frequency"],
            fundamental_amplitude=oscillatory_id2.oscillatory_properties["fundamental_amplitude"],
            fundamental_phase=oscillatory_id2.oscillatory_properties["fundamental_phase"]
        )
        projected2.update_oscillatory_state(time_delta=time_delta)
        
        # Calculate projected resonance and interference
        projected_resonance = projected1.calculate_resonance(projected2)
        projected_interference, projected_interference_type = projected1.calculate_interference(projected2)
        
        # Create prediction result
        prediction = {
            "entity_id1": entity_id1,
            "entity_id2": entity_id2,
            "current_interaction": {
                "resonance": current_resonance,
                "interference_value": current_interference,
                "interference_type": current_interference_type
            },
            "predicted_interaction": {
                "resonance": projected_resonance,
                "interference_value": projected_interference,
                "interference_type": projected_interference_type
            },
            "resonance_change": projected_resonance - current_resonance,
            "interference_change": projected_interference - current_interference,
            "time_delta": time_delta,
            "prediction_time": datetime.now().isoformat()
        }
        
        return prediction
