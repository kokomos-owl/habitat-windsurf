"""
Oscillatory Signature Service

This module provides services for creating, managing, and analyzing oscillatory signatures
for entities in the Habitat Evolution system. It integrates with existing components like
QueryActant and FieldStateModulator to extract oscillatory patterns from existing data.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
import uuid
from datetime import datetime
import logging
import numpy as np
from collections import deque

from ..query.query_actant import QueryActant
from ..transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint
from ..transformation.meaning_bridges import MeaningBridge
from ..id.adaptive_id import AdaptiveID
from ...field.field_state import TonicHarmonicFieldState

from .oscillatory_signature import OscillatorySignature, HarmonicComponent
from .signature_repository import OscillatorySignatureRepository

logger = logging.getLogger(__name__)

class OscillatorySignatureService:
    """
    Service for managing oscillatory signatures of entities in the system.
    
    This service:
    1. Creates signatures from existing entities and their data
    2. Updates signatures based on entity interactions and field state
    3. Analyzes signatures to detect patterns, resonance, and interference
    4. Predicts future states and interactions based on signatures
    """
    
    def __init__(self, repository: Optional[OscillatorySignatureRepository] = None):
        """
        Initialize the service.
        
        Args:
            repository: Repository for storing and retrieving signatures
        """
        self.repository = repository or OscillatorySignatureRepository()
        
        # Cache of recently accessed signatures to avoid database lookups
        self.signature_cache = {}
        self.cache_size = 100
        
        # History of signature interactions for pattern analysis
        self.interaction_history = deque(maxlen=50)
        
        logger.info("OscillatorySignatureService initialized")
    
    def create_signature_for_query_actant(self, query: QueryActant) -> OscillatorySignature:
        """
        Create an oscillatory signature for a query actant.
        
        This extracts oscillatory properties from the query's context, journey,
        and relationship properties to create a signature that captures its
        wave-like behavior in the semantic field.
        
        Args:
            query: The query actant to create a signature for
            
        Returns:
            A new OscillatorySignature for the query
        """
        logger.info(f"Creating oscillatory signature for query: {query.id}")
        
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
        
        # Create metadata with query information
        metadata = {
            "query_text": query.query_text,
            "modality": query.modality,
            "creation_time": datetime.now().isoformat()
        }
        
        # Add confidence if available
        if hasattr(query, 'last_result') and query.last_result and 'confidence' in query.last_result:
            metadata["confidence"] = query.last_result['confidence']
        
        # Create the signature
        signature = OscillatorySignature.create(
            entity_id=query.id,
            entity_type="query_actant",
            fundamental_frequency=base_frequency,
            fundamental_amplitude=amplitude,
            fundamental_phase=phase,
            harmonics=harmonics,
            metadata=metadata
        )
        
        # Save the signature
        self.repository.save(signature)
        
        # Update cache
        self.signature_cache[query.id] = signature
        
        logger.info(f"Created oscillatory signature {signature.id} for query {query.id}")
        return signature
    
    def create_signature_for_actant_journey(self, journey: ActantJourney) -> OscillatorySignature:
        """
        Create an oscillatory signature for an actant journey.
        
        This extracts oscillatory properties from the journey's points,
        transitions, and temporal patterns to create a signature that
        captures its wave-like behavior in the semantic field.
        
        Args:
            journey: The actant journey to create a signature for
            
        Returns:
            A new OscillatorySignature for the journey
        """
        logger.info(f"Creating oscillatory signature for actant journey: {journey.actant_name}")
        
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
        
        # Create metadata with journey information
        metadata = {
            "actant_name": journey.actant_name,
            "journey_length": journey_length,
            "domains": list(domains),
            "roles": list(roles),
            "creation_time": datetime.now().isoformat()
        }
        
        # Create the signature
        signature = OscillatorySignature.create(
            entity_id=journey.actant_name,
            entity_type="actant_journey",
            fundamental_frequency=base_frequency,
            fundamental_amplitude=amplitude,
            fundamental_phase=phase,
            harmonics=harmonics,
            metadata=metadata
        )
        
        # Save the signature
        self.repository.save(signature)
        
        # Update cache
        self.signature_cache[journey.actant_name] = signature
        
        logger.info(f"Created oscillatory signature {signature.id} for actant journey {journey.actant_name}")
        return signature
    
    def create_signature_for_meaning_bridge(self, bridge: MeaningBridge) -> OscillatorySignature:
        """
        Create an oscillatory signature for a meaning bridge.
        
        This extracts oscillatory properties from the bridge's strength,
        actants, and semantic properties to create a signature that
        captures its wave-like behavior in the semantic field.
        
        Args:
            bridge: The meaning bridge to create a signature for
            
        Returns:
            A new OscillatorySignature for the bridge
        """
        logger.info(f"Creating oscillatory signature for meaning bridge between {bridge.source_actant} and {bridge.target_actant}")
        
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
            for i, (prop, value) in enumerate(bridge.properties.items()[:3]):
                prop_hash = sum(ord(c) for c in prop) % 100
                prop_freq = base_frequency * (1 + (prop_hash / 50))
                
                # Use the property value to determine amplitude
                prop_amp = amplitude * 0.7 * float(value) if isinstance(value, (int, float)) else amplitude * 0.7
                
                harmonics.append({
                    "frequency": prop_freq,
                    "amplitude": prop_amp,
                    "phase": (phase + (i * np.pi / 6)) % (2 * np.pi)
                })
        
        # Create metadata with bridge information
        metadata = {
            "source_actant": bridge.source_actant,
            "target_actant": bridge.target_actant,
            "bridge_type": bridge.bridge_type if hasattr(bridge, 'bridge_type') else "unknown",
            "creation_time": datetime.now().isoformat()
        }
        
        # Create the signature
        bridge_id = f"{bridge.source_actant}_{bridge.target_actant}"
        signature = OscillatorySignature.create(
            entity_id=bridge_id,
            entity_type="meaning_bridge",
            fundamental_frequency=base_frequency,
            fundamental_amplitude=amplitude,
            fundamental_phase=phase,
            harmonics=harmonics,
            metadata=metadata
        )
        
        # Save the signature
        self.repository.save(signature)
        
        # Update cache
        self.signature_cache[bridge_id] = signature
        
        logger.info(f"Created oscillatory signature {signature.id} for meaning bridge {bridge_id}")
        return signature
    
    def create_signature_from_field_state(self, field_state: TonicHarmonicFieldState) -> OscillatorySignature:
        """
        Create an oscillatory signature from a field state.
        
        This extracts oscillatory properties from the field state metrics
        to create a signature that captures the overall field behavior.
        
        Args:
            field_state: The field state to create a signature for
            
        Returns:
            A new OscillatorySignature for the field state
        """
        logger.info("Creating oscillatory signature from field state")
        
        # Extract base frequency from field coherence
        # More coherent fields oscillate more quickly (higher frequency)
        coherence = field_state.get_metric('coherence') if hasattr(field_state, 'get_metric') else 0.5
        base_frequency = 0.1 + (coherence * 0.4)
        
        # Extract amplitude from field density
        density = field_state.get_metric('density') if hasattr(field_state, 'get_metric') else 0.5
        amplitude = 0.3 + (density * 0.7)
        
        # Extract phase from field stability
        stability = field_state.get_metric('stability') if hasattr(field_state, 'get_metric') else 0.5
        phase = stability * np.pi
        
        # Create harmonics based on field metrics
        harmonics = []
        
        # Add harmonic for turbulence
        turbulence = field_state.get_metric('turbulence') if hasattr(field_state, 'get_metric') else 0.3
        harmonics.append({
            "frequency": base_frequency * 2,
            "amplitude": amplitude * turbulence,
            "phase": (phase + np.pi/4) % (2 * np.pi)
        })
        
        # Add harmonic for dimensionality if available
        if hasattr(field_state, 'get_effective_dimensionality'):
            dim = field_state.get_effective_dimensionality()
            dim_normalized = min(1.0, dim / 10)  # Normalize to 0-1 range
            
            harmonics.append({
                "frequency": base_frequency * (1 + dim_normalized),
                "amplitude": amplitude * 0.6,
                "phase": (phase + np.pi/3) % (2 * np.pi)
            })
        
        # Create metadata with field state information
        metadata = {
            "field_id": field_state.id if hasattr(field_state, 'id') else str(uuid.uuid4()),
            "coherence": coherence,
            "density": density,
            "stability": stability,
            "turbulence": turbulence,
            "creation_time": datetime.now().isoformat()
        }
        
        # Add eigenvalues if available
        if hasattr(field_state, 'get_eigenvalues'):
            metadata["eigenvalues"] = field_state.get_eigenvalues()
        
        # Create the signature
        field_id = metadata["field_id"]
        signature = OscillatorySignature.create(
            entity_id=field_id,
            entity_type="field_state",
            fundamental_frequency=base_frequency,
            fundamental_amplitude=amplitude,
            fundamental_phase=phase,
            harmonics=harmonics,
            metadata=metadata
        )
        
        # Save the signature
        self.repository.save(signature)
        
        # Update cache
        self.signature_cache[field_id] = signature
        
        logger.info(f"Created oscillatory signature {signature.id} for field state {field_id}")
        return signature
    
    def get_signature(self, entity_id: str, entity_type: Optional[str] = None) -> Optional[OscillatorySignature]:
        """
        Get the oscillatory signature for an entity.
        
        Args:
            entity_id: ID of the entity
            entity_type: Optional type of the entity
            
        Returns:
            OscillatorySignature if found, None otherwise
        """
        # Check cache first
        if entity_id in self.signature_cache:
            return self.signature_cache[entity_id]
        
        # Find in repository
        signature = self.repository.find_by_entity_id(entity_id)
        
        # Update cache if found
        if signature:
            self.signature_cache[entity_id] = signature
            
            # Trim cache if too large
            if len(self.signature_cache) > self.cache_size:
                # Remove oldest entries
                oldest_keys = list(self.signature_cache.keys())[:len(self.signature_cache) - self.cache_size]
                for key in oldest_keys:
                    del self.signature_cache[key]
        
        return signature
    
    def update_signature(self, 
                        entity_id: str, 
                        time_delta: float = 1.0,
                        amplitude_change: float = 0.0,
                        energy_change: float = 0.0,
                        metadata_updates: Optional[Dict[str, Any]] = None) -> Optional[OscillatorySignature]:
        """
        Update the oscillatory signature for an entity.
        
        Args:
            entity_id: ID of the entity
            time_delta: Time elapsed since last update
            amplitude_change: Change in amplitude
            energy_change: Change in energy level
            metadata_updates: Updates to metadata
            
        Returns:
            Updated OscillatorySignature if found, None otherwise
        """
        # Get the signature
        signature = self.get_signature(entity_id)
        if not signature:
            return None
        
        # Update the signature
        signature.update_state(
            time_delta=time_delta,
            amplitude_change=amplitude_change,
            energy_change=energy_change,
            metadata_updates=metadata_updates
        )
        
        # Save the updated signature
        self.repository.save(signature)
        
        # Update cache
        self.signature_cache[entity_id] = signature
        
        return signature
    
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
        # Get the signature
        signature = self.get_signature(entity_id)
        if not signature:
            return []
        
        # Find resonant signatures
        resonant_sigs = self.repository.find_resonant_signatures(signature, threshold, limit)
        
        # Convert to entity IDs with resonance values
        return [(sig.entity_id, res) for sig, res in resonant_sigs]
    
    def predict_entity_state(self, 
                           entity_id: str, 
                           time_delta: float) -> Optional[Dict[str, Any]]:
        """
        Predict the future state of an entity based on its oscillatory signature.
        
        Args:
            entity_id: ID of the entity
            time_delta: Time to project forward
            
        Returns:
            Dictionary with predicted state, or None if signature not found
        """
        # Get the signature
        signature = self.get_signature(entity_id)
        if not signature:
            return None
        
        # Project the signature
        projected = signature.project_future_state(time_delta)
        
        # Create prediction result
        prediction = {
            "entity_id": entity_id,
            "current_state": {
                "fundamental_frequency": signature.fundamental_frequency,
                "fundamental_amplitude": signature.fundamental_amplitude,
                "fundamental_phase": signature.fundamental_phase,
                "energy_level": signature.energy_level,
                "current_value": signature.get_current_value()
            },
            "predicted_state": {
                "fundamental_frequency": projected.fundamental_frequency,
                "fundamental_amplitude": projected.fundamental_amplitude,
                "fundamental_phase": projected.fundamental_phase,
                "energy_level": projected.energy_level,
                "predicted_value": projected.get_current_value()
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
        Predict the interaction between two entities based on their oscillatory signatures.
        
        Args:
            entity_id1: ID of the first entity
            entity_id2: ID of the second entity
            time_delta: Time to project forward
            
        Returns:
            Dictionary with predicted interaction, or None if signatures not found
        """
        # Get the signatures
        sig1 = self.get_signature(entity_id1)
        sig2 = self.get_signature(entity_id2)
        
        if not sig1 or not sig2:
            return None
        
        # Project the signatures
        proj1 = sig1.project_future_state(time_delta)
        proj2 = sig2.project_future_state(time_delta)
        
        # Calculate current resonance and interference
        current_resonance = sig1.calculate_resonance(sig2)
        current_interference, current_interference_type = sig1.calculate_interference(sig2)
        
        # Calculate projected resonance and interference
        projected_resonance = proj1.calculate_resonance(proj2)
        projected_interference, projected_interference_type = proj1.calculate_interference(proj2)
        
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
