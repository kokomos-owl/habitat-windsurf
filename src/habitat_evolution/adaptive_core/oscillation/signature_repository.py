"""
Oscillatory Signature Repository

This module implements the repository for storing and retrieving oscillatory signatures
in ArangoDB. It provides methods for creating, updating, and querying signatures based on
their oscillatory properties.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import uuid
from datetime import datetime
import logging
import numpy as np

from ..persistence.arangodb.base_repository import ArangoDBBaseRepository
from .oscillatory_signature import OscillatorySignature, HarmonicComponent

logger = logging.getLogger(__name__)

class OscillatorySignatureRepository(ArangoDBBaseRepository):
    """Repository for managing oscillatory signatures in ArangoDB."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__()
        self.collection_name = "OscillatorySignature"
        
    def _dict_to_entity(self, entity_dict: Dict[str, Any]) -> OscillatorySignature:
        """Convert a dictionary to an OscillatorySignature entity."""
        return OscillatorySignature.from_dict(entity_dict)
    
    def find_by_entity_id(self, entity_id: str) -> Optional[OscillatorySignature]:
        """
        Find a signature by its entity ID.
        
        Args:
            entity_id: ID of the entity to find signature for
            
        Returns:
            OscillatorySignature if found, None otherwise
        """
        query = f"""
        FOR doc IN {self.collection_name}
            FILTER doc.entity_id == @entity_id
            RETURN doc
        """
        
        params = {"entity_id": entity_id}
        result = self.db.aql.execute(query, bind_vars=params)
        
        for doc in result:
            return self._dict_to_entity(doc)
        
        return None
    
    def find_by_entity_type(self, entity_type: str) -> List[OscillatorySignature]:
        """
        Find all signatures for a specific entity type.
        
        Args:
            entity_type: Type of entity to find signatures for
            
        Returns:
            List of OscillatorySignature objects
        """
        query = f"""
        FOR doc IN {self.collection_name}
            FILTER doc.entity_type == @entity_type
            RETURN doc
        """
        
        params = {"entity_type": entity_type}
        result = self.db.aql.execute(query, bind_vars=params)
        
        return [self._dict_to_entity(doc) for doc in result]
    
    def find_resonant_signatures(self, 
                                signature: OscillatorySignature, 
                                threshold: float = 0.7, 
                                limit: int = 10) -> List[Tuple[OscillatorySignature, float]]:
        """
        Find signatures that resonate with the given signature.
        
        Args:
            signature: The signature to find resonances for
            threshold: Minimum resonance value (0-1)
            limit: Maximum number of results
            
        Returns:
            List of tuples (signature, resonance_value) sorted by resonance
        """
        # This is a complex query that would ideally use vector similarity in ArangoDB
        # For the POC, we'll retrieve all signatures and calculate resonance in Python
        
        # Get all signatures except the one we're comparing against
        query = f"""
        FOR doc IN {self.collection_name}
            FILTER doc.id != @signature_id
            RETURN doc
        """
        
        params = {"signature_id": signature.id}
        result = self.db.aql.execute(query, bind_vars=params)
        
        # Calculate resonance for each signature
        resonances = []
        for doc in result:
            other_sig = self._dict_to_entity(doc)
            resonance = signature.calculate_resonance(other_sig)
            if resonance >= threshold:
                resonances.append((other_sig, resonance))
        
        # Sort by resonance (descending) and limit results
        resonances.sort(key=lambda x: x[1], reverse=True)
        return resonances[:limit]
    
    def find_by_frequency_range(self, 
                               min_freq: float, 
                               max_freq: float) -> List[OscillatorySignature]:
        """
        Find signatures with fundamental frequency in the given range.
        
        Args:
            min_freq: Minimum frequency
            max_freq: Maximum frequency
            
        Returns:
            List of OscillatorySignature objects
        """
        query = f"""
        FOR doc IN {self.collection_name}
            FILTER doc.fundamental_frequency >= @min_freq AND 
                   doc.fundamental_frequency <= @max_freq
            RETURN doc
        """
        
        params = {"min_freq": min_freq, "max_freq": max_freq}
        result = self.db.aql.execute(query, bind_vars=params)
        
        return [self._dict_to_entity(doc) for doc in result]
    
    def find_by_amplitude_range(self, 
                               min_amp: float, 
                               max_amp: float) -> List[OscillatorySignature]:
        """
        Find signatures with fundamental amplitude in the given range.
        
        Args:
            min_amp: Minimum amplitude
            max_amp: Maximum amplitude
            
        Returns:
            List of OscillatorySignature objects
        """
        query = f"""
        FOR doc IN {self.collection_name}
            FILTER doc.fundamental_amplitude >= @min_amp AND 
                   doc.fundamental_amplitude <= @max_amp
            RETURN doc
        """
        
        params = {"min_amp": min_amp, "max_amp": max_amp}
        result = self.db.aql.execute(query, bind_vars=params)
        
        return [self._dict_to_entity(doc) for doc in result]
    
    def find_active_signatures(self, min_energy: float = 0.5) -> List[OscillatorySignature]:
        """
        Find signatures with energy level above the given threshold.
        
        Args:
            min_energy: Minimum energy level
            
        Returns:
            List of OscillatorySignature objects
        """
        query = f"""
        FOR doc IN {self.collection_name}
            FILTER doc.energy_level >= @min_energy
            RETURN doc
        """
        
        params = {"min_energy": min_energy}
        result = self.db.aql.execute(query, bind_vars=params)
        
        return [self._dict_to_entity(doc) for doc in result]
    
    def find_by_harmonic_pattern(self, 
                                target_ratios: List[float], 
                                tolerance: float = 0.1) -> List[OscillatorySignature]:
        """
        Find signatures with harmonic patterns matching the target ratios.
        
        Args:
            target_ratios: List of frequency ratios to match
            tolerance: Tolerance for matching ratios
            
        Returns:
            List of OscillatorySignature objects
        """
        # This is a complex query that would require custom logic
        # For the POC, we'll retrieve all signatures and filter in Python
        
        all_signatures = self.find_all()
        matching_signatures = []
        
        for sig in all_signatures:
            if not sig.harmonics or len(sig.harmonics) < len(target_ratios):
                continue
                
            # Calculate ratios relative to fundamental
            actual_ratios = [h.frequency / sig.fundamental_frequency for h in sig.harmonics]
            
            # Check if ratios match within tolerance
            matches = 0
            for target in target_ratios:
                for actual in actual_ratios:
                    if abs(actual - target) <= tolerance:
                        matches += 1
                        break
            
            # If we found matches for all target ratios
            if matches == len(target_ratios):
                matching_signatures.append(sig)
        
        return matching_signatures
    
    def update_signature_state(self, 
                              signature_id: str, 
                              time_delta: float = 1.0,
                              amplitude_change: float = 0.0,
                              energy_change: float = 0.0,
                              metadata_updates: Optional[Dict[str, Any]] = None) -> Optional[OscillatorySignature]:
        """
        Update a signature's state based on time passage and changes.
        
        Args:
            signature_id: ID of the signature to update
            time_delta: Time elapsed since last update
            amplitude_change: Change in amplitude
            energy_change: Change in energy level
            metadata_updates: Updates to metadata
            
        Returns:
            Updated OscillatorySignature if found, None otherwise
        """
        # Find the signature
        signature = self.find_by_id(signature_id)
        if not signature:
            return None
        
        # Update the signature state
        signature.update_state(
            time_delta=time_delta,
            amplitude_change=amplitude_change,
            energy_change=energy_change,
            metadata_updates=metadata_updates
        )
        
        # Save the updated signature
        self.save(signature)
        
        return signature
    
    def add_harmonic_to_signature(self, 
                                 signature_id: str, 
                                 frequency: float, 
                                 amplitude: float, 
                                 phase: float = 0.0) -> Optional[OscillatorySignature]:
        """
        Add a harmonic component to a signature.
        
        Args:
            signature_id: ID of the signature to update
            frequency: Frequency of the harmonic
            amplitude: Amplitude of the harmonic
            phase: Phase of the harmonic
            
        Returns:
            Updated OscillatorySignature if found, None otherwise
        """
        # Find the signature
        signature = self.find_by_id(signature_id)
        if not signature:
            return None
        
        # Add the harmonic
        signature.add_harmonic(frequency, amplitude, phase)
        
        # Save the updated signature
        self.save(signature)
        
        return signature
