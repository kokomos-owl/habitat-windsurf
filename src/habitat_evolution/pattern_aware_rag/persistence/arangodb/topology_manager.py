"""
ArangoDB Topology Manager

This module provides an implementation of the TopologyManager that uses ArangoDB
for persistence of topology states, frequency domains, boundaries, and resonance points.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime
import uuid
import time

from src.habitat_evolution.pattern_aware_rag.topology.manager import TopologyManager, TopologyState
from src.habitat_evolution.pattern_aware_rag.topology.domain import FrequencyDomain
from src.habitat_evolution.pattern_aware_rag.topology.boundary import Boundary
from src.habitat_evolution.pattern_aware_rag.topology.resonance import ResonancePoint
from src.habitat_evolution.adaptive_core.models import Pattern

from .topology_repository import TopologyStateRepository
from .frequency_domain_repository import FrequencyDomainRepository
from .boundary_repository import BoundaryRepository
from .resonance_point_repository import ResonancePointRepository
from .pattern_repository import PatternRepository

logger = logging.getLogger(__name__)

class ArangoDBTopologyManager(TopologyManager):
    """
    ArangoDB implementation of the TopologyManager.
    
    This class handles the persistence of topology states, frequency domains,
    boundaries, and resonance points to ArangoDB.
    """
    
    def __init__(self):
        """Initialize the ArangoDB topology manager."""
        super().__init__()
        
        # Initialize repositories
        self.topology_repository = TopologyStateRepository()
        self.domain_repository = FrequencyDomainRepository()
        self.boundary_repository = BoundaryRepository()
        self.resonance_repository = ResonancePointRepository()
        self.pattern_repository = PatternRepository()
    
    def persist_topology_state(self, state: TopologyState) -> bool:
        """
        Persist a topology state to ArangoDB.
        
        Args:
            state: The topology state to persist
            
        Returns:
            True if the state was persisted successfully, False otherwise
        """
        try:
            # Special handling for test states
            is_test_state = getattr(state, 'is_test_state', False)
            
            # Persist frequency domains
            if hasattr(state, 'frequency_domains') and state.frequency_domains:
                for domain_id, domain in state.frequency_domains.items():
                    self.domain_repository.save(domain)
            
            # Persist boundaries
            if hasattr(state, 'boundaries') and state.boundaries:
                for boundary_id, boundary in state.boundaries.items():
                    self.boundary_repository.save(boundary)
                    
                    # Create connections between boundaries and domains
                    if hasattr(boundary, 'domain_ids') and len(boundary.domain_ids) == 2:
                        # The edge collections are handled by the graph definition
                        # in the schema manager, so we don't need to create them here
                        pass
            
            # Persist resonance points
            if hasattr(state, 'resonance_points') and state.resonance_points:
                for point_id, point in state.resonance_points.items():
                    self.resonance_repository.save(point)
                    
                    # Persist contributing patterns
                    if hasattr(point, 'contributing_pattern_ids') and point.contributing_pattern_ids:
                        for pattern_id, weight in point.contributing_pattern_ids.items():
                            # The edge collections are handled by the graph definition
                            # in the schema manager, so we don't need to create them here
                            pass
            
            # Persist pattern eigenspace properties
            if hasattr(state, 'pattern_eigenspace_properties') and state.pattern_eigenspace_properties:
                # Get current timestamp for temporal tracking
                current_timestamp = int(time.time() * 1000)  # milliseconds
                
                # Track learning window frequencies if available
                learning_window_frequencies = {}
                if hasattr(state, 'learning_windows') and state.learning_windows:
                    for window_id, window_info in state.learning_windows.items():
                        # Extract frequency from window properties
                        if 'duration_minutes' in window_info:
                            freq = 1.0 / (window_info['duration_minutes'] * 60)  # frequency in Hz
                            learning_window_frequencies[window_id] = freq
                
                # Default base frequency if no windows available
                base_frequency = 0.1  # Default 0.1 Hz (10-second cycle)
                if learning_window_frequencies:
                    # Use average of available frequencies
                    base_frequency = sum(learning_window_frequencies.values()) / len(learning_window_frequencies)
                
                # Process each pattern with temporal awareness
                for pattern_id, properties in state.pattern_eigenspace_properties.items():
                    # Check if the pattern exists
                    pattern = self.pattern_repository.find_by_id(pattern_id)
                    if not pattern:
                        # Create a new pattern
                        pattern = Pattern(
                            id=pattern_id,
                            pattern_type="eigenspace",
                            source="",
                            predicate="",
                            target=""
                        )
                    
                    # Calculate eigenspace stability based on dimensional coordinates
                    dimensional_coords = properties.get('dimensional_coordinates', [])
                    eigenspace_stability = 0.0
                    
                    if dimensional_coords:
                        # Calculate stability as inverse of variance in coordinate magnitudes
                        magnitudes = [abs(coord) for coord in dimensional_coords]
                        if len(magnitudes) > 1:
                            mean = sum(magnitudes) / len(magnitudes)
                            variance = sum((x - mean) ** 2 for x in magnitudes) / len(magnitudes)
                            eigenspace_stability = 1.0 / (1.0 + variance) if variance > 0 else 1.0
                    
                    # Calculate pattern frequency based on its appearance in learning windows
                    pattern_frequency = properties.get('frequency', base_frequency)
                    
                    # Calculate phase position in current cycle (0.0 to 1.0)
                    phase_position = (current_timestamp / 1000.0 * pattern_frequency) % 1.0
                    
                    # Calculate tonic value based on phase position
                    import math
                    tonic_value = 0.5 + 0.4 * math.sin(2 * math.pi * phase_position)
                    
                    # Calculate harmonic properties if frequency information is available
                    harmonic_ratio = 0.0
                    if 'frequency' in properties:
                        base_freq = properties.get('frequency', 0.0)
                        if base_freq > 0:
                            # Find nearest harmonic ratio (1:2, 2:3, 3:4, etc.)
                            for i in range(1, 10):
                                for j in range(i+1, i+10):
                                    harmonic = i / j
                                    if abs(base_freq - harmonic) < 0.05:
                                        harmonic_ratio = harmonic
                                        break
                    
                    # Calculate harmonic value (stability * tonic)
                    harmonic_value = eigenspace_stability * tonic_value
                    
                    # Update pattern properties
                    pattern.eigenspace_properties = {
                        'primary_dimensions': properties.get('primary_dimensions', []),
                        'dimensional_coordinates': dimensional_coords,
                        'eigenspace_centrality': properties.get('eigenspace_centrality', 0.0),
                        'eigenspace_stability': eigenspace_stability,
                        'dimensional_variance': properties.get('dimensional_variance', 0.0),
                        'resonance_groups': properties.get('resonance_groups', [])
                    }
                    
                    pattern.temporal_properties = {
                        'frequency': pattern_frequency,
                        'phase_position': phase_position,
                        'temporal_coherence': properties.get('temporal_coherence', eigenspace_stability * 0.8),
                        'harmonic_ratio': harmonic_ratio
                    }
                    
                    pattern.oscillatory_properties = {
                        'tonic_value': tonic_value,
                        'harmonic_value': harmonic_value,
                        'energy': properties.get('energy', 0.0)
                    }
                    
                    pattern.timestamp_ms = current_timestamp
                    
                    # Save the pattern
                    self.pattern_repository.save(pattern)
                    
                    # Connect patterns to their resonance groups
                    for group_id in properties.get('resonance_groups', []):
                        # The edge collections are handled by the graph definition
                        # in the schema manager, so we don't need to create them here
                        pass
            
            # Finally, persist the topology state itself
            self.topology_repository.save(state)
            
            logger.info(f"Successfully persisted topology state {state.id} to ArangoDB")
            return True
            
        except Exception as e:
            logger.error(f"Error persisting topology state to ArangoDB: {str(e)}")
            return False
    
    def persist_to_arango(self, state: TopologyState) -> bool:
        """
        Persist a topology state to ArangoDB.
        
        This is an alias for persist_topology_state to maintain compatibility
        with the existing API.
        
        Args:
            state: The topology state to persist
            
        Returns:
            True if the state was persisted successfully, False otherwise
        """
        return self.persist_topology_state(state)
    
    def load_topology_state(self, state_id: str) -> Optional[TopologyState]:
        """
        Load a topology state from ArangoDB.
        
        Args:
            state_id: The ID of the topology state to load
            
        Returns:
            The topology state if found, None otherwise
        """
        try:
            # Load the topology state
            state = self.topology_repository.find_by_id(state_id)
            if not state:
                logger.warning(f"Topology state {state_id} not found in ArangoDB")
                return None
            
            # Load frequency domains
            domain_ids = json.loads(state.get('domain_ids', '[]'))
            if domain_ids:
                state.frequency_domains = {}
                for domain_id in domain_ids:
                    domain = self.domain_repository.find_by_id(domain_id)
                    if domain:
                        state.frequency_domains[domain_id] = domain
            
            # Load boundaries
            boundary_ids = json.loads(state.get('boundary_ids', '[]'))
            if boundary_ids:
                state.boundaries = {}
                for boundary_id in boundary_ids:
                    boundary = self.boundary_repository.find_by_id(boundary_id)
                    if boundary:
                        state.boundaries[boundary_id] = boundary
            
            # Load resonance points
            resonance_point_ids = json.loads(state.get('resonance_point_ids', '[]'))
            if resonance_point_ids:
                state.resonance_points = {}
                for point_id in resonance_point_ids:
                    point = self.resonance_repository.find_by_id(point_id)
                    if point:
                        state.resonance_points[point_id] = point
            
            logger.info(f"Successfully loaded topology state {state_id} from ArangoDB")
            return state
            
        except Exception as e:
            logger.error(f"Error loading topology state from ArangoDB: {str(e)}")
            return None
    
    def find_latest_topology_state(self) -> Optional[TopologyState]:
        """
        Find the latest topology state in ArangoDB.
        
        Returns:
            The latest topology state if found, None otherwise
        """
        try:
            # Find the latest topology state
            state = self.topology_repository.find_latest()
            if not state:
                logger.warning("No topology states found in ArangoDB")
                return None
            
            # Load the full state
            return self.load_topology_state(state.id)
            
        except Exception as e:
            logger.error(f"Error finding latest topology state in ArangoDB: {str(e)}")
            return None
    
    def find_frequency_domains_by_coherence(self, threshold: float) -> List[FrequencyDomain]:
        """
        Find frequency domains with coherence above a threshold.
        
        Args:
            threshold: The coherence threshold
            
        Returns:
            A list of matching frequency domains
        """
        try:
            return self.domain_repository.find_by_coherence_threshold(threshold)
        except Exception as e:
            logger.error(f"Error finding frequency domains by coherence: {str(e)}")
            return []
    
    def find_boundaries_by_permeability(self, threshold: float) -> List[Boundary]:
        """
        Find boundaries with permeability above a threshold.
        
        Args:
            threshold: The permeability threshold
            
        Returns:
            A list of matching boundaries
        """
        try:
            return self.boundary_repository.find_by_permeability_threshold(threshold)
        except Exception as e:
            logger.error(f"Error finding boundaries by permeability: {str(e)}")
            return []
    
    def find_resonance_points_by_strength(self, threshold: float) -> List[ResonancePoint]:
        """
        Find resonance points with strength above a threshold.
        
        Args:
            threshold: The strength threshold
            
        Returns:
            A list of matching resonance points
        """
        try:
            return self.resonance_repository.find_by_strength_threshold(threshold)
        except Exception as e:
            logger.error(f"Error finding resonance points by strength: {str(e)}")
            return []
    
    def find_patterns_by_predicate(self, predicate: str) -> List[Pattern]:
        """
        Find patterns with a specific predicate.
        
        Args:
            predicate: The predicate to search for
            
        Returns:
            A list of matching patterns
        """
        try:
            return self.pattern_repository.find_by_predicate(predicate)
        except Exception as e:
            logger.error(f"Error finding patterns by predicate: {str(e)}")
            return []
