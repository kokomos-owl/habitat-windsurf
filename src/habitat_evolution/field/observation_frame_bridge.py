"""
Observation Frame Bridge

This module provides the ObservationFrameBridge class, which integrates observation frames
with the tonic-harmonic field architecture. It allows multiple perspectives on the same
underlying phenomena to contribute to field topology without imposing artificial domain
boundaries.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
import logging
from datetime import datetime
import uuid

from .field_state import TonicHarmonicFieldState
from .topological_field_analyzer import TopologicalFieldAnalyzer

logger = logging.getLogger(__name__)

class ObservationFrameBridge:
    """
    Bridges observation frames with the tonic-harmonic field architecture.
    
    This class allows multiple perspectives (ecological, indigenous, socioeconomic, etc.)
    to contribute to the field topology without imposing artificial domain boundaries.
    It enables the natural emergence of patterns across perspectives.
    """
    
    def __init__(
        self, 
        field_state: TonicHarmonicFieldState,
        field_analyzer: TopologicalFieldAnalyzer,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an observation frame bridge.
        
        Args:
            field_state: The tonic-harmonic field state to update
            field_analyzer: The topological field analyzer to use
            config: Optional configuration parameters
        """
        self.field_state = field_state
        self.field_analyzer = field_analyzer
        
        # Default configuration
        default_config = {
            "frame_influence_weight": 0.7,  # Weight of frame-specific observations
            "cross_frame_weight": 0.9,      # Weight of observations that cross frames
            "resonance_threshold": 0.65,    # Threshold for resonance detection
            "dimensionality_boost": 1.2     # Boost for effective dimensionality from multiple frames
        }
        
        # Update with provided config
        self.config = default_config
        if config:
            self.config.update(config)
            
        # Track registered observation frames
        self.observation_frames = {}
        
        # Track cross-frame relationships
        self.cross_frame_relationships = []
        
    def register_observation_frame(self, frame_data: Dict[str, Any]) -> str:
        """
        Register an observation frame with the bridge.
        
        Args:
            frame_data: Data describing the observation frame
            
        Returns:
            The ID of the registered frame
        """
        frame_id = frame_data.get("id", str(uuid.uuid4()))
        
        # Store the frame
        self.observation_frames[frame_id] = {
            "name": frame_data.get("name", f"frame_{frame_id}"),
            "description": frame_data.get("description", ""),
            "resonance_centers": frame_data.get("resonance_centers", []),
            "field_properties": frame_data.get("field_properties", {})
        }
        
        # Update field state with frame information
        self._update_field_with_frame(frame_id)
        
        logger.info(f"Registered observation frame: {self.observation_frames[frame_id]['name']}")
        return frame_id
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an observation through the appropriate frame(s).
        
        Args:
            observation: The observation data
            
        Returns:
            Processed observation with field-aware properties
        """
        # Extract the perspective (frame) from the observation
        perspective = observation.get("context", {}).get("perspective")
        
        # Find the corresponding frame
        frame = None
        for frame_id, frame_data in self.observation_frames.items():
            if frame_data["name"] == perspective:
                frame = frame_data
                break
        
        if not frame:
            logger.warning(f"No matching frame found for perspective: {perspective}")
            # Use default processing
            return self._default_observation_processing(observation)
        
        # Process the observation through the frame
        processed_observation = self._process_through_frame(observation, frame)
        
        # Check for cross-frame relationships
        self._check_cross_frame_relationships(processed_observation)
        
        # Update field state with the observation
        self._update_field_with_observation(processed_observation)
        
        return processed_observation
    
    def _process_through_frame(self, observation: Dict[str, Any], frame: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an observation through a specific frame.
        
        Args:
            observation: The observation data
            frame: The frame to process through
            
        Returns:
            Processed observation with frame-specific properties
        """
        # Create a copy of the observation to avoid modifying the original
        processed = observation.copy()
        
        # Add frame-specific field properties
        if "context" not in processed:
            processed["context"] = {}
            
        # Enhance with frame resonance centers if applicable
        source = observation.get("source", "")
        target = observation.get("target", "")
        
        # Check if source or target are resonance centers in this frame
        resonance_centers = frame.get("resonance_centers", [])
        
        # Calculate resonance influence
        resonance_influence = 0.0
        if source in resonance_centers:
            resonance_influence += 0.5
        if target in resonance_centers:
            resonance_influence += 0.5
            
        # Add field-aware properties
        processed["context"]["field_properties"] = {
            "frame": frame.get("name", ""),
            "resonance_influence": resonance_influence,
            "dimensional_alignment": self._calculate_dimensional_alignment(observation, frame),
            "frame_coherence": frame.get("field_properties", {}).get("coherence", 0.5)
        }
        
        return processed
    
    def _calculate_dimensional_alignment(self, observation: Dict[str, Any], frame: Dict[str, Any]) -> float:
        """
        Calculate how well an observation aligns with the frame's dimensional structure.
        
        Args:
            observation: The observation data
            frame: The frame to calculate alignment with
            
        Returns:
            Alignment score between 0 and 1
        """
        # Extract vector properties from observation
        vector_props = observation.get("context", {}).get("vector_properties", {})
        if not vector_props:
            return 0.5  # Default alignment
            
        # Extract dimensional structure from frame
        frame_dimensions = frame.get("field_properties", {}).get("dimensionality", 3)
        
        # Simple alignment calculation based on vector properties and frame dimensions
        # In a real implementation, this would use eigenspace projection
        vector_dim = len(vector_props)
        
        # Calculate alignment based on dimensional compatibility
        if vector_dim == 0:
            return 0.5
        
        # Higher alignment when dimensions match or exceed frame dimensions
        return min(1.0, vector_dim / frame_dimensions)
    
    def _check_cross_frame_relationships(self, observation: Dict[str, Any]) -> None:
        """
        Check if an observation creates relationships across frames.
        
        Args:
            observation: The processed observation
        """
        # This would identify when entities from different frames are related
        # For simplicity, we're just logging it
        source = observation.get("source", "")
        target = observation.get("target", "")
        perspective = observation.get("context", {}).get("perspective", "")
        
        # In a real implementation, this would track entities across frames
        # and identify when they form cross-frame relationships
        logger.debug(f"Checking cross-frame relationships for {source} -> {target} in {perspective}")
    
    def _update_field_with_frame(self, frame_id: str) -> None:
        """
        Update the field state with frame information.
        
        Args:
            frame_id: ID of the frame to incorporate
        """
        frame = self.observation_frames[frame_id]
        
        # Extract field properties from the frame
        field_props = frame.get("field_properties", {})
        
        # Create field analysis update with proper structure
        field_analysis_update = {
            "topology": {
                "effective_dimensionality": field_props.get("dimensionality", 3),
                "principal_dimensions": [i for i in range(field_props.get("dimensionality", 3))],
                "eigenvalues": np.array([0.8, 0.6, 0.4][:field_props.get("dimensionality", 3)]),
                "eigenvectors": np.eye(field_props.get("dimensionality", 3)),
                "resonance_centers": frame.get("resonance_centers", []),
                "flow_vectors": field_props.get("flow_vectors", [])
            },
            "density": {
                "density_centers": field_props.get("density_centers", []),
                "density_map": np.zeros((field_props.get("dimensionality", 3), field_props.get("dimensionality", 3)))
            },
            "field_properties": {
                "coherence": field_props.get("coherence", 0.6),
                "navigability_score": field_props.get("navigability_score", 0.65),
                "stability": field_props.get("stability", 0.7)
            },
            "patterns": {},
            "observation_frame": frame["name"]
        }
        
        # Update field state using the correct interface
        self.field_state.update_from_field_analysis(field_analysis_update)
        
        logger.debug(f"Updated field state with frame: {frame['name']}")
    
    def _update_field_with_observation(self, observation: Dict[str, Any]) -> None:
        """
        Update the field state with an observation.
        
        Args:
            observation: The processed observation
        """
        # Extract relevant information from the observation
        source = observation.get("source", "")
        target = observation.get("target", "")
        predicate = observation.get("predicate", "")
        context = observation.get("context", {})
        
        # Create a pattern entry for this observation
        pattern_id = f"pattern_{uuid.uuid4()}"
        
        # Extract vector properties or use defaults
        vector_props = context.get("vector_properties", {})
        position = [vector_props.get("x", 0.5), vector_props.get("y", 0.5), vector_props.get("z", 0.5)]
        
        # Create field analysis update with the new pattern
        field_analysis_update = {
            "patterns": {
                pattern_id: {
                    "source": source,
                    "predicate": predicate,
                    "target": target,
                    "position": position,
                    "confidence": context.get("confidence", 0.8),
                    "perspective": context.get("perspective", "default"),
                    "tonic_value": context.get("tonic_value", 0.5),
                    "timestamp": observation.get("timestamp", datetime.now().isoformat())
                }
            }
        }
        
        # Update field state with the new pattern
        self.field_state.update_patterns(field_analysis_update["patterns"])
        
        logger.debug(f"Updated field with observation: {observation.get('id', 'unknown')}")
    
    def _default_observation_processing(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an observation without a specific frame.
        
        Args:
            observation: The observation data
            
        Returns:
            Processed observation with default field properties
        """
        # Create a copy of the observation to avoid modifying the original
        processed = observation.copy()
        
        # Add default field properties
        if "context" not in processed:
            processed["context"] = {}
            
        processed["context"]["field_properties"] = {
            "frame": "default",
            "resonance_influence": 0.3,
            "dimensional_alignment": 0.5,
            "frame_coherence": 0.5
        }
        
        return processed
    
    def get_field_topology(self) -> Dict[str, Any]:
        """
        Get the current field topology influenced by all observation frames.
        
        Returns:
            The current field topology
        """
        # Extract topology information from field state
        topology = {
            "effective_dimensionality": self.field_state.effective_dimensionality,
            "principal_dimensions": self.field_state.principal_dimensions,
            "eigenvalues": self.field_state.eigenvalues,
            "resonance_centers": [],  # Would be extracted from field state
            "flow_vectors": []        # Would be extracted from field state
        }
        
        # In a full implementation, we would extract more detailed topology information
        return topology
    
    def get_cross_frame_resonance(self) -> List[Dict[str, Any]]:
        """
        Get resonance patterns that span multiple observation frames.
        
        Returns:
            List of cross-frame resonance patterns
        """
        # In a real implementation, this would analyze the field state
        # to identify resonance patterns that span multiple frames
        # For now, we're just returning a placeholder
        return []
