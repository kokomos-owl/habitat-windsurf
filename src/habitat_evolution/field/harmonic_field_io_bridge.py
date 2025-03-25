"""
Harmonic Field I/O Bridge

This bridge connects field components with the harmonic I/O service,
ensuring that eigenspace stability information from field components
is used to guide I/O operations.

The bridge implements the observer pattern to receive updates from
field components and propagate relevant metrics to the I/O service.
"""

from typing import Dict, List, Any, Optional
from ..adaptive_core.io.harmonic_io_service import HarmonicIOService


class HarmonicFieldIOBridge:
    """
    Bridge between field components and harmonic I/O.
    
    This bridge ensures that eigenspace stability information
    from field components is used to guide I/O operations.
    It observes field state changes and updates the harmonic
    I/O service with relevant metrics.
    """
    
    def __init__(self, io_service: HarmonicIOService):
        """
        Initialize the harmonic field I/O bridge.
        
        Args:
            io_service: Harmonic I/O service to update with field metrics
        """
        self.io_service = io_service
        self.field_metrics = {}
        self.observations = []
        
    def observe_field_state(self, field_state: Dict[str, Any]):
        """
        Observe field state changes and update I/O service.
        
        This method is called by field components when their state
        changes, allowing the bridge to extract relevant metrics
        and update the harmonic I/O service.
        
        Args:
            field_state: Current field state including eigenspace information
        """
        # Record observation
        self.observations.append({
            "context": field_state,
            "time": field_state.get("timestamp", None)
        })
        
        # Extract stability metrics from field state
        self._extract_eigenspace_metrics(field_state)
        
        # Update I/O service with current metrics
        self._update_io_service()
        
    def _extract_eigenspace_metrics(self, field_state: Dict[str, Any]):
        """
        Extract eigenspace metrics from field state.
        
        Args:
            field_state: Field state dictionary
        """
        # Extract eigenspace stability if available
        if "eigenspace" in field_state:
            eigenspace = field_state["eigenspace"]
            if "stability" in eigenspace:
                self.field_metrics["eigenspace_stability"] = eigenspace["stability"]
            if "coherence" in eigenspace:
                self.field_metrics["eigenspace_coherence"] = eigenspace["coherence"]
            if "uncertainty" in eigenspace:
                self.field_metrics["eigenspace_uncertainty"] = eigenspace["uncertainty"]
                
        # Extract tonic-harmonic metrics if available
        if "tonic_harmonic" in field_state:
            th = field_state["tonic_harmonic"]
            if "tonic" in th:
                self.field_metrics["tonic"] = th["tonic"]
            if "harmonic_stability" in th:
                self.field_metrics["harmonic_stability"] = th["harmonic_stability"]
                
        # Extract boundary metrics if available
        if "boundaries" in field_state:
            boundaries = field_state["boundaries"]
            if "fuzzy_boundary_strength" in boundaries:
                self.field_metrics["boundary_strength"] = boundaries["fuzzy_boundary_strength"]
            if "transition_zone_width" in boundaries:
                self.field_metrics["transition_width"] = boundaries["transition_zone_width"]
                
        # Extract pattern metrics if available
        if "patterns" in field_state:
            patterns = field_state["patterns"]
            if "coherence" in patterns:
                self.field_metrics["pattern_coherence"] = patterns["coherence"]
            if "resonance" in patterns:
                self.field_metrics["pattern_resonance"] = patterns["resonance"]
                
    def _update_io_service(self):
        """Update the I/O service with current field metrics."""
        # Update eigenspace stability
        if "eigenspace_stability" in self.field_metrics:
            self.io_service.update_eigenspace_stability(
                self.field_metrics["eigenspace_stability"]
            )
            
        # Update pattern coherence
        if "pattern_coherence" in self.field_metrics:
            self.io_service.update_pattern_coherence(
                self.field_metrics["pattern_coherence"]
            )
            
        # Update resonance level
        if "pattern_resonance" in self.field_metrics:
            self.io_service.update_resonance_level(
                self.field_metrics["pattern_resonance"]
            )
            
    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Get current field metrics.
        
        Returns:
            Dictionary of field metrics
        """
        return self.field_metrics.copy()
        
    def register_with_field_navigator(self, field_navigator):
        """
        Register this bridge with a field navigator.
        
        Args:
            field_navigator: Field navigator to register with
        """
        field_navigator.register_observer(self)
        
    def register_with_semantic_boundary_detector(self, boundary_detector):
        """
        Register this bridge with a semantic boundary detector.
        
        Args:
            boundary_detector: Semantic boundary detector to register with
        """
        boundary_detector.register_observer(self)
        
    def register_with_field_adaptive_id_bridge(self, field_adaptive_id_bridge):
        """
        Register this bridge with a field adaptive ID bridge.
        
        Args:
            field_adaptive_id_bridge: Field adaptive ID bridge to register with
        """
        field_adaptive_id_bridge.register_observer(self)
