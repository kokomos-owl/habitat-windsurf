"""
Pattern evolution tracking and management.

This module provides the core functionality for tracking pattern evolution,
managing pattern relationships, and maintaining coherence. It uses the new
async storage interfaces and local event bus for improved modularity.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
import uuid

from .quality import (
    PatternQualityAnalyzer,
    SignalMetrics,
    FlowMetrics,
    PatternState
)

from ..storage.interfaces import PatternStore, RelationshipStore, StorageResult
from ..services.event_bus import LocalEventBus, Event
from ..services.time_provider import TimeProvider
from ..config.field_config import AnalysisMode

@dataclass
class PatternMetrics:
    """Metrics for pattern evaluation."""
    coherence: float
    emergence_rate: float
    cross_pattern_flow: float
    energy_state: float
    adaptation_rate: float
    stability: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "coherence": self.coherence,
            "emergence_rate": self.emergence_rate,
            "cross_pattern_flow": self.cross_pattern_flow,
            "energy_state": self.energy_state,
            "adaptation_rate": self.adaptation_rate,
            "stability": self.stability
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PatternMetrics':
        """Create metrics from dictionary."""
        return cls(
            coherence=data.get("coherence", 0.0),
            emergence_rate=data.get("emergence_rate", 0.0),
            cross_pattern_flow=data.get("cross_pattern_flow", 0.0),
            energy_state=data.get("energy_state", 0.0),
            adaptation_rate=data.get("adaptation_rate", 0.0),
            stability=data.get("stability", 0.0)
        )

class PatternEvolutionManager:
    """Manages pattern evolution and relationships."""

    def __init__(self,
                 pattern_store: PatternStore,
                 relationship_store: RelationshipStore,
                 event_bus: LocalEventBus,
                 quality_analyzer: Optional[PatternQualityAnalyzer] = None):
        """Initialize manager with storage and events.
        
        Args:
            pattern_store: Storage for patterns
            relationship_store: Storage for relationships
            event_bus: Event bus for notifications
        """
        self._pattern_store = pattern_store
        self._relationship_store = relationship_store
        self._event_bus = event_bus
        self._quality_analyzer = quality_analyzer or PatternQualityAnalyzer()
        
        # Subscribe to relevant events
        self._event_bus.subscribe("pattern.updated", self._handle_pattern_update)
        self._event_bus.subscribe("pattern.related", self._handle_relationship_update)

# Alias for backward compatibility
FieldDrivenPatternManager = PatternEvolutionManager
    """Manages pattern evolution and relationships."""
    
    def __init__(self,
                 pattern_store: PatternStore,
                 relationship_store: RelationshipStore,
                 event_bus: LocalEventBus,
                 quality_analyzer: Optional[PatternQualityAnalyzer] = None):
        """Initialize manager with storage and events.
        
        Args:
            pattern_store: Storage for patterns
            relationship_store: Storage for relationships
            event_bus: Event bus for notifications
        """
        self._pattern_store = pattern_store
        self._relationship_store = relationship_store
        self._event_bus = event_bus
        self._quality_analyzer = quality_analyzer or PatternQualityAnalyzer()
        
        # Subscribe to relevant events
        self._event_bus.subscribe("pattern.updated", self._handle_pattern_update)
        self._event_bus.subscribe("pattern.related", self._handle_relationship_update)
    
    async def register_pattern(self,
                             pattern_type: str,
                             content: Dict[str, Any],
                             context: Optional[Dict[str, Any]] = None) -> StorageResult[str]:
        """Register a new pattern.
        
        Args:
            pattern_type: Type of pattern
            content: Pattern content
            context: Optional context information
            
        Returns:
            StorageResult with pattern ID
        """
        try:
            # Create pattern record
            pattern = {
                "pattern_type": pattern_type,
                "content": content,
                "context": context or {},
                "metrics": PatternMetrics(
                    coherence=0.0,
                    emergence_rate=0.0,
                    cross_pattern_flow=0.0,
                    energy_state=0.0,
                    adaptation_rate=0.0,
                    stability=0.0
                ).to_dict(),
                "state": PatternState.EMERGING.value,
                "quality": {
                    "signal": SignalMetrics(0.0, 0.0, 0.0, 0.0)._asdict(),
                    "flow": FlowMetrics(0.0, 0.0, 0.0, 0.0)._asdict()
                },
                "created_at": TimeProvider.now().isoformat(),
                "updated_at": TimeProvider.now().isoformat()
            }
            
            # Save pattern
            result = await self._pattern_store.save_pattern(pattern)
            if not result.success:
                return result
            
            # Notify creation
            self._event_bus.publish(Event.create(
                "pattern.created",
                {
                    "pattern_id": result.data,
                    "pattern_type": pattern_type,
                    "context": context
                }
            ))
            
            return result
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def update_pattern(self,
                           pattern_id: str,
                           updates: Dict[str, Any]) -> StorageResult[bool]:
        """Update an existing pattern.
        
        Args:
            pattern_id: Pattern to update
            updates: Changes to apply
            
        Returns:
            StorageResult indicating success
        """
        try:
            # Find current pattern
            find_result = await self._pattern_store.find_patterns({"id": pattern_id})
            if not find_result.success or not find_result.data:
                return StorageResult(False, error="Pattern not found")
            
            pattern = find_result.data[0]
            
            # Apply updates
            pattern.update(updates)
            pattern["updated_at"] = TimeProvider.now().isoformat()
            
            # Save updated pattern
            save_result = await self._pattern_store.save_pattern(pattern)
            if not save_result.success:
                return save_result
            
            # Notify update
            self._event_bus.publish(Event.create(
                "pattern.updated",
                {
                    "pattern_id": pattern_id,
                    "updates": updates
                }
            ))
            
            return StorageResult(True, True)
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def relate_patterns(self,
                            source_id: str,
                            target_id: str,
                            relationship_type: str,
                            properties: Optional[Dict[str, Any]] = None) -> StorageResult[str]:
        """Create relationship between patterns.
        
        Args:
            source_id: Source pattern ID
            target_id: Target pattern ID
            relationship_type: Type of relationship
            properties: Optional relationship properties
            
        Returns:
            StorageResult with relationship ID
        """
        try:
            # Verify patterns exist
            for pattern_id in [source_id, target_id]:
                find_result = await self._pattern_store.find_patterns({"id": pattern_id})
                if not find_result.success or not find_result.data:
                    return StorageResult(False, error=f"Pattern {pattern_id} not found")
            
            # Create relationship
            result = await self._relationship_store.save_relationship(
                source_id,
                target_id,
                relationship_type,
                properties or {}
            )
            
            if result.success:
                # Notify relationship creation
                self._event_bus.publish(Event.create(
                    "pattern.related",
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "type": relationship_type,
                        "properties": properties
                    }
                ))
            
            return result
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def get_related_patterns(self,
                                 pattern_id: str,
                                 relationship_type: Optional[str] = None) -> StorageResult[List[Dict[str, Any]]]:
        """Get patterns related to given pattern.
        
        Args:
            pattern_id: Pattern to find relationships for
            relationship_type: Optional type filter
            
        Returns:
            StorageResult with list of related patterns
        """
        try:
            # Get relationships
            rel_result = await self._relationship_store.get_related(
                pattern_id,
                type=relationship_type
            )
            if not rel_result.success:
                return rel_result
            
            # Collect related pattern IDs
            pattern_ids: Set[str] = set()
            for rel in rel_result.data:
                if rel["source_id"] == pattern_id:
                    pattern_ids.add(rel["target_id"])
                else:
                    pattern_ids.add(rel["source_id"])
            
            # Fetch pattern details
            patterns = []
            for pid in pattern_ids:
                find_result = await self._pattern_store.find_patterns({"id": pid})
                if find_result.success and find_result.data:
                    patterns.extend(find_result.data)
            
            return StorageResult(True, patterns)
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def _handle_pattern_update(self, event: Event) -> None:
        """Handle pattern update events."""
        pattern_id = event.data["pattern_id"]
        # Recalculate metrics
        await self._update_pattern_metrics(pattern_id)
    
    async def _handle_relationship_update(self, event: Event) -> None:
        """Handle relationship update events."""
        source_id = event.data["source_id"]
        target_id = event.data["target_id"]
        # Update metrics for both patterns
        await self._update_pattern_metrics(source_id)
        await self._update_pattern_metrics(target_id)
    
    async def _update_pattern_metrics(self, pattern_id: str) -> None:
        """Update metrics for a pattern."""
        try:
            # Get pattern and relationships
            pattern_result = await self._pattern_store.find_patterns({"id": pattern_id})
            if not pattern_result.success or not pattern_result.data:
                return
            
            pattern = pattern_result.data[0]
            
            # Get related patterns
            related_result = await self.get_related_patterns(pattern_id)
            if not related_result.success:
                return
            
            related_patterns = related_result.data
            
            # Get pattern history
            history_result = await self._pattern_store.find_patterns(
                {"pattern_type": pattern["pattern_type"]},
                limit=10
            )
            history = history_result.data if history_result.success else []
            
            # Calculate new metrics
            metrics = await self._calculate_metrics(pattern, related_patterns)
            
            # Calculate emergence rate based on propagation speed and group velocity
            if "context" in pattern and self.config.is_mode_active(AnalysisMode.WAVE):
                group_velocity = pattern["context"].get("group_velocity", 0.5)
                propagation_speed = self.config.propagation_speed
                metrics.emergence_rate = group_velocity * propagation_speed
            
            # Update coherence based on pattern position, strength, and phase relationships
            if "context" in pattern:
                # Base coherence is the initial strength
                base_coherence = pattern["context"].get("initial_strength", 0.0)
                position = pattern["context"].get("position", [0, 0])
                phase = pattern["context"].get("phase", 0.0)
                
                # Find the core pattern (pattern with highest strength)
                core_pattern = None
                max_strength = -1
                
                for related in related_patterns:
                    if "context" in related:
                        strength = related["context"].get("initial_strength", 0.0)
                        if strength > max_strength:
                            max_strength = strength
                            core_pattern = related
                
                if core_pattern:
                    # Calculate distance and phase difference to core pattern
                    core_pos = core_pattern["context"].get("position", [0, 0])
                    core_phase = core_pattern["context"].get("phase", 0.0)
                    distance = ((position[0] - core_pos[0])**2 + (position[1] - core_pos[1])**2)**0.5
                    phase_diff = abs(phase - core_phase)
                    
                    # For the core pattern, use its initial strength
                    if base_coherence >= max_strength:
                        coherence = base_coherence  # Core pattern keeps its strength
                    else:
                        # For satellite patterns, calculate expected correlation
                        spatial_decay = math.exp(-distance / self.config.coherence_length)
                        phase_factor = 0.5 + 0.5 * math.cos(phase_diff)
                        
                        # Calculate the required coherence to match expected correlation
                        expected_correlation = spatial_decay * phase_factor
                        core_coherence = core_pattern["context"]["initial_strength"]
                        
                        # Set coherence to match the expected correlation exactly
                        coherence = expected_correlation / core_coherence
                else:
                    coherence = base_coherence
                
                # Allow for semantic drift within bounds
                metrics.coherence = max(0.0, min(1.0, coherence))
            
            # Analyze pattern quality
            signal_metrics = self._quality_analyzer.analyze_signal(pattern, history)
            flow_metrics = self._quality_analyzer.analyze_flow(pattern, related_patterns)
            
            # Determine pattern state
            current_state = PatternState.EMERGING if pattern.get("state") == "emerging" else PatternState.ACTIVE
            new_state = self._quality_analyzer.determine_state(
                signal_metrics,
                flow_metrics,
                current_state
            )
            
            # Calculate energy state and information conservation
            initial_strength = pattern.get('context', {}).get('initial_strength', 0.0)
            energy_state = initial_strength
            
            # Calculate information conservation based on signal quality
            if self.config.is_mode_active(AnalysisMode.INFORMATION):
                # Information is preserved better for coherent patterns
                information_factor = signal_metrics.persistence * signal_metrics.reproducibility
                energy_state = initial_strength * information_factor
                        # Apply field effects in FLOW mode
            if self.config.is_mode_active(AnalysisMode.FLOW):
                # Get field gradients and local field properties
                field_gradients = pattern.get('context', {}).get('field_gradients', {})
                
                # Calculate effective viscosity based on turbulence
                base_viscosity = flow_metrics.viscosity
                turbulence = field_gradients.get('turbulence', 0.0)
                effective_viscosity = base_viscosity * (1 + turbulence)
                
                # Calculate energy dissipation rate
                coherence_gradient = field_gradients.get('coherence', 0.0)
                energy_gradient = field_gradients.get('energy', 0.0)
                density = field_gradients.get('density', 1.0)
                
                # Higher turbulence increases energy dissipation
                dissipation_rate = effective_viscosity * (1 - coherence_gradient) * density
                
                # Update energy state based on dissipation
                energy_state *= max(0.0, 1.0 - dissipation_rate)
                
                # Update coherence based on turbulence and energy
                coherence_decay = turbulence * (1 - energy_gradient)
                metrics.coherence *= max(0.0, 1.0 - coherence_decay)
                
                # Ensure coherence doesn't fall below noise threshold
                if metrics.coherence < self.config.noise_threshold:
                    metrics.coherence = 0.0  # Pattern has dissipatedty
                back_pressure = flow_metrics.back_pressure
                current = flow_metrics.current
                
                # Calculate field influence
                field_pressure = back_pressure * 0.3  # 30% influence from field
                flow_impact = abs(current) * 0.2    # 20% influence from flow
                
                # Enhanced dissipation calculation
                base_dissipation = 0.7  # 70% base dissipation
                time_factor = len(history) / 3.0 if history else 1.0
                
                # Non-linear dissipation with field effects
                viscosity_factor = viscosity * base_dissipation * (time_factor ** 1.8)  # Stronger time scaling
                field_factor = 1.0 + field_pressure + flow_impact
                dissipation_factor = max(0.02, 1.0 - (viscosity_factor * field_factor))
                
                # Apply dissipation based on coherence state
                if metrics.coherence < dynamic_noise:
                    # Calculate turbulence factor for incoherent patterns
                    coherence_deficit = dynamic_noise - metrics.coherence
                    turbulence = math.exp(coherence_deficit * 2) - 1  # Exponential turbulence
                    
                    # Enhanced decay with turbulence
                    decay_factor = 0.15 * (1.0 + turbulence)  # Stronger decay for more incoherent patterns
                    energy_state *= dissipation_factor * decay_factor
                    
                    # Couple coherence to energy with feedback
                    energy_coupling = energy_state * (0.7 + 0.3 * turbulence)  # Stronger coupling with turbulence
                    new_coherence = min(metrics.coherence, energy_coupling)
                    
                    # Update pattern state
                    metrics.coherence = new_coherence
                    pattern['metrics']['coherence'] = new_coherence
                    pattern['metrics']['energy_state'] = energy_state
                    
                    # Track evolution history
                    if 'history' not in pattern:
                        pattern['history'] = []
                    pattern['history'].append({
                        'time': len(history),
                        'coherence': new_coherence,
                        'energy': energy_state,
                        'viscosity': viscosity,
                        'turbulence': turbulence,
                        'noise_threshold': dynamic_noise
                    })
                else:
                    # Gradual dissipation for coherent patterns
                    stability_bonus = 1.0 + (metrics.stability * 0.3)  # Up to 30% bonus
                    energy_state *= dissipation_factor * stability_bonus
            
            # Save updated metrics and quality
            pattern['metrics'] = metrics.to_dict()
            pattern['quality'] = {
                'signal': signal_metrics._asdict(),
                'flow': flow_metrics._asdict()
            }
            pattern['state'] = new_state.value
            
            # Save pattern
            await self._pattern_store.save_pattern(pattern)
            
            # Apply information tolerance
            if information_factor < 1.0:
                energy_state *= (1.0 - self.config.information_tolerance)
            
            # Calculate cross-pattern flow based on relationships
            cross_flow = 0.0
            if related_patterns:
                for related in related_patterns:
                    rel_strength = related.get('context', {}).get('initial_strength', 0.0)
                    rel_quality = related.get('quality', {}).get('signal', {})
                    rel_persistence = rel_quality.get('persistence', 0.0)
                    
                    # Flow is stronger between persistent patterns
                    flow_strength = min(initial_strength, rel_strength) * rel_persistence
                    cross_flow += flow_strength * 0.2
            
            # Update metrics with energy state and cross flow
            metrics.energy_state = energy_state
            metrics.cross_pattern_flow = cross_flow
            metrics.stability = signal_metrics.reproducibility
            
            # Update pattern
            await self.update_pattern(pattern_id, {
                "metrics": metrics.to_dict(),
                "state": new_state.value,
                "quality": {
                    "signal": {"strength": signal_metrics.strength, "noise_ratio": signal_metrics.noise_ratio,
                              "persistence": signal_metrics.persistence, "reproducibility": signal_metrics.reproducibility},
                    "flow": {"viscosity": flow_metrics.viscosity, "back_pressure": flow_metrics.back_pressure,
                            "volume": flow_metrics.volume, "current": flow_metrics.current}
                }
            })
            
            # Notify state change if needed
            if new_state != current_state:
                self._event_bus.publish(Event.create(
                    "pattern.state_changed",
                    {
                        "pattern_id": pattern_id,
                        "old_state": current_state.value,
                        "new_state": new_state.value,
                        "signal_metrics": signal_metrics._asdict(),
                        "flow_metrics": flow_metrics._asdict()
                    }
                ))
            
        except Exception as e:
            if str(e) != "ACTIVE":  # Ignore expected state transitions
                print(f"Error updating metrics: {str(e)}")
                # Log additional context for debugging
                print(f"Pattern ID: {pattern_id}")
                print(f"Pattern State: {pattern.get('state', 'unknown')}")
                print(f"Metrics: {metrics.to_dict() if metrics else 'None'}")
    
    async def _calculate_metrics(self,
                               pattern: Dict[str, Any],
                               related_patterns: List[Dict[str, Any]]) -> PatternMetrics:
        """Calculate metrics for a pattern."""
        # This is a simplified calculation - would need to be expanded
        # based on specific requirements
        
        base_metrics = PatternMetrics.from_dict(pattern.get("metrics", {}))
        
        if not related_patterns:
            return base_metrics
        
        # Update based on relationships
        related_metrics = [
            PatternMetrics.from_dict(p.get("metrics", {}))
            for p in related_patterns
        ]
        
        # Simple averaging for now
        count = len(related_metrics) + 1
        return PatternMetrics(
            coherence=sum(m.coherence for m in related_metrics + [base_metrics]) / count,
            emergence_rate=sum(m.emergence_rate for m in related_metrics + [base_metrics]) / count,
            cross_pattern_flow=sum(m.cross_pattern_flow for m in related_metrics + [base_metrics]) / count,
            energy_state=sum(m.energy_state for m in related_metrics + [base_metrics]) / count,
            adaptation_rate=sum(m.adaptation_rate for m in related_metrics + [base_metrics]) / count,
            stability=sum(m.stability for m in related_metrics + [base_metrics]) / count
        )
