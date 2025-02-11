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
                        # For satellite patterns, start with initial coherence
                        # Then apply spatial and phase decay
                        spatial_decay = math.exp(-distance / self.config.coherence_length)
                        phase_factor = 0.5 + 0.5 * math.cos(phase_diff)
                        
                        # Combine initial coherence with decay factors
                        coherence = base_coherence * spatial_decay * phase_factor
                else:
                    coherence = base_coherence
                
                # Allow for semantic drift within bounds
                metrics.coherence = max(0.0, min(1.0, coherence))
            
            # Analyze pattern quality
            signal_metrics = self._quality_analyzer.analyze_signal(pattern, history)
            flow_metrics = self._quality_analyzer.analyze_flow(pattern, related_patterns)
            
            # Determine pattern state
            current_state = PatternState.EMERGING if pattern.get("state") == "EMERGING" else PatternState.ACTIVE
            new_state = self._quality_analyzer.determine_state(
                signal_metrics,
                flow_metrics,
                current_state
            )
            
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
            print(f"Error updating metrics: {e}")
    
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
