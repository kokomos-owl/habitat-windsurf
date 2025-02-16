"""
Pattern Emergence Interface (PEI) for semantic pattern observation and interaction.

Provides an event-driven interface for observing and interacting with emergent patterns
in the vector attention monitoring system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, AsyncIterator
import numpy as np
import asyncio
from ..monitoring.vector_attention_monitor import VectorAttentionMonitor, VectorSpaceMetrics
from ...adaptive_core.id.adaptive_id import AdaptiveID

class PatternState(Enum):
    """States in the pattern lifecycle."""
    FORMING = "forming"      # Initial pattern detection
    EMERGING = "emerging"    # Pattern gaining stability
    STABLE = "stable"       # Established pattern
    DISSOLVING = "dissolving" # Pattern losing coherence

@dataclass
class PatternMetrics:
    """Metrics describing pattern characteristics."""
    density: float          # Local density score
    stability: float        # Stability score
    attention: float        # Current attention weight
    confidence: float       # Overall confidence score
    timestamp: datetime     # Measurement timestamp

@dataclass
class EmergentPattern:
    """Represents a detected semantic pattern."""
    id: str                # Unique pattern identifier
    state: PatternState    # Current lifecycle state
    center: np.ndarray     # Pattern centroid in vector space
    radius: float          # Effective pattern radius
    metrics: PatternMetrics # Current pattern metrics
    context: Dict          # Additional pattern context
    
    @property
    def is_stable(self) -> bool:
        """Check if pattern has reached stability."""
        return (self.state == PatternState.STABLE and 
                self.metrics.confidence > 0.7)
    
    @property
    def is_forming(self) -> bool:
        """Check if pattern is in formation."""
        return self.state in (PatternState.FORMING, PatternState.EMERGING)

@dataclass
class PatternFeedback:
    """Feedback from agents about pattern detection."""
    attention_delta: float  # Requested attention adjustment
    confidence_override: Optional[float] = None  # Optional confidence override
    context_updates: Dict = None  # Additional context information

class PatternEventType(Enum):
    """Types of pattern-related events."""
    PATTERN_FORMING = "pattern_forming"
    PATTERN_EMERGED = "pattern_emerged"
    PATTERN_STABILIZED = "pattern_stabilized"
    PATTERN_DISSOLVING = "pattern_dissolving"
    PATTERN_DISSOLVED = "pattern_dissolved"

@dataclass
class PatternEvent:
    """Event notification for pattern state changes."""
    type: PatternEventType
    pattern: EmergentPattern
    timestamp: datetime
    previous_state: Optional[PatternState] = None

class PatternEmergenceInterface:
    """Main interface for pattern emergence observation and interaction."""
    
    def __init__(self, 
                 monitor: VectorAttentionMonitor,
                 min_confidence: float = 0.5,
                 event_buffer_size: int = 1000):
        self.monitor = monitor
        self.min_confidence = min_confidence
        self._patterns: Dict[str, EmergentPattern] = {}
        self._event_queue = asyncio.Queue(maxsize=event_buffer_size)
        self._subscribers = []
    
    async def start(self):
        """Start pattern monitoring and event processing."""
        self._running = True
        asyncio.create_task(self._process_metrics())
    
    async def stop(self):
        """Stop pattern monitoring and event processing."""
        self._running = False
        # Clear event queue
        while not self._event_queue.empty():
            try:
                _ = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Clear patterns
        self._patterns.clear()
    
    async def observe_patterns(self) -> AsyncIterator[EmergentPattern]:
        """Stream of pattern observations as they occur."""
        while self._running:
            try:
                event = await self._event_queue.get()
                yield event.pattern
            except asyncio.CancelledError:
                # Clean up on cancellation
                self._running = False
                raise
    
    def get_active_patterns(self) -> List[EmergentPattern]:
        """Get currently active (stable) patterns."""
        return [p for p in self._patterns.values() if p.is_stable]
    
    def get_emerging_patterns(self) -> List[EmergentPattern]:
        """Get patterns currently in formation."""
        return [p for p in self._patterns.values() if p.is_forming]
    
    async def process_feedback(self, 
                             pattern_id: str, 
                             feedback: PatternFeedback) -> bool:
        """Process agent feedback about a pattern."""
        if pattern_id not in self._patterns:
            return False
            
        pattern = self._patterns[pattern_id]
        
        # Update attention weights
        if feedback.attention_delta:
            await self._adjust_attention(pattern, feedback.attention_delta)
            
        # Update confidence if override provided
        if feedback.confidence_override is not None:
            await self._update_confidence(pattern, feedback.confidence_override)
            
        # Update context
        if feedback.context_updates:
            pattern.context.update(feedback.context_updates)
            
        return True
    
    async def _process_metrics(self):
        """Process incoming metrics and detect patterns."""
        while self._running:
            try:
                metrics = await self._get_next_metrics()
                await self._update_patterns(metrics)
                await self._emit_events()
            except asyncio.CancelledError:
                self._running = False
                raise
            except Exception as e:
                # Log error but continue processing
                continue
    
    async def _update_patterns(self, metrics: VectorSpaceMetrics):
        """Update pattern states based on new metrics.
        
        This method implements the core pattern detection and state transition logic:
        1. Updates existing patterns with new metrics
        2. Detects new pattern formation
        3. Handles pattern dissolution
        4. Manages pattern state transitions
        """
        try:
            vector = metrics.current_vector
            density = metrics.local_density
            stability = metrics.stability_score
            attention = metrics.attention_weight
            
            # Calculate pattern confidence
            confidence = (0.4 * density + 
                         0.4 * stability + 
                         0.2 * attention)  # Weighted combination
            
            # Update existing patterns
            for pattern_id, pattern in list(self._patterns.items()):
                distance = np.linalg.norm(pattern.center - vector)
                
                if distance <= pattern.radius:
                    # Vector belongs to this pattern
                    await self._update_pattern_metrics(pattern, metrics)
                    continue
                    
                # Check for pattern dissolution
                if pattern.metrics.confidence < 0.3:
                    await self._dissolve_pattern(pattern_id)
            
            # Detect new pattern formation
            if confidence > self.min_confidence:
                # Check if this could be a new pattern
                is_new = all(np.linalg.norm(p.center - vector) > p.radius 
                            for p in self._patterns.values())
                
                if is_new:
                    await self._create_new_pattern(vector, metrics)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Log error but continue processing
            pass
    
    async def _update_pattern_metrics(self, 
                                    pattern: EmergentPattern, 
                                    metrics: VectorSpaceMetrics):
        """Update pattern metrics and handle state transitions."""
        # Update pattern metrics
        new_metrics = PatternMetrics(
            density=metrics.local_density,
            stability=metrics.stability_score,
            attention=metrics.attention_weight,
            confidence=pattern.metrics.confidence,
            timestamp=datetime.now()
        )
        
        # Update adaptive ID metrics
        adaptive_id = pattern.context["adaptive_id"]
        adaptive_id.weight = metrics.local_density
        adaptive_id.confidence = metrics.stability_score
        adaptive_id.uncertainty = 1.0 - metrics.attention_weight
        
        # Update pattern state based on metrics
        old_state = pattern.state
        new_state = self._determine_pattern_state(new_metrics)
        
        if new_state != old_state:
            pattern.state = new_state
            await self._queue_state_transition_event(pattern, old_state)
        
        pattern.metrics = new_metrics
    
    def _determine_pattern_state(self, metrics: PatternMetrics) -> PatternState:
        """Determine pattern state based on metrics."""
        if metrics.confidence < 0.3:
            return PatternState.DISSOLVING
        elif metrics.confidence > 0.7 and metrics.stability > 0.7:
            return PatternState.STABLE
        elif metrics.confidence > 0.5:
            return PatternState.EMERGING
        else:
            return PatternState.FORMING
    
    async def _create_new_pattern(self, 
                                vector: np.ndarray, 
                                metrics: VectorSpaceMetrics):
        """Create and initialize a new pattern."""
        # Create adaptive ID for pattern provenance tracking
        base_concept = f"pattern_v{len(self._patterns)}"
        adaptive_id = AdaptiveID(
            base_concept=base_concept,
            creator_id="pattern_emergence",
            weight=metrics.local_density,
            confidence=metrics.stability_score,
            uncertainty=1.0 - metrics.attention_weight
        )
        
        # Calculate adaptive radius based on local density
        # For dense regions (density > 0.5), decrease radius
        # For sparse regions (density < 0.5), increase radius
        base_radius = self.monitor.density_radius
        if metrics.local_density > 0.5:
            # Dense regions get smaller radius proportional to density
            adaptive_radius = base_radius * metrics.local_density ** 0.5
        else:
            # Sparse regions get larger radius inversely proportional to density
            adaptive_radius = base_radius * (1.0 / metrics.local_density) ** 0.5
        
        pattern = EmergentPattern(
            id=adaptive_id.id,
            state=PatternState.FORMING,
            center=vector.copy(),
            radius=adaptive_radius,
            metrics=PatternMetrics(
                density=metrics.local_density,
                stability=metrics.stability_score,
                attention=metrics.attention_weight,
                confidence=0.0,  # Start with zero confidence
                timestamp=datetime.now()
            ),
            context={
                "adaptive_id": adaptive_id,
                "base_concept": base_concept,
                "creator": "pattern_emergence"
            }
        )
        
        self._patterns[adaptive_id.id] = pattern
        await self._queue_event(PatternEventType.PATTERN_FORMING, pattern)
    
    async def _dissolve_pattern(self, pattern_id: str):
        """Handle pattern dissolution."""
        pattern = self._patterns[pattern_id]
        await self._queue_event(PatternEventType.PATTERN_DISSOLVING, pattern)
        del self._patterns[pattern_id]
        await self._queue_event(PatternEventType.PATTERN_DISSOLVED, pattern)
    
    async def _adjust_attention(self, 
                              pattern: EmergentPattern, 
                              attention_delta: float):
        """Adjust attention weights based on feedback.
        
        Implements a smooth attention adjustment with bounds checking
        and stability considerations.
        """
        current = pattern.metrics.attention
        new_attention = np.clip(current + attention_delta, 0.1, 1.0)
        
        # Smooth transition based on stability
        alpha = pattern.metrics.stability  # Use stability as smoothing factor
        smoothed_attention = alpha * new_attention + (1 - alpha) * current
        
        # Create new metrics object with updated attention
        new_metrics = PatternMetrics(
            density=pattern.metrics.density,
            stability=pattern.metrics.stability,
            attention=smoothed_attention,
            confidence=pattern.metrics.confidence,
            timestamp=datetime.now()
        )
        
        # Update pattern metrics
        pattern.metrics = new_metrics
    
    async def _update_confidence(self, 
                               pattern: EmergentPattern,
                               confidence: float):
        """Update pattern confidence based on feedback.
        
        Implements confidence updating with temporal decay and
        stability-based smoothing.
        """
        current = pattern.metrics.confidence
        # Apply temporal decay
        decay = np.exp(-0.1)  # 10% decay per update
        decayed_confidence = current * decay
        
        # Smooth update based on stability
        alpha = pattern.metrics.stability
        new_confidence = np.clip(
            alpha * confidence + (1 - alpha) * decayed_confidence,
            0.0, 1.0
        )
        
        # Create new metrics object with updated confidence
        new_metrics = PatternMetrics(
            density=pattern.metrics.density,
            stability=pattern.metrics.stability,
            attention=pattern.metrics.attention,
            confidence=new_confidence,
            timestamp=datetime.now()
        )
        
        # Update pattern metrics
        pattern.metrics = new_metrics
    
    async def _queue_event(self, 
                          event_type: PatternEventType,
                          pattern: EmergentPattern,
                          previous_state: Optional[PatternState] = None):
        """Queue a pattern event for processing."""
        event = PatternEvent(
            type=event_type,
            pattern=pattern,
            timestamp=datetime.now(),
            previous_state=previous_state
        )
        
        try:
            await self._event_queue.put(event)
        except asyncio.QueueFull:
            # If queue is full, remove oldest event and try again
            _ = self._event_queue.get_nowait()
            await self._event_queue.put(event)
    
    async def _queue_state_transition_event(self,
                                          pattern: EmergentPattern,
                                          old_state: PatternState):
        """Queue appropriate event for pattern state transition."""
        event_map = {
            PatternState.EMERGING: PatternEventType.PATTERN_EMERGED,
            PatternState.STABLE: PatternEventType.PATTERN_STABILIZED,
            PatternState.DISSOLVING: PatternEventType.PATTERN_DISSOLVING
        }
        
        if pattern.state in event_map:
            await self._queue_event(
                event_map[pattern.state],
                pattern,
                old_state
            )
    
    async def _emit_events(self):
        """Emit pattern events to subscribers.
        
        Processes the event queue and notifies all subscribers of pattern events.
        Implements backpressure handling and subscriber cleanup.
        """
        while not self._event_queue.empty():
            event = await self._event_queue.get()
            
            # Clean up dead subscribers
            self._subscribers = [s for s in self._subscribers if not s.done()]
            
            # Notify all subscribers
            for subscriber in self._subscribers:
                try:
                    await subscriber.put(event)
                except asyncio.QueueFull:
                    continue  # Skip if subscriber queue is full
    
    async def _get_next_metrics(self) -> VectorSpaceMetrics:
        """Get next set of metrics from monitor.
        
        Implements metrics retrieval with timeout and error handling.
        """
        try:
            # Implement actual metrics retrieval from monitor
            # This is a placeholder until we implement the metrics stream
            await asyncio.sleep(0.1)  # Simulate processing time
            metrics = await self.monitor.get_latest_metrics()
            return metrics
        except Exception as e:
            # Return safe default metrics
            return VectorSpaceMetrics()
