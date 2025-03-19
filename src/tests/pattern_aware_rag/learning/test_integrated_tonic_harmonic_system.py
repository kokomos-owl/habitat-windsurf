"""
Integrated System Tests for Vector + Tonic-Harmonic Pattern Evolution.

These tests validate the complete integration of:
1. AdaptiveID-LearningWindow-PatternID bidirectional communication
2. Field-aware pattern detection with tonic-harmonic analysis
3. Pattern evolution tracking through learning windows
4. Natural boundary detection and resonance pattern identification
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import modules using relative imports
sys.path.append(os.path.join(src_path, 'src'))
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow, WindowState
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

# Import mock classes from other test files
from test_pattern_id_integration import PatternID, setup_adaptive_id_pattern_integration, setup_learning_window_pattern_integration
from test_field_harmonic_analysis import FieldObserver, ResonancePatternDetector, setup_learning_window_tonic_harmonic

# Mock Neo4j Bridge
class FieldNeo4jBridge:
    """Mock Neo4j Bridge for field state persistence."""
    
    def __init__(self, neo4j_client=None):
        """Initialize with optional Neo4j client."""
        self.neo4j_client = neo4j_client or MagicMock()
        self.stored_windows = []
        self.stored_patterns = []
    
    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Observe learning window state changes and persist to Neo4j."""
        # Store context for later retrieval
        if "window_state" in context:
            self.stored_windows.append(context)
            
            # Store in Neo4j
            query = """
            CREATE (w:LearningWindow {
                entity_id: $entity_id,
                window_state: $window_state,
                stability: $stability,
                timestamp: $timestamp
            })
            """
            params = {
                "entity_id": context.get("entity_id", "unknown"),
                "window_state": context.get("window_state", "unknown"),
                "stability": context.get("stability", 0.0),
                "timestamp": context.get("timestamp", datetime.now().isoformat())
            }
            
            self.neo4j_client.execute_query(query, params)
        
        return {"status": "persisted", "context": context}
    
    def store_learning_window_state(self, learning_window: LearningWindow) -> None:
        """Store learning window state in Neo4j."""
        context = {
            "window_id": id(learning_window),
            "window_state": learning_window.state.value,
            "stability_threshold": learning_window.stability_threshold,
            "coherence_threshold": learning_window.coherence_threshold,
            "max_changes_per_window": learning_window.max_changes_per_window,
            "change_count": learning_window.change_count,
            "start_time": learning_window.start_time.isoformat(),
            "end_time": learning_window.end_time.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.stored_windows.append(context)
        
        # Store in Neo4j
        query = """
        CREATE (w:LearningWindow {
            window_id: $window_id,
            window_state: $window_state,
            stability_threshold: $stability_threshold,
            coherence_threshold: $coherence_threshold,
            max_changes_per_window: $max_changes_per_window,
            change_count: $change_count,
            start_time: $start_time,
            end_time: $end_time,
            timestamp: $timestamp
        })
        """
        
        self.neo4j_client.execute_query(query, context)
    
    def store_pattern_evolution(self, pattern_id: PatternID, context: Dict[str, Any]) -> None:
        """Store pattern evolution in Neo4j."""
        evolution_context = {
            "pattern_id": pattern_id.id,
            "pattern_name": pattern_id.pattern_name,
            "evolution_count": pattern_id.evolution_count,
            **context,
            "timestamp": datetime.now().isoformat()
        }
        
        self.stored_patterns.append(evolution_context)
        
        # Store in Neo4j
        query = """
        CREATE (p:PatternEvolution {
            pattern_id: $pattern_id,
            pattern_name: $pattern_name,
            evolution_count: $evolution_count,
            entity_id: $entity_id,
            change_type: $change_type,
            window_state: $window_state,
            stability: $stability,
            timestamp: $timestamp
        })
        """
        
        self.neo4j_client.execute_query(query, evolution_context)


# Mock Event Coordinator
class EventCoordinator:
    """Event coordinator for learning windows."""
    
    def __init__(self, stability_threshold=0.7, coherence_threshold=0.6, max_changes_per_window=20):
        """Initialize event coordinator."""
        self.stability_threshold = stability_threshold
        self.coherence_threshold = coherence_threshold
        self.max_changes_per_window = max_changes_per_window
        self.learning_windows = []
        self.event_queue = []
        self.processed_events = []
    
    def create_learning_window(self, start_time=None, end_time=None):
        """Create a new learning window."""
        if start_time is None:
            start_time = datetime.now()
        
        if end_time is None:
            end_time = start_time + timedelta(minutes=10)
        
        window = LearningWindow(
            start_time=start_time,
            end_time=end_time,
            stability_threshold=self.stability_threshold,
            coherence_threshold=self.coherence_threshold,
            max_changes_per_window=self.max_changes_per_window
        )
        
        self.learning_windows.append(window)
        window.activate()  # Activate the window
        
        return window
    
    def queue_event(self, event_id, stability_score):
        """Queue an event for processing."""
        self.event_queue.append({
            "event_id": event_id,
            "stability_score": stability_score,
            "timestamp": datetime.now().isoformat()
        })
    
    def process_events(self):
        """Process queued events."""
        if not self.learning_windows:
            return
        
        # Find an active window
        active_window = None
        for window in self.learning_windows:
            if window.state != WindowState.CLOSED:
                active_window = window
                break
        
        if not active_window:
            # Create a new window if none are active
            active_window = self.create_learning_window()
        
        # Process events with back pressure
        for event in self.event_queue[:]:
            # Apply back pressure based on stability
            if event["stability_score"] < self.stability_threshold:
                # Skip low stability events if we're approaching saturation
                if active_window.change_count > active_window.max_changes_per_window * 0.7:
                    continue
            
            # Process the event
            active_window.stability_score = event["stability_score"]
            active_window.record_state_change(
                entity_id=event["event_id"],
                change_type="event_processing",
                old_value="queued",
                new_value="processed",
                origin="event_coordinator"
            )
            
            # Mark as processed
            self.processed_events.append(event)
            self.event_queue.remove(event)
            
            # Check if window is closed
            if active_window.state == WindowState.CLOSED:
                # Create a new window
                active_window = self.create_learning_window()


@pytest.fixture(scope="function")
def setup_integrated_system():
    """Set up integrated system with all components."""
    setup_adaptive_id_pattern_integration()
    setup_learning_window_pattern_integration()
    setup_learning_window_tonic_harmonic()
    
    # Return cleanup function
    def cleanup():
        # Restore original methods
        if hasattr(LearningWindow, '_original_record_state_change'):
            LearningWindow.record_state_change = LearningWindow._original_record_state_change
    
    return cleanup


@pytest.fixture(scope="function")
def mock_neo4j():
    """Create a mock Neo4j client."""
    return MagicMock()


class TestIntegratedTonicHarmonicSystem:
    """Test suite for integrated tonic-harmonic system."""
    
    def test_integrated_system(self, setup_integrated_system, mock_neo4j):
        """Test complete integration of all components."""
        # Setup with mocked datetime
        with patch('datetime.datetime') as mock_datetime:
            # Set fixed time for testing
            test_date = datetime(2025, 2, 27, 12, 0, 0)
            mock_datetime.now.return_value = test_date
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # Setup with mock Neo4j client
            neo4j_bridge = FieldNeo4jBridge(neo4j_client=mock_neo4j)
            
            # Create IDs with mock logger
            adaptive_id = AdaptiveID("test_concept", "test_creator")
            # Add mock logger with error method
            adaptive_id.logger = MagicMock()
            adaptive_id.logger.error = MagicMock()
            
            pattern_id = PatternID("test_pattern", "test_creator")
            
            # Register for bidirectional updates
            adaptive_id.register_with_pattern_id(pattern_id)
            pattern_id.register_with_adaptive_id(adaptive_id)
            
            # Create event coordinator
            coordinator = EventCoordinator(
                stability_threshold=0.7,
                coherence_threshold=0.6,
                max_changes_per_window=20
            )
            
            # Create field observer and resonance detector
            field_observer = FieldObserver()
            resonance_detector = ResonancePatternDetector()
            
            # Create learning window with fixed times
            window = coordinator.create_learning_window(
                start_time=test_date + timedelta(minutes=5),  # Start in the future
                end_time=test_date + timedelta(minutes=15)
            )
            
            # Ensure window is in CLOSED state
            assert window.state == WindowState.CLOSED
            
            # Register components
            adaptive_id.register_with_learning_window(window)
            window.register_pattern_observer(pattern_id)
            window.register_field_observer(field_observer)
            
            # Register Neo4j bridge as observer
            window.register_field_observer(neo4j_bridge)
        
        # Queue events with varying stability
        stability_pattern = [0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6]
        for i, stability in enumerate(stability_pattern):
            coordinator.queue_event(f"event_{i}", stability)
        
        # Process events
        coordinator.process_events()
        
        # Verify all components were updated
        assert pattern_id.evolution_count > 0
        assert len(pattern_id.evolution_history) > 0
        assert field_observer.analysis_count > 0
        assert mock_neo4j.execute_query.call_count > 0
        assert len(neo4j_bridge.stored_windows) > 0
        
        # Perform harmonic analysis on field observations
        if len(field_observer.observations) >= 3:
            last_observation = field_observer.observations[-1]
            if "harmonic_analysis" in last_observation:
                analysis = last_observation["harmonic_analysis"]
                
                # Detect patterns
                patterns = resonance_detector.detect_patterns(
                    analysis["harmonic_values"],
                    analysis["derivatives"]
                )
                
                # Store pattern evolution in Neo4j
                for pattern in patterns:
                    neo4j_bridge.store_pattern_evolution(pattern_id, {
                        "pattern_type": pattern["type"],
                        "confidence": pattern["confidence"],
                        "entity_id": "resonance_pattern",
                        "change_type": "pattern_detection",
                        "window_state": window.state.value,
                        "stability": pattern["confidence"]
                    })
        
        # Verify pattern evolution was stored in Neo4j
        assert len(neo4j_bridge.stored_patterns) > 0
    
    def test_pattern_evolution_tracking(self, setup_integrated_system):
        """Test full pattern evolution cycle through learning windows."""
        # Setup
        adaptive_id = AdaptiveID("test_concept", "test_creator")
        pattern_id = PatternID("test_pattern", "test_creator")
        
        # Register for bidirectional updates
        adaptive_id.register_with_pattern_id(pattern_id)
        pattern_id.register_with_adaptive_id(adaptive_id)
        
        # Create event coordinator
        coordinator = EventCoordinator(
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20
        )
        
        # Create field observer
        field_observer = FieldObserver()
        
        # Create learning window and register components
        window = coordinator.create_learning_window(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=10)
        )
        adaptive_id.register_with_learning_window(window)
        window.register_pattern_observer(pattern_id)
        window.register_field_observer(field_observer)
        
        # Simulate pattern evolution through multiple changes
        for i in range(5):
            # Set stability score
            window.stability_score = 0.7 + 0.05 * i
            
            # Notify state change
            adaptive_id.notify_state_change(
                change_type=f"evolution_{i}",
                old_value=f"state_{i}",
                new_value=f"state_{i+1}",
                origin="test"
            )
        
        # Verify pattern evolution was tracked
        assert pattern_id.evolution_count == 5
        assert len(pattern_id.evolution_history) == 5
        
        # Verify adaptive ID was updated with pattern evolution
        assert adaptive_id.has_context("pattern_evolution")
        
        # Verify field observer performed harmonic analysis
        assert field_observer.analysis_count > 0
        
        # Verify tonic-harmonic values were recorded
        assert len(field_observer.tonic_values) == 5
        assert len(field_observer.stability_scores) == 5
        
        # Verify harmonic values are increasing (as both stability and tonic increase)
        if "harmonic_analysis" in field_observer.observations[-1]:
            harmonic_values = field_observer.observations[-1]["harmonic_analysis"]["harmonic_values"]
            for i in range(1, len(harmonic_values)):
                assert harmonic_values[i] >= harmonic_values[i-1]
    
    def test_event_coordination_back_pressure(self, setup_integrated_system):
        """Test event coordination and back pressure control."""
        # Create event coordinator
        coordinator = EventCoordinator(
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=10  # Small value to test back pressure
        )
        
        # Create learning window
        window = coordinator.create_learning_window(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=10)
        )
        
        # Queue events with high stability
        for i in range(5):
            coordinator.queue_event(f"high_stability_event_{i}", 0.8)
        
        # Queue events with low stability
        for i in range(10):
            coordinator.queue_event(f"low_stability_event_{i}", 0.3)
        
        # Process events
        coordinator.process_events()
        
        # Verify high stability events were processed
        high_stability_processed = [e for e in coordinator.processed_events 
                                   if e["event_id"].startswith("high_stability_event_")]
        assert len(high_stability_processed) == 5
        
        # Verify some low stability events were skipped due to back pressure
        low_stability_processed = [e for e in coordinator.processed_events 
                                  if e["event_id"].startswith("low_stability_event_")]
        assert len(low_stability_processed) < 10
        
        # Verify window change count
        assert window.change_count <= window.max_changes_per_window
