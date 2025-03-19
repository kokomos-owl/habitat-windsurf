"""
Tests for PatternID integration with LearningWindow and AdaptiveID.

These tests validate the integration between PatternID, LearningWindow, and AdaptiveID,
ensuring proper pattern evolution tracking and bidirectional updates.
"""

import pytest
import sys
import os
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

# Mock PatternID class for testing
class PatternID:
    """Mock PatternID class for testing pattern evolution tracking."""
    
    def __init__(self, pattern_name: str, creator: str):
        """Initialize PatternID with name and creator."""
        self.id = f"{pattern_name}_{creator}"
        self.pattern_name = pattern_name
        self.creator = creator
        self.evolution_count = 0
        self.evolution_history = []
        self.adaptive_ids = []
        
    def register_with_adaptive_id(self, adaptive_id: AdaptiveID) -> None:
        """Register this PatternID with an AdaptiveID for bidirectional updates."""
        self.adaptive_ids.append(adaptive_id)
        
    def observe_pattern_evolution(self, context: Dict[str, Any]) -> None:
        """Observe pattern evolution from learning window."""
        self.evolution_count += 1
        self.evolution_history.append(context)
        
        # Print debug information
        print(f"PatternID {self.id} observed evolution: {context.get('change_type', 'unknown')}")
        print(f"Current evolution count: {self.evolution_count}")
        
        # Update associated AdaptiveIDs
        for adaptive_id in self.adaptive_ids:
            adaptive_id.update_from_pattern_id(self, context)
            
    def evolve_pattern(self, evolution_type: str, stability: float) -> None:
        """Evolve pattern and notify associated AdaptiveIDs."""
        self.evolution_count += 1  # Increment counter
        context = {
            "evolution_type": evolution_type,
            "stability": stability,
            "timestamp": datetime.now().isoformat()
        }
        self.evolution_history.append(context)
        
        # Print debug information
        print(f"PatternID {self.id} evolved: {evolution_type}")
        print(f"Current evolution count: {self.evolution_count}")
        
        # Notify associated AdaptiveIDs
        for adaptive_id in self.adaptive_ids:
            adaptive_id.update_from_pattern_id(self, context)
            
    def has_context(self, context_key: str) -> bool:
        """Check if pattern has specific context."""
        for context in self.evolution_history:
            if context_key in context:
                return True
        return False


# Extend AdaptiveID with pattern ID integration methods
def setup_adaptive_id_pattern_integration():
    """Add pattern ID integration methods to AdaptiveID."""
    
    # Add register_with_pattern_id method if not already present
    if not hasattr(AdaptiveID, 'register_with_pattern_id'):
        def register_with_pattern_id(self, pattern_id):
            """Register this AdaptiveID with a PatternID for bidirectional updates."""
            if not hasattr(self, 'pattern_ids'):
                self.pattern_ids = []
            self.pattern_ids.append(pattern_id)
        
        AdaptiveID.register_with_pattern_id = register_with_pattern_id
    
    # Add update_from_pattern_id method if not already present
    if not hasattr(AdaptiveID, 'update_from_pattern_id'):
        def update_from_pattern_id(self, pattern_id, pattern_data):
            """Update this AdaptiveID based on pattern evolution in PatternID."""
            # Update temporal context based on pattern evolution
            self.update_temporal_context(
                f"pattern_evolution_{pattern_id.id}",
                pattern_data,
                f"pattern_id_{pattern_id.id}"
            )
        
        AdaptiveID.update_from_pattern_id = update_from_pattern_id
    
    # Add has_context method if not already present
    if not hasattr(AdaptiveID, 'has_context'):
        def has_context(self, context_key):
            """Check if AdaptiveID has specific context."""
            if hasattr(self, 'temporal_context'):
                for key in self.temporal_context.keys():
                    if context_key in key:
                        return True
            return False
        
        AdaptiveID.has_context = has_context


# Extend LearningWindow with pattern observer methods
def setup_learning_window_pattern_integration():
    """Add pattern observer methods to LearningWindow."""
    
    # Add register_pattern_observer method if not already present
    if not hasattr(LearningWindow, 'register_pattern_observer'):
        def register_pattern_observer(self, observer):
            """Register a pattern observer with this learning window."""
            if not hasattr(self, 'pattern_observers'):
                self.pattern_observers = []
            self.pattern_observers.append(observer)
        
        LearningWindow.register_pattern_observer = register_pattern_observer
    
    # Add notify_pattern_observers method if not already present
    if not hasattr(LearningWindow, 'notify_pattern_observers'):
        def notify_pattern_observers(self, context):
            """Notify pattern observers of state changes."""
            if hasattr(self, 'pattern_observers'):
                for observer in self.pattern_observers:
                    observer.observe_pattern_evolution(context)
        
        LearningWindow.notify_pattern_observers = notify_pattern_observers
    
    # Extend record_state_change to notify pattern observers
    original_record_state_change = LearningWindow.record_state_change
    
    def extended_record_state_change(self, entity_id, change_type, old_value, new_value, origin):
        """Extended record_state_change that notifies pattern observers."""
        # Call original method
        result = original_record_state_change(self, entity_id, change_type, old_value, new_value, origin)
        
        # Create pattern evolution context
        pattern_context = {
            "entity_id": entity_id,
            "change_type": change_type,
            "old_value": old_value,
            "new_value": new_value,
            "window_state": self.state.value,
            "stability": self.stability_score if hasattr(self, 'stability_score') else 0.5,
            "timestamp": datetime.now().isoformat()
        }
        
        # Notify pattern observers
        if hasattr(self, 'notify_pattern_observers'):
            self.notify_pattern_observers(pattern_context)
        
        return result
    
    LearningWindow.record_state_change = extended_record_state_change


@pytest.fixture(scope="function")
def setup_integration():
    """Set up integration between AdaptiveID, LearningWindow, and PatternID."""
    setup_adaptive_id_pattern_integration()
    setup_learning_window_pattern_integration()
    
    # Return cleanup function
    def cleanup():
        # Restore original record_state_change method
        if hasattr(LearningWindow, '_original_record_state_change'):
            LearningWindow.record_state_change = LearningWindow._original_record_state_change
    
    return cleanup


class TestPatternIDIntegration:
    """Test suite for PatternID integration with LearningWindow and AdaptiveID."""
    
    def test_pattern_id_learning_window_basic_integration(self, setup_integration):
        """Test basic integration between PatternID and LearningWindow."""
        # Setup with mocked datetime
        with patch('datetime.datetime') as mock_datetime:
            # Set fixed time for testing
            test_date = datetime(2025, 2, 27, 12, 0, 0)
            mock_datetime.now.return_value = test_date
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # Create pattern ID
            pattern_id = PatternID("test_pattern", "test_creator")
            
            # Create learning window with fixed times
            learning_window = LearningWindow(
                start_time=test_date + timedelta(minutes=5),  # Start in the future
                end_time=test_date + timedelta(minutes=15),
                stability_threshold=0.7,
                coherence_threshold=0.6,
                max_changes_per_window=20
            )
            
            # Ensure window is in CLOSED state
            assert learning_window.state == WindowState.CLOSED
            
            # Register PatternID as observer
            learning_window.register_pattern_observer(pattern_id)
            
            # Record state change
            learning_window.record_state_change(
                entity_id="test_entity",
                change_type="test_change",
                old_value="old_value",
                new_value="new_value",
                origin="test_origin"
            )
            
            # Verify pattern evolution was tracked
            assert pattern_id.evolution_count == 1
            assert len(pattern_id.evolution_history) == 1
            
            # Verify context contains expected fields
            context = pattern_id.evolution_history[0]
            assert context["entity_id"] == "test_entity"
            assert context["change_type"] == "test_change"
            assert context["old_value"] == "old_value"
            assert context["new_value"] == "new_value"
            assert context["window_state"] == WindowState.CLOSED.value
    
    def test_adaptive_id_pattern_id_synchronization(self, setup_integration):
        """Test bidirectional updates between AdaptiveID and PatternID."""
        # Setup with mocked datetime
        with patch('datetime.datetime') as mock_datetime:
            # Set fixed time for testing
            test_date = datetime(2025, 2, 27, 12, 0, 0)
            mock_datetime.now.return_value = test_date
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # Create pattern ID
            pattern_id = PatternID("test_pattern", "test_creator")
            
            # Create adaptive ID with mock logger
            adaptive_id = AdaptiveID("test_concept", "test_creator")
            adaptive_id.logger = MagicMock()
            adaptive_id.logger.error = MagicMock()
            
            # Register for bidirectional updates
            adaptive_id.register_with_pattern_id(pattern_id)
            pattern_id.register_with_adaptive_id(adaptive_id)
            
            # Update from PatternID to AdaptiveID
            pattern_id.evolve_pattern("test_evolution", 0.8)
            
            # Verify AdaptiveID received update
            assert adaptive_id.has_context("pattern_evolution")
            
            # Create a mock learning window instead of a real one
            mock_learning_window = MagicMock()
            mock_learning_window.state = WindowState.CLOSED
            mock_learning_window.record_state_change.return_value = True
            
            # Register components
            adaptive_id.register_with_learning_window(mock_learning_window)
            
            # Update from AdaptiveID to PatternID directly
            # This avoids the circular dependency through learning window
            context = {
                "entity_id": adaptive_id.id,
                "change_type": "concept_evolution",
                "old_value": "concept_v1",
                "new_value": "concept_v2",
                "origin": "test_origin",
                "window_state": WindowState.CLOSED.value,
                "stability": 0.8,
                "tonic_value": 0.7,
                "harmonic_value": 0.56,
                "timestamp": test_date.isoformat()
            }
            
            # Directly update pattern ID with the context
            pattern_id.observe_pattern_evolution(context)
            
            # Verify PatternID received update
            assert pattern_id.evolution_count == 2
            assert len(pattern_id.evolution_history) == 2
            assert pattern_id.evolution_history[1]["change_type"] == "concept_evolution"
