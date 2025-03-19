"""
Verification script for AdaptiveID and LearningWindow integration.

This script tests the integration between AdaptiveID and LearningWindow without
relying on the test infrastructure, to verify our implementation works correctly.
It also tests window state transitions (CLOSED -> OPENING -> OPEN -> CLOSED).
"""

import sys
import os
import time
from datetime import datetime, timedelta
from unittest.mock import patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now we can import from the project modules
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow, WindowState

def test_basic_integration():
    """Test basic integration between AdaptiveID and LearningWindow."""
    print("\n=== Testing basic integration ===\n")
    
    # Create a learning window
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=10)
    learning_window = LearningWindow(
        start_time=start_time,
        end_time=end_time,
        stability_threshold=0.7,
        coherence_threshold=0.6,
        max_changes_per_window=20
    )
    
    # Create an AdaptiveID instance
    adaptive_id = AdaptiveID("test_concept", "test_creator")
    
    # Register the learning window with the AdaptiveID
    adaptive_id.register_with_learning_window(learning_window)
    
    # Verify registration
    print(f"Learning window registered: {learning_window in adaptive_id.learning_windows}")
    
    # Notify state change from AdaptiveID
    print("Notifying state change...")
    adaptive_id.notify_state_change(
        "test_change", 
        "old_value", 
        "new_value", 
        "test_origin"
    )
    
    # Verify learning window recorded the change
    print(f"Learning window change count: {learning_window.change_count}")
    print(f"Learning window stability metrics: {learning_window.stability_metrics}")
    
    # Test with temporal context update
    print("\nUpdating temporal context...")
    adaptive_id.update_temporal_context("test_key", "test_value", "test_origin")
    
    # Verify learning window recorded the change
    print(f"Learning window change count: {learning_window.change_count}")

def test_window_state_transitions():
    """Test window state transitions (CLOSED -> OPENING -> OPEN -> CLOSED)."""
    print("\n=== Testing window state transitions ===\n")
    
    # Use a fixed start time for testing
    test_date = datetime(2025, 2, 27, 12, 0, 0)
    
    # Create a mock datetime to control time during the test
    with patch('habitat_evolution.pattern_aware_rag.learning.learning_control.datetime') as mock_datetime:
        # Setup the mock datetime to return our test_date
        mock_datetime.now.return_value = test_date
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        # Create a learning window
        start_time = test_date
        end_time = start_time + timedelta(minutes=10)
        window = LearningWindow(
            start_time=start_time,
            end_time=end_time,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=5  # Small value to test saturation
        )
        
        # Initial state should be CLOSED before start time
        test_date = datetime(2025, 2, 27, 11, 59, 0)  # 1 minute before start
        mock_datetime.now.return_value = test_date
        print(f"Time: {test_date}, Window state: {window.state}")
        assert window.state == WindowState.CLOSED, "Window should be CLOSED before start time"
        
        # State should be OPENING at start time
        test_date = datetime(2025, 2, 27, 12, 0, 30)  # 30 seconds after start
        mock_datetime.now.return_value = test_date
        print(f"Time: {test_date}, Window state: {window.state}")
        assert window.state == WindowState.OPENING, "Window should be OPENING in the first minute"
        
        # State should be OPEN after 1 minute
        test_date = datetime(2025, 2, 27, 12, 1, 30)  # 1.5 minutes after start
        mock_datetime.now.return_value = test_date
        print(f"Time: {test_date}, Window state: {window.state}")
        assert window.state == WindowState.OPEN, "Window should be OPEN after the first minute"
        
        # Record changes until window is saturated
        for i in range(6):  # More than max_changes_per_window
            window.record_state_change(
                f"entity_{i}",
                "test_change",
                "old_value",
                "new_value",
                "test_origin"
            )
            print(f"Recorded change {i+1}, Window state: {window.state}")
            
            # Window should be CLOSED after saturation
            if i >= window.max_changes_per_window - 1:
                assert window.state == WindowState.CLOSED, "Window should be CLOSED after saturation"
        
        # State should be CLOSED after end time
        test_date = datetime(2025, 2, 27, 12, 15, 0)  # After end time
        mock_datetime.now.return_value = test_date
        print(f"Time: {test_date}, Window state: {window.state}")
        assert window.state == WindowState.CLOSED, "Window should be CLOSED after end time"

def main():
    """Run all verification tests."""
    print("Starting verification of AdaptiveID and LearningWindow integration...")
    
    # Run tests
    test_basic_integration()
    test_window_state_transitions()
    
    print("\nVerification complete! All tests passed.")

if __name__ == "__main__":
    main()
