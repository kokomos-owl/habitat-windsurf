"""Test suite for AdaptiveID implementation."""

import unittest
import logging
from unittest.mock import patch, MagicMock
import threading
import time
from datetime import datetime
from typing import Dict, Any
import sys
from io import StringIO

from habitat_evolution.adaptive_core.id.adaptive_id import (
    AdaptiveID,
    AdaptiveIDException,
    VersionNotFoundException,
    InvalidValueError
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_adaptive_id.log')
    ]
)
logger = logging.getLogger(__name__)

class TestOutputCapture:
    """Context manager to capture print statements and logging"""
    def __init__(self):
        self.held = StringIO()
        self.stream = sys.stdout

    def __enter__(self):
        sys.stdout = self.held
        return self.held

    def __exit__(self, *args):
        sys.stdout = self.stream

class TestAdaptiveID(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        logger.info("\n" + "="*50)
        logger.info(f"Starting test: {self._testMethodName}")
        logger.info("="*50)
        
        self.test_concept = "test_pattern"
        self.adaptive_id = AdaptiveID(
            base_concept=self.test_concept,
            weight=0.8,
            confidence=0.9,
            uncertainty=0.1
        )

    def test_initialization(self):
        """Test proper initialization of AdaptiveID."""
        self.assertEqual(self.adaptive_id.base_concept, self.test_concept)
        self.assertEqual(self.adaptive_id.weight, 0.8)
        self.assertEqual(self.adaptive_id.confidence, 0.9)
        self.assertEqual(self.adaptive_id.uncertainty, 0.1)
        self.assertIsNotNone(self.adaptive_id.id)
        self.assertEqual(len(self.adaptive_id.versions), 1)

    def test_temporal_context(self):
        """Test temporal context updates and retrieval."""
        test_key = "test_temporal"
        test_value = {"data": "test"}
        test_origin = "test_case"
        
        self.adaptive_id.update_temporal_context(test_key, test_value, test_origin)
        retrieved_value = self.adaptive_id.get_temporal_context(test_key)
        
        self.assertEqual(retrieved_value, test_value)
        self.assertGreater(len(self.adaptive_id.temporal_context[test_key]), 0)

    def test_spatial_context(self):
        """Test spatial context updates and retrieval."""
        test_key = "latitude"
        test_value = 45.0
        test_origin = "test_case"
        
        self.adaptive_id.update_spatial_context(test_key, test_value, test_origin)
        retrieved_value = self.adaptive_id.get_spatial_context(test_key)
        
        self.assertEqual(retrieved_value, test_value)
        
        # Test invalid key
        with self.assertRaises(InvalidValueError):
            self.adaptive_id.update_spatial_context("invalid_key", test_value, test_origin)

    def test_snapshot_restore(self):
        """Test snapshot creation and restoration."""
        # Update some values
        self.adaptive_id.update_temporal_context("test_key", "test_value", "test")
        self.adaptive_id.update_spatial_context("latitude", 45.0, "test")
        
        # Create snapshot
        snapshot = self.adaptive_id.create_snapshot()
        
        # Create new instance
        new_adaptive_id = AdaptiveID("new_concept")
        new_adaptive_id.restore_from_snapshot(snapshot)
        
        # Verify restoration
        self.assertEqual(self.adaptive_id.id, new_adaptive_id.id)
        self.assertEqual(self.adaptive_id.base_concept, new_adaptive_id.base_concept)
        self.assertEqual(
            self.adaptive_id.get_temporal_context("test_key"),
            new_adaptive_id.get_temporal_context("test_key")
        )
        self.assertEqual(
            self.adaptive_id.get_spatial_context("latitude"),
            new_adaptive_id.get_spatial_context("latitude")
        )

    def test_thread_safety(self):
        """Test thread safety of context updates."""
        def update_contexts():
            for i in range(10):
                self.adaptive_id.update_temporal_context(
                    f"key_{i}",
                    f"value_{i}",
                    "thread_test"
                )
                self.adaptive_id.update_spatial_context(
                    "latitude",
                    float(i),
                    "thread_test"
                )
                time.sleep(0.01)  # Simulate work
        
        # Create and start threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=update_contexts)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify no data corruption
        self.assertGreaterEqual(len(self.adaptive_id.temporal_context), 10)
        self.assertIsNotNone(self.adaptive_id.get_spatial_context("latitude"))

    def test_state_comparison(self):
        """Test state comparison functionality."""
        state1 = {
            "key1": "value1",
            "key2": "value2"
        }
        state2 = {
            "key1": "value1",
            "key2": "different",
            "key3": "new"
        }
        
        differences = self.adaptive_id.compare_states(state1, state2)
        
        self.assertIn("key2", differences)
        self.assertIn("key3", differences)
        self.assertNotIn("key1", differences)
        self.assertEqual(differences["key2"], ("value2", "different"))
        self.assertEqual(differences["key3"], (None, "new"))

if __name__ == '__main__':
    unittest.main()
