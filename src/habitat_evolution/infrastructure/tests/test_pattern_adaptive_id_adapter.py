"""
Tests for the PatternAdaptiveIDAdapter.

This module contains tests for the PatternAdaptiveIDAdapter, which bridges
between the Pattern class and AdaptiveID, enabling versioning, relationship tracking,
and context management capabilities for patterns.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, Any

from src.habitat_evolution.adaptive_core.models.pattern import Pattern
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.infrastructure.adapters.pattern_adaptive_id_adapter import (
    PatternAdaptiveIDAdapter,
    PatternAdaptiveIDFactory
)


class TestPatternAdaptiveIDAdapter(unittest.TestCase):
    """Test cases for the PatternAdaptiveIDAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pattern = Pattern(
            id="test-pattern-1",
            base_concept="climate risk",
            creator_id="test-user",
            weight=1.0,
            confidence=0.85,
            uncertainty=0.15,
            coherence=0.75,
            phase_stability=0.8,
            signal_strength=0.9
        )
        
        # Add some properties and metrics
        self.pattern.properties = {
            "domain": "climate",
            "category": "risk"
        }
        self.pattern.metrics = {
            "usage_count": 5,
            "relevance_score": 0.92
        }
        
        # Create the adapter
        self.adapter = PatternAdaptiveIDAdapter(self.pattern)
    
    def test_adapter_initialization(self):
        """Test that the adapter initializes correctly."""
        # Check that the pattern is stored
        self.assertEqual(self.adapter.pattern, self.pattern)
        
        # Check that an AdaptiveID was created
        self.assertIsInstance(self.adapter.adaptive_id, AdaptiveID)
        
        # Check that basic properties were transferred
        self.assertEqual(self.adapter.adaptive_id.base_concept, self.pattern.base_concept)
        self.assertEqual(self.adapter.adaptive_id.creator_id, self.pattern.creator_id)
        self.assertEqual(self.adapter.adaptive_id.weight, self.pattern.weight)
        self.assertEqual(self.adapter.adaptive_id.confidence, self.pattern.confidence)
        self.assertEqual(self.adapter.adaptive_id.uncertainty, self.pattern.uncertainty)
    
    def test_metadata_property(self):
        """Test that the metadata property works correctly."""
        metadata = self.adapter.metadata
        
        # Check that metadata contains expected fields
        self.assertEqual(metadata["coherence"], self.pattern.coherence)
        self.assertEqual(metadata["quality"], self.pattern.confidence)
        self.assertEqual(metadata["stability"], self.pattern.phase_stability)
        self.assertEqual(metadata["text"], self.pattern.base_concept)
        
        # Check that AdaptiveID-specific fields are included
        self.assertIn("version_count", metadata)
        self.assertIn("created_at", metadata)
        self.assertIn("last_modified", metadata)
    
    def test_text_property(self):
        """Test that the text property works correctly."""
        self.assertEqual(self.adapter.text, self.pattern.base_concept)
    
    def test_create_version(self):
        """Test creating a new version."""
        # Initial version count
        initial_version_count = self.adapter.adaptive_id.metadata.get("version_count", 0)
        
        # Create a new version
        new_data = {
            "base_concept": "enhanced climate risk",
            "confidence": 0.9,
            "uncertainty": 0.1
        }
        self.adapter.create_version(new_data, "test")
        
        # Check that a new version was created
        self.assertGreater(
            self.adapter.adaptive_id.metadata.get("version_count", 0),
            initial_version_count
        )
        
        # Check that the pattern was updated
        self.assertEqual(self.pattern.base_concept, "enhanced climate risk")
        self.assertEqual(self.pattern.confidence, 0.9)
        self.assertEqual(self.pattern.uncertainty, 0.1)
    
    def test_update_temporal_context(self):
        """Test updating temporal context."""
        # Update temporal context
        self.adapter.update_temporal_context("usage", {"query": "climate impact", "relevance": 0.95})
        
        # Check that context was updated in both Pattern and AdaptiveID
        self.assertIn("usage", self.pattern.temporal_context)
        self.assertIn("usage", self.adapter.adaptive_id.temporal_context)
        
        # Get the context value from AdaptiveID
        context_value = self.adapter.adaptive_id.get_temporal_context("usage")
        self.assertIsNotNone(context_value)
        self.assertEqual(context_value["query"], "climate impact")
    
    def test_update_spatial_context(self):
        """Test updating spatial context."""
        # Update spatial context
        self.adapter.update_spatial_context({
            "latitude": 37.7749,
            "longitude": -122.4194
        })
        
        # Check that context was updated in both Pattern and AdaptiveID
        self.assertEqual(self.pattern.spatial_context["latitude"], 37.7749)
        self.assertEqual(self.pattern.spatial_context["longitude"], -122.4194)
        
        # Get the context value from AdaptiveID
        self.assertEqual(self.adapter.adaptive_id.get_spatial_context("latitude"), 37.7749)
        self.assertEqual(self.adapter.adaptive_id.get_spatial_context("longitude"), -122.4194)
    
    def test_add_remove_relationship(self):
        """Test adding and removing relationships."""
        # Add a relationship
        self.adapter.add_relationship("related-pattern-1")
        
        # Check that relationship was added to Pattern
        self.assertIn("related-pattern-1", self.pattern.relationships)
        
        # Remove the relationship
        self.adapter.remove_relationship("related-pattern-1")
        
        # Check that relationship was removed from Pattern
        self.assertNotIn("related-pattern-1", self.pattern.relationships)
    
    def test_to_dict(self):
        """Test converting to dictionary representation."""
        # Convert to dict
        pattern_dict = self.adapter.to_dict()
        
        # Check that basic Pattern properties are included
        self.assertEqual(pattern_dict["id"], self.pattern.id)
        self.assertEqual(pattern_dict["base_concept"], self.pattern.base_concept)
        self.assertEqual(pattern_dict["confidence"], self.pattern.confidence)
        
        # Check that AdaptiveID-specific properties are included
        self.assertIn("version_history", pattern_dict)
        self.assertIn("version_count", pattern_dict)
        self.assertIn("adaptive_id", pattern_dict)
    
    def test_factory_create_adapter(self):
        """Test creating an adapter using the factory."""
        # Create a new pattern
        pattern = Pattern(
            id="test-pattern-2",
            base_concept="flood risk",
            creator_id="test-user"
        )
        
        # Create adapter using factory
        adapter = PatternAdaptiveIDFactory.create_adapter(pattern)
        
        # Check that adapter was created correctly
        self.assertIsInstance(adapter, PatternAdaptiveIDAdapter)
        self.assertEqual(adapter.pattern, pattern)
        self.assertEqual(adapter.adaptive_id.base_concept, "flood risk")
    
    def test_factory_create_from_adaptive_id(self):
        """Test creating an adapter from an AdaptiveID using the factory."""
        # Create an AdaptiveID
        adaptive_id = AdaptiveID(
            base_concept="drought risk",
            creator_id="test-user",
            confidence=0.88
        )
        
        # Create adapter using factory
        adapter = PatternAdaptiveIDFactory.create_from_adaptive_id(adaptive_id)
        
        # Check that adapter was created correctly
        self.assertIsInstance(adapter, PatternAdaptiveIDAdapter)
        self.assertEqual(adapter.adaptive_id, adaptive_id)
        self.assertEqual(adapter.pattern.base_concept, "drought risk")
        self.assertEqual(adapter.pattern.confidence, 0.88)


if __name__ == "__main__":
    unittest.main()
