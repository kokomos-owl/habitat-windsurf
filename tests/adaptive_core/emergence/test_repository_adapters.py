"""
Test suite for repository adapters.

This test suite verifies that the repository adapters correctly implement
the repository interfaces and work with the ArangoDB persistence layer.
"""

import unittest
import logging
import os
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime
import uuid

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s')

logger = logging.getLogger(__name__)

# Import the repository factory
from src.habitat_evolution.adaptive_core.emergence.repository_factory import (
    create_field_state_repository,
    create_pattern_repository,
    create_relationship_repository,
    create_topology_repository
)

# Import the interfaces
from src.habitat_evolution.adaptive_core.emergence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.topology_repository import TopologyRepositoryInterface

# Import the adapters
from src.habitat_evolution.adaptive_core.emergence.adapters.field_state_repository_adapter import FieldStateRepositoryAdapter
from src.habitat_evolution.adaptive_core.emergence.adapters.pattern_repository_adapter import PatternRepositoryAdapter
from src.habitat_evolution.adaptive_core.emergence.adapters.relationship_repository_adapter import RelationshipRepositoryAdapter
from src.habitat_evolution.adaptive_core.emergence.adapters.topology_repository_adapter import TopologyRepositoryAdapter


class TestRepositoryAdapters(unittest.TestCase):
    """Test the repository adapters."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the ArangoDB connection
        self.mock_db = MagicMock()
        
        # Create mock repositories
        self.mock_field_state_repo = MagicMock()
        self.mock_pattern_repo = MagicMock()
        self.mock_relationship_repo = MagicMock()
        self.mock_topology_repo = MagicMock()
    
    @patch('src.habitat_evolution.adaptive_core.emergence.adapters.field_state_repository_adapter.TonicHarmonicFieldStateRepository')
    def test_field_state_repository_adapter(self, mock_repo_class):
        """Test that the field state repository adapter correctly implements the interface."""
        # Configure the mock
        mock_repo_class.return_value = self.mock_field_state_repo
        
        # Create the adapter
        adapter = create_field_state_repository(self.mock_db)
        
        # Verify the adapter is the correct type
        self.assertIsInstance(adapter, FieldStateRepositoryInterface)
        self.assertIsInstance(adapter, FieldStateRepositoryAdapter)
        
        # Test save method
        field_state = {"id": "test_id", "coherence": 0.8}
        adapter.save(field_state)
        self.mock_field_state_repo.save.assert_called_once()
        
        # Test find_by_id method
        adapter.find_by_id("test_id")
        self.mock_field_state_repo.find_by_id.assert_called_once_with("test_id")
    
    @patch('src.habitat_evolution.adaptive_core.emergence.adapters.pattern_repository_adapter.PatternRepository')
    def test_pattern_repository_adapter(self, mock_repo_class):
        """Test that the pattern repository adapter correctly implements the interface."""
        # Configure the mock
        mock_repo_class.return_value = self.mock_pattern_repo
        
        # Create the adapter
        adapter = create_pattern_repository(self.mock_db)
        
        # Verify the adapter is the correct type
        self.assertIsInstance(adapter, PatternRepositoryInterface)
        self.assertIsInstance(adapter, PatternRepositoryAdapter)
        
        # Test save method
        pattern = {"id": "test_id", "confidence": 0.9}
        adapter.save(pattern)
        self.mock_pattern_repo.save.assert_called_once()
        
        # Test find_by_id method
        adapter.find_by_id("test_id")
        self.mock_pattern_repo.find_by_id.assert_called_once_with("test_id")
    
    @patch('src.habitat_evolution.adaptive_core.emergence.adapters.relationship_repository_adapter.PredicateRelationshipRepository')
    def test_relationship_repository_adapter(self, mock_repo_class):
        """Test that the relationship repository adapter correctly implements the interface."""
        # Configure the mock
        mock_repo_class.return_value = self.mock_relationship_repo
        
        # Create the adapter
        adapter = create_relationship_repository(self.mock_db)
        
        # Verify the adapter is the correct type
        self.assertIsInstance(adapter, RelationshipRepositoryInterface)
        self.assertIsInstance(adapter, RelationshipRepositoryAdapter)
        
        # Test save method
        relationship = {"source_id": "source", "target_id": "target", "type": "SIMILAR_TO"}
        adapter.save(relationship)
        self.mock_relationship_repo.save.assert_called_once()
        
        # Test find_by_id method
        adapter.find_by_id("test_id")
        self.mock_relationship_repo.find_by_id.assert_called_once_with("test_id")
    
    @patch('src.habitat_evolution.adaptive_core.emergence.adapters.topology_repository_adapter.TopologyRepository')
    def test_topology_repository_adapter(self, mock_repo_class):
        """Test that the topology repository adapter correctly implements the interface."""
        # Configure the mock
        mock_repo_class.return_value = self.mock_topology_repo
        
        # Create the adapter
        adapter = create_topology_repository(self.mock_db)
        
        # Verify the adapter is the correct type
        self.assertIsInstance(adapter, TopologyRepositoryInterface)
        self.assertIsInstance(adapter, TopologyRepositoryAdapter)
        
        # Test save method
        topology = {"field_id": "field_id", "clusters": 3}
        adapter.save(topology)
        self.mock_topology_repo.save.assert_called_once()
        
        # Test find_by_id method
        adapter.find_by_id("test_id")
        self.mock_topology_repo.find_by_id.assert_called_once_with("test_id")


if __name__ == "__main__":
    unittest.main()
