"""
Test suite for the new repository structure.

This test suite verifies that the repository interfaces, adapters, and factory
work correctly with the new import structure.
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
from habitat_evolution.adaptive_core.persistence.factory import (
    create_field_state_repository,
    create_pattern_repository,
    create_relationship_repository,
    create_topology_repository,
    create_repositories
)

# Import the interfaces
from habitat_evolution.adaptive_core.persistence.interfaces.field_state_repository import FieldStateRepositoryInterface
from habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepositoryInterface
from habitat_evolution.adaptive_core.persistence.interfaces.relationship_repository import RelationshipRepositoryInterface
from habitat_evolution.adaptive_core.persistence.interfaces.topology_repository import TopologyRepositoryInterface

# Import the adapters
from habitat_evolution.adaptive_core.persistence.adapters.field_state_repository_adapter import FieldStateRepositoryAdapter
from habitat_evolution.adaptive_core.persistence.adapters.pattern_repository_adapter import PatternRepositoryAdapter
from habitat_evolution.adaptive_core.persistence.adapters.relationship_repository_adapter import RelationshipRepositoryAdapter
from habitat_evolution.adaptive_core.persistence.adapters.topology_repository_adapter import TopologyRepositoryAdapter


class TestRepositoryStructure(unittest.TestCase):
    """Test the repository structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the ArangoDB connection
        self.mock_db = MagicMock()
        
        # Create mock repositories
        self.mock_field_state_repo = MagicMock()
        self.mock_pattern_repo = MagicMock()
        self.mock_relationship_repo = MagicMock()
        self.mock_topology_repo = MagicMock()
    
    @patch('habitat_evolution.pattern_aware_rag.persistence.arangodb.field_state_repository.TonicHarmonicFieldStateRepository')
    def test_field_state_repository_factory(self, mock_repo_class):
        """Test that the field state repository factory correctly creates an adapter."""
        # Configure the mock
        mock_repo_instance = MagicMock()
        mock_repo_class.return_value = mock_repo_instance
        
        # Create the repository
        repo = create_field_state_repository(self.mock_db)
        
        # Verify the repository is the correct type
        self.assertIsInstance(repo, FieldStateRepositoryInterface)
        self.assertIsInstance(repo, FieldStateRepositoryAdapter)
    
    @patch('habitat_evolution.pattern_aware_rag.persistence.arangodb.pattern_repository.PatternRepository')
    def test_pattern_repository_factory(self, mock_repo_class):
        """Test that the pattern repository factory correctly creates an adapter."""
        # Configure the mock
        mock_repo_instance = MagicMock()
        mock_repo_class.return_value = mock_repo_instance
        
        # Create the repository
        repo = create_pattern_repository(self.mock_db)
        
        # Verify the repository is the correct type
        self.assertIsInstance(repo, PatternRepositoryInterface)
        self.assertIsInstance(repo, PatternRepositoryAdapter)
    
    @patch('habitat_evolution.pattern_aware_rag.persistence.arangodb.predicate_relationship_repository.PredicateRelationshipRepository')
    def test_relationship_repository_factory(self, mock_repo_class):
        """Test that the relationship repository factory correctly creates an adapter."""
        # Configure the mock
        mock_repo_instance = MagicMock()
        mock_repo_class.return_value = mock_repo_instance
        
        # Create the repository
        repo = create_relationship_repository(self.mock_db)
        
        # Verify the repository is the correct type
        self.assertIsInstance(repo, RelationshipRepositoryInterface)
        self.assertIsInstance(repo, RelationshipRepositoryAdapter)
    
    @patch('habitat_evolution.pattern_aware_rag.persistence.arangodb.topology_repository.TopologyRepository')
    def test_topology_repository_factory(self, mock_repo_class):
        """Test that the topology repository factory correctly creates an adapter."""
        # Configure the mock
        mock_repo_instance = MagicMock()
        mock_repo_class.return_value = mock_repo_instance
        
        # Create the repository
        repo = create_topology_repository(self.mock_db)
        
        # Verify the repository is the correct type
        self.assertIsInstance(repo, TopologyRepositoryInterface)
        self.assertIsInstance(repo, TopologyRepositoryAdapter)
    
    @patch('habitat_evolution.adaptive_core.persistence.factory.create_field_state_repository')
    @patch('habitat_evolution.adaptive_core.persistence.factory.create_pattern_repository')
    @patch('habitat_evolution.adaptive_core.persistence.factory.create_relationship_repository')
    @patch('habitat_evolution.adaptive_core.persistence.factory.create_topology_repository')
    def test_create_repositories(self, mock_topology, mock_relationship, mock_pattern, mock_field_state):
        """Test that the create_repositories function correctly creates all repositories."""
        # Configure the mocks
        mock_field_state.return_value = self.mock_field_state_repo
        mock_pattern.return_value = self.mock_pattern_repo
        mock_relationship.return_value = self.mock_relationship_repo
        mock_topology.return_value = self.mock_topology_repo
        
        # Create the repositories
        repos = create_repositories(self.mock_db)
        
        # Verify all repositories were created
        self.assertEqual(repos["field_state_repository"], self.mock_field_state_repo)
        self.assertEqual(repos["pattern_repository"], self.mock_pattern_repo)
        self.assertEqual(repos["relationship_repository"], self.mock_relationship_repo)
        self.assertEqual(repos["topology_repository"], self.mock_topology_repo)
        
        # Verify all factory methods were called
        mock_field_state.assert_called_once_with(self.mock_db, None)
        mock_pattern.assert_called_once_with(self.mock_db, None)
        mock_relationship.assert_called_once_with(self.mock_db, None)
        mock_topology.assert_called_once_with(self.mock_db, None)


if __name__ == "__main__":
    unittest.main()
