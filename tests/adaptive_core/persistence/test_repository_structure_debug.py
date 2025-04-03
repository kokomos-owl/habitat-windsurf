"""
Test suite for the new repository structure with enhanced debugging.

This test suite verifies that the repository interfaces, adapters, and factory
work correctly with the new import structure, with detailed debugging information.
"""

import unittest
import logging
import os
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime
import uuid
import inspect

# Configure logging with more detailed format
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s')

logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Import the repository factory
from src.habitat_evolution.adaptive_core.persistence.factory import (
    create_field_state_repository,
    create_pattern_repository,
    create_relationship_repository,
    create_topology_repository,
    create_repositories
)

# Import the interfaces
from src.habitat_evolution.adaptive_core.persistence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.adaptive_core.persistence.interfaces.topology_repository import TopologyRepositoryInterface

# Import the adapters
from src.habitat_evolution.adaptive_core.persistence.adapters.field_state_repository_adapter import FieldStateRepositoryAdapter
from src.habitat_evolution.adaptive_core.persistence.adapters.pattern_repository_adapter import PatternRepositoryAdapter
from src.habitat_evolution.adaptive_core.persistence.adapters.relationship_repository_adapter import RelationshipRepositoryAdapter
from src.habitat_evolution.adaptive_core.persistence.adapters.topology_repository_adapter import TopologyRepositoryAdapter


def log_object_details(obj, name="Object"):
    """Log detailed information about an object."""
    logger.debug(f"{name} type: {type(obj)}")
    logger.debug(f"{name} dir: {dir(obj)}")
    if hasattr(obj, "__dict__"):
        logger.debug(f"{name} __dict__: {obj.__dict__}")
    if hasattr(obj, "__class__"):
        logger.debug(f"{name} __class__: {obj.__class__}")
        logger.debug(f"{name} MRO: {obj.__class__.__mro__}")


class TestRepositoryStructureDebug(unittest.TestCase):
    """Test the repository structure with enhanced debugging."""
    
    def setUp(self):
        """Set up test fixtures."""
        logger.debug("Setting up test fixtures")
        
        # Mock the ArangoDB connection
        self.mock_db = MagicMock()
        logger.debug(f"Created mock_db: {self.mock_db}")
        
        # Create mock repositories
        self.mock_field_state_repo = MagicMock()
        self.mock_pattern_repo = MagicMock()
        self.mock_relationship_repo = MagicMock()
        self.mock_topology_repo = MagicMock()
        
        logger.debug("Setup complete")
    
    def tearDown(self):
        """Tear down test fixtures."""
        logger.debug("Tearing down test fixtures")
    
    @patch('src.habitat_evolution.adaptive_core.persistence.adapters.field_state_repository_adapter.FieldStateRepositoryAdapter')
    def test_field_state_repository_factory(self, mock_adapter_class):
        """Test that the field state repository factory correctly creates an adapter."""
        logger.debug(f"Starting test_field_state_repository_factory with mock: {mock_adapter_class}")
        
        # Configure the mock
        mock_adapter_instance = MagicMock(spec=FieldStateRepositoryInterface)
        mock_adapter_class.return_value = mock_adapter_instance
        
        logger.debug(f"Configured mock_adapter_instance: {mock_adapter_instance}")
        
        # Create the repository
        logger.debug("Calling create_field_state_repository")
        repo = create_field_state_repository(self.mock_db)
        
        logger.debug(f"Repository created: {repo}")
        log_object_details(repo, "Repository")
        
        # Verify the repository is the correct type
        logger.debug(f"Verifying repo is instance of FieldStateRepositoryInterface")
        self.assertIsInstance(repo, FieldStateRepositoryInterface)
        
        logger.debug("test_field_state_repository_factory completed successfully")
    
    @patch('src.habitat_evolution.adaptive_core.persistence.adapters.pattern_repository_adapter.PatternRepositoryAdapter')
    def test_pattern_repository_factory(self, mock_adapter_class):
        """Test that the pattern repository factory correctly creates an adapter."""
        logger.debug(f"Starting test_pattern_repository_factory with mock: {mock_adapter_class}")
        
        # Configure the mock
        mock_adapter_instance = MagicMock(spec=PatternRepositoryInterface)
        mock_adapter_class.return_value = mock_adapter_instance
        
        logger.debug(f"Configured mock_adapter_instance: {mock_adapter_instance}")
        
        # Create the repository
        logger.debug("Calling create_pattern_repository")
        repo = create_pattern_repository(self.mock_db)
        
        logger.debug(f"Repository created: {repo}")
        log_object_details(repo, "Repository")
        
        # Verify the repository is the correct type
        logger.debug(f"Verifying repo is instance of PatternRepositoryInterface")
        self.assertIsInstance(repo, PatternRepositoryInterface)
        
        logger.debug("test_pattern_repository_factory completed successfully")
    
    @patch('src.habitat_evolution.adaptive_core.persistence.adapters.relationship_repository_adapter.RelationshipRepositoryAdapter')
    def test_relationship_repository_factory(self, mock_adapter_class):
        """Test that the relationship repository factory correctly creates an adapter."""
        logger.debug(f"Starting test_relationship_repository_factory with mock: {mock_adapter_class}")
        
        # Configure the mock
        mock_adapter_instance = MagicMock(spec=RelationshipRepositoryInterface)
        mock_adapter_class.return_value = mock_adapter_instance
        
        logger.debug(f"Configured mock_adapter_instance: {mock_adapter_instance}")
        
        # Create the repository
        logger.debug("Calling create_relationship_repository")
        repo = create_relationship_repository(self.mock_db)
        
        logger.debug(f"Repository created: {repo}")
        log_object_details(repo, "Repository")
        
        # Verify the repository is the correct type
        logger.debug(f"Verifying repo is instance of RelationshipRepositoryInterface")
        self.assertIsInstance(repo, RelationshipRepositoryInterface)
        
        logger.debug("test_relationship_repository_factory completed successfully")
    
    @patch('src.habitat_evolution.adaptive_core.persistence.adapters.topology_repository_adapter.TopologyRepositoryAdapter')
    def test_topology_repository_factory(self, mock_adapter_class):
        """Test that the topology repository factory correctly creates an adapter."""
        logger.debug(f"Starting test_topology_repository_factory with mock: {mock_adapter_class}")
        
        # Configure the mock
        mock_adapter_instance = MagicMock(spec=TopologyRepositoryInterface)
        mock_adapter_class.return_value = mock_adapter_instance
        
        logger.debug(f"Configured mock_adapter_instance: {mock_adapter_instance}")
        
        # Create the repository
        logger.debug("Calling create_topology_repository")
        repo = create_topology_repository(self.mock_db)
        
        logger.debug(f"Repository created: {repo}")
        log_object_details(repo, "Repository")
        
        # Verify the repository is the correct type
        logger.debug(f"Verifying repo is instance of TopologyRepositoryInterface")
        self.assertIsInstance(repo, TopologyRepositoryInterface)
        
        logger.debug("test_topology_repository_factory completed successfully")
    
    @patch('src.habitat_evolution.adaptive_core.persistence.factory.create_field_state_repository')
    @patch('src.habitat_evolution.adaptive_core.persistence.factory.create_pattern_repository')
    @patch('src.habitat_evolution.adaptive_core.persistence.factory.create_relationship_repository')
    @patch('src.habitat_evolution.adaptive_core.persistence.factory.create_topology_repository')
    def test_create_repositories(self, mock_topology, mock_relationship, mock_pattern, mock_field_state):
        """Test that the create_repositories function correctly creates all repositories."""
        logger.debug("Starting test_create_repositories")
        
        # Configure the mocks
        mock_field_state.return_value = self.mock_field_state_repo
        mock_pattern.return_value = self.mock_pattern_repo
        mock_relationship.return_value = self.mock_relationship_repo
        mock_topology.return_value = self.mock_topology_repo
        
        logger.debug("Configured all mocks")
        
        # Create the repositories
        logger.debug("Calling create_repositories")
        repos = create_repositories(self.mock_db)
        
        logger.debug(f"Repositories created: {repos}")
        
        # Verify all repositories were created
        logger.debug("Verifying all repositories were created correctly")
        self.assertEqual(repos["field_state_repository"], self.mock_field_state_repo)
        self.assertEqual(repos["pattern_repository"], self.mock_pattern_repo)
        self.assertEqual(repos["relationship_repository"], self.mock_relationship_repo)
        self.assertEqual(repos["topology_repository"], self.mock_topology_repo)
        
        # Verify all factory methods were called
        logger.debug("Verifying all factory methods were called correctly")
        mock_field_state.assert_called_once_with(self.mock_db, None)
        mock_pattern.assert_called_once_with(self.mock_db, None)
        mock_relationship.assert_called_once_with(self.mock_db, None)
        mock_topology.assert_called_once_with(self.mock_db, None)
        
        logger.debug("test_create_repositories completed successfully")


if __name__ == "__main__":
    # Print some system information
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Python path: {sys.path}")
    logger.debug(f"Current working directory: {os.getcwd()}")
    
    # Run the tests
    unittest.main()
