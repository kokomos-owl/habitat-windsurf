"""
System Initializer for Habitat Evolution.

This module provides utilities for initializing the entire Habitat Evolution system
in the correct order, tracking dependencies and errors during initialization.
"""

import logging
import json
import os
import traceback
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime

from src.habitat_evolution.infrastructure.initialization.dependency_tracker import (
    get_dependency_tracker,
    verify_initialization,
    verify_dependencies,
    initialize_with_dependencies
)

logger = logging.getLogger(__name__)

class SystemInitializer:
    """
    Initializes the Habitat Evolution system components in the correct order.
    
    This class provides utilities for initializing all components in the system,
    tracking dependencies and errors during initialization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new system initializer.
        
        Args:
            config: Optional configuration for the system
        """
        self.config = config or {}
        self.components = {}
        self.tracker = get_dependency_tracker()
        self.initialization_errors = {}
        
        logger.debug("SystemInitializer created")
    
    def initialize_system(self, log_dir: str = "logs") -> bool:
        """
        Initialize all components in the Habitat Evolution system.
        
        Args:
            log_dir: Directory to store initialization logs
            
        Returns:
            bool: True if all components were initialized successfully
        """
        logger.info("=== Starting Habitat Evolution System Initialization ===")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"system_initialization_{timestamp}.json")
        
        try:
            # Step 1: Initialize ArangoDB Connection
            logger.info("Step 1: Initializing ArangoDB Connection")
            success = self._initialize_arangodb_connection()
            if not success:
                logger.error("Failed to initialize ArangoDB Connection, aborting initialization")
                self._save_initialization_report(log_file)
                return False
            
            # Step 2: Initialize Event Service
            logger.info("Step 2: Initializing Event Service")
            success = self._initialize_event_service()
            if not success:
                logger.error("Failed to initialize Event Service, aborting initialization")
                self._save_initialization_report(log_file)
                return False
            
            # Step 3: Initialize Claude Adapter
            logger.info("Step 3: Initializing Claude Adapter")
            success = self._initialize_claude_adapter()
            if not success:
                logger.error("Failed to initialize Claude Adapter, aborting initialization")
                self._save_initialization_report(log_file)
                return False
            
            # Step 4: Initialize Repositories
            logger.info("Step 4: Initializing Repositories")
            success = self._initialize_repositories()
            if not success:
                logger.error("Failed to initialize Repositories, aborting initialization")
                self._save_initialization_report(log_file)
                return False
            
            # Step 5: Initialize Vector Tonic Service
            logger.info("Step 5: Initializing Vector Tonic Service")
            success = self._initialize_vector_tonic_service()
            if not success:
                logger.error("Failed to initialize Vector Tonic Service, aborting initialization")
                self._save_initialization_report(log_file)
                return False
            
            # Step 6: Initialize PatternAwareRAG Service
            logger.info("Step 6: Initializing PatternAwareRAG Service")
            success = self._initialize_pattern_aware_rag_service()
            if not success:
                logger.error("Failed to initialize PatternAwareRAG Service, aborting initialization")
                self._save_initialization_report(log_file)
                return False
            
            # Save initialization report
            self._save_initialization_report(log_file)
            
            logger.info("=== Habitat Evolution System Initialization Complete ===")
            return True
            
        except Exception as e:
            logger.error(f"Unhandled exception during system initialization: {e}")
            logger.error(traceback.format_exc())
            self._save_initialization_report(log_file)
            return False
    
    def _initialize_arangodb_connection(self) -> bool:
        """
        Initialize the ArangoDB connection.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Import the ArangoDB connection class
            try:
                from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
                logger.debug("Imported ArangoDBConnection from infrastructure.persistence.arangodb")
            except ImportError:
                logger.warning("Could not import ArangoDBConnection from infrastructure.persistence.arangodb")
                try:
                    from src.habitat_evolution.infrastructure.db.arangodb_connection import ArangoDBConnection
                    logger.debug("Imported ArangoDBConnection from infrastructure.db")
                except ImportError as e:
                    logger.error(f"Could not import ArangoDBConnection: {e}")
                    self.initialization_errors['arangodb_connection'] = f"Import error: {e}"
                    return False
            
            # Create the ArangoDB connection with configuration parameters
            db_config = self.config.get('arangodb', {})
            try:
                # Pass configuration parameters to the constructor
                db_connection = ArangoDBConnection(
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 8529),
                    username=db_config.get('username', 'root'),
                    password=db_config.get('password', ''),
                    database_name=db_config.get('database', 'habitat_evolution')
                )
                
                # Register with dependency tracker
                self.tracker.register_component('arangodb_connection', db_connection, [])
                
                # Initialize the connection
                logger.debug("Initializing ArangoDB connection")
                db_connection.initialize()
                logger.debug("ArangoDB connection initialized successfully")
                
                # Store the connection
                self.components['arangodb_connection'] = db_connection
                
                # Mark as initialized in the tracker
                self.tracker.mark_initialized('arangodb_connection', True)
                
                return True
            except Exception as e:
                logger.error(f"Error initializing ArangoDB connection: {e}")
                logger.error(traceback.format_exc())
                self.initialization_errors['arangodb_connection'] = f"Initialization error: {e}"
                self.tracker.mark_initialized('arangodb_connection', False, str(e))
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error initializing ArangoDB connection: {e}")
            logger.error(traceback.format_exc())
            self.initialization_errors['arangodb_connection'] = f"Unexpected error: {e}"
            return False
    
    def _initialize_event_service(self) -> bool:
        """
        Initialize the Event Service.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Import the Event Service class
            from src.habitat_evolution.infrastructure.services.event_service import EventService
            
            # Create the Event Service
            event_service = EventService()
            
            # Register with dependency tracker
            self.tracker.register_component('event_service', event_service, [])
            
            # Initialize the service
            try:
                logger.debug("Initializing Event Service")
                event_service.initialize()
                logger.debug("Event Service initialized successfully")
                
                # Store the service
                self.components['event_service'] = event_service
                
                # Mark as initialized in the tracker
                self.tracker.mark_initialized('event_service', True)
                
                return True
            except Exception as e:
                logger.error(f"Error initializing Event Service: {e}")
                logger.error(traceback.format_exc())
                self.initialization_errors['event_service'] = f"Initialization error: {e}"
                self.tracker.mark_initialized('event_service', False, str(e))
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error initializing Event Service: {e}")
            logger.error(traceback.format_exc())
            self.initialization_errors['event_service'] = f"Unexpected error: {e}"
            return False
    
    def _initialize_claude_adapter(self) -> bool:
        """
        Initialize the Claude Adapter.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Import the Claude Adapter class
            try:
                from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
                logger.debug("Imported ClaudeAdapter from infrastructure.adapters")
            except ImportError:
                logger.warning("Could not import ClaudeAdapter from infrastructure.adapters")
                try:
                    from src.habitat_evolution.infrastructure.services.claude_adapter import ClaudeAdapter
                    logger.debug("Imported ClaudeAdapter from infrastructure.services")
                except ImportError as e:
                    logger.error(f"Could not import ClaudeAdapter: {e}")
                    self.initialization_errors['claude_adapter'] = f"Import error: {e}"
                    return False
            
            # Create the Claude Adapter
            claude_config = self.config.get('claude', {})
            api_key = claude_config.get('api_key', 'mock_api_key_for_testing')
            claude_adapter = ClaudeAdapter()
            
            # Register with dependency tracker
            self.tracker.register_component('claude_adapter', claude_adapter, [])
            
            # Claude Adapter is already initialized in its constructor
            # Just mark it as initialized in our dependency tracker
            try:
                logger.debug("Claude Adapter already initialized in constructor")
                # No need to call initialize() as it doesn't exist
                logger.debug("Claude Adapter initialized successfully")
                
                # Store the adapter
                self.components['claude_adapter'] = claude_adapter
                
                # Mark as initialized in the tracker
                self.tracker.mark_initialized('claude_adapter', True)
                
                return True
            except Exception as e:
                logger.error(f"Error initializing Claude Adapter: {e}")
                logger.error(traceback.format_exc())
                self.initialization_errors['claude_adapter'] = f"Initialization error: {e}"
                self.tracker.mark_initialized('claude_adapter', False, str(e))
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error initializing Claude Adapter: {e}")
            logger.error(traceback.format_exc())
            self.initialization_errors['claude_adapter'] = f"Unexpected error: {e}"
            return False
    
    def _initialize_repositories(self) -> bool:
        """
        Initialize the Pattern and Relationship Repositories.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Check if ArangoDB connection is available
            if 'arangodb_connection' not in self.components:
                logger.error("Cannot initialize repositories: ArangoDB connection not available")
                self.initialization_errors['repositories'] = "Missing dependency: arangodb_connection"
                return False
            
            db_connection = self.components['arangodb_connection']
            
            # Import the repository classes
            try:
                from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository
                from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_graph_repository import ArangoDBGraphRepository
                logger.debug("Imported repository classes from infrastructure.persistence.arangodb")
            except ImportError as e:
                logger.error(f"Could not import repository classes: {e}")
                self.initialization_errors['repositories'] = f"Import error: {e}"
                return False
                
            # Initialize Pattern Repository
            try:
                logger.debug("Initializing Pattern Repository")
                # Get the dependencies
                db_connection = self.components.get('arangodb_connection')
                event_service = self.components.get('event_service')
                
                # Create the repository with dependencies
                pattern_repository = ArangoDBPatternRepository(
                    db_connection=db_connection,
                    event_service=event_service
                )
                
                # Register with dependency tracker
                self.tracker.register_component('pattern_repository', pattern_repository, 
                                              ['arangodb_connection', 'event_service'])
                
                # Store the repository
                self.components['pattern_repository'] = pattern_repository
                
                # Mark as initialized in the tracker
                self.tracker.mark_initialized('pattern_repository', True)
            except Exception as e:
                logger.error(f"Error initializing Pattern Repository: {e}")
                logger.error(traceback.format_exc())
                self.initialization_errors['pattern_repository'] = f"Initialization error: {e}"
                self.tracker.mark_initialized('pattern_repository', False, str(e))
                return False
            
            # Initialize Relationship Repository
            try:
                logger.debug("Initializing Relationship Repository")
                # Get the dependencies
                db_connection = self.components.get('arangodb_connection')
                event_service = self.components.get('event_service')
                
                # Create the repository with dependencies
                relationship_repository = ArangoDBGraphRepository(
                    node_collection_name="entities",
                    edge_collection_name="relationships",
                    graph_name="entity_graph",
                    db_connection=db_connection,
                    event_service=event_service,
                    entity_class=dict  # Using dict as a generic entity class
                )
                
                # Register with dependency tracker
                self.tracker.register_component('relationship_repository', relationship_repository, 
                                               ['arangodb_connection', 'event_service'])
                logger.debug("Relationship Repository initialized successfully")
                
                # Store the repository
                self.components['relationship_repository'] = relationship_repository
                
                # Mark as initialized in the tracker
                self.tracker.mark_initialized('relationship_repository', True)
            except Exception as e:
                logger.error(f"Error initializing Relationship Repository: {e}")
                logger.error(traceback.format_exc())
                self.initialization_errors['relationship_repository'] = f"Initialization error: {e}"
                self.tracker.mark_initialized('relationship_repository', False, str(e))
                return False
            
            return True
                
        except Exception as e:
            logger.error(f"Unexpected error initializing repositories: {e}")
            logger.error(traceback.format_exc())
            self.initialization_errors['repositories'] = f"Unexpected error: {e}"
            return False
    
    def _initialize_vector_tonic_service(self) -> bool:
        """
        Initialize the Vector Tonic Service.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Check if dependencies are available
            required_deps = ['arangodb_connection', 'event_service', 'pattern_repository']
            missing_deps = []
            
            for dep in required_deps:
                if dep not in self.components:
                    missing_deps.append(dep)
            
            if missing_deps:
                logger.error(f"Cannot initialize Vector Tonic Service: Missing dependencies: {missing_deps}")
                self.initialization_errors['vector_tonic_service'] = f"Missing dependencies: {missing_deps}"
                return False
            
            # For testing purposes, use the MockVectorTonicService to avoid ArangoDB client compatibility issues
            use_mock = self.config.get('use_mock_services', True)  # Default to mock for testing
            
            # Import the appropriate Vector Tonic Service implementation
            try:
                if use_mock:
                    from src.habitat_evolution.infrastructure.services.mock_vector_tonic_service import MockVectorTonicService as VectorTonicServiceClass
                    logger.debug("Using MockVectorTonicService for testing")
                else:
                    from src.habitat_evolution.infrastructure.services.vector_tonic_service import VectorTonicService as VectorTonicServiceClass
                    logger.debug("Using real VectorTonicService")
            except ImportError as e:
                logger.error(f"Could not import Vector Tonic Service: {e}")
                self.initialization_errors['vector_tonic_service'] = f"Import error: {e}"
                return False
                
            # Create the Vector Tonic Service
            try:
                vector_tonic_service = VectorTonicServiceClass(
                    db_connection=self.components['arangodb_connection'],
                    event_service=self.components['event_service'],
                    pattern_repository=self.components['pattern_repository']
                )
                
                # Register with dependency tracker
                self.tracker.register_component('vector_tonic_service', vector_tonic_service, 
                                              ['arangodb_connection', 'event_service', 'pattern_repository'])
                
                # Initialize the service
                logger.debug("Initializing Vector Tonic Service")
                vector_tonic_service.initialize()
                logger.debug("Vector Tonic Service initialized successfully")
                
                # Store the service
                self.components['vector_tonic_service'] = vector_tonic_service
                
                # Mark as initialized in the tracker
                self.tracker.mark_initialized('vector_tonic_service', True)
                
                return True
            except Exception as e:
                logger.error(f"Error initializing Vector Tonic Service: {e}")
                logger.error(traceback.format_exc())
                self.initialization_errors['vector_tonic_service'] = f"Initialization error: {e}"
                self.tracker.mark_initialized('vector_tonic_service', False, str(e))
                return False
        except Exception as e:
            logger.error(f"Unexpected error initializing Vector Tonic Service: {e}")
            logger.error(traceback.format_exc())
            self.initialization_errors['vector_tonic_service'] = f"Unexpected error: {e}"
            return False
    def _initialize_pattern_aware_rag_service(self) -> bool:
        """
        Initialize the PatternAwareRAG Service.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Check if dependencies are available
            required_deps = ['arangodb_connection', 'pattern_repository', 'relationship_repository', 
                           'vector_tonic_service', 'claude_adapter', 'event_service']
            missing_deps = []
            
            for dep in required_deps:
                if dep not in self.components:
                    missing_deps.append(dep)
            
            if missing_deps:
                logger.error(f"Cannot initialize PatternAwareRAG Service: Missing dependencies: {missing_deps}")
                self.initialization_errors['pattern_aware_rag'] = f"Missing dependencies: {missing_deps}"
                return False
            
            # Import the PatternAwareRAG Service class
            try:
                from src.habitat_evolution.infrastructure.services.pattern_aware_rag_service import PatternAwareRAGService
            except ImportError as e:
                logger.error(f"Could not import PatternAwareRAGService: {e}")
                self.initialization_errors['pattern_aware_rag'] = f"Import error: {e}"
                return False
            
            # Create the PatternAwareRAG Service
            pattern_aware_rag = PatternAwareRAGService(
                db_connection=self.components['arangodb_connection'],
                pattern_repository=self.components['pattern_repository'],
                vector_tonic_service=self.components['vector_tonic_service'],
                claude_adapter=self.components['claude_adapter'],
                event_service=self.components['event_service']
            )
            
            # Register with dependency tracker
            self.tracker.register_component('pattern_aware_rag', pattern_aware_rag, 
                                          ['arangodb_connection', 'pattern_repository', 'relationship_repository',
                                           'vector_tonic_service', 'claude_adapter', 'event_service'])
            
            # Initialize the service
            try:
                logger.debug("Initializing PatternAwareRAG Service")
                pattern_aware_rag.initialize()
                logger.debug("PatternAwareRAG Service initialized successfully")
                
                # Store the service
                self.components['pattern_aware_rag'] = pattern_aware_rag
                
                # Mark as initialized in the tracker
                self.tracker.mark_initialized('pattern_aware_rag', True)
                
                return True
            except Exception as e:
                logger.error(f"Error initializing PatternAwareRAG Service: {e}")
                logger.error(traceback.format_exc())
                self.initialization_errors['pattern_aware_rag'] = f"Initialization error: {e}"
                self.tracker.mark_initialized('pattern_aware_rag', False, str(e))
                return False
        except Exception as e:
            logger.error(f"Unexpected error initializing PatternAwareRAG Service: {e}")
            logger.error(traceback.format_exc())
            self.initialization_errors['pattern_aware_rag'] = f"Unexpected error: {e}"
            return False
    
    def _save_initialization_report(self, log_file: str) -> None:
        """
        Save the initialization report to a file.
        
        Args:
            log_file: Path to the log file
        """
        # Generate the report
        report = self.tracker.generate_initialization_report()
        
        # Add initialization errors
        report['initialization_errors'] = self.initialization_errors
        
        # Add timestamp
        report['timestamp'] = datetime.now().isoformat()
        
        # Save the report
        with open(log_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Initialization report saved to {log_file}")
    
    def get_components(self) -> Dict[str, Any]:
        """
        Get all initialized components.
        
        Returns:
            Dictionary mapping component keys to component instances
        """
        return self.components.copy()
    
    def get_component(self, component_key: str) -> Any:
        """
        Get a specific component.
        
        Args:
            component_key: The key of the component to get
            
        Returns:
            The component instance, or None if not found
        """
        return self.components.get(component_key)
    
    def get_initialization_errors(self) -> Dict[str, str]:
        """
        Get the initialization errors.
        
        Returns:
            Dictionary mapping component keys to error messages
        """
        return self.initialization_errors.copy()


def initialize_system(config: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any], Dict[str, str]]:
    """
    Initialize the Habitat Evolution system.
    
    Args:
        config: Optional configuration for the system
        
    Returns:
        Tuple containing:
            - Boolean indicating if initialization was successful
            - Dictionary of initialized components
            - Dictionary of initialization errors
    """
    initializer = SystemInitializer(config)
    success = initializer.initialize_system()
    
    return success, initializer.get_components(), initializer.get_initialization_errors()
