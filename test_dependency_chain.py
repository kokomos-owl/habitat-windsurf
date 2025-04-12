#!/usr/bin/env python
"""
Test script for demonstrating proper component initialization with dependency chain management.

This script shows how to properly initialize components with complex dependency chains
in the Habitat Evolution system, addressing the initialization issues identified in the error logs.
"""

import logging
import sys
import os
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import component initializer with error handling
try:
    from src.habitat_evolution.infrastructure.initialization.component_initializer import initialize_component
    COMPONENT_INITIALIZER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Component initializer not available: {e}")
    COMPONENT_INITIALIZER_AVAILABLE = False

# Import other components with error handling
try:
    from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
    ARANGODB_AVAILABLE = True
except ImportError as e:
    logger.error(f"ArangoDB connection not available: {e}")
    ARANGODB_AVAILABLE = False

try:
    from src.habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
    CLAUDE_ADAPTER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Claude adapter not available: {e}")
    CLAUDE_ADAPTER_AVAILABLE = False

def initialize_dependency_chain(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize the complete dependency chain for the Habitat Evolution system.
    
    This function demonstrates the proper order of initialization for components
    with complex dependencies, ensuring that each component has all required
    dependencies properly initialized before it is created.
    
    Args:
        config: Optional configuration for the components
        
    Returns:
        Dictionary containing all initialized components
    """
    components = {}
    config = config or {}
    
    # Check if component initializer is available
    if not COMPONENT_INITIALIZER_AVAILABLE:
        logger.error("Component initializer not available, cannot initialize dependency chain")
        return {
            "db_connection": None,
            "event_service": None,
            "claude_adapter": None,
            "vector_tonic_integrator": None,
            "event_bus": None,
            "harmonic_io_service": None,
            "pattern_repository": None,
            "pattern_aware_rag": None
        }
    
    # Step 1: Initialize foundation components
    logger.info("Step 1: Initializing foundation components")
    
    # Initialize ArangoDB connection
    if ARANGODB_AVAILABLE:
        try:
            db_config = config.get("db", {})
            db_connection = ArangoDBConnection(
                host=db_config.get("host", "localhost"),
                port=db_config.get("port", 8529),
                username=db_config.get("username", "root"),
                password=db_config.get("password", "habitat"),
                database_name=db_config.get("database_name", "habitat_evolution")
            )
            db_connection.initialize()
            components["db_connection"] = db_connection
            logger.info("ArangoDB connection initialized")
        except Exception as e:
            logger.error(f"Error initializing ArangoDB connection: {e}")
            components["db_connection"] = None
    else:
        logger.warning("ArangoDB not available, skipping connection initialization")
        components["db_connection"] = None
    
    # Step 2: Initialize service components
    logger.info("Step 2: Initializing service components")
    
    # Initialize EventService
    try:
        event_service = initialize_component("event_service", config.get("event_service"))
        components["event_service"] = event_service
        if event_service:
            logger.info("EventService initialized")
        else:
            logger.warning("EventService initialization returned None")
    except Exception as e:
        logger.error(f"Error initializing EventService: {e}")
        components["event_service"] = None
    
    # Initialize ClaudeAdapter
    if CLAUDE_ADAPTER_AVAILABLE:
        try:
            claude_adapter = ClaudeAdapter(api_key=config.get("claude_api_key"))
            components["claude_adapter"] = claude_adapter
            logger.info("ClaudeAdapter initialized")
        except Exception as e:
            logger.error(f"Error initializing ClaudeAdapter: {e}")
            components["claude_adapter"] = None
    else:
        logger.warning("ClaudeAdapter not available, skipping initialization")
        components["claude_adapter"] = None
    
    # Step 3: Initialize vector-tonic components
    logger.info("Step 3: Initializing vector-tonic components")
    
    try:
        # Initialize VectorTonicWindowIntegrator and related components
        vector_tonic_result = initialize_component(
            "vector_tonic",
            config.get("vector_tonic"),
            {"event_service": components["event_service"]}
        )
        
        # Unpack the result
        if vector_tonic_result and len(vector_tonic_result) == 3:
            vector_tonic_integrator, event_bus, harmonic_io_service = vector_tonic_result
            components["vector_tonic_integrator"] = vector_tonic_integrator
            components["event_bus"] = event_bus
            components["harmonic_io_service"] = harmonic_io_service
            
            if vector_tonic_integrator:
                logger.info("VectorTonicWindowIntegrator initialized")
            else:
                logger.warning("VectorTonicWindowIntegrator initialization returned None")
        else:
            logger.warning("Vector-tonic components initialization returned invalid result")
            components["vector_tonic_integrator"] = None
            components["event_bus"] = None
            components["harmonic_io_service"] = None
    except Exception as e:
        logger.error(f"Error initializing vector-tonic components: {e}")
        components["vector_tonic_integrator"] = None
        components["event_bus"] = None
        components["harmonic_io_service"] = None
    
    # Step 4: Initialize pattern-aware components
    logger.info("Step 4: Initializing pattern-aware components")
    
    # Mock pattern repository for testing
    try:
        class MockPatternRepository:
            def __init__(self):
                self._initialized = True
                logger.info("MockPatternRepository initialized")
                
            def find_by_id(self, pattern_id):
                return {"id": pattern_id, "content": "Mock pattern content"}
        
        pattern_repository = MockPatternRepository()
        components["pattern_repository"] = pattern_repository
    except Exception as e:
        logger.error(f"Error creating MockPatternRepository: {e}")
        components["pattern_repository"] = None
    
    # Initialize PatternAwareRAGService
    try:
        pattern_aware_rag = initialize_component(
            "pattern_aware_rag",
            config.get("pattern_aware_rag"),
            {
                "db_connection": components["db_connection"],
                "pattern_repository": components["pattern_repository"],
                "vector_tonic_service": components["vector_tonic_integrator"],
                "claude_adapter": components["claude_adapter"],
                "event_service": components["event_service"]
            }
        )
        components["pattern_aware_rag"] = pattern_aware_rag
        
        if pattern_aware_rag:
            logger.info("PatternAwareRAGService initialized")
        else:
            logger.warning("PatternAwareRAGService initialization returned None")
    except Exception as e:
        logger.error(f"Error initializing PatternAwareRAGService: {e}")
        components["pattern_aware_rag"] = None
    
    # Log initialization status
    logger.info("Dependency chain initialization complete")
    logger.info(f"Components initialized: {len([c for c in components.values() if c is not None])}/{len(components)}")
    
    return components

def main():
    """Run the dependency chain initialization test."""
    logger.info("Starting dependency chain initialization test")
    
    # Define test configuration
    test_config = {
        "db": {
            "host": "localhost",
            "port": 8529,
            "username": "root",
            "password": "habitat",
            "database_name": "habitat_evolution"
        },
        "event_service": {
            "buffer_size": 100,
            "flush_interval": 5
        },
        "vector_tonic": {
            "window_size": 10,
            "overlap": 2
        },
        "pattern_aware_rag": {
            "fallback_enabled": True,
            "max_patterns": 5
        }
    }
    
    # Initialize with test configuration
    logger.info("Initializing dependency chain with test configuration")
    components = initialize_dependency_chain(test_config)
    
    # Print component status with details
    logger.info("\n=== Component Initialization Status ===")
    for name, component in components.items():
        status = "SUCCESS" if component is not None else "FAILURE"
        component_type = type(component).__name__ if component is not None else "None"
        logger.info(f"{name}: {status} (Type: {component_type})")
    
    # Check if critical components were initialized
    critical_components = ["db_connection", "event_service", "claude_adapter", "pattern_aware_rag"]
    critical_initialized = [name for name in critical_components if components.get(name) is not None]
    critical_failed = [name for name in critical_components if components.get(name) is None]
    
    # Calculate success rate
    total_components = len(components)
    successful_components = len([c for c in components.values() if c is not None])
    success_rate = (successful_components / total_components) * 100 if total_components > 0 else 0
    
    # Print summary
    logger.info("\n=== Initialization Summary ===")
    logger.info(f"Total components: {total_components}")
    logger.info(f"Successfully initialized: {successful_components} ({success_rate:.1f}%)")
    logger.info(f"Critical components initialized: {len(critical_initialized)}/{len(critical_components)}")
    
    if critical_failed:
        logger.error(f"Failed critical components: {', '.join(critical_failed)}")
    
    # Determine exit code based on critical components
    critical_success = all(components.get(name) is not None for name in critical_components)
    
    if critical_success:
        logger.info("All critical components initialized successfully")
        logger.info("Test completed successfully")
        return 0
    else:
        logger.error("Some critical components failed to initialize")
        logger.info("Test completed with errors")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
