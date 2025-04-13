"""
Diagnostic test script for Habitat Evolution components.

This script provides detailed diagnostic information about the initialization
and verification status of all critical components in the Habitat Evolution system.
It uses the enhanced verification helpers to track errors in a more granular way.
"""

import os
import sys
import logging
import json
import pytest
from typing import Dict, Any, List

# Import our custom logging configuration
from tests.diagnostic.logging_config import configure_diagnostic_logging, log_component_status, log_exception_with_traceback

# Configure detailed logging
logger = configure_diagnostic_logging()

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import verification helpers
from tests.integration.climate_e2e.verification_helpers import (
    verify_all_critical_components,
    verify_component_initialization,
    verify_event_service,
    verify_vector_tonic_integration,
    verify_pattern_aware_rag,
    verify_claude_adapter
)

# Import the components we want to test - using the actual module paths
from habitat_evolution.infrastructure.services.event_service import EventService
from habitat_evolution.infrastructure.services.pattern_aware_rag_service import PatternAwareRAGService

# The VectorTonic integration is in the emergence module, not infrastructure
from habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegration

# Claude adapter is in the adapters directory
from habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter

# ArangoDB connection is in the persistence directory
from habitat_evolution.infrastructure.persistence.arangodb.connection import ArangoDBConnection

class TestComponentDiagnostics:
    """
    Diagnostic tests for Habitat Evolution components.
    
    These tests provide detailed information about the initialization and
    verification status of all critical components in the Habitat Evolution system.
    """
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create test context with all components
        self.test_context = {}
        self.initialization_errors = {}
        
        # Track component initialization order and dependencies
        self.initialization_order = []
        self.dependency_map = {}
        
        logger.info("=== STARTING COMPONENT INITIALIZATION ===")
        
        # Initialize foundation components first
        self._init_component('arangodb_connection', ArangoDBConnection, [])
        self._init_component('event_service', EventService, [])
        self._init_component('claude_adapter', ClaudeAdapter, [], init_args={'api_key': 'mock_api_key_for_testing'})
        
        # Initialize repositories that depend on ArangoDB
        # We'll need to import these here
        try:
            from habitat_evolution.infrastructure.repositories.pattern_repository import PatternRepository
            from habitat_evolution.infrastructure.repositories.relationship_repository import RelationshipRepository
            
            self._init_component('pattern_repository', PatternRepository, ['arangodb_connection'])
            self._init_component('relationship_repository', RelationshipRepository, ['arangodb_connection'])
        except ImportError as e:
            logger.error(f"Failed to import repositories: {e}")
        
        # Initialize Vector Tonic Integrator (depends on EventService)
        self._init_component('vector_tonic_integrator', VectorTonicWindowIntegration, ['event_service'])
        
        # Initialize PatternAwareRAG (depends on multiple components)
        self._init_component('pattern_aware_rag', PatternAwareRAGService, 
                            ['claude_adapter', 'pattern_repository', 'relationship_repository'])
        
        logger.info("=== COMPONENT INITIALIZATION COMPLETE ===")
        logger.info(f"Initialization order: {self.initialization_order}")
        logger.info(f"Dependency map: {self.dependency_map}")
        
    def _init_component(self, component_key, component_class, dependencies, init_args=None):
        """Initialize a component with proper dependency tracking and error logging."""
        logger.info(f"Initializing {component_key}...")
        
        # Check if dependencies are satisfied
        missing_deps = []
        for dep in dependencies:
            if dep not in self.test_context or self.test_context[dep] is None:
                missing_deps.append(dep)
                logger.error(f"Missing dependency for {component_key}: {dep}")
        
        if missing_deps:
            logger.error(f"Cannot initialize {component_key} due to missing dependencies: {missing_deps}")
            self.test_context[component_key] = None
            self.initialization_errors[component_key] = f"Missing dependencies: {missing_deps}"
            return
        
        # Store dependency information
        self.dependency_map[component_key] = dependencies
        
        # Create the component
        try:
            logger.debug(f"Creating {component_key} instance...")
            component = component_class()
            self.test_context[component_key] = component
            logger.debug(f"{component_key} instance created")
            
            # Log detailed component status
            log_component_status(logger, component, component_key)
            
            # Initialize the component if it has an initialize method
            if hasattr(component, 'initialize') and callable(component.initialize):
                try:
                    logger.debug(f"Calling initialize() on {component_key}...")
                    
                    # Pass dependencies to initialize if needed
                    if init_args:
                        component.initialize(**init_args)
                    else:
                        # Automatically pass dependencies based on parameter names
                        import inspect
                        sig = inspect.signature(component.initialize)
                        auto_args = {}
                        
                        for param_name in sig.parameters:
                            if param_name in self.test_context and self.test_context[param_name] is not None:
                                auto_args[param_name] = self.test_context[param_name]
                        
                        if auto_args:
                            logger.debug(f"Auto-passing dependencies to {component_key}.initialize(): {list(auto_args.keys())}")
                            component.initialize(**auto_args)
                        else:
                            component.initialize()
                    
                    logger.debug(f"{component_key} initialized successfully")
                    
                    # Record successful initialization
                    self.initialization_order.append(component_key)
                    
                    # Log detailed component status after initialization
                    log_component_status(logger, component, f"{component_key} (after initialization)")
                    
                except Exception as e:
                    log_exception_with_traceback(logger, e, f"Failed to initialize {component_key}")
                    self.initialization_errors[component_key] = f"Initialization error: {str(e)}"
            else:
                logger.debug(f"{component_key} has no initialize method")
                # Still record it in the initialization order
                self.initialization_order.append(component_key)
        except Exception as e:
            log_exception_with_traceback(logger, e, f"Failed to create {component_key}")
            self.test_context[component_key] = None
            self.initialization_errors[component_key] = f"Creation error: {str(e)}"
    
    def test_component_diagnostic(self):
        """
        Run diagnostic tests on all components to identify initialization issues.
        
        This test captures detailed information about each component's verification status
        and logs it for analysis.
        """
        logger.info("=== RUNNING COMPONENT DIAGNOSTIC TEST ===")
        
        # First, log any initialization errors that occurred during setup
        if self.initialization_errors:
            logger.warning("The following components had initialization errors:")
            for component, error in self.initialization_errors.items():
                logger.warning(f"  {component}: {error}")
        
        # Run verification in diagnostic mode (non-strict)
        diagnostic_results = verify_all_critical_components(
            self.test_context, 
            strict=False,
            diagnostic=True
        )
        
        # Log the diagnostic results
        logger.info("Diagnostic results:")
        for component_name, result in diagnostic_results.items():
            logger.info(f"Component: {component_name}")
            logger.info(f"  Present: {result.get('present', False)}")
            logger.info(f"  Initialized: {result.get('initialized', None)}")
            logger.info(f"  Verification passed: {result.get('verification_passed', False)}")
            
            # Log attribute status
            if 'attributes' in result:
                logger.info("  Attributes:")
                for attr, status in result['attributes'].items():
                    logger.info(f"    {attr}: {status}")
            
            # Log method status
            if 'methods' in result:
                logger.info("  Methods:")
                for method, status in result['methods'].items():
                    logger.info(f"    {method}: {status}")
            
            # Log errors
            if 'errors' in result and result['errors']:
                logger.info("  Errors:")
                for error in result['errors']:
                    logger.info(f"    - {error}")
        
        # Create a comprehensive diagnostic report
        diagnostic_report = {
            'timestamp': datetime.now().isoformat(),
            'initialization_order': self.initialization_order,
            'dependency_map': self.dependency_map,
            'initialization_errors': self.initialization_errors,
            'verification_results': diagnostic_results
        }
        
        # Save diagnostic results to a JSON file for further analysis
        with open('component_diagnostic_report.json', 'w') as f:
            json.dump(diagnostic_report, f, indent=2)
        
        logger.info("Comprehensive diagnostic report saved to component_diagnostic_report.json")
        
        # Assert that the test ran successfully (not that all components passed)
        assert True

    def test_dependency_chain_analysis(self):
        """
        Analyze the dependency chain of components.
        
        This test attempts to identify the dependency chain of components
        and logs any issues found.
        """
        logger.info("=== ANALYZING COMPONENT DEPENDENCY CHAIN ===")
        
        # Define the expected dependency chain based on our knowledge of the system
        expected_dependencies = {
            'arangodb_connection': [],
            'event_service': [],
            'claude_adapter': [],
            'pattern_repository': ['arangodb_connection'],
            'relationship_repository': ['arangodb_connection'],
            'vector_tonic_integrator': {
                'function': verify_vector_tonic_integration,
                'name': 'Vector Tonic Integration',
                'required_attributes': ['_initialized', 'process_window'],
                'required_methods': ['process_window']
            },
            'pattern_aware_rag': ['claude_adapter', 'pattern_repository', 'relationship_repository']
        }
        
        # Compare actual dependencies with expected dependencies
        dependency_analysis = {}
        
        for component, expected_deps in expected_dependencies.items():
            analysis = {
                'component': component,
                'exists': component in self.test_context and self.test_context[component] is not None,
                'expected_dependencies': expected_deps,
                'actual_dependencies': self.dependency_map.get(component, []),
                'missing_dependencies': [],
                'unexpected_dependencies': [],
                'initialization_error': self.initialization_errors.get(component, None),
                'initialized': False
            }
            
            # Check if component exists and is initialized
            if analysis['exists']:
                component_obj = self.test_context[component]
                if hasattr(component_obj, '_initialized'):
                    analysis['initialized'] = component_obj._initialized
                elif hasattr(component_obj, 'is_initialized') and callable(getattr(component_obj, 'is_initialized')):
                    try:
                        analysis['initialized'] = component_obj.is_initialized()
                    except Exception:
                        analysis['initialized'] = False
            
            # Check for missing dependencies
            for dep in expected_deps:
                if dep not in self.dependency_map.get(component, []):
                    analysis['missing_dependencies'].append(dep)
            
            # Check for unexpected dependencies
            for dep in self.dependency_map.get(component, []):
                if dep not in expected_deps:
                    analysis['unexpected_dependencies'].append(dep)
            
            dependency_analysis[component] = analysis
        
        # Log the dependency analysis
        logger.info("Dependency chain analysis:")
        for component, analysis in dependency_analysis.items():
            logger.info(f"Component: {component}")
            logger.info(f"  Exists: {analysis['exists']}")
            logger.info(f"  Initialized: {analysis['initialized']}")
            logger.info(f"  Expected dependencies: {analysis['expected_dependencies']}")
            logger.info(f"  Actual dependencies: {analysis['actual_dependencies']}")
            
            if analysis['missing_dependencies']:
                logger.warning(f"  Missing dependencies: {analysis['missing_dependencies']}")
            
            if analysis['unexpected_dependencies']:
                logger.warning(f"  Unexpected dependencies: {analysis['unexpected_dependencies']}")
            
            if analysis['initialization_error']:
                logger.error(f"  Initialization error: {analysis['initialization_error']}")
        
        # Identify circular dependencies
        def find_circular_dependencies(dependency_map):
            visited = set()
            path = []
            circular_deps = []
            
            def dfs(node):
                if node in path:
                    circular_deps.append(path[path.index(node):] + [node])
                    return
                
                if node in visited:
                    return
                
                visited.add(node)
                path.append(node)
                
                for neighbor in dependency_map.get(node, []):
                    dfs(neighbor)
                
                path.pop()
            
            for node in dependency_map:
                dfs(node)
            
            return circular_deps
        
        circular_dependencies = find_circular_dependencies(self.dependency_map)
        if circular_dependencies:
            logger.error("Circular dependencies detected:")
            for cycle in circular_dependencies:
                logger.error(f"  {' -> '.join(cycle)}")
        else:
            logger.info("No circular dependencies detected")
        
        # Save dependency analysis to a JSON file
        with open('dependency_analysis.json', 'w') as f:
            json.dump(dependency_analysis, f, indent=2)
        
        logger.info("Dependency analysis saved to dependency_analysis.json")
        
        # Assert that the test ran successfully
        assert True

    def test_component_method_tracing(self):
        """
        Trace method calls on critical components to identify initialization issues.
        
        This test attempts to call key methods on each component and logs any issues found.
        """
        logger.info("=== TRACING COMPONENT METHOD CALLS ===")
        
        # Define key methods to test for each component type
        test_methods = {
            'event_service': [
                {'method': 'publish', 'args': ['test.event', {'test': 'data'}]},
                {'method': 'subscribe', 'args': ['test.event', lambda x: None]}
            ],
            'claude_adapter': [
                {'method': 'query', 'args': ['Test query']}
            ],
            'pattern_aware_rag': [
                {'method': 'query', 'args': ['Test query']}
            ],
            'vector_tonic_integrator': [
                {'method': 'process_window', 'args': [{'data': 'test'}]}
            ]
        }
        
        method_results = {}
        
        # Test methods on each component
        for component_key, methods in test_methods.items():
            if component_key not in self.test_context or self.test_context[component_key] is None:
                logger.warning(f"Component {component_key} not available for method tracing")
                continue
            
            component = self.test_context[component_key]
            method_results[component_key] = []
            
            for method_info in methods:
                method_name = method_info['method']
                args = method_info['args']
                
                logger.info(f"Testing {component_key}.{method_name}()...")
                
                if not hasattr(component, method_name):
                    logger.error(f"Component {component_key} does not have method {method_name}")
                    method_results[component_key].append({
                        'method': method_name,
                        'success': False,
                        'error': f"Method {method_name} not found"
                    })
                    continue
                
                method = getattr(component, method_name)
                if not callable(method):
                    logger.error(f"Component {component_key}.{method_name} is not callable")
                    method_results[component_key].append({
                        'method': method_name,
                        'success': False,
                        'error': f"Attribute {method_name} is not callable"
                    })
                    continue
                
                try:
                    logger.debug(f"Calling {component_key}.{method_name}() with args: {args}")
                    result = method(*args)
                    logger.debug(f"Method call successful, result type: {type(result)}")
                    method_results[component_key].append({
                        'method': method_name,
                        'success': True,
                        'result_type': str(type(result))
                    })
                except Exception as e:
                    log_exception_with_traceback(logger, e, f"Error calling {component_key}.{method_name}()")
                    method_results[component_key].append({
                        'method': method_name,
                        'success': False,
                        'error': str(e)
                    })
        
        # Save method tracing results to a JSON file
        with open('method_tracing_results.json', 'w') as f:
            json.dump(method_results, f, indent=2)
        
        logger.info("Method tracing results saved to method_tracing_results.json")
        
        # Assert that the test ran successfully
        assert True

if __name__ == "__main__":
    # Run the tests directly if this script is executed
    pytest.main(["-v", __file__])
