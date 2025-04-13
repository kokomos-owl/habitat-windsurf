#!/usr/bin/env python
"""
Habitat Evolution Component Error Tracker

This script provides a detailed diagnostic approach to track errors in component initialization
and dependency chains in the Habitat Evolution system. It uses enhanced logging and
step-by-step verification to identify exactly where issues are occurring.
"""

import os
import sys
import logging
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"component_error_tracking_{timestamp}.log")

# Configure root logger with detailed formatting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("error_tracker")
logger.info(f"Starting component error tracking at {datetime.now().isoformat()}")
logger.info(f"Log file: {log_file}")

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Define our own verification helpers since we can't import them directly

def verify_component_initialization(component, component_name):
    """Verify that a component is properly initialized."""
    logger.debug(f"Verifying initialization of {component_name}")
    
    # Check if component exists
    if component is None:
        logger.error(f"Critical component {component_name} is None")
        return False
    
    # Log component type for debugging
    component_type = type(component).__name__
    logger.debug(f"Component {component_name} is of type {component_type}")
    
    # Check if component has _initialized attribute
    initialized_via_attribute = False
    if hasattr(component, '_initialized'):
        logger.debug(f"Component {component_name} has _initialized attribute: {component._initialized}")
        if not component._initialized:
            logger.error(f"Critical component {component_name} reports not initialized (_initialized is False)")
            return False
        initialized_via_attribute = True
    else:
        logger.debug(f"Component {component_name} does not have _initialized attribute")
    
    # Check if component has is_initialized() method
    initialized_via_method = False
    if hasattr(component, 'is_initialized') and callable(component.is_initialized):
        try:
            is_init = component.is_initialized()
            logger.debug(f"Component {component_name} is_initialized() returned: {is_init}")
            if not is_init:
                logger.error(f"Critical component {component_name} reports not initialized (is_initialized() returned False)")
                return False
            initialized_via_method = True
        except Exception as e:
            logger.error(f"Error calling is_initialized() on {component_name}: {e}")
            return False
    else:
        logger.debug(f"Component {component_name} does not have is_initialized() method")
    
    # Ensure at least one initialization check passed
    if not (initialized_via_attribute or initialized_via_method):
        logger.warning(f"Component {component_name} has no initialization status indicators")
        # If it's a mock component in a test, we'll be more lenient
        if component_type == 'MockComponent':
            logger.debug(f"MockComponent detected, assuming it's initialized for testing purposes")
        else:
            logger.error(f"Cannot verify initialization status of {component_name}")
            return False
    
    logger.debug(f"Component {component_name} is properly initialized")
    return True

def verify_event_service(event_service):
    """Verify that the EventService is properly initialized."""
    logger.debug("Verifying EventService initialization")
    
    # Step 1: Verify the EventService is not None
    if event_service is None:
        logger.error("EventService is None")
        return False
    
    # Step 2: Verify the EventService has _initialized attribute
    if not hasattr(event_service, '_initialized'):
        logger.error("EventService missing _initialized attribute")
        return False
    
    # Step 3: Verify the EventService is initialized
    if not event_service._initialized:
        logger.error("EventService is not initialized (_initialized is False)")
        return False
    
    # Step 4: Verify the EventService has the required methods
    required_methods = ['publish', 'subscribe', 'unsubscribe']
    for method in required_methods:
        if not hasattr(event_service, method):
            logger.error(f"EventService missing required method: {method}")
            return False
        if not callable(getattr(event_service, method)):
            logger.error(f"EventService attribute {method} is not callable")
            return False
    
    # Step 5: Verify the EventService can publish events (only if it's a real EventService)
    if event_service.__class__.__name__ != 'MockComponent':
        test_event_type = "test_event"
        test_event_data = {"test": "data"}
        try:
            event_service.publish(test_event_type, test_event_data)
            logger.debug("Successfully published test event")
        except Exception as e:
            logger.error(f"EventService failed to publish events: {e}")
            return False
    
    logger.debug("EventService is properly initialized")
    return True

def verify_vector_tonic_integration(vector_tonic_integrator):
    """Verify that Vector Tonic integration is properly initialized."""
    logger.debug("Verifying Vector Tonic Integration initialization")
    
    # Verify the Vector Tonic Integration is not None
    if not verify_component_initialization(vector_tonic_integrator, "Vector Tonic Integration"):
        return False
    
    # Verify the Vector Tonic Integration has the required methods
    required_methods = ['process_window']
    for method in required_methods:
        if not hasattr(vector_tonic_integrator, method):
            logger.error(f"Vector Tonic Integration missing required method: {method}")
            return False
        if not callable(getattr(vector_tonic_integrator, method)):
            logger.error(f"Vector Tonic Integration attribute {method} is not callable")
            return False
    
    logger.debug("Vector Tonic Integration is properly initialized")
    return True

def verify_pattern_aware_rag(pattern_aware_rag):
    """Verify that the PatternAwareRAG service is properly initialized."""
    logger.debug("Verifying PatternAwareRAG initialization")
    
    # Verify the PatternAwareRAG is not None
    if not verify_component_initialization(pattern_aware_rag, "PatternAwareRAG"):
        return False
    
    # Verify the PatternAwareRAG has the required attributes
    required_attributes = ['_pattern_repository', '_claude_adapter']
    for attr in required_attributes:
        if not hasattr(pattern_aware_rag, attr):
            logger.error(f"PatternAwareRAG missing required attribute: {attr}")
            return False
    
    # Verify the PatternAwareRAG has the required methods
    required_methods = ['query']
    for method in required_methods:
        if not hasattr(pattern_aware_rag, method):
            logger.error(f"PatternAwareRAG missing required method: {method}")
            return False
        if not callable(getattr(pattern_aware_rag, method)):
            logger.error(f"PatternAwareRAG attribute {method} is not callable")
            return False
    
    logger.debug("PatternAwareRAG is properly initialized")
    return True

def verify_claude_adapter(claude_adapter):
    """Verify that the ClaudeAdapter is properly initialized."""
    logger.debug("Verifying ClaudeAdapter initialization")
    
    # Verify the ClaudeAdapter is not None
    if not verify_component_initialization(claude_adapter, "ClaudeAdapter"):
        return False
    
    # Verify the ClaudeAdapter has the required methods
    required_methods = ['query', 'extract_patterns']
    for method in required_methods:
        if not hasattr(claude_adapter, method):
            logger.error(f"ClaudeAdapter missing required method: {method}")
            return False
        if not callable(getattr(claude_adapter, method)):
            logger.error(f"ClaudeAdapter attribute {method} is not callable")
            return False
    
    logger.debug("ClaudeAdapter is properly initialized")
    return True

def verify_all_critical_components(test_context, strict=True, diagnostic=False):
    """Verify all critical components in the test context."""
    logger.info("Performing comprehensive verification of all critical components")
    
    all_initialized = True
    verification_results = {}
    failed_verifications = []
    
    # Define verification functions for each component type
    verification_functions = {
        'event_service': {
            'function': verify_event_service,
            'name': 'EventService',
            'required_attributes': ['_initialized', 'publish', 'subscribe', 'unsubscribe'],
            'required_methods': ['publish', 'subscribe', 'unsubscribe']
        },
        'vector_tonic_integrator': {
            'function': verify_vector_tonic_integration,
            'name': 'Vector Tonic Integration',
            'required_attributes': ['_initialized', 'process_window'],
            'required_methods': ['process_window']
        },
        'pattern_aware_rag': {
            'function': verify_pattern_aware_rag,
            'name': 'PatternAwareRAG',
            'required_attributes': ['_initialized', '_pattern_repository', '_claude_adapter'],
            'required_methods': ['query']
        },
        'claude_adapter': {
            'function': verify_claude_adapter,
            'name': 'Claude Adapter',
            'required_attributes': ['_initialized'],
            'required_methods': ['query', 'extract_patterns']
        },
        'arangodb_connection': {
            'function': lambda x: verify_component_initialization(x, "ArangoDBConnection"),
            'name': 'ArangoDBConnection',
            'required_attributes': ['_initialized'],
            'required_methods': ['get_database']
        }
    }
    
    # Verify each component if present
    for component_key, verification_info in verification_functions.items():
        if component_key in test_context:
            component = test_context[component_key]
            component_name = verification_info['name']
            verification_function = verification_info['function']
            required_attributes = verification_info['required_attributes']
            required_methods = verification_info['required_methods']
            
            # Initialize component result in diagnostic mode
            if diagnostic:
                verification_results[component_name] = {
                    'present': True,
                    'initialized': None,
                    'attributes': {},
                    'methods': {},
                    'verification_passed': False,
                    'errors': []
                }
                
                # Check initialization status
                if hasattr(component, '_initialized'):
                    verification_results[component_name]['initialized'] = component._initialized
                elif hasattr(component, 'is_initialized') and callable(component.is_initialized):
                    try:
                        verification_results[component_name]['initialized'] = component.is_initialized()
                    except Exception as e:
                        verification_results[component_name]['initialized'] = False
                        verification_results[component_name]['errors'].append(f"Error calling is_initialized(): {e}")
                else:
                    verification_results[component_name]['initialized'] = None
                    verification_results[component_name]['errors'].append("No initialization status available")
                
                # Check required attributes
                for attr in required_attributes:
                    has_attr = hasattr(component, attr)
                    verification_results[component_name]['attributes'][attr] = has_attr
                    if not has_attr:
                        verification_results[component_name]['errors'].append(f"Missing required attribute: {attr}")
                
                # Check required methods
                for method in required_methods:
                    has_method = hasattr(component, method) and callable(getattr(component, method))
                    verification_results[component_name]['methods'][method] = has_method
                    if not has_method:
                        verification_results[component_name]['errors'].append(f"Missing required method: {method}")
            
            # Run the verification function
            try:
                logger.debug(f"Verifying {component_name}...")
                result = verification_function(component)
                if result:
                    logger.debug(f"{component_name} verification passed")
                    if diagnostic:
                        verification_results[component_name]['verification_passed'] = True
                else:
                    all_initialized = False
                    error_message = f"{component_name} verification failed"
                    failed_verifications.append((component_name, error_message))
                    
                    if diagnostic:
                        verification_results[component_name]['verification_passed'] = False
                        if not verification_results[component_name]['errors']:
                            verification_results[component_name]['errors'].append("Verification function returned False")
                    elif strict:
                        logger.error(error_message)
                        raise AssertionError(error_message)
                    else:
                        logger.warning(f"{error_message} but continuing")
            except Exception as e:
                all_initialized = False
                error_message = f"{component_name} verification failed: {e}"
                failed_verifications.append((component_name, str(e)))
                
                if diagnostic:
                    verification_results[component_name]['verification_passed'] = False
                    verification_results[component_name]['errors'].append(str(e))
                    logger.error(error_message)
                elif strict:
                    logger.error(error_message)
                    raise
                else:
                    logger.warning(f"{error_message} but continuing")
        elif diagnostic:
            # Component not present in test context
            verification_results[verification_info['name']] = {
                'present': False,
                'initialized': None,
                'verification_passed': False,
                'errors': ["Component not present in test context"]
            }
    
    # Log summary
    if all_initialized:
        logger.info("All critical components verified successfully")
    else:
        logger.warning(f"Some critical component verifications failed: {failed_verifications}")
    
    # Return appropriate result based on mode
    if diagnostic:
        return verification_results
    else:
        return all_initialized

def log_exception_with_traceback(e, message="Exception occurred"):
    """Log an exception with full traceback."""
    logger.error(f"{message}: {e}")
    logger.error(traceback.format_exc())

def log_component_details(component, component_name):
    """Log detailed information about a component."""
    logger.info(f"=== Component Details: {component_name} ===")
    
    if component is None:
        logger.error(f"Component {component_name} is None")
        return
    
    # Log component type
    component_type = type(component).__name__
    logger.info(f"Component type: {component_type}")
    
    # Log initialization status
    if hasattr(component, '_initialized'):
        logger.info(f"Initialization status (_initialized): {component._initialized}")
    else:
        logger.info("No _initialized attribute found")
    
    if hasattr(component, 'is_initialized') and callable(component.is_initialized):
        try:
            is_init = component.is_initialized()
            logger.info(f"Initialization status (is_initialized()): {is_init}")
        except Exception as e:
            logger.error(f"Error calling is_initialized(): {e}")
    else:
        logger.info("No is_initialized() method found")
    
    # Log attributes
    logger.debug(f"Attributes of {component_name}:")
    for attr in dir(component):
        if not attr.startswith('__') and not callable(getattr(component, attr)):
            try:
                value = getattr(component, attr)
                # Don't log large objects or sensitive data
                if isinstance(value, (str, int, float, bool, type(None))):
                    logger.debug(f"  {attr}: {value}")
                else:
                    logger.debug(f"  {attr}: {type(value)}")
            except Exception as e:
                logger.debug(f"  {attr}: Error accessing - {e}")
    
    # Log methods
    logger.debug(f"Methods of {component_name}:")
    for attr in dir(component):
        if not attr.startswith('__') and callable(getattr(component, attr)):
            logger.debug(f"  {attr}")

class ComponentErrorTracker:
    """Track errors in component initialization and dependencies."""
    
    def __init__(self):
        """Initialize the component error tracker."""
        self.components = {}
        self.initialization_errors = {}
        self.dependency_errors = {}
        self.verification_results = {}
        self.initialization_order = []
        
        logger.info("=== Component Error Tracker Initialized ===")
    
    def initialize_component(self, component_key, component_class, dependencies=None, init_args=None):
        """Initialize a component and track any errors."""
        logger.info(f"Initializing component: {component_key}")
        dependencies = dependencies or []
        
        # Check if dependencies are satisfied
        missing_deps = []
        for dep in dependencies:
            if dep not in self.components or self.components[dep] is None:
                missing_deps.append(dep)
                logger.error(f"Missing dependency for {component_key}: {dep}")
        
        if missing_deps:
            logger.error(f"Cannot initialize {component_key} due to missing dependencies: {missing_deps}")
            self.components[component_key] = None
            self.dependency_errors[component_key] = f"Missing dependencies: {missing_deps}"
            return None
        
        # Create the component
        try:
            logger.debug(f"Creating {component_key} instance...")
            component = component_class()
            self.components[component_key] = component
            logger.debug(f"{component_key} instance created")
            
            # Log detailed component status
            log_component_details(component, component_key)
            
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
                            if param_name in self.components and self.components[param_name] is not None:
                                auto_args[param_name] = self.components[param_name]
                        
                        if auto_args:
                            logger.debug(f"Auto-passing dependencies to {component_key}.initialize(): {list(auto_args.keys())}")
                            component.initialize(**auto_args)
                        else:
                            component.initialize()
                    
                    logger.debug(f"{component_key} initialized successfully")
                    self.initialization_order.append(component_key)
                    
                    # Log detailed component status after initialization
                    log_component_details(component, f"{component_key} (after initialization)")
                    
                    return component
                    
                except Exception as e:
                    log_exception_with_traceback(e, f"Failed to initialize {component_key}")
                    self.initialization_errors[component_key] = f"Initialization error: {str(e)}"
                    return component  # Return the component even though initialization failed
            else:
                logger.debug(f"{component_key} has no initialize method")
                self.initialization_order.append(component_key)
                return component
                
        except Exception as e:
            log_exception_with_traceback(e, f"Failed to create {component_key}")
            self.components[component_key] = None
            self.initialization_errors[component_key] = f"Creation error: {str(e)}"
            return None
    
    def verify_components(self):
        """Verify all components and track verification results."""
        logger.info("=== Verifying All Components ===")
        
        # Run verification in diagnostic mode to capture all issues
        self.verification_results = verify_all_critical_components(
            self.components,
            strict=False,
            diagnostic=True
        )
        
        # Log verification results
        for component_name, result in self.verification_results.items():
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
        
        return self.verification_results
    
    def test_component_methods(self):
        """Test key methods on each component to identify runtime issues."""
        logger.info("=== Testing Component Methods ===")
        
        method_results = {}
        
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
        
        # Test methods on each component
        for component_key, methods in test_methods.items():
            if component_key not in self.components or self.components[component_key] is None:
                logger.warning(f"Component {component_key} not available for method testing")
                continue
            
            component = self.components[component_key]
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
                    log_exception_with_traceback(e, f"Error calling {component_key}.{method_name}()")
                    method_results[component_key].append({
                        'method': method_name,
                        'success': False,
                        'error': str(e)
                    })
        
        return method_results
    
    def generate_diagnostic_report(self):
        """Generate a comprehensive diagnostic report."""
        logger.info("=== Generating Diagnostic Report ===")
        
        # Create report directory
        report_dir = "diagnostic_reports"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"component_diagnostic_{timestamp}.json")
        
        # Create the report
        report = {
            'timestamp': datetime.now().isoformat(),
            'initialization_order': self.initialization_order,
            'initialization_errors': self.initialization_errors,
            'dependency_errors': self.dependency_errors,
            'verification_results': self.verification_results
        }
        
        # Save the report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Diagnostic report saved to: {report_file}")
        return report_file

def main():
    """Run the component error tracking."""
    logger.info("=== Starting Component Error Tracking ===")
    
    tracker = ComponentErrorTracker()
    
    # Import the components we need to test
    try:
        from habitat_evolution.infrastructure.services.event_service import EventService
        from habitat_evolution.infrastructure.services.pattern_aware_rag_service import PatternAwareRAGService
        
        # Try different module paths for VectorTonic integration
        vector_tonic_integration_class = None
        try:
            from habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegration
            vector_tonic_integration_class = VectorTonicWindowIntegration
            logger.info("Found VectorTonicWindowIntegration in adaptive_core.emergence")
        except ImportError as e:
            logger.warning(f"Could not import VectorTonicWindowIntegration from adaptive_core.emergence: {e}")
            try:
                from habitat_evolution.infrastructure.services.vector_tonic_service import VectorTonicService
                vector_tonic_integration_class = VectorTonicService
                logger.info("Found VectorTonicService in infrastructure.services")
            except ImportError as e:
                logger.error(f"Could not import VectorTonicService: {e}")
        
        # Try different module paths for ClaudeAdapter
        claude_adapter_class = None
        try:
            from habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter
            claude_adapter_class = ClaudeAdapter
            logger.info("Found ClaudeAdapter in infrastructure.adapters")
        except ImportError as e:
            logger.warning(f"Could not import ClaudeAdapter from infrastructure.adapters: {e}")
            try:
                from habitat_evolution.infrastructure.services.claude_adapter import ClaudeAdapter
                claude_adapter_class = ClaudeAdapter
                logger.info("Found ClaudeAdapter in infrastructure.services")
            except ImportError as e:
                logger.error(f"Could not import ClaudeAdapter: {e}")
        
        # Try different module paths for ArangoDBConnection
        arangodb_connection_class = None
        try:
            from habitat_evolution.infrastructure.persistence.arangodb.connection import ArangoDBConnection
            arangodb_connection_class = ArangoDBConnection
            logger.info("Found ArangoDBConnection in infrastructure.persistence.arangodb")
        except ImportError as e:
            logger.warning(f"Could not import ArangoDBConnection from infrastructure.persistence.arangodb: {e}")
            try:
                from habitat_evolution.infrastructure.db.arangodb_connection import ArangoDBConnection
                arangodb_connection_class = ArangoDBConnection
                logger.info("Found ArangoDBConnection in infrastructure.db")
            except ImportError as e:
                logger.error(f"Could not import ArangoDBConnection: {e}")
        
        # Initialize foundation components first
        logger.info("Initializing foundation components...")
        if arangodb_connection_class:
            tracker.initialize_component('arangodb_connection', arangodb_connection_class, [])
        
        tracker.initialize_component('event_service', EventService, [])
        
        if claude_adapter_class:
            tracker.initialize_component('claude_adapter', claude_adapter_class, [], 
                                        init_args={'api_key': 'mock_api_key_for_testing'})
        
        # Initialize repositories that depend on ArangoDB
        logger.info("Initializing repositories...")
        try:
            from habitat_evolution.infrastructure.repositories.pattern_repository import PatternRepository
            from habitat_evolution.infrastructure.repositories.relationship_repository import RelationshipRepository
            
            tracker.initialize_component('pattern_repository', PatternRepository, ['arangodb_connection'])
            tracker.initialize_component('relationship_repository', RelationshipRepository, ['arangodb_connection'])
        except ImportError as e:
            logger.error(f"Failed to import repositories: {e}")
        
        # Initialize Vector Tonic Integration
        logger.info("Initializing Vector Tonic Integration...")
        if vector_tonic_integration_class:
            tracker.initialize_component('vector_tonic_integrator', vector_tonic_integration_class, ['event_service'])
        
        # Initialize PatternAwareRAG
        logger.info("Initializing PatternAwareRAG...")
        tracker.initialize_component('pattern_aware_rag', PatternAwareRAGService, 
                                    ['claude_adapter', 'pattern_repository', 'relationship_repository'])
        
        # Verify all components
        logger.info("Verifying all components...")
        verification_results = tracker.verify_components()
        
        # Test component methods
        logger.info("Testing component methods...")
        method_results = tracker.test_component_methods()
        
        # Generate diagnostic report
        report_file = tracker.generate_diagnostic_report()
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("COMPONENT ERROR TRACKING SUMMARY")
        logger.info("="*50)
        
        # Initialization errors
        if tracker.initialization_errors:
            logger.info("\nInitialization Errors:")
            for component, error in tracker.initialization_errors.items():
                logger.info(f"  - {component}: {error}")
        else:
            logger.info("\nNo initialization errors detected.")
        
        # Dependency errors
        if tracker.dependency_errors:
            logger.info("\nDependency Errors:")
            for component, error in tracker.dependency_errors.items():
                logger.info(f"  - {component}: {error}")
        else:
            logger.info("\nNo dependency errors detected.")
        
        # Verification failures
        verification_failures = []
        for component, result in verification_results.items():
            if not result.get('verification_passed', False):
                verification_failures.append(component)
        
        if verification_failures:
            logger.info("\nVerification Failures:")
            for component in verification_failures:
                errors = verification_results[component].get('errors', [])
                logger.info(f"  - {component}:")
                for error in errors:
                    logger.info(f"    * {error}")
        else:
            logger.info("\nNo verification failures detected.")
        
        logger.info("\n" + "="*50)
        logger.info(f"Diagnostic report saved to: {report_file}")
        logger.info(f"Log file: {log_file}")
        logger.info("="*50)
        
    except Exception as e:
        log_exception_with_traceback(e, "Error during component error tracking")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
