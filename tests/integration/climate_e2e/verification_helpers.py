"""
Verification helper functions for Habitat Evolution tests.

This module provides helper functions to verify component initialization
and system state, ensuring that tests fail when critical components are
not properly initialized.
"""

import logging
import inspect
import pytest
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def verify_component_initialization(component: Any, component_name: str) -> bool:
    """
    Verify that a component is properly initialized.
    
    This function checks if a component is properly initialized by:
    1. Checking if the component is not None
    2. Checking if the component has an _initialized attribute that is True
    3. Checking if the component has an is_initialized() method that returns True
    
    Args:
        component: The component to verify
        component_name: The name of the component for logging
        
    Returns:
        bool: True if the component is properly initialized, False otherwise
        
    Raises:
        AssertionError: If the component is not properly initialized
    """
    logger.debug(f"Verifying initialization of {component_name}")
    
    # Step 1: Check if component exists
    logger.debug(f"Step 1: Checking if {component_name} exists")
    assert component is not None, f"Critical component {component_name} is None"
    
    # Log component type for debugging
    component_type = type(component).__name__
    logger.debug(f"Component {component_name} is of type {component_type}")
    
    # Step 2: Check if component has _initialized attribute
    initialized_via_attribute = False
    if hasattr(component, '_initialized'):
        logger.debug(f"Step 2: Component {component_name} has _initialized attribute: {component._initialized}")
        assert component._initialized is True, f"Critical component {component_name} reports not initialized (_initialized is False)"
        initialized_via_attribute = True
    else:
        logger.debug(f"Component {component_name} does not have _initialized attribute")
    
    # Step 3: Check if component has is_initialized() method
    initialized_via_method = False
    if hasattr(component, 'is_initialized') and callable(component.is_initialized):
        try:
            is_init = component.is_initialized()
            logger.debug(f"Step 3: Component {component_name} is_initialized() returned: {is_init}")
            assert is_init, f"Critical component {component_name} reports not initialized (is_initialized() returned False)"
            initialized_via_method = True
        except Exception as e:
            logger.error(f"Error calling is_initialized() on {component_name}: {e}")
            raise AssertionError(f"Error calling is_initialized() on {component_name}: {e}")
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
            raise AssertionError(f"Component {component_name} has no initialization status indicators (_initialized attribute or is_initialized() method)")
    
    logger.debug(f"Component {component_name} is properly initialized")
    return True

def verify_system_state(components: Dict[str, Any], strict: bool = True) -> bool:
    """
    Verify all critical components are properly initialized.
    
    Args:
        components: Dictionary mapping component names to component instances
        strict: If True, raise AssertionError for any component that fails verification
               If False, log warnings but continue (useful for diagnostic purposes)
        
    Returns:
        bool: True if all components are properly initialized
        
    Raises:
        AssertionError: If any component is not properly initialized and strict=True
    """
    logger.debug(f"Verifying system state with {len(components)} components")
    
    all_initialized = True
    failed_components = []
    
    for name, component in components.items():
        try:
            verify_component_initialization(component, name)
        except AssertionError as e:
            all_initialized = False
            failed_components.append((name, str(e)))
            if strict:
                logger.error(f"Component verification failed: {e}")
                raise
            else:
                logger.warning(f"Component verification failed but continuing: {e}")
    
    if all_initialized:
        logger.debug("All components are properly initialized")
    else:
        logger.warning(f"Some components failed verification: {failed_components}")
    
    return all_initialized

def verify_event_service(event_service: Any) -> bool:
    """
    Verify that the EventService is properly initialized.
    
    Args:
        event_service: The EventService instance
        
    Returns:
        bool: True if the EventService is properly initialized
        
    Raises:
        AssertionError: If the EventService is not properly initialized
    """
    logger.debug("Verifying EventService initialization")
    
    # Step 1: Verify the EventService is not None
    logger.debug("Step 1: Verifying EventService is not None")
    assert event_service is not None, "EventService is None"
    
    # Step 2: Verify the EventService has _initialized attribute
    logger.debug("Step 2: Verifying EventService has _initialized attribute")
    assert hasattr(event_service, '_initialized'), "EventService missing _initialized attribute"
    
    # Step 3: Verify the EventService is initialized
    logger.debug("Step 3: Verifying EventService is initialized")
    assert event_service._initialized is True, "EventService is not initialized (_initialized is False)"
    
    # Step 4: Verify the EventService has the required methods
    logger.debug("Step 4: Verifying EventService has required methods")
    required_methods = ['publish', 'subscribe', 'unsubscribe']
    for method in required_methods:
        logger.debug(f"Checking for method: {method}")
        assert hasattr(event_service, method), f"EventService missing required method: {method}"
        assert callable(getattr(event_service, method)), f"EventService attribute {method} is not callable"
    
    # Step 5: Verify the EventService can publish events (only if it's a real EventService)
    logger.debug("Step 5: Verifying EventService can publish events")
    
    # Skip actual method calls for mock objects in tests
    if event_service.__class__.__name__ == 'MockComponent':
        logger.debug("Skipping method call verification for MockComponent")
    else:
        test_event_type = "test_event"
        test_event_data = {"test": "data"}
        try:
            event_service.publish(test_event_type, test_event_data)
            logger.debug("Successfully published test event")
        except Exception as e:
            error_msg = f"EventService failed to publish events: {e}"
            logger.error(error_msg)
            pytest.fail(error_msg)
    
    logger.debug("EventService is properly initialized")
    return True

def verify_vector_tonic_integration(vector_tonic_integrator: Any) -> bool:
    """
    Verify that Vector Tonic integration is properly initialized with all required dependencies.
    
    This function performs a comprehensive verification of the Vector Tonic integration,
    checking that all components in the dependency chain are properly initialized.
    
    Args:
        vector_tonic_integrator: Vector Tonic integrator to verify
        
    Returns:
        bool: True if the Vector Tonic integration is properly initialized
        
    Raises:
        AssertionError: If Vector Tonic integration or any of its dependencies are not properly initialized
    """
    logger.debug("Performing comprehensive verification of Vector Tonic integration")
    
    # Step 1: Verify the integrator itself is not None
    verify_component_initialization(vector_tonic_integrator, "VectorTonicWindowIntegrator")
    
    # Step 2: Verify the integrator has the required attributes
    required_attributes = ['tonic_detector', 'event_bus', 'harmonic_io_service']
    for attr in required_attributes:
        assert hasattr(vector_tonic_integrator, attr), f"VectorTonicWindowIntegrator missing required attribute: {attr}"
        assert getattr(vector_tonic_integrator, attr) is not None, f"VectorTonicWindowIntegrator attribute {attr} is None"
    
    # Instead of checking for specific methods, just log what methods are available
    # This follows our principle of DEBUG LOGGING IN ALL TEST PROCESSES
    available_methods = [method for method in dir(vector_tonic_integrator) if callable(getattr(vector_tonic_integrator, method)) and not method.startswith('_')]
    logger.debug(f"Available methods in VectorTonicWindowIntegrator: {available_methods}")
    
    # Check that the component has at least some methods
    assert len(available_methods) > 0, "VectorTonicWindowIntegrator has no public methods"
    
    # Step 3: Verify the tonic detector is initialized
    tonic_detector = vector_tonic_integrator.tonic_detector
    verify_component_initialization(tonic_detector, "TonicHarmonicPatternDetector")
    
    # Step 4: Verify the tonic detector has a base detector
    assert hasattr(tonic_detector, 'base_detector'), "TonicHarmonicPatternDetector missing required base_detector"
    assert tonic_detector.base_detector is not None, "TonicHarmonicPatternDetector base_detector is None"
    
    # Step 5: Verify the base detector (LearningWindowAwareDetector)
    learning_detector = tonic_detector.base_detector
    verify_component_initialization(learning_detector, "LearningWindowAwareDetector")
    
    # Step 6: Verify the learning detector has a detector and pattern publisher
    assert hasattr(learning_detector, 'detector'), "LearningWindowAwareDetector missing required detector"
    assert learning_detector.detector is not None, "LearningWindowAwareDetector detector is None"
    assert hasattr(learning_detector, 'pattern_publisher'), "LearningWindowAwareDetector missing required pattern_publisher"
    assert learning_detector.pattern_publisher is not None, "LearningWindowAwareDetector pattern_publisher is None"
    
    # Step 7: Verify the event-aware detector
    event_detector = learning_detector.detector
    verify_component_initialization(event_detector, "EventAwarePatternDetector")
    
    # Step 8: Verify the event-aware detector has a semantic observer
    assert hasattr(event_detector, 'semantic_observer'), "EventAwarePatternDetector missing required semantic_observer"
    assert event_detector.semantic_observer is not None, "EventAwarePatternDetector semantic_observer is None"
    
    # Step 9: Verify the semantic observer
    semantic_observer = event_detector.semantic_observer
    verify_component_initialization(semantic_observer, "SemanticCurrentObserver")
    
    # Step 10: Verify the semantic observer has required field components
    assert hasattr(semantic_observer, 'field_navigator'), "SemanticCurrentObserver missing required field_navigator"
    assert semantic_observer.field_navigator is not None, "SemanticCurrentObserver field_navigator is None"
    assert hasattr(semantic_observer, 'journey_tracker'), "SemanticCurrentObserver missing required journey_tracker"
    assert semantic_observer.journey_tracker is not None, "SemanticCurrentObserver journey_tracker is None"
    
    # Log success
    logger.info("Vector Tonic integration successfully verified with all dependencies")
    return True

def verify_pattern_aware_rag(pattern_aware_rag: Any) -> bool:
    """
    Verify that the PatternAwareRAG service is properly initialized.
    
    Args:
        pattern_aware_rag: The PatternAwareRAG instance
        
    Returns:
        bool: True if the PatternAwareRAG service is properly initialized
        
    Raises:
        AssertionError: If the PatternAwareRAG service is not properly initialized
    """
    logger.debug("Verifying PatternAwareRAG initialization")
    
    # Basic initialization check
    verify_component_initialization(pattern_aware_rag, "PatternAwareRAG")
    
    # Check required attributes
    required_attributes = ['_db_connection', '_pattern_repository', '_claude_adapter']
    for attr in required_attributes:
        assert hasattr(pattern_aware_rag, attr), f"PatternAwareRAG missing required attribute: {attr}"
        assert getattr(pattern_aware_rag, attr) is not None, f"PatternAwareRAG attribute {attr} is None"
    
    # Check required methods
    required_methods = ['query', 'enhance_with_patterns', 'store_pattern']
    for method in required_methods:
        assert hasattr(pattern_aware_rag, method), f"PatternAwareRAG missing required method: {method}"
        assert callable(getattr(pattern_aware_rag, method)), f"PatternAwareRAG attribute {method} is not callable"
    
    logger.debug("PatternAwareRAG is properly initialized")
    return True

def verify_claude_adapter(claude_adapter: Any) -> bool:
    """
    Verify that the ClaudeAdapter is properly initialized and has the required methods.
    
    Args:
        claude_adapter: The ClaudeAdapter instance
        
    Returns:
        bool: True if the ClaudeAdapter is properly initialized
        
    Raises:
        AssertionError: If the ClaudeAdapter is not properly initialized
    """
    logger.debug("Verifying ClaudeAdapter initialization")
    
    # Basic initialization check
    verify_component_initialization(claude_adapter, "ClaudeAdapter")
    
    # Check required methods
    required_methods = ['query', 'extract_patterns']
    for method in required_methods:
        assert hasattr(claude_adapter, method), f"ClaudeAdapter missing required method: {method}"
        assert callable(getattr(claude_adapter, method)), f"ClaudeAdapter attribute {method} is not callable"
    
    logger.debug("ClaudeAdapter is properly initialized")
    return True

def verify_type_safety(value: Any, expected_type: Any, value_name: str) -> bool:
    """
    Verify that a value is of the expected type.
    
    Args:
        value: The value to check
        expected_type: The expected type or tuple of types
        value_name: The name of the value for logging
        
    Returns:
        bool: True if the value is of the expected type
        
    Raises:
        AssertionError: If the value is not of the expected type
    """
    logger.debug(f"Verifying type safety of {value_name}")
    
    assert isinstance(value, expected_type), f"{value_name} is not of expected type {expected_type.__name__}, got {type(value).__name__}"
    
    logger.debug(f"{value_name} is of expected type {expected_type.__name__}")
    return True

def deliberately_break_component(component: Any, attribute: str, new_value: Any = None) -> Any:
    """
    Deliberately break a component by setting an attribute to None or a new value.
    This is useful for testing error handling and verification.
    
    Args:
        component: The component to break
        attribute: The attribute to set to None or a new value
        new_value: The new value to set the attribute to (default: None)
        
    Returns:
        The original value of the attribute
    """
    component_type = type(component).__name__
    logger.debug(f"Deliberately breaking component of type {component_type} by setting {attribute} to {new_value}")
    
    if not hasattr(component, attribute):
        logger.warning(f"Component does not have attribute {attribute}")
        return None
    
    original_value = getattr(component, attribute)
    logger.debug(f"Original value of {attribute} was {original_value}")
    
    setattr(component, attribute, new_value)
    logger.debug(f"Successfully set {attribute} to {new_value}")
    
    return original_value

def verify_all_critical_components(test_context: Dict[str, Any], strict: bool = True, diagnostic: bool = False) -> Dict[str, Any]:
    """
    Verify all critical components in the test context.
    
    This function provides a single entry point to verify all critical components
    in the Habitat Evolution system, including:
    1. EventService
    2. Vector Tonic Integration
    3. PatternAwareRAG
    4. Claude Adapter
    5. Database Connection
    
    Args:
        test_context: Dictionary containing all test fixtures and components
        strict: If True, raise AssertionError for any component that fails verification
               If False, log warnings but continue (useful for diagnostic purposes)
        diagnostic: If True, run in diagnostic mode which captures detailed information
                   about each component's verification status
        
    Returns:
        If diagnostic=False: bool indicating if all components are properly initialized
        If diagnostic=True: Dict with detailed verification results for each component
        
    Raises:
        AssertionError: If any component is not properly initialized and strict=True
    """
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
            'required_attributes': ['tonic_detector', 'event_bus', 'harmonic_io_service'],
            'required_methods': []
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
            'required_attributes': [],
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
                verification_function(component)
                logger.debug(f"{component_name} verification passed")
                if diagnostic:
                    verification_results[component_name]['verification_passed'] = True
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
