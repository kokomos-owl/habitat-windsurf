"""
Verification helper functions for Habitat Evolution tests.

This module provides helper functions to verify component initialization
and system state, ensuring that tests fail when critical components are
not properly initialized.
"""

import logging
import inspect
import pytest
from typing import Dict, Any, List, Optional, Callable

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
    
    # Check if component exists
    assert component is not None, f"Critical component {component_name} is None"
    
    # Check if component has _initialized attribute
    if hasattr(component, '_initialized'):
        assert component._initialized is True, f"Critical component {component_name} reports not initialized (_initialized is False)"
    
    # Check if component has is_initialized() method
    if hasattr(component, 'is_initialized') and callable(component.is_initialized):
        assert component.is_initialized(), f"Critical component {component_name} reports not initialized (is_initialized() returned False)"
    
    logger.debug(f"Component {component_name} is properly initialized")
    return True

def verify_system_state(components: Dict[str, Any]) -> bool:
    """
    Verify all critical components are properly initialized.
    
    Args:
        components: Dictionary mapping component names to component instances
        
    Returns:
        bool: True if all components are properly initialized
        
    Raises:
        AssertionError: If any component is not properly initialized
    """
    logger.debug("Verifying system state")
    
    for name, component in components.items():
        verify_component_initialization(component, name)
    
    logger.debug("All components are properly initialized")
    return True

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
    
    # Basic initialization check
    verify_component_initialization(event_service, "EventService")
    
    # Verify that event_service can publish events
    test_event_type = "test.event"
    test_event_data = {"test": "data"}
    
    try:
        event_service.publish(test_event_type, test_event_data)
        logger.debug("EventService can publish events")
    except Exception as e:
        pytest.fail(f"EventService failed to publish events: {e}")
    
    logger.debug("EventService is properly initialized")
    return True

def verify_vector_tonic_integration(vector_tonic_integrator: Any) -> bool:
    """
    Verify that the Vector Tonic integration is properly initialized.
    
    Args:
        vector_tonic_integrator: The VectorTonicWindowIntegrator instance
        
    Returns:
        bool: True if the Vector Tonic integration is properly initialized
        
    Raises:
        AssertionError: If the Vector Tonic integration is not properly initialized
    """
    logger.debug("Verifying Vector Tonic integration")
    
    # Basic initialization check
    verify_component_initialization(vector_tonic_integrator, "VectorTonicWindowIntegrator")
    
    # Check required attributes
    required_attributes = ['tonic_detector', 'event_bus', 'harmonic_io_service']
    for attr in required_attributes:
        assert hasattr(vector_tonic_integrator, attr), f"VectorTonicWindowIntegrator missing required attribute: {attr}"
        assert getattr(vector_tonic_integrator, attr) is not None, f"VectorTonicWindowIntegrator attribute {attr} is None"
    
    # Check required methods
    required_methods = ['create_learning_window', 'add_pattern_to_window', 'register_adaptive_id']
    for method in required_methods:
        assert hasattr(vector_tonic_integrator, method), f"VectorTonicWindowIntegrator missing required method: {method}"
        assert callable(getattr(vector_tonic_integrator, method)), f"VectorTonicWindowIntegrator attribute {method} is not callable"
    
    logger.debug("Vector Tonic integration is properly initialized")
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
    logger.debug(f"Deliberately breaking component by setting {attribute} to {new_value}")
    
    original_value = getattr(component, attribute, None)
    setattr(component, attribute, new_value)
    
    return original_value
