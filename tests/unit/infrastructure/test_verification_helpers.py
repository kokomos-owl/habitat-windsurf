"""
Unit tests for the verification helper functions.

This module contains tests that verify the verification helper functions
work correctly and can detect when critical components are not properly initialized.
"""

import pytest
import logging
from typing import Dict, Any

from tests.integration.climate_e2e.verification_helpers import (
    verify_component_initialization,
    verify_system_state,
    deliberately_break_component,
    verify_all_critical_components
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MockComponent:
    """Mock component for testing verification helpers."""
    
    def __init__(self, initialized=True, create_dependency=False):
        self._initialized = initialized
        # Avoid infinite recursion by only creating one level of dependency
        self.dependency = MockComponent(initialized, False) if initialized and create_dependency else None
    
    def is_initialized(self):
        return self._initialized


def test_verify_component_initialization():
    """Test that verify_component_initialization works correctly."""
    # Create a properly initialized component
    component = MockComponent(initialized=True, create_dependency=True)
    
    # Verify it passes verification
    assert verify_component_initialization(component, "TestComponent")
    
    # Create a component that is not initialized
    component = MockComponent(initialized=False)
    
    # Verify it fails verification
    with pytest.raises(AssertionError):
        verify_component_initialization(component, "TestComponent")
    
    # Test with None component
    with pytest.raises(AssertionError):
        verify_component_initialization(None, "NoneComponent")


def test_verify_system_state():
    """Test that verify_system_state works correctly."""
    # Create a set of properly initialized components
    components = {
        "component1": MockComponent(initialized=True, create_dependency=True),
        "component2": MockComponent(initialized=True, create_dependency=True),
        "component3": MockComponent(initialized=True, create_dependency=True)
    }
    
    # Verify they all pass verification
    assert verify_system_state(components, strict=True)
    
    # Break one component
    components["component2"] = MockComponent(initialized=False)
    
    # Verify it fails verification in strict mode
    with pytest.raises(AssertionError):
        verify_system_state(components, strict=True)
    
    # Verify it returns False but doesn't raise in non-strict mode
    assert not verify_system_state(components, strict=False)


def test_deliberately_break_component():
    """Test that deliberately_break_component works correctly."""
    # Create a properly initialized component
    component = MockComponent(initialized=True, create_dependency=True)
    
    # Verify it passes verification
    assert verify_component_initialization(component, "TestComponent")
    
    # Break it
    original_value = deliberately_break_component(component, "_initialized", False)
    
    # Verify it now fails verification
    with pytest.raises(AssertionError):
        verify_component_initialization(component, "TestComponent")
    
    # Verify the original value was returned correctly
    assert original_value is True
    
    # Restore it
    deliberately_break_component(component, "_initialized", original_value)
    
    # Verify it passes verification again
    assert verify_component_initialization(component, "TestComponent")


def test_verify_all_critical_components():
    """Test that verify_all_critical_components works correctly."""
    # Create mock components for all critical components
    test_context = {
        'arangodb_connection': MockComponent(initialized=True, create_dependency=True),
        'event_service': MockComponent(initialized=True, create_dependency=True),
        'pattern_evolution_service': MockComponent(initialized=True, create_dependency=True),
        'document_processing_service': MockComponent(initialized=True, create_dependency=True),
        'field_pattern_bridge': MockComponent(initialized=True, create_dependency=True),
        'claude_adapter': MockComponent(initialized=True, create_dependency=True),
        'pattern_aware_rag': MockComponent(initialized=True, create_dependency=True),
        'vector_tonic_integrator': MockComponent(initialized=True, create_dependency=True)
    }
    
    # Verify they all pass verification
    assert verify_all_critical_components(test_context, strict=True)
    
    # Break one component
    test_context['event_service'] = MockComponent(initialized=False, create_dependency=False)
    
    # Verify it fails verification in strict mode
    with pytest.raises(AssertionError):
        verify_all_critical_components(test_context, strict=True)
    
    # Verify it returns False but doesn't raise in non-strict mode
    assert not verify_all_critical_components(test_context, strict=False)
    
    # Test with missing components
    test_context = {
        'arangodb_connection': MockComponent(initialized=True, create_dependency=True),
        'event_service': MockComponent(initialized=True, create_dependency=True)
    }
    
    # Verify it still passes with only a subset of components
    assert verify_all_critical_components(test_context, strict=True)


def test_deliberate_failure_testing():
    """Test that deliberate failure testing works correctly."""
    # Create a test context with all components initialized
    test_context = {
        'arangodb_connection': MockComponent(initialized=True, create_dependency=True),
        'event_service': MockComponent(initialized=True, create_dependency=True),
        'pattern_evolution_service': MockComponent(initialized=True, create_dependency=True),
        'document_processing_service': MockComponent(initialized=True, create_dependency=True),
        'field_pattern_bridge': MockComponent(initialized=True, create_dependency=True),
        'claude_adapter': MockComponent(initialized=True, create_dependency=True),
        'pattern_aware_rag': MockComponent(initialized=True, create_dependency=True),
        'vector_tonic_integrator': MockComponent(initialized=True, create_dependency=True)
    }
    
    # Verify they all pass verification
    assert verify_all_critical_components(test_context, strict=True)
    
    # Deliberately break components one by one and verify they fail
    for component_name in test_context.keys():
        # Create a fresh test context
        fresh_context = {
            name: MockComponent(initialized=True, create_dependency=True) for name in test_context.keys()
        }
        
        # Break one component
        fresh_context[component_name] = MockComponent(initialized=False, create_dependency=False)
        
        # Verify it fails verification in strict mode
        with pytest.raises(AssertionError):
            verify_all_critical_components(fresh_context, strict=True)
        
        # Verify it returns False but doesn't raise in non-strict mode
        assert not verify_all_critical_components(fresh_context, strict=False)
