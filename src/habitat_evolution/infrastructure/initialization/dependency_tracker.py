"""
Dependency Tracker for Habitat Evolution.

This module provides utilities for tracking and verifying component dependencies
during initialization, helping to identify and debug initialization issues.
"""

import logging
import inspect
import traceback
from typing import Dict, Any, List, Optional, Set, Callable, Tuple

logger = logging.getLogger(__name__)

class DependencyTracker:
    """
    Tracks the initialization status of components and their dependencies.
    
    This class provides utilities for tracking component dependencies and
    verifying that all required dependencies are properly initialized before
    attempting to initialize a component.
    """
    
    def __init__(self):
        """Initialize a new dependency tracker."""
        self._components = {}
        self._initialization_status = {}
        self._dependency_map = {}
        self._initialization_order = []
        self._initialization_errors = {}
        
        logger.debug("DependencyTracker initialized")
    
    def register_component(self, 
                          component_key: str, 
                          component: Any, 
                          dependencies: List[str] = None):
        """
        Register a component with the dependency tracker.
        
        Args:
            component_key: The key to use for the component
            component: The component instance
            dependencies: List of component keys that this component depends on
        """
        dependencies = dependencies or []
        
        self._components[component_key] = component
        self._dependency_map[component_key] = dependencies
        
        # Check if the component is already initialized
        if hasattr(component, '_initialized') and component._initialized:
            self._initialization_status[component_key] = True
            logger.debug(f"Component {component_key} registered (already initialized)")
        elif hasattr(component, 'is_initialized') and callable(component.is_initialized):
            try:
                is_init = component.is_initialized()
                self._initialization_status[component_key] = is_init
                logger.debug(f"Component {component_key} registered (is_initialized={is_init})")
            except Exception as e:
                self._initialization_status[component_key] = False
                logger.error(f"Error checking initialization status of {component_key}: {e}")
        else:
            self._initialization_status[component_key] = False
            logger.debug(f"Component {component_key} registered (not initialized)")
    
    def verify_dependencies(self, component_key: str) -> Tuple[bool, List[str]]:
        """
        Verify that all dependencies for a component are available and initialized.
        
        Args:
            component_key: The key of the component to verify dependencies for
            
        Returns:
            Tuple containing:
                - Boolean indicating if all dependencies are satisfied
                - List of missing or uninitialized dependencies
        """
        if component_key not in self._dependency_map:
            logger.error(f"Component {component_key} not registered with dependency tracker")
            return False, ["Component not registered"]
        
        dependencies = self._dependency_map[component_key]
        missing_deps = []
        
        for dep in dependencies:
            if dep not in self._components:
                missing_deps.append(f"{dep} (missing)")
                logger.error(f"Dependency {dep} for {component_key} is missing")
            elif not self._initialization_status.get(dep, False):
                missing_deps.append(f"{dep} (not initialized)")
                logger.error(f"Dependency {dep} for {component_key} is not initialized")
        
        all_satisfied = len(missing_deps) == 0
        
        if all_satisfied:
            logger.debug(f"All dependencies for {component_key} are satisfied")
        else:
            logger.error(f"Dependencies for {component_key} not satisfied: {missing_deps}")
        
        return all_satisfied, missing_deps
    
    def mark_initialized(self, component_key: str, success: bool = True, error: str = None):
        """
        Mark a component as initialized.
        
        Args:
            component_key: The key of the component to mark as initialized
            success: Whether initialization was successful
            error: Error message if initialization failed
        """
        if component_key not in self._components:
            logger.error(f"Component {component_key} not registered with dependency tracker")
            return
        
        self._initialization_status[component_key] = success
        
        if success:
            self._initialization_order.append(component_key)
            logger.debug(f"Component {component_key} marked as initialized")
        else:
            self._initialization_errors[component_key] = error
            logger.error(f"Component {component_key} initialization failed: {error}")
    
    def get_initialization_order(self) -> List[str]:
        """
        Get the order in which components were initialized.
        
        Returns:
            List of component keys in the order they were initialized
        """
        return self._initialization_order.copy()
    
    def get_initialization_errors(self) -> Dict[str, str]:
        """
        Get the initialization errors for components.
        
        Returns:
            Dictionary mapping component keys to error messages
        """
        return self._initialization_errors.copy()
    
    def get_dependency_map(self) -> Dict[str, List[str]]:
        """
        Get the dependency map for components.
        
        Returns:
            Dictionary mapping component keys to lists of dependency keys
        """
        return self._dependency_map.copy()
    
    def get_initialization_status(self) -> Dict[str, bool]:
        """
        Get the initialization status for components.
        
        Returns:
            Dictionary mapping component keys to initialization status
        """
        return self._initialization_status.copy()
    
    def generate_initialization_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the initialization status.
        
        Returns:
            Dictionary containing initialization information
        """
        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies()
        
        # Calculate dependency depth for each component
        dependency_depths = self._calculate_dependency_depths()
        
        # Create the report
        report = {
            'components': list(self._components.keys()),
            'initialization_status': self._initialization_status.copy(),
            'initialization_order': self._initialization_order.copy(),
            'initialization_errors': self._initialization_errors.copy(),
            'dependency_map': self._dependency_map.copy(),
            'circular_dependencies': circular_deps,
            'dependency_depths': dependency_depths
        }
        
        return report
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """
        Find circular dependencies in the dependency map.
        
        Returns:
            List of circular dependency chains
        """
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
            
            for neighbor in self._dependency_map.get(node, []):
                dfs(neighbor)
            
            path.pop()
        
        for node in self._dependency_map:
            dfs(node)
        
        return circular_deps
    
    def _calculate_dependency_depths(self) -> Dict[str, int]:
        """
        Calculate the dependency depth for each component.
        
        Returns:
            Dictionary mapping component keys to dependency depths
        """
        depths = {}
        visited = set()
        
        def calculate_depth(node):
            if node in visited:
                return depths.get(node, 0)
            
            visited.add(node)
            
            if not self._dependency_map.get(node, []):
                depths[node] = 0
                return 0
            
            max_depth = 0
            for dep in self._dependency_map.get(node, []):
                if dep in self._dependency_map:
                    dep_depth = calculate_depth(dep)
                    max_depth = max(max_depth, dep_depth)
            
            depths[node] = max_depth + 1
            return depths[node]
        
        for node in self._dependency_map:
            if node not in visited:
                calculate_depth(node)
        
        return depths


# Global dependency tracker instance
_dependency_tracker = DependencyTracker()

def get_dependency_tracker() -> DependencyTracker:
    """
    Get the global dependency tracker instance.
    
    Returns:
        The global dependency tracker instance
    """
    return _dependency_tracker

def verify_initialization(component: Any, component_name: str) -> bool:
    """
    Verify that a component is properly initialized.
    
    Args:
        component: The component to verify
        component_name: The name of the component for logging
        
    Returns:
        bool: True if the component is properly initialized
        
    Raises:
        AssertionError: If the component is not properly initialized
    """
    logger.debug(f"Verifying initialization of {component_name}")
    
    # Check if component exists
    if component is None:
        error_msg = f"Critical component {component_name} is None"
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    # Log component type for debugging
    component_type = type(component).__name__
    logger.debug(f"Component {component_name} is of type {component_type}")
    
    # Check if component has _initialized attribute
    initialized_via_attribute = False
    if hasattr(component, '_initialized'):
        logger.debug(f"Component {component_name} has _initialized attribute: {component._initialized}")
        if not component._initialized:
            error_msg = f"Critical component {component_name} reports not initialized (_initialized is False)"
            logger.error(error_msg)
            raise AssertionError(error_msg)
        initialized_via_attribute = True
    
    # Check if component has is_initialized() method
    initialized_via_method = False
    if hasattr(component, 'is_initialized') and callable(component.is_initialized):
        try:
            is_init = component.is_initialized()
            logger.debug(f"Component {component_name} is_initialized() returned: {is_init}")
            if not is_init:
                error_msg = f"Critical component {component_name} reports not initialized (is_initialized() returned False)"
                logger.error(error_msg)
                raise AssertionError(error_msg)
            initialized_via_method = True
        except Exception as e:
            error_msg = f"Error calling is_initialized() on {component_name}: {e}"
            logger.error(error_msg)
            raise AssertionError(error_msg)
    
    # Ensure at least one initialization check passed
    if not (initialized_via_attribute or initialized_via_method):
        error_msg = f"Component {component_name} has no initialization status indicators (_initialized attribute or is_initialized() method)"
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    logger.debug(f"Component {component_name} is properly initialized")
    return True

def verify_dependencies(component_key: str) -> bool:
    """
    Verify that all dependencies for a component are available and initialized.
    
    Args:
        component_key: The key of the component to verify dependencies for
        
    Returns:
        bool: True if all dependencies are satisfied
        
    Raises:
        AssertionError: If any dependencies are missing or not initialized
    """
    tracker = get_dependency_tracker()
    all_satisfied, missing_deps = tracker.verify_dependencies(component_key)
    
    if not all_satisfied:
        error_msg = f"Dependencies for {component_key} not satisfied: {missing_deps}"
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    return True

def initialize_with_dependencies(component_key: str, 
                                initialize_func: Callable, 
                                *args, **kwargs) -> Any:
    """
    Initialize a component after verifying its dependencies.
    
    Args:
        component_key: The key of the component to initialize
        initialize_func: The function to call to initialize the component
        *args: Positional arguments to pass to the initialization function
        **kwargs: Keyword arguments to pass to the initialization function
        
    Returns:
        The result of the initialization function
        
    Raises:
        AssertionError: If dependencies are not satisfied or initialization fails
    """
    tracker = get_dependency_tracker()
    
    # Verify dependencies
    try:
        verify_dependencies(component_key)
    except AssertionError as e:
        tracker.mark_initialized(component_key, False, str(e))
        raise
    
    # Initialize the component
    try:
        logger.debug(f"Initializing component {component_key}")
        result = initialize_func(*args, **kwargs)
        tracker.mark_initialized(component_key, True)
        return result
    except Exception as e:
        error_msg = f"Error initializing {component_key}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        tracker.mark_initialized(component_key, False, str(e))
        raise AssertionError(error_msg) from e
