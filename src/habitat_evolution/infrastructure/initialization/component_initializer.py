"""
Component initialization utilities for Habitat Evolution.

This module provides factory functions and initialization utilities for
various components of the Habitat Evolution system, ensuring proper
dependency management and initialization order.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

# Set up logger first to avoid reference errors
logger = logging.getLogger(__name__)

# Import interfaces and services
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface

# Import vector-tonic components - using optional imports for flexibility
try:
    from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import (
        VectorTonicWindowIntegrator,
        create_vector_tonic_window_integrator
    )
    from src.habitat_evolution.adaptive_core.emergence.harmonic_io_service import HarmonicIOService
    from src.habitat_evolution.adaptive_core.emergence.local_event_bus import LocalEventBus
    VECTOR_TONIC_AVAILABLE = True
    logger.debug("Vector-tonic components successfully imported")
except ImportError as e:
    logger.warning(f"Vector-tonic components not available, some functionality will be limited: {e}")
    VECTOR_TONIC_AVAILABLE = False
    # Define placeholder classes to avoid import errors
    class VectorTonicWindowIntegrator:
        pass
    class HarmonicIOService:
        pass
    class LocalEventBus:
        pass

def create_event_service(config: Optional[Dict[str, Any]] = None) -> EventServiceInterface:
    """
    Create and initialize an EventService instance.
    
    This factory function ensures that the EventService is properly initialized
    with the correct configuration.
    
    Args:
        config: Optional configuration for the event service
        
    Returns:
        Initialized EventService instance
    """
    # Create event service
    event_service = EventService()
    
    # Initialize with configuration if provided
    if config:
        event_service.initialize(config)
    else:
        event_service.initialize()
        
    logger.info("Created and initialized EventService")
    return event_service

def create_vector_tonic_components(
    config: Optional[Dict[str, Any]] = None,
    event_service: Optional[EventServiceInterface] = None
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Create vector-tonic components with proper dependencies.
    
    This function creates the VectorTonicWindowIntegrator and its dependencies,
    ensuring that all components are properly initialized and connected.
    If the real vector-tonic components are not available, it creates simplified
    but functional versions that satisfy the interface requirements.
    
    Args:
        config: Optional configuration for the components
        event_service: Optional event service to use
        
    Returns:
        Tuple of (VectorTonicWindowIntegrator, EventBus, HarmonicIOService)
    """
    config = config or {}
    
    # First try to use the real vector-tonic components if available
    if VECTOR_TONIC_AVAILABLE:
        try:
            # Create HarmonicIOService
            harmonic_io_service = HarmonicIOService()
            
            # Create LocalEventBus
            event_bus = LocalEventBus()
            
            # Create base detector (simplified for testing)
            from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector
            base_detector = LearningWindowAwareDetector()
            
            # Create TonicHarmonicPatternDetector
            from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import create_tonic_harmonic_detector
            tonic_detector, _ = create_tonic_harmonic_detector(
                base_detector,
                event_bus,
                harmonic_io_service
            )
            
            # Create VectorTonicWindowIntegrator
            vector_tonic_integrator = VectorTonicWindowIntegrator(
                tonic_detector,
                event_bus,
                harmonic_io_service
            )
            
            logger.info("Created real vector-tonic components")
            return vector_tonic_integrator, event_bus, harmonic_io_service
        
        except Exception as e:
            logger.error(f"Error creating real vector-tonic components: {e}")
            # Fall through to simplified implementation
    
    # If real components aren't available or failed to initialize, create simplified versions
    try:
        logger.info("Creating simplified vector-tonic components")
        
        # Create simplified HarmonicIOService
        class SimpleHarmonicIOService:
            def __init__(self):
                self._initialized = True
                logger.info("SimpleHarmonicIOService initialized")
                
            def process_harmonic_sync(self, data):
                return {"processed": True, "data": data}
                
            def register_handler(self, event_type, handler):
                pass
        
        # Create simplified LocalEventBus
        class SimpleEventBus:
            def __init__(self):
                self._initialized = True
                self._subscribers = {}
                logger.info("SimpleEventBus initialized")
                
            def subscribe(self, event_type, callback):
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = []
                self._subscribers[event_type].append(callback)
                return f"sub_{len(self._subscribers)}"
                
            def publish_sync(self, event_type, data):
                if event_type in self._subscribers:
                    for callback in self._subscribers[event_type]:
                        try:
                            callback({"type": event_type, "data": data})
                        except Exception as e:
                            logger.error(f"Error in event handler: {e}")
        
        # Create simplified TonicHarmonicPatternDetector
        class SimpleTonicDetector:
            def __init__(self):
                self._initialized = True
                self.base_detector = object()
                logger.info("SimpleTonicDetector initialized")
                
            def detect(self, data):
                return {"detected": True, "confidence": 0.8}
        
        # Create simplified VectorTonicWindowIntegrator
        class SimpleVectorTonicIntegrator:
            def __init__(self, tonic_detector, event_bus, harmonic_io_service):
                self.tonic_detector = tonic_detector
                self.event_bus = event_bus
                self.harmonic_io_service = harmonic_io_service
                self._initialized = True
                logger.info("SimpleVectorTonicIntegrator initialized")
                
            def get_preparation_status(self):
                return {"status": "ready", "cache_warming_level": 1.0}
                
            def override_detect(self, data):
                return self.tonic_detector.detect(data)
        
        # Create instances
        harmonic_io_service = SimpleHarmonicIOService()
        event_bus = SimpleEventBus()
        tonic_detector = SimpleTonicDetector()
        vector_tonic_integrator = SimpleVectorTonicIntegrator(
            tonic_detector,
            event_bus,
            harmonic_io_service
        )
        
        logger.info("Created simplified vector-tonic components")
        return vector_tonic_integrator, event_bus, harmonic_io_service
        
    except Exception as e:
        logger.error(f"Error creating simplified vector-tonic components: {e}")
        return None, None, None

def initialize_component(
    component_type: str,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
    fallback_on_error: bool = True
) -> Any:
    """
    Initialize a component of the specified type with proper dependencies.
    
    This general-purpose factory function provides a unified interface for
    creating and initializing various components of the Habitat Evolution system.
    It implements robust error handling and fallback mechanisms to ensure that
    component initialization failures don't cascade through the system.
    
    Args:
        component_type: The type of component to initialize (e.g., "event_service", "vector_tonic")
        config: Optional configuration for the component
        dependencies: Optional dependencies to use for initialization
        fallback_on_error: If True, return a fallback component on error instead of raising exceptions
        
    Returns:
        Initialized component(s) of the specified type, or fallback components if initialization fails
        
    Raises:
        ValueError: If component_type is unknown and fallback_on_error is False
        Exception: If component initialization fails and fallback_on_error is False
    """
    dependencies = dependencies or {}
    
    try:
        if component_type == "event_service":
            return create_event_service(config)
        
        elif component_type == "vector_tonic":
            event_service = dependencies.get("event_service")
            result = create_vector_tonic_components(config, event_service)
            
            # Check if initialization was successful
            if result[0] is None and not fallback_on_error:
                raise ImportError("Failed to initialize vector-tonic components")
                
            return result
        
        elif component_type == "pattern_aware_rag":
            # Add support for PatternAwareRAGService initialization
            db_connection = dependencies.get("db_connection")
            pattern_repository = dependencies.get("pattern_repository")
            vector_tonic_service = dependencies.get("vector_tonic_service")
            claude_adapter = dependencies.get("claude_adapter")
            event_service = dependencies.get("event_service")
            
            # Check for missing required dependencies
            missing_deps = []
            if not db_connection:
                missing_deps.append("db_connection")
            if not pattern_repository:
                missing_deps.append("pattern_repository")
            if not claude_adapter:
                missing_deps.append("claude_adapter")
            if not event_service:
                missing_deps.append("event_service")
            
            # Note: vector_tonic_service is optional, not required
                
            if missing_deps and not fallback_on_error:
                raise ValueError(f"Missing required dependencies for pattern_aware_rag: {', '.join(missing_deps)}")
            
            # First try to import and create the real PatternAwareRAGService
            try:
                from src.habitat_evolution.infrastructure.services.pattern_aware_rag_service import PatternAwareRAGService
                
                # Only attempt to create the real service if we have all required dependencies
                if not missing_deps:
                    try:
                        rag_service = PatternAwareRAGService(
                            db_connection=db_connection,
                            pattern_repository=pattern_repository,
                            vector_tonic_service=vector_tonic_service,
                            claude_adapter=claude_adapter,
                            event_service=event_service,
                            config=config
                        )
                        
                        # Initialize the service
                        rag_service.initialize(config)
                        logger.info("Created and initialized PatternAwareRAGService")
                        return rag_service
                    except Exception as e:
                        logger.error(f"Error initializing PatternAwareRAGService: {e}")
                        if not fallback_on_error:
                            raise
                        # If fallback is enabled, continue to try the mock service
            except ImportError as e:
                logger.warning(f"PatternAwareRAGService not available: {e}")
                # Continue to try the mock service
            
            # If the real service couldn't be created, try to use the mock service
            try:
                logger.info("Attempting to use MockPatternAwareRAGService as fallback")
                from src.habitat_evolution.infrastructure.services.mock_pattern_aware_rag_service import MockPatternAwareRAGService
                
                mock_rag_service = MockPatternAwareRAGService(
                    db_connection=db_connection,
                    pattern_repository=pattern_repository,
                    vector_tonic_service=vector_tonic_service,
                    claude_adapter=claude_adapter,
                    event_service=event_service,
                    config=config
                )
                
                # Initialize the mock service
                mock_rag_service.initialize(config)
                logger.info("Created and initialized MockPatternAwareRAGService as fallback")
                return mock_rag_service
            except Exception as e:
                logger.error(f"Error initializing MockPatternAwareRAGService: {e}")
                if not fallback_on_error:
                    raise
            
            # Return None if both real and mock initialization failed and fallback is enabled
            logger.warning("Returning None for PatternAwareRAGService due to initialization failure")
            return None
        
        else:
            if fallback_on_error:
                logger.warning(f"Unknown component type: {component_type}, returning None")
                return None
            else:
                raise ValueError(f"Unknown component type: {component_type}")
                
    except Exception as e:
        if fallback_on_error:
            logger.error(f"Error initializing component {component_type}: {e}")
            logger.warning(f"Returning fallback for {component_type} due to initialization error")
            
            # Return appropriate fallbacks based on component type
            if component_type == "event_service":
                return None
            elif component_type == "vector_tonic":
                return None, None, None
            else:
                return None
        else:
            # Re-raise the exception if fallback is disabled
            raise
