"""
Integration test for Vector-Tonic-Harmonic Learning Window Integration.

This test demonstrates the enhanced integration between learning windows and
vector-tonic-harmonic validation, focusing on progressive preparation during
the OPENING state and adaptive soak periods.
"""

import logging
import unittest
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Any, Optional
import numpy as np

# Use absolute imports to avoid module path issues
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.adaptive_core.emergence.event_aware_detector import EventAwarePatternDetector
from src.habitat_evolution.adaptive_core.emergence.event_bus_integration import PatternEventPublisher
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector, create_tonic_harmonic_detector
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator, create_vector_tonic_window_integrator
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController, LearningWindow
from src.habitat_evolution.adaptive_core.emergence.integration_service import EventBusIntegrationService

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s')

logger = logging.getLogger(__name__)


class MockSemanticCurrentObserver:
    """
    A simple mock of SemanticCurrentObserver for testing purposes.
    Provides the minimum functionality needed for pattern detection.
    """
    
    def __init__(self, adaptive_id: AdaptiveID):
        self.adaptive_id = adaptive_id
        self.observed_relationships = {}
        self.relationship_frequency = {}
        self.observation_history = []
    
    def observe_relationship(self, source: str, predicate: str, target: str, context: Dict[str, Any] = None):
        """Record an observed relationship with source, predicate, target and optional context."""
        # Create a unique key for this relationship
        rel_key = f"{source}_{predicate}_{target}"
        
        # Create a relationship dictionary with all required fields
        relationship = {
            'source': source,
            'predicate': predicate,
            'target': target,
            'frequency': 1,
            'last_observed': datetime.now().isoformat(),
            'first_observed': datetime.now().isoformat(),
            'contexts': [context or {}],
            'context': context or {}
        }
        
        # Store the relationship
        if rel_key not in self.observed_relationships:
            self.observed_relationships[rel_key] = relationship
            self.relationship_frequency[rel_key] = 1
        else:
            # Update existing relationship
            self.observed_relationships[rel_key]['frequency'] += 1
            self.observed_relationships[rel_key]['last_observed'] = datetime.now().isoformat()
            if context:
                self.observed_relationships[rel_key]['contexts'].append(context)
                self.observed_relationships[rel_key]['context'] = context  # Update with most recent context
            self.relationship_frequency[rel_key] += 1
        
        # Record in observation history
        self.observation_history.append({
            'relationship': rel_key,
            'timestamp': datetime.now()
        })
        
        return rel_key


class TestVectorTonicWindowIntegration:
    """Test vector-tonic-harmonic learning window integration."""
    
    def __init__(self):
        """Initialize test components."""
        # Create event bus
        self.event_bus = LocalEventBus()
        
        # Create AdaptiveIDs for components
        self.semantic_observer_id = AdaptiveID("test_semantic_observer", creator_id="test_system")
        self.pattern_detector_id = AdaptiveID("test_pattern_detector", creator_id="test_system")
        self.publisher_id = AdaptiveID("test_publisher", creator_id="test_system")
        
        # Create a mock semantic observer
        self.semantic_observer = MockSemanticCurrentObserver(self.semantic_observer_id)
        
        # Initialize event bus integration
        self._init_event_bus_integration()
        
        # Create pattern detector components
        self._create_pattern_detector()
        
        # Create harmonic services
        self._create_harmonic_services()
        
        # Create vector-tonic window integrator
        self._create_vector_tonic_integrator()
        
        # Load test data
        self.relationships = self._load_test_data()
        
    def _init_event_bus_integration(self):
        """Initialize event bus integration."""
        # Create integration service
        self.integration_service = EventBusIntegrationService(self.event_bus)
        logger.info("Initialized event bus integration service")
        
        # Integrate AdaptiveIDs with event bus
        self.integration_service.integrate_adaptive_id(self.semantic_observer_id)
        self.integration_service.integrate_adaptive_id(self.pattern_detector_id)
        
        # Note: We're using the MockSemanticCurrentObserver created in __init__
        # No need to create a new semantic observer here
        
        # Create pattern publisher
        self.pattern_publisher = self.integration_service.create_pattern_publisher(self.pattern_detector_id.id)
        logger.info(f"Created pattern publisher for {self.pattern_detector_id.id}")
        
        # Subscribe to events
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        logger.info("Subscribed to events on the event bus")
        
    def _create_pattern_detector(self):
        """Create pattern detector components."""
        # Create base pattern detector
        self.base_detector = EventAwarePatternDetector(
            semantic_observer=self.semantic_observer,
            event_bus=self.event_bus,
            pattern_publisher=self.pattern_publisher,
            threshold=3  # Require at least 3 observations for proper pattern detection
        )
        logger.info(f"Integrated pattern detector {self.pattern_detector_id.id} with event bus")
        
        # Create learning window aware detector
        self.learning_detector = LearningWindowAwareDetector(
            detector=self.base_detector,
            pattern_publisher=self.pattern_publisher,
            back_pressure_controller=BackPressureController()
        )
        
        # Create publisher for field events
        self.field_publisher = PatternEventPublisher(self.event_bus)
        logger.info(f"Created pattern publisher for {self.publisher_id.id}")
        
    def _create_harmonic_services(self):
        """Create harmonic services."""
        # Create harmonic I/O service
        self.harmonic_io_service = HarmonicIOService(self.event_bus)
        
        # Create tonic-harmonic metrics
        self.metrics = TonicHarmonicMetrics()
        
        # Create field bridge
        field_bridge = HarmonicFieldIOBridge(self.harmonic_io_service)
        
        # Create tonic-harmonic detector directly instead of using create_tonic_harmonic_detector
        self.tonic_detector = TonicHarmonicPatternDetector(
            base_detector=self.learning_detector,
            harmonic_io_service=self.harmonic_io_service,
            event_bus=self.event_bus,
            field_bridge=field_bridge,
            metrics=self.metrics
        )
        
    def _create_vector_tonic_integrator(self):
        """Create vector-tonic window integrator."""
        # Create integrator
        self.integrator = create_vector_tonic_window_integrator(
            tonic_detector=self.tonic_detector,
            event_bus=self.event_bus,
            harmonic_io_service=self.harmonic_io_service,
            metrics=self.metrics,
            adaptive_soak_period=True
        )
        
        # Create learning window
        self.learning_window = LearningWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=10),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=50,
            _state=WindowState.OPENING,
            field_aware_transitions=True
        )
        
        # Set learning window
        self.learning_detector.set_learning_window(self.learning_window)
        logger.info(f"Set learning window with state: {self.learning_detector.window_state}")
        
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data from climate risk files using the ClimateDataLoader.
        
        Returns:
            List of relationship dictionaries
        """
        # Find climate risk data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 
                               "data", "climate_risk")
        
        # Use the ClimateDataLoader to properly load and process climate risk data
        from src.habitat_evolution.adaptive_core.emergence.climate_data_loader import ClimateDataLoader
        climate_loader = ClimateDataLoader(data_dir)
        
        # Load all climate risk files
        climate_loader.load_all_files()
        
        # Get extracted relationships
        relationships = climate_loader.relationships
        logger.info(f"Loaded {len(relationships)} relationships from climate risk data")
        
        # Generate additional synthetic relationships to ensure pattern detection
        synthetic_relationships = climate_loader.generate_synthetic_relationships(count=15)
        logger.info(f"Generated {len(synthetic_relationships)} synthetic relationships")
        
        # Combine real and synthetic relationships
        relationships.extend(synthetic_relationships)
        
        logger.info(f"Loaded {len(relationships)} relationships from climate risk data")
        
        # Generate some synthetic relationships for testing
        synthetic = self._generate_synthetic_relationships(15)
        relationships.extend(synthetic)
        logger.info(f"Generated {len(synthetic)} synthetic relationships")
        
        return relationships
    
    def _generate_synthetic_relationships(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate synthetic relationships for testing.
        
        Args:
            count: Number of relationships to generate
            
        Returns:
            List of synthetic relationships
        """
        sources = ["coastal_community", "infrastructure", "natural_systems", 
                  "policy_makers", "local_government", "businesses"]
        predicates = ["faces", "adapts_to", "mitigates", "implements", 
                     "develops", "needs", "buffer"]
        targets = ["sea_level_rise", "storm_surge", "adaptation", 
                  "adaptation_measures", "resilience_plans", "climate_impacts"]
        
        relationships = []
        for _ in range(count):
            source = np.random.choice(sources)
            predicate = np.random.choice(predicates)
            target = np.random.choice(targets)
            
            relationships.append({
                "source": source,
                "predicate": predicate,
                "target": target,
                "confidence": np.random.uniform(0.7, 1.0)
            })
        
        return relationships
    
    def _on_pattern_detected(self, event: Event):
        """
        Handle pattern detection events.
        
        Args:
            event: Pattern detection event
        """
        pattern_id = event.data.get('pattern_id', 'unknown')
        relationship = event.data.get('relationship', {})
        
        source = relationship.get('source', 'None')
        predicate = relationship.get('predicate', 'None')
        target = relationship.get('target', 'None')
        
        logger.info(f"Pattern detected: {pattern_id}")
        logger.info(f"Pattern relationship: {source} {predicate} {target}")
    
    def _publish_field_gradient(self, coherence: float, stability: float):
        """
        Publish field gradient event.
        
        Args:
            coherence: Field coherence (0.0-1.0)
            stability: Field stability (0.0-1.0)
        """
        # Calculate turbulence as inverse of stability
        turbulence = 1.0 - stability
        
        # Create field gradient data
        gradient_data = {
            "metrics": {
                "coherence": coherence,
                "stability": stability,
                "turbulence": turbulence,
                "density": np.random.uniform(0.4, 0.8)
            },
            "vectors": {
                f"vector_{i}": np.random.rand(128).tolist() for i in range(10)
            }
        }
        
        # Publish field gradient event
        self.event_bus.publish(Event.create(
            type="field.gradient.update",
            source="test_field_service",
            data={
                "gradient": gradient_data,
                "timestamp": datetime.now().isoformat()
            }
        ))
        
        # Also publish field state update
        self.event_bus.publish(Event.create(
            type="field.state.updated",
            source="test_field_service",
            data={
                "field_state": {
                    "metrics": gradient_data["metrics"],
                    "effective_dimensionality": 64,
                    "timestamp": datetime.now().isoformat()
                }
            }
        ))
    
    def _observe_relationships(self, relationships: List[Dict[str, Any]]):
        """
        Observe relationships for pattern detection.
        
        Args:
            relationships: List of relationships to observe
        """
        for relationship in relationships:
            # Create observation context
            context = {
                "timestamp": datetime.now().isoformat(),
                "source": "test_observer"
            }
            
            # Add any additional context from the relationship
            if 'context' in relationship:
                context.update(relationship['context'])
            
            # Record in our mock observer using the new method signature
            self.semantic_observer.observe_relationship(
                source=relationship['source'],
                predicate=relationship['predicate'],
                target=relationship['target'],
                context=context
            )
            
            # Publish observation event
            self.event_bus.publish(Event.create(
                type="semantic.observation",
                source=self.semantic_observer_id.id,
                data={
                    "relationship": {
                        "source": relationship['source'],
                        "predicate": relationship['predicate'],
                        "target": relationship['target']
                    },
                    "context": context,
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
            logger.info(f"Observed relationship: {relationship['source']} {relationship['predicate']} {relationship['target']}")
    
    def test_progressive_preparation(self):
        """Test progressive preparation during OPENING state."""
        logger.info("Testing progressive preparation during OPENING state")
        
        # Ensure window is in OPENING state
        self.learning_detector.update_window_state(WindowState.OPENING)
        logger.info(f"Window state: {self.learning_detector.window_state}")
        
        # Set next transition time
        self.learning_detector.current_window.next_transition_time = datetime.now() + timedelta(seconds=10)
        
        # Publish field gradients to trigger progressive preparation
        logger.info("Publishing field gradients for progressive preparation")
        
        # Simulate gradual improvement in field metrics
        for i in range(5):
            coherence = 0.6 + (i * 0.05)
            stability = 0.7 + (i * 0.05)
            
            self._publish_field_gradient(coherence, stability)
            
            # Get preparation status
            status = self.integrator.get_preparation_status()
            logger.info(f"Preparation status: {status}")
            
            # Wait a bit
            time.sleep(1)
        
        # Check if vector cache was warmed
        logger.info(f"Final vector cache size: {len(self.integrator.vector_cache)}")
        
        # Observe some relationships to generate pattern candidates
        logger.info("Observing relationships to generate pattern candidates")
        self._observe_relationships(self.relationships[:10])
        
        # Wait for pattern candidate identification
        time.sleep(1)
        
        # Check if pattern candidates were identified
        logger.info(f"Pattern candidates identified: {len(self.integrator.pattern_candidates)}")
        
        # Wait for window to transition to OPEN
        remaining_time = (self.learning_detector.current_window.next_transition_time - datetime.now()).total_seconds()
        if remaining_time > 0:
            logger.info(f"Waiting {remaining_time:.1f}s for window to transition to OPEN")
            time.sleep(remaining_time + 0.5)
        
        # Check if window transitioned to OPEN
        logger.info(f"Final window state: {self.learning_detector.window_state}")
    
    def test_adaptive_soak_period(self):
        """Test adaptive soak period based on field volatility."""
        logger.info("Testing adaptive soak period based on field volatility")
        
        # Reset window to CLOSED
        self.learning_detector.update_window_state(WindowState.CLOSED)
        
        # Generate stable field history
        logger.info("Generating stable field history")
        for _ in range(5):
            self._publish_field_gradient(0.8, 0.9)  # High stability, low volatility
            time.sleep(0.1)
        
        # Transition to OPENING and check soak period
        self.learning_detector.update_window_state(WindowState.OPENING)
        soak_period = self.integrator._calculate_adaptive_soak_period()
        logger.info(f"Adaptive soak period for stable field: {soak_period:.1f}s")
        
        # Reset window to CLOSED
        self.learning_detector.update_window_state(WindowState.CLOSED)
        
        # Generate volatile field history
        logger.info("Generating volatile field history")
        for i in range(10):
            # Oscillating metrics to create volatility
            coherence = 0.5 + (0.3 * np.sin(i))
            stability = 0.5 + (0.3 * np.cos(i))
            self._publish_field_gradient(coherence, stability)
            time.sleep(0.1)
        
        # Transition to OPENING and check soak period
        self.learning_detector.update_window_state(WindowState.OPENING)
        soak_period = self.integrator._calculate_adaptive_soak_period()
        logger.info(f"Adaptive soak period for volatile field: {soak_period:.1f}s")
    
    def _create_primary_cascade_relationships(self):
        """Create explicit primary cascade relationships for testing."""
        primary_cascades = [
            {
                "cascade": "Climate Change → Sea Level Rise → Coastal Flooding",
                "importance": "high",
                "cascade_type": "primary"
            },
            {
                "cascade": "Habitat Loss → Biodiversity Decline → Ecosystem Collapse",
                "importance": "critical",
                "cascade_type": "primary"
            },
            {
                "cascade": "Policy Implementation → Community Adaptation → Resilience Building",
                "importance": "high",
                "cascade_type": "primary"
            }
        ]
        
        primary_relationships = []
        
        for cascade_info in primary_cascades:
            cascade = cascade_info["cascade"]
            steps = cascade.split(" → ")
            
            if len(steps) >= 3:
                first_step = steps[0]
                second_step = steps[1]
                third_step = steps[2]
                
                # Create first relationship in cascade
                primary_relationships.append({
                    "source": first_step,
                    "predicate": "leads_to",
                    "target": second_step,
                    "context": {
                        "relationship_type": "cascade_first_step",
                        "cascade_type": "primary",
                        "importance": cascade_info["importance"],
                        "full_cascade": cascade
                    }
                })
                
                # Create second relationship in cascade
                primary_relationships.append({
                    "source": second_step,
                    "predicate": "leads_to",
                    "target": third_step,
                    "context": {
                        "relationship_type": "cascade_second_step",
                        "cascade_type": "primary",
                        "importance": cascade_info["importance"],
                        "full_cascade": cascade
                    }
                })
        
        return primary_relationships

    def test_full_integration(self):
        """Test full integration with pattern detection."""
        logger.info("Testing full integration with pattern detection")
        
        # Reset window to CLOSED
        self.learning_detector.update_window_state(WindowState.CLOSED)
        
        # Check pattern detection with CLOSED window
        logger.info("Testing pattern detection with window CLOSED")
        self._observe_relationships(self.relationships[:5])
        patterns = self.tonic_detector.detect_patterns()
        logger.info(f"Patterns detected with window CLOSED: {len(patterns)}")
        
        # Transition to OPENING with adaptive soak period
        logger.info("Transitioning to OPENING with adaptive soak period")
        self.learning_detector.update_window_state(WindowState.OPENING)
        
        # Publish improving field metrics during OPENING
        for i in range(3):
            coherence = 0.7 + (i * 0.05)
            stability = 0.8 + (i * 0.03)
            self._publish_field_gradient(coherence, stability)
            time.sleep(1)
            
            # Check preparation status
            status = self.integrator.get_preparation_status()
            logger.info(f"Preparation status during OPENING: {status}")
        
        # Manually transition to OPEN for testing
        logger.info("Manually transitioning to OPEN state")
        self.learning_detector.update_window_state(WindowState.OPEN)
        
        # Create and observe primary cascade relationships
        primary_cascades = self._create_primary_cascade_relationships()
        logger.info(f"\nPrimary Cascades")
        for relationship in primary_cascades:
            # Observe each primary cascade relationship multiple times to exceed threshold
            for _ in range(5):  # Higher frequency for primary cascades
                self._observe_relationships([relationship])
                time.sleep(0.1)  # Small delay between observations
        
        # Test pattern detection with OPEN window (secondary cascades)
        logger.info("\nSecondary Cascades")
        self._observe_relationships(self.relationships[5:15])
        patterns = self.tonic_detector.detect_patterns()
        
        # Log detected patterns with cascade type
        logger.info(f"Patterns detected with window OPEN: {len(patterns)}")
        for pattern in patterns:
            cascade_type = pattern.get('context', {}).get('cascade_type', 'secondary')
            logger.info(f"Pattern ({cascade_type}): {pattern.get('source')} {pattern.get('predicate')} {pattern.get('target')}")
        
        # Test back pressure control
        logger.info("Testing back pressure control")
        detected_count = 0
        for i in range(5):
            patterns = self.tonic_detector.detect_patterns()
            detected_count += 1 if patterns else 0
            logger.info(f"Iteration {i+1}: Patterns detected: {len(patterns)}")
            time.sleep(0.1)
        
        logger.info(f"Detected patterns in {detected_count} out of 5 iterations")


if __name__ == "__main__":
    logger.info("Starting vector-tonic window integration test")
    
    # Run tests
    test = TestVectorTonicWindowIntegration()
    
    # Test progressive preparation
    test.test_progressive_preparation()
    
    # Test adaptive soak period
    test.test_adaptive_soak_period()
    
    # Test full integration
    test.test_full_integration()
    
    logger.info("Vector-tonic window integration test completed")
