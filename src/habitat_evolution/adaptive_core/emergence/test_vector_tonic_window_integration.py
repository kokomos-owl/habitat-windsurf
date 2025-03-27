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
        
        # Track detected patterns
        self.detected_patterns = []
        
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
        pattern_data = event.data
        pattern_id = pattern_data.get('pattern_id', pattern_data.get('id', 'unknown'))
        pattern_type = pattern_data.get('type', 'standard')
        relationship = pattern_data.get('relationship', {})
        
        # Store the detected pattern for later use in topology metrics
        self.detected_patterns.append(pattern_data)
        
        # Handle meta-patterns differently
        if pattern_type == "meta" or pattern_id.startswith("meta_pattern"):
            evolution_type = pattern_data.get('evolution_type', 'object_evolution')
            frequency = pattern_data.get('frequency', 0)
            confidence = pattern_data.get('confidence', 0.0)
            examples = pattern_data.get('examples', [])
            
            logger.info(f"Detected meta-pattern: {pattern_id}")
            logger.info(f"  Evolution type: {evolution_type}")
            logger.info(f"  Frequency: {frequency}")
            logger.info(f"  Confidence: {confidence}")
            logger.info(f"  Examples: {len(examples)} instances")
            
            # Log examples
            for i, example in enumerate(examples[:5]):  # Show up to 5 examples
                logger.info(f"  Example {i+1}: {example}")
                
            # Apply feedback loop adjustments if we have harmonic services
            if hasattr(self, 'harmonic_io_service') and self.harmonic_io_service is not None:
                self._apply_meta_pattern_feedback(pattern_data)
        else:
            # Standard pattern detection
            source = relationship.get('source', pattern_data.get('source', 'None'))
            predicate = relationship.get('predicate', pattern_data.get('predicate', 'None'))
            target = relationship.get('target', pattern_data.get('target', 'None'))
            
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
            },
            "topology": {
                "resonance_centers": [
                    {
                        "id": f"center_{i}",
                        "position": [np.random.uniform(0, 1), np.random.uniform(0, 1)],
                        "strength": np.random.uniform(0.5, 0.9),
                        "radius": np.random.uniform(0.1, 0.3),
                        "stability": stability
                    } for i in range(3)
                ],
                "interference_patterns": [
                    {
                        "id": f"interference_{i}",
                        "position": [np.random.uniform(0, 1), np.random.uniform(0, 1)],
                        "strength": np.random.uniform(0.3, 0.7),
                        "type": np.random.choice(["constructive", "destructive", "neutral"])
                    } for i in range(2)
                ],
                "field_density_centers": [
                    {
                        "id": f"density_{i}",
                        "position": [np.random.uniform(0, 1), np.random.uniform(0, 1)],
                        "density": np.random.uniform(0.6, 0.9)
                    } for i in range(4)
                ],
                "flow_vectors": [
                    {
                        "id": f"flow_{i}",
                        "start": [np.random.uniform(0, 1), np.random.uniform(0, 1)],
                        "end": [np.random.uniform(0, 1), np.random.uniform(0, 1)],
                        "strength": np.random.uniform(0.4, 0.8)
                    } for i in range(5)
                ]
            }
        }
        
        # Publish field gradient event with carefully structured data
        # Ensure all topology data is properly formatted as dictionaries
        # Convert any list-based topology data to dictionary format
        
        # Create a copy of the topology data to avoid modifying the original
        topology_dict = {}
        
        # Ensure resonance_centers is a dictionary
        resonance_centers_dict = {}
        for i, center in enumerate(gradient_data["topology"]["resonance_centers"]):
            resonance_centers_dict[f"center_{i}"] = center
        topology_dict["resonance_centers"] = resonance_centers_dict
        
        # Ensure interference_patterns is a dictionary
        interference_patterns_dict = {}
        for i, pattern in enumerate(gradient_data["topology"]["interference_patterns"]):
            interference_patterns_dict[f"pattern_{i}"] = pattern
        topology_dict["interference_patterns"] = interference_patterns_dict
        
        # Ensure field_density_centers is a dictionary
        density_centers_dict = {}
        for i, center in enumerate(gradient_data["topology"]["field_density_centers"]):
            density_centers_dict[f"density_{i}"] = center
        topology_dict["field_density_centers"] = density_centers_dict
        
        # Ensure flow_vectors is a dictionary
        flow_vectors_dict = {}
        for i, vector in enumerate(gradient_data["topology"]["flow_vectors"]):
            flow_vectors_dict[f"flow_{i}"] = vector
        topology_dict["flow_vectors"] = flow_vectors_dict
        
        # Publish the event with properly structured data
        self.event_bus.publish(Event.create(
            type="field.gradient.update",
            source="test_field_service",
            data={
                "gradients": gradient_data["metrics"],  # Match what event_aware_detector expects
                "gradient": {  # Match what vector_tonic_window_integration expects
                    "metrics": gradient_data["metrics"],
                    "vectors": gradient_data["vectors"],
                    "topology": topology_dict  # Use dictionary-based topology data
                },
                "topology": topology_dict,  # Include dictionary-based topology at the top level
                "vectors": gradient_data["vectors"],   # Include vectors at the top level
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
                    "timestamp": datetime.now().isoformat(),
                    "topology": {
                        # Include topology data in the expected structure
                        "effective_dimensionality": 64,
                        "principal_dimensions": [
                            {
                                "eigenvalue": 0.8,
                                "explained_variance": 0.5,
                                "eigenvector": [0.1, 0.2, 0.3]
                            },
                            {
                                "eigenvalue": 0.6,
                                "explained_variance": 0.3,
                                "eigenvector": [0.4, 0.5, 0.6]
                            },
                            {
                                "eigenvalue": 0.4,
                                "explained_variance": 0.2,
                                "eigenvector": [0.7, 0.8, 0.9]
                            }
                        ],
                        "eigenvalues": {"0": 0.8, "1": 0.6, "2": 0.4},
                        "eigenvectors": {"0": [0.1, 0.2, 0.3], "1": [0.4, 0.5, 0.6], "2": [0.7, 0.8, 0.9]},
                        # Include the topology data from gradient_data
                        "resonance_centers": gradient_data["topology"]["resonance_centers"],
                        "interference_patterns": gradient_data["topology"]["interference_patterns"],
                        "field_density_centers": gradient_data["topology"]["field_density_centers"],
                        "flow_vectors": gradient_data["topology"]["flow_vectors"]
                    },
                    # Add density information required by TonicHarmonicFieldState
                    "density": {
                        "density_centers": [
                            {
                                "index": i,
                                "position": [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)],
                                "density": center["density"],
                                "influence_radius": np.random.uniform(0.1, 0.3)
                            } for i, center in enumerate(gradient_data["topology"]["field_density_centers"])
                        ],
                        "density_map": [[np.random.uniform(0, 1) for _ in range(3)] for _ in range(3)]
                    },
                    # Add field properties required by TonicHarmonicFieldState
                    "field_properties": {
                        "coherence": gradient_data["metrics"]["coherence"],
                        "navigability_score": np.random.uniform(0.5, 0.9),
                        "stability": gradient_data["metrics"]["stability"]
                    },
                    # Add patterns required by TonicHarmonicFieldState
                    "patterns": [],
                    # Include additional field state properties needed by handlers
                    "resonance_history": [],
                    "resonance_relationships": {},
                    "pattern_relationships": {}
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


    def _apply_meta_pattern_feedback(self, pattern_data):
        """
        Apply feedback loop adjustments based on meta-pattern detection.
        
        Args:
            pattern_data: Meta-pattern data
        """
        pattern_id = pattern_data.get("id", pattern_data.get("pattern_id", "unknown"))
        evolution_type = pattern_data.get("evolution_type", "object_evolution")
        confidence = pattern_data.get("confidence", 0.0)
        frequency = pattern_data.get("frequency", 0)
        
        # Calculate impact score based on confidence and frequency
        frequency_factor = min(frequency / 10.0, 1.0)  # Normalize frequency to 0.0-1.0
        impact_score = confidence * frequency_factor
        
        logger.info(f"Meta-pattern impact score: {impact_score:.4f}")
        logger.info(f"  Based on confidence: {confidence:.2f}, frequency factor: {frequency_factor:.2f}")
        
        # Get current harmonic parameters
        current_base_freq = self.harmonic_io_service.base_frequency
        current_stability = self.harmonic_io_service.eigenspace_stability
        current_coherence = self.harmonic_io_service.pattern_coherence
        
        # Adjust parameters based on pattern type and impact score
        new_base_freq = current_base_freq
        new_stability = current_stability
        new_coherence = current_coherence
        
        if evolution_type == "object_evolution":
            # Object evolution: increase frequency and coherence
            new_base_freq = current_base_freq * (1.0 + (impact_score * 0.5))
            new_stability = current_stability * (1.0 + (impact_score * 0.2))
            new_coherence = current_coherence * (1.0 + (impact_score * 0.3))
        elif evolution_type == "causal_cascade":
            # Causal cascade: increase stability and coherence
            new_base_freq = current_base_freq * (1.0 + (impact_score * 0.3))
            new_stability = current_stability * (1.0 + (impact_score * 0.5))
            new_coherence = current_coherence * (1.0 + (impact_score * 0.2))
        elif evolution_type == "convergent_influence":
            # Convergent influence: increase coherence and frequency
            new_base_freq = current_base_freq * (1.0 + (impact_score * 0.4))
            new_stability = current_stability * (1.0 + (impact_score * 0.1))
            new_coherence = current_coherence * (1.0 + (impact_score * 0.4))
        
        # Apply the adjusted parameters
        self.harmonic_io_service.base_frequency = new_base_freq
        self.harmonic_io_service.eigenspace_stability = new_stability
        self.harmonic_io_service.pattern_coherence = new_coherence
        
        logger.info(f"Adjusted harmonic parameters based on meta-pattern: {evolution_type}")
        logger.info(f"  Base frequency: {current_base_freq:.4f} → {new_base_freq:.4f}")
        logger.info(f"  Eigenspace stability: {current_stability:.4f} → {new_stability:.4f}")
        logger.info(f"  Pattern coherence: {current_coherence:.4f} → {new_coherence:.4f}")
        
        # Generate and publish updated topology metrics after adjustment
        self._publish_field_gradient_with_topology(new_coherence, new_stability)
    
    def _publish_field_gradient_with_topology(self, coherence: float, stability: float):
        """
        Publish field gradient with enhanced topology metrics.
        
        Args:
            coherence: Field coherence (0.0-1.0)
            stability: Field stability (0.0-1.0)
        """
        # Calculate topology metrics based on detected patterns
        pattern_count = len(self.detected_patterns)
        meta_pattern_count = sum(1 for p in self.detected_patterns if p.get('type') == 'meta' or str(p.get('id', '')).startswith('meta_pattern'))
        
        # Enhanced topology metrics
        topology_metrics = {
            "pattern_count": pattern_count,
            "meta_pattern_count": meta_pattern_count,
            "resonance_density": min(0.3 + (pattern_count * 0.05), 0.9),
            "interference_complexity": min(0.2 + (meta_pattern_count * 0.1), 0.8),
            "flow_coherence": coherence,
            "stability_trend": stability,
            "effective_dimensionality": min(3 + (pattern_count // 5), 7)
        }
        
        logger.info(f"Publishing enhanced topology metrics:")
        for key, value in topology_metrics.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Call the standard field gradient publisher with the enhanced metrics
        self._publish_field_gradient(coherence, stability)
    
    def test_feedback_loop_with_topology(self):
        """
        Test feedback loop with topology metrics extraction and visualization.
        """
        logger.info("\n=== Testing Feedback Loop with Topology Metrics ===\n")
        
        # Reset window to CLOSED
        self.learning_detector.update_window_state(WindowState.CLOSED)
        
        # Store initial parameters
        initial_frequency = self.harmonic_io_service.base_frequency
        initial_stability = self.harmonic_io_service.eigenspace_stability
        initial_coherence = self.harmonic_io_service.pattern_coherence
        
        logger.info(f"Initial parameters:")
        logger.info(f"  Base frequency: {initial_frequency:.4f}")
        logger.info(f"  Eigenspace stability: {initial_stability:.4f}")
        logger.info(f"  Pattern coherence: {initial_coherence:.4f}")
        
        # Transition to OPENING
        logger.info("\nTransitioning to OPENING state")
        self.learning_detector.update_window_state(WindowState.OPENING)
        
        # Publish initial field gradient with topology
        logger.info("\nPublishing initial field gradient with topology")
        self._publish_field_gradient_with_topology(0.7, 0.7)
        time.sleep(1)
        
        # Transition to OPEN
        logger.info("\nTransitioning to OPEN state")
        self.learning_detector.update_window_state(WindowState.OPEN)
        
        # Create and observe primary cascade relationships
        primary_cascades = self._create_primary_cascade_relationships()
        logger.info(f"\nObserving Primary Cascades")
        for relationship in primary_cascades:
            # Observe each primary cascade relationship multiple times to exceed threshold
            for _ in range(5):  # Higher frequency for primary cascades
                self._observe_relationships([relationship])
                time.sleep(0.1)  # Small delay between observations
        
        # Test pattern detection with OPEN window (secondary cascades)
        logger.info("\nObserving Secondary Cascades")
        self._observe_relationships(self.relationships[5:15])
        patterns = self.tonic_detector.detect_patterns()
        
        # Publish updated field gradient with topology after pattern detection
        logger.info("\nPublishing updated field gradient with topology")
        self._publish_field_gradient_with_topology(0.8, 0.8)
        time.sleep(1)
        
        # Get final metrics
        if hasattr(self.harmonic_io_service, 'get_metrics'):
            final_metrics = self.harmonic_io_service.get_metrics()
            
            logger.info("\nFinal System State:")
            if "system_state" in final_metrics:
                for key, value in final_metrics["system_state"].items():
                    logger.info(f"  {key}: {value}")
            
            logger.info("\nFinal Topology Metrics:")
            if "topology" in final_metrics:
                for key, value in final_metrics["topology"].items():
                    logger.info(f"  {key}: {value}")
        
        # Verify parameter adjustments
        logger.info("\nParameter adjustments summary:")
        logger.info(f"  Base frequency: {initial_frequency:.4f} → {self.harmonic_io_service.base_frequency:.4f}")
        logger.info(f"  Eigenspace stability: {initial_stability:.4f} → {self.harmonic_io_service.eigenspace_stability:.4f}")
        logger.info(f"  Pattern coherence: {initial_coherence:.4f} → {self.harmonic_io_service.pattern_coherence:.4f}")
        
        # Close the window
        logger.info("\nClosing learning window")
        self.learning_detector.update_window_state(WindowState.CLOSED)


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
    
    # Test feedback loop with topology metrics
    test.test_feedback_loop_with_topology()
    
    logger.info("Vector-tonic window integration test completed")
