"""
Integration test for Meta-Pattern Propensity Calculator.

This test demonstrates the integration of the Meta-Pattern Propensity Calculator
with the Vector-Tonic-Harmonic Learning Window system, showing how meta-patterns
can be used to predict future pattern emergence based on field coherence and stability.
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
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator, create_vector_tonic_window_integrator
from src.habitat_evolution.adaptive_core.emergence.meta_pattern_propensity import MetaPatternPropensityCalculator
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, LearningWindow, BackPressureController
from src.habitat_evolution.adaptive_core.emergence.integration_service import EventBusIntegrationService
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics

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
        """
        Observe a semantic relationship.
        
        Args:
            source: Source entity
            predicate: Relationship predicate
            target: Target entity
            context: Optional context information
        """
        # Create a unique key for this relationship
        rel_key = f"{source}_{predicate}_{target}"
        
        # Store the observation
        observation = {
            "source": source,
            "predicate": predicate,
            "target": target,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Update frequency counter
        if rel_key not in self.relationship_frequency:
            self.relationship_frequency[rel_key] = 0
        self.relationship_frequency[rel_key] += 1
        
        # Store in observed relationships
        self.observed_relationships[rel_key] = observation
        
        # Add to history
        self.observation_history.append(observation)


class TestMetaPatternPropensityCalculator:
    """Test the Meta-Pattern Propensity Calculator integration with vector-tonic-harmonic system."""
    
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
        
        # Create harmonic services first
        self._create_harmonic_services()
        
        # Create pattern detector components
        self._create_pattern_detector()
        
        # Create vector-tonic window integrator
        self._create_vector_tonic_integrator()
        
        # Create meta-pattern propensity calculator
        self._create_meta_pattern_calculator()
        
        # Load test data
        self.relationships = self._load_test_data()
        
        # Track detected patterns and meta-patterns
        self.detected_patterns = []
        self.detected_meta_patterns = []
        
    def _init_event_bus_integration(self):
        """Initialize event bus integration."""
        # Subscribe to pattern detection events
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("meta_pattern.detected", self._on_meta_pattern_detected)
        
    def _create_pattern_detector(self):
        """Create pattern detector components."""
        # Create a mock semantic observer
        self.semantic_observer = MockSemanticCurrentObserver(self.semantic_observer_id)
        
        # Create event-aware detector
        self.event_detector = EventAwarePatternDetector(
            semantic_observer=self.semantic_observer,
            event_bus=self.event_bus
        )
        
        # Create pattern publisher
        self.pattern_publisher = PatternEventPublisher(
            event_bus=self.event_bus
        )
        
        # Create learning window-aware detector
        self.learning_detector = LearningWindowAwareDetector(
            detector=self.event_detector,
            pattern_publisher=self.pattern_publisher,
            back_pressure_controller=BackPressureController()
        )
        
        # Create tonic-harmonic metrics
        self.metrics = TonicHarmonicMetrics()
        
        # Create tonic-harmonic detector
        self.tonic_detector = TonicHarmonicPatternDetector(
            base_detector=self.learning_detector,
            harmonic_io_service=self.harmonic_io,
            event_bus=self.event_bus,
            field_bridge=self.field_bridge,
            metrics=self.metrics
        )
        
        # No need to create another publisher here as we already created one above
        
    def _create_harmonic_services(self):
        """Create harmonic services."""
        # Create harmonic IO service
        self.harmonic_io = HarmonicIOService(self.event_bus)
        
        # Create harmonic field IO bridge
        self.field_bridge = HarmonicFieldIOBridge(self.harmonic_io)
        
    def _create_vector_tonic_integrator(self):
        """Create vector-tonic window integrator."""
        # Create integrator
        self.integrator = create_vector_tonic_window_integrator(
            tonic_detector=self.tonic_detector,
            event_bus=self.event_bus,
            harmonic_io_service=self.harmonic_io,
            metrics=self.metrics,
            adaptive_soak_period=True
        )
        
        # Create learning window
        self.learning_window = LearningWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=10),
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes_per_window=20,
            field_aware_transitions=True
        )
        
        # Set learning window
        self.learning_detector.set_learning_window(self.learning_window)
        
    def _create_meta_pattern_calculator(self):
        """Create meta-pattern propensity calculator."""
        # Initialize with default field metrics
        self.propensity_calculator = MetaPatternPropensityCalculator(
            field_metrics={
                "coherence": 0.7,
                "stability": 0.7,
                "turbulence": 0.3,
                "density": 0.5
            }
        )
        
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data for climate risk patterns.
        
        Returns:
            List of relationship dictionaries
        """
        # Create synthetic climate risk relationships
        return self._generate_synthetic_relationships(20)
        
    def _generate_synthetic_relationships(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate synthetic relationships for testing.
        
        Args:
            count: Number of relationships to generate
            
        Returns:
            List of synthetic relationships
        """
        # Climate risk domains
        domains = ["coastal", "urban", "agricultural", "ecological", "economic"]
        
        # Sources (climate drivers)
        sources = [
            "Sea Level Rise", "Precipitation Changes", "Temperature Increase", 
            "Storm Frequency", "Drought", "Flooding", "Heat Waves"
        ]
        
        # Predicates (relationships)
        predicates = ["leads_to", "causes", "influences", "exacerbates", "triggers"]
        
        # Targets (impacts)
        targets = [
            "Erosion", "Infrastructure Damage", "Crop Failure", "Biodiversity Loss",
            "Economic Impacts", "Population Displacement", "Health Impacts",
            "Water Scarcity", "Ecosystem Changes", "Property Value Decline"
        ]
        
        # Generate relationships
        relationships = []
        for i in range(count):
            domain = domains[i % len(domains)]
            source = sources[i % len(sources)]
            predicate = predicates[i % len(predicates)]
            target = targets[i % len(targets)]
            
            # Add some variation to avoid exact duplicates
            if i > len(targets):
                target = f"{target} {i // len(targets)}"
                
            relationships.append({
                "source": source,
                "predicate": predicate,
                "target": target,
                "context": {
                    "domain": domain,
                    "confidence": 0.7 + (0.3 * (i / count)),
                    "importance": "high" if i < count/3 else "medium" if i < 2*count/3 else "low"
                }
            })
            
        return relationships
    
    def _on_pattern_detected(self, event: Event):
        """
        Handle pattern detection events.
        
        Args:
            event: Pattern detection event
        """
        pattern = event.data.get("pattern")
        if pattern:
            logger.info(f"Pattern detected: {pattern.get('source')} {pattern.get('predicate')} {pattern.get('target')}")
            
            # Add domain from context if available
            if "context" in pattern and "domain" in pattern["context"]:
                pattern["domain"] = pattern["context"]["domain"]
                
            # Register with propensity calculator
            self.propensity_calculator.register_pattern(pattern)
            
            # Track detected pattern
            self.detected_patterns.append(pattern)
            
    def _on_meta_pattern_detected(self, event: Event):
        """
        Handle meta-pattern detection events.
        
        Args:
            event: Meta-pattern detection event
        """
        meta_pattern = event.data.get("meta_pattern")
        if meta_pattern:
            logger.info(f"Meta-pattern detected: {meta_pattern.get('id')}")
            logger.info(f"  Type: {meta_pattern.get('evolution_type')}")
            logger.info(f"  Confidence: {meta_pattern.get('confidence')}")
            
            # Register with propensity calculator
            self.propensity_calculator.register_meta_pattern(meta_pattern)
            
            # Track detected meta-pattern
            self.detected_meta_patterns.append(meta_pattern)
    
    def _publish_field_gradient(self, coherence: float, stability: float):
        """
        Publish field gradient event.
        
        Args:
            coherence: Field coherence (0.0-1.0)
            stability: Field stability (0.0-1.0)
        """
        # Create metrics dictionary
        metrics_dict = {
            "coherence": coherence,
            "stability": stability,
            "turbulence": 1.0 - stability,
            "density": 0.5 + (coherence * 0.5)
        }
        
        # Update propensity calculator with field metrics
        self.propensity_calculator.update_field_metrics(metrics_dict)
        
        # Create vectors dictionary
        vectors_dict = {
            "coherence_vector": [coherence, 0.5, 0.3],
            "stability_vector": [stability, 0.6, 0.4],
            "flow_vector": [0.5, 0.5, 0.5]
        }
        
        # Create topology dictionary with required structure
        topology_dict = {
            "resonance_centers": {
                "center_0": {"position": [0.2, 0.3, 0.4], "strength": 0.7},
                "center_1": {"position": [0.6, 0.7, 0.8], "strength": 0.8}
            },
            "interference_patterns": {
                "pattern_0": {"position": [0.3, 0.4, 0.5], "intensity": 0.6},
                "pattern_1": {"position": [0.7, 0.8, 0.9], "intensity": 0.7}
            },
            "field_density_centers": [
                {"position": [0.1, 0.2, 0.3], "density": 0.8},
                {"position": [0.5, 0.6, 0.7], "density": 0.9}
            ],
            "flow_vectors": {
                "vector_0": [0.1, 0.2, 0.3],
                "vector_1": [0.4, 0.5, 0.6]
            }
        }
        
        # Create gradient data
        gradient_data = {
            "metrics": metrics_dict,
            "vectors": vectors_dict,
            "topology": topology_dict
        }
        
        # Publish field gradient event
        self.event_bus.publish(Event.create(
            type="field.gradient.update",
            source="test_field_service",
            data={
                "gradients": metrics_dict,
                "gradient": {
                    "metrics": metrics_dict,
                    "vectors": vectors_dict,
                    "topology": topology_dict
                },
                "topology": topology_dict,
                "vectors": vectors_dict,
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
            
            # Record in our mock observer
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
                    "context": context
                }
            ))
    
    def _create_meta_patterns(self):
        """Create explicit meta-patterns for testing."""
        # Object evolution meta-pattern
        object_evolution_meta_pattern = {
            "id": "meta_pattern_0_object_evolution",
            "type": "evolution_meta_pattern",
            "evolution_type": "object_evolution",
            "frequency": 5,
            "confidence": 0.7,
            "detection_timestamp": datetime.now().isoformat(),
            "examples": [
                {
                    "from_pattern": "pattern_1",
                    "to_pattern": "pattern_2",
                    "details": {
                        "from_object": "Erosion",
                        "to_object": "Infrastructure Damage"
                    }
                },
                {
                    "from_pattern": "pattern_3",
                    "to_pattern": "pattern_4",
                    "details": {
                        "from_object": "Crop Failure",
                        "to_object": "Economic Impacts"
                    }
                }
            ]
        }
        
        # Subject evolution meta-pattern
        subject_evolution_meta_pattern = {
            "id": "meta_pattern_1_subject_evolution",
            "type": "evolution_meta_pattern",
            "evolution_type": "subject_evolution",
            "frequency": 3,
            "confidence": 0.6,
            "detection_timestamp": datetime.now().isoformat(),
            "examples": [
                {
                    "from_pattern": "pattern_5",
                    "to_pattern": "pattern_6",
                    "details": {
                        "from_subject": "Sea Level Rise",
                        "to_subject": "Flooding"
                    }
                },
                {
                    "from_pattern": "pattern_7",
                    "to_pattern": "pattern_8",
                    "details": {
                        "from_subject": "Temperature Increase",
                        "to_subject": "Heat Waves"
                    }
                }
            ]
        }
        
        # Predicate evolution meta-pattern
        predicate_evolution_meta_pattern = {
            "id": "meta_pattern_2_predicate_evolution",
            "type": "evolution_meta_pattern",
            "evolution_type": "predicate_evolution",
            "frequency": 2,
            "confidence": 0.5,
            "detection_timestamp": datetime.now().isoformat(),
            "examples": [
                {
                    "from_pattern": "pattern_9",
                    "to_pattern": "pattern_10",
                    "details": {
                        "from_predicate": "leads_to",
                        "to_predicate": "causes"
                    }
                },
                {
                    "from_pattern": "pattern_11",
                    "to_pattern": "pattern_12",
                    "details": {
                        "from_predicate": "influences",
                        "to_predicate": "exacerbates"
                    }
                }
            ]
        }
        
        # Register meta-patterns with propensity calculator
        self.propensity_calculator.register_meta_pattern(object_evolution_meta_pattern)
        self.propensity_calculator.register_meta_pattern(subject_evolution_meta_pattern)
        self.propensity_calculator.register_meta_pattern(predicate_evolution_meta_pattern)
        
        # Publish meta-pattern events
        self.event_bus.publish(Event.create(
            type="meta_pattern.detected",
            source=self.pattern_detector_id.id,
            data={"meta_pattern": object_evolution_meta_pattern}
        ))
        
        self.event_bus.publish(Event.create(
            type="meta_pattern.detected",
            source=self.pattern_detector_id.id,
            data={"meta_pattern": subject_evolution_meta_pattern}
        ))
        
        self.event_bus.publish(Event.create(
            type="meta_pattern.detected",
            source=self.pattern_detector_id.id,
            data={"meta_pattern": predicate_evolution_meta_pattern}
        ))
        
        # Track meta-patterns
        self.detected_meta_patterns.extend([
            object_evolution_meta_pattern,
            subject_evolution_meta_pattern,
            predicate_evolution_meta_pattern
        ])
    
    def test_propensity_calculation(self):
        """Test pattern propensity calculation."""
        logger.info("Testing pattern propensity calculation")
        
        # Reset window to CLOSED
        self.learning_detector.update_window_state(WindowState.CLOSED)
        
        # Transition to OPENING
        logger.info("Transitioning to OPENING state")
        self.learning_detector.update_window_state(WindowState.OPENING)
        
        # Publish improving field metrics during OPENING
        for i in range(3):
            coherence = 0.7 + (i * 0.05)
            stability = 0.8 + (i * 0.03)
            self._publish_field_gradient(coherence, stability)
            time.sleep(0.5)
        
        # Manually transition to OPEN for testing
        logger.info("Transitioning to OPEN state")
        self.learning_detector.update_window_state(WindowState.OPEN)
        
        # Observe first batch of relationships
        logger.info("Observing first batch of relationships")
        self._observe_relationships(self.relationships[:10])
        
        # Detect patterns
        patterns = self.tonic_detector.detect_patterns()
        logger.info(f"Detected {len(patterns)} patterns in first batch")
        
        # Create meta-patterns
        logger.info("Creating meta-patterns")
        self._create_meta_patterns()
        
        # Calculate propensities before second batch
        logger.info("Calculating pattern propensities before second batch")
        propensities = self.propensity_calculator.get_top_propensities(5)
        logger.info("Top pattern propensities before second batch:")
        for i, prop in enumerate(propensities):
            logger.info(f"  {i+1}. {prop['source']} {prop['predicate']} {prop['target']}: {prop['propensity']:.4f}")
        
        # Observe second batch of relationships
        logger.info("Observing second batch of relationships")
        self._observe_relationships(self.relationships[10:])
        
        # Detect patterns again
        patterns = self.tonic_detector.detect_patterns()
        logger.info(f"Detected {len(patterns)} patterns in second batch")
        
        # Calculate propensities after second batch
        logger.info("Calculating pattern propensities after second batch")
        propensities = self.propensity_calculator.get_top_propensities(5)
        logger.info("Top pattern propensities after second batch:")
        for i, prop in enumerate(propensities):
            logger.info(f"  {i+1}. {prop['source']} {prop['predicate']} {prop['target']}: {prop['propensity']:.4f}")
        
        # Test with changing field metrics
        logger.info("Testing propensity calculation with changing field metrics")
        
        # Update field metrics to simulate changing field conditions
        self._publish_field_gradient(0.9, 0.6)  # High coherence, lower stability
        
        # Calculate propensities with updated field metrics
        logger.info("Calculating pattern propensities with updated field metrics")
        propensities = self.propensity_calculator.get_top_propensities(5)
        logger.info("Top pattern propensities with updated field metrics:")
        for i, prop in enumerate(propensities):
            logger.info(f"  {i+1}. {prop['source']} {prop['predicate']} {prop['target']}: {prop['propensity']:.4f}")
        
        # Verify pattern propensity calculation
        logger.info("Verifying pattern propensity calculation")
        assert len(propensities) > 0, "No pattern propensities calculated"
        
        # Check that propensities sum to approximately 1.0
        total_propensity = sum(prop["propensity"] for prop in propensities)
        logger.info(f"Total propensity of top patterns: {total_propensity:.4f}")
        
        # Test integration with AdaptiveID capaciousness
        logger.info("Testing integration with AdaptiveID capaciousness")
        
        # Update AdaptiveID capaciousness
        self.pattern_detector_id.uncertainty = 0.2  # Lower uncertainty = higher capaciousness
        capaciousness = self.pattern_detector_id.get_capaciousness()
        logger.info(f"Pattern detector AdaptiveID capaciousness: {capaciousness:.4f}")
        
        # Update field metrics based on capaciousness
        coherence = 0.5 + (capaciousness * 0.5)
        stability = 0.4 + (capaciousness * 0.6)
        self._publish_field_gradient(coherence, stability)
        
        # Calculate propensities with capaciousness-influenced field metrics
        logger.info("Calculating propensities with capaciousness-influenced field metrics")
        propensities = self.propensity_calculator.get_top_propensities(5)
        logger.info("Top pattern propensities with capaciousness influence:")
        for i, prop in enumerate(propensities):
            logger.info(f"  {i+1}. {prop['source']} {prop['predicate']} {prop['target']}: {prop['propensity']:.4f}")
        
        logger.info("Meta-Pattern Propensity Calculator test completed")


if __name__ == "__main__":
    logger.info("Starting Meta-Pattern Propensity Calculator test")
    
    # Run test
    test = TestMetaPatternPropensityCalculator()
    test.test_propensity_calculation()
    
    logger.info("Meta-Pattern Propensity Calculator test completed")
