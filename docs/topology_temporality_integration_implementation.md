# Topology-Temporality Integration Implementation

## Overview

This document outlines the implementation plan for integrating topology and temporality in the Habitat Evolution system. The integration follows a field-first philosophy, treating both dimensions as emergent properties of a unified semantic field while leveraging the existing AdaptiveID system as the central reference point.

## Core Principles

1. **Field-First Integration**: Topology and temporality are emergent properties of the same underlying field.
2. **Oscillatory Foundations**: All relationships are fundamentally oscillatory, with natural frequencies, amplitudes, and phases.
3. **Adaptive Homeostasis**: The system maintains stability while evolving through feedback mechanisms.
4. **Bidirectional Coherence**: Changes propagate bidirectionally across dimensions, maintaining coherence.
5. **Gradient-Based Evolution**: Evolution follows natural gradients in the field, not imposed structural changes.
6. **AdaptiveID Centrality**: AdaptiveID serves as the central reference point across both dimensions.

## Implementation Phases

### Phase 1: Core Components Enhancement

#### Definition
Extend existing core components with oscillatory properties and bidirectional awareness.

#### Tasks

1. **Extend TonicHarmonicFieldState**
   - Add oscillatory properties to field state
   - Implement energy distribution system
   - Add homeostasis mechanisms

2. **Enhance AdaptiveID**
   - Add oscillatory properties
   - Implement back pressure mechanism
   - Add energy regulation

3. **Create Oscillatory Relationship Base Class**
   - Define common oscillatory properties for relationships
   - Implement phase coherence calculations
   - Support bidirectional propagation

#### Test Cases

1. **Field State Oscillation Tests**
   - Test energy distribution across dimensions
   - Verify homeostasis mechanisms maintain stability
   - Validate coherence regulation

2. **AdaptiveID Oscillation Tests**
   - Test oscillatory property updates
   - Verify back pressure prevents system shocks
   - Validate energy regulation maintains stability

3. **Relationship Oscillation Tests**
   - Test bidirectional propagation of changes
   - Verify phase coherence calculations
   - Validate relationship stability under load

### Phase 2: Topology-Temporality Bridge

#### Definition
Implement bidirectional bridge between topology and temporality components.

#### Tasks

1. **Create TopologyTemporalityBridge**
   - Implement bidirectional update mechanisms
   - Add oscillatory property propagation
   - Support event-based communication

2. **Enhance Topology Manager**
   - Add temporal awareness to topology constructs
   - Implement oscillatory frequency domains
   - Support bidirectional updates

3. **Enhance Temporal Pattern Manager**
   - Add topology awareness to temporal patterns
   - Implement oscillatory pattern evolution
   - Support bidirectional updates

#### Test Cases

1. **Bridge Communication Tests**
   - Test bidirectional updates between dimensions
   - Verify oscillatory property propagation
   - Validate event handling

2. **Topology-Temporal Coherence Tests**
   - Test coherence between dimensions under various loads
   - Verify stability during rapid changes
   - Validate pattern emergence across dimensions

3. **Oscillatory Synchronization Tests**
   - Test phase synchronization between dimensions
   - Verify frequency entrainment
   - Validate harmonic relationships

### Phase 3: Actant Journey Integration

#### Definition
Implement actant journey tracking across topology and temporality.

#### Tasks

1. **Create TopologyTemporalActantJourney**
   - Leverage AdaptiveID for identity
   - Implement journey point tracking
   - Support oscillatory transitions

2. **Enhance ActantJourneyRepository**
   - Add topology-temporality awareness
   - Implement oscillatory journey tracking
   - Support back pressure mechanisms

3. **Create ActantJourneyAnalyzer**
   - Implement pattern detection across dimensions
   - Add stability analysis for journeys
   - Support feedback to field state

#### Test Cases

1. **Actant Journey Tracking Tests**
   - Test journey point recording across dimensions
   - Verify transition detection and analysis
   - Validate oscillatory property tracking

2. **Journey Stability Tests**
   - Test journey stability under various conditions
   - Verify homeostasis mechanisms maintain coherence
   - Validate back pressure prevents system shocks

3. **Cross-Dimensional Pattern Tests**
   - Test pattern detection across dimensions
   - Verify feedback to field state
   - Validate emergent meta-patterns

### Phase 4: Field State Integration

#### Definition
Integrate field state with topology and temporality for coherent field evolution.

#### Tasks

1. **Enhance Field State Manager**
   - Add topology-temporality awareness
   - Implement oscillatory field metrics
   - Support gradient-based evolution

2. **Create Field Gradient System**
   - Implement natural gradients across dimensions
   - Add energy flow along gradients
   - Support coherence regulation

3. **Implement Field Observer System**
   - Create observers for topology-temporality
   - Implement bidirectional feedback
   - Support field homeostasis

#### Test Cases

1. **Field Evolution Tests**
   - Test field evolution under various conditions
   - Verify gradient-based changes
   - Validate stability during evolution

2. **Coherence Regulation Tests**
   - Test coherence regulation mechanisms
   - Verify stability under perturbations
   - Validate homeostasis across dimensions

3. **Observer Feedback Tests**
   - Test observer notifications
   - Verify bidirectional feedback
   - Validate system response to observations

## Implementation Classes

### 1. OscillatoryFieldState

```python
class OscillatoryFieldState(TonicHarmonicFieldState):
    """
    Extends TonicHarmonicFieldState with enhanced oscillatory properties
    that integrate topology and temporality.
    """
    
    def __init__(self, field_analysis):
        super().__init__(field_analysis)
        
        # Oscillatory system properties
        self.oscillatory_properties = {
            "resonance_frequency": 0.1,  # Base resonance frequency
            "phase_coherence": 0.5,      # Phase coherence across the field
            "energy_distribution": {     # Current energy distribution
                "topology": 0.33,
                "temporality": 0.33,
                "identity": 0.33,
                "reserve": 0.01
            }
        }
        
        # Homeostasis parameters
        self.homeostasis = {
            "coherence_threshold": 0.3,   # Minimum coherence threshold
            "rigidity_threshold": 0.8,    # Maximum rigidity threshold
            "energy_reserve": 0.1,        # Energy reserve for stability
            "recovery_rate": 0.05         # Recovery rate after perturbations
        }
    
    def regulate_coherence(self):
        """
        Apply gentle regulation to maintain field coherence within healthy bounds.
        """
        current_coherence = self.get_coherence_metric()
        
        if current_coherence < self.homeostasis["coherence_threshold"]:
            # Apply coherence-increasing feedback
            self._increase_coherence()
        elif current_coherence > self.homeostasis["rigidity_threshold"]:
            # Apply rigidity-reducing feedback
            self._reduce_rigidity()
    
    def distribute_energy(self, event_energy):
        """
        Distribute energy across the field based on current needs.
        """
        # Get current energy distribution
        distribution = self.oscillatory_properties["energy_distribution"]
        
        # Calculate topology energy need based on boundary activity
        topology_need = self._calculate_topology_energy_need()
        
        # Calculate temporality energy need based on pattern evolution
        temporality_need = self._calculate_temporality_energy_need()
        
        # Calculate identity energy need based on actant activity
        identity_need = self._calculate_identity_energy_need()
        
        # Calculate total need
        total_need = topology_need + temporality_need + identity_need
        
        # Calculate new distribution
        if total_need > 0:
            new_distribution = {
                "topology": topology_need / total_need,
                "temporality": temporality_need / total_need,
                "identity": identity_need / total_need,
                "reserve": distribution["reserve"]  # Maintain reserve
            }
            
            # Update distribution
            self.oscillatory_properties["energy_distribution"] = new_distribution
        
        return self.oscillatory_properties["energy_distribution"]
```

### 2. OscillatoryAdaptiveID

```python
class OscillatoryAdaptiveID(AdaptiveID):
    """
    AdaptiveID with oscillatory properties and back pressure mechanism.
    
    Maintains stability through energy regulation and back pressure
    while enabling bidirectional integration with topology and temporality.
    """
    
    def __init__(self, base_concept, creator_id, weight=1.0, confidence=1.0):
        super().__init__(base_concept, creator_id, weight, confidence)
        
        # Oscillatory properties
        self.oscillatory_properties = {
            "frequency": 0.1,  # Natural frequency
            "phase": 0.0,      # Current phase
            "amplitude": 1.0,  # Current amplitude
            "harmonics": []    # Harmonic components
        }
        
        # Energy system
        self.energy_system = {
            "current_energy": 1.0,  # Current energy level
            "capacity": 2.0,        # Maximum energy capacity
            "consumption_rate": 0.01, # Base energy consumption
            "production_rate": 0.02   # Base energy production
        }
        
        # Back pressure system
        self.back_pressure = {
            "current_pressure": 0,    # Current back pressure
            "threshold": 10,          # Pressure threshold
            "recovery_rate": 0.5      # Recovery rate per update
        }
    
    def update(self, change_type, value, origin):
        """
        Update the ID with back pressure control.
        """
        # Check back pressure before proceeding
        if self.back_pressure["current_pressure"] > self.back_pressure["threshold"]:
            # Apply back pressure - defer update
            self._queue_update(change_type, value, origin)
            return False
        
        # Calculate energy required for this change
        energy_required = self._calculate_energy_requirement(change_type, value)
        
        # Check if we have sufficient energy
        if self.energy_system["current_energy"] < energy_required:
            # Not enough energy - scale down change
            value = self._scale_change_to_energy(
                change_type, value, self.energy_system["current_energy"]
            )
            energy_required = self.energy_system["current_energy"]
        
        # Consume energy
        self.energy_system["current_energy"] -= energy_required
        
        # Apply update
        result = super().update(change_type, value, origin)
        
        # Increase back pressure
        self.back_pressure["current_pressure"] += self._calculate_pressure_increase(
            change_type, value
        )
        
        # Generate energy based on activity
        self._generate_energy()
        
        return result
```

### 3. TopologyTemporalityBridge

```python
class TopologyTemporalityBridge:
    """
    Bridge between topology and temporality components.
    
    Enables bidirectional updates between topology constructs and
    temporal patterns, maintaining oscillatory properties.
    """
    
    def __init__(self, topology_manager, temporal_pattern_manager, field_state):
        self.topology_manager = topology_manager
        self.temporal_pattern_manager = temporal_pattern_manager
        self.field_state = field_state
        self.event_bus = None  # Will be set during initialization
    
    def initialize(self, event_bus):
        """Initialize the bridge with event bus for notifications."""
        self.event_bus = event_bus
        
        # Subscribe to relevant events
        self.event_bus.subscribe("topology.state.updated", self._on_topology_updated)
        self.event_bus.subscribe("temporal.pattern.updated", self._on_temporal_updated)
        self.event_bus.subscribe("field.state.updated", self._on_field_updated)
    
    def _on_topology_updated(self, event):
        """Handle topology state updates."""
        topology_state = event.data.get("topology_state")
        if not topology_state:
            return
        
        # Update temporal patterns based on topology changes
        self._update_temporal_from_topology(topology_state)
        
        # Update field state
        self._update_field_state_from_topology(topology_state)
    
    def _on_temporal_updated(self, event):
        """Handle temporal pattern updates."""
        temporal_pattern = event.data.get("temporal_pattern")
        if not temporal_pattern:
            return
        
        # Update topology based on temporal changes
        self._update_topology_from_temporal(temporal_pattern)
        
        # Update field state
        self._update_field_state_from_temporal(temporal_pattern)
```

### 4. TopologyTemporalActantJourney

```python
class TopologyTemporalActantJourney:
    """
    Tracks actant journeys across both topology and temporality.
    
    Maintains coherent identity as actants traverse domains, boundaries,
    and temporal patterns, with oscillatory properties.
    """
    
    def __init__(self, adaptive_id):
        """
        Initialize with an existing AdaptiveID.
        
        Args:
            adaptive_id: The AdaptiveID instance to track
        """
        # Use the existing AdaptiveID rather than creating a new ID
        self.adaptive_id = adaptive_id
        self.journey_points = []
        self.domain_transitions = []
        self.temporal_transitions = []
        self.oscillatory_history = []
        
        # Homeostasis parameters
        self.homeostasis = {
            "stability_threshold": 0.3,  # Minimum stability threshold
            "energy_reserve": 0.2,       # Energy reserve for transitions
            "recovery_rate": 0.1         # Recovery rate after transitions
        }
    
    def add_journey_point(self, point_data):
        """
        Add a journey point with topology and temporality information.
        """
        # Create journey point with timestamp
        journey_point = {
            # Use AdaptiveID's versioning system instead of creating new UUIDs
            "id": self.adaptive_id.create_version("journey_point", point_data, "topology_temporal_journey"),
            "timestamp": datetime.now().isoformat(),
            "data": point_data,
            "oscillatory_properties": point_data.get("oscillatory_properties", {})
        }
        
        # Add to journey points
        self.journey_points.append(journey_point)
        
        # Record oscillatory state
        self._record_oscillatory_state(journey_point)
        
        # Check for domain transition
        if "domain_id" in point_data and self.journey_points:
            previous_point = self.journey_points[-2] if len(self.journey_points) > 1 else None
            if previous_point and previous_point["data"].get("domain_id") != point_data["domain_id"]:
                self._record_domain_transition(previous_point, journey_point)
        
        # Check for temporal transition
        if "temporal_pattern_id" in point_data and self.journey_points:
            previous_point = self.journey_points[-2] if len(self.journey_points) > 1 else None
            if previous_point and previous_point["data"].get("temporal_pattern_id") != point_data["temporal_pattern_id"]:
                self._record_temporal_transition(previous_point, journey_point)
        
        return journey_point
```

### 5. TopologyTemporalActantJourneyRepository

```python
class TopologyTemporalActantJourneyRepository:
    """
    Repository for managing actant journeys across topology and temporality.
    """
    
    def __init__(self, arangodb_client):
        self.db = arangodb_client
        self.collection_name = "ActantJourney"
        self.adaptive_id_repository = AdaptiveIDRepository(arangodb_client)
    
    def find_by_adaptive_id(self, adaptive_id_value):
        """
        Find journey by AdaptiveID.
        
        Args:
            adaptive_id_value: The AdaptiveID.id value
            
        Returns:
            The journey data if found, None otherwise
        """
        query = f"""
        FOR journey IN {self.collection_name}
            FILTER journey.adaptive_id == @adaptive_id
            RETURN journey
        """
        
        bind_vars = {"adaptive_id": adaptive_id_value}
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        result = list(cursor)
        
        return result[0] if result else None
    
    def create_journey(self, adaptive_id):
        """
        Create a new journey for an AdaptiveID.
        
        Args:
            adaptive_id: The AdaptiveID instance
            
        Returns:
            The created journey document
        """
        # Ensure the AdaptiveID exists in the repository
        self.adaptive_id_repository.ensure_exists(adaptive_id)
        
        # Create journey document
        journey_doc = {
            "_key": adaptive_id.id,  # Use AdaptiveID.id as the document key
            "adaptive_id": adaptive_id.id,
            "base_concept": adaptive_id.base_concept,
            "created_at": datetime.now().isoformat(),
            "journey_points": [],
            "domain_transitions": [],
            "temporal_transitions": [],
            "oscillatory_history": []
        }
        
        # Insert document
        self.db.collection(self.collection_name).insert(journey_doc)
        
        return journey_doc
```

## Test-Driven Implementation

### Test Suite 1: OscillatoryFieldState Tests

```python
import unittest
from datetime import datetime
from src.habitat_evolution.field.field_state import OscillatoryFieldState

class TestOscillatoryFieldState(unittest.TestCase):
    
    def setUp(self):
        # Create mock field analysis
        self.field_analysis = {
            "topology": {
                "effective_dimensionality": 3,
                "principal_dimensions": [0, 1, 2],
                "eigenvalues": [0.5, 0.3, 0.2],
                "eigenvectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            },
            "metrics": {
                "coherence": 0.7,
                "turbulence": 0.2,
                "stability": 0.8
            },
            "density": {
                "density_centers": [[0.1, 0.2], [0.3, 0.4]],
                "density_map": {"0,0": 0.1, "0,1": 0.2}
            },
            "field_properties": {
                "coherence": 0.7,
                "navigability_score": 0.6,
                "stability": 0.8
            },
            "patterns": {}
        }
        
        # Create field state
        self.field_state = OscillatoryFieldState(self.field_analysis)
    
    def test_energy_distribution(self):
        """Test energy distribution across dimensions."""
        # Initial distribution should be equal
        self.assertEqual(self.field_state.oscillatory_properties["energy_distribution"]["topology"], 0.33)
        self.assertEqual(self.field_state.oscillatory_properties["energy_distribution"]["temporality"], 0.33)
        self.assertEqual(self.field_state.oscillatory_properties["energy_distribution"]["identity"], 0.33)
        
        # Simulate event energy
        event_energy = 1.0
        
        # Mock energy needs
        self.field_state._calculate_topology_energy_need = lambda: 0.5
        self.field_state._calculate_temporality_energy_need = lambda: 0.3
        self.field_state._calculate_identity_energy_need = lambda: 0.2
        
        # Distribute energy
        distribution = self.field_state.distribute_energy(event_energy)
        
        # Check distribution
        self.assertEqual(distribution["topology"], 0.5)
        self.assertEqual(distribution["temporality"], 0.3)
        self.assertEqual(distribution["identity"], 0.2)
    
    def test_coherence_regulation(self):
        """Test coherence regulation mechanisms."""
        # Mock coherence metrics
        self.field_state.get_coherence_metric = lambda: 0.2  # Below threshold
        
        # Mock coherence increase method
        increase_called = [False]
        def mock_increase():
            increase_called[0] = True
        self.field_state._increase_coherence = mock_increase
        
        # Test regulation with low coherence
        self.field_state.regulate_coherence()
        self.assertTrue(increase_called[0])
        
        # Reset and test with high coherence
        increase_called[0] = False
        self.field_state.get_coherence_metric = lambda: 0.9  # Above threshold
        
        # Mock rigidity reduction method
        reduce_called = [False]
        def mock_reduce():
            reduce_called[0] = True
        self.field_state._reduce_rigidity = mock_reduce
        
        # Test regulation with high coherence
        self.field_state.regulate_coherence()
        self.assertTrue(reduce_called[0])
```

### Test Suite 2: OscillatoryAdaptiveID Tests

```python
import unittest
from datetime import datetime
from src.habitat_evolution.adaptive_core.id.oscillatory_adaptive_id import OscillatoryAdaptiveID

class TestOscillatoryAdaptiveID(unittest.TestCase):
    
    def setUp(self):
        # Create adaptive ID
        self.adaptive_id = OscillatoryAdaptiveID(
            base_concept="test_concept",
            creator_id="test_creator",
            weight=1.0,
            confidence=0.8
        )
    
    def test_back_pressure(self):
        """Test back pressure mechanism."""
        # Set high back pressure
        self.adaptive_id.back_pressure["current_pressure"] = 20  # Above threshold
        
        # Mock queue update method
        queue_called = [False]
        def mock_queue(change_type, value, origin):
            queue_called[0] = True
            return False
        self.adaptive_id._queue_update = mock_queue
        
        # Try to update
        result = self.adaptive_id.update("test_change", "test_value", "test_origin")
        
        # Check that update was queued
        self.assertTrue(queue_called[0])
        self.assertFalse(result)
    
    def test_energy_regulation(self):
        """Test energy regulation during updates."""
        # Set low energy
        self.adaptive_id.energy_system["current_energy"] = 0.1
        
        # Mock energy calculation
        self.adaptive_id._calculate_energy_requirement = lambda change_type, value: 0.2
        
        # Mock scale method
        scaled_value = "scaled_test_value"
        def mock_scale(change_type, value, available_energy):
            return scaled_value
        self.adaptive_id._scale_change_to_energy = mock_scale
        
        # Mock super update
        super_called = [False]
        super_value = [None]
        def mock_super_update(change_type, value, origin):
            super_called[0] = True
            super_value[0] = value
            return True
        self.adaptive_id.__class__.__bases__[0].update = mock_super_update
        
        # Try to update
        result = self.adaptive_id.update("test_change", "test_value", "test_origin")
        
        # Check that update was scaled
        self.assertTrue(super_called[0])
        self.assertEqual(super_value[0], scaled_value)
        self.assertTrue(result)
        
        # Check energy was consumed
        self.assertEqual(self.adaptive_id.energy_system["current_energy"], 0)
```

### Test Suite 3: TopologyTemporalActantJourney Tests

```python
import unittest
from datetime import datetime
from unittest.mock import MagicMock
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.pattern_aware_rag.topology.actant_journey import TopologyTemporalActantJourney

class TestTopologyTemporalActantJourney(unittest.TestCase):
    
    def setUp(self):
        # Create mock AdaptiveID
        self.adaptive_id = MagicMock(spec=AdaptiveID)
        self.adaptive_id.id = "test_id"
        self.adaptive_id.base_concept = "test_concept"
        self.adaptive_id.create_version.return_value = "test_version_id"
        
        # Create journey
        self.journey = TopologyTemporalActantJourney(self.adaptive_id)
    
    def test_add_journey_point(self):
        """Test adding journey points."""
        # Add journey point
        point_data = {
            "domain_id": "domain1",
            "temporal_pattern_id": "pattern1",
            "position": [0.1, 0.2, 0.3],
            "oscillatory_properties": {
                "frequency": 0.1,
                "phase": 0.2,
                "amplitude": 0.8
            }
        }
        
        journey_point = self.journey.add_journey_point(point_data)
        
        # Check journey point was added
        self.assertEqual(len(self.journey.journey_points), 1)
        self.assertEqual(journey_point["id"], "test_version_id")
        self.assertEqual(journey_point["data"], point_data)
        
        # Check AdaptiveID version was created
        self.adaptive_id.create_version.assert_called_once()
    
    def test_domain_transition(self):
        """Test domain transition detection."""
        # Add first journey point
        point1 = {
            "domain_id": "domain1",
            "temporal_pattern_id": "pattern1"
        }
        self.journey.add_journey_point(point1)
        
        # Mock record method
        record_called = [False]
        def mock_record(from_point, to_point):
            record_called[0] = True
            self.assertEqual(from_point["data"]["domain_id"], "domain1")
            self.assertEqual(to_point["data"]["domain_id"], "domain2")
        self.journey._record_domain_transition = mock_record
        
        # Add second journey point with different domain
        point2 = {
            "domain_id": "domain2",
            "temporal_pattern_id": "pattern1"
        }
        self.journey.add_journey_point(point2)
        
        # Check transition was recorded
        self.assertTrue(record_called[0])
    
    def test_temporal_transition(self):
        """Test temporal transition detection."""
        # Add first journey point
        point1 = {
            "domain_id": "domain1",
            "temporal_pattern_id": "pattern1"
        }
        self.journey.add_journey_point(point1)
        
        # Mock record method
        record_called = [False]
        def mock_record(from_point, to_point):
            record_called[0] = True
            self.assertEqual(from_point["data"]["temporal_pattern_id"], "pattern1")
            self.assertEqual(to_point["data"]["temporal_pattern_id"], "pattern2")
        self.journey._record_temporal_transition = mock_record
        
        # Add second journey point with different pattern
        point2 = {
            "domain_id": "domain1",
            "temporal_pattern_id": "pattern2"
        }
        self.journey.add_journey_point(point2)
        
        # Check transition was recorded
        self.assertTrue(record_called[0])
```

## Conclusion

This implementation plan provides a comprehensive approach to integrating topology and temporality in the Habitat Evolution system. By following a field-first philosophy and leveraging the existing AdaptiveID system, we ensure coherent identity across both dimensions while maintaining system stability through homeostasis mechanisms and back pressure.

The test-driven implementation approach ensures that each component functions correctly and integrates seamlessly with the rest of the system. By breaking the implementation into clear phases with defined tasks and test cases, we can systematically build and validate the integration while maintaining the system's overall coherence and stability.
