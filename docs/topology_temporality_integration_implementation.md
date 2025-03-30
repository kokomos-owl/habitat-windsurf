# Topology-Temporality Integration Implementation

## Overview

This document outlines the comprehensive implementation plan for integrating topology and temporality in the Habitat Evolution system. This integration represents a critical advancement in our pattern evolution and co-evolution capabilities, enabling the system to detect and evolve coherent patterns while observing semantic change across both spatial (topological) and temporal dimensions.

The integration follows a field-first philosophy, treating both dimensions as emergent properties of a unified semantic field while leveraging the existing AdaptiveID system as the central reference point. This approach ensures that we maintain coherent identity across both dimensions without introducing parallel identification structures or disruptive architectural changes.

By implementing bidirectional relationships between topology constructs (frequency domains, boundaries, resonance points) and temporal patterns (learning windows, pattern evolution trajectories), we enable seamless tracking of actant journeys through the semantic landscape. This integration also addresses the current challenges of mixed database usage (Neo4j and ArangoDB) by proposing a unified persistence layer that maintains oscillatory properties across all relationships.

## Core Principles

1. **Field-First Integration**: Topology and temporality are emergent properties of the same underlying field, not separate systems that need to be connected. The field itself is primary, with topology (spatial relationships) and temporality (pattern evolution over time) emerging naturally from field dynamics. This principle ensures that integration isn't merely connecting separate systems but recognizing their inherent unity within the semantic field.

2. **Oscillatory Foundations**: All relationships are fundamentally oscillatory, with natural frequencies, amplitudes, and phases that respond to system conditions. These oscillatory properties enable natural resonance between topology and temporality, creating coherent patterns that evolve across both dimensions. The oscillatory nature allows for:
   - Natural synchronization between topology and temporal patterns
   - Harmonic relationships that maintain coherence during evolution
   - Wave-like propagation of changes across the field
   - Interference patterns that create emergent meta-patterns

3. **Adaptive Homeostasis**: The system maintains stability while evolving through feedback mechanisms that regulate energy flow and pattern formation. Like biological systems, Habitat Evolution implements:
   - Energy distribution based on system needs
   - Coherence regulation to prevent excessive disorder or rigidity
   - Back pressure mechanisms to prevent system shocks
   - Recovery mechanisms after perturbations
   - Energy reserves for maintaining stability during transitions

4. **Bidirectional Coherence**: Changes propagate bidirectionally across dimensions, maintaining coherence without forcing synchronization. This principle ensures that:
   - Topology changes influence temporal pattern evolution
   - Temporal pattern evolution shapes topology
   - Changes flow naturally along field gradients
   - Local changes can have global effects through resonance
   - The system maintains overall coherence despite local perturbations

5. **Gradient-Based Evolution**: Evolution follows natural gradients in the field, not imposed structural changes. The system evolves along:
   - Coherence gradients (toward greater pattern coherence)
   - Energy gradients (toward optimal energy distribution)
   - Information density gradients (toward meaningful pattern formation)
   - This ensures organic evolution without artificial boundaries or disruptive changes

6. **AdaptiveID Centrality**: AdaptiveID serves as the central reference point across both dimensions, maintaining coherent identity throughout the system. This principle ensures that:
   - We avoid creating parallel identification structures
   - Identity remains coherent across topology and temporality
   - Actant journeys can be tracked seamlessly across dimensions
   - Versioning and context tracking span both dimensions
   - Relationships maintain oscillatory properties across dimensions

## Implementation Phases

### Phase 1: Core Components Enhancement

#### Definition

Extend existing core components with oscillatory properties and bidirectional awareness. This phase focuses on enhancing the fundamental building blocks of the system to support oscillatory behavior and adaptive homeostasis, laying the groundwork for bidirectional integration between topology and temporality.

#### Tasks

1. **Extend TonicHarmonicFieldState**
   - Add oscillatory properties to field state:
     - Implement resonance frequency tracking across the field
     - Add phase coherence metrics between topology and temporality
     - Implement amplitude modulation based on field activity
     - Create harmonic component tracking for complex oscillatory patterns
   - Implement energy distribution system:
     - Create energy allocation mechanisms across dimensions
     - Implement energy flow tracking along field gradients
     - Add energy need calculation based on activity levels
     - Create energy reserve management for stability
   - Add homeostasis mechanisms:
     - Implement coherence regulation to maintain field stability
     - Add rigidity detection and reduction mechanisms
     - Create recovery mechanisms after perturbations
     - Implement adaptive thresholds based on field conditions

2. **Enhance AdaptiveID**
   - Add oscillatory properties:
     - Implement natural frequency calculation based on concept properties
     - Add phase tracking relative to field state
     - Create amplitude modulation based on confidence and weight
     - Implement harmonic components for complex concept relationships
   - Implement back pressure mechanism:
     - Create pressure monitoring system
     - Implement update queuing under high pressure
     - Add pressure recovery based on time and system conditions
     - Create adaptive pressure thresholds based on importance
   - Add energy regulation:
     - Implement energy consumption for state changes
     - Create energy production based on activity and relevance
     - Add energy capacity management
     - Implement energy-based scaling of changes

3. **Create Oscillatory Relationship Base Class**
   - Define common oscillatory properties for relationships:
     - Implement frequency matching between related entities
     - Add phase difference tracking and adjustment
     - Create amplitude correlation based on relationship strength
     - Implement harmonic resonance detection between entities
   - Implement phase coherence calculations:
     - Create phase synchronization mechanisms
     - Add phase lock detection for strong relationships
     - Implement phase drift monitoring for weakening relationships
     - Create phase reset mechanisms for relationship changes
   - Support bidirectional propagation:
     - Implement wave-like propagation of changes
     - Add interference pattern detection between waves
     - Create resonance amplification for important changes
     - Implement damping for potentially destabilizing changes

#### Test Cases

1. **Field State Oscillation Tests**
   - Test energy distribution across dimensions:
     - Verify energy allocation based on system needs
     - Test energy flow along field gradients
     - Validate energy reserve management under stress
     - Verify adaptive energy distribution during topology-temporal transitions
   - Verify homeostasis mechanisms maintain stability:
     - Test coherence regulation under various perturbations
     - Verify recovery after significant system changes
     - Test adaptation to changing field conditions
     - Validate stability maintenance during rapid evolution
   - Validate coherence regulation:
     - Test detection of excessive disorder
     - Verify response to excessive rigidity
     - Validate coherence maintenance across dimensions
     - Test adaptive threshold adjustment

2. **AdaptiveID Oscillation Tests**
   - Test oscillatory property updates:
     - Verify frequency adaptation based on context
     - Test phase alignment with related concepts
     - Validate amplitude modulation based on confidence
     - Test harmonic component evolution
   - Verify back pressure prevents system shocks:
     - Test update queuing under high pressure
     - Verify pressure recovery over time
     - Validate adaptive pressure thresholds
     - Test system stability under sustained pressure
   - Validate energy regulation maintains stability:
     - Test energy consumption during updates
     - Verify energy production mechanisms
     - Validate energy-based scaling of changes
     - Test stability under energy constraints

3. **Relationship Oscillation Tests**
   - Test bidirectional propagation of changes:
     - Verify wave-like propagation across the field
     - Test interference pattern formation
     - Validate resonance amplification for important changes
     - Test damping of potentially destabilizing changes
   - Verify phase coherence calculations:
     - Test phase synchronization between related entities
     - Verify phase lock detection for strong relationships
     - Validate phase drift monitoring
     - Test phase reset mechanisms
   - Validate relationship stability under load:
     - Test relationship maintenance during rapid changes
     - Verify stability under conflicting updates
     - Validate graceful degradation under extreme conditions
     - Test relationship recovery after system stress

### Phase 2: Topology-Temporality Bridge

#### Definition

Implement bidirectional bridge between topology and temporality components. This phase creates the core integration layer that enables seamless communication and synchronization between topological constructs and temporal patterns, ensuring coherent evolution across both dimensions.

#### Tasks

1. **Create TopologyTemporalityBridge**
   - Implement bidirectional update mechanisms:
     - Create topology-to-temporality update pipeline
     - Implement temporality-to-topology update pipeline
     - Add conflict resolution for bidirectional updates
     - Create transaction-like update batching for consistency
   - Add oscillatory property propagation:
     - Implement frequency synchronization across dimensions
     - Create phase coherence maintenance mechanisms
     - Add amplitude modulation based on cross-dimensional relevance
     - Implement harmonic relationship detection and enhancement
   - Support event-based communication:
     - Create event subscription system for topology and temporality events
     - Implement event filtering based on relevance and priority
     - Add event aggregation for related changes
     - Create event replay capabilities for system recovery

2. **Enhance Topology Manager**
   - Add temporal awareness to topology constructs:
     - Implement temporal versioning for topology states
     - Create temporal evolution tracking for frequency domains
     - Add temporal context to boundary transitions
     - Implement temporal pattern recognition within topology
   - Implement oscillatory frequency domains:
     - Create frequency-based domain classification
     - Add phase coherence metrics within domains
     - Implement resonance detection between domains
     - Create harmonic relationship tracking across domains
   - Support bidirectional updates:
     - Implement update receivers for temporal pattern changes
     - Create topology state evolution based on temporal patterns
     - Add feedback mechanisms for topology changes
     - Implement coherence maintenance during updates

3. **Enhance Temporal Pattern Manager**
   - Add topology awareness to temporal patterns:
     - Implement spatial context for temporal patterns
     - Create topology-based pattern segmentation
     - Add topology boundary awareness to pattern evolution
     - Implement topology-influenced pattern prediction
   - Implement oscillatory pattern evolution:
     - Create frequency-based pattern classification
     - Add phase tracking for pattern evolution stages
     - Implement amplitude modulation based on pattern strength
     - Create harmonic pattern detection and enhancement
   - Support bidirectional updates:
     - Implement update receivers for topology changes
     - Create pattern evolution trajectories based on topology
     - Add feedback mechanisms for pattern changes
     - Implement coherence maintenance during updates

#### Test Cases

1. **Bridge Communication Tests**
   - Test bidirectional updates between dimensions:
     - Verify topology changes propagate to temporal patterns
     - Test temporal pattern changes affect topology
     - Validate update consistency across dimensions
     - Test conflict resolution for simultaneous updates
   - Verify oscillatory property propagation:
     - Test frequency synchronization across dimensions
     - Verify phase coherence maintenance
     - Validate amplitude modulation based on relevance
     - Test harmonic relationship enhancement
   - Validate event handling:
     - Test event subscription and delivery
     - Verify event filtering and prioritization
     - Test event aggregation for related changes
     - Validate event replay during recovery

2. **Topology-Temporal Coherence Tests**
   - Test coherence between dimensions under various loads:
     - Verify coherence maintenance during high update volumes
     - Test cross-dimensional consistency under stress
     - Validate adaptive coherence thresholds
     - Test graceful degradation under extreme conditions
   - Verify stability during rapid changes:
     - Test system response to rapid topology changes
     - Verify pattern stability during temporal evolution
     - Validate boundary stability during transitions
     - Test recovery after sudden perturbations
   - Validate pattern emergence across dimensions:
     - Test meta-pattern detection spanning dimensions
     - Verify cross-dimensional pattern reinforcement
     - Validate emergent property detection
     - Test pattern prediction accuracy

3. **Oscillatory Synchronization Tests**
   - Test phase synchronization between dimensions:
     - Verify phase alignment between related constructs
     - Test phase lock detection for strong relationships
     - Validate phase drift monitoring for weakening relationships
     - Test phase reset mechanisms during major changes
   - Verify frequency entrainment:
     - Test natural frequency alignment between dimensions
     - Verify frequency adaptation based on interaction strength
     - Validate frequency band emergence for related constructs
     - Test resonance detection and amplification
   - Validate harmonic relationships:
     - Test harmonic pattern detection across dimensions
     - Verify harmonic reinforcement mechanisms
     - Validate interference pattern analysis
     - Test harmonic stability under perturbations

### Phase 3: Actant Journey Integration

#### Definition

Implement actant journey tracking across topology and temporality, focusing on the essential movement patterns of actants (human, system, or intelligent agents) while maintaining coherent identity.

#### Tasks

1. **Create TopologyTemporalActantJourney**
   - Leverage AdaptiveID for identity:
     - Use existing AdaptiveID as the central reference point
     - Add minimal versioning for cross-dimensional tracking
     - Implement essential context attributes for both dimensions
   - Implement journey point tracking:
     - Create lightweight journey points with position references
     - Build simple path reconstruction across dimensions
     - Focus on critical transition points rather than complete paths
   - Support basic oscillatory properties:
     - Track frequency and phase changes at domain boundaries
     - Implement simple resonance detection for related actants

2. **Enhance ActantJourneyRepository**
   - Add topology-temporality awareness:
     - Create unified storage schema for cross-dimensional journeys
     - Implement essential indexing for efficient retrieval
     - Add basic query capabilities for common access patterns
   - Implement streamlined oscillatory tracking:
     - Store only essential oscillatory properties (frequency, phase)
     - Track significant phase shifts at transition points
   - Support minimal back pressure:
     - Implement simple load monitoring
     - Add basic queuing for high-volume periods

3. **Create ActantJourneyAnalyzer**
   - Implement essential pattern detection:
     - Focus on identifying common journey patterns
     - Detect significant deviations from expected paths
   - Add focused stability analysis:
     - Track journey continuity across dimensional boundaries
     - Identify unstable transition points
   - Support targeted feedback:
     - Provide key journey insights to field state
     - Identify high-traffic pathways for optimization

#### Test Cases

1. **Actant Journey Tracking Tests**
   - Test essential journey recording:
     - Verify basic topology position recording
     - Test temporal context preservation
     - Validate cross-dimensional linking
   - Verify key transition detection:
     - Test domain boundary crossing detection
     - Verify pattern transition recognition
   - Validate basic oscillatory tracking:
     - Test frequency changes at transition points
     - Verify phase relationship maintenance

2. **Journey Stability Tests**
   - Test core stability requirements:
     - Verify identity coherence during transitions
     - Test basic journey continuity during perturbations
     - Validate journey reconstruction with incomplete data
   - Verify essential coherence maintenance:
     - Test basic stability during domain transitions
     - Verify recovery after simple perturbations
   - Validate minimal back pressure functionality:
     - Test behavior under moderate load
     - Verify basic recovery after interruptions

3. **Cross-Dimensional Pattern Tests**
   - Test essential pattern recognition:
     - Verify detection of common cross-dimensional patterns
     - Test basic anomaly detection
   - Verify targeted feedback mechanisms:
     - Test key updates to field state
     - Verify basic gradient influence
   - Validate primary pattern emergence:
     - Test detection of fundamental repeated patterns
     - Verify stability of detected patterns

### Phase 4: Field State Integration

#### Definition

Integrate field state with topology and temporality for coherent field evolution, focusing on essential connections that enable natural system evolution while maintaining stability.

#### Tasks

1. **Enhance Field State Manager**
   - Add topology-temporality awareness:
     - Implement basic unified field representation
     - Create simple dimensional projections for analysis
     - Add core coherence metrics between dimensions
   - Implement essential oscillatory metrics:
     - Track primary resonance frequencies
     - Monitor phase relationships between key regions
   - Support basic gradient-based evolution:
     - Implement simplified coherence gradients
     - Create fundamental energy flow paths

2. **Create Field Gradient System**
   - Implement essential gradients:
     - Create basic coherence gradients between dimensions
     - Implement simplified semantic density tracking
   - Add streamlined energy flow:
     - Implement core energy distribution mechanisms
     - Track energy flow along primary pathways
   - Support minimal coherence regulation:
     - Create basic coherence thresholds
     - Implement simple enhancement for weak connections

3. **Implement Field Observer System**
   - Create focused observers:
     - Implement key change detection between dimensions
     - Monitor critical boundary transitions
   - Implement essential feedback:
     - Create primary feedback channels between components
     - Implement basic journey feedback mechanisms
   - Support core homeostasis:
     - Implement simple energy regulation
     - Create basic stability mechanisms

#### Test Cases

1. **Field Evolution Tests**
   - Test essential field evolution:
     - Verify basic adaptation to topology changes
     - Test response to key temporal pattern shifts
     - Validate fundamental coherence maintenance
   - Verify core gradient functionality:
     - Test basic flow along primary gradients
     - Verify energy distribution patterns
   - Validate basic stability:
     - Test field response to simple perturbations
     - Verify recovery after disruptions

2. **Coherence Regulation Tests**
   - Test basic regulation mechanisms:
     - Verify enhancement in key weak regions
     - Test simple stability mechanisms
   - Verify essential stability:
     - Test response to common disruptions
     - Verify recovery patterns
   - Validate core homeostasis:
     - Test basic energy regulation
     - Verify fundamental stability maintenance

3. **Observer Feedback Tests**
   - Test core notifications:
     - Verify detection of significant changes
     - Test accuracy of critical notifications
   - Verify essential feedback paths:
     - Test primary feedback channels
     - Verify basic journey feedback
   - Validate fundamental responses:
     - Test key field adjustments from observations
     - Verify basic learning from repeated patterns

## Implementation Classes

### 1. OscillatoryFieldState

```python
class OscillatoryFieldState(TonicHarmonicFieldState):
    """
    Extends TonicHarmonicFieldState with essential oscillatory properties
    that integrate topology and temporality within a unified field.
    """
    
    def __init__(self, field_analysis):
        super().__init__(field_analysis)
        
        # Core oscillatory properties - minimal set needed for functionality
        self.oscillatory_properties = {
            "resonance_frequency": 0.1,  # Base resonance frequency
            "phase_coherence": 0.5,     # Phase coherence across the field
            "energy_distribution": {     # Simplified energy distribution
                "topology": 0.33,
                "temporality": 0.33,
                "identity": 0.33,
                "reserve": 0.01
            }
        }
        
        # Streamlined homeostasis parameters
        self.homeostasis = {
            "coherence_threshold": 0.3,  # Minimum coherence threshold
            "rigidity_threshold": 0.8,   # Maximum rigidity threshold
            "energy_reserve": 0.1        # Energy reserve for stability
        }
        
        # Essential cross-dimensional metrics
        self.cross_dimensional_metrics = {
            "topology_temporal_coherence": 0.5  # Core coherence metric
        }
        
        # Minimal observer system
        self.observers = []
    
    def register_observer(self, observer):
        """Register an observer for essential field state changes."""
        if observer not in self.observers:
            self.observers.append(observer)
    
    def notify_observers(self, context):
        """Notify observers with minimal context information."""
        for observer in self.observers:
            if hasattr(observer, "observe"):
                observer.observe(context)
    
    def regulate_coherence(self):
        """
        Apply basic regulation to maintain field coherence within bounds.
        """
        current_coherence = self.get_coherence_metric()
        
        # Simplified coherence regulation
        if current_coherence < self.homeostasis["coherence_threshold"]:
            # Apply coherence-increasing feedback
            self._increase_coherence(current_coherence)
            
            # Minimal notification
            self.notify_observers({
                "type": "coherence_regulation",
                "action": "increase"
            })
        elif current_coherence > self.homeostasis["rigidity_threshold"]:
            # Apply rigidity-reducing feedback
            self._reduce_rigidity(current_coherence)
            
            # Minimal notification
            self.notify_observers({
                "type": "coherence_regulation",
                "action": "reduce_rigidity"
            })
        
        return self.get_coherence_metric()
    
    def _increase_coherence(self, current_coherence):
        """
        Apply targeted feedback to increase field coherence.
        """
        # Calculate coherence deficit
        coherence_deficit = self.homeostasis["coherence_threshold"] - current_coherence
        
        # Focus only on most important relationships
        for pattern_id, pattern in self.patterns.items():
            if "relationships" in pattern:
                # Identify and strengthen only high-potential relationships
                for rel_id, relationship in pattern["relationships"].items():
                    if relationship.get("potential", 0) > 0.5:
                        relationship["strength"] = min(1.0, relationship.get("strength", 0) + 0.1)
        
        # Adjust field coherence directly
        self.field_properties["coherence"] = min(1.0, 
                                              self.field_properties.get("coherence", 0) + 0.1)
    
    def _reduce_rigidity(self, current_coherence):
        """
        Apply simple feedback to reduce field rigidity.
        """
        # Identify most rigid patterns
        for pattern_id, pattern in self.patterns.items():
            if pattern.get("stability", 0) > 0.8:
                # Apply fixed perturbation to rigid patterns
                pattern["stability"] = max(0.5, pattern.get("stability", 0) - 0.1)
                pattern["adaptability"] = min(1.0, pattern.get("adaptability", 0) + 0.1)
        
        # Slightly reduce field coherence
        self.field_properties["coherence"] = max(0.0, 
                                             self.field_properties.get("coherence", 0) - 0.1)
    
    # Removed _adapt_coherence_thresholds method to simplify implementation
    
    def _calculate_field_activity(self):
        """
        Calculate overall field activity level.
        
        Returns:
            Activity level between 0.0 and 1.0
        """
        # Count recent pattern changes
        pattern_change_count = sum(1 for pattern in self.patterns.values() 
                                 if "last_modified" in pattern and 
                                 (datetime.now() - datetime.fromisoformat(pattern["last_modified"])).total_seconds() < 3600)
        
        # Count recent perturbations
        perturbation_count = len(self.homeostasis["perturbation_memory"])
        
        # Calculate normalized activity level
        max_expected_changes = max(len(self.patterns), 1) * 0.5
        activity_level = (pattern_change_count + perturbation_count) / max_expected_changes
        
        return min(1.0, activity_level)
    
    def distribute_energy(self, event_energy):
        """
        Distribute energy across the field based on basic needs.
        
        Args:
            event_energy: Energy introduced by an event
        """
        # Get current distribution
        distribution = self.oscillatory_properties["energy_distribution"]
        
        # Simple fixed distribution based on current field state
        # Allocate more energy to the dimension with lowest coherence
        topo_coherence = self.get_topology_coherence()
        temp_coherence = self.get_temporality_coherence()
        
        if topo_coherence < temp_coherence:
            new_distribution = {
                "topology": 0.5,
                "temporality": 0.3,
                "identity": 0.15,
                "reserve": 0.05
            }
        else:
            new_distribution = {
                "topology": 0.3,
                "temporality": 0.5,
                "identity": 0.15,
                "reserve": 0.05
            }
        
        # Update distribution
        self.oscillatory_properties["energy_distribution"] = new_distribution
        
        return new_distribution
    
    def _calculate_topology_energy_need(self):
        """
        Calculate energy need for topology based on current activity.
        
        Returns:
            Normalized energy need (0.0-1.0)
        """
        # Base need starts at moderate level
        base_need = 0.3
        
        # Increase need based on boundary activity
        boundary_activity = self._get_boundary_activity()
        
        # Increase need based on domain formation activity
        domain_formation = self._get_domain_formation_activity()
        
        # Combine factors with appropriate weights
        topology_need = base_need + (boundary_activity * 0.4) + (domain_formation * 0.3)
        
        # Normalize to 0.0-1.0 range
        return min(1.0, topology_need)
    
    def _calculate_temporality_energy_need(self):
        """
        Calculate energy need for temporality based on current activity.
        
        Returns:
            Normalized energy need (0.0-1.0)
        """
        # Base need starts at moderate level
        base_need = 0.3
        
        # Increase need based on pattern evolution activity
        pattern_evolution = self._get_pattern_evolution_activity()
        
        # Increase need based on learning window activity
        learning_window_activity = self._get_learning_window_activity()
        
        # Combine factors with appropriate weights
        temporality_need = base_need + (pattern_evolution * 0.4) + (learning_window_activity * 0.3)
        
        # Normalize to 0.0-1.0 range
        return min(1.0, temporality_need)
    
    def _calculate_identity_energy_need(self):
        """
        Calculate energy need for identity based on current activity.
        
        Returns:
            Normalized energy need (0.0-1.0)
        """
        # Base need starts at moderate level
        base_need = 0.3
        
        # Increase need based on actant activity
        actant_activity = self._get_actant_activity()
        
        # Increase need based on relationship formation
        relationship_formation = self._get_relationship_formation_activity()
        
        # Combine factors with appropriate weights
        identity_need = base_need + (actant_activity * 0.4) + (relationship_formation * 0.3)
        
        # Normalize to 0.0-1.0 range
        return min(1.0, identity_need)
    
    def _calculate_reserve_energy_need(self):
        """
        Calculate energy need for reserves based on system stability.
        
        Returns:
            Normalized energy need (0.0-1.0)
        """
        # Base need is proportional to instability
        stability = self.field_properties.get("stability", 0.5)
        base_need = 0.1 + ((1.0 - stability) * 0.2)
        
        # Increase need based on recent perturbations
        recent_perturbations = len(self.homeostasis["perturbation_memory"])
        perturbation_factor = min(0.5, recent_perturbations * 0.05)
        
        # Combine factors
        reserve_need = base_need + perturbation_factor
        
        # Normalize to 0.0-1.0 range
        return min(1.0, reserve_need)
    
    def update_gradients(self):
        """
        Update essential field gradients.
        """
        # Update only the most critical gradient - coherence
        self._update_coherence_gradients()
        
        return self.gradients
        
    def _update_coherence_gradients(self):
        """Calculate basic coherence gradients between key field regions."""
        # Simplified implementation focusing only on core regions
        pass
        
    def get_topology_coherence(self):
        """Get topology dimension coherence level."""
        return self.cross_dimensional_metrics.get("topology_coherence", 0.5)
        
    def get_temporality_coherence(self):
        """Get temporality dimension coherence level."""
        return self.cross_dimensional_metrics.get("temporality_coherence", 0.5)
    
    def _update_coherence_gradients(self):
        """
        Update coherence gradients across the field.
        """
        # Implementation calculates coherence gradients between regions
        pass
    
    def _update_energy_gradients(self):
        """
        Update energy gradients across the field.
        """
        # Implementation calculates energy potential gradients
        pass
    
    def _update_information_gradients(self):
        """
        Update information density gradients across the field.
        """
        # Implementation calculates information density gradients
        pass
    
    def _update_stability_gradients(self):
        """
        Update stability gradients across the field.
        """
        # Implementation calculates stability gradients
        pass
    
    def get_flow_direction(self, location):
        """
        Get the natural flow direction at a field location.
        
        Args:
            location: Location in the field
            
        Returns:
            Flow direction vector
        """
        # Get gradients at location
        coherence_gradient = self._get_gradient_at_location("coherence", location)
        energy_gradient = self._get_gradient_at_location("energy", location)
        info_gradient = self._get_gradient_at_location("information", location)
        stability_gradient = self._get_gradient_at_location("stability", location)
        
        # Combine gradients with appropriate weights
        # Weight depends on current energy distribution
        weights = {
            "coherence": 0.4,
            "energy": 0.3,
            "information": 0.2,
            "stability": 0.1
        }
        
        # Calculate combined gradient
        combined_gradient = {
            "x": (coherence_gradient["x"] * weights["coherence"] +
                 energy_gradient["x"] * weights["energy"] +
                 info_gradient["x"] * weights["information"] +
                 stability_gradient["x"] * weights["stability"]),
            "y": (coherence_gradient["y"] * weights["coherence"] +
                 energy_gradient["y"] * weights["energy"] +
                 info_gradient["y"] * weights["information"] +
                 stability_gradient["y"] * weights["stability"])
        }
        
        return combined_gradient
    
    def _get_gradient_at_location(self, gradient_type, location):
        """
        Get gradient value at a specific location.
        
        Args:
            gradient_type: Type of gradient ("coherence", "energy", etc.)
            location: Location in the field
            
        Returns:
            Gradient vector at location
        """
        # Default gradient (no flow)
        default_gradient = {"x": 0, "y": 0}
        
        # Get gradient map
        gradient_map = self.gradients.get(gradient_type, {})
        
        # Find closest location in gradient map
        # In a real implementation, this would use spatial indexing
        closest_location = None
        min_distance = float('inf')
        
        for loc in gradient_map:
            # Parse location string "x,y"
            loc_parts = loc.split(",")
            loc_x = float(loc_parts[0])
            loc_y = float(loc_parts[1])
            
            # Calculate distance
            distance = ((loc_x - location["x"]) ** 2 + (loc_y - location["y"]) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_location = loc
        
        # Return gradient at closest location, or default if none found
        return gradient_map.get(closest_location, default_gradient) if closest_location else default_gradient
```

### 2. OscillatoryAdaptiveID

```python
class OscillatoryAdaptiveID(AdaptiveID):
    """
    AdaptiveID with essential oscillatory properties and minimal back pressure.
    
    Enables coherent identity across topology and temporality dimensions.
    """
    
    def __init__(self, base_concept, creator_id, weight=1.0, confidence=1.0):
        super().__init__(base_concept, creator_id, weight, confidence)
        
        # Core oscillatory properties - only what's needed
        self.oscillatory_properties = {
            "frequency": 0.1,  # Natural frequency
            "phase": 0.0      # Current phase
        }
        
        # Simplified energy system
        self.energy_system = {
            "current_energy": 1.0,  # Current energy level
            "capacity": 2.0        # Maximum energy capacity
        }
        
        # Minimal back pressure
        self.back_pressure = {
            "current_pressure": 0,  # Current back pressure
            "threshold": 10        # Pressure threshold
        }
    
    def reduce_pressure(self):
        """
        Reduce back pressure periodically.
        """
        # Simple pressure reduction
        self.back_pressure["current_pressure"] = max(0, self.back_pressure["current_pressure"] - 1)
    
    def adjust_oscillatory_properties(self, field_state):
        """
        Adjust basic oscillatory properties based on field state.
        """
        # Update phase
        self.oscillatory_properties["phase"] += self.oscillatory_properties["frequency"]
        
        # Normalize phase to [0, 2π]
        self.oscillatory_properties["phase"] %= (2 * math.pi)
        
        # Simple frequency adjustment if field state is available
        if hasattr(field_state, "oscillatory_properties"):
            field_frequency = field_state.oscillatory_properties.get("resonance_frequency", 0.1)
            # Gradual shift toward field frequency
            self.oscillatory_properties["frequency"] = (
                0.9 * self.oscillatory_properties["frequency"] +
                0.1 * field_frequency
            )

    def update(self, change_type, value, origin):
        """
        Update the ID with simplified back pressure control.
        """
        # Basic back pressure check
        if self.back_pressure["current_pressure"] > self.back_pressure["threshold"]:
            # Defer update when pressure is too high
            return False
        
        # Simple energy check - fixed cost per update
        energy_cost = 0.1
        
        # Check if we have enough energy
        if self.energy_system["current_energy"] < energy_cost:
            return False
        
        # Consume energy
        self.energy_system["current_energy"] -= energy_cost
        
        # Apply update
        result = super().update(change_type, value, origin)
        
        # Increase back pressure by fixed amount
        self.back_pressure["current_pressure"] += 1
        
        # Generate fixed energy amount
from datetime import datetime
from unittest.mock import MagicMock
from src.habitat_evolution.field.oscillatory_field_state import OscillatoryFieldState

class TestOscillatoryFieldState(unittest.TestCase):
    
    def setUp(self):
        # Create minimal field analysis with only essential properties
        self.field_analysis = {
            "topology": {
                "effective_dimensionality": 3,
                "eigenvalues": [0.5, 0.3, 0.2]
            },
            "metrics": {
                "coherence": 0.7,
                "stability": 0.8
            },
            "field_properties": {
                "coherence": 0.7
            }
        }
        
        # Create field state
        self.field_state = OscillatoryFieldState(self.field_analysis)
    
    def test_coherence_regulation(self):
        """Test essential coherence regulation."""
        # Mock coherence metrics
        self.field_state.get_coherence_metric = lambda: 0.2  # Below threshold
        
        # Mock coherence increase method
        increase_called = [False]
        def mock_increase(current):
            increase_called[0] = True
        self.field_state._increase_coherence = mock_increase
        
        # Test regulation with low coherence
        self.field_state.regulate_coherence()
        self.assertTrue(increase_called[0], "Should increase coherence when below threshold")

from src.habitat_evolution.adaptive_core.id.oscillatory_adaptive_id import OscillatoryAdaptiveID

class TestOscillatoryAdaptiveID(unittest.TestCase):
    
    def setUp(self):
        # Create adaptive ID with minimal parameters
        self.adaptive_id = OscillatoryAdaptiveID(
            base_concept="test_concept",
            creator_id="test_creator"
        )
    
    def test_simplified_back_pressure(self):
        """Test simplified back pressure mechanism."""
        # Set high back pressure
        self.adaptive_id.back_pressure["current_pressure"] = 20  # Above threshold
        
        # Attempt update with high pressure
        result = self.adaptive_id.update("test_change", "test_value", "test_origin")
        
        # Check that update was rejected due to high pressure
        self.assertFalse(result, "Update should be rejected when pressure is high")
        
        # Reset pressure
        self.adaptive_id.back_pressure["current_pressure"] = 0  # Below threshold
        
        # Ensure we have enough energy
        self.adaptive_id.energy_system["current_energy"] = 1.0
        
        # Attempt update with normal pressure
        result = self.adaptive_id.update("test_change", "test_value", "test_origin")
        
        # Check that update was processed
        self.assertTrue(result, "Update should be processed when pressure is normal")

from unittest.mock import MagicMock
from src.habitat_evolution.integration.topology_temporal_actant_journey import TopologyTemporalActantJourney

class TestTopologyTemporalActantJourney(unittest.TestCase):
    
    def setUp(self):
        # Create minimal mock adaptive ID
        self.adaptive_id = MagicMock()
        self.adaptive_id.create_version = MagicMock(return_value="journey_point_1")
        self.adaptive_id.update = MagicMock(return_value=True)
        
        # Create journey
        self.journey = TopologyTemporalActantJourney(self.adaptive_id)
    
    def test_journey_tracking(self):
        """Test essential journey tracking functionality."""
        # Create minimal point data
        point_data = {
            "domain_id": "domain_1",
            "temporal_pattern_id": "pattern_1"
        }
        
        # Add first point
        result = self.journey.add_journey_point(point_data)
        
        # Verify point was added correctly
        self.assertEqual(len(self.journey.journey_points), 1)
        self.assertEqual(result["id"], "journey_point_1")
        
        # Set up for second point with domain transition
        self.adaptive_id.create_version = MagicMock(return_value="journey_point_2")
        point_data2 = {
            "domain_id": "domain_2",
            "temporal_pattern_id": "pattern_1"
        }
        
        # Add second point
        self.journey.add_journey_point(point_data2)
        
        # Verify domain transition was tracked
        self.adaptive_id.update.assert_called_with("domain_transition", {
            "from": "domain_1",
            "to": "domain_2"
        }, "journey")
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
