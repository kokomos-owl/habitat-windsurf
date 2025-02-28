# Field Awareness in Habitat

*Date: February 27, 2025*

## Introduction

Field Awareness represents a revolutionary approach to pattern recognition in the Habitat system. Unlike traditional pattern matching that relies on rigid boundaries and explicit relationships, Field Awareness observes the natural emergence of patterns through their contextual relationships and interactions. This document outlines the core principles, implementation details, and integration points of Field Awareness in the Habitat system.

## Core Principles

### Natural Emergence vs. Enforcement

Field Awareness operates on the principle that patterns naturally emerge from the semantic field rather than being enforced by the system. This allows for:

1. **Organic Pattern Formation**: Patterns form based on natural relationships and semantic proximity
2. **Self-Regulation**: The system adjusts its observation without imposing artificial constraints
3. **Adaptive Boundaries**: Pattern boundaries shift naturally as the field evolves

### Coherence as a Meta-Interface

Coherence serves as a higher-order interface that bridges system components and external systems:

1. **Bidirectional Boundary Adherence**: 
   - External systems must adhere to Habitat's coherence boundaries
   - Habitat itself adheres to the natural boundaries of emergent patterns
   - This creates a homeostatic relationship between the system and its environment

2. **Observation Without Intervention**:
   - The system observes field states without disrupting natural pattern formation
   - Changes in field coherence are recorded rather than directed

## Health Integration

### System Health and Field Awareness

The integration between system health and field awareness creates a bidirectional relationship:

1. **Health-Aware Field Observation**:
   - `HealthFieldObserver` connects field metrics with system health metrics
   - Tonic values derived from health metrics influence field observations
   - Field state transitions adapt to system health conditions

2. **Field-Aware Health Monitoring**:
   - System health adapts to observed field coherence
   - Pattern shifts trigger health metric adjustments
   - Health service receives field context during observations

Here's an example of how the `HealthFieldObserver` observes and integrates with system health:

```python
async def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """Add observation and notify health service if configured."""
    # Record observation
    now = datetime.now()
    self.observations.append({"context": context, "time": now})
    
    # Get stability from context (handle both field name formats)
    stability = context.get("stability", context.get("stability_score", 0.8))
    
    # Add to wave history for analysis
    self.wave_history.append(stability)
    
    # Call parent observe method
    parent_response = await super().observe(context)
    
    # Update field metrics
    self.update_field_metrics(context)
    
    # Request health observation if service available
    if self.health_service:
        health_data = self.health_service.observe(context)
        
        # Extract tonic value if available
        if "rhythm_patterns" in health_data and "stability" in health_data["rhythm_patterns"]:
            tonic = health_data["rhythm_patterns"]["stability"].get("tonic", 0.5)
            self.tonic_history.append(tonic)
            self.field_metrics["tonic"] = tonic
            
            # Add resonance metrics if available
            if "resonance" in health_data["rhythm_patterns"]["stability"]:
                self.field_metrics["resonance"] = health_data["rhythm_patterns"]["stability"]["resonance"]
        
        return {**parent_response, **health_data}
    
    return parent_response
```

### Tonic-Harmonic Pattern Detection

The tonic-harmonic approach represents a breakthrough in natural boundary detection:

1. **Stability Wave Analysis**:
   - Field stability creates a "wave" pattern over time
   - System observes natural rhythms in stability fluctuations
   - Wave patterns correlate with meaningful semantic shifts

2. **Tonic Value Integration**:
   - Tonic values represent baseline system state
   - Health service provides tonic values for each observation context
   - Correlation between stability waves and tonic values reveals natural boundaries

3. **Harmonic Boundary Detection**:
   - `perform_harmonic_analysis` method identifies contextual boundaries
   - Natural transitions emerge at points of harmonic resonance
   - Field-aware state transitions occur at these detected boundaries

The following code shows how the system performs harmonic analysis to detect contextual boundaries:

```python
def perform_harmonic_analysis(self, wave: List[float], tonic: List[float]) -> Dict[str, Any]:
    """Analyze harmonic patterns to detect contextual boundaries.
    
    Uses correlation between wave and tonic to identify significant shifts.
    
    Args:
        wave: Wave pattern (typically stability scores)
        tonic: Tonic pattern from field
        
    Returns:
        Analysis results including detected boundaries
    """
    import numpy as np
    
    # Ensure equal lengths
    min_length = min(len(wave), len(tonic))
    if min_length < 3:
        return {"boundaries": [], "harmonic": []}
        
    wave = wave[:min_length]
    tonic = tonic[:min_length]
    
    # Calculate harmonic relationship
    harmonic = [w * t for w, t in zip(wave, tonic)]
    
    # Find significant changes using derivatives
    derivatives = [harmonic[i+1] - harmonic[i] for i in range(len(harmonic)-1)]
    
    # Use rolling window to identify significant shifts
    window_size = 2
    deltas = []
    
    for i in range(len(derivatives) - window_size + 1):
        window = derivatives[i:i+window_size]
        deltas.append(sum(window) / window_size)
    
    # Find the largest absolute changes
    abs_deltas = [abs(d) for d in deltas]
    
    # Use percentile for dynamic threshold
    if len(abs_deltas) >= 3:
        threshold = np.percentile(abs_deltas, 75)
        
        # Find all peaks above threshold
        boundaries = [
            i for i in range(len(deltas))
            if abs(deltas[i]) > threshold
        ]
    else:
        # If not enough data, use index of max change
        boundaries = [abs_deltas.index(max(abs_deltas))] if abs_deltas else []
    
    if not boundaries and deltas:
        # Always return at least the max shift point if available
        max_delta_index = deltas.index(max(deltas, key=abs))
        boundaries = [max_delta_index]
        
    return {
        "boundaries": boundaries,
        "harmonic": harmonic,
        "derivatives": derivatives,
        "deltas": deltas,
        "threshold": threshold if len(abs_deltas) >= 3 else None
    }
```

In practice, this is how a test might detect a contextual boundary using tonic-harmonic analysis:

```python
# Set specific tonic pattern for testing
health_service.set_tonic_pattern([0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5])

# Create window with field observer
window = event_coordinator.create_learning_window(
    duration_minutes=5,
    stability_threshold=0.7,
    coherence_threshold=0.6,
    max_changes=20  # Increased to prevent early saturation
)
window.register_field_observer(field_observer)

# Generate semantic wave with known pattern
# Context shift after event 5
semantic_pattern = [
    {"score": 0.8, "id": "concept_1"},
    {"score": 0.82, "id": "concept_2"},
    {"score": 0.85, "id": "concept_3"},
    {"score": 0.87, "id": "concept_4"},
    {"score": 0.84, "id": "concept_5"},
    {"score": 0.75, "id": "concept_6"},  # Context shift
    {"score": 0.73, "id": "concept_7"},
    {"score": 0.68, "id": "concept_8"},  # Clear downward trend
    {"score": 0.65, "id": "concept_9"},
]

# Process events through the window
for event in semantic_pattern:
    window.record_change(
        event_id=event["id"],
        stability_score=event["score"]
    )
    
    # Allow time for cycle progression and model processing
    import time
    time.sleep(0.1)
    
    # Request health observation to get tonic
    context = {
        "state": window.state.value,
        "stability": event["score"],
        "coherence": window.coherence_threshold,
        "saturation": window.change_count / window.max_changes_per_window
    }
    
    # Add direct observation to field observer
    field_observer.observations.append({"context": context, "time": datetime.now()})
    
    # Get health report to ensure tonic values
    health_report = health_service.observe(context)
    
    # Manually update field metrics with tonic
    if "rhythm_patterns" in health_report and "stability" in health_report["rhythm_patterns"]:
        field_observer.field_metrics["tonic"] = health_report["rhythm_patterns"]["stability"]["tonic"]

# Extract stability scores (base wave)
stability_scores = [event["score"] for event in semantic_pattern]

# Get tonic values
tonic_values = field_observer.tonic_history

# If not enough tonic values, get from health service
if len(tonic_values) < len(stability_scores):
    tonic_values = health_service.tonic_pattern[:len(stability_scores)]
    
# Perform harmonic analysis to detect context shift
analysis = field_observer.perform_harmonic_analysis(
    stability_scores, 
    tonic_values[:len(stability_scores)]
)

# Check if boundaries were detected in expected range (after index 4-6)
print(f"Detected boundaries: {analysis.get('boundaries', [])}")
boundaries = analysis.get('boundaries', [])

# Find boundaries in the expected range
relevant_boundaries = [b for b in boundaries if 4 <= b <= 6]

assert relevant_boundaries, f"No harmonic boundaries detected in expected range (4-6). Found: {boundaries}"
```

### Health Metric Integration

The integration with system health metrics provides a comprehensive view of system state:

1. **Health Metrics in Field Context**:
   - Stability scores influence health metrics
   - Coherence thresholds adapt based on system health
   - Field pressure adjusts according to health metrics

2. **Rhythm Patterns**:
   - System detects rhythm patterns in both health and field metrics
   - Resonance measured between different metric dimensions
   - Cross-dimensional harmony indicates optimal system state

The following code demonstrates how the `HealthIntegratedBackPressure` controller calculates delay based on system health:

```python
def calculate_delay_with_health(self, stability_score: float) -> float:
    """Calculate delay incorporating system health metrics."""
    # Calculate base delay from regular algorithm
    base_delay = self.calculate_delay(stability_score)
    
    # If no health service, return base delay
    if not self.health_service:
        return base_delay
        
    # Get observation about current back pressure state
    observation = {
        "stability_score": stability_score,
        "base_delay": base_delay,
        "current_pressure": self.current_pressure,
        "component": "back_pressure_controller"
    }
    
    # Get health report
    self.last_health_report = self.health_service.observe(observation)
    
    # Extract health metrics that influence pressure
    current_status = self.last_health_report.get('current_status', {})
    system_stress = current_status.get('system_stress', 0.5)
    
    # Get resonance for natural rhythm alignment
    resonance = self.last_health_report.get('resonance_levels', {})
    system_resonance = resonance.get('cross_dimensional', 0.5)
    self.system_resonance = system_resonance
    
    # Modify delay based on health metrics
    # - High stress increases delay (protective)
    # - High resonance decreases delay (things are flowing naturally)
    stress_factor = 1.0 + (system_stress - 0.5) * self.health_influence
    resonance_factor = 1.0 - (system_resonance - 0.5) * self.health_influence
    
    # Combine factors with natural alignment
    health_adjusted_delay = base_delay * stress_factor * resonance_factor
    
    # Ensure within bounds
    return np.clip(health_adjusted_delay, self.base_delay, self.max_delay)
```

## Implementation Details

### HealthFieldObserver

The `HealthFieldObserver` class serves as the primary integration point between field observation and system health:

```python
class HealthFieldObserver(FieldObserver):
    """Field observer that integrates with system health.
    
    Connects field observation with system health metrics to create a
    tonic-harmonic approach to detecting contextual boundaries.
    """
    
    def __init__(self, field_id: str, health_service: SystemHealthService):
        super().__init__(field_id)
        self.health_service = health_service
```

Key functionality includes:

1. **Health-Aware Observation**:
   - Records field observations with health context
   - Extracts tonic values from health data
   - Updates field metrics with resonance patterns

2. **Optimal Transition Detection**:
   - Analyzes tonic-harmonic patterns to find natural boundaries
   - Identifies peak indices in tonic patterns
   - Calculates optimal transition times for learning windows

The `HealthFieldObserver` implements the `get_optimal_transition_time` method, which identifies natural boundaries for window transitions:

```python
def get_optimal_transition_time(self, 
                              window_start: datetime,
                              window_duration: timedelta) -> Optional[datetime]:
    """Find optimal time for window transition based on field harmonics.
    
    Analyzes tonic-harmonic patterns to detect natural boundaries for transitions.
    """
    if len(self.tonic_history) < 3 or "observation_rhythm" not in self.field_metrics:
        return None
        
    # Estimate tonic cycle period from observations
    if len(self.field_metrics["observation_rhythm"]) > 2:
        rhythm_mean = np.mean(self.field_metrics["observation_rhythm"])
        
        # Find pattern in tonic values
        tonic_values = np.array(self.tonic_history)
        peak_indices = []
        
        # Find peaks in tonic pattern
        for i in range(1, len(tonic_values)-1):
            if tonic_values[i] > tonic_values[i-1] and tonic_values[i] > tonic_values[i+1]:
                peak_indices.append(i)
        
        # Calculate tonic period if peaks found
        if len(peak_indices) >= 2:
            avg_peak_distance = np.mean([peak_indices[i+1] - peak_indices[i] 
                                       for i in range(len(peak_indices)-1)])
            tonic_period = avg_peak_distance * rhythm_mean
            
            # Calculate next peak time
            now = datetime.now()
            last_peak_time = self.observations[peak_indices[-1]]["time"]
            time_since_peak = (now - last_peak_time).total_seconds()
            time_to_next_peak = tonic_period - (time_since_peak % tonic_period)
            
            # Calculate optimal transition time (at tonic peak)
            return now + timedelta(seconds=time_to_next_peak)
    
    # Default to half the window duration if pattern analysis failed
    return window_start + (window_duration / 2)
```

### HealthAwareLearningWindow

The `HealthAwareLearningWindow` extends the base Learning Window with health awareness:

```python
class HealthAwareLearningWindow(LearningWindow):
    """Learning window with integrated health awareness.
    
    Extends the basic learning window to incorporate system health metrics
    and adapts its behavior based on system health patterns.
    """
```

Key functionality includes:

1. **Health Metric Recording**:
   - Tracks learning-related health metrics
   - Maintains a metric profile for trend analysis
   - Updates dimension resonance across different context dimensions

2. **Adaptive Behavior**:
   - Adjusts window parameters based on health metrics
   - Modifies state transitions based on system health
   - Adapts coherence thresholds to maintain system integrity

### HealthIntegratedBackPressure

The `HealthIntegratedBackPressure` controller manages system pressure with health awareness:

```python
class HealthIntegratedBackPressure(BackPressureController):
    """Back pressure controller with system health integration.
    
    Extends the basic back pressure controller to incorporate system health
    metrics and adapts its delay calculations based on health patterns.
    """
```

Key functionality includes:

1. **Health-Aware Delay Calculation**:
   - Incorporates system health into delay calculations
   - Adjusts pressure based on health metrics
   - Maintains system resonance through adaptive delays

## Integration Points

### With Pattern-Aware RAG

Field Awareness integrates with the Pattern-Aware RAG system through:

1. **Learning Window Integration**:
   - Field observers attach to learning windows
   - Pattern emergence aligns with field observations
   - Window transitions correspond to field boundaries

2. **Dual-Mode Support**:
   - Field awareness works in both Neo4j persistence mode and Direct LLM mode
   - Natural boundaries detected regardless of persistence mechanism
   - Coherence maintained across operational modes

### With External Systems

Field Awareness creates integration points for external systems:

1. **Coherence Boundary Interface**:
   - External systems interact through coherence boundaries
   - Field metrics provide visibility into system state
   - Natural pattern boundaries serve as integration points

2. **Health Metric Exchange**:
   - External systems can provide and consume health metrics
   - Tonic-harmonic patterns expose system rhythm
   - Cross-system resonance enables natural integration

## Observable Products of Homeostasis

Though homeostasis itself is difficult to directly observe, its products provide visibility:

1. **Field Metrics**:
   - Stability scores over time
   - Coherence measurements
   - Tonic values and resonance patterns

2. **Pattern Emergence**:
   - Natural boundary formation
   - Pattern visibility through field state
   - Relationship strength based on field coherence

3. **System Adaptations**:
   - Window transition timing
   - Pressure response curves
   - Rhythm pattern evolution

## Conclusion

Field Awareness represents a paradigm shift in pattern recognition, moving from enforced structures to observed emergence. By integrating system health with semantic understanding through coherence as a meta-interface, Habitat creates a self-regulating ecosystem that maintains integrity while adapting to natural pattern boundaries.

The tonic-harmonic approach to boundary detection provides a natural way to identify meaningful transitions in the semantic field, enabling more effective pattern recognition without artificial constraints. This approach aligns with the core principle that coherence emerges naturally through observation rather than enforcement.

Field Awareness establishes a foundation for future enhancements to the Habitat system, enabling more natural pattern recognition and more effective integration with external systems through coherence boundaries.
