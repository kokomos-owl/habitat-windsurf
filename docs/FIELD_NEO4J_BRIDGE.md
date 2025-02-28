# Field-Neo4j Bridge

## Overview

The Field-Neo4j Bridge provides a unified interface for integrating pattern-aware RAG processing with both field theory and Neo4j graph persistence. This component is essential for maintaining coherence in the Habitat system while supporting dual-mode operation.

## Architecture

### Core Components

![Field-Neo4j Bridge Architecture](assets/field_neo4j_bridge.png)

1. **FieldStateNeo4jBridge**: Central coordinator between field observers and persistence
2. **FieldObserver**: Monitors field metrics and state
3. **MockPatternDB**: Testing implementation of the Neo4j pattern database
4. **PatternAdaptiveID**: Visualization-specific implementation of AdaptiveID

### Key Interfaces

```python
class FieldStateNeo4jBridge:
    def __init__(self, 
                 field_observer: Optional[FieldObserver] = None, 
                 persistence_mode: str = "neo4j", 
                 pattern_db = None, 
                 provenance_tracker = None):
        """
        Initialize the bridge between field state and Neo4j persistence.
        
        Args:
            field_observer: Observer for field metrics and state
            persistence_mode: 'neo4j' or 'direct'
            pattern_db: Pattern database interface
            provenance_tracker: Optional tracker for provenance
        """
        self.field_observer = field_observer
        self.persistence_mode = persistence_mode
        self.pattern_db = pattern_db or MockPatternDB()
        self.provenance_tracker = provenance_tracker
        self.logger = logging.getLogger(__name__)
```

## Dual Mode Operation

The bridge supports two operational modes that accommodate different deployment scenarios:

### Neo4j Persistence Mode

In Neo4j mode, all patterns and relationships are persisted to the Neo4j database, maintaining full history and provenance:

```python
def persist_pattern_to_neo4j(self, pattern_data: Dict[str, Any]) -> str:
    """
    Persist a pattern to Neo4j.
    
    Args:
        pattern_data: Pattern data to persist
        
    Returns:
        ID of the persisted pattern
    """
    if self.persistence_mode != "neo4j" or not self.pattern_db:
        return None
        
    try:
        # Prepare pattern for Neo4j
        neo4j_pattern = self.prepare_for_neo4j(pattern_data)
        
        # Persist pattern
        pattern_id = self.pattern_db.save_pattern(neo4j_pattern)
        
        # Track provenance if available
        if self.provenance_tracker:
            self.provenance_tracker.add_provenance(
                entity_id=pattern_id,
                origin="pattern_aware_rag",
                user_id=pattern_data.get("provenance", {}).get("user_id"),
                operation="persist_pattern"
            )
            
        return pattern_id
    except Exception as e:
        self.logger.error(f"Failed to persist pattern to Neo4j: {e}")
        return None
```

### Direct Mode ("Trust Mode")

In direct mode, patterns are processed without persistence, reducing overhead for real-time interactions:

```python
def process_in_direct_mode(self, pattern_data: Dict[str, Any], 
                          user_id: str) -> Dict[str, Any]:
    """
    Process a pattern in direct mode without Neo4j persistence.
    
    Args:
        pattern_data: Pattern data to process
        user_id: ID of the user processing this pattern
        
    Returns:
        Processed pattern with field state
    """
    # Align with field state
    aligned_pattern = self.align_incoming_pattern(pattern_data, user_id)
    
    # Add processing timestamp
    aligned_pattern["processed_at"] = datetime.now().isoformat()
    
    # Add alignment metrics for consistency with Neo4j mode
    aligned_pattern["neo4j_alignment"] = {
        "is_aligned": True,
        "metrics": {"direct_mode": 1.0}
    }
    
    return aligned_pattern
```

## Pattern Processing

### Aligning Incoming Patterns

When a pattern is received, the bridge aligns it with current field state:

```python
def align_incoming_pattern(self, pattern_data: Dict[str, Any], 
                          user_id: str) -> Dict[str, Any]:
    """
    Align an incoming pattern with field state and Neo4j state.
    
    Args:
        pattern_data: Raw pattern data
        user_id: ID of the user who generated this pattern
        
    Returns:
        Aligned pattern with field state integration
    """
    # Create AdaptiveID for provenance tracking
    pattern_id = pattern_data.get("id", f"pattern_{datetime.now().isoformat()}")
    
    # Create adaptive ID if not exists
    if not isinstance(pattern_data.get("adaptive_id"), AdaptiveID):
        pattern_data["adaptive_id"] = AdaptiveID(
            base_concept=pattern_id,
            creator_id=user_id
        )
    
    # Apply field state
    if self.field_observer:
        # Get current field metrics
        field_metrics = self.field_observer.get_field_metrics()
        
        # Get tonic-harmonic analysis if available
        if len(self.field_observer.wave_history) >= 3 and len(self.field_observer.tonic_history) >= 3:
            harmonic_analysis = self.field_observer.perform_harmonic_analysis(
                self.field_observer.wave_history, 
                self.field_observer.tonic_history
            )
            
            # Apply field state metrics
            pattern_data["field_state"] = {
                "stability": self.field_observer.wave_history[-1] if self.field_observer.wave_history else 0.8,
                "tonic": self.field_observer.tonic_history[-1] if self.field_observer.tonic_history else 0.5,
                "harmonic": harmonic_analysis.get("harmonic", [])[-1] if harmonic_analysis.get("harmonic") else 0.4,
                "coherence": field_metrics.get("coherence", 0.7),
                "boundaries": harmonic_analysis.get("boundaries", []),
                "observed_at": datetime.now().isoformat()
            }
            
            # Update adaptive ID with field state for provenance
            if isinstance(pattern_data["adaptive_id"], AdaptiveID):
                pattern_data["adaptive_id"].add_version(
                    {
                        "field_state": pattern_data["field_state"],
                        "confidence": pattern_data.get("confidence", 0.8)
                    },
                    origin="field_bridge"
                )
        
    return pattern_data
```

### Processing Generated Content

The bridge can process both single patterns and lists of patterns:

```python
def process_prompt_generated_content(self, 
                                    content: Union[Dict[str, Any], List[Dict[str, Any]]], 
                                    user_id: str) -> Dict[str, Any]:
    """
    Process prompt-generated content to prepare for Neo4j integration.
    
    Args:
        content: Raw content from pattern-aware RAG
        user_id: ID of the user who generated this content
        
    Returns:
        Processed content with field state metrics and alignment info
    """
    # Handle single pattern or list of patterns
    patterns = content if isinstance(content, list) else [content]
    processed_patterns = []
    
    for pattern in patterns:
        # Align with field state
        aligned_pattern = self.align_incoming_pattern(pattern, user_id)
        
        # Check alignment with Neo4j state
        is_aligned, alignment_metrics = self.check_state_alignment(aligned_pattern)
        aligned_pattern["neo4j_alignment"] = {
            "is_aligned": is_aligned,
            "metrics": alignment_metrics
        }
        
        # Prepare for Neo4j if in Neo4j mode
        if self.persistence_mode == "neo4j":
            neo4j_pattern = self.prepare_for_neo4j(aligned_pattern)
            if neo4j_pattern:
                aligned_pattern["neo4j_pattern"] = neo4j_pattern.to_dict()
            
        processed_patterns.append(aligned_pattern)
        
    # Return as same type as input
    if isinstance(content, list):
        return {
            "patterns": processed_patterns,
            "field_state": self._get_current_field_state(),
            "processed_at": datetime.now().isoformat()
        }
    else:
        return processed_patterns[0] if processed_patterns else {}
```

## Neo4j Integration

### Preparing Patterns for Neo4j

Patterns are transformed for Neo4j persistence using PatternAdaptiveID:

```python
def prepare_for_neo4j(self, pattern_data: Dict[str, Any]) -> Optional[PatternAdaptiveID]:
    """
    Prepare a pattern for Neo4j storage using PatternAdaptiveID.
    
    Args:
        pattern_data: Pattern data to prepare
        
    Returns:
        PatternAdaptiveID instance ready for Neo4j
    """
    try:
        # Extract pattern type and hazard type
        pattern_type = pattern_data.get("pattern_type", "unknown")
        hazard_type = pattern_data.get("hazard_type", "unknown")
        
        pattern_visualization = PatternAdaptiveID(
            pattern_type=pattern_type,
            hazard_type=hazard_type,
            creator_id=pattern_data["adaptive_id"].creator_id if "adaptive_id" in pattern_data else "pattern_bridge",
            confidence=pattern_data.get("confidence", 0.8)
        )
        
        # Update metrics
        field_state = pattern_data.get("field_state", {})
        pattern_visualization.update_metrics(
            position=(0, 0),  # Default position
            field_state=field_state.get("stability", 0.8) * field_state.get("tonic", 0.5),
            coherence=field_state.get("coherence", 0.7),
            energy_state=field_state.get("stability", 0.8)
        )
        
        # Update temporal context if available
        if "temporal_context" in pattern_data:
            pattern_visualization.temporal_context = (
                pattern_data["temporal_context"] 
                if isinstance(pattern_data["temporal_context"], str) 
                else json.dumps(pattern_data["temporal_context"])
            )
            
        return pattern_visualization
        
    except Exception as e:
        self.logger.error(f"Failed to prepare pattern for Neo4j: {e}")
        return None
```

### Checking State Alignment

Patterns are checked for alignment with the current Neo4j state:

```python
def check_state_alignment(self, 
                          pattern_data: Dict[str, Any],
                          threshold: float = 0.7) -> Tuple[bool, Dict[str, float]]:
    """
    Check if the incoming pattern aligns with current Neo4j state.
    
    Args:
        pattern_data: Pattern data to check
        threshold: Minimum alignment threshold (0.0-1.0)
        
    Returns:
        Tuple of (is_aligned, alignment_metrics)
    """
    if not self.pattern_db or self.persistence_mode != "neo4j":
        # In direct mode, always consider aligned
        return True, {"direct_mode": 1.0}
        
    # Get field state from pattern
    pattern_field_state = pattern_data.get("field_state", {})
    
    # Get current Neo4j state - this would call the pattern DB interface
    try:
        neo4j_state = self.pattern_db.get_current_state()
        
        # Calculate alignment metrics
        alignment = {}
        
        # Check pattern name against existing patterns
        if "name" in pattern_data:
            existing_pattern = self.pattern_db.find_by_name(pattern_data["name"])
            if existing_pattern:
                alignment["name_match"] = 1.0
            else:
                # Check similarity to existing pattern names
                name_similarity = self.pattern_db.calculate_name_similarity(pattern_data["name"])
                if name_similarity > 0.8:
                    alignment["name_similarity"] = name_similarity
                
        # Check location if available
        if "location" in pattern_data:
            location_match = self.pattern_db.find_by_location(pattern_data["location"])
            if location_match:
                alignment["location_match"] = 1.0
                
        # Check relationship potential
        if "type" in pattern_data and "impact" in pattern_data:
            relationship_potential = self.pattern_db.calculate_relationship_potential(
                pattern_type=pattern_data["type"],
                impact=pattern_data["impact"]
            )
            alignment["relationship_potential"] = relationship_potential
            
        # Calculate overall alignment score
        if alignment:
            avg_alignment = sum(alignment.values()) / len(alignment)
            is_aligned = avg_alignment >= threshold
        else:
            # No specific alignment checks could be performed
            is_aligned = True
            alignment["default"] = 0.8
            
        return is_aligned, alignment
        
    except Exception as e:
        self.logger.error(f"Failed to check state alignment: {e}")
        return True, {"error": 0.0}
```

## Field State Integration

### Getting Field Metrics

Field metrics are retrieved from the field observer:

```python
def _get_current_field_state(self) -> Dict[str, Any]:
    """Get current field state metrics."""
    if not self.field_observer:
        return {}
        
    field_metrics = self.field_observer.get_field_metrics()
    return {
        "coherence": field_metrics.get("coherence", 0.7),
        "stability": field_metrics.get("stability", 0.8),
        "energy": field_metrics.get("energy", 0.5),
        "timestamp": datetime.now().isoformat()
    }
```

## Testing

### Setup Test Components

When testing the bridge, several components need to be properly configured:

```python
@pytest.fixture
def field_observer():
    """Create a field observer for testing."""
    return FieldObserver(field_id="test_field")

@pytest.fixture
def health_service():
    """Create a system health service for testing."""
    return SystemHealthService()

@pytest.fixture
def health_field_observer(health_service):
    """Create a health-aware field observer for testing."""
    return HealthFieldObserver(field_id="test_field", health_service=health_service)

@pytest.fixture
def pattern_db():
    """Create a mock Neo4j pattern DB for testing."""
    return MockPatternDB()

@pytest.fixture
def neo4j_bridge(health_field_observer, pattern_db):
    """Create a bridge with Neo4j persistence for testing."""
    return FieldStateNeo4jBridge(
        field_observer=health_field_observer,
        persistence_mode="neo4j",
        pattern_db=pattern_db
    )
```

### Test Pattern Alignment

Test that patterns align properly with field state:

```python
def test_align_incoming_pattern(self, neo4j_bridge):
    """Test aligning incoming pattern data with Neo4j state."""
    # Test pattern data
    pattern_data = {
        "name": "Sea Level Rise",
        "type": "climate_risk",
        "probability": 0.85,
        "impact": "high",
        "location": "Martha's Vineyard",
        "temporal_horizon": "2050"
    }
    
    # Define user ID for provenance tracking
    user_id = "test_user_123"
    
    # Align pattern with Neo4j state
    aligned_pattern = neo4j_bridge.align_incoming_pattern(pattern_data, user_id)
    
    # Verify the pattern has been processed correctly
    assert "adaptive_id" in aligned_pattern
    assert aligned_pattern["name"] == "Sea Level Rise"
    assert aligned_pattern["type"] == "climate_risk"
    assert aligned_pattern["location"] == "Martha's Vineyard"
    
    # Field state might or might not be added depending on the observer state
    if "field_state" in aligned_pattern:
        assert "stability" in aligned_pattern["field_state"]
        assert "coherence" in aligned_pattern["field_state"]
```

### Test Content Processing

Test processing generated content:

```python
def test_process_prompt_generated_content_single(self, neo4j_bridge):
    """Test processing single pattern from prompt-generated content."""
    # Single pattern from prompt
    content = {
        "name": "Coastal Erosion",
        "type": "climate_risk",
        "probability": 0.75,
        "impact": "medium",
        "location": "Eastern Shore"
    }
    
    user_id = "test_user_456"
    
    # Process the content
    processed = neo4j_bridge.process_prompt_generated_content(content, user_id)
    
    # Verify processing
    assert isinstance(processed, dict)
    assert "adaptive_id" in processed
    assert processed["name"] == "Coastal Erosion"
    
    # Check neo4j alignment is included
    assert "neo4j_alignment" in processed
    assert "is_aligned" in processed["neo4j_alignment"]
    assert "metrics" in processed["neo4j_alignment"]
```

## Conclusion

The Field-Neo4j Bridge provides a critical integration point between pattern-aware RAG processing, field theory, and Neo4j persistence. By supporting dual-mode operation, it enables both high-performance real-time interactions and comprehensive pattern tracking.

Key benefits include:

1. **Unified Interface**: Consistent pattern handling across modes
2. **Field-Aware Processing**: All patterns integrate with field state
3. **Flexible Deployment**: Support for both Neo4j and direct modes
4. **Provenance Tracking**: Complete history of pattern evolution
5. **Coherent Visualization**: Patterns naturally emerge in visualization

This component is essential for maintaining coherence in the Habitat system while providing the flexibility needed for diverse deployment scenarios.
