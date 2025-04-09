from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime

from ...infrastructure.services.pattern_evolution_service import PatternEvolutionService
from ..field_state.simple_field_analyzer import SimpleFieldStateAnalyzer
from ..field_state.multi_scale_analyzer import MultiScaleAnalyzer

class FieldPatternBridge:
    """
    Bridges field-state analysis with the pattern evolution service.
    """
    
    def __init__(self, pattern_evolution_service: PatternEvolutionService):
        """
        Initialize the field pattern bridge.
        
        Args:
            pattern_evolution_service: Pattern evolution service to connect with
        """
        self.pattern_evolution_service = pattern_evolution_service
        self.field_analyzer = SimpleFieldStateAnalyzer()
        self.multi_scale_analyzer = MultiScaleAnalyzer()
        
    def process_time_series(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process time series data and bridge to pattern evolution.
        
        Args:
            data: Time series data
            metadata: Optional metadata
            
        Returns:
            Processing results
        """
        # Analyze data with field state analyzer
        field_analysis = self.field_analyzer.analyze_time_series(data)
        
        # Analyze at multiple scales
        multi_scale = self.multi_scale_analyzer.analyze(data)
        
        # Convert field patterns to evolution patterns
        patterns = []
        
        # Add field patterns
        for field_pattern in field_analysis["patterns"]:
            pattern = self._convert_to_evolution_pattern(field_pattern, "field_state", metadata)
            patterns.append(pattern)
            
        # Add cross-scale patterns
        for cross_pattern in multi_scale["cross_scale_patterns"]:
            pattern = self._convert_to_evolution_pattern(cross_pattern, "multi_scale", metadata)
            patterns.append(pattern)
            
        # Store patterns in evolution service
        stored_patterns = []
        for pattern in patterns:
            try:
                # Create pattern in evolution service
                result = self.pattern_evolution_service.create_pattern(
                    pattern_data=pattern,
                    context={
                        "source": "field_pattern_bridge",
                        "created_at": datetime.now().isoformat()
                    }
                )
                stored_patterns.append(result)
            except Exception as e:
                print(f"Error storing pattern: {e}")
                
        return {
            "field_analysis": field_analysis,
            "multi_scale": multi_scale,
            "patterns": stored_patterns
        }
        
    def _convert_to_evolution_pattern(self, source_pattern: Dict[str, Any], 
                                     pattern_source: str,
                                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert a field pattern to an evolution pattern."""
        pattern_id = source_pattern.get("id", f"field_pattern_{uuid.uuid4().hex[:8]}")
        
        # Map field pattern types to evolution pattern types
        type_mapping = {
            "warming_trend": "climate_warming_trend",
            "cooling_trend": "climate_cooling_trend",
            "consistent_warming": "multi_scale_warming",
            "consistent_cooling": "multi_scale_cooling"
        }
        
        pattern_type = type_mapping.get(source_pattern.get("type", ""), "unknown_pattern")
        
        # Create evolution pattern
        pattern = {
            "id": pattern_id,
            "type": pattern_type,
            "source": pattern_source,
            "confidence": source_pattern.get("confidence", 0.7),
            "magnitude": source_pattern.get("magnitude", 0.5),
            "quality_state": "emergent",
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "original_pattern": source_pattern,
                "user_metadata": metadata or {}
            }
        }
        
        # Add position if available
        if "position" in source_pattern:
            pattern["field_position"] = source_pattern["position"]
            
        return pattern
