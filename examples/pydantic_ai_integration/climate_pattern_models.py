"""
Example of using PydanticAI to define climate pattern models for Habitat Evolution.
This demonstrates how PydanticAI can be used to create structured, validated pattern
representations that can be integrated with Windsurf and Habitat.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
import pydanticai  # This would be the PydanticAI extension


class SpatialContext(BaseModel):
    """Spatial context for a climate pattern."""
    region: str = Field(..., description="Geographic region name (e.g., 'Cape Cod', 'Boston Harbor')")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Latitude/longitude coordinates if available")
    area_type: str = Field(..., description="Type of area (e.g., 'coastal', 'inland', 'island')")
    
    @pydanticai.ai_validator
    def normalize_region_name(cls, region: str) -> str:
        """Use AI to normalize region names to a standard format."""
        # This would use AI to standardize region names
        # For example, "Boston Harbor Islands" and "Boston Harbor" might be normalized
        return region


class TemporalContext(BaseModel):
    """Temporal context for a climate pattern."""
    start_date: datetime = Field(..., description="Start date of the pattern observation")
    end_date: Optional[datetime] = Field(None, description="End date of the pattern observation")
    time_scale: str = Field(..., description="Time scale of the pattern (e.g., 'daily', 'monthly', 'annual', 'decadal')")
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        if v and 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class PatternConfidence(BaseModel):
    """Confidence metrics for a climate pattern."""
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    method: str = Field(..., description="Method used to calculate confidence")
    factors: List[str] = Field(default_factory=list, description="Factors influencing confidence")


class ClimatePattern(BaseModel):
    """Base model for climate patterns detected by Habitat Evolution."""
    id: str = Field(..., description="Unique identifier for the pattern")
    name: str = Field(..., description="Descriptive name of the pattern")
    description: str = Field(..., description="Detailed description of the pattern")
    pattern_type: str = Field(..., description="Type of pattern (e.g., 'trend', 'anomaly', 'relationship')")
    quality_state: Literal["hypothetical", "emergent", "stable"] = Field(
        ..., description="Current quality state of the pattern"
    )
    spatial_context: SpatialContext = Field(..., description="Spatial context of the pattern")
    temporal_context: TemporalContext = Field(..., description="Temporal context of the pattern")
    confidence: PatternConfidence = Field(..., description="Confidence metrics for the pattern")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @pydanticai.ai_field_extractor
    def extract_related_patterns(self, text: str) -> List[str]:
        """Use AI to extract related pattern IDs from text descriptions."""
        # This would use AI to identify mentions of other patterns
        # and return their IDs
        return []


class StatisticalPattern(ClimatePattern):
    """Statistical pattern detected in climate data."""
    data_source: str = Field(..., description="Source of the data (e.g., 'NOAA temperature data')")
    statistical_method: str = Field(..., description="Statistical method used to detect the pattern")
    magnitude: float = Field(..., description="Magnitude of the pattern")
    unit: str = Field(..., description="Unit of measurement")
    baseline: Optional[Dict[str, Any]] = Field(None, description="Baseline for comparison")


class SemanticPattern(ClimatePattern):
    """Semantic pattern detected in climate risk documents."""
    document_source: str = Field(..., description="Source document reference")
    key_terms: List[str] = Field(..., description="Key terms associated with the pattern")
    sentiment: Optional[float] = Field(None, description="Sentiment score if applicable")
    
    @pydanticai.ai_field_extractor
    def extract_key_terms(self, text: str) -> List[str]:
        """Use AI to extract key terms from text."""
        # This would use AI to identify important terms
        return []


class PatternRelationship(BaseModel):
    """Relationship between two climate patterns."""
    id: str = Field(..., description="Unique identifier for the relationship")
    source_pattern_id: str = Field(..., description="ID of the source pattern")
    target_pattern_id: str = Field(..., description="ID of the target pattern")
    relationship_type: str = Field(..., description="Type of relationship")
    strength: float = Field(..., ge=0.0, le=1.0, description="Strength of the relationship")
    description: str = Field(..., description="Description of the relationship")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting the relationship")
    
    @pydanticai.ai_validator
    def validate_relationship_type(cls, relationship_type: str) -> str:
        """Use AI to validate and normalize relationship types."""
        # This would use AI to ensure relationship types follow a standard taxonomy
        return relationship_type


class AdaptivePatternID(BaseModel):
    """Adaptive ID for tracking pattern evolution over time."""
    core_id: str = Field(..., description="Core identifier that remains stable across versions")
    version: int = Field(..., ge=1, description="Version number of the pattern")
    previous_versions: List[str] = Field(default_factory=list, description="IDs of previous versions")
    creation_date: datetime = Field(..., description="Date when this version was created")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Coherence score for this version")
    context_hash: str = Field(..., description="Hash representing the context in which this version exists")


# Example of how these models could be used with Windsurf and Habitat

class HabitatPatternService:
    """Example service that would use PydanticAI models with Habitat and Windsurf."""
    
    def create_pattern_from_climate_data(self, data: Dict[str, Any]) -> StatisticalPattern:
        """Create a statistical pattern from climate data."""
        # Process the data and create a StatisticalPattern
        pattern = StatisticalPattern(
            id=f"stat-{datetime.now().timestamp()}",
            name=f"Temperature Trend in {data['region']}",
            description=f"Temperature trend detected in {data['region']} from {data['start_date']} to {data['end_date']}",
            pattern_type="trend",
            quality_state="emergent",
            spatial_context=SpatialContext(
                region=data['region'],
                coordinates=data.get('coordinates'),
                area_type=data.get('area_type', 'coastal')
            ),
            temporal_context=TemporalContext(
                start_date=datetime.fromisoformat(data['start_date']),
                end_date=datetime.fromisoformat(data['end_date']),
                time_scale=data.get('time_scale', 'monthly')
            ),
            confidence=PatternConfidence(
                score=0.85,
                method="statistical_analysis",
                factors=["data_quality", "sample_size"]
            ),
            data_source=data['source'],
            statistical_method="linear_regression",
            magnitude=data['magnitude'],
            unit=data['unit'],
            baseline=data.get('baseline')
        )
        return pattern
    
    def create_pattern_from_document(self, document: Dict[str, Any]) -> SemanticPattern:
        """Create a semantic pattern from a climate risk document."""
        # Process the document and create a SemanticPattern
        pattern = SemanticPattern(
            id=f"sem-{datetime.now().timestamp()}",
            name=f"Sea Level Rise Impact in {document['region']}",
            description=document['content'],
            pattern_type="impact",
            quality_state="hypothetical",
            spatial_context=SpatialContext(
                region=document['region'],
                area_type=document.get('area_type', 'coastal')
            ),
            temporal_context=TemporalContext(
                start_date=datetime.fromisoformat(document['publication_date']),
                time_scale="annual"
            ),
            confidence=PatternConfidence(
                score=0.7,
                method="semantic_analysis",
                factors=["source_credibility", "recency"]
            ),
            document_source=document['source'],
            key_terms=document.get('key_terms', [])
        )
        return pattern
    
    def create_pattern_relationship(self, source_pattern: ClimatePattern, target_pattern: ClimatePattern, 
                                   relationship_info: Dict[str, Any]) -> PatternRelationship:
        """Create a relationship between two patterns."""
        relationship = PatternRelationship(
            id=f"rel-{source_pattern.id}-{target_pattern.id}",
            source_pattern_id=source_pattern.id,
            target_pattern_id=target_pattern.id,
            relationship_type=relationship_info['type'],
            strength=relationship_info['strength'],
            description=relationship_info['description'],
            evidence=relationship_info.get('evidence', [])
        )
        return relationship


# Example of how this might be used with Windsurf for persistence

class WindsurfPatternPersistence:
    """Example of how Windsurf could be used to persist PydanticAI pattern models."""
    
    def __init__(self, connection_string: str):
        """Initialize with Windsurf connection string."""
        self.connection_string = connection_string
        # In a real implementation, this would initialize Windsurf connection
    
    def save_pattern(self, pattern: ClimatePattern) -> str:
        """Save a pattern to Windsurf."""
        # In a real implementation, this would:
        # 1. Convert the Pydantic model to a dict
        pattern_dict = pattern.dict()
        # 2. Use Windsurf to persist the pattern
        # windsurf_client.save_entity("climate_patterns", pattern_dict)
        return pattern.id
    
    def save_relationship(self, relationship: PatternRelationship) -> str:
        """Save a pattern relationship to Windsurf."""
        # In a real implementation, this would:
        # 1. Convert the Pydantic model to a dict
        relationship_dict = relationship.dict()
        # 2. Use Windsurf to persist the relationship
        # windsurf_client.save_relationship(
        #     "pattern_relationships", 
        #     relationship_dict,
        #     source_id=relationship.source_pattern_id,
        #     target_id=relationship.target_pattern_id
        # )
        return relationship.id
    
    def get_pattern(self, pattern_id: str) -> ClimatePattern:
        """Retrieve a pattern from Windsurf."""
        # In a real implementation, this would:
        # 1. Use Windsurf to retrieve the pattern
        # pattern_dict = windsurf_client.get_entity("climate_patterns", pattern_id)
        # 2. Determine the pattern type and create the appropriate model
        # if pattern_dict["pattern_type"] == "statistical":
        #     return StatisticalPattern(**pattern_dict)
        # else:
        #     return SemanticPattern(**pattern_dict)
        pass
    
    def get_related_patterns(self, pattern_id: str) -> List[PatternRelationship]:
        """Get all relationships for a pattern."""
        # In a real implementation, this would:
        # 1. Use Windsurf to retrieve relationships
        # relationships = windsurf_client.get_relationships("pattern_relationships", pattern_id)
        # 2. Convert to PatternRelationship models
        # return [PatternRelationship(**rel) for rel in relationships]
        pass


# Example usage
if __name__ == "__main__":
    # This would demonstrate how to use these models in practice
    pattern_service = HabitatPatternService()
    persistence = WindsurfPatternPersistence("windsurf://localhost:8000")
    
    # Create a statistical pattern from climate data
    climate_data = {
        "region": "Cape Cod",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "source": "NOAA temperature data",
        "magnitude": 1.2,
        "unit": "°C/decade",
        "area_type": "coastal",
        "time_scale": "monthly",
        "coordinates": {"lat": 41.6688, "lon": -70.2962}
    }
    
    stat_pattern = pattern_service.create_pattern_from_climate_data(climate_data)
    persistence.save_pattern(stat_pattern)
    
    # Create a semantic pattern from a document
    document_data = {
        "region": "Cape Cod",
        "content": "Sea level rise in Cape Cod is projected to increase coastal flooding frequency by 50% by 2050.",
        "publication_date": "2023-05-15",
        "source": "Massachusetts Climate Change Report 2023",
        "key_terms": ["sea level rise", "coastal flooding", "projection"]
    }
    
    sem_pattern = pattern_service.create_pattern_from_document(document_data)
    persistence.save_pattern(sem_pattern)
    
    # Create a relationship between the patterns
    relationship_info = {
        "type": "causal",
        "strength": 0.8,
        "description": "Temperature increase contributes to sea level rise through thermal expansion and ice melt",
        "evidence": ["IPCC AR6 Report", "Massachusetts Climate Change Report 2023"]
    }
    
    relationship = pattern_service.create_pattern_relationship(stat_pattern, sem_pattern, relationship_info)
    persistence.save_relationship(relationship)
