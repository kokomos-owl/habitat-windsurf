"""Natural structure analysis tests."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.core.analysis.structure_analysis import (
    StructureAnalyzer,
    StructuralElement,
    StructureContext
)
from src.core.types import DensityMetrics

@pytest.fixture
def structure_analyzer():
    """Create StructureAnalyzer with natural dependencies."""
    timestamp_service = Mock()
    timestamp_service.get_timestamp.return_value = datetime.now()
    
    temporal_core = Mock()
    
    return StructureAnalyzer(
        timestamp_service=timestamp_service,
        temporal_core=temporal_core
    )

@pytest.fixture
def sample_elements():
    """Create sample structural elements."""
    now = datetime.now()
    
    return [
        StructuralElement(
            element_id="elem1",
            element_type="section",
            content="Sample content 1",
            density=0.5,
            emergence_time=now
        ),
        StructuralElement(
            element_id="elem2",
            element_type="section",
            content="Sample content 2",
            density=0.7,
            emergence_time=now + timedelta(seconds=1)
        ),
        StructuralElement(
            element_id="elem3",
            element_type="paragraph",
            content="Sample content 3",
            density=0.3,
            emergence_time=now + timedelta(seconds=2)
        )
    ]

@pytest.mark.asyncio
async def test_natural_discovery(structure_analyzer):
    """Test natural structure discovery."""
    content = "Sample content for analysis"
    result = await structure_analyzer.analyze_content(content, "text")
    
    assert "elements" in result
    assert "relationships" in result
    assert "densities" in result
    assert "timestamp" in result

@pytest.mark.asyncio
async def test_relationship_emergence(structure_analyzer, sample_elements):
    """Test natural relationship emergence."""
    # Calculate relationships
    relationships = structure_analyzer._observe_relationships(sample_elements)
    
    # Verify natural grouping of similar elements
    assert "elem1" in relationships
    assert "elem2" in relationships
    assert len(relationships["elem1"]) > 0  # Should find some relationships
    
    # Similar elements should have stronger relationships
    strength1 = structure_analyzer._calculate_relationship_strength(
        sample_elements[0], sample_elements[1]
    )
    strength2 = structure_analyzer._calculate_relationship_strength(
        sample_elements[0], sample_elements[2]
    )
    assert strength1 > strength2  # Same type should have stronger relationship

@pytest.mark.asyncio
async def test_density_emergence(structure_analyzer, sample_elements):
    """Test natural density emergence."""
    # Calculate densities
    densities = structure_analyzer._calculate_densities(sample_elements)
    
    # Verify density metrics
    assert len(densities) == len(sample_elements)
    assert all(isinstance(d, DensityMetrics) for d in densities.values())
    
    # Verify density calculations
    for element_id, metrics in densities.items():
        assert 0 <= metrics.local_density <= 1
        assert 0 <= metrics.global_density <= 1
