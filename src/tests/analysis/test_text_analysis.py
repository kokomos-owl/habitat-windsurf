"""Natural text analysis tests."""

import pytest
from datetime import datetime

from src.core.analysis.text_analysis import TextAnalyzer, TextSegment
from src.core.analysis.structure_analysis import StructureContext

@pytest.fixture
def text_analyzer():
    """Create TextAnalyzer instance."""
    return TextAnalyzer()

@pytest.fixture
def analysis_context():
    """Create analysis context."""
    return StructureContext(
        start_time=datetime.now(),
        content_type="text",
        analysis_depth=2
    )

@pytest.mark.asyncio
async def test_natural_segmentation(text_analyzer, analysis_context):
    """Test natural text segmentation."""
    content = """# Section 1
    
This is a paragraph with some natural text.
It continues for a bit to form coherent content.

## Subsection
- List item 1
- List item 2

Another paragraph with different content
to test natural coherence detection."""

    elements = await text_analyzer.analyze_text(content, analysis_context)
    
    # Verify natural emergence
    assert len(elements) > 0
    
    # Check segment types
    segment_types = {e.element_type for e in elements}
    assert 'section' in segment_types
    assert 'list' in segment_types
    
    # Verify depth emergence
    depths = [e.metadata['depth'] for e in elements]
    assert any(d > 0 for d in depths)  # Some segments should have depth

@pytest.mark.asyncio
async def test_coherence_detection(text_analyzer, analysis_context):
    """Test natural coherence detection."""
    content = """First paragraph about a specific topic
that continues with related information.

Second paragraph that's related to the first
and maintains topical coherence.

A completely different paragraph about
an unrelated topic."""

    elements = await text_analyzer.analyze_text(content, analysis_context)
    
    # Get coherence scores
    coherence_scores = [e.metadata['coherence'] for e in elements]
    
    # First two paragraphs should have higher coherence
    assert coherence_scores[0] > 0.3
    assert coherence_scores[1] > 0.3
    
    # Last paragraph should have lower coherence
    assert coherence_scores[-1] < coherence_scores[0]

@pytest.mark.asyncio
async def test_density_emergence(text_analyzer, analysis_context):
    """Test natural density emergence."""
    content = """# Main Topic

First paragraph with substantial content
that forms a coherent block of text with
multiple sentences and ideas.

## Related Subtopic
- Related point 1
- Related point 2

## Different Subtopic
Shorter, less connected paragraph."""

    elements = await text_analyzer.analyze_text(content, analysis_context)
    
    # Check density values
    densities = [e.density for e in elements]
    
    # Verify natural density patterns
    assert all(0 <= d <= 1 for d in densities)
    assert any(d > 0.5 for d in densities)  # Some segments should have high density

@pytest.mark.asyncio
async def test_segment_validation(text_analyzer):
    """Test natural segment validation."""
    # Too short
    short_segment = TextSegment(
        content="Short",
        segment_type="text",
        start_pos=0,
        end_pos=5
    )
    assert not text_analyzer._is_valid_segment(short_segment)
    
    # Valid length
    valid_segment = TextSegment(
        content="This is a valid segment with enough content to analyze",
        segment_type="text",
        start_pos=0,
        end_pos=50
    )
    assert text_analyzer._is_valid_segment(valid_segment)
    
    # Empty content
    empty_segment = TextSegment(
        content="   ",
        segment_type="text",
        start_pos=0,
        end_pos=20
    )
    assert not text_analyzer._is_valid_segment(empty_segment)
