"""Tests for climate risk document processor."""

import pytest
from pathlib import Path
from datetime import datetime
from src.core.processor import (
    ClimateRiskProcessor,
    SemanticPatternExtractor,
    RiskMetric,
    ProcessingResult
)

@pytest.fixture
def sample_content():
    return """
    The likelihood of extreme drought will increase from 8.5% to 26% by late-century.
    Wildfire danger days are expected to increase 44% by mid-century.
    Storm risk will increase by 15% by late-century.
    """

@pytest.fixture
def processor():
    return ClimateRiskProcessor()

@pytest.fixture
def extractor():
    return SemanticPatternExtractor()

class TestRiskMetric:
    """Test RiskMetric validation."""
    
    def test_valid_metric(self):
        metric = RiskMetric(
            value=44.0,
            unit='percentage',
            timeframe='mid',
            risk_type='wildfire',
            confidence=0.8,
            source_text='44% by mid-century',
            semantic_weight=0.9
        )
        assert metric.validate() is True
        
    def test_invalid_confidence(self):
        metric = RiskMetric(
            value=44.0,
            unit='percentage',
            timeframe='mid',
            risk_type='wildfire',
            confidence=1.5,  # Invalid
            source_text='44% by mid-century',
            semantic_weight=0.9
        )
        assert metric.validate() is False

class TestSemanticPatternExtractor:
    """Test semantic pattern extraction."""
    
    @pytest.mark.asyncio
    async def test_extract_metrics(self, extractor, sample_content):
        metrics = await extractor.extract(sample_content)
        assert len(metrics) == 3
        
        # Verify drought metric
        drought = next(m for m in metrics if m.risk_type == 'drought')
        assert drought.value == 26.0
        assert drought.timeframe == 'late'
        assert drought.semantic_weight > 0
        
        # Verify wildfire metric
        wildfire = next(m for m in metrics if m.risk_type == 'wildfire')
        assert wildfire.value == 44.0
        assert wildfire.timeframe == 'mid'
        assert wildfire.semantic_weight > 0
        
        # Verify storm metric
        storm = next(m for m in metrics if m.risk_type == 'storm')
        assert storm.value == 15.0
        assert storm.timeframe == 'late'
        assert storm.semantic_weight > 0
        
    @pytest.mark.asyncio
    async def test_extract_empty_content(self, extractor):
        metrics = await extractor.extract("")
        assert len(metrics) == 0

    @pytest.mark.asyncio
    async def test_semantic_weights(self, extractor):
        content = "Drought risk will significantly increase from 10% to 30% by late-century"
        metrics = await extractor.extract(content)
        assert len(metrics) == 1
        assert metrics[0].semantic_weight > 0.5  # Higher weight due to "significantly"

class TestClimateRiskProcessor:
    """Test document processing with semantic analysis."""
    
    @pytest.mark.asyncio
    async def test_process_missing_document(self, processor):
        result = await processor.process_document("")
        assert not result.is_valid()
        assert "Missing document path" in result.validation_errors
        
    @pytest.mark.asyncio
    async def test_process_nonexistent_document(self, processor):
        result = await processor.process_document("/nonexistent/path")
        assert not result.is_valid()
        assert any("not found" in e for e in result.validation_errors)
        
    @pytest.mark.asyncio
    async def test_process_valid_document(self, processor, tmp_path):
        # Create test document
        doc_path = tmp_path / "test.txt"
        doc_path.write_text("""
        The likelihood of extreme drought will increase from 8.5% to 26% by late-century.
        Wildfire danger days are expected to increase 44% by mid-century.
        """)
        
        result = await processor.process_document(str(doc_path))
        assert result.is_valid()
        assert len(result.metrics) > 0
        assert result.coherence_score > 0
        
        # Verify metrics
        drought = next(m for m in result.metrics if m.risk_type == 'drought')
        assert drought.value == 26.0
        assert drought.timeframe == 'late'
        
        wildfire = next(m for m in result.metrics if m.risk_type == 'wildfire')
        assert wildfire.value == 44.0
        assert wildfire.timeframe == 'mid'
        
    @pytest.mark.asyncio
    async def test_coherence_scoring(self, processor, tmp_path):
        # Create test document with strong semantic relationships
        doc_path = tmp_path / "test.txt"
        doc_path.write_text("""
        Climate risks are increasing significantly:
        Drought probability will rise from 10% to 30% by late-century.
        Wildfire danger will surge by 25% by late-century.
        Storm intensity will increase dramatically by 40% by late-century.
        """)
        
        result = await processor.process_document(str(doc_path))
        assert result.is_valid()
        assert result.coherence_score > 0.7  # High coherence due to strong relationships
