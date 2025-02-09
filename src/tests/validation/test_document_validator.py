"""Tests for climate risk document validator."""

import pytest
from datetime import datetime
from pathlib import Path
from src.core.validation.document_validator import (
    ClimateDocumentValidator,
    DocumentSection,
    ValidationReport
)

@pytest.fixture
def validator():
    """Create fresh validator for each test."""
    return ClimateDocumentValidator()

@pytest.fixture
def sample_document(tmp_path):
    """Create a sample climate risk document."""
    content = """
INTRODUCTION

The impacts of climate change on Martha's Vineyard from 2020-2050 show significant trends.

METHODOLOGY

Analysis conducted over 2001-2020 period shows:
- Temperature increase of 2.5 degrees
- Precipitation change of 55%
- Drought conditions 8.5% of the time

FINDINGS

By mid-century (2050):
- Drought likelihood increases to 13%
- Fire danger days increase by 94%
- Heavy precipitation events up 55%

RECOMMENDATIONS

Short-term (2025-2030):
1. Implement water conservation
2. Enhance fire prevention
3. Improve flood resilience

Long-term (2040-2050):
1. Infrastructure adaptation
2. Ecosystem restoration
"""
    doc_path = tmp_path / "test_document.txt"
    doc_path.write_text(content)
    return str(doc_path)

def test_document_validation(validator, sample_document):
    """Test basic document validation."""
    report = validator.validate_document(sample_document)
    
    assert report.is_valid
    assert len(report.sections) == 4
    assert report.temporal_range == (2001, 2050)
    assert report.confidence_score >= 0.8
    assert not report.errors
    assert not report.warnings

def test_section_extraction(validator, sample_document):
    """Test section extraction and content."""
    report = validator.validate_document(sample_document)
    
    # Check all required sections present
    section_names = [s.name.lower() for s in report.sections]
    for required in validator.required_sections:
        assert required in section_names
        
    # Check section content
    findings = next(s for s in report.sections if s.name.lower() == "findings")
    assert "mid-century" in findings.content
    assert "13%" in findings.content

def test_temporal_marker_extraction(validator, sample_document):
    """Test temporal marker extraction."""
    report = validator.validate_document(sample_document)
    
    # Get all temporal markers
    all_markers = []
    for section in report.sections:
        all_markers.extend(section.temporal_markers)
    
    expected_markers = [
        "2020-2050", "2001-2020", "2050",
        "2025-2030", "2040-2050"
    ]
    
    for marker in expected_markers:
        assert marker in all_markers

def test_metric_extraction(validator, sample_document):
    """Test metric extraction from content."""
    report = validator.validate_document(sample_document)
    
    # Get all metrics
    all_metrics = []
    for section in report.sections:
        if section.metrics:
            all_metrics.extend(section.metrics)
    
    expected_values = [2.5, 55, 8.5, 13, 94, 55]
    found_values = [m['value'] for m in all_metrics]
    
    for value in expected_values:
        assert value in found_values

def test_invalid_document(validator, tmp_path):
    """Test validation of invalid document."""
    # Create document missing sections
    content = """
INTRODUCTION

Some introductory text.

FINDINGS

Some findings.
"""
    doc_path = tmp_path / "invalid_document.txt"
    doc_path.write_text(content)
    
    report = validator.validate_document(str(doc_path))
    
    assert not report.is_valid
    assert report.confidence_score < 0.7
    assert len(report.warnings) > 0
    assert "methodology" in report.warnings[0].lower()

def test_nonexistent_document(validator):
    """Test validation of nonexistent document."""
    report = validator.validate_document("nonexistent.txt")
    
    assert not report.is_valid
    assert report.confidence_score == 0.0
    assert len(report.errors) == 1
    assert "Failed to read document" in report.errors[0]

def test_confidence_calculation(validator, tmp_path):
    """Test confidence score calculation."""
    # Create document with varying completeness
    documents = [
        # Complete document
        """
        INTRODUCTION
        Text from 2020-2050
        Value: 10%
        
        METHODOLOGY
        Analysis from 2001-2020
        Value: 20%
        
        FINDINGS
        Results show 30%
        
        RECOMMENDATIONS
        Suggest 40% improvement
        """,
        
        # Missing temporal markers
        """
        INTRODUCTION
        Some text
        Value: 10%
        
        METHODOLOGY
        Some methods
        Value: 20%
        
        FINDINGS
        Results
        
        RECOMMENDATIONS
        Suggestions
        """,
        
        # Missing metrics
        """
        INTRODUCTION
        Text from 2020-2050
        
        METHODOLOGY
        Analysis from 2001-2020
        
        FINDINGS
        Results
        
        RECOMMENDATIONS
        Suggestions
        """
    ]
    
    scores = []
    for i, content in enumerate(documents):
        doc_path = tmp_path / f"doc_{i}.txt"
        doc_path.write_text(content)
        report = validator.validate_document(str(doc_path))
        scores.append(report.confidence_score)
    
    # Complete document should have highest score
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    
    # Document without temporal markers should have lower score
    assert scores[1] < 0.9
    
    # Document without metrics should have lower score
    assert scores[2] < 0.9
