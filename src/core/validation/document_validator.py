"""Climate risk document validator."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import re
from pathlib import Path

@dataclass
class DocumentSection:
    """Represents a section in the climate risk document."""
    name: str
    content: str
    start_line: int
    end_line: int
    temporal_markers: List[str] = None
    metrics: List[Dict] = None

@dataclass
class ValidationReport:
    """Validation report for document processing."""
    is_valid: bool
    sections: List[DocumentSection]
    temporal_range: tuple
    confidence_score: float
    errors: List[str]
    warnings: List[str]
    timestamp: datetime = None

class ClimateDocumentValidator:
    """Validates climate risk document structure and content."""
    
    def __init__(self):
        self.required_sections = [
            "introduction",
            "methodology",
            "findings",
            "recommendations"
        ]
        
        self.temporal_patterns = [
            r'\d{4}-\d{4}',  # Year range (e.g., 2020-2050)
            r'\d{4}',        # Single year
            r'(?:mid|late)-century',  # Century references
            r'(?:short|medium|long)-term',  # Term references
            r'\d{4}-\d{2}'  # Year-month
        ]
        
        self.metric_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:percent|%)',  # Percentages
            r'(\d+(?:\.\d+)?)\s*(?:degrees?|°)',  # Temperature
            r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m)',    # Precipitation
            r'(\d+(?:\.\d+)?)\s*(?:days?|months?|years?)'  # Time periods
        ]
    
    def validate_document(self, doc_path: str) -> ValidationReport:
        """Validate climate risk document."""
        try:
            content = Path(doc_path).read_text()
        except Exception as e:
            return ValidationReport(
                is_valid=False,
                sections=[],
                temporal_range=None,
                confidence_score=0.0,
                errors=[f"Failed to read document: {str(e)}"],
                warnings=[],
                timestamp=datetime.utcnow()
            )
        
        # Extract and validate sections
        sections = self._extract_sections(content)
        if not sections:
            return ValidationReport(
                is_valid=False,
                sections=[],
                temporal_range=None,
                confidence_score=0.0,
                errors=["No valid sections found"],
                warnings=[],
                timestamp=datetime.utcnow()
            )
        
        # Validate required sections
        missing_sections = [
            section for section in self.required_sections
            if not any(s.name.lower() == section for s in sections)
        ]
        
        # Extract temporal markers
        temporal_range = self._extract_temporal_range(sections)
        
        # Calculate confidence score
        errors = []
        warnings = []
        
        if missing_sections:
            warnings.append(f"Missing sections: {', '.join(missing_sections)}")
            
        confidence_score = self._calculate_confidence(
            sections=sections,
            missing_sections=missing_sections,
            temporal_range=temporal_range
        )
        
        return ValidationReport(
            is_valid=confidence_score >= 0.7,
            sections=sections,
            temporal_range=temporal_range,
            confidence_score=confidence_score,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.utcnow()
        )
    
    def _extract_sections(self, content: str) -> List[DocumentSection]:
        """Extract document sections."""
        lines = content.split('\n')
        sections = []
        current_section = None
        section_content = []
        start_line = 0
        
        for i, line in enumerate(lines):
            # Check for section header
            if line.strip() and (line.isupper() or line.startswith('#')):
                # Save previous section
                if current_section:
                    sections.append(DocumentSection(
                        name=current_section,
                        content='\n'.join(section_content),
                        start_line=start_line,
                        end_line=i-1,
                        temporal_markers=self._find_temporal_markers('\n'.join(section_content)),
                        metrics=self._extract_metrics('\n'.join(section_content))
                    ))
                
                # Start new section
                current_section = line.strip('#').strip()
                section_content = []
                start_line = i
            elif current_section:
                section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            sections.append(DocumentSection(
                name=current_section,
                content='\n'.join(section_content),
                start_line=start_line,
                end_line=len(lines)-1,
                temporal_markers=self._find_temporal_markers('\n'.join(section_content)),
                metrics=self._extract_metrics('\n'.join(section_content))
            ))
        
        return sections
    
    def _find_temporal_markers(self, content: str) -> List[str]:
        """Find temporal markers in content."""
        markers = []
        
        # Find year ranges (e.g., 2020-2050)
        year_ranges = re.finditer(r'\b(\d{4})[-–](\d{4})\b', content)
        markers.extend(f"{m.group(1)}-{m.group(2)}" for m in year_ranges)
        
        # Find single years
        single_years = re.finditer(r'\b\d{4}\b(?![-–]\d{4})', content)
        markers.extend(m.group(0) for m in single_years)
        
        # Find special references
        special_refs = re.finditer(r'(?:mid|late)-century|(?:short|medium|long)-term', content)
        markers.extend(m.group(0) for m in special_refs)
        
        return sorted(set(markers))
    
    def _extract_metrics(self, content: str) -> List[Dict]:
        """Extract metrics from content."""
        metrics = []
        for pattern in self.metric_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                context = content[max(0, match.start()-50):min(len(content), match.end()+50)]
                metrics.append({
                    'value': float(match.group(1)),
                    'unit': match.group(0)[len(match.group(1)):].strip(),
                    'context': context.strip()
                })
        return metrics
    
    def _extract_temporal_range(self, sections: List[DocumentSection]) -> Optional[tuple]:
        """Extract document temporal range."""
        all_markers = []
        for section in sections:
            if section.temporal_markers:
                all_markers.extend(section.temporal_markers)
                
        if not all_markers:
            return None
            
        # Convert to years where possible
        years = []
        for marker in all_markers:
            # Handle year ranges (e.g., 2020-2050)
            if re.match(r'\d{4}-\d{4}', marker):
                start, end = marker.split('-')
                years.extend([int(start), int(end)])
            # Handle single years
            elif re.match(r'\d{4}', marker):
                years.append(int(marker))
            # Handle special cases
            elif 'mid-century' in marker:
                years.append(2050)
            elif 'late-century' in marker:
                years.append(2100)
            elif 'short-term' in marker:
                current_year = datetime.utcnow().year
                years.extend([current_year, current_year + 5])
            elif 'medium-term' in marker:
                current_year = datetime.utcnow().year
                years.extend([current_year, current_year + 10])
            elif 'long-term' in marker:
                current_year = datetime.utcnow().year
                years.extend([current_year, current_year + 20])
                
        if years:
            return (min(years), max(years))
        return None
    
    def _calculate_confidence(self, sections: List[DocumentSection], 
                            missing_sections: List[str],
                            temporal_range: Optional[tuple]) -> float:
        """Calculate validation confidence score."""
        score = 1.0
        
        # Penalize for missing sections
        if missing_sections:
            score *= (1 - 0.15 * len(missing_sections))  # Increased penalty
        
        # Check temporal coverage
        if not temporal_range:
            score *= 0.7  # More severe penalty
        
        # Check metric presence
        total_metrics = sum(1 for s in sections if s.metrics)
        if total_metrics < len(sections) / 2:
            score *= 0.8  # More severe penalty
        elif total_metrics == 0:
            score *= 0.6  # Even more severe for no metrics
        
        # Additional penalties
        if len(sections) < 3:  # Too few sections
            score *= 0.7
        
        # Check metric density
        total_metric_count = sum(len(s.metrics) if s.metrics else 0 for s in sections)
        if total_metric_count < 5:  # Need at least 5 metrics
            score *= 0.85
        
        return max(0.0, min(1.0, score))
