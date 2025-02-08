"""
Natural text analysis service.

Discovers structural elements in text without enforcing rigid patterns,
allowing natural organization and meaning to emerge.
"""

from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
from collections import defaultdict

from src.core.analysis.structure_analysis import StructuralElement, StructureContext

@dataclass
class TextSegment:
    """Natural text segment discovered during analysis."""
    content: str
    segment_type: str
    start_pos: int
    end_pos: int
    depth: int = 0
    coherence_score: float = 0.0
    
    @property
    def length(self) -> int:
        """Natural segment length."""
        return self.end_pos - self.start_pos

class TextAnalyzer:
    """Natural text analyzer that discovers structure without forcing patterns."""
    
    def __init__(self):
        # Natural break patterns (not enforced, just observed)
        self.potential_breaks = {
            'paragraph': r'\n\s*\n+',
            'section': r'\n\s*#{1,6}\s',
            'list': r'\n\s*[-*]\s',
            'code': r'```\w*\n',
        }
        
        # Coherence thresholds (learned from observation)
        self.min_segment_length = 10
        self.max_segment_length = 5000
        self.coherence_threshold = 0.3
        
    async def analyze_text(
        self,
        content: str,
        context: StructureContext
    ) -> List[StructuralElement]:
        """
        Analyze text content naturally.
        
        Args:
            content: Text content to analyze
            context: Analysis context
            
        Returns:
            List of discovered structural elements
        """
        # Discover initial segments
        segments = await self._discover_segments(content, context)
        
        # Observe relationships and coherence
        segments = self._observe_coherence(segments)
        
        # Convert to structural elements
        elements = []
        for segment in segments:
            # Calculate natural density based on coherence and relationships
            density = self._calculate_segment_density(segment, segments)
            
            # Create structural element
            element = StructuralElement(
                element_id=f"text_{segment.start_pos}_{segment.end_pos}",
                element_type=segment.segment_type,
                content=segment.content,
                metadata={
                    "start_pos": segment.start_pos,
                    "end_pos": segment.end_pos,
                    "depth": segment.depth,
                    "coherence": segment.coherence_score
                },
                density=density,
                emergence_time=context.start_time
            )
            elements.append(element)
            
        return elements
    
    async def _discover_segments(
        self,
        content: str,
        context: StructureContext
    ) -> List[TextSegment]:
        """Discover natural text segments."""
        segments = []
        
        # First pass: Find potential break points
        break_points = self._find_break_points(content)
        
        # Second pass: Create initial segments
        current_pos = 0
        for pos, break_type in sorted(break_points.items()):
            if pos - current_pos < self.min_segment_length:
                continue
                
            segment = TextSegment(
                content=content[current_pos:pos].strip(),
                segment_type=break_type,
                start_pos=current_pos,
                end_pos=pos
            )
            
            if self._is_valid_segment(segment):
                segments.append(segment)
            current_pos = pos
            
        # Add final segment if needed
        if current_pos < len(content):
            final_segment = TextSegment(
                content=content[current_pos:].strip(),
                segment_type='text',
                start_pos=current_pos,
                end_pos=len(content)
            )
            if self._is_valid_segment(final_segment):
                segments.append(final_segment)
        
        return segments
    
    def _find_break_points(self, content: str) -> Dict[int, str]:
        """Find natural break points in text."""
        break_points = {}
        
        for break_type, pattern in self.potential_breaks.items():
            for match in re.finditer(pattern, content):
                break_points[match.start()] = break_type
                
        return break_points
    
    def _is_valid_segment(self, segment: TextSegment) -> bool:
        """Check if segment is naturally valid."""
        return (
            segment.length >= self.min_segment_length and
            segment.length <= self.max_segment_length and
            len(segment.content.strip()) > 0
        )
    
    def _observe_coherence(
        self,
        segments: List[TextSegment]
    ) -> List[TextSegment]:
        """Observe natural coherence between segments."""
        for i, segment in enumerate(segments):
            # Calculate coherence based on surrounding segments
            prev_coherence = self._calculate_coherence(
                segment,
                segments[i-1] if i > 0 else None
            )
            next_coherence = self._calculate_coherence(
                segment,
                segments[i+1] if i < len(segments)-1 else None
            )
            
            # Natural coherence score
            segment.coherence_score = (prev_coherence + next_coherence) / 2
            
            # Determine depth based on coherence
            segment.depth = self._determine_depth(segment, segments[:i])
            
        return segments
    
    def _calculate_coherence(
        self,
        segment1: TextSegment,
        segment2: Optional[TextSegment]
    ) -> float:
        """Calculate natural coherence between segments."""
        if not segment2:
            return 0.5  # Base coherence for edges
            
        # Consider multiple factors
        type_coherence = float(segment1.segment_type == segment2.segment_type)
        
        # Length similarity
        length_ratio = min(
            segment1.length,
            segment2.length
        ) / max(segment1.length, segment2.length)
        
        # Content similarity (simple for now)
        common_words = len(
            set(segment1.content.lower().split()) &
            set(segment2.content.lower().split())
        )
        total_words = len(set(segment1.content.lower().split()) |
                         set(segment2.content.lower().split()))
        content_similarity = common_words / max(1, total_words)
        
        return (
            0.4 * type_coherence +
            0.3 * length_ratio +
            0.3 * content_similarity
        )
    
    def _determine_depth(
        self,
        segment: TextSegment,
        previous_segments: List[TextSegment]
    ) -> int:
        """Determine natural depth of segment."""
        depth = 0
        
        # Consider header patterns
        if segment.segment_type == 'section':
            header_match = re.match(r'#+', segment.content.lstrip())
            if header_match:
                depth = len(header_match.group(0))
                return depth
        
        # Consider list patterns
        if segment.segment_type == 'list':
            indent_match = re.match(r'^\s*', segment.content)
            if indent_match:
                depth = 1 + (len(indent_match.group(0)) // 2)
                return depth
        
        # Consider indentation for other types
        indent_match = re.match(r'^\s+', segment.content)
        if indent_match:
            depth = 1 + (len(indent_match.group(0)) // 2)
        
        # Consider previous context
        if previous_segments:
            prev_segment = previous_segments[-1]
            if prev_segment.segment_type == 'section':
                depth = max(depth, prev_segment.depth + 1)
            elif self._calculate_coherence(segment, prev_segment) > 0.7:
                depth = prev_segment.depth
        
        return max(1, depth)  # Ensure minimum depth of 1
    
    def _calculate_segment_density(
        self,
        segment: TextSegment,
        all_segments: List[TextSegment]
    ) -> float:
        """Calculate natural density of text segment."""
        if not all_segments:
            return 0.0
            
        # Content density - richer content has higher density
        words = segment.content.split()
        unique_words = len(set(words))
        content_density = unique_words / max(len(words), 1)
        
        # Structure density - well-structured content has higher density
        structure_patterns = [
            r'\b\w+[.!?]\s',  # Sentences
            r'\n\s*[-*]\s',   # List items
            r'\b\w+:\s',      # Definitions/labels
            r'\n\s*\d+\.\s',  # Numbered items
            r'\n\s*#{1,6}\s',  # Headers
            r'`[^`]+`',        # Code snippets
            r'\[.+?\]\(.+?\)'  # Links
        ]
        structure_matches = sum(
            len(re.findall(pattern, segment.content))
            for pattern in structure_patterns
        )
        structure_density = min(1.0, structure_matches / max(len(words) / 5, 1))
        
        # Type-based density boost
        type_boost = {
            'section': 0.8,  # Headers are naturally dense
            'list': 0.7,     # Lists are structured
            'code': 0.7,     # Code blocks are dense
            'paragraph': 0.5  # Base density for paragraphs
        }.get(segment.segment_type, 0.3)
        
        # Relationship density - more coherent relationships = higher density
        coherent_relations = sum(
            1 for s in all_segments
            if s != segment and self._calculate_coherence(segment, s) > 0.5
        )
        relationship_density = coherent_relations / max(len(all_segments) - 1, 1)
        
        # Depth contribution - deeper segments often have higher density
        depth_factor = min(1.0, segment.depth / 2)
        
        # Calculate base density
        base_density = (
            0.25 * content_density +
            0.25 * structure_density +
            0.25 * relationship_density +
            0.25 * depth_factor
        )
        
        # Apply type boost
        return min(1.0, base_density + (1 - base_density) * type_boost)
