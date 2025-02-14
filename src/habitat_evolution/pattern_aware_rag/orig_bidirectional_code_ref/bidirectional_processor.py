"""
Bidirectional processor for structure-meaning relationships in Habitat.
Handles bidirectional influence between structural and semantic aspects.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

# Use relative import for mock settings
from .tests.config.mock_settings import MockSettings
from .mock_state_processor import MockStateProcessor
from .mock_coherence import CoherenceChecker
from .logging_manager import LoggingManager
from .models.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Metrics for processing evaluation."""
    
    structural_score: float = 0.0
    semantic_score: float = 0.0
    influence_score: float = 0.0
    adaptation_score: float = 0.0
    evolution_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for logging."""
        return {
            "structural_score": self.structural_score,
            "semantic_score": self.semantic_score,
            "influence_score": self.influence_score,
            "adaptation_score": self.adaptation_score,
            "evolution_score": self.evolution_score
        }

class BidirectionalProcessor:
    """Processes documents with bidirectional structure-meaning influence."""
    
    def __init__(self, settings: MockSettings):
        """Initialize processor with settings."""
        self.settings = settings
        self.state_processor = MockStateProcessor()
        self.coherence_checker = CoherenceChecker()
        self.metrics = ProcessingMetrics()
        self.logger = LoggingManager("bidirectional_processor")
        
        # Thresholds
        self.STRUCTURE_THRESHOLD = 0.3
        self.MEANING_THRESHOLD = 0.3
        self.INFLUENCE_THRESHOLD = 0.3
        self.EVOLUTION_THRESHOLD = 0.1
    
    async def process_structure(self, document: Document) -> Dict[str, Any]:
        """Process structural elements with meaning awareness."""
        try:
            content = document.content
            if not content:
                raise ValueError("Document has no content")
            
            # Extract basic structure
            hierarchy = self._extract_hierarchy(content)
            relationships = self._extract_relationships(content)
            cross_refs = self._extract_cross_references(content)
            
            # Calculate structural score
            structural_score = await self.coherence_checker.check_structural_coherence(
                hierarchy,
                relationships
            )
            
            # Calculate influence on meaning
            influence_score = self._calculate_structure_influence(
                hierarchy,
                relationships,
                cross_refs
            )
            
            # Update metrics
            self.metrics.structural_score = structural_score
            self.metrics.influence_score = influence_score
            
            return {
                "score": structural_score,
                "elements": [
                    "hierarchy",
                    "relationships",
                    "cross_references"
                ] if all([hierarchy, relationships, cross_refs]) else [],
                "influence": influence_score,
                "data": {
                    "hierarchy": hierarchy,
                    "relationships": relationships,
                    "cross_references": cross_refs
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in structure processing: {str(e)}")
            raise
    
    async def process_meaning(self, document: Document) -> Dict[str, Any]:
        """Process meaning with structural context."""
        try:
            content = document.content
            if not content:
                raise ValueError("Document has no content")
            
            # Extract semantic elements
            domains = self._extract_domains(content)
            temporal = self._extract_temporal_context(content)
            impacts = self._extract_impact_relationships(content)
            
            # Calculate semantic score
            semantic_score = await self.coherence_checker.check_semantic_coherence(
                domains,
                impacts
            )
            
            # Calculate influence on structure
            influence_score = self._calculate_meaning_influence(
                domains,
                impacts,
                temporal
            )
            
            # Update metrics
            self.metrics.semantic_score = semantic_score
            self.metrics.influence_score = influence_score
            
            return {
                "score": semantic_score,
                "elements": [
                    "domain_context",
                    "temporal_context",
                    "impact_relationships"
                ] if all([domains, temporal, impacts]) else [],
                "influence": influence_score,
                "data": {
                    "domains": domains,
                    "temporal": temporal,
                    "impacts": impacts
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in meaning processing: {str(e)}")
            raise
    
    async def adapt_rag(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt RAG based on bidirectional state."""
        try:
            # Extract context
            structure = context.get("structure", {})
            meaning = context.get("meaning", {})
            
            # Enhance query
            enhanced_query = self._enhance_query(
                query,
                structure["data"],
                meaning["data"]
            )
            
            # Calculate adaptation score
            adaptation_score = (
                structure["score"] +
                meaning["score"] +
                self.metrics.influence_score
            ) / 3
            
            # Calculate evolution
            evolution = adaptation_score - max(
                self.metrics.adaptation_score,
                self.EVOLUTION_THRESHOLD
            )
            
            # Update metrics
            self.metrics.adaptation_score = adaptation_score
            self.metrics.evolution_score = evolution
            
            return {
                "score": adaptation_score,
                "adaptations": [
                    "query_enhancement",
                    "context_integration",
                    "coherence_preservation"
                ],
                "evolution": evolution,
                "enhanced_query": enhanced_query
            }
            
        except Exception as e:
            self.logger.error(f"Error in RAG adaptation: {str(e)}")
            raise
    
    def _extract_hierarchy(self, content: str) -> List[Dict[str, Any]]:
        """Extract hierarchical structure from content."""
        hierarchy = []
        current_section = None
        
        for line in content.split('\n'):
            if line.strip() and any(c.isdigit() for c in line):
                level = len(line) - len(line.lstrip())
                hierarchy.append({
                    "level": level,
                    "content": line.strip(),
                    "parent": current_section
                })
                current_section = line.strip()
        
        return hierarchy
    
    def _extract_relationships(self, content: str) -> List[Dict[str, Any]]:
        """Extract relationships from content."""
        relationships = []
        in_relationships = False
        
        for line in content.split('\n'):
            if "RELATIONSHIP NETWORKS" in line:
                in_relationships = True
            elif in_relationships and "→" in line:
                source, target = line.split("→")
                relationships.append({
                    "source": source.strip(),
                    "target": target.strip(),
                    "type": "direct"
                })
            elif in_relationships and "-" in line:
                # Capture implicit relationships
                parts = line.split("-")
                if len(parts) == 2:
                    relationships.append({
                        "source": parts[0].strip(),
                        "target": parts[1].strip(),
                        "type": "implicit"
                    })
        
        return relationships
    
    def _extract_cross_references(self, content: str) -> List[str]:
        """Extract cross-references from content."""
        refs = []
        for line in content.split('\n'):
            if "Cross-References:" in line:
                _, refs_str = line.split(":")
                refs = [r.strip() for r in refs_str.split(",")]
                break
        return refs
    
    def _extract_domains(self, content: str) -> List[Dict[str, Any]]:
        """Extract knowledge domains and their relationships."""
        domains = []
        in_domains = False
        
        for line in content.split('\n'):
            if "KNOWLEDGE DOMAIN RELATIONSHIPS" in line:
                in_domains = True
            elif in_domains and ":" in line:
                domain_type, values = line.split(":")
                domains.extend([
                    {
                        "type": domain_type.strip(),
                        "name": v.strip(),
                        "level": "primary" if "Primary" in domain_type else "related"
                    }
                    for v in values.split(",")
                ])
            elif in_domains and line.strip() == "":
                in_domains = False
        
        return domains
    
    def _extract_temporal_context(self, content: str) -> Dict[str, Any]:
        """Extract temporal context and relationships."""
        # Implementation would extract dates, periods, and temporal relationships
        return {
            "type": "temporal_context",
            "periods": [],  # Would contain actual temporal periods
            "relationships": []  # Would contain temporal relationships
        }
    
    def _extract_impact_relationships(self, content: str) -> List[Dict[str, Any]]:
        """Extract impact relationships from content."""
        impacts = []
        in_impacts = False
        
        for line in content.split('\n'):
            if "STRUCTURE-MEANING RELATIONSHIPS" in line:
                in_impacts = True
            elif in_impacts and "→" in line:
                source, target = line.split("→")
                impacts.append({
                    "source": source.strip(),
                    "target": target.strip(),
                    "type": "impact"
                })
        
        return impacts
    
    def _enhance_with_structure(
        self,
        domains: List[Dict[str, Any]],
        hierarchy: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance domains with structural context."""
        enhanced = []
        for domain in domains:
            # Find related structural elements
            related_elements = [
                h for h in hierarchy
                if domain["name"].lower() in h["content"].lower()
            ]
            enhanced.append({
                **domain,
                "structural_context": [
                    {
                        "level": e["level"],
                        "content": e["content"]
                    }
                    for e in related_elements
                ]
            })
        return enhanced
    
    def _enhance_relationships(
        self,
        impacts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance impact relationships with structural relationships."""
        enhanced = []
        for impact in impacts:
            # Find related structural relationships
            related_rels = [
                r for r in relationships
                if impact["source"] in r["source"] or impact["target"] in r["target"]
            ]
            enhanced.append({
                **impact,
                "structural_context": related_rels
            })
        return enhanced
    
    def _enhance_query(
        self,
        query: str,
        structure: Dict[str, Any],
        meaning: Dict[str, Any]
    ) -> str:
        """Enhance query using structure and meaning context."""
        # Implementation would enhance query based on context
        enhanced_terms = []
        
        # Add structural context
        if "hierarchy" in structure:
            relevant_sections = [
                h["content"] for h in structure["hierarchy"]
                if any(term.lower() in h["content"].lower() 
                      for term in query.split())
            ]
            enhanced_terms.extend(relevant_sections)
        
        # Add semantic context
        if "domains" in meaning:
            relevant_domains = [
                d["name"] for d in meaning["domains"]
                if any(term.lower() in d["name"].lower() 
                      for term in query.split())
            ]
            enhanced_terms.extend(relevant_domains)
        
        # Combine original query with enhancements
        return f"{query} {' '.join(enhanced_terms)}"
    
    def _calculate_structure_influence(
        self,
        hierarchy: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        cross_refs: List[str]
    ) -> float:
        """Calculate structure's influence on meaning."""
        # Implementation would calculate actual influence score
        base_score = 0.3
        
        if hierarchy:
            base_score += 0.1
        if relationships:
            base_score += 0.1
        if cross_refs:
            base_score += 0.1
            
        return min(base_score, 1.0)
    
    def _calculate_meaning_influence(
        self,
        domains: List[Dict[str, Any]],
        impacts: List[Dict[str, Any]],
        temporal: Dict[str, Any]
    ) -> float:
        """Calculate meaning's influence on structure."""
        # Implementation would calculate actual influence score
        base_score = 0.3
        
        if domains:
            base_score += 0.1
        if impacts:
            base_score += 0.1
        if temporal and (temporal["periods"] or temporal["relationships"]):
            base_score += 0.1
            
        return min(base_score, 1.0)


"""
Bidirectional processor for testing purposes.
Uses mock implementations for testing.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from .mock_coherence import CoherenceChecker, CoherenceThresholds
from .mock_observable import MockObservable
from .mock_adaptive_id import MockAdaptiveID

class MockBidirectionalProcessor:
    """Mock bidirectional processor for testing."""
    
    def __init__(self):
        """Initialize processor with mock components."""
        self.coherence_checker = CoherenceChecker()
        self.observable = MockObservable()
        self.processing_history = []
        
    async def process_structure(self, data: Dict[str, Any], adaptive_id: Optional[MockAdaptiveID] = None) -> Dict[str, Any]:
        """Process structural aspects of data."""
        result = {
            "structure_score": 0.8,
            "structure_analysis": {
                "completeness": 0.85,
                "consistency": 0.75,
                "validity": 0.80
            }
        }
        
        # Track processing
        self.processing_history.append({
            "type": "structure",
            "data": data,
            "result": result
        })
        
        return result
        
    async def process_meaning(self, data: Dict[str, Any], adaptive_id: Optional[MockAdaptiveID] = None) -> Dict[str, Any]:
        """Process semantic meaning of data."""
        result = {
            "meaning_score": 0.85,
            "semantic_analysis": {
                "relevance": 0.9,
                "coherence": 0.8,
                "context_alignment": 0.85
            }
        }
        
        # Track processing
        self.processing_history.append({
            "type": "meaning",
            "data": data,
            "result": result
        })
        
        return result
        
    async def process_bidirectional(self, data: Dict[str, Any], adaptive_id: Optional[MockAdaptiveID] = None) -> Dict[str, Any]:
        """Process both structure and meaning with bidirectional influence."""
        structure_result = await self.process_structure(data, adaptive_id)
        meaning_result = await self.process_meaning(data, adaptive_id)
        
        # Combine results
        combined_result = {
            "structure": structure_result,
            "meaning": meaning_result,
            "overall_score": (structure_result["structure_score"] + meaning_result["meaning_score"]) / 2
        }
        
        # Update visualization
        self.observable.update_cell("bidirectional_result", combined_result)
        
        return combined_result
        
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get history of processing operations."""
        return self.processing_history
