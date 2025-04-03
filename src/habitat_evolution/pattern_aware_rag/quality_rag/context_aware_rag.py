"""
Context-aware RAG with quality assessment paths.

This module provides the ContextAwareRAG class which extends PatternAwareRAG
with context-aware pattern extraction and quality assessment capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG
from src.habitat_evolution.adaptive_core.emergence.context_aware_extraction.context_aware_extractor import ContextAwareExtractor
from src.habitat_evolution.adaptive_core.emergence.context_aware_extraction.quality_assessment import QualityAssessment
from src.habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext
from src.habitat_evolution.adaptive_core.models import Pattern, Relationship
from src.habitat_evolution.adaptive_core.persistence.interfaces.repository_adapter import PatternRepository
from src.habitat_evolution.core.pattern import PatternState

from .quality_enhanced_retrieval import QualityEnhancedRetrieval, RetrievalResult

logger = logging.getLogger(__name__)

class ContextAwareRAG:
    """Context-aware RAG with quality assessment paths.
    
    This class implements context-aware pattern extraction and quality assessment
    capabilities, creating a self-reinforcing feedback mechanism that improves
    pattern extraction and retrieval capabilities over time.
    """
    
    def __init__(
        self,
        pattern_repository: PatternRepository,
        window_sizes: List[int] = None,
        quality_threshold: float = 0.7,
        quality_weight: float = 0.7,
        coherence_threshold: float = 0.6,
        data_dir: Optional[Path] = None
    ):
        """Initialize context-aware RAG.
        
        Args:
            pattern_repository: Repository for storing and retrieving patterns
            window_sizes: List of window sizes for context-aware extraction
            quality_threshold: Threshold for "good" quality state
            quality_weight: Weight to give to quality in ranking
            coherence_threshold: Threshold for coherence filtering
            data_dir: Optional directory for climate risk data
        """
        # Store the pattern repository
        self.pattern_repository = pattern_repository
        
        # Initialize context-aware extraction components
        self.context_aware_extractor = ContextAwareExtractor(
            window_sizes=window_sizes,
            quality_threshold=quality_threshold,
            data_dir=data_dir
        )
        
        # Initialize quality-enhanced retrieval
        self.quality_retrieval = QualityEnhancedRetrieval(
            quality_weight=quality_weight,
            coherence_threshold=coherence_threshold
        )
        
        logger.info(f"Initialized ContextAwareRAG with quality_threshold={quality_threshold}")
    
    def process_with_context_aware_patterns(
        self, 
        query: str, 
        document: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process query with context-aware pattern extraction.
        
        Args:
            query: Query string
            document: Document text to process
            context: Optional context dictionary
            
        Returns:
            Dictionary with processing results
        """
        # Extract context-aware patterns from document
        extraction_results = self.context_aware_extractor.process_document(document)
        
        # Create quality-aware context
        quality_context = QualityAwarePatternContext()
        
        # Update context from extraction results
        quality_context.context_aware_extraction_results = extraction_results
        quality_context.update_from_quality_assessment(self.context_aware_extractor.quality_assessor)
        
        # Retrieve with quality awareness
        retrieval_result = self.quality_retrieval.retrieve_with_quality(
            query=query,
            context=quality_context
        )
        
        # Store high-quality patterns in repository
        self._store_high_quality_patterns(quality_context)
        
        # Generate response with quality-aware patterns
        response = self._generate_response_with_quality(
            query=query,
            retrieval_result=retrieval_result,
            quality_context=quality_context
        )
        
        return {
            "query": query,
            "response": response,
            "quality_context": quality_context.to_dict(),
            "retrieval_result": {
                "patterns": [p.to_dict() for p in retrieval_result.patterns],
                "quality_distribution": retrieval_result.quality_distribution,
                "confidence": retrieval_result.confidence,
                "retrieval_explanation": retrieval_result.retrieval_explanation
            },
            "extraction_results": {
                "entities_count": len(extraction_results["entities"]),
                "relationships_count": len(extraction_results["relationships"]),
                "quality_summary": extraction_results["quality_summary"]
            }
        }
    
    def process_document_for_patterns(self, document: str) -> Dict[str, Any]:
        """Process a document to extract context-aware patterns.
        
        Args:
            document: Document text to process
            
        Returns:
            Dictionary with extraction results
        """
        # Extract context-aware patterns from document
        extraction_results = self.context_aware_extractor.process_document(document)
        
        # Create quality-aware context
        quality_context = QualityAwarePatternContext()
        
        # Update context from extraction results
        quality_context.context_aware_extraction_results = extraction_results
        quality_context.update_from_quality_assessment(self.context_aware_extractor.quality_assessor)
        
        # Store high-quality patterns in repository
        stored_patterns = self._store_high_quality_patterns(quality_context)
        
        return {
            "extraction_results": extraction_results,
            "quality_context": quality_context.to_dict(),
            "stored_patterns_count": len(stored_patterns)
        }
    
    def _store_high_quality_patterns(self, quality_context: QualityAwarePatternContext) -> List[Pattern]:
        """Store high-quality patterns in the repository.
        
        Args:
            quality_context: Quality-aware pattern context
            
        Returns:
            List of stored patterns
        """
        # Get high-quality patterns
        high_quality_patterns = quality_context.get_high_quality_patterns()
        
        # Store in repository
        stored_patterns = []
        for pattern in high_quality_patterns:
            # Check if pattern already exists
            existing = self.pattern_repository.find_by_text(pattern.base_concept)
            
            if existing:
                # Update existing pattern with new quality information
                existing.properties.update(pattern.properties)
                self.pattern_repository.update(existing)
                stored_patterns.append(existing)
                logger.info(f"Updated existing pattern '{pattern.base_concept}' with new quality information")
            else:
                # Store new pattern
                self.pattern_repository.store(pattern)
                stored_patterns.append(pattern)
                logger.info(f"Stored new high-quality pattern '{pattern.base_concept}'")
        
        return stored_patterns
    
    def _generate_response_with_quality(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        quality_context: QualityAwarePatternContext
    ) -> str:
        """Generate response with quality-aware patterns.
        
        Args:
            query: Query string
            retrieval_result: Result of quality-enhanced retrieval
            quality_context: Quality-aware pattern context
            
        Returns:
            Generated response
        """
        # In a real implementation, this would use an LLM to generate a response
        # Here we'll create a simple template-based response
        
        patterns_text = ", ".join(p.text for p in retrieval_result.patterns[:5])
        confidence = retrieval_result.confidence
        
        response = f"Based on the context-aware patterns extracted ({patterns_text}), "
        
        if confidence > 0.8:
            response += "I can provide a high-confidence answer to your query."
        elif confidence > 0.5:
            response += "I can provide a moderately confident answer to your query."
        else:
            response += "I can provide a tentative answer based on emerging patterns."
        
        response += f"\n\nConfidence: {confidence:.2f}"
        response += f"\nQuality distribution: Good={retrieval_result.quality_distribution['good']}, "
        response += f"Uncertain={retrieval_result.quality_distribution['uncertain']}, "
        response += f"Poor={retrieval_result.quality_distribution['poor']}"
        
        return response
    
    def analyze_quality_evolution(self, document_ids: List[str]) -> Dict[str, Any]:
        """Analyze quality evolution across multiple documents.
        
        Args:
            document_ids: List of document IDs to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # In a real implementation, this would load documents from a repository
        # Here we'll return a placeholder for demonstration
        
        return {
            "documents_analyzed": len(document_ids),
            "quality_evolution": {
                "improvement_ratio": 0.75,
                "pattern_quality_trend": "improving",
                "coherence_trend": "stable",
                "emergence_rate": "high"
            },
            "pattern_state_transitions": {
                "EMERGENT_to_COHERENT": 12,
                "COHERENT_to_STABLE": 8,
                "DEGRADING_to_EMERGENT": 3
            }
        }
    
    def get_quality_assessment_path(self, entity: str) -> Dict[str, Any]:
        """Get quality assessment path for an entity.
        
        Args:
            entity: Entity to get path for
            
        Returns:
            Dictionary with quality assessment path
        """
        # Get transitions from quality assessor
        transitions = self.context_aware_extractor.quality_assessor.get_transitions(entity)
        
        if not transitions:
            return {"entity": entity, "has_path": False}
        
        # Get current quality state
        quality_states = self.context_aware_extractor.quality_assessor.get_quality_states()
        current_state = None
        
        for state, entities in quality_states.items():
            if entity in entities:
                current_state = state
                break
        
        return {
            "entity": entity,
            "has_path": True,
            "transitions": transitions,
            "current_state": current_state,
            "transition_count": len(transitions)
        }
