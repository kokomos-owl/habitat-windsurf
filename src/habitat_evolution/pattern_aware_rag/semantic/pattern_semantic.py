"""
Semantic content enhancement for patterns in the Habitat Evolution framework.

This module provides functionality to extract, enhance, and manage semantic content
for patterns, enabling direct semantic space representation without abstractions.
"""

import json
import re
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime

# Common English stopwords to filter out from keywords
STOPWORDS = {
    "the", "and", "a", "an", "in", "on", "at", "to", "for", "with", "by", 
    "of", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "that", "this", "these",
    "those", "it", "its", "they", "them", "their", "we", "us", "our", "you",
    "your", "he", "him", "his", "she", "her", "hers"
}

class PatternSemanticEnhancer:
    """
    Enhances patterns with semantic content capabilities.
    
    This class provides methods to extract semantic content and keywords from
    different pattern implementations, ensuring a consistent semantic interface
    regardless of the underlying pattern class.
    """
    
    @staticmethod
    def get_semantic_content(pattern: Any) -> str:
        """
        Extract semantic content from a pattern.
        
        Works with Pattern, PatternAdaptiveID, or any object with semantic properties.
        
        Args:
            pattern: Any pattern-like object
            
        Returns:
            Extracted semantic content as string
        """
        # If pattern has direct semantic_content attribute
        if hasattr(pattern, 'semantic_content') and pattern.semantic_content:
            return pattern.semantic_content
            
        # If pattern has base_concept (from Pattern class)
        if hasattr(pattern, 'base_concept') and pattern.base_concept:
            return pattern.base_concept
            
        # If pattern has hazard_type (from PatternAdaptiveID)
        if hasattr(pattern, 'hazard_type') and pattern.hazard_type:
            if hasattr(pattern, 'pattern_type') and pattern.pattern_type:
                return f"{pattern.pattern_type} pattern for {pattern.hazard_type}"
            return f"Pattern for {pattern.hazard_type}"
            
        # If pattern has text_fragments
        if hasattr(pattern, 'text_fragments') and pattern.text_fragments:
            if isinstance(pattern.text_fragments, list):
                return " ".join(pattern.text_fragments)
            return str(pattern.text_fragments)
            
        # If pattern has adaptive_id
        if hasattr(pattern, 'adaptive_id'):
            adaptive_id = pattern.adaptive_id
            if hasattr(adaptive_id, 'base_concept') and adaptive_id.base_concept:
                return adaptive_id.base_concept
                
        # Default fallback
        if hasattr(pattern, 'id'):
            return f"Pattern {pattern.id}"
        return "Unnamed pattern"
    
    @staticmethod
    def get_keywords(pattern: Any, max_keywords: int = 5) -> List[str]:
        """
        Extract keywords from a pattern.
        
        Args:
            pattern: Any pattern-like object
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        # If pattern already has keywords
        if hasattr(pattern, 'keywords') and pattern.keywords:
            if isinstance(pattern.keywords, list):
                return pattern.keywords[:max_keywords]
            elif isinstance(pattern.keywords, str):
                return [pattern.keywords]
        
        # If pattern has temporal_context (from Pattern class)
        if hasattr(pattern, 'temporal_context') and pattern.temporal_context:
            if isinstance(pattern.temporal_context, dict):
                # Extract keys from temporal context
                return list(pattern.temporal_context.keys())[:max_keywords]
            
        # If pattern has adaptive_id with temporal_context
        if hasattr(pattern, 'adaptive_id'):
            adaptive_id = pattern.adaptive_id
            if hasattr(adaptive_id, 'temporal_context'):
                if isinstance(adaptive_id.temporal_context, dict):
                    return list(adaptive_id.temporal_context.keys())[:max_keywords]
                elif isinstance(adaptive_id.temporal_context, str):
                    try:
                        # Try to parse JSON temporal context
                        temporal_data = json.loads(adaptive_id.temporal_context)
                        if isinstance(temporal_data, dict):
                            return list(temporal_data.keys())[:max_keywords]
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        # Extract keywords from semantic content
        content = PatternSemanticEnhancer.get_semantic_content(pattern)
        return PatternSemanticEnhancer.extract_keywords_from_text(content, max_keywords)
    
    @staticmethod
    def extract_keywords_from_text(text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract keywords from text content.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stopwords and short words
        keywords = [w for w in words if len(w) > 3 and w not in STOPWORDS]
        
        # Get unique keywords
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        # Return top N keywords
        return unique_keywords[:max_keywords]
    
    @staticmethod
    def to_neo4j_properties(pattern: Any) -> Dict[str, Any]:
        """
        Convert pattern to Neo4j properties with semantic content.
        
        Args:
            pattern: Any pattern-like object
            
        Returns:
            Dictionary of properties for Neo4j
        """
        # Get semantic content and keywords
        semantic_content = PatternSemanticEnhancer.get_semantic_content(pattern)
        keywords = PatternSemanticEnhancer.get_keywords(pattern)
        
        # Base properties
        properties = {
            "semantic_content": semantic_content,
            "keywords": json.dumps(keywords),
            "last_updated": datetime.now().timestamp()
        }
        
        # Add tonic-harmonic properties if available
        if hasattr(pattern, 'tonic_value'):
            properties["tonic_value"] = pattern.tonic_value
            
        if hasattr(pattern, 'stability'):
            properties["stability"] = pattern.stability
            
        if hasattr(pattern, 'phase_position'):
            properties["phase_position"] = pattern.phase_position
            
        # Calculate harmonic value if both tonic and stability are available
        if "tonic_value" in properties and "stability" in properties:
            properties["harmonic_value"] = properties["tonic_value"] * properties["stability"]
            
        # Add confidence if available
        if hasattr(pattern, 'confidence'):
            properties["confidence"] = pattern.confidence
            
        # Add pattern type if available
        if hasattr(pattern, 'pattern_type'):
            properties["pattern_type"] = pattern.pattern_type
        
        return properties
    
    @staticmethod
    def enhance_pattern(pattern: Any) -> Any:
        """
        Enhance a pattern with semantic content methods.
        
        This method adds semantic content capabilities to an existing pattern object
        without modifying its class definition.
        
        Args:
            pattern: Any pattern-like object
            
        Returns:
            The same pattern object with enhanced capabilities
        """
        # Add semantic_content property if not present
        if not hasattr(pattern, 'semantic_content'):
            pattern.semantic_content = PatternSemanticEnhancer.get_semantic_content(pattern)
            
        # Add keywords property if not present
        if not hasattr(pattern, 'keywords'):
            pattern.keywords = PatternSemanticEnhancer.get_keywords(pattern)
            
        # Add to_neo4j_properties method if not present
        if not hasattr(pattern, 'to_neo4j_properties'):
            pattern.to_neo4j_properties = lambda: PatternSemanticEnhancer.to_neo4j_properties(pattern)
            
        return pattern
