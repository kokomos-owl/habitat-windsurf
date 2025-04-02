"""
Semantic Topology Enhancer for the Habitat Evolution framework.

This module enhances the TopologyManager with semantic content capabilities,
enabling direct semantic space representation without abstractions.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from src.habitat_evolution.pattern_aware_rag.semantic.pattern_semantic import PatternSemanticEnhancer

logger = logging.getLogger(__name__)

class SemanticTopologyEnhancer:
    """
    Enhances the TopologyManager with semantic content capabilities.
    
    This class provides methods to enhance the persistence of topology states,
    patterns, and relationships with semantic content, enabling direct semantic
    space representation without abstractions.
    """
    
    @staticmethod
    def enhance_pattern_persistence(session, pattern_id: str, pattern: Any, state_id: str) -> None:
        """
        Enhance pattern persistence with semantic content.
        
        Args:
            session: Neo4j session
            pattern_id: ID of the pattern
            pattern: Pattern object
            state_id: ID of the topology state
        """
        # Extract semantic content and keywords
        semantic_content = PatternSemanticEnhancer.get_semantic_content(pattern)
        keywords = PatternSemanticEnhancer.get_keywords(pattern)
        
        # Update pattern with semantic content
        session.run("""
            MATCH (p:Pattern {id: $pattern_id})
            SET p.semantic_content = $semantic_content,
                p.keywords = $keywords,
                p.last_semantic_update = timestamp()
            WITH p
            MATCH (ts:TopologyState {id: $state_id})
            MERGE (ts)-[:HAS_PATTERN]->(p)
        """,
            pattern_id=pattern_id,
            semantic_content=semantic_content,
            keywords=json.dumps(keywords),
            state_id=state_id
        )
        
        logger.debug(f"Enhanced pattern {pattern_id} with semantic content: {semantic_content[:50]}...")
    
    @staticmethod
    def enhance_resonance_group_persistence(session, group_id: str, patterns: List[Any], state_id: str) -> None:
        """
        Enhance resonance group persistence with semantic content.
        
        Args:
            session: Neo4j session
            group_id: ID of the resonance group
            patterns: List of patterns in the group
            state_id: ID of the topology state
        """
        # Extract semantic content from all patterns in the group
        semantic_contents = []
        all_keywords = set()
        
        for pattern in patterns:
            content = PatternSemanticEnhancer.get_semantic_content(pattern)
            keywords = PatternSemanticEnhancer.get_keywords(pattern)
            
            semantic_contents.append(content)
            all_keywords.update(keywords)
        
        # Create a combined semantic representation for the group
        combined_content = " | ".join(semantic_contents)
        group_keywords = list(all_keywords)[:10]  # Limit to top 10 keywords
        
        # Calculate semantic coherence based on keyword overlap
        semantic_coherence = 0.0
        if patterns:
            # Count how many patterns share each keyword
            keyword_counts = {}
            for pattern in patterns:
                pattern_keywords = PatternSemanticEnhancer.get_keywords(pattern)
                for kw in pattern_keywords:
                    if kw in keyword_counts:
                        keyword_counts[kw] += 1
                    else:
                        keyword_counts[kw] = 1
            
            # Calculate average keyword sharing ratio
            if keyword_counts:
                avg_sharing = sum(count / len(patterns) for count in keyword_counts.values()) / len(keyword_counts)
                semantic_coherence = avg_sharing
        
        # Update resonance group with semantic content
        session.run("""
            MATCH (rg:ResonanceGroup {id: $group_id})
            SET rg.semantic_content = $semantic_content,
                rg.keywords = $keywords,
                rg.semantic_coherence = $semantic_coherence,
                rg.last_semantic_update = timestamp()
            WITH rg
            MATCH (ts:TopologyState {id: $state_id})
            MERGE (ts)-[:HAS_RESONANCE_GROUP]->(rg)
        """,
            group_id=group_id,
            semantic_content=combined_content,
            keywords=json.dumps(group_keywords),
            semantic_coherence=semantic_coherence,
            state_id=state_id
        )
        
        logger.debug(f"Enhanced resonance group {group_id} with semantic content")
    
    @staticmethod
    def enhance_relationship_persistence(
        session, 
        pattern1_id: str, 
        pattern2_id: str, 
        relationship_type: str,
        properties: Dict[str, Any]
    ) -> None:
        """
        Enhance relationship persistence with semantic properties.
        
        Args:
            session: Neo4j session
            pattern1_id: ID of the first pattern
            pattern2_id: ID of the second pattern
            relationship_type: Type of relationship
            properties: Relationship properties
        """
        # Get patterns
        result = session.run("""
            MATCH (p1:Pattern {id: $pattern1_id})
            MATCH (p2:Pattern {id: $pattern2_id})
            RETURN p1.semantic_content as content1, p1.keywords as keywords1,
                   p2.semantic_content as content2, p2.keywords as keywords2
        """,
            pattern1_id=pattern1_id,
            pattern2_id=pattern2_id
        )
        
        record = result.single()
        if not record:
            logger.warning(f"Could not find patterns {pattern1_id} and {pattern2_id} for relationship enhancement")
            return
        
        # Extract semantic content and keywords
        content1 = record["content1"] if record["content1"] else ""
        content2 = record["content2"] if record["content2"] else ""
        
        try:
            keywords1 = json.loads(record["keywords1"]) if record["keywords1"] else []
        except (json.JSONDecodeError, TypeError):
            keywords1 = []
            
        try:
            keywords2 = json.loads(record["keywords2"]) if record["keywords2"] else []
        except (json.JSONDecodeError, TypeError):
            keywords2 = []
        
        # Calculate semantic relationship properties
        semantic_similarity = 0.0
        semantic_overlap = []
        
        # Calculate keyword overlap
        if keywords1 and keywords2:
            set1 = set(keywords1)
            set2 = set(keywords2)
            overlap = set1.intersection(set2)
            union = set1.union(set2)
            
            semantic_similarity = len(overlap) / max(1, len(union))
            semantic_overlap = list(overlap)
        
        # Update relationship with semantic properties
        session.run(f"""
            MATCH (p1:Pattern {{id: $pattern1_id}})
            MATCH (p2:Pattern {{id: $pattern2_id}})
            MERGE (p1)-[r:{relationship_type}]->(p2)
            SET r.semantic_similarity = $semantic_similarity,
                r.semantic_overlap = $semantic_overlap,
                r.last_semantic_update = timestamp()
        """,
            pattern1_id=pattern1_id,
            pattern2_id=pattern2_id,
            semantic_similarity=semantic_similarity,
            semantic_overlap=json.dumps(semantic_overlap)
        )
        
        logger.debug(f"Enhanced relationship between {pattern1_id} and {pattern2_id} with semantic properties")
    
    @staticmethod
    def enhance_topology_state_persistence(session, state_id: str, patterns: Dict[str, Any]) -> None:
        """
        Enhance topology state persistence with semantic summary.
        
        Args:
            session: Neo4j session
            state_id: ID of the topology state
            patterns: Dictionary of patterns in the state
        """
        # Extract semantic content from all patterns
        semantic_contents = []
        all_keywords = set()
        
        for pattern_id, pattern in patterns.items():
            content = PatternSemanticEnhancer.get_semantic_content(pattern)
            keywords = PatternSemanticEnhancer.get_keywords(pattern)
            
            semantic_contents.append(content)
            all_keywords.update(keywords)
        
        # Create a semantic summary for the topology state
        semantic_summary = "Topology state representing: " + ", ".join(list(all_keywords)[:10])
        
        # Update topology state with semantic summary
        session.run("""
            MATCH (ts:TopologyState {id: $state_id})
            SET ts.semantic_summary = $semantic_summary,
                ts.keywords = $keywords,
                ts.last_semantic_update = timestamp()
        """,
            state_id=state_id,
            semantic_summary=semantic_summary,
            keywords=json.dumps(list(all_keywords)[:20])  # Limit to top 20 keywords
        )
        
        logger.debug(f"Enhanced topology state {state_id} with semantic summary")
    
    @staticmethod
    def enhance_frequency_domain_persistence(
        session, 
        domain_id: str, 
        patterns: List[Any], 
        state_id: str
    ) -> None:
        """
        Enhance frequency domain persistence with semantic content.
        
        Args:
            session: Neo4j session
            domain_id: ID of the frequency domain
            patterns: List of patterns in the domain
            state_id: ID of the topology state
        """
        # Extract semantic content from all patterns in the domain
        semantic_contents = []
        all_keywords = set()
        
        for pattern in patterns:
            content = PatternSemanticEnhancer.get_semantic_content(pattern)
            keywords = PatternSemanticEnhancer.get_keywords(pattern)
            
            semantic_contents.append(content)
            all_keywords.update(keywords)
        
        # Create a combined semantic representation for the domain
        domain_content = " | ".join(semantic_contents[:5])  # Limit to 5 patterns
        domain_keywords = list(all_keywords)[:10]  # Limit to top 10 keywords
        
        # Update frequency domain with semantic content
        session.run("""
            MATCH (fd:FrequencyDomain {id: $domain_id})
            SET fd.semantic_content = $semantic_content,
                fd.keywords = $keywords,
                fd.last_semantic_update = timestamp()
            WITH fd
            MATCH (ts:TopologyState {id: $state_id})
            MERGE (ts)-[:HAS_DOMAIN]->(fd)
        """,
            domain_id=domain_id,
            semantic_content=domain_content,
            keywords=json.dumps(domain_keywords),
            state_id=state_id
        )
        
        logger.debug(f"Enhanced frequency domain {domain_id} with semantic content")
    
    @staticmethod
    def enhance_boundary_persistence(
        session, 
        boundary_id: str, 
        domain_ids: List[str], 
        state_id: str
    ) -> None:
        """
        Enhance boundary persistence with semantic content.
        
        Args:
            session: Neo4j session
            boundary_id: ID of the boundary
            domain_ids: IDs of domains connected by the boundary
            state_id: ID of the topology state
        """
        if len(domain_ids) != 2:
            logger.warning(f"Boundary {boundary_id} does not connect exactly 2 domains")
            return
            
        # Get domains
        result = session.run("""
            MATCH (fd1:FrequencyDomain {id: $domain1_id})
            MATCH (fd2:FrequencyDomain {id: $domain2_id})
            RETURN fd1.semantic_content as content1, fd1.keywords as keywords1,
                   fd2.semantic_content as content2, fd2.keywords as keywords2
        """,
            domain1_id=domain_ids[0],
            domain2_id=domain_ids[1]
        )
        
        record = result.single()
        if not record:
            logger.warning(f"Could not find domains {domain_ids[0]} and {domain_ids[1]} for boundary enhancement")
            return
        
        # Extract semantic content and keywords
        content1 = record["content1"] if record["content1"] else ""
        content2 = record["content2"] if record["content2"] else ""
        
        try:
            keywords1 = json.loads(record["keywords1"]) if record["keywords1"] else []
        except (json.JSONDecodeError, TypeError):
            keywords1 = []
            
        try:
            keywords2 = json.loads(record["keywords2"]) if record["keywords2"] else []
        except (json.JSONDecodeError, TypeError):
            keywords2 = []
        
        # Calculate semantic boundary properties
        set1 = set(keywords1)
        set2 = set(keywords2)
        overlap = set1.intersection(set2)
        unique1 = set1 - set2
        unique2 = set2 - set1
        
        # Create semantic representation of the boundary
        boundary_content = f"Boundary between {content1[:50]} and {content2[:50]}"
        boundary_keywords = list(overlap)
        
        # Calculate semantic permeability based on keyword overlap
        semantic_permeability = len(overlap) / max(1, len(set1.union(set2)))
        
        # Update boundary with semantic content
        session.run("""
            MATCH (b:Boundary {id: $boundary_id})
            SET b.semantic_content = $semantic_content,
                b.keywords = $keywords,
                b.semantic_permeability = $semantic_permeability,
                b.domain1_unique_keywords = $unique1,
                b.domain2_unique_keywords = $unique2,
                b.shared_keywords = $shared,
                b.last_semantic_update = timestamp()
            WITH b
            MATCH (ts:TopologyState {id: $state_id})
            MERGE (ts)-[:HAS_BOUNDARY]->(b)
        """,
            boundary_id=boundary_id,
            semantic_content=boundary_content,
            keywords=json.dumps(boundary_keywords),
            semantic_permeability=semantic_permeability,
            unique1=json.dumps(list(unique1)[:5]),
            unique2=json.dumps(list(unique2)[:5]),
            shared=json.dumps(list(overlap)),
            state_id=state_id
        )
        
        logger.debug(f"Enhanced boundary {boundary_id} with semantic content")
