"""
Concept-Predicate-Syntax Model for the Habitat Evolution system.

This module implements a co-evolutionary model of language where concepts and
predicates co-evolve through their interactions, with syntax emerging as
momentary intentionality.
"""
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import asyncio
from datetime import datetime, timedelta

# Import from our local implementation
from ..persistence.semantic_potential_calculator import (
    ConceptNode, 
    PatternState, 
    GraphService,
    SemanticPotentialCalculator
)


class ConceptPredicateSyntaxModel:
    """
    Model for the co-evolutionary concept-predicate-syntax space in Habitat.
    
    This model captures how concepts and predicates co-evolve through their
    interactions, with syntax emerging as momentary intentionality.
    """
    
    def __init__(self, graph_service: GraphService, potential_calculator: SemanticPotentialCalculator):
        """
        Initialize the concept-predicate-syntax model.
        
        Args:
            graph_service: The graph service for accessing persistence data
            potential_calculator: The semantic potential calculator
        """
        self.graph_service = graph_service
        self.potential_calculator = potential_calculator
        
    async def map_co_resonance_field(self, window_id: str = None) -> Dict[str, Any]:
        """
        Map the co-resonance field between concepts and predicates.
        
        This identifies how concepts and predicates are co-evolving
        and influencing each other's identity.
        
        Args:
            window_id: Optional specific window to focus on
            
        Returns:
            Co-resonance field map
        """
        # Get concepts (stabilized patterns)
        concepts = await self._get_stabilized_patterns(window_id)
        
        # Get predicates (transformative relationships)
        predicates = await self._get_transformative_relationships(window_id)
        
        # Calculate co-resonance between each concept-predicate pair
        co_resonances = []
        for concept in concepts:
            for predicate in predicates:
                resonance = await self._calculate_co_resonance(concept, predicate)
                if resonance["strength"] > 0.3:  # Threshold for meaningful resonance
                    co_resonances.append(resonance)
        
        # Map the field of co-resonances
        field_map = self._build_co_resonance_field(co_resonances)
        
        return field_map
    
    async def detect_intentionality_vectors(self, window_id: str = None) -> Dict[str, Any]:
        """
        Detect momentary intentionality vectors in the syntax space.
        
        These vectors represent the current direction of attention and
        intention in the concept-predicate space.
        
        Args:
            window_id: Optional specific window to focus on
            
        Returns:
            Intentionality vectors
        """
        # Get potential gradients
        field_potential = await self.potential_calculator.calculate_field_potential(window_id)
        topo_potential = await self.potential_calculator.calculate_topological_potential(window_id)
        
        # Calculate intentionality vectors from gradients
        vectors = self._derive_intentionality_from_gradients(field_potential, topo_potential)
        
        return vectors
    
    async def generate_co_evolutionary_expression(
        self, seed_concepts: List[str] = None, intentionality: Dict[str, Any] = None, window_id: str = None
    ) -> Dict[str, Any]:
        """
        Generate an expression from the co-evolutionary space.
        
        This creates a meaningful expression based on the current
        state of the concept-predicate co-resonance field and
        intentionality vectors.
        
        Args:
            seed_concepts: Optional concept IDs to start with
            intentionality: Optional intentionality to guide expression
            window_id: Optional specific window to focus on
            
        Returns:
            Generated expression
        """
        # Get co-resonance field
        field = await self.map_co_resonance_field(window_id)
        
        # Get intentionality vectors if not provided
        if intentionality is None:
            intentionality = await self.detect_intentionality_vectors(window_id)
        
        # Select concepts based on seed or intentionality
        selected_concepts = await self._select_concepts(seed_concepts, intentionality, field)
        
        # Select predicates that co-resonate with the concepts
        selected_predicates = await self._select_co_resonant_predicates(selected_concepts, field)
        
        # Compose expression based on momentary syntax (intentionality)
        expression = self._compose_expression(selected_concepts, selected_predicates, intentionality)
        
        return expression
    
    async def _get_stabilized_patterns(self, window_id: str = None) -> List[PatternState]:
        """
        Get stabilized patterns that can serve as concepts.
        
        Args:
            window_id: Optional specific window to focus on
            
        Returns:
            List of stabilized patterns
        """
        try:
            # Find patterns with good quality state
            patterns = await asyncio.to_thread(
                self.graph_service.repository.find_nodes_by_quality,
                "good",
                node_type="pattern"
            )
            
            # Filter by window if specified
            if window_id and patterns:
                patterns = [
                    p for p in patterns 
                    if p.attributes.get("window_id") == window_id
                ]
                
            return patterns
        except Exception as e:
            print(f"Error getting stabilized patterns: {e}")
            return []
    
    async def _get_transformative_relationships(self, window_id: str = None) -> List[Dict[str, Any]]:
        """
        Get transformative relationships that can serve as predicates.
        
        Args:
            window_id: Optional specific window to focus on
            
        Returns:
            List of transformative relationships
        """
        try:
            # Find relations with uncertain or good quality state
            relations = await asyncio.to_thread(
                self.graph_service.repository.find_relations_by_quality,
                ["uncertain", "good"]
            )
            
            # Convert to predicate format
            predicates = []
            for relation in relations:
                # Filter by window if specified
                if window_id and relation.attributes.get("window_id") != window_id:
                    continue
                    
                predicates.append({
                    "id": relation.id,
                    "type": relation.relation_type,
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "weight": relation.weight,
                    "attributes": relation.attributes
                })
                
            return predicates
        except Exception as e:
            print(f"Error getting transformative relationships: {e}")
            return []
    
    async def _calculate_co_resonance(
        self, concept: PatternState, predicate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the co-resonance between a concept and predicate.
        
        This measures how they co-evolve and influence each other.
        
        Args:
            concept: Concept pattern
            predicate: Predicate relationship
            
        Returns:
            Co-resonance metrics
        """
        # Calculate historical co-occurrence
        co_occurrences = await self._get_historical_co_occurrences(concept, predicate)
        
        # Calculate semantic compatibility
        compatibility = self._calculate_semantic_compatibility(concept, predicate)
        
        # Calculate mutual information
        mutual_info = self._calculate_mutual_information(concept, predicate)
        
        # Calculate evolutionary influence (how they've changed each other)
        concept_influence = await self._calculate_evolutionary_influence(predicate, concept)
        predicate_influence = await self._calculate_evolutionary_influence(concept, predicate)
        
        # Combine metrics
        strength = (
            compatibility * 0.3 +
            mutual_info * 0.3 +
            concept_influence * 0.2 +
            predicate_influence * 0.2
        )
        
        return {
            "concept_id": concept.id,
            "predicate_id": predicate["id"],
            "strength": strength,
            "compatibility": compatibility,
            "mutual_information": mutual_info,
            "concept_influence": concept_influence,
            "predicate_influence": predicate_influence,
            "co_occurrences": co_occurrences
        }
    
    async def _get_historical_co_occurrences(
        self, concept: PatternState, predicate: Dict[str, Any]
    ) -> int:
        """
        Get historical co-occurrences of concept and predicate.
        
        Args:
            concept: Concept pattern
            predicate: Predicate relationship
            
        Returns:
            Number of co-occurrences
        """
        # In a real implementation, this would query the database
        # For now, we'll use a simple implementation
        try:
            # Check if the concept is related to the predicate
            if (concept.id == predicate["source_id"] or 
                concept.id == predicate["target_id"]):
                return 1
            else:
                return 0
        except Exception as e:
            print(f"Error getting historical co-occurrences: {e}")
            return 0
    
    def _calculate_semantic_compatibility(
        self, concept: PatternState, predicate: Dict[str, Any]
    ) -> float:
        """
        Calculate semantic compatibility between concept and predicate.
        
        Args:
            concept: Concept pattern
            predicate: Predicate relationship
            
        Returns:
            Compatibility score (0-1)
        """
        # In a real implementation, this would use embeddings or other semantic measures
        # For now, we'll use a simple implementation
        try:
            # Check if the concept type is compatible with the predicate type
            concept_type = concept.metadata.get("type", "")
            predicate_type = predicate["type"]
            
            # Define compatibility rules (simplified)
            compatibility_map = {
                "entity": ["has_property", "related_to", "instance_of"],
                "event": ["occurs_in", "causes", "involves"],
                "property": ["describes", "measured_by", "compared_to"],
                "concept": ["broader_than", "narrower_than", "similar_to"],
                "statistical": ["correlates_with", "predicts", "distributes_as"]
            }
            
            # Check compatibility
            compatible_predicates = compatibility_map.get(concept_type, [])
            if predicate_type in compatible_predicates:
                return 0.8  # High compatibility
            else:
                return 0.3  # Low compatibility
        except Exception as e:
            print(f"Error calculating semantic compatibility: {e}")
            return 0.5  # Default compatibility
    
    def _calculate_mutual_information(
        self, concept: PatternState, predicate: Dict[str, Any]
    ) -> float:
        """
        Calculate mutual information between concept and predicate.
        
        Args:
            concept: Concept pattern
            predicate: Predicate relationship
            
        Returns:
            Mutual information (0-1)
        """
        # In a real implementation, this would calculate actual mutual information
        # For now, we'll use a simple implementation
        try:
            # Use concept confidence and predicate weight as proxies
            concept_confidence = float(concept.confidence)
            predicate_weight = float(predicate["weight"])
            
            # Calculate mutual information (simplified)
            mutual_info = (concept_confidence * predicate_weight) ** 0.5
            
            return mutual_info
        except Exception as e:
            print(f"Error calculating mutual information: {e}")
            return 0.5  # Default mutual information
    
    async def _calculate_evolutionary_influence(
        self, source: Any, target: Any
    ) -> float:
        """
        Calculate evolutionary influence of source on target.
        
        This measures how much the source has influenced the evolution of the target.
        
        Args:
            source: Source entity (concept or predicate)
            target: Target entity (concept or predicate)
            
        Returns:
            Influence score (0-1)
        """
        # In a real implementation, this would analyze historical transitions
        # For now, we'll use a simple implementation
        try:
            # Get quality transitions for the target
            target_id = target.id if hasattr(target, "id") else target["id"]
            transitions = await asyncio.to_thread(
                self.graph_service.repository.find_quality_transitions_by_node_id,
                target_id
            )
            
            if not transitions:
                return 0.5  # Default influence
                
            # Count transitions that mention the source in context
            source_id = source.id if hasattr(source, "id") else source["id"]
            influenced_transitions = sum(
                1 for t in transitions
                if source_id in str(t.context)
            )
            
            # Calculate influence ratio
            influence = influenced_transitions / len(transitions) if transitions else 0.5
            
            return influence
        except Exception as e:
            print(f"Error calculating evolutionary influence: {e}")
            return 0.5  # Default influence
    
    def _build_co_resonance_field(self, co_resonances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a map of the co-resonance field.
        
        Args:
            co_resonances: List of co-resonance measurements
            
        Returns:
            Co-resonance field map
        """
        if not co_resonances:
            return {
                "concepts": {},
                "predicates": {},
                "resonances": [],
                "clusters": [],
                "field_strength": 0
            }
            
        # Extract unique concepts and predicates
        concept_ids = set(r["concept_id"] for r in co_resonances)
        predicate_ids = set(r["predicate_id"] for r in co_resonances)
        
        # Calculate average resonance strength
        avg_strength = sum(r["strength"] for r in co_resonances) / len(co_resonances)
        
        # Identify clusters of strongly resonating concept-predicate pairs
        clusters = self._identify_resonance_clusters(co_resonances)
        
        # Build the field map
        field_map = {
            "concepts": {cid: {} for cid in concept_ids},
            "predicates": {pid: {} for pid in predicate_ids},
            "resonances": co_resonances,
            "clusters": clusters,
            "field_strength": avg_strength
        }
        
        return field_map
    
    def _identify_resonance_clusters(self, co_resonances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify clusters of strongly resonating concept-predicate pairs.
        
        Args:
            co_resonances: List of co-resonance measurements
            
        Returns:
            List of resonance clusters
        """
        # In a real implementation, this would use clustering algorithms
        # For now, we'll use a simple threshold-based approach
        strong_resonances = [r for r in co_resonances if r["strength"] > 0.6]
        
        if not strong_resonances:
            return []
            
        # Group by concepts
        concept_groups = {}
        for resonance in strong_resonances:
            concept_id = resonance["concept_id"]
            if concept_id not in concept_groups:
                concept_groups[concept_id] = []
            concept_groups[concept_id].append(resonance)
        
        # Create clusters
        clusters = []
        for concept_id, resonances in concept_groups.items():
            if len(resonances) > 1:  # Only create clusters with multiple resonances
                clusters.append({
                    "concept_id": concept_id,
                    "predicate_ids": [r["predicate_id"] for r in resonances],
                    "strength": sum(r["strength"] for r in resonances) / len(resonances)
                })
        
        return clusters
    
    def _derive_intentionality_from_gradients(
        self, field_potential: Dict[str, Any], topo_potential: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Derive intentionality vectors from potential gradients.
        
        Args:
            field_potential: Field potential metrics
            topo_potential: Topological potential metrics
            
        Returns:
            Intentionality vectors
        """
        # Extract gradient information
        gradient_field = field_potential["gradient_field"]
        manifold_curvature = topo_potential["manifold_curvature"]
        
        # Calculate primary direction from gradient field
        primary_direction = gradient_field["direction"]
        
        # Calculate magnitude from gradient magnitude and curvature
        magnitude = (
            gradient_field["magnitude"] * 0.7 +
            manifold_curvature["average_curvature"] * 0.3
        )
        
        # Calculate focus from gradient uniformity
        focus = gradient_field["uniformity"]
        
        # Calculate stability from topological depth
        stability = manifold_curvature["topological_depth"]
        
        # Create intentionality vectors
        vectors = {
            "primary_direction": primary_direction,
            "magnitude": magnitude,
            "focus": focus,
            "stability": stability,
            "composite_vector": {
                "direction": primary_direction,
                "strength": magnitude * focus,
                "stability": stability
            }
        }
        
        return vectors
    
    async def _select_concepts(
        self, seed_concepts: List[str], intentionality: Dict[str, Any], field: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Select concepts based on seed or intentionality.
        
        Args:
            seed_concepts: Optional concept IDs to start with
            intentionality: Intentionality vectors
            field: Co-resonance field
            
        Returns:
            Selected concepts
        """
        selected = []
        
        # If seed concepts are provided, use them
        if seed_concepts:
            for concept_id in seed_concepts:
                try:
                    concept = await asyncio.to_thread(
                        self.graph_service.repository.find_node_by_id,
                        concept_id
                    )
                    if concept:
                        selected.append({
                            "id": concept.id,
                            "content": concept.name,
                            "confidence": float(concept.attributes.get("confidence", 0.5)),
                            "attributes": concept.attributes
                        })
                except Exception as e:
                    print(f"Error finding concept {concept_id}: {e}")
        
        # If no seed concepts or not enough found, select based on intentionality
        if len(selected) < 2:
            # Get concepts from clusters aligned with intentionality
            for cluster in field.get("clusters", []):
                # Check if cluster aligns with intentionality
                if self._cluster_aligns_with_intentionality(cluster, intentionality):
                    try:
                        concept = await asyncio.to_thread(
                            self.graph_service.repository.find_node_by_id,
                            cluster["concept_id"]
                        )
                        if concept:
                            selected.append({
                                "id": concept.id,
                                "content": concept.name,
                                "confidence": float(concept.attributes.get("confidence", 0.5)),
                                "attributes": concept.attributes
                            })
                    except Exception as e:
                        print(f"Error finding concept {cluster['concept_id']}: {e}")
                        
                # Stop when we have enough concepts
                if len(selected) >= 3:
                    break
        
        return selected
    
    def _cluster_aligns_with_intentionality(
        self, cluster: Dict[str, Any], intentionality: Dict[str, Any]
    ) -> bool:
        """
        Check if a cluster aligns with intentionality.
        
        Args:
            cluster: Resonance cluster
            intentionality: Intentionality vectors
            
        Returns:
            True if aligned, False otherwise
        """
        # In a real implementation, this would use more sophisticated alignment metrics
        # For now, we'll use a simple threshold-based approach
        
        # Check if cluster strength aligns with intentionality magnitude
        strength_alignment = abs(cluster["strength"] - intentionality["magnitude"]) < 0.3
        
        return strength_alignment
    
    async def _select_co_resonant_predicates(
        self, concepts: List[Dict[str, Any]], field: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Select predicates that co-resonate with the concepts.
        
        Args:
            concepts: Selected concepts
            field: Co-resonance field
            
        Returns:
            Selected predicates
        """
        if not concepts:
            return []
            
        selected = []
        concept_ids = [c["id"] for c in concepts]
        
        # Find predicates that resonate with the selected concepts
        for resonance in field.get("resonances", []):
            if resonance["concept_id"] in concept_ids and resonance["strength"] > 0.5:
                try:
                    # Get the predicate details
                    predicate_id = resonance["predicate_id"]
                    
                    # Check if we already selected this predicate
                    if any(p["id"] == predicate_id for p in selected):
                        continue
                        
                    # Get predicate from repository
                    # In a real implementation, this would get the actual predicate
                    # For now, we'll create a mock predicate
                    selected.append({
                        "id": predicate_id,
                        "type": "unknown",  # This would be filled from repository
                        "strength": resonance["strength"],
                        "connects": [
                            c["id"] for c in concepts
                            if any(r["concept_id"] == c["id"] and r["predicate_id"] == predicate_id
                                 for r in field.get("resonances", []))
                        ]
                    })
                except Exception as e:
                    print(f"Error selecting predicate {resonance['predicate_id']}: {e}")
        
        return selected
    
    def _compose_expression(
        self, concepts: List[Dict[str, Any]], predicates: List[Dict[str, Any]], intentionality: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compose an expression based on concepts, predicates, and intentionality.
        
        Args:
            concepts: Selected concepts
            predicates: Selected predicates
            intentionality: Intentionality vectors
            
        Returns:
            Composed expression
        """
        if not concepts or not predicates:
            return {
                "expression": "",
                "components": [],
                "intentionality": intentionality,
                "coherence": 0
            }
            
        # Sort concepts by confidence
        sorted_concepts = sorted(concepts, key=lambda c: c["confidence"], reverse=True)
        
        # Sort predicates by strength
        sorted_predicates = sorted(predicates, key=lambda p: p["strength"], reverse=True)
        
        # Build components list
        components = []
        for concept in sorted_concepts:
            components.append({
                "type": "concept",
                "id": concept["id"],
                "content": concept["content"],
                "confidence": concept["confidence"]
            })
            
        for predicate in sorted_predicates:
            components.append({
                "type": "predicate",
                "id": predicate["id"],
                "content": predicate["type"],
                "strength": predicate["strength"],
                "connects": predicate["connects"]
            })
            
        # Build expression text
        # In a real implementation, this would use more sophisticated language generation
        # For now, we'll use a simple template-based approach
        expression_text = self._generate_expression_text(sorted_concepts, sorted_predicates, intentionality)
        
        # Calculate coherence
        coherence = self._calculate_expression_coherence(sorted_concepts, sorted_predicates, intentionality)
        
        return {
            "expression": expression_text,
            "components": components,
            "intentionality": intentionality,
            "coherence": coherence
        }
    
    def _generate_expression_text(
        self, concepts: List[Dict[str, Any]], predicates: List[Dict[str, Any]], intentionality: Dict[str, Any]
    ) -> str:
        """
        Generate expression text from concepts and predicates.
        
        Args:
            concepts: Selected concepts
            predicates: Selected predicates
            intentionality: Intentionality vectors
            
        Returns:
            Expression text
        """
        if not concepts or not predicates:
            return ""
            
        # Get primary concept and predicate
        primary_concept = concepts[0]["content"]
        primary_predicate = predicates[0]["type"]
        
        # Build expression based on intentionality direction
        direction = intentionality["primary_direction"]
        
        if direction == "increasing":
            # Forward-looking expression
            if len(concepts) > 1:
                secondary_concept = concepts[1]["content"]
                return f"{primary_concept} {primary_predicate} {secondary_concept}"
            else:
                return f"{primary_concept} {primary_predicate}"
        elif direction == "decreasing":
            # Backward-looking expression
            if len(concepts) > 1:
                secondary_concept = concepts[1]["content"]
                return f"{secondary_concept} {primary_predicate} {primary_concept}"
            else:
                return f"{primary_concept} {primary_predicate}"
        else:  # "stable"
            # Balanced expression
            if len(concepts) > 1:
                secondary_concept = concepts[1]["content"]
                return f"{primary_concept} and {secondary_concept} {primary_predicate}"
            else:
                return f"{primary_concept} {primary_predicate}"
    
    def _calculate_expression_coherence(
        self, concepts: List[Dict[str, Any]], predicates: List[Dict[str, Any]], intentionality: Dict[str, Any]
    ) -> float:
        """
        Calculate coherence of the expression.
        
        Args:
            concepts: Selected concepts
            predicates: Selected predicates
            intentionality: Intentionality vectors
            
        Returns:
            Coherence score (0-1)
        """
        if not concepts or not predicates:
            return 0
            
        # Calculate average concept confidence
        avg_confidence = sum(c["confidence"] for c in concepts) / len(concepts)
        
        # Calculate average predicate strength
        avg_strength = sum(p["strength"] for p in predicates) / len(predicates)
        
        # Factor in intentionality focus
        focus = intentionality["focus"]
        
        # Calculate coherence
        coherence = (avg_confidence * 0.4) + (avg_strength * 0.4) + (focus * 0.2)
        
        return coherence
