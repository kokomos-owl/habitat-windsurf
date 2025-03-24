"""
Emergent Propensity Module

This module implements the concept of emergent propensities - the dynamic, emergent
properties that arise from the interaction between semantic propositions and domains.
These represent both state-changes of patterns and state-conditions of communities.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from habitat_evolution.adaptive_core.transformation.semantic_proposition_patterns import SemanticProposition


class EmergentPropensity:
    """
    Represents the emergent propensities within semantic patterns.
    
    This class captures the dynamic, emergent properties that arise from the interaction
    between semantic propositions and domains - the propensities "of" patterns rather
    than just the patterns themselves.
    
    These propensities represent both:
    1. State-changes of patterns (how patterns transform across domains)
    2. State-conditions of communities (how communities embody semantic flows)
    """
    
    def __init__(self, source_proposition: SemanticProposition, domain_context: Dict[str, Any]):
        """
        Initialize an emergent propensity.
        
        Args:
            source_proposition: The proposition from which propensities emerge
            domain_context: The context in which propensities manifest
        """
        self.source_proposition = source_proposition
        self.domain_context = domain_context
        self.capaciousness = source_proposition.capaciousness
        self.directionality = source_proposition.directionality.copy()
        
        # Emergent properties - these are the emergent quotients
        self.manifestation_potential = self._calculate_manifestation_potential()
        self.state_change_vectors = self._calculate_state_change_vectors()
        self.community_condition_indices = self._calculate_community_condition_indices()
    
    def _calculate_manifestation_potential(self) -> float:
        """
        Calculate the potential for this propensity to manifest in the domain.
        
        This is an emergent quotient representing how strongly the pattern
        wants to manifest in this specific context.
        
        Returns:
            A manifestation potential between 0 and 1
        """
        # Base potential from capaciousness
        potential = self.capaciousness * 0.5
        
        # Adjust based on domain context compatibility
        context_compatibility = self._assess_context_compatibility()
        potential *= (1.0 + context_compatibility)
        
        # Adjust based on actant presence
        actant_presence = self._assess_actant_presence()
        potential *= (0.5 + actant_presence)
        
        return min(potential, 1.0)
    
    def _assess_context_compatibility(self) -> float:
        """
        Assess how compatible the domain context is with the proposition.
        
        Returns:
            A compatibility score between 0 and 1
        """
        # Simple implementation - can be enhanced
        compatibility = 0.5  # Default moderate compatibility
        
        # Check for domain context emphasis that aligns with directionality
        if isinstance(self.domain_context, dict) and 'context' in self.domain_context:
            context = self.domain_context['context']
            if isinstance(context, dict) and 'emphasis' in context:
                emphasis = context['emphasis']
                # Check if any directional emphasis aligns with proposition directionality
                for direction, weight in self.directionality.items():
                    if direction in emphasis:
                        compatibility += weight * emphasis[direction] * 0.3
        
        return min(compatibility, 1.0)
    
    def _assess_actant_presence(self) -> float:
        """
        Assess the presence of relevant actants in the domain.
        
        Returns:
            An actant presence score between 0 and 1
        """
        # Simple implementation - can be enhanced
        if 'actants' not in self.domain_context:
            return 0.5  # Default moderate presence
            
        domain_actants = set(self.domain_context['actants'])
        proposition_actants = set(self.source_proposition.constituent_actants)
        
        # Calculate overlap
        if not proposition_actants:
            return 0.5
            
        overlap = len(domain_actants.intersection(proposition_actants)) / len(proposition_actants)
        return min(overlap, 1.0)
    
    def _calculate_state_change_vectors(self) -> Dict[str, float]:
        """
        Calculate vectors representing potential state changes of the pattern.
        
        These vectors represent how the pattern might transform when instantiated.
        
        Returns:
            A dictionary mapping directions to their transformed weights
        """
        vectors = {}
        
        # Start with base directionality
        for direction, weight in self.directionality.items():
            vectors[direction] = weight
            
        # Apply transformation rules based on domain context
        for direction, weight in list(vectors.items()):
            # Find transformation rules that apply to this direction
            for rule in self._get_applicable_transformation_rules(direction):
                target_direction = rule.get('target_direction')
                transform_weight = rule.get('weight', 0.3)
                
                if target_direction and transform_weight > 0:
                    # Create or update target direction with transformed weight
                    vectors[target_direction] = vectors.get(target_direction, 0) + (weight * transform_weight)
        
        # Normalize
        total = sum(vectors.values())
        if total > 0:
            vectors = {d: w/total for d, w in vectors.items()}
            
        return vectors
    
    def _get_applicable_transformation_rules(self, direction: str) -> List[Dict[str, Any]]:
        """
        Get transformation rules that apply to a given direction in this domain.
        
        Returns:
            A list of applicable transformation rules
        """
        # Simple implementation - can be enhanced with more sophisticated rule matching
        rules = []
        
        # Example rule: adapts -> supports in community contexts
        if direction == 'adapts' and 'community' in self.domain_context.get('actants', []):
            rules.append({
                'target_direction': 'supports',
                'weight': 0.4
            })
            
        # Example rule: supports -> funds in economic contexts
        if direction == 'supports' and 'economy' in self.domain_context.get('actants', []):
            rules.append({
                'target_direction': 'funds',
                'weight': 0.3
            })
            
        return rules
    
    def _calculate_community_condition_indices(self) -> Dict[str, float]:
        """
        Calculate indices representing potential state-conditions of a community.
        
        These indices represent how a community might embody certain semantic flows.
        
        Returns:
            A dictionary of community condition indices
        """
        indices = {}
        
        # Base indices from directionality
        for direction, weight in self.directionality.items():
            indices[f"{direction}_index"] = weight * self.manifestation_potential
            
        # Calculate composite indices
        if 'adapts_index' in indices and 'supports_index' in indices:
            indices['resilience_index'] = (indices['adapts_index'] + indices['supports_index']) / 2
            
        if 'funds_index' in indices and 'invests_index' in indices:
            indices['economic_vitality_index'] = (indices['funds_index'] + indices['invests_index']) / 2
            
        return indices
    
    def get_emergent_quotients(self) -> Dict[str, float]:
        """
        Get all emergent quotients as a dictionary.
        
        Returns:
            A dictionary of all emergent quotients
        """
        quotients = {
            'manifestation_potential': self.manifestation_potential
        }
        
        # Add state change vector magnitudes
        for direction, weight in self.state_change_vectors.items():
            quotients[f"{direction}_vector"] = weight
            
        # Add community condition indices
        quotients.update(self.community_condition_indices)
        
        return quotients
    
    def project_state_change(self) -> Dict[str, Any]:
        """
        Project how the pattern will change when manifested in this domain.
        
        Returns:
            A dictionary describing the projected state change
        """
        # Find primary direction (highest weight in state change vectors)
        primary_direction = max(self.state_change_vectors.items(), key=lambda x: x[1])
        
        # Calculate transformation magnitude
        original_weight = self.directionality.get(primary_direction[0], 0)
        transformed_weight = primary_direction[1]
        transformation_magnitude = abs(transformed_weight - original_weight)
        
        return {
            'primary_direction': primary_direction[0],
            'original_weight': original_weight,
            'transformed_weight': transformed_weight,
            'transformation_magnitude': transformation_magnitude,
            'transformation_type': 'amplification' if transformed_weight > original_weight else 'attenuation',
            'state_change_description': f"The pattern's '{primary_direction[0]}' direction will " +
                                       f"{'strengthen' if transformed_weight > original_weight else 'weaken'} " +
                                       f"by {transformation_magnitude:.2f} when manifested in this domain."
        }
    
    def project_community_condition(self) -> Dict[str, Any]:
        """
        Project how the community will embody this pattern when manifested.
        
        Returns:
            A dictionary describing the projected community condition
        """
        # Find primary condition index
        primary_index = max(self.community_condition_indices.items(), key=lambda x: x[1])
        
        # Determine condition strength category
        strength = primary_index[1]
        if strength > 0.7:
            strength_category = "strong"
        elif strength > 0.4:
            strength_category = "moderate"
        else:
            strength_category = "weak"
            
        return {
            'primary_condition': primary_index[0],
            'condition_strength': strength,
            'strength_category': strength_category,
            'condition_description': f"The community will exhibit a {strength_category} " +
                                    f"{primary_index[0]} of {strength:.2f} when this pattern manifests."
        }
