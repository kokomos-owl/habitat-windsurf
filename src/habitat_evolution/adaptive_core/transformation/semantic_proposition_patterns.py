"""
Semantic Proposition Patterns Module

This module implements the concept of semantic patterns as propositions - 
capacious structures that carry code and can be instantiated or negatively
instantiated as conductive structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, Any
import numpy as np
from scipy import sparse
import networkx as nx

from habitat_evolution.adaptive_core.transformation.predicate_sublimation import ConceptualFramework


class SemanticProposition:
    """
    A semantic proposition derived from a conceptual framework.
    
    Propositions carry code - they are executable patterns that can be
    instantiated across domains or negatively instantiated to identify
    conductive gaps.
    """
    
    def __init__(self, framework: ConceptualFramework):
        """
        Initialize a semantic proposition from a conceptual framework.
        
        Args:
            framework: The conceptual framework to derive the proposition from
        """
        self.source_framework = framework
        self.name = framework.name
        self.description = framework.description
        self.capaciousness = framework.capaciousness_index
        self.directionality = framework.semantic_directionality.copy()
        self.constituent_predicates = framework.constituent_predicates.copy()
        self.constituent_actants = framework.constituent_actants.copy()
        
        # Derived properties
        self.flow_signature = self._compute_flow_signature()
        self.proposition_code = self._derive_proposition_code()
        
    def _compute_flow_signature(self) -> np.ndarray:
        """
        Compute a numerical signature of the semantic flow directionality.
        
        Returns:
            A normalized vector representing the flow signature
        """
        # Sort directions by name for consistency
        directions = sorted(self.directionality.keys())
        signature = np.array([self.directionality[d] for d in directions])
        
        # Normalize to unit vector
        norm = np.linalg.norm(signature)
        if norm > 0:
            signature = signature / norm
            
        return signature
    
    def _derive_proposition_code(self) -> Dict[str, Any]:
        """
        Derive executable proposition code from the framework.
        
        This code encapsulates the pattern's behavior when instantiated.
        
        Returns:
            A dictionary of code components that define the proposition
        """
        # Extract primary directionality (verb with highest weight)
        primary_direction = max(self.directionality.items(), key=lambda x: x[1])[0]
        
        # Create code structure
        code = {
            'primary_verb': primary_direction,
            'flow_weights': self.directionality,
            'actant_roles': self._derive_actant_roles(),
            'transformation_rules': self._derive_transformation_rules(),
            'boundary_conditions': {
                'min_capaciousness': self.capaciousness * 0.8,  # 80% of original
                'similarity_threshold': 0.7
            }
        }
        
        return code
    
    def _derive_actant_roles(self) -> Dict[str, str]:
        """
        Derive the roles of actants in this proposition.
        
        Returns:
            A dictionary mapping actant names to their roles
        """
        # Simple implementation - can be enhanced
        roles = {}
        
        # Get the primary verb (direction with highest weight)
        primary_verb = max(self.directionality.items(), key=lambda x: x[1])[0]
        
        # Assign roles based on predicate structure
        # This is a simplified approach
        for actant in self.constituent_actants:
            if actant in self.name.lower():
                roles[actant] = 'primary'
            else:
                roles[actant] = 'secondary'
                
        return roles
    
    def _derive_transformation_rules(self) -> List[Dict[str, Any]]:
        """
        Derive transformation rules that govern how this proposition transforms.
        
        Returns:
            A list of transformation rule dictionaries
        """
        # Extract primary and secondary directions
        sorted_directions = sorted(self.directionality.items(), key=lambda x: x[1], reverse=True)
        
        rules = []
        
        # Create rules based on top directionality pairs
        for i in range(min(len(sorted_directions) - 1, 2)):  # Top 2 pairs if available
            source_verb, source_weight = sorted_directions[i]
            target_verb, target_weight = sorted_directions[i + 1]
            
            rules.append({
                'source_verb': source_verb,
                'target_verb': target_verb,
                'transition_weight': source_weight / (source_weight + target_weight),
                'conditions': {
                    'min_source_presence': source_weight * 0.6,  # 60% of original
                    'context_similarity': 0.5
                }
            })
            
        return rules
    
    def instantiate(self, target_domain: Dict[str, Any]) -> 'PropositionInstance':
        """
        Instantiate this proposition in a target domain.
        
        Args:
            target_domain: The domain to instantiate the proposition in
            
        Returns:
            A PropositionInstance representing the instantiated proposition
        """
        # Calculate compatibility with target domain
        compatibility = self._calculate_domain_compatibility(target_domain)
        
        # Create instance with adjusted properties based on domain
        instance = PropositionInstance(
            proposition=self,
            target_domain=target_domain,
            compatibility=compatibility
        )
        
        return instance
    
    def negative_instantiate(self, target_domain: Dict[str, Any]) -> 'ConductiveGap':
        """
        Negatively instantiate this proposition to identify conductive gaps.
        
        Args:
            target_domain: The domain to check for gaps
            
        Returns:
            A ConductiveGap representing the negative instantiation
        """
        # Project expected flows in target domain
        expected_flows = self._project_flows(target_domain)
        
        # Extract actual flows in target domain
        actual_flows = self._extract_actual_flows(target_domain)
        
        # Calculate gaps between expected and actual
        gaps = {}
        for direction, expected in expected_flows.items():
            actual = actual_flows.get(direction, 0.0)
            if expected - actual > 0.1:  # Minimum gap threshold
                gaps[direction] = {
                    'expected': expected,
                    'actual': actual,
                    'gap_size': expected - actual
                }
        
        # Create conductive gap
        gap = ConductiveGap(
            proposition=self,
            target_domain=target_domain,
            expected_flows=expected_flows,
            actual_flows=actual_flows,
            gaps=gaps
        )
        
        return gap
    
    def _calculate_domain_compatibility(self, domain: Dict[str, Any]) -> float:
        """
        Calculate compatibility of this proposition with a target domain.
        
        Args:
            domain: The target domain
            
        Returns:
            A compatibility score between 0 and 1
        """
        # Simple implementation - can be enhanced
        # Check actant overlap
        domain_actants = set(domain.get('actants', []))
        shared_actants = domain_actants.intersection(self.constituent_actants)
        
        if not domain_actants:
            actant_score = 0.0
        else:
            actant_score = len(shared_actants) / len(self.constituent_actants)
        
        # Check verb/direction overlap
        domain_verbs = set(domain.get('verbs', []))
        proposition_verbs = set(self.directionality.keys())
        shared_verbs = domain_verbs.intersection(proposition_verbs)
        
        if not proposition_verbs:
            verb_score = 0.0
        else:
            verb_score = len(shared_verbs) / len(proposition_verbs)
        
        # Combined score with weights
        return 0.7 * actant_score + 0.3 * verb_score
    
    def _project_flows(self, domain: Dict[str, Any]) -> Dict[str, float]:
        """
        Project expected semantic flows into a target domain.
        
        Args:
            domain: The target domain
            
        Returns:
            Dictionary of expected flow strengths by direction
        """
        # Calculate domain compatibility
        compatibility = self._calculate_domain_compatibility(domain)
        
        # Adjust flow strengths based on compatibility
        projected_flows = {}
        for direction, strength in self.directionality.items():
            # Adjust strength based on compatibility and domain context
            context_factor = self._calculate_context_factor(direction, domain)
            projected_flows[direction] = strength * compatibility * context_factor
            
        return projected_flows
    
    def _calculate_context_factor(self, direction: str, domain: Dict[str, Any]) -> float:
        """
        Calculate a context factor for a direction in a domain.
        
        Args:
            direction: The semantic direction
            domain: The target domain
            
        Returns:
            A context factor between 0 and 1.5
        """
        # Simple implementation - can be enhanced
        # Check if direction is mentioned in domain context
        domain_context = domain.get('context', {})
        context_description = domain_context.get('description', '') if isinstance(domain_context, dict) else ''
        
        if direction.lower() in context_description.lower():
            return 1.2  # Boost if explicitly mentioned
        
        # Check if semantically related terms are present
        # This could be enhanced with word embeddings or semantic similarity
        related_terms = {
            'adapts': ['adapt', 'adjustment', 'flexibility', 'response'],
            'supports': ['support', 'assist', 'help', 'aid', 'enable'],
            'fails': ['fail', 'failure', 'breakdown', 'collapse'],
            'funds': ['fund', 'finance', 'invest', 'allocate'],
            'invests': ['invest', 'investment', 'funding', 'capital'],
            'regulates': ['regulate', 'regulation', 'control', 'govern'],
            'influences': ['influence', 'impact', 'affect', 'shape'],
            'damages': ['damage', 'harm', 'destroy', 'impair'],
            'protects': ['protect', 'protection', 'shield', 'defend']
        }
        
        if direction in related_terms:
            for term in related_terms[direction]:
                if term in domain_context:
                    return 1.1  # Slight boost for related terms
        
        return 1.0  # Default factor
    
    def _extract_actual_flows(self, domain: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract actual semantic flows in a target domain.
        
        Args:
            domain: The target domain
            
        Returns:
            Dictionary of actual flow strengths by direction
        """
        # Extract from domain if available
        if 'semantic_flows' in domain:
            return domain['semantic_flows'].copy()
        
        # Otherwise, estimate from domain predicates
        actual_flows = {}
        predicates = domain.get('predicates', [])
        
        # Count verb occurrences in predicates
        verb_counts = {}
        total_count = 0
        
        for predicate in predicates:
            verb = predicate.get('verb', '')
            if verb:
                verb_counts[verb] = verb_counts.get(verb, 0) + 1
                total_count += 1
        
        # Normalize to get flow strengths
        if total_count > 0:
            for verb, count in verb_counts.items():
                actual_flows[verb] = count / total_count
                
        return actual_flows


class PropositionInstance:
    """
    An instance of a semantic proposition in a specific domain.
    """
    
    def __init__(self, proposition: SemanticProposition, 
                 target_domain: Dict[str, Any],
                 compatibility: float):
        """
        Initialize a proposition instance.
        
        Args:
            proposition: The source proposition
            target_domain: The domain this is instantiated in
            compatibility: Compatibility score with the domain
        """
        self.proposition = proposition
        self.target_domain = target_domain
        self.compatibility = compatibility
        
        # Adjusted properties based on domain
        self.adjusted_directionality = self._adjust_directionality()
        self.instantiation_strength = self._calculate_instantiation_strength()
        
    def _adjust_directionality(self) -> Dict[str, float]:
        """
        Adjust directionality based on target domain.
        
        Returns:
            Adjusted directionality dictionary
        """
        adjusted = {}
        domain_context = self.target_domain.get('context', {})
        
        for direction, strength in self.proposition.directionality.items():
            # Adjust based on domain context and compatibility
            context_factor = self._calculate_context_adjustment(direction, domain_context)
            adjusted[direction] = strength * self.compatibility * context_factor
            
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            for direction in adjusted:
                adjusted[direction] /= total
                
        return adjusted
    
    def _calculate_context_adjustment(self, direction: str, context: Dict) -> float:
        """
        Calculate adjustment factor based on context.
        
        Args:
            direction: The semantic direction
            context: The domain context
            
        Returns:
            An adjustment factor
        """
        # Simple implementation - can be enhanced
        # Default adjustment
        adjustment = 1.0
        
        # Handle string or dict context
        if not isinstance(context, dict):
            return adjustment
            
        # Adjust based on domain emphasis
        emphasis = context.get('emphasis', {})
        if direction in emphasis:
            adjustment *= 1.0 + emphasis[direction]
        
        # Adjust based on domain constraints
        constraints = context.get('constraints', {})
        if direction in constraints:
            adjustment *= 1.0 - constraints[direction]
            
        return max(0.1, min(adjustment, 2.0))  # Clamp between 0.1 and 2.0
    
    def _calculate_instantiation_strength(self) -> float:
        """
        Calculate the overall strength of this instantiation.
        
        Returns:
            Instantiation strength between 0 and 1
        """
        # Weighted combination of compatibility and capaciousness
        return 0.7 * self.compatibility + 0.3 * self.proposition.capaciousness
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute this proposition instance to generate effects in the domain.
        
        Returns:
            Dictionary of effects generated by this execution
        """
        # Only execute if strong enough
        if self.instantiation_strength < 0.3:
            return {'status': 'too_weak', 'effects': {}}
        
        # Generate effects based on proposition code
        effects = {}
        
        # Apply primary verb effect
        primary_verb = max(self.adjusted_directionality.items(), key=lambda x: x[1])[0]
        effects[primary_verb] = {
            'strength': self.adjusted_directionality[primary_verb],
            'targets': self._identify_effect_targets(primary_verb),
            'description': f"Instantiated '{primary_verb}' flow with strength {self.adjusted_directionality[primary_verb]:.2f}"
        }
        
        # Apply secondary effects
        for verb, strength in sorted(self.adjusted_directionality.items(), 
                                    key=lambda x: x[1], reverse=True)[1:3]:  # Next 2 strongest
            if strength > 0.1:  # Minimum threshold for secondary effects
                effects[verb] = {
                    'strength': strength,
                    'targets': self._identify_effect_targets(verb),
                    'description': f"Instantiated '{verb}' flow with strength {strength:.2f}"
                }
        
        return {
            'status': 'executed',
            'instantiation_strength': self.instantiation_strength,
            'effects': effects
        }
    
    def _identify_effect_targets(self, verb: str) -> List[str]:
        """
        Identify targets for a verb effect.
        
        Args:
            verb: The verb to identify targets for
            
        Returns:
            List of target actants
        """
        # Simple implementation - can be enhanced
        targets = []
        
        # Get actants from domain
        domain_actants = self.target_domain.get('actants', [])
        
        # Get proposition actants
        prop_actants = list(self.proposition.constituent_actants)
        
        # Prioritize shared actants
        shared = [a for a in domain_actants if a in prop_actants]
        if shared:
            targets.extend(shared[:2])  # Up to 2 shared actants
        
        # Add domain-specific actants if needed
        if len(targets) < 2 and domain_actants:
            for actant in domain_actants:
                if actant not in targets:
                    targets.append(actant)
                    if len(targets) >= 2:
                        break
        
        return targets


class ConductiveGap:
    """
    A conductive gap identified through negative instantiation.
    
    Conductive gaps represent areas where semantic flows are expected
    but absent, creating potential for meaning to flow into.
    """
    
    def __init__(self, proposition: SemanticProposition,
                 target_domain: Dict[str, Any],
                 expected_flows: Dict[str, float],
                 actual_flows: Dict[str, float],
                 gaps: Dict[str, Dict[str, float]]):
        """
        Initialize a conductive gap.
        
        Args:
            proposition: The source proposition
            target_domain: The domain with the gap
            expected_flows: Expected semantic flows
            actual_flows: Actual semantic flows
            gaps: Dictionary of identified gaps
        """
        self.proposition = proposition
        self.target_domain = target_domain
        self.expected_flows = expected_flows
        self.actual_flows = actual_flows
        self.gaps = gaps
        
        # Derived properties
        self.conductivity = self._calculate_conductivity()
        self.primary_gap = self._identify_primary_gap()
        
    def _calculate_conductivity(self) -> float:
        """
        Calculate the overall conductivity of this gap.
        
        Higher conductivity means greater potential for meaning to flow.
        
        Returns:
            Conductivity score between 0 and 1
        """
        if not self.gaps:
            return 0.0
        
        # Sum of gap sizes weighted by expected flow strength
        weighted_sum = sum(gap['gap_size'] * self.expected_flows[direction]
                          for direction, gap in self.gaps.items())
        
        # Normalize by sum of expected flows
        total_expected = sum(self.expected_flows.values())
        if total_expected > 0:
            return min(1.0, weighted_sum / total_expected)
        return 0.0
    
    def _identify_primary_gap(self) -> Tuple[str, Dict[str, float]]:
        """
        Identify the primary gap - the one with highest potential.
        
        Returns:
            Tuple of (direction, gap_info)
        """
        if not self.gaps:
            return ('', {})
        
        # Find gap with highest product of gap size and expected flow
        primary = max(self.gaps.items(),
                     key=lambda x: x[1]['gap_size'] * self.expected_flows[x[0]])
        
        return primary
    
    def induce_behavior_propensities(self) -> Dict[str, Any]:
        """
        Induce behavior propensities from this conductive gap.
        
        Returns:
            Dictionary of induced propensities
        """
        if self.conductivity < 0.2:
            return {'status': 'insufficient_conductivity', 'propensities': {}}
        
        propensities = {}
        
        # Generate propensities for each significant gap
        for direction, gap_info in sorted(self.gaps.items(),
                                         key=lambda x: x[1]['gap_size'] * self.expected_flows[x[0]],
                                         reverse=True):
            # Calculate propensity strength
            strength = gap_info['gap_size'] * self.expected_flows[direction] * self.conductivity
            
            if strength > 0.1:  # Minimum threshold for propensities
                propensities[direction] = {
                    'strength': strength,
                    'targets': self._identify_propensity_targets(direction),
                    'description': f"Induced '{direction}' propensity with strength {strength:.2f}"
                }
        
        return {
            'status': 'induced',
            'conductivity': self.conductivity,
            'propensities': propensities
        }
    
    def _identify_propensity_targets(self, direction: str) -> List[str]:
        """
        Identify targets for a propensity.
        
        Args:
            direction: The direction to identify targets for
            
        Returns:
            List of target actants
        """
        # Similar to effect targets but prioritizing proposition actants
        targets = []
        
        # Get actants from domain
        domain_actants = self.target_domain.get('actants', [])
        
        # Get proposition actants
        prop_actants = list(self.proposition.constituent_actants)
        
        # Prioritize proposition actants that are also in domain
        shared = [a for a in prop_actants if a in domain_actants]
        if shared:
            targets.extend(shared[:2])  # Up to 2 shared actants
        
        # Add proposition-specific actants if needed
        if len(targets) < 2 and prop_actants:
            for actant in prop_actants:
                if actant not in targets:
                    targets.append(actant)
                    if len(targets) >= 2:
                        break
        
        return targets


class PropositionPatternRegistry:
    """
    Registry for semantic proposition patterns.
    
    Maintains a collection of proposition patterns derived from
    conceptual frameworks and provides methods to query and apply them.
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self.propositions = {}
        self.proposition_vectors = []
        self.proposition_keys = []
        
    def register_proposition(self, framework: ConceptualFramework) -> SemanticProposition:
        """
        Register a new proposition from a conceptual framework.
        
        Args:
            framework: The conceptual framework to derive the proposition from
            
        Returns:
            The created semantic proposition
        """
        # Create proposition
        proposition = SemanticProposition(framework)
        
        # Generate a unique key
        key = f"{proposition.name}_{len(self.propositions)}"
        
        # Store proposition
        self.propositions[key] = proposition
        self.proposition_vectors.append(proposition.flow_signature)
        self.proposition_keys.append(key)
        
        return proposition
    
    def register_batch(self, frameworks: List[ConceptualFramework]) -> List[SemanticProposition]:
        """
        Register multiple propositions from a list of frameworks.
        
        Args:
            frameworks: List of conceptual frameworks
            
        Returns:
            List of created semantic propositions
        """
        return [self.register_proposition(f) for f in frameworks]
    
    def find_similar_propositions(self, query_proposition: SemanticProposition, 
                                 threshold: float = 0.7,
                                 max_results: int = 5) -> List[Tuple[str, SemanticProposition, float]]:
        """
        Find propositions similar to a query proposition.
        
        Args:
            query_proposition: The proposition to find similar ones to
            threshold: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of tuples (key, proposition, similarity)
        """
        if not self.proposition_vectors:
            return []
        
        # Convert list to array for vectorized operations
        prop_matrix = np.array(self.proposition_vectors)
        
        # Calculate cosine similarities
        query_vector = query_proposition.flow_signature
        similarities = np.dot(prop_matrix, query_vector)
        
        # Find propositions above threshold
        results = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                key = self.proposition_keys[i]
                results.append((key, self.propositions[key], float(sim)))
        
        # Sort by similarity (descending) and limit results
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]
    
    def find_propositions_for_domain(self, domain: Dict[str, Any],
                                    min_compatibility: float = 0.5,
                                    max_results: int = 3) -> List[Tuple[str, SemanticProposition, float]]:
        """
        Find propositions compatible with a domain.
        
        Args:
            domain: The target domain
            min_compatibility: Minimum compatibility threshold
            max_results: Maximum number of results
            
        Returns:
            List of tuples (key, proposition, compatibility)
        """
        results = []
        
        for key, prop in self.propositions.items():
            compatibility = prop._calculate_domain_compatibility(domain)
            if compatibility >= min_compatibility:
                results.append((key, prop, compatibility))
        
        # Sort by compatibility (descending) and limit results
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]
    
    def instantiate_in_domain(self, domain: Dict[str, Any],
                             min_compatibility: float = 0.5) -> List[PropositionInstance]:
        """
        Instantiate compatible propositions in a domain.
        
        Args:
            domain: The target domain
            min_compatibility: Minimum compatibility threshold
            
        Returns:
            List of proposition instances
        """
        # Find compatible propositions
        compatible_props = self.find_propositions_for_domain(
            domain, min_compatibility)
        
        # Instantiate them
        instances = []
        for _, prop, compatibility in compatible_props:
            instance = prop.instantiate(domain)
            instances.append(instance)
            
        return instances
    
    def find_conductive_gaps(self, domain: Dict[str, Any],
                            min_conductivity: float = 0.3) -> List[ConductiveGap]:
        """
        Find conductive gaps in a domain.
        
        Args:
            domain: The target domain
            min_conductivity: Minimum conductivity threshold
            
        Returns:
            List of conductive gaps
        """
        gaps = []
        
        for key, prop in self.propositions.items():
            gap = prop.negative_instantiate(domain)
            if gap.conductivity >= min_conductivity:
                gaps.append(gap)
        
        # Sort by conductivity (descending)
        gaps.sort(key=lambda x: x.conductivity, reverse=True)
        return gaps
