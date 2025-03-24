"""
Multi-Proposition Dynamics Module

This module implements the dynamics of multiple propositions interacting within a domain,
capturing emergent behaviors that arise from proposition interactions.
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
import networkx as nx
from enum import Enum

from habitat_evolution.adaptive_core.transformation.semantic_proposition_patterns import SemanticProposition
from habitat_evolution.adaptive_core.emergence.emergent_propensity import EmergentPropensity


class InteractionType(Enum):
    """Types of interactions between propositions."""
    REINFORCING = "reinforcing"  # Propositions strengthen each other
    CONFLICTING = "conflicting"  # Propositions weaken each other
    CATALYZING = "catalyzing"    # One proposition enables another
    INHIBITING = "inhibiting"    # One proposition suppresses another
    TRANSFORMING = "transforming"  # One proposition changes another
    NEUTRAL = "neutral"          # No significant interaction


class PropositionInteraction:
    """
    Represents an interaction between two propositions.
    
    Captures how propositions influence each other when co-present in a domain.
    """
    
    def __init__(self, source_prop: EmergentPropensity, target_prop: EmergentPropensity):
        """
        Initialize a proposition interaction.
        
        Args:
            source_prop: The source proposition propensity
            target_prop: The target proposition propensity
        """
        self.source = source_prop
        self.target = target_prop
        self.domain_context = source_prop.domain_context  # Assuming same domain
        
        # Calculate interaction properties
        self.interaction_type = self._determine_interaction_type()
        self.interaction_strength = self._calculate_interaction_strength()
        self.effects = self._calculate_effects()
    
    def _determine_interaction_type(self) -> InteractionType:
        """
        Determine the type of interaction between the propositions.
        
        Returns:
            The interaction type
        """
        # Compare directionality to determine interaction type
        source_dirs = self.source.state_change_vectors
        target_dirs = self.target.state_change_vectors
        
        # Calculate directional alignment
        alignment = 0
        total_weight = 0
        
        for direction in set(source_dirs.keys()).union(target_dirs.keys()):
            source_weight = source_dirs.get(direction, 0)
            target_weight = target_dirs.get(direction, 0)
            
            # Weighted contribution to alignment
            weight = max(source_weight, target_weight)
            if source_weight > 0 and target_weight > 0:
                # Both have this direction - reinforcing
                alignment += weight
            elif source_weight > 0 and target_weight == 0:
                # Source has direction, target doesn't - potential catalyzing
                alignment += 0  # Neutral for alignment
            elif source_weight == 0 and target_weight > 0:
                # Target has direction, source doesn't - potential catalyzing
                alignment += 0  # Neutral for alignment
            
            total_weight += weight
        
        # Normalize alignment
        if total_weight > 0:
            alignment_score = alignment / total_weight
        else:
            alignment_score = 0
            
        # Check for opposing directions (conflicting)
        opposing_dirs = {
            "supports": "undermines",
            "adapts": "fails",
            "funds": "withdraws",
            "invests": "divests",
            "protects": "threatens"
        }
        
        for dir1, dir2 in opposing_dirs.items():
            if (dir1 in source_dirs and dir2 in target_dirs) or (dir2 in source_dirs and dir1 in target_dirs):
                if source_dirs.get(dir1, 0) * target_dirs.get(dir2, 0) > 0 or \
                   source_dirs.get(dir2, 0) * target_dirs.get(dir1, 0) > 0:
                    return InteractionType.CONFLICTING
        
        # Determine type based on alignment score
        if alignment_score > 0.7:
            return InteractionType.REINFORCING
        elif alignment_score < 0.3:
            # Further differentiate between inhibiting and transforming
            if self._has_transformation_potential():
                return InteractionType.TRANSFORMING
            else:
                return InteractionType.INHIBITING
        else:
            # Check for catalyzing relationship
            if self._has_catalyzing_potential():
                return InteractionType.CATALYZING
            else:
                return InteractionType.NEUTRAL
    
    def _has_transformation_potential(self) -> bool:
        """
        Check if there's potential for transformation between propositions.
        
        Returns:
            True if transformation potential exists
        """
        # Simple implementation - can be enhanced
        source_dirs = set(self.source.state_change_vectors.keys())
        target_dirs = set(self.target.state_change_vectors.keys())
        
        # If source has directions that target doesn't, potential transformation
        return len(source_dirs.difference(target_dirs)) > 0
    
    def _has_catalyzing_potential(self) -> bool:
        """
        Check if there's potential for catalysis between propositions.
        
        Returns:
            True if catalyzing potential exists
        """
        # Simple implementation - can be enhanced
        # If source's primary direction enables target's manifestation
        source_primary = max(self.source.state_change_vectors.items(), key=lambda x: x[1])[0]
        
        # Catalyzing relationships
        catalysts = {
            "funds": ["adapts", "supports", "invests"],
            "supports": ["adapts", "protects"],
            "adapts": ["protects", "survives"]
        }
        
        if source_primary in catalysts:
            target_dirs = set(self.target.state_change_vectors.keys())
            return any(d in target_dirs for d in catalysts[source_primary])
            
        return False
    
    def _calculate_interaction_strength(self) -> float:
        """
        Calculate the strength of the interaction.
        
        Returns:
            Interaction strength between 0 and 1
        """
        # Base strength from manifestation potentials
        base_strength = (self.source.manifestation_potential + self.target.manifestation_potential) / 2
        
        # Adjust based on interaction type
        type_multipliers = {
            InteractionType.REINFORCING: 1.2,
            InteractionType.CONFLICTING: 1.1,
            InteractionType.CATALYZING: 1.3,
            InteractionType.INHIBITING: 0.9,
            InteractionType.TRANSFORMING: 1.4,
            InteractionType.NEUTRAL: 0.7
        }
        
        adjusted_strength = base_strength * type_multipliers[self.interaction_type]
        
        # Adjust based on actant overlap
        source_actants = set(self.source.source_proposition.constituent_actants)
        target_actants = set(self.target.source_proposition.constituent_actants)
        
        if source_actants and target_actants:
            overlap_ratio = len(source_actants.intersection(target_actants)) / len(source_actants.union(target_actants))
            adjusted_strength *= (0.7 + (0.6 * overlap_ratio))
        
        return min(adjusted_strength, 1.0)
    
    def _calculate_effects(self) -> Dict[str, Any]:
        """
        Calculate the effects of this interaction.
        
        Returns:
            Dictionary of effects
        """
        effects = {
            "type": self.interaction_type.value,
            "strength": self.interaction_strength,
            "source_to_target": {},
            "target_to_source": {},
            "emergent_directions": {}
        }
        
        # Calculate effects based on interaction type
        if self.interaction_type == InteractionType.REINFORCING:
            # Reinforcing interactions strengthen shared directions
            source_dirs = self.source.state_change_vectors
            target_dirs = self.target.state_change_vectors
            
            for direction in set(source_dirs.keys()).intersection(target_dirs.keys()):
                source_weight = source_dirs[direction]
                target_weight = target_dirs[direction]
                
                # Mutual reinforcement
                boost = self.interaction_strength * 0.3
                effects["source_to_target"][direction] = boost * source_weight
                effects["target_to_source"][direction] = boost * target_weight
                
                # Emergent strengthening of shared direction
                effects["emergent_directions"][direction] = (source_weight + target_weight) / 2 * (1 + boost)
                
        elif self.interaction_type == InteractionType.CONFLICTING:
            # Conflicting interactions weaken opposing directions
            source_dirs = self.source.state_change_vectors
            target_dirs = self.target.state_change_vectors
            
            opposing_dirs = {
                "supports": "undermines",
                "adapts": "fails",
                "funds": "withdraws",
                "invests": "divests",
                "protects": "threatens"
            }
            
            for dir1, dir2 in opposing_dirs.items():
                if dir1 in source_dirs and dir2 in target_dirs:
                    inhibition = self.interaction_strength * 0.4
                    effects["source_to_target"][dir2] = -inhibition * source_dirs[dir1]
                
                if dir2 in source_dirs and dir1 in target_dirs:
                    inhibition = self.interaction_strength * 0.4
                    effects["source_to_target"][dir1] = -inhibition * source_dirs[dir2]
                    
        elif self.interaction_type == InteractionType.CATALYZING:
            # Catalyzing interactions enable or amplify certain directions
            source_dirs = self.source.state_change_vectors
            target_dirs = self.target.state_change_vectors
            
            # Catalyzing relationships
            catalysts = {
                "funds": ["adapts", "supports", "invests"],
                "supports": ["adapts", "protects"],
                "adapts": ["protects", "survives"]
            }
            
            for catalyst, enabled in catalysts.items():
                if catalyst in source_dirs:
                    catalyst_strength = source_dirs[catalyst]
                    for direction in enabled:
                        if direction in target_dirs:
                            boost = self.interaction_strength * 0.5 * catalyst_strength
                            effects["source_to_target"][direction] = boost
                            
        elif self.interaction_type == InteractionType.INHIBITING:
            # Inhibiting interactions suppress certain directions
            source_dirs = self.source.state_change_vectors
            target_dirs = self.target.state_change_vectors
            
            # Simple inhibition - suppress all directions
            inhibition = self.interaction_strength * 0.3
            for direction, weight in target_dirs.items():
                effects["source_to_target"][direction] = -inhibition
                
        elif self.interaction_type == InteractionType.TRANSFORMING:
            # Transforming interactions change directions
            source_dirs = self.source.state_change_vectors
            target_dirs = self.target.state_change_vectors
            
            # Source directions not in target can emerge as new directions
            for direction, weight in source_dirs.items():
                if direction not in target_dirs:
                    transform_strength = weight * self.interaction_strength * 0.4
                    effects["emergent_directions"][direction] = transform_strength
        
        return effects
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this interaction.
        
        Returns:
            Dictionary summarizing the interaction
        """
        return {
            "source_proposition": self.source.source_proposition.name,
            "target_proposition": self.target.source_proposition.name,
            "interaction_type": self.interaction_type.value,
            "interaction_strength": self.interaction_strength,
            "effects": self.effects,
            "description": self._generate_description()
        }
    
    def _generate_description(self) -> str:
        """
        Generate a human-readable description of the interaction.
        
        Returns:
            Description string
        """
        source_name = self.source.source_proposition.name
        target_name = self.target.source_proposition.name
        
        if self.interaction_type == InteractionType.REINFORCING:
            return f"{source_name} reinforces {target_name}, strengthening shared semantic directions."
            
        elif self.interaction_type == InteractionType.CONFLICTING:
            return f"{source_name} conflicts with {target_name}, weakening opposing semantic directions."
            
        elif self.interaction_type == InteractionType.CATALYZING:
            return f"{source_name} catalyzes {target_name}, enabling or amplifying certain semantic directions."
            
        elif self.interaction_type == InteractionType.INHIBITING:
            return f"{source_name} inhibits {target_name}, suppressing its semantic directions."
            
        elif self.interaction_type == InteractionType.TRANSFORMING:
            return f"{source_name} transforms {target_name}, introducing new semantic directions."
            
        else:  # NEUTRAL
            return f"{source_name} and {target_name} coexist with minimal interaction."


class PropositionEcosystem:
    """
    Represents an ecosystem of interacting propositions within a domain.
    
    Captures the collective dynamics of multiple propositions, including
    emergent behaviors that arise from their interactions.
    """
    
    def __init__(self, domain_context: Dict[str, Any]):
        """
        Initialize a proposition ecosystem.
        
        Args:
            domain_context: The domain context in which propositions interact
        """
        self.domain_context = domain_context
        self.propositions = []
        self.interactions = []
        self.interaction_network = nx.DiGraph()
    
    def add_proposition(self, proposition: SemanticProposition):
        """
        Add a proposition to the ecosystem.
        
        Args:
            proposition: The proposition to add
        """
        # Create emergent propensity for this proposition in this domain
        propensity = EmergentPropensity(proposition, self.domain_context)
        self.propositions.append(propensity)
        
        # Add to interaction network
        self.interaction_network.add_node(proposition.name, propensity=propensity)
        
        # Create interactions with existing propositions
        for existing_prop in self.propositions[:-1]:  # All except the one just added
            interaction = PropositionInteraction(propensity, existing_prop)
            self.interactions.append(interaction)
            
            # Add to interaction network
            self.interaction_network.add_edge(
                propensity.source_proposition.name,
                existing_prop.source_proposition.name,
                interaction=interaction
            )
            
            # Add reverse interaction
            reverse_interaction = PropositionInteraction(existing_prop, propensity)
            self.interactions.append(reverse_interaction)
            
            self.interaction_network.add_edge(
                existing_prop.source_proposition.name,
                propensity.source_proposition.name,
                interaction=reverse_interaction
            )
    
    def calculate_emergent_directions(self) -> Dict[str, float]:
        """
        Calculate the emergent semantic directions from all interactions.
        
        Returns:
            Dictionary mapping directions to their emergent weights
        """
        emergent_dirs = {}
        
        # Collect base directions from all propositions
        for prop in self.propositions:
            for direction, weight in prop.state_change_vectors.items():
                if direction in emergent_dirs:
                    emergent_dirs[direction] += weight / len(self.propositions)
                else:
                    emergent_dirs[direction] = weight / len(self.propositions)
        
        # Apply interaction effects
        for interaction in self.interactions:
            for direction, effect in interaction.effects.get("emergent_directions", {}).items():
                if direction in emergent_dirs:
                    emergent_dirs[direction] += effect / len(self.interactions)
                else:
                    emergent_dirs[direction] = effect / len(self.interactions)
        
        # Normalize
        total = sum(emergent_dirs.values())
        if total > 0:
            emergent_dirs = {d: w/total for d, w in emergent_dirs.items()}
            
        return emergent_dirs
    
    def calculate_ecosystem_stability(self) -> float:
        """
        Calculate the stability of the proposition ecosystem.
        
        Returns:
            Stability index between 0 and 1
        """
        if not self.interactions:
            return 1.0  # Single proposition or empty ecosystem is stable
            
        # Calculate based on interaction types
        type_weights = {
            InteractionType.REINFORCING: 1.0,
            InteractionType.CATALYZING: 0.8,
            InteractionType.NEUTRAL: 0.7,
            InteractionType.TRANSFORMING: 0.5,
            InteractionType.INHIBITING: 0.3,
            InteractionType.CONFLICTING: 0.2
        }
        
        stability = 0
        total_weight = 0
        
        for interaction in self.interactions:
            weight = interaction.interaction_strength
            stability += type_weights[interaction.interaction_type] * weight
            total_weight += weight
            
        if total_weight > 0:
            return stability / total_weight
        else:
            return 1.0
    
    def calculate_ecosystem_capaciousness(self) -> float:
        """
        Calculate the capaciousness of the proposition ecosystem.
        
        Returns:
            Capaciousness index between 0 and 1
        """
        if not self.propositions:
            return 0.0
            
        # Base capaciousness from individual propositions
        base_capaciousness = sum(p.capaciousness for p in self.propositions) / len(self.propositions)
        
        # Adjust based on interaction diversity
        interaction_types = [i.interaction_type for i in self.interactions]
        type_diversity = len(set(interaction_types)) / len(InteractionType) if self.interactions else 0
        
        # Adjust based on emergent directions
        emergent_dirs = self.calculate_emergent_directions()
        direction_diversity = len(emergent_dirs) / 10  # Normalize assuming max ~10 directions
        
        # Combine factors
        return (base_capaciousness * 0.4) + (type_diversity * 0.3) + (direction_diversity * 0.3)
    
    def get_dominant_propositions(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get the most dominant propositions in the ecosystem.
        
        Args:
            top_n: Number of top propositions to return
            
        Returns:
            List of dominant proposition information
        """
        if not self.propositions:
            return []
            
        # Calculate dominance scores
        dominance_scores = {}
        
        for prop in self.propositions:
            name = prop.source_proposition.name
            
            # Base score from manifestation potential
            score = prop.manifestation_potential
            
            # Adjust based on interactions
            for interaction in self.interactions:
                if interaction.source.source_proposition.name == name:
                    # This proposition is the source
                    target = interaction.target.source_proposition.name
                    
                    # Add influence on other propositions
                    if interaction.interaction_type in [InteractionType.REINFORCING, InteractionType.CATALYZING, InteractionType.TRANSFORMING]:
                        score += 0.1 * interaction.interaction_strength
                    elif interaction.interaction_type in [InteractionType.INHIBITING, InteractionType.CONFLICTING]:
                        score += 0.05 * interaction.interaction_strength
            
            dominance_scores[name] = score
            
        # Sort by score
        sorted_props = sorted(dominance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        result = []
        for name, score in sorted_props[:top_n]:
            prop = next(p for p in self.propositions if p.source_proposition.name == name)
            result.append({
                "name": name,
                "dominance_score": score,
                "manifestation_potential": prop.manifestation_potential,
                "primary_direction": max(prop.state_change_vectors.items(), key=lambda x: x[1])[0]
            })
            
        return result
    
    def get_ecosystem_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the proposition ecosystem.
        
        Returns:
            Dictionary summarizing the ecosystem
        """
        emergent_dirs = self.calculate_emergent_directions()
        primary_direction = max(emergent_dirs.items(), key=lambda x: x[1])[0] if emergent_dirs else None
        
        return {
            "proposition_count": len(self.propositions),
            "interaction_count": len(self.interactions),
            "stability_index": self.calculate_ecosystem_stability(),
            "capaciousness_index": self.calculate_ecosystem_capaciousness(),
            "emergent_directions": emergent_dirs,
            "primary_direction": primary_direction,
            "dominant_propositions": self.get_dominant_propositions(),
            "interaction_types": {
                t.value: sum(1 for i in self.interactions if i.interaction_type == t)
                for t in InteractionType
            }
        }
