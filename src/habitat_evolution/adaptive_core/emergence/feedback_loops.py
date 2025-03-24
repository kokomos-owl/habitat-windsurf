"""
Feedback Loops Module

This module implements feedback loops between propositions and domains,
capturing how instantiated propositions affect their domains over time
and how these changes in turn affect proposition manifestation.
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
import copy
from enum import Enum

from habitat_evolution.adaptive_core.transformation.semantic_proposition_patterns import SemanticProposition, PropositionInstance
from habitat_evolution.adaptive_core.emergence.emergent_propensity import EmergentPropensity
from habitat_evolution.adaptive_core.emergence.multi_proposition_dynamics import PropositionEcosystem


class FeedbackType(Enum):
    """Types of feedback loops."""
    POSITIVE = "positive"  # Self-reinforcing feedback
    NEGATIVE = "negative"  # Self-regulating feedback
    OSCILLATING = "oscillating"  # Alternating feedback
    COMPLEX = "complex"  # Complex, multi-factor feedback


class DomainChangeVector:
    """
    Represents a vector of changes to a domain.
    
    Captures how a domain changes in response to proposition instantiation.
    """
    
    def __init__(self, domain_context: Dict[str, Any]):
        """
        Initialize a domain change vector.
        
        Args:
            domain_context: The original domain context
        """
        self.original_domain = copy.deepcopy(domain_context)
        self.changes = {
            "actants": {},  # Actant presence changes
            "emphasis": {},  # Emphasis changes
            "semantic_flows": {},  # Semantic flow changes
            "predicates": []  # New predicates
        }
    
    def add_actant_change(self, actant: str, change: float):
        """
        Add a change to an actant's presence.
        
        Args:
            actant: The actant name
            change: The change in presence (-1 to 1)
        """
        if actant in self.changes["actants"]:
            self.changes["actants"][actant] += change
        else:
            self.changes["actants"][actant] = change
    
    def add_emphasis_change(self, emphasis: str, change: float):
        """
        Add a change to a domain emphasis.
        
        Args:
            emphasis: The emphasis name
            change: The change in emphasis (-1 to 1)
        """
        if emphasis in self.changes["emphasis"]:
            self.changes["emphasis"][emphasis] += change
        else:
            self.changes["emphasis"][emphasis] = change
    
    def add_semantic_flow_change(self, flow: str, change: float):
        """
        Add a change to a semantic flow.
        
        Args:
            flow: The flow name
            change: The change in flow strength (-1 to 1)
        """
        if flow in self.changes["semantic_flows"]:
            self.changes["semantic_flows"][flow] += change
        else:
            self.changes["semantic_flows"][flow] = change
    
    def add_predicate(self, predicate: Dict[str, str]):
        """
        Add a new predicate to the domain.
        
        Args:
            predicate: Dictionary with subject, verb, object
        """
        self.changes["predicates"].append(predicate)
    
    def apply_to_domain(self) -> Dict[str, Any]:
        """
        Apply changes to create a new domain context.
        
        Returns:
            The updated domain context
        """
        # Start with a copy of the original
        new_domain = copy.deepcopy(self.original_domain)
        
        # Apply actant changes
        domain_actants = set(new_domain.get("actants", []))
        for actant, change in self.changes["actants"].items():
            if change > 0.5 and actant not in domain_actants:
                # Add new actant
                domain_actants.add(actant)
            elif change < -0.5 and actant in domain_actants:
                # Remove actant
                domain_actants.remove(actant)
        new_domain["actants"] = list(domain_actants)
        
        # Apply emphasis changes
        if "context" not in new_domain:
            new_domain["context"] = {}
        if "emphasis" not in new_domain["context"]:
            new_domain["context"]["emphasis"] = {}
            
        for emphasis, change in self.changes["emphasis"].items():
            current = new_domain["context"]["emphasis"].get(emphasis, 0)
            new_value = max(0, min(1, current + change))  # Clamp to 0-1
            new_domain["context"]["emphasis"][emphasis] = new_value
            
        # Apply semantic flow changes
        if "semantic_flows" not in new_domain:
            new_domain["semantic_flows"] = {}
            
        for flow, change in self.changes["semantic_flows"].items():
            current = new_domain["semantic_flows"].get(flow, 0)
            new_value = max(0, min(1, current + change))  # Clamp to 0-1
            new_domain["semantic_flows"][flow] = new_value
            
        # Apply predicate changes
        if "predicates" not in new_domain:
            new_domain["predicates"] = []
            
        for predicate in self.changes["predicates"]:
            # Check if predicate already exists
            exists = False
            for existing in new_domain["predicates"]:
                if (existing.get("subject") == predicate.get("subject") and
                    existing.get("verb") == predicate.get("verb") and
                    existing.get("object") == predicate.get("object")):
                    exists = True
                    break
                    
            if not exists:
                new_domain["predicates"].append(predicate)
                
        return new_domain


class FeedbackLoop:
    """
    Represents a feedback loop between propositions and a domain.
    
    Captures how propositions affect a domain and how these changes
    in turn affect proposition manifestation.
    """
    
    def __init__(self, initial_domain: Dict[str, Any], propositions: List[SemanticProposition]):
        """
        Initialize a feedback loop.
        
        Args:
            initial_domain: The initial domain context
            propositions: List of propositions in the loop
        """
        self.initial_domain = copy.deepcopy(initial_domain)
        self.current_domain = copy.deepcopy(initial_domain)
        self.propositions = propositions
        self.history = []
        self.current_step = 0
        
        # Initialize with current state
        self._record_current_state()
    
    def _record_current_state(self):
        """Record the current state in history."""
        # Create ecosystem for current domain
        ecosystem = PropositionEcosystem(self.current_domain)
        for prop in self.propositions:
            ecosystem.add_proposition(prop)
            
        # Record state
        state = {
            "step": self.current_step,
            "domain": copy.deepcopy(self.current_domain),
            "ecosystem": ecosystem.get_ecosystem_summary(),
            "propensities": {}
        }
        
        # Record propensities
        for prop in self.propositions:
            propensity = EmergentPropensity(prop, self.current_domain)
            state["propensities"][prop.name] = {
                "manifestation_potential": propensity.manifestation_potential,
                "state_change_vectors": propensity.state_change_vectors,
                "community_condition_indices": propensity.community_condition_indices
            }
            
        self.history.append(state)
    
    def step(self, instantiation_threshold: float = 0.5):
        """
        Advance the feedback loop by one step.
        
        Args:
            instantiation_threshold: Threshold for proposition instantiation
        """
        # Create change vector for domain
        change_vector = DomainChangeVector(self.current_domain)
        
        # Process each proposition
        for prop in self.propositions:
            # Calculate propensity in current domain
            propensity = EmergentPropensity(prop, self.current_domain)
            
            # Check if proposition manifests
            if propensity.manifestation_potential >= instantiation_threshold:
                # Instantiate proposition with compatibility parameter
                # Calculate compatibility based on manifestation potential
                compatibility = propensity.manifestation_potential
                instance = PropositionInstance(prop, self.current_domain, compatibility)
                instance.instantiation_strength = propensity.manifestation_potential
                
                # Apply effects to domain
                self._apply_proposition_effects(propensity, change_vector)
        
        # Apply all changes to domain
        self.current_domain = change_vector.apply_to_domain()
        
        # Increment step counter
        self.current_step += 1
        
        # Record new state
        self._record_current_state()
    
    def _apply_proposition_effects(self, propensity: EmergentPropensity, change_vector: DomainChangeVector):
        """
        Apply the effects of a proposition to a domain change vector.
        
        Args:
            propensity: The emergent propensity of the proposition
            change_vector: The domain change vector to modify
        """
        # Get primary direction
        primary_direction = max(propensity.state_change_vectors.items(), key=lambda x: x[1])
        direction = primary_direction[0]
        strength = primary_direction[1] * propensity.manifestation_potential
        
        # Apply effects based on direction
        if direction == "adapts":
            # Adaptation increases adaptive capacity and resilience emphasis
            change_vector.add_emphasis_change("adaptation", strength * 0.2)
            change_vector.add_emphasis_change("resilience", strength * 0.15)
            change_vector.add_semantic_flow_change("adapts", strength * 0.25)
            
            # May add new adaptation-related predicates
            if strength > 0.6:
                change_vector.add_predicate({
                    "subject": "Community",
                    "verb": "adapts to",
                    "object": "changing conditions"
                })
                
        elif direction == "supports":
            # Support increases community emphasis and support flows
            change_vector.add_emphasis_change("community_engagement", strength * 0.2)
            change_vector.add_emphasis_change("social_cohesion", strength * 0.15)
            change_vector.add_semantic_flow_change("supports", strength * 0.25)
            
            # May add new support-related predicates
            if strength > 0.6:
                change_vector.add_predicate({
                    "subject": "Policy",
                    "verb": "supports",
                    "object": "community initiatives"
                })
                
        elif direction == "funds":
            # Funding increases economic emphasis and funding flows
            change_vector.add_emphasis_change("economic_growth", strength * 0.2)
            change_vector.add_emphasis_change("investment", strength * 0.15)
            change_vector.add_semantic_flow_change("funds", strength * 0.25)
            
            # May add new funding-related predicates
            if strength > 0.6:
                change_vector.add_predicate({
                    "subject": "Economy",
                    "verb": "funds",
                    "object": "infrastructure development"
                })
                
        elif direction == "protects":
            # Protection increases safety emphasis and protection flows
            change_vector.add_emphasis_change("safety", strength * 0.2)
            change_vector.add_emphasis_change("protection", strength * 0.15)
            change_vector.add_semantic_flow_change("protects", strength * 0.25)
            
            # May add new protection-related predicates
            if strength > 0.6:
                change_vector.add_predicate({
                    "subject": "Infrastructure",
                    "verb": "protects",
                    "object": "vulnerable assets"
                })
                
        elif direction == "fails":
            # Failure increases vulnerability emphasis and threatening flows
            change_vector.add_emphasis_change("vulnerability", strength * 0.2)
            change_vector.add_emphasis_change("risk", strength * 0.15)
            change_vector.add_semantic_flow_change("threatens", strength * 0.25)
            
            # May add new failure-related predicates
            if strength > 0.6:
                change_vector.add_predicate({
                    "subject": "System",
                    "verb": "fails to protect",
                    "object": "critical assets"
                })
    
    def run_simulation(self, steps: int, instantiation_threshold: float = 0.5):
        """
        Run the feedback loop for multiple steps.
        
        Args:
            steps: Number of steps to run
            instantiation_threshold: Threshold for proposition instantiation
        """
        for _ in range(steps):
            self.step(instantiation_threshold)
    
    def analyze_feedback_type(self) -> FeedbackType:
        """
        Analyze the type of feedback loop.
        
        Returns:
            The feedback type
        """
        if len(self.history) < 3:
            return FeedbackType.COMPLEX  # Not enough data
            
        # Extract manifestation potentials for each proposition
        potentials = {}
        for prop in self.propositions:
            potentials[prop.name] = [
                state["propensities"][prop.name]["manifestation_potential"]
                for state in self.history
            ]
            
        # Analyze trends
        trends = {}
        for name, values in potentials.items():
            # Calculate differences between consecutive steps
            diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
            
            # Check if consistently increasing (positive feedback)
            if all(d > 0 for d in diffs):
                trends[name] = "increasing"
            # Check if consistently decreasing (negative feedback)
            elif all(d < 0 for d in diffs):
                trends[name] = "decreasing"
            # Check if alternating (oscillating feedback)
            elif all(diffs[i] * diffs[i+1] < 0 for i in range(len(diffs)-1)):
                trends[name] = "oscillating"
            else:
                trends[name] = "complex"
                
        # Determine overall feedback type
        if all(t == "increasing" for t in trends.values()):
            return FeedbackType.POSITIVE
        elif all(t == "decreasing" for t in trends.values()):
            return FeedbackType.NEGATIVE
        elif all(t == "oscillating" for t in trends.values()):
            return FeedbackType.OSCILLATING
        else:
            return FeedbackType.COMPLEX
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the feedback loop.
        
        Returns:
            Dictionary summarizing the feedback loop
        """
        if not self.history:
            return {"status": "no_data"}
            
        # Get first and last states
        first_state = self.history[0]
        last_state = self.history[-1]
        
        # Calculate changes in manifestation potentials
        potential_changes = {}
        for prop in self.propositions:
            name = prop.name
            first_potential = first_state["propensities"][name]["manifestation_potential"]
            last_potential = last_state["propensities"][name]["manifestation_potential"]
            potential_changes[name] = last_potential - first_potential
            
        # Calculate changes in domain
        domain_changes = {
            "emphasis": {},
            "semantic_flows": {}
        }
        
        # Emphasis changes
        first_emphasis = first_state["domain"].get("context", {}).get("emphasis", {})
        last_emphasis = last_state["domain"].get("context", {}).get("emphasis", {})
        
        for emphasis in set(first_emphasis.keys()).union(last_emphasis.keys()):
            first_value = first_emphasis.get(emphasis, 0)
            last_value = last_emphasis.get(emphasis, 0)
            domain_changes["emphasis"][emphasis] = last_value - first_value
            
        # Semantic flow changes
        first_flows = first_state["domain"].get("semantic_flows", {})
        last_flows = last_state["domain"].get("semantic_flows", {})
        
        for flow in set(first_flows.keys()).union(last_flows.keys()):
            first_value = first_flows.get(flow, 0)
            last_value = last_flows.get(flow, 0)
            domain_changes["semantic_flows"][flow] = last_value - first_value
            
        return {
            "steps": self.current_step,
            "feedback_type": self.analyze_feedback_type().value,
            "potential_changes": potential_changes,
            "domain_changes": domain_changes,
            "ecosystem_stability_change": last_state["ecosystem"]["stability_index"] - first_state["ecosystem"]["stability_index"],
            "ecosystem_capaciousness_change": last_state["ecosystem"]["capaciousness_index"] - first_state["ecosystem"]["capaciousness_index"]
        }


class PropensityGradient:
    """
    Represents a gradient of propensities across multiple domains.
    
    Captures how propensities vary across different contexts, allowing
    for the visualization of propensity topologies.
    """
    
    def __init__(self, proposition: SemanticProposition, domains: List[Dict[str, Any]]):
        """
        Initialize a propensity gradient.
        
        Args:
            proposition: The proposition to analyze
            domains: List of domains to calculate propensities for
        """
        self.proposition = proposition
        self.domains = domains
        self.propensities = []
        
        # Calculate propensities for each domain
        for domain in domains:
            propensity = EmergentPropensity(proposition, domain)
            self.propensities.append(propensity)
    
    def get_manifestation_gradient(self) -> List[float]:
        """
        Get the gradient of manifestation potentials.
        
        Returns:
            List of manifestation potentials
        """
        return [p.manifestation_potential for p in self.propensities]
    
    def get_direction_gradient(self, direction: str) -> List[float]:
        """
        Get the gradient of a specific direction's weight.
        
        Args:
            direction: The direction to get gradient for
            
        Returns:
            List of direction weights
        """
        return [p.state_change_vectors.get(direction, 0) for p in self.propensities]
    
    def get_condition_gradient(self, condition: str) -> List[float]:
        """
        Get the gradient of a specific community condition.
        
        Args:
            condition: The condition to get gradient for
            
        Returns:
            List of condition indices
        """
        return [p.community_condition_indices.get(condition, 0) for p in self.propensities]
    
    def calculate_gradient_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a gradient.
        
        Args:
            values: List of gradient values
            
        Returns:
            Dictionary of statistics
        """
        if not values:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0,
                "range": 0
            }
            
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "std": np.std(values),
            "range": max(values) - min(values)
        }
    
    def get_gradient_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the propensity gradient.
        
        Returns:
            Dictionary summarizing the gradient
        """
        # Get all directions and conditions
        directions = set()
        conditions = set()
        
        for p in self.propensities:
            directions.update(p.state_change_vectors.keys())
            conditions.update(p.community_condition_indices.keys())
            
        # Calculate gradients
        manifestation_gradient = self.get_manifestation_gradient()
        direction_gradients = {d: self.get_direction_gradient(d) for d in directions}
        condition_gradients = {c: self.get_condition_gradient(c) for c in conditions}
        
        # Calculate statistics
        return {
            "proposition_name": self.proposition.name,
            "domain_count": len(self.domains),
            "manifestation_gradient": {
                "values": manifestation_gradient,
                "stats": self.calculate_gradient_statistics(manifestation_gradient)
            },
            "direction_gradients": {
                d: {
                    "values": g,
                    "stats": self.calculate_gradient_statistics(g)
                }
                for d, g in direction_gradients.items()
            },
            "condition_gradients": {
                c: {
                    "values": g,
                    "stats": self.calculate_gradient_statistics(g)
                }
                for c, g in condition_gradients.items()
            }
        }
