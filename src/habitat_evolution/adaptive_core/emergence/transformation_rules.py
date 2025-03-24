"""
Transformation Rules Module

This module implements sophisticated rules for how semantic directions transform
across domains, accounting for contextual factors, actant relationships, and
domain-specific conditions.
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Callable
import numpy as np


class TransformationRule:
    """
    Represents a rule for how a semantic direction transforms across domains.
    
    Transformation rules capture the grammar of our emergence language - how
    meaning evolves and transforms as it moves between contexts.
    """
    
    def __init__(self, source_direction: str, target_direction: str, 
                 base_weight: float = 0.3, name: str = None):
        """
        Initialize a transformation rule.
        
        Args:
            source_direction: The original semantic direction
            target_direction: The direction it transforms into
            base_weight: Base transformation weight (0-1)
            name: Optional name for the rule
        """
        self.source_direction = source_direction
        self.target_direction = target_direction
        self.base_weight = base_weight
        self.name = name or f"{source_direction}_to_{target_direction}"
        self.conditions = []
        self.modifiers = []
    
    def add_condition(self, condition_func: Callable[[Dict[str, Any]], bool], 
                     description: str = None):
        """
        Add a condition for when this rule applies.
        
        Args:
            condition_func: Function that takes domain context and returns boolean
            description: Optional description of the condition
        """
        self.conditions.append({
            'function': condition_func,
            'description': description or "Unnamed condition"
        })
        return self  # For method chaining
    
    def add_modifier(self, modifier_func: Callable[[Dict[str, Any]], float], 
                    description: str = None):
        """
        Add a modifier that adjusts the transformation weight.
        
        Args:
            modifier_func: Function that takes domain context and returns a weight modifier
            description: Optional description of the modifier
        """
        self.modifiers.append({
            'function': modifier_func,
            'description': description or "Unnamed modifier"
        })
        return self  # For method chaining
    
    def applies_to(self, domain_context: Dict[str, Any]) -> bool:
        """
        Check if this rule applies to a given domain context.
        
        Args:
            domain_context: The domain context to check
            
        Returns:
            True if all conditions are met, False otherwise
        """
        # If no conditions, rule always applies
        if not self.conditions:
            return True
            
        # Check all conditions
        return all(cond['function'](domain_context) for cond in self.conditions)
    
    def calculate_weight(self, domain_context: Dict[str, Any]) -> float:
        """
        Calculate the transformation weight for a given domain context.
        
        Args:
            domain_context: The domain context to calculate weight for
            
        Returns:
            The adjusted transformation weight
        """
        weight = self.base_weight
        
        # Apply all modifiers
        for mod in self.modifiers:
            modifier = mod['function'](domain_context)
            weight *= modifier
            
        return min(1.0, max(0.0, weight))  # Clamp to 0-1 range


class TransformationRuleRegistry:
    """
    Registry for transformation rules.
    
    Maintains a collection of rules and provides methods to apply them
    to domain contexts.
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self.rules = {}
    
    def register_rule(self, rule: TransformationRule):
        """
        Register a transformation rule.
        
        Args:
            rule: The rule to register
        """
        source = rule.source_direction
        if source not in self.rules:
            self.rules[source] = []
        self.rules[source].append(rule)
    
    def get_applicable_rules(self, direction: str, domain_context: Dict[str, Any]) -> List[TransformationRule]:
        """
        Get all applicable rules for a direction in a domain context.
        
        Args:
            direction: The source direction
            domain_context: The domain context
            
        Returns:
            List of applicable transformation rules
        """
        if direction not in self.rules:
            return []
            
        return [rule for rule in self.rules[direction] if rule.applies_to(domain_context)]
    
    def transform_direction(self, direction: str, weight: float, domain_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Transform a direction according to applicable rules.
        
        Args:
            direction: The source direction
            weight: The original weight
            domain_context: The domain context
            
        Returns:
            Dictionary mapping transformed directions to weights
        """
        result = {direction: weight}
        
        # Get applicable rules
        rules = self.get_applicable_rules(direction, domain_context)
        
        # Apply each rule
        for rule in rules:
            transform_weight = rule.calculate_weight(domain_context)
            target = rule.target_direction
            
            # Calculate transformed weight
            transformed_weight = weight * transform_weight
            
            # Add or update target direction
            if target in result:
                result[target] += transformed_weight
            else:
                result[target] = transformed_weight
                
            # Reduce source weight proportionally
            result[direction] -= transformed_weight * 0.5  # Only reduce by half to allow for multiple transformations
            
        # Ensure no negative weights
        result = {d: max(0, w) for d, w in result.items()}
        
        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {d: w/total for d, w in result.items()}
            
        return result


# Common conditions
def has_actant(actant_name: str) -> Callable[[Dict[str, Any]], bool]:
    """Create a condition that checks if a domain has a specific actant."""
    def condition(domain: Dict[str, Any]) -> bool:
        return actant_name in domain.get('actants', [])
    return condition

def has_emphasis(emphasis_key: str, min_value: float = 0.5) -> Callable[[Dict[str, Any]], bool]:
    """Create a condition that checks if a domain has a specific emphasis."""
    def condition(domain: Dict[str, Any]) -> bool:
        context = domain.get('context', {})
        if isinstance(context, dict):
            emphasis = context.get('emphasis', {})
            return emphasis.get(emphasis_key, 0) >= min_value
        return False
    return condition

def has_flow(flow_name: str, min_value: float = 0.3) -> Callable[[Dict[str, Any]], bool]:
    """Create a condition that checks if a domain has a specific semantic flow."""
    def condition(domain: Dict[str, Any]) -> bool:
        flows = domain.get('semantic_flows', {})
        return flows.get(flow_name, 0) >= min_value
    return condition

# Common modifiers
def emphasis_modifier(emphasis_key: str, factor: float = 0.5) -> Callable[[Dict[str, Any]], float]:
    """Create a modifier based on emphasis strength."""
    def modifier(domain: Dict[str, Any]) -> float:
        context = domain.get('context', {})
        if isinstance(context, dict):
            emphasis = context.get('emphasis', {})
            value = emphasis.get(emphasis_key, 0)
            return 1.0 + (value * factor)
        return 1.0
    return modifier

def actant_presence_modifier(actant_names: List[str], factor: float = 0.2) -> Callable[[Dict[str, Any]], float]:
    """Create a modifier based on presence of multiple actants."""
    def modifier(domain: Dict[str, Any]) -> float:
        domain_actants = set(domain.get('actants', []))
        overlap = len([a for a in actant_names if a in domain_actants])
        return 1.0 + (overlap * factor / len(actant_names))
    return modifier


# Create standard rule set
def create_standard_ruleset() -> TransformationRuleRegistry:
    """Create a standard set of transformation rules."""
    registry = TransformationRuleRegistry()
    
    # adapts -> supports (when community is present)
    adapts_to_supports = TransformationRule(
        source_direction="adapts",
        target_direction="supports",
        base_weight=0.4,
        name="adaptation_leads_to_support"
    )
    adapts_to_supports.add_condition(
        has_actant("community"),
        "Community is present"
    ).add_modifier(
        emphasis_modifier("adaptation", 0.6),
        "Stronger when adaptation is emphasized"
    )
    registry.register_rule(adapts_to_supports)
    
    # supports -> funds (when economy is present)
    supports_to_funds = TransformationRule(
        source_direction="supports",
        target_direction="funds",
        base_weight=0.35,
        name="support_leads_to_funding"
    )
    supports_to_funds.add_condition(
        has_actant("economy"),
        "Economy is present"
    ).add_modifier(
        emphasis_modifier("economic_growth", 0.7),
        "Stronger when economic growth is emphasized"
    )
    registry.register_rule(supports_to_funds)
    
    # funds -> invests (when infrastructure is present)
    funds_to_invests = TransformationRule(
        source_direction="funds",
        target_direction="invests",
        base_weight=0.45,
        name="funding_leads_to_investment"
    )
    funds_to_invests.add_condition(
        has_actant("infrastructure"),
        "Infrastructure is present"
    ).add_modifier(
        actant_presence_modifier(["economy", "policy"], 0.3),
        "Stronger when economy and policy are present"
    )
    registry.register_rule(funds_to_invests)
    
    # threatens -> adapts (when community and policy are present)
    threatens_to_adapts = TransformationRule(
        source_direction="threatens",
        target_direction="adapts",
        base_weight=0.3,
        name="threats_lead_to_adaptation"
    )
    threatens_to_adapts.add_condition(
        lambda domain: has_actant("community")(domain) and has_actant("policy")(domain),
        "Both community and policy are present"
    ).add_modifier(
        emphasis_modifier("vulnerability", 0.5),
        "Stronger when vulnerability is emphasized"
    )
    registry.register_rule(threatens_to_adapts)
    
    # adapts -> fails (when resources are insufficient)
    adapts_to_fails = TransformationRule(
        source_direction="adapts",
        target_direction="fails",
        base_weight=0.25,
        name="adaptation_may_fail"
    )
    adapts_to_fails.add_condition(
        lambda domain: "economy" not in domain.get('actants', []),
        "Economy is absent (insufficient resources)"
    )
    registry.register_rule(adapts_to_fails)
    
    return registry
