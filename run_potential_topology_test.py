"""
Test runner for the Topological-Temporal Potential framework.

This script demonstrates the capabilities of the Topological-Temporal Potential
framework in the Habitat Evolution system.
"""
import asyncio
import json
from datetime import datetime
from pprint import pprint

# Import from src directory
import sys
sys.path.append("src")

from habitat_evolution.adaptive_core.persistence.models.pattern_state import PatternState
from habitat_evolution.field.persistence.semantic_potential_calculator import SemanticPotentialCalculator
from habitat_evolution.field.emergence.concept_predicate_syntax_model import ConceptPredicateSyntaxModel


class MockGraphService:
    """Mock implementation of GraphService for testing."""
    
    def __init__(self):
        """Initialize with mock data."""
        self.repository = self
        self.patterns = {}
        self.concepts = {}
        self.relations = {}
        self.transitions = {}
        
        # Create some mock patterns
        self._create_mock_patterns()
        
    def _create_mock_patterns(self):
        """Create mock patterns for testing."""
        # Create semantic patterns
        self.patterns["sem1"] = PatternState(
            id="sem1",
            content="Climate change is affecting global temperatures",
            metadata={"type": "semantic", "category": "climate"},
            timestamp=datetime.now(),
            confidence=0.85
        )
        
        self.patterns["sem2"] = PatternState(
            id="sem2",
            content="Renewable energy sources are becoming more affordable",
            metadata={"type": "semantic", "category": "energy"},
            timestamp=datetime.now(),
            confidence=0.78
        )
        
        # Create statistical patterns
        self.patterns["stat1"] = PatternState(
            id="stat1",
            content="Temperature anomaly correlation with CO2 levels",
            metadata={"type": "statistical", "category": "climate"},
            timestamp=datetime.now(),
            confidence=0.92
        )
        
        self.patterns["stat2"] = PatternState(
            id="stat2",
            content="Solar panel efficiency improvement trend",
            metadata={"type": "statistical", "category": "energy"},
            timestamp=datetime.now(),
            confidence=0.88
        )
        
        # Create relations
        self.relations["rel1"] = {
            "id": "rel1",
            "relation_type": "correlates_with",
            "source_id": "sem1",
            "target_id": "stat1",
            "weight": 0.87,
            "attributes": {
                "confidence": "0.87",
                "type": "correlation"
            }
        }
        
        self.relations["rel2"] = {
            "id": "rel2",
            "relation_type": "supports",
            "source_id": "sem2",
            "target_id": "stat2",
            "weight": 0.82,
            "attributes": {
                "confidence": "0.82",
                "type": "support"
            }
        }
        
        # Create transitions
        self.transitions["trans1"] = [
            {
                "id": "t1",
                "node_id": "sem1",
                "from_quality": "uncertain",
                "to_quality": "good",
                "timestamp": datetime.now(),
                "context": {"evidence": "Multiple sources confirm"}
            }
        ]
        
        self.transitions["trans2"] = [
            {
                "id": "t2",
                "node_id": "stat1",
                "from_quality": "poor",
                "to_quality": "uncertain",
                "timestamp": datetime.now(),
                "context": {"evidence": "Initial statistical correlation"}
            },
            {
                "id": "t3",
                "node_id": "stat1",
                "from_quality": "uncertain",
                "to_quality": "good",
                "timestamp": datetime.now(),
                "context": {"evidence": "Confirmed by additional data"}
            }
        ]
    
    def find_nodes_by_quality(self, quality_state, node_type=None):
        """Find nodes by quality state."""
        # For testing, return all patterns as "good" quality
        if quality_state == "good":
            return list(self.patterns.values())
        return []
    
    def find_relations_by_quality(self, quality_states):
        """Find relations by quality states."""
        # For testing, return all relations
        return [
            type("Relation", (), {
                "id": r["id"],
                "relation_type": r["relation_type"],
                "source_id": r["source_id"],
                "target_id": r["target_id"],
                "weight": r["weight"],
                "attributes": r["attributes"]
            })
            for r in self.relations.values()
        ]
    
    def find_node_by_id(self, node_id):
        """Find node by ID."""
        return self.patterns.get(node_id)
    
    def find_quality_transitions_by_node_id(self, node_id):
        """Find quality transitions by node ID."""
        for transitions in self.transitions.values():
            node_transitions = [
                type("Transition", (), {
                    "id": t["id"],
                    "node_id": t["node_id"],
                    "from_quality": t["from_quality"],
                    "to_quality": t["to_quality"],
                    "timestamp": t["timestamp"],
                    "context": t["context"]
                })
                for t in transitions
                if t["node_id"] == node_id
            ]
            if node_transitions:
                return node_transitions
        return []


async def test_semantic_potential_calculator():
    """Test the SemanticPotentialCalculator."""
    print("\n=== Testing SemanticPotentialCalculator ===")
    
    # Create mock services
    graph_service = MockGraphService()
    
    # Create calculator
    calculator = SemanticPotentialCalculator(graph_service)
    
    # Test pattern potential calculation
    print("\nCalculating pattern potential for 'sem1'...")
    pattern_potential = await calculator.calculate_pattern_potential("sem1")
    print(f"Pattern Potential: {json.dumps(pattern_potential, indent=2, default=str)}")
    
    # Test field potential calculation
    print("\nCalculating field potential...")
    field_potential = await calculator.calculate_field_potential()
    print(f"Field Potential: {json.dumps(field_potential, indent=2, default=str)}")
    
    # Test topological potential calculation
    print("\nCalculating topological potential...")
    topo_potential = await calculator.calculate_topological_potential()
    print(f"Topological Potential: {json.dumps(topo_potential, indent=2, default=str)}")
    
    return calculator


async def main():
    """Run the test demonstration."""
    print("=== Habitat Evolution: Topological-Temporal Potential Framework ===")
    
    # Test semantic potential calculator
    calculator = await test_semantic_potential_calculator()
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
