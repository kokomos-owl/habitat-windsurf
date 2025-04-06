"""
Integration test for the relational accretion model in Habitat Evolution.

This test validates the complete integration of the relational accretion model,
ensuring that queries properly accrete significance through interactions with patterns
and that this significance influences pattern evolution.
"""

import os
import sys
import pytest
import asyncio
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.services.event_service import EventService
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.core.services.field.field_state_service import ConcreteFieldStateService
from src.habitat_evolution.core.services.field.gradient_service import GradientService
from src.habitat_evolution.core.services.field.flow_dynamics_service import FlowDynamicsService
from src.habitat_evolution.adaptive_core.services.metrics_service import MetricsService
from src.habitat_evolution.adaptive_core.services.quality_metrics_service import QualityMetricsService
from src.habitat_evolution.pattern_aware_rag.core.coherence_analyzer import CoherenceAnalyzer
from src.habitat_evolution.pattern_aware_rag.interfaces.pattern_emergence import PatternEmergenceFlow
from src.habitat_evolution.pattern_aware_rag.services.graph_service import GraphService
from src.habitat_evolution.pattern_aware_rag.services.claude_baseline_service import ClaudeBaselineService
from src.habitat_evolution.pattern_aware_rag.services.significance_accretion_service import SignificanceAccretionService
from src.habitat_evolution.pattern_aware_rag.accretive_pattern_rag import AccretivePatternRAG

class TestRelationalAccretionIntegration:
    """Test the integration of the relational accretion model."""
    
    @pytest.fixture
    async def setup_services(self):
        """Set up the services needed for testing."""
        # Create event service
        event_service = EventService()
        
        # Create mock ArangoDB connection
        db_connection = MockArangoDBConnection()
        
        # Create significance accretion service
        significance_service = SignificanceAccretionService(
            db_connection=db_connection,
            event_service=event_service
        )
        
        # Create other required services
        pattern_evolution_service = MockPatternEvolutionService(
            db_connection=db_connection,
            event_service=event_service
        )
        
        field_state_service = ConcreteFieldStateService(
            event_service=event_service
        )
        
        gradient_service = GradientService()
        
        flow_dynamics_service = FlowDynamicsService()
        
        metrics_service = MockMetricsService()
        
        quality_metrics_service = MockQualityMetricsService()
        
        coherence_analyzer = MockCoherenceAnalyzer()
        
        emergence_flow = MockPatternEmergenceFlow()
        
        graph_service = MockGraphService()
        
        # Create accretive pattern RAG
        rag = AccretivePatternRAG(
            pattern_evolution_service=pattern_evolution_service,
            field_state_service=field_state_service,
            gradient_service=gradient_service,
            flow_dynamics_service=flow_dynamics_service,
            metrics_service=metrics_service,
            quality_metrics_service=quality_metrics_service,
            event_service=event_service,
            coherence_analyzer=coherence_analyzer,
            emergence_flow=emergence_flow,
            settings={},
            graph_service=graph_service,
            db_connection=db_connection
        )
        
        return {
            "event_service": event_service,
            "db_connection": db_connection,
            "significance_service": significance_service,
            "pattern_evolution_service": pattern_evolution_service,
            "rag": rag
        }
    
    @pytest.mark.asyncio
    async def test_query_significance_accretion(self, setup_services):
        """Test that query significance accretes through interactions."""
        services = await setup_services
        rag = services["rag"]
        significance_service = services["significance_service"]
        
        # Process a sequence of related queries
        queries = [
            "What are the projected sea level rise impacts for Martha's Vineyard?",
            "How will sea level rise affect coastal properties on Martha's Vineyard?",
            "What adaptation strategies are recommended for sea level rise on Martha's Vineyard?",
            "How does Martha's Vineyard's sea level rise compare to other coastal areas?"
        ]
        
        # Track significance over time
        significance_history = []
        
        # Process each query and track significance
        for query in queries:
            result = await rag.query(query)
            query_id = result["query_id"]
            
            # Get significance after processing
            significance = await significance_service.get_query_significance(query_id)
            significance_history.append(significance)
            
            # Log significance metrics
            print(f"Query: {query}")
            print(f"Accretion Level: {significance['accretion_level']:.2f}")
            print(f"Semantic Stability: {significance['semantic_stability']:.2f}")
            print(f"Relational Density: {significance['relational_density']:.2f}")
            print(f"Emergence Potential: {significance['emergence_potential']:.2f}")
            print(f"Interaction Count: {significance['interaction_count']}")
            print(f"Vector Size: {len(significance['significance_vector'])}")
            print("---")
        
        # Verify that significance accretes over time
        assert significance_history[-1]["accretion_level"] > significance_history[0]["accretion_level"]
        assert significance_history[-1]["semantic_stability"] > significance_history[0]["semantic_stability"]
        assert significance_history[-1]["relational_density"] > significance_history[0]["relational_density"]
        
        # Verify that patterns are generated once significance reaches threshold
        pattern_ids = services["pattern_evolution_service"].get_created_pattern_ids()
        assert len(pattern_ids) > 0, "No patterns were generated from query significance"
        
        # Verify bidirectional flow through event system
        events = services["event_service"].get_published_events()
        assert any(e["event_type"] == "query.significance.updated" for e in events)
        assert any(e["event_type"] == "pattern.created" for e in events)
        
        # Verify that the final query has high enough significance to generate patterns
        final_significance = significance_history[-1]
        assert final_significance["accretion_level"] >= 0.3, "Final query did not accrete enough significance"
    
    @pytest.mark.asyncio
    async def test_significance_to_pattern_generation(self, setup_services):
        """Test that sufficient significance leads to pattern generation."""
        services = await setup_services
        rag = services["rag"]
        significance_service = services["significance_service"]
        pattern_evolution_service = services["pattern_evolution_service"]
        
        # Process a query with high initial significance
        query = "What comprehensive adaptation strategies exist for sea level rise on Martha's Vineyard?"
        
        # Process the query
        result = await rag.query(query)
        query_id = result["query_id"]
        
        # Get the significance
        significance = await significance_service.get_query_significance(query_id)
        
        # Manually increase significance to trigger pattern generation
        significance["accretion_level"] = 0.8
        significance["semantic_stability"] = 0.7
        significance["relational_density"] = 0.6
        significance["significance_vector"] = {
            "pattern-1": 0.9,
            "pattern-2": 0.8,
            "pattern-3": 0.7
        }
        
        # Update significance
        await significance_service.update_significance(
            query_id=query_id,
            interaction_metrics={
                "coherence_score": 0.9,
                "retrieval_quality": 0.8,
                "pattern_relevance": significance["significance_vector"]
            },
            accretion_rate=0.5
        )
        
        # Process the query again to trigger pattern generation
        result = await rag.query(query)
        
        # Verify that a pattern was generated
        pattern_ids = pattern_evolution_service.get_created_pattern_ids()
        assert len(pattern_ids) > 0, "No patterns were generated from high significance query"
        
        # Get the generated pattern
        pattern = pattern_evolution_service.get_pattern(pattern_ids[-1])
        
        # Verify pattern properties
        assert pattern["confidence"] > 0.7, "Generated pattern has low confidence"
        assert pattern["coherence"] > 0.6, "Generated pattern has low coherence"
        assert "query_origin" in pattern["properties"], "Pattern does not have query origin property"
        assert pattern["properties"]["query_origin"] is True, "Pattern is not marked as originating from a query"
        
        # Verify that the pattern has related patterns from the significance vector
        assert len(pattern["properties"]["related_patterns"]) > 0, "Pattern has no related patterns"


# Mock implementations for testing

class MockArangoDBConnection:
    """Mock ArangoDB connection for testing."""
    
    def __init__(self):
        self.collections = {}
        self.documents = {}
    
    def collection_exists(self, collection_name):
        return collection_name in self.collections
    
    def create_collection(self, collection_name, edge=False):
        self.collections[collection_name] = {"is_edge": edge}
        self.documents[collection_name] = {}
    
    async def execute_query(self, query, bind_vars=None):
        # Simple mock implementation that handles basic operations
        if "INSERT" in query and "RETURN NEW" in query:
            # Extract collection name
            collection_name = query.split("INTO")[1].split("RETURN")[0].strip()
            
            # Extract document
            doc_start = query.find("{")
            doc_end = query.rfind("}")
            doc_str = query[doc_start:doc_end+1]
            doc = json.loads(doc_str)
            
            # Store document
            doc_id = doc.get("_key", str(uuid.uuid4()))
            self.documents.setdefault(collection_name, {})[doc_id] = doc
            return [doc]
            
        elif "FOR doc IN" in query and "RETURN doc" in query:
            # Extract collection name
            collection_name = query.split("FOR doc IN")[1].split("FILTER")[0].strip()
            
            # Extract filter condition
            if "FILTER" in query:
                filter_condition = query.split("FILTER")[1].split("RETURN")[0].strip()
                
                # Simple handling for doc.query_id == @query_id
                if "doc.query_id == @query_id" in filter_condition and bind_vars and "query_id" in bind_vars:
                    query_id = bind_vars["query_id"]
                    
                    # Find document with matching query_id
                    for doc_id, doc in self.documents.get(collection_name, {}).items():
                        if doc.get("query_id") == query_id:
                            return [doc]
            
            # If no filter or no match, return all documents
            return list(self.documents.get(collection_name, {}).values())
            
        elif "UPDATE" in query and "RETURN NEW" in query:
            # Extract bind vars for the document
            if bind_vars and "new_significance" in bind_vars:
                doc = bind_vars["new_significance"]
                doc_id = doc.get("_key", str(uuid.uuid4()))
                collection_name = query.split("IN")[1].split("RETURN")[0].strip()
                
                # Update document
                self.documents.setdefault(collection_name, {})[doc_id] = doc
                return [doc]
        
        # Default return empty list
        return []


class MockPatternEvolutionService:
    """Mock pattern evolution service for testing."""
    
    def __init__(self, db_connection, event_service):
        self.db_connection = db_connection
        self.event_service = event_service
        self.patterns = {}
        self.created_pattern_ids = []
    
    async def create_pattern(self, pattern_data):
        pattern_id = pattern_data.get("id", f"pattern-{str(uuid.uuid4())}")
        self.patterns[pattern_id] = pattern_data
        self.created_pattern_ids.append(pattern_id)
        
        # Publish event
        if self.event_service:
            self.event_service.publish(
                "pattern.created",
                {
                    "pattern_id": pattern_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        return pattern_id
    
    def get_created_pattern_ids(self):
        return self.created_pattern_ids
    
    def get_pattern(self, pattern_id):
        return self.patterns.get(pattern_id, {})


class MockMetricsService:
    """Mock metrics service for testing."""
    
    def __init__(self):
        pass
    
    async def record_metric(self, metric_name, value, context=None):
        pass


class MockQualityMetricsService:
    """Mock quality metrics service for testing."""
    
    def __init__(self):
        pass
    
    async def calculate_quality_metrics(self, data, context=None):
        return {
            "coherence": 0.8,
            "stability": 0.7,
            "signal_strength": 0.75
        }


class MockCoherenceAnalyzer:
    """Mock coherence analyzer for testing."""
    
    def __init__(self):
        pass
    
    async def analyze_coherence(self, patterns, context=None):
        return {
            "coherence_score": 0.8,
            "pattern_coherence": {p.get("id", "unknown"): 0.75 for p in patterns}
        }


class MockPatternEmergenceFlow:
    """Mock pattern emergence flow for testing."""
    
    def __init__(self):
        self.flow_state = "stable"
    
    def get_flow_state(self):
        return self.flow_state
    
    async def update_flow_state(self, metrics):
        pass


class MockGraphService:
    """Mock graph service for testing."""
    
    def __init__(self):
        pass
    
    async def create_graph_from_patterns(self, patterns):
        return {
            "nodes": [{"id": p.get("id"), "label": p.get("base_concept")} for p in patterns],
            "edges": []
        }
