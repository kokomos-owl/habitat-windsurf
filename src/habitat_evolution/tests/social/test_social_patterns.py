"""Test social pattern evolution through real-world scenarios."""

import pytest
from typing import Dict, Any
import asyncio
from datetime import datetime

from ...social.services.social_pattern_service import SocialPatternService, SocialMetrics
from ...core.storage.interfaces import PatternStore, RelationshipStore
from ...core.services.event_bus import LocalEventBus
from ...core.pattern.quality import PatternQualityAnalyzer

class MockPatternStore(PatternStore):
    """Mock pattern storage for testing."""
    def __init__(self):
        self.patterns = {}
        
    async def store(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        pattern_id = pattern_data.get("id", str(len(self.patterns)))
        self.patterns[pattern_id] = pattern_data
        return {"id": pattern_id}
        
    async def get(self, pattern_id: str) -> Dict[str, Any]:
        return self.patterns.get(pattern_id, {})
        
    async def update(self, pattern_id: str, updates: Dict[str, Any]) -> bool:
        if pattern_id in self.patterns:
            self.patterns[pattern_id].update(updates)
            return True
        return False

class MockRelationshipStore(RelationshipStore):
    """Mock relationship storage for testing."""
    def __init__(self):
        self.relationships = {}
        
    async def store(self, relationship_data: Dict[str, Any]) -> Dict[str, Any]:
        rel_id = str(len(self.relationships))
        self.relationships[rel_id] = relationship_data
        return {"id": rel_id}
        
    async def get_related(self, pattern_id: str, rel_type: str = None) -> List[Dict[str, Any]]:
        return [r for r in self.relationships.values() 
                if r["source_id"] == pattern_id 
                and (rel_type is None or r["type"] == rel_type)]

@pytest.fixture
def event_bus():
    return LocalEventBus()

@pytest.fixture
def pattern_store():
    return MockPatternStore()

@pytest.fixture
def relationship_store():
    return MockRelationshipStore()

@pytest.fixture
def social_pattern_service(pattern_store, relationship_store, event_bus):
    return SocialPatternService(pattern_store, relationship_store, event_bus)

class TestCommunityPracticeEmergence:
    """Test emergence of community practices from individual patterns."""
    
    async def test_local_food_network_emergence(self, social_pattern_service):
        """Test how individual food sharing evolves into community practice."""
        
        # Individual pattern: Personal garden sharing
        garden_pattern = await social_pattern_service.register_social_pattern(
            pattern_type="resource_sharing",
            content={
                "resource_type": "food",
                "sharing_frequency": "weekly",
                "participants": 3,
                "location": "neighborhood"
            },
            field_state={
                "energy": 0.4,
                "coherence": 0.3,
                "flow": 0.5
            }
        )
        
        # Track evolution as more people join
        for week in range(12):
            await social_pattern_service.track_practice_evolution(
                pattern_id=garden_pattern.data["id"],
                practice_data={
                    "participants": 3 + week * 2,
                    "sharing_frequency": "weekly",
                    "resource_types": ["vegetables", "fruits", "herbs"],
                    "organization_level": "informal"
                }
            )
        
        # Verify emergence of stable community practice
        pattern = await social_pattern_service.get_pattern(garden_pattern.data["id"])
        metrics = SocialMetrics.from_dict(pattern.data["social_metrics"])
        
        assert metrics.practice_maturity > 0.7
        assert metrics.institutionalization > 0.5
        
    async def test_skill_sharing_network(self, social_pattern_service):
        """Test evolution of informal skill sharing into structured practice."""
        
        # Individual pattern: Peer teaching
        teaching_pattern = await social_pattern_service.register_social_pattern(
            pattern_type="knowledge_sharing",
            content={
                "knowledge_type": "practical_skills",
                "format": "informal_meetup",
                "participants": 5,
                "frequency": "monthly"
            },
            field_state={
                "energy": 0.5,
                "coherence": 0.4,
                "flow": 0.6
            }
        )
        
        # Track evolution as format becomes more structured
        stages = [
            {
                "format": "scheduled_workshops",
                "participants": 8,
                "organization_level": "coordinated"
            },
            {
                "format": "regular_programs",
                "participants": 15,
                "organization_level": "structured"
            },
            {
                "format": "community_school",
                "participants": 25,
                "organization_level": "institutional"
            }
        ]
        
        for stage in stages:
            await social_pattern_service.track_practice_evolution(
                pattern_id=teaching_pattern.data["id"],
                practice_data=stage
            )
            
        # Verify institutional emergence
        pattern = await social_pattern_service.get_pattern(teaching_pattern.data["id"])
        metrics = SocialMetrics.from_dict(pattern.data["social_metrics"])
        
        assert metrics.practice_maturity > 0.8
        assert metrics.institutionalization > 0.7
        
    async def test_climate_response_network(self, social_pattern_service):
        """Test emergence of climate response practices from individual actions."""
        
        # Individual pattern: Coastal monitoring
        monitoring_pattern = await social_pattern_service.register_social_pattern(
            pattern_type="environmental_practice",
            content={
                "activity_type": "coastal_monitoring",
                "frequency": "weekly",
                "participants": 4,
                "data_collection": "informal"
            },
            field_state={
                "energy": 0.6,
                "coherence": 0.5,
                "flow": 0.4
            }
        )
        
        # Evolution stages
        stages = [
            {
                "participants": 8,
                "data_collection": "structured",
                "coordination": "informal_network"
            },
            {
                "participants": 15,
                "data_collection": "systematic",
                "coordination": "working_group"
            },
            {
                "participants": 30,
                "data_collection": "scientific",
                "coordination": "formal_organization",
                "partnerships": ["research_institutions", "local_government"]
            }
        ]
        
        for stage in stages:
            await social_pattern_service.track_practice_evolution(
                pattern_id=monitoring_pattern.data["id"],
                practice_data=stage
            )
            
        # Verify emergence of institutional practice
        pattern = await social_pattern_service.get_pattern(monitoring_pattern.data["id"])
        metrics = SocialMetrics.from_dict(pattern.data["social_metrics"])
        
        assert metrics.practice_maturity > 0.8
        assert metrics.institutionalization > 0.7
        assert metrics.influence_reach > 0.6
