"""Functional tests for user integration with adaptive patterns."""

import unittest
from datetime import datetime
from typing import Dict, Any

from habitat_evolution.adaptive_core.id.user_id import UserID
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

class TestUserPatternIntegration(unittest.TestCase):
    """Test user interaction with patterns and knowledge media."""
    
    def setUp(self):
        """Initialize test environment."""
        self.user = UserID(
            username="test_user",
            email="test@example.com"
        )
        self.collaborator = UserID(
            username="collaborator",
            email="collaborator@example.com"
        )
        
        # Create a test pattern owned by the user
        self.pattern = AdaptiveID(
            base_concept="test_pattern",
            creator_id=self.user.id,
            weight=0.8,
            confidence=0.9
        )

    def test_pattern_ownership(self):
        """Test pattern creation and ownership."""
        # Register pattern with user
        self.user.register_pattern(
            self.pattern.id,
            {"type": "test", "description": "Test pattern"}
        )
        
        # Verify pattern ownership
        self.assertIn(self.pattern.id, self.user.context.discovered_patterns)
        self.assertEqual(self.pattern.creator_id, self.user.id)
        
        # Verify user interaction tracking
        self.assertIn(self.user.id, self.pattern.user_interactions)
        self.assertEqual(
            self.pattern.user_interactions[self.user.id]["role"],
            "creator"
        )

    def test_pattern_sharing(self):
        """Test pattern sharing between users."""
        # Register and share pattern
        self.user.register_pattern(
            self.pattern.id,
            {"type": "test", "description": "Test pattern"}
        )
        self.user.share_pattern(self.pattern.id, [self.collaborator.id])
        
        # Verify sharing
        self.assertTrue(
            self.user.can_access_pattern(self.pattern.id, self.collaborator.id)
        )
        self.assertIn(self.pattern.id, self.user.context.shared_patterns)

    def test_pattern_collaboration(self):
        """Test collaboration on patterns."""
        # Setup collaboration
        self.user.register_pattern(
            self.pattern.id,
            {"type": "test", "description": "Test pattern"}
        )
        self.user.share_pattern(self.pattern.id, [self.collaborator.id])
        self.user.add_collaboration(self.pattern.id, self.collaborator.id)
        
        # Verify collaboration tracking
        self.assertIn(self.collaborator.id, self.user.context.collaborations)

    def test_user_context_preservation(self):
        """Test preservation of user context in pattern snapshots."""
        # Create pattern and register
        self.user.register_pattern(
            self.pattern.id,
            {"type": "test", "description": "Test pattern"}
        )
        
        # Create snapshot
        snapshot = self.pattern.create_snapshot()
        
        # Create new pattern from snapshot
        new_pattern = AdaptiveID(
            base_concept="new_pattern",
            creator_id=self.user.id
        )
        new_pattern.restore_from_snapshot(snapshot)
        
        # Verify user context preserved
        self.assertEqual(
            new_pattern.user_interactions[self.user.id]["role"],
            "creator"
        )

    def test_user_preferences_impact(self):
        """Test how user preferences affect pattern interaction."""
        # Set user preferences
        self.user.update_preferences({
            "pattern_visibility": "public",
            "collaboration_mode": "open",
            "learning_style": "interactive"
        })
        
        # Register pattern
        self.user.register_pattern(
            self.pattern.id,
            {
                "type": "test",
                "description": "Test pattern",
                "visibility": self.user.context.preferences["pattern_visibility"]
            }
        )
        
        # Verify preferences applied
        self.assertEqual(
            self.user.context.preferences["pattern_visibility"],
            "public"
        )

if __name__ == '__main__':
    unittest.main()
