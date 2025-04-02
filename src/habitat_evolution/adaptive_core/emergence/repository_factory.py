"""
Factory for creating repository instances.

This module provides factory methods for creating repository instances
that implement the repository interfaces defined in the interfaces package.

NOTE: This module is deprecated and will be removed in a future version.
Please use habitat_evolution.adaptive_core.persistence.factory instead.
"""

import logging
from typing import Optional, Dict, Any

# Import from the new location
from habitat_evolution.adaptive_core.persistence.factory import (
    create_field_state_repository,
    create_pattern_repository,
    create_relationship_repository,
    create_topology_repository,
    create_repositories
)

logger = logging.getLogger(__name__)

# The implementation has been moved to habitat_evolution.adaptive_core.persistence.factory
# This file now imports from there for backward compatibility
