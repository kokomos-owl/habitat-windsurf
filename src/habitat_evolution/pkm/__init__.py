"""
Pattern Knowledge Medium (PKM) Package for Habitat Evolution.

This package provides components for creating, managing, and sharing PKM files,
which encapsulate patterns detected by the Habitat Evolution system along with
their relationships, metadata, and user attribution.

The PKM system enables bidirectional flow between patterns and knowledge,
allowing patterns to drive query generation and knowledge synthesis, while
also capturing the resulting knowledge in a structured format that can be
shared and evolved collaboratively.
"""

from src.habitat_evolution.pkm.pkm_repository import PKMFile, PKMRepository, create_pkm_from_claude_response
from src.habitat_evolution.pkm.pkm_bidirectional_integration import PKMBidirectionalIntegration
from src.habitat_evolution.pkm.pkm_factory import create_pkm_repository, create_pkm_bidirectional_integration

__all__ = [
    "PKMFile", 
    "PKMRepository", 
    "create_pkm_from_claude_response", 
    "PKMBidirectionalIntegration",
    "create_pkm_repository",
    "create_pkm_bidirectional_integration"
]
