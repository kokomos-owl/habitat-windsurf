"""
Oscillatory Properties Module

This module implements oscillatory properties for entities in the Habitat Evolution system,
enabling pattern recognition, coherence maintenance, and predictive capabilities through
wave-like signatures.
"""

from .oscillatory_signature import OscillatorySignature, HarmonicComponent
from .signature_repository import OscillatorySignatureRepository
from .signature_service import OscillatorySignatureService

__all__ = [
    'OscillatorySignature',
    'HarmonicComponent',
    'OscillatorySignatureRepository',
    'OscillatorySignatureService'
]
