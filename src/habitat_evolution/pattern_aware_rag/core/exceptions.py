"""
Exceptions for pattern-aware RAG core functionality.
"""

class PatternAwareRAGError(Exception):
    """Base exception class for pattern-aware RAG errors."""
    pass

class InvalidStateError(PatternAwareRAGError):
    """Raised when a state is invalid."""
    pass

class StateValidationError(PatternAwareRAGError):
    """Raised when state validation fails."""
    pass

class WindowStateError(PatternAwareRAGError):
    """Raised when a learning window operation fails due to invalid state."""
    pass
