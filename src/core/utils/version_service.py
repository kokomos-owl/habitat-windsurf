class VersionService:
    """Service for managing versions."""
    
    def __init__(self):
        self._version = "1.0.0"
    
    def get_version(self) -> str:
        """Get current version."""
        return self._version
