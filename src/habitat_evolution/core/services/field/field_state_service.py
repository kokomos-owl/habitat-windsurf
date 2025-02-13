"""
Concrete implementation of the field state service.
"""

from typing import Dict, Optional
from datetime import datetime

from .interfaces import FieldStateService, FieldState
from ...storage.field_repository import FieldRepository
from ...services.event_bus import EventBus
from ...quality.metrics import calculate_field_stability

class ConcreteFieldStateService(FieldStateService):
    """
    Concrete implementation of FieldStateService that manages field states
    and coordinates with the event system.
    """
    
    def __init__(
        self,
        field_repository: FieldRepository,
        event_bus: EventBus
    ):
        self.repository = field_repository
        self.event_bus = event_bus
        
    async def get_field_state(self, field_id: str) -> Optional[FieldState]:
        """Get the current state of a field"""
        try:
            state = await self.repository.get_field_state(field_id)
            if not state:
                return None
                
            return FieldState(
                field_id=field_id,
                timestamp=state.timestamp,
                potential=state.potential,
                gradient=state.gradient,
                stability=state.stability,
                metadata=state.metadata
            )
        except Exception as e:
            await self.event_bus.emit("field.state.error", {
                "field_id": field_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
            
    async def update_field_state(self, field_id: str, state: FieldState) -> None:
        """Update the state of a field"""
        try:
            # Calculate new stability before update
            new_stability = await self.calculate_field_stability(field_id)
            state.stability = new_stability
            
            # Update state in repository
            await self.repository.update_field_state(field_id, state)
            
            # Emit state change event
            await self.event_bus.emit("field.state.updated", {
                "field_id": field_id,
                "timestamp": datetime.now().isoformat(),
                "stability": new_stability,
                "potential": state.potential
            })
            
        except Exception as e:
            await self.event_bus.emit("field.state.error", {
                "field_id": field_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
            
    async def calculate_field_stability(self, field_id: str) -> float:
        """Calculate the stability metric for a field"""
        try:
            state = await self.get_field_state(field_id)
            if not state:
                raise ValueError(f"No state found for field {field_id}")
                
            # Calculate stability using quality metrics
            stability = calculate_field_stability(
                potential=state.potential,
                gradient=state.gradient,
                metadata=state.metadata
            )
            
            # Emit stability calculation event
            await self.event_bus.emit("field.stability.calculated", {
                "field_id": field_id,
                "stability": stability,
                "timestamp": datetime.now().isoformat()
            })
            
            return stability
            
        except Exception as e:
            await self.event_bus.emit("field.stability.error", {
                "field_id": field_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
