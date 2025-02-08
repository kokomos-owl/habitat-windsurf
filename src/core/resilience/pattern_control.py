"""Pattern control with version control, backpressure, and circuit breaking."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List
import asyncio
from enum import Enum

class VersionState(Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"

@dataclass
class Version:
    major: int
    minor: int
    patch: int
    timestamp: datetime
    state: VersionState = VersionState.ACTIVE

    def __gt__(self, other: 'Version') -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing
    HALF_OPEN = "half_open"  # Testing recovery

class PatternCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 30):
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time: Optional[datetime] = None

    async def execute(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError()

        try:
            result = await func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failures = 0
            return result
        except Exception as e:
            self._handle_failure()
            raise e

    def _handle_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now()
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.reset_timeout

class BackpressureController:
    def __init__(self, max_queue_size: int = 100):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = False

    async def submit(self, pattern: 'Pattern') -> bool:
        try:
            await asyncio.wait_for(self.queue.put(pattern), timeout=1.0)
            return True
        except asyncio.TimeoutError:
            return False

    async def start_processing(self, processor_func):
        self.processing = True
        while self.processing:
            try:
                pattern = await self.queue.get()
                await processor_func(pattern)
                self.queue.task_done()
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing pattern: {e}")

    def stop_processing(self):
        self.processing = False

class VersionedPatternManager:
    def __init__(self):
        self.patterns: Dict[str, List[tuple[Version, 'Pattern']]] = {}
        self.circuit_breaker = PatternCircuitBreaker()
        self.backpressure = BackpressureController()

    async def register_pattern(self, pattern_id: str, pattern: 'Pattern', version: Version):
        async def _register():
            if pattern_id not in self.patterns:
                self.patterns[pattern_id] = []
            
            # Supersede older versions
            for v, _ in self.patterns[pattern_id]:
                if version > v:
                    v.state = VersionState.SUPERSEDED
            
            self.patterns[pattern_id].append((version, pattern))
            self.patterns[pattern_id].sort(key=lambda x: x[0], reverse=True)

        # Use circuit breaker for registration
        await self.circuit_breaker.execute(_register)

    async def get_pattern(self, pattern_id: str, version: Optional[Version] = None) -> Optional['Pattern']:
        if pattern_id not in self.patterns:
            return None
            
        if version:
            for v, p in self.patterns[pattern_id]:
                if v == version:
                    return p
        else:
            # Get latest active version
            for v, p in self.patterns[pattern_id]:
                if v.state == VersionState.ACTIVE:
                    return p
        
        return None

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass
