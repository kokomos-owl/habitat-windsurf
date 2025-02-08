"""IO system for pattern processing with adaptive windows."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import asyncio
from enum import Enum
import logging

from src.core.temporal.adaptive_window import (
    WindowState,
    AdaptiveWindow,
    WindowStateManager
)

class ProcessingState(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    PROCESSING = "processing"
    THROTTLED = "throttled"
    ERROR = "error"

@dataclass
class ProcessingMetrics:
    """Metrics for pattern processing."""
    patterns_processed: int = 0
    patterns_queued: int = 0
    average_processing_time: float = 0.0
    error_count: int = 0
    last_update: datetime = None
    
    def update_processing_time(self, processing_time: float):
        """Update average processing time."""
        if self.patterns_processed == 0:
            self.average_processing_time = processing_time
        else:
            self.average_processing_time = (
                (self.average_processing_time * self.patterns_processed + processing_time) /
                (self.patterns_processed + 1)
            )
        self.patterns_processed += 1
        self.last_update = datetime.now()

class PatternProcessor:
    """Handles IO processing of patterns with adaptive windows."""
    
    def __init__(
        self,
        max_concurrent: int = 5,
        processing_timeout: float = 30.0
    ):
        self.processing_state = ProcessingState.IDLE
        self.metrics = ProcessingMetrics()
        self.max_concurrent = max_concurrent
        self.processing_timeout = processing_timeout
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)
        self.pattern_queue = asyncio.Queue()
        self.window_manager = WindowStateManager()
        self.processing_tasks: List[asyncio.Task] = []
        self._shutdown = False
        
        # Handlers for different pattern types
        self.pattern_handlers: Dict[str, Callable] = {}
        
    async def start(self):
        """Start the pattern processor."""
        self._shutdown = False
        self.processing_tasks.append(
            asyncio.create_task(self._process_queue())
        )
        
    async def stop(self):
        """Stop the pattern processor."""
        self._shutdown = True
        # Wait for processing to complete
        await self.pattern_queue.join()
        for task in self.processing_tasks:
            task.cancel()
        
    def register_handler(
        self,
        pattern_type: str,
        handler: Callable
    ):
        """Register a handler for a specific pattern type."""
        self.pattern_handlers[pattern_type] = handler
        
    async def submit_pattern(
        self,
        pattern: Any,
        window_id: str
    ) -> bool:
        """Submit a pattern for processing."""
        try:
            # Check window state first
            window = self.window_manager.windows.get(window_id)
            if not window:
                window = await self.window_manager.create_window(window_id)
                
            # Update window with pattern
            success = await window.process_pattern(pattern)
            if not success:
                return False
                
            # Queue pattern for processing
            await self.pattern_queue.put((pattern, window_id))
            self.metrics.patterns_queued += 1
            
            return True
        except Exception as e:
            logging.error(f"Error submitting pattern: {e}")
            self.metrics.error_count += 1
            return False
            
    async def _process_queue(self):
        """Process patterns from the queue."""
        while not self._shutdown:
            try:
                # Get next pattern
                pattern, window_id = await self.pattern_queue.get()
                
                # Process with concurrency control
                async with self.processing_semaphore:
                    await self._process_pattern(pattern, window_id)
                    
                self.pattern_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error processing pattern: {e}")
                self.metrics.error_count += 1
                
    async def _process_pattern(
        self,
        pattern: Any,
        window_id: str
    ):
        """Process a single pattern."""
        start_time = datetime.now()
        
        try:
            # Get handler for pattern type
            handler = self.pattern_handlers.get(
                getattr(pattern, 'pattern_type', None)
            )
            if not handler:
                raise ValueError(f"No handler for pattern type: {pattern.pattern_type}")
                
            # Process with timeout
            async with asyncio.timeout(self.processing_timeout):
                await handler(pattern)
                
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics.update_processing_time(processing_time)
            
        except asyncio.TimeoutError:
            logging.error(f"Pattern processing timeout: {pattern}")
            self.metrics.error_count += 1
        except Exception as e:
            logging.error(f"Error processing pattern: {e}")
            self.metrics.error_count += 1
            
    async def get_window_status(
        self,
        window_id: str
    ) -> Optional[Dict]:
        """Get status of a specific window."""
        window = self.window_manager.windows.get(window_id)
        if not window:
            return None
            
        return {
            'state': window.state.value,
            'density': window.density_metrics.current_density,
            'trend': window.density_metrics.trend,
            'acceleration': window.density_metrics.acceleration
        }
        
    def get_processing_metrics(self) -> Dict:
        """Get current processing metrics."""
        return {
            'state': self.processing_state.value,
            'patterns_processed': self.metrics.patterns_processed,
            'patterns_queued': self.metrics.patterns_queued,
            'average_processing_time': self.metrics.average_processing_time,
            'error_count': self.metrics.error_count,
            'last_update': self.metrics.last_update
        }
